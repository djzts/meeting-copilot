"""Local speech-to-text backends (faster-whisper / mlx-whisper) + hallucination filter."""
from __future__ import annotations

import glob
import importlib.util
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("stt")


@dataclass
class Utterance:
    id: int
    t: float  # seconds since the session started (speech start)
    text: str
    is_question: bool = False
    t_end: float = 0.0
    wall_end: float = 0.0  # time.time() when the speaker stopped
    stt_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("wall_end", None)
        return d


class Transcript:
    """Thread-safe list of utterances."""

    def __init__(self, max_items: int = 5000):
        self._items: list[Utterance] = []
        self._lock = threading.Lock()
        self._next_id = 1
        self.max_items = max_items

    def add(self, t: float, text: str, **kw) -> Utterance:
        with self._lock:
            u = Utterance(id=self._next_id, t=t, text=text, **kw)
            self._next_id += 1
            self._items.append(u)
            if len(self._items) > self.max_items:
                del self._items[: len(self._items) - self.max_items]
            return u

    def get(self, utt_id: int) -> Optional[Utterance]:
        with self._lock:
            for u in reversed(self._items):
                if u.id == utt_id:
                    return u
        return None

    def last(self) -> Optional[Utterance]:
        with self._lock:
            return self._items[-1] if self._items else None

    def previous(self, utt_id: int) -> Optional[Utterance]:
        with self._lock:
            for i in range(len(self._items) - 1, -1, -1):
                if self._items[i].id == utt_id:
                    return self._items[i - 1] if i > 0 else None
        return None

    def window(self, end_t: float, minutes: float) -> list[Utterance]:
        """Utterances in [end_t - minutes, end_t]."""
        start = end_t - minutes * 60
        with self._lock:
            return [u for u in self._items if start <= u.t <= end_t + 0.01]

    def all(self, limit: Optional[int] = None) -> list[Utterance]:
        with self._lock:
            return list(self._items[-limit:] if limit else self._items)


@dataclass
class STTConfig:
    backend: str = "faster-whisper"  # or "mlx"
    model: str = "large-v3-turbo"
    language: str = ""  # "" = auto-detect, or "zh" / "en"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "int8"
    initial_prompt: str = ""
    beam_size: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "STTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Hallucination filter
# --------------------------------------------------------------------------- #
# Whole-utterance phrases Whisper tends to invent on silence / noise / music.
_EXACT_HALLUCINATIONS = {
    "thank you", "thank you very much", "thank you so much", "thanks", "thanks for watching",
    "thank you for watching", "thanks for listening", "thank you for listening",
    "please subscribe", "subscribe", "like and subscribe", "you", "bye", "bye bye",
    "the end", "music", "applause", "silence", "foreign",
    "谢谢", "谢谢观看", "谢谢大家", "谢谢收看", "感谢观看", "请订阅", "订阅", "字幕",
}
# Substrings that only ever appear in subtitle credits.
_SUBSTRING_HALLUCINATIONS = [
    "字幕由", "字幕提供", "字幕志愿者", "字幕组", "中文字幕", "小编字幕", "请不吝点赞", "点赞 订阅",
    "订阅 转发", "明镜与点点", "amara.org", "subtitles by", "subtitled by", "captions by",
    "transcribed by", "translated by",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    return text.strip()


def collapse_repeats(text: str, max_repeats: int = 3) -> str:
    """Collapse Whisper's runaway loops ('the the the the ...', '好的好的好的好的')."""
    # repeated words / short phrases separated by spaces or punctuation
    pattern = re.compile(r"((?:\b[\w']+\b[\s,.!?]*){1,4}?)\1{%d,}" % max_repeats, re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(lambda m: m.group(1) * max_repeats, text)
    # repeated CJK chunks without separators
    text = re.sub(r"([一-鿿]{1,6})\1{%d,}" % max_repeats, lambda m: m.group(1) * max_repeats, text)
    return text


def clean_text(text: str) -> Optional[str]:
    """Return cleaned text, or None if it is empty or a known hallucination."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    text = collapse_repeats(text)
    norm = _normalize(text)
    if not norm:
        return None  # only punctuation / symbols (e.g. "...", "♪")
    if norm in _EXACT_HALLUCINATIONS:
        return None
    low = text.lower()
    if any(s in low for s in _SUBSTRING_HALLUCINATIONS):
        return None
    if re.fullmatch(r"[\s♪♫🎵*\-–—.…]+", text):
        return None
    return text


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _add_windows_cuda_dll_dirs() -> None:
    """faster-whisper (CTranslate2) needs cuBLAS / cuDNN DLLs on Windows.

    They are often already shipped by PyTorch or the nvidia-* pip wheels, so we
    add those folders to the DLL search path without importing torch.
    """
    if sys.platform != "win32":
        return
    dirs = []
    for pkg in ("torch", "nvidia"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            if pkg == "torch":
                dirs.append(os.path.join(loc, "lib"))
            else:
                dirs.extend(glob.glob(os.path.join(loc, "*", "bin")))
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    name = "base"

    def transcribe(self, audio: np.ndarray) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self) -> None:
        try:
            self.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up failed: %s", e)


class FasterWhisperTranscriber(Transcriber):
    name = "faster-whisper"

    def __init__(self, cfg: STTConfig):
        _add_windows_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.cfg = cfg
        device = cfg.device
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        compute_type = cfg.compute_type or "default"
        try:
            self.model = WhisperModel(cfg.model, device=device, compute_type=compute_type)
        except Exception as e:  # noqa: BLE001
            if device != "cpu" and cfg.device == "auto":
                log.warning("CUDA init failed (%s); falling back to CPU", e)
                device = "cpu"
                self.model = WhisperModel(cfg.model, device="cpu", compute_type="int8")
            else:
                raise
        self.device = device
        log.info("faster-whisper model=%s device=%s compute=%s", cfg.model, device, compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language or None,
            beam_size=max(1, self.cfg.beam_size),
            condition_on_previous_text=False,
            initial_prompt=self.cfg.initial_prompt or None,
            vad_filter=False,
            without_timestamps=True,
        )
        return "".join(s.text for s in segments).strip()


class MlxTranscriber(Transcriber):
    name = "mlx"

    def __init__(self, cfg: STTConfig):
        import mlx_whisper  # noqa: F401  (Apple Silicon only)

        self.cfg = cfg
        self.repo = cfg.model if "/" in cfg.model else f"mlx-community/whisper-{cfg.model}"
        self.device = "mlx"
        log.info("mlx-whisper model=%s", self.repo)

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        # mlx-whisper only implements greedy decoding (equivalent to beam_size=1)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.cfg.language or None,
            initial_prompt=self.cfg.initial_prompt or None,
            condition_on_previous_text=False,
            temperature=0.0,
            verbose=None,
        )
        return (result.get("text") or "").strip()


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def create_transcriber(cfg: STTConfig) -> Transcriber:
    backend = cfg.backend.lower().replace("_", "-")
    if backend in ("mlx", "mlx-whisper"):
        return MlxTranscriber(cfg)
    if backend in ("faster-whisper", "fasterwhisper", "whisper"):
        return FasterWhisperTranscriber(cfg)
    raise ValueError(f"unknown stt backend {cfg.backend!r}")


class EchoTranscriber(Transcriber):
    """Test helper: returns a fixed list of texts in order."""

    name = "echo"

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.device = "none"

    def transcribe(self, audio: np.ndarray) -> str:
        return self.texts.pop(0) if self.texts else ""


def transcribe_segment(tr: Transcriber, audio: np.ndarray) -> tuple[Optional[str], int]:
    """Transcribe and filter; returns (text or None, elapsed ms)."""
    t0 = time.perf_counter()
    raw = tr.transcribe(audio)
    ms = int((time.perf_counter() - t0) * 1000)
    return clean_text(raw), ms
