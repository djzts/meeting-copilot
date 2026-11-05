"""Audio capture, resampling to 16 kHz and silence-based segmentation.

The pipeline is: source (sound device or WAV file) -> mono mix -> resample to
16 kHz -> 30 ms RMS voice-activity frames -> Segment objects on a queue.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("audio")

TARGET_RATE = 16000
FRAME_SEC = 0.03

# Preferred host APIs when the same device shows up several times (Windows lists
# every device under MME, DirectSound, WASAPI and WDM-KS).
_HOSTAPI_PREFERENCE = ["Windows WASAPI", "MME", "Windows DirectSound", "Core Audio", "ALSA", "JACK"]


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """A chunk of speech, 16 kHz mono float32."""

    audio: np.ndarray
    t_start: float  # seconds since the stream started
    t_end: float  # end of the last voiced frame, seconds since stream start
    wall_end: float = 0.0  # time.time() when the speaker stopped talking
    wall_ready: float = 0.0  # time.time() when the segment was cut

    @property
    def duration(self) -> float:
        return len(self.audio) / TARGET_RATE


# --------------------------------------------------------------------------- #
# Signal processing helpers
# --------------------------------------------------------------------------- #
def to_mono(block: np.ndarray) -> np.ndarray:
    """Mix (frames, channels) audio down to a 1-D float32 array."""
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1, dtype=np.float32)


class Resampler:
    """Streaming resampler: windowed-sinc low-pass + linear interpolation.

    Good enough for speech recognition and has no SciPy dependency. Keeps state
    between calls so block boundaries do not produce clicks.
    """

    def __init__(self, src_rate: float, dst_rate: float = TARGET_RATE, taps: int = 63):
        self.src_rate = float(src_rate)
        self.dst_rate = float(dst_rate)
        self.passthrough = abs(self.src_rate - self.dst_rate) < 1e-6
        self.step = self.src_rate / self.dst_rate  # input samples per output sample
        if not self.passthrough:
            if self.src_rate > self.dst_rate:
                fc = 0.45 * self.dst_rate / self.src_rate  # cycles/sample
            else:
                fc = 0.45
            n = np.arange(taps) - (taps - 1) / 2
            h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
            self.kernel = (h / h.sum()).astype(np.float32)
            self.history = np.zeros(taps - 1, dtype=np.float32)
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0  # position of the next output sample inside self.buf

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.passthrough or len(x) == 0:
            return x
        padded = np.concatenate([self.history, x])
        filtered = np.convolve(padded, self.kernel, mode="valid").astype(np.float32)
        self.history = padded[-(len(self.kernel) - 1):]
        self.buf = np.concatenate([self.buf, filtered])
        # interpolation needs buf[i] and buf[i+1]
        last = len(self.buf) - 1
        if self.pos > last - 1e-9:
            return np.zeros(0, dtype=np.float32)
        n_out = int(np.floor((last - self.pos) / self.step)) + 1
        idx = self.pos + self.step * np.arange(n_out)
        idx = idx[idx <= last]
        out = np.interp(idx, np.arange(len(self.buf)), self.buf).astype(np.float32)
        next_pos = self.pos + self.step * len(idx)
        drop = int(np.floor(next_pos))
        drop = min(drop, len(self.buf) - 1)
        self.buf = self.buf[drop:]
        self.pos = next_pos - drop
        return out


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class Segmenter:
    """RMS voice-activity detector that cuts the 16 kHz stream into utterances.

    * 30 ms frames; a frame is voiced when RMS > silence_threshold
    * 0.3 s of pre-roll is kept before the first voiced frame
    * a segment ends after >= end_silence_sec of silence, or when it reaches
      max_segment_sec (then the next segment starts immediately)
    * segments whose voiced span is shorter than min_segment_sec are dropped
    """

    def __init__(
        self,
        rate: int = TARGET_RATE,
        silence_threshold: float = 0.01,
        end_silence_sec: float = 0.8,
        max_segment_sec: float = 20.0,
        min_segment_sec: float = 0.6,
        preroll_sec: float = 0.3,
        frame_sec: float = FRAME_SEC,
    ):
        self.rate = rate
        self.frame_len = int(round(rate * frame_sec))
        self.frame_sec = self.frame_len / rate
        self.threshold = silence_threshold
        self.end_silence_frames = max(1, int(round(end_silence_sec / self.frame_sec)))
        self.max_frames = max(1, int(round(max_segment_sec / self.frame_sec)))
        self.min_voiced_sec = min_segment_sec
        self.preroll = deque(maxlen=max(0, int(round(preroll_sec / self.frame_sec))))
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_index = 0  # index of the next frame to be processed
        self._seg: Optional[list[np.ndarray]] = None
        self._seg_first_frame = 0
        self._first_voiced = 0
        self._last_voiced = 0
        self._silent_run = 0
        self.last_level = 0.0

    @property
    def stream_time(self) -> float:
        """Seconds of audio consumed so far (frame aligned)."""
        return self._frame_index * self.frame_sec

    @property
    def in_speech(self) -> bool:
        return self._seg is not None

    def feed(self, samples: np.ndarray) -> list[Segment]:
        out: list[Segment] = []
        samples = np.asarray(samples, dtype=np.float32)
        if len(self._pending):
            samples = np.concatenate([self._pending, samples])
        n_frames = len(samples) // self.frame_len
        for i in range(n_frames):
            frame = samples[i * self.frame_len:(i + 1) * self.frame_len]
            seg = self._process_frame(frame)
            if seg is not None:
                out.append(seg)
        self._pending = samples[n_frames * self.frame_len:].copy()
        return out

    def flush(self) -> list[Segment]:
        """End of stream: emit whatever is in progress."""
        out = []
        if self._seg is not None:
            seg = self._finish()
            if seg is not None:
                out.append(seg)
        return out

    # -- internals --------------------------------------------------------- #
    def _process_frame(self, frame: np.ndarray) -> Optional[Segment]:
        idx = self._frame_index
        self._frame_index += 1
        level = rms(frame)
        self.last_level = level
        voiced = level > self.threshold
        result = None
        if self._seg is None:
            if voiced:
                self._seg = list(self.preroll) + [frame]
                self._seg_first_frame = idx - len(self.preroll)
                self._first_voiced = idx
                self._last_voiced = idx
                self._silent_run = 0
                self.preroll.clear()
            else:
                self.preroll.append(frame)
            return None

        self._seg.append(frame)
        if voiced:
            self._last_voiced = idx
            self._silent_run = 0
        else:
            self._silent_run += 1

        if self._silent_run >= self.end_silence_frames:
            result = self._finish()
        elif len(self._seg) >= self.max_frames:
            result = self._finish()
            # keep going: the speaker is still talking, start a new segment now
            if voiced:
                self._seg = []
                self._seg_first_frame = idx + 1
                self._first_voiced = idx + 1
                self._last_voiced = idx
                self._silent_run = 0
        return result

    def _finish(self) -> Optional[Segment]:
        frames = self._seg or []
        self._seg = None
        self.preroll.clear()
        if not frames:
            return None
        voiced_sec = (self._last_voiced - self._first_voiced + 1) * self.frame_sec
        if voiced_sec < self.min_voiced_sec:
            return None
        # Keep a short tail of silence (0.2 s) and drop the rest.
        keep_frames = (self._last_voiced - self._seg_first_frame + 1) + int(round(0.2 / self.frame_sec))
        frames = frames[:max(1, keep_frames)]
        audio = np.concatenate(frames).astype(np.float32)
        t_start = max(0, self._seg_first_frame) * self.frame_sec
        t_end = (self._last_voiced + 1) * self.frame_sec
        return Segment(audio=audio, t_start=t_start, t_end=t_end)


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def _sd():
    import sounddevice as sd  # imported lazily so tests do not need PortAudio

    return sd


def list_devices() -> str:
    """Return a printable table of input-capable devices."""
    sd = _sd()
    apis = sd.query_hostapis()
    lines = ["Input devices (use part of the name as [audio].device):", ""]
    lines.append(f"{'idx':>4}  {'host API':<20} {'ch':>3} {'rate':>7}  name")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = apis[d["hostapi"]]["name"]
        lines.append(f"{i:>4}  {api:<20} {d['max_input_channels']:>3} {int(d['default_samplerate']):>7}  {d['name']}")
    lines.append("")
    lines.append("Capture the meeting's audio (system output), not your own microphone:")
    lines.append("  Windows: 'CABLE Output' (VB-Audio Virtual Cable) or 'Stereo Mix'")
    lines.append("  macOS:   'BlackHole'")
    return "\n".join(lines)


def find_device(name_part: str) -> int:
    """Find an input device whose name contains name_part (case-insensitive).

    When several host APIs expose the same device, WASAPI/MME are preferred.
    """
    sd = _sd()
    if name_part.strip().isdigit():
        return int(name_part)
    apis = sd.query_hostapis()
    needle = name_part.lower()
    matches = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and needle in d["name"].lower():
            api = apis[d["hostapi"]]["name"]
            rank = _HOSTAPI_PREFERENCE.index(api) if api in _HOSTAPI_PREFERENCE else len(_HOSTAPI_PREFERENCE)
            matches.append((rank, i))
    if not matches:
        raise ValueError(f"No input device matches {name_part!r}.\n\n" + list_devices())
    matches.sort()
    return matches[0][1]


# --------------------------------------------------------------------------- #
# Pipeline (one thread per source)
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    device: str = ""
    silence_threshold: float = 0.01
    end_silence_sec: float = 0.8
    max_segment_sec: float = 20.0
    min_segment_sec: float = 0.6
    preroll_sec: float = 0.3
    channels: int = 0  # 0 = min(2, device max)

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class AudioPipeline:
    """Runs a source in a background thread and puts Segments on out_q."""

    def __init__(
        self,
        cfg: AudioConfig,
        out_q: "queue.Queue[Segment]",
        simulate: Optional[str] = None,
        on_level: Optional[Callable[[float, bool], None]] = None,
        realtime: bool = True,
    ):
        self.cfg = cfg
        self.out_q = out_q
        self.simulate = simulate
        self.on_level = on_level
        self.realtime = realtime
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.source_name = ""
        self.error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._segmenter = Segmenter(
            silence_threshold=cfg.silence_threshold,
            end_silence_sec=cfg.end_silence_sec,
            max_segment_sec=cfg.max_segment_sec,
            min_segment_sec=cfg.min_segment_sec,
            preroll_sec=cfg.preroll_sec,
        )
        self._last_level_report = 0.0

    def start(self) -> None:
        target = self._run_file if self.simulate else self._run_device
        self._thread = threading.Thread(target=target, name="audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    # -- shared processing -------------------------------------------------- #
    def _consume(self, mono16k: np.ndarray) -> None:
        segs = self._segmenter.feed(mono16k)
        self._emit(segs)
        now = time.time()
        if self.on_level and now - self._last_level_report > 0.5:
            self._last_level_report = now
            self.on_level(self._segmenter.last_level, self._segmenter.in_speech)

    def _emit(self, segs: list[Segment]) -> None:
        if not segs:
            return
        now = time.time()
        stream_now = self._segmenter.stream_time
        for s in segs:
            s.wall_ready = now
            s.wall_end = now - max(0.0, stream_now - s.t_end)
            log.debug("segment %.2f-%.2f (%.2fs)", s.t_start, s.t_end, s.duration)
            self.out_q.put(s)

    # -- live device -------------------------------------------------------- #
    def _run_device(self) -> None:
        try:
            sd = _sd()
            idx = find_device(self.cfg.device)
            info = sd.query_devices(idx)
            rate = float(info["default_samplerate"])
            channels = self.cfg.channels or min(2, int(info["max_input_channels"]))
            self.source_name = info["name"]
            resampler = Resampler(rate, TARGET_RATE)
            blocks: "queue.Queue[np.ndarray]" = queue.Queue()

            def callback(indata, frames, time_info, status):  # PortAudio thread
                if status:
                    log.debug("audio status: %s", status)
                blocks.put(indata.copy())

            log.info("recording from [%d] %s @ %d Hz, %d ch", idx, info["name"], rate, channels)
            with sd.InputStream(
                device=idx,
                samplerate=rate,
                channels=channels,
                dtype="float32",
                blocksize=int(rate * 0.05),
                callback=callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        block = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self._consume(resampler.process(to_mono(block)))
        except Exception as e:  # noqa: BLE001 - surface any device error to the UI
            self.error = f"audio: {e}"
            log.error("%s", self.error)
        finally:
            self._emit(self._segmenter.flush())
            self.finished.set()

    # -- WAV simulation ----------------------------------------------------- #
    def _run_file(self) -> None:
        try:
            rate, data = read_wav(self.simulate)
            self.source_name = f"simulate: {self.simulate}"
            resampler = Resampler(rate, TARGET_RATE)
            block = int(rate * 0.05)
            t0 = time.monotonic()
            log.info("simulating %s (%.1f s) at %s speed", self.simulate, len(data) / rate,
                     "real-time" if self.realtime else "max")
            for start in range(0, len(data), block):
                if self.stop_event.is_set():
                    break
                chunk = data[start:start + block]
                if self.realtime:
                    due = t0 + (start + len(chunk)) / rate
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                self._consume(resampler.process(to_mono(chunk)))
            # push a little silence so the final utterance gets cut normally
            tail = np.zeros(int(TARGET_RATE * (self.cfg.end_silence_sec + 0.1)), dtype=np.float32)
            self._consume(tail)
        except Exception as e:  # noqa: BLE001
            self.error = f"simulate: {e}"
            log.error("%s", self.error)
        finally:
            self._emit(self._segmenter.flush())
            self.finished.set()
            log.info("simulation finished")


def read_wav(path: str) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file into float32 (frames, channels) in [-1, 1]."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width {width}")
    return rate, data.reshape(-1, ch)


def write_wav(path: str, audio: np.ndarray, rate: int = TARGET_RATE) -> None:
    """Write mono float32 audio as 16-bit PCM."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
