"""meeting-copilot: live meeting transcription + answer suggestions from your Obsidian vault.

    python copilot.py --config config.toml
    python copilot.py --list-devices
    python copilot.py --simulate meeting.wav --dry-run
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import os
import queue
import sys
import threading
import time
import tomllib
import webbrowser
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audio  # noqa: E402
import detector  # noqa: E402
import settings  # noqa: E402
import stt  # noqa: E402
from answer import AnswerEngine, AskRequest, ClaudeConfig  # noqa: E402
from vault import Vault  # noqa: E402

log = logging.getLogger("copilot")

DEFAULTS: dict = {
    "audio": {"device": "", "silence_threshold": 0.01, "end_silence_sec": 0.8, "max_segment_sec": 20,
              "min_segment_sec": 0.6, "preroll_sec": 0.3},
    "stt": {"backend": "faster-whisper", "model": "large-v3-turbo", "language": "", "device": "auto",
            "compute_type": "int8", "initial_prompt": ""},
    "vault": {"path": "", "name": "", "exclude": [], "embedding_model": "", "top_k": 6, "per_note": 2},
    "claude": {"cli": "claude", "model": "haiku", "context_minutes": 3, "auto_answer": True,
               "allow_vault_tools": True, "about_me": ""},
    "ui": {"host": "127.0.0.1", "port": 8765},
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> tuple[dict, Optional[Path]]:
    candidates = [Path(path)] if path else [Path("config.toml"), HERE / "config.toml",
                                             HERE / "config.example.toml"]
    for p in candidates:
        if p.is_file():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            if p.name == "config.example.toml" and not path:
                log.warning("config.toml not found, using config.example.toml (copy it to config.toml)")
            cfg = deep_merge(DEFAULTS, data)
            # a relative vault path is relative to the config file
            vpath = str(cfg["vault"].get("path") or "").strip()
            cfg["vault"]["_path_raw"] = vpath
            cfg["vault"]["path"] = resolve_vault_path(vpath, p)
            return cfg, p
    if path:
        raise SystemExit(f"config file not found: {path}")
    return copy.deepcopy(DEFAULTS), None


def resolve_vault_path(vpath: str, config_path: Optional[Path]) -> str:
    """A relative vault path is relative to the config file."""
    vpath = (vpath or "").strip()
    if not vpath:
        return ""
    vp = Path(os.path.expanduser(vpath))
    if not vp.is_absolute():
        vp = ((config_path.resolve().parent if config_path else HERE) / vp).resolve()
    return str(vp)


# --------------------------------------------------------------------------- #
# Event hub: keeps UI state and fans events out to SSE clients
# --------------------------------------------------------------------------- #
class EventHub:
    MAX_UTTS = 1000
    MAX_ANSWERS = 50

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: set[queue.Queue] = set()
        self.utts: deque = deque(maxlen=self.MAX_UTTS)
        self.answers: "OrderedDict[int, dict]" = OrderedDict()
        self.status: dict = {}
        self.info: dict = {}

    def publish(self, event: str, data: dict) -> None:
        with self._lock:
            self._apply(event, data)
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait((event, data))
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.discard(q)

    def _apply(self, event: str, data: dict) -> None:
        if event == "utt":
            self.utts.append(data)
        elif event == "answer_start":
            self.answers[data["id"]] = {**data, "text": "", "state": "streaming", "tool": None}
            while len(self.answers) > self.MAX_ANSWERS:
                self.answers.popitem(last=False)
        elif event == "answer_delta":
            a = self.answers.get(data.get("id"))
            if a is not None:
                if data.get("reset"):
                    a["text"] = ""
                if data.get("tool"):
                    a["tool"] = data["tool"]
                if data.get("text"):
                    a["text"] += data["text"]
        elif event == "answer_end":
            a = self.answers.get(data.get("id"))
            if a is not None:
                a.update(data)
                a["state"] = "error" if data.get("error") else "done"
        elif event == "answer_skip":
            self.answers.pop(data.get("id"), None)
        elif event == "status":
            self.status.update(data)
        elif event == "info":
            self.info.update(data)
        elif event == "error":
            self.status["last_error"] = data.get("message")

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=5000)
        with self._lock:
            snapshot = {
                "info": self.info,
                "utts": list(self.utts),
                "answers": list(self.answers.values()),
                "status": dict(self.status),
            }
            q.put(("snapshot", snapshot))
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    def snapshot(self) -> dict:
        with self._lock:
            return {"info": self.info, "utts": list(self.utts), "answers": list(self.answers.values()),
                    "status": dict(self.status)}


# --------------------------------------------------------------------------- #
# HTTP / SSE
# --------------------------------------------------------------------------- #
def make_handler(app: "App"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MeetingCopilot/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            log.debug("http: " + fmt, *args)

        def _host_ok(self) -> bool:
            if app.cfg["ui"]["host"] not in ("127.0.0.1", "localhost", "::1"):
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            return host in ("127.0.0.1", "localhost", "::1")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (HERE / "ui.html").read_bytes()
                return self._send(200, body, "text/html; charset=utf-8")
            if path == "/events":
                return self._sse()
            if path == "/state":
                return self._json(200, app.hub.snapshot())
            if path == "/config":
                return self._json(200, app.settings_payload())
            return self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if ctype != "application/json":  # forces a CORS pre-flight for cross-site requests
                return self._json(415, {"error": "use Content-Type: application/json"})
            try:
                data = json.loads(raw or b"{}")
                if not isinstance(data, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            path = self.path.split("?", 1)[0]
            if path == "/ask":
                ok, msg = app.ask(data.get("utt_id"), data.get("text"))
                return self._json(200 if ok else 400, {"ok": ok, "message": msg})
            if path == "/reindex":
                started = app.reindex()
                return self._json(200, {"ok": True, "started": started})
            if path == "/config":
                try:
                    result = app.apply_settings(data.get("changes", {}), save=bool(data.get("save", True)))
                except settings.SettingsError as e:
                    return self._json(400, {"ok": False, "errors": e.errors, "message": str(e)})
                except OSError as e:
                    return self._json(500, {"ok": False, "message": f"could not save the config: {e}"})
                return self._json(200, result)
            return self._json(404, {"error": "not found"})

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = app.hub.subscribe()
            try:
                while not app.stopping.is_set():
                    try:
                        event, data = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(data, ensure_ascii=False)
                    self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                app.hub.unsubscribe(q)
                self.close_connection = True

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, cfg: dict, args: argparse.Namespace, config_path: Optional[Path] = None):
        self.cfg = cfg
        self.args = args
        self.config_path = config_path
        self.raw_vault_path = str(cfg["vault"].pop("_path_raw", cfg["vault"].get("path", "")) or "")
        self.stopping = threading.Event()
        self.hub = EventHub()
        self.transcript = stt.Transcript()
        self.vault = self._make_vault()
        self.vault_name = cfg["vault"].get("name") or self.vault.name
        self.claude_cfg = ClaudeConfig.from_dict(cfg["claude"])
        self.stt_cfg = stt.STTConfig.from_dict(cfg["stt"])  # shared with the transcriber: edits apply live
        self._stt_reload = threading.Event()
        self._settings_lock = threading.Lock()
        self._unsaved: dict[str, dict] = {}  # applied from the UI but not written to the config file
        self.engine = AnswerEngine(self.claude_cfg, self.vault, self.transcript, self.hub.publish,
                                   dry_run=args.dry_run, vault_name=self.vault_name)
        self.seg_q: "queue.Queue[audio.Segment]" = queue.Queue()
        self.utt_q: "queue.Queue[stt.Utterance]" = queue.Queue()
        self.pipeline: Optional[audio.AudioPipeline] = None
        self.httpd: Optional[Server] = None

    def status(self, **kw) -> None:
        self.hub.publish("status", kw)

    # -- settings (web UI) ---------------------------------------------------- #
    def _make_vault(self) -> Vault:
        vcfg = self.cfg["vault"]
        return Vault(
            vcfg.get("path", ""),
            exclude=vcfg.get("exclude", []),
            embedding_model=vcfg.get("embedding_model", ""),
            top_k=int(vcfg.get("top_k", 6)),
            per_note=int(vcfg.get("per_note", 2)),
        )

    def _save_target(self) -> Path:
        """Where UI changes are saved: the loaded config, but never the example template."""
        p = self.config_path
        if p is None or p.name == "config.example.toml":
            return HERE / "config.toml"
        return p

    def _live_values(self) -> dict:
        """Current settings; Claude / STT come from the objects the running code uses."""
        view = {**self.cfg,
                "claude": {**self.cfg["claude"], **dataclasses.asdict(self.claude_cfg)},
                "stt": {**self.cfg["stt"], **dataclasses.asdict(self.stt_cfg)}}
        return settings.current_values(view, self.raw_vault_path)

    def settings_payload(self) -> dict:
        pipe = self.pipeline
        loopback_names = None
        if pipe is not None and not pipe.simulate and not pipe.finished.is_set() \
                and audio.parse_loopback(pipe.cfg.device) is not None:
            loopback_names = pipe.loopback_names  # don't re-initialise PortAudio under a live stream
        return {
            **settings.schema(),
            "values": self._live_values(),
            "unsaved": sorted(f"{s}.{k}" for s, vals in self._unsaved.items() for k in vals),
            "config_path": str(self._save_target()),
            "device_options": audio.device_options(loopback_names),
            "simulate": bool(self.args.simulate),
            "dry_run": bool(self.args.dry_run),
        }

    def apply_settings(self, changes: dict, save: bool = True) -> dict:
        """Apply settings edited in the UI to the running app; optionally save them.

        Raises settings.SettingsError (nothing applied) when a value is invalid.
        """
        new = settings.validate_changes(changes)
        with self._settings_lock:
            current = self._live_values()
            diff = {s: {k: v for k, v in vals.items() if current.get(s, {}).get(k) != v}
                    for s, vals in new.items()}
            diff = {s: vals for s, vals in diff.items() if vals}
            for s, vals in diff.items():
                for k, v in vals.items():
                    if (s, k) == ("vault", "path"):
                        self.raw_vault_path = v
                        self.cfg["vault"]["path"] = resolve_vault_path(v, self.config_path)
                    else:
                        self.cfg[s][k] = v
                    self._unsaved.setdefault(s, {})[k] = v
            effects = sorted({settings.BY_ID[f"{s}.{k}"].effect for s, vals in diff.items() for k in vals})
            notes = self._apply_effects(diff)
            saved = None
            if save and self._unsaved:
                saved = str(settings.save_changes(self._save_target(), self._unsaved,
                                                  template=HERE / "config.example.toml"))
                self._unsaved = {}
                log.info("settings saved to %s", saved)
        changed = [f"{s}.{k}" for s, vals in diff.items() for k in vals]
        if changed:
            log.info("settings changed: %s", ", ".join(f"{c}={diff[c.split('.')[0]][c.split('.')[1]]!r}"
                                                         for c in changed))
        return {"ok": True, "changed": changed, "effects": effects, "saved": saved, "notes": notes,
                "unsaved": sorted(f"{s}.{k}" for s, vals in self._unsaved.items() for k in vals)}

    def _apply_effects(self, diff: dict) -> list[str]:
        notes: list[str] = []
        restart_claude = False

        c = diff.get("claude", {})
        for k, v in c.items():
            setattr(self.claude_cfg, k, v)  # the engine shares this object
            if settings.BY_ID[f"claude.{k}"].effect == "claude":
                restart_claude = True
        if {"auto_answer", "model"} & c.keys():
            self.hub.publish("info", {"auto_answer": self.claude_cfg.auto_answer, "model": self.claude_cfg.model})

        v = diff.get("vault", {})
        if {"path", "exclude", "embedding_model"} & v.keys():
            self.vault = self._make_vault()
            self.engine.vault = self.vault
            self.reindex()
            restart_claude = True  # its readable folder and deny rules come from the vault
        else:
            if "top_k" in v:
                self.vault.top_k = v["top_k"]
            if "per_note" in v:
                self.vault.per_note = v["per_note"]
        name = self.cfg["vault"].get("name") or self.vault.name
        if name != self.vault_name:
            self.vault_name = self.engine.vault_name = name
            self.hub.publish("info", {"vault_name": name})

        s = diff.get("stt", {})
        for k, val in s.items():
            setattr(self.stt_cfg, k, val)  # language / initial_prompt are read for every utterance
        if any(settings.BY_ID[f"stt.{k}"].effect == "stt" for k in s):
            self._stt_reload.set()

        a = diff.get("audio", {})
        if "device" in a:
            if self.args.simulate:
                notes.append("The audio device is not used while simulating a WAV file.")
            else:
                threading.Thread(target=self._restart_audio, name="audio-restart", daemon=True).start()
        elif a and self.pipeline is not None:
            self.pipeline.update_vad(audio.AudioConfig.from_dict(self.cfg["audio"]))

        if diff.get("ui"):
            notes.append("Web UI host / port change after restarting meeting-copilot.")
        if restart_claude:
            if self.args.dry_run:
                notes.append("Claude is off in dry-run mode; the new Claude settings apply on the next normal start.")
            else:
                self.engine.restart()
        return notes

    # -- API ---------------------------------------------------------------- #
    def ask(self, utt_id, text) -> tuple[bool, str]:
        if text is not None and str(text).strip():
            self.engine.submit(AskRequest(text=str(text).strip()[:2000], manual=True))
            return True, "queued"
        if utt_id is not None:
            try:
                utt_id = int(utt_id)
            except (TypeError, ValueError):
                return False, "utt_id must be an integer"
            if self.transcript.get(utt_id) is None:
                return False, f"no utterance {utt_id}"
            self.engine.submit(AskRequest(utt_id=utt_id, manual=True))
            return True, "queued"
        last = self.transcript.last()
        if last is None:
            return False, "nothing has been said yet"
        self.engine.submit(AskRequest(utt_id=last.id, manual=True))
        return True, "queued"

    def reindex(self) -> bool:
        if self.vault.building:
            return False
        threading.Thread(target=self._build_index, name="index", daemon=True).start()
        return True

    def _build_index(self) -> None:
        v = self.vault
        if not v.root:
            self.status(index_chunks=0, index_files=0, indexing=False, vectors=False,
                        index_error="no [vault].path configured", index_progress=None)
            return
        self.status(indexing=True, index_error=None)

        def progress(msg: str) -> None:
            if v is self.vault:  # a newer vault replaced this one: stay quiet
                self.status(index_progress=msg, **self._index_stats())

        try:
            v.build(progress=progress)
            if v is self.vault:
                st = v.stats()
                self.status(indexing=False, index_chunks=st["chunks"], index_files=st["files"],
                            vectors=st["vectors"], index_error=None, index_progress=None)
        except Exception as e:  # noqa: BLE001
            log.error("indexing failed: %s", e)
            if v is self.vault:
                self.status(indexing=False, index_error=str(e))
                self.hub.publish("error", {"message": f"indexing failed: {e}"})

    def _index_stats(self) -> dict:
        st = self.vault.stats()
        return {"index_chunks": st["chunks"], "index_files": st["files"], "vectors": st["vectors"]}

    # -- threads ------------------------------------------------------------ #
    def _load_transcriber(self) -> Optional["stt.Transcriber"]:
        self.status(stt_state="loading", stt_model=self.stt_cfg.model)
        try:
            tr = stt.create_transcriber(self.stt_cfg)
            tr.warmup()
        except Exception as e:  # noqa: BLE001
            log.exception("could not load speech model")
            self.status(stt_state="error")
            self.hub.publish("error", {"message": f"speech model failed to load: {e}"})
            return None
        self.status(stt_state="ready", stt_model=self.stt_cfg.model, stt_device=getattr(tr, "device", ""),
                    stt_backend=tr.name)
        return tr

    def _stt_thread(self) -> None:
        tr = self._load_transcriber()
        if tr is None:
            self.status(recording=False)
            return
        self._start_audio()
        while not self.stopping.is_set():
            if self._stt_reload.is_set():
                self._stt_reload.clear()
                log.info("reloading speech model: %s on %s (%s)", self.stt_cfg.model, self.stt_cfg.device,
                         self.stt_cfg.compute_type)
                new = self._load_transcriber()
                if new is not None:
                    tr = new
                else:
                    self.status(stt_state="ready")  # keep using the previous model
            try:
                seg = self.seg_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text, ms = stt.transcribe_segment(tr, seg.audio)
            except Exception as e:  # noqa: BLE001
                log.exception("transcription failed")
                self.hub.publish("error", {"message": f"transcription failed: {e}"})
                continue
            lag_ms = int((time.time() - seg.wall_end) * 1000) if seg.wall_end else ms
            if not text:
                log.debug("dropped empty/hallucinated segment (%.1fs)", seg.duration)
                continue
            u = self.transcript.add(seg.t_start, text, t_end=seg.t_end, wall_end=seg.wall_end, stt_ms=ms)
            log.info("[%s] %s  (stt %d ms, lag %d ms)", _ts(u.t), text, ms, lag_ms)
            self.status(stt_ms=ms, stt_lag_ms=lag_ms)
            self.utt_q.put(u)

    def _detector_thread(self) -> None:
        while not self.stopping.is_set():
            try:
                u = self.utt_q.get(timeout=0.5)
            except queue.Empty:
                continue
            u.is_question = detector.is_question(u.text)
            self.hub.publish("utt", u.to_dict())
            if u.is_question and self.claude_cfg.auto_answer:
                self.engine.submit(AskRequest(utt_id=u.id))

    def _start_audio(self, time_offset: float = 0.0) -> None:
        acfg = audio.AudioConfig.from_dict(self.cfg["audio"])
        pipe = audio.AudioPipeline(acfg, self.seg_q, simulate=self.args.simulate,
                                   on_level=self._on_level,
                                   on_source=lambda name: self.status(source=name),
                                   time_offset=time_offset)
        self.pipeline = pipe
        pipe.start()

        def watch():
            # report the source once it is open, and errors / end of simulation
            for _ in range(50):
                if pipe.source_name or pipe.finished.is_set():
                    break
                time.sleep(0.1)
            if pipe is self.pipeline:
                self.status(recording=not pipe.finished.is_set(), source=pipe.source_name)
            pipe.finished.wait()
            if pipe is self.pipeline:  # not when it was replaced by a reopened device
                self.status(recording=False, level=0, speaking=False)
            if pipe.error:
                self.hub.publish("error", {"message": pipe.error})

        threading.Thread(target=watch, name="audio-watch", daemon=True).start()

    def _restart_audio(self) -> None:
        old = self.pipeline
        if old is None:  # not started yet (speech model still loading): it will use the new cfg
            return
        old.stop()
        old.finished.wait(5)
        log.info("reopening audio: %s", self.cfg["audio"]["device"])
        self._start_audio(time_offset=old.stream_time)

    def _on_level(self, level: float, speaking: bool) -> None:
        self.hub.publish("status", {"level": round(level, 4), "speaking": speaking})

    # -- lifecycle ---------------------------------------------------------- #
    def run(self) -> None:
        ui = self.cfg["ui"]
        host, port = ui.get("host", "127.0.0.1"), int(self.args.port or ui.get("port", 8765))
        self.hub.info = {
            "vault_name": self.vault_name,
            "dry_run": self.args.dry_run,
            "simulate": self.args.simulate,
            "auto_answer": self.claude_cfg.auto_answer,
            "model": self.claude_cfg.model,
            "config_path": str(self._save_target()),
        }
        self.hub.status.update({"recording": False, "dry_run": self.args.dry_run,
                                "claude_state": "dry-run" if self.args.dry_run else "starting"})
        self.httpd = Server((host, port), make_handler(self))
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()
        url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
        log.info("UI: %s", url)

        self.reindex()
        self.engine.start()
        threading.Thread(target=self._detector_thread, name="detector", daemon=True).start()
        threading.Thread(target=self._stt_thread, name="stt", daemon=True).start()
        if self.args.open:
            webbrowser.open(url)
        try:
            while not self.stopping.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("stopping…")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.stopping.set()
        if self.pipeline:
            self.pipeline.stop()
        self.engine.stop()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def _ts(t: float) -> str:
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local real-time meeting Q&A assistant")
    p.add_argument("--config", help="path to config.toml (default: ./config.toml)")
    p.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    p.add_argument("--simulate", metavar="FILE.wav", help="feed a WAV file at real-time speed instead of a device")
    p.add_argument("--dry-run", action="store_true", help="do not call Claude; only show retrieval results")
    p.add_argument("--port", type=int, help="override [ui].port")
    p.add_argument("--open", action="store_true", help="open the UI in the default browser")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3",
                  "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_devices:
        print(audio.list_devices())
        return 0
    cfg, path = load_config(args.config)
    if path:
        log.info("config: %s", path)
    if args.simulate:
        if not os.path.isfile(args.simulate):
            print(f"file not found: {args.simulate}", file=sys.stderr)
            return 2
    elif not str(cfg["audio"].get("device", "")).strip():
        print("[audio].device is empty. Pick a device and put part of its name in the config.\n")
        print(audio.list_devices())
        return 0
    App(cfg, args, config_path=path).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
