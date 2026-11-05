"""meeting-copilot: live meeting transcription + answer suggestions from your Obsidian vault.

    python copilot.py --config config.toml
    python copilot.py --list-devices
    python copilot.py --simulate meeting.wav --dry-run
"""
from __future__ import annotations

import argparse
import copy
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
            if vpath:
                vp = Path(os.path.expanduser(vpath))
                if not vp.is_absolute():
                    vp = (p.resolve().parent / vp).resolve()
                cfg["vault"]["path"] = str(vp)
            return cfg, p
    if path:
        raise SystemExit(f"config file not found: {path}")
    return copy.deepcopy(DEFAULTS), None


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
    def __init__(self, cfg: dict, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.stopping = threading.Event()
        self.hub = EventHub()
        self.transcript = stt.Transcript()
        vcfg = cfg["vault"]
        self.vault = Vault(
            vcfg.get("path", ""),
            exclude=vcfg.get("exclude", []),
            embedding_model=vcfg.get("embedding_model", ""),
            top_k=int(vcfg.get("top_k", 6)),
            per_note=int(vcfg.get("per_note", 2)),
        )
        self.vault_name = vcfg.get("name") or self.vault.name
        self.claude_cfg = ClaudeConfig.from_dict(cfg["claude"])
        self.engine = AnswerEngine(self.claude_cfg, self.vault, self.transcript, self.hub.publish,
                                   dry_run=args.dry_run, vault_name=self.vault_name)
        self.seg_q: "queue.Queue[audio.Segment]" = queue.Queue()
        self.utt_q: "queue.Queue[stt.Utterance]" = queue.Queue()
        self.pipeline: Optional[audio.AudioPipeline] = None
        self.httpd: Optional[Server] = None

    def status(self, **kw) -> None:
        self.hub.publish("status", kw)

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
        if not self.vault.root:
            self.status(index_chunks=0, index_files=0, indexing=False, index_error="no [vault].path configured")
            return
        self.status(indexing=True)
        try:
            self.vault.build(progress=lambda msg: self.status(index_progress=msg, **self._index_stats()))
            st = self.vault.stats()
            self.status(indexing=False, index_chunks=st["chunks"], index_files=st["files"],
                        vectors=st["vectors"], index_error=None, index_progress=None)
        except Exception as e:  # noqa: BLE001
            log.error("indexing failed: %s", e)
            self.status(indexing=False, index_error=str(e))
            self.hub.publish("error", {"message": f"indexing failed: {e}"})

    def _index_stats(self) -> dict:
        st = self.vault.stats()
        return {"index_chunks": st["chunks"], "index_files": st["files"], "vectors": st["vectors"]}

    # -- threads ------------------------------------------------------------ #
    def _stt_thread(self) -> None:
        scfg = stt.STTConfig.from_dict(self.cfg["stt"])
        self.status(stt_state="loading", stt_model=scfg.model)
        try:
            tr = stt.create_transcriber(scfg)
            tr.warmup()
        except Exception as e:  # noqa: BLE001
            log.exception("could not load speech model")
            self.status(stt_state="error", recording=False)
            self.hub.publish("error", {"message": f"speech model failed to load: {e}"})
            return
        self.status(stt_state="ready", stt_device=getattr(tr, "device", ""), stt_backend=tr.name)
        self._start_audio()
        while not self.stopping.is_set():
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

    def _start_audio(self) -> None:
        acfg = audio.AudioConfig.from_dict(self.cfg["audio"])
        self.pipeline = audio.AudioPipeline(acfg, self.seg_q, simulate=self.args.simulate,
                                            on_level=self._on_level,
                                            on_source=lambda name: self.status(source=name))
        self.pipeline.start()

        def watch():
            # report the source once it is open, and errors / end of simulation
            for _ in range(50):
                if self.pipeline.source_name or self.pipeline.finished.is_set():
                    break
                time.sleep(0.1)
            self.status(recording=not self.pipeline.finished.is_set(), source=self.pipeline.source_name)
            self.pipeline.finished.wait()
            self.status(recording=False, level=0, speaking=False)
            if self.pipeline.error:
                self.hub.publish("error", {"message": self.pipeline.error})

        threading.Thread(target=watch, name="audio-watch", daemon=True).start()

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
    App(cfg, args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
