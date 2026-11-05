# meeting-copilot

A local, real-time meeting Q&A assistant. During a Teams / Zoom / Meet call it:

1. captures the **system audio** (the other people's voices, not your microphone),
2. transcribes it **locally** with Whisper,
3. detects when someone asks a question,
4. searches your **Obsidian vault** for relevant notes,
5. asks your local **Claude Code CLI** (your Claude subscription, no API key) for an answer you can say out loud,
6. streams the transcript and answer cards to a browser window on your second screen.

```
 system audio ─▶ audio.py ─▶ stt.py ─▶ detector.py ─▶ answer.py ─▶ claude (persistent process)
 (BlackHole /    16 kHz,     Whisper   question?      vault.py       │ stream-json
  VB-Cable)      VAD cuts    (local)                  BM25+vectors   ▼
                                                       copilot.py ── SSE ──▶ ui.html (second screen)
```

Each stage runs in its own thread, connected by queues.

---

## 1. Install

### Claude Code (required)

1. Install Claude Code: <https://docs.claude.com/en/docs/claude-code/setup>.
2. Run `claude` once in a terminal and log in with your **Claude subscription account** (`/login`).
   Check the login with:
   ```bash
   claude auth status
   ```
   It must say `"loggedIn": true`.
3. Answers count against your subscription usage limits, not API credits. meeting-copilot removes
   `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` from the environment of the `claude` child process, so
   a stray API key can't switch you to API billing.

### Python

Python 3.11+ is required (`tomllib`).

```bash
pip install -r requirements.txt
```

On Windows / Linux this installs `faster-whisper`; on Apple Silicon it installs `mlx-whisper`.
`sentence-transformers` is optional (semantic search). Set `[vault].embedding_model = ""` to skip it.

**NVIDIA GPU on Windows.** faster-whisper (CTranslate2) needs the CUDA 12 cuBLAS and cuDNN 9 DLLs.
If PyTorch with CUDA is installed in the same environment, meeting-copilot finds its DLLs automatically.
Otherwise run `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.

---

## 2. Route the meeting audio

meeting-copilot only listens to **system audio**, never your own microphone. You need a loopback device.

### macOS

1. Install [BlackHole 2ch](https://existential.audio/blackhole/).
2. Open **Audio MIDI Setup**, click **+** and choose **Create Multi-Output Device**. Tick your speakers/headphones
   **and** BlackHole 2ch.
3. Set the system output (or the meeting app's speaker) to that Multi-Output Device. You still hear the
   call, and BlackHole gets a copy.
4. In `config.toml`:
   ```toml
   [audio]
   device = "BlackHole"
   [stt]
   backend = "mlx"
   model = "mlx-community/whisper-large-v3-turbo"
   ```

### Windows

**Recommended: WASAPI loopback**, no extra software:

```toml
[audio]
device = "loopback"          # or "loopback:JBL" to pin one output device
```

This records whatever the **default output device** is playing: headphones, a USB headset, speakers,
or *Remote Audio* when you work over Remote Desktop. It needs `PyAudioWPatch` (in `requirements.txt`).

- When nothing plays, Windows delivers no audio at all, so meeting-copilot fills those gaps with silence.
  That lets the end of an utterance be detected and keeps the timestamps right.
- While the device is silent, it rechecks the default output every 5 s, so switching to a headset
  mid-meeting is picked up.
- The capture buffer is 2 s, so a busy moment elsewhere in the app (for example loading the embedding
  model) doesn't drop audio. If a dropout does happen, the console shows `loopback: … s of audio lost`.

Alternatives:

- **VB-Audio Virtual Cable** (<https://vb-audio.com/Cable/>). Set the meeting app's speaker to
  *CABLE Input*. To keep hearing the call, open *Sound settings → Recording → CABLE Output → Properties →
  Listen* and tick "Listen to this device" with your headphones. Use `device = "CABLE Output"`.
- **Stereo Mix** (some Realtek drivers). Enable it in *Sound settings → Recording* (show disabled devices),
  then use `device = "Stereo Mix"`. It only hears the Realtek card, though: not a USB/Bluetooth headset,
  and nothing at all over Remote Desktop.

Speech model:

- NVIDIA GPU: `model = "large-v3-turbo"`, `device = "cuda"` (or `"auto"`). Use `compute_type = "int8"`
  for Pascal cards such as the GTX 10xx and Quadro P-series. On RTX cards, `"int8_float16"` or
  `"float16"` is faster.
- CPU only: `model = "small"`, `device = "cpu"`, `compute_type = "int8"`.

List the input devices (any OS):

```bash
python copilot.py --list-devices
```

---

## 3. Configure

```bash
cp config.example.toml config.toml
```

Settings you'll usually change:

| key | meaning |
|---|---|
| `[audio] device` | `"loopback"` (Windows), or part of an input device name; empty → list devices and exit |
| `[stt] model`, `language`, `initial_prompt` | Whisper model, `""`/`"en"`/`"zh"`, domain terms to help spelling |
| `[vault] path` | your Obsidian vault folder (relative paths are relative to the config file) |
| `[vault] exclude` | globs to skip, e.g. `["NDA/**", "Private/**", "*.excalidraw.md"]` |
| `[vault] embedding_model` | `""` = BM25 only; default multilingual MiniLM; `"BAAI/bge-m3"` is better but slower |
| `[claude] model` | `haiku` (fastest) or `sonnet` |
| `[claude] about_me` | your role, so answers sound like you |
| `[claude] auto_answer` | `false` = only answer when you click / press Space |
| `[claude] allow_vault_tools` | let Claude `Grep`/`Read` the vault itself when the retrieved notes aren't enough |
| `[claude] thinking`, `effort` | default `false` / `"low"`; extended thinking makes Haiku spend 5–10 s before the first word |

---

## 4. Run

```bash
python copilot.py --config config.toml --open
```

Open <http://127.0.0.1:8765/> on your second screen (`--open` does that for you).

| flag | |
|---|---|
| `--config FILE` | config file (default `./config.toml`, falls back to `config.example.toml`) |
| `--list-devices` | list audio input devices and exit |
| `--simulate file.wav` | play a WAV file through the pipeline at real-time speed instead of a device |
| `--dry-run` | don't call Claude; only show transcripts, detected questions and retrieved notes |
| `--port N`, `--open`, `-v` | port override, open the browser, debug logging |

### The UI

- **Left (30%)**: live transcript. Detected questions are highlighted. **Click a line** to get an answer to it.
  Use the box at the bottom to type a question yourself.
- **Right**: answer cards, newest on top. Each card shows the question, the streamed answer and its sources.
  The `[n]` citations and source titles are `obsidian://open?vault=…&file=…` links.
- **Space**: answer the latest utterance. **Rebuild index**: re-scan the vault after you edit notes.
- **⚙ Settings**: change the configuration while it runs. Each field shows when a change takes effect:
  | badge | settings |
  |---|---|
  | instant | auto-answer, about me, transcript context, notes per question, sections per note, vault name |
  | next utterance | speech threshold, end-of-speech silence, segment lengths, pre-roll, Whisper language, names and jargon |
  | restarts Claude (~2 s) | model, effort, extended thinking, vault tools, process mode, timeouts, CLI path |
  | re-indexes notes | notes folder, exclude globs, embedding model (Claude is restarted too, for its folder and deny rules) |
  | reloads speech model (~10 s) | Whisper model, backend, device, precision; transcription continues with the old model until the new one is ready |
  | reopens audio | audio source; transcript timestamps continue |
  | after restarting the app | web UI host and port |

  **Save to file** (on by default) writes the changes into the config file that was loaded. Only the edited lines
  change; comments and everything else stay as they were. If meeting-copilot was started with `config.example.toml`,
  the changes go to `config.toml` instead, so the template stays clean. Untick it to try a change for this session only.
  Invalid values are rejected and nothing is changed.
- **Status bar**: recording state and input level, Whisper time per segment, Claude time to first token,
  time from end of speech to first answer text, and the number of indexed chunks.

Manual requests (click, Space, typed question) are never `SKIP`ped. Automatic ones are skipped when Claude
decides the line isn't really a question for you (small talk, rhetorical questions, and so on). Skipped
cards disappear.

### How Claude is called

At start-up a single long-lived process is launched:

```
claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages
       --model haiku --append-system-prompt "<assistant rules>"
       --allowedTools Read,Grep,Glob --add-dir <vault> --tools Read,Grep,Glob
       --disallowedTools "Read(./NDA/**)" ...        # one deny rule per [vault].exclude glob
       --no-session-persistence --strict-mcp-config --disable-slash-commands
       --effort low --settings '{"alwaysThinkingEnabled": false}'   # + MAX_THINKING_TOKENS=0
```

- Each question is one JSON `user` message on its stdin. Text deltas from stdout are forwarded to the UI as they
  arrive. The first 6 characters are held back, and an answer that starts with `SKIP` is dropped.
- Questions that arrive while Claude is still answering are merged into a single request.
- A short warm-up turn at start-up checks the login and primes the prompt cache.
- If the process dies it is restarted. After `session_max_minutes` (30 min) it's replaced by a fresh session,
  because a growing conversation gets slower.
- On Windows, the npm `claude.cmd` shim is bypassed and the bundled `claude.exe` is run directly, so the
  multi-line system prompt survives intact.
- **Fallback**: if the installed CLI has no `--input-format stream-json`, or the persistent process keeps
  dying (`mode = "auto"`), every question runs its own `claude -p --output-format stream-json --verbose`
  process with the prompt on stdin. That works, but **each answer then pays the CLI start-up time**:
  measured first token 2.0–2.3 s instead of 0.5–1.2 s, so about +1.5 s per answer. You can force either
  mode with `[claude] mode`.

---

## 5. Privacy

- Audio and transcription never leave your machine. Whisper runs locally.
- Claude receives only the recent transcript lines (`context_minutes`), the detected question, and the
  retrieved note sections (`top_k`, at most `per_note` sections per note).
- Use `[vault] exclude` for sensitive folders. Excluded notes are never indexed, and every exclude glob is
  also passed to `claude` as a deny rule (`--disallowedTools "Read(./NDA/**)"`), so Claude's own
  Read/Grep/Glob can't open them either. Tested: asked to read `NDA/Acme Contract.md`, Claude was denied
  and answered that it had no access. For extra safety, set `allow_vault_tools = false`: Claude then has no
  file tools at all and sees only the retrieved sections.
- The web UI binds to `127.0.0.1`, rejects other `Host` headers and only accepts JSON POSTs.
- **Tell the other participants** that you are transcribing the meeting, and follow your company's rules and
  local law on recording.

---

## 6. Tests and the acceptance checks

```bash
python -m pytest -q
```

The tests cover retrieval against the fake vault in `tests/fixtures/vault` (including excluded and dot folders),
English and Chinese question detection, VAD segmentation and resampling, the hallucination filter, stream-json
parsing (`SKIP`, tool calls, errors), and the persistent / one-shot runners. The runners are tested against a
fake `claude` (`tests/fake_claude.py`), so the tests need no login or network.

End-to-end with a generated meeting recording:

```bash
python tests/make_demo_wav.py demo_meeting.wav
```

Dry run (no Claude): transcripts, highlighted questions and retrieved sources:

```bash
python copilot.py --config tests/demo_config.toml --simulate demo_meeting.wav --dry-run --open
```

Live answers through Claude:

```bash
python copilot.py --config tests/demo_config.toml --simulate demo_meeting.wav --open
```

`make_demo_wav.py` uses the OS text-to-speech: System.Speech on Windows, `say` on macOS, `espeak-ng` on Linux.
The demo contains five questions and one piece of small talk ("how was your weekend?") that should be `SKIP`ped.

### Measured latency

Measured with the demo recording on Windows 11, Quadro P4000 (8 GB, Pascal), faster-whisper `large-v3-turbo` /
`int8` on CUDA, Claude Code 2.1.272 with `model = "haiku"` in persistent mode:

| stage | time |
|---|---|
| end-of-speech detection (`end_silence_sec`) | 0.8 s |
| Whisper per utterance | 0.6–0.8 s |
| end of speech → transcript in the UI | ≈ 1.45 s |
| Claude: request → first answer token | 0.46–1.16 s |
| Claude: full answer | 1.1–1.8 s |
| **end of speech → first answer text on screen** | **1.9–2.6 s** (goal: < 8 s) |
| `SKIP` for small talk ("how was your weekend?") | 0.5 s, card removed |

With extended thinking on (`thinking = true`, `effort = ""`), Haiku first spent 1,300–1,900 thinking tokens
and the first token took 7.6–13 s. That's why thinking is off by default.

The status bar and every answer card show these numbers live.

---

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `Failed to authenticate` / `loggedIn: false` | run `claude` in a terminal and `/login` with your subscription account |
| status bar says "Speech model error" | see the console; on Windows + CUDA install `nvidia-cublas-cu12 nvidia-cudnn-cu12`, or set `[stt] device = "cpu"` |
| nothing gets transcribed | check the input level meter; raise the meeting volume or lower `silence_threshold`. With Stereo Mix, the sound must play through the Realtek card (not a USB headset, not Remote Desktop). Use `device = "loopback"` instead |
| words missing mid-sentence | look for `loopback: … s of audio lost` in the console; something stalled the capture thread for > 2 s |
| words cut in half | raise `end_silence_sec` (e.g. 1.0) |
| answers are slow | keep `model = "haiku"`, lower `top_k`, set `allow_vault_tools = false` |
| "Thank you." / subtitle credits appear | Whisper hallucinations on noise are filtered in `stt.py` (`_EXACT_HALLUCINATIONS`); add your own |

## Project layout

```
copilot.py           entry point: CLI flags, threads, HTTP/SSE server
audio.py             capture, resample to 16 kHz, silence-based segmentation
stt.py               Whisper backends (faster-whisper / mlx) + hallucination filter
detector.py          rule-based question detection (English + Chinese)
vault.py             Obsidian indexing: BM25 + optional embeddings, RRF fusion
settings.py          settings editable from the UI: schema, validation, comment-preserving config.toml writer
answer.py            persistent claude process, prompt building, stream parsing, SKIP
ui.html              single-file UI
config.example.toml  configuration template
tests/               pytest suite, fake vault, fake claude, demo WAV generator
```
