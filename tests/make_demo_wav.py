"""Generate a fake meeting recording (questions + statements) with the OS's TTS.

    python tests/make_demo_wav.py demo_meeting.wav

Windows: System.Speech (built in). macOS: `say`. Linux: `espeak-ng`.
The sentences match the notes in tests/fixtures/vault, so
    python copilot.py --config tests/demo_config.toml --simulate demo_meeting.wav --dry-run
shows retrieval results for each question.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio import TARGET_RATE, Resampler, read_wav, to_mono, write_wav  # noqa: E402

LINES = [
    "Good morning everyone, thanks for joining the weekly sync.",
    "So, what is the current timeline for the Apollo migration?",
    "We had some disk issues on the staging cluster last week, but that is fixed now.",
    "How much budget do we have left for the migration project?",
    "Can you walk me through the main risks before the cut-over?",
    "By the way, how was your weekend?",
    "Okay, let's move on to the search service.",
    "What is the p95 latency of the search service right now?",
]
GAP_SEC = 2.5


def synth_windows(text: str, out: str) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate = 0;"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak([Console]::In.ReadToEnd());"
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text.encode("utf-8"), check=True)


def synth_mac(text: str, out: str) -> None:
    subprocess.run(["say", "--data-format=LEI16@16000", "-o", out, text], check=True)


def synth_linux(text: str, out: str) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise SystemExit("install espeak-ng to generate the demo audio")
    subprocess.run([exe, "-w", out, text], check=True)


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "demo_meeting.wav"
    synth = {"win32": synth_windows, "darwin": synth_mac}.get(sys.platform, synth_linux)
    parts = [np.zeros(int(TARGET_RATE * 1.0), dtype=np.float32)]
    with tempfile.TemporaryDirectory() as td:
        for i, line in enumerate(LINES):
            p = os.path.join(td, f"{i}.wav")
            synth(line, p)
            rate, data = read_wav(p)
            mono = Resampler(rate, TARGET_RATE).process(to_mono(data))
            peak = float(np.max(np.abs(mono))) or 1.0
            parts.append((mono / peak * 0.5).astype(np.float32))
            parts.append(np.zeros(int(TARGET_RATE * GAP_SEC), dtype=np.float32))
    audio = np.concatenate(parts)
    write_wav(out_path, audio)
    print(f"wrote {out_path} ({len(audio) / TARGET_RATE:.1f} s, {len(LINES)} lines)")


if __name__ == "__main__":
    main()
