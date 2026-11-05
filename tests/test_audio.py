import queue

import numpy as np
import pytest

from audio import (TARGET_RATE, AudioConfig, AudioPipeline, Resampler, Segmenter, read_wav, to_mono,
                   write_wav)

SR = TARGET_RATE


def tone(sec, amp=0.2, freq=300.0, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(sec, sr=SR):
    return np.zeros(int(sec * sr), dtype=np.float32)


def run(seg: Segmenter, audio, block=0.05):
    out = []
    n = int(block * SR)
    for i in range(0, len(audio), n):
        out += seg.feed(audio[i:i + n])
    return out + seg.flush()


def test_two_utterances_split_on_silence():
    audio = np.concatenate([silence(1.0), tone(1.5), silence(1.2), tone(2.0), silence(1.0)])
    segs = run(Segmenter(), audio)
    assert len(segs) == 2
    assert segs[0].t_end == pytest.approx(2.5, abs=0.05)
    assert segs[1].t_end == pytest.approx(5.7, abs=0.05)


def test_preroll_is_kept():
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.0)])
    seg = run(Segmenter(preroll_sec=0.3), audio)[0]
    assert seg.t_start == pytest.approx(0.7, abs=0.035)
    # 0.3 s pre-roll + 1.0 s speech + 0.2 s kept tail
    assert seg.duration == pytest.approx(1.5, abs=0.07)


def test_short_pause_does_not_split():
    audio = np.concatenate([tone(1.0), silence(0.5), tone(1.0), silence(1.0)])
    assert len(run(Segmenter(end_silence_sec=0.8), audio)) == 1


def test_short_blip_is_dropped():
    audio = np.concatenate([silence(0.5), tone(0.3), silence(1.5), tone(0.9), silence(1.0)])
    segs = run(Segmenter(min_segment_sec=0.6), audio)
    assert len(segs) == 1
    assert segs[0].t_end == pytest.approx(3.2, abs=0.05)


def test_long_speech_is_cut_at_max_segment():
    audio = np.concatenate([tone(45.0), silence(1.0)])
    segs = run(Segmenter(max_segment_sec=20), audio)
    assert len(segs) == 3
    assert all(s.duration <= 20.25 for s in segs)
    total = sum(s.t_end - s.t_start for s in segs)
    assert total == pytest.approx(45.0, abs=0.2)


def test_quiet_noise_is_ignored():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 5) * 0.002).astype(np.float32)
    assert run(Segmenter(silence_threshold=0.01), noise) == []


@pytest.mark.parametrize("src", [44100, 48000, 16000, 8000])
def test_resampler_length_and_pitch(src):
    sec, freq = 2.0, 440.0
    x = tone(sec, freq=freq, sr=src)
    r = Resampler(src, SR)
    out = np.concatenate([r.process(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    assert abs(len(out) - sec * SR) <= 2 + 64  # filter delay tail
    spec = np.abs(np.fft.rfft(out[2000:2000 + SR]))
    peak_hz = np.argmax(spec) * SR / SR
    assert peak_hz == pytest.approx(freq, abs=2)


def test_resampler_removes_content_above_nyquist():
    x = tone(1.0, freq=12000, sr=48000)  # above the 8 kHz Nyquist of 16 kHz
    out = Resampler(48000, SR).process(x)
    assert np.sqrt(np.mean(out[200:] ** 2)) < 0.01


def test_to_mono():
    st = np.stack([np.ones(10), -np.ones(10)], axis=1).astype(np.float32)
    assert np.allclose(to_mono(st), 0)
    assert to_mono(np.ones((5, 1))).shape == (5,)


def test_wav_roundtrip_and_simulated_pipeline(tmp_path):
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(1.2), silence(0.3)])
    # write at 48 kHz stereo-ish via mono file at 16 kHz, then simulate at max speed
    p = tmp_path / "x.wav"
    write_wav(str(p), audio)
    rate, data = read_wav(str(p))
    assert rate == SR and data.shape[1] == 1
    assert np.allclose(data[:, 0], audio, atol=1e-4)

    q: queue.Queue = queue.Queue()
    pipe = AudioPipeline(AudioConfig(), q, simulate=str(p), realtime=False)
    pipe.start()
    assert pipe.finished.wait(10)
    segs = []
    while not q.empty():
        segs.append(q.get())
    assert len(segs) == 2
    assert all(s.wall_end <= s.wall_ready for s in segs)


# --------------------------------------------------------------------------- #
# WASAPI loopback, with a fake PyAudioWPatch and a fake clock
# --------------------------------------------------------------------------- #
import types  # noqa: E402

import audio  # noqa: E402
from audio import parse_loopback  # noqa: E402


class FakeClock:
    def __init__(self, stop_at, stop_event=None):
        self.t = 0.0
        self.stop_at = stop_at
        self.stop_event = stop_event

    def monotonic(self):
        return self.t

    def time(self):
        return 1_000_000.0 + self.t

    def sleep(self, dt):
        self.t += dt
        if self.t >= self.stop_at and self.stop_event is not None:
            self.stop_event.set()


class FakeStream:
    """Like WASAPI loopback: frames are only available while something plays."""

    def __init__(self, clock, rate, channels, plays, stall=None):
        self.clock, self.rate, self.channels, self.plays = clock, rate, channels, plays
        self.read_until = clock.t
        self.closed = False
        self.stall = stall  # (at, duration, buffer_sec): the reader freezes, the buffer overflows

    def _available_sec(self):
        t = self.clock.t
        return sum(max(0.0, min(e, t) - max(s, self.read_until)) for s, e in self.plays)

    def get_read_available(self):
        if self.stall and self.clock.t >= self.stall[0]:
            at, dur, buffer_sec = self.stall
            self.stall = None
            self.clock.t += dur  # this thread was frozen; only the last buffer_sec survived
            self.read_until = max(self.read_until, self.clock.t - buffer_sec)
        return int(self._available_sec() * self.rate)

    def read(self, n, exception_on_overflow=True):
        self.read_until = self.clock.t
        x = (0.2 * np.sin(2 * np.pi * 300 * np.arange(n) / self.rate)).astype(np.float32)
        return np.repeat(x[:, None], self.channels, axis=1).tobytes()

    def close(self):
        self.closed = True


def fake_pa(clock, devices, plays, default_names, stall=None):
    """A stand-in for the pyaudiowpatch module. default_names[i] is the default
    output for the i-th PyAudio() instance (the last one repeats)."""
    state = {"opened": 0, "streams": []}

    class FakePyAudio:
        def __init__(self):
            self.n = state["opened"]
            state["opened"] += 1

        def get_loopback_device_info_generator(self):
            yield from devices

        def get_default_wasapi_loopback(self):
            name = default_names[min(self.n, len(default_names) - 1)]
            return next(d for d in devices if d["name"].startswith(name))

        def open(self, format, channels, rate, input, input_device_index, frames_per_buffer):
            s = FakeStream(clock, rate, channels, plays, stall)
            state["streams"].append(s)
            return s

        def terminate(self):
            pass

    return types.SimpleNamespace(PyAudio=FakePyAudio, paFloat32=1, state=state)


DEVICES = [
    {"index": 5, "name": "Remote Audio [Loopback]", "maxInputChannels": 2, "defaultSampleRate": 44100.0},
    {"index": 7, "name": "Speakers (JBL Quantum 400) [Loopback]", "maxInputChannels": 2, "defaultSampleRate": 48000.0},
]


def run_loopback(monkeypatch, device, plays, stop_at, default_names=("Remote Audio",), stall=None):
    q = queue.Queue()
    sources = []
    pipe = AudioPipeline(AudioConfig(device=device), q, on_source=sources.append)
    clock = FakeClock(stop_at, pipe.stop_event)
    pa = fake_pa(clock, DEVICES, plays, list(default_names), stall)
    monkeypatch.setattr(audio, "_pa", lambda: pa)
    monkeypatch.setattr(audio, "time", clock)
    pipe._run_device()  # runs in this thread until the fake clock reaches stop_at
    segs = []
    while not q.empty():
        segs.append(q.get())
    return pipe, segs, sources, pa


@pytest.mark.parametrize("spec,expected", [
    ("loopback", ""), ("Loopback", ""), ("loopback:JBL", "JBL"), (" loopback: Remote ", "Remote"),
    ("Speakers (Realtek) [Loopback]", "Speakers (Realtek) [Loopback]"),
    ("Stereo Mix", None), ("CABLE Output", None), ("", None),
])
def test_parse_loopback(spec, expected):
    assert parse_loopback(spec) == expected


def test_loopback_pads_silence_so_the_utterance_is_cut(monkeypatch):
    # audio plays 0.5-2.0 s; afterwards WASAPI delivers nothing at all
    pipe, segs, _, _ = run_loopback(monkeypatch, "loopback", plays=[(0.5, 2.0)], stop_at=4.0)
    assert pipe.error is None
    assert pipe.source_name == "Remote Audio [Loopback]"
    assert len(segs) == 1
    s = segs[0]
    assert s.t_end == pytest.approx(2.0, abs=0.1)  # stream time tracks wall-clock time
    # cut by the VAD ~0.8 s after the audio stopped, not by the final flush at 4.0 s
    assert s.wall_ready - 1_000_000.0 < 3.2


def test_loopback_two_utterances_with_idle_gap(monkeypatch):
    pipe, segs, _, _ = run_loopback(monkeypatch, "loopback:JBL", plays=[(0.2, 1.4), (3.0, 4.0)], stop_at=6.0)
    assert pipe.source_name.startswith("Speakers (JBL")
    # stream time stays in step with wall-clock time across the idle gap
    assert [s.t_end for s in segs] == pytest.approx([1.4, 4.0], abs=0.04)


def test_loopback_follows_a_new_default_output(monkeypatch):
    # nothing plays for > LOOPBACK_RECHECK_SEC, then the default output changes to the JBL headset
    pipe, _, sources, pa = run_loopback(monkeypatch, "loopback", plays=[], stop_at=12.0,
                                        default_names=("Remote Audio", "Speakers (JBL"))
    assert pa.state["opened"] >= 2
    assert all(s.closed for s in pa.state["streams"][:-1])
    assert sources == ["Speakers (JBL Quantum 400) [Loopback]"]
    assert pipe.source_name.startswith("Speakers (JBL")


def test_loopback_unknown_device_reports_error(monkeypatch):
    pipe, segs, _, _ = run_loopback(monkeypatch, "loopback:Nope", plays=[], stop_at=1.0)
    assert "No output device matches 'Nope'" in pipe.error
    assert "Remote Audio [Loopback]" in pipe.error
    assert pipe.finished.is_set() and segs == []


def test_loopback_dropout_keeps_time_and_is_counted(monkeypatch):
    # audio 0.5-3.0 s; at 1.5 s the reader stalls for 0.6 s and only 0.1 s survives in the buffer
    pipe, segs, _, _ = run_loopback(monkeypatch, "loopback", plays=[(0.5, 3.0)], stop_at=5.0,
                                    stall=(1.5, 0.6, 0.1))
    assert pipe.dropped_sec == pytest.approx(0.5, abs=0.05)
    assert len(segs) == 1
    assert segs[0].t_end == pytest.approx(3.0, abs=0.04)  # still aligned with wall-clock time
