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
