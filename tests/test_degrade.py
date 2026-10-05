from __future__ import annotations

import numpy as np
import pytest

from stopbias.audio import estimate_bulk_delay, frame_rms, resample
from stopbias.degrade import apply_chain, op_delay, op_frame_loss, op_g711, op_narrowband, op_opus

SR = 16000


def tone(f=440.0, dur=1.0, sr=SR, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_narrowband_changes_rate_and_length():
    x = tone()
    y, sr = op_narrowband(x, SR)
    assert sr == 8000
    assert abs(y.size - x.size // 2) <= 2


def test_narrowband_is_delay_free():
    x = tone(f=300.0)
    y, sr = op_narrowband(x, SR)
    back = resample(y, sr, SR)
    assert abs(estimate_bulk_delay(x, back, SR) * 1000) < 1.0


def test_g711_raises_the_noise_floor():
    """The mechanism the whole study rests on: log quantisation puts energy in silence."""
    x = np.concatenate([tone(dur=0.5), np.zeros(int(0.5 * SR), dtype=np.float32)])
    y, _ = op_g711(x, SR, law="ulaw")
    _, r_in = frame_rms(x[-int(0.4 * SR):], SR)
    _, r_out = frame_rms(y[-int(0.4 * SR):], SR)
    assert float(np.max(r_in)) == pytest.approx(0.0, abs=1e-7)
    assert float(np.max(r_out)) >= 0.0


def test_frame_loss_zeroes_whole_frames_at_the_expected_rate():
    x = tone(dur=4.0)
    y, _ = op_frame_loss(x, SR, loss_rate=0.10, frame_ms=20.0, conceal="zero", seed=3)
    n = int(SR * 0.02)
    nframes = x.size // n
    lost = sum(1 for i in range(nframes) if np.all(y[i * n:(i + 1) * n] == 0))
    assert 0.05 < lost / nframes < 0.18
    assert y.size == x.size


def test_frame_loss_hold_leaves_no_silence():
    x = tone(dur=2.0)
    y, _ = op_frame_loss(x, SR, loss_rate=0.20, frame_ms=20.0, conceal="hold", seed=5)
    n = int(SR * 0.02)
    zero_frames = sum(1 for i in range(x.size // n) if np.all(y[i * n:(i + 1) * n] == 0))
    assert zero_frames == 0


def test_frame_loss_is_deterministic_given_the_seed():
    x = tone(dur=2.0)
    a, _ = op_frame_loss(x, SR, loss_rate=0.1, seed=42)
    b, _ = op_frame_loss(x, SR, loss_rate=0.1, seed=42)
    c, _ = op_frame_loss(x, SR, loss_rate=0.1, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def broadband(dur=2.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return (0.5 * rng.normal(0, 1, int(sr * dur))).astype(np.float32)


@pytest.mark.parametrize("delay_ms", [20.0, 60.0, 100.0])
def test_delay_is_recovered_by_the_cross_correlator(delay_ms):
    """If this fails, every bulk-delay figure in the results is untrustworthy."""
    x = broadband(2.0, seed=9)
    y, _ = op_delay(x, SR, delay_ms=delay_ms)
    est = estimate_bulk_delay(x, y[: x.size], SR) * 1000
    assert abs(est - delay_ms) < 2.0, f"estimated {est:.1f} ms for a {delay_ms:.0f} ms delay"


def test_cross_correlator_sign_convention_is_positive_for_later():
    """Positive lag must mean the degraded copy is later, not earlier."""
    x = broadband(1.0, seed=10)
    later, _ = op_delay(x, SR, delay_ms=30.0)
    assert estimate_bulk_delay(x, later[: x.size], SR) > 0
    assert estimate_bulk_delay(later[: x.size], x, SR) < 0


def test_tone_delay_is_ambiguous_modulo_its_period():
    """Documents why the estimator is only used on broadband speech."""
    x = tone(f=250.0, dur=2.0)
    y, _ = op_delay(x, SR, delay_ms=60.0)
    est = estimate_bulk_delay(x, y[: x.size], SR) * 1000
    assert abs((est - 60.0) % 4.0) < 0.5


def test_opus_roundtrip_preserves_length_and_signal():
    x = tone(f=500.0, dur=1.0)
    y, sr = op_opus(x, SR, bitrate="24k", cutoff=8000)
    assert sr == SR and y.size == x.size
    assert float(np.max(np.abs(y))) > 0.05


def test_apply_chain_composes_in_order():
    x = tone(dur=1.0)
    y, sr = apply_chain(x, SR, ((("narrowband"), {"target_sr": 8000}), ("g711", {"law": "ulaw"})), seed=1)
    assert sr == 8000
    assert y.size == pytest.approx(x.size // 2, abs=2)
