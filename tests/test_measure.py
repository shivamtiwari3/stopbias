"""Harness validity: does the measurement chain recover ground truth it was handed?

These are the tests that make the results believable. If a synthetic trial with a known
60 ms buffer delay does not read back as 60 ms, no result about G.711 means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from stopbias.audio import energy_onset
from stopbias.conditions import BY_NAME, Cell
from stopbias.measure import Degraded, channel_seed, degrade_trial, measure
from stopbias.trials import SR, Trial, build
from stopbias.vad import DETECTORS, EnergyDetector, merge_segments


def speechlike(dur=1.0, sr=SR, seed=0):
    """Broadband amplitude-modulated noise: energy where speech has energy, no lexical content."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    car = rng.normal(0, 1, n)
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    x = (car * env).astype(np.float32)
    x /= max(float(np.max(np.abs(x))), 1e-9)
    return (0.7 * x).astype(np.float32)


def make_trial(onset=1.0, lat=0.5, tail=1.0):
    user = speechlike(1.2, seed=1)
    agent = speechlike(6.0, seed=2)
    cut = onset + lat
    trial = Trial("t000", "u", "a", SR, onset, cut, lat, cut + tail, tail)
    agent_ch, user_ch = build(user, agent, onset, cut, tail)
    return trial, agent_ch, user_ch


def test_build_places_boundaries_exactly():
    trial, agent_ch, user_ch = make_trial(onset=1.0, lat=0.5)
    ncut = int(round(trial.cut_s * SR))
    assert np.any(agent_ch[ncut - 100:ncut] != 0)
    assert np.all(agent_ch[ncut:] == 0)
    a = int(round(trial.onset_s * SR))
    assert np.all(user_ch[:a] == 0)
    assert np.any(user_ch[a:a + 100] != 0)
    assert agent_ch.size == user_ch.size


def test_energy_onset_finds_the_insertion_point():
    _trial, _agent, user_ch = make_trial(onset=1.0, lat=0.5)
    t0 = energy_onset(user_ch, SR)
    assert t0 is not None and abs(t0 - 1.0) < 0.01


# Framing sets the floor on how exact any of this can be: the energy detector uses a 20 ms
# window on a 5 ms hop, so a boundary is only ever located to within roughly one window.
FRAME_TOL_MS = 45.0


def test_reference_condition_recovers_true_stop_latency():
    """Synthetic stimulus, so the stimulus-agnostic detector is the one under test here.
    Silero is validated on real speech in test_corpus_integration.py."""
    trial, agent_ch, user_ch = make_trial(onset=1.2, lat=0.6)
    cond = BY_NAME["ref_16k"]
    deg = degrade_trial(trial, agent_ch, user_ch, cond)
    row = measure(trial, agent_ch, user_ch, deg, Cell(cond, "wire", "energy"))
    assert row.status == "ok"
    assert abs(row.stop_err_ms) < FRAME_TOL_MS, f"off by {row.stop_err_ms:.1f} ms on clean 16 kHz"


def test_silero_rejects_modulated_noise():
    """Documents why the synthetic-stimulus tests do not use Silero: it is a speech model,
    and correctly declines to call amplitude-modulated noise speech."""
    trial, agent_ch, user_ch = make_trial(onset=1.2, lat=0.6)
    cond = BY_NAME["ref_16k"]
    deg = degrade_trial(trial, agent_ch, user_ch, cond)
    row = measure(trial, agent_ch, user_ch, deg, Cell(cond, "wire", "silero"))
    assert row.status != "ok"
    assert row.stop_err_ms is None


def test_jitter_buffer_shifts_both_boundaries_equally_in_the_wire_viewpoint():
    """60 ms of playout delay on both channels must cancel out of the difference."""
    trial, agent_ch, user_ch = make_trial(onset=1.2, lat=0.6)
    cond = BY_NAME["nb8_ulaw_jb60"]
    deg = degrade_trial(trial, agent_ch, user_ch, cond)
    wire = measure(trial, agent_ch, user_ch, deg, Cell(cond, "wire", "energy"))
    assert abs(wire.onset_err_ms - 60.0) < FRAME_TOL_MS
    assert abs(wire.offset_err_ms - 60.0) < FRAME_TOL_MS
    assert abs(wire.stop_err_ms) < FRAME_TOL_MS, (
        f"a delay applied to both channels must cancel, got {wire.stop_err_ms:+.1f} ms")


def test_jitter_buffer_biases_the_local_user_viewpoint_by_the_full_delay():
    """Knowing your own emission time does not help if the return path is delayed."""
    trial, agent_ch, user_ch = make_trial(onset=1.2, lat=0.6)
    cond = BY_NAME["nb8_ulaw_jb60"]
    deg = degrade_trial(trial, agent_ch, user_ch, cond)
    local = measure(trial, agent_ch, user_ch, deg, Cell(cond, "local_user", "energy"))
    assert abs(local.stop_err_ms - 60.0) < FRAME_TOL_MS, (
        f"expected the full +60 ms to survive, got {local.stop_err_ms:+.1f} ms")


def test_bulk_delay_is_reported():
    trial, agent_ch, user_ch = make_trial()
    deg = degrade_trial(trial, agent_ch, user_ch, BY_NAME["nb8_ulaw_jb60"])
    assert abs(deg.bulk_delay_ms - 60.0) < 5.0


def test_channel_seeds_differ_per_channel_and_condition():
    a = channel_seed("t000", "c1", "agent")
    b = channel_seed("t000", "c1", "user")
    c = channel_seed("t001", "c1", "agent")
    assert len({a, b, c}) == 3


def test_merge_segments_joins_only_small_gaps():
    segs = [(0.0, 1.0), (1.2, 2.0), (3.0, 3.5)]
    assert merge_segments(segs, 0.3) == [(0.0, 2.0), (3.0, 3.5)]
    assert merge_segments(segs, 0.05) == segs
    assert merge_segments([], 0.5) == []


def test_missing_speech_is_reported_not_silently_zero():
    trial, agent_ch, user_ch = make_trial()
    cond = BY_NAME["ref_16k"]
    silent = np.zeros_like(user_ch)
    deg = Degraded(agent_ch, SR, silent, SR, 0.0)
    row = measure(trial, agent_ch, silent, deg, Cell(cond, "wire", "energy"))
    assert row.status == "no_user_speech"
    assert row.stop_err_ms is None


def test_energy_detector_offset_lands_near_a_hard_cut():
    x = np.concatenate([speechlike(1.0, seed=4), np.zeros(int(0.5 * SR), dtype=np.float32)])
    off = DETECTORS["energy"].last_offset(x, SR)
    assert off is not None and abs(off - 1.0) < 0.15


def test_energy_detector_threshold_tracks_a_raised_floor():
    """A noise floor 40 dB up must not turn the whole file into one speech segment."""
    x = np.concatenate([speechlike(1.0, seed=6), np.zeros(int(0.5 * SR), dtype=np.float32)])
    rng = np.random.default_rng(0)
    x = x + rng.normal(0, 10 ** (-40 / 20), x.size).astype(np.float32)
    det = EnergyDetector()
    segs = merge_segments(det.segments(x, SR), 0.5)
    assert segs and segs[-1][1] < 1.25
