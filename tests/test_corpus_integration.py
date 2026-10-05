"""Integration tests on the real corpus. Skipped when the corpus has not been prepared.

These are the tests that validate Silero, which cannot be exercised on synthetic stimuli
because it is a speech model and correctly rejects non-speech.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stopbias.audio import energy_onset, load_mono
from stopbias.conditions import BY_NAME, Cell
from stopbias.measure import degrade_trial, measure
from stopbias.trials import SR, build, load_plan

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"
PLAN = ROOT / "data" / "trials" / "plan.json"

pytestmark = pytest.mark.skipif(
    not (MANIFEST.exists() and PLAN.exists()),
    reason="corpus not prepared; run `python -m stopbias.cli prep && ... plan`",
)


def _trials(n=6):
    return load_plan(PLAN)[:n]


def _render(trial):
    user, _ = load_mono(trial.user_clip)
    agent, _ = load_mono(trial.agent_clip)
    return build(user, agent, trial.onset_s, trial.cut_s, trial.tail_s)


def test_speaker_pools_are_disjoint():
    man = json.loads(MANIFEST.read_text())
    assert not (set(man["user_speakers"]) & set(man["agent_speakers"]))
    users = {c["speaker_id"] for c in man["clips"] if c["role"] == "user"}
    agents = {c["speaker_id"] for c in man["clips"] if c["role"] == "agent"}
    assert not (users & agents)
    assert len(users) > 5 and len(agents) > 5


def test_every_user_clip_starts_exactly_at_its_energy_onset():
    """The ground truth of the whole experiment. If a clip has leading silence, the
    'true onset' recorded in the plan is not where the audio actually begins."""
    man = json.loads(MANIFEST.read_text())
    for c in man["clips"]:
        if c["role"] != "user":
            continue
        x, sr = load_mono(c["path"])
        assert energy_onset(x, sr) == pytest.approx(0.0, abs=1e-9), c["clip_id"]


def test_planned_cut_lands_in_loud_agent_speech():
    for t in _trials():
        agent_ch, _ = _render(t)
        n = int(round(t.cut_s * SR))
        pre = agent_ch[max(0, n - int(0.05 * SR)) : n]
        peak = float(np.max(np.abs(agent_ch)))
        rms = float(np.sqrt(np.mean(pre.astype(np.float64) ** 2)))
        assert rms >= peak * 10 ** (-30 / 20), f"{t.trial_id}: cut landed in a pause"
        assert np.all(agent_ch[n:] == 0)


@pytest.mark.parametrize("detector", ["silero", "energy"])
def test_detectors_find_real_speech_in_every_trial(detector):
    cond = BY_NAME["ref_16k"]
    for t in _trials():
        agent_ch, user_ch = _render(t)
        deg = degrade_trial(t, agent_ch, user_ch, cond)
        row = measure(t, agent_ch, user_ch, deg, Cell(cond, "wire", detector))
        assert row.status == "ok", f"{detector} found no speech in {t.trial_id}"


def test_silero_recovers_true_stop_latency_on_clean_speech():
    """Silero's median error on 16 kHz files must be small, or the reference condition is
    not a usable baseline to compare transports against."""
    cond = BY_NAME["ref_16k"]
    errs = []
    for t in _trials(12):
        agent_ch, user_ch = _render(t)
        deg = degrade_trial(t, agent_ch, user_ch, cond)
        row = measure(t, agent_ch, user_ch, deg, Cell(cond, "wire", "silero"))
        errs.append(row.stop_err_ms)
    med = float(np.median(errs))
    assert abs(med) < 40.0, f"Silero median error {med:+.1f} ms on clean 16 kHz speech"


def test_jitter_buffer_control_holds_on_real_speech():
    """The experiment's internal control, on the actual stimuli rather than synthetic ones."""
    jb, base = BY_NAME["nb8_ulaw_jb60"], BY_NAME["nb8_ulaw"]
    d_wire, d_local = [], []
    for t in _trials(12):
        agent_ch, user_ch = _render(t)
        for det in ("silero", "energy"):
            b = measure(t, agent_ch, user_ch, degrade_trial(t, agent_ch, user_ch, base), Cell(base, "wire", det))
            j = measure(t, agent_ch, user_ch, degrade_trial(t, agent_ch, user_ch, jb), Cell(jb, "wire", det))
            if b.stop_err_ms is not None and j.stop_err_ms is not None:
                d_wire.append(j.stop_err_ms - b.stop_err_ms)
            bl = measure(t, agent_ch, user_ch, degrade_trial(t, agent_ch, user_ch, base), Cell(base, "local_user", det))
            jl = measure(t, agent_ch, user_ch, degrade_trial(t, agent_ch, user_ch, jb), Cell(jb, "local_user", det))
            if bl.stop_err_ms is not None and jl.stop_err_ms is not None:
                d_local.append(jl.stop_err_ms - bl.stop_err_ms)
    assert abs(float(np.median(d_wire))) < 20.0, "a delay on both channels must cancel"
    assert abs(float(np.median(d_local)) - 60.0) < 20.0, "a delay on one channel must show in full"
