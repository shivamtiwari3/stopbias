"""Phase 2 recovery, validated offline against simulated calls with known ground truth.

This is the validation a paid pilot cannot do: a real call gives no ground truth to check
against, so if the recovery code is wrong the pilot spends money and reports a plausible
number. Everything here runs for free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stopbias.recover import cancel_reference, locate, recover
from stopbias.simcall import SR, simulate
from stopbias.stimulus import build_caller_track, delay_probe

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="corpus not prepared; run `python -m stopbias.cli prep`"
)

# The energy detector's framing convention costs a fixed ~33 ms (Phase 1, RESULTS.md S9) and the
# agent cut is a hard edge, so recovery lands late by about that much. These tolerances bound
# total error, not bias, and are deliberately tighter than the 100 ms effects Phase 2 tests for.
TOL_MS = 60.0
TOL_MS_DEGRADED = 90.0

G711_NB = (("narrowband", {"target_sr": 8000}), ("g711", {"law": "ulaw"}))


def _clips():
    man = json.loads(MANIFEST.read_text())
    agent = next(c["path"] for c in man["clips"] if c["role"] == "agent")
    user = next(c["path"] for c in man["clips"] if c["role"] == "user")
    return agent, user


# --- stimulus ---------------------------------------------------------------------------

def test_probe_is_broadband_so_delay_estimation_is_unambiguous():
    """Phase 1 established a tone's delay is only recoverable modulo its period. The probe must
    not have that failure mode, so its autocorrelation needs one dominant peak."""
    p = delay_probe(SR)
    ac = np.correlate(p, p, mode="full")
    ac = np.abs(ac[ac.size // 2 :])
    ac /= ac[0]
    assert float(np.max(ac[int(0.005 * SR) :])) < 0.35, "probe autocorrelation has strong sidelobes"


def test_caller_track_records_its_own_event_times_exactly():
    agent, user = _clips()
    from stopbias.audio import load_mono
    inter, _ = load_mono(user)
    t = build_caller_track(inter, SR, interjection_at_s=2.5)
    assert t.interjection_start_s == pytest.approx(2.5, abs=1e-9)
    assert t.probe_start_s == pytest.approx(0.5, abs=1e-9)
    i0 = int(round(2.5 * SR))
    assert np.max(np.abs(t.audio[i0 : i0 + 100])) > 0
    assert np.all(t.audio[i0 - 200 : i0 - 100] == 0), "interjection must start at its own onset"


def test_interjection_must_follow_the_probe():
    agent, user = _clips()
    from stopbias.audio import load_mono
    inter, _ = load_mono(user)
    with pytest.raises(ValueError):
        build_caller_track(inter, SR, interjection_at_s=0.6)


# --- matched filtering ------------------------------------------------------------------

@pytest.mark.parametrize("delay_ms", [0.0, 37.0, 120.0])
def test_locate_finds_a_known_signal_to_the_sample(delay_ms):
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 0.2, size=int(0.2 * SR)).astype(np.float32)
    n = int(round(delay_ms / 1000.0 * SR))
    sig = np.concatenate([np.zeros(n, dtype=np.float32), ref, np.zeros(SR, dtype=np.float32)])
    at, corr = locate(ref, sig, SR)
    assert at == pytest.approx(delay_ms / 1000.0, abs=1e-3)
    assert corr > 0.9


def test_locate_reports_low_correlation_when_the_signal_is_absent():
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 0.2, size=int(0.2 * SR)).astype(np.float32)
    sig = rng.normal(0, 0.2, size=SR).astype(np.float32)
    _, corr = locate(ref, sig, SR)
    assert abs(corr) < 0.5


# --- cancellation -----------------------------------------------------------------------

def test_cancellation_suppresses_a_known_reference():
    rng = np.random.default_rng(8)
    ref = rng.normal(0, 0.1, size=3 * SR).astype(np.float32)
    mix = 0.7 * ref + rng.normal(0, 1e-4, size=ref.size).astype(np.float32)
    resid, sup = cancel_reference(mix, ref, SR, quality_from_s=0.0, quality_to_s=1.0)
    assert sup > 20.0, f"only {sup:.1f} dB suppression on a clean linear channel"
    assert float(np.sqrt(np.mean(resid**2))) < float(np.sqrt(np.mean(mix**2))) / 10


def test_cancellation_needs_full_band_support_not_just_the_probe():
    """Why the fit window is every reference-active sample rather than the probe alone. Fitted on
    a 350-3300 Hz chirp the filter is unconstrained outside that band, so it cancels full-band
    speech far worse than its own probe score suggests. This test pins the failure mode that
    caused a real bug: probe suppression of 85 dB alongside only ~25 dB on speech."""
    rng = np.random.default_rng(11)
    probe = delay_probe(SR)
    speech = rng.normal(0, 0.1, size=int(1.5 * SR)).astype(np.float32)  # full-band
    ref = np.zeros(3 * SR, dtype=np.float32)
    ref[int(0.5 * SR) : int(0.5 * SR) + probe.size] = probe
    ref[int(1.2 * SR) : int(1.2 * SR) + speech.size] = speech
    mix = 0.7 * ref

    # Fitted on the whole active reference, the speech region cancels properly.
    resid, _ = cancel_reference(mix, ref, SR, quality_from_s=0.5, quality_to_s=0.75)
    a, b = int(1.3 * SR), int(2.5 * SR)
    got = 10 * np.log10(np.sum(mix[a:b] ** 2) / max(float(np.sum(resid[a:b] ** 2)), 1e-30))
    assert got > 40.0, f"only {got:.1f} dB on the full-band region"


def test_cancellation_handles_a_bulk_delay_it_is_told_about():
    """A 96-tap filter is 6 ms at 16 kHz and cannot represent a 60 ms delay, so the alignment
    argument is doing real work rather than being decorative."""
    rng = np.random.default_rng(9)
    ref = rng.normal(0, 0.1, size=3 * SR).astype(np.float32)
    d = int(0.060 * SR)
    mix = np.concatenate([np.zeros(d, dtype=np.float32), 0.7 * ref])
    ref_pad = np.pad(ref, (0, d))
    _, sup_aligned = cancel_reference(mix, ref_pad, SR, quality_from_s=0.060, quality_to_s=1.0,
                                      align_delay_s=0.060)
    _, sup_blind = cancel_reference(mix, ref_pad, SR, quality_from_s=0.060, quality_to_s=1.0,
                                    align_delay_s=0.0)
    assert sup_aligned > 20.0
    assert sup_aligned > sup_blind + 10.0


# --- end to end -------------------------------------------------------------------------

@pytest.mark.parametrize("true_ms", [200.0, 500.0, 900.0])
def test_dual_channel_recovers_the_true_latency(true_ms):
    agent, user = _clips()
    c = simulate(agent, user, true_ms, d_out_ms=40.0, d_ret_ms=40.0)
    r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=c.provider_agent)
    assert r.status == "ok"
    assert r.stop_latency_ms == pytest.approx(true_ms, abs=TOL_MS)


@pytest.mark.parametrize("true_ms", [200.0, 500.0, 900.0])
def test_mono_mix_recovers_the_true_latency_on_a_clean_leg(true_ms):
    """Cancellation does work when the channel is linear -- which is why the path exists at all
    and why its failure over telephony is a property of the transport, not a coding bug."""
    agent, user = _clips()
    c = simulate(agent, user, true_ms, d_out_ms=40.0, d_ret_ms=40.0)
    r = recover(c.provider_mix, c.sr, c.track, "energy", allow_mono_mix=True)
    assert r.stop_latency_ms == pytest.approx(true_ms, abs=TOL_MS)


def test_mono_mix_is_refused_unless_explicitly_opted_into():
    """The default must not hand back a number that is +680 ms wrong over a real leg."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0)
    r = recover(c.provider_mix, c.sr, c.track, "energy")
    assert r.status == "mono_mix_refused"
    assert r.stop_latency_ms is None


def test_mono_mix_is_never_reported_as_ok_even_when_it_is_right():
    """It is right here, but nothing observable in the call distinguishes this from the legs where
    it is 685 ms wrong, so the status must not claim validity it cannot demonstrate."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0)
    r = recover(c.provider_mix, c.sr, c.track, "energy", allow_mono_mix=True)
    assert r.status == "mono_mix_unvalidated"
    assert r.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS)


@pytest.mark.parametrize("chain,label", [
    (G711_NB, "mu-law 8 kHz"),
    ((("pstn_band", {"lo": 300.0, "hi": 3400.0}),), "PSTN passband"),
])
def test_mono_mix_fails_over_a_telephony_leg_while_dual_channel_holds(chain, label):
    """The finding that makes dual-channel recording a hard Phase 2 requirement. Pinned as a test
    so that a future change to the canceller which claims to fix this has to prove it here."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0, chain=chain, seed=3)

    dual = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=c.provider_agent)
    assert dual.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS_DEGRADED), label

    mono = recover(c.provider_mix, c.sr, c.track, "energy", allow_mono_mix=True)
    assert mono.stop_latency_ms is not None
    err = mono.stop_latency_ms - 500.0
    assert err > 300.0, f"{label}: mono error unexpectedly small ({err:+.0f} ms) -- if the " \
                        "canceller really improved, update the module docstring and the preregistration"
    # The signature of the failure: the recovered "stop" is the end of the interjection, because
    # leakage holds the detector open past the agent's real cessation.
    inter_end_ms = c.track.interjection_dur_s * 1000.0
    assert mono.stop_latency_ms == pytest.approx(inter_end_ms, abs=120.0)


@pytest.mark.parametrize("chain,label", [
    ((), "clean"),
    (G711_NB, "mu-law 8 kHz"),
])
def test_no_in_band_diagnostic_separates_usable_mono_from_unusable(chain, label):
    """Why the failure is handled by refusing the path rather than by flagging trials: the two
    diagnostics the recording affords do not discriminate. Clean scores 74 dB / 0.025 leak and is
    right; the mu-law leg scores 38 dB / 0.028 leak and is 685 ms wrong. This test records the
    overlap, so a genuinely discriminating diagnostic would break it and prompt a rethink."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0, chain=chain, seed=3)
    r = recover(c.provider_mix, c.sr, c.track, "energy", allow_mono_mix=True)
    assert r.leak_corr < 0.10, f"{label}: leak_corr={r.leak_corr:.4f}"
    assert r.suppression_db > 20.0, f"{label}: suppression={r.suppression_db:.1f} dB"


def test_wire_recovery_is_immune_to_the_transport_delay():
    """The point of locating the onset in the recording rather than trusting the local clock:
    the answer must not move when the route gets slower."""
    agent, user = _clips()
    out = []
    for d in (0.0, 60.0, 150.0):
        c = simulate(agent, user, 500.0, d_out_ms=d, d_ret_ms=d)
        r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=c.provider_agent)
        assert r.status == "ok"
        out.append(r.stop_latency_ms)
    assert max(out) - min(out) < 25.0, f"delay leaked into the wire measurement: {out}"


def test_local_only_is_biased_by_the_full_round_trip_and_the_correction_removes_it():
    """The arithmetic the module docstring derives: the raw local measurement is L + RTT, so the
    whole round trip comes off -- not half of it."""
    agent, user = _clips()
    true_ms, d_out, d_ret = 500.0, 30.0, 90.0
    c = simulate(agent, user, true_ms, d_out_ms=d_out, d_ret_ms=d_ret)

    raw = recover(c.local_inbound, c.sr, c.track, "energy", local_only_rtt_ms=0.0)
    assert raw.stop_latency_ms == pytest.approx(true_ms + c.rtt_ms, abs=TOL_MS)

    fixed = recover(c.local_inbound, c.sr, c.track, "energy", local_only_rtt_ms=c.rtt_ms)
    assert fixed.stop_latency_ms == pytest.approx(true_ms, abs=TOL_MS)


def test_local_only_correction_needs_no_symmetry_assumption():
    """Same RTT split three different ways must give the same corrected answer."""
    agent, user = _clips()
    got = []
    for d_out, d_ret in ((10.0, 110.0), (60.0, 60.0), (110.0, 10.0)):
        c = simulate(agent, user, 500.0, d_out_ms=d_out, d_ret_ms=d_ret)
        r = recover(c.local_inbound, c.sr, c.track, "energy", local_only_rtt_ms=c.rtt_ms)
        got.append(r.stop_latency_ms)
    assert max(got) - min(got) < 25.0, f"asymmetry leaked into the correction: {got}"


def test_recovery_survives_a_narrowband_g711_leg():
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0, chain=G711_NB, seed=3)
    r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=c.provider_agent)
    assert r.status == "ok"
    assert r.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS_DEGRADED)


def test_agent_silent_at_the_interjection_is_void_not_zero():
    """A trial where the agent had already finished has no stop latency. Reporting 0 ms would be
    a fabricated data point in the direction of 'infinitely fast'."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, agent_start_s=0.8, interjection_at_s=2.5)
    # Blank the agent channel around the interjection so nothing is active there.
    killed = c.provider_agent.copy()
    killed[int(1.2 * c.sr) :] = 0.0
    r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=killed)
    assert r.status == "agent_not_speaking_at_interjection"
    assert r.stop_latency_ms is None


def test_missing_interjection_is_reported_not_guessed():
    agent, user = _clips()
    c = simulate(agent, user, 500.0)
    r = recover(np.zeros_like(c.provider_mix), c.sr, c.track, "energy", allow_mono_mix=True)
    assert r.status in ("interjection_not_located", "agent_not_speaking_at_interjection")
    assert r.stop_latency_ms is None


@pytest.mark.parametrize("detector", ["energy", "silero"])
def test_a_cut_inside_a_natural_pause_is_unmeasurable(detector):
    """A Phase 2 void condition, not a measurement. If the agent's own speech happens to pause
    across the moment it was cut, there is no acoustic stop to find and every detector reads
    early -- measured at 320 ms early for Silero on otherwise clean audio. So the harness must
    confirm the agent was continuously speaking, and `simulate` models that by default
    (`_voiced_only`); this test constructs the other case on purpose.

    Both detectors must be wrong here. If one were right it would mean the pause was not really
    silent and the test had stopped testing anything."""
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=40.0, d_ret_ms=40.0)
    sr, cut_s = c.sr, 2.5 + 0.040 + 0.500

    ok = recover(c.provider_mix, sr, c.track, detector, agent_channel=c.provider_agent)
    assert ok.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS), \
        "the continuously-voiced control must be measurable, else this proves nothing"

    # Silence the agent for the 300 ms before its cut: it had already stopped talking, so the cut
    # is inaudible. Relying on a clip's own pauses is not enough -- for this clip and this latency
    # the cut lands in voiced speech and the case would not be constructed at all.
    quiet = c.provider_agent.copy()
    quiet[int((cut_s - 0.300) * sr) : int(cut_s * sr)] = 0.0

    bad = recover(c.provider_mix, sr, c.track, detector, agent_channel=quiet)
    assert bad.stop_latency_ms is None or bad.stop_latency_ms < 500.0 - 150.0, \
        f"a cut in a pause should read early, got {bad.stop_latency_ms}"


def test_agent_resumption_is_flagged_and_not_counted_as_the_stop():
    """An agent that stops, then restarts, must be scored on the first cessation. Taking the
    global last offset would inflate the latency by the whole gap."""
    agent, user = _clips()
    c = simulate(agent, user, 400.0, d_out_ms=0.0, d_ret_ms=0.0)
    ch = c.provider_agent.copy()
    src = ch[int(1.0 * c.sr) : int(1.5 * c.sr)]
    start = int(4.0 * c.sr)
    if start + src.size < ch.size:
        ch[start : start + src.size] = src
    r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=ch)
    assert r.status == "ok"
    assert r.agent_resumed
    assert r.stop_latency_ms == pytest.approx(400.0, abs=TOL_MS)


def test_probe_delay_is_measured_and_matches_the_outbound_leg():
    agent, user = _clips()
    c = simulate(agent, user, 500.0, d_out_ms=75.0, d_ret_ms=25.0)
    r = recover(c.provider_mix, c.sr, c.track, "energy", agent_channel=c.provider_agent)
    assert r.probe_delay_ms == pytest.approx(75.0, abs=5.0)
