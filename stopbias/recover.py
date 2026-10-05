"""Recovering stop latency from a live call recording.

Phase 2's gating unknown is what the provider hands back. Three cases, three methods, all
validated offline in `tests/test_recover.py` against synthetic calls with known ground truth.

1. **Dual-channel recording.** Both parties separated on the provider's clock. Read boundaries
   directly. This is the `wire` viewpoint from Phase 1 and needs no correction.

2. **Mono mixed recording.** Both parties in one channel, overlapping during exactly the
   barge-in moment that has to be measured. **Measured offline to be unusable over any
   telephony leg, and refused by default.** The onset half works fine: the harness knows what it
   emitted, so the onset is *located in the recording* by matched filtering, putting both
   boundaries on one clock so transport delay cancels -- this is still the unbiased `wire`
   viewpoint. The offset half does not work. Recovering it needs the emitted audio subtracted out
   to expose the agent-only residual, and on simulated calls with known ground truth:

   | leg | mono error | dual-channel error |
   |---|---|---|
   | clean | +15 ms | +15 ms |
   | 20 dB line noise | +15 ms | +15 ms |
   | mu-law 8 kHz | **+685 ms** | +15 ms |
   | 300-3400 Hz passband | **+680 ms** | +20 ms |
   | Opus NB 12 kbit/s | **+695 ms** | +35 ms |
   | full PSTN leg | **+685 ms** | +20 ms |

   The failures are not noise: the recovered "stop" is the end of the interjection, so leakage
   holds the detector open past the agent's real cessation. The physical reason is that the
   interjection sits ~15 dB above the agent in the mix, after the PSTN filter both are speech
   inside the same 300-3400 Hz band, and the channel is nonlinear (companding) and time-varying
   (loss, jitter). Suppressing the user by more than 15 dB across the whole band at every instant
   is what would be required. Two independent cancellers were tried -- a delay-aligned,
   reweighted least-squares FIR and a time-varying per-bin STFT gain -- and neither manages it
   (the STFT version rescues bursty loss but still errs +565 ms on the companded legs).

   Worse, **no in-band diagnostic separates the good cases from the bad**, so this cannot be
   handled by flagging. Probe suppression does not: a mu-law leg scores 38 dB and still errs
   +685 ms, while a 5% bursty-loss leg scores 7.1 dB. Residual-vs-reference correlation does not
   either: clean scores 0.025 and the failing mu-law leg 0.028, because the leakage is
   spectrally distorted -- low correlation, ample energy. Both are still reported per trial as
   `suppression_db` and `leak_corr`, but as diagnostics, not as gates.

   Hence `allow_mono_mix` defaults to False and an opted-in measurement is returned with status
   `mono_mix_unvalidated`, never `ok`. This makes dual-channel recording a hard requirement for
   Phase 2 rather than a preference, and makes the one-call gating experiment decisive.

3. **Local recording only.** No provider recording. `recording` must be the harness's *inbound*
   stream -- what it received, recorded separately from what it sent, which is how a WebRTC track
   or a media websocket delivers it. A locally mixed file would need the case-2 canceller first.

   The harness knows when it emitted and when it *heard* the agent stop. Writing T for the
   emission time, the agent receives the onset at
   T+d_out, stops at T+d_out+L, and the harness hears that at T+d_out+L+d_ret. So the raw local
   measurement is

       L + d_out + d_ret  =  L + RTT

   The correction is therefore the **whole** round trip, not half of it, and no path-symmetry
   assumption is needed -- which is fortunate, because asymmetric routing is common. Phase 1
   measured this bias to be exactly additive (+40.0/+60.0/+80.0 ms at 40/60/80 ms one-way),
   which is what licenses subtracting it.

   The catch is that RTT is **not observable from the in-call probe**: nothing in the call
   echoes the probe back, so the harness can measure neither d_out nor d_ret from it. RTT has
   to come from outside -- an echo-endpoint calibration call on the same route, or provider
   RTCP round-trip statistics. The assumption that remains is that the supplied RTT applies to
   the measurement call, which jitter-buffer adaptation can violate. Case 3 is the fallback of
   last resort for that reason.

The stop is taken from the agent segment active at the interjection, mirroring
Full-Duplex-Bench's overlap rule, not from the global last offset -- an agent that stops and
then resumes must not be credited with the resumption's end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .audio import EPS
from .stimulus import CallerTrack
from .vad import DETECTORS, merge_segments

MS = 1000.0
DEFAULT_TAPS = 96
RIDGE = 1e-6
MIN_MATCH_CORR = 0.15


@dataclass(frozen=True)
class Recovery:
    """One trial's recovered latency plus everything needed to judge whether to trust it."""

    stop_latency_ms: float | None
    user_onset_s: float | None
    agent_stop_s: float | None
    viewpoint: str
    status: str
    match_corr: float = float("nan")
    suppression_db: float = float("nan")
    leak_corr: float = float("nan")
    probe_delay_ms: float = float("nan")
    n_agent_segments: int = 0
    agent_resumed: bool = False


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean() if x.size else x


def locate(reference: np.ndarray, signal: np.ndarray, sr: int,
           search_from_s: float = 0.0, search_to_s: float | None = None) -> tuple[float, float]:
    """Find where `reference` occurs in `signal` by normalised matched filtering.

    Returns (start_time_s, peak_normalised_correlation). The correlation is normalised by the
    energy of both signals over the matched span, so its magnitude is comparable across trials
    and usable as a confidence diagnostic.
    """
    r, s = _norm(reference), _norm(signal)
    if r.size == 0 or s.size < r.size:
        return float("nan"), 0.0
    cc = fftconvolve(s, r[::-1], mode="valid")
    lo = max(0, int(round(search_from_s * sr)))
    hi = cc.size if search_to_s is None else min(cc.size, int(round(search_to_s * sr)) + 1)
    if hi <= lo:
        lo, hi = 0, cc.size
    seg = cc[lo:hi]
    k = int(np.argmax(np.abs(seg))) + lo
    win = s[k : k + r.size]
    denom = float(np.sqrt(np.sum(r**2) * np.sum(win**2)))
    corr = float(cc[k] / denom) if denom > EPS else 0.0
    return k / float(sr), corr


def _active_mask(ref: np.ndarray, sr: int, rel_db: float = -40.0, win_ms: float = 20.0) -> np.ndarray:
    """Samples where the reference actually has energy, via a smoothed power envelope."""
    w = max(1, int(round(sr * win_ms / 1000.0)))
    env = np.convolve(ref**2, np.ones(w) / w, mode="same")
    peak = float(env.max()) if env.size else 0.0
    if peak <= EPS:
        return np.zeros(ref.size, dtype=bool)
    return env >= peak * 10.0 ** (rel_db / 10.0)


def cancel_reference(mix: np.ndarray, reference: np.ndarray, sr: int,
                     quality_from_s: float, quality_to_s: float, align_delay_s: float = 0.0,
                     n_taps: int = DEFAULT_TAPS) -> tuple[np.ndarray, float]:
    """Subtract a known `reference` from `mix` with a least-squares FIR channel estimate.

    `align_delay_s` carries the reference's transport delay into the recording, measured from
    the probe. A short FIR cannot represent a 60 ms bulk delay -- at 16 kHz that would need 960
    taps -- so the reference is coarsely shifted first and the filter only models the residual
    channel response.

    **The filter is fitted over every sample where the reference is active, not over a
    reference-only window.** Fitting on the probe alone was tried and fails badly: the probe is a
    350-3300 Hz chirp, so it leaves the filter unconstrained outside that band, and the fitted
    filter then cancels full-band speech by only ~25 dB despite achieving 85 dB on the probe
    itself. Fitting across the interjection as well gives the filter full spectral support.

    The agent's speech is present during the interjection and is *not* excluded. That is sound:
    least squares with interference that is uncorrelated with the reference -- a different
    speaker -- gives an unbiased estimate of the channel, merely a noisier one. Ridge
    regularisation keeps it stable.

    Suppression is therefore reported on [quality_from_s, quality_to_s), which must be a window
    where the reference is genuinely the only thing present (the probe). Measuring it on the fit
    region would fold in the agent's energy and understate the cancellation.

    Returns (residual, suppression_db). Low suppression means the channel-stationarity assumption
    failed and the trial should be flagged, not silently trusted.
    """
    mix = np.asarray(mix, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    shift = int(round(align_delay_s * sr))
    if shift > 0:
        ref = np.concatenate([np.zeros(shift), ref])
    elif shift < 0:
        ref = ref[-shift:]
    # Pad or trim the reference to the recording's length. Never trim the recording: the agent's
    # stop can fall after the harness has finished emitting.
    n = mix.size
    if n == 0:
        return mix.astype(np.float32), 0.0
    ref = np.pad(ref, (0, max(0, n - ref.size)))[:n]

    sel = np.flatnonzero(_active_mask(ref, sr))
    if sel.size < n_taps * 8:
        return mix.astype(np.float32), 0.0

    pad = np.concatenate([np.zeros(n_taps - 1), ref])
    idx = sel[:, None] + np.arange(n_taps)[None, :]
    X = pad[idx][:, ::-1]
    y = mix[sel]

    def _solve(wt: np.ndarray | None) -> np.ndarray | None:
        Xw, yw = (X, y) if wt is None else (X * wt[:, None], y * wt)
        gram = Xw.T @ Xw
        gram += RIDGE * np.trace(gram) / n_taps * np.eye(n_taps)
        try:
            return np.linalg.solve(gram, Xw.T @ yw)
        except np.linalg.LinAlgError:
            return None

    h = _solve(None)
    if h is None:
        return mix.astype(np.float32), 0.0
    residual = mix - fftconvolve(ref, h, mode="full")[:n]

    # One reweighted pass. The agent's speech is uncorrelated interference, so the unweighted fit
    # is unbiased but its variance scales with the agent's level -- a loud agent measurably
    # degrades the channel estimate and hence the recovered stop. The first-pass residual *is* an
    # estimate of that interference, so it can be used to down-weight the samples where the agent
    # is loud and lean on the ones where it is quiet. The floor keeps weights bounded.
    w_env = max(1, int(round(sr * 0.020)))
    env = np.convolve(residual**2, np.ones(w_env) / w_env, mode="same")[sel]
    floor = float(np.quantile(env, 0.25))
    if floor > EPS:
        h2 = _solve(1.0 / np.sqrt(env + floor))
        if h2 is not None:
            r2 = mix - fftconvolve(ref, h2, mode="full")[:n]
            # Keep the refinement only if it actually cancels better where the truth is known.
            qa, qb = int(round(quality_from_s * sr)), int(round(quality_to_s * sr))
            qa, qb = max(0, qa), min(n, qb)
            if qb - qa > 2 and np.sum(r2[qa:qb] ** 2) < np.sum(residual[qa:qb] ** 2):
                residual = r2

    a, b = int(round(quality_from_s * sr)), int(round(quality_to_s * sr))
    a, b = max(0, a), min(n, b)
    if b - a < 2:
        a, b = 0, n
    e0 = float(np.sum(mix[a:b] ** 2))
    e1 = float(np.sum(residual[a:b] ** 2))
    sup = 10.0 * np.log10(e0 / max(e1, EPS)) if e0 > EPS else 0.0
    return residual.astype(np.float32), float(sup)


def residual_leak(residual: np.ndarray, reference: np.ndarray, sr: int,
                  at_s: float) -> float:
    """How much of `reference` is still present in `residual` at a known position.

    The honest quality guard for the mono-mix path. Probe suppression turned out not to predict
    usability at all -- measured on simulated calls, a G.711 narrowband leg scores 38 dB on the
    probe and still mismeasures the stop by 685 ms, while a 5% bursty-loss leg scores 7.1 dB,
    which clears a 6 dB threshold and would be silently trusted. Both fail for the same reason:
    a stationary FIR cannot model companding, per-direction resampling, or time-varying loss.

    Correlating the residual against the emitted interjection at its known location measures the
    thing that actually matters -- is the user's speech still in there -- rather than a proxy.
    """
    a = int(round(at_s * sr))
    seg = residual[a : a + reference.size]
    if seg.size < reference.size // 2 or reference.size == 0:
        return float("nan")
    r = _norm(reference[: seg.size])
    s = _norm(seg)
    denom = float(np.sqrt(np.sum(r**2) * np.sum(s**2)))
    return abs(float(np.dot(r, s) / denom)) if denom > EPS else 0.0


def _stop_after(segs: list[tuple[float, float]], t_user: float,
                tol_s: float = 0.05) -> tuple[float | None, bool]:
    """End of the agent segment active at `t_user`, plus whether the agent resumed later.

    Mirrors Full-Duplex-Bench's overlap rule. If the agent was not speaking at the
    interjection there is no stop to measure and the trial is void, which is a different
    outcome from "stopped instantly".
    """
    if not segs:
        return None, False
    for i, (s, e) in enumerate(segs):
        if s - tol_s <= t_user <= e + tol_s:
            return e, i + 1 < len(segs)
    return None, any(s > t_user for s, _ in segs)


def recover(recording: np.ndarray, sr: int, track: CallerTrack, detector: str = "energy",
            agent_channel: np.ndarray | None = None,
            local_only_rtt_ms: float | None = None,
            allow_mono_mix: bool = False) -> Recovery:
    """Recover one trial's stop latency, choosing the method from what was supplied.

    - `agent_channel` given -> case 1, dual channel. The only path validated over a telephony leg.
    - `local_only_rtt_ms` given -> case 3, harness inbound stream, corrected by the full RTT.
    - neither -> case 2, mono mix, which is refused unless `allow_mono_mix` is set and then
      returns status `mono_mix_unvalidated`. See the module docstring for the measurements.
    """
    spec = DETECTORS[detector]
    if agent_channel is None and local_only_rtt_ms is None and not allow_mono_mix:
        return Recovery(None, None, None, "wire", "mono_mix_refused")
    rec = np.asarray(recording, dtype=np.float32)
    probe_from = track.probe_start_s
    probe = track.audio[int(round(probe_from * sr)) :
                        int(round((probe_from + 0.25) * sr))]
    probe_at, probe_corr = locate(probe, rec, sr)
    probe_delay_ms = (probe_at - probe_from) * MS if np.isfinite(probe_at) else float("nan")

    inter_ref = track.audio[int(round(track.interjection_start_s * sr)) :
                            int(round((track.interjection_start_s + track.interjection_dur_s) * sr))]

    if local_only_rtt_ms is not None:
        # Case 3: the harness's own clock. Onset is known exactly; the raw measurement is
        # L + RTT, so the whole round trip comes off (see the module docstring).
        t_user = track.interjection_start_s
        agent_est = rec
        viewpoint = "local_user_corrected"
        corr = float("nan")
        sup = float("nan")
        leak = float("nan")
        correction_ms = float(local_only_rtt_ms)
    else:
        correction_ms = 0.0
        viewpoint = "wire"
        # Locate the interjection on the recording's own clock so transport delay cancels.
        t_user, corr = locate(inter_ref, rec, sr)
        if not np.isfinite(t_user) or abs(corr) < MIN_MATCH_CORR:
            return Recovery(None, None, None, viewpoint, "interjection_not_located",
                            corr, float("nan"), float("nan"), probe_delay_ms)
        if agent_channel is not None:
            agent_est, sup, leak = np.asarray(agent_channel, dtype=np.float32), float("inf"), 0.0
        else:
            # Hand the canceller the measured transport delay so the short FIR only models the
            # residual channel, and score its quality on the probe window -- the one stretch of
            # the recording where the harness's audio is provably the only thing present.
            align_s = 0.0 if not np.isfinite(probe_delay_ms) else probe_delay_ms / MS
            agent_est, sup = cancel_reference(
                rec, track.audio, sr,
                quality_from_s=probe_at, quality_to_s=probe_at + 0.25,
                align_delay_s=align_s,
            )
            leak = residual_leak(agent_est, inter_ref, sr, t_user)

    segs = merge_segments(spec.detector.segments(agent_est, sr), spec.model_merge_gap_s)
    stop_s, resumed = _stop_after(segs, t_user)
    m_corr = corr if local_only_rtt_ms is None else float("nan")
    if stop_s is None:
        return Recovery(None, float(t_user), None, viewpoint, "agent_not_speaking_at_interjection",
                        m_corr, sup, leak, probe_delay_ms, len(segs), resumed)

    latency_ms = (stop_s - t_user) * MS - correction_ms
    status = "ok"
    if local_only_rtt_ms is None and agent_channel is None:
        # Never `ok`: neither diagnostic separates the usable legs from the +680 ms ones, so the
        # honest label is "unvalidated" and the caller must decide what to do with it.
        status = "mono_mix_unvalidated"
    return Recovery(float(latency_ms), float(t_user), float(stop_s), viewpoint, status,
                    m_corr, sup, leak, probe_delay_ms, len(segs), resumed)
