"""Bridged calls with known ground truth: validating the bridge before it costs anything.

`simcall` simulates a call where the agent is a hard cut chosen by the experimenter and the
recording comes from the provider. This module does the same job for the bridged path, and the
difference matters: here the **real bridge code runs inside the simulated call**. The packets are
really packetised, really lost, really reordered, really concealed by
`bridge.FixedJitterBuffer`, and really transcoded. So the numbers this produces test the bridge,
not a description of it.

The timeline, on the bridge's wall clock, with `t_u` the moment the caller's interjection reaches
the bridge, `D_in` the playout buffer, `D_out` the outbound framing quantum and `L` the model's
true stop latency:

    caller onset arrives at the SIP socket        t_u
    the model hears it                           t_u + D_in
    the model stops emitting                     t_u + D_in + L
    the stop leaves the SIP socket               t_u + D_in + L + D_out

`L` is what the study wants and it is set here by construction, so both viewpoints can be scored
against truth:

    dual_channel("model") -> L                    (the bridge's delay is common mode)
    dual_channel("sip")   -> L + D_in + D_out     (it is not)

Two things this arrangement gets for free, both of which cost `simcall` real effort:

- **The bridge is the recorder.** Dual-channel separation is structural, so the mono-mix failure
  that makes Phase 1's gating call decisive cannot occur in this cell at all. That risk survives
  only for the cascaded arm, where the provider owns the recording.
- **The return leg cannot bias the measurement.** The tap is upstream of it, so whatever the
  carrier does to the audio on its way back to the harness is outside the measurement path.
  `simcall`'s provider-recording viewpoint has the same property; its local viewpoint does not.

Set `voiced_agent=False` to construct the void case from `PREREGISTRATION.md` S3 deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import load_mono, resample
from .bridge import BridgeConfig, BridgeTap, MediaBridge, RtpPacket, packetise
from .degrade import apply_chain
from .recover import Recovery, recover
from .simcall import _voiced_only
from .stimulus import CallerTrack, build_caller_track

MEAS_SR = 16000
MS = 1000.0


@dataclass(frozen=True)
class BridgedCall:
    """One simulated bridged call and the truth its recovery must reproduce."""

    track: CallerTrack
    tap: BridgeTap
    cfg: BridgeConfig
    true_stop_latency_ms: float
    d_out_ms: float
    model_onset_s: float
    dropped: int = 0

    @property
    def bridge_delay_ms(self) -> float:
        return self.cfg.delay_ms

    def expected_ms(self, viewpoint: str) -> float:
        """What a correct recovery should return from each viewpoint."""
        if viewpoint == "model":
            return self.true_stop_latency_ms
        if viewpoint == "sip":
            return self.true_stop_latency_ms + self.bridge_delay_ms
        raise ValueError(f"unknown viewpoint {viewpoint!r}")


def _time_scale(x: np.ndarray, factor: float) -> np.ndarray:
    """Stretch or squeeze a stream, modelling two clocks that do not agree.

    Linear interpolation is enough: the point is the timescale, and the error a resampling
    filter's shape would contribute is far below the drift being modelled.
    """
    if abs(factor - 1.0) < 1e-12 or x.size == 0:
        return np.asarray(x, dtype=np.float32)
    n = max(1, int(round(x.size * factor)))
    src = np.arange(x.size, dtype=np.float64)
    return np.interp(np.linspace(0.0, x.size - 1, n), src, x).astype(np.float32)


def _lose(packets: list[RtpPacket], loss_rate: float, burst_p: float,
          rng: np.random.Generator) -> list[RtpPacket]:
    """Drop whole packets with the Gilbert-Elliott model Phase 1 used for frame loss.

    Dropping real packets rather than blanking an array is what lets the bridge's own
    concealment be the thing under test.
    """
    if loss_rate <= 0.0:
        return list(packets)
    kept, prev = [], False
    for pkt in packets:
        p = burst_p if (prev and burst_p > 0.0) else loss_rate
        prev = bool(rng.random() < p)
        if not prev:
            kept.append(pkt)
    return kept


def _arrivals(packets: list[RtpPacket], cfg: BridgeConfig, jitter_ms: float,
              rng: np.random.Generator) -> tuple[list[RtpPacket], list[float]]:
    """Nominal arrival times plus bounded delay jitter, sorted into real arrival order.

    Sorting is what makes reordering genuine: once the jitter exceeds the 20 ms packet spacing,
    packets physically arrive out of sequence and the buffer has to cope.
    """
    base = [i * cfg.frame_s for i in range(len(packets))]
    if jitter_ms > 0.0:
        base = [t + float(rng.uniform(0.0, jitter_ms / MS)) for t in base]
    order = sorted(range(len(packets)), key=lambda i: base[i])
    return [packets[i] for i in order], [base[i] for i in order]


def simulate_bridged(agent_clip_path: str, interjection_path: str, true_stop_latency_ms: float,
                     cfg: BridgeConfig | None = None, agent_start_s: float = 1.0,
                     interjection_at_s: float = 2.5, d_out_ms: float = 0.0,
                     carrier_ops: tuple = (), loss_rate: float = 0.0, burst_p: float = 0.0,
                     jitter_ms: float = 0.0, clock_ppm: float = 0.0, seed: int = 0,
                     agent_tail_s: float = 2.0, voiced_agent: bool = True) -> BridgedCall:
    """Run one call through the real bridge with a model whose stop latency is known.

    `carrier_ops` carries only analogue-domain impairments -- the passband, line noise -- because
    the sample rate and the G.711 quantisation belong to the RTP stream itself and are applied by
    `bridge.packetise`. Loss, jitter and reordering are packet-level and are given as parameters
    rather than as chain ops for the same reason.

    The model is modelled as continuously voiced up to a hard cut, exactly as in Phase 1, and its
    stop is placed relative to when the audio *reached it* rather than when the harness emitted
    it. That is the causal ordering a real barge-in path has, and it is why the bridge's inbound
    delay pushes the stop later in absolute time without changing `L`.
    """
    cfg = cfg or BridgeConfig()
    rng = np.random.default_rng(seed)
    bridge = MediaBridge(cfg)

    agent_src, sr_a = load_mono(agent_clip_path)
    inter, sr_i = load_mono(interjection_path)
    if sr_a != MEAS_SR or sr_i != MEAS_SR:
        raise ValueError(f"clips must be {MEAS_SR} Hz, got {sr_a} and {sr_i}")
    if voiced_agent:
        agent_src = _voiced_only(agent_src, MEAS_SR)

    track = build_caller_track(inter, MEAS_SR, interjection_at_s=interjection_at_s)

    t_u = interjection_at_s + d_out_ms / MS
    model_onset_s = t_u + cfg.inbound_delay_ms / MS
    stop_at_s = model_onset_s + true_stop_latency_ms / MS
    if stop_at_s <= agent_start_s:
        raise ValueError("the model must be speaking when it hears the interjection")
    total_s = max(track.duration_s, stop_at_s) + agent_tail_s

    # --- carrier, harness -> bridge -------------------------------------------------------
    caller16 = np.pad(track.audio, (0, max(0, int(round(total_s * MEAS_SR)) - track.audio.size)))
    caller8 = resample(caller16, MEAS_SR, cfg.pstn_sr)
    if carrier_ops:
        caller8, sr8 = apply_chain(caller8, cfg.pstn_sr, carrier_ops, seed=seed * 2 + 1)
        if sr8 != cfg.pstn_sr:
            raise ValueError("carrier_ops must not change the sample rate; that is the RTP layer's job")
    lead = int(round(d_out_ms / MS * cfg.pstn_sr))
    caller8 = np.concatenate([np.zeros(lead, dtype=np.float32), caller8])
    caller8 = _time_scale(caller8, 1.0 + clock_ppm / 1e6)

    packets = _lose(packetise(caller8, cfg), loss_rate, burst_p, rng)
    packets, arrival_times = _arrivals(packets, cfg, jitter_ms, rng)
    sip_user, model_user, stats = bridge.carrier_to_model(packets, arrival_times)

    # --- the model ------------------------------------------------------------------------
    agent_m = resample(agent_src, MEAS_SR, cfg.model_sr)
    total_m = int(round(total_s * cfg.model_sr))
    model_agent = np.zeros(total_m, dtype=np.float32)
    a0 = int(round(agent_start_s * cfg.model_sr))
    a1 = min(total_m, int(round(stop_at_s * cfg.model_sr)))
    span = a1 - a0
    if span <= 0:
        raise ValueError("the model must be speaking when it hears the interjection")
    reps = int(np.ceil(span / max(1, agent_m.size)))
    model_agent[a0:a1] = np.tile(agent_m, reps)[:span]

    # --- bridge -> carrier ----------------------------------------------------------------
    _, sip_agent = bridge.model_to_carrier(model_agent)

    return BridgedCall(
        track=track,
        tap=BridgeTap(sip_user=sip_user, sip_agent=sip_agent,
                      model_user=model_user, model_agent=model_agent,
                      pstn_sr=cfg.pstn_sr, model_sr=cfg.model_sr, inbound=stats),
        cfg=cfg,
        true_stop_latency_ms=float(true_stop_latency_ms),
        d_out_ms=float(d_out_ms),
        model_onset_s=float(model_onset_s),
        dropped=stats.lost,
    )


def measure(call: BridgedCall, detector: str = "energy", viewpoint: str = "model") -> Recovery:
    """Recover one bridged call with Phase 1's code, from either tap.

    The user channel is handed in as the recording rather than a mix of the two: with a
    dual-channel tap there is no reason to locate the interjection inside the agent's speech as
    well, and `recover` only uses the recording to find that onset.
    """
    user, agent = call.tap.dual_channel(viewpoint, MEAS_SR)
    return recover(user, MEAS_SR, call.track, detector, agent_channel=agent)


# --- the validation grid ------------------------------------------------------------------

PSTN_BAND = (("pstn_band", {"lo": 300.0, "hi": 3400.0}),)

#: Each leg is one thing a carrier does to a call, plus two that combine them. The names match
#: Phase 1's conditions where they describe the same impairment, so the two tables can be read
#: together.
LEGS: tuple[tuple[str, dict], ...] = (
    ("clean", {}),
    ("pstn_band", {"carrier_ops": PSTN_BAND}),
    ("noise_20db", {"carrier_ops": PSTN_BAND + (("noise", {"snr_db": 20.0}),)}),
    ("loss_1pct", {"loss_rate": 0.01}),
    ("loss_5pct_bursty", {"loss_rate": 0.05, "burst_p": 0.4}),
    ("jitter_15ms", {"jitter_ms": 15.0}),
    ("jitter_80ms", {"jitter_ms": 80.0}),
    ("drift_200ppm", {"clock_ppm": 200.0}),
    ("pstn_full", {"carrier_ops": PSTN_BAND + (("noise", {"snr_db": 20.0}),),
                   "loss_rate": 0.03, "burst_p": 0.4, "jitter_ms": 15.0, "clock_ppm": 100.0}),
)

LATENCIES_MS = (100.0, 200.0, 500.0, 900.0, 1200.0)


def validate(agent_clip_path: str, interjection_path: str,
             latencies_ms: tuple[float, ...] = LATENCIES_MS,
             detectors: tuple[str, ...] = ("energy", "silero"),
             viewpoints: tuple[str, ...] = ("model", "sip"),
             cfg: BridgeConfig | None = None, d_out_ms: float = 40.0, seed: int = 3):
    """Score every leg x latency x detector x viewpoint against the latency that was built in.

    Returns a DataFrame of per-trial errors. The column that matters is `err_ms`, the signed
    difference from what that viewpoint *should* return -- `L` for the model tap, `L + D_bridge`
    for the SIP tap -- so a single column is comparable across viewpoints that measure different
    quantities.
    """
    import pandas as pd

    rows = []
    for leg, kwargs in LEGS:
        for true_ms in latencies_ms:
            call = simulate_bridged(agent_clip_path, interjection_path, true_ms, cfg=cfg,
                                    d_out_ms=d_out_ms, seed=seed, **kwargs)
            for det in detectors:
                got = {}
                for vp in viewpoints:
                    r = measure(call, det, vp)
                    got[vp] = r.stop_latency_ms
                    rows.append({
                        "leg": leg, "true_ms": true_ms, "detector": det, "viewpoint": vp,
                        "status": r.status,
                        "recovered_ms": r.stop_latency_ms,
                        "expected_ms": call.expected_ms(vp),
                        "err_ms": (None if r.stop_latency_ms is None
                                   else r.stop_latency_ms - call.expected_ms(vp)),
                        "bridge_delay_ms": call.bridge_delay_ms,
                        "measured_bridge_delay_ms": None,
                        "concealed_pct": 100.0 * call.tap.inbound.concealed_fraction,
                        "late": call.tap.inbound.late,
                        "lost": call.tap.inbound.lost,
                        "reordered": call.tap.inbound.reordered,
                    })
                if got.get("model") is not None and got.get("sip") is not None:
                    d = got["sip"] - got["model"]
                    for row in rows[-len(viewpoints):]:
                        row["measured_bridge_delay_ms"] = d
    return pd.DataFrame(rows)


def markdown(df) -> str:
    """The bridge's validation report, one question per table."""
    out = [
        "# Phase 2 bridge validation, offline against known ground truth\n",
        f"{len(df)} recoveries: {df.leg.nunique()} carrier legs x {df.true_ms.nunique()} true "
        f"latencies x {df.detector.nunique()} detectors x {df.viewpoint.nunique()} viewpoints. "
        "The real bridge code is in the path -- real packetisation, real loss, real concealment, "
        "real transcode.\n",
        "\n`err_ms` is signed error against what each viewpoint should return: `L` at the model "
        "tap, `L + D_bridge` at the SIP tap.\n",
    ]
    for det in sorted(df.detector.unique()):
        out.append(f"\n## detector = `{det}`\n\n")
        out.append("| leg | model err (ms) | sip err (ms) | D_bridge measured | concealed | late | reord |\n")
        out.append("|---|--:|--:|--:|--:|--:|--:|\n")
        for leg, _ in LEGS:
            s = df[(df.detector == det) & (df.leg == leg)]
            if s.empty:
                continue
            m = s[s.viewpoint == "model"].err_ms.dropna()
            p = s[s.viewpoint == "sip"].err_ms.dropna()
            d = s.measured_bridge_delay_ms.dropna()
            out.append(
                f"| {leg} "
                f"| {m.median():+.1f} ({m.min():+.0f} to {m.max():+.0f}) "
                f"| {p.median():+.1f} ({p.min():+.0f} to {p.max():+.0f}) "
                f"| {d.median():.1f} of {s.bridge_delay_ms.iloc[0]:.0f} "
                f"| {s.concealed_pct.iloc[0]:.1f}% "
                f"| {int(s.late.iloc[0])} | {int(s.reordered.iloc[0])} |\n"
            )
    bad = df[df.status != "ok"]
    out.append(f"\nNon-`ok` recoveries: {len(bad)} of {len(df)}"
               + (f" ({', '.join(sorted(bad.status.unique()))})" if len(bad) else "") + ".\n")
    return "".join(out)
