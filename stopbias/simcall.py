"""Synthetic live calls with known ground truth.

Phase 2's recovery code has to be validated before any money is spent on real calls, and a
pilot cannot validate it: a real call gives no ground truth to check against. So calls are
simulated here with the delays, codecs and losses of Phase 1 applied per direction, and the
recovery methods are checked against the latency that was built in.

The timing model, with T the harness's local emission time and L the agent's true stop latency:

    harness emits interjection            T                    (harness clock)
    agent receives its onset              T + d_out
    agent stops emitting                  T + d_out + L
    harness hears the stop                T + d_out + L + d_ret

Two observation points follow, and they are not interchangeable:

    provider recording (agent side) = shift(caller, d_out) + agent
        user onset at T + d_out, agent stop at T + d_out + L, so the difference is L exactly.

    harness inbound stream (local)  = shift(agent, d_ret)
        onset known from its own clock at T, agent stop heard at T + d_out + L + d_ret, so the
        difference is L + RTT. Inbound is recorded separately from what the harness sends, so
        the harness's own audio is not in it.

That asymmetry is the whole reason Phase 1's two viewpoints exist, and simulating it is what
makes the offline validation meaningful rather than circular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import load_mono
from .degrade import apply_chain
from .stimulus import CallerTrack, build_caller_track

SR = 16000


@dataclass(frozen=True)
class SimCall:
    """A simulated call plus the ground truth the recovery must reproduce."""

    track: CallerTrack
    provider_mix: np.ndarray          # mono mixed, agent-side clock
    provider_user: np.ndarray         # dual-channel: user leg, agent-side clock
    provider_agent: np.ndarray        # dual-channel: agent leg, agent-side clock
    local_inbound: np.ndarray         # harness-side: only what it received
    local_mixed: np.ndarray           # harness-side: inbound plus its own emission
    sr: int
    true_stop_latency_ms: float
    d_out_ms: float
    d_ret_ms: float

    @property
    def rtt_ms(self) -> float:
        return self.d_out_ms + self.d_ret_ms


def _voiced_only(x: np.ndarray, sr: int, rel_db: float = -30.0,
                 frame_ms: float = 20.0) -> np.ndarray:
    """Concatenate the frames of `x` that are genuinely loud, dropping its natural pauses.

    Phase 1 could place the agent's cut wherever it liked and simply nudged it into loud speech
    (`trials._active_before`). Phase 2 cannot: the latency is the agent's own behaviour, so the
    cut lands where it lands. If it lands inside a natural pause then no detector can see the
    stop -- the agent had already gone quiet -- and the ground truth is unobservable rather than
    merely hard to measure. Measured directly: a cut in a pause makes Silero read 320 ms *early*
    on otherwise clean audio.

    So the simulated agent is modelled as mid-utterance and continuously voiced, which is the
    condition a real trial must also satisfy. `test_a_cut_inside_a_natural_pause_is_unmeasurable`
    pins the other case, and it is a void condition for Phase 2, not a measurement.
    """
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-9:
        return x
    trimmed = x[: x.size - (x.size % n)].reshape(-1, n)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    keep = trimmed[rms >= peak * 10.0 ** (rel_db / 20.0)]
    return keep.reshape(-1).astype(np.float32) if keep.size else x


def _shift(x: np.ndarray, sr: int, delay_s: float, total: int) -> np.ndarray:
    n = int(round(delay_s * sr))
    out = np.zeros(total, dtype=np.float32)
    if n >= total:
        return out
    take = min(x.size, total - n)
    out[n : n + take] = x[:take]
    return out


def simulate(agent_clip_path: str, interjection_path: str, true_stop_latency_ms: float,
             agent_start_s: float = 1.0, interjection_at_s: float = 2.5,
             d_out_ms: float = 0.0, d_ret_ms: float = 0.0,
             chain: tuple = (), seed: int = 0, sr: int = SR,
             agent_tail_s: float = 2.0, voiced_agent: bool = True) -> SimCall:
    """Build one simulated call.

    `chain` is a Phase 1 impairment chain applied to each direction independently, with
    different seeds, because the two directions of a real call are not correlated.

    `voiced_agent` tiles the agent's pauses out so the hard cut is observable (see
    `_voiced_only`). Pass False to construct the unmeasurable case deliberately.
    """
    agent_src, sr_a = load_mono(agent_clip_path)
    inter, sr_i = load_mono(interjection_path)
    if sr_a != sr or sr_i != sr:
        raise ValueError(f"clips must be {sr} Hz, got {sr_a} and {sr_i}")
    if voiced_agent:
        agent_src = _voiced_only(agent_src, sr)

    track = build_caller_track(inter, sr, interjection_at_s=interjection_at_s)

    # The agent hears the onset at interjection_at_s + d_out and stops L later. Its own emission
    # timeline is the provider clock, so the cut is placed at that absolute time.
    stop_at_s = interjection_at_s + d_out_ms / 1000.0 + true_stop_latency_ms / 1000.0
    total_s = max(track.duration_s, stop_at_s) + agent_tail_s
    total = int(round(total_s * sr))

    agent = np.zeros(total, dtype=np.float32)
    a0 = int(round(agent_start_s * sr))
    a_end = int(round(stop_at_s * sr))
    span = a_end - a0
    if span <= 0:
        raise ValueError("agent must be speaking when the interjection lands")
    reps = int(np.ceil(span / max(1, agent_src.size)))
    agent[a0:a_end] = np.tile(agent_src, reps)[:span]

    caller = np.pad(track.audio, (0, max(0, total - track.audio.size)))[:total]

    # Apply the transport, per direction, then place each stream on the right clock.
    up, sr_up = apply_chain(caller, sr, chain, seed=seed * 2 + 1)
    down, sr_dn = apply_chain(agent, sr, chain, seed=seed * 2 + 2)
    if sr_up != sr or sr_dn != sr:
        from .audio import resample
        up, down = resample(up, sr_up, sr), resample(down, sr_dn, sr)
    up = np.pad(up, (0, max(0, total - up.size)))[:total]
    down = np.pad(down, (0, max(0, total - down.size)))[:total]

    # `agent` already holds emission times on the shared absolute clock, and the agent's stop was
    # placed at T + d_out + L. So reaching the harness costs d_ret alone -- adding d_out here
    # would count the outbound leg twice.
    prov_user = _shift(up, sr, d_out_ms / 1000.0, total)
    prov_agent = down
    # The harness receives the agent's stream and nothing else: it records inbound separately
    # from what it sends, so there is no local mixing. `local_mixed` is kept for the SDKs that
    # hand back a single mixed file, where case 2's canceller has to run first.
    local_in = _shift(down, sr, d_ret_ms / 1000.0, total)

    return SimCall(
        track=track,
        provider_mix=(prov_user + prov_agent).astype(np.float32),
        provider_user=prov_user,
        provider_agent=prov_agent,
        local_inbound=local_in.astype(np.float32),
        local_mixed=(caller[:total] + local_in).astype(np.float32),
        sr=sr,
        true_stop_latency_ms=float(true_stop_latency_ms),
        d_out_ms=float(d_out_ms),
        d_ret_ms=float(d_ret_ms),
    )
