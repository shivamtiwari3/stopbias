"""Measurement: run a detector over a degraded trial and record every error term.

The decomposition that matters:

    stop_err = offset_err - onset_err

A condition can be badly wrong on both boundaries and still look fine on stop latency if
the two errors happen to cancel. Reporting only the stop-latency error hides that, so all
three terms are kept. `bulk_delay_ms` separates a codec's or buffer's constant transport
delay from the detector's own error.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .audio import estimate_bulk_delay, resample
from .conditions import Cell, Condition
from .degrade import apply_chain
from .trials import SR, Trial
from .vad import DETECTORS, merge_segments

MS = 1000.0


def channel_seed(trial_id: str, condition: str, channel: str) -> int:
    return zlib.crc32(f"{trial_id}|{condition}|{channel}".encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class Degraded:
    """A condition applied once per trial, then shared by every viewpoint and detector."""

    agent: np.ndarray
    agent_sr: int
    user: np.ndarray
    user_sr: int
    bulk_delay_ms: float


def degrade_trial(trial: Trial, agent_ch: np.ndarray, user_ch: np.ndarray, cond: Condition) -> Degraded:
    a, sr_a = apply_chain(agent_ch, SR, cond.chain, seed=channel_seed(trial.trial_id, cond.name, "agent"))
    u, sr_u = apply_chain(user_ch, SR, cond.chain, seed=channel_seed(trial.trial_id, cond.name, "user"))
    ref16 = agent_ch
    got16 = resample(a, sr_a, SR)
    bulk = estimate_bulk_delay(ref16, got16, SR) * MS
    return Degraded(a, sr_a, u, sr_u, float(bulk))


@dataclass(frozen=True)
class Row:
    trial_id: str
    condition: str
    viewpoint: str
    detector: str
    analysis_sr_user: int
    analysis_sr_agent: int
    true_onset_ms: float
    true_cut_ms: float
    true_stop_ms: float
    det_onset_ms: float | None
    det_offset_ms: float | None
    meas_stop_ms: float | None
    onset_err_ms: float | None
    offset_err_ms: float | None
    stop_err_ms: float | None
    bulk_delay_ms: float
    n_agent_segments: int
    status: str


def measure(
    trial: Trial,
    agent_ch: np.ndarray,
    user_ch: np.ndarray,
    deg: Degraded,
    cell: Cell,
    cache: dict | None = None,
) -> Row:
    """`cache` memoises boundaries within a trial. The agent-side analysis does not depend
    on the viewpoint, and the local_user user-side analysis does not depend on the
    condition, so without a cache each is recomputed several times over identical audio."""
    cond = cell.condition
    spec = DETECTORS[cell.detector]
    cache = cache if cache is not None else {}

    if cell.viewpoint == "local_user":
        # The measuring party emitted this audio; it reads its own pristine copy.
        user_x, user_sr = user_ch, SR
        user_key = (trial.trial_id, "local_user", cell.detector, SR)
    else:
        user_x, user_sr = resample(deg.user, deg.user_sr, cond.analysis_sr), cond.analysis_sr
        user_key = (trial.trial_id, f"wire:{cond.name}", cell.detector, user_sr)

    agent_sr = cond.analysis_sr
    agent_key = (trial.trial_id, f"agent:{cond.name}", cell.detector, agent_sr)

    if user_key in cache:
        onset = cache[user_key]
    else:
        onset = spec.first_onset(user_x, user_sr)
        cache[user_key] = onset

    if agent_key in cache:
        offset, n_segs = cache[agent_key]
    else:
        agent_x = resample(deg.agent, deg.agent_sr, agent_sr)
        merged = merge_segments(spec.detector.segments(agent_x, agent_sr), spec.model_merge_gap_s)
        offset = merged[-1][1] if merged else None
        n_segs = len(merged)
        cache[agent_key] = (offset, n_segs)

    status = "ok"
    if onset is None:
        status = "no_user_speech"
    elif offset is None:
        status = "no_agent_speech"

    onset_ms = None if onset is None else onset * MS
    offset_ms = None if offset is None else offset * MS
    true_onset_ms = trial.onset_s * MS
    true_cut_ms = trial.cut_s * MS
    true_stop_ms = trial.true_stop_latency_s * MS

    onset_err = None if onset_ms is None else onset_ms - true_onset_ms
    offset_err = None if offset_ms is None else offset_ms - true_cut_ms
    meas_stop = None if (onset_ms is None or offset_ms is None) else offset_ms - onset_ms
    stop_err = None if meas_stop is None else meas_stop - true_stop_ms

    return Row(
        trial_id=trial.trial_id,
        condition=cond.name,
        viewpoint=cell.viewpoint,
        detector=cell.detector,
        analysis_sr_user=user_sr,
        analysis_sr_agent=agent_sr,
        true_onset_ms=round(true_onset_ms, 4),
        true_cut_ms=round(true_cut_ms, 4),
        true_stop_ms=round(true_stop_ms, 4),
        det_onset_ms=None if onset_ms is None else round(onset_ms, 4),
        det_offset_ms=None if offset_ms is None else round(offset_ms, 4),
        meas_stop_ms=None if meas_stop is None else round(meas_stop, 4),
        onset_err_ms=None if onset_err is None else round(onset_err, 4),
        offset_err_ms=None if offset_err is None else round(offset_err, 4),
        stop_err_ms=None if stop_err is None else round(stop_err, 4),
        bulk_delay_ms=round(deg.bulk_delay_ms, 4),
        n_agent_segments=n_segs,
        status=status,
    )


def row_to_dict(r: Row) -> dict:
    return asdict(r)
