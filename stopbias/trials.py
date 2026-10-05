"""Trial construction.

A trial is a two-channel scene with exactly known ground truth:

    agent channel :  agent speech from t=0, hard-cut to digital silence at t_cut
    user  channel :  digital silence until t_onset, then the user interjection

    true stop latency  =  t_cut - t_onset

This is the same quantity Full-Duplex-Bench v1.5 defines as
`t_stop = t_model_stop - t_user_start`, except that here both terms are known by
construction instead of being estimated by a VAD. That is the whole point: it lets the
detector's error be measured rather than assumed.

The hard cut is the right model of a barge-in stop. A voice agent that yields does not fade
out; it invalidates its media queue and stops emitting frames, which is a discontinuity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import EPS, load_mono

SR = 16000


@dataclass(frozen=True)
class Trial:
    trial_id: str
    user_clip: str
    agent_clip: str
    sample_rate: int
    onset_s: float
    cut_s: float
    true_stop_latency_s: float
    duration_s: float
    tail_s: float


def _active_before(x: np.ndarray, sr: int, t: float, window_s: float = 0.05, rel_db: float = -30.0) -> bool:
    """Is the audio genuinely loud right up to `t`? A cut in a pause is not a stop event."""
    a = max(0, int(round((t - window_s) * sr)))
    b = min(x.size, int(round(t * sr)))
    if b - a < int(round(0.01 * sr)):
        return False
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < EPS:
        return False
    seg = x[a:b]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) >= peak * 10.0 ** (rel_db / 20.0)


def build(
    user: np.ndarray,
    agent: np.ndarray,
    onset_s: float,
    cut_s: float,
    tail_s: float = 1.0,
    sr: int = SR,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the two channels. Both come back the same length."""
    total = int(round((cut_s + tail_s) * sr))
    total = max(total, int(round((onset_s * sr))) + user.size)
    agent_ch = np.zeros(total, dtype=np.float32)
    ncut = int(round(cut_s * sr))
    take = min(ncut, agent.size)
    agent_ch[:take] = agent[:take]

    user_ch = np.zeros(total, dtype=np.float32)
    a = int(round(onset_s * sr))
    take_u = min(user.size, total - a)
    user_ch[a : a + take_u] = user[:take_u]
    return agent_ch, user_ch


def plan(
    manifest_path: Path,
    n_trials: int = 60,
    onset_range: tuple[float, float] = (1.00, 2.20),
    latency_range: tuple[float, float] = (0.20, 1.20),
    tail_s: float = 1.0,
    seed: int = 20260825,
) -> list[Trial]:
    """Pair clips 1:1 and draw (onset, true latency) on a fixed grid.

    Onset and true latency are drawn from a deterministic low-discrepancy sweep rather than
    at random, so the true-latency axis is covered evenly and the plan is reproducible
    without depending on a PRNG implementation.
    """
    man = json.loads(Path(manifest_path).read_text())
    users = [c for c in man["clips"] if c["role"] == "user"]
    agents = [c for c in man["clips"] if c["role"] == "agent"]
    n = min(n_trials, len(users), len(agents))

    trials: list[Trial] = []
    for i in range(n):
        u, g = users[i], agents[i]
        frac_o = ((i * 7) % n) / max(n - 1, 1)
        frac_l = ((i * 11) % n) / max(n - 1, 1)
        onset = onset_range[0] + frac_o * (onset_range[1] - onset_range[0])
        lat = latency_range[0] + frac_l * (latency_range[1] - latency_range[0])
        cut = onset + lat

        gx, _ = load_mono(g["path"])
        # The cut must land in loud agent speech, else the "stop" is not observable.
        if not _active_before(gx, SR, cut):
            for delta in np.arange(0.02, 0.60, 0.02):
                for cand in (cut + delta, cut - delta):
                    if cand > onset + 0.15 and _active_before(gx, SR, cand):
                        cut = float(cand)
                        break
                else:
                    continue
                break
        ux, _ = load_mono(u["path"])
        dur = max(cut + tail_s, onset + ux.size / SR)
        trials.append(Trial(
            trial_id=f"t{i:03d}", user_clip=u["path"], agent_clip=g["path"], sample_rate=SR,
            onset_s=round(float(onset), 6), cut_s=round(float(cut), 6),
            true_stop_latency_s=round(float(cut - onset), 6),
            duration_s=round(float(dur), 6), tail_s=tail_s,
        ))
    return trials


def save_plan(trials: list[Trial], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(t) for t in trials], indent=2))


def load_plan(path: Path) -> list[Trial]:
    return [Trial(**d) for d in json.loads(Path(path).read_text())]
