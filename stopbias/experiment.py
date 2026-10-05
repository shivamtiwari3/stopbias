"""Phase 1 runner: every trial x every condition x every viewpoint x every detector."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .audio import load_mono
from .conditions import CONDITIONS, Cell, Condition
from .measure import Row, degrade_trial, measure, row_to_dict
from .trials import build, load_plan


def run(
    plan_path: Path,
    out_path: Path,
    detectors: tuple[str, ...] = ("silero", "energy"),
    only: tuple[str, ...] | None = None,
    progress: bool = True,
) -> list[Row]:
    trials = load_plan(plan_path)
    conds: list[Condition] = [c for c in CONDITIONS if not only or c.name in only]
    rows: list[Row] = []
    t0 = time.time()
    total = len(trials)
    for i, trial in enumerate(trials, 1):
        user, _ = load_mono(trial.user_clip)
        agent, _ = load_mono(trial.agent_clip)
        agent_ch, user_ch = build(user, agent, trial.onset_s, trial.cut_s, trial.tail_s)
        cache: dict = {}  # per-trial; boundaries are shared across viewpoints
        for cond in conds:
            deg = degrade_trial(trial, agent_ch, user_ch, cond)
            for vp in cond.viewpoints:
                for det in detectors:
                    rows.append(measure(trial, agent_ch, user_ch, deg, Cell(cond, vp, det), cache))
        if progress:
            el = time.time() - t0
            eta = el / i * (total - i)
            print(f"\r  trial {i}/{total}  rows={len(rows)}  elapsed={el:6.1f}s  eta={eta:6.1f}s",
                  end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(row_to_dict(r)) + "\n")
    return rows
