"""Power analysis for Phase 2, driven by Phase 1's measured error distributions.

Phase 2 costs money per trial, so the sample size has to be justified before spending it
rather than defended afterwards. The usual approach is to assume normality and plug an
estimated SD into a t-test formula. That is wrong here for two reasons Phase 1 measured
directly: the per-trial error distributions are heavy-tailed (P95 absolute error up to 3x the
IQR) and Silero's are quantised to a 32 ms frame grid, which makes a signed-rank test on
discrete data behave nothing like the normal approximation.

So power is estimated by simulation from the empirical Phase 1 residuals: resample observed
per-trial errors, add the effect being tested, and count how often the test rejects. The
resulting n inherits the real tail behaviour and the real quantisation.

The test mirrored here is the one the preregistration commits to: a paired Wilcoxon
signed-rank test on within-trial differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
TARGET_POWER = 0.80
N_SIM = 4000
SEED = 20260825

# Effect sizes worth detecting, in ms. 25 ms is roughly the granularity at which vendors
# currently claim to rank systems; 50 ms is the granularity at which a difference is
# plausibly audible in turn-taking; 100 ms is a difference nobody would dispute matters.
EFFECTS_MS = (10.0, 25.0, 50.0, 100.0)
N_GRID = (10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480)


@dataclass(frozen=True)
class PowerCell:
    detector: str
    viewpoint: str
    condition: str
    effect_ms: float
    n_required: int | None
    power_at_max_n: float
    iqr_ms: float
    n_source_trials: int


def _tie_term(ad: np.ndarray) -> np.ndarray:
    """sum(t^3 - t) over groups of tied values, per row. Needed because Silero's 32 ms
    quantisation produces heavy ties, which shrink the variance of the rank sum; ignoring it
    would overstate power."""
    n_sim, n = ad.shape
    s = np.sort(ad, axis=1)
    new = np.ones_like(s, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    gid = np.cumsum(new, axis=1) - 1
    flat = (np.arange(n_sim)[:, None] * n + gid).ravel()
    counts = np.bincount(flat, minlength=n_sim * n).reshape(n_sim, n).astype(np.float64)
    return np.sum(counts**3 - counts, axis=1)


def _wilcoxon_reject(d: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Vectorised two-sided Wilcoxon signed-rank test over rows of `d`, normal approximation
    with tie correction and `zero_method="wilcox"` (zeros dropped).

    Validated against scipy row by row in tests/test_power.py. Hand-rolled because the
    simulation needs ~10^6 tests and scipy's per-call overhead makes that intractable.

    Ranks are taken over |d| including zeros and then shifted down by the number of zeros in
    the row: since |d|=0 sorts smallest, excluding zeros only shifts the nonzero ranks by a
    per-row constant, which makes the whole thing vectorisable.
    """
    n_sim, n = d.shape
    ad = np.abs(d)
    ranks = stats.rankdata(ad, axis=1)
    z = np.count_nonzero(d == 0, axis=1)
    n_eff = (n - z).astype(np.float64)
    ranks = ranks - z[:, None]

    r_plus = np.sum(np.where(d > 0, ranks, 0.0), axis=1)
    r_minus = np.sum(np.where(d < 0, ranks, 0.0), axis=1)
    T = np.minimum(r_plus, r_minus)

    mn = n_eff * (n_eff + 1.0) / 4.0
    ties = _tie_term(ad) - (z.astype(np.float64) ** 3 - z)
    var = (n_eff * (n_eff + 1.0) * (2.0 * n_eff + 1.0) - 0.5 * ties) / 24.0

    ok = (n_eff >= 6) & (var > 0)
    zstat = np.zeros(n_sim, dtype=np.float64)
    np.divide(T - mn, np.sqrt(np.where(ok, var, 1.0)), out=zstat, where=ok)
    p = np.where(ok, 2.0 * stats.norm.sf(np.abs(zstat)), 1.0)
    return p < alpha


def simulate_power(errors: np.ndarray, effect_ms: float, n: int, rng: np.random.Generator,
                   n_sim: int = N_SIM) -> float:
    """Power to detect `effect_ms` with `n` paired trials.

    Each simulated trial draws a measurement error from the empirical distribution for each
    arm, so the paired difference carries the real spread of the *difference*, not of one arm.
    The effect is added to one arm.
    """
    if errors.size == 0:
        return float("nan")
    a = rng.choice(errors, size=(n_sim, n), replace=True)
    b = rng.choice(errors, size=(n_sim, n), replace=True) + effect_ms
    return float(np.mean(_wilcoxon_reject(b - a)))


def required_n(errors: np.ndarray, effect_ms: float, rng: np.random.Generator,
               target: float = TARGET_POWER) -> tuple[int | None, float]:
    """Smallest n on the grid reaching `target` power. Power is monotone in n up to simulation
    noise, so this bisects the grid rather than scanning it."""
    lo, hi = 0, len(N_GRID) - 1
    p_hi = simulate_power(errors, effect_ms, N_GRID[hi], rng)
    if p_hi < target:
        return None, p_hi
    best, best_p = hi, p_hi
    while lo <= hi:
        mid = (lo + hi) // 2
        p = simulate_power(errors, effect_ms, N_GRID[mid], rng)
        if p >= target:
            best, best_p = mid, p
            hi = mid - 1
        else:
            lo = mid + 1
    return N_GRID[best], best_p


def analyse(raw_jsonl: Path, conditions: tuple[str, ...] = ("ref_16k", "nb8_ulaw", "pstn_typical",
                                                            "pstn_poor")) -> pd.DataFrame:
    df = pd.read_json(Path(raw_jsonl), lines=True)
    df = df[(df.status == "ok") & df.stop_err_ms.notna()]
    rows: list[PowerCell] = []
    rng = np.random.default_rng(SEED)
    for det in sorted(df.detector.unique()):
        for vp in sorted(df.viewpoint.unique()):
            for cond in conditions:
                sub = df[(df.detector == det) & (df.viewpoint == vp) & (df.condition == cond)]
                if sub.empty:
                    continue
                e = sub.stop_err_ms.to_numpy(dtype=float)
                iqr = float(np.subtract(*np.percentile(e, [75, 25])))
                for eff in EFFECTS_MS:
                    n, p = required_n(e, eff, rng)
                    rows.append(PowerCell(det, vp, cond, eff, n, round(p, 3),
                                          round(iqr, 1), int(e.size)))
    return pd.DataFrame([r.__dict__ for r in rows])


def markdown(pw: pd.DataFrame) -> str:
    out = [
        "# Phase 2 sample size, simulated from Phase 1 residuals\n",
        f"Paired Wilcoxon signed-rank, alpha={ALPHA}, target power={TARGET_POWER:.0%}, "
        f"{N_SIM} simulations per point.\n",
        f"`n` is paired trials per arm. `--` means {max(N_GRID)} trials were not enough.\n",
    ]
    for det in sorted(pw.detector.unique()):
        for vp in sorted(pw[pw.detector == det].viewpoint.unique()):
            s = pw[(pw.detector == det) & (pw.viewpoint == vp)]
            out.append(f"\n## detector = `{det}`, viewpoint = `{vp}`\n\n")
            effs = sorted(s.effect_ms.unique())
            out.append("| condition | IQR | " + " | ".join(f"n for {e:.0f} ms" for e in effs) + " |\n")
            out.append("|---|--:|" + "--:|" * len(effs) + "\n")
            for cond in s.condition.unique():
                g = s[s.condition == cond].set_index("effect_ms")
                cells = []
                for e in effs:
                    n = g.loc[e, "n_required"]
                    cells.append("--" if pd.isna(n) else f"{int(n)}")
                out.append(f"| {cond} | {g.iloc[0].iqr_ms:.1f} | " + " | ".join(cells) + " |\n")
    return "".join(out)
