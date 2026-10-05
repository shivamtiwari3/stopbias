"""Aggregation, preregistered.

Two headline quantities per cell:

  absolute error   stop_err = measured stop latency - true stop latency, in ms.
                   This is the number a benchmark reader cares about: how wrong is the
                   published figure?

  relative bias    d = stop_err(condition) - stop_err(reference), paired within trial.
                   This is the number an implementer cares about: what does moving from an
                   offline 16 kHz file to this transport do to my measurement?

Both are summarised by the median, because stop-latency distributions are skewed and the
prior art reports medians. Confidence intervals are percentile bootstraps over trials, and
the paired test is Wilcoxon signed-rank, which assumes neither normality nor symmetry of
the underlying latencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conditions import BY_NAME, REFERENCE

N_BOOT = 10_000
BOOT_SEED = 20260825


def load(path: Path) -> pd.DataFrame:
    return pd.read_json(Path(path), lines=True)


def boot_median_ci(x: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = BOOT_SEED) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    return (float(np.percentile(meds, 100 * alpha / 2)), float(np.percentile(meds, 100 * (1 - alpha / 2))))


def _wilcoxon(d: np.ndarray) -> tuple[float, float]:
    from scipy.stats import wilcoxon

    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return (float("nan"), float("nan"))
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return (float(res.statistic), float(res.pvalue))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (detector, viewpoint, condition)."""
    out = []
    for det, dsub in df.groupby("detector", sort=True):
        ref = dsub[(dsub.condition == REFERENCE) & (dsub.viewpoint == "wire")]
        ref_by_trial = ref.set_index("trial_id")["stop_err_ms"]
        for (vp, cond), g in dsub.groupby(["viewpoint", "condition"], sort=True):
            n_all = len(g)
            ok = g[g.status == "ok"]
            se = ok["stop_err_ms"].to_numpy(dtype=float)
            on = ok["onset_err_ms"].to_numpy(dtype=float)
            off = ok["offset_err_ms"].to_numpy(dtype=float)

            joined = ok.set_index("trial_id")["stop_err_ms"].sub(ref_by_trial, fill_value=np.nan).dropna()
            d = joined.to_numpy(dtype=float)
            lo, hi = boot_median_ci(se)
            dlo, dhi = boot_median_ci(d)
            _w, p = _wilcoxon(d)

            out.append({
                "detector": det,
                "viewpoint": vp,
                "condition": cond,
                "label": BY_NAME[cond].label if cond in BY_NAME else cond,
                "n": n_all,
                "n_ok": len(ok),
                "miss_rate": round(1.0 - len(ok) / n_all, 4) if n_all else np.nan,
                "bulk_delay_ms": round(float(np.median(ok["bulk_delay_ms"])), 2) if len(ok) else np.nan,
                "stop_err_median_ms": round(float(np.median(se)), 2) if se.size else np.nan,
                "stop_err_ci_lo": round(lo, 2),
                "stop_err_ci_hi": round(hi, 2),
                "stop_err_iqr_ms": round(float(np.percentile(se, 75) - np.percentile(se, 25)), 2) if se.size else np.nan,
                "stop_abs_err_p95_ms": round(float(np.percentile(np.abs(se), 95)), 2) if se.size else np.nan,
                "onset_err_median_ms": round(float(np.median(on)), 2) if on.size else np.nan,
                "offset_err_median_ms": round(float(np.median(off)), 2) if off.size else np.nan,
                "rel_bias_median_ms": round(float(np.median(d)), 2) if d.size else np.nan,
                "rel_bias_ci_lo": round(dlo, 2),
                "rel_bias_ci_hi": round(dhi, 2),
                "wilcoxon_p": (float(f"{p:.3g}") if np.isfinite(p) else np.nan),
                "n_paired": int(d.size),
                "agent_segments_median": float(np.median(ok["n_agent_segments"])) if len(ok) else np.nan,
            })
    res = pd.DataFrame(out)
    order = {c: i for i, c in enumerate(BY_NAME)}
    res["_o"] = res["condition"].map(order).fillna(999)
    return res.sort_values(["detector", "viewpoint", "_o"]).drop(columns="_o").reset_index(drop=True)


def markdown_table(s: pd.DataFrame, detector: str, viewpoint: str) -> str:
    sub = s[(s.detector == detector) & (s.viewpoint == viewpoint)]
    if sub.empty:
        return f"_no rows for {detector} / {viewpoint}_\n"
    head = ("| Condition | n | miss | bulk delay | onset err | offset err | **stop err (95% CI)** | IQR | P95 abs | "
            "vs 16 kHz | p |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    lines = []
    for _, r in sub.iterrows():
        rel = "—" if r.condition == REFERENCE else f"{r.rel_bias_median_ms:+.1f} [{r.rel_bias_ci_lo:+.1f}, {r.rel_bias_ci_hi:+.1f}]"
        p = "—" if r.condition == REFERENCE or not np.isfinite(r.wilcoxon_p) else f"{r.wilcoxon_p:.2g}"
        lines.append(
            f"| {r.label} | {r.n_ok} | {r.miss_rate:.0%} | {r.bulk_delay_ms:+.1f} | {r.onset_err_median_ms:+.1f} | "
            f"{r.offset_err_median_ms:+.1f} | **{r.stop_err_median_ms:+.1f}** [{r.stop_err_ci_lo:+.1f}, {r.stop_err_ci_hi:+.1f}] | "
            f"{r.stop_err_iqr_ms:.1f} | {r.stop_abs_err_p95_ms:.1f} | {rel} | {p} |"
        )
    return head + "\n".join(lines) + "\n\nAll figures in milliseconds. Positive = measured later than truth.\n"
