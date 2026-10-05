"""Figures. One question per figure; no decoration that does not carry information."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .conditions import BY_NAME, REFERENCE  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

ORDER = [c for c in BY_NAME]
SHORT = {
    "ref_16k": "16 kHz file",
    "nb8_ulaw": "µ-law 8k\n(VAD@8k)",
    "nb8_ulaw_up16": "µ-law 8k\n(VAD@16k)",
    "nb8_alaw": "A-law 8k",
    "nb8_ulaw_band": "µ-law\n+300-3400",
    "nb8_ulaw_loss1": "+1% loss",
    "nb8_ulaw_loss3": "+3% loss",
    "nb8_ulaw_loss5": "+5% loss",
    "nb8_ulaw_loss5_burst": "+5% burst",
    "nb8_ulaw_loss5_hold": "+5% loss\n(PLC hold)",
    "nb8_ulaw_jb60": "+60 ms JB\n(control)",
    "opus_nb_12k": "Opus 12k\nNB",
    "opus_wb_24k": "Opus 24k\nWB",
    "nb8_ulaw_snr20": "µ-law\n+20 dB SNR",
    "pstn_typical": "full PSTN\n(typical)",
    "pstn_poor": "full PSTN\n(poor)",
}


def _order(df: pd.DataFrame) -> list[str]:
    return [c for c in ORDER if c in set(df.condition)]


def fig_stop_error(df: pd.DataFrame, out: Path) -> Path:
    dets = sorted(df.detector.unique())
    fig, axes = plt.subplots(len(dets), 1, figsize=(10, 3.2 * len(dets)), sharex=True)
    axes = np.atleast_1d(axes)
    conds = _order(df)
    for ax, det in zip(axes, dets):
        sub = df[(df.detector == det) & (df.viewpoint == "wire") & (df.status == "ok")]
        data = [sub[sub.condition == c]["stop_err_ms"].dropna().to_numpy() for c in conds]
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="black", lw=1.4))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#c9d9ec" if conds[i] != REFERENCE else "#e6e6e6")
            patch.set_edgecolor("#5a6b7d")
        for i, d in enumerate(data, start=1):
            if d.size:
                ax.plot(np.full(d.size, i) + np.linspace(-0.18, 0.18, d.size), d, ".",
                        ms=2.5, color="#2b3a4a", alpha=0.45, zorder=3)
        ax.axhline(0, color="crimson", lw=1.0, ls="--", zorder=1)
        ax.set_ylabel("stop-latency error (ms)")
        ax.set_title(f"detector = {det}   (wire viewpoint)", loc="left", fontsize=9)
    axes[-1].set_xticks(range(1, len(conds) + 1))
    axes[-1].set_xticklabels([SHORT.get(c, c) for c in conds], fontsize=7.5)
    fig.suptitle("Measured minus true stop latency, by transport condition", y=0.995, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_boundary_decomposition(df: pd.DataFrame, out: Path) -> Path:
    """stop_err = offset_err - onset_err. Show the two terms so cancellation is visible."""
    dets = sorted(df.detector.unique())
    conds = _order(df)
    fig, axes = plt.subplots(len(dets), 1, figsize=(10, 3.0 * len(dets)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(conds))
    for ax, det in zip(axes, dets):
        sub = df[(df.detector == det) & (df.viewpoint == "wire") & (df.status == "ok")]
        on = [np.median(sub[sub.condition == c]["onset_err_ms"].dropna()) if (sub.condition == c).any() else np.nan for c in conds]
        off = [np.median(sub[sub.condition == c]["offset_err_ms"].dropna()) if (sub.condition == c).any() else np.nan for c in conds]
        stop = [np.median(sub[sub.condition == c]["stop_err_ms"].dropna()) if (sub.condition == c).any() else np.nan for c in conds]
        ax.bar(x - 0.26, on, 0.25, label="user onset error", color="#7fa8d0")
        ax.bar(x, off, 0.25, label="agent offset error", color="#d08f7f")
        ax.bar(x + 0.26, stop, 0.25, label="stop-latency error", color="#4a5c6e")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("median error (ms)")
        ax.set_title(f"detector = {det}   (wire viewpoint)", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, ncols=3, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([SHORT.get(c, c) for c in conds], fontsize=7.5)
    fig.suptitle("Where the error comes from: onset, offset, and their difference", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_viewpoints(df: pd.DataFrame, out: Path) -> Path:
    """Does knowing your own emission time rescue the measurement?"""
    conds = [c for c in _order(df) if c != REFERENCE]
    dets = sorted(df.detector.unique())
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, det in zip(axes, dets):
        sub = df[(df.detector == det) & (df.status == "ok")]
        w = [np.median(sub[(sub.condition == c) & (sub.viewpoint == "wire")]["stop_err_ms"].dropna()) for c in conds]
        lu = [np.median(sub[(sub.condition == c) & (sub.viewpoint == "local_user")]["stop_err_ms"].dropna()) for c in conds]
        y = np.arange(len(conds))
        for i, (a, b) in enumerate(zip(w, lu)):
            ax.plot([a, b], [i, i], "-", color="#b0b8c0", lw=1.2, zorder=1)
        ax.plot(w, y, "o", ms=5, color="#2f6fb0", label="wire (both channels degraded)", zorder=3)
        ax.plot(lu, y, "s", ms=5, color="#b0602f", label="local_user (own copy of user audio)", zorder=3)
        ax.axvline(0, color="crimson", lw=1.0, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(c, c).replace("\n", " ") for c in conds], fontsize=7.5)
        ax.set_xlabel("median stop-latency error (ms)")
        ax.set_title(f"detector = {det}", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Measurement viewpoint: does a local copy of the user audio remove the bias?", y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_true_vs_measured(df: pd.DataFrame, out: Path, detector: str = "energy") -> Path:
    """Is the bias a constant offset or does it scale with the latency being measured?"""
    show = [REFERENCE, "nb8_ulaw", "opus_nb_12k", "pstn_poor"]
    show = [c for c in show if c in set(df.condition)]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = ["#7a7a7a", "#2f6fb0", "#b0602f", "#3f8f5a"]
    sub = df[(df.detector == detector) & (df.viewpoint == "wire") & (df.status == "ok")]
    for c, col in zip(show, colors):
        g = sub[sub.condition == c]
        ax.plot(g.true_stop_ms, g.meas_stop_ms, "o", ms=3.5, alpha=0.7, color=col,
                label=SHORT.get(c, c).replace("\n", " "))
    lim = [0, float(np.nanmax(sub.true_stop_ms)) * 1.15]
    ax.plot(lim, lim, "--", color="crimson", lw=1.0, label="perfect measurement")
    ax.set_xlim(lim)
    ax.set_xlabel("true stop latency (ms)")
    ax.set_ylabel("measured stop latency (ms)")
    ax.set_title(f"detector = {detector}, wire viewpoint", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all(raw_jsonl: Path, out_dir: Path) -> list[Path]:
    df = pd.read_json(Path(raw_jsonl), lines=True)
    out_dir = Path(out_dir)
    return [
        fig_stop_error(df, out_dir / "fig1_stop_error_by_condition.png"),
        fig_boundary_decomposition(df, out_dir / "fig2_boundary_decomposition.png"),
        fig_viewpoints(df, out_dir / "fig3_viewpoints.png"),
        fig_true_vs_measured(df, out_dir / "fig4_true_vs_measured.png"),
    ]
