"""Audio primitives. Everything downstream is float32 mono in [-1, 1]."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

EPS = 1e-12


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(x.mean(axis=1), dtype=np.float32), int(sr)


def save_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling. Deterministic, zero bulk delay (linear phase, compensated)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def from_int16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).astype(np.float32)


def band_pass(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 3400.0, order: int = 4) -> np.ndarray:
    """Zero-phase telephony band limit, so it adds no bulk delay of its own."""
    nyq = sr / 2.0
    hi = min(hi, nyq * 0.995)
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def peak_normalize(x: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < EPS:
        return x.astype(np.float32, copy=False)
    return (x * (10.0 ** (target_dbfs / 20.0) / peak)).astype(np.float32)


def frame_rms(x: np.ndarray, sr: int, win_ms: float = 20.0, hop_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_start_times_s, rms). Frame time is the frame's start, not its centre:
    a boundary question ("when did audio stop?") is about frame starts."""
    win = max(1, int(round(sr * win_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return (hop * np.arange(n) / sr).astype(np.float64), rms


def dbfs(rms: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(rms, EPS))


def energy_onset(x: np.ndarray, sr: int, rel_db: float = -40.0, win_ms: float = 5.0) -> float | None:
    """Detector-independent physical onset: first frame within `rel_db` of the clip peak.

    This is the ground-truth reference for trial construction. It is deliberately not a
    VAD, so VAD error can be measured against it without circularity.
    """
    t, rms = frame_rms(x, sr, win_ms=win_ms, hop_ms=win_ms / 2.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < EPS:
        return None
    thr = peak * 10.0 ** (rel_db / 20.0)
    hits = np.flatnonzero(rms >= thr)
    return float(t[hits[0]]) if hits.size else None


def estimate_bulk_delay(ref: np.ndarray, deg: np.ndarray, sr: int, max_ms: float = 200.0) -> float:
    """Cross-correlation lag (seconds) of `deg` relative to `ref`, positive = deg is later.

    Separates a codec's or buffer's constant delay from detector error.
    """
    n = min(ref.size, deg.size)
    a = ref[:n] - float(np.mean(ref[:n]))
    b = deg[:n] - float(np.mean(deg[:n]))
    max_lag = int(round(sr * max_ms / 1000.0))
    if n == 0 or max_lag == 0:
        return 0.0
    lags = np.arange(-max_lag, max_lag + 1)
    fa = np.fft.rfft(a, 2 * n)
    fb = np.fft.rfft(b, 2 * n)
    cc = np.fft.irfft(fb * np.conj(fa), 2 * n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    return float(lags[int(np.argmax(cc))]) / sr
