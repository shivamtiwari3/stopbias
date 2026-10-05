"""Transport degradations: the difference between a 16 kHz file and a phone call.

Every op takes and returns (x, sr) so ops compose in a declared order. Ops that change
the sample rate say so in their return value; ops that add bulk delay do it explicitly.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from . import g711
from .audio import band_pass, from_int16, resample, to_int16

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def op_narrowband(x: np.ndarray, sr: int, target_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Resample to the telephony rate. Anti-aliased, so it also band-limits to 4 kHz."""
    return resample(x, sr, target_sr), target_sr


def op_pstn_band(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 3400.0) -> tuple[np.ndarray, int]:
    """The 300-3400 Hz passband a real PSTN leg imposes, including the high-pass."""
    return band_pass(x, sr, lo=lo, hi=hi), sr


def op_g711(x: np.ndarray, sr: int, law: str = "ulaw") -> tuple[np.ndarray, int]:
    """8-bit logarithmic quantisation. Raises the noise floor, which is what moves boundaries."""
    return from_int16(g711.roundtrip(to_int16(x), law)), sr


def op_frame_loss(
    x: np.ndarray,
    sr: int,
    loss_rate: float = 0.0,
    frame_ms: float = 20.0,
    conceal: str = "zero",
    seed: int = 0,
    burst_p: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Drop whole RTP frames.

    conceal='zero' is a gateway that emits silence for a lost frame; conceal='hold' is a
    zero-order-hold repeat of the previous frame, the crudest form of packet loss
    concealment. burst_p is the probability that a frame following a loss is also lost
    (a two-state Gilbert model); 0 gives independent Bernoulli loss.
    """
    if loss_rate <= 0.0:
        return x, sr
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    rng = np.random.default_rng(seed)
    nframes = int(np.ceil(x.size / n))
    lost = np.zeros(nframes, dtype=bool)
    prev = False
    for i in range(nframes):
        p = burst_p if (prev and burst_p > 0.0) else loss_rate
        prev = bool(rng.random() < p)
        lost[i] = prev
    y = x.copy()
    for i in np.flatnonzero(lost):
        a, b = i * n, min((i + 1) * n, x.size)
        if conceal == "hold" and i > 0:
            src = y[(i - 1) * n : (i - 1) * n + (b - a)]
            y[a:b] = src if src.size == (b - a) else 0.0
        else:
            y[a:b] = 0.0
    return y, sr


def op_delay(x: np.ndarray, sr: int, delay_ms: float = 0.0) -> tuple[np.ndarray, int]:
    """A fixed playout/jitter-buffer delay: pure translation of every boundary."""
    n = int(round(sr * delay_ms / 1000.0))
    return (np.concatenate([np.zeros(n, dtype=np.float32), x]) if n > 0 else x), sr


def op_noise(x: np.ndarray, sr: int, snr_db: float = 30.0, seed: int = 0) -> tuple[np.ndarray, int]:
    """Additive white noise at a stated SNR against the signal's active-speech power."""
    active = x[np.abs(x) > (np.max(np.abs(x)) * 1e-3)] if x.size else x
    p_sig = float(np.mean(active.astype(np.float64) ** 2)) if active.size else 0.0
    if p_sig <= 0.0:
        return x, sr
    p_n = p_sig / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0.0, np.sqrt(p_n), x.size).astype(np.float32)).astype(np.float32), sr


def op_opus(
    x: np.ndarray,
    sr: int,
    bitrate: str = "24k",
    cutoff: int = 8000,
    frame_ms: int = 20,
    application: str = "voip",
) -> tuple[np.ndarray, int]:
    """Encode and decode through libopus at telephony settings, back at the same rate.

    Opus is the WebRTC path's codec. It resamples internally to 48 kHz, so this op
    exercises a real transcode chain rather than a simple quantiser.
    """
    pcm = to_int16(x).tobytes()
    enc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
         "-i", "pipe:0", "-c:a", "libopus", "-b:a", bitrate, "-application", application,
         "-frame_duration", str(frame_ms), "-cutoff", str(cutoff), "-f", "ogg", "pipe:1"],
        input=pcm, capture_output=True, check=True,
    )
    dec = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(sr), "-ac", "1", "pipe:1"],
        input=enc.stdout, capture_output=True, check=True,
    )
    y = from_int16(np.frombuffer(dec.stdout, dtype="<i2"))
    # Keep length comparable to the input; Opus framing can pad the tail.
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    return y[: x.size].astype(np.float32), sr


OPS = {
    "narrowband": op_narrowband,
    "pstn_band": op_pstn_band,
    "g711": op_g711,
    "frame_loss": op_frame_loss,
    "delay": op_delay,
    "noise": op_noise,
    "opus": op_opus,
}


def apply_chain(x: np.ndarray, sr: int, chain: tuple, seed: int = 0) -> tuple[np.ndarray, int]:
    """Apply a declared op chain. `seed` is threaded into stochastic ops for reproducibility."""
    for name, kwargs in chain:
        fn = OPS[name]
        kw = dict(kwargs)
        if name in ("frame_loss", "noise"):
            kw.setdefault("seed", seed)
        x, sr = fn(x, sr, **kw)
    return x, sr
