"""Boundary detectors.

Two families, because the prior art splits along exactly this line:
  - Silero-VAD, as used by Full-Duplex-Bench to define t_stop on 16 kHz files.
  - A frame-energy threshold, which is what a call-capture harness uses to decide
    when returned audio actually stopped.

Both expose the same interface so a condition can be measured with either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .audio import dbfs, frame_rms

Segments = list[tuple[float, float]]

SILERO_RATES = (8000, 16000)


@lru_cache(maxsize=1)
def _silero_model():
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    return load_silero_vad()


def merge_segments(segs: Segments, gap_s: float) -> Segments:
    """Join segments separated by <= gap_s. Mirrors Full-Duplex-Bench's merge step."""
    if not segs:
        return []
    out = [tuple(segs[0])]
    for s, e in sorted(segs)[1:]:
        ps, pe = out[-1]
        if s - pe <= gap_s:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return [(float(s), float(e)) for s, e in out]


@dataclass(frozen=True)
class SileroDetector:
    """Silero-VAD boundaries.

    speech_pad_ms=0 by default: Silero's own default of 30 ms pads segments outward,
    which shifts a measured onset 30 ms early and a measured offset 30 ms late. That is
    fine for cutting speech out of audio and wrong for timing it.
    """

    name: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 100
    min_silence_ms: int = 100
    speech_pad_ms: int = 0

    def segments(self, x: np.ndarray, sr: int) -> Segments:
        if sr not in SILERO_RATES:
            raise ValueError(f"Silero supports {SILERO_RATES}, got {sr}")
        import torch
        from silero_vad import get_speech_timestamps

        model = _silero_model()
        model.reset_states()
        ts = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        return [(t["start"] / sr, t["end"] / sr) for t in ts]


@dataclass(frozen=True)
class EnergyDetector:
    """Frame-energy boundaries with a noise-floor-relative threshold.

    The floor is estimated from the signal's own quiet frames, so a codec that raises the
    noise floor raises the threshold with it. That is the honest version of this detector:
    a harness that hard-codes an absolute dBFS threshold does strictly worse.
    """

    name: str = "energy"
    margin_db: float = 12.0
    floor_pctile: float = 10.0
    win_ms: float = 20.0
    hop_ms: float = 5.0
    min_speech_ms: float = 100.0
    hangover_ms: float = 100.0
    abs_floor_dbfs: float = -70.0

    def segments(self, x: np.ndarray, sr: int) -> Segments:
        t, rms = frame_rms(x, sr, win_ms=self.win_ms, hop_ms=self.hop_ms)
        lvl = np.asarray(dbfs(rms), dtype=np.float64)
        floor = max(float(np.percentile(lvl, self.floor_pctile)), self.abs_floor_dbfs)
        thr = floor + self.margin_db
        active = lvl >= thr
        hang = max(1, int(round(self.hangover_ms / self.hop_ms)))
        segs: Segments = []
        i, n = 0, active.size
        while i < n:
            if not active[i]:
                i += 1
                continue
            start = i
            last = i
            j = i
            while j < n and (j - last) <= hang:
                if active[j]:
                    last = j
                j += 1
            end_t = float(t[last] + self.win_ms / 1000.0)
            if (end_t - float(t[start])) * 1000.0 >= self.min_speech_ms:
                segs.append((float(t[start]), end_t))
            i = last + 1
        return segs


@dataclass(frozen=True)
class DetectorSpec:
    """A detector plus the segment-merge policy used to read boundaries off it."""

    detector: object
    user_merge_gap_s: float = 0.6
    model_merge_gap_s: float = 0.5
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.detector.name

    def first_onset(self, x: np.ndarray, sr: int) -> float | None:
        segs = merge_segments(self.detector.segments(x, sr), self.user_merge_gap_s)
        return segs[0][0] if segs else None

    def last_offset(self, x: np.ndarray, sr: int) -> float | None:
        segs = merge_segments(self.detector.segments(x, sr), self.model_merge_gap_s)
        return segs[-1][1] if segs else None


DETECTORS: dict[str, DetectorSpec] = {
    "silero": DetectorSpec(SileroDetector()),
    "silero_fdb": DetectorSpec(  # Full-Duplex-Bench's configuration, for comparability
        SileroDetector(name="silero_fdb", min_speech_ms=250, min_silence_ms=100, speech_pad_ms=30)
    ),
    "energy": DetectorSpec(EnergyDetector()),
}
