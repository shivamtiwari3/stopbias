"""Phase 2 stimuli: what the harness emits into a live call.

Two signals, with different jobs.

`delay_probe` measures the transport delay of the call. It is a band-limited linear chirp, not
a tone and not a click: Phase 1's tests established that cross-correlation delay estimation is
ambiguous modulo the period for a tone (`test_tone_delay_is_ambiguous_modulo_its_period`), and
a click loses most of its energy to the 300-3400 Hz passband. A chirp sweeping the passband is
broadband, survives G.711, and has a sharp autocorrelation peak.

`interjection` is the barge-in stimulus. It reuses Phase 1's corpus clips, which are already
trimmed so their physical energy onset is sample 0 -- so the moment the harness starts playing
one is the moment user speech begins, with no separate onset estimate needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import chirp

from .audio import band_pass, peak_normalize

PROBE_MS = 250.0
PROBE_LO_HZ = 350.0
PROBE_HI_HZ = 3300.0


def delay_probe(sr: int, dur_ms: float = PROBE_MS, lo: float = PROBE_LO_HZ,
                hi: float = PROBE_HI_HZ, target_dbfs: float = -6.0) -> np.ndarray:
    """A band-limited up-chirp, raised-cosine windowed so it does not click.

    Kept inside 350-3300 Hz rather than 300-3400 so that a real carrier's passband edges do
    not eat the chirp's endpoints and skew the correlation peak.
    """
    n = max(1, int(round(sr * dur_ms / 1000.0)))
    t = np.arange(n) / float(sr)
    x = chirp(t, f0=lo, f1=min(hi, sr / 2.0 * 0.95), t1=t[-1] if n > 1 else 1.0, method="linear")
    win = np.hanning(n) if n > 1 else np.ones(1)
    return peak_normalize((x * win).astype(np.float32), target_dbfs)


@dataclass(frozen=True)
class CallerTrack:
    """The harness's outbound audio, with every event time known exactly by construction.

    `probe_start_s` and `interjection_start_s` are the ground truth the recovery code is
    checked against. In a live call the interjection time is chosen reactively (once the agent
    is heard speaking), so the runtime records the actual value here rather than assuming it.
    """

    audio: np.ndarray
    sr: int
    probe_start_s: float
    interjection_start_s: float
    interjection_dur_s: float

    @property
    def duration_s(self) -> float:
        return self.audio.size / float(self.sr)


def build_caller_track(interjection: np.ndarray, sr: int, interjection_at_s: float,
                       probe_at_s: float = 0.5, total_s: float | None = None,
                       probe_dbfs: float = -6.0, interjection_dbfs: float = -6.0) -> CallerTrack:
    """Assemble silence + probe + silence + interjection + silence.

    The probe sits early, while the agent is still silent or greeting, because the cancellation
    path in `recover` needs a window where the harness's own audio is the only thing present.
    """
    if interjection_at_s <= probe_at_s + PROBE_MS / 1000.0:
        raise ValueError("interjection must start after the probe finishes")
    probe = delay_probe(sr, target_dbfs=probe_dbfs)
    inter = peak_normalize(np.asarray(interjection, dtype=np.float32), interjection_dbfs)
    end_s = interjection_at_s + inter.size / sr + 1.0
    n = int(round(sr * (total_s if total_s is not None else end_s)))
    track = np.zeros(max(n, int(round(sr * end_s))), dtype=np.float32)

    p0 = int(round(sr * probe_at_s))
    track[p0 : p0 + probe.size] += probe[: max(0, track.size - p0)]
    i0 = int(round(sr * interjection_at_s))
    take = min(inter.size, track.size - i0)
    track[i0 : i0 + take] += inter[:take]

    return CallerTrack(
        audio=track, sr=sr,
        probe_start_s=p0 / sr,
        interjection_start_s=i0 / sr,
        interjection_dur_s=take / sr,
    )
