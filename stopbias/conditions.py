"""The transport conditions under test.

A condition is a declared op chain plus the rate the detector runs at plus the measurement
viewpoint. Nothing here is discovered at runtime: the grid is frozen so the analysis plan
can be preregistered.

Viewpoints
----------
wire       : both channels are read off the wire, so both are degraded. This is what a SIP
             capture harness sees, and what a remote observer of a phone call sees.
local_user : the measuring party emitted the user audio itself, so it knows the user onset
             from its own pristine 16 kHz copy, and only the returned agent audio is
             degraded. This is telnyx-onset's setup, and the question is how much of the
             bias it removes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REFERENCE = "ref_16k"


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    chain: tuple = ()
    analysis_sr: int = 16000
    viewpoints: tuple[str, ...] = ("wire", "local_user")
    expected_bulk_delay_ms: float = 0.0
    notes: str = ""


G711U = ("g711", {"law": "ulaw"})
G711A = ("g711", {"law": "alaw"})
NB = ("narrowband", {"target_sr": 8000})

CONDITIONS: tuple[Condition, ...] = (
    Condition(
        REFERENCE, "16 kHz PCM (offline file)", chain=(), analysis_sr=16000,
        viewpoints=("wire",),
        notes="What Full-Duplex-Bench measures on. The reference the field's numbers come from.",
    ),
    Condition(
        "nb8_ulaw", "8 kHz G.711 mu-law, VAD at 8 kHz", chain=(NB, G711U), analysis_sr=8000,
        notes="A PCMU telephony leg, analysed natively.",
    ),
    Condition(
        "nb8_ulaw_up16", "8 kHz G.711 mu-law, upsampled to 16 kHz before VAD",
        chain=(NB, G711U), analysis_sr=16000,
        notes="The common integration: narrowband wire, wideband model. Isolates the "
              "detector's rate from the transport's rate.",
    ),
    Condition(
        "nb8_alaw", "8 kHz G.711 A-law, VAD at 8 kHz", chain=(NB, G711A), analysis_sr=8000,
        notes="A-law has a finer floor near zero than mu-law; tests whether the law matters.",
    ),
    Condition(
        "nb8_ulaw_band", "8 kHz G.711 mu-law + 300-3400 Hz PSTN band",
        chain=(("pstn_band", {"lo": 300.0, "hi": 3400.0}), NB, G711U), analysis_sr=8000,
        notes="Adds the high-pass a real PSTN leg imposes, which removes onset energy below 300 Hz.",
    ),
    Condition(
        "nb8_ulaw_loss1", "8 kHz mu-law + 1% frame loss (zero-fill)",
        chain=(NB, G711U, ("frame_loss", {"loss_rate": 0.01, "frame_ms": 20.0, "conceal": "zero"})),
        analysis_sr=8000,
    ),
    Condition(
        "nb8_ulaw_loss3", "8 kHz mu-law + 3% frame loss (zero-fill)",
        chain=(NB, G711U, ("frame_loss", {"loss_rate": 0.03, "frame_ms": 20.0, "conceal": "zero"})),
        analysis_sr=8000,
    ),
    Condition(
        "nb8_ulaw_loss5", "8 kHz mu-law + 5% frame loss (zero-fill)",
        chain=(NB, G711U, ("frame_loss", {"loss_rate": 0.05, "frame_ms": 20.0, "conceal": "zero"})),
        analysis_sr=8000,
        notes="Zero-filled gaps mid-utterance can split a segment and move the measured offset early.",
    ),
    Condition(
        "nb8_ulaw_loss5_burst", "8 kHz mu-law + 5% bursty loss (Gilbert, zero-fill)",
        chain=(NB, G711U, ("frame_loss", {"loss_rate": 0.05, "frame_ms": 20.0, "conceal": "zero",
                                          "burst_p": 0.5})),
        analysis_sr=8000,
    ),
    Condition(
        "nb8_ulaw_loss5_hold", "8 kHz mu-law + 5% frame loss (zero-order-hold PLC)",
        chain=(NB, G711U, ("frame_loss", {"loss_rate": 0.05, "frame_ms": 20.0, "conceal": "hold"})),
        analysis_sr=8000,
        notes="Concealment fills gaps with energy, so it should protect the offset and can "
              "extend it past the true cut.",
    ),
    Condition(
        "nb8_ulaw_jb60", "8 kHz mu-law + 60 ms fixed jitter buffer",
        chain=(NB, G711U, ("delay", {"delay_ms": 60.0})), analysis_sr=8000,
        expected_bulk_delay_ms=60.0,
        notes="Validation condition. In the wire viewpoint both boundaries shift by 60 ms and "
              "the stop latency should be unchanged; in the local_user viewpoint only the "
              "agent boundary shifts and the bias should be +60 ms. If the harness does not "
              "recover that, the harness is wrong.",
    ),
    Condition(
        "opus_nb_12k", "Opus 12 kbps, 4 kHz cutoff, at 8 kHz", chain=(NB, ("opus", {"bitrate": "12k", "cutoff": 4000})),
        analysis_sr=8000,
        notes="A narrowband WebRTC leg.",
    ),
    Condition(
        "opus_wb_24k", "Opus 24 kbps, 8 kHz cutoff, at 16 kHz",
        chain=(("opus", {"bitrate": "24k", "cutoff": 8000}),), analysis_sr=16000,
        notes="A wideband WebRTC leg: keeps the rate, changes the waveform.",
    ),
    Condition(
        "nb8_ulaw_snr20", "8 kHz mu-law + 20 dB SNR line noise",
        chain=(("noise", {"snr_db": 20.0}), NB, G711U), analysis_sr=8000,
    ),
    # Composites. Every condition above varies one thing, which cannot answer whether the
    # single-impairment biases add. A real call applies all of them at once, so if they
    # interact, no amount of per-impairment characterisation predicts the field.
    Condition(
        "pstn_typical", "Full PSTN leg: band + mu-law + 1% loss + 40 ms jitter",
        chain=(("pstn_band", {"lo": 300.0, "hi": 3400.0}), NB, G711U,
               ("frame_loss", {"loss_rate": 0.01, "frame_ms": 20.0, "conceal": "hold"}),
               ("delay", {"delay_ms": 40.0})),
        analysis_sr=8000, expected_bulk_delay_ms=40.0,
        notes="A healthy carrier leg. Tests additivity against nb8_ulaw_band, "
              "nb8_ulaw_loss1 and nb8_ulaw_jb60 measured separately.",
    ),
    Condition(
        "pstn_poor", "Degraded PSTN leg: band + mu-law + 5% bursty loss + 80 ms jitter",
        chain=(("pstn_band", {"lo": 300.0, "hi": 3400.0}), NB, G711U,
               ("frame_loss", {"loss_rate": 0.05, "frame_ms": 20.0, "conceal": "zero",
                               "burst_p": 0.5}),
               ("delay", {"delay_ms": 80.0})),
        analysis_sr=8000, expected_bulk_delay_ms=80.0,
        notes="A congested mobile leg. The worst realistic case a deployed agent meets.",
    ),
)

BY_NAME = {c.name: c for c in CONDITIONS}


@dataclass(frozen=True)
class Cell:
    condition: Condition
    viewpoint: str
    detector: str
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.condition.name}|{self.viewpoint}|{self.detector}"


def grid(detectors: tuple[str, ...] = ("silero", "energy")) -> list[Cell]:
    return [
        Cell(c, vp, det)
        for c in CONDITIONS
        for vp in c.viewpoints
        for det in detectors
    ]
