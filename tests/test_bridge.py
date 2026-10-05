"""The bridge, validated offline before it is put in the measurement path.

Same principle as `test_recover.py`: the bridge sits inside the quantity Phase 2 measures, so a
mistake in it produces a plausible wrong number rather than a visible failure. Everything here
runs for free, and the bridged-call tests run the real bridge code -- real packetisation, real
loss, real concealment, real transcode -- against a latency that was built in.

The load-bearing test is `test_buffer_depth_does_not_leak_into_the_model_viewpoint`. Everything
else supports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stopbias.audio import band_pass, resample
from stopbias.bridge import (
    PT_PCMA,
    PT_PCMU,
    BridgeConfig,
    CodecMismatch,
    FixedJitterBuffer,
    RtpPacket,
    Transcoder,
    accept_sdp_answer,
    packetise,
    rtp_decode,
    rtp_encode,
    sdp_offer,
)
from stopbias.bridgecall import measure, simulate_bridged

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"

needs_corpus = pytest.mark.skipif(
    not MANIFEST.exists(), reason="corpus not prepared; run `python -m stopbias.cli prep`"
)

# As in test_recover.py these bound total error, not bias: the energy detector's framing costs a
# fixed ~15-20 ms against a hard cut. Deliberately tighter than the 50 ms effects Phase 2 tests.
TOL_MS = 60.0
PSTN_BAND = (("pstn_band", {"lo": 300.0, "hi": 3400.0}),)
NOISY_PSTN = PSTN_BAND + (("noise", {"snr_db": 20.0}),)


def _clips() -> tuple[str, str]:
    man = json.loads(MANIFEST.read_text())
    agent = next(c["path"] for c in man["clips"] if c["role"] == "agent")
    user = next(c["path"] for c in man["clips"] if c["role"] == "user")
    return agent, user


def _tone(sr: int, dur_s: float, dbfs: float = -6.0, seed: int = 0) -> np.ndarray:
    """Band-limited noise: speech-like enough for the codec, deterministic enough to compare."""
    rng = np.random.default_rng(seed)
    x = band_pass(rng.normal(0, 1.0, int(sr * dur_s)).astype(np.float32), sr, 300.0, 3400.0)
    peak = float(np.max(np.abs(x)))
    return (x / peak * 10.0 ** (dbfs / 20.0)).astype(np.float32)


def _through_g711(x: np.ndarray, cfg: BridgeConfig) -> np.ndarray:
    """What the codec alone does to a signal, so the buffer's contribution can be isolated."""
    from stopbias.audio import from_int16, to_int16
    from stopbias.g711 import roundtrip
    return from_int16(roundtrip(to_int16(x), cfg.law))


# --- RTP wire format --------------------------------------------------------------------

def test_rtp_roundtrip_preserves_every_field():
    pkt = RtpPacket(seq=41234, timestamp=987654321, ssrc=0xDEADBEEF,
                    payload=bytes(range(160)), payload_type=PT_PCMA, marker=True)
    got = rtp_decode(rtp_encode(pkt))
    assert (got.seq, got.timestamp, got.ssrc, got.payload_type, got.marker) == \
           (pkt.seq, pkt.timestamp, pkt.ssrc, pkt.payload_type, pkt.marker)
    assert got.payload == pkt.payload


def test_rtp_decode_skips_csrcs_and_the_extension_header():
    """A mis-parsed header offset would shift the payload and every frame boundary with it, which
    is a timing error that looks like audio content."""
    payload = bytes(range(160))
    body = rtp_encode(RtpPacket(seq=7, timestamp=1600, ssrc=1, payload=payload))
    head = bytearray(body[:12])
    head[0] |= 0x02          # two CSRC identifiers
    head[0] |= 0x10          # and an extension header
    ext = bytes([0xBE, 0xDE, 0x00, 0x01]) + bytes(4)
    got = rtp_decode(bytes(head) + bytes(8) + ext + payload)
    assert got.payload == payload


def test_rtp_decode_strips_padding():
    payload = bytes(range(160))
    body = bytearray(rtp_encode(RtpPacket(seq=1, timestamp=0, ssrc=1,
                                          payload=payload + b"\x00\x00\x00\x04")))
    body[0] |= 0x20
    assert rtp_decode(bytes(body)).payload == payload


@pytest.mark.parametrize("bad", [b"", b"\x80\x00\x00", bytes([0x40] + [0] * 20)])
def test_rtp_rejects_what_it_cannot_parse(bad):
    """Refusing a malformed packet beats guessing at one: a silently mangled frame is a timing
    error nothing downstream can detect."""
    with pytest.raises(ValueError):
        rtp_decode(bad)


def test_packetise_makes_whole_frames_and_pads_the_tail():
    cfg = BridgeConfig()
    pkts = packetise(_tone(cfg.pstn_sr, 0.5)[: cfg.pstn_frame * 3 + 40], cfg)
    assert len(pkts) == 4, "a partial tail frame must be padded up, not dropped"
    assert all(len(p.payload) == cfg.pstn_frame for p in pkts)
    assert [p.seq for p in pkts] == [0, 1, 2, 3]
    assert [p.timestamp for p in pkts] == [0, 160, 320, 480]
    assert pkts[0].marker and not any(p.marker for p in pkts[1:])
    assert all(p.payload_type == PT_PCMU for p in pkts)


# --- the fixed jitter buffer ------------------------------------------------------------

def _drain(cfg: BridgeConfig, packets, arrivals=None):
    buf = FixedJitterBuffer(cfg)
    for i, p in enumerate(packets):
        buf.push(p, None if arrivals is None else arrivals[i])
    return buf.drain()


@pytest.mark.parametrize("depth", [0, 1, 2, 5])
def test_buffer_delay_is_exactly_its_depth(depth):
    """And the buffer contributes nothing else. The output is the input through G.711 and nothing
    more, checked by exact equality against the codec alone rather than by a tolerance -- a
    tolerance would hide a buffer that resampled, faded or reordered a frame."""
    cfg = BridgeConfig(jitter_frames=depth)
    x = _tone(cfg.pstn_sr, 1.0)
    out, stats = _drain(cfg, packetise(x, cfg))
    assert stats.added_delay_ms == pytest.approx(depth * cfg.frame_ms)
    lead = depth * cfg.pstn_frame
    assert np.all(out[:lead] == 0.0)
    np.testing.assert_array_equal(out[lead : lead + x.size], _through_g711(x, cfg))


def test_buffer_puts_reordered_packets_back_in_sequence():
    cfg = BridgeConfig(jitter_frames=3)
    x = _tone(cfg.pstn_sr, 0.4)
    pkts = packetise(x, cfg)
    shuffled = list(pkts)
    shuffled[1], shuffled[3] = shuffled[3], shuffled[1]
    in_order, _ = _drain(cfg, pkts)
    out_of_order, stats = _drain(cfg, shuffled)
    assert stats.reordered > 0, "the test did not actually reorder anything"
    assert stats.lost == 0 and stats.late == 0
    np.testing.assert_allclose(out_of_order, in_order, atol=1e-6)


def test_buffer_drops_duplicates_rather_than_playing_them_twice():
    cfg = BridgeConfig()
    pkts = packetise(_tone(cfg.pstn_sr, 0.2), cfg)
    out, stats = _drain(cfg, pkts + pkts[2:5])
    assert stats.duplicate == 3
    assert stats.expected == len(pkts)
    assert out.size == (len(pkts) + cfg.jitter_frames) * cfg.pstn_frame


def test_lost_packets_are_concealed_and_counted():
    cfg = BridgeConfig(conceal="zero")
    pkts = packetise(_tone(cfg.pstn_sr, 1.0), cfg)
    # Never drop the first packet: it establishes the base sequence number, so losing it shortens
    # the call rather than punching a hole in it, which is a different (and untestable) thing.
    kept = [p for i, p in enumerate(pkts) if i % 7 != 3]
    out, stats = _drain(cfg, kept)
    assert stats.lost == len(pkts) - len(kept) > 0
    assert stats.played == len(kept)
    assert stats.expected == len(pkts)
    assert stats.concealed_fraction == pytest.approx(stats.lost / stats.expected)
    # A lost frame under zero-fill really is silent, so the gap is where it was expected.
    gap = (cfg.jitter_frames + 3) * cfg.pstn_frame
    assert np.all(out[gap : gap + cfg.pstn_frame] == 0.0)


def test_hold_concealment_repeats_the_previous_frame():
    cfg = BridgeConfig(conceal="hold")
    pkts = packetise(_tone(cfg.pstn_sr, 0.5), cfg)
    out, _ = _drain(cfg, [p for i, p in enumerate(pkts) if i != 4])
    n, lead = cfg.pstn_frame, cfg.jitter_frames * cfg.pstn_frame
    prev = out[lead + 3 * n : lead + 4 * n]
    held = out[lead + 4 * n : lead + 5 * n]
    np.testing.assert_allclose(held, prev, atol=1e-6)


def test_a_late_packet_is_concealed_and_the_delay_does_not_change():
    """Rule 2. An adaptive buffer would grow to accommodate the late packet, and its delay would
    then vary with network conditions -- which are the treatment. This one refuses, so lateness
    shows up as a countable concealed frame and the constant stays constant."""
    cfg = BridgeConfig(jitter_frames=2)
    pkts = packetise(_tone(cfg.pstn_sr, 0.6), cfg)
    arrivals = [i * cfg.frame_s for i in range(len(pkts))]
    arrivals[5] += 0.100  # 100 ms late, far beyond the 40 ms buffer
    out, stats = _drain(cfg, pkts, arrivals)
    assert stats.late == 1 and stats.lost == 0
    assert stats.added_delay_ms == pytest.approx(cfg.inbound_delay_ms)
    assert out.size == (len(pkts) + cfg.jitter_frames) * cfg.pstn_frame


def test_jitter_within_the_buffer_costs_nothing():
    cfg = BridgeConfig(jitter_frames=2)
    pkts = packetise(_tone(cfg.pstn_sr, 0.6), cfg)
    clean, _ = _drain(cfg, pkts)
    jittered, stats = _drain(cfg, pkts, [i * cfg.frame_s + 0.015 for i in range(len(pkts))])
    assert stats.late == 0
    np.testing.assert_allclose(jittered, clean, atol=1e-6)


def test_sequence_wraparound_does_not_truncate_the_call():
    """Carriers start the sequence number at random (RFC 3550 S5.1), so a call beginning near
    65535 wraps about 20 s in. Reading that as a backwards jump would discard the rest of it --
    including, quite possibly, the barge-in."""
    cfg = BridgeConfig(jitter_frames=0)
    x = _tone(cfg.pstn_sr, 0.4)
    pkts = packetise(x, cfg, seq0=65530)
    assert [p.seq for p in pkts][:8] == [65530, 65531, 65532, 65533, 65534, 65535, 0, 1]
    out, stats = _drain(cfg, pkts)
    assert stats.expected == len(pkts) and stats.lost == 0 and stats.played == len(pkts)
    np.testing.assert_array_equal(out[: x.size], _through_g711(x, cfg))


def test_an_empty_stream_is_empty_not_an_exception():
    out, stats = _drain(BridgeConfig(), [])
    assert out.size == 0 and stats.expected == 0 and stats.concealed_fraction == 0.0


# --- rate conversion --------------------------------------------------------------------

def test_transcode_roundtrip_adds_no_bulk_delay():
    """Rule 4. A resampler with group delay would shift every boundary by an amount nothing in
    the measurement accounts for."""
    cfg = BridgeConfig()
    t = Transcoder(cfg)
    x = _tone(cfg.pstn_sr, 1.0)
    back = t.from_model(t.to_model(x))
    k = min(x.size, back.size)
    from stopbias.audio import estimate_bulk_delay
    assert estimate_bulk_delay(x[:k], back[:k], cfg.pstn_sr) == pytest.approx(0.0, abs=1e-4)
    err = float(np.sqrt(np.mean((back[:k] - x[:k]) ** 2)))
    assert err < 0.01 * float(np.sqrt(np.mean(x[:k] ** 2)))


def test_stateless_per_frame_resampling_is_worse_than_block_mode():
    """Why rule 4 says a streaming implementation must keep resampler state. Converting each 20 ms
    frame independently gives every frame boundary its own filter transient: measured here at
    ~10x the round-trip error, i.e. -30 dB against -51 dB. Pinned so that an implementation which
    resamples frame-at-a-time cannot pass for free."""
    cfg = BridgeConfig()
    t = Transcoder(cfg)
    x = _tone(cfg.pstn_sr, 2.0)
    block = t.from_model(t.to_model(x))

    up = np.concatenate([resample(x[i : i + cfg.pstn_frame], cfg.pstn_sr, cfg.model_sr)
                         for i in range(0, x.size, cfg.pstn_frame)])
    per_frame = np.concatenate([resample(up[i : i + cfg.model_frame], cfg.model_sr, cfg.pstn_sr)
                                for i in range(0, up.size, cfg.model_frame)])

    k = min(x.size, block.size, per_frame.size)
    e_block = float(np.sqrt(np.mean((block[:k] - x[:k]) ** 2)))
    e_frame = float(np.sqrt(np.mean((per_frame[:k] - x[:k]) ** 2)))
    assert e_frame > 5.0 * e_block, f"per-frame {e_frame:.2e} vs block {e_block:.2e}"


def test_bridge_does_not_gate_quiet_audio():
    """Rule 3, pinned from the outside because it is a rule about absent code. A bridge with DTX,
    comfort noise or a transmit VAD would insert a second endpointing decision into the path
    whose endpointing Phase 2 is measuring, and no analysis could separate the two."""
    cfg = BridgeConfig()
    quiet = _tone(cfg.pstn_sr, 2.0, dbfs=-40.0, seed=2)
    out, stats = _drain(cfg, packetise(quiet, cfg))
    out = out[cfg.jitter_frames * cfg.pstn_frame :]
    k = min(quiet.size, out.size)
    rms_in = float(np.sqrt(np.mean(quiet[:k] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[:k] ** 2)))
    assert 20 * np.log10(rms_out / rms_in) == pytest.approx(0.0, abs=1.0)
    frames = out[: k - k % cfg.pstn_frame].reshape(-1, cfg.pstn_frame)
    assert not np.any(np.all(frames == 0.0, axis=1)), "a frame was suppressed to silence"
    assert stats.concealed == 0


# --- SDP: one codec, verified ------------------------------------------------------------

def _answer(pt: int = PT_PCMU, name: str = "PCMU/8000", port: int = 40002,
            ptime: int | None = 20) -> str:
    lines = ["v=0", "o=carrier 1 1 IN IP4 203.0.113.9", "s=-", "c=IN IP4 203.0.113.9", "t=0 0",
             f"m=audio {port} RTP/AVP {pt}", f"a=rtpmap:{pt} {name}"]
    if ptime is not None:
        lines.append(f"a=ptime:{ptime}")
    return "\r\n".join(lines + ["a=sendrecv", ""])


def test_offer_names_exactly_one_codec():
    offer = sdp_offer("198.51.100.7", 40000)
    m = next(li for li in offer.split("\r\n") if li.startswith("m=audio"))
    assert m == "m=audio 40000 RTP/AVP 0", "offering more than one codec invites renegotiation"
    assert "a=rtpmap:0 PCMU/8000" in offer and "a=ptime:20" in offer


def test_a_matching_answer_is_accepted():
    assert accept_sdp_answer(_answer()) == ("203.0.113.9", 40002)


def test_alaw_config_offers_and_accepts_alaw():
    cfg = BridgeConfig(law="alaw")
    assert "RTP/AVP 8" in sdp_offer("198.51.100.7", 40000, cfg)
    assert accept_sdp_answer(_answer(PT_PCMA, "PCMA/8000"), cfg) == ("203.0.113.9", 40002)


@pytest.mark.parametrize("pt,name", [(9, "G722/8000"), (111, "opus/48000/2"), (8, "PCMA/8000")])
def test_an_answer_that_switched_codec_is_refused(pt, name):
    """The failure this exists to prevent: a carrier answers G.722, the cell runs a wideband leg,
    and every number downstream still looks entirely reasonable."""
    with pytest.raises(CodecMismatch):
        accept_sdp_answer(_answer(pt, name))


def test_an_answer_that_changed_the_packet_time_is_refused():
    with pytest.raises(CodecMismatch):
        accept_sdp_answer(_answer(ptime=30))


def test_an_answer_that_declined_audio_is_refused():
    with pytest.raises(ValueError):
        accept_sdp_answer(_answer(port=0))


def test_an_answer_that_relabels_the_payload_type_is_refused():
    """PT 0 is statically assigned to PCMU, but an answer that maps it to something else is either
    broken or hostile; either way it is not the preregistered transport."""
    with pytest.raises(CodecMismatch):
        accept_sdp_answer(_answer(PT_PCMU, "G722/8000"))


# --- bridged calls, end to end -----------------------------------------------------------

@needs_corpus
@pytest.mark.parametrize("true_ms", [200.0, 500.0, 900.0])
def test_model_viewpoint_recovers_the_true_latency(true_ms):
    agent, user = _clips()
    c = simulate_bridged(agent, user, true_ms, d_out_ms=40.0)
    r = measure(c, "energy", "model")
    assert r.status == "ok"
    assert r.stop_latency_ms == pytest.approx(true_ms, abs=TOL_MS)


@needs_corpus
@pytest.mark.parametrize("depth", [0, 1, 2, 4, 6])
def test_buffer_depth_does_not_leak_into_the_model_viewpoint(depth):
    """Rule 1, and the reason this module exists.

    The bridge is present in exactly one cell of the design -- speech-native over PSTN -- so any
    delay it contributes to the measurement is indistinguishable from that cell's transport
    effect. Measured across buffer depths 0 to 6 frames, i.e. 20 to 140 ms of bridge delay:

        depth  D_bridge   model error   sip error   sip - model
            0     20 ms       +15 ms      +35 ms       20 ms
            1     40 ms       +20 ms      +60 ms       40 ms
            2     60 ms       +20 ms      +80 ms       60 ms
            4    100 ms       +15 ms     +115 ms      100 ms
            6    140 ms       +15 ms     +155 ms      140 ms

    All 140 ms lands in the SIP viewpoint and none of it in the model viewpoint. At the default
    depth that is 60 ms of pure fabrication, against the 50 ms effect H1 is powered to detect --
    so tapping the wrong interface would not merely add noise, it would manufacture H1.
    """
    agent, user = _clips()
    cfg = BridgeConfig(jitter_frames=depth)
    c = simulate_bridged(agent, user, 500.0, cfg=cfg, d_out_ms=40.0, seed=3)

    model = measure(c, "energy", "model")
    sip = measure(c, "energy", "sip")
    assert model.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS), "bridge delay leaked"
    assert sip.stop_latency_ms == pytest.approx(500.0 + c.bridge_delay_ms, abs=TOL_MS)
    # The two viewpoints differ by the bridge's own constant, which is therefore measurable per
    # call without trusting this docstring's arithmetic.
    assert sip.stop_latency_ms - model.stop_latency_ms == pytest.approx(c.bridge_delay_ms, abs=6.0)


@needs_corpus
def test_silero_cannot_resolve_the_bridge_delay_so_the_check_uses_energy():
    """Phase 1 S9 measured Silero's 32 ms frame quantisation, and it lands here: the per-call
    D_bridge check misses a true 60 ms by up to a frame in either direction with Silero -- 40 ms on
    this trial, a median of 72 ms across `results/phase2_bridge.md` -- while energy returns exactly
    60 ms on every leg. Same reason S4 of the preregistration makes energy the primary detector,
    showing up in a second place. Silero remains a valid robustness check on the latency itself,
    which is the quantity it is quantised relative to rather than comparable in size to."""
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, d_out_ms=40.0, seed=3)
    e = measure(c, "energy", "sip").stop_latency_ms - measure(c, "energy", "model").stop_latency_ms
    s = measure(c, "silero", "sip").stop_latency_ms - measure(c, "silero", "model").stop_latency_ms
    assert e == pytest.approx(c.bridge_delay_ms, abs=6.0)
    assert abs(s - c.bridge_delay_ms) > 10.0, \
        "Silero resolved it after all -- if so, update PREREGISTRATION.md S4 and this test"


@needs_corpus
@pytest.mark.parametrize("kwargs,label", [
    ({}, "clean"),
    ({"carrier_ops": PSTN_BAND}, "300-3400 Hz passband"),
    ({"carrier_ops": NOISY_PSTN}, "passband + 20 dB SNR"),
    ({"loss_rate": 0.05, "burst_p": 0.4}, "5% bursty loss"),
    ({"jitter_ms": 15.0}, "15 ms jitter, inside the buffer"),
])
def test_the_model_viewpoint_holds_over_every_carrier_leg(kwargs, label):
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, d_out_ms=40.0, seed=3, **kwargs)
    r = measure(c, "energy", "model")
    assert r.status == "ok", label
    assert r.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS), label


@needs_corpus
def test_jitter_that_overruns_the_buffer_costs_frames_but_not_delay():
    """80 ms of jitter against a 40 ms buffer: 42% of frames arrive too late to play and are
    concealed. That is a wrecked call and the concealment rate says so -- but the bridge's delay
    has not moved, so the measurement is still on a fixed constant rather than a drifting one."""
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, d_out_ms=40.0, jitter_ms=80.0, seed=3)
    stats = c.tap.inbound
    assert stats.late > 0 and stats.reordered > 0
    assert stats.concealed_fraction > 0.2
    assert stats.added_delay_ms == pytest.approx(c.cfg.inbound_delay_ms)
    r = measure(c, "energy", "model")
    assert r.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS)


@needs_corpus
@pytest.mark.parametrize("ppm", [-200.0, 200.0])
def test_clock_drift_is_bounded_and_negligible(ppm):
    """Why the bridge does not resample to correct drift: it does not need to, and correcting it
    means inserting or dropping samples inside the window being measured. Drift is a timescale
    error, so it costs `t * ppm` -- at 200 ppm and an interjection 2.5 s in, 0.5 ms. Measured at
    0.6 ms, three orders of magnitude below the effects in scope."""
    agent, user = _clips()
    ref = measure(simulate_bridged(agent, user, 500.0, d_out_ms=40.0, seed=3), "energy", "model")
    got = measure(simulate_bridged(agent, user, 500.0, d_out_ms=40.0, seed=3, clock_ppm=ppm),
                  "energy", "model")
    assert abs(got.stop_latency_ms - ref.stop_latency_ms) < 3.0


@needs_corpus
def test_the_model_viewpoint_is_immune_to_the_carrier_delay():
    """The same property Phase 1's `wire` viewpoint has, and the reason the harness never needs to
    know the route's one-way delay for this arm."""
    agent, user = _clips()
    got = []
    for d in (0.0, 60.0, 150.0):
        c = simulate_bridged(agent, user, 500.0, d_out_ms=d, seed=3)
        r = measure(c, "energy", "model")
        assert r.status == "ok"
        got.append(r.stop_latency_ms)
    assert max(got) - min(got) < 25.0, f"carrier delay leaked into the measurement: {got}"


@needs_corpus
@pytest.mark.parametrize("model_sr,who", [(16000, "Gemini Live"), (24000, "GPT-Realtime")])
def test_both_model_socket_rates_measure_the_same(model_sr, who):
    """The two speech-native systems under test speak different rates. If the bridge's rate
    conversion biased the measurement, the two would not be comparable and H3's ranking would be
    partly an artefact of the socket format."""
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, cfg=BridgeConfig(model_sr=model_sr),
                         d_out_ms=40.0, seed=3)
    r = measure(c, "energy", "model")
    assert r.stop_latency_ms == pytest.approx(500.0, abs=TOL_MS), who


@needs_corpus
def test_a_model_that_was_not_speaking_is_void_not_instant():
    """Same exclusion as `PREREGISTRATION.md` S7, enforced on the bridged path too."""
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, d_out_ms=40.0)
    killed = c.tap.model_agent.copy()
    killed[int(1.2 * c.cfg.model_sr) :] = 0.0
    tap = type(c.tap)(sip_user=c.tap.sip_user, sip_agent=c.tap.sip_agent,
                      model_user=c.tap.model_user, model_agent=killed,
                      pstn_sr=c.tap.pstn_sr, model_sr=c.tap.model_sr, inbound=c.tap.inbound)
    r = measure(type(c)(track=c.track, tap=tap, cfg=c.cfg,
                        true_stop_latency_ms=c.true_stop_latency_ms, d_out_ms=c.d_out_ms,
                        model_onset_s=c.model_onset_s), "energy", "model")
    assert r.status == "agent_not_speaking_at_interjection"
    assert r.stop_latency_ms is None


@needs_corpus
def test_the_bridge_tap_is_dual_channel_by_construction():
    """The gating risk that makes Phase 1's one-call experiment decisive does not exist for this
    cell: the bridge is the recorder, so the two parties are never mixed and `recover` never has
    to fall back to the path it refuses."""
    agent, user = _clips()
    c = simulate_bridged(agent, user, 500.0, d_out_ms=40.0)
    u, a = c.tap.dual_channel("model", 16000)
    assert u.size == a.size
    from stopbias.recover import locate
    inter = c.track.audio[int(2.5 * 16000) : int(2.5 * 16000) + int(0.5 * 16000)]
    _, corr_user = locate(inter, u, 16000)
    _, corr_agent = locate(inter, a, 16000)
    assert abs(corr_user) > 0.5, "the interjection must be findable in the user channel"
    # The agent channel is a different speaker, so its best match against the interjection is the
    # spurious correlation any two speech signals share (~0.18 here) -- not leakage. There is no
    # path by which the caller's audio reaches this channel, which is what makes the separation
    # structural rather than achieved. Scored comparatively for exactly the reason the mono-mix
    # `leak_corr` diagnostic was uninformative: the absolute number alone says little.
    assert abs(corr_agent) < 0.4 * abs(corr_user), \
        f"the caller leaked into the agent channel (user {corr_user:.3f}, agent {corr_agent:.3f})"


def test_configs_that_would_silently_misframe_are_rejected():
    with pytest.raises(ValueError):
        BridgeConfig(model_sr=11025, frame_ms=20.0)   # 220.5 samples, not a whole frame
    with pytest.raises(ValueError):
        BridgeConfig(law="g722")
    with pytest.raises(ValueError):
        BridgeConfig(conceal="interpolate")
    with pytest.raises(ValueError):
        BridgeConfig(jitter_frames=-1)
