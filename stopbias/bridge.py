"""SIP<->WebSocket media bridge: Phase 2's fourth cell, and the hazards it introduces.

Phase 2 crosses 2 architectures with 2 transports. Novis (cascaded) is reachable both over a
socket and over the PSTN. GPT-Realtime and Gemini Live (speech-native) speak PCM over a
WebSocket and nothing else, so without a bridge the speech-native arm exists only on the socket
side, architecture and transport are perfectly confounded, and H2/H3 cannot be tested at all
(`PREREGISTRATION.md` S2 calls this out and says the bridge is not optional).

This module is that bridge's **media plane**: RTP on the carrier side, linear PCM on the model
side, and the timing bookkeeping in between. It is the part that can change the measurement, so
it is the part that is built and validated here. The **signalling plane** -- the SIP dialog,
registration, digest auth, re-INVITE -- is delegated to an existing stack (pjsua/baresip/
Asterisk), because it cannot be exercised without a carrier and getting it wrong fails loudly at
call setup rather than quietly at measurement time. What the delegate needs from this module is
`sdp_offer` and `accept_sdp_answer`; what it hands back is a UDP socket pair.

The bridge sits *inside the measured path*, which makes it a confound unless it is built to
specific rules. Each rule below exists because breaking it corrupts a Phase 2 number.

1. **The dual-channel tap is taken at the model-facing interface, not the SIP-facing one.**
   Both taps are recorded, and they measure different quantities. Writing D_in for the inbound
   delay and D_out for the outbound, with L the model's true stop latency:

       model-side tap:  user onset at D_in,          agent stop at D_in + L        -> L
       SIP-side tap:    user onset at 0,             agent stop at D_in + L + D_out -> L + D_bridge

   The bridge's own delay is common mode at the model interface and cancels exactly, the same
   argument that makes Phase 1's `wire` viewpoint unbiased. At the SIP interface it does not
   cancel, and it lands *entirely inside one cell of the design* -- the speech-native PSTN cell,
   the only cell with a bridge in it. At the default configuration that is 60 ms of pure
   fabrication in the same direction and of the same order as the 50 ms effect H1 is powered to
   detect. So the tap point is not a matter of convenience. `dual_channel("sip")` is retained
   because the difference between the two viewpoints *measures* D_bridge per call, which is a
   free validity check on this docstring's arithmetic.

2. **The jitter buffer is fixed-depth, never adaptive.** A fixed buffer contributes a constant
   delay, which cancels at the model tap and is subtractable at the SIP tap. An adaptive buffer
   contributes a time-varying delay that does neither, and it would vary *with network
   conditions* -- correlated with the treatment. Jitter beyond the buffer's depth is therefore
   absorbed as concealed frames, which are counted and reported, rather than as a silent change
   in delay.

3. **No voice activity detection, no DTX, no comfort noise, no gain control.** The bridge is
   transmit-always. Phase 2 measures when a model decides to stop talking; a bridge that made
   its own speech/silence decision would insert a second endpointer into the path being measured
   and there would be no way to attribute the result. This is a rule about what the module does
   *not* contain, so `test_bridge_does_not_gate_quiet_audio` pins it from the outside.

4. **Rate conversion is linear-phase and adds no bulk delay**, via `audio.resample`, so the
   timeline is not shifted by an unmeasured amount. A streaming implementation must keep
   resampler state across frames: converting each 20 ms frame independently is measurably worse
   (`test_stateless_per_frame_resampling_is_worse_than_block_mode`), because every frame boundary
   gets its own filter transient.

5. **One codec, verified at answer time.** `sdp_offer` offers exactly one codec and
   `accept_sdp_answer` raises if the answer picks anything else. A carrier that quietly answers
   G.722 would put a wideband leg in a cell preregistered as narrowband, and the measurement
   would still look fine.

Clock drift between the carrier's 8 kHz and the model socket's clock is *not* compensated
during a call. Compensation means inserting or dropping samples, which moves the very boundaries
being measured. Instead drift is left alone and bounded: it is a timescale error, so over the
~1 s that separates the interjection from the stop, 200 ppm is 0.2 ms -- three orders of
magnitude below the effects in scope. `bridgecall` measures this rather than asserting it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import g711
from .audio import from_int16, resample, to_int16

RTP_VERSION = 2
RTP_HEADER_BYTES = 12
PT_PCMU = 0
PT_PCMA = 8
SEQ_MOD = 1 << 16
FRAME_MS = 20.0
PSTN_SR = 8000
DEFAULT_JITTER_FRAMES = 2
MS = 1000.0


class CodecMismatch(ValueError):
    """The answer selected a codec the study did not preregister. Fail, do not measure."""


@dataclass(frozen=True)
class BridgeConfig:
    """Everything about the bridge that affects timing, in one auditable place.

    `model_sr` is the model socket's rate: GPT-Realtime speaks 24 kHz PCM16, Gemini Live 16 kHz.
    `jitter_frames` is the fixed playout depth in frames; 2 frames = 40 ms absorbs ordinary
    carrier jitter while keeping the constant it contributes small and known.
    """

    model_sr: int = 24000
    law: str = "ulaw"
    frame_ms: float = FRAME_MS
    jitter_frames: int = DEFAULT_JITTER_FRAMES
    conceal: str = "hold"
    pstn_sr: int = PSTN_SR
    ssrc: int = 0x5A11B00B

    def __post_init__(self) -> None:
        if self.law not in ("ulaw", "alaw"):
            raise ValueError(f"law must be 'ulaw' or 'alaw', got {self.law!r}")
        if self.conceal not in ("zero", "hold"):
            raise ValueError(f"conceal must be 'zero' or 'hold', got {self.conceal!r}")
        if self.jitter_frames < 0:
            raise ValueError("jitter_frames must be >= 0")
        for sr in (self.model_sr, self.pstn_sr):
            n = sr * self.frame_ms / MS
            if abs(n - round(n)) > 1e-9:
                raise ValueError(f"{self.frame_ms} ms is not a whole number of samples at {sr} Hz")

    @property
    def frame_s(self) -> float:
        return self.frame_ms / MS

    @property
    def pstn_frame(self) -> int:
        return int(round(self.pstn_sr * self.frame_s))

    @property
    def model_frame(self) -> int:
        return int(round(self.model_sr * self.frame_s))

    @property
    def payload_type(self) -> int:
        return PT_PCMU if self.law == "ulaw" else PT_PCMA

    @property
    def encoding_name(self) -> str:
        return "PCMU" if self.law == "ulaw" else "PCMA"

    @property
    def inbound_delay_ms(self) -> float:
        """Carrier -> model: the fixed playout buffer, and nothing else."""
        return self.jitter_frames * self.frame_ms

    @property
    def outbound_delay_ms(self) -> float:
        """Model -> carrier: aggregating the socket's PCM into whole RTP frames costs one frame."""
        return self.frame_ms

    @property
    def delay_ms(self) -> float:
        """The whole constant the bridge contributes, which the SIP-side tap carries and the
        model-side tap does not."""
        return self.inbound_delay_ms + self.outbound_delay_ms


# --- RTP wire format ------------------------------------------------------------------------

@dataclass(frozen=True)
class RtpPacket:
    seq: int
    timestamp: int
    ssrc: int
    payload: bytes
    payload_type: int = PT_PCMU
    marker: bool = False


def rtp_encode(pkt: RtpPacket) -> bytes:
    """Serialise a minimal RFC 3550 packet: no CSRCs, no extension, no padding."""
    b0 = RTP_VERSION << 6
    b1 = (0x80 if pkt.marker else 0x00) | (pkt.payload_type & 0x7F)
    head = struct.pack("!BBHII", b0, b1, pkt.seq & 0xFFFF,
                       pkt.timestamp & 0xFFFFFFFF, pkt.ssrc & 0xFFFFFFFF)
    return head + pkt.payload


def rtp_decode(data: bytes) -> RtpPacket:
    """Parse a packet from the wire, honouring the fields a real carrier actually sets.

    CSRC count, the extension header and the padding bit are all handled: they are rare on a
    carrier's G.711 stream but a mis-parse would silently shift the payload and every frame
    boundary with it.
    """
    if len(data) < RTP_HEADER_BYTES:
        raise ValueError(f"RTP packet too short: {len(data)} bytes")
    b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:RTP_HEADER_BYTES])
    if (b0 >> 6) != RTP_VERSION:
        raise ValueError(f"unsupported RTP version {b0 >> 6}")
    off = RTP_HEADER_BYTES + 4 * (b0 & 0x0F)
    if (b0 >> 4) & 0x01:
        if len(data) < off + 4:
            raise ValueError("truncated RTP extension header")
        words = struct.unpack("!H", data[off + 2 : off + 4])[0]
        off += 4 + 4 * words
    payload = data[off:]
    if (b0 >> 5) & 0x01 and payload:
        pad = payload[-1]
        if pad == 0 or pad > len(payload):
            raise ValueError("bad RTP padding length")
        payload = payload[:-pad]
    return RtpPacket(seq=seq, timestamp=ts, ssrc=ssrc, payload=payload,
                     payload_type=b1 & 0x7F, marker=bool(b1 >> 7))


def packetise(pcm: np.ndarray, cfg: BridgeConfig, seq0: int = 0,
              timestamp0: int = 0) -> list[RtpPacket]:
    """Cut float32 PCM at `cfg.pstn_sr` into G.711 RTP frames.

    The tail is zero-padded to a whole frame rather than dropped: a short final frame would make
    the stream's duration depend on its content, and durations are what this study measures.
    """
    x = np.asarray(pcm, dtype=np.float32)
    n = cfg.pstn_frame
    if x.size % n:
        x = np.pad(x, (0, n - x.size % n))
    codes = g711.encode(to_int16(x), cfg.law).astype(np.uint8).tobytes()
    return [
        RtpPacket(seq=(seq0 + i) % SEQ_MOD,
                  timestamp=(timestamp0 + i * n) % (1 << 32),
                  ssrc=cfg.ssrc,
                  payload=codes[i * n : (i + 1) * n],
                  payload_type=cfg.payload_type,
                  marker=(i == 0))
        for i in range(x.size // n)
    ]


# --- fixed-depth jitter buffer --------------------------------------------------------------

@dataclass(frozen=True)
class JitterStats:
    """What the buffer had to do, so a trial can be judged without re-reading the audio."""

    expected: int = 0
    played: int = 0
    lost: int = 0
    late: int = 0
    duplicate: int = 0
    reordered: int = 0
    depth_frames: int = 0
    added_delay_ms: float = 0.0

    @property
    def concealed(self) -> int:
        """Frames the model heard as concealment rather than as carrier audio."""
        return self.lost + self.late

    @property
    def concealed_fraction(self) -> float:
        return self.concealed / self.expected if self.expected else 0.0


class FixedJitterBuffer:
    """Reordering, de-duplicating playout buffer with a delay that never changes.

    Playout of the frame `k` frames after the first is due at `t(first) + (k + depth) * frame`.
    A frame that arrives after its own deadline is *late*, which is indistinguishable from lost
    as far as the model is concerned, so it is concealed and counted -- not played early, and not
    accommodated by growing the buffer. That refusal to adapt is the whole point (rule 2 in the
    module docstring): the delay stays a constant that cancels at the model tap.
    """

    def __init__(self, cfg: BridgeConfig) -> None:
        self.cfg = cfg
        self._payloads: dict[int, bytes] = {}
        self._arrivals: dict[int, float] = {}
        self._base: int | None = None
        self._base_arrival = 0.0
        self._top = 0
        self._wraps = 0
        self._last_raw: int | None = None
        self._duplicate = 0
        self._reordered = 0

    def _extend(self, seq: int) -> int:
        """Undo 16-bit sequence wraparound.

        Carriers start the sequence number at random (RFC 3550 S5.1), so a call beginning near
        65535 wraps about 20 s in. Treating the wrapped values as a huge backwards jump would
        discard the rest of the call.
        """
        if self._last_raw is not None:
            if seq < self._last_raw - SEQ_MOD // 2:
                self._wraps += 1
            elif seq > self._last_raw + SEQ_MOD // 2:
                self._wraps -= 1
        self._last_raw = seq
        return seq + SEQ_MOD * self._wraps

    def push(self, pkt: RtpPacket, arrival_s: float | None = None) -> None:
        """Offer a received packet. `arrival_s` defaults to its nominal on-time arrival."""
        seq = self._extend(pkt.seq)
        if self._base is None:
            self._base = seq
            self._base_arrival = 0.0 if arrival_s is None else float(arrival_s)
        nominal = self._base_arrival + (seq - self._base) * self.cfg.frame_s
        at = nominal if arrival_s is None else float(arrival_s)
        if seq in self._payloads:
            self._duplicate += 1
            return
        if seq < self._top:
            self._reordered += 1
        self._top = max(self._top, seq)
        self._payloads[seq] = pkt.payload
        self._arrivals[seq] = at

    def drain(self) -> tuple[np.ndarray, JitterStats]:
        """Produce the continuous stream the model hears, plus what it cost."""
        n = self.cfg.pstn_frame
        depth = self.cfg.jitter_frames
        if self._base is None:
            return np.zeros(0, dtype=np.float32), JitterStats(depth_frames=depth,
                                                              added_delay_ms=self.cfg.inbound_delay_ms)
        expected = self._top - self._base + 1
        out = [np.zeros(n * depth, dtype=np.float32)]  # the buffer's own fixed delay, made real
        prev = np.zeros(n, dtype=np.float32)
        lost = late = played = 0
        for k in range(expected):
            seq = self._base + k
            deadline = self._base_arrival + (k + depth) * self.cfg.frame_s
            payload = self._payloads.get(seq)
            if payload is None:
                lost += 1
                frame = prev if self.cfg.conceal == "hold" else np.zeros(n, dtype=np.float32)
            elif self._arrivals[seq] > deadline + 1e-9:
                late += 1
                frame = prev if self.cfg.conceal == "hold" else np.zeros(n, dtype=np.float32)
            else:
                played += 1
                frame = from_int16(g711.decode(np.frombuffer(payload, dtype=np.uint8), self.cfg.law))
                if frame.size != n:
                    frame = np.pad(frame, (0, max(0, n - frame.size)))[:n]
                prev = frame
            out.append(frame)
        return np.concatenate(out).astype(np.float32), JitterStats(
            expected=expected, played=played, lost=lost, late=late,
            duplicate=self._duplicate, reordered=self._reordered,
            depth_frames=depth, added_delay_ms=self.cfg.inbound_delay_ms,
        )


# --- rate conversion ------------------------------------------------------------------------

class Transcoder:
    """Rate conversion between the telephony leg and the model socket.

    Block mode: whole streams in, whole streams out, which is what a correctly stateful
    streaming resampler converges to and which adds no bulk delay. Doing it per frame without
    carrying filter state is not equivalent -- see rule 4 in the module docstring.
    """

    def __init__(self, cfg: BridgeConfig) -> None:
        self.cfg = cfg

    def to_model(self, pcm8: np.ndarray) -> np.ndarray:
        return resample(np.asarray(pcm8, dtype=np.float32), self.cfg.pstn_sr, self.cfg.model_sr)

    def from_model(self, pcm: np.ndarray) -> np.ndarray:
        return resample(np.asarray(pcm, dtype=np.float32), self.cfg.model_sr, self.cfg.pstn_sr)


# --- the bridge -----------------------------------------------------------------------------

def _pad_to(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(np.asarray(x, dtype=np.float32), (0, max(0, n - x.size)))[:n]


@dataclass(frozen=True)
class BridgeTap:
    """Four channels recorded at the bridge's two interfaces, all on the bridge's own clock.

    The bridge is the recorder for the speech-native PSTN cell, which incidentally settles the
    gating question for that cell: dual-channel separation is guaranteed by construction here,
    so the mono-mix failure mode that `recover` refuses cannot arise. It remains an open risk
    only for the cascaded arm, where the provider owns the recording.
    """

    sip_user: np.ndarray
    sip_agent: np.ndarray
    model_user: np.ndarray
    model_agent: np.ndarray
    pstn_sr: int
    model_sr: int
    inbound: JitterStats = field(default_factory=JitterStats)

    def dual_channel(self, viewpoint: str = "model", sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
        """(user, agent) at a common rate, ready for `recover`.

        `model` cancels the bridge's delay; `sip` carries it (`BridgeConfig.delay_ms`). The
        resampling here is linear-phase and delay-free, so it does not move a boundary -- the
        same step Phase 1's `*_up` conditions took to measure 8 kHz audio at 16 kHz.
        """
        if viewpoint == "model":
            user, agent, src = self.model_user, self.model_agent, self.model_sr
        elif viewpoint == "sip":
            user, agent, src = self.sip_user, self.sip_agent, self.pstn_sr
        else:
            raise ValueError(f"viewpoint must be 'model' or 'sip', got {viewpoint!r}")
        u, a = resample(user, src, sr), resample(agent, src, sr)
        n = max(u.size, a.size)
        return _pad_to(u, n), _pad_to(a, n)


class MediaBridge:
    """The media plane. Carrier RTP on one side, linear PCM on the other, taps on both.

    Deliberately not a network object: it converts streams and keeps the clocks honest. The
    delegated SIP stack supplies received packets and sends returned ones, which is also what
    makes the whole path testable offline against known ground truth.
    """

    def __init__(self, cfg: BridgeConfig | None = None) -> None:
        self.cfg = cfg or BridgeConfig()
        self.transcoder = Transcoder(self.cfg)

    def carrier_to_model(self, packets: list[RtpPacket],
                         arrivals: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, JitterStats]:
        """Inbound leg. Returns (sip_side_pcm8, model_side_pcm, buffer stats).

        Both taps start at the instant the first packet arrived. The model-side tap is the
        playout stream, so events in it sit `inbound_delay_ms` later than they arrived. The
        SIP-side tap is that same stream re-originated to arrival time -- identical to a capture
        of the inbound RTP except inside concealed frames, where a capture would show a gap and
        this shows the concealment. That difference cannot move the user onset, which is found by
        matched filtering, and it keeps the two viewpoints differing by exactly `delay_ms` so the
        per-call check in rule 1 stays meaningful.
        """
        buf = FixedJitterBuffer(self.cfg)
        for i, pkt in enumerate(packets):
            buf.push(pkt, None if arrivals is None else arrivals[i])
        played, stats = buf.drain()
        skip = self.cfg.jitter_frames * self.cfg.pstn_frame
        return played[skip:], self.transcoder.to_model(played), stats

    def model_to_carrier(self, model_pcm: np.ndarray, seq0: int = 0,
                         timestamp0: int = 0) -> tuple[list[RtpPacket], np.ndarray]:
        """Outbound leg. Returns (packets to send, the SIP-side tap of what was sent).

        One frame of silence is prepended because the socket delivers PCM in chunks of its own
        choosing and whole 20 ms frames have to be assembled before they can go out. The tap
        records that delay rather than pretending it away.
        """
        pcm8 = self.transcoder.from_model(model_pcm)
        paced = np.concatenate([np.zeros(self.cfg.pstn_frame, dtype=np.float32), pcm8])
        packets = packetise(paced, self.cfg, seq0=seq0, timestamp0=timestamp0)
        n = len(packets) * self.cfg.pstn_frame
        return packets, _pad_to(paced, n)


# --- SDP, the one thing the delegated SIP stack needs from here -----------------------------

def sdp_offer(local_ip: str, local_port: int, cfg: BridgeConfig | None = None,
              session_id: int = 0) -> str:
    """A single-codec offer.

    Offering one codec is the point. If the offer lists G.711 alongside G.722 or Opus, the
    carrier may answer with the wideband one, and the call then runs a transport the
    preregistration does not describe while every downstream number still looks plausible.
    """
    cfg = cfg or BridgeConfig()
    return "\r\n".join([
        "v=0",
        f"o=stopbias {session_id} {session_id} IN IP4 {local_ip}",
        "s=stopbias-phase2",
        f"c=IN IP4 {local_ip}",
        "t=0 0",
        f"m=audio {local_port} RTP/AVP {cfg.payload_type}",
        f"a=rtpmap:{cfg.payload_type} {cfg.encoding_name}/{cfg.pstn_sr}",
        f"a=ptime:{int(cfg.frame_ms)}",
        "a=sendrecv",
        "",
    ])


def accept_sdp_answer(sdp: str, cfg: BridgeConfig | None = None) -> tuple[str, int]:
    """Validate the answer and return the remote (ip, port), or raise.

    Raises `CodecMismatch` if the answer selected anything other than the offered codec, or if
    it came back with a different clock rate or packet time. A rejected call is a reported
    failure; a silently renegotiated one is a corrupted cell.
    """
    cfg = cfg or BridgeConfig()
    ip: str | None = None
    port: int | None = None
    chosen: list[int] = []
    rtpmap: dict[int, str] = {}
    ptime: int | None = None
    for raw in sdp.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("c=IN IP4 "):
            ip = line[len("c=IN IP4 ") :].strip().split("/")[0]
        elif line.startswith("m=audio "):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"malformed m= line: {line!r}")
            port = int(parts[1])
            chosen = [int(p) for p in parts[3:] if p.isdigit()]
        elif line.startswith("a=rtpmap:"):
            spec = line[len("a=rtpmap:") :].split(None, 1)
            if len(spec) == 2:
                rtpmap[int(spec[0])] = spec[1].strip()
        elif line.startswith("a=ptime:"):
            ptime = int(float(line[len("a=ptime:") :]))
    if ip is None or port is None:
        raise ValueError("answer has no connection address or audio port")
    if port == 0:
        raise ValueError("answer declined the audio stream (port 0)")
    if not chosen:
        raise CodecMismatch("answer selected no payload type")
    if chosen[0] != cfg.payload_type:
        name = rtpmap.get(chosen[0], f"PT {chosen[0]}")
        raise CodecMismatch(
            f"answer selected {name} but the study is preregistered on "
            f"{cfg.encoding_name}/{cfg.pstn_sr}"
        )
    declared = rtpmap.get(cfg.payload_type)
    if declared is not None:
        want = f"{cfg.encoding_name}/{cfg.pstn_sr}"
        if declared.split("/")[0].upper() != cfg.encoding_name or not declared.startswith(want):
            raise CodecMismatch(f"answer maps PT {cfg.payload_type} to {declared}, expected {want}")
    if ptime is not None and ptime != int(cfg.frame_ms):
        raise CodecMismatch(f"answer asked for {ptime} ms packets, bridge is built for "
                            f"{int(cfg.frame_ms)} ms")
    return ip, port
