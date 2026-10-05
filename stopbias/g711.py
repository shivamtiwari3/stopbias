"""ITU-T G.711 mu-law and A-law, implemented from the reference decode tables.

The decoders are the canonical bit-exact algorithms. The encoders are nearest-neighbour
quantisers over the decode codebook, which is the optimal encoder for that codebook and
what practical implementations (including ffmpeg's generated tables) converge to.
`tests/test_g711.py` checks exact inverse-on-codebook behaviour and agreement with ffmpeg.
"""

from __future__ import annotations

import numpy as np

_SIGN_BIT = 0x80
_QUANT_MASK = 0x0F
_SEG_SHIFT = 4
_SEG_MASK = 0x70
_BIAS = 0x84


def _ulaw2linear(code: int) -> int:
    u = (~code) & 0xFF
    t = ((u & _QUANT_MASK) << 3) + _BIAS
    t <<= (u & _SEG_MASK) >> _SEG_SHIFT
    return (_BIAS - t) if (u & _SIGN_BIT) else (t - _BIAS)


def _alaw2linear(code: int) -> int:
    a = code ^ 0x55
    t = (a & _QUANT_MASK) << 4
    seg = (a & _SEG_MASK) >> _SEG_SHIFT
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t = (t + 0x108) << (seg - 1)
    return t if (a & _SIGN_BIT) else -t


ULAW_DECODE = np.array([_ulaw2linear(i) for i in range(256)], dtype=np.int32)
ALAW_DECODE = np.array([_alaw2linear(i) for i in range(256)], dtype=np.int32)


def _build_encoder(decode: np.ndarray):
    order = np.argsort(decode, kind="stable").astype(np.uint8)
    vals = decode[order].astype(np.int64)
    # Nearest-neighbour decision boundaries between adjacent reconstruction levels.
    bounds = (vals[:-1] + vals[1:] + 1) // 2
    codes = order

    def encode(pcm16: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(bounds, pcm16.astype(np.int64), side="left")
        return codes[idx]

    return encode


_encode_ulaw = _build_encoder(ULAW_DECODE)
_encode_alaw = _build_encoder(ALAW_DECODE)


def encode(pcm16: np.ndarray, law: str) -> np.ndarray:
    """int16 PCM -> uint8 G.711 codes."""
    enc = _encode_ulaw if law == "ulaw" else _encode_alaw if law == "alaw" else None
    if enc is None:
        raise ValueError(f"law must be 'ulaw' or 'alaw', got {law!r}")
    return enc(np.asarray(pcm16))


def decode(codes: np.ndarray, law: str) -> np.ndarray:
    """uint8 G.711 codes -> int16 PCM."""
    table = ULAW_DECODE if law == "ulaw" else ALAW_DECODE if law == "alaw" else None
    if table is None:
        raise ValueError(f"law must be 'ulaw' or 'alaw', got {law!r}")
    return table[np.asarray(codes, dtype=np.uint8)].astype(np.int16)


def roundtrip(pcm16: np.ndarray, law: str) -> np.ndarray:
    """The quantisation a PSTN leg imposes: 16-bit linear -> 8-bit log -> 16-bit linear."""
    return decode(encode(pcm16, law), law)
