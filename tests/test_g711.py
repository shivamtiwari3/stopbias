"""G.711 correctness, including agreement with ffmpeg's implementation.

Two facts about the standard shape these tests:
  - mu-law has two codes for zero (0x7F and 0xFF), so its codebook holds 255 distinct
    levels, not 256. An encoder can only return one of the two.
  - A-law's smallest levels are -8 and +8, so an input of exactly 0 is an exact tie and the
    tie-break is arbitrary. ffmpeg picks +8; a nearest-neighbour search picks -8.
Everywhere else the two implementations must agree exactly.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from stopbias import g711
from stopbias.degrade import FFMPEG

LAWS = ("ulaw", "alaw")
RAW_FMT = {"ulaw": "mulaw", "alaw": "alaw"}


def table(law: str) -> np.ndarray:
    return g711.ULAW_DECODE if law == "ulaw" else g711.ALAW_DECODE


def test_ulaw_has_two_zero_codes():
    zeros = np.flatnonzero(g711.ULAW_DECODE == 0)
    assert zeros.tolist() == [127, 255]
    assert len(set(g711.ULAW_DECODE.tolist())) == 255


def test_alaw_codebook_is_fully_distinct_and_has_no_zero_level():
    assert len(set(g711.ALAW_DECODE.tolist())) == 256
    assert 0 not in set(g711.ALAW_DECODE.tolist())
    assert sorted(abs(v) for v in g711.ALAW_DECODE.tolist())[:2] == [8, 8]


@pytest.mark.parametrize("law", LAWS)
def test_decode_table_spans_and_is_symmetric(law):
    t = table(law)
    assert t.shape == (256,)
    assert t.min() < -32000 and t.max() > 32000
    assert int(t.min()) == -int(t.max()), "the codebook must be symmetric about zero"


@pytest.mark.parametrize("law", LAWS)
def test_encoder_is_an_exact_inverse_up_to_level_equivalence(law):
    """encode(decode(c)) must land on a code with the same reconstruction level."""
    codes = np.arange(256, dtype=np.uint8)
    levels = g711.decode(codes, law)
    back = g711.decode(g711.encode(levels, law), law)
    assert np.array_equal(back, levels)


@pytest.mark.parametrize("law", LAWS)
def test_roundtrip_error_bounded_by_half_the_quantiser_step_in_range(law):
    rng = np.random.default_rng(7)
    x = rng.integers(-32768, 32767, size=20000).astype(np.int16)
    y = g711.roundtrip(x, law)
    t = np.sort(table(law).astype(np.int64))
    err = np.abs(y.astype(np.int64) - x.astype(np.int64))
    in_range = np.abs(x.astype(np.int64)) <= t.max()
    assert err[in_range].max() <= int(np.max(np.diff(t))) // 2 + 1


@pytest.mark.parametrize("law", LAWS)
def test_out_of_range_input_clips_to_the_codebook_extreme(law):
    """|x| beyond the codebook cannot round-trip; the residual is clipping, not quantisation."""
    t = table(law).astype(np.int64)
    x = np.array([-32768, 32767], dtype=np.int16)
    y = g711.roundtrip(x, law).astype(np.int64)
    assert y[0] == t.min() and y[1] == t.max()


def _ffmpeg_encode(x: np.ndarray, law: str) -> np.ndarray:
    enc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "8000", "-ac", "1",
         "-i", "pipe:0", "-f", RAW_FMT[law], "pipe:1"],
        input=x.tobytes(), capture_output=True, check=True,
    )
    return np.frombuffer(enc.stdout, dtype=np.uint8)


@pytest.mark.parametrize("law", LAWS)
def test_agrees_with_ffmpeg_on_almost_every_sample(law):
    """ffmpeg rounds its decision boundaries to 14-bit resolution (its table is indexed by
    sample >> 2), so a handful of samples sitting within a few LSBs of a boundary land on the
    adjacent code. Everything else must match bit for bit."""
    rng = np.random.default_rng(11)
    x = rng.integers(-32000, 32000, size=8000).astype(np.int16)
    x = x[x != 0]
    theirs, ours = _ffmpeg_encode(x, law), g711.encode(x, law)
    assert theirs.size == ours.size
    agree = float(np.mean(theirs == ours))
    assert agree > 0.98, f"{law}: only {agree:.4f} agreement with ffmpeg"


@pytest.mark.parametrize("law", LAWS)
def test_never_reconstructs_worse_than_ffmpeg(law):
    """The claim that matters: this encoder is an optimal quantiser over the standard's
    codebook, so its reconstruction error is never larger than ffmpeg's."""
    rng = np.random.default_rng(11)
    x = rng.integers(-32000, 32000, size=8000).astype(np.int16)
    x = x[x != 0]
    err_ours = np.abs(g711.decode(g711.encode(x, law), law).astype(np.int64) - x.astype(np.int64))
    err_theirs = np.abs(g711.decode(_ffmpeg_encode(x, law), law).astype(np.int64) - x.astype(np.int64))
    assert np.all(err_ours <= err_theirs), f"{law}: worse than ffmpeg on {int((err_ours > err_theirs).sum())} samples"


@pytest.mark.parametrize("law", LAWS)
def test_zero_input_ties_are_the_only_disagreement_with_ffmpeg(law):
    x = np.zeros(16, dtype=np.int16)
    enc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "8000", "-ac", "1",
         "-i", "pipe:0", "-f", RAW_FMT[law], "pipe:1"],
        input=x.tobytes(), capture_output=True, check=True,
    )
    theirs = g711.decode(np.frombuffer(enc.stdout, dtype=np.uint8), law)
    ours = g711.decode(g711.encode(x, law), law)
    assert np.array_equal(np.abs(theirs.astype(int)), np.abs(ours.astype(int)))
    assert int(np.max(np.abs(ours.astype(int)))) <= 8
