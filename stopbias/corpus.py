"""Corpus preparation.

Source: LibriSpeech dev-clean (OpenSLR resource 12, CC BY 4.0), 16 kHz read speech,
40 speakers. Speakers are partitioned into disjoint agent and user pools, so no trial ever
pairs a speaker with themselves and no acoustic identity is shared across the two channels
of a trial.

Every emitted clip is trimmed so that sample 0 is its physical energy onset. That is what
makes the insertion point in a constructed trial the exact ground-truth onset time.

Read speech is a stimulus, not a conversation: these clips stand in for the *acoustics* of
a barge-in onset, which is all a boundary detector responds to. Lexical realism belongs to
the live phases, not to this one.
"""

from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import energy_onset, load_mono, peak_normalize, resample, save_wav
from .vad import EnergyDetector, merge_segments

DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TARGET_SR = 16000
UA = "stopbias-research/0.1"

USER_CLIP_S = 1.20
AGENT_CLIP_S = 6.00
MIN_LEAD_SILENCE_S = 0.30
ONSET_ACTIVE_S = 0.25


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    role: str
    speaker_id: str
    source_id: str
    source_offset_s: float
    duration_s: float
    path: str


def fetch_dev_clean(root: Path) -> Path:
    """Download and extract dev-clean if not already present. Returns the LibriSpeech dir."""
    root = Path(root)
    extracted = root / "LibriSpeech" / "dev-clean"
    if extracted.is_dir():
        return extracted
    tgz = root / "dev-clean.tar.gz"
    if not (tgz.exists() and tgz.stat().st_size > 100_000_000):
        root.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DEV_CLEAN_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=1800) as r, open(tgz, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(root, filter="data")
    return extracted


def _segments(x: np.ndarray) -> list[tuple[float, float]]:
    det = EnergyDetector(margin_db=12.0, min_speech_ms=150.0, hangover_ms=60.0)
    return merge_segments(det.segments(x, TARGET_SR), gap_s=0.20)


def _cut_user(x: np.ndarray, segs: list[tuple[float, float]]) -> tuple[np.ndarray, float] | None:
    """A short interjection whose start is a genuine acoustic onset preceded by silence."""
    need = int(round(USER_CLIP_S * TARGET_SR))
    for k, (s, e) in enumerate(segs):
        prev_end = segs[k - 1][1] if k > 0 else 0.0
        if k > 0 and (s - prev_end) < MIN_LEAD_SILENCE_S:
            continue
        if (e - s) < ONSET_ACTIVE_S:
            continue
        a = int(round(s * TARGET_SR))
        if a + need > x.size:
            continue
        clip = peak_normalize(x[a : a + need])
        t0 = energy_onset(clip, TARGET_SR)
        if t0 is None or t0 > 0.05:
            continue
        clip = np.ascontiguousarray(clip[int(round(t0 * TARGET_SR)) :])
        if clip.size < int(round(0.80 * TARGET_SR)):
            continue
        return clip, s + t0
    return None


def _cut_agent(x: np.ndarray, segs: list[tuple[float, float]]) -> tuple[np.ndarray, float] | None:
    """A long stretch of agent-like speech, starting at an onset, long enough to cut into."""
    need = int(round(AGENT_CLIP_S * TARGET_SR))
    for s, _e in segs:
        a = int(round(s * TARGET_SR))
        if a + need > x.size:
            continue
        clip = peak_normalize(x[a : a + need])
        t0 = energy_onset(clip, TARGET_SR)
        if t0 is None or t0 > 0.05:
            continue
        return np.ascontiguousarray(clip[int(round(t0 * TARGET_SR)) :]), s + t0
    return None


def prepare(out_dir: Path, n_pairs: int = 60, corpus_root: Path | None = None, seed: int = 20260825) -> list[ClipMeta]:
    out_dir = Path(out_dir)
    src = fetch_dev_clean(Path(corpus_root or out_dir))

    speakers = sorted((p.name for p in src.iterdir() if p.is_dir()), key=int)
    if len(speakers) < 4:
        raise RuntimeError(f"expected LibriSpeech speaker dirs under {src}, found {speakers}")
    rng = random.Random(seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    user_spk, agent_spk = sorted(shuffled[:half], key=int), sorted(shuffled[half:], key=int)

    def utterances(spks: list[str]) -> list[Path]:
        files: list[Path] = []
        for s in spks:
            files.extend(sorted((src / s).rglob("*.flac")))
        rng.shuffle(files)
        return files

    def harvest(spks: list[str], role: str, cutter, target: int) -> list[ClipMeta]:
        made: list[ClipMeta] = []
        for f in utterances(spks):
            if len(made) >= target:
                break
            x, sr = load_mono(f)
            if sr != TARGET_SR:
                x = resample(x, sr, TARGET_SR)
            segs = _segments(x)
            if not segs:
                continue
            got = cutter(x, segs)
            if not got:
                continue
            clip, off = got
            cid = f"{role}{len(made):03d}"
            p = out_dir / role / f"{cid}.wav"
            save_wav(p, clip, TARGET_SR)
            made.append(ClipMeta(cid, role, f.parts[-3], f.stem, round(off, 4),
                                 round(clip.size / TARGET_SR, 4), str(p)))
        return made

    users = harvest(user_spk, "user", _cut_user, n_pairs)
    agents = harvest(agent_spk, "agent", _cut_agent, n_pairs)
    n = min(len(users), len(agents))
    if n < n_pairs:
        raise RuntimeError(f"only {n} usable pairs (user={len(users)}, agent={len(agents)})")
    metas = users[:n] + agents[:n]

    (out_dir / "manifest.json").write_text(json.dumps(
        {"source": DEV_CLEAN_URL, "sample_rate": TARGET_SR, "n_pairs": n, "seed": seed,
         "user_speakers": user_spk, "agent_speakers": agent_spk,
         "user_clip_s": USER_CLIP_S, "agent_clip_s": AGENT_CLIP_S,
         "clips": [asdict(m) for m in metas]}, indent=2))
    return metas
