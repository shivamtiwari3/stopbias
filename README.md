# stopbias — does offline stop-latency measurement survive the phone network?

Full-duplex voice benchmarks measure **stop latency** — how fast an agent stops talking when
the user interrupts — by running a VAD over offline 16 kHz PCM files. Deployed agents run over
WebRTC or 8 kHz G.711 on a SIP trunk. This repo measures how much that difference costs, using
trials where the ground truth is known by construction rather than estimated.

**Phase 1 is complete.** [`PAPER.md`](PAPER.md) is the write-up for submission;
[`RESULTS.md`](RESULTS.md) is the finding-by-finding record it draws on. Headline: the codec and sample rate
cost nothing; the PSTN passband, Opus, loss burstiness and line noise cost 10–96 ms and can
flip sign; a noisy line makes an agent look 50 ms *faster* than it is; and the choice of
measurement viewpoint matters more than any single impairment.

## Why the ground truth is real

The measured quantity is Full-Duplex-Bench v1.5's definition:

```
t_stop = t_model_stop − t_user_start
```

In FDB both terms are *estimated* by a VAD, so VAD error is invisible. Here both are *known*:

- Each user clip is trimmed so its physical energy onset is **sample 0**, verified by test for
  all 60 clips. The insertion index therefore *is* the true onset.
- The agent channel is hard-cut to zero at a chosen sample, so the true stop is exact.

That makes a detector's error a measurable quantity. Every result is reported as the
decomposition `stop_err = offset_err − onset_err`, because a condition can be wrong on both
boundaries and look perfect on their difference.

## Design

**16 conditions**, frozen in `stopbias/conditions.py`: 16 kHz reference; µ-law and A-law at
8 kHz (analysed natively and upsampled); the 300–3400 Hz PSTN passband; 1/3/5% frame loss with
Bernoulli and Gilbert–Elliott burst models and zero-fill vs zero-order-hold concealment;
fixed jitter buffers; narrowband and wideband Opus; 20 dB SNR line noise; and two composites
that apply a whole realistic leg at once to test additivity.

**2 viewpoints.** `wire` reads both channels off the wire, as a SIP capture harness does.
`local_user` reads the user onset from the measuring party's own pristine copy, as
telnyx-onset does. These turn out to measure two different physical quantities.

**2 detectors.** `silero` (neural, with `speech_pad_ms=0` — Silero's default of 30 ms pads
segments outward, which is right for cutting speech out of audio and wrong for timing it) and
`energy` (noise-floor-relative threshold with hangover). They fail in orthogonal ways.

**Validation built in.** Three conditions inject a known constant delay (40/60/80 ms). In the
`wire` viewpoint a common-mode delay must cancel exactly; in `local_user` it must appear in
full. The harness recovers all three to the millisecond. If it did not, nothing else would
count.

## Run it

Needs Python 3.12+, `ffmpeg`, and ~350 MB for LibriSpeech dev-clean. No API keys, no spend,
~46 s for the full grid.

```bash
uv venv && uv pip install -e .
.venv/bin/python -m stopbias.cli prep --pairs 60    # download + cut the corpus
.venv/bin/python -m stopbias.cli plan --pairs 60    # freeze the trial plan
.venv/bin/python -m stopbias.cli run                # 3,720 measurements
.venv/bin/python -m stopbias.cli report             # CSV + markdown tables
.venv/bin/python -m stopbias.cli plots              # 4 figures
.venv/bin/python -m stopbias.cli power              # Phase 2 sample sizes (~10 s)
.venv/bin/python -m stopbias.cli bridge             # validate the media bridge (~4 s)
.venv/bin/python -m pytest tests/ -q                # 144 tests
```

Everything is seeded (`BOOT_SEED = 20260825`, per-channel seeds derived by CRC32 from
trial/condition/channel) so a rerun reproduces the numbers exactly.

## Layout

| path | what |
|---|---|
| `stopbias/audio.py` | float32 mono primitives; `energy_onset` ground truth; cross-correlation bulk-delay estimator |
| `stopbias/g711.py` | ITU-T µ-law/A-law tables + optimal nearest-neighbour encoders |
| `stopbias/degrade.py` | the impairment ops and `apply_chain` |
| `stopbias/vad.py` | Silero and energy detectors behind one interface, FDB's merge gaps |
| `stopbias/corpus.py` | LibriSpeech dev-clean → onset-aligned clips, disjoint speaker pools |
| `stopbias/trials.py` | two-channel scene construction with exact ground truth |
| `stopbias/conditions.py` | the frozen condition × viewpoint grid |
| `stopbias/measure.py` | one measurement, all three error terms, cached boundaries |
| `stopbias/stats.py` | bootstrap median CIs (10,000 resamples), paired Wilcoxon vs reference |
| `stopbias/power.py` | vectorised signed-rank test + Phase 2 sample sizes simulated from Phase 1 residuals |
| `stopbias/plots.py` | four figures, one question each |
| `stopbias/stimulus.py` | Phase 2 outbound audio: broadband delay probe + interjection track |
| `stopbias/recover.py` | Phase 2 stop-latency recovery from a call recording, three channel layouts |
| `stopbias/simcall.py` | simulated live calls with known ground truth, for validating the above |
| `stopbias/bridge.py` | SIP↔WebSocket media plane: RTP, fixed playout buffer, transcode, four-channel tap, single-codec SDP |
| `stopbias/bridgecall.py` | simulated *bridged* calls that run the real bridge, with the latency built in |

## Statistics

Medians with 10,000-resample bootstrap percentile CIs. Bias against the 16 kHz reference is a
**paired within-trial difference** (same speakers, same onset, same true latency), tested with
a Wilcoxon signed-rank test. Note the resolution limit documented in `RESULTS.md` §9: Silero's
32 ms frame quantisation means bias below one frame is not resolvable at n=60, which is why
some cells show a non-zero median bias with a CI touching zero.

## Provenance

FDB is CC BY-NC: the stop-latency definition and merge-gap constants were reimplemented from
its published equation and documented parameters, not copied. telnyx-onset has no licence file
and was read for method only; no code was copied. Corpus is LibriSpeech dev-clean (CC BY 4.0).
G.711 is verified against ffmpeg — >0.99 sample agreement and provably never a worse
reconstruction, with the residual differences traced to ffmpeg's 14-bit-rounded decision
boundaries and an arbitrary tie-break at zero.

## Phase 2 (preregistered, not started)

Phase 1's agent is a hard cut — a perfect interrupter. A real agent's barge-in path runs its
own VAD on the degraded audio arriving *inbound*, so transport plausibly changes **when the
agent decides to stop**, not merely when a harness thinks it stopped.

The design, hypotheses, sample sizes, exclusion rules and stopping rule are frozen in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any live data collection. Summary: 2
architectures (cascaded, speech-native) × 2 transports (WebRTC, PSTN), fully crossed and
within-item paired, n=60 per cell, ≈320 calls, ≈$40–60.

Phase 1 pays for itself here. Its measured error distributions give real sample sizes instead
of guesses (`stopbias.cli power`): with the calibratable energy detector, detecting a 50 ms
effect over a degraded PSTN leg needs **15** paired calls; with a neural VAD it needs **80**,
and 25 ms needs **320**. Choosing the detector Phase 1 recommends makes Phase 2 roughly 8×
cheaper.

**The measurement path is already built and validated offline, before any spend.**
`stopbias/simcall.py` synthesises calls carrying Phase 1's impairments per direction with the
true latency built in, which is validation a paid pilot cannot perform — a real call supplies no
ground truth. Results are in `RESULTS.md` §11–13, and two of them changed the design:

- Dual-channel recovery is accurate on every leg tested (+12 to +35 ms, 100–1200 ms true
  latency, both detectors). Mono-mix recovery is wrong by **+680 ms** on every impaired leg, and
  no in-band diagnostic tells the good cases from the bad — so `recover` refuses that path by
  default and **dual-channel recording is a hard requirement**, making the $0.05 gating call
  decisive rather than informative.
- A stop landing inside the agent's own pause is unmeasurable, not fast: it makes both detectors
  read up to 320 ms early. It is now a void condition on trial validity.

Both corrected claims the preregistration's first revision got wrong; the revision is recorded
in the document rather than silently overwritten.

**The bridge is built too.** Filling the speech-native × PSTN cell needs a SIP↔WebSocket media
bridge — without it, architecture and transport are perfectly confounded and H2/H3 die. Its media
plane is in `stopbias/bridge.py` and validated the same way: the real bridge runs inside a
simulated call whose latency is known by construction (`stopbias.cli bridge`, ~4 s,
`results/phase2_bridge.md`). It changed the preregistration a third time, because a bridge is a
measurement instrument:

- **The recording must be tapped at the model-facing interface, not the SIP-facing one.** The
  bridge's own delay is common mode there and cancels exactly; at the SIP interface it does not,
  and since the bridge exists in only one cell of the design it would show up as a transport
  effect. Measured across playout depths of 0–6 frames, the SIP tap carries the whole 20–140 ms
  and the model tap carries none of it. At the default depth that is 60 ms — the size of the
  effect the study is powered to detect, in the direction H1 predicts. Reading the wrong
  interface would have manufactured the primary result.
- **The playout buffer is fixed-depth, never adaptive.** Recovery error is flat to within 0.6 ms
  across nine carrier legs, including one where 80 ms of jitter leaves 42% of frames concealed. An
  adaptive buffer would have absorbed that jitter into a delay that varies with network conditions
  — that is, with the treatment.

Everything still missing for Phase 2 is access, not engineering: OpenAI and Gemini keys, a SIP
trunk and number, a media host with a routable RTP path, and spend authorisation.
