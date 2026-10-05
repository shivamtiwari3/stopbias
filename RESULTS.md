# Phase 1 — Does transport bias offline stop-latency measurement?

**Status:** complete. 3,720 measurements, 60 trials × 16 conditions × 2 viewpoints × 2 detectors.
Zero missed detections in every cell. Runtime 46 s, no API keys, no spend. (Repository-wide the
suite is now 144 tests, covering Phase 2's recovery code and bridge as well.)

Reproduce: `prep → plan → run → report → plots` (see `README.md`).
Raw: `results/raw/phase1.jsonl`. Tables: `results/phase1_tables.md`. Figures: `results/figures/`.

---

## The question

Full-Duplex-Bench v1.5 defines stop latency as

```
t_stop = t_model_stop − t_user_start
```

and measures it by running a VAD over **offline 16 kHz PCM files**. Every published
full-duplex stop-latency number the field has comes from that setup. Deployed voice agents
do not run on 16 kHz files — they run over WebRTC or over 8 kHz G.711 on a SIP trunk.

Nobody has tested whether the offline number predicts the transported one. That is the gap
this phase measures, and the reason it is measurable at all is that here **both terms of the
equation are known by construction** rather than estimated: the user clip is trimmed so its
physical energy onset is sample 0, and the agent channel is hard-cut to zero at a chosen
sample. So a detector's error is a measurable quantity, not a definitional one.

Everything is reported as the decomposition

```
stop_err = offset_err − onset_err
```

because a condition can be badly wrong on **both** boundaries and still look perfect on stop
latency if the two errors cancel. Reporting only stop latency hides that, and as it turns
out, cancellation is the single most important effect in these results.

---

## First: the harness is correct

No finding below is worth anything unless the instrument is verified. Three conditions inject
a known constant delay into the transport (40, 60, 80 ms). The harness must recover each one
in a specific, non-obvious pattern:

| injected delay | measured bulk delay | `energy` wire | `energy` local_user |
|---|---|---|---|
| 40 ms (`pstn_typical`) | **+40.0** | **0.0** | **+40.0** |
| 60 ms (`nb8_ulaw_jb60`) | **+60.0** | **0.0** | **+60.0** |
| 80 ms (`pstn_poor`) | **+80.0** | −5.0 | **+80.0** |

Read the `wire` column: a delay applied to both channels shifts both boundaries equally and
**cancels exactly** out of the difference. In `nb8_ulaw_jb60` the onset error moves
−15.7 → +44.3 ms and the offset error 16.9 → 77.0 ms — both by +60.0 — while the stop-latency
error does not move at all (32.8 vs 33.1 ms reference). Read the `local_user` column: the same
delay applied to one channel only shows up **in full**, to the millisecond, at all three delays
(p ≤ 4.6e-12).

Recovering the right answer in the right place at three delays, with exact cancellation where
cancellation is predicted, is what makes the rest of the table trustworthy. Figure 2 shows it
directly: at the jitter conditions both boundary bars jump and the stop-latency bar stays flat.

---

## Findings

### 1. G.711 narrowband does not bias the median. At all.

This is the surprise. 8 kHz µ-law, A-law, analysed natively or upsampled to 16 kHz first:
**relative bias 0.0 ms for both detectors, in both viewpoints.**

| condition | `silero` wire bias | p | `energy` wire bias |
|---|--:|--:|--:|
| µ-law 8 kHz, VAD @ 8 kHz | +0.0 [0, 0] | 0.76 | +0.0 |
| µ-law 8 kHz, upsampled to 16 kHz | +0.0 [0, 0] | 0.96 | +0.0 |
| A-law 8 kHz | +0.0 [0, 0] | 0.41 | +0.0 |

Log-companded quantisation raises the noise floor substantially, and it does not move an
adaptive-threshold boundary or a neural VAD's boundary. The plain codec is not the problem, and
neither is the sample rate: whether the VAD runs at 8 kHz or sees upsampled audio makes no
measurable difference. **A µ-law-only leg needs no correction.**

### 2. But it doubles the spread, which is a different problem

| detector, condition | IQR | P95 abs error |
|---|--:|--:|
| `silero`, 16 kHz file | 31.3 ms | 67.9 ms |
| `silero`, µ-law 8 kHz | **63.5 ms** | **128.6 ms** |

The median is unbiased and the per-call number is twice as noisy. Since the width of a
confidence interval on the median goes as spread/√n, this **roughly quadruples the number of
calls** needed to resolve a given difference between two systems. Accuracy and precision
degrade independently here, and only accuracy is zero.

### 3. The PSTN passband is what actually biases a neural VAD

Adding the 300–3400 Hz band limit a real carrier leg imposes:

| detector / viewpoint | offset error | relative bias | p |
|---|--:|--:|--:|
| `silero` / wire | 13.7 → **83.2 ms** | **+32.0** [0, 64] | 1.0e-04 |
| `silero` / local_user | 13.7 → **83.2 ms** | **+64.0** [32, 64] | 1.4e-08 |
| `energy` / wire | 16.9 → 18.1 ms | +0.0 | 7.7e-04 |

Removing energy below 300 Hz makes Silero hold the speech segment open ~70 ms past the true
cut. The energy detector is untouched. So the impairment that matters is not the codec — it is
the high-pass, and it only matters if your detector is a speech model.

### 4. Opus is the largest single offender

| condition | `silero` wire bias | p | `energy` wire bias | p |
|---|--:|--:|--:|--:|
| Opus 12 kbps, 4 kHz cutoff | **+80.0** [64, 96] | 2.8e-09 | +12.5 [10, 15] | 1.2e-06 |
| Opus 24 kbps, 8 kHz cutoff | **+32.0** [32, 64] | 7.7e-09 | +10.0 [5, 12.5] | 1.5e-07 |

Silero's offset error under narrowband Opus is **105 ms**. Part of this is genuinely not
detector error: a codec's decoder emits a real energy tail after a hard cut, so some of the
audio the VAD is detecting actually exists on the wire. That makes it a transport artefact
either way — and it means WebRTC-transported measurements are biased ~2.5× more than
G.711/SIP ones, the opposite of what "PSTN is the degraded path" would suggest.

### 5. Packet-loss concealment matters more than the loss rate

| condition | `silero` offset err | `silero` wire bias | p |
|---|--:|--:|--:|
| 1% loss, zero-fill | 15.9 ms | +0.0 | 0.71 |
| 5% loss, zero-fill | 32.5 ms | +0.0 [0, 32] | 0.064 |
| 5% **bursty** loss, zero-fill | **56.5 ms** | **+16.0** [0, 48] | 0.0025 |
| 5% loss, zero-order-hold PLC | 21.9 ms | +0.0 | 0.86 |

Same 5% loss rate, three different answers. Bursty loss shifts the offset and inflates IQR to
117 ms; a trivial zero-order-hold concealer removes the bias entirely. Reporting a loss
percentage without stating the concealment strategy does not specify the condition.

### 6. Line noise makes an agent look *faster* than it is

The only condition producing a **negative** bias, and it is large:

| detector / viewpoint | onset err | offset err | relative bias | IQR | P95 abs |
|---|--:|--:|--:|--:|--:|
| `energy` / wire, 20 dB SNR | −15.7 → **+1.8** | 16.9 → **+8.3** | **−50.0** [−70, −30] | 90.0 | **428.9 ms** |

Noise raises the floor, the adaptive threshold rises with it, and the onset is found *later*
while the offset is found *earlier*. Both errors push the measured stop latency **down**. An
energy-gated harness on a noisy line will credit an agent with 50 ms it never earned, with a
P95 error of 429 ms. Silero moves the other way (+48 ms) and does not break.

This is the most dangerous result in the set, because it biases in the flattering direction.

### 7. The impairments are approximately additive

Two composite conditions apply everything a real leg applies at once, so the single-impairment
numbers can be checked against the combination:

| detector / viewpoint | typical leg: observed / predicted | poor leg: observed / predicted |
|---|--:|--:|
| `silero` / wire | **+32.0 / +32.0** | +64.0 / +48.0 |
| `silero` / local_user | +96.0 / +106.7 | +160.0 / +181.3 |
| `energy` / wire | **+0.0 / +0.0** | −5.0 / +0.0 |
| `energy` / local_user | **+40.0 / +40.0** | **+80.0 / +80.0** |

Residuals are ≤32 ms, which is exactly one Silero analysis frame — at or below the resolution
of the measurement. There is **no evidence of interaction between impairments**, which is a
useful negative: characterising impairments one at a time is a valid way to predict a
composite leg, and nobody needs to run the full cross-product.

The realistic worst case is worth stating plainly. A congested mobile leg (`pstn_poor`:
300–3400 Hz + µ-law + 5% bursty loss + 80 ms jitter) measured with Silero gives a **+64 ms
median bias with a 270 ms P95 absolute error** in the wire viewpoint, and **+160 ms** in the
local-user viewpoint.

### 8. A local copy of the user audio makes the bias *worse*, not better

telnyx-onset's design uses the measuring party's own pristine copy of the user audio to get
the onset. Intuitively that should help — one less degraded signal. It does the opposite: for
`silero`, the local-user bias is at least as large as the wire bias in **every** condition,
often double (band limit +64 vs +32; poor leg +160 vs +64; jitter +64 vs 0).

The reason is finding 0: the wire viewpoint gets **error cancellation for free**, because a
common-mode transport delay hits both boundaries and subtracts out. The local-user viewpoint
destroys that cancellation and exposes the full one-way return delay.

But this is not simply "wire is the better method," and that is the conceptual point of the
phase. **The two viewpoints measure two different physical quantities:**

- **wire** ≈ how long the agent took to stop *at the agent*, transport delay cancelled out.
- **local_user** ≈ how long the interruption took *as the user experienced it*, transport delay
  included — because the user really does keep hearing the agent for the duration of the
  return path.
- **offline 16 kHz files** measure the first quantity with no transport in the system at all.

Neither is wrong. Conflating them is, and the field currently does: a published offline number
is an agent-internal reaction time being read as a user-experienced one. The gap between them
is not noise — it is the return-path delay plus codec tail, and it is what these tables
quantify.

### 9. The two detectors fail in orthogonal ways, and one of them is fixable

| detector | reference bias | IQR range (all conditions) | P95 range |
|---|--:|--:|--:|
| `energy` | **+33.1 ms** [32.5, 34.0] | **3.5 – 90.0 ms** | 36.6 – 428.9 ms |
| `silero` | **−6.3 ms** [−10.8, +4.0] | 31.3 – 139.5 ms | 67.9 – 271.5 ms |

The energy detector carries a large but extremely stable systematic offset — +33 ms with an
IQR of 3.6 ms. That is a framing convention, and a constant is **subtractable**. Silero is
nearly unbiased at the median and has ~10× the spread, which is **not** subtractable.

Practical consequence, and the most directly actionable result here: with n=60, the energy
detector resolves bias to ~1 ms, while Silero's 32 ms frame quantisation means a 60-trial study
**cannot resolve a bias below one frame**. This shows up in the tables as apparent conflicts —
`pstn_typical`/`silero`/wire has a +32 ms median bias with a bootstrap CI of [0, 64] and
p=0.061. The sign is consistent and Wilcoxon (which uses all the paired information, not just
the median) often rejects where the median CI cannot, but the honest reading is that
**Silero-based bias estimates below ~32 ms are not resolvable at this sample size.**

Anyone building a stop-latency harness should calibrate an energy gate and subtract its
constant, not reach for a neural VAD for its apparent accuracy.

### 10. The bias is additive, not multiplicative

Figure 4 plots measured against true stop latency across 200–1200 ms. The condition clouds sit
**parallel to the identity line** — offset vertically, not rotated. So a single constant per
condition describes the bias; it does not scale with the latency being measured. The visible
exception is a handful of `pstn_poor` outliers pulled 100–400 ms *below* the line, where a loss
burst split the final speech segment and the harness took an early segment end as the stop.

---

## Phase 2's measurement path, validated offline

Phase 2 has to recover stop latency from whatever a live provider hands back, and Novis exposes
no per-turn timestamps, so it must come from audio. What the provider returns is therefore the
gating unknown. All three possible answers were implemented and tested against simulated calls
carrying Phase 1's impairments per direction, with the true latency built in — validation a paid
pilot cannot do, because a real call supplies no ground truth to check against.

### 11. Dual-channel recovery is accurate; mono-mix recovery is not, by ~680 ms

Recovered minus true latency, median over true latencies of 200/500/900 ms × 3 seeds:

| transport leg | dual channel | mono mix |
|---|--:|--:|
| clean | +15 ms | +15 ms |
| 20 dB line noise | +15 ms | +15 ms |
| µ-law 8 kHz | +15 ms | **+685 ms** |
| 300–3400 Hz passband | +20 ms | **+680 ms** |
| 5% bursty loss, zero-fill | +15 ms | **+680 ms** |
| Opus NB 12 kbit/s | +35 ms | **+695 ms** |
| full PSTN leg (composite) | +20 ms | **+685 ms** |

Dual-channel holds from 100 to 1200 ms of true latency with both detectors; the residual +12 to
+35 ms is the framing constant of §9, which is subtractable.

The mono failures are not noise. The recovered "stop" is the *end of the interjection*: residual
leakage from the harness's own audio holds the detector open past the agent's real cessation. The
onset half of the mono method works — because the harness knows what it emitted, the onset is
located in the provider's recording by matched filtering, so both boundaries sit on one clock and
transport delay cancels (verified: spread < 25 ms across 0/60/150 ms of delay). It is the offset
that cannot be recovered, because the interjection sits ~15 dB above the agent, after the PSTN
filter both are speech inside the same 300–3400 Hz band, and the channel is nonlinear (companding)
and time-varying (loss, jitter). Suppressing the user by >15 dB across the whole band at every
instant is what would be needed. Two independent cancellers were tried — a delay-aligned,
reweighted least-squares FIR and a time-varying per-bin STFT gain — and neither manages it.

### 12. No in-band diagnostic distinguishes the usable mono cases from the broken ones

This is what makes §11 a hard constraint rather than a quality-control problem. Both diagnostics
the recording affords overlap between the working and failing cases:

| leg | probe suppression | residual/reference correlation | error |
|---|--:|--:|--:|
| clean | 74.7 dB | 0.025 | +15 ms |
| µ-law 8 kHz | 38.1 dB | 0.028 | +685 ms |
| 5% bursty loss | 7.1 dB | 0.058 | +680 ms |
| Opus NB 12 kbit/s | 5.0 dB | 0.031 | +695 ms |

Suppression is not monotone in the error, and the correlation guard fails because the leakage is
*spectrally distorted* — low correlation with the clean reference, ample energy. So bad trials
cannot be flagged and dropped. `stopbias.recover` refuses the mono path by default and, when
opted into, labels the result `mono_mix_unvalidated` rather than `ok`. **Dual-channel recording is
a hard requirement for Phase 2, which makes the one-call gating experiment decisive.**

Two claims in the preregistration's first revision were falsified by this work and corrected
before any data collection: that a mono mix was a survivable fallback, and that the in-call probe
could supply the round-trip delay the local-only correction needs (nothing echoes it back, so it
cannot).

### 13. A stop that lands in the agent's own pause is unmeasurable, not fast

Constructing this case makes **both** detectors read up to 320 ms early on otherwise clean audio,
because there is no acoustic cessation to find. Phase 1 could avoid it by nudging the cut into
loud speech; Phase 2 cannot, since the latency is the agent's own behaviour. So it becomes a
precondition on trial validity — the harness must confirm continuous agent voicing across the
window and void the trial otherwise. Left unhandled it would bias Phase 2 *toward* finding agents
faster than they are, in the same direction as §6.

---

## Phase 2's bridge, validated offline

Phase 2's fourth cell — a speech-native model reached over the PSTN — requires a SIP↔WebSocket
media bridge, because GPT-Realtime and Gemini Live speak PCM over a socket and nothing else.
Without it, architecture and transport are perfectly confounded and H2/H3 cannot be tested at all.

The bridge is now built (`stopbias/bridge.py`) and validated the same way the recovery code was:
by putting the real bridge inside a simulated call whose latency is known by construction
(`stopbias/bridgecall.py`, 54 tests in `tests/test_bridge.py`, report in
`results/phase2_bridge.md`, reproducible via `stopbias.cli bridge` in ~4 s). Real packetisation,
real packet loss, real reordering, real concealment, real transcode — not a description of them.

The finding that matters is that a bridge is a measurement instrument, and it can fabricate the
study's primary effect if it is read at the wrong point.

### 14. Tapping the bridge at the SIP interface would manufacture H1

The bridge exists in exactly **one cell of the design**. So any delay it contributes is
arithmetically indistinguishable from that cell's transport effect — which is what H1 claims to
measure. Writing `D_in` for the playout buffer, `D_out` for the outbound framing quantum and `L`
for the model's true stop latency:

```
model-side tap:  onset at D_in,  stop at D_in + L          → L
SIP-side tap:    onset at 0,     stop at D_in + L + D_out   → L + D_bridge
```

Sweeping the buffer depth from 0 to 6 frames, i.e. 20 to 140 ms of bridge delay, at a true
latency of 500 ms with the `energy` detector:

| depth | D_bridge | model-tap error | SIP-tap error | SIP − model |
|--:|--:|--:|--:|--:|
| 0 | 20 ms | +15 ms | +35 ms | 20 ms |
| 1 | 40 ms | +20 ms | +60 ms | 40 ms |
| 2 | 60 ms | +20 ms | +80 ms | 60 ms |
| 4 | 100 ms | +15 ms | +115 ms | 100 ms |
| 6 | 140 ms | +15 ms | +155 ms | 140 ms |

**All 140 ms lands in the SIP viewpoint and none of it in the model viewpoint.** The bridge's
delay is common mode at the model interface and cancels exactly — the same argument that makes
Phase 1's `wire` viewpoint unbiased, arrived at independently. At the default depth the SIP tap
carries 60 ms of pure fabrication, against the 50 ms effect §5 of the preregistration is powered
to detect. That is not extra noise; it is an effect of the right size and the right sign,
appearing in exactly the cell H1 predicts it in.

So the tap point is a preregistered measurement decision, not an implementation detail, and
`PREREGISTRATION.md` §3 now fixes it. Both taps are still recorded, because their difference
*measures* `D_bridge` per call: recovered as exactly 60.0 ms of a true 60 ms on all nine carrier
legs. That is a per-call validity check on the arithmetic above rather than a claim about it.

The check must use `energy`. Silero's 32 ms frame quantisation (§9) leaves it unable to resolve
60 ms to better than a frame — it reads a median 72 ms across the grid and 40 ms on one trial —
which is §4's detector argument turning up in a second, unrelated place.

### 15. A fixed playout buffer keeps the bridge's delay constant even when the call is wrecked

Median model-tap error over 5 true latencies (100–1200 ms) per leg, `energy` detector:

| carrier leg | model-tap error | concealed frames |
|---|--:|--:|
| clean | +15.0 ms | 0.0% |
| 300–3400 Hz passband | +15.0 ms | 0.0% |
| passband + 20 dB SNR | +15.0 ms | 0.0% |
| 1% loss | +15.0 ms | 1.2% |
| 5% bursty loss | +15.0 ms | 5.4% |
| 15 ms jitter | +15.0 ms | 0.0% |
| **80 ms jitter** | **+15.0 ms** | **42.1%** |
| ±200 ppm clock drift | +14.4 ms | 0.0% |
| full PSTN composite | +14.8 ms | 3.9% |

The error is flat to within 0.6 ms across every leg, and the 80 ms jitter row is the point: 142
packets arrive too late to play and 42% of frames reach the model as concealment, yet the bridge's
delay has not moved and the measurement is unaffected. An *adaptive* buffer would have grown to
absorb that jitter, and its delay would then vary with network conditions — which are the
treatment. A fixed buffer converts jitter into a countable concealment rate instead, which is
reportable per trial and cannot masquerade as latency. This is why the buffer refuses to adapt.

Clock drift is left uncorrected for the same reason: correcting it means inserting or dropping
samples inside the window being measured. It does not need correcting. Drift is a timescale error
costing `t × ppm`, so at 200 ppm and an interjection 2.5 s in it is 0.5 ms — measured at 0.6 ms,
three orders of magnitude below the effects in scope.

One thing the bridge gets for free: **it is the recorder.** Dual-channel separation is structural
in this cell, so the mono-mix failure of §11–12 cannot arise in it at all, and the return leg
cannot bias the measurement because the tap is upstream of it. The gating risk that makes the
one-call Novis experiment decisive survives only for the cascaded arm, where the provider owns the
recording.

### 16. Three things the bridge deliberately does not do

Each is a rule about absent code, so each is pinned by a test that would fail if someone added it.

- **No VAD, no DTX, no comfort noise, no gain control.** Phase 2 measures when a model decides to
  stop talking. A bridge that made its own speech/silence decision would put a second endpointer
  inside the path whose endpointing is the measurement, and no analysis could separate them.
  Verified by passing −40 dBFS audio through and requiring 0.0 dB of change and no suppressed
  frame.
- **No stateless per-frame resampling.** Converting each 20 ms frame independently gives every
  frame boundary its own filter transient: measured at 10× the round-trip error of block mode
  (−30 dB against −51 dB). A streaming implementation must carry resampler state across frames.
- **No multi-codec offer.** The SDP offer names exactly one codec and the answer is rejected if it
  selected anything else, or changed the packet time. A carrier that quietly answers G.722 would
  put a wideband leg in a cell preregistered as narrowband, and every number downstream would
  still look entirely reasonable.

**What is not built.** The SIP signalling plane — dialog, registration, digest auth, re-INVITE —
is delegated to an existing stack, because it cannot be exercised without a carrier and it fails
loudly at call setup rather than quietly at measurement time. What the delegate needs from this
module is `sdp_offer` and `accept_sdp_answer`; what it returns is a socket pair. If the carrier
offers bidirectional media over a WebSocket instead, the media plane attaches to that directly and
the RTP layer here goes unused — the timing rules above are what matter either way, and they are
independent of how the frames arrive.

**What this does not validate.** The simulated model is a hard cut, so this establishes that the
bridge does not distort the *measurement* of a stop. It says nothing about how a real model's
stop behaviour responds to 42% concealment — that response is H1, and it needs live calls.

---

## What this does and does not establish

**Establishes.** Transport is a real, measurable, reproducible confound in offline stop-latency
measurement, and it is not where you would guess: the codec and sample rate contribute nothing,
while the passband, Opus, loss burstiness and line noise contribute 10–96 ms and can flip sign.
The measurement viewpoint is a bigger design decision than any single impairment. Impairments
do not interact, so they can be characterised independently. The detector choice dominates
everything else, and the better choice is the calibratable one.

**Does not establish.** These are simulated transports applied to LibriSpeech read speech, not
live calls. Read speech has cleaner onsets than spontaneous conversational speech, so the
onset-error terms are likely optimistic. Most importantly, **the agent here is a hard cut —
a perfect interrupter.** A real agent's stop behaviour interacts with the degraded audio it
receives on the way *in*: a VAD inside the agent's own barge-in path faces the same impairments
measured here, which plausibly changes when the agent decides to stop, not merely when a
harness thinks it stopped. That coupling is Phase 2 and it needs live systems.

For scale: telnyx-onset reports a 640 ms gap between local-VAD and transcript-based stop
latency. The transport effects here top out near 96 ms. So transport does **not** explain that
gap — the VAD-vs-transcript method choice dominates it. Transport is a second-order confound
that adds up to ~100 ms of bias on top, which still matters when systems are ranked at 50 ms
granularity, but it should not be oversold as the main story.

## Provenance and licensing

The stop-latency definition and the merge-gap constants (0.6 s user, 0.5 s model) were
reimplemented from Full-Duplex-Bench's published equation and documented parameters, not copied
— FDB is CC BY-NC. telnyx-onset carries no licence file and was read for method only; no code
was copied from it. Corpus is LibriSpeech dev-clean (CC BY 4.0), 20 user / 20 agent speakers,
disjoint pools, verified by test. G.711 is implemented from ITU-T tables and verified against
ffmpeg (>0.99 sample agreement, and never a worse reconstruction — the residual disagreements
are ffmpeg's 14-bit-rounded decision boundaries and an arbitrary zero tie-break).
