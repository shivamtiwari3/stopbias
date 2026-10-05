# Does an offline stop-latency number survive the phone network?

**Transport bias in full-duplex voice benchmarks, measured against ground truth by construction**

*Draft, 2026-09-02. Phase 1 complete; Phase 2 preregistered and not yet run. Numbers in this
draft are reproducible from this repository (`RESULTS.md`, `results/`).*

---

## Abstract

Full-duplex spoken dialogue benchmarks report **stop latency** — how quickly an agent stops
talking when a user interrupts — by running a voice activity detector over offline 16 kHz PCM
files. Deployed voice agents do not run on files. They run over WebRTC, or over 8 kHz G.711 on a
SIP trunk. Whether the offline number predicts the transported one has not been tested, and it
cannot be tested with the standard setup, because there both terms of the stop-latency equation
are *estimated* by the same VAD whose error is in question.

We build a harness in which both terms are **known by construction**: each user clip is trimmed
so its physical energy onset is sample 0, and the agent channel is hard-cut to zero at a chosen
sample. Detector error therefore becomes a measurable quantity rather than a definitional one.
Across 16 transport conditions × 2 measurement viewpoints × 2 detectors (3,720 measurements,
zero missed detections), we find that transport bias is real, reproducible, and not located where
intuition puts it.

The codec and the sample rate cost nothing: 8 kHz µ-law and A-law shift the median error by
**0.0 ms** for both detectors in both viewpoints, whether the detector runs natively at 8 kHz or
on upsampled audio. What costs is everything else. The 300–3400 Hz telephony passband biases a
neural VAD by **+32 to +64 ms**; narrowband Opus by **+80 to +96 ms**, making a WebRTC leg worse
than a G.711 leg — the reverse of the usual assumption; loss *burstiness* matters more than the
loss rate, and a trivial zero-order-hold concealer removes the bias a 5% bursty loss condition
creates. Line noise at 20 dB SNR is the most dangerous condition in the set, because it biases in
the flattering direction: it makes an agent look **50 ms faster than it is** (P95 absolute error
429 ms) by moving the onset later and the offset earlier at once.

Two design-level results outweigh the individual impairments. First, **the measurement viewpoint
matters more than any single impairment**, and the field currently conflates two different
physical quantities: reading both boundaries off the wire cancels common-mode transport delay and
measures an agent-internal reaction time, while reading the user onset from a local pristine copy
exposes the full return path and measures what the user experienced. Offline files measure the
first and are read as the second. Second, **the choice of detector dominates everything**: a
plain energy gate carries a large but extremely stable bias (+33.1 ms, IQR 3.6 ms) which is a
*subtractable constant*, whereas a neural VAD is near-unbiased at the median with ~10× the spread
and a 32 ms frame quantisation which is not subtractable. Choosing the calibratable detector
makes a downstream study up to **8× cheaper** at equal power.

We then build and validate offline — before any spend — the measurement path for a live
follow-up, and report two instrument-design findings that changed our preregistration. Recovering
stop latency from a **mono-mixed** call recording fails by **+680 ms** on every impaired leg, and
no in-band diagnostic separates the working cases from the broken ones, making dual-channel
recording a hard requirement rather than a preference. And a SIP↔WebSocket media bridge, needed
to break the confound between architecture and transport, **would manufacture the study's primary
effect if tapped at the wrong interface**: 60 ms of the bridge's own delay lands in exactly one
cell of the design, in the predicted direction, at the size the study is powered to detect.

Everything is seeded and reproducible in ~46 s with no API keys and no spend. Contributions: a
ground-truth methodology for validating stop-latency harnesses; a quantified map of where
transport bias comes from; concrete recommendations for benchmark designers; and a preregistered,
powered protocol for the behavioural question these measurements cannot answer.

---

## 1. Introduction

Full-duplex spoken dialogue systems are evaluated in part on how gracefully they yield the floor.
Full-Duplex-Bench (FDB) formalises this as stop latency,

```
t_stop = t_model_stop − t_user_start,                                     (1)
```

the interval between the user beginning to speak and the model ceasing to speak. FDB measures
both terms with a voice activity detector applied to offline 16 kHz PCM, and to our knowledge
every published full-duplex stop-latency figure originates in that setup.

Deployment does not look like that. A voice agent reached by phone is heard through a 300–3400 Hz
passband, log-companded to 8 bits at 8 kHz, packetised into 20 ms frames, subject to loss,
reordering and jitter buffering. A voice agent reached in a browser is heard through Opus, whose
decoder emits a real energy tail after speech stops. Both differ from a 16 kHz file in ways that
plausibly move a VAD boundary by tens of milliseconds — the same granularity at which systems are
ranked.

The obvious experiment, running a benchmark over a phone call and comparing, cannot settle it. In
equation (1) as normally instantiated, `t_user_start` and `t_model_stop` are both VAD estimates,
so a change in the reported number could be a change in the system, a change in the detector's
behaviour under degradation, or a change in the transport delay between the two reference points —
and nothing in the recording distinguishes them. A live call supplies no ground truth to check
against.

**Our approach is to remove the estimation from the ground truth.** We construct two-channel
scenes in which the true value of each term of (1) is known exactly:

- Every user clip is trimmed so that its physical energy onset is sample 0, verified by test for
  all 60 clips in the corpus. The insertion index therefore *is* the true `t_user_start`.
- The agent channel is hard-cut to zero at a chosen sample, so the true `t_model_stop` is exact
  by construction.

A detector's error is then a measurable quantity, and every result below is reported as the
decomposition

```
stop_err = offset_err − onset_err,                                       (2)
```

because a condition can be badly wrong on *both* boundaries and still look perfect on their
difference. As it turns out, that cancellation is the single most important effect in our results,
and reporting stop latency alone conceals it.

### Contributions

1. **A ground-truth methodology** for validating stop-latency measurement, in which the harness
   itself is verified before any claim is made: three conditions inject a known constant delay,
   which must cancel exactly in one viewpoint and appear in full in the other. It does, to the
   millisecond, at all three delays (§3.4).
2. **A quantified map of transport bias** over 16 conditions, 2 viewpoints and 2 detectors,
   3,720 measurements with zero missed detections (§4). Several results are counterintuitive; two
   are useful negatives.
3. **A viewpoint distinction** which we argue is a conceptual error in current practice, not
   merely a source of noise (§4.8).
4. **A detector recommendation** with its cost consequence made explicit through a simulated power
   analysis driven by the measured error distributions (§4.9, §6).
5. **An offline-validated measurement path for live systems**, including two findings that
   changed our preregistration before any money was spent: mono-mix recovery is unrecoverable by
   ~680 ms with no usable in-band diagnostic (§5.1–5.2), and a media bridge tapped at the wrong
   interface fabricates the primary effect (§5.4).
6. **A preregistered protocol** for the behavioural question — does transport change when an agent
   *decides* to stop? — with sample sizes derived from measured residuals rather than assumed
   (§6).

### What we deliberately do not claim

We measure simulated transports applied to read speech, not live calls, and our agent is a hard
cut: a perfect interrupter. This paper is about the *measurement*, and we are explicit throughout
about where that boundary lies (§7). We also decline to oversell the effect size. For scale, a
public telephony experiment reports a ~640 ms gap between local-VAD and transcript-based stop
latency; our transport effects top out near 96 ms for a single impairment. Transport does not
explain that gap — the VAD-versus-transcript method choice dominates it. Transport is a
second-order confound worth roughly 100 ms, which still matters when systems are separated by
50 ms, and should not be inflated into the main story.

---

## 2. Background and related work

**Full-Duplex-Bench** supplies the definition in (1) and the segment merge-gap constants (0.6 s
for the user channel, 0.5 s for the model channel) used to turn frame-level VAD output into
speech segments. We reimplemented the definition and those constants from the published equation
and documented parameters rather than copying code; FDB is licensed CC BY-NC.

**Neural VAD in evaluation harnesses.** FDB and comparable harnesses use a neural VAD. We use
Silero VAD as the representative of that class, with one deliberate change: `speech_pad_ms=0`.
Silero's default pads detected segments outward by 30 ms, which is correct when the goal is to
excise speech from audio without clipping it, and wrong when the goal is to time its edges.
Leaving the default in place would have contributed a known systematic offset to every number.

**Energy-based endpointing** is the other detector we characterise: a noise-floor-relative
threshold with a hangover. It is the detector a practitioner writes in an afternoon, and it is
usually treated as the inferior option. Our results argue the opposite for this task.

**Telephony measurement practice.** A publicly documented onset-measurement harness
(`telnyx-onset`) takes the user onset from the measuring party's own pristine local copy of the
audio rather than from the transported stream. This is intuitively an improvement — one fewer
degraded signal — and §4.8 shows it systematically *increases* measured bias, for a reason that
turns out to be structural rather than incidental. We read that work for method only; no code was
copied and it carries no licence file.

**Impairment modelling.** Loss is modelled both as i.i.d. Bernoulli frame loss and with a
two-state Gilbert–Elliott burst channel; concealment as either zero-fill or zero-order hold. The
codec path uses ITU-T G.711 µ-law and A-law implemented from the published tables, and Opus at
narrowband and wideband settings. Packetisation, sequencing and timestamping in the live bridge
follow RFC 3550.

---

## 3. Method

### 3.1 Corpus and scene construction

Sixty two-channel scenes are built from LibriSpeech dev-clean, with **disjoint speaker pools** for
the user and agent roles (20 speakers each, verified by test). The agent speaks continuously; a
user interjection is inserted at a chosen time; the agent channel is hard-cut some known latency
later. True stop latencies span 200–1200 ms.

One construction detail is load-bearing. Natural read speech contains pauses, and if the agent's
hard cut lands inside one there is no acoustic cessation for any detector to find. Phase 1 avoids
this by construction (the cut is placed in voiced speech); §5.3 shows what happens when it cannot
be avoided, and why this becomes a validity precondition rather than a nuisance.

### 3.2 The 16 conditions

Frozen in `stopbias/conditions.py` before analysis:

| family | conditions |
|---|---|
| reference | 16 kHz PCM offline file |
| codec / rate | µ-law and A-law at 8 kHz, analysed natively and upsampled to 16 kHz first |
| passband | 300–3400 Hz PSTN band limit |
| loss | 1 / 3 / 5% frame loss, Bernoulli; 5% bursty (Gilbert–Elliott); zero-fill vs zero-order-hold concealment |
| buffering | a 60 ms fixed jitter buffer standalone; 40 ms and 80 ms inside the composites |
| wideband codec | Opus at 12 kbps / 4 kHz cutoff and 24 kbps / 8 kHz cutoff |
| noise | 20 dB SNR line noise |
| composite | a full typical PSTN leg and a degraded one, applied as whole chains |

The two composites exist to test additivity: they let the single-impairment numbers be checked
against the combination without running a full cross-product.

### 3.3 Viewpoints and detectors

**Two viewpoints.** `wire` reads both boundaries off the transported stream, as a SIP capture
harness does. `local_user` reads the user onset from the measuring party's own pristine copy, as
`telnyx-onset` does. §4.8 argues these measure two different physical quantities.

**Two detectors.** `silero` (neural, `speech_pad_ms=0`) and `energy` (noise-floor-relative
threshold with hangover), behind one interface, with FDB's merge gaps applied identically to both.

Everything is seeded — `BOOT_SEED = 20260825`, with per-channel seeds derived by CRC32 from
(trial, condition, channel) — so a rerun reproduces every number exactly.

### 3.4 Validating the harness before trusting it

No finding is worth anything unless the instrument is verified, so three conditions inject a known
constant transport delay and the harness must recover each one in a specific, non-obvious pattern:

| injected delay | measured bulk delay | `energy` wire | `energy` local_user |
|---|--:|--:|--:|
| 40 ms (`pstn_typical`) | **+40.0** | **0.0** | **+40.0** |
| 60 ms (`nb8_ulaw_jb60`) | **+60.0** | **0.0** | **+60.0** |
| 80 ms (`pstn_poor`) | **+80.0** | −5.0 | **+80.0** |

The `wire` column is the prediction that matters: a delay applied to both channels shifts both
boundaries equally and must **cancel exactly** out of their difference. In the 60 ms condition the
onset error moves −15.7 → +44.3 ms and the offset error +16.9 → +77.0 ms — both by exactly
+60.0 — while the stop-latency error does not move at all (+32.8 vs +33.1 ms at the reference).
The `local_user` column is the complementary prediction: the same delay applied to one channel
only must appear in full. It does, to the millisecond, at all three delays (p ≤ 4.6 × 10⁻¹²).

Recovering the right answer in the right place at three delays, with exact cancellation where
cancellation is predicted, is what licenses the rest of the results. Figure 2 shows it directly:
at the jitter conditions both boundary bars jump and the stop-latency bar stays flat.

### 3.5 Statistics

Medians with 10,000-resample bootstrap percentile confidence intervals. Bias against the 16 kHz
reference is a **paired within-trial difference** — same speakers, same onset, same true latency —
tested with a Wilcoxon signed-rank test. Because Wilcoxon uses all the paired information rather
than only the median, it sometimes rejects where a median CI cannot; §4.9 explains the resolution
limit that produces this pattern for the neural detector, and we report it rather than smoothing
it over.

---

## 4. Results

3,720 measurements, 60 trials × 16 conditions × 2 viewpoints × 2 detectors. **Zero missed
detections in every cell.** Full tables in `results/phase1_tables.md`; figures in
`results/figures/`.

### 4.1 G.711 narrowband does not bias the median at all

This is the surprise. 8 kHz µ-law, A-law, analysed natively or upsampled to 16 kHz first:
**relative bias 0.0 ms for both detectors in both viewpoints.**

| condition | `silero` wire bias | p | `energy` wire bias |
|---|--:|--:|--:|
| µ-law 8 kHz, VAD @ 8 kHz | +0.0 [0, 0] | 0.76 | +0.0 |
| µ-law 8 kHz, upsampled to 16 kHz | +0.0 [0, 0] | 0.96 | +0.0 |
| A-law 8 kHz | +0.0 [0, 0] | 0.41 | +0.0 |

Log-companded quantisation raises the noise floor substantially and moves neither an
adaptive-threshold boundary nor a neural VAD's boundary. The sample rate is equally innocent:
whether the detector runs at 8 kHz or sees upsampled audio makes no measurable difference. **A
µ-law-only leg needs no correction** — which also means a harness that "handles telephony" by
upsampling has not addressed anything.

### 4.2 But it doubles the spread, which is a different problem

| detector, condition | IQR | P95 absolute error |
|---|--:|--:|
| `silero`, 16 kHz file | 31.3 ms | 67.9 ms |
| `silero`, µ-law 8 kHz | **63.5 ms** | **128.6 ms** |

The median is unbiased and the per-call number is twice as noisy. Since a CI on the median widens
as spread/√n, this **roughly quadruples the number of calls** needed to resolve a given difference
between two systems. Accuracy and precision degrade independently here, and only accuracy is zero.

### 4.3 The passband is what actually biases a neural VAD

| detector / viewpoint | offset error | relative bias | p |
|---|--:|--:|--:|
| `silero` / wire | 13.7 → **83.2 ms** | **+32.0** [0, 64] | 1.0 × 10⁻⁴ |
| `silero` / local_user | 13.7 → **83.2 ms** | **+64.0** [32, 64] | 1.4 × 10⁻⁸ |
| `energy` / wire | 16.9 → 18.1 ms | +0.0 | 7.7 × 10⁻⁴ |

Removing energy below 300 Hz makes Silero hold the speech segment open ~70 ms past the true cut.
The energy detector is untouched. So the impairment that matters on a telephony leg is not the
codec but the high-pass, and it only matters if the detector is a speech model.

### 4.4 Opus is the largest single offender

| condition | `silero` wire bias | p | `energy` wire bias | p |
|---|--:|--:|--:|--:|
| Opus 12 kbps, 4 kHz cutoff | **+80.0** [64, 96] | 2.8 × 10⁻⁹ | +12.5 [10, 15] | 1.2 × 10⁻⁶ |
| Opus 24 kbps, 8 kHz cutoff | **+32.0** [32, 64] | 7.7 × 10⁻⁹ | +10.0 [5, 12.5] | 1.5 × 10⁻⁷ |

Silero's offset error under narrowband Opus reaches **105 ms**. Part of this is not detector error
at all: a codec's decoder emits a real energy tail after a hard cut, so some of what the VAD
detects genuinely exists on the wire. Either way it is a transport artefact, and it means
**WebRTC-transported measurements are biased ~2.5× more than G.711/SIP ones** — the opposite of
what "PSTN is the degraded path" would predict, and directly relevant because WebRTC is the
transport most evaluation harnesses reach for when they move off files.

### 4.5 Concealment matters more than the loss rate

| condition | `silero` offset err | `silero` wire bias | p |
|---|--:|--:|--:|
| 1% loss, zero-fill | 15.9 ms | +0.0 | 0.71 |
| 5% loss, zero-fill | 32.5 ms | +0.0 [0, 32] | 0.064 |
| 5% **bursty** loss, zero-fill | **56.5 ms** | **+16.0** [0, 48] | 0.0025 |
| 5% loss, zero-order-hold PLC | 21.9 ms | +0.0 | 0.86 |

The same 5% loss rate gives three different answers. Bursty loss shifts the offset and inflates
the IQR to 117 ms; a trivial zero-order-hold concealer removes the bias entirely. **Reporting a
loss percentage without stating the burst model and the concealment strategy does not specify the
condition** — which is a reporting requirement, not a footnote.

### 4.6 Line noise makes an agent look *faster* than it is

The only condition producing a **negative** bias, and it is large:

| detector / viewpoint | onset err | offset err | relative bias | IQR | P95 abs |
|---|--:|--:|--:|--:|--:|
| `energy` / wire, 20 dB SNR | −15.7 → **+1.8** | 16.9 → **+8.3** | **−50.0** [−70, −30] | 90.0 | **428.9 ms** |

Noise raises the floor; the adaptive threshold rises with it; the onset is found *later* and the
offset *earlier*. Both errors push measured stop latency **down**. An energy-gated harness on a
noisy line credits an agent with 50 ms it never earned, with a P95 absolute error of 429 ms.
Silero moves the other way (+48 ms) and does not break.

This is the most dangerous result in the set precisely because it biases in the flattering
direction: a system evaluated on a noisier line looks better, so the artefact will not be
questioned.

### 4.7 The impairments are approximately additive

| detector / viewpoint | typical leg: observed / predicted | poor leg: observed / predicted |
|---|--:|--:|
| `silero` / wire | **+32.0 / +32.0** | +64.0 / +48.0 |
| `silero` / local_user | +96.0 / +106.7 | +160.0 / +181.3 |
| `energy` / wire | **+0.0 / +0.0** | −5.0 / +0.0 |
| `energy` / local_user | **+40.0 / +40.0** | **+80.0 / +80.0** |

Residuals are ≤32 ms, exactly one Silero analysis frame — at or below the resolution of the
measurement. There is **no evidence of interaction between impairments**, which is a useful
negative: characterising impairments one at a time is a valid way to predict a composite leg, and
nobody needs to run the full cross-product.

The realistic worst case is worth stating plainly. A congested mobile leg (300–3400 Hz + µ-law +
5% bursty loss + 80 ms jitter) measured with Silero gives a **+64 ms median bias with a 270 ms P95
absolute error** in the wire viewpoint, and **+160 ms** in the local-user viewpoint.

### 4.8 A local copy of the user audio makes the bias *worse*

For `silero`, the local-user bias is at least as large as the wire bias in **every** condition and
often double: passband +64 vs +32, poor leg +160 vs +64, 60 ms jitter +64 vs 0.

The mechanism is §3.4: the wire viewpoint gets **error cancellation for free**, because a
common-mode transport delay hits both boundaries and subtracts out. Taking the onset from a
pristine local copy destroys that cancellation and exposes the full one-way return delay.

But the conclusion is not "wire is the better method", and this is the conceptual point of the
paper. **The two viewpoints measure two different physical quantities:**

- **wire** ≈ how long the agent took to stop *at the agent*, with transport delay cancelled.
- **local_user** ≈ how long the interruption took *as the user experienced it*, transport delay
  included — because the user really does keep hearing the agent for the duration of the return
  path.
- **offline 16 kHz files** measure the first quantity, in a system containing no transport at all.

Neither is wrong. Conflating them is, and current practice does: a published offline number is an
agent-internal reaction time being read as a user-experienced one. The gap between them is not
noise — it is the return-path delay plus codec tail, and it is what these tables quantify. A
benchmark should state which quantity it reports.

### 4.9 The detectors fail in orthogonal ways, and one failure is fixable

| detector | reference bias | IQR range (all conditions) | P95 range |
|---|--:|--:|--:|
| `energy` | **+33.1 ms** [32.5, 34.0] | **3.5 – 90.0 ms** | 36.6 – 428.9 ms |
| `silero` | **−6.3 ms** [−10.8, +4.0] | 31.3 – 139.5 ms | 67.9 – 271.5 ms |

The energy detector carries a large but extremely stable systematic offset — +33 ms with an IQR of
3.6 ms. That is a framing convention, and a constant is **subtractable**. Silero is nearly
unbiased at the median with ~10× the spread, which is **not** subtractable.

The practical consequence is the most directly actionable result here. At n=60 the energy detector
resolves bias to ~1 ms, while Silero's 32 ms frame quantisation means a 60-trial study **cannot
resolve a bias below one frame**. That is the source of the apparent conflicts in the tables — the
typical PSTN leg under `silero`/wire shows a +32 ms median bias with a bootstrap CI of [0, 64] and
p = 0.061. The sign is consistent and Wilcoxon often rejects where the median CI cannot, but the
honest reading is that **Silero-based bias estimates below ~32 ms are not resolvable at this
sample size.**

Anyone building a stop-latency harness should calibrate an energy gate and subtract its constant
rather than reaching for a neural VAD for its apparent accuracy.

### 4.10 The bias is additive, not multiplicative

Figure 4 plots measured against true stop latency across 200–1200 ms. The condition clouds sit
**parallel to the identity line** — offset vertically, not rotated — so a single constant per
condition describes the bias and it does not scale with the latency being measured. The visible
exception is a handful of degraded-leg outliers pulled 100–400 ms *below* the line, where a loss
burst split the final speech segment and the harness took an early segment end as the stop. That
failure mode is worth naming because it, too, errs fast.

### Figures

| figure | question it answers |
|---|---|
| `fig1_stop_error_by_condition.png` | which conditions move the stop-latency error, and by how much |
| `fig2_boundary_decomposition.png` | the cancellation of §3.4: both boundaries jump, the difference does not |
| `fig3_viewpoints.png` | wire vs local_user, per condition |
| `fig4_true_vs_measured.png` | additive vs multiplicative bias across 200–1200 ms |

---

## 5. From measurement bias to a live protocol

Sections 1–4 characterise the measurement. The behavioural question they raise — does transport
change when an agent *decides* to stop, not merely when a harness thinks it stopped? — needs live
systems. Before spending anything, we built the measurement path for that study and validated it
the same way: by putting the **real** code inside simulated calls whose latency is known by
construction. This is validation a paid pilot cannot perform, because a real call supplies no
ground truth. Two of the results changed the design.

### 5.1 Mono-mix recovery fails by ~680 ms

A live provider returns a recording; whether it separates the parties is not ours to choose. All
three possible layouts were implemented and tested. Recovered minus true latency, median over
true latencies of 200/500/900 ms × 3 seeds:

| transport leg | dual channel | mono mix |
|---|--:|--:|
| clean | +15 ms | +15 ms |
| 20 dB line noise | +15 ms | +15 ms |
| µ-law 8 kHz | +15 ms | **+685 ms** |
| 300–3400 Hz passband | +20 ms | **+680 ms** |
| 5% bursty loss, zero-fill | +15 ms | **+680 ms** |
| Opus NB 12 kbit/s | +35 ms | **+695 ms** |
| full PSTN leg (composite) | +20 ms | **+685 ms** |

Dual-channel recovery holds from 100 to 1200 ms of true latency with both detectors; the residual
+12 to +35 ms is the subtractable framing constant of §4.9.

The mono failures are not noise, and they are structured. The recovered "stop" is the *end of the
interjection*: residual leakage of the harness's own audio holds the detector open past the
agent's real cessation. Notably the *onset* half of the mono method works — because the harness
knows exactly what it emitted, the onset can be located in the provider's recording by matched
filtering, so both boundaries sit on one clock and transport delay cancels (verified: spread
< 25 ms across 0/60/150 ms of delay). It is the offset that cannot be recovered. The interjection
sits ~15 dB above the agent; after the passband both are speech inside the same 300–3400 Hz band;
and the channel is nonlinear (companding) and time-varying (loss, jitter). Cancellation would
require suppressing the user by >15 dB across the whole band at every instant. Two independent
cancellers were tried — a delay-aligned, reweighted least-squares FIR and a time-varying per-bin
STFT gain — and neither manages it.

### 5.2 No in-band diagnostic separates the usable cases from the broken ones

This is what makes §5.1 a hard constraint rather than a quality-control problem. Both diagnostics
the recording affords overlap between working and failing cases:

| leg | probe suppression | residual/reference correlation | error |
|---|--:|--:|--:|
| clean | 74.7 dB | 0.025 | +15 ms |
| µ-law 8 kHz | 38.1 dB | 0.028 | +685 ms |
| 5% bursty loss | 7.1 dB | 0.058 | +680 ms |
| Opus NB 12 kbit/s | 5.0 dB | 0.031 | +695 ms |

Suppression is not monotone in the error, and the correlation guard fails because the leakage is
*spectrally distorted* — low correlation against the clean reference, ample energy. Bad trials
therefore cannot be flagged and dropped. Our implementation refuses the mono path by default and,
when explicitly opted into, labels the result `mono_mix_unvalidated` rather than `ok`.
**Dual-channel recording is a hard requirement**, which turns a one-call, $0.05 check on the
provider's recording format from an informative step into a decisive one.

### 5.3 A stop that lands in the agent's own pause is unmeasurable, not fast

Constructing this case makes **both** detectors read up to 320 ms early on otherwise clean audio,
because there is no acoustic cessation to find. Phase 1 avoids it by placing the cut in voiced
speech; a live study cannot, since the latency is the agent's own behaviour. It therefore becomes
a **precondition on trial validity**: the runner must confirm continuous agent voicing across the
measured window and void the trial otherwise. Left unhandled it would bias a live study *toward*
finding agents faster than they are — the same direction as §4.6, which is the direction that does
not get questioned.

### 5.4 A media bridge tapped at the wrong interface manufactures the primary effect

The live design is 2 architectures × 2 transports. Its fourth cell — a speech-native model reached
over the PSTN — requires a SIP↔WebSocket media bridge, because speech-native models speak PCM over
a socket and nothing else. Without that bridge, architecture and transport are perfectly
confounded and the architectural hypotheses cannot be tested at all.

The bridge's media plane is built and validated by the same method: the real bridge — real
packetisation, real loss, real reordering, real concealment, real transcode — placed inside a call
whose latency is known by construction. The finding is that **a bridge is a measurement
instrument.** It exists in exactly one cell of the design, so any delay it contributes is
arithmetically indistinguishable from that cell's transport effect. Writing `D_in` for the playout
buffer, `D_out` for the outbound framing quantum and `L` for the model's true stop latency:

```
model-side tap:  onset at D_in,  stop at D_in + L         →  L
SIP-side tap:    onset at 0,     stop at D_in + L + D_out  →  L + D_bridge
```

Sweeping the buffer depth from 0 to 6 frames (20 to 140 ms of bridge delay), at a true latency of
500 ms with the `energy` detector:

| depth | D_bridge | model-tap error | SIP-tap error | SIP − model |
|--:|--:|--:|--:|--:|
| 0 | 20 ms | +15 ms | +35 ms | 20 ms |
| 1 | 40 ms | +20 ms | +60 ms | 40 ms |
| 2 | 60 ms | +20 ms | +80 ms | 60 ms |
| 4 | 100 ms | +15 ms | +115 ms | 100 ms |
| 6 | 140 ms | +15 ms | +155 ms | 140 ms |

**All 140 ms lands in the SIP viewpoint and none of it in the model viewpoint.** The bridge's
delay is common mode at the model interface and cancels exactly — the same argument that makes the
`wire` viewpoint of §4.8 unbiased, arrived at independently in a different layer of the system. At
the default depth the SIP tap carries 60 ms of pure fabrication against a 50 ms target effect.
That is not extra noise: it is an effect of the right size and the right sign, appearing in
exactly the cell where the hypothesis predicts one.

The tap point is therefore a preregistered measurement decision rather than an implementation
detail. Both taps are still recorded, because their difference *measures* `D_bridge` per call —
recovered as exactly 60.0 ms of a true 60 ms on all nine carrier legs — giving a per-call validity
check on the arithmetic above instead of a claim about it. That check must use `energy`: Silero's
32 ms frame quantisation leaves it unable to resolve 60 ms to better than a frame (it reads a
median 72 ms across the grid), which is §4.9's detector argument turning up in a second, unrelated
place.

### 5.5 A fixed playout buffer keeps the delay constant even when the call is wrecked

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

Error is flat to within 0.6 ms across every leg, and the 80 ms jitter row is the point: 142
packets arrive too late to play and 42% of frames reach the model as concealment, yet the bridge's
delay has not moved and the measurement is unaffected. An **adaptive** buffer would have grown to
absorb that jitter, and its delay would then vary with network conditions — which are the
treatment. A fixed buffer converts jitter into a countable concealment rate instead, which is
reportable per trial and cannot masquerade as latency.

Clock drift is left uncorrected for the same reason: correcting it means inserting or dropping
samples inside the window being measured, and it does not need correcting. Drift is a timescale
error costing `t × ppm`, so at 200 ppm with an interjection 2.5 s in it is 0.5 ms — measured at
0.6 ms, three orders of magnitude below the effects in scope.

Three further rules are enforced by tests that fail if the code is added: **no VAD, DTX, comfort
noise or gain control** anywhere in the bridge, because a bridge that made its own speech/silence
decision would place a second endpointer inside a path whose endpointing is the measurement;
**no stateless per-frame resampling**, which gives every frame boundary its own filter transient
and measured 10× the round-trip error of block mode (−30 dB against −51 dB); and **no multi-codec
SDP offer**, since a carrier that quietly answers a wideband codec would put a wideband leg in a
cell preregistered as narrowband and every downstream number would still look reasonable.

One property comes for free: in this cell **the bridge is the recorder**, so dual-channel
separation is structural and the mono-mix failure of §5.1–5.2 cannot arise there at all. The
return leg cannot bias the measurement either, because the tap is upstream of it.

What this does *not* validate: the simulated model is a hard cut, so this establishes that the
bridge does not distort the *measurement* of a stop. It says nothing about how a real model's stop
behaviour responds to 42% concealment. That response is the live hypothesis.

---

## 6. The live study, preregistered

The design, hypotheses, sample sizes, exclusions and stopping rule are frozen before any data
collection (`PREREGISTRATION.md`, three revisions, each recorded rather than overwritten — two
revisions were forced by §5.1–5.2 and one by §5.4).

**Hypotheses.** H1: transport degradation increases *true* stop latency, over and above the
measurement bias of §4. H2: the increase is larger for cascaded (STT→LLM→TTS) systems than for
speech-native ones, because a cascaded barge-in path depends on a transcription-side endpointer
whose input is degraded. H3: architecture × transport interact, so the ranking of systems measured
offline does not survive a phone leg. Direction is predicted for all three. A null result on H3 is
useful: it would mean offline benchmarks are safe for ranking, which nobody has shown.

**Sample size from measured residuals, not assumptions.** Phase 1's empirical per-trial error
distributions are heavy-tailed (P95 up to 3× the IQR) and, for Silero, quantised, so power is
simulated from them rather than derived under normality (paired Wilcoxon, α = 0.05, 80% power,
4,000 simulations per point). Calls needed **per arm** on a degraded PSTN leg:

| detector | to detect 25 ms | to detect 50 ms | to detect 100 ms |
|---|--:|--:|--:|
| `energy`, wire | **40** | **15** | **10** |
| `silero`, wire | 320 | 80 | 30 |

Neither detector resolves 10 ms over a degraded leg at any n ≤ 480, so effects below 25 ms are
declared out of scope in advance. **This is where §4.9 pays for itself: choosing the calibratable
detector makes the study up to 8× cheaper at equal power.** The committed target is n = 60 paired
trials per cell — ≈320 calls including a barge-in-disabled positive control, pilot and the gating
call, at ≈$40–60 total.

One observation the power table raises without settling: a 16 kHz reference condition with a
neural VAD needs ~40 paired trials to resolve 25 ms *even offline with no transport at all*.
Published offline stop-latency comparisons that rank systems on fewer trials than that, with a
neural VAD, may be underpowered for the differences they report. We have not audited any specific
paper and make no claim about one; checking is cheap, and the check is now available.

---

## 7. Limitations

**These are simulated transports, not live calls.** Each impairment is implemented from its
specification and the codec path is verified against `ffmpeg` (>0.99 sample agreement, and
provably never a worse reconstruction), but a real carrier leg is a composition of impairments we
have not enumerated. §4.7's additivity result makes the composition tractable; it does not make
the enumeration complete.

**Read speech has cleaner onsets than spontaneous speech.** LibriSpeech utterances begin more
crisply than conversational interruptions, so the onset-error terms are likely optimistic. The
direction of this limitation is known, which is the most that can be said without a conversational
corpus.

**The agent is a hard cut.** This is the important one. Our agent is a perfect interrupter whose
stop time we chose, so everything above is about the *measurement* of a stop. A real agent's
barge-in path runs its own VAD on the degraded audio arriving inbound, over exactly the
impairments measured here, so transport plausibly changes when the agent decides to stop. That
coupling is §6 and it needs live systems.

**One neural VAD.** Silero represents the neural class; a different model would have a different
frame rate and a different passband sensitivity. The specific numbers in §4.3–4.5 are Silero's.
The structural results — cancellation by viewpoint, additivity, constant-versus-spread — are not
detector-specific, and §4.9's recommendation is about *calibratability*, a property any
constant-offset detector has and any frame-quantised one lacks.

**Effect size in context.** As stated in §1, transport contributes ~100 ms of bias on top of
method choices that contribute several hundred. We report it as a second-order confound that
matters at 50 ms ranking granularity, not as the dominant term in stop-latency error.

---

## 8. Recommendations for benchmark designers

Ordered by how much they change a reported number for how little work.

1. **State which quantity you report.** Agent-internal reaction time (both boundaries off one
   stream) and user-experienced interruption time (transport delay included) differ by the
   return-path delay plus codec tail. An offline file measures the first. Saying which one is free
   and currently absent.
2. **Prefer a calibrated energy gate to a neural VAD for timing.** +33 ms with a 3.6 ms IQR is
   better than −6 ms with a 31–140 ms IQR, because the first is subtractable and the second is
   not, and the difference is up to 8× in sample size (§4.9, §6).
3. **If you use a neural VAD, set its padding to zero and report its frame rate.** A 30 ms
   outward pad is a systematic offset; a 32 ms frame is a resolution floor that no amount of
   averaging within a small n removes.
4. **Do not claim to have handled telephony by upsampling.** The codec and the sample rate cost
   0.0 ms. The passband costs +32 to +64 ms for a neural VAD (§4.1, §4.3).
5. **Specify loss conditions as (rate, burst model, concealment).** One rate gives three answers
   (§4.5).
6. **Treat WebRTC as the more biased transport, not the less.** Narrowband Opus is ~2.5× worse
   than G.711 (§4.4).
7. **Be suspicious of clean, fast numbers from noisy lines.** Noise biases an energy-gated
   measurement 50 ms *fast*, with a 429 ms P95 (§4.6).
8. **Validate the harness with an injected known delay before reporting anything.** It costs
   three extra conditions, it must cancel in one viewpoint and appear in full in the other, and if
   it doesn't, nothing else in the table means anything (§3.4).
9. **Require dual-channel recordings from any provider you measure.** Mono-mix recovery is wrong
   by ~680 ms over a phone leg and cannot be diagnosed from the recording (§5.1–5.2).
10. **If a bridge, proxy or gateway sits in one arm of your comparison, characterise its delay and
    tap upstream of it.** Otherwise the instrument supplies the effect (§5.4).

---

## 9. Reproducibility

Python 3.12+, `ffmpeg`, and ~350 MB for LibriSpeech dev-clean. No API keys, no spend, ~46 s for
the full grid; 144 tests.

```bash
uv venv && uv pip install -e .
.venv/bin/python -m stopbias.cli prep --pairs 60    # download + cut the corpus
.venv/bin/python -m stopbias.cli plan --pairs 60    # freeze the trial plan
.venv/bin/python -m stopbias.cli run                # 3,720 measurements
.venv/bin/python -m stopbias.cli report             # CSV + markdown tables
.venv/bin/python -m stopbias.cli plots              # 4 figures
.venv/bin/python -m stopbias.cli power              # sample sizes for §6
.venv/bin/python -m stopbias.cli bridge             # validate the media bridge (§5.4–5.5)
.venv/bin/python -m pytest tests/ -q
```

Every seed is fixed (`BOOT_SEED = 20260825`, per-channel seeds by CRC32 over trial/condition/
channel), so a rerun reproduces the numbers in this paper exactly. Raw per-measurement records are
in `results/raw/phase1.jsonl`. Each finding above that constrains the design is additionally
pinned by a test, so a regression that would invalidate a claim in this paper fails the suite
rather than silently changing a table.

## 10. Provenance and licensing

The stop-latency definition and merge-gap constants (0.6 s user, 0.5 s model) were reimplemented
from Full-Duplex-Bench's published equation and documented parameters, not copied — FDB is CC
BY-NC and is not used commercially here. `telnyx-onset` carries no licence file, was read for
method only, and no code was copied from it. The corpus is LibriSpeech dev-clean (CC BY 4.0), 20
user and 20 agent speakers in disjoint pools, verified by test. G.711 is implemented from the
ITU-T tables and verified against `ffmpeg`: >0.99 sample agreement and provably never a worse
reconstruction, with residual disagreements traced to `ffmpeg`'s 14-bit-rounded decision
boundaries and an arbitrary tie-break at zero.

---

## References

- Full-Duplex-Bench: a benchmark for evaluating full-duplex spoken dialogue models on turn-taking
  capabilities. (Source of equation (1) and the merge-gap constants; v1.5 parameters used.)
- Silero VAD. Silero Team, pre-trained enterprise-grade voice activity detector.
- V. Panayotov, G. Chen, D. Povey, S. Khudanpur. *LibriSpeech: an ASR corpus based on public
  domain audio books.* ICASSP 2015.
- ITU-T Recommendation G.711. *Pulse code modulation (PCM) of voice frequencies.*
- H. Schulzrinne, S. Casner, R. Frederick, V. Jacobson. *RTP: A Transport Protocol for Real-Time
  Applications.* RFC 3550, 2003.
- J.-M. Valin, K. Vos, T. Terriberry. *Definition of the Opus Audio Codec.* RFC 6716, 2012.
- E. N. Gilbert. *Capacity of a burst-noise channel.* Bell System Technical Journal, 1960; E. O.
  Elliott, 1963. (Two-state burst loss model.)
- F. Wilcoxon. *Individual comparisons by ranking methods.* Biometrics Bulletin, 1945.
- `telnyx-onset`. Public telephony onset-measurement harness; read for method only.

*Citation details for Full-Duplex-Bench, Silero VAD and `telnyx-onset` need to be completed with
exact author lists, venues and versions before submission — they are cited here by title and role
only, which is what we can state without checking the sources.*
