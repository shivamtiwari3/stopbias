# Phase 2 preregistration — does transport change when an agent *decides* to stop?

**Frozen:** 2026-08-25, before any live data collection. Phase 1 (`RESULTS.md`) is complete and
its results inform the design below, in particular the sample sizes.

**Revision 2, 2026-08-25, still before any live data collection.** §3 was rewritten after the
recovery code was built and validated offline against simulated calls with known ground truth
(`stopbias/recover.py`, `stopbias/simcall.py`, `tests/test_recover.py`). That work falsified two
claims in revision 1 and both are corrected below rather than quietly dropped: the mono-mix
fallback does not work over a telephony leg, and the in-call probe cannot measure the round-trip
delay that the `local_user` correction needs. The hypotheses, sample sizes, exclusion rules and
stopping rule are unchanged.

**Revision 3, 2026-09-02, still before any live data collection.** The SIP↔WebSocket bridge's
media plane has been built and validated offline against calls with known ground truth
(`stopbias/bridge.py`, `stopbias/bridgecall.py`, `tests/test_bridge.py`,
`results/phase2_bridge.md`). That work added one measurement decision to §3 that revision 2 did
not contain and could not have: **which interface of the bridge the recording is tapped at.**
Measured, the wrong choice contributes up to 140 ms — 60 ms at the default configuration — to
exactly one cell of the design, in the direction H1 predicts and at the size §5 is powered to
detect. It is fixed below before any call is placed. §2 and §9 are updated for what is now built
and what is still missing. Hypotheses, sample sizes, the analysis plan, exclusions and the
stopping rule are unchanged.

**Not yet authorised.** Phase 2 spends money on live API and telephony calls. Nothing in this
document has been executed. Section 9 lists what must be approved and what it costs.

---

## 1. The question Phase 1 could not answer

Phase 1 established that transport biases the *measurement* of stop latency by 0–96 ms, with
the bias coming from the passband, Opus, loss burstiness and line noise rather than from the
codec or sample rate. But Phase 1's agent was a hard cut — a perfect interrupter whose stop
time was chosen by the experimenter.

A real agent is not a hard cut. Its barge-in path runs its own VAD or endpointer on the
degraded audio arriving **inbound**, over exactly the impairments Phase 1 characterised. So
transport plausibly changes **when the agent decides to stop**, not merely when a harness
thinks it stopped. Those are different effects and Phase 1 can only see the second.

**H1 (behavioural).** Transport degradation increases true stop latency — the agent genuinely
reacts later over a degraded leg, over and above measurement bias.

**H2 (architectural).** The increase is larger for cascaded (STT→LLM→TTS) systems than for
speech-native ones, because a cascaded barge-in path depends on a transcription-side endpointer
whose input is degraded, whereas a speech-native model consumes audio directly.

**H3 (interaction).** Architecture × transport interact: the ranking of systems measured
offline does not preserve the ranking measured over a phone leg. This is the hypothesis that
matters for whether published offline numbers are usable, and it is the one Phase 1 could
not test at all.

**Direction is predicted for all three.** H1 and H2 predict positive effects; H3 predicts at
least one rank inversion. A null result on H3 is a publishable and useful outcome: it would mean
offline benchmarks are safe to trust for ranking, which nobody has shown.

## 2. Design

2 architectures × 2 transports, fully crossed, within-item paired.

| | socket / WebRTC | PSTN (8 kHz G.711 over SIP) |
|---|---|---|
| **cascaded** | Novis via embed WebRTC (coturn) | Novis via `/telephony/initiate-call` |
| **speech-native** | GPT-Realtime, Gemini Live via WebSocket | same models bridged to SIP |

**The confound this design exists to avoid.** Novis is reachable both ways; GPT-Realtime and
Gemini Live natively are not. If the cascaded arm ran only on PSTN and the speech-native arm
only on sockets, architecture and transport would be perfectly confounded and no comparison
would mean anything. Filling the bottom-right cell therefore requires a SIP↔WebSocket media
bridge, and that bridge is the single largest piece of engineering in Phase 2. **It is not
optional.** If it cannot be built, the study reduces to two separate within-architecture
transport comparisons and H2/H3 are abandoned — stated here in advance so that outcome cannot
be quietly reframed as success.

**Bridge status (revision 3).** The media plane is built and validated offline: RTP framing, a
fixed-depth playout buffer with concealment, G.711 transcoding to and from the model socket's
rate, a four-channel tap, and single-codec SDP negotiation. Its contribution to the measurement
is characterised rather than assumed — flat to within 0.6 ms across nine carrier legs including
one where 42% of frames are concealed (`RESULTS.md` §14–16). The **signalling** plane (SIP dialog,
registration, digest auth) is delegated to an existing stack: it cannot be exercised without a
carrier, and unlike a timing error it fails loudly, at call setup. If the carrier instead offers
bidirectional media over a WebSocket, the media plane attaches to that and the RTP layer is
unused; the timing rules below are unchanged either way. **The bridge is therefore no longer a
risk to the design, only to the schedule.**

**Pairing.** The same interjection stimulus, at the same offset into the same agent turn, is
used in every cell. Analysis is on within-item differences, which is what the sample sizes in
§5 assume.

**Barge-in as a manipulation check.** Novis exposes `allow_interrupt` as a per-node boolean.
Running matched trials with it off gives a positive control: stop latency must be
indistinguishable from "never stops" when barge-in is disabled. If it isn't, the harness is
measuring something other than barge-in.

## 3. What is measured, and the gating problem

Novis exposes `recording_url`, `transcript_url`, `usage_info`, `cost_info` and
`duration_seconds` per run. It exposes **no per-turn timestamps and no latency telemetry** — I
checked all 175 endpoints and every schema property; the only `latency_ms` field in the API
belongs to a model-test endpoint, not a conversation. So stop latency must be recovered from
audio, using the same measurement code as Phase 1. That is good for comparability and it
creates one hard dependency:

> **Gating experiment (1 call, ~$0.05).** Determine whether `recording_url` is dual-channel
> (user and agent separated) or a mono mix.

**This experiment is decisive, not merely informative: if the recording is a mono mix, Phase 2
as designed cannot proceed on that transport.** Revision 1 treated a mono mix as a survivable
fallback. Building the recovery code and validating it against simulated calls with known ground
truth showed that it is not, so the gate is now a hard one.

- **Dual-channel** → the `wire` viewpoint is available. Both boundaries come from the same
  degraded stream and common-mode delay cancels. Validated offline across a clean leg, µ-law
  8 kHz, the 300–3400 Hz passband, 5% bursty loss, 20 dB line noise, Opus NB 12 kbit/s and a
  full PSTN leg: recovered latency is within +12 to +35 ms of truth on every one, with both
  detectors, at true latencies from 100 to 1200 ms. **Required.**
- **Mono mix** → not usable. Half of it works: because the harness knows exactly what it emitted,
  the user onset can be *located in the provider's recording* by matched filtering instead of
  taken from the local clock, so both boundaries sit on one clock and this is still the unbiased
  `wire` viewpoint (revision 1 wrongly said `wire` was unavailable). The agent offset is the
  problem. Recovering it needs the emitted audio cancelled out of the mix, and that fails over
  every impaired leg tested — error +680 to +695 ms, where the recovered "stop" is simply the end
  of the interjection because residual leakage holds the detector open past the agent's real
  cessation. The interjection sits ~15 dB above the agent, after the PSTN filter both are speech
  in the same 300–3400 Hz band, and the channel is nonlinear (companding) and time-varying (loss,
  jitter). Two independent cancellers were tried, a delay-aligned reweighted least-squares FIR
  and a time-varying per-bin STFT gain; neither suffices.

  Critically, **no in-band diagnostic separates the usable cases from the +680 ms ones**, so this
  cannot be managed by flagging bad trials. Probe suppression does not discriminate (a µ-law leg
  scores 38 dB and is still 685 ms wrong; a 5% bursty-loss leg scores 7.1 dB). Residual-vs-
  reference correlation does not either (clean 0.025, failing µ-law leg 0.028 — the leakage is
  spectrally distorted, so low correlation but ample energy). `stopbias.recover` therefore
  refuses the mono path unless explicitly opted into, and then labels the result
  `mono_mix_unvalidated` rather than `ok`.

- **Local recording only** → last resort, and weaker than revision 1 claimed. Phase 1 did
  calibrate the correction: it recovered injected one-way delay exactly and additively at 40, 60
  and 80 ms (bias = +40.0, +60.0, +80.0 ms, p ≤ 4.6e-12), and deriving the timeline shows the raw
  local measurement is `L + d_out + d_ret = L + RTT`, so the **whole** round trip comes off and no
  path-symmetry assumption is needed. But revision 1 asserted the one-way delay "is obtained per
  call by cross-correlating a broadband probe burst played at call start", and that is wrong:
  nothing in the call echoes the probe back, so a harness holding only its own recording can
  measure neither `d_out` nor `d_ret` from it. The probe measures `d_out` only when there *is* a
  provider-side recording to find it in — in which case this case does not apply. RTT must
  therefore come from outside the measurement: an echo-endpoint calibration call on the same
  route, or provider RTCP round-trip statistics. The residual assumption is that the supplied RTT
  applies to the measurement call, which jitter-buffer adaptation can violate.

**Where the bridged arm's recording is tapped (new in revision 3).** For the speech-native PSTN
cell the bridge is the recorder, which settles two things and raises a third.

- Dual-channel separation is **structural** there — the two parties are never mixed — so the
  mono-mix failure above cannot arise in that cell, and the return leg cannot bias it either,
  because the tap is upstream of the return leg. The gating risk survives only for the cascaded
  arm, where Novis owns the recording.
- **The tap is taken at the model-facing interface, not the SIP-facing one.** This is a
  preregistered measurement decision. The bridge's own delay is common mode at the model
  interface and cancels exactly; at the SIP interface it does not, and because the bridge exists
  in only one cell of the design, it would appear there as a transport effect. Measured across
  playout depths of 0 to 6 frames, the SIP tap carries the whole 20–140 ms and the model tap
  carries none of it (`RESULTS.md` §14). At the default depth that is 60 ms — the size of the
  effect §5 is powered to detect, in the direction H1 predicts.
- Both taps are recorded anyway, because their difference measures the bridge's delay per call
  (recovered as 60.0 ms of a true 60 ms on every leg). That is a per-call validity check, reported
  per cell. It uses `energy`; Silero cannot resolve it to better than its 32 ms frame (§4).

**Concealment rate is a reported covariate, not an exclusion.** The bridge counts lost, late,
reordered and concealed frames per trial and all four are reported per cell. They are *not*
grounds for dropping a trial. A call where 42% of frames arrive too late to play is a badly
degraded transport, which is the treatment — excluding it would select for calls that went well
and bias the transport effect toward zero, the same error §7 forbids for latency. The one thing
the concealment counters are used for is diagnosis: a cell whose concealment rate differs greatly
between architectures is not a fair transport comparison, and that is reported rather than
adjusted for.

**Contingency, stated in advance.** If the gating call returns a mono mix and no dual-channel
option exists on that transport, that transport arm is dropped and reported as dropped, with the
reason. It is not silently replaced by mono-mix numbers or by uncalibrated local-only numbers.

**Trial validity precondition.** The interjection may only be injected while the agent is
*continuously* speaking, and a trial in which the agent's own speech pauses across the measured
stop is **void, not a fast stop**. This is not a hypothetical: constructing that case offline
makes both detectors read up to 320 ms early on otherwise clean audio, because there is no
acoustic cessation to find. The runner must verify continuous agent voicing over the window and
void the trial otherwise.

**Primary outcome.** `t_stop = t_model_stop − t_user_start`, per trial, in ms, measured from
audio with the `energy` detector and its calibrated constant subtracted (see §4).

**Secondary outcomes.** Onset and offset error separately (`stop_err = offset_err − onset_err`,
as in Phase 1); whether the agent stopped at all; number of agent speech segments after the
interjection (restart behaviour); per-run `stt_cost_usd` / `llm_cost_usd` / `tts_cost_usd` for
a cost-per-millisecond-saved analysis; one-way delay per call.

## 4. Detector choice, decided in advance

Phase 1 measured that the two detectors fail orthogonally: `energy` carries a large but
extremely stable bias (+33.1 ms, IQR 3.6 ms — a subtractable constant), while `silero` is
near-unbiased at the median but with ~10× the spread and a 32 ms frame quantisation that is not
subtractable.

**Primary analysis uses `energy` with its constant subtracted.** `silero` is reported alongside
as a robustness check, and the two must agree in sign and rough magnitude; disagreement is
reported, not resolved by picking the nicer one.

This is a substantive decision and §5 shows it changes the cost of the study by up to 8×. It
runs against common practice — Full-Duplex-Bench uses a neural VAD — and Phase 1 is the
justification.

## 5. Sample size

Simulated from Phase 1's empirical per-trial error distributions rather than assumed normal,
because those distributions are heavy-tailed (P95 up to 3× the IQR) and, for Silero, quantised.
Paired Wilcoxon signed-rank, α=0.05, 80% power, 4,000 simulations per point
(`results/phase2_power.md`, reproducible via `stopbias/power.py`; the vectorised test is
verified to agree with scipy row-for-row on continuous, quantised, zero-heavy and shifted data).

Calls needed **per arm**, on a degraded PSTN leg (`pstn_poor`):

| detector | to detect 25 ms | to detect 50 ms | to detect 100 ms |
|---|--:|--:|--:|
| `energy`, wire | **40** | **15** | **10** |
| `silero`, wire | 320 | 80 | 30 |

Neither detector can resolve 10 ms over a degraded leg at any n ≤ 480.

**Committed target: n = 60 paired trials per cell.** That gives >80% power for a 50 ms effect
with the primary detector in every condition with margin, and ~80% power for 25 ms in the
cleaner conditions. 4 cells × 60 = 240 calls, plus 60 for the `allow_interrupt` control and
~20 for pilot and gating = **≈320 calls**.

**Effects below 25 ms are declared out of scope in advance.** Detecting them needs 320+ calls
per arm with the robustness detector and the study will not claim them.

A note on the existing literature: `ref_16k` with `silero` needs ~40 paired trials to resolve
25 ms even offline with no transport at all. Published offline stop-latency comparisons that
rank systems on fewer trials than that, using a neural VAD, may be underpowered for the
differences they report. I have not audited any specific paper's sample size and make no claim
about one; this is a hypothesis the power table raises, and checking it is cheap.

## 6. Analysis plan

- Primary: paired Wilcoxon signed-rank on within-item differences, two-sided, α=0.05.
- Medians with 10,000-resample bootstrap percentile CIs, as in Phase 1.
- H1: transport effect within each architecture (PSTN − socket).
- H2: difference of those differences between architectures.
- H3: Kendall's τ between the socket-measured and PSTN-measured system rankings, with a
  bootstrap CI; plus explicit reporting of any pair whose order inverts.
- Multiplicity: H1–H3 are three preregistered primary tests, Holm-corrected across them.
  Everything else is exploratory and labelled as such.
- Both detectors reported for every cell. Both viewpoints reported when both are available.

## 7. Exclusions and attrition, fixed in advance

- A trial is excluded only if: the call failed to connect, the recording is missing or
  truncated before the interjection, the agent produced no speech before the interjection, the
  emitted interjection could not be located in the recording (`interjection_not_located`), or the
  agent's own speech was not continuous across the measured stop (§3, void by precondition).
- These are all conditions on whether a stop event *exists and is observable*, decided without
  reference to the recovered latency. That is the line: a trial can be voided for having no
  measurable stop, never for having an inconvenient one.
- **Failed trials are not replaced.** Attrition is reported per cell with reasons. Replacing
  failures silently selects for calls that went well, which biases latency downward — a live
  call that degrades badly is exactly the observation of interest.
- No trial is excluded on the basis of its measured latency. No outlier removal.
- If attrition exceeds 20% in any cell, that cell's result is reported as unreliable rather
  than analysed as if complete.

## 8. Stopping rule

n = 60 per cell is fixed. No interim analysis of the primary outcome, no stopping early on a
significant result, no extending n after seeing the data. If the pilot (§9, 20 calls) shows
the harness is broken, the pilot is discarded and rerun after fixing it — pilot data does not
enter the analysis.

## 9. Authorisation required, with costs

Nothing below has been run.

| step | calls | est. cost | needs |
|---|--:|--:|---|
| Gating: recording channel layout | 1 | ~$0.05 | Novis key (held) + a workflow |
| Pilot: end-to-end harness validation | 20 | ~$2 | above |
| Cascaded arm, both transports | 120 | ~$15 | above + a number to call |
| Speech-native arm, socket | 60 | ~$8 | OpenAI + Gemini keys (**not held**) |
| Speech-native arm, PSTN via bridge | 60 | ~$10 | above + a SIP trunk; bridge media plane built |
| `allow_interrupt` control | 60 | ~$7 | Novis only |

Estimates are order-of-magnitude from per-minute STT+LLM+TTS+telephony rates for ~1-minute
calls; Novis reports actual per-run `stt_cost_usd` / `llm_cost_usd` / `tts_cost_usd`, so real
spend will be reported per cell rather than estimated. **Total ≈ $40–60 and ≈320 calls.**

Outstanding blockers, as of revision 3:

1. **OpenAI and Gemini API keys.** Not present in this project's environment. Both arms of the
   speech-native cell need them, so 120 of the 320 calls are blocked on credentials alone. A
   Bedrock token does not substitute: Bedrock hosts neither GPT-Realtime nor Gemini Live.
2. **A SIP trunk and a number.** Registrar, credentials, a DID, and G.711 µ-law at 20 ms with no
   wideband codec in the answer (§3, enforced by `bridge.accept_sdp_answer`).
3. **A media host reachable for RTP.** Whatever runs the bridge needs a routable media path. A
   laptop behind NAT is not one, so this is a small cloud instance or a carrier that delivers
   media over a WebSocket instead.
4. **A SIP user agent** for the signalling plane (pjsua2, baresip or equivalent), unless the
   carrier's WebSocket media path is used, in which case none is needed.
5. **Spend authorisation** for the ≈$40–60 above.

The bridge is no longer on this list. Everything remaining is access, not engineering.

## 10. What would falsify this

- **H1 false** if transport-corrected true stop latency does not differ between socket and PSTN
  beyond the Phase 1 measurement bias.
- **H2 false** if the cascaded and speech-native arms show statistically indistinguishable
  transport sensitivity.
- **H3 false** if Kendall's τ between socket and PSTN rankings has a CI excluding values below
  ~0.8 and no pair inverts — which would mean offline benchmarks rank systems correctly and the
  field can keep using them.

Any of these outcomes is reported. H3 being false is the most likely single result and the most
useful negative, because it is currently an untested assumption that the whole field relies on.

## 11. Provenance

Stop-latency definition and merge-gap constants reimplemented from Full-Duplex-Bench's
published equation and documented parameters (FDB is CC BY-NC; not copied, not used
commercially). telnyx-onset carries no licence file, was read for method only, and no code was
copied from it. Phase 1 corpus is LibriSpeech dev-clean (CC BY 4.0). Any Phase 2 stimulus set
will be documented with its licence before use.
