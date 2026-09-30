# Kairos 1.0 — Architecture Specification (Design Draft)



**Status of this document:** design proposal, not a validated result. Nothing below

has been trained or benchmarked yet except where explicitly marked

`[VALIDATED — 2026-09-XX text/banking77 run]`, referring to the real experiment

already run in this project (linear_probe 63.36% / equal_capacity 71.75% ECE 0.033 /

kairos 71.62% ECE 0.463). Every other layer is marked with a status tag so nobody —

including future-you — mistakes a design idea for a measured result. This mirrors

the CIDA paper's own Author Checklist (Appendix F): state plainly what is designed

vs what is measured, always.



**Status legend**

- `[VALIDATED]` — measured on real data in this project already.

- `[PRINCIPLED FIX]` — directly addresses a real, diagnosed failure from the

  validated run above. Strong prior it helps, but still needs its own ablation.

- `[SPECULATIVE]` — plausible, standard technique, but has no evidence for or

  against it in this project yet. Must be ablated before it ships in Kairos 1.0.

- `[RISK-FLAGGED]` — known failure modes from the broader literature that this

  component specifically invites. Listed so they aren't discovered in production.



---



## 0a. Grounding in current research (searched Sept 2026)



Before adding anything below, here is what was actually found — real, dated, with

arXiv/source IDs — not "current trends" asserted from memory:



- **Probabilistic Tiny Recursive Model (PTRM)**, arXiv 2605.19943, May 2026.

  Tiny Recursive Models (5–7M params) iteratively refine a *latent state*, no

  tokens. Their problem: deterministic recursion gets trapped in bad solution

  basins. PTRM's fix: inject Gaussian noise at every recursion step, run several

  parallel noisy rollouts, and use the model's own existing early-stopping head

  (a "Q head") to pick the best trajectory — near-doubling accuracy on

  constraint-satisfaction puzzles (62.6% → 91.2%) with **no retraining**, at a

  tiny fraction of the inference cost of frontier LLMs. This is directly

  relevant — see L5b below.

- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth

  Approach**, Geiping et al., arXiv 2502.05171. A decoder-only model with a

  Prelude → Recurrent Core → Coda structure, iterating the core block an

  arbitrary number of times at test time, trained with a *randomly sampled*

  number of iterations per example for robustness across compute budgets.

  Scaled to 3.5B params / 800B tokens — established, not a toy result. This is

  the closest published precedent for cyclic latent refinement without CoT

  tokens, and grounds L5's original design.

- **Conformal Prediction** (well-established field, actively revisited through

  2026 — e.g. arXiv 2603.10731 comparing it against MC-Dropout on calibration).

  Distribution-free, finite-sample **coverage guarantee**: given a held-out

  calibration set, it produces prediction *sets* with a provable probability of

  containing the true label — a strictly stronger honesty property than a

  softmax-threshold or ECE number, which are both diagnostic, not guaranteed.

  Directly relevant to "хладнокровно, без галлюцинаций" — see the new gate layer

  below.

- **MoE capacity-allocation research** (Nemotron 3, "Scaling Embeddings

  Outperforms Scaling Experts", Step 3.5 Flash — per Sebastian Raschka's 2026

  paper roundup) — active area on *how* to allocate MoE capacity, relevant

  context for L9, not yet dug into deeply enough here to prescribe specifics.

- **Hybrid state-space/attention backbones** (Mamba-3, Nemotron 3) and

  **activation sparsity** work ("The Spike, the Sparse and the Sink") — relevant

  if you ever train a custom backbone instead of reusing Laya; noted for later,

  not load-bearing for Kairos 1.0 since Laya is already the chosen backbone.



---



## 0. Design principle



Every layer below exists to answer one specific question, and has a stated way to

find out if it's earning its place. A layer that only exists to sound advanced and

has no ablation plan does not belong in Kairos 1.0 — that is exactly the failure

mode `role_prior_blend` fell into yesterday (a mechanism inspired by a binary

medical-imaging task, silently broken on a 77-class problem, shipped without a

multi-class-specific test). This document exists so that doesn't happen twice.



---



## 1. Layer-by-layer architecture



### L0 — Typed Action Space (prerequisite, not a model component)

A finite, versioned enum of `{tool_id, slot enums, confidence_threshold}` per

domain. Nothing downstream works without this being defined first — CIDA and Laya

can only score what's enumerated, they cannot invent new actions.

**Status:** prerequisite, no ablation needed — it's a spec, not a trained component.



### L1 — Laya Backbone (421M, frozen or LoRA)

`[VALIDATED]` as a feature extractor — this is the exact model used in yesterday's

run. Fast, single forward pass, typed-decision-native.



### L2 — Conditional Gate (early-exit)

Before touching CIDA at all: if Laya's own softmax confidence on the typed head

clears a threshold, skip straight to L9 (Typed Decision Head) on Laya's own output.

This is your "agentность и скорость" requirement made literal — most easy,

unambiguous cases never pay CIDA's compute cost at all.

**Status:** `[SPECULATIVE]` mechanism (standard early-exit), but the threshold and

skip-rate need measuring on real data — report "% skipped" and "accuracy on

skipped vs escalated" separately, exactly like the confident/uncertain split we

discussed for the agentic benchmark.



### L2b — Conformal Calibration Gate (upgrades L2's threshold to a guarantee)

L2's "confidence high enough?" check, as designed, is a softmax threshold —

diagnostic, tunable, but not *guaranteed* to mean anything in particular. Conformal

prediction gives a strictly stronger version of the same gate: hold out a real

calibration split (separate from train/eval — a third split, not reused), compute

nonconformity scores from CIDA's `p_final`, and derive a threshold `q` such that

the resulting prediction *set* contains the true action with probability ≥ 1-α by

construction, for any α you choose (e.g. 90%, 95%). The escalation rule becomes

precise and falsifiable: **escalate whenever the conformal prediction set contains

more than one action** — "genuinely ambiguous by a provable guarantee," not "the

softmax number is below 0.7 because that felt right." This is exactly the honesty

property "хладнокровно, без галлюцинаций" is asking for, made mathematically

precise instead of asserted.

**Status:** `[SPECULATIVE]` in this exact pipeline position, but the underlying

method is decades-old, standard, and model-agnostic — the risk isn't "does

conformal prediction work," it's "does wiring it after CIDA's specific output

break anything." Needs: a real third calibration split (do not reuse the eval

split you already used for accuracy/ECE — that invalidates the coverage

guarantee), and a report of *empirical* coverage vs the *target* coverage as the

actual validation metric, not accuracy.



### L3 — VAE-Perspective Layer (replaces the broken role priors)

`[PRINCIPLED FIX]`. This directly answers your question — **is a VAE here

tupizm or not** — and the honest answer is: not tupizm, but not free either.



Yesterday's diagnosed problem: `ROLE_PRIOR_POSITIVE = [0.85, 0.15, 0.50, None]`

hard-codes a *binary* "positive/negative" semantic into agent initialization.

On banking77 (77 classes, no meaningful "positive class") this pulls two of four

agents toward an arbitrary index every round, and ECE degraded from 0.033

(equal-capacity control) to 0.463 (kairos) while accuracy barely moved — consistent

with corrupted probability calibration rather than corrupted discrimination.



A VAE-style perspective layer replaces the hard-coded prior with a **learned

stochastic diversity mechanism**: instead of `s_i^0 = W_i^proj h + r_i` (Eq. 1 in

the CIDA paper, deterministic per-agent projection + fixed role embedding), learn

`(mu_i, logvar_i) = f_i(h)` per agent and sample `s_i^0 = mu_i + eps * exp(0.5*logvar_i)`.

The KL term against a shared prior keeps the four agents from collapsing onto the

same point (replacing, or complementing, the orthogonality regularizer, Eq. 2),

**without asserting anything about what "positive" means** — the diversity is

learned from data, not hand-coded from a medical-imaging intuition that doesn't

transfer.



Where it could go wrong (`[RISK-FLAGGED]`): posterior collapse (all four agents

converge to the same mu, KL term goes to zero, you're back to identical agents —

this is a well-known VAE failure mode, not hypothetical), and an extra

hyperparameter (KL weight) to tune badly. Ablation plan: three-way comparison —

old hard role priors vs VAE-perspective vs orthogonality-only (no stochasticity at

all) — on the same banking77 setup, same equal-capacity control discipline as

yesterday. If VAE doesn't beat orthogonality-only on ECE, don't ship it — the

complexity isn't earning its place.



### L4 — CIDA Core (4 agents, T rounds, disagreement-biased attention)

Unmodified from the paper: perspective projection (now stochastic, via L3),

contrastive communication (Eq. 3–4), state update, reliability-weighted consensus.

**Status:** `[VALIDATED]` as a mechanism that measurably changes calibration

(direction currently bad, but the effect is real and large — 14x ECE — so the

machinery clearly has leverage; L3 is the attempt to point that leverage the

right way).



### L5 — Cyclic Refinement Loop ("CIDA as the heart")

Your instinct here was right and it's already half-described in the CIDA paper

itself, just for a different backbone: instead of CIDA collapsing straight to a

scalar belief, it returns **P=2–4 plan vectors** (Section 3.6's plan-token idea,

here applied to Laya instead of Qwen). These vectors are prepended to Laya's input

and Laya re-encodes *with that context*, optionally looping up to a small

`max_cycles` (2–3), each cycle re-checked against the same halting/disagreement

threshold as L2. This is the literal implementation of "CIDA возвращает несколько

векторов, Лая эти подсказки использует и снова пропускает через себя" — it's not

a new idea, it's Coconut-style continuous feedback (already cited as related work

in the CIDA paper, Section 2.3) applied to your specific backbone pair.

**Status:** `[SPECULATIVE]` — architecturally coherent, directly descended from a

cited, published technique, but the interaction with L2/L3 in this exact stack has

never been run. Ablate cycle count (1 vs 2 vs 3) against wall-clock latency — this

is exactly the axis your paper's own Section 9.2 lists as unevaluated ("Latent

Rollback... would make T adaptive per-input... at the cost of variable inference

latency").



### L5b — Stochastic Rollout Refinement (PTRM-style upgrade to L5)

L5's cyclic loop as designed is still deterministic — same input, same rounds,

same output every time. The Probabilistic Tiny Recursive Model paper (arXiv

2605.19943, May 2026) diagnoses exactly this failure mode in a closely analogous

architecture (tiny model, latent iterative refinement, no tokens): deterministic

recursion gets stuck in bad solution basins and *cannot escape them by running

more rounds*, only by injecting randomness and exploring several trajectories in

parallel. Their fix, adapted to Kairos: at each CIDA round, inject Gaussian noise

into the agent states `s_i^t` before the update step; run `N` (e.g. 8–32) noisy

rollouts of the L4→L5 loop in parallel; use the existing `HaltingPredictor`

(`consensus.py`) as the verifier that scores each rollout's trajectory and selects

the best one — this is structurally the same role PTRM's "Q head" plays, and you

already have the component, you'd just be repurposing it as a selector across

rollouts instead of a single-trajectory stop signal. PTRM reports near-doubling

accuracy this way **with no retraining**, purely as an inference-time addition —

which means this can be prototyped and ablated cheaply, without touching L1–L4's

weights at all.

**Status:** `[SPECULATIVE]` in this exact stack, but the source technique is real,

recent (May 2026), peer-reviewable, and reports large, specific, checkable numbers

on its own benchmark — this is about as strong a prior as an unvalidated layer can

have. Ablate rollout width N against latency directly: N noisy rollouts cost

roughly N× the compute of L4/L5 alone, so this belongs strictly behind the L2b

conformal gate (only pay for it on inputs already flagged ambiguous), never on the

fast path.



### L6 — Consensus Compression

The several plan vectors CIDA emits (L5) get compressed into one vector the

downstream head consumes cleanly. You already have the right tool for this —

`LatentAEProjector` in `projector.py` (Perceiver-IO-style cross-attention

bottleneck) — this isn't a new component, it's reusing what's already built rather

than inventing a parallel mechanism.

**Status:** `[VALIDATED]` as existing, working code; `[SPECULATIVE]` as wired into

this specific pipeline position — needs its own smoke test once L3/L5 exist.



### L7 — Typed Decision Head

Final enumerated-action scoring + calibrated confidence. No token generation

anywhere in L1–L7 — this is what makes the whole stack fast.

**Status:** `[VALIDATED]` mechanism (same head style as yesterday's run).



### L8 — Escalation / Fallback

When confidence stays low even after L2→L7, hand off to an external generative

model or a human queue. Architecturally outside the fast path — its latency does

not count against Kairos's speed claims, and it should never be silently skipped

just because it's slow.

**Status:** interface contract, not a trained component.



---



## 2. Optional extensions (Kairos-Pro, not core 1.0 — each needs its own domain to justify existing)



### L9 — Multi-CIDA / Mixture-of-Deliberators (MoE-style)

Instead of one shared CIDA instance, run several **domain-specialized** CIDA heads

(e.g. `ToolCallCIDA`, `ModerationCIDA`, `RoutingCIDA`) behind a lightweight router

that activates only the relevant one per input — sparse activation keeps average

compute low even as total specialized capacity grows, which is how you reconcile

"universal" with a 500M budget (same principle as Qwen3-30B-A3B or Gemma's MoE

variants — real, current technique, not invented). Current research is actively

rethinking *how* to allocate MoE capacity rather than just how many experts to add

(Nemotron 3's hybrid design, and papers like "Scaling Embeddings Outperforms

Scaling Experts" per the 2026 architecture-paper roundups found in this search) —

worth reading before picking a router design, not detailed enough here to

prescribe one.

**Status:** `[SPECULATIVE]`, and only relevant once you actually have ≥2 distinct

agentic domains with real data (not before — one domain means the router has

nothing to route between). `[RISK-FLAGGED]`: router collapse (all inputs routed to

one expert, the others starve and undertrain) and load-imbalance are the standard

MoE failure modes — budget real eval time for this specifically, not just

end-task accuracy.



### L10 — Seed Ensemble (conditional, escalation-tier only)

Run 3–5 independently-seeded CIDA heads **only when L2's gate has already flagged

the input as uncertain** — the ensemble's cost is paid exclusively on the hard

cases, not on every request, preserving the fast path. This also directly answers

your own paper's stated single-seed limitation (Section 8.1) — instead of

reporting one seed and hoping it generalizes, the model's own escalation decision

becomes multi-seed by construction, and ensemble variance becomes an *additional*

uncertainty signal on top of CIDA's internal disagreement.

**Status:** `[SPECULATIVE]` mechanism, but the motivation (fix single-seed

fragility) is `[PRINCIPLED FIX]`-grade, directly citing your own paper's

acknowledged gap.



### L11 — UniEncoder Bridge (multimodality)

A shared projection layer mapping Laya's text-space and a vision encoder's

image-space (JEPA or similar) into one common space CIDA agents can jointly

perceive — generalizing Eq. 1's per-agent projection across modalities, not just

across agents. **This is not a new idea you're bolting on for novelty — it is

explicitly listed in the CIDA paper itself as unevaluated future work**

(Section 9.2, "Multimodal CIDA... allowing the disagreement-seeking communication

mechanism to operate across, rather than only within, a single modality"). Doing

this for real, with a genuine ablation, would be closing a gap the paper's own

authors flagged, not chasing a buzzword.

**Status:** `[SPECULATIVE]`, explicitly future work in the source paper — budget

for this only after L1–L8 are validated singly; a vision backbone under 500M is

its own separate sourcing problem (I-JEPA-Huge alone is 630M, already over

budget — a smaller ViT-B/L variant would need to be found or trained).



---



## 3. Training plan, stage by stage



**Stage 0 — Fix before scaling.** Rerun yesterday's exact banking77 setup with

`role_prior_blend=0` (control) vs the new VAE-perspective layer (L3) vs old

hard priors, same equal-capacity-control discipline. Do not proceed to Stage 1

until one of these clearly beats the equal-capacity MLP on ECE with more than one

seed — this is the load-bearing fix; everything else is decoration on top of it.



**Stage 1 — Core validation on real agentic data.** Replace banking77 with

`Salesforce/xlam-function-calling-60k` (real, 60k examples, verified on HF) and

the abstain-category subset of `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

(real, verified, explicitly designed to test abstain/escalate behavior in

multi-step agentic settings — this is the actual target use case, not a proxy).

Train L1–L4 + L6–L7 (no cycling, no rollouts, no ensemble, no MoE yet). Report

accuracy, ECE, and — new metric — confident-decision accuracy vs escalation rate,

separately. Split the data three ways here, not two: train / eval / a held-out

**conformal calibration split**, reserved untouched for L2b — reusing the eval

split for calibration invalidates its coverage guarantee before you've even

measured it.



**Stage 1b — Conformal gate (L2b).** Fit the conformal threshold on the reserved

calibration split from Stage 1. Validation metric here is *empirical coverage vs

target coverage* (e.g. "did the 90%-target set actually contain the true action

90% of the time on held-out data"), not accuracy — a well-calibrated gate can

still have mediocre accuracy, and a badly-calibrated one can look accurate by

luck. Do this before Stage 2, since Stage 2's rollout budget should only be spent

behind this gate.



**Stage 2 — Cyclic loop (L5) and stochastic rollouts (L5b).** Only after Stage 1

and 1b are solid. Two separate ablations, don't conflate them: (a) deterministic

cycle count 1/2/3 (L5) against accuracy and latency; (b) rollout width N (L5b)

against accuracy and latency, gated behind L2b so its cost is only paid on

already-ambiguous inputs. Report both against the PTRM-style claim explicitly —

if N rollouts don't meaningfully beat 1 deterministic pass at N× the latency

budget, that's a real negative result worth reporting, not a reason to quietly

drop the comparison.



**Stage 3 — Escalation-tier ensemble (L10).** Add only on inputs the Stage-2 gate

already flags uncertain; measure whether ensemble variance improves calibration

beyond CIDA's own disagreement signal (and beyond L5b's rollout-selection signal,

now that it exists), or is redundant with either.



**Stage 4 — MoE / multimodality (L9, L11).** Only once you have ≥2 real

agentic domains with real datasets for L9, and a real sub-500M-compatible vision

backbone sourced for L11. Do not build the router or the modality bridge before

the data that would justify them exists — an untrained router with one expert is

not a mixture of experts, it's a linear layer with extra steps.



---



## 4. Release checklist — before calling anything "Kairos 1.0"



Mirroring your own paper's Appendix F discipline, publicly and in the model card:



- [ ] Which layers above are `[VALIDATED]` at release time, listed explicitly —

      not implied by the architecture diagram existing.

- [ ] Multi-seed results (≥3 seeds) for every reported accuracy/ECE number —

      the single-seed gap flagged in the CIDA paper's own Section 8.1 does not get

      repeated in Kairos 1.0.

- [ ] Conformal coverage reported empirically (target vs actual) for L2b, on a

      calibration split that was never used for training, ECE, or accuracy

      reporting — a coverage guarantee claimed on a reused split is not a

      guarantee.

- [ ] Rollout width (L5b) reported against latency explicitly, including the

      negative case if N rollouts didn't beat 1 pass at N× cost — do not report

      only the configuration that looked best.

- [ ] Equal-capacity control reported for every claimed "deliberation helps" line,

      exactly like yesterday — no headline number without its parameter-matched

      control sitting next to it.

- [ ] Explicit non-goals: state plainly this is not a general chat model, it is a

      typed-decision agentic core with a fast path and an honest escalation path.

- [ ] Known-broken/unvalidated layers named, not silently omitted (e.g. if L9/L11

      ship disabled by default because their data isn't ready, say so).
