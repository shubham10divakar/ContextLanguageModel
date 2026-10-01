# Research guide: variants of Context Language Models

Written 1 Oct 2026, two days after the CLM paper appeared. It ranks your nine design-doc ideas together with
five new ones against what the paper now covers, and says what to run first with this codebase.

## 0. What the CLM paper changes in your design doc

1. **Name clash.** The paper's diagnostic suite is already called *ContextBench* (four tasks, release "coming
   soon"). Rename yours (for example *KeepSet* or *ContextBench-Oracle*) and position it as complementary. Its
   real differentiators are oracle keep-sets, knowledge updates, distractors, dead ends, cross-session memory
   and injection.
2. **Your #4's premise now runs against the paper's main claim.** The paper argues that unrestricted edits
   (bash on a file) beat constrained action spaces. It also shows that closed API models work *zero-shot* as
   CLMs, which removes the "the reader can't edit itself" motivation. A JSON-op editor with a frozen reader is
   still a valid study, but it is no longer the obvious main paper.
3. **Your #8 now has a concrete, novel attack surface.** Building this replication surfaced two issues:
   header forgery and role laundering (`docs/REPLICATION.md` §4, with a passing test). The paper itself names
   safety of editable context as future work.
4. **Your #1 is nearly free in the CLM framing.** The sandbox already has git. "Fork / revert / merge" are
   `git branch / checkout / merge` on the context file. No new op grammar is needed, so you can test it
   zero-shot before any training.
5. **Edit cost depends on the architecture.** From Eq. 9 (computed with `clm.flops`): a mid-context edit that
   drops 10K of 20K tokens and leaves a 5K suffix only pays back its re-prefill after **≈66 later turns on the
   hybrid Qwen3.6-27B, but ≈16 turns on dense Qwen3-8B** (1K new tokens per turn). The hybrid has only 16
   full-attention layers, so shorter contexts save it little. Prefix caching already makes the linear terms
   independent of context length. Two consequences: CLM's FLOPs savings in the paper probably come from
   trajectory effects (fewer turns, no overflow re-summarization, shorter generations) rather than from
   cheaper attention; and the optimal edit policy should differ across architectures. Nobody has measured
   either.

## 1. Ranking

Criteria, each weighted equally: novelty after the CLM paper, cost on your hardware (RTX 3060 12 GB, with an
occasional rented A100), weeks to a first real result, fit with your attention work (IMHA entropy weighting,
TopoSA/GASM), and paper strength.

| Rank | Idea | Origin | Novelty risk | Compute | First result | Fit | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Self-editing context security**: header forgery, role laundering, rule erosion, signed headers | your #8, sharpened | low | inference only | 1–2 wks | med | **do first** |
| 2 | **Stale KV after edits**: which heads go stale under SCR; entropy-predicted selective recompute | new | medium (CacheBlend / EPIC / PIE are adjacent) | 3060 (0.6–4B) | 2 wks | **high** | **do first** |
| 3 | **Attention-annotated context files**: show the CLM where its own attention went | your #7, moved into CLM | medium (H2O / SnapKV work on KV, not on CLM edits) | 3060 + proxy scorer | 2–3 wks | **high** | strong |
| 4 | **Cost model of edits**: when and where a CLM should edit; architecture-aware policies | new | low | CPU + traces | 1–2 wks | med | strong, short |
| 5 | **How zero-shot is "zero-shot" CLM?** Scaffolding ablation plus context-length awareness | new | low | inference | 1–2 wks | low | fast analysis paper |
| 6 | **Versioned context via git** (fork, revert with lessons) | your #1, reframed | medium-high (Context-Folding) | inference, then RL | 2–3 wks | low | good zero-shot study |
| 7 | **Counterfactual edit credit** for CLM RL | your #6 | medium | RL (A100) | 4+ wks | med | after #3 |
| 8 | **Oracle keep-set task families** (KU, DX, MH, DE, XS, INJ) | your #9, renamed | medium (name clash) | CPU | 2 wks | med | infrastructure for #3 and #7 |
| 9 | Learned handoff briefs (subagent = new context file) | your #5 | medium | inference / RL | 3 wks | low | later |
| 10 | Small editor, frozen reader | your #4 | high (ACON, MemAgent, MEM1, now CLM) | RL (A100) | 4 wks | low | keep as a baseline arm |
| 11 | Context→weights consolidation ("sleep") | your #3 | medium-high (SEAL) | A100 | 3 wks | low | must beat disk + grep, which is already near-perfect |
| 12 | Graph context + structural attention bias | your #2 | medium | reader fine-tuning | 4+ wks | **high** (TopoSA) | long-term; reframe below |

Execution order differs from rank where infrastructure is shared. Section 3 has the order.

## 2. Idea cards

Each card gives the hypothesis, why now, what to build on in this repo, the key experiments, and when to stop.

### Rank 1: Security of self-editing context

**Hypothesis.** Context-as-a-file adds three integrity failures that compaction harnesses don't have, and a
harness-level defence fixes them at near-zero utility cost:

* **(a) header forgery:** untrusted text containing `[[CTX_TURN n role=assistant]]` becomes a forged turn after
  any later edit;
* **(b) role laundering:** invented roles and model-written summaries of untrusted text come back as
  user-authority text;
* **(c) rule erosion:** with an efficiency reward (Eq. 6), constraints in the editable region are deleted
  because they cost tokens. No attacker is needed for this one.

**Why now.** The paper flags this as future work and cites a vendor report of self-generated injections in
compaction summaries. (a) is mechanical and reproducible today (`tests/test_harness.py`).

**Build on.**

* `clm/context_file.py` (`escape_headers`) and `clm/harness.py`.
* Add **signed headers**: the harness writes `[[CTX_TURN 5 role=tool sig=<hmac(content, role)>]]`. On
  parse-back, a turn keeps a privileged role only if its signature verifies. Edited or new turns get
  `role=model_note` provenance and never `user`.
* Add a taint bit that propagates through summaries (your #8 design applies directly).

**Experiments.**

* Attack success rate for forgery, laundering and rule deletion across 3 model families (local Qwen3-8B and two
  API models).
* Conditions: no defence, escaping, signed headers, signed headers + taint.
* Utility on ContextBench with and without attacks.
* Rule-retention curve under the efficiency advantage, using `clm/rl` with a small model.

**Kill.** Forged turns don't change behaviour (attack success < 5%) even for strong attacks. Publish (a) as a
short vulnerability note with the defence, and move the effort to (c).

### Rank 2: Stale KV after context edits

**Hypothesis.** After an edit, relocated KV states were computed under the old context. The error concentrates
in a small, predictable set of layers and heads: those whose old-pass attention from the relocated span to the
edited span was large and sharp, which is an entropy signal. Recomputing only those heads or layers for the
relocated tokens recovers exact-prefill quality at a small fraction of re-prefill cost.

**Why now.** The paper bounds the damage only by relocating K ≤ 6 spans and re-prefilling 16-token tails. It
never asks *where* the approximation error lives. Your entropy-guided head weighting (IMHA) is the natural
predictor.

**Build on.** `clm/scr/reference.py` (`scr_prefill`, `scr_divergence`) already gives exact vs. relocated caches
per layer. Add:

1. per-layer and per-head error maps on real CLM traces (from `clm bench` runs);
2. a predictor from the previous turn's attention to the edited span (hooks as in your #7);
3. a `partial` mode that recomputes chosen layers or heads.

**Experiments.**

* Next-token KL and task accuracy vs. recompute FLOPs Pareto on Qwen3-0.6B, 1.7B and 4B.
* Baselines: full re-prefill, vanilla SCR at K ∈ {1, 6, 64}, CacheBlend-style token selection.
* Whether the predictor transfers across models.

**Kill.** Vanilla SCR's KL is already < 1e-3 on real traces and accuracy is unchanged. Then publish the
measurement as a short note ("SCR is safe, and here is why").

### Rank 3: Attention-annotated context files

**Hypothesis.** A CLM picks edits by reading text, blind to what it actually attends to. Writing a per-turn
salience score into the headers improves keep-set recall and lowers FLOPs at equal accuracy. For example,
`[[CTX_TURN 7 role=tool attn=0.004 H=0.91]]`, computed as entropy-weighted attention mass from recent probe
tokens, using your #7 scorer. A harness that auto-drops low-salience turns is the training-free baseline.

**Build on.**

* Extend `HEADER_RE` to accept extra `key=value` attributes; parse-back ignores them.
* Add `clm/salience.py` with the hook-based scorer (Q and K captured after RoPE, GQA-aware), run on a proxy model.

**Experiments.**

* Needle, KV and Log Triage, plus your KU / DR families (rank 8), with and without annotations.
* Head weighting: entropy-weighted vs. uniform vs. single sharpest head (your IMHA figure).
* Proxy-scorer transfer: 0.6B scorer, 8B CLM.

**Kill.** Annotations don't change which turns get edited (measure the edit diff). Then report the harness-side
auto-drop policy as the result.

### Rank 4: A cost model for context edits

**Hypothesis.** Eq. 9 gives a closed-form break-even horizon for an edit at position e that drops Δ tokens:

    T* ≈ [C_token (P' − e) + ½ C_attn (P'² − e²)] / [C_attn Δ (U + G)]

This predicts that FLOPs-optimal CLMs should edit rarely, edit late in the context, and edit more often on dense
models than on hybrid ones. SCR changes the numerator, which makes edits cheaper and shifts the optimum.

**Build on.** `clm/flops` (exact replay, SCR estimate) and the README sanity table, where the always-rewrite
oracle loses to append-only on FLOPs.

**Experiments.**

1. Validate T* against replayed traces.
2. Decompose the paper-style CLM savings on your runs into attention vs. turn-count vs. generation effects.
3. Steer a CLM with a one-paragraph cost rule (like Fig. 7) and check whether it moves toward the predicted
   optimum.
4. Cost-aware edit scheduling in the harness: defer pending compactions until they pay off.

**Kill.** Measured savings match neither the model nor the decomposition. Then fold the cost model into rank 2
as its motivation section.

### Rank 5: How zero-shot is "zero-shot" CLM?

**Hypothesis.** A large share of CLM's gains comes from scaffolding: the token readout, nudges, rollback with
ledger, and prompt advice ("don't cat", "batch edits"). This fits App. G: models estimate their own context
length poorly.

**Build on.** Every component is already a config flag: `nudge_ratios=[]`, `urgent_mode=none`,
`max_num_retry_on_limit=0`, `final_turns_warning=0`. Add a footer-off flag and prompt variants in
`clm/prompts.py`.

**Experiments.**

* Factorial ablation on ContextBench at 1–4× pressure with 2–3 models.
* Then a small SFT of a 0.6–1.7B model to predict its own token count, and test whether the footer can be
  removed afterwards.

**Kill.** None needed. Every outcome is publishable as an analysis.

### Rank 6: Versioned context via git

Put `LIVE_CTX` in a git repo and tell the model it may `git commit / branch / checkout / revert` the file, and
must write a ≤ 64-token lesson when it abandons a branch.

* **Tasks:** Countdown, your DE rooms task, small debugging tasks.
* **Baselines:** plain CLM, restart-with-notes, Context-Folding style branch-and-return.
* **Free systems result:** checking out a branch whose prefix matches an earlier prompt hits the prefix cache.
  `PrefixCacheSim` matches against *all* earlier prompts, so the replay measures this directly.
* **Kill:** models never revert even when prompted, or git gives no gain over restart-with-notes.

### Rank 7: Counterfactual edit credit

The paper's dual-channel advantage puts efficiency credit uniformly on edit turns. The paper admits that
outcome rewards supervise editing weakly.

* **Method:** use the saved snapshots (`context_snapshots/`) to replay from turn t with the edit undone. On
  ContextBench the stream is fixed, so score with the answer log-likelihood instead of exact match.
* **Payoff:** this gives per-edit advantages; plug them into `clm.rl.segments.assign_advantages` as a third mode.
* **Requirements:** rank 8's oracle keep-sets for credit-accuracy AUROC, and an A100 for RL.
* **Kill:** no gain over a larger rollout group G at equal compute.

### Rank 8: Oracle keep-set families

Add KU (knowledge updates), DX (distractors), MH (multi-hop), DE (dead ends), XS (cross-session) and INJ as
`StreamTask` subclasses in `clm/contextbench/`. The existing classes show the pattern: `generate`, `score_op`,
an oracle in `oracles.py`.

* Report keep-set precision and recall from the final context.
* This is infrastructure for ranks 3 and 7. Publish it as a benchmark only once the official ContextBench is
  out and you can show complementarity.

### Ranks 9–12, briefly

* **Handoff briefs (9).** In CLM terms, a subagent is a new context file and the brief is its initial content.
  The paper found subagents add little on single-repo EdgeBench, so first show headroom on tasks with hidden
  dependencies (your MH-delegate, constraint-carrying code).
* **Small editor, frozen reader (10).** Keep it as an arm in ranks 3 and 7: "a 1.7B editor vs. the reader
  editing itself". It is weak as a standalone paper after CLM.
* **Consolidation (11).** In KV Store, disk plus grep is already exact. Weights can only win on fuzzy or
  semantic recall, so test that first in a week with fixed rules before building anything.
* **Graph context (12).** Reframe it: let the CLM *choose* to keep a graph-structured file (it already invents
  trackers and ledgers, Fig. 3). Then test whether a TopoSA-style structural bias helps a fine-tuned small
  reader read such files. This is your most personal idea but the most expensive one; keep it for after
  papers 1–2.

## 3. Eight-week plan

| Week | Work | Uses |
|---|---|---|
| 1 | Rung 1 of the replication ladder with a local 4–8B model (Base / Summary / CLM, 4 seeds). Start rank-5 ablations on the same runs | `clm bench` |
| 2 | Rank 1: attack suite plus signed headers; rank 4: cost model validated on week-1 traces | `clm/context_file.py`, `clm/flops` |
| 3–4 | Rank 2: stale-KV maps and selective recompute on 0.6–4B | `clm/scr/reference.py` |
| 5–6 | Rank 8 families (KU, DR, MH), then rank 3 annotations | `clm/contextbench`, new `clm/salience.py` |
| 7–8 | Write paper A (rank 1) and paper B (ranks 2 + 4); decide on the A100 for rank 7 | — |

**Paper bundles**

* **A.** Attacking and defending context-as-a-file (rank 1).
* **B.** The systems of self-editing context: edit costs and stale caches (ranks 2 + 4).
* **C.** Attention-guided CLMs (rank 3 + rank-8 families).
* **D.** How zero-shot is zero-shot (rank 5). Workshop or findings track.

**Hygiene for every result**

* At least 3 seeds and a paired bootstrap.
* Report prefix-reuse FLOPs from server `cached_tokens` when available.
* Report results with the clean-room prompt *and* with one prompt variant, since the prompt is an uncontrolled
  variable relative to the paper.
* Before each idea, run the novelty check in your design doc against works citing arXiv 2609.37725.
