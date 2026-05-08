# v0 Phase 1 — Baseline-first paired-experiment design

**Status:** approved by user 2026-05-08; ready for implementation planning
**Date:** 2026-05-08
**Author:** Patrick Reed (with Claude as collaborator)
**Supersedes:** the implicit "iterate Cerberus multi-task config until it converges" plan that drove the previous ~15 sessions

## Context

The cerberus-neuro project's v0 deliverable is a paired-experiment comparison: brightfield-only multi-task `CerberusModel` vs all-channel single-task `BaselineDiseaseClassifier`, both trained on the same NeuroPainting subset with the same training infrastructure. The headline result is the gap between the two disease classifiers, which quantifies "how much of the all-channel disease signal is recoverable from brightfield alone."

After ~15 sessions of iteration on the multi-task model's training config (loss function, learning rate, encoder LR ratio, gradient clipping, batch size, GPU utilization), the multi-task setup has not yet converged. The cell-type-only sanity check works at 0.96 val accuracy in 1 epoch with the validated infrastructure; the segmentation-only sanity check shows slow IoU growth; the baseline disease classifier has never been trained.

The conversation that produced this spec recognized two failure modes:

1. **A patch-run-patch cycle** on the multi-task config without a clear v0 success criterion. Each iteration was treating a symptom rather than asking "is multi-task the right v0 deliverable, given what we've learned?".
2. **A gap in the project's empirical foundation:** the headline-result baseline classifier has never been trained, so we don't know whether the disease signal is recoverable at our current data scale at all. Without that number, even a converged multi-task run can't be interpreted.

This spec sets a v0 phase plan that addresses both failure modes by sequencing the work: train the simpler baseline first, gate the multi-task work on that result.

## Goal

Validate the v0 paired-experiment narrative is achievable on the current 16k-crop training subset before committing more time to multi-task tuning. Phase 1 (this spec) trains the baseline disease classifier; Phase 2 (separate spec, after Phase 1 outcome) trains the multi-task Cerberus model and reports the gap.

## v0 phase sequencing (overall plan, for context)

This spec covers Phase 1 only. The full v0 phase plan, decided alongside this spec, is:

- **Phase 0.5 — Cell-type single-task deployment.** Train `CellTypeOnlyModel` to stable val accuracy on the shared `ResNet34Encoder` architecture; push to Hugging Face Hub as `patrickjreed/cerberus-neuro-cell-type-v0`. Already validated at 0.96 val acc in the sanity check; this phase just packages it as a shippable v0 artifact. ~30 minutes wall-clock. Runs in parallel with Phase 1.
- **Phase 1 — Baseline disease classifier (this spec).** Train `BaselineDiseaseClassifier` to interpretable val accuracy as the gate for Phase 2. ~2 hours wall-clock.
- **Phase 2 — 2-head multi-task Cerberus + paired-experiment evaluation (separate spec, after Phase 1 outcome).** Train a Cerberus model with **two heads** (organelle segmentation + disease classification) — cell-type is shipped separately in Phase 0.5 because (a) it's already independently validated and (b) including it in multi-task adds gradient interference for a problem that's already solved. Compare Cerberus disease accuracy to Phase 1's baseline; report the gap. ~3–6 hours wall-clock depending on Phase 2 design.

The "Cerberus" architectural pattern (shared encoder + multi-head decoder) is preserved with 2 heads. The Lyu et al. paper had three binary-segmentation decoders combined into one output; Patrick's adaptation has always used heterogeneous heads. The pattern is the value, not the literal head count.

The v0 narrative becomes: "Three task-specific models share the same `ResNet34Encoder` architecture, each independently validated. Disease and segmentation are jointly optimized in the multi-task `CerberusModel`; cell-type is shipped as a separate single-task model. Cell-type integration into the multi-task model is v1."

## Success criteria

### Project-level (v0 overall)

`baseline_val_disease_acc > cerberus_val_disease_acc > 0.50`, with a meaningful gap (≥ 5 percentage points). This validates the operational claim that brightfield carries a recoverable fraction of the all-channel disease signal.

### Phase 1 (this spec)

The baseline disease classifier converges to a val accuracy interpretable enough to set a useful upper bound for the Cerberus model. Three regimes determine what happens next:

| baseline val_acc | Interpretation | Next action |
|---|---|---|
| ≥ 0.65 | disease signal recoverable from full 6 channels at this data scale; ceiling exists for Cerberus to recover from BF | proceed to Phase 2 (multi-task) at current scope |
| 0.55 – 0.65 | marginal disease signal; multi-task may not produce a meaningful gap | scope-up data first (Phase 1.5 spec), OR proceed cautiously and accept tighter gap |
| ≤ 0.55 | disease signal not recoverable at this scale regardless of architecture | scope-up data is mandatory before continuing v0 |

This decision tree is the entire reason Phase 1 happens before Phase 2.

## Phase 1 architecture

**Model:** `BaselineDiseaseClassifier(in_channels=6, n_classes=2)` from `src/cerberus_neuro/model.py`. Already implemented and never modified. ResNet34 encoder (ImageNet-pretrained, conv1 adapted via tile-and-rescale for 6-channel input) + `ClassifierHead(512, 2)`.

**Data:** identical subset to the current `02_train.ipynb` configuration:
- `wells_per_cell_type=48, sites_per_well=9, crops_per_site=12`
- Well-level 80/20 train/val split, stratified by `(cell_type, line_condition)`
- 16,416 distinct training crops, 4,096 distinct val crops
- Reuses the populated `/content/cerberus-cache/` so no S3 download

**Loss:** `F.cross_entropy(logits, line_condition)`. Single task — no Kendall, no segmentation, no virtual staining.

**Training recipe** (validated infrastructure; no new tuning):
- AdamW, encoder LR `3e-5`, head LR `3e-4` (discriminative, matching the cell-type-only sanity-check config that achieved 0.96 val acc in 1 epoch)
- Linear warmup (5% of total steps) → cosine annealing to 0
- AMP autocast + GradScaler
- Gradient norm clip at 1.0
- batch_size=64, num_workers=8, prefetch_factor=4, pin_memory=True
- 15 epochs (cell-type-only converged in 1 epoch; disease is harder, so give it room)
- D4 dihedral augmentation on the 6-channel input

**Wall-clock budget:** ~2 hours on a Colab Pro L4 GPU. ~200 steps/epoch × 15 epochs = ~3,000 total steps × ~250 ms/step ≈ 13 min compute, plus per-epoch validation passes and HF Hub uploads.

## Implementation surface

The minimum viable code change to run Phase 1 is to make `02_train.ipynb` § 4 (the baseline cell) self-contained, so it can run without first running § 3 (the Cerberus cell). § 3 currently defines `LR` and `WARMUP_STEPS` that § 4 references; § 4 needs to define them itself.

**Files changed:** `notebooks/02_train.ipynb` only. No package code changes. No new files.

**Commits expected:** one commit that makes § 4 self-contained.

**Procedure to run Phase 1 in Colab Pro:**
1. File → Revert in Colab to pull the latest `02_train.ipynb`.
2. Run § 1 setup (install + sys.modules purge + Drive mount + GPU check).
3. Run § 2 manifest construction + parallel S3 prefetch (uses cached data; near-instant).
4. **Skip § 3 entirely** (Cerberus multi-task; not part of Phase 1).
5. Run § 4 (baseline disease classifier training).
6. Run § 5 (final summary; will report only the baseline number).

**Artifacts produced:**
- `/content/drive/MyDrive/cerberus-neuro/cache/v0/baseline/latest.pt` (durable checkpoint)
- `/content/drive/MyDrive/cerberus-neuro/cache/v0/baseline/train_log.jsonl` (per-epoch val record)
- `patrickjreed/cerberus-neuro-v0-baseline` HF Hub repo with per-epoch checkpoint history

## Decision gate

When Phase 1 finishes (~2 hours wall-clock), `train_log.jsonl` has 15 val records with `acc_line_condition` per epoch. The headline number is the best-epoch val accuracy.

The decision gate maps this number to the next action via the table in the Phase 1 success criteria above.

If `acc ≥ 0.65`: launch Phase 2 (separate spec). Phase 2's design will cover the 2-head multi-task config decisions (BCE+Dice loss, full encoder LR, 10–25 epoch budget, etc.). Phase 2 has its own success criterion (Cerberus disease > 0.50, gap to baseline > 5 percentage points).

If `acc 0.55–0.65`: pause and decide between (a) accepting the narrower gap as v0's headline number, or (b) writing a Phase 1.5 spec for data scope-up before continuing. Concrete scope-up paths considered: include 63× neurons batch (would require resolution adaptation), drop `wells_per_cell_type` cap entirely (modest gain since 48 is already near-saturating), pivot to per-cell crops (v1 stretch).

If `acc ≤ 0.55`: project as currently scoped cannot deliver the v0 success criterion. Conversation pivots to "is the cohort scale we have actually sufficient for a 22q11.2 detection task at all" — a real research question rather than a v0 hyperparameter question.

## Subset-size question (addressing the user's stated concern)

The user explicitly asked whether to expand the subset before running Phase 1 to improve odds of a good result. The decision: **do not expand before Phase 1.**

Reasoning:
1. The baseline is simpler than Cerberus. If it converges on 16k crops, scope-up was unnecessary. If it doesn't, knowing that is more informative than scoping up first and confounding the result.
2. Per-(cell_type, condition) stratum we have ~2k crops. For a single-task binary classification with a pretrained encoder, this is in the documented working range for similar Cell Painting CNN literature.
3. Astrocytes specifically (the disease-relevant cell type per Tegtmeyer 2025) carry ~25% of crops = ~1k per condition. If 1k crops per condition is genuinely insufficient, the bottleneck is cohort-scale not subset-scale, and scoping within v0 won't fix it.
4. Run-first-then-scope-up is principled experimental discipline. Scope-up-first introduces a confound between data effect and architecture effect.

If Phase 1 results require scope-up, Phase 1.5 will be a separate brainstormed spec.

## Out of scope (explicit)

Phase 1 deliberately excludes the following. Each is reserved for a later spec when its prerequisites are met:

- 2-head Multi-task Cerberus training (Phase 2; gated on Phase 1 outcome).
- Cell-type single-task deployment (Phase 0.5; brief separate sub-task — package the validated `CellTypeOnlyModel` and push to HF Hub. Not in this spec because it's a deployment task, not a research question; can run in parallel with Phase 1).
- Any modification to `BaselineDiseaseClassifier`, `data.py`, or `training.py`. The baseline run uses validated, in-place infrastructure end-to-end.
- The segmentation head's role in v0. The Phase 1 success criterion is about disease accuracy and doesn't depend on segmentation quality. Segmentation will be revisited as part of the Phase 2 design (for the 2-head Cerberus multi-task model) or, if Phase 2 is gated negatively, dropped from v0 entirely.
- Cell-type head's role in multi-task Cerberus. By design, the 2-head Phase 2 model has segmentation + disease only. Cell-type integration is v1.
- Hyperparameter sweeps. Phase 1 uses one config (the same recipe that produced the cell-type-only 0.96 val acc result) and reports what happens. No sweeps.
- Per-cell crops, 63× resolution, MAE encoder pretraining. All v1 stretch goals already documented in `README.md`.

## Open questions

None blocking implementation. One worth flagging for the eventual writeup:

- **What's a "meaningful gap" between baseline and Cerberus disease accuracy in Phase 2?** The success criterion says "≥ 5 percentage points" but this is somewhat arbitrary. For the v1 writeup, a more rigorous test would be a permutation test or McNemar's test on per-sample agreement. Out of scope for this spec; flag for the eval notebook (notebook 03) design.

## Related work and prior decisions

- The "shared encoder + multi-headed decoder" architecture is from Lyu, Alonso, Pintó, *Cerberus*, BrainLes 2020.
- The pretrained-ImageNet-encoder + conv1 channel adaptation recipe is the standard microscopy-CV transfer learning pattern (Christiansen 2018, Ounkomol 2018).
- The current data subset (`wells_per_cell_type=48, sites=9, crops=12 = 16k crops`) was set during the iteration cycle this spec is correcting against.
- The cell-type-only sanity check at 0.96 val acc (epoch 0) is the empirical validation that the encoder, data pipeline, and discriminative-LR training recipe all work end-to-end.
- All prior multi-task config iterations (loss function, encoder LR, batch size, gradient clipping) are not deleted — the code paths remain available for Phase 2's design.
