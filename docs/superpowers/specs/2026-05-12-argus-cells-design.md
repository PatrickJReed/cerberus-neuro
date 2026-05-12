# argus-cells — Design Spec

**Status:** Approved design, pending implementation plan.
**Date:** 2026-05-12
**Predecessor:** `cerberus-neuro` (multi-task BMS-reproduction framing, now archived).
**Successor of:** `docs/superpowers/specs/2026-05-08-v0-baseline-first-paired-experiment-design.md` (Phase 1 of that spec shipped at val_acc 0.7311; superseded by this reframe).

## 0. Reframe summary

`cerberus-neuro` was framed as a public reproduction of a BMS-internal three-headed multi-task vision model. That framing is retired. The actual scientific question worth pursuing on this dataset (Broad NeuroPainting iPSC Cell Painting, 22q11.2 deletion vs control) is narrower and stronger: **build an honest 6-channel disease classifier, then explain what the model is using to call disease.** This spec is the design for that reframed project under the new name `argus-cells`.

The cerberus / three-heads / virtual-staining / segmentation framing is dead. Multi-task is no longer a goal in itself; it remains in the toolbox if a future phase finds it useful.

## 1. Project identity

**Name.** `argus-cells`. Repo: `argus-cells`. Python package: `argus_cells`. PyPI name: `argus-cells`. Mythological many-eyed-watcher metaphor for a 6-channel input model attending to cellular morphology. Namespace verified available on PyPI and GitHub as of 2026-05-12.

**Mission.** A 6-channel (5 Cell Painting fluorescence + 1 brightfield) disease classifier on Broad NeuroPainting iPSC data (22q11.2 deletion vs control), paired with an interpretability harness that answers, honestly and architecture-robustly, what drives the disease call. The headline deliverable is the interpretability story; the classifier is table stakes.

**Goals, ranked.**
1. **Headline.** An open-ended interpretability story explaining what drives the disease classification, with **cell type** and **Cell Painting channel** as core axes of investigation, and **donor identity** as a confound to rule out.
2. **Table stakes.** A credible classifier — at minimum matching the existing 0.73 baseline from the predecessor project — that the interpretability harness has something real to interpret.

**Audience.** Both ML-methods readers (Anthropic Applied AI, ML-engineering roles) and biology readers (Recursion, Insitro, Iambic, pharma comp-bio). The same artifact serves both via the methods-spine framing (rigorous attribution-method comparison) plus the biology-payoff structure (whatever the data says about disease biology).

**Explicit non-goals.**
- Not a BMS reproduction. The cerberus framing is retired.
- Not multi-task. No virtual staining head, no segmentation head, no Kendall uncertainty weighting.
- Not chasing any pre-committed published claim (including Tegtmeyer 2025). Biology output is whatever the data says.
- Not an architecture tournament for raw accuracy. Two architectures are chosen for *complementary interpretability surfaces,* not for best-accuracy-wins.

## 2. Architecture: paired models

Two classifiers train on the same data, the same head structure, and the same loss. They differ only in backbone and initialization, and they are chosen specifically because their interpretability methods are complementary.

| Model | Backbone | Initialization | Input handling | Native attribution method |
|---|---|---|---|---|
| **`Argus-RN34`** | ResNet34 | ImageNet1K_V1 pretrained | 6-channel `conv1` adapted via tile-and-scale | GradCAM (last conv stage) |
| **`Argus-CCT`** | Compact Convolutional Transformer (CCT-7/3×1 default; size confirmed in Phase 0 based on data scale) | From scratch | 6-channel native conv tokenizer | Attention rollout (transformer encoder) |

Both produce:
- **Disease logits** — binary classification head, trained with `BCEWithLogitsLoss` on the line-condition label (22q11.2 deletion vs control).
- **Final embedding** — pre-classifier feature vector. Used by the donor probe and by UMAP visualizations.
- **Pre-classifier feature map** — used by GradCAM (Argus-RN34) and attention rollout (Argus-CCT).

Both report best-epoch val_acc per the Phase 1 best-epoch convention. Both ship to HF Hub as `patrickjreed/argus-rn34-v0` and `patrickjreed/argus-cct-v0`.

## 3. Data pipeline

**Reuse from predecessor.** The existing `NeuroPaintingDataset` (S3 streaming, CellProfiler-centroid tile selection, well-level train/val split stratified by cell_type × condition, D4 augmentation) is the production data path. It has been validated on Phase 0.5 (cell-type at val_acc 0.99) and Phase 1 (baseline disease at val_acc 0.73).

**Strip from predecessor.** Remove all virtual-staining target assembly, segmentation-mask construction, and 3-head Cerberus label batching. Keep cell_type labels in the batch as a **stratification key,** not as a training target. Keep donor identity in the batch as a **confound key,** not as a training target.

**Crop budget.** Production training scales from the current ~16k crops toward ~50-100k crops. Exact number is set during Phase 0 based on the donor-balance audit. CCT-from-scratch needs more data than pretrained ResNet34 to converge.

**Phase 0 prerequisite audit.** Before any training, audit cpg0038's donor structure: count donor lines per condition, verify minimum N ≥ 3 donors per condition (otherwise the donor probe is uninformative), identify cell-type × donor × condition imbalance. Report findings as `docs/superpowers/results/2026-05-XX-phase-0-donor-audit.md`. Hard gate: if N < 3 donors per condition, escalate to user for scope decision before Phase 1 begins.

## 4. Interpretability harness

The harness is the actual headline deliverable. It is designed to be reusable across both architectures and any future encoder.

**Module layout.**

```
src/argus_cells/
  data.py                       — reused; segmentation/VS components removed
  models/
    resnet34.py                 — Argus-RN34 (6-ch, ImageNet-pretrained)
    cct.py                      — Argus-CCT (6-ch, from scratch)
  training.py                   — single train() entrypoint; dispatches on model.model_kind
  attribution/
    base.py                     — common interface for all attribution methods
    gradcam.py                  — GradCAM on Argus-RN34 last conv stage
    attention_rollout.py        — attention rollout on Argus-CCT transformer encoder
    integrated_gradients.py     — IG; works on either model
    channel_ablation.py         — perturbation: zero one of 6 channels, measure val_acc drop
  probes/
    donor_probe.py              — linear probe on frozen encoder embeddings → donor identity
  analysis/
    stratification.py           — apply any attribution method, group by cell_type, summarize
    agreement.py                — saliency-map agreement between Argus-RN34 and Argus-CCT
    figures.py                  — paper-style figure generators
```

**Common attribution interface.** Each method in `attribution/` exposes a single function with this signature:

```python
def compute_attribution(
    model: nn.Module,
    batch: dict,            # {"images": Tensor[B,6,H,W], "labels": Tensor[B], "cell_type": ...}
    target_class: int = 1,  # disease class
) -> AttributionResult:
    ...
```

where `AttributionResult` is a dataclass:

```python
@dataclass
class AttributionResult:
    saliency: torch.Tensor             # [B, 6, H, W] or [B, H, W] depending on method
    channel_scores: torch.Tensor       # [B, 6] per-channel importance, summarized
    metadata: dict                     # method name, hyperparams, runtime
```

This uniform interface lets `analysis/` modules treat all four methods identically. A method that does not produce per-pixel saliency (e.g., channel ablation) returns `saliency=None` and populates `channel_scores` only.

**Donor probe.** A 1-hidden-layer MLP fit on frozen encoder embeddings to predict donor identity. Reported metrics: top-1 donor classification accuracy, top-1 disease classification accuracy on the same probe, and ratio (donor / disease) accuracy as a confound-strength indicator. Probe is fit on the val split donor subset to avoid leakage from training.

**Cell-type stratification.** Every attribution method is run per cell type. Production output is a per-cell-type × per-channel matrix: how much does each channel matter for disease classification within each cell type? This is the core finding-shaped output of the project.

**Cross-architecture agreement.** For methods that produce per-pixel saliency (GradCAM, attention rollout, IG), compute the Spearman correlation between Argus-RN34 and Argus-CCT saliency maps over the same val-set crops. Reported as a scalar agreement score per method, plus per-cell-type breakdown.

## 5. Phased execution plan

Approach 2 from brainstorming: **tooling-first, training-in-parallel.** Build the interpretability harness against the existing 0.73 baseline first, validate every code path, then bring up the two production models and run them through the already-working harness.

| Phase | Duration | Deliverable | Compute |
|---|---|---|---|
| **0.** Donor + dataset audit | 1 wk | `docs/superpowers/results/2026-05-XX-phase-0-donor-audit.md` with donor counts, cell-type × donor × condition imbalance, decision on crop scale. Hard gate before Phase 1. | Colab Free |
| **1.** Build interpretability harness on existing 0.73 baseline | 2 wk | All four attribution methods + donor probe + cell-type stratification + cross-architecture-agreement scaffolding (against single architecture for now) producing sensible outputs end-to-end on the existing `cerberus-neuro-v0-baseline` checkpoint. Harness is reusable on any 6-channel encoder. | Colab Pro |
| **2.** Train paired classifiers | 1 wk | `argus-rn34-v0` (ResNet34 retrained on scaled crop budget) + `argus-cct-v0` (CCT-from-scratch) on HF Hub. Best-epoch val_acc reported for each. | Colab Pro (RN34) + 1-2 hr cloud A100 (CCT) |
| **3.** Run full analysis | 2 wk | (a) Per-cell-type × per-channel ablation tables; (b) cross-architecture saliency-map agreement scores; (c) donor-probe results; (d) honest open-ended biology writeup. | Colab Pro (inference only) |
| **4.** Polish + ship | 1 wk | `argus-cells` repo live on GitHub, `argus-cells` on PyPI, README rewrites the project story, HF artifacts published with model cards. | Local |

**Total budget:** 7 weeks of evening work, ~$10-15 cloud compute (one A100 session for CCT-from-scratch).

## 6. Success criteria

- **Phase 0.** Donor structure characterized in a one-page report. Hard gate: if N < 3 donors per condition, escalate to user before continuing.
- **Phase 1.** Every attribution method + donor probe + cell-type stratification runs end-to-end on the existing 0.73 baseline without manual intervention. No new accuracy claims at this phase; the goal is harness validation.
- **Phase 2.** At least one of {`argus-rn34-v0`, `argus-cct-v0`} achieves val_acc ≥ 0.73 (matches the existing baseline; bonus if better). If *both* significantly underperform, treat as a training problem rather than an interpretability problem and pause Phase 3 for scope review.
- **Phase 3.** Four concrete numerical deliverables:
  1. Ranked channel-ablation table: 6 channels × 4 cell types × 2 architectures = 48 cells of per-channel importance scores.
  2. Cross-architecture saliency-map agreement: Spearman correlation between Argus-RN34 and Argus-CCT saliency maps, reported per method (GradCAM-vs-attention-rollout makes no sense; IG-vs-IG and ablation-vs-ablation do).
  3. Donor probe results: donor classification accuracy vs disease classification accuracy from the same probe; ratio is the confound-strength scalar.
  4. At least one honest biology claim with confidence framing, expressed as "the model says X" not "we proved X."
- **Phase 4.** README narrates the methods comparison and biology findings honestly, with explicit caveats; HF artifacts loadable in one cell; `pip install argus-cells` works on a clean environment.

## 7. Repo migration + existing-artifact disposition

**Existing `cerberus-neuro` GitHub repo.** Mark archived. Add README banner pointing to `argus-cells`. Do not delete. The cell-type model (val_acc 0.99) and baseline disease model (val_acc 0.73) shipped from this repo are real artifacts that should remain reachable.

**Existing HF Hub repos** (`patrickjreed/cerberus-neuro-cell-type-v0`, `patrickjreed/cerberus-neuro-v0-baseline`). Leave published. Add a banner to each model card README pointing to `argus-cells` as the current line of work. They remain useful as the historical predecessor; do not delete.

**New `argus-cells` GitHub repo.** Fresh repo, fresh `main` branch. Migrate reusable code in clean form:
- Port `src/cerberus_neuro/data.py` → `src/argus_cells/data.py`, stripping segmentation/VS components.
- Port `src/cerberus_neuro/model.py` → `src/argus_cells/models/resnet34.py` (just the `BaselineDiseaseClassifier`, renamed; no Cerberus, no virtual staining, no cell-type-only variant).
- Add `src/argus_cells/models/cct.py` as the new CCT implementation.
- Port `src/cerberus_neuro/training.py` → `src/argus_cells/training.py`, stripping segmentation loss and multi-head dispatch.
- Move all `docs/superpowers/specs/`, `plans/`, `results/` from the cerberus-neuro era into `docs/archive/cerberus-neuro-era/` for transparency; start fresh `docs/superpowers/` tree.

**Migration order.** Do not port until Phase 0 audit clears. Phase 1 (interpretability harness build) still runs against the existing `cerberus-neuro` checkpoint in the existing repo; the new `argus-cells` repo is initialized at the start of Phase 2 when production training begins.

**Package name on HF Hub.** Future artifacts published under `patrickjreed/argus-*` namespace.

## 8. Open risks

- **CCT-from-scratch convergence.** Untested on Cell Painting at this data scale; may need more epochs or crops than the budget allows. Mitigation: train Argus-RN34 in parallel as fallback. If CCT fails, the harness still runs on Argus-RN34 alone, the methods spine becomes "two attribution methods on one architecture" rather than "two architectures comparison," and the project still ships.
- **Donor confound severity.** Phase 0 may surface that the donor structure is too thin for the probe to be informative. Mitigation: explicit user-escalation gate before Phase 1 commits; possible scope change to a different dataset subset or a different confound (cell_type-only or batch-only) if donor structure is insufficient.
- **Cross-architecture attribution disagreement.** Argus-RN34 and Argus-CCT may attend to different features of the same crops. This is interesting to *report,* not a project failure. Disagreement between credible attribution methods is itself a finding ("attribution method choice influences the biological claim, so single-method claims should be treated with caution").
- **Channel ablation hides per-cell-type nuance.** If the model strongly relies on one or two channels globally, the ablation result is "channel X dominates" with weak intra-cell-type structure. Mitigation: report per-cell-type ablation separately. Nuance emerges when one cell type uses different channels than another, which is exactly the kind of finding we want.
- **The existing 0.73 baseline is not Phase 1 production.** Phase 1 builds the harness against the predecessor's `cerberus-neuro-v0-baseline` checkpoint, which used the old training recipe and ~16k crops. The harness must remain reusable when swapped to the Phase 2 production checkpoints; this is enforced by the common attribution interface in §4.

## 9. Out of scope (not in this v0)

- Multi-task (Cerberus) revival in any form.
- Architecture tournament beyond the two specified models. No ViT-B, no ResNet50, no ConvNeXt unless Phase 2 surfaces a specific reason and the user approves.
- Hyperparameter search. One recipe per architecture, validated on the existing Phase 0.5 / Phase 1 patterns.
- Leave-one-donor-out cross-validation. Donor confound is handled by the embedding probe; LODO can be added in v1 if the probe surfaces a real confound that needs full quantification.
- Classical CellProfiler-feature comparison. Considered and dropped during brainstorming.
- Gradio demo / HF Spaces app. v1 stretch.
- Preprint or workshop submission. v1+ stretch contingent on results.

## 10. Glossary

- **Argus-RN34.** ResNet34 backbone, ImageNet1K_V1 pretrained, 6-channel input via tile-and-scale `conv1`, single disease classification head. Trained on Cell Painting + brightfield crops with BCE loss.
- **Argus-CCT.** Compact Convolutional Transformer (Hassani et al., 2021), 6-channel native input, trained from scratch. Single disease classification head. Same loss and training schedule as Argus-RN34.
- **Attribution method.** A function that, given a trained model and an input batch, produces a saliency map and/or per-channel importance scores indicating which inputs the model relied on for its prediction. The four methods in scope: GradCAM, attention rollout, Integrated Gradients, channel ablation.
- **Donor probe.** A linear (or shallow MLP) classifier fit on the frozen encoder embeddings to predict donor identity. High accuracy on this probe indicates the encoder has learned donor-discriminating features, which is a confound for the disease classification claim.
- **Cell-type stratification.** Running every attribution method per cell type, producing a per-cell-type × per-channel matrix of importance scores.
- **Cross-architecture agreement.** The Spearman correlation between Argus-RN34 and Argus-CCT saliency maps on the same val-set crops, computed per method (IG-vs-IG, ablation-vs-ablation). High agreement strengthens any claim derived from those maps.
