# argus-cells Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement the code tasks task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Compute and infrastructure tasks (training runs, new-repo creation) are human-gated and marked **[GATE]**.

**Goal:** Train the two paired production disease classifiers (Argus-RN34, Argus-CCT), complete the cross-architecture interpretability harness (attention rollout + saliency agreement), and migrate the project to the `argus-cells` repo/package.

**Architecture:** Two single-head 6-channel classifiers chosen for complementary attribution surfaces: Argus-RN34 (ResNet34, ImageNet-pretrained, GradCAM) and Argus-CCT (Compact Convolutional Transformer, from scratch, attention rollout). Both reuse the existing `data.py` pipeline (now with `yield_donor`) and report best-epoch val accuracy. The harness gains attention rollout and a cross-architecture saliency-agreement module so every method runs identically on both.

**Tech Stack:** PyTorch + torchvision, the existing `cerberus_neuro` package (to be renamed `argus_cells`), Colab Pro for RN34, a rented A100 (1-2 hr) for CCT-from-scratch, Hugging Face Hub for checkpoints.

**Source spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md) §2, §4, §5, §7. **Predecessor result:** [`docs/superpowers/results/2026-06-09-phase-1-harness-result.md`](../results/2026-06-09-phase-1-harness-result.md).

---

## Open decisions to confirm before execution

These three decisions gate the detailed code below. Each has a recommendation; confirm or redirect before the corresponding tasks run.

**D1. Argus-CCT implementation source.**
- **Recommendation: minimal from-scratch CCT-7/3×1 in `models/cct.py` (~150 lines, no new dependency).** A Compact Convolutional Transformer is a small, well-documented architecture (conv tokenizer → N transformer encoder layers → sequence pooling → linear head). Hand-rolling it keeps the repo dependency-light, gives us direct access to attention weights for rollout, and matches the CCT screening models Patrick built at BMS.
- Alternatives: vendor the SHI-Labs `Compact-Transformers` reference (heavier, less control over attention capture), or use a `timm` CCT if available (adds a dependency). Choose only if you want a known-good reference over a self-contained one.

**D2. Repo/package migration timing.** Spec §7 says the `argus-cells` repo is initialized at the start of Phase 2.
- **Recommendation: build Phase 2 code in the current `cerberus-neuro` repo/package first, then do the rename + new-repo migration as the final Phase 2 step (Task Group D).** This avoids a context-switch in the middle of implementation and keeps the working test suite intact while CCT + harness land. The HF artifacts still publish under `patrickjreed/argus-*` regardless.
- Alternative: migrate first (fresh repo, rename `cerberus_neuro`→`argus_cells`), then build. Cleaner lineage, but you implement against a brand-new untested tree.

**D3. Compute for the CCT-from-scratch run.** Spec budgets ~1-2 A100 hr.
- **Recommendation: Colab Pro L4 for the Argus-RN34 retrain (pretrained, converges fast), a rented A100 (Lambda/RunPod, ~$2-15) for Argus-CCT.** Confirm you want to spend the A100 hour before Task Group C runs; if not, Argus-CCT can attempt L4 first and escalate only if it fails to converge.

---

## Sequencing and dependencies

```
A. CCT model  ──►  B. attention_rollout (needs a transformer to attribute)
   (code, no compute)        │
                             ▼
C. [GATE] train Argus-RN34 + Argus-CCT  ──►  E. agreement (needs 2 trained models)
   (compute)                                     │
                                                 ▼
                                     F. full analysis (Phase 3 overlap)
D. [GATE] repo migration  ── runs last, after code is proven green ──
```

Group A and B are pure code (TDD, no compute) and are the immediate work. Group C is compute-gated (D3). Group D is infra-gated (D2). Group E depends on C. The donor probe is already wired (`yield_donor`), so the confound analysis just needs the production checkpoints.

---

## File structure

Built in the current package now (per D2 recommendation); paths rename to `argus_cells/` at migration.

- Create: `src/cerberus_neuro/models/__init__.py` — re-exports the model classes.
- Create: `src/cerberus_neuro/models/cct.py` — `ArgusCCT` (conv tokenizer + transformer encoder + seq pool + head), exposing `forward`, `extract_embedding`, `parameter_count`, and captured `attention_maps`.
- Modify: `src/cerberus_neuro/model.py` — leave `BaselineDiseaseClassifier` in place; it is Argus-RN34. (Optionally add an `ArgusRN34 = BaselineDiseaseClassifier` alias at migration.)
- Create: `src/cerberus_neuro/attribution/attention_rollout.py` — `compute_attention_rollout(model, images, target_class=1)` returning an `AttributionResult`.
- Create: `src/cerberus_neuro/analysis/agreement.py` — `saliency_agreement(result_a, result_b)` returning per-sample and mean Spearman correlation between two methods' saliency maps.
- Modify: `src/cerberus_neuro/attribution/__init__.py`, `analysis/__init__.py` — re-export the new functions.
- Test: `tests/test_cct.py`, `tests/test_attention_rollout.py`, `tests/test_agreement.py`.
- Notebook (Group C/F): `notebooks/05_phase_2_train.ipynb`, `notebooks/06_phase_3_analysis.ipynb`.

The harness's common `AttributionResult` interface (`attribution/base.py`) is unchanged; attention rollout and agreement conform to it, so `analysis/stratification.py` and the figure generators already work on them.

---

## Task Group A: Argus-CCT model (code, no compute) — gated on D1

**Files:** Create `src/cerberus_neuro/models/cct.py`, `src/cerberus_neuro/models/__init__.py`, `tests/test_cct.py`.

The model must satisfy the harness contract that IG and channel ablation already rely on, plus expose attention for rollout:
- `forward(x: Tensor[B,6,H,W]) -> Tensor[B,2]` (disease logits)
- `extract_embedding(x) -> Tensor[B,D]` (pre-head pooled features, for the donor/disease probes and UMAP)
- `parameter_count() -> dict[str,int]` (mirror `BaselineDiseaseClassifier.parameter_count`)
- `attention_maps` populated on the last forward: list of `Tensor[B, heads, T, T]` per encoder layer (T = tokens + 1 if a class token is used; CCT uses seq-pool, no class token, so T = tokens)

### Task A1: CCT forward-pass shape contract

- [ ] **Step 1: Write the failing test** (`tests/test_cct.py`)

```python
from __future__ import annotations

import torch

from cerberus_neuro.models.cct import ArgusCCT


def test_forward_emits_two_logits():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    out = model(torch.randn(3, 6, 64, 64))
    assert out.shape == (3, 2)


def test_extract_embedding_shape():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    emb = model.extract_embedding(torch.randn(3, 6, 64, 64))
    assert emb.ndim == 2 and emb.shape[0] == 3
```

- [ ] **Step 2: Run, verify it fails** — `python -m pytest tests/test_cct.py -q` → ImportError / module missing.
- [ ] **Step 3: Implement `ArgusCCT`** — minimal CCT-7/3×1: a `ConvTokenizer` (2 conv+ReLU+maxpool blocks mapping `[B,6,H,W]`→`[B,T,d]` token sequence), 7 `TransformerEncoderLayer`s (multi-head attention with attention-weight capture + MLP, pre-norm), a `SeqPool` (learned attention pooling over tokens → `[B,d]`), and a `Linear(d, n_classes)` head. `extract_embedding` returns the SeqPool output. Default `embed_dim=256, num_layers=7, num_heads=4, mlp_ratio=2`. Use `nn.MultiheadAttention(..., batch_first=True)` with `need_weights=True, average_attn_weights=False` so per-head weights can be captured.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(models): ArgusCCT (Compact Convolutional Transformer, 6-ch, from scratch)`.

### Task A2: attention-map capture + embedding determinism

- [ ] **Step 1: Failing test** — assert that after a forward pass, `model.attention_maps` is a list of length `num_layers`, each `Tensor[B, num_heads, T, T]`, and that two forwards on the same eval-mode input give identical logits (no dropout leak).

```python
def test_attention_maps_captured():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64, num_layers=7, num_heads=4).eval()
    _ = model(torch.randn(2, 6, 64, 64))
    assert len(model.attention_maps) == 7
    a = model.attention_maps[0]
    assert a.shape[0] == 2 and a.shape[1] == 4 and a.shape[2] == a.shape[3]


def test_eval_forward_is_deterministic():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    x = torch.randn(2, 6, 64, 64)
    assert torch.allclose(model(x), model(x))
```

- [ ] **Step 2-4:** Verify fail, implement capture (store per-layer weights in `self.attention_maps`, reset at the start of each forward), verify pass.
- [ ] **Step 5: Commit** — `feat(models): ArgusCCT attention-map capture for rollout`.

### Task A3: re-export

- [ ] **Step 1:** `models/__init__.py` re-exports `ArgusCCT` (and `BaselineDiseaseClassifier` for symmetry). **Step 2:** `python -c "from cerberus_neuro.models import ArgusCCT"`. **Step 3: Commit.**

---

## Task Group B: attention rollout (code, no compute) — depends on A

**Files:** Create `src/cerberus_neuro/attribution/attention_rollout.py`, `tests/test_attention_rollout.py`; modify `attribution/__init__.py`.

`compute_attention_rollout(model, images, target_class=1)` runs a forward (populating `model.attention_maps`), applies the Abnar-Zuidema rollout (per layer: `A_hat = 0.5*A.mean(heads) + 0.5*I`, then matrix-multiply across layers), maps the resulting token-importance vector back to a `[B,1,H,W]` spatial saliency via the tokenizer's spatial grid, and returns an `AttributionResult` with `saliency` and `channel_scores` (rollout is spatial, not per-channel, so `channel_scores` is the saliency summed over space and broadcast across the 6 input channels, documented as such).

### Task B1: rollout math on synthetic attention

- [ ] **Step 1: Failing test** — build a tiny model whose `attention_maps` are set to known matrices; assert rollout produces the analytically-expected token weights (identity attention → uniform; a peaked attention → that token dominates). Assert output is a valid `AttributionResult` with `saliency.shape == (B,1,H,W)`.
- [ ] **Step 2-4:** verify fail, implement, verify pass.
- [ ] **Step 5: Commit** — `feat(attribution): attention rollout for ArgusCCT`.

### Task B2: end-to-end on ArgusCCT + re-export

- [ ] **Step 1: Failing test** — run `compute_attention_rollout` on a real `ArgusCCT` over a `[2,6,64,64]` batch; assert shapes and finite values. **Step 2-4.** **Step 5:** re-export from `attribution/__init__.py`, commit.

---

## Task Group C: [GATE] train the paired classifiers (compute) — gated on D3

**Files:** Create `notebooks/05_phase_2_train.ipynb` (mirror the structure of `02_cell_type_deploy.ipynb` / `02_train.ipynb`; cell-1 install uses the `--no-deps` pattern).

Not bite-sized code — these are orchestration + compute steps run on Colab/A100, reusing the existing `train()` entrypoint in `training.py`.

- [ ] **C1.** Scale the crop budget: `subset_manifest(..., crops_per_site=10)` per the Phase 0 audit recommendation (~63-95k crops). Same well-level split (seed=0).
- [ ] **C2. [GATE]** Train Argus-RN34 (= `BaselineDiseaseClassifier`, pretrained) with the validated recipe (AdamW, discriminative LR 3e-4/3e-5, 5% warmup + cosine, AMP, grad-clip 1.0, ~15 epochs, best-epoch reporting). Push per-epoch checkpoints to `patrickjreed/argus-rn34-v0`.
- [ ] **C3. [GATE]** Train Argus-CCT from scratch (no pretrain) on the same crops. Expect more epochs than RN34; if it stalls at `crops_per_site=10`, escalate to 30 per the Phase 0 audit before changing the recipe. Push to `patrickjreed/argus-cct-v0`.
- [ ] **C4.** Record best-epoch val_acc for each. **Success gate (spec §6):** at least one of {RN34, CCT} ≥ 0.73. If both materially underperform, pause and treat as a training problem (scope review), not an interpretability problem.

---

## Task Group E: cross-architecture saliency agreement (code) — depends on C

**Files:** Create `src/cerberus_neuro/analysis/agreement.py`, `tests/test_agreement.py`; modify `analysis/__init__.py`.

`saliency_agreement(result_a, result_b)` takes two `AttributionResult`s with per-pixel `saliency` over the same crops and returns `{"per_sample": Tensor[B], "mean": float}` of Spearman correlations (flatten each sample's saliency, rank-correlate). Only valid for like-for-like methods (IG-vs-IG, ablation-vs-ablation); document that GradCAM-vs-rollout is not comparable.

### Task E1: Spearman agreement on synthetic saliency

- [ ] **Step 1: Failing test** — identical saliency → corr 1.0; reversed-rank saliency → corr -1.0; shape mismatch → `ValueError`.
- [ ] **Step 2-4:** verify fail, implement (use `scipy.stats.spearmanr` per sample, already a dependency), verify pass.
- [ ] **Step 5:** re-export, commit — `feat(analysis): cross-architecture saliency agreement`.

---

## Task Group F: full analysis (Phase 3 overlap) — depends on C, E

**Files:** Create `notebooks/06_phase_3_analysis.ipynb`.

- [ ] **F1.** Run the full harness (channel ablation, GradCAM on RN34, attention rollout on CCT, IG on both, donor probe via `yield_donor`, cell-type stratification) on both production checkpoints, using the balanced-collection logic.
- [ ] **F2.** Produce the four spec §6 deliverables: (1) 6-channel × 4-cell-type × 2-architecture ablation table, (2) cross-architecture agreement (IG-vs-IG, ablation-vs-ablation), (3) the now-valid donor/disease probe ratio, (4) one honestly-framed biology claim ("the model says X").
- [ ] **F3.** Write `docs/superpowers/results/<date>-phase-3-analysis-result.md`.

---

## Task Group D: [GATE] repo + package migration (infra) — gated on D2, runs last

Per spec §7, after the code above is green. Not bite-sized code.

- [ ] **D1. [GATE]** Create the fresh `argus-cells` GitHub repo (user action).
- [ ] **D2.** Rename the package `cerberus_neuro` → `argus_cells` (directory, `pyproject.toml` name/packages, all imports, notebook install lines). Run the full test suite green under the new name.
- [ ] **D3.** Move `docs/superpowers/{specs,plans,results}` from the cerberus era into `docs/archive/cerberus-neuro-era/`; start a fresh `docs/superpowers/` tree for argus.
- [ ] **D4.** Archive the old `cerberus-neuro` GitHub repo with a README banner pointing to `argus-cells`; add banners to the two predecessor HF model cards. Do not delete anything.
- [ ] **D5.** Publish `argus-cells` to PyPI; confirm `pip install argus-cells` works clean.

---

## Success criteria (from spec §6)

- **Phase 2 (Group C):** at least one of `argus-rn34-v0`, `argus-cct-v0` reaches val_acc ≥ 0.73.
- **Harness completeness (Groups B, E):** attention rollout and saliency agreement run end-to-end via the common `AttributionResult` interface, unit-tested.
- **Analysis (Group F):** the four numerical deliverables produced, with the donor probe now a valid confound measurement (thanks to `yield_donor`).
- **Migration (Group D):** `pip install argus-cells` works; HF artifacts loadable in one cell; predecessor artifacts preserved with banners.

---

## Self-review

- **Spec coverage:** §2 paired models → Groups A (CCT) + existing RN34; §4 harness (attention rollout, agreement, donor probe, stratification) → Groups B, E, F + already-shipped `yield_donor`; §5 phase plan → Groups C, F; §7 migration → Group D. Covered.
- **Dependency consistency:** `ArgusCCT.attention_maps` (A2) is consumed by `compute_attention_rollout` (B); `saliency_agreement` (E) consumes `AttributionResult.saliency` (existing). Names consistent across tasks.
- **Proportionality note (deliberate, not a placeholder):** Group A/B/E carry full TDD test specs and interface contracts; the CCT's internal layer code is specified to the architecture + test-contract level rather than line-by-line because it is gated on decision D1. On confirmation of D1, these expand to fully-inlined bite-sized code (or dispatch to subagent-driven-development). Groups C, D, F are compute/infra and are procedural by nature.

---

## Execution handoff

This plan has an unusual shape: a code core (Groups A, B, E) that can start immediately once **D1** is confirmed, and compute/infra tails (C, D, F) gated on **D2/D3** and your go-ahead on spend.

Recommended next step: confirm D1-D3, then implement Group A (Argus-CCT) via subagent-driven-development, since it is the foundation everything else depends on and needs no compute.
