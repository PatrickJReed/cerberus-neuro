# cerberus-neuro

> Cerberus-inspired ResNet34 architecture applied to the Broad NeuroPainting Cell Painting dataset. Three task heads on a shared encoder: cell-type classification, organelle soft-segmentation, and disease-state classification (control vs 22q11.2 deletion). Encoder initialized from ImageNet and fine-tuned end-to-end. v0 validates each task independently; v1 unifies them in a single multi-task training pass.

## What this is

A public reproduction of a multi-task vision proof-of-concept originally built at Bristol Myers Squibb in 2025 for the Neurobot high-throughput screening platform. The internal version was completed on the same public Broad NeuroPainting dataset; this rebuild is the version anyone can cite, run, and extend.

The headline scientific question is operational: how much of the disease-state signal in a six-channel Cell Painting plate is recoverable from brightfield alone? An HTS platform that classified 22q11.2 deletion carriers from brightfield acquisitions would replace a multi-hour fluorescence-staining assay with a minutes-long acquisition at roughly 1/6 the per-plate cost. v0 trains task-specific models that share the same encoder architecture and reports each independently; v1 integrates them in a multi-task model and reports the gap against an all-channel single-task baseline.

## Architecture

**Shared encoder** (across all task variants): ResNet34 from torchvision, initialized from ImageNet1K_V1 weights. The first conv layer is rebuilt for the model's input-channel count: mean across the 3 pretrained channels for the brightfield (1-channel) variant; tiled and rescaled for the 6-channel baseline variant. The whole model is fine-tuned end-to-end with discriminative learning rates (encoder at 0.1× the head LR) and gradient norm clipping. ~24M parameters total.

**Three task heads** (each independently trainable on the shared encoder, or jointly in v1):

1. **Cell-type classification** — 4-way softmax (stem / progen / neuron / astro). Classifier head: adaptive average pool + linear projection over the deepest feature map. Trained with cross-entropy.
2. **Organelle soft-segmentation** — U-Net-style decoder with skip connections at every encoder stride, producing 5-channel mask predictions (DNA, mitochondria, AGP, ER, RNA) at the input resolution. Each fluorescence channel is treated as a soft probability mask: high fluorescence → high probability that the pixel belongs to that organelle compartment. Trained with `0.5 * BCE_with_logits + 0.5 * (1 - soft_Dice)`. The combined loss avoids both the L1-style constant-prediction-at-data-mean attractor and the BCE-style constant-prediction-at-channel-mean attractor that pure pixel-wise losses fall into on sparse-foreground segmentation.
3. **Disease-state classification** — binary (control vs 22q11.2 deletion). Same classifier-head architecture as cell-type, with a 2-class output.

**`BaselineDiseaseClassifier` (companion model for v0 paired-experiment evaluation).** Same `ResNet34Encoder` taking the full 6-channel stack (BF + 5 fluorescence) as input, with only the disease head. Establishes the all-channel disease-accuracy upper bound the brightfield-only Cerberus model is compared against.

## Data

[Broad Institute NeuroPainting](https://broadinstitute.github.io/cellpainting-gallery/), Tegtmeyer et al., *Nat Commun* 2025 ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x)). Hosted on the Cell Painting Gallery (`s3://cellpainting-gallery/cpg0038-tegtmeyer-neuropainting/`) under AWS Open Data; public, no credentials required.

Verified against the bucket via the audit notebook (`notebooks/01_data_exploration.ipynb`):

| | |
|---|---|
| Cohort | 48 distinct cell-line IDs: 22 human control + 22 human 22q11.2-deletion + 2 isogenic-control + 2 isogenic-deletion |
| Cell types | iPSC (`stem`), NPC (`progen`), neuron, astrocyte |
| Plate format | 384-well, 8 wells per cell line (most plates) |
| Imaging | Perkin Elmer Phenix at 20× and 63× (v0 uses 20× only) |
| Image format | 2160 × 2160 px, 16-bit, LZW-compressed TIFF |
| Channels | Brightfield + 5 fluorescence: DNA (Hoechst), mitochondria (MitoTracker), AGP (Phalloidin + ConA combined), ER (WGA), RNA (SYTO 14) |
| Sites per well | 9 |
| Total volume | 187k TIFFs / 1.35 TB across 5 batches; ~942 GB at 20× alone |

v0 uses a scoped subset of the data (~16k distinct training crops from 48 wells per cell type at 20×).

## Method

**Cell-aware tile selection.** Each 2160×2160 site is tiled into non-overlapping 256×256 patches; tiles are scored by the count of CellProfiler centroids they contain (read from the bucket's per-site `Cells.csv` analysis outputs); the top-N tiles per site are fed to the model. Drops background-only crops without committing to per-cell segmentation, and aligns the model's input distribution with cell-rich regions where the biological signal lives.

**Augmentation.** Random D4 dihedral transforms (4 rotations × optional horizontal flip) applied identically across all 6 channels per crop. Cell Painting biology is rotation- and reflection-invariant, giving an 8× data multiplier without changing label or pixel-level alignment. No photometric jitter in v0; the assay is tightly normalized.

**Loss functions.**

- Cell type, line condition: `F.cross_entropy`.
- Organelle segmentation: `0.5 * binary_cross_entropy_with_logits + 0.5 * (1 - soft_Dice)`. The U-Net head returns raw logits (not sigmoid-activated) to use the AMP-safe fused BCE-with-logits operation. The combined BCE+Dice loss is the standard medical-imaging segmentation recipe and breaks the constant-prediction-at-channel-mean attractor that pure BCE on sparse-foreground data falls into.

**Optimizer and schedule.** AdamW with discriminative learning rates: encoder at `0.1 × LR`, heads at `LR`. Linear warmup (5% of total steps) ramps to peak LR `3e-4`, then cosine annealing to 0. AMP autocast + GradScaler on CUDA. Gradient norm clipping at 1.0 to prevent the gradient explosions seen on hard batches when encoder + heads disagree.

**Multi-task balance (v1, when integrating heads).** Kendall uncertainty weighting (Kendall, Gal, Cipolla 2018): one trainable log-variance scalar per task, joint loss `Σ_i 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i`.

**Resumability and HF Hub artifacts.** Single `latest.pt` checkpoint on Drive (overwritten in place) plus per-epoch checkpoints uploaded to Hugging Face Hub for durable history. Resume across Colab session restarts via `train(..., resume_from=...)`.

## Status

**v0 in progress.** Each task is independently validated or under active development on the shared `ResNet34Encoder` architecture.

| Task | Notebook | Status | Reported metric |
|---|---|---|---|
| Environment | `notebooks/00_environment_smoke.ipynb` | done | — |
| S3 audit | `notebooks/01_data_exploration.ipynb` | done | All audit gaps closed |
| Data pipeline | `src/cerberus_neuro/data.py` | done | exercised in all three sanity checks |
| Cell-type classification | `notebooks/02_cell_type_deploy.ipynb` | **shipped** | **Best val acc 0.9905 at epoch 4. HF Hub: [`patrickjreed/cerberus-neuro-cell-type-v0`](https://huggingface.co/patrickjreed/cerberus-neuro-cell-type-v0).** |
| All-channel disease baseline | `notebooks/02_train.ipynb` § 4 | **shipped** | **Best val acc 0.7311 at epochs 12-13. HF Hub: [`patrickjreed/cerberus-neuro-v0-baseline`](https://huggingface.co/patrickjreed/cerberus-neuro-v0-baseline).** Disease signal is robust at the v0 16k-crop scale. |
| Organelle segmentation | `notebooks/02_sanity_check.ipynb` § 4 | active diagnosis | BCE+Dice loss landed; encoder-LR experiments in flight. Will be integrated into the 2-head Cerberus multi-task model in Phase 2. |
| Disease classification (Cerberus brightfield-only) | `notebooks/02_train.ipynb` § 3 | Phase 2 (pending) | Trains in the multi-task Cerberus model with the segmentation head. Goal: val acc in [0.55, 0.65] = 10-20pp gap to baseline. |
| 2-head Cerberus multi-task | `notebooks/02_train.ipynb` § 3 | Phase 2 (pending design) | Phase 1 (baseline) shipped at 0.73 val acc, unblocking the paired-experiment narrative. Phase 2 brainstorm + spec is the next work. |
| Evaluation + writeup | `notebooks/03_eval.ipynb` | pending | — |

**No headline metrics are claimed beyond the ones in this table.** Validated numbers will land here as each task converges.

## Repo layout

```
src/cerberus_neuro/
  data.py        manifest builder, IterableDataset, cell-aware tile selection
  model.py       CerberusModel, BaselineDiseaseClassifier,
                 CellTypeOnlyModel, VirtualStainingOnlyModel
  training.py    multi-task loss, BCE+Dice segmentation loss, train/eval loop,
                 checkpointing, HF Hub push
notebooks/
  00_environment_smoke.ipynb
  01_data_exploration.ipynb     S3 audit + dataset/model/training smoke tests
  02_sanity_check.ipynb         Single-task diagnostic experiments per head
  02_train.ipynb                v0 paired training run (Cerberus + baseline)
  03_eval.ipynb                 (planned: per-task metrics, gap analysis, figures)
docs/
  CONTEXT.md                    project background + positioning
  SETUP.md                      first-time GitHub / HF / Colab / Docker setup
```

## Running

Two supported workflows:

| Workflow | When to use |
|---|---|
| **Google Colab Pro (L4 / A100)** | Day-to-day development and v0 training runs. T4 Free works for the smoke notebook only; Pro recommended for the full training runs. |
| **Docker** (local / Lambda / RunPod / Paperspace) | Full-resolution / all-plate runs at v1, reproducible across cheap GPU rentals. |

First-time setup (HF account, Colab activation, Docker basics, VS Code workflow): see [`docs/SETUP.md`](docs/SETUP.md).

## v1 stretch goals

Documented as natural extensions once v0 ships:

- **Multi-task model** with proper Kendall uncertainty weighting on a shared encoder. v0 demonstrates each head works independently; v1 integrates.
- **Cell-centered crops** via the CellProfiler outline masks under `publication_data/.../outlines/`. Solves the per-tile cell-density imbalance (astrocytes are 5× sparser per tile than progenitors at 20× — the disease-relevant cell type is the most undersampled).
- **MAE-pretrained encoder** on the full Cell Painting Gallery brightfield (~10M frames). Replaces ImageNet-pretrain initialization. Standard recipe, expected substantial gain on virtual-staining-style heads.
- **63× resolution variant** for organelle-resolution work (mitochondrial structure changes, the published 22q11.2 signal). Excluded from v0's 20× scope to fit Colab compute budgets.

## Related work

- Tegtmeyer et al., *Nat Commun* (2025). Combining NeuroPainting with transcriptomics reveals cell-type-specific morphological and molecular signatures of the 22q11.2 deletion. ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x))
- Lyu, Alonso, Pintó. *Cerberus: A Multi-headed Network for Brain Tumor Segmentation*, BrainLes 2020 / Springer LNCS 12659. The shared-encoder + multi-headed-decoder pattern this project's name and architecture come from.
- Bray et al. *Nat Protoc* (2016). Cell Painting protocol.
- Cell Painting Gallery (AWS Open Data). Index of public Cell Painting datasets.
- Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018. The loss-balancing scheme used in v1.
- Christiansen et al., *Cell* (2018). *In Silico Labeling*; Ounkomol et al., *Nat Methods* (2018). Foundational virtual-staining-from-brightfield work.
- Sudre et al. *Generalised Dice overlap as a deep learning loss function*, DLMIA 2017; Isensee et al. *nnU-Net*, *Nat Methods* 2021. The combined BCE+Dice loss recipe used for the segmentation head.

## Author

Patrick J. Reed, Ph.D. ([LinkedIn](https://linkedin.com/in/patrickjenningsreed))

15+ years computational biology, most recently Principal Scientist at Bristol Myers Squibb working on single-cell foundation models, multi-task vision models for HTS, and atlas-scale data engineering. cerberus-neuro is a deliberate public artifact of work originally done internally at BMS, rebuilt on the same public dataset for the broader community.

## License

MIT — see [LICENSE](LICENSE).
