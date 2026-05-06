# cerberus-neuro

> Cerberus-inspired multi-task ResNet34 on the Broad NeuroPainting Cell Painting dataset. Three task heads on a shared image encoder: cell-type classification, virtual staining (brightfield → 5 fluorescence channels), and disease-state classification (control vs 22q11.2 deletion). Trained from scratch as a clean public reproduction.

## What this is

A public reproduction of a multi-task vision proof-of-concept originally built at Bristol Myers Squibb in 2025 for the Neurobot high-throughput screening platform. The internal version was completed on the same public Broad NeuroPainting dataset; this rebuild is the version anyone can cite, run, and extend.

The headline scientific question is operational, not academic: how much of the disease-state signal in a six-channel Cell Painting plate is recoverable from brightfield alone? An HTS platform that classified 22q11.2 deletion carriers from brightfield acquisitions would replace a multi-hour fluorescence-staining assay with a minutes-long acquisition at roughly 1/6 the per-plate cost. To answer that, v0 trains two paired models on the same data and reports the gap.

## Architecture

**`CerberusModel` (the headline model).** Single ResNet34 encoder consuming a 1-channel brightfield crop, feeding three heterogeneous task heads:

1. **Cell-type classification** — 4-way softmax (stem / progen / neuron / astro).
2. **Virtual staining** — U-Net-style decoder with skip connections at every encoder stride, producing a 5-channel fluorescence prediction (DNA, mitochondria, AGP, ER, RNA) at the input resolution.
3. **Disease-state classification** — binary (control vs 22q11.2 deletion).

ResNet34 follows the torchvision implementation but with a single-channel input conv. ~24M parameters total, trained end-to-end from scratch with no ImageNet weights.

**`BaselineDiseaseClassifier` (the upper-bound baseline).** Same ResNet34 encoder, takes the full 6-channel stack (BF + 5 fluorescence), exposes only the disease head. Establishes the all-channel disease accuracy you can reach when the model has direct access to mitochondrial and ER intensity rather than having to infer it.

The Cerberus model's value claim is reported relative to this baseline: "X% of the all-channel disease signal recovered from brightfield alone, at 1/6 the assay cost at inference."

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

v0 uses a scoped subset (a handful of plates per cell type at 20×) so a full training run fits a Colab Free 12-hour session.

## Method

**Cell-aware tile selection.** Each 2160×2160 site is tiled into non-overlapping 256×256 patches; tiles are scored by the count of CellProfiler centroids they contain (read from the bucket's `Cells.csv` analysis outputs); the top-N tiles per site are fed to the model. This drops background-only crops without committing to per-cell segmentation, and aligns the model's input distribution with cell-rich regions where the biological signal lives.

**Multi-task loss.** Kendall uncertainty weighting (Kendall, Gal, Cipolla 2018): one trainable log-variance scalar per task, joint loss `Σ_i 0.5·exp(-log_var_i)·L_i + 0.5·log_var_i`. Per-task losses:

- Cell type, line condition: `F.cross_entropy`.
- Virtual staining: `0.85 · L1 + 0.15 · (1 − SSIM)` per channel, summed across the 5 fluorescence outputs. SSIM via `pytorch-msssim`.

This avoids hand-tuning the relative scale between per-pixel regression and per-batch classification losses, which differ by an order of magnitude at init and would otherwise let classification dominate the gradient.

**Optimizer.** AdamW with `lr=3e-4`, `weight_decay=1e-4`, cosine annealing across `n_epochs × steps_per_epoch`. Mixed-precision autocast + GradScaler on CUDA.

**Resumability.** Checkpoints (model + optimizer + scheduler + Kendall + AMP scaler) are written to Drive every N steps and at the end of each epoch; a `latest.pt` is overwritten in-place for one-line resume after a Colab session restart. Per-epoch checkpoints are pushed to Hugging Face Hub (`patrickjreed/cerberus-neuro-v0` and `…-v0-baseline`) for durable artifacts.

## Status

v0 in progress. Implemented and exercised end-to-end on a small subset:

| Step | Module | Status |
|---|---|---|
| Environment | `notebooks/00_environment_smoke.ipynb` | done |
| S3 audit | `notebooks/01_data_exploration.ipynb` (Stages A–F) | done |
| Data pipeline | `src/cerberus_neuro/data.py` | done |
| Architecture | `src/cerberus_neuro/model.py` | done |
| Training loop | `src/cerberus_neuro/training.py` | done |
| Scoped training run | `notebooks/02_train.ipynb` | pending |
| Evaluation | `notebooks/03_eval.ipynb` | pending |
| Polish + writeup | README + figures | pending |

No trained metrics yet; numbers will land here only after the scoped v0 runs converge and produce them. See [`docs/CONTEXT.md`](docs/CONTEXT.md) for project background and [`CLAUDE.md`](CLAUDE.md) for working conventions.

## Repo layout

```
src/cerberus_neuro/
  data.py        manifest builder, IterableDataset, cell-aware tile selection
  model.py       CerberusModel, BaselineDiseaseClassifier
  training.py    multi-task loss, train/eval loop, checkpointing
notebooks/
  00_environment_smoke.ipynb
  01_data_exploration.ipynb     S3 audit + dataset/model/training smoke tests
  02_train.ipynb                (planned: scoped v0 paired runs)
  03_eval.ipynb                 (planned: metrics, figures, gap analysis)
docs/
  CONTEXT.md                    project background + positioning
  SETUP.md                      first-time GitHub / HF / Colab / Docker setup
```

## Running

Two supported workflows:

| Workflow | When to use |
|---|---|
| **Google Colab** | Day-to-day development and the scoped v0 training run. T4 Free is enough for v0; Pro recommended for longer convergence runs. |
| **Docker** (local / Lambda / RunPod / Paperspace) | Full-scale runs at full resolution / all plates, reproducible across cheap GPU rentals. |

First-time setup (HF account, Colab activation, Docker basics, VS Code workflow): see [`docs/SETUP.md`](docs/SETUP.md).

## Related work

- Tegtmeyer et al., *Nat Commun* (2025). Combining NeuroPainting with transcriptomics reveals cell-type-specific morphological and molecular signatures of the 22q11.2 deletion. ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x))
- Lyu, Alonso, Pintó. *Cerberus: A Multi-headed Network for Brain Tumor Segmentation*, BrainLes 2020 / Springer LNCS 12659. The shared-encoder + multi-headed-decoder pattern this project's name and architecture come from.
- Bray et al. *Nat Protoc* (2016). Cell Painting protocol.
- Cell Painting Gallery (AWS Open Data). Index of public Cell Painting datasets.
- Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018. The loss-balancing scheme used here.
- Christiansen et al., *Cell* (2018). *In Silico Labeling*; Ounkomol et al., *Nat Methods* (2018). Foundational virtual-staining-from-brightfield work.

## Author

Patrick J. Reed, Ph.D. ([LinkedIn](https://linkedin.com/in/patrickjenningsreed))

15+ years computational biology, most recently Principal Scientist at Bristol Myers Squibb working on single-cell foundation models, multi-task vision models for HTS, and atlas-scale data engineering. cerberus-neuro is a deliberate public artifact of work originally done internally at BMS, rebuilt on the same public dataset for the broader community.

## License

MIT — see [LICENSE](LICENSE).
