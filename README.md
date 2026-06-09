# argus-cells

> A 6-channel Cell Painting disease classifier on the Broad NeuroPainting iPSC dataset (22q11.2 deletion vs control), paired with an interpretability harness that asks, honestly and across architectures, what the model actually uses to call disease. The classifier is table stakes; the interpretability story is the deliverable.

## What this is

`argus-cells` trains a disease-state classifier on Cell Painting morphology and then explains the prediction. The scientific question is not "can a model tell deletion from control" (it can, at val acc 0.73) but "what is it looking at": which fluorescence channels carry the signal, whether that differs by cell type, and whether the model is quietly riding donor identity rather than disease biology.

The name is the many-eyed watcher Argus: a 6-channel input model attending to cellular morphology, then made to show where it looked.

The project began as `cerberus-neuro`, a three-headed multi-task reproduction of a Bristol Myers Squibb proof-of-concept. That framing was retired in favor of the sharper interpretability question. See [Project lineage](#project-lineage) for why the repo and package are still named `cerberus_neuro` for now.

## The question

A disease classifier that reaches 0.73 val accuracy on Cell Painting crops is only interesting if you can say what it learned. `argus-cells` is built around four concrete, numerical interpretability outputs:

1. **Per-channel importance.** Zero each of the 6 channels (brightfield + DNA, mitochondria, AGP, ER, RNA) and measure the accuracy drop. Which channels does the disease call depend on?
2. **Cell-type stratification.** Run every attribution method per cell type (stem / progen / neuron / astro). A per-cell-type by per-channel importance matrix is the core finding-shaped output: does the model read disease from different channels in astrocytes than in neurons?
3. **Donor-confound probe.** A linear probe on frozen embeddings predicts donor identity; a parallel probe predicts disease. The donor / disease accuracy ratio is a confound-strength scalar. Ratio near or above 1 is a red flag that the classifier may be partly reading donor line rather than disease.
4. **Cross-architecture agreement.** Two backbones with complementary attribution surfaces are compared on the same crops. Where they agree, a claim is stronger; where they disagree, single-method attribution claims deserve caution, which is itself a finding worth reporting.

## Architecture: paired classifiers

Two classifiers train on the same data, the same single disease head, and the same loss. They differ only in backbone and initialization, chosen because their native attribution methods are complementary.

| Model | Backbone | Init | Input handling | Native attribution |
|---|---|---|---|---|
| **Argus-RN34** | ResNet34 (torchvision) | ImageNet1K_V1 pretrained | 6-channel `conv1` via tile-and-scale | GradCAM (last conv stage) |
| **Argus-CCT** | Compact Convolutional Transformer (Hassani et al. 2021) | from scratch | 6-channel native conv tokenizer | Attention rollout |

Both emit disease logits (binary, `BCEWithLogitsLoss` on the line-condition label), a pre-classifier embedding (used by the donor probe and UMAP), and a pre-classifier feature map (used by GradCAM / attention rollout). Both report best-epoch val accuracy. Argus-RN34 is the production line that exists today as the 0.73 baseline checkpoint; Argus-CCT arrives in Phase 2.

## Data

[Broad Institute NeuroPainting](https://broadinstitute.github.io/cellpainting-gallery/), Tegtmeyer et al., *Nat Commun* 2025 ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x)). Hosted on the Cell Painting Gallery (`s3://cellpainting-gallery/cpg0038-tegtmeyer-neuropainting/`) under AWS Open Data; public, no credentials required.

Verified against the bucket via `notebooks/01_data_exploration.ipynb` and the Phase 0 donor audit (`notebooks/03_donor_audit.ipynb`):

| | |
|---|---|
| Cohort | 48 cell-line IDs, 24 per condition: 22 human control + 2 isogenic control, 22 human 22q11.2-deletion + 2 isogenic deletion |
| Cell types | iPSC (`stem`), NPC (`progen`), neuron, astrocyte |
| Donor structure | 24 donor lines per condition; disjoint donor ranges (control 1-24, deletion 25-48), so donor identity and disease label are independent given cell type |
| Plate format | 384-well, 8 wells per cell line (most plates) |
| Imaging | Perkin Elmer Phenix at 20x and 63x (v0 uses 20x only) |
| Image format | 2160 x 2160 px, 16-bit, LZW-compressed TIFF |
| Channels | Brightfield + 5 fluorescence: DNA (Hoechst), mitochondria (MitoTracker), AGP (Phalloidin + ConA), ER (WGA), RNA (SYTO 14) |
| Sites per well | 9 |
| Total footprint | 10,532 sites across 1,724 wells across 48 lines; 187k TIFFs / 1.35 TB across 5 batches (942 GB at 20x) |

The current baseline trains on a ~16k-crop scope (48 wells per cell type at 20x). Production training scales toward 50-100k crops; the Phase 0 audit confirmed the crop budget supports this (190-285k crops available at `crops_per_site=30`).

## Interpretability harness

The harness is the reusable core, designed to run on any 6-channel encoder. Every attribution method exposes one function returning a common `AttributionResult` (per-pixel saliency, per-channel importance scores, and method metadata), so the analysis layer treats all methods identically.

Implemented and unit-tested today (`src/cerberus_neuro/`):

| Component | Module | What it does |
|---|---|---|
| Channel ablation | `attribution/channel_ablation.py` | Zero one of 6 channels, measure val-accuracy drop. Batch and per-sample variants. |
| GradCAM | `attribution/gradcam.py` | Hook-based activation/gradient capture on the last conv stage. |
| Integrated Gradients | `attribution/integrated_gradients.py` | Zero-baseline IG, configurable `n_steps`. Works on either backbone. |
| Donor probe | `probes/donor_probe.py` | Logistic-regression probe on frozen embeddings; donor vs disease accuracy ratio. |
| Cell-type stratification | `analysis/stratification.py` | Group any method's per-sample channel scores by cell type. |
| Figures | `analysis/figures.py` | Channel-ablation heatmap, probe-comparison bar chart. |

Arriving with Argus-CCT in Phase 2: attention rollout (`attribution/attention_rollout.py`) and cross-architecture saliency agreement (`analysis/agreement.py`, Spearman correlation between Argus-RN34 and Argus-CCT maps per method).

## Method

**Cell-aware tile selection.** Each 2160x2160 site is tiled into non-overlapping 256x256 patches, scored by the count of CellProfiler centroids each contains (read from the bucket's per-site `Cells.csv` outputs), and the top-N tiles per site are kept. This drops background-only crops without committing to per-cell segmentation, aligning the input distribution with cell-rich regions where the signal lives.

**Augmentation.** Random D4 dihedral transforms (4 rotations, optional horizontal flip) applied identically across all 6 channels per crop. Cell Painting morphology is rotation- and reflection-invariant, so this gives an 8x data multiplier without changing the label. No photometric jitter; the assay is tightly normalized.

**Optimizer and schedule.** AdamW with discriminative learning rates (encoder at `0.1 x LR`, head at `LR`). Linear warmup over 5% of steps to peak LR `3e-4`, then cosine annealing to 0. AMP autocast plus GradScaler on CUDA, gradient-norm clipping at 1.0. Best-epoch reporting is the convention, established when the baseline's epoch-1/2 LR-transition transient (val acc dipped to 0.50, then recovered to 0.73 by epoch 12) showed that 3 noisy data points do not call a trend.

**Resumability and artifacts.** A single `latest.pt` on Drive (overwritten in place) plus per-epoch checkpoints uploaded to Hugging Face Hub for durable history. Resume across Colab session restarts via `train(..., resume_from=...)`.

## Status

Tooling-first build: the interpretability harness is validated against the existing 0.73 baseline before the two production classifiers are trained, so every analysis code path is proven before it interprets a production model.

| Phase | Work | Status |
|---|---|---|
| Phase 0.5 (predecessor) | Cell-type classifier | **shipped.** Best val acc **0.9905**. HF: [`patrickjreed/cerberus-neuro-cell-type-v0`](https://huggingface.co/patrickjreed/cerberus-neuro-cell-type-v0) |
| Phase 1 baseline (predecessor) | 6-channel disease classifier | **shipped.** Best val acc **0.7311** (epochs 12-13). HF: [`patrickjreed/cerberus-neuro-v0-baseline`](https://huggingface.co/patrickjreed/cerberus-neuro-v0-baseline) |
| Phase 0 | Donor-structure audit | **done.** 24 donors/condition, imbalance CV near 0, crop budget sufficient. PROCEED gate cleared. |
| Phase 1 | Interpretability harness (code) | **built and unit-tested** (33 tests passing). Channel ablation, GradCAM, IG, donor probe, cell-type stratification, figures. |
| Phase 1 | Harness run on Colab + results doc | **pending.** `notebooks/04_phase_1_harness.ipynb` is scaffolded but not yet executed against the 0.73 checkpoint. |
| Phase 2 | Train Argus-RN34 + Argus-CCT on scaled crops | pending |
| Phase 3 | Full analysis: per-cell-type x per-channel tables, cross-architecture agreement, donor probe, biology writeup | pending |
| Phase 4 | Polish, PyPI, model cards, README narrative | pending |

No headline metrics are claimed beyond the two shipped numbers above. Every number traces to a logged training run.

## Project lineage

This repo started as `cerberus-neuro`: a three-headed multi-task ResNet34 (cell-type + organelle soft-segmentation + disease) framed as a public reproduction of a BMS internal proof-of-concept. Two task-specific models shipped under that framing (the cell-type and baseline-disease checkpoints above), validating the data pipeline and training infrastructure end-to-end on real biology.

The multi-task framing was then retired. The disease classifier plus an honest interpretability harness is a stronger, more focused artifact than three heads competing for encoder capacity. Per the [design spec](docs/superpowers/specs/2026-05-12-argus-cells-design.md), the GitHub repo, the Python package, and the HF namespace migrate to `argus-cells` / `argus_cells` / `patrickjreed/argus-*` at the start of Phase 2 (production training). Until then, the Phase 1 harness runs in this repo against the existing `cerberus-neuro-v0-baseline` checkpoint, and the package name stays `cerberus_neuro`. The two shipped checkpoints keep their `cerberus-neuro-*` names as the historical predecessor.

## Repo layout

```
src/cerberus_neuro/
  data.py                       manifest builder, IterableDataset, cell-aware tile selection
  model.py                      BaselineDiseaseClassifier (6-ch), ResNet34Encoder, extract_embedding()
  training.py                   train/eval loop, checkpointing, HF Hub push
  audit.py                      donor-structure audit helpers (Phase 0)
  attribution/
    base.py                     AttributionResult dataclass + channel-score helper
    channel_ablation.py         per-channel accuracy-drop attribution
    gradcam.py                  GradCAM on the last conv stage
    integrated_gradients.py     zero-baseline IG
  probes/
    donor_probe.py              linear donor / disease probe on frozen embeddings
  analysis/
    stratification.py           per-cell-type channel-score grouping
    figures.py                  paper-style figure generators
notebooks/
  00_environment_smoke.ipynb    runtime check (GPU, HF login, Drive)
  01_data_exploration.ipynb     S3 audit + per-channel statistics
  02_cell_type_deploy.ipynb     cell-type model training + HF deploy (shipped)
  02_train.ipynb                baseline disease classifier training (shipped)
  03_donor_audit.ipynb          Phase 0 donor-structure audit
  04_phase_1_harness.ipynb      interpretability-harness validation (pending run)
docs/
  CONTEXT.md                    project background + positioning
  SETUP.md                      first-time GitHub / HF / Colab / Docker setup
  superpowers/                  specs, plans, and results docs per phase
```

## Running

| Workflow | When to use |
|---|---|
| **Google Colab Pro (L4 / A100)** | Day-to-day development and training. T4 Free covers the smoke notebook; Pro is recommended for training and the harness run. |
| **Docker** (local / Lambda / RunPod / Paperspace) | Full-resolution / scaled-crop runs, reproducible across cheap GPU rentals. |

First-time setup (HF account, Colab activation, Docker basics, VS Code workflow): see [`docs/SETUP.md`](docs/SETUP.md).

## Roadmap

- **Phase 2.** Train Argus-RN34 (ResNet34, retrained on 50-100k crops) and Argus-CCT (CCT from scratch, ~1-2 A100 hours) to HF as `patrickjreed/argus-rn34-v0` and `patrickjreed/argus-cct-v0`. Initialize the `argus-cells` repo and package.
- **Phase 3.** Run the harness on both production models: a 6-channel by 4-cell-type by 2-architecture channel-importance table, cross-architecture saliency agreement, donor-probe confound result, and an honest biology writeup framed as "the model says X" rather than "we proved X".
- **Phase 4.** `pip install argus-cells`, model cards on HF, README rewritten around the methods comparison and findings.

Out of scope for v0: multi-task revival, architecture tournament beyond the two specified models, hyperparameter sweeps, leave-one-donor-out cross-validation, Gradio demo.

## Related work

- Tegtmeyer et al., *Nat Commun* (2025). NeuroPainting plus transcriptomics reveals cell-type-specific morphological and molecular signatures of the 22q11.2 deletion. ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x))
- Hassani et al. *Escaping the Big Data Paradigm with Compact Transformers* (2021). The Compact Convolutional Transformer used for Argus-CCT.
- Selvaraju et al. *Grad-CAM*, ICCV 2017. Gradient-weighted class activation mapping (Argus-RN34 attribution).
- Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks*, ICML 2017. Integrated Gradients.
- Abnar, Zuidema. *Quantifying Attention Flow in Transformers*, ACL 2020. Attention rollout (Argus-CCT attribution).
- Bray et al. *Nat Protoc* (2016). Cell Painting protocol.
- Cell Painting Gallery (AWS Open Data). Index of public Cell Painting datasets.

## Author

Patrick J. Reed, Ph.D. ([LinkedIn](https://linkedin.com/in/patrickjenningsreed))

15+ years computational biology, most recently Principal Scientist at Bristol Myers Squibb working on single-cell foundation models, multi-task vision for high-throughput screening, and atlas-scale data engineering. `argus-cells` extends that imaging-AI work as a public, citable artifact, including the Compact Convolutional Transformer backbone that echoes the CCT screening models built in that prior work.

## License

MIT, see [LICENSE](LICENSE).
