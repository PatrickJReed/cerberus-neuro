# CLAUDE.md — context for Claude Code sessions in this repo

This file seeds the next Claude Code session with project context that isn't obvious from the code or README alone. Read this before doing meaningful work.

For the **strategic background** behind the project (why it exists, public-version-of-BMS-internal narrative, role-class positioning, connection to other portfolio projects, short- and long-term goals, honestly-acknowledged gaps), read `docs/CONTEXT.md`. This file (`CLAUDE.md`) is for *how to work in the codebase*; `docs/CONTEXT.md` is for *why the codebase exists*.

## What this repo is

`cerberus-neuro` is a portfolio research artifact authored by Patrick J. Reed, Ph.D. (computational biologist, 15+ years; recent Principal Scientist at Bristol Myers Squibb). The project is a **public reproduction** of a multi-task vision proof-of-concept originally built at BMS in 2025 on the same public Broad NeuroPainting dataset. Internal version was not transferred or productionized before April 2026; the public rebuild here exists so the work is citable and extensible.

Three-headed (Cerberus-inspired) ResNet34 with a shared image encoder and three task heads: cell-type classification, virtual staining of 5 Cell Painting channels from brightfield, and disease-state classification on 22q11.2 deletion vs control iPSC lines. v0 trains this alongside an all-channel single-task disease-classifier baseline (same ResNet34 encoder, 6-channel input, single line-condition head) so the Cerberus disease number is reported against a meaningful upper bound: "what fraction of the all-channel disease signal is recoverable from brightfield alone, at 1/6 the assay cost?".

The artifact is positioned for senior IC roles at:
- **Recursion, Insitro, Iambic Therapeutics** — image-based CRISPR perturbation / Cell Painting / multi-task vision specifically
- **Anthropic Applied AI Engineer Life Sciences** — builder credibility on a non-trivial bio analysis (paired with cellduet)
- **Bio-AI imaging companies generally** (Atomwise, Genentech gRED, Generate:Biomedicines)
- **Pharma comp-bio roles** that emphasize Cell Painting / phenotypic screening (BMS, Lilly, Pfizer)

See `README.md` for the public-facing technical framing.

## Execution model

Two-runtime model: **Colab for prototyping + small training, Docker for full-scale training and portability.**

```
VS Code (local)  →  git push  →  GitHub
                                  │
                           ┌──────┴──────┐
                           ▼             ▼
                     Colab notebooks   Docker container
                     (T4 16 GB,        (local / Lambda /
                      ~12 hr Free      RunPod / Paperspace,
                      ~24 hr Pro)      A100 hours when needed)
                           │             │
                           └──────┬──────┘
                                  ▼
                          Hugging Face Hub
                          (datasets, model checkpoints)
```

- **Code lives locally** and is edited in VS Code.
- **Day-to-day work runs on Google Colab.** Patrick has Colab Pro available; default to Free first to enforce scope discipline, escalate to Pro when actually needed.
- **Full-scale / longer training runs use the `Dockerfile`** in this repo. Built locally for testing; deployed to Lambda Cloud / RunPod / Paperspace Pro when an A100 hour or two is genuinely required (~$2-15 per converged run).
- **Each Colab notebook reinstalls `cerberus_neuro` from GitHub** at the top: `!pip install -q git+https://github.com/PatrickJReed/cerberus-neuro.git@main`.
- **Artifacts persist on Hugging Face Hub** (account: `patrickjreed`). Model checkpoints → HF model repo. Aggregated derived datasets → HF dataset repo.

Onboarding details: `docs/SETUP.md`.

## Hard scope discipline

These constraints exist because the project must ship in 6–8 weeks of evening work without burning excessive compute money. Violate only after explicit user approval.

### Compute discipline

- **Colab Free first.** v0 must demonstrate the model works on a scoped subset (lower image resolution, subset of plates, fewer epochs) within a single Colab Free 12-hour session. This forces good engineering: efficient data loading, batch-size discipline, frequent checkpointing, resumable training.
- **Colab Pro is available, not required.** Escalate when scope demands it (full-resolution training, longer convergence runs). Don't default to Pro for prototype iteration.
- **Lambda / RunPod / Paperspace are for full-scale runs only.** Spin up an A100 only when v0 has shipped and a full-scale training run is genuinely needed. Total full-train cost should be $5–15, not recurring.
- **No multi-GPU training in v0.** Single-GPU runs only. Multi-GPU adds engineering complexity that doesn't earn its place until v1.

### Modeling discipline

- **ResNet34 with three task heads, period.** Don't swap to ViT, EfficientNet, or other architectures in v0. The point is to faithfully reproduce the BMS POC architecture; architecture variants are v1 stretch.
- **Don't try to beat the BMS internal numbers.** Internal version had access to internal dataset preparation, BMS-tuned hyperparameters, and BMS infrastructure. Reproduction targets reasonable performance on the public data, not performance parity with the internal version.
- **No hyperparameter sweeps in v0.** One config, train, evaluate, ship. Sweeps are v1.
- **ImageNet-pretrained encoder by default** (`ResNet34_Weights.IMAGENET1K_V1`), `conv1` adapted for the input-channel count of each model variant (mean across channels for 1-ch BF; tile-and-scale for 6-ch baseline). Standard microscopy-CV transfer-learning recipe; the dossier flags the BMS internal version's pretraining choice as unknown, so adopting the standard practice is honest. `pretrained_encoder=False` available for a v1 from-scratch ablation if useful for the writeup.

### Repo discipline

- **Lean module architecture.** Do not pre-create empty `data/`, `models/`, `training/`, `eval/` Python packages until there is real code to put in them. Module structure should emerge from working code.
- **Notebooks are the execution surface, not the library.** `src/cerberus_neuro/` is reusable Python utilities; notebooks orchestrate analyses and produce figures. Keep notebooks thin (call into `src/`); keep `src/` thin (utilities only, not workflow logic).
- **Commit notebooks with cleared outputs.**
- **Persistent caches** go to `/content/drive/MyDrive/cerberus-neuro/cache/` (Colab) or HF datasets, never to repo paths. Cache directory should be configurable.

## Voice and writing style

Patrick has explicit voice preferences captured at `/Users/patrickreed/NewRoleEfforts/PatrickReed_Writing_Style.md`. Read that file before writing any user-facing prose. Highlights:

- **No em-dashes** as connectors. Use commas, parentheses, or sentence breaks.
- **No banned cliches**: "leverage", "robust", "seamless", "cutting-edge", "passionate about", "world-class", "state-of-the-art", "perfect fit", "I am writing to express", "Furthermore" / "Moreover" sentence openers.
- **Once-per-document rule** for signature phrases like "convergent" / "convergent evidence". Rotate to "concordant", "orthogonal lines of evidence", "multi-stream validation" if needed.
- **Demonstrate, don't state.** Specific numbers + named tools + concrete outcomes beat adjectives.
- **Honest framing.** Never invent metrics, never claim experience that doesn't exist. If a result is preliminary, say so. If a method has known caveats, name them. The README's "What this is" section is explicit that this is a public reproduction; don't blur the line.

## Coding conventions

- Python ≥ 3.10. Type hints encouraged but not enforced.
- Ruff for lint + format (`ruff check . && ruff format .`). Config in `pyproject.toml`.
- Build via `hatchling`. Editable install: `pip install -e ".[dev,training]"`.
- PyTorch is required for training (in `[training]` extra). Default `pip install -e .` is the lighter package without torch for those who only want to run analysis utilities.
- Notebooks live in `notebooks/`, named with a `NN_short_description.ipynb` convention. Each notebook starts with: (a) markdown header with title + Open-In-Colab badge, (b) `!pip install -q git+...cerberus-neuro.git@main` cell, (c) HF login cell (Colab Secret `HF_TOKEN`), (d) optional Drive mount cell, (e) work cells. Pattern is established in `notebooks/00_environment_smoke.ipynb` — copy that as the template.
- Dockerfile is at the repo root. Tag local builds as `cerberus-neuro:latest`. See `docs/SETUP.md` for cloud-platform specifics.

## What NOT to do

- Do **not** pre-architect a deep module hierarchy without code. Empty packages are technical debt.
- Do **not** add CI workflows, pre-commit hooks, or `tests/` scaffolding until there is something to test or run.
- Do **not** invent or estimate metrics. Every reported number must trace to an actual training run with a logged config.
- Do **not** scale beyond v0 scope without explicit user approval. Multi-GPU, hyperparameter sweeps, architecture variants are out of scope unless promoted.
- Do **not** rely on Colab Pro / Lambda compute being available. v0 must work on Colab Free.
- Do **not** claim performance parity with the BMS internal version. Reproduction targets reasonable public-data performance.
- Do **not** create new top-level documentation files (`*.md`) without explicit ask. Edit `README.md` for public-facing changes; edit this file (`CLAUDE.md`) for Claude-context changes.

## Cross-references

- **Sibling portfolio project**: `~/Sandbox/cellduet/` — multimodal perturbation concordance (transcriptomic + morphological), with overlapping methodology stack
- **Parent strategy doc**: `~/NewRoleEfforts/docs/portfolio_project_brainstorm.md` — full portfolio rationale and project rankings
- **Voice rules**: `~/NewRoleEfforts/PatrickReed_Writing_Style.md`
- **Master resume**: `~/NewRoleEfforts/PatrickReed_Master_Resume.md` (gold-standard claims source)
- **Accomplishments bank**: `~/NewRoleEfforts/PatrickReed_Accomplishments_Bank.md` — search `Neurobot Multi-Task Foundation Model` for the BMS-internal version's accomplishment entry, including the "RxRx3 reference, NOT training data" note
- **Claims audit**: `~/NewRoleEfforts/PatrickReed_Claims_Audit.md` — flagged claims; consult before writing about Patrick's prior work
- **NeuroPainting dossier**: `~/NewRoleEfforts/docs/NeuroPainting_MultiTask_Model_Dossier.md` — comprehensive reference on the BMS POC including dataset details, Cerberus architecture inspiration, and connections to upstream Tegtmeyer et al. publication

## v0 task list

Rough order; revisit as work proceeds. All compute steps run on Colab (or local Docker for testing); results push to HF for persistence.

0. **Run `notebooks/00_environment_smoke.ipynb` on Colab** to confirm the runtime is good (GPU, HF login, Drive mount).
1. **Data exploration** (`notebooks/01_data_exploration.ipynb`): pull a small NeuroPainting subset from the Cell Painting Gallery, inspect images per cell type / per condition, document the data shape and per-channel statistics. Key v0 unblocker: confirm the data pipeline works end-to-end on Colab Free.
2. **Data pipeline** (`src/cerberus_neuro/data.py`): efficient image loading + per-channel normalization + per-task label assembly. Use `torch.utils.data.IterableDataset` over S3 streaming to avoid full local download. Cell-aware crop selection: tile each site into non-overlapping `crop_size` patches, score each by CellProfiler centroid count, yield the top `crops_per_site` tiles.
3. **Architecture** (`src/cerberus_neuro/model.py`): ResNet34 backbone + three task heads (`CerberusModel`). Plus `BaselineDiseaseClassifier` (same encoder, 6-channel input, single disease head) as the all-channel upper-bound baseline. Both from scratch (no pretrained ImageNet weights) for the clean-public-reproduction angle.
4. **Training loop** (`src/cerberus_neuro/training.py`): multi-task loss with Kendall uncertainty weighting (or fixed magnitude-scaled weights); checkpoint to Drive + HF every N steps; resumable across Colab session restarts. Single training entry point that handles both `CerberusModel` (3 heads) and `BaselineDiseaseClassifier` (1 head) configs.
5. **Scoped v0 training runs** (`notebooks/02_train.ipynb`): two paired runs on the same scoped subset (plates, resolution, epochs): (a) `CerberusModel` brightfield-only multi-task → HF as `patrickjreed/cerberus-neuro-v0`; (b) `BaselineDiseaseClassifier` 6-channel single-task → HF as `patrickjreed/cerberus-neuro-v0-baseline`.
6. **Evaluation** (`notebooks/03_eval.ipynb`): for Cerberus, cell-type accuracy + per-channel virtual-staining MSE/SSIM + disease-state accuracy + AUC. For baseline, disease-state accuracy + AUC. Report the gap (Cerberus disease AUC vs baseline disease AUC) and the inference-cost ratio as the headline result. Worked-through cases with figures.
7. **Repo polish + writeup**: clean notebooks, figures, paper-style README extension or blog post framing the paired-experiment narrative ("Cerberus recovers Z% of the all-channel disease signal at 1/6 the assay cost"). Ship.

Stretch (v1+):

- **Full-scale training run** on rented A100 (Lambda / RunPod / Paperspace) with full resolution, all plates, full convergence. Checkpoint to `patrickjreed/cerberus-neuro-v1`.
- **Architecture variants** (ViT-family, EfficientNet swaps); ablations on task-head structure (independent heads vs shared MLP; learned vs fixed task weights)
- **Hyperparameter search** for task-loss weights and learning rate
- **HF Spaces Gradio demo**: upload a brightfield image, get virtual-staining predictions + cell-type call live
- **Possible preprint or workshop submission** if v1 results are publication-quality

## Status

Scaffold only. No analysis or training code yet. Files: `README.md`, `CLAUDE.md`, `pyproject.toml`, `LICENSE`, `.gitignore`, `Dockerfile`, `src/cerberus_neuro/__init__.py`, `docs/CONTEXT.md`, `docs/SETUP.md`, `notebooks/00_environment_smoke.ipynb`. Initial commit on `main`.
