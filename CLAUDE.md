# CLAUDE.md — context for Claude Code sessions in this repo

This file seeds the next Claude Code session with project context that isn't obvious from the code or README alone. Read this before doing meaningful work.

For the **strategic background** behind the project (why it exists, role-class positioning, connection to other portfolio projects, short- and long-term goals, honestly-acknowledged gaps), read `docs/CONTEXT.md`. This file (`CLAUDE.md`) is for *how to work in the codebase*; `docs/CONTEXT.md` is for *why the codebase exists*. **Note:** `docs/CONTEXT.md` still describes the retired cerberus three-headed framing and is pending a reframe update; treat its "why" content as historical until then.

## What this repo is

This repo is a portfolio research artifact authored by Patrick J. Reed, Ph.D. (computational biologist, 15+ years; recent Principal Scientist at Bristol Myers Squibb). The project is **`argus-cells`**: a 6-channel Cell Painting disease classifier on the Broad NeuroPainting iPSC dataset (22q11.2 deletion vs control), paired with an interpretability harness that explains, honestly and across architectures, what drives the disease call. The classifier is table stakes (target: match the existing 0.73 baseline); the interpretability story is the headline deliverable.

**The repo is still named `cerberus-neuro` and the package is still `cerberus_neuro`.** Per the [argus-cells design spec](docs/superpowers/specs/2026-05-12-argus-cells-design.md) §7, the rename to `argus-cells` / `argus_cells` / `patrickjreed/argus-*` happens at the **start of Phase 2** (production training). Until then, Phase 1 (interpretability harness) runs in this repo against the existing `patrickjreed/cerberus-neuro-v0-baseline` checkpoint. Do not rename anything until Phase 2 begins.

**Retired framing (do not revive without explicit user approval):** the project was originally `cerberus-neuro`, a three-headed multi-task ResNet34 (cell-type + organelle soft-segmentation + disease) framed as a public reproduction of a BMS internal POC. That framing is dead. No virtual-staining head, no segmentation head, no Kendall uncertainty weighting, no brightfield-only-recovery question. Multi-task stays in the toolbox only if a future phase finds it useful.

**Current architecture: two paired single-head disease classifiers**, chosen for complementary attribution surfaces, not for an accuracy tournament.
- **Argus-RN34** — ResNet34, ImageNet1K_V1 pretrained, 6-channel `conv1` via tile-and-scale, GradCAM attribution. Exists today as the 0.73 baseline checkpoint (`BaselineDiseaseClassifier` in `model.py`).
- **Argus-CCT** — Compact Convolutional Transformer, from scratch, 6-channel native, attention-rollout attribution. Arrives in Phase 2.

The interpretability harness (the deliverable) lives in `src/cerberus_neuro/{attribution,probes,analysis}/`: channel ablation, GradCAM, Integrated Gradients, donor-confound probe, and cell-type stratification, all behind a common `AttributionResult` interface.

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

- **Two single-head disease classifiers, period: Argus-RN34 and Argus-CCT.** No third architecture, no ResNet50 / ViT-B / ConvNeXt, unless Phase 2 surfaces a specific reason and the user approves. The two are chosen for complementary attribution surfaces, not for an accuracy tournament.
- **Interpretability is the deliverable; accuracy is table stakes.** Success at Phase 2 is "at least one model matches the existing 0.73 baseline." Do not chase raw accuracy past that. If both models significantly underperform 0.73, treat it as a training problem and pause the analysis phase for a scope review.
- **Not a BMS reproduction, not chasing any published claim.** The biology output is whatever the data says, framed as "the model says X" rather than "we proved X". No pre-committed result (including Tegtmeyer 2025).
- **No hyperparameter sweeps.** One recipe per architecture, reusing the validated Phase 0.5 / Phase 1 patterns. Sweeps are v1.
- **6-channel input only.** Argus-RN34 adapts `conv1` (ImageNet1K_V1 pretrained) via tile-and-scale; Argus-CCT uses a native 6-channel conv tokenizer from scratch. The 1-channel brightfield variant from the cerberus era is retired.

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
- **Honest framing.** Never invent metrics, never claim experience that doesn't exist. If a result is preliminary, say so. If a method has known caveats, name them. Biology claims are framed as "the model says X", not "we proved X". The README's "Project lineage" section is explicit that this began as a BMS-POC reproduction and was reframed; don't blur that line or revive the retired framing.

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

## Phased plan

Source of truth: the [argus-cells design spec](docs/superpowers/specs/2026-05-12-argus-cells-design.md) §5 and the per-phase plans/specs/results under `docs/superpowers/`. Tooling-first: build and validate the interpretability harness against the existing 0.73 baseline before training the production models.

| Phase | Deliverable | Compute |
|---|---|---|
| **0.** Donor + dataset audit | Donor counts, imbalance, crop-budget decision. Hard gate before Phase 1. | Colab Free |
| **1.** Interpretability harness on the 0.73 baseline | All attribution methods + donor probe + cell-type stratification running end-to-end on `cerberus-neuro-v0-baseline`. Harness reusable on any 6-channel encoder. No new accuracy claims. | Colab Pro |
| **2.** Train paired classifiers | `argus-rn34-v0` + `argus-cct-v0` on scaled crops, best-epoch val acc each. Initialize the `argus-cells` repo/package here. | Colab Pro + ~1-2 hr A100 (CCT) |
| **3.** Full analysis | Per-cell-type x per-channel ablation tables, cross-architecture saliency agreement, donor-probe confound result, honest biology writeup. | Colab Pro (inference) |
| **4.** Polish + ship | `argus-cells` on GitHub + PyPI, README narrative, HF model cards. | Local |

## Status (as of 2026-06-09)

The CLAUDE.md you are reading was just refreshed from the stale "scaffold only" state; trust this section and `git log` over older doc prose.

**Shipped (predecessor cerberus-era artifacts, real and on HF):**
- Cell-type classifier, val acc **0.9905** → `patrickjreed/cerberus-neuro-cell-type-v0`.
- 6-channel baseline disease classifier, val acc **0.7311** (best epoch) → `patrickjreed/cerberus-neuro-v0-baseline`. This is the Phase 1 harness target and the Argus-RN34 predecessor.

**Phase 0 (done):** donor audit cleared the PROCEED gate (24 donors/condition, imbalance CV near 0, crop budget sufficient). See `docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`.

**Phase 1 (in flight):** the interpretability harness is **code-complete and unit-tested** (33 tests passing) in `src/cerberus_neuro/{attribution,probes,analysis}/`, and `notebooks/04_phase_1_harness.ipynb` is fully scaffolded (14 cells, outputs cleared). Remaining Phase 1 work (plan Tasks 15-17): **(1)** run the notebook on Colab against the 0.73 checkpoint and pull the executed copy back; **(2)** write the Phase 1 interpretability results doc with the Phase 2 gate decision; **(3)** announce. Attention rollout and cross-architecture agreement are deliberately deferred to Phase 2 (they need Argus-CCT).

**Known debt:** `docs/CONTEXT.md` still reflects the retired cerberus three-headed framing (pending reframe). Repo carries ~100 cosmetic ruff lint findings (import sorting, unused imports, notebook E402); no functional bugs. Package/repo rename to `argus-cells` is intentionally deferred to Phase 2.
