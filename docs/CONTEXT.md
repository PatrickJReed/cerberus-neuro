# cerberus-neuro — project context

The strategic background behind this project: why it exists, why it's a public reproduction of internal BMS work, dataset and architecture rationale, and what the short- and long-term goals are. Companion to `README.md` (public-facing technical framing) and `CLAUDE.md` (working context for Claude Code sessions).

## Why this project exists

This project is part of a deliberate portfolio strategy by **Patrick J. Reed, Ph.D.** (computational biologist, 15+ years; most recently Principal Scientist at Bristol Myers Squibb). The project closes a specific portfolio gap: **prior work that lives only inside a former employer cannot be cited or shown to recruiters**.

At BMS in 2025, Patrick built a multi-task foundation-model proof-of-concept for the Neurobot high-throughput screening platform: a Cerberus-inspired ResNet34 with shared encoder and three task heads (cell-type classification, virtual staining, disease-state classification) trained on the publicly available Broad NeuroPainting dataset. The internal version was completed as a POC but not transferred to BMS internal data or productionized into Neurobot screening workflows before the April 2026 layoff. As a result, Patrick describes the work in interviews and on resumes but cannot point recruiters to a citable artifact.

`cerberus-neuro` is a **clean public reproduction** built from scratch on the same public dataset, using the same architectural pattern. No BMS internal data, no internal hyperparameters, no internal infrastructure — just the architecture, the public NeuroPainting data, and a public training run. The artifact is the version anyone can run, cite, fork, or extend.

The decision to rebuild publicly was made during the 2026-05-05 portfolio refresh session, alongside the related but distinct `cellduet` project (multimodal perturbation concordance).

## Why these design choices

### Why a public reproduction (not new research)

Two reasons:

1. **The internal version was already done on public data.** The Broad NeuroPainting dataset is publicly available on the Cell Painting Gallery. The BMS POC trained on this public data, so a public rebuild has no IP or cleanroom constraints.
2. **Faithful reproduction has more credentialing value than novel architecture exploration for this specific gap.** The point is to demonstrate *what Patrick built at BMS*, in a form anyone can run and verify. Architecture variants are v1 stretch, not v0.

### Why this dataset (Broad NeuroPainting)

- **Public**: Available on the Cell Painting Gallery on AWS Open Data; permissive licensing for non-commercial use.
- **Disease-relevant**: 22q11.2 deletion is a well-characterized neuropsychiatric risk factor; iPSC + Cell Painting framing applies broadly to neurodegenerative and neurodevelopmental disorders.
- **Multi-modal in the right way**: Cell Painting fluorescence channels + brightfield input enable the virtual-staining task naturally. Not all imaging datasets support that.
- **Manageable scale**: 44 lines × 4 cell types × Cell Painting plates × multiple fields. Large enough to be non-trivial, small enough to fit on Colab Free with subset scoping for v0.
- **Cited and traceable**: Tegtmeyer et al., *Nat Commun* 2025 ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x)) is the canonical reference; readers can locate the project in the published literature.

### Why this architecture (Cerberus-inspired ResNet34 + 3 heads)

- **Faithful to the BMS POC**: this is the architectural choice Patrick actually shipped internally. The point of a public reproduction is to demonstrate that work, not invent a new architecture.
- **Three-headed shared-encoder pattern** is well-supported in the multi-task vision literature (Caruana 1997 onward; the Cerberus framing comes from various ML-systems papers).
- **ResNet34 is small enough to train on a T4 16GB**, large enough to be non-toy. Avoids the "I trained a 200M-parameter model on a tiny dataset" anti-pattern.
- **Three tasks span a useful gradient**: classification (cell type), regression (virtual staining), binary classification (disease state). Single architecture, three task types, gives reviewers a complete multi-task vision artifact.

### Why Colab + Docker (not pure cloud or pure local)

This is documented in `docs/SETUP.md`. Short version:
- Colab handles iteration speed and Free-tier compute for v0.
- Docker captures the env so the same code runs locally, on Lambda/RunPod for full-scale training, or in production-style deployment.
- Multi-platform Docker portability is itself a credentialing signal that's missing from Patrick's portfolio; baked in from day 1 is cheap, retrofitted later is expensive.
- **No Colab-Pro or paid-cloud commitment in v0**. Pro is available as a fallback if Free becomes a bottleneck; full-scale training runs on a rented A100 ($2-15 per converged run) only when v1 is in scope.

## Strategic positioning

The artifact is built to support applications to:

- **Recursion** — image-based CRISPR perturbation and Cell Painting are core to their platform. cerberus-neuro is on-domain.
- **Insitro** — cellular phenotyping + ML; multi-task vision on iPSC imaging maps directly.
- **Iambic Therapeutics** — generative + multi-task vision in drug discovery.
- **Anthropic Applied AI Engineer Life Sciences** — pairs with cellduet to demonstrate breadth: Patrick can ship multimodal *and* multi-task biological vision artifacts. The HHMI / Allen Institute partner work specifically values applied imaging-AI.
- **Pharma comp-bio roles** with Cell Painting / phenotypic screening (BMS, Lilly, Pfizer, AbbVie). Patrick's BMS work gets a citable public version.
- **Generally**: any role asking "show me your prior work on multi-task vision in biology" — the cerberus-neuro repo + writeup is the answer.

## Connection to other portfolio work

- **`~/Sandbox/cellduet/`** — sibling project. cellduet analyzes pre-computed embeddings across modalities (transcriptomic + morphological); cerberus-neuro trains a multi-task model on the morphological side. The two together demonstrate Patrick can both (a) train domain-specific models and (b) reason analytically across pre-trained embedding spaces.
- **`docs/NeuroPainting_MultiTask_Model_Dossier.md`** in NewRoleEfforts — comprehensive reference on the BMS internal version, including dataset details, Cerberus inspiration, connections to Tegtmeyer et al.
- **`PatrickReed_Accomplishments_Bank.md`** "Neurobot Multi-Task Foundation Model: Proof-of-Concept (BMS, 2025)" entry — internal version's accomplishment record. Patrick should reference cerberus-neuro from that entry once v0 ships.
- **TDP43/STMN2 CCT screening model** (BMS, 2024-2025) — different architecture (Compact Convolutional Transformer), same Cell Painting / iPSC neuron domain. Both demonstrate Patrick can ship vision models in HTS contexts.

## Honestly acknowledged gaps

These are intentionally surfaced, not buried:

- **This is a reproduction, not novel research.** The README and CLAUDE.md say so. The artifact's value is "Patrick can build and ship multi-task vision models on bio imaging data," not "Patrick discovered something new."
- **Performance numbers will not match the BMS internal version**, because the internal version had access to BMS infrastructure, internal hyperparameter tuning, and internal data preparation tooling. Reproduction targets reasonable public-data performance, not parity. This is named in the writeup.
- **The Broad NeuroPainting dataset is non-commercial-licensed for some uses**; check licensing before any commercial repurposing of the trained model.
- **22q11.2 deletion is one disease state**; the model does not generalize to other disease conditions without retraining. Generalization to other conditions is a v2+ direction, not a v0 claim.
- **Patrick is not credited as an author on the upstream Tegtmeyer et al. publication.** This is a reproduction of a model architecture *applied to* their public data, not a re-implementation of their methodology.

## Short-term goals (v0, ~6–8 weeks evening work)

1. Establish data pipeline against the Cell Painting Gallery on AWS S3.
2. Implement Cerberus-inspired ResNet34 + 3 task heads from scratch.
3. Multi-task training loop with checkpointing, resumability, and per-task loss weighting.
4. Scoped training run (subset of plates, reduced resolution) on Colab Free demonstrating convergence on all three tasks.
5. Evaluation: cell-type accuracy, per-channel virtual-staining MSE/SSIM, disease-state accuracy + AUC.
6. Repo polish: clean notebooks, figures, paper-style writeup, HF model checkpoint pushed.

## Long-term goals (v1+, opportunistic)

- **Full-scale training run** on rented A100 (Lambda / RunPod / Paperspace) — full resolution, all plates, full convergence. Push to HF as `patrickjreed/cerberus-neuro-v1`.
- **Architecture variants**: ViT-family backbone, EfficientNet swaps, learned vs fixed task-loss weights.
- **HF Spaces Gradio demo**: upload a brightfield image, predict virtual stains + cell type live.
- **Possible preprint or workshop submission** if v1 quality justifies it.

## Source documents

- `~/Sandbox/cellduet/docs/CONTEXT.md` — sibling project's strategic background; shares the broader portfolio rationale
- `~/NewRoleEfforts/docs/portfolio_project_brainstorm.md` — overall portfolio strategy and project rankings
- `~/NewRoleEfforts/docs/NeuroPainting_MultiTask_Model_Dossier.md` — comprehensive reference on the BMS internal POC
- `~/NewRoleEfforts/PatrickReed_Master_Resume.md` — gold-standard claims source for any prose about Patrick's prior work
- `~/NewRoleEfforts/PatrickReed_Accomplishments_Bank.md` — extended detail on BMS, Ionis, DNAtrix, Salk projects
- `~/NewRoleEfforts/PatrickReed_Claims_Audit.md` — claims flagged for revision; consult before writing about prior work
- `~/NewRoleEfforts/PatrickReed_Writing_Style.md` — voice rules
