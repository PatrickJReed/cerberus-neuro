# cerberus-neuro

> Cerberus-inspired multi-task ResNet34 on the Broad NeuroPainting Cell Painting dataset. Three task heads on a shared image encoder: cell-type classification, virtual staining (brightfield → 5 fluorescence channels), and disease-state classification (control vs 22q11.2 deletion).

## What this is

A public reproduction of a multi-task foundation-model proof-of-concept originally built at Bristol Myers Squibb in 2025 for the Neurobot high-throughput screening platform. The internal version was completed on the same public Broad NeuroPainting dataset; the rebuild here is the version anyone can cite, run, and extend.

## Architecture

Three-headed (Cerberus-inspired) ResNet34 sharing a single image encoder across:

1. **Cell-type classification** — iPSC / NPC / Neuron / Astrocyte (4-class softmax)
2. **Virtual staining** — predicting 5 Cell Painting fluorescence channels (nuclei / ER / nucleoli + RNA / actin + Golgi + plasma membrane / mitochondria) from brightfield input (per-channel regression)
3. **Disease-state classification** — control vs 22q11.2 deletion carrier (binary)

Multi-task losses are weighted; shared encoder trained end-to-end. Implementation details and training config are in `src/cerberus_neuro/` and `notebooks/` as the project develops.

## Data

[Broad Institute NeuroPainting](https://broadinstitute.github.io/cellpainting-gallery/) (Tegtmeyer et al., *Nat Commun* 2025; [10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x))

- 44 iPSC lines: 22 control, 22 with 22q11.2 deletion
- Four cell types: iPSC, NPC, Neuron, Astrocyte
- Cell Painting imaging: 5 fluorescence channels + brightfield, multiple fields per well

Hosted on the Cell Painting Gallery on AWS S3 (public, free egress).

## Status

Scaffold only. v0 in progress. See [`docs/CONTEXT.md`](docs/CONTEXT.md) for project background and [`CLAUDE.md`](CLAUDE.md) for working conventions.

## Running

Two supported workflows:

| Workflow | When to use |
|---|---|
| **Google Colab** | Day-to-day development and small training runs. Free tier handles scoped v0 demos; Pro recommended for longer training. |
| **Docker** (local / Lambda / RunPod / Paperspace) | Full-scale training, reproducible runs, portability across cheap GPU rentals. |

First-time setup (HF account, Colab activation, Docker basics, VS Code workflow): see [`docs/SETUP.md`](docs/SETUP.md).

## Related work

- Tegtmeyer et al., *Nat Commun* (2025). Combining NeuroPainting with transcriptomics reveals cell-type-specific morphological and molecular signatures of the 22q11.2 deletion. ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x))
- Bray et al. *Nat Protoc* (2016). Cell Painting protocol.
- Cell Painting Gallery (AWS Open Data) — large-scale Cell Painting datasets.
- Recursion RxRx3 + Phenom embeddings — image-based CRISPR perturbation.
- Sun et al., *Cell* (2025). Perturb-Multimodal: pooled genetic screens with imaging and sequencing.

## Author

Patrick J. Reed, Ph.D. ([LinkedIn](https://linkedin.com/in/patrickjenningsreed))

15+ years computational biology, most recently Principal Scientist at Bristol Myers Squibb working on single-cell foundation models, multi-task vision models for HTS, and atlas-scale data engineering. cerberus-neuro is a deliberate public artifact of work originally done internally at BMS, rebuilt on the same public dataset for the broader community.

## License

MIT — see [LICENSE](LICENSE).
