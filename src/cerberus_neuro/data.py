"""NeuroPainting data pipeline (cpg0038-tegtmeyer-neuropainting).

Two entry points:

- :func:`build_manifest` — concatenate per-plate ``load_data.csv`` URL columns
  with biological labels from per-plate ``platemap.txt`` files, returning one
  row per (plate, well, site).
- :class:`NeuroPaintingDataset` — IterableDataset streaming
  ``(brightfield[1, h, w], fluorescence[5, h, w], cell_type, line_condition)``
  tuples. v0 strategy: tile each 2160×2160 site into non-overlapping
  ``crop_size`` patches, score each by CellProfiler centroid count, yield
  the top ``crops_per_site`` tiles (filtered to ``>= min_cells_per_crop``).
  Same coordinates are applied to all 6 channels so brightfield and
  fluorescence stay aligned. Per-cell crops via the segmentation outlines
  under ``publication_data/`` remain a v1 option.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from torch.utils.data import IterableDataset

BUCKET = "cellpainting-gallery"
ROOT_PREFIX = "cpg0038-tegtmeyer-neuropainting/"
WORKSPACE_PREFIX = ROOT_PREFIX + "broad/workspace/"
IMAGES_PREFIX = ROOT_PREFIX + "broad/images/"

# Default v0 batch set excludes the 63× neurons batch (different magnification
# breaks the cell-type-classification task; reserved for v1 resolution-invariance
# work).
BATCHES_V0 = [
    "NCP_ASTROCYTES_1",
    "NCP_NEURONS_2_20x",
    "NCP_PROGENITORS_1",
    "NCP_STEM_1",
]

# Brightfield is the model input; the five fluorescence channels are virtual-
# staining targets. This order is what the dataset stacks into the channel axis.
CHANNEL_INPUT = "OrigBrightfield"
CHANNELS_FLUORESCENCE = ["OrigDNA", "OrigMito", "OrigAGP", "OrigER", "OrigRNA"]
CHANNEL_ORDER = [CHANNEL_INPUT] + CHANNELS_FLUORESCENCE

CELL_TYPES = ["stem", "progen", "neuron", "astro"]
CELL_TYPE_TO_IDX = {c: i for i, c in enumerate(CELL_TYPES)}

LINE_CONDITIONS = ["control", "deletion"]
LINE_CONDITION_TO_IDX = {c: i for i, c in enumerate(LINE_CONDITIONS)}


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _list_recursive(s3, prefix: str) -> list[tuple[str, int]]:
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 1000}
    ):
        out.extend((o["Key"], o["Size"]) for o in page.get("Contents", []))
    return out


def _download(s3, key: str, local: Path) -> Path:
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local))
    return local


def _key_from_url(url: str) -> str:
    prefix = f"s3://{BUCKET}/"
    return url[len(prefix) :] if url.startswith(prefix) else url.lstrip("/")


def build_manifest(cache_dir: Path, batches: list[str] | None = None) -> pd.DataFrame:
    """Build a per-(plate, well, site) manifest.

    Joins per-plate ``load_data.csv`` (URL_* image columns + Metadata_Site) with
    per-plate ``platemap.txt`` (Metadata_cell_type, Metadata_line_ID,
    Metadata_line_condition, Metadata_line_source).

    Parameters
    ----------
    cache_dir
        Local directory used to cache downloaded CSVs. Files land at
        ``cache_dir / <s3_key>``.
    batches
        Subset of batches to include. Defaults to :data:`BATCHES_V0` (excludes
        the 63× neurons batch).

    Returns
    -------
    DataFrame indexed by (Metadata_Plate, Metadata_Well, Metadata_Site) with
    columns: URL_OrigBrightfield, URL_OrigDNA, URL_OrigMito, URL_OrigAGP,
    URL_OrigER, URL_OrigRNA, Metadata_cell_type, Metadata_line_ID,
    Metadata_line_condition, Metadata_line_source, batch.
    """
    cache_dir = Path(cache_dir)
    s3 = _s3_client()
    batches = batches if batches is not None else BATCHES_V0

    platemap_frames = []
    for batch in batches:
        for key, _ in _list_recursive(s3, f"{WORKSPACE_PREFIX}metadata/{batch}/platemap/"):
            if not key.endswith(".txt"):
                continue
            local = _download(s3, key, cache_dir / key)
            df = pd.read_csv(local, sep="\t")
            df["batch"] = batch
            platemap_frames.append(df)
    pmap = pd.concat(platemap_frames, ignore_index=True).drop_duplicates(
        subset=["Metadata_Plate", "Metadata_Well"]
    )

    keep_cols = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"] + [
        f"URL_{c}" for c in CHANNEL_ORDER
    ]
    load_frames = []
    for batch in batches:
        for key, _ in _list_recursive(s3, f"{WORKSPACE_PREFIX}load_data_csv/{batch}/"):
            if not key.endswith("/load_data.csv"):
                continue
            local = _download(s3, key, cache_dir / key)
            df = pd.read_csv(local, usecols=keep_cols)
            df["batch"] = batch
            load_frames.append(df)
    loads = pd.concat(load_frames, ignore_index=True).drop_duplicates(
        subset=["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    )

    pmap_cols = [
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_cell_type",
        "Metadata_line_ID",
        "Metadata_line_condition",
        "Metadata_line_source",
    ]
    return loads.merge(pmap[pmap_cols], on=["Metadata_Plate", "Metadata_Well"], how="inner")


def subset_manifest(
    manifest: pd.DataFrame,
    wells_per_cell_type: int | None = None,
    sites_per_well: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic subsampling for Colab-Free-friendly v0 runs.

    ``wells_per_cell_type`` is split evenly across line_condition (control vs
    deletion). ``sites_per_well`` caps the per-well site count. Both are
    optional; passing neither returns the input.
    """
    rng = np.random.default_rng(seed)
    if wells_per_cell_type is not None:
        kept = []
        per_cond = max(1, wells_per_cell_type // 2)
        for _, ct_df in manifest.groupby("Metadata_cell_type"):
            wells = ct_df[
                ["Metadata_Plate", "Metadata_Well", "Metadata_line_condition"]
            ].drop_duplicates()
            for _, cond_df in wells.groupby("Metadata_line_condition"):
                idx = rng.choice(len(cond_df), size=min(per_cond, len(cond_df)), replace=False)
                kept.append(cond_df.iloc[idx][["Metadata_Plate", "Metadata_Well"]])
        kept_keys = pd.concat(kept).drop_duplicates().reset_index(drop=True)
        manifest = manifest.merge(kept_keys, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    if sites_per_well is not None:
        manifest = (
            manifest.groupby(["Metadata_Plate", "Metadata_Well"], group_keys=False)
            .apply(lambda g: g.sample(min(sites_per_well, len(g)), random_state=seed))
            .reset_index(drop=True)
        )
    return manifest


def well_level_split(
    manifest: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``val_frac`` of wells per ``(cell_type, line_condition)`` for validation.

    Splits at the (Metadata_Plate, Metadata_Well) level so all sites from a
    given well land in the same split (no within-well leakage between train
    and val). Stratified by cell type x condition so each combination is
    represented in both splits.

    Returns
    -------
    (train_manifest, val_manifest) — same columns as the input manifest.
    """
    rng = np.random.default_rng(seed)
    train_keys, val_keys = [], []
    for _, grp in manifest.groupby(["Metadata_cell_type", "Metadata_line_condition"]):
        wells = grp[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().reset_index(drop=True)
        n_val = max(1, int(round(len(wells) * val_frac)))
        idx = rng.permutation(len(wells))
        val_keys.append(wells.iloc[idx[:n_val]])
        train_keys.append(wells.iloc[idx[n_val:]])
    train_keys_df = pd.concat(train_keys, ignore_index=True)
    val_keys_df = pd.concat(val_keys, ignore_index=True)
    train = manifest.merge(train_keys_df, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    val = manifest.merge(val_keys_df, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    return train, val


def _load_image(s3, url: str, cache_dir: Path) -> np.ndarray:
    key = _key_from_url(url)
    local = _download(s3, key, cache_dir / key)
    with Image.open(local) as im:
        return np.asarray(im)


_CENTROID_COL_CANDIDATES = [
    ("AreaShape_Center_Y", "AreaShape_Center_X"),
    ("Location_Center_Y", "Location_Center_X"),
    ("Center_Y", "Center_X"),
]


def load_cell_centroids(
    s3,
    batch: str,
    plate: str,
    well: str,
    site,
    cache_dir: Path,
) -> np.ndarray:
    """Return an ``Nx2`` array of ``(y, x)`` cell centroids in image-pixel space.

    Reads CellProfiler ``Cells.csv`` from
    ``workspace/analysis/<batch>/<plate>/analysis/<plate>-<well>-<site>/Cells.csv``.
    Tries common centroid column-name variants (CP version dependent).
    """
    site_str = str(site)
    key = f"{WORKSPACE_PREFIX}analysis/{batch}/{plate}/analysis/{plate}-{well}-{site_str}/Cells.csv"
    local = _download(s3, key, Path(cache_dir) / key)
    df = pd.read_csv(local)
    for y_col, x_col in _CENTROID_COL_CANDIDATES:
        if y_col in df.columns and x_col in df.columns:
            return np.stack([df[y_col].to_numpy(), df[x_col].to_numpy()], axis=1)
    raise KeyError(f"No centroid columns in {key}; first columns: {list(df.columns)[:20]}")


def crop_cell_count(centroids: np.ndarray | None, y: int, x: int, size: int) -> int:
    """How many centroids fall inside the ``[y, y+size) x [x, x+size)`` crop."""
    if centroids is None or len(centroids) == 0:
        return 0
    inside = (
        (centroids[:, 0] >= y)
        & (centroids[:, 0] < y + size)
        & (centroids[:, 1] >= x)
        & (centroids[:, 1] < x + size)
    )
    return int(inside.sum())


def apply_dihedral(channels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random D4 (4 rotations × optional horizontal flip) over a ``(C, H, W)`` crop.

    Cell Painting biology is invariant to rotation and reflection, so the D4
    dihedral group is a free 8× data multiplier. The transform is applied
    uniformly across the channel axis so brightfield input and fluorescence
    target stay registered.
    """
    k = int(rng.integers(0, 4))
    if k:
        channels = np.rot90(channels, k, axes=(-2, -1))
    if rng.random() < 0.5:
        channels = channels[:, :, ::-1]
    return np.ascontiguousarray(channels)


def tile_top_cells(
    centroids: np.ndarray | None,
    h: int,
    w: int,
    crop_size: int,
    n: int,
    stride: int | None = None,
    min_cells: int = 1,
) -> list[tuple[int, int, int]]:
    """Pick the top ``n`` non-overlapping tiles by cell count.

    Tiles ``(h, w)`` at ``stride`` (defaults to ``crop_size``, i.e.
    non-overlapping), scores each tile by the number of centroids it contains,
    drops tiles with fewer than ``min_cells``, and returns the top ``n``
    ``(y, x, n_cells)`` tuples in descending count order.
    """
    stride = stride or crop_size
    candidates: list[tuple[int, int, int]] = []
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            count = crop_cell_count(centroids, y, x, crop_size)
            if count >= min_cells:
                candidates.append((y, x, count))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:n]


@dataclass
class NeuroPaintingDataset(IterableDataset):
    """IterableDataset streaming (brightfield, fluorescence, cell_type, line_condition).

    Each yielded sample is a tuple of:

    - ``brightfield`` — float32 tensor, shape ``(1, crop_size, crop_size)``,
      values in ``[0, 1]``.
    - ``fluorescence`` — float32 tensor, shape ``(5, crop_size, crop_size)``,
      values in ``[0, 1]``, channel order ``[DNA, Mito, AGP, ER, RNA]``.
    - ``cell_type`` — int in ``[0, 4)`` (index into :data:`CELL_TYPES`).
    - ``line_condition`` — int in ``{0, 1}`` (control vs deletion).

    For each (plate, well, site) row in the manifest, the 2160×2160 image is
    tiled into non-overlapping ``crop_size`` patches; each tile is scored by
    the count of CellProfiler centroids it contains, and the top
    ``crops_per_site`` tiles (filtered to ``>= min_cells_per_crop``) are
    yielded in descending-count order. Same coordinates are applied to all 6
    channels so brightfield input and fluorescence target stay aligned. Sites
    missing a ``Cells.csv`` are skipped; sites with fewer qualifying tiles
    than ``crops_per_site`` yield however many are available.

    When ``augment=True`` (default), each yielded crop is run through a random
    D4 transform (one of 4 rotations × optional horizontal flip) applied
    identically across all 6 channels. Geometric augmentation only; no
    photometric jitter in v0 since Cell Painting acquisition is tightly
    normalized.
    """

    manifest: pd.DataFrame
    cache_dir: Path
    crop_size: int = 256
    crops_per_site: int = 4
    min_cells_per_crop: int = 1
    tile_stride: int | None = None
    augment: bool = True
    shuffle: bool = True
    seed: int = 0
    yield_donor: bool = False

    def __iter__(self) -> Iterator[tuple]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        n_workers = worker.num_workers if worker else 1

        s3 = _s3_client()
        cache = Path(self.cache_dir)
        stride = self.tile_stride or self.crop_size

        # Mix in time so each __iter__ call (i.e., each epoch) gets a fresh
        # shuffle and a fresh augmentation RNG. Using only self.seed + worker_id
        # would replay the identical batch sequence every epoch.
        epoch_seed = self.seed + worker_id + (time.time_ns() & 0xFFFFFFFF)
        rng = np.random.default_rng(epoch_seed)

        rows = self.manifest.iloc[worker_id::n_workers].reset_index(drop=True)
        if self.shuffle:
            rows = rows.sample(frac=1.0, random_state=int(epoch_seed % (2**32))).reset_index(
                drop=True
            )

        for _, row in rows.iterrows():
            try:
                channels = np.stack(
                    [_load_image(s3, row[f"URL_{c}"], cache) for c in CHANNEL_ORDER],
                    axis=0,
                )
                centroids = load_cell_centroids(
                    s3,
                    row["batch"],
                    row["Metadata_Plate"],
                    row["Metadata_Well"],
                    row["Metadata_Site"],
                    cache,
                )
            except Exception:
                continue

            channels = channels.astype(np.float32) / 65535.0
            ct = CELL_TYPE_TO_IDX[row["Metadata_cell_type"]]
            cond = LINE_CONDITION_TO_IDX[row["Metadata_line_condition"]]
            donor = int(row["Metadata_line_ID"])
            _, h, w = channels.shape

            selected = tile_top_cells(
                centroids,
                h,
                w,
                crop_size=self.crop_size,
                n=self.crops_per_site,
                stride=stride,
                min_cells=self.min_cells_per_crop,
            )
            for y, x, _ncells in selected:
                crop = channels[:, y : y + self.crop_size, x : x + self.crop_size]
                if self.augment:
                    crop = apply_dihedral(crop, rng)
                sample = (
                    torch.from_numpy(crop[:1].copy()),
                    torch.from_numpy(crop[1:].copy()),
                    ct,
                    cond,
                )
                # When yield_donor is set, append the crop's true donor ID so the
                # donor probe can pair each embedding with its line. Default off
                # keeps the 4-tuple every other consumer (training, eval) expects.
                if self.yield_donor:
                    sample = sample + (donor,)
                yield sample
