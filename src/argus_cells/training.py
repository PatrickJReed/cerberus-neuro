"""Multi-task training loop for cerberus-neuro v0.

Single :func:`train` entry point dispatches on model type:

- :class:`~argus_cells.model.CerberusModel`: 3-head loss with Kendall
  uncertainty weighting; CE for cell type, CE for line condition,
  ``0.85 * L1 + 0.15 * (1 - SSIM)`` for virtual staining.
- :class:`~argus_cells.model.BaselineDiseaseClassifier`: single-task CE
  on disease, all 6 channels stacked as input.

AdamW + cosine annealing, AMP enabled by default on CUDA. Resumable across
Colab session restarts via a single ``latest.pt`` checkpoint on Drive (the
retention policy keeps only the most recent state locally, since each
checkpoint is ~290 MB and Drive Free is 15 GB). Per-epoch history lives on
the HF Hub: ``latest.pt`` is uploaded as ``epoch_NNN.pt`` at the end of
each epoch when ``hf_repo`` is set.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias
from pytorch_msssim import ssim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from argus_cells.model import (
    CerberusOutput,
)


@dataclass
class TrainConfig:
    n_epochs: int = 5
    steps_per_epoch: int = 200  # IterableDataset: caller computes from manifest size
    lr: float = 3e-4  # learning rate for new (head) parameters
    encoder_lr_ratio: float = 1.0  # ratio for pretrained encoder; e.g. 0.1 -> encoder lr = 0.1 * lr
    weight_decay: float = 1e-4
    warmup_steps: int = 0  # 0 disables warmup; otherwise linear ramp 0 -> lr
    grad_clip_norm: float = 1.0  # 0 disables gradient norm clipping
    amp: bool = True
    log_every_steps: int = 25
    ckpt_every_steps: int = 250
    seed: int = 0


class KendallMultiTaskLoss(nn.Module):
    """Learned uncertainty weighting (Kendall, Gal, Cipolla 2018).

    ``total = sum_i 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i``

    Each task gets one trainable scalar ``log_var_i``; the optimizer adjusts
    it so high-noise tasks contribute less to the joint gradient.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = losses[0].new_zeros(())
        for lv, L in zip(self.log_vars, losses, strict=False):  # noqa: N806 — L is a loss scalar
            total = total + 0.5 * torch.exp(-lv) * L + 0.5 * lv
        return total


def virtual_staining_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.85,
    ssim_weight: float = 0.15,
) -> torch.Tensor:
    """Legacy image-generation loss: ``l1_weight * L1 + ssim_weight * (1 - SSIM)``.

    Kept available for ablation against the segmentation framing but no longer
    used by ``train()`` / ``evaluate()`` by default. Predicting per-pixel
    fluorescence intensity tends to collapse the decoder to a constant near
    the data mean (the local minimum of L1); the segmentation-style BCE loss
    below avoids this failure mode.
    """
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target, data_range=1.0, size_average=True)
    return l1_weight * l1 + ssim_weight * (1.0 - s)


def soft_dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - soft Dice averaged across channels. Probs in [0, 1], target in [0, 1].

    Dice = 2 * sum(p*t) / (sum(p) + sum(t)). Direct overlap-based supervision —
    its gradient is non-zero only when the prediction overlaps the target,
    which prevents the constant-prediction-at-data-mean attractor BCE alone
    falls into on sparse-foreground segmentation.
    """
    # probs, target: (B, C, H, W)
    intersection = (probs * target).sum(dim=(0, 2, 3))
    p_sum = probs.sum(dim=(0, 2, 3))
    t_sum = target.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (p_sum + t_sum + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined BCE-with-logits + soft Dice for organelle soft-segmentation.

    Pure BCE on sparse-foreground data has a constant-prediction local minimum
    at the per-channel mean: predicting ``p ≈ data_mean`` everywhere minimizes
    the per-pixel BCE without any spatial localization. Empirically, the
    decoder converges to this attractor and IoU stays flat near
    ``data_mean / 2`` across epochs.

    Adding a Dice term (overlap-based) supplies the missing spatial supervision.
    Dice gradient is zero only when intersection is zero, which directly
    penalizes "predict the mean everywhere" (no overlap with actual organelle
    locations).

    ``dice_weight=0.0`` falls back to pure BCE-with-logits. Default 0.5 follows
    the standard medical-imaging segmentation recipe (``0.5 * BCE + 0.5 *
    (1 - Dice)``).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    if dice_weight <= 0:
        return bce
    dice = soft_dice_loss(torch.sigmoid(logits), target)
    return (1.0 - dice_weight) * bce + dice_weight * dice


def soft_iou(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft intersection-over-union per channel.

    Returns a tensor of shape ``(C,)`` with one IoU per fluorescence channel.
    Soft (no thresholding): treats the [0, 1] predictions and targets as
    membership scores. ``intersection = sum(probs * target)``,
    ``union = sum(probs + target - probs * target)``.

    Caller is responsible for applying ``torch.sigmoid()`` to the model's
    raw logit output before passing to this function.
    """
    # probs, target: (B, C, H, W) in [0, 1]
    intersection = (probs * target).sum(dim=(0, 2, 3))
    union = (probs + target - probs * target).sum(dim=(0, 2, 3))
    return intersection / (union + eps)


def _cerberus_step(
    out: CerberusOutput,
    fluo: torch.Tensor,
    ct: torch.Tensor,
    cond: torch.Tensor,
    kendall: KendallMultiTaskLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    L_ct = F.cross_entropy(out.cell_type_logits, ct)  # noqa: N806 — L denotes a loss term
    L_cond = F.cross_entropy(out.line_condition_logits, cond)  # noqa: N806 — L denotes a loss term
    L_seg = segmentation_loss(out.fluorescence_logits, fluo)  # noqa: N806 — L denotes a loss term
    total = kendall([L_ct, L_cond, L_seg])
    return total, {
        "L_cell_type": L_ct.item(),
        "L_line_condition": L_cond.item(),
        "L_segmentation": L_seg.item(),
        "log_var_ct": kendall.log_vars[0].item(),
        "log_var_cond": kendall.log_vars[1].item(),
        "log_var_seg": kendall.log_vars[2].item(),
    }


def _baseline_step(
    logits: torch.Tensor,
    cond: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    L = F.cross_entropy(logits, cond)  # noqa: N806 — L denotes a loss term
    return L, {"L_line_condition": L.item()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    kendall: KendallMultiTaskLoss | None,
    scaler,
    step: int,
    epoch: int,
    cfg: TrainConfig,
) -> Path:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "kendall": kendall.state_dict() if kendall is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": asdict(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    kendall: KendallMultiTaskLoss | None = None,
    scaler=None,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Strict load: refuse to silently overwrite a freshly-initialized model
    # (e.g., ImageNet-pretrained weights) with state from an architecturally
    # incompatible checkpoint. If the keys or shapes don't match, raise.
    incompat = model.load_state_dict(state["model"], strict=True)
    if hasattr(incompat, "missing_keys") and (incompat.missing_keys or incompat.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint at {path} doesn't match current model: "
            f"missing={incompat.missing_keys[:5]}, unexpected={incompat.unexpected_keys[:5]}. "
            f"Delete latest.pt to start from a fresh init, or fix the architecture mismatch."
        )
    # Each subordinate state load is wrapped: a config change between runs
    # (e.g., switching CosineAnnealingLR -> SequentialLR for warmup) will
    # produce an incompatible state dict for that one component. Warn and
    # continue with a freshly-initialized component rather than crash.
    if optimizer and state.get("optimizer"):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"WARN: optimizer state incompatible ({e}); using fresh optimizer")
    if scheduler and state.get("scheduler"):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as e:
            print(f"WARN: scheduler state incompatible ({e}); using fresh schedule")
    if kendall is not None and state.get("kendall"):
        try:
            kendall.load_state_dict(state["kendall"])
        except Exception as e:
            print(f"WARN: kendall state incompatible ({e}); using fresh log-vars")
    if scaler is not None and state.get("scaler"):
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"WARN: scaler state incompatible ({e}); using fresh scaler")
    return state.get("step", 0), state.get("epoch", 0)


def _push_to_hf(local_path: Path, repo_id: str, path_in_repo: str | None = None) -> None:
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo or local_path.name,
        repo_id=repo_id,
        repo_type="model",
    )


# Display names for the 5 fluorescence channels in CHANNEL_ORDER position.
# (The data pipeline stacks brightfield first, then these in this order.)
_FLUO_CH_NAMES = ["DNA", "Mito", "AGP", "ER", "RNA"]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    is_cerberus: bool,
) -> dict[str, float]:
    model.eval()
    n = 0
    correct_ct = correct_cond = 0
    sum_l_ct = sum_l_cond = sum_l_vs = 0.0
    n_fluo = len(_FLUO_CH_NAMES)
    sum_bce_per_ch = torch.zeros(n_fluo)
    sum_intersection = torch.zeros(n_fluo)
    sum_union = torch.zeros(n_fluo)

    for bf, fluo, ct, cond in loader:
        bf, fluo, ct, cond = (t.to(device) for t in (bf, fluo, ct, cond))
        if is_cerberus:
            out = model(bf)
            sum_l_ct += F.cross_entropy(out.cell_type_logits, ct).item() * bf.size(0)
            sum_l_cond += F.cross_entropy(out.line_condition_logits, cond).item() * bf.size(0)
            sum_l_vs += segmentation_loss(out.fluorescence_logits, fluo).item() * bf.size(0)
            correct_ct += (out.cell_type_logits.argmax(dim=1) == ct).sum().item()
            correct_cond += (out.line_condition_logits.argmax(dim=1) == cond).sum().item()

            # Per-channel segmentation metrics: BCE measures per-pixel agreement;
            # soft IoU measures how well the predicted soft mask overlaps the
            # target fluorescence-as-probability mask. Apply sigmoid to logits
            # before computing IoU since IoU expects probabilities in [0, 1].
            logits = out.fluorescence_logits.float()
            probs = torch.sigmoid(logits)
            target = fluo.float()
            for c in range(n_fluo):
                lc, pc, tc = logits[:, c : c + 1], probs[:, c : c + 1], target[:, c : c + 1]
                sum_bce_per_ch[c] += F.binary_cross_entropy_with_logits(lc, tc).item() * bf.size(0)
                sum_intersection[c] += (pc * tc).sum().item()
                sum_union[c] += (pc + tc - pc * tc).sum().item()
        else:
            logits = model(torch.cat([bf, fluo], dim=1))
            sum_l_cond += F.cross_entropy(logits, cond).item() * bf.size(0)
            correct_cond += (logits.argmax(dim=1) == cond).sum().item()
        n += bf.size(0)

    metrics: dict[str, float] = {
        "n_samples": n,
        "acc_line_condition": correct_cond / max(n, 1),
        "L_line_condition": sum_l_cond / max(n, 1),
    }
    if is_cerberus:
        metrics.update(
            {
                "acc_cell_type": correct_ct / max(n, 1),
                "L_cell_type": sum_l_ct / max(n, 1),
                "L_segmentation": sum_l_vs / max(n, 1),
            }
        )
        for c, name in enumerate(_FLUO_CH_NAMES):
            metrics[f"BCE_{name}"] = sum_bce_per_ch[c].item() / max(n, 1)
            metrics[f"IoU_{name}"] = (sum_intersection[c] / (sum_union[c] + 1e-6)).item()
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    checkpoint_dir: Path,
    val_loader: DataLoader | None = None,
    log_path: Path | None = None,
    hf_repo: str | None = None,
    device: str | None = None,
    resume_from: Path | None = None,
) -> dict:
    """Train Cerberus or baseline; returns final state summary."""
    # Dispatch on a stable class attribute rather than isinstance, so that
    # sys.modules cache resets in long-running notebooks (where the model
    # was instantiated before training.py was reimported) don't break.
    kind = getattr(model, "model_kind", None)
    is_cerberus = kind == "cerberus"
    # "baseline" (ResNet34) and "argus_cct" (Compact Convolutional Transformer)
    # are both single-head 6-channel disease classifiers: they share the same
    # training/eval path (cross-entropy on the disease logits over the
    # concatenated 6-channel input, uniform or discriminative LR depending on
    # whether the model exposes an `encoder` attribute).
    is_baseline = kind in ("baseline", "argus_cct")
    if not (is_cerberus or is_baseline):
        raise ValueError(
            f"Unsupported model type: {type(model).__name__} (model_kind={kind!r}). "
            "Expected CerberusModel, BaselineDiseaseClassifier, or ArgusCCT."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model = model.to(device)
    kendall = KendallMultiTaskLoss(n_tasks=3).to(device) if is_cerberus else None

    # Discriminative LR: pretrained encoder gets a slower LR than newly-initialized
    # heads. This protects ImageNet features while letting the heads (random init)
    # take big optimizer steps to fit the task. Default ratio 1.0 = uniform LR.
    if cfg.encoder_lr_ratio != 1.0 and hasattr(model, "encoder"):
        encoder_params = list(model.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
        if kendall is not None:
            head_params += list(kendall.parameters())
        optimizer = AdamW(
            [
                {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_ratio},
                {"params": head_params, "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        params = list(model.parameters()) + (list(kendall.parameters()) if kendall else [])
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.n_epochs * cfg.steps_per_epoch
    if cfg.warmup_steps > 0:
        warmup = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - cfg.warmup_steps))
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler("cuda") if cfg.amp and device == "cuda" else None

    step, epoch_start = 0, 0
    if resume_from is not None and Path(resume_from).exists():
        step, epoch_start = load_checkpoint(
            Path(resume_from), model, optimizer, scheduler, kendall, scaler
        )
        print(f"resumed from {resume_from} at step={step}, epoch={epoch_start}")

    checkpoint_dir = Path(checkpoint_dir)
    log_path = Path(log_path) if log_path else (checkpoint_dir / "train_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epoch_start, cfg.n_epochs):
        model.train()
        if kendall:
            kendall.train()

        for bf, fluo, ct, cond in train_loader:
            bf, fluo, ct, cond = (t.to(device, non_blocking=True) for t in (bf, fluo, ct, cond))
            optimizer.zero_grad(set_to_none=True)

            ctx = torch.amp.autocast("cuda") if scaler else nullcontext()
            with ctx:
                if is_cerberus:
                    out = model(bf)
                    loss, metrics = _cerberus_step(out, fluo, ct, cond, kendall)
                else:
                    logits = model(torch.cat([bf, fluo], dim=1))
                    loss, metrics = _baseline_step(logits, cond)

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    if kendall is not None:
                        torch.nn.utils.clip_grad_norm_(kendall.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    if kendall is not None:
                        torch.nn.utils.clip_grad_norm_(kendall.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            scheduler.step()

            step += 1
            if step % cfg.log_every_steps == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    **metrics,
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[step {step:6d}] loss={loss.item():.4f}  "
                    f"lr={rec['lr']:.2e}  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                )

            if step % cfg.ckpt_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    kendall,
                    scaler,
                    step,
                    epoch,
                    cfg,
                )

            if step >= (epoch + 1) * cfg.steps_per_epoch:
                break

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, is_cerberus)
            print(
                f"[epoch {epoch}] val: " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )
            with log_path.open("a") as f:
                f.write(json.dumps({"epoch": epoch, "split": "val", **val_metrics}) + "\n")

        # Retention policy: keep only latest.pt locally. HF Hub holds the
        # per-epoch history (latest.pt is uploaded as epoch_NNN.pt to the repo).
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            kendall,
            scaler,
            step,
            epoch,
            cfg,
        )
        if hf_repo:
            try:
                _push_to_hf(
                    checkpoint_dir / "latest.pt", hf_repo, path_in_repo=f"epoch_{epoch:03d}.pt"
                )
                print(f"pushed latest.pt to HF {hf_repo} as epoch_{epoch:03d}.pt")
            except Exception as e:
                print(f"HF push failed: {e}")

    return {
        "final_step": step,
        "final_epoch": epoch_start + cfg.n_epochs - 1,
        "checkpoint_dir": str(checkpoint_dir),
    }
