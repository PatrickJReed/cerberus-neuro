"""Multi-task training loop for cerberus-neuro v0.

Single :func:`train` entry point dispatches on model type:

- :class:`~cerberus_neuro.model.CerberusModel`: 3-head loss with Kendall
  uncertainty weighting; CE for cell type, CE for line condition,
  ``0.85 * L1 + 0.15 * (1 - SSIM)`` for virtual staining.
- :class:`~cerberus_neuro.model.BaselineDiseaseClassifier`: single-task CE
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
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from cerberus_neuro.model import (
    BaselineDiseaseClassifier,
    CerberusModel,
    CerberusOutput,
)


@dataclass
class TrainConfig:
    n_epochs: int = 5
    steps_per_epoch: int = 200          # IterableDataset: caller computes from manifest size
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 0                # 0 disables warmup; otherwise linear ramp 0 -> lr
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
        for lv, L in zip(self.log_vars, losses):
            total = total + 0.5 * torch.exp(-lv) * L + 0.5 * lv
        return total


def virtual_staining_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.85,
    ssim_weight: float = 0.15,
) -> torch.Tensor:
    """``l1_weight * L1 + ssim_weight * (1 - SSIM)`` over (B, C, H, W) in [0, 1]."""
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target, data_range=1.0, size_average=True)
    return l1_weight * l1 + ssim_weight * (1.0 - s)


def _cerberus_step(
    out: CerberusOutput,
    fluo: torch.Tensor,
    ct: torch.Tensor,
    cond: torch.Tensor,
    kendall: KendallMultiTaskLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    L_ct = F.cross_entropy(out.cell_type_logits, ct)
    L_cond = F.cross_entropy(out.line_condition_logits, cond)
    L_vs = virtual_staining_loss(out.fluorescence_pred, fluo)
    total = kendall([L_ct, L_cond, L_vs])
    return total, {
        "L_cell_type": L_ct.item(),
        "L_line_condition": L_cond.item(),
        "L_virtual_staining": L_vs.item(),
        "log_var_ct": kendall.log_vars[0].item(),
        "log_var_cond": kendall.log_vars[1].item(),
        "log_var_vs": kendall.log_vars[2].item(),
    }


def _baseline_step(
    logits: torch.Tensor,
    cond: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    L = F.cross_entropy(logits, cond)
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
    for bf, fluo, ct, cond in loader:
        bf, fluo, ct, cond = (t.to(device) for t in (bf, fluo, ct, cond))
        if is_cerberus:
            out = model(bf)
            sum_l_ct += F.cross_entropy(out.cell_type_logits, ct).item() * bf.size(0)
            sum_l_cond += F.cross_entropy(out.line_condition_logits, cond).item() * bf.size(0)
            sum_l_vs += virtual_staining_loss(out.fluorescence_pred, fluo).item() * bf.size(0)
            correct_ct += (out.cell_type_logits.argmax(dim=1) == ct).sum().item()
            correct_cond += (out.line_condition_logits.argmax(dim=1) == cond).sum().item()
        else:
            logits = model(torch.cat([bf, fluo], dim=1))
            sum_l_cond += F.cross_entropy(logits, cond).item() * bf.size(0)
            correct_cond += (logits.argmax(dim=1) == cond).sum().item()
        n += bf.size(0)

    out = {
        "n_samples": n,
        "acc_line_condition": correct_cond / max(n, 1),
        "L_line_condition": sum_l_cond / max(n, 1),
    }
    if is_cerberus:
        out.update({
            "acc_cell_type": correct_ct / max(n, 1),
            "L_cell_type": sum_l_ct / max(n, 1),
            "L_virtual_staining": sum_l_vs / max(n, 1),
        })
    return out


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
    is_baseline = kind == "baseline"
    if not (is_cerberus or is_baseline):
        raise ValueError(
            f"Unsupported model type: {type(model).__name__} (model_kind={kind!r}). "
            "Expected CerberusModel or BaselineDiseaseClassifier."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model = model.to(device)
    kendall = KendallMultiTaskLoss(n_tasks=3).to(device) if is_cerberus else None

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
        step, epoch_start = load_checkpoint(Path(resume_from), model, optimizer, scheduler, kendall, scaler)
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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            step += 1
            if step % cfg.log_every_steps == 0:
                rec = {
                    "step": step, "epoch": epoch,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    **metrics,
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[step {step:6d}] loss={loss.item():.4f}  "
                      f"lr={rec['lr']:.2e}  " +
                      "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

            if step % cfg.ckpt_every_steps == 0:
                save_checkpoint(checkpoint_dir / "latest.pt",
                                model, optimizer, scheduler, kendall, scaler, step, epoch, cfg)

            if step >= (epoch + 1) * cfg.steps_per_epoch:
                break

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, is_cerberus)
            print(f"[epoch {epoch}] val: " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
            with log_path.open("a") as f:
                f.write(json.dumps({"epoch": epoch, "split": "val", **val_metrics}) + "\n")

        # Retention policy: keep only latest.pt locally. HF Hub holds the
        # per-epoch history (latest.pt is uploaded as epoch_NNN.pt to the repo).
        save_checkpoint(checkpoint_dir / "latest.pt",
                        model, optimizer, scheduler, kendall, scaler, step, epoch, cfg)
        if hf_repo:
            try:
                _push_to_hf(checkpoint_dir / "latest.pt", hf_repo,
                            path_in_repo=f"epoch_{epoch:03d}.pt")
                print(f"pushed latest.pt to HF {hf_repo} as epoch_{epoch:03d}.pt")
            except Exception as e:
                print(f"HF push failed: {e}")

    return {"final_step": step, "final_epoch": epoch_start + cfg.n_epochs - 1,
            "checkpoint_dir": str(checkpoint_dir)}
