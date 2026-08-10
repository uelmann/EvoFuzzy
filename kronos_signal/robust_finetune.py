"""Resumable, leakage-free Kronos fine-tuning, one epoch per Modal call.

The original upstream loop is suitable for a stable multi-hour process.  This
version persists model, optimizer, scheduler, RNG-independent epoch metadata,
and validation history after every epoch so an interrupted GPU call can resume
without restarting training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .forecast import _ensure_kronos_on_path
from .official_config import OfficialConfig
from .official_dataset import OfficialKronosDataset

Phase = Literal["tokenizer", "predictor"]


def robust_config(root: str | Path, predictor_size: str = "base") -> OfficialConfig:
    """Configuration with disjoint train/validation/OOS periods."""
    cfg = OfficialConfig(root=root, predictor_size=predictor_size, epochs=30)
    cfg.train_time_range = ["2016-01-01", "2022-12-31"]
    cfg.val_time_range = ["2023-01-01", "2024-06-30"]
    cfg.test_time_range = ["2024-07-01", "2026-08-08"]
    cfg.backtest_time_range = ["2024-07-01", "2026-08-08"]

    # Conservative full-model adaptation.  The upstream 4e-5 OneCycle schedule
    # overfit this crypto validation set immediately; use plateau decay instead.
    cfg.tokenizer_learning_rate = 1e-4
    cfg.predictor_learning_rate = 1e-5
    cfg.adam_weight_decay = 0.05
    cfg.batch_size = 50
    cfg.n_train_iter = 100_000
    cfg.n_val_iter = 20_000

    # Separate names from all earlier experiments.
    cfg.tokenizer_save_folder_name = "finetune_tokenizer_base_robust"
    cfg.predictor_save_folder_name = "finetune_predictor_base_robust"
    cfg.finetuned_tokenizer_path = (
        f"{cfg.save_path}/{cfg.tokenizer_save_folder_name}/checkpoints/best_model"
    )
    cfg.finetuned_predictor_path = (
        f"{cfg.save_path}/{cfg.predictor_save_folder_name}/checkpoints/best_model"
    )
    return cfg


def _phase_paths(cfg: OfficialConfig, phase: Phase) -> dict[str, Path]:
    folder = (
        cfg.tokenizer_save_folder_name
        if phase == "tokenizer"
        else cfg.predictor_save_folder_name
    )
    base = Path(cfg.save_path) / folder
    return {
        "base": base,
        "best": base / "checkpoints" / "best_model",
        "latest": base / "checkpoints" / "latest_model",
        "state": base / "training_state.pt",
        "meta": base / "training_state.json",
        "summary": base / "summary.json",
    }


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_epochs": 0,
            "best_val_loss": math.inf,
            "bad_epochs": 0,
            "history": [],
            "stopped_early": False,
        }
    return json.loads(path.read_text())


def _loader(cfg: OfficialConfig, kind: str) -> tuple[OfficialKronosDataset, DataLoader]:
    ds = OfficialKronosDataset(kind, cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=(kind == "train"),
        pin_memory=True,
    )
    return ds, loader


def _optimizer(model: torch.nn.Module, cfg: OfficialConfig, phase: Phase):
    lr = (
        cfg.tokenizer_learning_rate
        if phase == "tokenizer"
        else cfg.predictor_learning_rate
    )
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
    )


def train_one_epoch(
    cfg: OfficialConfig,
    phase: Phase,
    device: str = "cuda",
    kronos_root: str | None = None,
    patience: int = 4,
    min_delta: float = 1e-4,
) -> dict:
    """Train exactly one epoch and atomically persist resumable state."""
    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosTokenizer

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    paths = _phase_paths(cfg, phase)
    paths["base"].mkdir(parents=True, exist_ok=True)
    meta = _load_meta(paths["meta"])
    if meta.get("stopped_early"):
        return {**meta, "phase": phase, "status": "already_stopped"}

    epoch = int(meta["completed_epochs"])
    train_ds, train_loader = _loader(cfg, "train")
    val_ds, val_loader = _loader(cfg, "val")
    train_ds.set_epoch_seed(epoch * 10_000)

    if phase == "tokenizer":
        legacy_crypto_tokenizer = (
            Path("/data/crypto/official_runs_base")
            / "models"
            / "finetune_tokenizer_official_base"
            / "checkpoints"
            / "best_model"
        )
        if paths["latest"].exists():
            source = str(paths["latest"])
        elif legacy_crypto_tokenizer.exists():
            # Preserve the useful first-stage work: it only saw train data in
            # gradient updates.  Re-evaluate it on the new disjoint validation
            # set before accepting it as the robust baseline.
            source = str(legacy_crypto_tokenizer)
        else:
            source = cfg.pretrained_tokenizer_path
        model = KronosTokenizer.from_pretrained(source).to(device)
        tokenizer = None
    else:
        source = (
            str(paths["latest"])
            if paths["latest"].exists()
            else cfg.pretrained_predictor_path
        )
        model = Kronos.from_pretrained(source).to(device)
        tokenizer = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path)
        tokenizer.eval().to(device)
        for param in tokenizer.parameters():
            param.requires_grad_(False)

    opt = _optimizer(model, cfg, phase)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    if paths["state"].exists():
        state = torch.load(paths["state"], map_location="cpu", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    def validate() -> float:
        # OfficialKronosDataset samples windows with an internal RNG. Reset it
        # so epoch-0 and every later epoch use the identical validation windows.
        val_ds.set_epoch_seed(0)
        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)
                if phase == "tokenizer":
                    zs, _, _, _ = model(batch_x)
                    _, z = zs
                    val_loss = F.mse_loss(z, batch_x)
                else:
                    assert tokenizer is not None
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                    logits = model(
                        token_seq_0[:, :-1],
                        token_seq_1[:, :-1],
                        batch_stamp[:, :-1, :],
                    )
                    val_loss, _, _ = model.head.compute_loss(
                        logits[0],
                        logits[1],
                        token_seq_0[:, 1:],
                        token_seq_1[:, 1:],
                    )
                val_sum += float(val_loss.item())
                val_n += 1
        return val_sum / max(val_n, 1)

    # Establish a deterministic epoch-0 control.  Fine-tuning is accepted only
    # when it beats this exact pretrained/warm-start model on the same fixed
    # disjoint validation samples.
    if int(meta["completed_epochs"]) == 0 and "initial_val_loss" not in meta:
        initial_val = validate()
        meta["initial_val_loss"] = initial_val
        meta["best_val_loss"] = initial_val
        meta["best_epoch"] = 0
        meta["source_model"] = source
        paths["best"].mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(paths["best"]))
        paths["meta"].write_text(json.dumps(meta, indent=2))
        print(
            f"[robust-{phase}] epoch=0 baseline_val={initial_val:.5f} "
            f"source={source}",
            flush=True,
        )

    model.train()
    train_sum = 0.0
    train_n = 0
    for i, (batch_x, batch_stamp) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_stamp = batch_stamp.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if phase == "tokenizer":
            zs, bsq_loss, _, _ = model(batch_x)
            z_pre, z = zs
            loss = (
                F.mse_loss(z_pre, batch_x)
                + F.mse_loss(z, batch_x)
                + bsq_loss
            ) / 2
            clip = 2.0
        else:
            assert tokenizer is not None
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            logits = model(
                token_seq_0[:, :-1],
                token_seq_1[:, :-1],
                batch_stamp[:, :-1, :],
            )
            loss, _, _ = model.head.compute_loss(
                logits[0],
                logits[1],
                token_seq_0[:, 1:],
                token_seq_1[:, 1:],
            )
            clip = 3.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        train_sum += float(loss.item())
        train_n += 1
        if (i + 1) % cfg.log_interval == 0:
            print(
                f"[robust-{phase}] epoch={epoch + 1} "
                f"step={i + 1}/{len(train_loader)} "
                f"loss={train_sum / train_n:.5f} lr={opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    train_loss = train_sum / max(train_n, 1)
    val_loss = validate()
    scheduler.step(val_loss)

    # Always persist latest for exact optimizer/model resume.
    paths["latest"].mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(paths["latest"]))

    improved = val_loss < (float(meta["best_val_loss"]) - min_delta)
    if improved:
        paths["best"].mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(paths["best"]))
        meta["best_val_loss"] = val_loss
        meta["best_epoch"] = epoch + 1
        meta["bad_epochs"] = 0
    else:
        meta["bad_epochs"] = int(meta["bad_epochs"]) + 1

    meta["completed_epochs"] = epoch + 1
    meta["stopped_early"] = int(meta["bad_epochs"]) >= patience
    meta["history"].append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": opt.param_groups[0]["lr"],
            "improved": improved,
        }
    )
    paths["meta"].write_text(json.dumps(meta, indent=2))
    torch.save(
        {
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
        },
        paths["state"],
    )
    paths["summary"].write_text(json.dumps(meta, indent=2))
    print(
        f"[robust-{phase}] epoch={epoch + 1} train={train_loss:.5f} "
        f"val={val_loss:.5f} best={meta['best_val_loss']:.5f} "
        f"bad={meta['bad_epochs']}/{patience} stopped={meta['stopped_early']}",
        flush=True,
    )
    return {**meta, "phase": phase, "status": "completed_epoch"}
