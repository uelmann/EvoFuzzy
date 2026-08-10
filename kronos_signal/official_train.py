"""Single-GPU official Kronos FT: tokenizer then predictor (no DDP/torchrun)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .forecast import _ensure_kronos_on_path
from .official_config import OfficialConfig
from .official_dataset import OfficialKronosDataset


def _format_time(seconds: float) -> str:
    return str(__import__("datetime").timedelta(seconds=int(seconds)))


def train_tokenizer(cfg: OfficialConfig, device: str = "cuda", kronos_root: str | None = None) -> dict:
    _ensure_kronos_on_path(kronos_root)
    from model import KronosTokenizer

    save_dir = Path(cfg.save_path) / cfg.tokenizer_save_folder_name
    ckpt_dir = save_dir / "checkpoints" / "best_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OfficialKronosDataset("train", cfg)
    val_ds = OfficialKronosDataset("val", cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False
    )

    model = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.tokenizer_learning_rate, weight_decay=cfg.adam_weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.tokenizer_learning_rate,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=cfg.epochs,
        pct_start=0.03,
        div_factor=10,
    )

    best = float("inf")
    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)
        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.to(device)
            opt.zero_grad()
            total_loss = 0.0
            for j in range(cfg.accumulation_steps):
                bs = ori_batch_x.shape[0] // cfg.accumulation_steps
                batch_x = ori_batch_x[j * bs : (j + 1) * bs]
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs
                recon_pre = F.mse_loss(z_pre, batch_x)
                recon_all = F.mse_loss(z, batch_x)
                loss = ((recon_pre + recon_all) + bsq_loss) / 2
                (loss / cfg.accumulation_steps).backward()
                total_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            if (step % cfg.log_interval) == 0:
                print(
                    f"[tok] epoch {epoch+1}/{cfg.epochs} step {i+1}/{len(train_loader)} "
                    f"loss={total_loss/cfg.accumulation_steps:.4f}",
                    flush=True,
                )

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                zs, _, _, _ = model(batch_x)
                _, z = zs
                val_sum += F.mse_loss(z, batch_x).item() * batch_x.size(0)
                val_n += batch_x.size(0)
        avg = val_sum / max(val_n, 1)
        print(f"[tok] epoch {epoch+1} val={avg:.4f} elapsed={_format_time(time.time()-t0)}", flush=True)
        if avg < best:
            best = avg
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"[tok] saved best → {ckpt_dir}", flush=True)

    summary = {"best_val_loss": best, "epochs": cfg.epochs, "path": str(ckpt_dir)}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_predictor(cfg: OfficialConfig, device: str = "cuda", kronos_root: str | None = None) -> dict:
    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosTokenizer

    save_dir = Path(cfg.save_path) / cfg.predictor_save_folder_name
    ckpt_dir = save_dir / "checkpoints" / "best_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OfficialKronosDataset("train", cfg)
    val_ds = OfficialKronosDataset("val", cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False
    )

    tokenizer = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path)
    tokenizer.eval().to(device)
    model = Kronos.from_pretrained(cfg.pretrained_predictor_path).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.predictor_learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.predictor_learning_rate,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=cfg.epochs,
        pct_start=0.03,
        div_factor=10,
    )

    best = float("inf")
    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)
        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            step += 1
            if (step % cfg.log_interval) == 0:
                print(
                    f"[pred] epoch {epoch+1}/{cfg.epochs} step {i+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} s1={s1_loss.item():.4f} s2={s2_loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device)
                batch_x_stamp = batch_x_stamp.to(device)
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_sum += float(val_loss.item())
                val_n += 1
        avg = val_sum / max(val_n, 1)
        print(f"[pred] epoch {epoch+1} val={avg:.4f} elapsed={_format_time(time.time()-t0)}", flush=True)
        if avg < best:
            best = avg
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"[pred] saved best → {ckpt_dir}", flush=True)
        # Persist progress marker each epoch (Modal volume commit is done by caller)
        (save_dir / "last_epoch.json").write_text(
            json.dumps({"epoch": epoch + 1, "val": avg, "best": best}, indent=2)
        )

    summary = {"best_val_loss": best, "epochs": cfg.epochs, "path": str(ckpt_dir)}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_official_finetune(
    cfg: OfficialConfig,
    device: str | None = None,
    kronos_root: str | None = None,
    skip_tokenizer: bool = False,
    skip_predictor: bool = False,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"device": device, "pretrained_predictor": cfg.pretrained_predictor_path}
    if not skip_tokenizer:
        out["tokenizer"] = train_tokenizer(cfg, device=device, kronos_root=kronos_root)
    if not skip_predictor:
        out["predictor"] = train_predictor(cfg, device=device, kronos_root=kronos_root)
    return out
