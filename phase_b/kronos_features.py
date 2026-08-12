"""Kronos frozen-feature extraction (paths kept, not averaged)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


KRONOS_FEATURE_COLS = [
    "kr_mu_h7",
    "kr_sigma_h7",
    "kr_p_up_h7",
    "kr_q10_h7",
    "kr_q90_h7",
    "kr_skew_h7",
    "kr_vol_h7",
    "kr_conv_h7",
    "kr_mu_h10",
    "kr_sigma_h10",
    "kr_p_up_h10",
    "kr_q10_h10",
    "kr_q90_h10",
    "kr_skew_h10",
    "kr_vol_h10",
    "kr_conv_h10",
    "kr_nll",
]


def project_budget(
    n_rows: int,
    n_paths: int,
    context: int,
    sec_per_row: float | None = None,
    max_gpu_hours: float = 30.0,
) -> dict:
    """Project GPU-hours; defaults scale roughly with paths * context."""
    if sec_per_row is None:
        # A10G Kronos-small, parallel path expand, pred_len=10, batch≈8
        sec_per_row = 0.022 * n_paths * (context / 300.0)
    gpu_hours = n_rows * sec_per_row / 3600.0
    return {
        "n_rows": int(n_rows),
        "n_paths": int(n_paths),
        "context": int(context),
        "sec_per_row_assumed": float(sec_per_row),
        "gpu_hours": float(gpu_hours),
        "over_budget": bool(gpu_hours > max_gpu_hours),
        "max_gpu_hours": float(max_gpu_hours),
    }


def choose_budget_settings(
    n_rows: int,
    n_paths: int,
    context: int,
    n_paths_fallback: int,
    context_fallback: int,
    sec_per_row: float | None = None,
    max_gpu_hours: float = 30.0,
) -> dict:
    """Apply hard budget guard: abort projection >30h then reduce N/context and re-project."""
    steps = []
    cur_n, cur_c = int(n_paths), int(context)
    proj = project_budget(n_rows, cur_n, cur_c, sec_per_row=sec_per_row, max_gpu_hours=max_gpu_hours)
    steps.append({"stage": "initial", **proj})
    if proj["over_budget"]:
        cur_n = int(n_paths_fallback)
        proj = project_budget(n_rows, cur_n, cur_c, sec_per_row=sec_per_row, max_gpu_hours=max_gpu_hours)
        steps.append({"stage": "reduce_n_paths", **proj})
    if proj["over_budget"]:
        cur_c = int(context_fallback)
        proj = project_budget(n_rows, cur_n, cur_c, sec_per_row=sec_per_row, max_gpu_hours=max_gpu_hours)
        steps.append({"stage": "reduce_context", **proj})
    if proj["over_budget"]:
        # last resort: both fallbacks already applied; still over → fail
        return {
            "ok": False,
            "n_paths": cur_n,
            "context": cur_c,
            "projection": proj,
            "steps": steps,
            "abort_reason": f"projected {proj['gpu_hours']:.2f} GPU-h > {max_gpu_hours}",
        }
    return {
        "ok": True,
        "n_paths": cur_n,
        "context": cur_c,
        "projection": proj,
        "steps": steps,
        "abort_reason": None,
    }


def _calc_time_stamps(x_timestamp: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "minute": x_timestamp.dt.minute,
            "hour": x_timestamp.dt.hour,
            "weekday": x_timestamp.dt.weekday,
            "day": x_timestamp.dt.day,
            "month": x_timestamp.dt.month,
        }
    )


def auto_regressive_inference_paths(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.9,
    sample_count=10,
):
    """Like upstream auto_regressive_inference but returns all paths (no mean)."""
    from phase_b.vendor.kronos_model.kronos import sample_from_logits

    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        dtype = next(model.parameters()).dtype
        x = x.to(dtype=torch.float32)
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        use_autocast = device.type == "cuda" and dtype == torch.bfloat16
        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                    s1_logits = s1_logits[:, -1, :].float()
                    sample_pre = sample_from_logits(
                        s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
                    )
                    s2_logits = model.decode_s2(context, sample_pre)
                    s2_logits = s2_logits[:, -1, :].float()
                    sample_post = sample_from_logits(
                        s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
                    )
            else:
                s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1, :]
                sample_pre = sample_from_logits(
                    s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
                )
                s2_logits = model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1, :]
                sample_post = sample_from_logits(
                    s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
                )
            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        return z.cpu().numpy()  # [B, sample_count, T, F]


def observed_token_nll(tokenizer, model, x_norm: np.ndarray, x_stamp: np.ndarray, n_last: int = 30) -> float:
    """Mean per-token CE NLL of the last n_last observed bars (teacher forcing)."""
    with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(device)
        stamp = torch.from_numpy(x_stamp.astype(np.float32)).unsqueeze(0).to(device)
        tokens = tokenizer.encode(x, half=True)
        s1, s2 = tokens[0], tokens[1]
        L = s1.size(1)
        n = min(n_last, L - 1)
        if n < 5:
            return float("nan")
        s1_in = s1[:, -(n + 1) : -1]
        s2_in = s2[:, -(n + 1) : -1]
        stamp_in = stamp[:, -(n + 1) : -1, :]
        s1_tgt = s1[:, -n:]
        s2_tgt = s2[:, -n:]
        try:
            s1_logits, s2_logits = model(
                s1_in, s2_in, stamp=stamp_in, use_teacher_forcing=True, s1_targets=s1_tgt
            )
            ce1 = F.cross_entropy(
                s1_logits.reshape(-1, s1_logits.size(-1)).float(),
                s1_tgt.reshape(-1),
                reduction="mean",
            )
            ce2 = F.cross_entropy(
                s2_logits.reshape(-1, s2_logits.size(-1)).float(),
                s2_tgt.reshape(-1),
                reduction="mean",
            )
            return float(((ce1 + ce2) / 2).item())
        except Exception:
            return float("nan")


def _horizon_stats(path_closes: np.ndarray, last_close: float, h: int) -> dict:
    """path_closes: [N, pred_len] absolute closes for forecast steps."""
    term = path_closes[:, h - 1]
    logret = np.log(np.clip(term, 1e-12, None) / max(last_close, 1e-12))
    full = np.concatenate([np.full((path_closes.shape[0], 1), last_close), path_closes[:, :h]], axis=1)
    step_lr = np.diff(np.log(np.clip(full, 1e-12, None)), axis=1)
    per_path_vol = step_lr.std(axis=1)
    mu = float(np.mean(logret))
    sigma = float(np.std(logret)) + 1e-12
    skew = float(pd.Series(logret).skew()) if len(logret) > 2 else 0.0
    if not np.isfinite(skew):
        skew = 0.0
    return {
        f"kr_mu_h{h}": mu,
        f"kr_sigma_h{h}": sigma,
        f"kr_p_up_h{h}": float(np.mean(logret > 0)),
        f"kr_q10_h{h}": float(np.quantile(logret, 0.10)),
        f"kr_q90_h{h}": float(np.quantile(logret, 0.90)),
        f"kr_skew_h{h}": skew,
        f"kr_vol_h{h}": float(np.mean(per_path_vol)),
        f"kr_conv_h{h}": float(mu / sigma),
    }


def _nan_features() -> dict[str, float]:
    return {c: float("nan") for c in KRONOS_FEATURE_COLS}


class KronosFeatureExtractor:
    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-small",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cuda:0",
        max_context: int = 512,
        context: int = 400,
        min_context: int = 200,
        n_paths: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        bf16: bool = True,
        model=None,
        tokenizer=None,
    ):
        from phase_b.vendor.kronos_model import Kronos, KronosTokenizer

        self.context = context
        self.min_context = min_context
        self.n_paths = n_paths
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_context = max_context
        self.device = device
        if tokenizer is None:
            self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = Kronos.from_pretrained(model_id)
        else:
            self.model = model
        self.tokenizer = self.tokenizer.to(device)
        self.model = self.model.to(device)
        if bf16 and str(device).startswith("cuda"):
            self.model = self.model.bfloat16()
        self.model.eval()
        self.clip = 5

    def _prepare_ctx(self, df_hist: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
        """Return (tensors_dict, None) or (None, nan_features)."""
        if len(df_hist) < self.min_context:
            return None, _nan_features()
        ctx = df_hist.tail(self.context).copy()
        if "amount" not in ctx.columns:
            ctx["amount"] = ctx["volume"] * ctx[["open", "high", "low", "close"]].mean(axis=1)
        x_timestamp = pd.to_datetime(ctx["date"], utc=True)
        last = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(
            pd.date_range(last + pd.Timedelta(days=1), periods=10, freq="D", tz="UTC")
        )
        price_cols = ["open", "high", "low", "close"]
        x_raw = ctx[price_cols + ["volume", "amount"]].values.astype(np.float32)
        x_mean, x_std = x_raw.mean(axis=0), x_raw.std(axis=0)
        x_norm = np.clip((x_raw - x_mean) / (x_std + 1e-5), -self.clip, self.clip)
        x_stamp = _calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = _calc_time_stamps(y_timestamp).values.astype(np.float32)
        return {
            "x_norm": x_norm,
            "x_stamp": x_stamp,
            "y_stamp": y_stamp,
            "x_mean": x_mean,
            "x_std": x_std,
            "last_close": float(ctx["close"].iloc[-1]),
            "seq_len": int(len(ctx)),
        }, None

    def features_for_row(self, df_hist: pd.DataFrame) -> dict[str, float]:
        prep, nan_out = self._prepare_ctx(df_hist)
        if nan_out is not None:
            return nan_out
        assert prep is not None
        return self._features_from_prep_batch([prep])[0]

    def features_for_batch(self, hist_list: list[pd.DataFrame]) -> list[dict[str, float]]:
        """Batch rows that share the same context length; variable lengths processed in groups."""
        out: list[dict[str, float] | None] = [None] * len(hist_list)
        groups: dict[int, list[tuple[int, dict]]] = {}
        for i, df_hist in enumerate(hist_list):
            prep, nan_out = self._prepare_ctx(df_hist)
            if nan_out is not None:
                out[i] = nan_out
                continue
            assert prep is not None
            groups.setdefault(prep["seq_len"], []).append((i, prep))
        for _L, items in groups.items():
            idxs = [i for i, _ in items]
            preps = [p for _, p in items]
            feats = self._features_from_prep_batch(preps)
            for i, f in zip(idxs, feats):
                out[i] = f
        return [o if o is not None else _nan_features() for o in out]

    def _features_from_prep_batch(self, preps: list[dict]) -> list[dict[str, float]]:
        if not preps:
            return []
        device = self.device
        x = torch.from_numpy(np.stack([p["x_norm"] for p in preps], axis=0)).to(device)
        xs = torch.from_numpy(np.stack([p["x_stamp"] for p in preps], axis=0)).to(device)
        ys = torch.from_numpy(np.stack([p["y_stamp"] for p in preps], axis=0)).to(device)
        preds = auto_regressive_inference_paths(
            self.tokenizer,
            self.model,
            x,
            xs,
            ys,
            self.max_context,
            pred_len=10,
            clip=self.clip,
            T=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            sample_count=self.n_paths,
        )  # [B, N, T_full, F]
        close_idx = 3
        results = []
        for bi, p in enumerate(preps):
            fut = preds[bi, :, -10:, :]
            close_paths = fut[:, :, close_idx] * (p["x_std"][close_idx] + 1e-5) + p["x_mean"][close_idx]
            row = {}
            row.update(_horizon_stats(close_paths, p["last_close"], 7))
            row.update(_horizon_stats(close_paths, p["last_close"], 10))
            row["kr_nll"] = observed_token_nll(
                self.tokenizer, self.model, p["x_norm"], p["x_stamp"], n_last=30
            )
            results.append(row)
        return results


def extract_symbol_features(
    panel_sym: pd.DataFrame,
    dates: list[pd.Timestamp],
    extractor: KronosFeatureExtractor,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Extract Kronos features for one symbol at requested dates. panel_sym sorted by date."""
    panel_sym = panel_sym.sort_values("date").reset_index(drop=True)
    panel_sym["date"] = pd.to_datetime(panel_sym["date"], utc=True)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_sym["date"])}
    rows = []
    skipped_short = 0
    computed = 0
    t0 = time.time()
    pending_hist: list[pd.DataFrame] = []
    pending_meta: list[tuple[pd.Timestamp, str]] = []

    def _flush():
        nonlocal computed
        if not pending_hist:
            return
        feats = extractor.features_for_batch(pending_hist)
        for (dt, sym), feat in zip(pending_meta, feats):
            rec = {"date": dt, "symbol": sym, **feat}
            rows.append(rec)
            if all(not (isinstance(feat[c], float) and math.isnan(feat[c])) for c in ("kr_mu_h7", "kr_nll")):
                computed += 1
            elif np.isfinite(feat.get("kr_mu_h7", np.nan)):
                computed += 1
        pending_hist.clear()
        pending_meta.clear()

    symbol = str(panel_sym["symbol"].iloc[0]) if len(panel_sym) else ""
    for dt in dates:
        dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        if dt not in date_to_idx:
            skipped_short += 1
            continue
        end_i = date_to_idx[dt]
        start_i = max(0, end_i + 1 - extractor.context)
        hist = panel_sym.iloc[start_i : end_i + 1]
        if len(hist) < extractor.min_context:
            skipped_short += 1
            rows.append({"date": dt, "symbol": symbol, **_nan_features()})
            continue
        pending_hist.append(hist)
        pending_meta.append((dt, symbol))
        if len(pending_hist) >= batch_size:
            _flush()
    _flush()
    out = pd.DataFrame(rows)
    stats = {
        "symbol": symbol,
        "n_requested": len(dates),
        "n_rows": int(len(out)),
        "n_computed": int(computed),
        "n_skipped_short": int(skipped_short),
        "elapsed_sec": float(time.time() - t0),
    }
    return out, stats
