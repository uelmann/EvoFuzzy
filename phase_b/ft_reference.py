"""Fine-tuned Kronos head-to-head — CONTAMINATED REFERENCE only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import evaluate_predictions


def try_load_ft_extractor(
    predictor_dir: str | Path,
    tokenizer_dir: str | Path,
    device: str = "cuda:0",
    context: int = 90,
    n_paths: int = 1,
):
    """Load local FT Kronos-base checkpoint. Returns extractor or raises."""
    from phase_b.vendor.kronos_model import Kronos, KronosTokenizer
    from phase_b.kronos_features import KronosFeatureExtractor

    pred_dir = Path(predictor_dir)
    tok_dir = Path(tokenizer_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"FT predictor missing: {pred_dir}")
    # tokenizer may live under checkpoints/best_model
    if not tok_dir.exists():
        alt = tok_dir.parent.parent / "checkpoints" / "best_model"
        if alt.exists():
            tok_dir = alt
        else:
            # fall back to HF base tokenizer matching FT source
            tok_dir = None

    model = Kronos.from_pretrained(str(pred_dir))
    if tok_dir is not None and Path(tok_dir).exists():
        tokenizer = KronosTokenizer.from_pretrained(str(tok_dir))
    else:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    return KronosFeatureExtractor(
        model_id=str(pred_dir),
        tokenizer_id=str(tok_dir) if tok_dir else "NeoQuasar/Kronos-Tokenizer-base",
        device=device,
        context=context,
        min_context=min(50, context),
        n_paths=n_paths,
        bf16=True,
        model=model,
        tokenizer=tokenizer,
    )


def regenerate_ft_scores(
    panel: pd.DataFrame,
    oos_keys: pd.DataFrame,
    extractor,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    For each OOS (date, symbol), score = kr_mu_h10 from FT model.
    CONTAMINATED — FT trained on full-sample data.
    """
    from phase_b.kronos_features import extract_symbol_features

    keys = oos_keys[["date", "symbol"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"], utc=True)
    rows = []
    stats = []
    for sym, g in keys.groupby("symbol"):
        dates = sorted(pd.to_datetime(g["date"], utc=True).unique())
        panel_sym = panel[panel["symbol"] == sym].copy()
        if panel_sym.empty:
            continue
        feat_df, st = extract_symbol_features(panel_sym, list(dates), extractor, batch_size=batch_size)
        stats.append(st)
        if feat_df.empty:
            continue
        tmp = feat_df[["date", "symbol", "kr_mu_h10"]].rename(columns={"kr_mu_h10": "score"})
        tmp["horizon"] = 10
        tmp["model_name"] = "kronos_ft_contaminated"
        rows.append(tmp)
        print(f"[ft-ref] {sym} n={len(tmp)} elapsed={st['elapsed_sec']:.1f}s", flush=True)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["score"])
    return out


def evaluate_ft_reference(
    ft_preds: pd.DataFrame,
    feat: pd.DataFrame,
    pit20: pd.DataFrame,
    pit120: pd.DataFrame,
) -> dict:
    if ft_preds is None or ft_preds.empty:
        return {"status": "unavailable", "reason": "empty predictions"}
    ycol = "y_h10"
    df = ft_preds.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if ycol not in df.columns:
        df = df.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    cutoff = pd.Timestamp("2025-08-17", tz="UTC")
    out = {"status": "ok", "label": "CONTAMINATED REFERENCE — fine-tuned on full-sample data"}
    for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
        for window, mask in (
            ("full", df["date"].notna()),
            ("pre", df["date"] < cutoff),
            ("post", df["date"] >= cutoff),
        ):
            sub = df.loc[mask].copy()
            if sub.empty:
                out[f"{uni_name}_{window}"] = {"mean_ic": float("nan"), "n_days": 0}
                continue
            ev = evaluate_predictions(sub, 10, universe=uni, label=uni_name)
            out[f"{uni_name}_{window}"] = {
                k: v for k, v in ev.items() if k != "ic_series"
            }
    return out


def run_ft_reference_safe(
    panel: pd.DataFrame,
    oos_pred_h10: pd.DataFrame,
    feat: pd.DataFrame,
    pit20: pd.DataFrame,
    pit120: pd.DataFrame,
    out_path: Path,
    predictor_dir: str,
    tokenizer_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Non-blocking FT reference. On failure, return status and continue."""
    try:
        # Prefer regenerating; if GPU/model missing, fall back to note
        extractor = try_load_ft_extractor(predictor_dir, tokenizer_dir, device=device)
        # Limit to symbols present in OOS preds to bound cost
        keys = oos_pred_h10[["date", "symbol"]].drop_duplicates()
        # Further downsample if huge: keep all — OOS ~170k is heavy for Kronos-base;
        # use top20 PIT intersection only for reference speed
        uni = pit20.copy()
        uni["date"] = pd.to_datetime(uni["date"], utc=True)
        keys["date"] = pd.to_datetime(keys["date"], utc=True)
        keys = keys.merge(uni[["date", "symbol"]], on=["date", "symbol"], how="inner")
        print(f"[ft-ref] regenerating for {len(keys)} top20 OOS rows (CONTAMINATED)", flush=True)
        ft_preds = regenerate_ft_scores(panel, keys, extractor, batch_size=4)
        if ft_preds.empty:
            return {"status": "unavailable", "reason": "regeneration produced empty frame"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ft_preds.to_parquet(out_path, index=False)
        ev = evaluate_ft_reference(ft_preds, feat, pit20, pit120)
        ev["n_rows"] = int(len(ft_preds))
        ev["out"] = str(out_path)
        return ev
    except Exception as e:
        return {
            "status": "unavailable",
            "reason": str(e),
            "label": "CONTAMINATED REFERENCE — checkpoint unusable; skipped",
        }
