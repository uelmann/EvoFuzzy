"""End-to-end official-style Kronos recipe on crypto panel."""

from __future__ import annotations

import json
from pathlib import Path

from .official_config import OfficialConfig
from .official_topk_bt import load_full_panel, panels_from_full, roc_scores, topk_dropout_backtest
from .official_train import run_official_finetune
from .prepare_official_pickles import prepare_pickles


def run_official_pipeline(
    csv_path: Path,
    root: Path,
    predictor_size: str = "small",
    epochs: int = 30,
    device: str | None = None,
    kronos_root: str | None = None,
    skip_train: bool = False,
    skip_tokenizer: bool = False,
    skip_predictor: bool = False,
) -> dict:
    cfg = OfficialConfig(root=root, predictor_size=predictor_size, epochs=epochs)
    meta = prepare_pickles(csv_path, cfg)

    train_info = None
    if not skip_train:
        train_info = run_official_finetune(
            cfg,
            device=device,
            kronos_root=kronos_root,
            skip_tokenizer=skip_tokenizer,
            skip_predictor=skip_predictor,
        )

    # Baseline TopkDropout with ROC scores (same engine as FT model will use)
    data = load_full_panel(cfg)
    panels = panels_from_full(data)
    scores = roc_scores(panels["close"], window=cfg.predict_window)
    bt = topk_dropout_backtest(
        scores, panels["close"], cfg, universe_mcap=panels["marketCap"]
    )
    summary = {
        "recipe": "official_kronos_mirror",
        "notes": {
            "market": "crypto_panel_not_csi300",
            "ft_start": cfg.pretrained_predictor_path,
            "lookback": cfg.lookback_window,
            "predict_window": cfg.predict_window,
            "topk": cfg.backtest_n_symbol_hold,
            "n_drop": cfg.backtest_n_symbol_drop,
            "hold_thresh": cfg.backtest_hold_thresh,
            "long_only": True,
            "not_long_short": True,
        },
        "dataset_meta": meta,
        "train": train_info,
        "topk_roc_baseline": {
            "total_return": bt["total_return"],
            "max_drawdown": bt["max_drawdown"],
            "sharpe": bt["sharpe"],
            "btc_total_return": bt["btc_total_return"],
            "n_days": bt["n_days"],
        },
        "paths": {
            "tokenizer": cfg.finetuned_tokenizer_path,
            "predictor": cfg.finetuned_predictor_path,
            "dataset": cfg.dataset_path,
        },
    }
    out = Path(root) / "last_official_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # strip non-json
    out.write_text(json.dumps(summary, indent=2, default=str))
    return summary
