"""CPU tests for official pickle prep + TopkDropout engine."""

from __future__ import annotations

from pathlib import Path

from kronos_signal.official_config import OfficialConfig
from kronos_signal.official_topk_bt import panels_from_full, roc_scores, topk_dropout_backtest
from kronos_signal.panel_data import DEFAULT_CSV
from kronos_signal.prepare_official_pickles import prepare_pickles


def test_prepare_and_topk():
    assert DEFAULT_CSV.exists()
    root = Path("/tmp/kronos_official_test")
    cfg = OfficialConfig(root=root, predictor_size="small", epochs=1)
    meta = prepare_pickles(DEFAULT_CSV, cfg)
    assert meta["n_train_symbols"] >= 10
    import pickle

    with open(Path(cfg.dataset_path) / "full_panel.pkl", "rb") as f:
        data = pickle.load(f)
    panels = panels_from_full(data)
    scores = roc_scores(panels["close"], window=10)
    bt = topk_dropout_backtest(scores, panels["close"], cfg, panels["marketCap"])
    assert bt["n_days"] > 50
    # long-only: weights never negative
    assert (bt["weights"].min().min() >= -1e-12)


if __name__ == "__main__":
    test_prepare_and_topk()
    print("ok")
