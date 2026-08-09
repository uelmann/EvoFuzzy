"""Official Kronos finetune config — mirrored from Kronos/finetune/config.py.

Market: crypto panel (CMC/KuCoin) instead of Qlib CSI300, same training recipe.
Backtest: long-only TopkDropout scaled to universe_n=30 (topk=5, n_drop=1).
Start checkpoint: Kronos-small (exact demo default). Use predictor_size=base to override.
"""

from __future__ import annotations

from pathlib import Path


class OfficialConfig:
    def __init__(
        self,
        root: str | Path | None = None,
        predictor_size: str = "small",
        epochs: int | None = None,
    ):
        root = Path(root) if root else Path(__file__).resolve().parent / "official_runs"
        self.root = root

        # --- Data (same windows as upstream demo) ---
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512
        self.feature_list = ["open", "high", "low", "close", "vol", "amt"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        # Crypto calendar splits mirroring their train/val/test structure
        # Extended to 2016 to match full KuCoin/CMC history download.
        self.dataset_begin_time = "2016-01-01"
        self.dataset_end_time = "2026-08-08"
        self.train_time_range = ["2016-01-01", "2022-12-31"]
        self.val_time_range = ["2022-09-01", "2024-06-30"]
        self.test_time_range = ["2024-04-01", "2026-08-08"]
        self.backtest_time_range = ["2024-07-01", "2026-08-08"]

        self.dataset_path = str(root / "processed_datasets")
        self.universe_n = 30  # point-in-time top-N by mcap for BT (training uses all pickles)

        # --- Training (upstream defaults) ---
        self.clip = 5.0
        self.epochs = 30 if epochs is None else epochs
        self.log_interval = 100
        self.batch_size = 50
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.seed = 100
        self.use_comet = False
        self.num_workers = 0

        self.save_path = str(root / "models")
        # Keep legacy folder names for small so existing ckpts still resolve;
        # base (and others) get a size suffix to avoid overwriting.
        if predictor_size == "small":
            self.tokenizer_save_folder_name = "finetune_tokenizer_official"
            self.predictor_save_folder_name = "finetune_predictor_official"
        else:
            self.tokenizer_save_folder_name = f"finetune_tokenizer_official_{predictor_size}"
            self.predictor_save_folder_name = f"finetune_predictor_official_{predictor_size}"

        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        if predictor_size == "base":
            self.pretrained_predictor_path = "NeoQuasar/Kronos-base"
        else:
            # Exact demo default
            self.pretrained_predictor_path = "NeoQuasar/Kronos-small"

        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # --- Backtest (scaled TopkDropout) ---
        # Upstream CSI300: hold 50 / drop 5 / hold_thresh 5 on ~300 names.
        # Scale ≈ hold 5 / drop 1 on 30 names (~same fractions).
        self.backtest_n_symbol_hold = 5
        self.backtest_n_symbol_drop = 1
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 64
        self.open_cost = 0.001
        self.close_cost = 0.0015

    def as_dict(self) -> dict:
        return dict(self.__dict__)
