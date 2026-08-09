"""Plot equity comparison: raw vs meta (zero-shot / finetuned)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _equity(steps: list[dict]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    dates = pd.to_datetime([s["asof"] for s in steps])
    rets = np.array([s["strategy_return"] for s in steps], dtype=float)
    eq = np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    x = pd.to_datetime([dates[0] - pd.Timedelta(days=5)] + list(dates))
    return x, eq


def plot_improve(
    path: Path | str = Path(__file__).with_name("last_improve.json"),
    out_path: Path | str | None = None,
) -> Path:
    data = json.loads(Path(path).read_text())
    out = Path(out_path) if out_path else Path(__file__).with_name("improve_equity.png")

    series = {
        "ZS raw": data["zeroshot"]["compare"]["raw_aligned"]["steps"],
        "ZS meta": data["zeroshot"]["compare"]["meta"]["steps"],
        "FT raw": data["finetuned"]["compare"]["raw_aligned"]["steps"],
        "FT meta": data["finetuned"]["compare"]["meta"]["steps"],
    }
    colors = {
        "ZS raw": "#8b949e",
        "ZS meta": "#1f6feb",
        "FT raw": "#bf8700",
        "FT meta": "#1a7f37",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, steps in series.items():
        if not steps:
            continue
        x, eq = _equity(steps)
        ret = eq[-1] - 1.0
        ax.plot(x, eq, lw=2.0, color=colors[name], label=f"{name} ({ret:+.1%})")
    ax.axhline(1.0, color="#d0d7de", lw=1)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title("Improve backtest: raw Kronos rule vs meta-model (zero-shot & fine-tuned)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Prefer improve JSON; fall back to local zero-shot compare only.
    improve = Path(__file__).with_name("last_improve.json")
    if improve.exists():
        print(plot_improve())
    else:
        print("last_improve.json not found; run modal --mode improve first")
