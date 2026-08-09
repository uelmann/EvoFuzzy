"""Plot annual-retrain meta equity and per-year bars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_annual(
    path: Path | str = Path(__file__).with_name("last_long_annual.json"),
    out_path: Path | str | None = None,
) -> Path:
    data = json.loads(Path(path).read_text())
    out = Path(out_path) if out_path else Path(__file__).with_name("long_annual_equity.png")
    steps = data["steps"]
    dates = pd.to_datetime([s["asof"] for s in steps])
    rets = np.array([s["strategy_return"] for s in steps], dtype=float)
    bh_rets = np.array([s["realized_return"] for s in steps], dtype=float)
    eq = np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    eq_bh = np.cumprod(np.concatenate([[1.0], 1.0 + bh_rets]))
    x = pd.to_datetime([dates[0] - pd.Timedelta(days=5)] + list(dates))

    years = [y["year"] for y in data["by_year"]]
    y_ret = [y["total_return"] * 100 for y in data["by_year"]]
    y_bh = [y["buy_hold_return"] * 100 for y in data["by_year"]]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2.0, 1.1]})
    ax = axes[0]
    ov = data["overall"]
    ax.plot(x, eq, color="#1f6feb", lw=2.2, label=f"Annual-retrain meta ({ov['total_return']:+.1%})")
    ax.plot(x, eq_bh, color="#8b949e", lw=1.6, ls="--", label=f"Buy&hold rolls ({ov['buy_hold_return']:+.1%})")
    ax.axhline(1.0, color="#d0d7de", lw=1)
    ax.set_title(
        f"Binance BTCUSDT daily — annual meta retrain\n"
        f"{ov['start'][:10]} → {ov['end'][:10]}  hit={ov['hit_rate']}  maxDD={ov['max_drawdown']:.1%}"
    )
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax2 = axes[1]
    xpos = np.arange(len(years))
    w = 0.38
    ax2.bar(xpos - w / 2, y_ret, width=w, color="#1f6feb", label="Meta")
    ax2.bar(xpos + w / 2, y_bh, width=w, color="#8b949e", label="B&H")
    ax2.axhline(0, color="#d0d7de", lw=1)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([str(y) for y in years])
    ax2.set_ylabel("Year return %")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_annual())
