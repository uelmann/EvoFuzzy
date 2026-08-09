"""Plot equity curve from last_backtest.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_backtest(
    json_path: Path | str = Path(__file__).with_name("last_backtest.json"),
    out_path: Path | str | None = None,
) -> Path:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    steps = data["steps"]

    dates = pd.to_datetime([s["asof"] for s in steps])
    strat_rets = np.array([s["strategy_return"] for s in steps], dtype=float)
    realized = np.array([s["realized_return"] for s in steps], dtype=float)
    signals = [s["signal"] for s in steps]
    correct = [s["correct"] for s in steps]

    eq_strat = np.cumprod(np.concatenate([[1.0], 1.0 + strat_rets]))
    eq_bh = np.cumprod(np.concatenate([[1.0], 1.0 + realized]))
    x = pd.to_datetime([dates[0] - pd.Timedelta(days=data["pred_len"])] + list(dates))

    out = Path(out_path) if out_path else Path(__file__).with_name("backtest_equity.png")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True
    )

    ax = axes[0]
    ax.plot(x, eq_strat, color="#1f6feb", lw=2.2, label="Strategy (L/H/S)")
    ax.plot(x, eq_bh, color="#8b949e", lw=1.8, ls="--", label="Buy & hold (5d rolls)")
    ax.axhline(1.0, color="#d0d7de", lw=1)
    ax.fill_between(x, eq_strat, 1.0, where=eq_strat >= 1.0, color="#1f6feb", alpha=0.08)
    ax.fill_between(x, eq_strat, 1.0, where=eq_strat < 1.0, color="#cf222e", alpha=0.08)
    ax.set_ylabel("Equity (start = 1.0)")
    hr = data["hit_rate"]
    ax.set_title(
        f"Kronos-base BTCUSDT daily walk-forward\n"
        f"{data['start'][:10]} → {data['end'][:10]}  |  "
        f"hit={hr:.1%}  strat={data['total_return']:.1%}  "
        f"B&H={data['buy_hold_return']:.1%}  maxDD={data['max_drawdown']:.1%}"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = []
    for s, c in zip(signals, correct):
        if s == "HOLD":
            colors.append("#8b949e")
        elif c is True:
            colors.append("#1a7f37")
        else:
            colors.append("#cf222e")
    ax2.bar(dates, strat_rets * 100, color=colors, width=3.5, alpha=0.85)
    ax2.axhline(0, color="#d0d7de", lw=1)
    ax2.set_ylabel("Step return %")
    ax2.set_xlabel("Decision date")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(
        handles=[
            Patch(color="#1a7f37", label="Active correct"),
            Patch(color="#cf222e", label="Active wrong"),
            Patch(color="#8b949e", label="HOLD"),
        ],
        loc="upper right",
        frameon=False,
        ncol=3,
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_diagnostics(
    json_path: Path | str = Path(__file__).with_name("last_backtest.json"),
    out_path: Path | str | None = None,
) -> Path:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    steps = data["steps"]
    pred = np.array([s["mean_return"] for s in steps], dtype=float) * 100
    real = np.array([s["realized_return"] for s in steps], dtype=float) * 100
    p_up = np.array([s["p_up"] for s in steps], dtype=float)

    out = Path(out_path) if out_path else Path(__file__).with_name("backtest_diagnostics.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(pred, real, c="#1f6feb", alpha=0.65, edgecolors="none", s=36)
    lim = max(5, np.percentile(np.abs(np.concatenate([pred, real])), 95))
    ax.plot([-lim, lim], [-lim, lim], color="#8b949e", ls="--", lw=1, label="y = x")
    ax.axhline(0, color="#d0d7de", lw=1)
    ax.axvline(0, color="#d0d7de", lw=1)
    ax.set_xlabel("Predicted mean return %")
    ax.set_ylabel("Realized return %")
    corr = np.corrcoef(pred, real)[0, 1] if len(pred) > 1 else float("nan")
    ax.set_title(f"Pred vs realized (corr={corr:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax2 = axes[1]
    ax2.hist(p_up, bins=np.linspace(0, 1, 11), color="#1f6feb", alpha=0.85, edgecolor="white")
    ax2.axvline(0.6, color="#1a7f37", ls="--", label="LONG thr 0.60")
    ax2.axvline(0.4, color="#cf222e", ls="--", label="SHORT thr 0.40")
    ax2.set_xlabel("p_up")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Upside-probability mass (mean={p_up.mean():.2%})")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(frameon=False)

    fig.suptitle("Why the signal fails: overconfident paths + ~zero ranking power", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_backtest())
    print(plot_diagnostics())
