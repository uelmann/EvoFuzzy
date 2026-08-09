"""Plot equity comparison across improve / improve_v2 results."""

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
    return _plot_series(series, colors, out, "Improve v1: raw vs meta (zero-shot & AR fine-tune)")


def plot_improve_v2(
    path: Path | str = Path(__file__).with_name("last_improve_v2.json"),
    out_path: Path | str | None = None,
) -> Path:
    data = json.loads(Path(path).read_text())
    out = Path(out_path) if out_path else Path(__file__).with_name("improve_v2_equity.png")
    series = {
        "Raw": data["meta_v2"]["raw_aligned"]["steps"],
        "Meta logistic": data["meta_v2"]["meta_logistic"]["steps"],
        "Meta embargo": data["meta_v2"]["meta_logistic_embargo"]["steps"],
        "Market-only": data["meta_v2"]["meta_market_only"]["steps"],
        "Sup head": data["supervised"]["head_alone"]["steps"],
        "Meta+sup": data["supervised"]["meta_with_sup"]["steps"],
    }
    colors = {
        "Raw": "#8b949e",
        "Meta logistic": "#1f6feb",
        "Meta embargo": "#9a6700",
        "Market-only": "#cf222e",
        "Sup head": "#8250df",
        "Meta+sup": "#1a7f37",
    }
    return _plot_series(
        series,
        colors,
        out,
        "Improve v2: logistic meta + supervised Kronos direction head",
    )


def _plot_series(series, colors, out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, steps in series.items():
        if not steps:
            continue
        x, eq = _equity(steps)
        ret = eq[-1] - 1.0
        ax.plot(x, eq, lw=2.0, color=colors.get(name, None), label=f"{name} ({ret:+.1%})")
    ax.axhline(1.0, color="#d0d7de", lw=1)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    v2 = Path(__file__).with_name("last_improve_v2.json")
    v1 = Path(__file__).with_name("last_improve.json")
    if v2.exists():
        print(plot_improve_v2())
    elif v1.exists():
        print(plot_improve())
    else:
        print("No improve JSON found")
