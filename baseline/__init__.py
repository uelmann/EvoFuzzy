"""Baseline package — Phase A0 price-only LightGBM pipeline."""

from __future__ import annotations

__all__ = ["load_config"]


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    from pathlib import Path

    p = Path(path)
    with open(p) as f:
        return yaml.safe_load(f)
