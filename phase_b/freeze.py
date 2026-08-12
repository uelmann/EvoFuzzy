"""Phase B — freeze helpers for locked A0 config."""

from __future__ import annotations

import hashlib
from pathlib import Path


FROZEN_CONFIG_PATH = Path("config_frozen_a0.yaml")
FROZEN_HASH_PATH = Path("config_frozen_a0.sha256")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_frozen_snapshot(src: Path = Path("config.yaml")) -> dict:
    text = src.read_text()
    h = sha256_text(text)
    FROZEN_CONFIG_PATH.write_text(text)
    FROZEN_HASH_PATH.write_text(h + "\n")
    return {"path": str(FROZEN_CONFIG_PATH), "sha256": h}


def verify_frozen(expected_hash: str | None = None) -> dict:
    if not FROZEN_CONFIG_PATH.exists() or not FROZEN_HASH_PATH.exists():
        raise RuntimeError("Frozen A0 config missing — refuse to run Phase B")
    text = FROZEN_CONFIG_PATH.read_text()
    file_hash = FROZEN_HASH_PATH.read_text().strip()
    calc = sha256_text(text)
    if calc != file_hash:
        raise RuntimeError(f"Frozen config hash mismatch: file={file_hash} calc={calc}")
    if expected_hash is not None and calc != expected_hash:
        raise RuntimeError(f"Frozen config does not match expected hash {expected_hash}")
    # live config.yaml must match frozen snapshot
    live = Path("config.yaml")
    if live.exists():
        live_h = sha256_text(live.read_text())
        if live_h != calc:
            raise RuntimeError(
                f"config.yaml drifted from frozen A0 (live={live_h} frozen={calc}) — task failure"
            )
    return {"ok": True, "sha256": calc}
