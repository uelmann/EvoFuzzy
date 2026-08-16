"""Firewall: PI formula never referenced; warm-start provenance only from RULE-FORGE."""

from __future__ import annotations

import ast
from pathlib import Path

from nfn_va.constants import FORBIDDEN_TOKENS

NFN_ROOT = Path(__file__).resolve().parent

SKIP_NAMES = {"firewall.py", "constants.py"}


def miner_source_files() -> list[Path]:
    files = []
    for p in sorted(NFN_ROOT.glob("*.py")):
        if p.name in SKIP_NAMES:
            continue
        files.append(p)
    pipe = NFN_ROOT.parent / "btcb_phase7d_pipeline.py"
    if pipe.exists():
        files.append(pipe)
    return files


def scan_forbidden(text: str) -> list[str]:
    hits = []
    low = text.lower()
    for tok in FORBIDDEN_TOKENS:
        if tok.lower() in low:
            hits.append(tok)
    return hits


def assert_no_manuel2_import(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "btcb.manuel2" or alias.name.startswith("btcb.manuel2."):
                    raise RuntimeError(f"firewall: {path.name} imports {alias.name}")
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "btcb.manuel2" or mod.startswith("btcb.manuel2."):
                raise RuntimeError(f"firewall: {path.name} imports from {mod}")
            if mod == "btcb" and any(a.name == "manuel2" for a in node.names):
                raise RuntimeError(f"firewall: {path.name} imports btcb.manuel2")


def assert_firewall() -> dict:
    hits_all = []
    files = miner_source_files()
    for p in files:
        text = p.read_text()
        hits = scan_forbidden(text)
        if hits:
            hits_all.append({"file": p.name, "tokens": hits})
        assert_no_manuel2_import(p)
    if hits_all:
        raise RuntimeError(f"firewall: hand-formula tokens in miner source: {hits_all}")
    rec = {"passed": True, "n_files": len(files), "files": [p.name for p in files]}
    print(f"[p7d-firewall] PASS files={rec['n_files']}", flush=True)
    return rec
