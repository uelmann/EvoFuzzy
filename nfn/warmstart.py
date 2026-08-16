"""Map a VIABLE Phase-6 RULE-FORGE bank onto 8/24 NFN rules. Provenance asserted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from btcb.constants import STAGE_S_COLS
from nfn.constants import N_INIT_PRIMITIVES, N_MEMBERSHIPS, N_RULES, WARMSTART_RULES

BANK_CANDIDATES = (
    Path("/data/quant/reports/btcb_phase6_ruleforge.json"),
    Path("/data/quant/btcb/phase6/ruleforge.json"),
    Path("/root/btcb_phase6_ruleforge.json"),
    Path("reports/btcb_phase6_ruleforge.json"),
)

CENTER_K = 1  # membership with c_init = 0.0


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def primitive_index(feat_idx: int, mem_k: int, complement: bool) -> int:
    n_feat = len(STAGE_S_COLS)
    base = int(feat_idx) * int(N_MEMBERSHIPS) + int(mem_k)
    if complement:
        return n_feat * int(N_MEMBERSHIPS) + base
    return base


def _viable(blob: dict) -> bool:
    if not isinstance(blob, dict):
        return False
    v = blob.get("verdict") if isinstance(blob.get("verdict"), dict) else blob
    for key in ("viable", "VIABLE", "is_viable"):
        if key in v and bool(v[key]):
            return True
    label = str(v.get("label") or blob.get("label") or blob.get("mechanical") or "")
    return "VIABLE" in label.upper() and "NOT" not in label.upper()


def _rules_from_blob(blob: dict) -> list[dict]:
    bank = blob.get("rule_bank") or blob.get("bank") or blob
    rules = bank.get("rules") if isinstance(bank, dict) else None
    if not isinstance(rules, list):
        return []
    out = []
    name_to_i = {n: i for i, n in enumerate(STAGE_S_COLS)}
    for rec in rules:
        if not isinstance(rec, dict):
            continue
        w = float(rec.get("weight", 0.0))
        lits = rec.get("literals") or rec.get("primitives") or []
        mapped = []
        for lit in lits:
            if not isinstance(lit, dict):
                continue
            if "feature" in lit:
                feat = str(lit["feature"])
                if feat not in name_to_i:
                    continue
                fi = name_to_i[feat]
            else:
                fi = int(lit.get("feature_idx", lit.get("j", -1)))
            if fi < 0 or fi >= len(STAGE_S_COLS):
                continue
            comp = bool(lit.get("complement", False))
            mapped.append({"feature_idx": fi, "complement": comp})
        out.append({"weight": w, "literals": mapped})
    return out


def find_ruleforge_bank() -> tuple[dict | None, dict]:
    meta = {"path": None, "sha256": None, "viable": False, "n_rules": 0, "reason": "not_found"}
    for p in BANK_CANDIDATES:
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception as exc:
            meta = {"path": str(p), "sha256": None, "viable": False, "n_rules": 0, "reason": f"read_fail:{exc}"}
            continue
        rules = _rules_from_blob(blob)
        ok = _viable(blob) and len(rules) > 0
        meta = {
            "path": str(p),
            "sha256": _file_sha256(p),
            "viable": bool(ok),
            "n_rules": int(len(rules)),
            "reason": "ok" if ok else "not_viable_or_empty",
        }
        if ok:
            return blob, meta
        # keep scanning; a later candidate may be viable
    return None, meta


def apply_warmstart(model, blob: dict, n_take: int = WARMSTART_RULES) -> int:
    """Copy up to n_take RULE-FORGE product rules onto the first NFN rules."""
    import torch

    rules = _rules_from_blob(blob)[: int(n_take)]
    if not rules:
        return 0
    with torch.no_grad():
        # zero the reserved rules' exponents, then set mapped literals to e=1
        n_prim = int(model.e_raw.shape[1])
        for k, rec in enumerate(rules):
            model.e_raw.data[k].fill_(-12.0)
            for lit in rec["literals"][:N_INIT_PRIMITIVES]:
                p = primitive_index(int(lit["feature_idx"]), CENTER_K, bool(lit["complement"]))
                if 0 <= p < n_prim:
                    # softplus(x)=1 => x = log(e-1)
                    model.e_raw.data[k, p] = float(np.log(np.e - 1.0))
            model.w.data[k] = float(np.clip(rec.get("weight", 0.0), -1.0, 1.0))
    return int(len(rules))
