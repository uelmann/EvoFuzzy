"""Phase 7.b — FUZZY-STACK helpers.

BACKTEST / ANALYSIS ONLY. Nothing adopted.
Product library of fixed CDF memberships + optional RULE-FORGE/NFN stack.
The PI hand formula is quarantined; rule features must carry provenance.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from btcb.constants import (
    PHASE7B_CLOSED,
    PHASE7B_ES_CAP,
    PHASE7B_ES_FLOOR,
    PHASE7B_ES_PATIENCE,
    PHASE7B_H,
    PHASE7B_HAND_FORMULA_NEEDLES,
    PHASE7B_KEEP_K,
    PHASE7B_LOG_EVERY,
    PHASE7B_NFN_N,
    PHASE7B_NFN_VERDICT,
    PHASE7B_OVERLAP_DELTA,
    PHASE7B_RANKIC_DELTA,
    PHASE7B_RULEFORGE_MAX,
    PHASE7B_TAIL_IC_DELTA,
    PHASE7B_UNDERTRAINED_LT,
    SEED,
)
from btcb.model import FoldSpec, fit_predict_fold, merge_twin_preds
from btcb.phase4b import (
    _cache_dump,
    _cache_load,
    _finish_null,
    _fold_cell,
    fold_tail_pack,
)
from btcb.phase4v2 import _utc, collapse_fold_preds

PROVENANCE_ORIG = "stage_s_original"
PROVENANCE_LIBRARY = "product_library_cdf"
PROVENANCE_RULEFORGE = "ruleforge"
PROVENANCE_NFN = "nfn"
ALLOWED_RULE_PROVENANCE = frozenset({PROVENANCE_RULEFORGE, PROVENANCE_NFN})

HYGIENE = {
    "es_floor": PHASE7B_ES_FLOOR,
    "patience": PHASE7B_ES_PATIENCE,
    "cap": PHASE7B_ES_CAP,
    "log_every": PHASE7B_LOG_EVERY,
    "undertrained_lt": PHASE7B_UNDERTRAINED_LT,
}

LGBM_P7B = {"num_threads": 8, "n_estimators": PHASE7B_ES_CAP, "early_stopping_rounds": PHASE7B_ES_PATIENCE}

RULEFORGE_SEARCH = (
    Path("/data/quant/btcb/phase6"),
    Path("/data/quant/btcb/ruleforge"),
    Path("/data/quant/btcb/rule_forge"),
)
NFN_SEARCH = (
    Path("/data/quant/btcb/phase7"),
    Path("/data/quant/btcb/nfn"),
)
REPORT_SEARCH = (
    Path("/data/quant/reports"),
    Path("/data/quant/btcb/phase6"),
    Path("/data/quant/btcb/phase7"),
    Path("/data/quant/btcb/ruleforge"),
    Path("/data/quant/btcb/nfn"),
)


def _log(msg: str) -> None:
    print(f"[p7b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_spec_to_dict(fold: FoldSpec) -> dict:
    return {
        "fold_id": int(fold.fold_id),
        "train_start": str(fold.train_start),
        "train_end": str(fold.train_end),
        "purge_end": str(fold.purge_end),
        "embargo_end": str(fold.embargo_end),
        "val_start": str(fold.val_start),
        "val_end": str(fold.val_end),
        "horizon": int(fold.horizon),
    }


def fold_spec_from_dict(d: dict) -> FoldSpec:
    return FoldSpec(
        fold_id=int(d["fold_id"]),
        train_start=pd.Timestamp(d["train_start"]),
        train_end=pd.Timestamp(d["train_end"]),
        purge_end=pd.Timestamp(d["purge_end"]),
        embargo_end=pd.Timestamp(d["embargo_end"]),
        val_start=pd.Timestamp(d["val_start"]),
        val_end=pd.Timestamp(d["val_end"]),
        horizon=int(d["horizon"]),
    )


def primitive_catalog(cols: list[str]) -> list[tuple[str, str, str]]:
    """(kind, feature, name) for μ and 1−μ of each column."""
    out = []
    for c in cols:
        out.append(("mu", c, f"mu_{c}"))
        out.append(("nmu", c, f"nmu_{c}"))
    return out


def printed_literal(kind: str, col: str) -> str:
    return f"μ({col})" if kind == "mu" else f"(1−μ({col}))"


def product_catalog(cols: list[str]) -> tuple[list[tuple[str, str, str]], list[dict]]:
    prims = primitive_catalog(cols)
    specs = []
    for i, j in combinations(range(len(prims)), 2):
        a, b = prims[i], prims[j]
        name = f"P__{a[2]}__{b[2]}"
        formula = f"{printed_literal(a[0], a[1])} × {printed_literal(b[0], b[1])}"
        specs.append({"name": name, "i": i, "j": j, "formula": formula, "left": a[2], "right": b[2]})
    return prims, specs


def assert_library_size(n_features: int, n_products: int) -> None:
    n_prim = 2 * int(n_features)
    expect = n_prim * (n_prim - 1) // 2
    if int(n_products) != expect:
        raise RuntimeError(f"product library size {n_products} != C({n_prim},2)={expect}")


def assert_firewall(feature_cols: list[str], provenance: dict[str, str], originals: list[str]) -> dict:
    """Reject the PI hand formula; rule columns must be RULE-FORGE/NFN."""
    needles = tuple(s.lower() for s in PHASE7B_HAND_FORMULA_NEEDLES)
    bad = []
    for c in feature_cols:
        low = str(c).lower()
        if any(n in low for n in needles):
            bad.append(c)
    if bad:
        raise RuntimeError(f"FIREWALL: PI hand-formula needles in features: {bad}")
    orig_set = set(originals)
    for c in feature_cols:
        src = provenance.get(c)
        if src is None:
            raise RuntimeError(f"FIREWALL: missing provenance for {c}")
        if src == PROVENANCE_ORIG and c not in orig_set:
            raise RuntimeError(f"FIREWALL: original tag on non-Stage-S column {c}")
        if src == PROVENANCE_LIBRARY and not str(c).startswith("P__"):
            raise RuntimeError(f"FIREWALL: library tag on non-product column {c}")
        if src in ALLOWED_RULE_PROVENANCE and c in orig_set:
            raise RuntimeError(f"FIREWALL: rule provenance on original {c}")
        if src not in {PROVENANCE_ORIG, PROVENANCE_LIBRARY, *ALLOWED_RULE_PROVENANCE}:
            raise RuntimeError(f"FIREWALL: unknown provenance {src} for {c}")
    leaked_rules = [c for c, s in provenance.items() if s in ALLOWED_RULE_PROVENANCE and c not in feature_cols]
    rec = {
        "name": "phase7b_firewall",
        "passed": True,
        "n_feat": len(feature_cols),
        "n_orig": sum(1 for c in feature_cols if provenance.get(c) == PROVENANCE_ORIG),
        "n_library": sum(1 for c in feature_cols if provenance.get(c) == PROVENANCE_LIBRARY),
        "n_ruleforge": sum(1 for c in feature_cols if provenance.get(c) == PROVENANCE_RULEFORGE),
        "n_nfn": sum(1 for c in feature_cols if provenance.get(c) == PROVENANCE_NFN),
        "leaked_unused_rules": leaked_rules,
    }
    _log(f"firewall PASS {rec}")
    return rec


def build_product_block(df: pd.DataFrame, cols: list[str], specs: list[dict]) -> pd.DataFrame:
    """Pairwise products of Φ(z) and 1−Φ(z). `cols` are already CS-z in feat_s."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"library missing originals: {missing}")
    z = df[cols].to_numpy(dtype=np.float64)
    mu = ndtr(z)
    nmu = 1.0 - mu
    prim = np.empty((z.shape[0], z.shape[1] * 2), dtype=np.float32)
    prim[:, 0::2] = mu.astype(np.float32, copy=False)
    prim[:, 1::2] = nmu.astype(np.float32, copy=False)
    n_prod = len(specs)
    assert_library_size(len(cols), n_prod)
    prod = np.empty((z.shape[0], n_prod), dtype=np.float32)
    for k, spec in enumerate(specs):
        prod[:, k] = prim[:, int(spec["i"])] * prim[:, int(spec["j"])]
    names = [s["name"] for s in specs]
    out = pd.DataFrame(prod, index=df.index, columns=names)
    return out


def provenance_for(cols: list[str], originals: list[str], library: list[str], rules: dict[str, str]) -> dict[str, str]:
    orig_set = set(originals)
    lib_set = set(library)
    out = {}
    for c in cols:
        if c in orig_set:
            out[c] = PROVENANCE_ORIG
        elif c in lib_set:
            out[c] = PROVENANCE_LIBRARY
        elif c in rules:
            out[c] = rules[c]
        else:
            raise RuntimeError(f"unassigned provenance: {c}")
    return out


def _read_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return None
    return blob if isinstance(blob, dict) else None


def _extract_verdict(blob: dict | None) -> str | None:
    if not blob:
        return None
    for key in ("verdict", "label", "status", "gate"):
        v = blob.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for k2 in ("verdict", "label", "status"):
                if isinstance(v.get(k2), str) and v[k2].strip():
                    return v[k2].strip()
    nested = blob.get("mechanical") or blob.get("verdicts") or blob.get("extra") or {}
    if isinstance(nested, dict):
        for key in ("verdict", "label", "status"):
            v = nested.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _verdict_ge_viable(verdict: str | None) -> bool:
    if not verdict:
        return False
    v = str(verdict).strip().upper()
    if any(tok in v for tok in ("NOT VIABLE", "NON-VIABLE", "BARREN", "PARKED", "FAIL", "REFUTED", "SKIP")):
        return False
    return "VIABLE" in v or v in {
        "PRODUCT-GRADE",
        "EXTRACTS",
        "COMPOSITION-WINS",
        "STRONG",
        "SLEEVE-GRADE",
        "CONFIRMED",
        "LIVE",
    }


def _verdict_is_live(verdict: str | None) -> bool:
    return str(verdict or "").strip().upper() == PHASE7B_NFN_VERDICT


def _find_reports(stems: tuple[str, ...]) -> list[Path]:
    hits = []
    for root in REPORT_SEARCH:
        if not root.exists():
            continue
        for stem in stems:
            hits.extend(sorted(root.glob(f"{stem}*.json")))
            hits.extend(sorted(root.glob(f"{stem}*.md")))
    # de-dupe, prefer json
    seen = set()
    out = []
    for p in hits:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    return out


def _find_parquets(roots: tuple[Path, ...], globs: tuple[str, ...]) -> list[Path]:
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for g in globs:
            hits.extend(sorted(root.glob(g)))
            hits.extend(sorted(root.rglob(g)))
    uniq = []
    seen = set()
    for p in hits:
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        uniq.append(p)
    return uniq


def _activation_cols(df: pd.DataFrame, prefixes: tuple[str, ...], max_n: int) -> list[str]:
    skip = {"date", "id", "symbol", "fold_id", "horizon", "p", "spread", "score"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if not prefixes or any(str(c).startswith(pref) for pref in prefixes):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(str(c))
    if not cols:
        cols = [
            str(c)
            for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]
    return cols[: int(max_n)]


def _standalone_tail_ic(blob: dict | None) -> float:
    if not blob:
        return float("nan")
    for key in (
        "tail_ic_top",
        "tail_ic_top_half",
        "standalone_tail_ic_top",
        "parent_tail_ic_top",
    ):
        v = blob.get(key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
    grid = blob.get("grid") or blob.get("metrics") or {}
    if isinstance(grid, dict):
        for _k, rec in grid.items():
            if isinstance(rec, dict) and rec.get("tail_ic_top") is not None:
                try:
                    return float(rec["tail_ic_top"])
                except (TypeError, ValueError):
                    continue
    extra = blob.get("extra") or {}
    if isinstance(extra, dict) and extra.get("tail_ic_top") is not None:
        try:
            return float(extra["tail_ic_top"])
        except (TypeError, ValueError):
            pass
    return float("nan")


def load_rule_stack() -> dict:
    """Conditional Arm B sources. Never invents activations; asserts provenance."""
    reasons = []
    rule_frames = []
    rule_prov: dict[str, str] = {}
    parents = {}

    rf_reports = _find_reports(("btcb_phase6", "btcb_ruleforge", "btcb_rule_forge", "ruleforge"))
    rf_banks = _find_parquets(
        RULEFORGE_SEARCH + (Path("/data/quant/reports"),),
        ("*bank*.parquet", "*rules*.parquet", "*activations*.parquet", "preds_*.parquet"),
    )
    rf_json = next((p for p in rf_reports if p.suffix == ".json"), None)
    rf_blob = _read_json(rf_json) if rf_json else None
    rf_verdict = _extract_verdict(rf_blob)
    rf_ok = bool(rf_banks) and _verdict_ge_viable(rf_verdict)
    if not rf_banks:
        reasons.append("RULE-FORGE bank missing")
    elif not _verdict_ge_viable(rf_verdict):
        reasons.append(
            f"RULE-FORGE verdict={rf_verdict!r} does not meet ≥ {PHASE7B_RULEFORGE_MIN_VERDICT}"
        )
    else:
        bank = pd.read_parquet(rf_banks[0])
        bank["date"] = _utc(bank["date"])
        bank["id"] = bank["id"].astype(int)
        cols = _activation_cols(bank, ("rf_", "ruleforge_", "rule_"), PHASE7B_RULEFORGE_MAX)
        if not cols:
            reasons.append("RULE-FORGE bank has no numeric rule columns")
            rf_ok = False
        else:
            keep = ["date", "id"] + cols
            sl = bank[keep].copy()
            rename = {}
            for c in cols:
                nc = c if str(c).startswith("rf_") else f"rf_{c}"
                rename[c] = nc
                rule_prov[nc] = PROVENANCE_RULEFORGE
            sl = sl.rename(columns=rename)
            rule_frames.append(sl)
            parents["ruleforge"] = {
                "verdict": rf_verdict,
                "path": str(rf_banks[0]),
                "report": str(rf_json) if rf_json else None,
                "n_rules": len(rename),
                "tail_ic_top": _standalone_tail_ic(rf_blob),
                "columns": list(rename.values()),
            }
            _log(f"RULE-FORGE accepted n={len(rename)} verdict={rf_verdict} from {rf_banks[0]}")

    nfn_reports = [
        p
        for p in _find_reports(("btcb_phase7", "btcb_nfn", "nfn"))
        if "phase7b" not in p.name.lower() and "fuzzystack" not in p.name.lower()
    ]
    nfn_banks = [
        p
        for p in _find_parquets(
            NFN_SEARCH + (Path("/data/quant/reports"),),
            ("*nfn*.parquet", "*ensemble*.parquet", "*activations*.parquet", "preds_*.parquet"),
        )
        if "phase7b" not in str(p).lower()
    ]
    nfn_json = next((p for p in nfn_reports if p.suffix == ".json"), None)
    nfn_blob = _read_json(nfn_json) if nfn_json else None
    nfn_verdict = _extract_verdict(nfn_blob)
    nfn_ok = bool(nfn_banks) and _verdict_is_live(nfn_verdict)
    if not nfn_banks:
        reasons.append("NFN bank missing")
    elif not _verdict_is_live(nfn_verdict):
        reasons.append(f"NFN verdict={nfn_verdict!r} is not {PHASE7B_NFN_VERDICT}")
    else:
        bank = pd.read_parquet(nfn_banks[0])
        bank["date"] = _utc(bank["date"])
        bank["id"] = bank["id"].astype(int)
        cols = _activation_cols(bank, ("nfn_", "ens_", "rule_"), PHASE7B_NFN_N)
        if not cols:
            reasons.append("NFN bank has no numeric rule columns")
            nfn_ok = False
        else:
            keep = ["date", "id"] + cols[: PHASE7B_NFN_N]
            sl = bank[keep].copy()
            rename = {}
            for c in keep[2:]:
                nc = c if str(c).startswith("nfn_") else f"nfn_{c}"
                rename[c] = nc
                rule_prov[nc] = PROVENANCE_NFN
            sl = sl.rename(columns=rename)
            rule_frames.append(sl)
            parents["nfn"] = {
                "verdict": nfn_verdict,
                "path": str(nfn_banks[0]),
                "report": str(nfn_json) if nfn_json else None,
                "n_rules": len(rename),
                "tail_ic_top": _standalone_tail_ic(nfn_blob),
                "columns": list(rename.values()),
            }
            _log(f"NFN accepted n={len(rename)} verdict={nfn_verdict} from {nfn_banks[0]}")

    skipped = not (rf_ok or nfn_ok)
    rules_df = None
    if rule_frames:
        rules_df = rule_frames[0]
        for extra in rule_frames[1:]:
            rules_df = rules_df.merge(extra, on=["date", "id"], how="outer")
        for c in list(rule_prov):
            if c in rules_df.columns:
                rules_df[c] = pd.to_numeric(rules_df[c], errors="coerce")
        # firewall: drop anything that looks like the hand formula
        drop = []
        for c in list(rule_prov):
            low = c.lower()
            if any(n in low for n in PHASE7B_HAND_FORMULA_NEEDLES):
                drop.append(c)
        if drop:
            raise RuntimeError(f"FIREWALL: rule stack carried hand-formula columns {drop}")
    rec = {
        "ruleforge_ok": bool(rf_ok),
        "nfn_ok": bool(nfn_ok),
        "skipped": skipped,
        "reasons": reasons,
        "parents": parents,
        "rule_provenance": rule_prov,
        "n_rule_features": int(len(rule_prov)),
        "ruleforge_verdict": rf_verdict,
        "nfn_verdict": nfn_verdict,
    }
    if skipped:
        print(f"STACK-SKIPPED reasons={reasons}", flush=True)
    return {"record": rec, "frame": rules_df}


def total_gain_by_feature(metas: list[dict], names: list[str]) -> dict[str, float]:
    acc = {n: 0.0 for n in names}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for n in names:
            acc[n] += float(gi.get(n, 0.0) or 0.0)
    return acc


def prune_library(
    gain_top: dict[str, float],
    gain_bot: dict[str, float],
    library_names: list[str],
    k: int = PHASE7B_KEEP_K,
) -> dict:
    ranked_top = sorted(library_names, key=lambda n: -float(gain_top.get(n, 0.0)))
    ranked_bot = sorted(library_names, key=lambda n: -float(gain_bot.get(n, 0.0)))
    top_k = ranked_top[: int(k)]
    bot_k = ranked_bot[: int(k)]
    union = sorted(
        set(top_k) | set(bot_k),
        key=lambda n: -(float(gain_top.get(n, 0.0)) + float(gain_bot.get(n, 0.0))),
    )
    return {
        "top_k": top_k,
        "bot_k": bot_k,
        "kept": union,
        "n_top": len(top_k),
        "n_bot": len(bot_k),
        "n_union": len(union),
        "k": int(k),
    }


def hygiene_rows(metas: list[dict], tag: str) -> list[dict]:
    rows = []
    for m in metas:
        bi = m.get("best_iteration")
        try:
            bi_i = int(bi) if bi is not None else None
        except (TypeError, ValueError):
            bi_i = None
        under = bool(m.get("undertrained"))
        if bi_i is not None and m.get("undertrained") is None:
            under = bi_i < int(PHASE7B_UNDERTRAINED_LT)
        rows.append(
            {
                "tag": tag,
                "fold_id": m.get("fold_id"),
                "head": m.get("head"),
                "best_iteration": bi_i,
                "undertrained": under,
                "status": m.get("status"),
                "elapsed": m.get("elapsed"),
                "n_train": m.get("n_train"),
                "n_holdout": m.get("n_holdout"),
            }
        )
    return rows


def count_undertrained(rows: list[dict]) -> int:
    return int(sum(1 for r in rows if r.get("undertrained")))


def gain_share(metas: list[dict], provenance: dict[str, str]) -> dict:
    totals = {PROVENANCE_ORIG: 0.0, PROVENANCE_LIBRARY: 0.0, PROVENANCE_RULEFORGE: 0.0, PROVENANCE_NFN: 0.0}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for feat, g in gi.items():
            src = provenance.get(feat)
            if src in totals:
                totals[src] += float(g or 0.0)
    grand = float(sum(totals.values()))
    share = {k: (v / grand if grand else 0.0) for k, v in totals.items()}
    share["rules"] = share[PROVENANCE_RULEFORGE] + share[PROVENANCE_NFN]
    share["total_gain"] = grand
    share["originals"] = share[PROVENANCE_ORIG]
    share["products"] = share[PROVENANCE_LIBRARY]
    return share


def join_spread(top: pd.DataFrame, bot: pd.DataFrame, horizon: int = PHASE7B_H) -> pd.DataFrame:
    twin = merge_twin_preds(top, bot, horizon)
    return twin


def collapse_spread(twin: pd.DataFrame) -> pd.DataFrame:
    if twin is None or twin.empty:
        return pd.DataFrame(columns=["date", "id", "spread"])
    return collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})


def _delta(a, b) -> float:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return float("nan")
    return fa - fb


def mechanical_verdicts(
    grid: dict,
    null_best: dict | None,
    best_arm: str | None,
    parents: dict,
    ran: dict,
) -> dict:
    base = grid.get("frozen_spread") or {}
    arms = [k for k in ("arm_a", "arm_b", "arm_ab") if ran.get(k)]
    per = {}
    any_extracts = False
    any_whole = False
    any_comp = False
    null_pass = bool((null_best or {}).get("passed")) if null_best else False
    for name in arms:
        met = grid.get(name) or {}
        d_ic = _delta(met.get("tail_ic_top"), base.get("tail_ic_top"))
        d_ov = _delta(met.get("overlap"), base.get("overlap"))
        d_ric = _delta(met.get("rankic"), base.get("rankic"))
        this_null = bool(null_pass and name == best_arm)
        clears = bool(
            np.isfinite(d_ic)
            and np.isfinite(d_ov)
            and d_ic >= float(PHASE7B_TAIL_IC_DELTA)
            and d_ov >= float(PHASE7B_OVERLAP_DELTA)
        )
        extracts = bool(clears and this_null)
        parent_ics = []
        if name in ("arm_b", "arm_ab"):
            for src, rec in (parents or {}).items():
                tic = rec.get("tail_ic_top")
                try:
                    tic_f = float(tic)
                except (TypeError, ValueError):
                    tic_f = float("nan")
                if np.isfinite(tic_f):
                    parent_ics.append((src, tic_f))
        beats_parents = True
        if parent_ics:
            arm_ic = met.get("tail_ic_top")
            try:
                arm_ic_f = float(arm_ic)
            except (TypeError, ValueError):
                arm_ic_f = float("nan")
            beats_parents = bool(np.isfinite(arm_ic_f) and all(arm_ic_f > p for _, p in parent_ics))
        else:
            beats_parents = False
        composition_wins = bool(extracts and beats_parents and parent_ics)
        whole = bool(
            (not extracts)
            and this_null
            and np.isfinite(d_ric)
            and d_ric >= float(PHASE7B_RANKIC_DELTA)
        )
        if extracts:
            label = "COMPOSITION-WINS" if composition_wins else "EXTRACTS"
        elif whole:
            label = "WHOLE-RANKING LEAD"
        else:
            label = "FAIL"
        per[name] = {
            "label": label,
            "delta_tail_ic_top": d_ic,
            "delta_overlap": d_ov,
            "delta_rankic": d_ric,
            "clears_deltas": clears,
            "null_pass": this_null,
            "extracts": extracts,
            "composition_wins": composition_wins,
            "whole_ranking_lead": whole,
            "beats_parents": beats_parents,
            "n_parents": len(parent_ics),
        }
        any_extracts = any_extracts or extracts
        any_whole = any_whole or whole
        any_comp = any_comp or composition_wins
    closed = None
    if arms and (not any_extracts) and (not any_whole) and (not any_comp):
        closed = PHASE7B_CLOSED
    if not arms:
        closed = PHASE7B_CLOSED
    return {
        "per_arm": per,
        "best_arm": best_arm,
        "null_pass": null_pass,
        "any_extracts": any_extracts,
        "any_composition_wins": any_comp,
        "any_whole_ranking_lead": any_whole,
        "closed": closed,
        "nothing_adopted": True,
        "stack_skipped": not bool(ran.get("arm_b")),
    }


def pick_best_arm(grid: dict, ran: dict) -> str | None:
    cands = []
    for name in ("arm_a", "arm_b", "arm_ab"):
        if not ran.get(name):
            continue
        tic = (grid.get(name) or {}).get("tail_ic_top")
        try:
            tic_f = float(tic)
        except (TypeError, ValueError):
            continue
        if np.isfinite(tic_f):
            cands.append((name, tic_f))
    if not cands:
        return None
    cands.sort(key=lambda kv: -kv[1])
    return cands[0][0]


def gate_vol_matched_spread_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real: dict[int, dict],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    y_top: str,
    y_bot: str,
    cache_dir: Path | None = None,
    commit_fn=None,
    vol_col: str = "yz_vol_30",
    hygiene: dict | None = None,
    lgbm_params: dict | None = None,
) -> dict:
    """Twin classifier heads, joint vol-matched label shuffle (house tail null)."""
    from btcb.constants import NULL_REPLICATES, NULL_SHUFFLE_SEEDS

    use_seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    hyg = hygiene if hygiene is not None else HYGIENE
    params = lgbm_params if lgbm_params is not None else LGBM_P7B
    for fold in folds:
        ics, ovs, mons = [], [], []
        for i, ss in enumerate(use_seeds):
            cached = None
            if cache_dir is not None:
                cached = _cache_load(cache_dir / f"fold{fold.fold_id}_seed{ss}.json")
            if cached is not None:
                ics.append(cached.get("tail_ic_top"))
                ovs.append(cached.get("overlap"))
                mons.append(cached.get("monster"))
                continue
            _log(f"spread-null fold={fold.fold_id} rep={i+1}/{len(use_seeds)} seed={ss}")
            pred_t, meta_t = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                early_stop="per_date_auc",
                ycol=y_top,
                hygiene=hyg,
                lgbm_params=params,
            )
            pred_b, meta_b = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                early_stop="per_date_auc",
                ycol=y_bot,
                hygiene=hyg,
                lgbm_params=params,
            )
            if (
                pred_t.empty
                or pred_b.empty
                or meta_t.get("status") != "ok"
                or meta_b.get("status") != "ok"
            ):
                rec = {
                    "tail_ic_top": None,
                    "overlap": None,
                    "monster": None,
                    "status": f"{meta_t.get('status')}/{meta_b.get('status')}",
                }
            else:
                twin = merge_twin_preds(pred_t, pred_b, fold.horizon)
                sm = fold_tail_pack(twin, labeled, close, btc_id, "spread")
                rec = {k: sm.get(k) for k in ("tail_ic_top", "overlap", "monster")}
                rec["status"] = "ok"
            ics.append(rec.get("tail_ic_top"))
            ovs.append(rec.get("overlap"))
            mons.append(rec.get("monster"))
            if cache_dir is not None:
                _cache_dump(
                    cache_dir / f"fold{fold.fold_id}_seed{ss}.json",
                    rec,
                    commit_fn if (i + 1) % 5 == 0 else None,
                )
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
        st = cells["tail_ic_top"][-1]
        _log(
            f"spread-null fold={fold.fold_id} mean={st['mean']:.4f} p95={st['p95']:.4f} "
            f"real={st.get('real_tail_ic_top')} bias_ok={st['bias_ok']}"
        )
        if cache_dir is not None and commit_fn is not None:
            commit_fn()
    return _finish_null(
        "fuzzy_spread_vol_matched_null",
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )


def null_one_replicate(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed_s: int,
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    y_top: str,
    y_bot: str,
    vol_col: str,
) -> dict:
    pred_t, meta_t = fit_predict_fold(
        df,
        fold,
        seed=SEED,
        shuffle_labels=True,
        shuffle_seed=int(seed_s),
        shuffle_mode="vol_matched",
        vol_col=vol_col,
        feature_cols=feature_cols,
        early_stop="per_date_auc",
        ycol=y_top,
        hygiene=HYGIENE,
        lgbm_params=LGBM_P7B,
    )
    pred_b, meta_b = fit_predict_fold(
        df,
        fold,
        seed=SEED,
        shuffle_labels=True,
        shuffle_seed=int(seed_s),
        shuffle_mode="vol_matched",
        vol_col=vol_col,
        feature_cols=feature_cols,
        early_stop="per_date_auc",
        ycol=y_bot,
        hygiene=HYGIENE,
        lgbm_params=LGBM_P7B,
    )
    if pred_t.empty or pred_b.empty or meta_t.get("status") != "ok" or meta_b.get("status") != "ok":
        return {
            "tail_ic_top": None,
            "overlap": None,
            "monster": None,
            "status": f"{meta_t.get('status')}/{meta_b.get('status')}",
            "fold_id": fold.fold_id,
            "shuffle_seed": int(seed_s),
        }
    twin = merge_twin_preds(pred_t, pred_b, fold.horizon)
    sm = fold_tail_pack(twin, labeled, close, btc_id, "spread")
    return {
        "tail_ic_top": sm.get("tail_ic_top"),
        "overlap": sm.get("overlap"),
        "monster": sm.get("monster"),
        "status": "ok",
        "fold_id": fold.fold_id,
        "shuffle_seed": int(seed_s),
        "best_iteration_top": meta_t.get("best_iteration"),
        "best_iteration_bot": meta_b.get("best_iteration"),
        "undertrained_top": meta_t.get("undertrained"),
        "undertrained_bot": meta_b.get("undertrained"),
    }


def assemble_null_from_replicates(rows: list[dict], folds: list[FoldSpec], real: dict[int, dict]) -> dict:
    by_fold: dict[int, list[dict]] = {}
    for rec in rows:
        by_fold.setdefault(int(rec["fold_id"]), []).append(rec)
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    for fold in folds:
        pack = by_fold.get(int(fold.fold_id), [])
        pack = sorted(pack, key=lambda r: int(r.get("shuffle_seed") or 0))
        ics = [r.get("tail_ic_top") for r in pack]
        ovs = [r.get("overlap") for r in pack]
        mons = [r.get("monster") for r in pack]
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
    return _finish_null(
        "fuzzy_spread_vol_matched_null",
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )
