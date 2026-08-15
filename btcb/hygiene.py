"""Redenom/split cleaning, investability floor, naive contribution autopsy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    CONTRIB_SHARE_FLAG,
    FLOOR_DV_MED_30,
    FLOOR_MAX_ABS_RET_30,
    FLOOR_MIN_PRICE,
    FLOOR_MIN_SESSIONS,
    JUMP_ABS_RET,
    PIT_DV_MIN_PERIODS,
    PIT_DV_WINDOW,
    STABLE_OR_WRAP,
)
from btcb.universe import build_pit_topn_ids


_SPLIT_FACTORS = np.array(
    [10.0 ** k for k in range(1, 10)]
    + [2.0 * 10.0 ** k for k in range(0, 9)]
    + [5.0 * 10.0 ** k for k in range(0, 8)],
    dtype=float,
)
_SPLIT_CANDS = np.concatenate([_SPLIT_FACTORS, 1.0 / _SPLIT_FACTORS])


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.normalize()


def contribution_table(
    contrib: dict[int, float],
    panel: pd.DataFrame,
    id_to_sym: dict,
    btc_id: int,
    top_n: int = 10,
) -> dict:
    rows = []
    tot = float(sum(contrib.values())) if contrib else 0.0
    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    p = panel.copy()
    p["date"] = _utc(p["date"])
    p = p.sort_values(["id", "date"])
    p["dret"] = p.groupby("id")["close"].pct_change()
    max_abs = p.groupby("id")["dret"].apply(lambda s: float(s.abs().max()) if s.notna().any() else float("nan"))
    max_pos = p.groupby("id")["dret"].max()
    for iid, c in ranked[: int(top_n)]:
        iid = int(iid)
        share = (c / tot) if tot else 0.0
        rows.append(
            {
                "id": iid,
                "symbol": id_to_sym.get(iid),
                "contrib": float(c),
                "share": float(share),
                "is_btc": iid == int(btc_id),
                "max_abs_daily_ret": float(max_abs.get(iid, np.nan)) if iid in max_abs.index else float("nan"),
                "max_daily_ret": float(max_pos.get(iid, np.nan)) if iid in max_pos.index else float("nan"),
            }
        )
    top_share = max((abs(r["share"]) for r in rows if not r["is_btc"]), default=0.0)
    flagged = bool(top_share > CONTRIB_SHARE_FLAG)
    return {
        "top": rows,
        "total_contrib": tot,
        "top_alt_share": float(top_share),
        "flag_single_name_gt_25pct": flagged,
        "n_names": int(len(contrib)),
    }


def _round_split(ratio: float, tol: float = 0.30) -> tuple[bool, float]:
    if not np.isfinite(ratio) or ratio <= 0:
        return False, float("nan")
    log_r = np.log(ratio)
    err = np.abs(np.log(_SPLIT_CANDS) - log_r)
    j = int(np.argmin(err))
    if err[j] <= abs(np.log(1.0 + tol)):
        return True, float(_SPLIT_CANDS[j])
    return False, float(ratio)


def find_jump_days(panel: pd.DataFrame, thresh: float = JUMP_ABS_RET) -> pd.DataFrame:
    df = panel.sort_values(["id", "date"]).copy()
    df["date"] = _utc(df["date"])
    df["dret"] = df.groupby("id")["close"].pct_change()
    cols = [c for c in ("id", "symbol", "slug", "date", "close", "dret", "mcap", "dv") if c in df.columns]
    jumps = df[df["dret"].abs() > float(thresh)][cols].copy()
    return jumps


def clean_panel(panel: pd.DataFrame, btc_id: int, thresh: float = JUMP_ABS_RET) -> tuple[pd.DataFrame, list[dict]]:
    """Splice round splits / stable-mcap jumps; otherwise truncate at the breakpoint."""
    df = panel.copy()
    df["date"] = _utc(df["date"])
    df["id"] = df["id"].astype(int)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    log: list[dict] = []
    keep_parts = []
    px_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]

    for iid, g in df.groupby("id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        iid = int(iid)
        if iid == int(btc_id):
            keep_parts.append(g)
            continue
        dret = g["close"].pct_change()
        jump_idx = [int(i) for i in g.index[dret.abs() > float(thresh)].tolist()]
        if not jump_idx:
            keep_parts.append(g)
            continue
        g_work = g
        actions = []
        truncated = False
        # re-scan after each splice; stop on truncate
        guard = 0
        while guard < 20:
            guard += 1
            dret = g_work["close"].pct_change()
            hits = [int(i) for i in g_work.index[dret.abs() > float(thresh)].tolist()]
            if not hits:
                break
            i = hits[0]
            if i <= 0:
                g_work = g_work.iloc[1:].reset_index(drop=True)
                actions.append({"action": "drop_first", "date": str(pd.Timestamp(g.loc[0, "date"]).date())})
                continue
            prev_c = float(g_work.loc[i - 1, "close"])
            cur_c = float(g_work.loc[i, "close"])
            ratio = cur_c / prev_c if prev_c else np.nan
            n_after = int(len(g_work) - i)
            mcap_ratio = np.nan
            if "mcap" in g_work.columns:
                m0 = float(g_work.loc[i - 1, "mcap"]) if np.isfinite(g_work.loc[i - 1, "mcap"]) else np.nan
                m1 = float(g_work.loc[i, "mcap"]) if np.isfinite(g_work.loc[i, "mcap"]) else np.nan
                if m0 and np.isfinite(m0) and m0 > 0 and np.isfinite(m1) and m1 > 0:
                    mcap_ratio = m1 / m0
            round_ok, factor = _round_split(ratio)
            mcap_stable = bool(np.isfinite(mcap_ratio) and abs(np.log(mcap_ratio)) < 0.5)
            can_splice = bool(n_after >= 30 and (round_ok or mcap_stable) and np.isfinite(ratio) and ratio > 0)
            rec = {
                "id": iid,
                "symbol": str(g_work.loc[i, "symbol"]),
                "slug": str(g_work.loc[i, "slug"]) if "slug" in g_work.columns else None,
                "date": str(pd.Timestamp(g_work.loc[i, "date"]).date()),
                "ratio": float(ratio) if np.isfinite(ratio) else None,
                "mcap_ratio": float(mcap_ratio) if np.isfinite(mcap_ratio) else None,
                "n_after": n_after,
                "round_split": bool(round_ok),
                "split_factor": float(factor) if np.isfinite(factor) else None,
                "mcap_stable": mcap_stable,
            }
            if can_splice:
                scale = prev_c / cur_c
                g_work.loc[i:, px_cols] = g_work.loc[i:, px_cols].astype(float) * scale
                # Coin units scale inversely; USD dv is split-invariant and is left alone.
                if "volume" in g_work.columns:
                    g_work.loc[i:, "volume"] = g_work.loc[i:, "volume"].astype(float) / scale
                if "mcap" in g_work.columns and np.isfinite(mcap_ratio) and abs(np.log(mcap_ratio)) >= 0.5:
                    g_work.loc[i:, "mcap"] = g_work.loc[i:, "mcap"].astype(float) * scale
                rec["action"] = "splice"
                rec["scale"] = float(scale)
                actions.append(rec)
            else:
                g_work = g_work.iloc[:i].reset_index(drop=True)
                rec["action"] = "truncate"
                actions.append(rec)
                truncated = True
                break
        if len(g_work):
            keep_parts.append(g_work)
        log.append(
            {
                "id": iid,
                "symbol": str(g["symbol"].iloc[-1]),
                "n_jumps_raw": int(len(jump_idx)),
                "truncated": truncated,
                "actions": actions,
            }
        )
    out = pd.concat(keep_parts, ignore_index=True) if keep_parts else df.iloc[0:0]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    n_splice = sum(1 for r in log for a in r["actions"] if a.get("action") == "splice")
    n_trunc = sum(1 for r in log for a in r["actions"] if a.get("action") == "truncate")
    print(
        f"[HB] clean ids_touched={len(log)} splice_events={n_splice} truncate_events={n_trunc} "
        f"rows {len(panel)}→{len(out)}",
        flush=True,
    )
    return out.reset_index(drop=True), log


def eligibility_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """Long frame date,id,eligible under the frozen investability floor (causal)."""
    df = panel.copy()
    df["date"] = _utc(df["date"])
    df["id"] = df["id"].astype(int)
    df = df.sort_values(["id", "date"])
    df["dret"] = df.groupby("id")["close"].pct_change()
    df["dv_med30"] = df.groupby("id")["dv"].transform(
        lambda s: s.rolling(PIT_DV_WINDOW, min_periods=min(PIT_DV_MIN_PERIODS, PIT_DV_WINDOW)).median()
    )
    df["prior_sess"] = df.groupby("id").cumcount()  # 0 on first session
    df["max_abs_ret30"] = df.groupby("id")["dret"].transform(
        lambda s: s.abs().rolling(30, min_periods=5).max()
    )
    ok = (
        (df["dv_med30"] >= FLOOR_DV_MED_30)
        & (df["close"] >= FLOOR_MIN_PRICE)
        & (df["prior_sess"] >= FLOOR_MIN_SESSIONS)
        & (df["max_abs_ret30"].fillna(0.0) <= FLOOR_MAX_ABS_RET_30)
        & (~df["symbol"].str.upper().isin(STABLE_OR_WRAP))
    )
    out = df[["date", "id", "symbol"]].copy()
    out["eligible"] = ok.astype(bool)
    out["dv_med30"] = df["dv_med30"]
    out["mcap_med30"] = df.groupby("id")["mcap"].transform(
        lambda s: s.rolling(PIT_DV_WINDOW, min_periods=min(PIT_DV_MIN_PERIODS, PIT_DV_WINDOW)).median()
    )
    return out


def build_floored_pit(panel: pd.DataFrame, n: int, score: str = "dv") -> tuple[pd.DataFrame, dict]:
    elig = eligibility_mask(panel)
    df = panel.copy()
    df["date"] = _utc(df["date"])
    df["id"] = df["id"].astype(int)
    score_col = "mcap_med30" if score == "mcap" else "dv_med30"
    merged = df.merge(elig[["date", "id", "eligible", score_col]], on=["date", "id"], how="left")
    merged["eligible"] = merged["eligible"].fillna(False)
    score_src = merged[merged["eligible"]].copy()
    if score_src.empty:
        return pd.DataFrame(columns=["date", "id", "symbol", "rank", "score"]), {"n_eligible_rows": 0}
    wide = score_src.pivot(index="date", columns="id", values=score_col).sort_index()
    last_sym = score_src.sort_values("date").groupby("id")["symbol"].last()
    pit = build_pit_topn_ids(wide, n=n, last_sym=last_sym)
    by_d = elig[elig["eligible"]].groupby("date").size()
    extra = {
        "n_eligible_rows": int(elig["eligible"].sum()),
        "n_eligible_ids": int(elig.loc[elig["eligible"], "id"].nunique()),
        "median_eligible_per_date": float(by_d.median()) if len(by_d) else 0.0,
        "min_eligible_per_date": int(by_d.min()) if len(by_d) else 0,
        "n_dates_ge_n": int((by_d >= n).sum()) if len(by_d) else 0,
        "n_dates": int(by_d.shape[0]) if len(by_d) else 0,
        "pit_rows": int(len(pit)),
        "pit_n": int(n),
        "score": str(score),
        "score_col": score_col,
    }
    print(
        f"[HB] floored PIT n={n} score={score} rows={len(pit)} eligible_ids={extra['n_eligible_ids']} "
        f"med_elig/day={extra['median_eligible_per_date']:.0f}",
        flush=True,
    )
    return pit, extra
