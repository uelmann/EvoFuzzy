"""
BTC-BEATER SPREAD-LS universe sensitivity — top-30 vs top-50 vs top-100.

BACKTEST ONLY. Portfolio layer only. CPU only. Frozen products untouched.
2.c spread cache reused byte-identical. Funding-off (3.b has not run).
Usage: modal run --detach btcb_universe_sens_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-univ"
VOL_Q = "quant-baseline"

quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "lightgbm",
        "matplotlib",
        "httpx",
        "pyyaml",
        "scikit-learn",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file(
        "reports/btcb_universe_sensitivity_addendum.md",
        remote_path="/root/btcb_universe_sensitivity_addendum.md",
    )
    .add_local_file("universe/btcb_top50_floor.parquet", remote_path="/root/btcb_top50_floor.parquet")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
)

app = modal.App(APP_NAME, image=image)


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {
        "daily_ret",
        "btc_ret",
        "equity",
        "n_long",
        "n_short",
        "n_shortable",
        "incomplete",
        "long_gross",
        "short_gross",
        "cash",
        "realized_beta_90d",
        "id_to_sym",
        "contrib",
    }
    if isinstance(x, dict):
        return {str(k): _jsonable(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, drop) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    return x


@app.function(
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_univ() -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import load_panel
    from baseline.seedutil import seed_everything
    from btcb.constants import (
        PHASE2_PRIMARY_H,
        PHASE3_FUNDING_CAVEAT,
        SEED,
        UNIVERSE_FUNDING_ON,
        UNIVERSE_NS,
        UNIVERSE_PRIMARY_H,
        UNIVERSE_SENS_CRITERION,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import build_floored_pit, clean_panel
    from btcb.spread_ls import (
        attach_beta,
        build_shortable,
        choose_production_universe,
        ew_basket,
        hash_pred_dir,
        load_twin_from_cache,
        mcap_overrides_volume,
        run_spread_ls,
        squeeze_table,
        within_universe_rankic,
    )
    from btcb.universe_sens_report import plot_three_equity, write_universe_sensitivity

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_universe_sensitivity_addendum.md").read_text()
    if UNIVERSE_SENS_CRITERION not in addendum:
        raise RuntimeError("universe-sensitivity addendum missing verbatim criterion")
    print("[HB] BTC-BEATER universe sensitivity BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {UNIVERSE_SENS_CRITERION}", flush=True)
    print(f"[HB] {PHASE3_FUNDING_CAVEAT}", flush=True)
    print(f"[HB] FUNDING_ON={UNIVERSE_FUNDING_ON} (3.b has not run; all books funding-off)", flush=True)
    if UNIVERSE_FUNDING_ON:
        raise RuntimeError("funding-on requested but 3.b has not run")

    def commit():
        quant_vol.commit()

    panel = pd.read_parquet("/data/quant/btcb/full/panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id}", flush=True)

    def _load_pit(name: str):
        for p in (
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/root/{name}"),
        ):
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                print(f"[HB] pit {name} from {p} rows={len(df)}", flush=True)
                return df
        return None

    pit50 = _load_pit("btcb_top50_floor.parquet")
    pit100 = _load_pit("btcb_top100_floor.parquet")
    if pit50 is None or pit100 is None:
        raise RuntimeError("missing reused floored PIT top-50/100")

    print("[HB] re-applying frozen 2.b cleaner (no new hygiene)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)

    feat = pd.read_parquet("/data/quant/btcb/phase2b/feat_s.parquet")
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n_files={pred_hash['n_files']}", flush=True)
    twin = load_twin_from_cache(pred_dir, UNIVERSE_PRIMARY_H)
    print(f"[HB] twin h={UNIVERSE_PRIMARY_H} rows={len(twin)}", flush=True)

    raw_dir = Path("/data/quant/raw/klines")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    kline_panel = load_panel(raw_dir, kline_syms)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    shortable = build_shortable(cleaned, kline_panel, btc_id)
    print(f"[HB] shortable dates={len(shortable)}", flush=True)

    close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    btc_simple = close[btc_id].pct_change(fill_method=None)

    uni_dir = Path("/data/quant/btcb/universe")
    uni_dir.mkdir(parents=True, exist_ok=True)
    pit30, extra30 = build_floored_pit(cleaned, 30, score="dv")
    pit30.to_parquet(uni_dir / "btcb_top30_floor.parquet", index=False)
    commit()

    pits_dv = {30: pit30, 50: pit50, 100: pit100}
    pits_mcap = {}
    for n in UNIVERSE_NS:
        p, ex = build_floored_pit(cleaned, int(n), score="mcap")
        pits_mcap[int(n)] = p
        print(f"[HB] mcap PIT n={n} rows={ex.get('pit_rows')}", flush=True)

    def _run_one(pit, n, tag):
        print(f"[HB] book {tag} U={n} k_dec={n//10} k_q={n//5} beta_matched=1 funding=0...", flush=True)
        packed = run_spread_ls(
            cleaned,
            pit,
            twin,
            feat,
            shortable,
            btc_id,
            h=int(PHASE2_PRIMARY_H),
            beta_matched=True,
            decile_k=int(n) // 10,
            quintile_k=int(n) // 5,
        )
        if packed.get("error"):
            raise RuntimeError(f"{tag} failed: {packed}")
        packed = attach_beta(packed, btc_simple)
        packed["rankic"] = within_universe_rankic(twin, pit, horizon=UNIVERSE_PRIMARY_H)
        members = {
            pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
            for d, v in pit.groupby("date")["id"]
        }
        all_dates = [d for d in close.index if d in members]
        basket = ew_basket(close, members, all_dates)
        sq = squeeze_table(packed["daily_ret"], basket)
        vals = [float(r["spread_ls"]) for r in sq if np.isfinite(r.get("spread_ls", float("nan")))]
        packed["squeeze"] = sq
        packed["squeeze_mean"] = float(np.mean(vals)) if vals else float("nan")
        packed["universe_n"] = int(n)
        packed["ranking"] = tag.split("-")[0]
        packed["funding_on"] = False
        print(
            f"[HB] {tag} sharpe={packed.get('net_sharpe')} trail={packed.get('net_sharpe_trail18m')} "
            f"rankic={packed.get('rankic')} squeeze_mean={packed.get('squeeze_mean')} "
            f"top5={ (packed.get('concentration') or {}).get('top5_pnl_share') }",
            flush=True,
        )
        return packed

    dv, mcap = {}, {}
    btc_hits = 0
    for n in UNIVERSE_NS:
        dv[int(n)] = _run_one(pits_dv[int(n)], int(n), f"dv-{n}")
        btc_hits += int(dv[int(n)].get("btc_in_book_hits") or 0)
        commit()
    for n in UNIVERSE_NS:
        mcap[int(n)] = _run_one(pits_mcap[int(n)], int(n), f"mcap-{n}")
        btc_hits += int(mcap[int(n)].get("btc_in_book_hits") or 0)
        commit()
    if btc_hits:
        raise RuntimeError(f"BTC leaked into a book: hits={btc_hits}")

    choice = choose_production_universe(dv)
    ranking = mcap_overrides_volume(choice, dv, mcap)
    choice["ranking"] = ranking["ranking"]
    print(f"[HB] choice {choice}", flush=True)
    print(f"[HB] ranking {ranking}", flush=True)

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "btc_hits_total": int(btc_hits),
        "pred_sha256": pred_hash["sha256"],
        "pred_n_files": pred_hash["n_files"],
        "funding_on": False,
        "pit30_rows": int(len(pit30)),
        "extra30": extra30,
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_universe_sensitivity(
        rep_dir / "btcb_universe_sensitivity.md",
        choice=choice,
        ranking=ranking,
        dv=dv,
        mcap=mcap,
        extra=extra,
    )
    plot_three_equity(dv, chart_dir / "btcb_universe_sens_equity.png")
    payload = {
        "criterion": UNIVERSE_SENS_CRITERION,
        "funding_caveat": PHASE3_FUNDING_CAVEAT,
        "funding_on": False,
        "choice": _jsonable(choice),
        "ranking": _jsonable(ranking),
        "dv": {str(k): _jsonable(v) for k, v in dv.items()},
        "mcap": {str(k): _jsonable(v) for k, v in mcap.items()},
        "pred_hash": {"sha256": pred_hash["sha256"], "n_files": pred_hash["n_files"]},
        "extra": _jsonable(extra),
        "gpu_used": False,
    }
    (rep_dir / "btcb_universe_sensitivity.json").write_text(json.dumps(payload, indent=2, default=str))
    commit()

    u = choice["chosen_u"]
    print(f"CHOSEN UNIVERSE: top-{u} ranking={ranking.get('ranking')} fallback={choice.get('fallback')}", flush=True)
    for n in UNIVERSE_NS:
        print(
            f"DV top-{n}: full={dv[n].get('net_sharpe')} trail={dv[n].get('net_sharpe_trail18m')}",
            flush=True,
        )
    print(
        f"RankIC DV: 30={dv[30].get('rankic')} 50={dv[50].get('rankic')} 100={dv[100].get('rankic')}",
        flush=True,
    )
    print(
        f"RankIC mcap: 30={mcap[30].get('rankic')} 50={mcap[50].get('rankic')} 100={mcap[100].get('rankic')}",
        flush=True,
    )
    print("FUNDING=0. COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "chosen_u": u,
        "ranking": ranking.get("ranking"),
        "fallback": bool(choice.get("fallback")),
        "dv_full": {str(n): dv[n].get("net_sharpe") for n in UNIVERSE_NS},
        "dv_trail": {str(n): dv[n].get("net_sharpe_trail18m") for n in UNIVERSE_NS},
        "rankic_dv": {str(n): dv[n].get("rankic") for n in UNIVERSE_NS},
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "pred_sha256": pred_hash["sha256"],
    }


@app.local_entrypoint()
def main():
    print("[local] starting universe sensitivity (spawn, then wait)...", flush=True)
    fc = run_btcb_univ.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    Path("universe").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_universe_sensitivity.md", "reports"),
        ("reports/btcb_universe_sensitivity.json", "reports"),
        ("charts/btcb_universe_sens_equity.png", "charts"),
        ("btcb/universe/btcb_top30_floor.parquet", "universe"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            out = Path(kind) / name
            out.parent.mkdir(exist_ok=True)
            shutil.copy2(candidate, out)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("btcb_universe*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_universe*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] universe sensitivity complete.", flush=True)
