"""
PROJECT FORGE F0 — evolutionary strategy miner.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot per stage.
Frozen products untouched. Master only.
Usage:
  modal run --detach btcb_forge_f0_pipeline.py --mode mine
  modal run btcb_forge_f0_pipeline.py --mode judge
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-forge-f0"
VOL_Q = "quant-baseline"
quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "matplotlib",
        "httpx",
        "pyyaml",
        "scikit-learn",
        "numba",
        "lightgbm",
    )
    .env({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/forge_f0_addendum.md", remote_path="/root/forge_f0_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
)
_champs = Path("reports/forge_f0_champions.json")
if _champs.exists():
    _image = _image.add_local_file(str(_champs), remote_path="/root/forge_f0_champions.json")
_sha = Path("reports/forge_f0_code_sha.txt")
if _sha.exists():
    _image = _image.add_local_file(str(_sha), remote_path="/root/forge_f0_code_sha.txt")

image = _image
app = modal.App(APP_NAME, image=image)
CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class KillHeartbeat:
    """60s heartbeat; KILL the process if silent for 20 minutes (after optional checkpoint)."""

    def __init__(self, stage: str, ckpt_fn=None, silence_sec: float = 20 * 60):
        self.stage = stage
        self.ckpt_fn = ckpt_fn
        self.silence_sec = float(silence_sec)
        self.t0 = time.time()
        self.last = time.time()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        print(f"[HB] STAGE START {stage}", flush=True)

    def ping(self, msg: str = "") -> None:
        self.last = time.time()
        extra = f" {msg}" if msg else ""
        print(f"[HB] {self.stage}{extra} elapsed={time.time() - self.t0:.0f}s", flush=True)

    def _run(self) -> None:
        while not self.stop.wait(60.0):
            now = time.time()
            silent = now - self.last
            print(
                f"[HB] {self.stage} heartbeat elapsed={now - self.t0:.0f}s silent={silent:.0f}s",
                flush=True,
            )
            if silent >= self.silence_sec:
                print(
                    f"[KILL] STAGE {self.stage} silent={silent:.0f}s ≥ {self.silence_sec:.0f}s — checkpoint then exit",
                    flush=True,
                )
                if self.ckpt_fn is not None:
                    try:
                        self.ckpt_fn()
                    except Exception as e:
                        print(f"[KILL] checkpoint failed: {e}", flush=True)
                os._exit(2)

    def close(self) -> None:
        self.stop.set()
        print(f"[HB] STAGE END {self.stage} elapsed={time.time() - self.t0:.0f}s", flush=True)


def _load_close_long(raw_dir: Path, symbols: list[str]):
    import pandas as pd

    parts = []
    for sym in symbols:
        pq = raw_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if df is None or df.empty or "close" not in df.columns:
            continue
        if "symbol" not in df.columns:
            df["symbol"] = sym
        parts.append(df[["date", "close", "symbol"]])
    if not parts:
        return pd.DataFrame(columns=["date", "close", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"])


def _code_sha() -> str:
    for p in (Path("/root/forge_f0_code_sha.txt"), Path("reports/forge_f0_code_sha.txt")):
        if p.exists():
            return p.read_text().strip()
    return "UNKNOWN"


def _load_champions() -> dict:
    for p in (Path("/root/forge_f0_champions.json"), Path("reports/forge_f0_champions.json")):
        if p.exists():
            return json.loads(p.read_text())
    raise RuntimeError("committed champions json missing; JUDGE cannot run")


@app.function(
    timeout=7 * 60 * 60,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=32,
    memory=131072,
)
def run_forge(mode: str = "mine") -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    import numpy as np
    import pandas as pd

    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        FORGE_BUDGET_SEC,
        FORGE_CRITERION,
        FORGE_FITNESS,
        FORGE_GENS,
        FORGE_HEADLINE_K,
        FORGE_HEADLINE_REBAL,
        FORGE_HEADLINE_UNIV,
        FORGE_JUDGE_END,
        FORGE_JUDGE_START,
        FORGE_MINE_END,
        FORGE_MINE_START,
        FORGE_NONCONTAMINATION,
        FORGE_NULL_GENS,
        FORGE_POP,
        FORGE_SELECT_END,
        FORGE_SELECT_START,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
    )
    from btcb.features import btc_id_from_panel
    from btcb.forge import (
        champion_passes,
        cube_dir,
        eval_tree,
        fitness_details,
        formula_str,
        ingredient_census,
        load_cube,
        mechanical_forge_verdict,
        null_select_floor,
        parse_formula,
        prepare_cubes,
        run_book,
        run_ew_book,
        save_cube,
        select_champions,
        shuffle_fwd_vol,
        summarize_net_book,
        evolve,
    )
    from btcb.forge_report import (
        jsonable,
        plot_fitness_gen,
        plot_judge_equity,
        plot_mine_vs_select,
        update_ledger_forge,
        write_forge_f0,
    )
    from btcb.hygiene import clean_panel
    from btcb.manuel2 import cmc_close_wide, hybrid_close_wide
    from btcb.oracle_ladder import _utc_idx, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4v2 import collapse_fold_preds, preds_to_score_at
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from btcb.academic_factor import pit_members

    t0 = time.time()
    mode = str(mode or "mine").strip().lower()
    addendum = Path("/root/forge_f0_addendum.md").read_text()
    for txt in (FORGE_NONCONTAMINATION, FORGE_FITNESS, FORGE_CRITERION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"FORGE addendum missing freeze text: {txt[:80]}")
    print("[HB] FORGE F0 BACKTEST ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {FORGE_NONCONTAMINATION}", flush=True)
    print(f"[HB] {FORGE_FITNESS}", flush=True)
    print(f"[HB] {FORGE_CRITERION}", flush=True)
    print(f"[HB] mode={mode}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    hb = KillHeartbeat(mode)
    ckpt_holder = {"fn": None}

    def _ckpt():
        fn = ckpt_holder.get("fn")
        if fn:
            fn()

    hb.ckpt_fn = _ckpt

    def ping(msg=""):
        hb.ping(msg)

    try:
        end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
        panel = pd.read_parquet(CMC_PANEL)
        panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        panel["id"] = panel["id"].astype(int)
        panel = panel[panel["date"] <= end].copy()
        btc_id = btc_id_from_panel(panel)
        ping(f"btc_id={btc_id} rows={len(panel)}")

        pit = None
        for p in (
            Path("/data/quant/btcb/universe/btcb_top100_floor.parquet"),
            Path("/data/quant/universe/btcb_top100_floor.parquet"),
            Path("/root/btcb_top100_floor.parquet"),
        ):
            if p.exists():
                pit = pd.read_parquet(p)
                pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                pit["id"] = pit["id"].astype(int)
                pit = pit[pit["date"] <= end].copy()
                ping(f"pit from {p} rows={len(pit)}")
                break
        if pit is None:
            raise RuntimeError("missing floored PIT top-100")

        ping("re-applying frozen 2.b cleaner (no CMC writes)")
        cleaned, _ = clean_panel(panel, btc_id=btc_id)
        cleaned = cleaned[cleaned["date"] <= end].copy()

        pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
        pred_hash = hash_pred_dir(pred_dir_2c)
        ping(f"2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}")
        if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
            raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")

        ping("loading Binance spot close (hybrid)")
        spot_dir = Path("/data/quant/raw/spot_klines")
        spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
        all_ids = sorted(set(int(i) for i in cleaned["id"].unique()) | {int(btc_id)})
        id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), set(spot_syms))
        id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in set(spot_syms) else id_to_spot.get(int(btc_id))
        spot_needed = sorted({s for s in id_to_spot.values() if s})
        spot_long = _load_close_long(spot_dir, spot_needed)
        spot_wide = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
        cmc_px = cmc_close_wide(cleaned)
        close = hybrid_close_wide(cmc_px, spot_wide)
        if int(btc_id) not in close.columns:
            raise RuntimeError("BTC missing from hybrid close")
        close = close[close.index <= end].sort_index()
        close.index = _utc_idx(close.index)
        btc_ok = close[int(btc_id)].astype(float)
        close = close.loc[np.isfinite(btc_ok) & (btc_ok > 0)].copy()
        ping(f"hybrid close {close.shape} {close.index.min().date()}→{close.index.max().date()}")

        work = cube_dir()
        work.mkdir(parents=True, exist_ok=True)
        cube_path = work / "cubes.npz"
        if cube_path.exists() and mode != "prepare":
            ping("loading existing cubes")
            cube = load_cube(cube_path)
        else:
            ping("preparing primitive cubes")
            cube = prepare_cubes(
                panel=panel,
                cleaned=cleaned,
                pit=pit,
                close=close,
                btc_id=int(btc_id),
                ping=ping,
            )
            save_cube(cube, cube_path)
            quant_vol.commit()
            ping("cubes saved")

        if mode == "prepare":
            hb.close()
            return {"mode": "prepare", "n_dates": int(len(cube.dates)), "n_ids": int(len(cube.ids))}

        mine_cube = cube.slice_window(FORGE_MINE_START, FORGE_MINE_END)
        select_cube = cube.slice_window(FORGE_SELECT_START, FORGE_SELECT_END)
        judge_cube = cube.slice_window(FORGE_JUDGE_START, FORGE_JUDGE_END)
        ping(
            f"windows mine={len(mine_cube.dates)} select={len(select_cube.dates)} judge={len(judge_cube.dates)}"
        )

        n_jobs = min(32, int(os.cpu_count() or 1), 50)
        smoke = mode == "smoke"
        n_pop = 16 if smoke else int(FORGE_POP)
        n_gen = 2 if smoke else int(FORGE_GENS)
        n_null = 1 if smoke else int(FORGE_NULL_GENS)

        extra = {
            "pred_sha256": pred_hash["sha256"],
            "cmc_panel_sha256": cmc_panel_sha0,
            "cmc_readonly_ok": cmc_panel_sha0 == CMC_PANEL_SHA256,
            "gpu_used": False,
            "n_jobs": n_jobs,
            "mean_dv": cube.meta.get("mean_dv"),
            "mean_mcap": cube.meta.get("mean_mcap"),
            "code_sha": _code_sha(),
            "smoke": smoke,
        }

        rep_dir = Path("/data/quant/reports")
        chart_dir = Path("/data/quant/charts")
        for d in (work, rep_dir, chart_dir):
            d.mkdir(parents=True, exist_ok=True)

        if mode in ("mine", "smoke"):
            ckpt_path = work / ("checkpoint_smoke.json" if smoke else "checkpoint_mine.json")

            def _save_progress():
                quant_vol.commit()

            ckpt_holder["fn"] = _save_progress
            ping(f"MINE start pop={n_pop} gens={n_gen} jobs={n_jobs}")
            mined = evolve(
                mine_cube,
                n_pop=n_pop,
                n_gen=n_gen,
                checkpoint_path=ckpt_path,
                ping=ping,
                n_jobs=n_jobs,
                t0=t0,
                budget_sec=float(FORGE_BUDGET_SEC),
                label="mine",
            )
            quant_vol.commit()
            ping("SELECT")
            selected = select_champions(
                mined["population"],
                mined["scored"],
                select_cube,
                ping=ping,
                n_jobs=n_jobs,
            )
            census = ingredient_census(selected.get("top50") or [])
            ping(f"census {census.get('one_liner')}")

            null_payload = {
                "best_select_fitness": None,
                "best_expr": None,
                "n_gen": n_null,
                "skipped": False,
            }
            if (time.time() - t0) >= float(FORGE_BUDGET_SEC):
                null_payload["skipped"] = True
                extra["null_budget_flag"] = True
                ping("NULL skipped (budget)")
            else:
                ping("NULL arm")
                null_cube = shuffle_fwd_vol(mine_cube)
                null_ck = work / ("checkpoint_null_smoke.json" if smoke else "checkpoint_null.json")
                nulled = evolve(
                    null_cube,
                    n_pop=n_pop,
                    n_gen=n_null,
                    seed=42,
                    checkpoint_path=null_ck,
                    ping=ping,
                    n_jobs=n_jobs,
                    t0=t0,
                    budget_sec=float(FORGE_BUDGET_SEC),
                    label="null",
                )
                extra["null_budget_flag"] = bool(nulled.get("budget_flag"))
                floor = null_select_floor(
                    nulled["population"],
                    nulled["scored"],
                    select_cube,
                    ping=ping,
                    n_jobs=n_jobs,
                )
                null_payload.update(floor)
                null_payload["n_gen"] = n_null
                extra["null_history"] = nulled.get("history")

            extra["elapsed_sec"] = time.time() - t0
            extra["budget_flag"] = bool(mined.get("budget_flag"))
            extra["last_generation"] = (mined.get("history") or [{}])[-1].get("generation") if mined.get("history") else None
            extra["best_mine_expr"] = (mined.get("history") or [{}])[-1].get("best_expr") if mined.get("history") else None
            extra["champions_sha"] = "PENDING_COMMIT"

            champs = selected.get("champions") or []
            payload = {
                "stage": "pre_judge",
                "noncontamination": FORGE_NONCONTAMINATION,
                "fitness": FORGE_FITNESS,
                "criterion": FORGE_CRITERION,
                "champions": champs,
                "decay_top20": selected.get("decay_top20"),
                "top50": selected.get("top50"),
                "census": census,
                "null": jsonable(null_payload),
                "history": mined.get("history"),
                "extra": jsonable(extra),
                "judge_touched": False,
            }
            (work / "forge_f0_champions.json").write_text(json.dumps(payload, indent=2, default=str))
            (rep_dir / "forge_f0_champions.json").write_text(json.dumps(payload, indent=2, default=str))
            write_forge_f0(
                rep_dir / "forge_f0.md",
                addendum=addendum,
                extra=extra,
                history=mined.get("history"),
                decay=selected.get("decay_top20"),
                champions=champs,
                census=census,
                null=null_payload,
                stage="pre_judge",
            )
            (rep_dir / "forge_f0.json").write_text(json.dumps(jsonable(payload), indent=2, default=str))
            (rep_dir / "forge_f0_addendum.md").write_text(addendum)
            plot_fitness_gen(mined.get("history") or [], chart_dir / "forge_f0_fitness_gen.png")
            plot_mine_vs_select(selected.get("decay_top20") or [], chart_dir / "forge_f0_mine_vs_select.png")
            quant_vol.commit()

            print("FORGE F0 PRE-JUDGE (JUDGE not touched)", flush=True)
            for ch in champs:
                print(
                    f"C{ch.get('rank')} nodes={ch.get('nodes')} MINE={ch.get('mine_fitness')} "
                    f"SELECT={ch.get('select_fitness')} `{ch.get('expr')}`",
                    flush=True,
                )
            print(
                f"null floor SELECT={null_payload.get('best_select_fitness')} vs champions",
                flush=True,
            )
            print(f"census {census.get('one_liner')}", flush=True)
            print(f"[HB] DONE mode={mode} elapsed={time.time() - t0:.1f}s gpu=false", flush=True)
            hb.close()
            return {
                "mode": mode,
                "n_champions": len(champs),
                "null_floor": null_payload.get("best_select_fitness"),
                "census": census.get("one_liner"),
                "budget_flag": extra.get("budget_flag"),
                "elapsed_sec": time.time() - t0,
                "gpu_used": False,
            }

        if mode != "judge":
            raise RuntimeError(f"unknown mode {mode}")

        frozen = _load_champions()
        if frozen.get("judge_touched"):
            raise RuntimeError("champions json already marked judge_touched")
        champs = frozen.get("champions") or []
        if not champs:
            raise RuntimeError("no committed champions")
        ping(f"JUDGE loading {len(champs)} committed champions")
        for ch in champs:
            parse_formula(ch["expr"])
            print(f"[HB] FROZEN C{ch.get('rank')} `{ch.get('expr')}`", flush=True)

        judge_rows = []
        for ch in champs:
            tree = parse_formula(ch["expr"])
            score = eval_tree(tree, judge_cube.prims, judge_cube.mask_dv)
            packed = run_book(
                score,
                judge_cube,
                k=int(FORGE_HEADLINE_K),
                rebal_mode=str(FORGE_HEADLINE_REBAL),
                univ=str(FORGE_HEADLINE_UNIV),
            )
            book = summarize_net_book(packed, judge_cube)
            ping(
                f"JUDGE C{ch.get('rank')} total={book.get('total')} rel={book.get('rel_sharpe')} "
                f"maxdd={book.get('maxdd')}"
            )
            judge_rows.append({"rank": ch.get("rank"), "expr": ch["expr"], "nodes": ch.get("nodes"), "book": book})

        btc_rets = None
        if judge_rows:
            btc_rets = judge_rows[0]["book"].get("btc_ret")
        from btcb.manuel2 import btc_bh_book

        idx = judge_rows[0]["book"]["daily_ret"].index if judge_rows else judge_cube.dates[1:]
        btc = btc_bh_book(close, int(btc_id), pd.DatetimeIndex(idx))
        per = []
        for row in judge_rows:
            bits = champion_passes(row["book"], btc)
            row["pass_bits"] = bits
            row["book_total"] = bits["book_total"]
            row["btc_total"] = bits["btc_total"]
            row["rel_sharpe"] = bits["rel_sharpe"]
            row["maxdd"] = bits["maxdd"]
            row["btc_maxdd"] = bits["btc_maxdd"]
            per.append(bits)
        verdict = mechanical_forge_verdict(per)

        ping("JUDGE benchmarks EW + frozen-spread")
        ew_dv = summarize_net_book(run_ew_book(judge_cube, "dv100", "daily"), judge_cube)
        ew_mc = summarize_net_book(run_ew_book(judge_cube, "mcap50", "daily"), judge_cube)

        from btcb.academic_factor import pit_members as _pm
        from btcb.oracle_ladder import ffill_members as _ff

        twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
        twin = twin[twin["date"] <= end].copy()
        frozen_pred = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
        members_pit = _ff(_pm(pit, btc_id), list(close.index))
        oos = [d for d in list(close.index) if d >= pd.Timestamp(FORGE_JUDGE_START, tz="UTC")]
        pairs14 = formation_dates(oos, 14)
        scores = preds_to_score_at(frozen_pred, "spread", [t for t, _, _ in pairs14])
        spr = run_periodic_long(
            close,
            members_pit,
            btc_id,
            scores,
            pairs14,
            cost_bps=float(ALT_BPS),
            label="frozen_spread",
        )
        spr_rets = spr["daily_ret"]
        spr_rets.index = _utc_idx(spr_rets.index)
        from btcb.book import summarize_book as _sum

        btc_r = btc["daily_ret"].reindex(spr_rets.index).fillna(0.0)
        z = pd.Series(0.0, index=spr_rets.index)
        nn = pd.Series(float("nan"), index=spr_rets.index)
        spr_sum = _sum(spr_rets, btc_r, z, nn, [])
        spr["rel_sharpe"] = spr_sum.get("rel_sharpe")
        spr["total"] = spr.get("total", spr_sum.get("book_total"))
        spr["sharpe"] = spr_sum.get("book_sharpe")
        spr["maxdd"] = spr_sum.get("maxdd")
        spr["corr_btc"] = float(spr_rets.corr(btc_r)) if len(spr_rets) >= 10 else float("nan")
        spr["equity"] = spr_sum.get("equity")
        spr["daily_ret"] = spr_rets
        spr["book_total"] = spr.get("total")

        bench = {"btc_bh": btc, "ew_dv100": ew_dv, "ew_mcap50": ew_mc, "frozen_spread": spr}
        extra["elapsed_sec"] = time.time() - t0
        extra["budget_flag"] = False
        extra["champions_sha"] = _code_sha()
        extra["judge_start"] = FORGE_JUDGE_START
        extra["judge_end"] = FORGE_JUDGE_END

        one_liners = []
        for row in judge_rows:
            bits = row["pass_bits"]
            line = (
                f"C{row.get('rank')} `{row.get('expr')}` total={bits.get('book_total')} "
                f"vs BTC={bits.get('btc_total')} relSharpe={bits.get('rel_sharpe')} "
                f"MaxDD={bits.get('maxdd')} pass={bits.get('pass')}"
            )
            one_liners.append(line)
            print(line, flush=True)

        census_line = (frozen.get("census") or {}).get("one_liner", "")
        null_floor = (frozen.get("null") or {}).get("best_select_fitness")
        print(verdict.get("label"), flush=True)
        print(f"null floor SELECT={null_floor} vs champions SELECT", flush=True)
        print(f"census {census_line}", flush=True)

        ledger_path = Path("/root/numbers_ledger.md")
        update_ledger_forge(
            ledger_path,
            verdict=verdict,
            extra={
                "null_floor": null_floor,
                "census_line": census_line,
                "champion_one_liners": "\n".join(f"- {x}" for x in one_liners),
            },
        )
        (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())

        write_forge_f0(
            rep_dir / "forge_f0.md",
            addendum=addendum,
            extra=extra,
            history=frozen.get("history"),
            decay=frozen.get("decay_top20"),
            champions=champs,
            census=frozen.get("census") or {},
            null=frozen.get("null") or {},
            judge={"champions": judge_rows},
            bench=bench,
            verdict=verdict,
            stage="judge",
        )
        plot_fitness_gen(frozen.get("history") or [], chart_dir / "forge_f0_fitness_gen.png")
        plot_mine_vs_select(frozen.get("decay_top20") or [], chart_dir / "forge_f0_mine_vs_select.png")
        plot_judge_equity(judge_rows, btc, chart_dir / "forge_f0_judge_equity.png")
        out_json = {
            "stage": "judge",
            "verdict": verdict,
            "champions": jsonable(champs),
            "judge": jsonable({"champions": judge_rows}),
            "bench": jsonable(bench),
            "null": frozen.get("null"),
            "census": frozen.get("census"),
            "history": frozen.get("history"),
            "extra": jsonable(extra),
            "criterion": FORGE_CRITERION,
            "fitness": FORGE_FITNESS,
            "noncontamination": FORGE_NONCONTAMINATION,
        }
        (rep_dir / "forge_f0.json").write_text(json.dumps(out_json, indent=2, default=str))
        (rep_dir / "forge_f0_addendum.md").write_text(addendum)
        (work / "forge_f0_judge.json").write_text(json.dumps(out_json, indent=2, default=str))
        quant_vol.commit()
        print(f"[HB] DONE mode=judge elapsed={time.time() - t0:.1f}s gpu=false verdict={verdict.get('label')}", flush=True)
        hb.close()
        return {
            "mode": "judge",
            "label": verdict.get("label"),
            "n_pass": verdict.get("n_pass"),
            "elapsed_sec": time.time() - t0,
            "gpu_used": False,
        }
    except Exception:
        hb.close()
        raise


def _pull_artifacts():
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/forge_f0.md", "reports"),
        ("reports/forge_f0.json", "reports"),
        ("reports/forge_f0_champions.json", "reports"),
        ("reports/forge_f0_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/forge_f0_fitness_gen.png", "charts"),
        ("charts/forge_f0_mine_vs_select.png", "charts"),
        ("charts/forge_f0_judge_equity.png", "charts"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            Path(kind).mkdir(exist_ok=True)
            shutil.copy2(candidate, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in Path("charts").glob("forge_f0*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in Path("reports").glob("forge_f0*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = Path("reports/numbers_ledger.md")
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())


@app.local_entrypoint()
def main(mode: str = "mine"):
    print(f"[local] FORGE F0 mode={mode}", flush=True)
    fc = run_forge.spawn(mode)
    summary = fc.get()
    _pull_artifacts()
    print(json.dumps(summary, indent=2, default=str))
    print("[local] FORGE F0 complete.", flush=True)
