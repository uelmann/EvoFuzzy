"""Self-contained HTML + PDF export of the Phase 2.c report.

Reads reports/btcb_phase2c_report.json and charts/btcb_p2c_*.png.
Does not recompute any verdicts.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path


def _pct(x, nd=1) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _num(x, nd=3) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img(path: Path, alt: str) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return (
        f'<figure class="chart"><img src="data:{mime};base64,{_b64(path)}" '
        f'alt="{alt}"/><figcaption>{alt}</figcaption></figure>'
    )


def _bool_pill(ok: bool, yes="YES", no="NO") -> str:
    cls = "ok" if ok else "no"
    return f'<span class="pill {cls}">{yes if ok else no}</span>'


def _rows(headers: list[str], body: list[list[str]], highlight_last=False) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    n = len(body)
    for i, row in enumerate(body):
        cls = ' class="hl"' if highlight_last and i == n - 1 else ""
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def build_html(root: Path, data: dict) -> str:
    skill = data["skill"]
    v = data["verdicts"]
    h = data["headline"]
    ng = data["null_gate"]
    extra = data.get("extra") or {}
    naive = data.get("naive_v4_record") or {}
    charts = root / "charts"

    equity = _img(charts / "btcb_p2c_equity.png", "MODEL-V3 equity vs BTC (log) · relative line · gate ON")
    rankic = _img(charts / "btcb_p2c_rankic.png", "OOS RankIC of spread vs p_top control")
    calib = _img(charts / "btcb_p2c_calibration.png", "Twin-head reliability (isotonic, OOS)")

    auc_cells = ng.get("auc_cells") or []
    ric_cells = ng.get("rankic_cells") or []
    auc_n = ng.get("auc") or {}
    ric_n = ng.get("rankic") or {}

    auc_tbl = _rows(
        ["fold", "n", "null mean", "SD", "95th", "real", "bias_ok", "exceeds p95"],
        [
            [
                str(c.get("fold_id")),
                str(c.get("n")),
                _num(c.get("mean"), 4),
                _num(c.get("sd"), 4),
                _num(c.get("p95"), 4),
                _num(c.get("real_auc"), 4),
                str(c.get("bias_ok")),
                str(c.get("exceeds_p95")),
            ]
            for c in auc_cells
        ],
    )
    ric_tbl = _rows(
        ["fold", "n", "null mean", "SD", "95th", "real", "bias_ok", "exceeds p95"],
        [
            [
                str(c.get("fold_id")),
                str(c.get("n")),
                _num(c.get("mean"), 4),
                _num(c.get("sd"), 4),
                _num(c.get("p95"), 4),
                _num(c.get("real_rankic"), 4),
                str(c.get("bias_ok")),
                str(c.get("exceeds_p95")),
            ]
            for c in ric_cells
        ],
    )

    fm = data.get("fold_metrics") or {}
    fold_rows = []
    for hz in ("14", "30"):
        for row in fm.get(hz) or []:
            fold_rows.append(
                [
                    hz,
                    str(row.get("fold_id", row.get("fold"))),
                    _num(row.get("rankic_spread"), 4),
                    _num(row.get("rankic_ptop"), 4),
                    _num(row.get("auc_spread"), 4),
                    _num(row.get("auc_ptop"), 4),
                ]
            )
    fold_tbl = _rows(
        ["h", "fold", "RankIC spread", "RankIC p_top", "AUC spread vs top-q", "AUC p_top"],
        fold_rows,
    )

    book_tbl = _rows(
        ["book", "total", "CAGR", "USD Sharpe", "rel Sharpe", "MaxDD", "avg #names", "% BTC", "gate ON", "ann TO", "forced"],
        [
            [
                f"MODEL-V3 h={h.get('horizon')} θ={h.get('theta')}",
                _pct(h.get("book_total")),
                _pct(h.get("book_cagr")),
                _num(h.get("book_sharpe")),
                _num(h.get("rel_sharpe")),
                _pct(h.get("maxdd")),
                _num(h.get("avg_n_names"), 2),
                _pct(h.get("avg_w_btc")),
                _pct(h.get("gate_on_frac")),
                _num(h.get("ann_turnover"), 2),
                str((h.get("forced_exits") or {}).get("n_events")),
            ],
            [
                "BTC B&H",
                _pct(h.get("btc_total")),
                _pct(h.get("btc_cagr")),
                _num(h.get("btc_sharpe")),
                "0.000",
                _pct(h.get("btc_maxdd")),
                "0.00",
                "100.0%",
                "0.0%",
                "0.00",
                "0",
            ],
        ],
    )

    grid_rows = []
    for g in data.get("grid") or []:
        th = g.get("theta")
        mark = " ← median" if float(th) == float(h.get("theta", 0.2)) else ""
        grid_rows.append(
            [
                f"{th}{mark}",
                _num(g.get("rel_sharpe")),
                _pct(g.get("book_total")),
                _pct(g.get("maxdd")),
                _pct(g.get("avg_w_btc")),
                _num(g.get("avg_n_names"), 2),
                _pct(g.get("gate_on_frac")),
            ]
        )
    grid_tbl = _rows(
        ["θ", "rel Sharpe", "total", "MaxDD", "% BTC", "avg #names", "gate ON"],
        grid_rows,
        highlight_last=True,
    )

    cyc_rows = []
    for name, c in (h.get("cycles") or {}).items():
        cyc_rows.append(
            [
                name,
                str(c.get("n")),
                _pct(c.get("book_total")),
                _pct(c.get("btc_total")),
                _num(c.get("book_sharpe")),
                _num(c.get("rel_sharpe")),
                _pct(c.get("maxdd")),
                _pct(c.get("avg_w_btc")),
            ]
        )
    cyc_tbl = _rows(
        ["cycle", "n", "book tot", "BTC tot", "USD Sharpe", "rel Sharpe", "MaxDD", "% BTC"],
        cyc_rows,
    )

    def _imp_tbl(items):
        rows = []
        for i, item in enumerate(items or [], start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                feat, gain = item[0], item[1]
            elif isinstance(item, dict):
                feat, gain = item.get("feature"), item.get("gain", item.get("mean_gain"))
            else:
                feat, gain = str(item), None
            rows.append([str(i), f"<code>{feat}</code>", _num(gain, 2)])
        return _rows(["rank", "feature", "mean gain"], rows)

    css = """
:root {
  --ink: #1b1f24;
  --muted: #5b6570;
  --rule: #d8dde3;
  --paper: #f7f4ee;
  --card: #fffdf8;
  --ok: #1f7a4d;
  --ok-bg: #e6f4ec;
  --no: #9b2c2c;
  --no-bg: #f8e8e8;
  --accent: #243b55;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.45;
  font-size: 15px;
}
.wrap { max-width: 920px; margin: 0 auto; padding: 28px 22px 64px; }
header.hero {
  border-bottom: 2px solid var(--accent);
  padding-bottom: 18px;
  margin-bottom: 22px;
}
.kicker { letter-spacing: .14em; text-transform: uppercase; font-size: 11px; color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; }
h1 { font-size: 28px; line-height: 1.2; margin: 6px 0 8px; font-weight: 700; }
.sub { color: var(--muted); font-size: 14px; }
.toolbar { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; font-family: ui-sans-serif, system-ui, sans-serif; }
button, .hint {
  font-size: 13px;
}
button {
  background: var(--accent); color: white; border: 0; border-radius: 6px;
  padding: 8px 14px; cursor: pointer;
}
.hint { color: var(--muted); align-self: center; }
.banner {
  background: #243b55; color: #f7f4ee; padding: 10px 14px; border-radius: 6px;
  font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12.5px; margin: 14px 0 20px;
}
.verdicts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 16px 0 22px; }
.card {
  background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px;
}
.card .lbl { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.card .val { font-size: 18px; font-weight: 700; margin-top: 4px; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 22px; }
.metrics .card .val { font-variant-numeric: tabular-nums; font-size: 16px; }
.pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; font-weight: 700; }
.pill.ok { background: var(--ok-bg); color: var(--ok); }
.pill.no { background: var(--no-bg); color: var(--no); }
h2 { font-size: 18px; margin: 28px 0 8px; border-top: 1px solid var(--rule); padding-top: 18px; }
h3 { font-size: 15px; margin: 16px 0 6px; }
blockquote {
  margin: 8px 0 16px; padding: 10px 14px; background: #efebe3; border-left: 3px solid var(--accent);
  font-size: 13.5px; color: #333;
}
p { margin: 8px 0; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: ui-sans-serif, system-ui, sans-serif; margin: 8px 0 16px; font-variant-numeric: tabular-nums; }
th { text-align: left; background: #eae4d8; font-size: 11px; letter-spacing: .03em; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 6px; }
tr.hl td { background: #e6f4ec; font-weight: 600; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
.chart { margin: 12px 0 20px; }
.chart img { width: 100%; height: auto; border: 1px solid var(--rule); background: white; }
figcaption { font-size: 12px; color: var(--muted); margin-top: 4px; font-family: ui-sans-serif, system-ui, sans-serif; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.note { font-size: 13px; color: var(--muted); }
footer { margin-top: 28px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 12px; font-family: ui-sans-serif, system-ui, sans-serif; }
@media print {
  .toolbar { display: none !important; }
  body { background: white; }
  .wrap { max-width: none; padding: 0; }
  .card, .banner, blockquote { break-inside: avoid; }
  .chart { break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
  @page { size: A4; margin: 14mm 12mm; }
}
@media (max-width: 800px) {
  .verdicts, .metrics, .two { grid-template-columns: 1fr 1fr; }
}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC-BEATER Phase 2.c — export report</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div class="kicker">BTC-BEATER · Phase 2.c · backtest only</div>
  <h1>Twin-head spread + repowered skill null</h1>
  <div class="sub">OOS {h.get('start')} → {h.get('end')} · n={h.get('n_days')} days · CPU only · COMBO untouched (v2.0-combo-final)</div>
  <div class="toolbar">
    <button onclick="window.print()">Print / Save as PDF</button>
    <span class="hint">Single file. Charts are embedded. Use the browser print dialog to export PDF.</span>
  </div>
</header>

<div class="banner">BACKTEST ONLY. Cleaned+floored 2.b data reused as-is. Stage T frozen. Context excluded. Zero GPU. A verdict is not overridden by any single cycle. Operative floor is BTC.</div>

<div class="verdicts">
  <div class="card"><div class="lbl">Spread selection skill</div><div class="val">{_bool_pill(bool(skill.get('has_skill')), 'TRUE', 'FALSE')}</div><div class="note">AUC failed the 0.52 bar</div></div>
  <div class="card"><div class="lbl">MODEL-V3 viable</div><div class="val">{_bool_pill(bool(v.get('viable')))}</div><div class="note">rel Sharpe &gt; 0, total ≥ BTC, MaxDD ≤ BTC</div></div>
  <div class="card"><div class="lbl">MODEL-V3 product-grade</div><div class="val">{_bool_pill(bool(v.get('product_grade')))}</div><div class="note">rel Sharpe ≥ 0.30 and avg alt ≥ 5%</div></div>
</div>

<div class="metrics">
  <div class="card"><div class="lbl">Rel Sharpe</div><div class="val">{_num(h.get('rel_sharpe'))}</div></div>
  <div class="card"><div class="lbl">Book vs BTC</div><div class="val">{_pct(h.get('book_total'))} / {_pct(h.get('btc_total'))}</div></div>
  <div class="card"><div class="lbl">MaxDD vs BTC</div><div class="val">{_pct(h.get('maxdd'))} / {_pct(h.get('btc_maxdd'))}</div></div>
  <div class="card"><div class="lbl">Median θ</div><div class="val">{h.get('theta')}</div></div>
  <div class="card"><div class="lbl">% time in BTC</div><div class="val">{_pct(h.get('avg_w_btc'))}</div></div>
  <div class="card"><div class="lbl">Avg alt / #names</div><div class="val">{_pct(v.get('avg_alt'))} · {_num(h.get('avg_n_names'), 2)}</div></div>
  <div class="card"><div class="lbl">Gate ON</div><div class="val">{_pct(h.get('gate_on_frac'))}</div></div>
  <div class="card"><div class="lbl">Forced exits</div><div class="val">{(h.get('forced_exits') or {}).get('n_events', 0)}</div></div>
</div>

<h2>Pre-registered criteria (verbatim, before results)</h2>
<blockquote>{data.get('criterion')}</blockquote>
<h3>Repowered skill null</h3>
<blockquote>{data.get('null_gate_text')}</blockquote>
<h3>Death-in-position convention</h3>
<blockquote>{data.get('death_convention')}</blockquote>

<h2>Mechanical verdicts</h2>
<p>
<strong>SPREAD has SELECTION SKILL: {skill.get('has_skill')}</strong>
(h=14 RankIC={_num(skill.get('rankic_h14'), 4)} AUC={_num(skill.get('auc_h14'), 4)};
h=30 RankIC={_num(skill.get('rankic_h30'), 4)} AUC={_num(skill.get('auc_h30'), 4)};
§2 spread RankIC {skill.get('null_verdict')} {skill.get('n_exceed')}/6 Stouffer z={_num(skill.get('stouffer_z'), 3)}).
</p>
<p>
(a) rel Sharpe {_num(v.get('rel_sharpe'))} &gt; 0 → {v.get('a_rel_sharpe_gt0')}.
(b) book {_pct(v.get('book_total'))} vs BTC {_pct(v.get('btc_total'))} → {v.get('b_total_ge_btc')}.
(c) MaxDD {_pct(v.get('maxdd'))} vs BTC {_pct(v.get('btc_maxdd'))} → {v.get('c_maxdd_le_btc')}.
Product-grade need rel ≥ {_num(v.get('need_product_rel'))} and alt ≥ {_pct(v.get('need_product_alt'))}; avg alt {_pct(v.get('avg_alt'))}.
</p>
<p class="note">uncertainty↔yz_vol_30 mean per-date RankIC = {_num(extra.get('uncert_vol_rankic'), 4)} (lottery diagnostic; not used in trading). Naive v4 (record only): rel Sharpe={_num(naive.get('rel_sharpe'))}, live_benchmark={naive.get('live_benchmark')}.</p>

<h2>Charts</h2>
{equity}
{rankic}
{calib}

<h2>§2 null — p_top per-date AUC (h=14)</h2>
<p>Bias pass={auc_n.get('bias_pass')}; skill {auc_n.get('verdict')}; {auc_n.get('n_exceed')}/{auc_n.get('n_folds')} exceed p95; Stouffer z={_num(auc_n.get('stouffer_z'), 3)}. Judged signal is spread RankIC, not this control.</p>
{auc_tbl}

<h2>§2 null — spread per-date RankIC (h=14, judged signal)</h2>
<p>Bias pass={ric_n.get('bias_pass')}; skill {ric_n.get('verdict')}; {ric_n.get('n_exceed')}/{ric_n.get('n_folds')} exceed p95; Stouffer z={_num(ric_n.get('stouffer_z'), 3)}. Failure = PARKED, no override, no retest with different folds.</p>
{ric_tbl}

<h2>Per-fold selection metrics (floored PIT top-100)</h2>
{fold_tbl}
<p class="note">Aggregate (last-fold-wins OOS): h=14 RankIC(spread)={_num(skill.get('rankic_h14'), 4)} RankIC(p_top)={_num(skill.get('rankic_ptop_h14'), 4)} AUC(spread)={_num(skill.get('auc_h14'), 4)}; h=30 RankIC(spread)={_num(skill.get('rankic_h30'), 4)} RankIC(p_top)={_num(skill.get('rankic_ptop_h30'), 4)} AUC(spread)={_num(skill.get('auc_h30'), 4)}.</p>

<h2>MODEL-V3 book vs BTC (same OOS window)</h2>
{book_tbl}

<h2>θ grid (h=14, median convention)</h2>
{grid_tbl}

<h2>Per-cycle honesty (headline h=14)</h2>
{cyc_tbl}

<h2>Feature importances (mean gain, h=14)</h2>
<div class="two">
  <div><h3>Head TOP</h3>{_imp_tbl(data.get('importances_top'))}</div>
  <div><h3>Head BOTTOM</h3>{_imp_tbl(data.get('importances_bot'))}</div>
</div>

<footer>
Elapsed s={_num(extra.get('elapsed_sec'), 1)}. GPU={extra.get('gpu_used', False)}. n_features={extra.get('n_features')}.
COMBO untouched (v2.0-combo-final). Source: reports/btcb_phase2c_report.json — numbers are not recomputed.
</footer>
</div>
</body>
</html>
"""
    return html


def write_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = Path("/usr/local/bin/google-chrome")
    if not chrome.exists():
        chrome = Path("/usr/bin/google-chrome")
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "reports" / "btcb_phase2c_report.json").read_text())
    html = build_html(root, data)
    html_path = root / "reports" / "btcb_phase2c_export.html"
    pdf_path = root / "reports" / "btcb_phase2c_export.pdf"
    html_path.write_text(html)
    write_pdf(html_path, pdf_path)
    print(f"wrote {html_path} ({html_path.stat().st_size} bytes)")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
