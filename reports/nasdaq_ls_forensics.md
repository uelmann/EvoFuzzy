# Why the Nasdaq LightGBM long-short failed (and why crypto COMBO is not a counterexample)

**Not a product. Not a new freeze.** Diagnosis of NASDAQ-LS / NASDAQ-LS21 vs frozen crypto COMBO. Numbers: `reports/nasdaq_ls_forensics.json`.

The Yahoo file is usable. An oracle and a textbook 12–1 momentum factor both work on it. Crypto alpha did not “vanish” on stocks. We copied the booster and dropped the parts that make COMBO a book: **BTC hedge, τ (can be flat), a universe that dies, and a 7–10 day residual that actually exists in perps.** On Nasdaq we trained a QQQ-residual, then paid ourselves in unhedged dollars among **today’s index winners**. The model learned **low-beta short-term reversal**. That is the crypto pattern. It is the wrong factor on this panel. The long leg is fine. The short leg is the whole loss.

---

## Direct answers

**Are the data wrong?** No, not in a way that explains Sharpe −0.5. Adj Close is wired. Splits do not jump. 0 negative prices, 0 duplicate rows. AAPL 4-for-1 on 2020-08-31: Adj 121.0 → 125.1. Oracle lookahead LS on the same PIT-30: Sharpe **12.6** (h=10) / **9.4** (h=21). If ranking were correct, this universe prints.

**Is there no stock alpha in this file?** There is. Classic **12–1 momentum**, same PIT top-30, long/short 10%, costless 1-day: Sharpe **0.57**, total **+504%**. RankIC of 12–1 vs simple 21d return = **+0.020**. 21-session reversal on the same names: Sharpe **−0.03**. The crypto-ish 2–4 week fade is not an equity factor here. 6–12 month momentum is.

**Then how can crypto have alpha and this not?** COMBO is not “LightGBM on 33 columns.” It is residual-vs-BTC **plus** a daily BTC hedge **plus** τ (flat unless |score| is extreme) **plus** a PIT universe where losers disappear. We ran residual-vs-QQQ **without** a QQQ overlay, **always in**, among names that **won** for twenty years. That is the equity analog of training residual vs BTC and then running an unhedged long-short through a BTC bull.

**Would a longer/shorter horizon have saved it?** Horizon was not the first-order bug. h=10 residual RankIC is real but small (**+0.022**, NW-t 3.14). h=21 is noise (**+0.008**, NW-t 1.07). Stretching 10 → 21 made RankIC **worse**, not better. The clock that already pays on this file is **252 skip 21**, not 10 or 21 sessions.

**Would the COMBO hedge have saved it?** Partially, not enough. See §3.

---

## 1. The data are not the failure

| check | result |
|--------|--------|
| Working close | Yahoo **Adj Close** (splits + dividends). `close == adj_close` on every row |
| Adj ≠ raw Close | 69% of rows (dividends). Intended |
| AAPL 2020-08-31 4-for-1 | Adj 121.0 → 125.1. No 4× jump. Yahoo Close is already split-adjusted |
| Neg/zero prices | 0 |
| Duplicate date×symbol | 0 |
| Span | 1990-01-02 → 2026-08-14, 100 names, 653k rows |
| Names alive in 2005 | median **67** / 100. The rest IPO later (PLTR, ARM, ABNB, …) |

One Yahoo quirk: `EA` starts 2026-07-17 (ticker reuse / mapping). Not material to 2007–2025 PnL.

**Survivorship is real and it is a bias**, not the Sharpe −0.5 by itself. 12–1 still works on the same survivor list. What survivorship does is make **always-in shorts** among today’s winners a structurally ugly book (NVDA/NFLX/MU are the top name-PnL holes on NASDAQ-LS21). Crypto PIT includes coins that go to zero. This list does not.

---

## 2. What the model actually learned

From 2007, PIT top-30, OOS scores:

| | h=10 expanding (v1) | h=21, 5y rolling (v21) |
|--|---------------------|-------------------------|
| RankIC vs **residual** y (train target) | **+0.022** (NW-t 3.14) | **+0.008** (NW-t 1.07) |
| RankIC vs **simple** 10d/21d return (what the book is paid in) | **−0.009** | **−0.005** |
| Spearman(score, QQQ-beta) | **−0.46** (NW-t −53) | **−0.24** (NW-t −24) |
| Spearman(score, ret_7) | **−0.26** | **−0.11** |
| Spearman(score, rev_1) = −yesterday | **+0.24** | **+0.07** |
| Spearman(score, mom_90_skip14) | **−0.06** | **−0.05** |

Crypto A0 RankIC vs its residual/ratio target is about **0.06**. Here the residual RankIC is **one-third to one-eighth** of that, and the dollar RankIC has the **wrong sign**.

Quintile mean **simple** forward return (1 = lowest score = our shorts):

- h=10: Q1 **1.09%** > Q5 **0.82%**
- h=21: Q1 **2.36%** > Q5 **1.91%**

Low score → **higher** subsequent dollar return. Residual quintiles are almost flat and not monotone (h=10 Q4 is the best residual, not Q5). There is no residual spread to harvest even if the book were paid in residual.

**Mechanism.** Label is `y = r_i − β_QQQ · r_QQQ`. After de-betaing, high y prefers **low-beta recent losers** (crypto fade). The booster learns that: inverse beta, inverse 7–28d return, positive 1-day reversal. The book then **longs low-beta and shorts high-beta** with no QQQ overlay, for twenty years of QQQ compounding.

Mean QQQ-beta of the 1-day 10% book (h=10): longs **0.98**, shorts **1.53**, portfolio beta **−0.27**. OLS beta of that LS vs QQQ: **−0.15**. Short NVDA/NFLX/MU is not a backtest bug. It is the residual label, unhedged.

Crypto A0 features `ret_7/14/28/90` are 7–90 **calendar** days on 24/7 perps. On this panel the same integers are **sessions** (≈ 1.5 weeks–4.5 months). Equity momentum that already works here is **12–1** (252 skip 21). We asked the model for a crypto fade and it found one. 21d reversal Sharpe on this file is ≈ 0. The paying factor is the opposite direction and a longer clock.

---

## 3. Longs work. Shorts are the product. Hedge helps, it does not create COMBO.

Equal-weight, 1-day hold, from 2007, **costless**:

| book | Sharpe long | Sharpe short | Sharpe LS | COMBO-style QQQ overlay Sharpe | mean port β |
|------|-------------|--------------|-----------|--------------------------------|-------------|
| h=10, top/worst 10% (k≈3) | **+0.80** | **−0.90** | −0.24 | **+0.06** | −0.27 |
| h=10, k=10 (v1 shape) | **+0.91** | **−0.79** | −0.12 | **+0.39** | −0.24 |
| h=21, top/worst 10% | **+0.62** | **−0.94** | −0.42 | **−0.33** | −0.14 |

Overlay = dollar LS − (portfolio β) × r_QQQ, same construction COMBO uses vs BTC. Because the book is already short beta, the overlay **buys QQQ**. That cancels part of the bull-market hole. It does **not** turn a 0.02 residual IC into a 1.7 Sharpe book. Residual quintiles stay flat. h=21 stays negative even hedged.

Long-only top 10% (no shorts), costless 1d: Sharpe **0.80**, excess vs QQQ **+0.40** (h=10). The ranking has mild long-side content. Forcing a dollar-neutral short book in a megacap bull turns it into a negative-beta machine.

Recorded overlapping books (5 bps, the actual scouts): Sharpe **−0.49** (h=10 k=10) and **−0.58** (h=21 10%). Cost drag 0.27 / 0.20 in return units. Costs make it worse; costless LS is already negative. Top name holes on v21: NVDA, NFLX, MU.

---

## 4. Crypto COMBO is not what we copied

Frozen COMBO (`reports/system_card.md`) Sharpe 1.711 is **not** “LightGBM, always in, top 10 minus worst 10, no hedge.”

| | Crypto COMBO (what worked) | Nasdaq scout (what we ran) |
|--|----------------------------|----------------------------|
| Label | residual vs **BTC** | residual vs **QQQ** |
| Book hedge | **trailing BTC beta, every day** | **none** |
| Entry | **τ-threshold**. Can be flat. P1 ≈ 6 names, P2 ≈ 16 | **always-in** 10 names or 10% every session |
| Universe | PIT top-120 train / top-20 or 40 exec, **churn**, losers disappear | **today’s NDX 100**, 67 names already alive in 2005 that **won** |
| Clock | `ret_7…90` = 7–90 **calendar** days on 24/7 perps | same integers = 7–90 **sessions** |
| Horizon that paid | h=7 and h=10 with BTC residual **and** hedge | 10 and 21 sessions; equity momentum here is **6–12 months** |
| Residual RankIC | A0 ≈ **0.06** | **0.022** / **0.008** |
| Shorts | coins that often go to zero | NVDA/META/TSLA-class survivors |
| Cross-section | high idio vol after BTC | median CS std of 21d simple ≈ 7.7%; after QQQ residual still 7.1%, **score does not rank it** |

Crypto alpha is: **BTC-residual signal + BTC-hedged sparse book + a universe that dies.**  
It is not a theorem that 33 A0 columns + Huber produce CS alpha on any 100 names.

Copying the booster and dropping the hedge is the equity analog of training residual vs BTC and then running unhedged LS through a BTC bull.

---

## 5. Same-panel controls (so this is not “stocks have no CS”)

| control | Sharpe | RankIC vs simple 21d | RankIC vs residual 21d |
|---------|--------|----------------------|------------------------|
| 12–1 momentum LS 10% | **+0.57** | **+0.020** | −0.005 |
| 21d reversal LS 10% | −0.03 | — | — |
| LightGBM h=21 | −0.58 recorded | **−0.005** | +0.008 |
| Oracle simple LS 10% | +9.4 | (lookahead) | — |

12–1’s RankIC vs simple 21d (**0.020**) is the same *size* as LightGBM’s residual RankIC (0.022 at h=10). 12–1 **pays in dollars** because label and book match. LightGBM’s residual RankIC is real and small, then the unhedged book pays the opposite dollar.

---

## 6. What to adapt (if the next run is a test, not another knob)

Do not retune 10 vs 21 vs 5y vs 500 trees and call it research. The failure is **label/book mismatch + crypto clock + always-in shorts among survivors**. A next scout is only worth running if it changes those, with a freeze addendum, and with 12–1 as the **control that must still print**.

1. **Align label and book.** Either (a) residual vs QQQ **and** overlay a QQQ/NDX beta hedge like COMBO, or (b) predict **simple** return and do not pretend it is market-neutral. Mixing (a)’s label with (b)’s book is how you get IC_resid > 0 and IC_simple < 0. Hedge-on-copy of v1 k=10 is costless Sharpe **+0.39**, not 1.7. Necessary, not sufficient.
2. **Do not force always-in shorts.** COMBO only shorts when score < −τ. On this panel, long-only top 10% already has +0.40 excess Sharpe vs QQQ (h=10, costless daily). That is the first analog of “is there a factor.”
3. **Point-in-time index members**, not today’s NDX. Survivorship puts future winners in the 2005 shortable set.
4. **Change the clock.** Replace crypto 7/14/28/90 with **21/63/126/252**. Put **12–1** on the feature list as a control column. If LightGBM cannot beat 12–1 on the same book, there is no ML claim.
5. **Predict 6–12 months, or use 12–1 as the control book.** 21-day reversal Sharpe ≈ 0 here. Asking LightGBM for a 10–21 day residual is asking for a crypto horizon in an equity panel.
6. Yahoo Adj Close can stay. It is not the failure mode.

None of this says “equities have no CS alpha.” It says this copy asked a BTC-residual, high-churn, 10-day machine to short the winners of a 20-year QQQ bull, without the hedge that makes COMBO COMBO.

Suggested freeze name if we run one more scout: **NASDAQ-ADAPT-1** = simple (or residual+QQQ-hedge) label, 21/63/126/252 features, long-only or τ-sparse, PIT membership when available, 12–1 control. Not another always-in 10−10 on today’s list.
