"""Round F pre-registered constants."""

from baseline.features import FEATURE_COLS

LEDGER_PATH = "reports/numbers_ledger.md"

KEEP_CRITERION = (
    "Block X is KEPT on universe U only if trailing-18m ΔRankIC on U ≥ +0.005 at h=7 or h=10 "
    "AND full-OOS ΔRankIC on U ≥ 0 AND Δ positive in ≥60% of trailing-18m folds on U AND the "
    "corresponding portfolio trailing-18m net Sharpe Δ on U ≥ 0. F4 (pruning) uses the same "
    "criterion with thresholds ΔRankIC ≥ 0 (trailing) and ≥ −0.002 (full): pruning is KEPT if "
    "it does not hurt. Verdicts per-universe, mechanical, no post-hoc adjustment."
)

COMBO_CRITERION = (
    "COMBO is ADOPTED as the reference book only if its trailing-18m net Sharpe ≥ max(P1, P2 trailing) "
    "− 0.10 AND its full-period net Sharpe ≥ max(P1, P2 full) − 0.10. Otherwise the adopted book "
    "remains P2 with P1 as reference."
)

CTX_COLS = [
    "ctx_disp",
    "ctx_score_disp",
    "ctx_btc_vol",
    "ctx_btc_trend",
    "ctx_funding_agg",
    "ctx_breadth",
    "ctx_corr",
]

CATCH22_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]
C22_COLS = [f"c22_{n}" for n in CATCH22_NAMES]
EXTRA_CX_COLS = ["hurst_90", "vr_5", "perm_entropy_90", "mr_halflife_90"]
CX_COLS = C22_COLS + EXTRA_CX_COLS

# Adopted books (ledger)
P1_H, P1_TAU, P1_UNI = 7, 80.0, "top20"
P2_H, P2_TAU, P2_UNI = 10, 70.0, "top40"

A0_FEATURE_COLS = list(FEATURE_COLS)
N_PRUNE = 8
