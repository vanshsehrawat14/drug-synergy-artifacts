"""Recompute the numbers stated in manuscript.tex and supplement.tex from their source artifacts.

Each entry gives the text as written, the artifact and the key path (or a computation), and the number of
decimals used in the text. A check passes when the source value rounds to the stated value and the stated text
occurs in the manuscript or the supplement (LaTeX source with \\% read as %). Exit status is non-zero if any
check fails.

python manuscript/scripts/check_numbers.py [--resolution RESOLUTION_DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "manuscript"


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def get(o, path):
    for k in path.split("/"):
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


IA = "manuscript/audit/outputs/interaction_audit.json"
PS = "manuscript/audit/outputs/pairseen_audit.json"
LA = "manuscript/audit/outputs/loewe_audit.json"
SA = "manuscript/audit/outputs/screening_audit.json"
G2 = "manuscript/audit/outputs/gate2_reproduction.json"
VC = "results/interaction_gap_20261110/variance_components.json"
AE = "results/interaction_gap_20261110/anchor_experiment.json"
GR = "results/interaction_gap_20261110/grounding.json"
BT = "results/interaction_gap_20261110/batch_transfer.json"
DS = "results/interaction_gap_20261110/dose_shape_check.json"
BP = "results/interaction_gap_20261110/b3_paired_summary.json"
NC = "results/mechanisms_20261108/historical_controls/failed_ncats_confirmation.json"
CL = "results/final_20261110/claim_ledger.json"

# (stated text, source, key path, scale, decimals). scale converts the source to the stated unit.
CHECKS = [
    # Response benchmark (results/compare/final.csv handled below).
    # Interaction audit.
    ("9.8%", IA, "results/pair_cell:bliss_residual/noise_share_grid_split", 100, 1),
    ("8.9, 10.8", IA, "results/pair_cell:bliss_residual/noise_share_grid_pair_interval/0", 100, 1),
    ("drug 6.2, 13.3", IA, "results/pair_cell:bliss_residual/noise_share_grid_drug_interval/1", 100, 1),
    ("0.061 [0.022, 0.098]", IA, "results/pair_cell:bliss_residual/models/neural_F2/r2", 1, 3),
    ("0.068", IA, "results/pair_cell:bliss_residual/models/neural_F2/noise_corrected_r2_grid", 1, 3),
    ("CF 0.032", IA, "results/pair_cell:bliss_residual/models/cf_selected/r2", 1, 3),
    ("GP 0.028", IA, "results/pair_cell:bliss_residual/models/gp_F2/r2", 1, 3),
    ("−0.050", IA, "results/pair_cell:bliss_residual/models/neural_base_F2/r2", 1, 3),
    ("−0.056", IA, "results/pair_cell:bliss_residual/models/neural_base_F2/noise_corrected_r2_grid", 1, 3),
    ("0.035", IA, "results/pair_cell:bliss_residual/models/cf_selected/noise_corrected_r2_grid", 1, 3),
    ("0.031", IA, "results/pair_cell:bliss_residual/models/gp_F2/noise_corrected_r2_grid", 1, 3),
    ("36% [34, 38]", IA, "results/dose_shape:bliss_residual/noise_share_grid_split", 100, 0),
    ("0.4473", IA, "results/dose_shape:bliss_residual/noise_share_panel_split", 1, 4),
    ("0.240", IA, "results/dose_shape:bliss_residual/models/gp_F2/noise_corrected_r2_grid", 1, 3),
    ("0.195", IA, "results/dose_shape:bliss_residual/models/neural_F2/noise_corrected_r2_grid", 1, 3),
    # Pair-seen campaign.
    ("0.268", PS, "results/pair_cell:bliss_residual/difference", 1, 3),
    ("[0.209, 0.330]", PS, "results/pair_cell:bliss_residual/difference_pair_interval/0", 1, 3),
    ("0.2182355", PS, "results/pair_cell:bliss_residual/pair_seen_r2", 1, 7),
    # Interaction-gap analysis.
    ("0.242 [0.197, 0.293", VC, "components/bliss/pair_wide", 1, 3),
    ("0.660 [0.613, 0.702]", VC, "components/bliss/pair_by_cell", 1, 3),
    ("drug 0.151, 0.334", VC, "components/bliss/pair_wide_drug_interval/1", 1, 3),
    ("0.988", VC, "controls/zero_interaction/noise_share_mean", 1, 3),
    ("0.0007 ± 0.0016", VC, "controls/pair_permutation/pair_wide_mean", 1, 4),
    ("$r=0.009$", VC, "components/pair_mean_correlations/bliss~ref", 1, 3),
    ("$r=0.81$", VC, "components/pair_mean_correlations/bliss~raw", 1, 2),
    ("1.6%", VC, "controls/reference_perturbation/5.0/induced_energy_share", 100, 1),
    ("0.359", VC, "loco/bliss/B2/r2", 1, 3),
    ("0.144 [0.116, 0.172]", VC, "loco/bliss/B2_minus_B1/r2", 1, 3),
    ("−0.28 (production F2)", VC, "held_out_pair_models/f2_base/pair_mean_r2", 1, 2),
    ("−0.14", VC, "held_out_pair_models/f2_selected/pair_mean_r2", 1, 2),
    ("reaches 0.13", VC, "held_out_pair_models/f2_selected/residual_r2", 1, 2),
    ("88% [65%, 101%]", VC, "cross_batch/bliss_frozen_reference/cell_shift/replication_overall", 100, 0),
    ("4.5 times", VC, "cross_batch/bliss_frozen_reference/none/msd_over_within_noise", 1, 1),
    ("0.89 [0.52, 0.97]", VC, "cross_batch/bliss_frozen_reference/cell_shift/replication_pair_wide", 1, 2),
    ("0.85 [0.62, 1.16]", VC, "cross_batch/bliss_frozen_reference/cell_shift/replication_pair_by_cell", 1, 2),
    ("0.033 at one anchor", AE, "curves/bliss:k=1/B2-B1/r2_difference", 1, 3),
    ("0.086 [0.071, 0.102; drug 0.048, 0.124]", AE, "curves/bliss:k=16/B2-B1/r2_difference", 1, 3),
    ("0.087 at 16", AE, "curves/train:k=16/B2-B1/r2_difference", 1, 3),
    ("−0.27", AE, "curves/bliss:k=16/B4_B2/r2", 1, 2),
    ("0.085, below B1", AE, "curves/bliss:k=16/B5b_B2/r2", 1, 3),
    ("0.12 $R^2$ with four", AE, "curves/bliss:k=4/B1/r2", 1, 2),
    ("10.7% [8.9, 12.6]", AE, "curves/bliss:k=16/experiment_c/residual_r2_B2_vs_B1", 100, 1),
    ("(Spearman 0.62)", GR, "bliss/split_half_reliability_spearman", 1, 2),
    ("0.069", GR, "bliss/mantel_spearman", 1, 3),
    ("1% of B2", GR, "bliss/expression_share_of_B2_gain", 100, 0),
    ("0.030 [−0.015, 0.086]", BT, "bliss/scored_against_batch2/B2_minus_B1", 1, 3),
    ("0.3633", DS, "dose_shape_bliss_noise_share_grid_split", 1, 4),
    ("0.3631", DS, "dose_shape_bliss_noise_share_batch_split", 1, 4),
    ("0.177", BP, "models/B1_pair_intercept/r2", 1, 3),
    ("[0.143, 0.212]", BP, "models/B1_pair_intercept/pair_interval/0", 1, 3),
    ("0.216", BP, "models/B2_cross_cell_covariance/r2", 1, 3),
    ("[0.162, 0.274]", BP, "models/B3_pair_seen_network/pair_interval/0", 1, 3),
    ("[−0.058, 0.141]", BP, "differences/B3-B1/drug_interval/0", 1, 3),
    ("0.003", BP, "differences/B3-B2/r2_difference", 1, 3),
    ("[0.029, 0.049]", BP, "differences/B2-B1/pair_interval/0", 1, 3),
    ("81%", BP, "b1_share_of_b3/value", 100, 0),
    ("only 39% of pairs", BP, "differences/B3-B1/pairs_where_first_is_better", 100, 0),
    # Loewe.
    ("310,928", LA, "existence_criterion/agreements", 1, 0),
    ("1,872", LA, "existence_criterion/co_decreasing", 1, 0),
    ("180,364", LA, "populations/status_by_fallback/certified_root|accepted", 1, 0),
    ("15,620", LA, "populations/fallback_certified_roots", 1, 0),
    ("14,197", LA, "populations/C1", 1, 0),
    ("12,477", LA, "populations/historical_class/artifact: solver at root, answer discarded", 1, 0),
    ("1,720", LA, "populations/historical_class/artifact: root exists, solver failed", 1, 0),
    ("1,423", LA, "populations/C2_recovered", 1, 0),
    ("117,824", LA, "populations/certificate_status/certified_no_root", 1, 0),
    ("49,184", LA, "populations/certificate_status/undefined_opposite_or_flat", 1, 0),
    ("4,160", LA, "populations/certificate_status/undefined_empty_interval", 1, 0),
    ("1,883", LA, "populations/status_by_fallback/undefined_opposite_or_flat|accepted", 1, 0),
    ("15,588 of 15,588", LA, "independent_bracket_replay/fallback_roots/sign_change_confirmed", 1, 0),
    ("1,994 of 1,994", LA, "independent_bracket_replay/accepted_controls/sign_change_confirmed", 1, 0),
    ("median 4.17 points", LA, "fallback_error_on_rejected_roots/C1_14197/summed_dose/median", 1, 2),
    ("(mean 20.85)", LA, "fallback_error_on_rejected_roots/C1_14197/summed_dose/mean", 1, 2),
    ("median 0.0040", LA, "fallback_error_on_rejected_roots/C1_14197/hsa/median", 1, 4),
    ("71.7%", LA, "fallback_error_on_rejected_roots/C1_14197/share_hsa_closer", 100, 1),
    ("positive in 58%", LA, "c1_location/jump_positive_share_plateau_below_100", 100, 0),
    ("13,277", LA, "boundary_layer/cases", 1, 0),
    ("0.998", LA, "boundary_layer/loglog_pearson_r", 1, 3),
    ("93.8% against 43.4%", LA, "boundary_layer/share_abs_slope_above_2/C1_co_increasing_LL4", 100, 1),
    ("68.3%", LA, "c1_location/share_of_absolute_shift_upper", 100, 1),
    # Screening.
    ("RMSE 0.704", SA, "checks/native_vs_published/rmse", 1, 3),
    ("0.9990", SA, "checks/native_vs_published/pearson", 1, 4),
    ("20,276", SA, "block_level/changed_by_C1", 1, 0),
    ("1,998", SA, "block_level/sign_reversals_C1", 1, 0),
    ("5.89%", SA, "budget_0.05/N_to_C1_turnover", 100, 2),
    ("67 of 1,137", SA, "budget_0.05/N_to_C1_replaced", 1, 0),
    ("6.95%", SA, "budget_0.05/N_to_I_turnover", 100, 2),
    ("30.87%", SA, "budget_0.05/N_to_H_turnover", 100, 2),
    ("median 3.60", SA, "c1_top5/entrant_native_margin_below_cutoff/median", 1, 2),
    ("5.87% [5.10, 6.68]", SA, "replicate_resampling/within_resample_N_to_C1_turnover_bootstrap/mean", 100, 2),
    ("16.6%", SA, "replicate_resampling/noise_turnover_consecutive_bootstrap_N/mean", 100, 1),
    ("28.5%", SA, "replicate_resampling/noise_turnover_half_depth_N/mean", 100, 1),
    # Gate 2.
    ("0.772", G2, "r18_gate2.json/key_values/P2.cr2_M_DC/rerun", 1, 3),
    ("0.488", G2, "r18_gate2.json/key_values/P4.runs.cell_independent:loewe_sim.knn.cr2/rerun", 1, 3),
    ("0.515", G2, "r18_gate2.json/key_values/P4.runs.cell_independent:loewe_sim.krr.cr2/rerun", 1, 3),
    ("−0.099", G2, "r18_gate2.json/key_values/P4.runs.cell_independent:bliss_sim.knn.cr2/rerun", 1, 3),
    ("79.8%", G2, "r18_gate2.json/key_values/P3.fraction_units_with_fallback/rerun", 100, 1),
    # Negative results.
    ("0.057", NC, "H1/interaction_residual_r2", 1, 3),
    ("[drug −0.006, 0.120]", NC, "H1/drug_jackknife/1", 1, 3),
    ("+4.61%", CL, "9/pooled_2026_2029", 100, 2),
    ("+3.23%", CL, "9/seed_2029", 100, 2),
]


def fmt(v, d):
    s = f"{abs(v):,.{d}f}" if d == 0 else f"{abs(v):.{d}f}"
    return ("−" if v < 0 and float(s.replace(",", "")) != 0 else "") + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default=None, help="frozen resolution cache (for the variance-share check)")
    a = ap.parse_args()
    text = "".join((M / f"{n}.tex").read_text(encoding="utf-8") for n in ("manuscript", "supplement"))
    text = text.replace(chr(92) + "%", "%")
    cache, fails, rows = {}, 0, []
    for stated, src, path, scale, dec in CHECKS:
        o = cache.setdefault(src, load(src))
        v = float(get(o, path)) * scale
        want = fmt(v, dec)
        ok_value = want in stated.replace(",", "") or want in stated
        ok_text = stated in text
        status = "ok" if (ok_value and ok_text) else "FAIL"
        fails += status != "ok"
        rows.append((status, stated, want, src.split("/")[-1], path))
    b1 = get(load(AE), "matched_pair_seen/bliss:leak_free/B1:projected/r2")
    b3 = get(load(PS), "results/pair_cell:bliss_residual/pair_seen_r2")
    want = fmt(b3 - b1, 3)
    status = "ok" if (want == "0.041" and "B3 − B1 & 0.041 &" in text) else "FAIL"
    fails += status != "ok"
    rows.append((status, "B3 − B1 0.041", want, "full precision", "pairseen B3 minus leak-free B1"))
    fc = pd.read_csv(ROOT / "results/compare/final.csv").set_index("model")
    for stated, model, col, dec in [("11.84", "selected:F2", "rmse", 2), ("0.878", "selected:F2", "r2", 3),
                                    ("11.75", "gp_f2", "rmse", 2), ("0.880", "gp_f2", "r2", 3), ("12.78", "cf_selected", "rmse", 2),
                                    ("14.03", "bliss_reference", "rmse", 2), ("0.829", "bliss_reference", "r2", 3),
                                    ("18.19", "deepsynergy_dose", "rmse", 2), ("0.283", "deepsynergy_dose", "contrast_r2", 3),
                                    ("0.315", "bliss_reference", "contrast_r2", 3)]:
        want = fmt(float(fc.loc[model, col]), dec)
        status = "ok" if (want == stated and stated in text) else "FAIL"
        fails += status != "ok"
        rows.append((status, stated, want, "compare/final.csv", f"{model}/{col}"))
    if a.resolution:
        var = json.loads((Path(a.resolution) / "variance.json").read_text())["shares"]
        for stated, v in [("81.8%", var["between_panel_additive"] + var["within_panel_additive_dose"]),
                          ("2.6%", var["between_panel_contrast"]), ("10.2%", var["within_panel_dose_contrast"]),
                          ("5.4%", var["replicate_deviations"])]:
            want = fmt(100 * v, 1) + "%"
            status = "ok" if (want == stated and stated in text) else "FAIL"
            fails += status != "ok"
            rows.append((status, stated, want, "resolution/variance.json", "shares"))
    # Post-review Supplement S15: recompute the five-fold summary from individual folds.
    # Keep the original checks above unchanged; published-label and dose-response tasks remain separate.
    import numpy as np
    import re
    supplement = (M / "supplement.tex").read_text(encoding="utf-8")
    ds = pd.read_csv(ROOT / "results/neural_sprint/deepsynergy.csv")
    folds = ds[ds.fold.isin([str(k) for k in range(5)])]
    if len(folds) != 5:
        raise ValueError("Original-label reproduction must contain five individual folds")
    published = ds.set_index("fold").loc["paper_table2"]
    table = re.search(r"Existing original-label port & (.*?) \\\\", supplement).group(1)
    paper = re.search(r"Published DeepSynergy & (.*?) \\\\", supplement).group(1)
    review_checks = [
        ("260.74", folds.mse.mean(), 2, table, "mean fold MSE"),
        ("16.07", np.sqrt(folds.mse).mean(), 2, table, "mean fold RMSE"),
        ("1.81", np.sqrt(folds.mse).std(ddof=1), 2, table, "sample SD fold RMSE"),
        ("0.726", folds.pearson.mean(), 3, table, "mean fold Pearson"),
        ("0.037", folds.pearson.std(ddof=1), 3, table, "sample SD fold Pearson"),
    ]
    for stated, col, dec in [("255.49", "mse", 2), ("15.91", "rmse", 2),
                             ("1.56", "rmse_sd", 2), ("0.73", "pearson", 2),
                             ("0.04", "pearson_sd", 2)]:
        review_checks.append((stated, published[col], dec, paper, f"published {col}"))
    bench = pd.read_csv(ROOT / "results/neural_sprint/benchmark.csv").set_index("model").loc["deepsynergy_dose"]
    for stated, col, dec, literal in [
        ("17.87", "rmse_ci_low", 2, "[17.87, 18.53]"),
        ("18.53", "rmse_ci_high", 2, "[17.87, 18.53]"),
        ("−0.179", "interaction_residual_r2_jackknife_low", 3, "[$-0.179$, 0.085]"),
        ("0.085", "interaction_residual_r2_jackknife_high", 3, "[$-0.179$, 0.085]"),
        ("−0.047", "interaction_residual_r2", 3, "$-0.047$"),
        ("0.713", "r2", 3, "$R^2$ 0.713"),
    ]:
        review_checks.append((stated, bench[col], dec, literal if literal in supplement else "", col))
    for stated, value, dec, context, key in review_checks:
        want = fmt(float(value), dec)
        # Signed values in S15 are typeset with a mathematical minus.
        ok_text = stated in context or stated.replace("−", "-") in context
        status = "ok" if want == stated and ok_text else "FAIL"
        fails += status != "ok"
        rows.append((status, stated, want, "Supplement S15 / neural_sprint", key))
    for r in rows:
        if r[0] != "ok":
            print(" | ".join(r))
    print(f"{len(rows) - fails}/{len(rows)} checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
