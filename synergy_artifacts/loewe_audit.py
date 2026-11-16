"""Native Loewe fallback audit: the 98 probe cases, all 367,152 fixed-coefficient cases, and score/prediction effects.

python -m synergy_artifacts.loewe_audit --output results/loewe_audit [--resume]

Inputs are existing artifacts from the scoring audit on the pipeline branch:
  results/neural/runs/phase3_native_attribution_v3/strict_cases.parquet  (200-block native-fit probe, 3,200 cases)
  results/neural/runs/scientific_native_full/{surface_cases,blocks}.parquet (all 92,208 blocks, cached fits, native solver)
  results/neural/runs/phase2_panel_neural/predictions.parquet              (existing out-of-fold panel predictions)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from .metrics import interaction_contrast, regression_metrics

RUNS = Path("results/neural/runs")


def strict_reference(xa, xb, model_a, pa, model_b, pb):
    """Same rule as experiments/phase3/attribution.py: root of the Loewe additivity equation on the
    joint positive-inverse domain of two increasing curves; undefined otherwise."""
    bounds = []
    for model, p in [(model_a, pa), (model_b, pb)]:
        b, c, d, e = p
        if model not in {"LL.4", "L.4"} or not np.isfinite(p).all() or (model == "LL.4" and e <= 0):
            return np.nan, "invalid_parameters"
        if -b * (d - c) <= 0:
            return np.nan, "not_increasing"
        bounds.append((min(c, d), max(c, d)))
    low, high = max(x[0] for x in bounds), min(x[1] for x in bounds)
    if high <= low:
        return np.nan, "disjoint_domains"
    margin = max(1e-10, (high - low) * 1e-10)

    def equation(effect):
        terms = []
        for dose, model, p in [(xa, model_a, pa), (xb, model_b, pb)]:
            b, c, d, e = p
            log_inverse = (np.log(e) if model == "LL.4" else e) + np.log((d - effect) / (effect - c)) / b
            terms.append(np.log(dose) - log_inverse)
        return float(logsumexp(terms))
    lo, hi = low + margin, high - margin
    if lo >= hi or equation(lo) < 0 or equation(hi) > 0:
        return np.nan, "no_joint_root"
    return float(brentq(equation, lo, hi, xtol=1e-12)), "defined"


def loewe_residual(effect, xa, xb, model_a, pa, model_b, pb):
    """Scorer's own equation (eq.*.* in calculate_synergy_score.R): sum of dose / equivalent dose minus 1."""
    total = -1.0
    for dose, model, (b, c, d, e) in [(xa, model_a, pa), (xb, model_b, pb)]:
        ratio = ((effect - d) / (c - effect)) ** (1 / b)
        total += dose / (e * ratio) if model == "LL.4" else dose / np.exp(e + np.log((effect - d) / (c - effect)) / b)
    return total


def params(row, arm):
    return np.array([getattr(row, f"{arm}_{k}") for k in "bcde"], float)


def solve_all(frame):
    roots, status = [], []
    for row in frame.itertuples():
        r, s = strict_reference(row.dose_a, row.dose_b, row.model_a, params(row, "a"), row.model_b, params(row, "b"))
        roots.append(r)
        status.append(s)
    return np.array(roots), np.array(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="reuse full_cases.parquet from a previous run")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. The 98 probe cases (native R 3.6.3 fits, official scorer outputs).
    probe = pd.read_parquet(RUNS / "phase3_native_attribution_v3/strict_cases.parquet")
    probe["strict_check"], probe["status_check"] = solve_all(probe)
    f98 = probe[probe.fallback & probe.strict_reference.notna()].copy()
    f98["equation_residual_at_strict"] = [loewe_residual(r.strict_check, r.dose_a, r.dose_b, r.model_a, params(r, "a"), r.model_b, params(r, "b"))
                                          for r in f98.itertuples()]
    f98["solver_x_minus_strict"] = f98.solution - f98.strict_check
    f98["shift"] = np.minimum(f98.strict_check, 100) - f98.native_reference
    f98.to_csv(out / "probe_98_cases.csv", index=False)
    code3 = f98.native_code == 3
    report["probe_98"] = {
        "cases": len(f98), "blocks": int(f98.block_id.nunique()), "pairs": int(f98.pair_id.nunique()),
        "native_codes": {str(k): int(v) for k, v in f98.native_code.value_counts().items()},
        "strict_root_recomputed_max_abs_diff": float((f98.strict_check - f98.strict_reference).abs().max()),
        "status_recomputed_all_defined": bool((f98.status_check == "defined").all()),
        "max_abs_equation_residual_at_root": float(np.nanmax(np.abs(f98.equation_residual_at_strict))),
        "native_reference_equals_official": bool(np.allclose(f98.native_reference, f98.official_reference, atol=1e-9)),
        "native_reference_equals_summed_dose_fallback_capped": bool(np.allclose(f98.native_reference, np.minimum(f98.fallback_sum, 100), atol=1e-6)),
        "code3_solver_x_minus_strict_abs": f98.loc[code3, "solver_x_minus_strict"].abs().describe().to_dict(),
        "code3_solver_x_within_0.05_of_root": float((f98.loc[code3, "solver_x_minus_strict"].abs() < .05).mean()),
        "reference_shift_capped_root_minus_native": f98["shift"].describe().to_dict(),
        "reference_shift_rmse": float(np.sqrt(np.mean(f98["shift"] ** 2))),
    }
    print(json.dumps(report["probe_98"], indent=1), flush=True)

    # 2. All 367,152 unique fixed-coefficient cases (22,947 surfaces) behind the 92,208 blocks.
    if (out / "full_cases.parquet").exists() and args.resume:
        cases = pd.read_parquet(out / "full_cases.parquet")
    else:
        cases = pd.read_parquet(RUNS / "scientific_native_full/surface_cases.parquet")
        cases["strict"], cases["strict_status"] = solve_all(cases)
    defined = cases.strict_status == "defined"
    cases["corrected_reference"] = np.where(cases.fallback & defined, np.minimum(cases.strict, 100), cases.native_below_3)
    cases["shift"] = cases.corrected_reference - cases.native_below_3
    accepted_defined = ~cases.fallback & defined & (cases.native_code < 3)
    cases["accepted_minus_strict"] = np.where(accepted_defined, cases.native_solution - cases.strict, np.nan)
    cases.to_parquet(out / "full_cases.parquet", index=False)
    strata = (cases.groupby(["fallback", "strict_status"]).agg(cases=("case_id", "size"), surfaces=("surface_id", "nunique"),
                                                                mean_abs_shift=("shift", lambda s: float(s.abs().mean())))
              .reset_index())
    fd = cases[cases.fallback & defined]
    report["full"] = {
        "cases": len(cases), "surfaces": int(cases.surface_id.nunique()), "fallback_cases": int(cases.fallback.sum()),
        "strata": strata.to_dict("records"),
        "fallback_with_defined_root": {"cases": len(fd), "surfaces": int(fd.surface_id.nunique()),
                                       "codes": {str(k): int(v) for k, v in fd.native_code.value_counts().items()},
                                       "shift": fd["shift"].describe().to_dict(), "shift_rmse": float(np.sqrt(np.mean(fd["shift"] ** 2))),
                                       "solver_x_within_0.05_of_root": float(((fd.native_solution - fd.strict).abs() < .05).mean()),
                                       "code3_solver_x_within_0.05_of_root": float(((fd.native_solution - fd.strict).abs() < .05)[fd.native_code == 3].mean())},
        "accepted_defined_max_abs_solution_minus_strict": float(np.nanmax(np.abs(cases.accepted_minus_strict))),
    }
    print(json.dumps(report["full"], indent=1, default=str), flush=True)

    # 3. Block and pair-cell Loewe scores with valid roots in place of the fallback where a root exists.
    blocks = pd.read_parquet(RUNS / "scientific_native_full/blocks.parquet",
                             columns=["block_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "response", "synergy_loewe",
                                      "native_below_3", "native_sum_score", "surface_id"])
    surface_shift = cases.groupby("surface_id").agg(mean_shift=("shift", "mean"), n_corrected=("shift", lambda s: int((s.abs() > 1e-9).sum())))
    blocks = blocks.merge(surface_shift, on="surface_id", validate="many_to_one")
    blocks["corrected_score"] = blocks.native_sum_score - blocks.mean_shift
    blocks["score_shift"] = blocks.corrected_score - blocks.native_sum_score
    affected = blocks.n_corrected > 0
    blocks.to_parquet(out / "blocks_corrected.parquet", index=False)
    for c in ["pair_id", "drug_a", "drug_b", "cell_line"]:
        blocks[c] = blocks[c].astype(str)
    panel = blocks.groupby(["pair_id", "drug_a", "drug_b", "cell_line", "fold"], sort=True).agg(
        published=("synergy_loewe", "mean"), native=("native_sum_score", "mean"), corrected=("corrected_score", "mean"),
        affected_blocks=("n_corrected", lambda s: int((s > 0).sum())), blocks=("block_id", "size")).reset_index()
    panel["shift"] = panel.corrected - panel.native
    report["scores"] = {
        "blocks": len(blocks), "affected_blocks": int(affected.sum()), "affected_pairs": int(blocks.pair_id[affected].nunique()),
        "affected_cells": int(blocks.cell_line[affected].nunique()),
        "block_score_shift_affected": blocks.loc[affected, "score_shift"].describe().to_dict(),
        "blocks_abs_shift_ge": {str(t): int((blocks.score_shift.abs() >= t).sum()) for t in [1, 5, 10]},
        "block_sign_flips": int(((np.sign(blocks.native_sum_score) != np.sign(blocks.corrected_score)) & affected).sum()),
        "native_vs_published_block_rmse": float(np.sqrt(np.mean((blocks.native_sum_score - blocks.synergy_loewe) ** 2))),
        "panel_units": len(panel), "affected_panel_units": int((panel.affected_blocks > 0).sum()),
        "panel_shift_affected": panel.loc[panel.affected_blocks > 0, "shift"].describe().to_dict(),
        "panel_abs_shift_ge": {str(t): int((panel["shift"].abs() >= t).sum()) for t in [1, 5, 10]},
        "panel_spearman_native_corrected": float(panel.native.rank().corr(panel.corrected.rank())),
    }
    print(json.dumps(report["scores"], indent=1, default=str), flush=True)

    # 4. Prediction artifacts: existing out-of-fold panel predictions of the published Loewe label.

    pred = pd.read_parquet(RUNS / "phase2_panel_neural/predictions.parquet")
    pred = pred[pred.target == "synergy_loewe"].drop(columns="target").astype({"pair_id": str, "cell_line": str, "drug_a": str, "drug_b": str})
    m = pred.merge(panel, on=["pair_id", "drug_a", "drug_b", "cell_line", "fold"], validate="one_to_one")
    assert len(m) == len(pred) and np.allclose(m.truth, m.published, atol=1e-6), "panel labels differ from the prediction truth"
    hit = m.affected_blocks > 0
    artifacts = {"units": len(m), "affected_units": int(hit.sum())}
    for label in ["published", "corrected"]:
        artifacts[label] = {"all": regression_metrics(m[label].to_numpy(), m.prediction.to_numpy()),
                            "affected": regression_metrics(m.loc[hit, label].to_numpy(), m.loc[hit, "prediction"].to_numpy()),
                            "unaffected": regression_metrics(m.loc[~hit, label].to_numpy(), m.loc[~hit, "prediction"].to_numpy()),
                            "contrast_r2": interaction_contrast(m.rename(columns={label: "target"}), "target")}
    artifacts["prediction_error_vs_label_shift_corr_affected"] = float(np.corrcoef((m.prediction - m.published)[hit], m["shift"][hit])[0, 1])
    report["prediction_artifacts"] = artifacts
    print(json.dumps(artifacts, indent=1, default=str), flush=True)
    (out / "loewe_audit.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
