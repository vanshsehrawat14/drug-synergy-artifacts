"""Independent check of the Loewe fallback finding (configs/loewe_verification.json).

python -m cs622.loewe_verify --cases results/loewe_audit/full_cases.parquet --output results/loewe_verify
python -m cs622.loewe_verify --cases ... --output ... --retrained <panel identity predictions on corrected labels>

1. Admissibility under the scorer's own definition: the root solves the Loewe equation of
   calculate_synergy_score.R (residual) and lies strictly inside both fitted effect ranges, where
   that equation is real-valued.
2. A second root finder that never inverts a curve: with theta = dose_a / D_a(y), the root is the theta in
   (0, 1) where f_a(dose_a / theta) = f_b(dose_b / (1 - theta)); bisection on logit(theta). It covers
   both co-increasing and co-decreasing curve pairs.
3. Every native fallback is classed as a numerical artifact (an admissible root exists) or as
   definition-dependent (no root, or curves of opposite direction); score effects are given per class
   next to the effect of switching between published fallback conventions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .loewe_audit import loewe_residual, params
from .metrics import interaction_contrast, regression_metrics

RUNS = Path("results/neural/runs")
NEAR = 0.05  # inhibition points: native solver answer counted as "at the root"


def curve(log_dose, model, b, c, d, e):
    """drc LL.4 / L.4 (SynergyFinder fits L.4 on log dose) evaluated forward."""
    centre = np.where(model == "LL.4", np.log(np.where(e > 0, e, np.nan)), e)
    return c + (d - c) * expit(-b * (log_dose - centre))


def theta_root(frame, steps=200):
    """Loewe expectation from forward curves only. Returns (effect, status) arrays."""
    ma, mb = frame.model_a.to_numpy(), frame.model_b.to_numpy()
    pa = [frame[f"a_{k}"].to_numpy() for k in "bcde"]
    pb = [frame[f"b_{k}"].to_numpy() for k in "bcde"]
    xa, xb = np.log(frame.dose_a.to_numpy()), np.log(frame.dose_b.to_numpy())
    slope_a, slope_b = -pa[0] * (pa[2] - pa[1]), -pb[0] * (pb[2] - pb[1])
    finite = np.isfinite(np.column_stack(pa + pb)).all(1) & np.isin(ma, ["LL.4", "L.4"]) & np.isin(mb, ["LL.4", "L.4"])
    finite &= ~((ma == "LL.4") & (pa[3] <= 0)) & ~((mb == "LL.4") & (pb[3] <= 0))
    direction = np.sign(slope_a)
    same = finite & (np.sign(slope_a) == np.sign(slope_b)) & (slope_a != 0)

    def g(t):  # decreasing in t for co-increasing curves; sign-flipped for co-decreasing
        log_theta, log_rest = -np.logaddexp(0, -t), -np.logaddexp(0, t)
        with np.errstate(all="ignore"):
            return direction * (curve(xa - log_theta, ma, *pa) - curve(xb - log_rest, mb, *pb))
    lo, hi = np.full(len(frame), -60.), np.full(len(frame), 60.)
    g_lo, g_hi = g(lo), g(hi)
    bracket = same & (g_lo > 0) & (g_hi < 0)
    for _ in range(steps):
        mid = (lo + hi) / 2
        positive = g(mid) > 0
        lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
    t = (lo + hi) / 2
    with np.errstate(all="ignore"):
        effect = (curve(xa + np.logaddexp(0, -t), ma, *pa) + curve(xb + np.logaddexp(0, t), mb, *pb)) / 2
    status = np.select([~finite, ~same & (slope_a * slope_b < 0), ~same, ~bracket],
                       ["invalid_parameters", "opposite_directions", "flat_curve", "no_root"],
                       np.where(direction > 0, "root_increasing", "root_decreasing"))
    return np.where(bracket, effect, np.nan), status


def admissible(frame, effect):
    """Residual of the scorer's equation at effect, and whether effect is strictly inside both effect ranges."""
    with np.errstate(all="ignore"):
        residual = np.array([loewe_residual(y, r.dose_a, r.dose_b, r.model_a, params(r, "a"), r.model_b, params(r, "b"))
                             if np.isfinite(y) else np.nan for y, r in zip(effect, frame.itertuples())])
    inside = np.ones(len(frame), bool)
    for arm in "ab":
        c, d = frame[f"{arm}_c"].to_numpy(), frame[f"{arm}_d"].to_numpy()
        inside &= (effect > np.minimum(c, d)) & (effect < np.maximum(c, d))
    return residual, inside & np.isfinite(effect) & (np.abs(residual) < 1e-6)


def classify(cases):
    found = cases.independent_status.str.startswith("root")
    root = found & cases.admissible
    near = (cases.native_solution - cases.independent).abs() <= NEAR
    fb = cases.fallback
    return np.select(
        [fb & root & near, fb & root & ~near, fb & found, fb & (cases.independent_status == "no_root"),
         fb & cases.independent_status.isin(["opposite_directions", "flat_curve"]), fb,
         ~fb & root & near, ~fb & root, ~fb],
        ["artifact: solver at root, answer discarded", "artifact: root exists, solver failed",
         "boundary: root at a curve asymptote, scorer residual >= 1e-6", "definition: no root",
         "definition: opposite or flat curves", "definition: invalid fit",
         "accepted: matches root", "accepted: differs from root", "accepted: no admissible root"], "unclassified")


def score_effects(cases, blocks, mask, label):
    """Block and pair-cell Loewe scores with the admissible root in place of the native reference on mask."""
    corrected = np.where(mask, np.minimum(cases.independent, 100), cases.native_below_3)
    shift = pd.Series(corrected - cases.native_below_3).groupby(cases.surface_id.to_numpy()).mean()
    s = blocks.surface_id.map(shift).to_numpy()
    new = blocks.native_sum_score.to_numpy() - s
    affected = np.abs(s) > 1e-9
    flips = affected & (np.sign(new) != np.sign(blocks.native_sum_score.to_numpy()))
    return {"label": label, "cases": int(mask.sum()), "surfaces": int(cases.surface_id[mask].nunique()),
            "blocks_affected": int(affected.sum()), "blocks_affected_share": float(affected.mean()),
            "block_shift_mean_affected": float(-s[affected].mean()) if affected.any() else 0.,
            "block_abs_shift_ge": {str(t): int((np.abs(s) >= t).sum()) for t in [1, 5, 10]},
            "block_sign_flips": int(flips.sum()), "max_block_shift": float(np.abs(s).max())}, -s


def convention_effects(blocks):
    """Ordinary definitional difference: the same native solver with the HSA fallback instead of the summed-dose fallback."""
    d = blocks.native_hsa_fallback_score.to_numpy() - blocks.native_sum_score.to_numpy()
    affected = np.abs(d) > 1e-9
    return {"label": "convention: HSA fallback instead of summed-dose fallback (all fallbacks)",
            "blocks_affected": int(affected.sum()), "blocks_affected_share": float(affected.mean()),
            "block_shift_mean_affected": float(d[affected].mean()), "block_abs_shift_ge": {str(t): int((np.abs(d) >= t).sum()) for t in [1, 5, 10]},
            "block_sign_flips": int((affected & (np.sign(blocks.native_hsa_fallback_score) != np.sign(blocks.native_sum_score))).sum()),
            "max_block_shift": float(np.abs(d).max())}


def panel_labels(blocks, shifts):
    frame = blocks[["pair_id", "drug_a", "drug_b", "cell_line", "fold", "synergy_loewe"]].astype({c: str for c in ["pair_id", "drug_a", "drug_b", "cell_line"]})
    for name, s in shifts.items():
        frame[name] = s
    panel = frame.groupby(["pair_id", "drug_a", "drug_b", "cell_line", "fold"], sort=True).mean().reset_index()
    for name in shifts:
        panel[name] = panel.synergy_loewe + panel[name]
    return panel.rename(columns={"synergy_loewe": "published"})


def model_effects(predictions, panel, labels):
    """Score one set of out-of-fold predictions against each label version; units touched by any correction separately."""
    m = predictions.merge(panel, on=["pair_id", "drug_a", "drug_b", "cell_line", "fold"], validate="one_to_one")
    hit = np.zeros(len(m), bool)
    for name in labels:
        hit |= np.abs(m[name] - m.published) > 1e-9
    out = {"units": len(m), "affected_units": int(hit.sum())}
    for name in ["published"] + labels:
        out[name] = {"all": regression_metrics(m[name].to_numpy(), m.prediction.to_numpy()),
                     "affected": regression_metrics(m.loc[hit, name].to_numpy(), m.loc[hit, "prediction"].to_numpy()),
                     "contrast_r2": interaction_contrast(m.rename(columns={name: "target"}), "target")}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cases", required=True, help="full_cases.parquet written by cs622.loewe_audit")
    p.add_argument("--output", required=True)
    p.add_argument("--retrained", help="out-of-fold panel predictions with targets published/valid_root/solver_root")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(a.cases)
    cases["independent"], cases["independent_status"] = theta_root(cases)
    cases["residual_at_independent"], cases["admissible"] = admissible(cases, cases.independent.to_numpy())
    cases["class"] = classify(cases)
    both = cases.strict_status.eq("defined") & cases.independent_status.eq("root_increasing")
    report = {"cases": len(cases), "near_tolerance": NEAR,
              "independent_vs_strict": {"both_defined": int(both.sum()),
                                        "max_abs_difference": float((cases.independent - cases.strict)[both].abs().max()),
                                        "strict_only": int((cases.strict_status.eq("defined") & ~cases.independent_status.str.startswith("root")).sum()),
                                        "independent_increasing_only": int((~cases.strict_status.eq("defined") & cases.independent_status.eq("root_increasing")).sum()),
                                        "independent_decreasing": int(cases.independent_status.eq("root_decreasing").sum())},
              "admissibility": {"roots": int(cases.independent_status.str.startswith("root").sum()),
                                "admissible": int(cases.admissible.sum()),
                                "max_abs_scorer_residual_admissible": float(cases.residual_at_independent[cases.admissible].abs().max()),
                                "roots_failing_residual": int((cases.independent_status.str.startswith("root") & ~cases.admissible).sum())},
              "independent_status": cases.independent_status.value_counts().to_dict()}
    table = (cases.assign(native_minus_root=(cases.native_solution - cases.independent).abs(),
                          root_minus_native=np.minimum(cases.independent, 100) - cases.native_below_3)
             .groupby("class").agg(cases=("case_id", "size"), surfaces=("surface_id", "nunique"),
                                   native_code_3=("native_code", lambda s: int((s == 3).sum())),
                                   native_error_999=("native_code", lambda s: int((s == 999).sum())),
                                   decreasing=("independent_status", lambda s: int((s == "root_decreasing").sum())),
                                   median_abs_native_minus_root=("native_minus_root", "median"),
                                   mean_root_minus_native=("root_minus_native", "mean"),
                                   median_root_minus_native=("root_minus_native", "median")).reset_index())
    table.to_csv(out / "classes.csv", index=False, float_format="%.6g", lineterminator="\n")
    report["classes"] = table.to_dict("records")

    blocks = pd.read_parquet(RUNS / "scientific_native_full/blocks.parquet",
                             columns=["block_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "synergy_loewe",
                                      "native_sum_score", "native_hsa_fallback_score", "surface_id"])
    artifact_at_root = cases["class"].eq("artifact: solver at root, answer discarded").to_numpy()
    artifact_all = cases["class"].str.startswith("artifact").to_numpy()
    increasing = cases.independent_status.eq("root_increasing").to_numpy()
    effects, shifts = [], {}
    for mask, label, key in [(artifact_at_root, "artifact: solver answer at the root discarded", "solver_root"),
                             (artifact_all & increasing, "artifact: every admissible root, co-increasing curves (overnight audit set)", "valid_root"),
                             (artifact_all, "artifact: every admissible root, incl. co-decreasing curves", "valid_root_any")]:
        e, s = score_effects(cases, blocks, mask, label)
        effects.append(e)
        shifts[key] = s
    effects.append(convention_effects(blocks))
    report["score_effects"] = effects
    panel = panel_labels(blocks, shifts)
    panel.to_parquet(out / "panel_labels.parquet", index=False)
    report["panel"] = {k: {"units_changed": int((np.abs(panel[k] - panel.published) > 1e-9).sum()),
                           "units_abs_shift_ge_5": int((np.abs(panel[k] - panel.published) >= 5).sum()),
                           "spearman_with_published": float(panel[k].rank().corr(panel.published.rank()))} for k in shifts}

    labels = list(shifts)
    pred = pd.read_parquet(RUNS / "phase2_panel_neural/predictions.parquet")
    pred = pred[pred.target == "synergy_loewe"].drop(columns=["target", "truth", "observation_id"]).astype({c: str for c in ["pair_id", "drug_a", "drug_b", "cell_line"]})
    report["evaluation_effect"] = model_effects(pred, panel, labels)
    if a.retrained:
        r = pd.read_parquet(a.retrained).astype({c: str for c in ["pair_id", "drug_a", "drug_b", "cell_line"]})
        training = {}
        for target in ["published"] + labels:
            own = r[r.target == target].drop(columns=["target", "observation_id"])
            truth_check = own.merge(panel, on=["pair_id", "drug_a", "drug_b", "cell_line", "fold"])
            assert np.allclose(truth_check.truth, truth_check[target], atol=1e-6), target
            training[target] = model_effects(own.drop(columns="truth"), panel, labels)[target]
        reproduced = r[r.target == "published"].merge(pred, on=["pair_id", "drug_a", "drug_b", "cell_line", "fold"], suffixes=("", "_existing"))
        report["training_effect"] = {"reproduces_existing_predictions_max_abs": float((reproduced.prediction - reproduced.prediction_existing).abs().max()),
                                     "by_training_label": training}
    cases[["case_id", "surface_id", "native_code", "fallback", "strict", "strict_status", "independent", "independent_status",
           "residual_at_independent", "admissible", "class"]].to_parquet(out / "cases_verified.parquet", index=False)
    (out / "loewe_verify.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ["independent_vs_strict", "admissibility", "independent_status", "panel"]}, indent=1, default=str))
    print(table.to_string())
    print(pd.DataFrame(effects).drop(columns="block_abs_shift_ge").to_string())


if __name__ == "__main__":
    main()
