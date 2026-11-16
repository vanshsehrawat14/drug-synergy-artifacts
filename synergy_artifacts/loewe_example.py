"""Worked Loewe example and root-level experiment on the 98 native-fit probe cases.

python -m synergy_artifacts.loewe_example --output results/loewe_audit [--cases FULL_CASES --verified CASES_VERIFIED]

Input: results/loewe_audit/probe_98_cases.csv. These are native drc fits and SynergyFinder 2.0.12 outputs
(R 3.6.3, nleqslv 3.3.2); every case is a native fallback with an admissible root. Nothing here needs R.

For each probe case, from the fitted curve coefficients alone:
- the root of the scorer's equation by two solvers (bisection on forward curves; Brent on inverses);
- the scorer's residual at the native nleqslv iterate and at the root, and the slope dr/dy at the root;
- the summed-dose fallback recomputed and compared with the reference the scorer returned.

The worked example is the median fallback-minus-root gap among the code-3 cases whose iterate lies outside the
joint effect range and whose recomputed fallback matches the returned reference.

With --cases and --verified (loewe_audit/full_cases.parquet and loewe_verify/cases_verified.parquet), every case's
root is located relative to the nearer edge of the joint effect range, by verification class. The upper edge is the
weaker drug's plateau; the lower edge is the larger lower asymptote.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_json
from .loewe_audit import loewe_residual, params, strict_reference
from .loewe_verify import curve, theta_root


def joint_range(frame):
    lo = np.maximum(np.minimum(frame.a_c, frame.a_d), np.minimum(frame.b_c, frame.b_d))
    hi = np.minimum(np.maximum(frame.a_c, frame.a_d), np.maximum(frame.b_c, frame.b_d))
    return lo.to_numpy(float), hi.to_numpy(float)


def recompute(probe):
    rows = []
    bisect, bisect_status = theta_root(probe)
    for r, root_b in zip(probe.itertuples(), bisect):
        pa, pb = params(r, "a"), params(r, "b")
        root, status = strict_reference(r.dose_a, r.dose_b, r.model_a, pa, r.model_b, pb)
        res = lambda y: loewe_residual(y, r.dose_a, r.dose_b, r.model_a, pa, r.model_b, pb)
        h = 1e-6
        slope = (res(root + h) - res(root - h)) / (2 * h) if np.isfinite(root) else np.nan
        total = np.log(r.dose_a + r.dose_b)
        fallback = max(curve(total, np.array(r.model_a), *pa), curve(total, np.array(r.model_b), *pb))
        rows.append({"case_id": r.case_id, "block_id": r.block_id, "drug_a": r.drug_a, "drug_b": r.drug_b,
                     "cell_line": r.cell_line, "dose_a": r.dose_a, "dose_b": r.dose_b, "native_code": r.native_code,
                     "native_iterate": r.solution, "root_brent": root, "root_bisection": root_b,
                     "residual_at_iterate": res(r.solution), "residual_at_root": res(root), "slope_at_root": slope,
                     "iterate_minus_root": r.solution - root, "fallback_recomputed": float(min(fallback, 100)),
                     "reference_returned": r.native_reference, "fallback_minus_root": float(min(fallback, 100)) - root})
    table = pd.DataFrame(rows)
    lo, hi = joint_range(probe)
    table["iterate_outside_range"] = (probe.solution.to_numpy() <= lo) | (probe.solution.to_numpy() >= hi)
    table["root_to_edge"] = np.minimum(table.root_brent - lo, hi - table.root_brent)
    return table


def edge_by_class(cases_path, verified_path):
    cases = pd.read_parquet(cases_path, columns=["case_id", "native_solution", "a_c", "a_d", "b_c", "b_d"])
    verified = pd.read_parquet(verified_path, columns=["case_id", "class", "independent"])
    if not (cases.case_id.to_numpy() == verified.case_id.to_numpy()).all():
        raise ValueError("Case tables are not aligned")
    lo, hi = joint_range(cases)
    upper, lower = hi - verified.independent, verified.independent - lo
    edge = np.minimum(upper, lower)
    outside = (cases.native_solution <= lo) | (cases.native_solution >= hi)
    frame = pd.DataFrame({"cls": verified["class"], "edge": edge, "upper": upper < 0.01, "lower": lower < 0.01, "outside": outside})
    bins = np.arange(-14, 2.5, 0.5)
    hist = []
    for cls, g in frame.groupby("cls"):
        e = g.edge[g.edge > 0]
        if len(e):
            counts = np.histogram(np.log10(e), bins=bins)[0]
            hist += [{"class": cls, "log10_low": lo_, "log10_high": hi_, "cases": int(n)} for lo_, hi_, n in zip(bins[:-1], bins[1:], counts)]
    out = {"histogram": hist}
    for cls, g in frame.groupby("cls"):
        e = g.edge.dropna()
        out[cls] = {"cases": len(g), "root_to_edge_median": float(e.median()) if len(e) else None,
                    "within_0.01": int((e < 0.01).sum()), "within_0.01_of_upper_edge": int(g.upper.sum()),
                    "within_0.01_of_lower_edge": int(g.lower.sum()), "within_0.1": int((e < 0.1).sum()),
                    "native_iterate_outside_range": int(g.outside.sum())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--probe", default="results/loewe_audit/probe_98_cases.csv")
    p.add_argument("--blocks", default="results/loewe_audit/affected_blocks.csv")
    p.add_argument("--output", default="results/loewe_audit")
    p.add_argument("--cases", help="loewe_audit/full_cases.parquet")
    p.add_argument("--verified", help="loewe_verify/cases_verified.parquet")
    a = p.parse_args()
    probe = pd.read_csv(a.probe)
    table = recompute(probe)
    code3 = table[(table.native_code == 3) & table.iterate_outside_range
                  & ((table.fallback_recomputed - table.reference_returned).abs() < 1e-6)].sort_values(["fallback_minus_root", "case_id"])
    example = code3.iloc[(len(code3) - 1) // 2]
    blocks = pd.read_csv(a.blocks)
    block = blocks[blocks.block_id == example.block_id]
    summary = {
        "cases": len(table), "native_codes": {str(k): int(v) for k, v in table.native_code.value_counts().items()},
        "solvers_max_abs_difference": float(np.nanmax(np.abs(table.root_brent - table.root_bisection))),
        "max_abs_residual_at_root": float(np.nanmax(np.abs(table.residual_at_root))),
        "code3": {"cases": int((table.native_code == 3).sum()),
                  "iterate_outside_range": int(table[table.native_code == 3].iterate_outside_range.sum()),
                  "median_abs_iterate_minus_root": float(table[table.native_code == 3].iterate_minus_root.abs().median()),
                  "median_root_to_edge": float(table[table.native_code == 3].root_to_edge.median()),
                  "median_abs_slope_at_root": float(table[table.native_code == 3].slope_at_root.abs().median())},
        "error_cases_median_root_to_edge": float(table[table.native_code == 999].root_to_edge.median()),
        "fallback_recomputed_vs_returned": {"max_abs_difference": float((table.fallback_recomputed - table.reference_returned).abs().max()),
                                            "cases_within_1e-6": int(((table.fallback_recomputed - table.reference_returned).abs() < 1e-6).sum())},
        "fallback_minus_root": {"median": float(table.fallback_minus_root.median()), "min": float(table.fallback_minus_root.min()),
                                "max": float(table.fallback_minus_root.max()), "positive": int((table.fallback_minus_root > 0).sum())},
        "example": {k: (None if isinstance(v, float) and not np.isfinite(v) else (v.item() if hasattr(v, "item") else v))
                    for k, v in example.items()},
        "example_block": block.drop(columns=["pair_id"]).to_dict("records")[0] if len(block) else None,
    }
    out = Path(a.output)
    if a.cases and a.verified:
        by_class = edge_by_class(a.cases, a.verified)
        pd.DataFrame(by_class.pop("histogram")).to_csv(out / "edge_histogram.csv", index=False, lineterminator="\n")
        summary["all_cases_by_class"] = by_class
    table.to_csv(out / "probe_recomputed.csv", index=False, float_format="%.10g", lineterminator="\n")
    write_json(out / "example.json", summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
