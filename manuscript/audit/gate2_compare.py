"""Compare a re-execution of the October Gate 2 pipeline (main, ae55c68) with its committed outputs.

Every numeric leaf of the committed JSON summaries is compared with the re-executed value; CSV run tables
are compared column by column. Prints and writes the largest absolute differences.

python manuscript/audit/gate2_compare.py --committed COMMITTED/results/preliminary --rerun RERUN/results/preliminary \
    --output manuscript/audit/outputs/gate2_reproduction.json
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

FILES = ["r08_audit.json", "r12_summary.json", "r13_summary.json", "r18_gate2.json", "r19_summary.json", "r20_fallback_mechanism.json"]
TABLES = ["r12_runs.csv", "r13_runs.csv", "r19_runs.csv", "r17_fits.csv"]
KEY = {"r18_gate2.json": ["P2.cr2_M_DC", "components.M_alg.cr2_vs_Y_L", "components.M_impl.cr2_vs_Y_L",
                          "components.M_exist.cr2_vs_Y_L", "P3.s0", "P3.gate2_pass", "P3.M_impl_energy_share_in_fallback_units",
                          "P3.fraction_units_with_fallback", "P4.runs.cell_independent:loewe_sim.knn.cr2",
                          "P4.runs.cell_independent:loewe_sim.krr.cr2", "P4.runs.cell_independent:bliss_sim.knn.cr2",
                          "P4.runs.cell_independent:bliss_sim.krr.cr2", "fidelity_vs_DC_block.exist_rule",
                          "fidelity_vs_DC_block_exact_frac.exist_rule"]}


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    c, r = Path(a.committed), Path(a.rerun)
    out = {}
    for f in FILES:
        A = dict(leaves(json.loads((c / f).read_text())))
        B = dict(leaves(json.loads((r / f).read_text())))
        diffs, mismatched, missing = [], [], sorted(set(A) ^ set(B))
        for k in sorted(set(A) & set(B)):
            x, y = A[k], B[k]
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                if x != y:
                    mismatched.append(k)
                continue
            if math.isnan(x) and math.isnan(y):
                continue
            diffs.append((abs(x - y), k))
        diffs.sort(reverse=True)
        out[f] = {"numeric_leaves": len(diffs), "max_abs_diff": diffs[0][0] if diffs else 0.0,
                  "largest": [[k, d] for d, k in diffs[:3]], "non_numeric_mismatches": mismatched[:10], "key_set_differences": missing[:10]}
        if f in KEY:
            out[f]["key_values"] = {k: {"committed": A.get(k), "rerun": B.get(k)} for k in KEY[f]}
    for f in TABLES:
        if not (c / f).exists() or not (r / f).exists():
            out[f] = "missing"
            continue
        x, y = pd.read_csv(c / f), pd.read_csv(r / f)
        num = [k for k in x.columns if k in y.columns and pd.api.types.is_numeric_dtype(x[k]) and pd.api.types.is_numeric_dtype(y[k])]
        same_shape = x.shape == y.shape
        md = max(((x[k] - y[k]).abs().max() for k in num), default=0.0) if same_shape else None
        out[f] = {"shape_committed": list(x.shape), "shape_rerun": list(y.shape), "max_abs_diff_numeric": None if md is None else float(md)}
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
