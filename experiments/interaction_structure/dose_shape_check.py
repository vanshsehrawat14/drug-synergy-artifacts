"""Recheck of the dose-shape noise share under the frozen grid split and the batch-aware split.

Uses the contrast operators of manuscript/audit/interaction_audit.py on the unit-level Bliss residual.

python experiments/interaction_structure/dose_shape_check.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --output results/interaction_structure
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "manuscript" / "audit"))
from core import Data  # noqa: E402
from interaction_audit import Operators  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "output"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data = Data(a.observations, a.resolution)
    u = data.units.copy()
    u["pc"] = pd.factorize(u.pair_id + "|" + u.cell_line, sort=True)[0]
    u["pair"] = pd.factorize(u.pair_id, sort=True)[0]
    first = data.comb.groupby("unit").head(1).sort_values("unit")
    for c in ["drug_a", "drug_b", "fold"]:
        u[c] = first[c].to_numpy()
    ops = Operators(u, np.ones(len(u), bool))
    target = (data.truth_u - data.cache["bliss"])[:, None]
    _, ew, _ = ops.energies(target)
    T = ew[:, 0].sum()
    out = {"target_energy_per_row": float(T / data.n_u.sum())}
    for kind in ["grid", "batch", "panel"]:
        signs = data.split_signs(50, 97, kind)
        noise = 0.0
        for r0 in range(0, 50, 10):
            cols = []
            for r in range(r0, r0 + 10):
                s = signs[data.row_block, r].astype(float)
                cols.append(np.bincount(data.unit_of_row, weights=s * data.y) / data.n_u)
            _, wn, _ = ops.energies(np.column_stack(cols))
            noise += wn.sum() / 50
        out[f"dose_shape_bliss_noise_share_{kind}_split"] = float(noise / T)
        print(kind, out[f"dose_shape_bliss_noise_share_{kind}_split"], flush=True)
    out["frozen_reference"] = {"grid": 0.36334704817338515, "panel": 0.4473364977574429}
    Path(a.output, "dose_shape_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
