"""Build raw O'Neil combination-inhibition targets for Neighborhood CF."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "drugcomb_oneil"
SOURCE = DATA_DIR / "oneil_dose_level.npz"
MANIFEST = DATA_DIR / "manifest.json"
OUT_DIR = ROOT / "results" / "neighborhood_cf"

sys.path.insert(0, str(ROOT / "experiments" / "preliminary"))
from common import load_oneil

ALIASES = {
    "NCI-H460": "NCIH460",
    "SK-OV-3": "SKOV3",
    "SW-620": "SW620",
    "T-47D": "T47D",
    "UWB1289+BRCA1": "UWB1289BRCA1",
}


def main():
    if not SOURCE.exists() or not MANIFEST.exists():
        raise FileNotFoundError(
            "Missing processed O'Neil data. Run the team's "
            "r16_extract_oneil_raw.py extraction script first."
        )

    manifest = json.loads(MANIFEST.read_text())
    drugs = np.array(manifest["drugs"])
    cells = np.array(manifest["cells"])

    with np.load(SOURCE, allow_pickle=False) as z:
        block_id = z["block_id"]
        conc_r = z["conc_r"]
        conc_c = z["conc_c"]
        inhibition = z["inhibition"]
        drug_row = z["drug_row"]
        drug_col = z["drug_col"]
        cell_id = z["cell"]

    if not np.isfinite(inhibition).all():
        raise ValueError("Inhibition contains non-finite values.")

    unique_blocks, inverse, counts = np.unique(
        block_id, return_inverse=True, return_counts=True
    )

    if not np.all(counts == 25):
        raise ValueError("Expected exactly 25 observations per block.")

    positive = (conc_r > 0) & (conc_c > 0)
    positive_counts = np.bincount(
        inverse[positive], minlength=len(unique_blocks)
    )

    if not np.all(positive_counts == 16):
        raise ValueError("Expected exactly 16 positive-positive rows per block.")

    # Verify that every block represents one drug pair and one cell line.
    first = np.full(len(unique_blocks), len(block_id), dtype=int)
    np.minimum.at(first, inverse, np.arange(len(block_id)))

    for arr, name in [
        (drug_row, "row drug"),
        (drug_col, "column drug"),
        (cell_id, "cell line"),
    ]:
        if not np.all(arr == arr[first][inverse]):
            raise ValueError(f"Inconsistent {name} within a replicate block.")

    # Average the 16 positive-positive dose observations within each block.
    block_sums = np.bincount(
        inverse[positive],
        weights=inhibition[positive],
        minlength=len(unique_blocks),
    )
    block_means = block_sums / positive_counts

    drug_a = drugs[drug_row[first]]
    drug_b = drugs[drug_col[first]]
    pair_a = np.where(drug_a <= drug_b, drug_a, drug_b)
    pair_b = np.where(drug_a <= drug_b, drug_b, drug_a)
    block_cells = np.array([
        ALIASES.get(c, c) for c in cells[cell_id[first]]
    ])

    # Average replicate-block means for each pair and cell line.
    df = pd.DataFrame({
        "a": pair_a,
        "b": pair_b,
        "cell": block_cells,
        "block_mean": block_means,
    })

    grouped = df.groupby(["a", "b", "cell"], sort=True).agg(
        response=("block_mean", "mean"),
        n_blocks=("block_mean", "size"),
    )

    # Align exactly with the team's shared pair and cell ordering.
    shared = load_oneil()
    pairs = [tuple(p) for p in shared["pairs"]]
    shared_cells = list(shared["cells"])
    folds = shared["folds"].astype(int)

    Y = np.full((len(pairs), len(shared_cells)), np.nan)
    n_blocks = np.zeros(Y.shape, dtype=int)

    for i, (a, b) in enumerate(pairs):
        for j, cell in enumerate(shared_cells):
            key = (a, b, cell)
            if key in grouped.index:
                Y[i, j] = grouped.loc[key, "response"]
                n_blocks[i, j] = grouped.loc[key, "n_blocks"]

    if Y.shape != (583, 39):
        raise ValueError(f"Unexpected target shape: {Y.shape}")

    if not np.isfinite(Y).all():
        raise ValueError("Some pair-cell targets are missing.")

    if np.any(n_blocks == 0):
        raise ValueError("Some pair-cell units have no replicate blocks.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_DIR / "raw_response_targets.npz",
        Y=Y,
        n_blocks=n_blocks,
        pairs=np.array(pairs, dtype=str),
        cells=np.array(shared_cells, dtype=str),
        folds=folds,
    )

    report = {
        "target": (
            "mean raw combination inhibition across 16 positive-positive "
            "dose cells, then across replicate blocks"
        ),
        "n_pairs": len(pairs),
        "n_cells": len(shared_cells),
        "target_shape": list(Y.shape),
        "missing_pair_cell_targets": int(np.isnan(Y).sum()),
        "complete_pair_cell_targets": int(np.isfinite(Y).sum()),
        "min_blocks_per_observed_unit": int(n_blocks.min()),
        "max_blocks_per_unit": int(n_blocks.max()),
        "fold_counts": {
            str(f): int((folds == f).sum())
            for f in sorted(np.unique(folds))
        },
        "response_min": float(Y.min()),
        "response_max": float(Y.max()),
        "response_mean": float(Y.mean()),
    }

    (OUT_DIR / "target_validation.json").write_text(
        json.dumps(report, indent=2)
    )

    print(json.dumps(report, indent=2))
    print("\nTarget matrix and validation report saved.")


if __name__ == "__main__":
    main()
