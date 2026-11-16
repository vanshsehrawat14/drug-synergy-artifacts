"""Transparent response/reference controls. No Loewe or ZIP approximation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, write_json


def bliss_reference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a + b - a * b / 100


def hsa_reference(a, b):
    return np.maximum(a, b)


def bliss_null(a, b, seed=2026, noise_sd=0.):
    """Bliss-independent inhibition plus explicit iid Gaussian measurement noise.

    This generates a null under Bliss, not under Loewe or every synergy model.
    """
    ref = bliss_reference(a, b)
    if noise_sd < 0:
        raise ValueError("Noise SD must be nonnegative")
    return ref + np.random.default_rng(seed).normal(0, noise_sd, size=ref.shape)


def audit_reported_scores(dataset, output):
    """Retrospective within-block reconstruction; not an inference-time feature."""
    directory, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    columns = ["block_id", "drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition", "is_combination", "is_monotherapy"]
    raw = pd.read_parquet(directory / "observations.parquet", columns=columns)
    combos = raw[raw.is_combination].copy()
    for endpoint, other in [("a", "b"), ("b", "a")]:
        edges = raw[(raw["dose_" + endpoint] > 0) & (raw["dose_" + other] == 0)]
        keys = ["block_id", "dose_" + endpoint]
        edges = edges[keys + ["inhibition"]].rename(columns={"inhibition": "mono_" + endpoint})
        combos = combos.merge(edges, on=keys, how="left", validate="many_to_one")
    if not np.isfinite(combos[["mono_a", "mono_b"]].to_numpy(float)).all():
        raise ValueError("Incomplete block-edge reference reconstruction")
    combos["reference_bliss"] = bliss_reference(combos.mono_a, combos.mono_b)
    combos["reference_hsa"] = hsa_reference(combos.mono_a, combos.mono_b)
    observed = combos.groupby("block_id", sort=True)[["inhibition", "reference_bliss", "reference_hsa"]].mean()
    blocks = pd.read_parquet(directory / "blocks.parquet")
    observed = blocks.merge(observed, on="block_id", how="left", validate="one_to_one")
    for name in ["bliss", "hsa"]:
        observed["reconstructed_" + name] = observed.inhibition - observed["reference_" + name]
        observed["difference_" + name] = observed["reconstructed_" + name] - observed["synergy_" + name]
    # T - reported Loewe is explicitly outcome-dependent accounting, not a predictor.
    observed["retrospective_loewe_reference"] = observed.response - observed.synergy_loewe
    observed["loewe_minus_bliss_reported"] = observed.synergy_loewe - observed.synergy_bliss
    output.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(output / "block_comparison.parquet", index=False)
    report = {"scope": "retrospective score audit; held-out-block edges are never model features",
              "blocks": len(observed), "source_sha256": digest(directory / "observations.parquet"),
              "max_abs_difference": {name: float(observed["difference_" + name].abs().max()) for name in ["bliss", "hsa"]},
              "loewe": "reported; T minus label is outcome-dependent accounting, not independent reconstruction",
              "zip": "reported only; reconstruction not implemented", "S": "reported block statistic; not dose-level score"}
    write_json(output / "audit.json", report)
    print(report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/processed/oneil_v1")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit_reported_scores(args.dataset, args.output)


if __name__ == "__main__":
    main()
