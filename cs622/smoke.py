"""Run a self-contained synthetic data-to-prediction experiment without downloads."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .data import prepare
from .runner import run


def smoke(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    assignments = ["AB CD EF", "AC BE DF", "AD BF CE", "AE BD CF", "AF BC DE"]
    folds = {pair: fold for fold, group in enumerate(assignments) for pair in group.split()}
    raw, summary, labels = [], [], []
    for block, (a, b) in enumerate(itertools.combinations("ABCDEF", 2), 1):
        for x, y in itertools.product([0., .1, 1., 10., 100.], repeat=2):
            raw.append(dict(block_id=block, conc_r=x, conc_c=y,
                inhibition=10*np.log1p(x+y)+ord(a)-65+ord(b)-65,
                drug_row=a, drug_col=b, conc_r_unit="uM", conc_c_unit="uM",
                cell_line_name="SYNTHETIC", drug_row_cid=ord(a), drug_col_cid=ord(b),
                cellosaurus_accession="SYNTHETIC", study_name="ONEIL"))
        # These zero placeholders exercise joins; no simulated synergy claim.
        summary.append(dict(block_id=block, drug_row=a, drug_col=b, cell_line_name="SYNTHETIC",
            conc_r_unit="uM", conc_c_unit="uM", synergy_bliss=0., synergy_hsa=0.,
            synergy_loewe=0., synergy_zip=0., S=0., css=0.))
        labels.append(dict(drug_a_name=a, drug_b_name=b, cell_line="SYNTHETIC", synergy=0., fold=folds[a+b]))
    paths = [output / name for name in ["synthetic_raw.csv", "synthetic_summary.csv", "synthetic_labels.csv"]]
    for frame, path in zip([raw, summary, labels], paths):
        pd.DataFrame(frame).to_csv(path, index=False)
    prepared = output / "prepared"
    prepare(dict(raw=str(paths[0]), summary=str(paths[1]), labels=str(paths[2]), output=str(prepared), chunksize=100))
    config = dict(dataset=str(prepared), table="observations", target="inhibition", test_fold=0, seed=2026,
        features={"regime": "F0"}, model={"kind": "dose", "hidden": [16, 8], "encoder_dim": 8, "dropout": 0.},
        training={"epochs": 3, "batch_size": 32, "patience": 3, "threads": 1, "device": "cpu"}, bootstrap=20)
    return run(config, output / "run")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    _, metrics = smoke(a.output)
    print({"scope": "synthetic engineering check, not a scientific benchmark", "pooled": metrics["pooled"]})


if __name__ == "__main__":
    main()
