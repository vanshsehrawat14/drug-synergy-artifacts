"""Checks on GP out-of-fold predictions before they enter the shared comparison.

python -m synergy_artifacts.gp_verify --output results/gp/verification.json --method gp=DIR [--reproduce CELL:FOLD ...]
    [--decision results/gp/decision.json --config configs/gp_study.json --study results/neural/runs/gp_study]

For each method directory:
- folds 0-4 each cover exactly their test rows once, finite, all 39 cell lines; 1,475,328 rows in total;
- every pair lies in one published fold, so the training folds of model k share no pair with fold k or with its
  validation fold (k+1)%5;
- selection.csv: validation fold (f+1)%5, training rows equal to the cell's rows outside folds k and (k+1)%5
  (or a cap below that), one selected family per cell and fold;
- --reproduce refits the named cell and fold with the saved settings (synergy_artifacts.gp_shared or synergy_artifacts.gp_gpu) and
  compares the predictions with the saved file.
With --decision, the frozen selection is recomputed from the study files and compared with the saved one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_json
from .neural_report import load_truth


def check_method(directory, df):
    directory = Path(directory)
    out = {"directory": str(directory), "folds": {}}
    total, cells = 0, set()
    for k in range(5):
        frame = pd.read_parquet(directory / f"fold_{k}.parquet")
        expected = set(df.observation_id[df.fold == k])
        ok = (not frame.observation_id.duplicated().any()) and set(frame.observation_id) == expected
        finite = bool(np.isfinite(frame.prediction).all() and ("predictive_std" not in frame or np.isfinite(frame.predictive_std).all()))
        out["folds"][k] = {"rows": len(frame), "covers_test_rows_once": ok, "finite": finite,
                           "cells": int(frame.cell_line.nunique())}
        total += len(frame)
        cells |= set(frame.cell_line)
    pair_folds = df.groupby("pair_id", observed=True).fold.nunique()
    out["rows"], out["cells"], out["pairs_in_one_fold"] = total, len(cells), bool((pair_folds == 1).all())
    sel = pd.read_csv(directory / "selection.csv")
    if "selected" not in sel:  # earlier adapter layout: one row per cell and fold, the chosen family
        sel = sel.assign(selected=True)
    exact = sel[sel.predictor == "exact"] if "predictor" in sel else sel
    counts = df.groupby(["cell_line", "fold"], observed=True).size().unstack(fill_value=0)
    available = {(c, k): int(counts.loc[c].drop([k, (k + 1) % 5]).sum()) for c in counts.index for k in range(5)}
    col = "n_train_available" if "n_train_available" in exact else "n_rows"
    out["selection"] = {
        "validation_fold_is_next": bool((exact.validation_fold == (exact.fold + 1) % 5).all()),
        "training_rows_match_or_cap": bool(all(getattr(r, col) <= available[(r.cell_line, r.fold)] for r in exact.itertuples())),
        "training_rows_equal_available": bool(all(getattr(r, col) == available[(r.cell_line, r.fold)] for r in exact.itertuples())),
        "one_selected_family_per_cell_fold": bool((exact[exact.selected].groupby(["cell_line", "fold"]).size() == 1).all()
                                                  and exact[exact.selected].groupby(["cell_line", "fold"]).ngroups == 39 * 5),
    }
    out["passed"] = bool(total == len(df) and len(cells) == 39 and out["pairs_in_one_fold"]
                         and all(f["covers_test_rows_once"] and f["finite"] for f in out["folds"].values())
                         and out["selection"]["validation_fold_is_next"] and out["selection"]["one_selected_family_per_cell_fold"])
    return out


def reproduce(directory, cell_name, fold):
    directory = Path(directory)
    info = json.loads((directory / "run.json").read_text())
    saved = pd.read_parquet(directory / f"fold_{fold}.parquet")
    saved = saved[saved.cell_line == cell_name].set_index("observation_id")
    if "config" in info:
        from .gp_gpu import DEFAULT, fit_cell, load_cells, load_plan
        plan = load_plan(info["config"])
        cells, _, _ = load_cells(plan["dataset"], [cell_name])
        cell, mono = cells[cell_name]
        method = {k: v for k, v in info["settings"].items() if k in DEFAULT}
        pred, _, _ = fit_cell(cell, mono, fold, method, info["settings"]["seed"])
    else:
        from .gp_shared import HIS, _cells, fit_cell
        cells, _, _ = _cells("data/processed/oneil_v1")
        settings = info["settings"]
        if "cap" in settings:
            method = {k: settings[k] for k in ["cap", "families", "interaction_weight", "shared_similarity", "predictor"]}
        else:  # earlier adapter record: his procedure with a row cap
            method = {**HIS, "cap": settings["training_cap_rows"], "families": settings["families"]}
        pred, _ = fit_cell(cells[cell_name], fold, method["cap"], settings["seed"], settings.get("threads", 2), method)
    pred = pred.set_index("observation_id").reindex(saved.index)
    diff = np.abs(pred.prediction.to_numpy() - saved.prediction.to_numpy())
    return {"cell_line": cell_name, "fold": fold, "rows": len(saved), "max_abs_difference": float(diff.max()),
            "max_abs_std_difference": float(np.abs(pred.predictive_std.to_numpy() - saved.predictive_std.to_numpy()).max())}


def check_decision(decision, config, study):
    from .gp_select import decide, summarize
    summary, _ = summarize(study)
    again = decide(summary, json.loads(Path(config).read_text()))
    saved = json.loads(Path(decision).read_text())
    return {"selected_saved": saved["selected"], "selected_recomputed": again["selected"],
            "method_matches": saved["method"] == again["method"], "agrees": saved["selected"] == again["selected"]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--method", action="append", required=True, help="NAME=DIR")
    p.add_argument("--reproduce", nargs="*", default=[], help="CELL:FOLD, refit for every method")
    p.add_argument("--decision")
    p.add_argument("--config")
    p.add_argument("--study")
    a = p.parse_args()
    df, _, _ = load_truth("data/processed/oneil_v1")
    report = {}
    for spec in a.method:
        name, directory = spec.split("=", 1)
        report[name] = check_method(directory, df)
        report[name]["reproduced"] = [reproduce(directory, *(lambda c, f: (c, int(f)))(*r.split(":"))) for r in a.reproduce]
    if a.decision:
        report["frozen_selection"] = check_decision(a.decision, a.config, a.study)
    write_json(a.output, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
