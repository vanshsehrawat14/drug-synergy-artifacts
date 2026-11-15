"""Shar's neighborhood collaborative filtering on the shared five-fold benchmark, and a validation-only extension.

Her model (experiments/neighborhood_cf/run_neighborhood_cf.py, on the residual kNN in
experiments/preliminary/pairlearners.py) predicts one value per drug pair and cell line: the mean inhibition over the
16 combination doses and the replicate blocks. A drug-in-cell additive ridge is fitted on training pairs; for a
held-out pair (x, y) it adds the similarity-weighted mean residual of the k training pairs (x, z) whose z is most like
y, averaged over both orientations. Drug similarity is the Pearson correlation of training residuals over common
partners, shrunk by n / (n + shrink). k, shrink and the ridge lambda are chosen on validation fold (k+1)%5; nothing is
refitted. The held-out pair enters neither the ridge, the similarities nor the neighbour sets.

python experiments/neighborhood_cf/build_raw_response_targets.py
python -m cs622.cf study --config configs/cf_study.json --output results/neural/runs/cf_study
python -m cs622.cf select --config configs/cf_study.json --study results/neural/runs/cf_study --output results/cf
python -m cs622.cf run --config configs/cf_study.json --cf shar --dose none --output results/neural/runs/cf_shar
python -m cs622.cf run --config configs/cf_study.json --decision results/cf/decision.json --output results/neural/runs/cf_selected

`--cf shar --dose none` is her method unchanged: her predict_cf per test fold, its pair-cell value given to every dose
row of that pair and cell line. The extension (dose profile and the variants in configs/cf_study.json) is Vansh's.
"""
import argparse
import importlib.metadata
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .io import digest, read_json, write_json
from .runner import git_state
from .sprint_report import index, load_truth

ROOT = Path(__file__).resolve().parents[1]
DATASET = "data/processed/oneil_v1"
TARGETS = ROOT / "results" / "neighborhood_cf" / "raw_response_targets.npz"
SOURCES = [ROOT / "experiments" / "neighborhood_cf" / "run_neighborhood_cf.py",
           ROOT / "experiments" / "preliminary" / "pairlearners.py", ROOT / "experiments" / "preliminary" / "common.py"]
COLUMNS = ["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "block_id", "dose_a", "dose_b"]


def shar():
    """Her evaluation module and the preliminary helpers it imports, unchanged."""
    for d in ["preliminary", "neighborhood_cf"]:
        path = str(ROOT / "experiments" / d)
        if path not in sys.path:
            sys.path.insert(0, path)
    import run_neighborhood_cf
    return run_neighborhood_cf


def norm(name):
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def load(dataset=DATASET, targets=TARGETS):
    """Her pair-cell matrix aligned to the benchmark rows, with each drug's dose rank within its block.

    Checks that her pair, cell and fold order is the shared one, that her folds equal the benchmark folds, that every
    target equals the mean of that pair-cell's benchmark rows and that every block is a full 4 x 4 dose grid."""
    cf = shar()
    with np.load(targets, allow_pickle=False) as z:
        Y, folds = z["Y"].astype(float), z["folds"].astype(int)
        pairs, cells = [tuple(p) for p in z["pairs"].tolist()], list(z["cells"].tolist())
    shared = cf.load_oneil()
    if [tuple(p) for p in shared["pairs"]] != pairs or list(shared["cells"]) != cells or not np.array_equal(shared["folds"], folds):
        raise ValueError("Target matrix is not in the shared pair, cell and fold order")
    df, dataset_sha, _ = load_truth(dataset)
    t = index(df)
    pair_of = {frozenset(map(norm, p)): i for i, p in enumerate(pairs)}
    pi = np.array([pair_of[frozenset((norm(a), norm(b)))] for a, b in t["pair_drugs"]])[t["pair"]]
    cj = np.array([cells.index(c) for c in t["cells"]])[t["cell"]]
    fold = df.fold.to_numpy(int)
    if len(pair_of) != len(pairs) or not np.array_equal(folds[pi], fold):
        raise ValueError("Her pair folds differ from the benchmark folds")
    pc = pi * len(cells) + cj
    n = np.bincount(pc, minlength=Y.size)
    if (n == 0).any() or not np.allclose(np.bincount(pc, weights=t["y"], minlength=Y.size) / n, Y.ravel(), rtol=0, atol=1e-9):
        raise ValueError("Her targets differ from the pair-cell means of the benchmark rows")
    block = df.groupby("block_id")
    ra = block.dose_a.rank(method="dense").to_numpy(int) - 1
    rb = block.dose_b.rank(method="dense").to_numpy(int) - 1
    _, b = np.unique(df.block_id.to_numpy(), return_inverse=True)
    if ra.max() > 3 or rb.max() > 3 or (np.bincount(b * 16 + ra * 4 + rb, minlength=16 * (b.max() + 1)) != 1).any():
        raise ValueError("Every block must be a full 4 x 4 grid of dose ranks")
    drugs = sorted(set(df.drug_a.astype(str)) | set(df.drug_b.astype(str)))
    code = {d: i for i, d in enumerate(drugs)}
    return {"Y": Y, "H": shared["H"], "aa": shared["aa"], "bb": shared["bb"], "folds": folds, "cells": cells,
            "pc_rows": n.reshape(Y.shape), "df": df, "dataset_sha256": dataset_sha, "y": t["y"], "fold": fold, "pi": pi,
            "cell": cj, "e": t["y"] - Y[pi, cj], "ra": ra, "rb": rb, "n_drugs": len(drugs),
            "da": df.drug_a.astype(str).map(code).to_numpy(int), "db": df.drug_b.astype(str).map(code).to_numpy(int)}


# ---- pair-cell stage: her predict_cf with a declared grid and similarity --------------------------------------------

def similarity(M, W, shrink, kind):
    """Drug similarity of training residuals over common partners, shrunk by n / (n + shrink).

    pearson is her pairlearners.drug_sim_from_resid; cosine drops the centring; cell returns one matrix per cell line,
    the mean of the same-cell Pearson similarity and the all-cell one."""
    L = shar().L
    if kind == "pearson":
        return L.drug_sim_from_resid(M, W, shrink=shrink)
    nd, C = M.shape[0], M.shape[2]
    base = L.drug_sim_from_resid(M, W, shrink=shrink) if kind == "cell" else None
    S = np.eye(nd) if kind == "cosine" else np.repeat(np.eye(nd)[None], C, 0)
    for i in range(nd):
        for j in range(i + 1, nd):
            com = (W[i] * W[j]) > 0
            com[i] = com[j] = False
            n = com.sum()
            if n < 2:
                continue
            x, y = M[i, com], M[j, com]
            if kind == "cosine":
                x, y = x.ravel(), y.ravel()
                d = np.sqrt((x @ x) * (y @ y))
                S[i, j] = S[j, i] = (x @ y / d if d > 0 else 0.0) * n / (n + shrink)
            else:
                xc, yc = x - x.mean(0), y - y.mean(0)
                d = np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0))
                r = np.divide((xc * yc).sum(0), d, out=np.zeros(C), where=d > 0)
                S[:, i, j] = S[:, j, i] = 0.5 * (r * n / (n + shrink) + base[i, j])
    return S


def neighbours(S, M, W, rows, k, aa, bb, kind):
    L = shar().L
    if kind != "cell":
        return L.knn(S, M, W, rows, k, aa=aa, bb=bb)
    return np.column_stack([L.knn(S[c], M[:, :, c:c + 1], W, rows, k, aa=aa, bb=bb)[:, 0] for c in range(M.shape[2])])


def pair_cell(data, k, spec, lambdas, predict_test=False):
    """Her predict_cf for test fold k with the declared grid and similarity: validation (and test) pair-cell predictions."""
    cf = shar()
    Y, H, aa, bb, kind = data["Y"], data["H"], data["aa"], data["bb"], spec["similarity"]
    tr, va, te = cf.split(data["folds"], k)
    lam, beta = cf.ridge_additive(H, Y, tr, va, lams=lambdas)
    base = H @ beta
    M, W = cf.L.residual_rows(Y - base, tr, aa=aa, bb=bb, nd=H.shape[1])
    best = None
    for shrink in spec["shrink"]:
        S = similarity(M, W, shrink, kind)
        for kk in spec["k"]:
            v = base[va] + neighbours(S, M, W, va, kk, aa, bb, kind)
            mse = float(np.mean((v - Y[va]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, kk, shrink, S, v)
    mse, kk, shrink, S, v = best
    out = {"va": va, "te": te, "validation": v, "k": kk, "shrink": shrink, "lambda": lam, "validation_mse": mse}
    if predict_test:
        out["test"] = base[te] + neighbours(S, M, W, te, kk, aa, bb, kind)
    return out


# ---- dose stage: within-pair-cell profile from training rows --------------------------------------------------------

def design(rows, data, partner):
    """Drug-by-dose-rank one-hot features (partner: also by the partner's rank) minus their mean over the pair-cell's
    balanced 4 x 4 grid, so every profile sums to zero within its pair-cell. Columns are local to one cell line."""
    da, db, ra, rb = (data[x][rows] for x in ["da", "db", "ra", "rb"])
    levels, n, r = (16 if partner else 4), len(rows), np.arange(len(rows))
    cols = np.r_[da * levels + (ra * 4 + rb if partner else ra), db * levels + (rb * 4 + ra if partner else rb)]
    shape, grid = (n, data["n_drugs"] * levels), np.arange(levels)
    hot = sparse.csr_matrix((np.ones(2 * n), (np.r_[r, r], cols)), shape=shape)
    every = np.r_[(da[:, None] * levels + grid).ravel(), (db[:, None] * levels + grid).ravel()]
    mean = sparse.csr_matrix((np.full(2 * n * levels, 1 / levels), (np.repeat(np.r_[r, r], levels), every)), shape=shape)
    return (hot - mean).tocsr()


def dose_profile(data, k, partner, lambdas, predict_test=False):
    """Ridge fit per cell line of each training row's deviation from its pair-cell mean; one lambda per fold, chosen on
    validation rows. Test rows contribute only their drugs, cell line and dose ranks."""
    fold, cell, e = data["fold"], data["cell"], data["e"]
    train, val, test = ~np.isin(fold, [k, (k + 1) % 5]), fold == (k + 1) % 5, fold == k
    pred = np.zeros((len(lambdas), len(fold)))
    for c in range(len(data["cells"])):
        rows = np.flatnonzero(train & (cell == c))
        X = design(rows, data, partner)
        s, V = np.linalg.eigh((X.T @ X).toarray())
        proj = V.T @ (X.T @ e[rows])
        targets = [np.flatnonzero(val & (cell == c))] + ([np.flatnonzero(test & (cell == c))] if predict_test else [])
        for where in targets:
            Xw = design(where, data, partner)
            for i, lam in enumerate(lambdas):
                pred[i, where] = Xw @ (V @ (proj / (s + lam)))
    sse = ((pred[:, val] - e[val]) ** 2).sum(1)
    best = int(np.argmin(sse))
    out = {"lambda": lambdas[best], "validation_sse": float(sse[best]), "validation_rows": int(val.sum()), "validation": pred[best, val]}
    if predict_test:
        out["test"] = pred[best, test]
    return out


# ---- study, selection and five-fold run ---------------------------------------------------------------------------

def _provenance(data):
    return {"dataset_sha256": data["dataset_sha256"], "target_sha256": digest(TARGETS),
            "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in SOURCES}, "git": git_state(),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy"]}}


def study(config, output):
    """Validation-only runs of every declared variant; no test-fold prediction is made."""
    plan, out = read_json(config), Path(output)
    if (out / "study.csv").exists():
        raise FileExistsError("Study exists; choose a new output")
    data, rows = load(), []
    started = time.perf_counter()
    for name, spec in plan["cf_stage"]["variants"].items():
        for k in range(5):
            t0 = time.perf_counter()
            r = pair_cell(data, k, spec, plan["cf_stage"]["ridge_lambdas"])
            w = data["pc_rows"][r["va"]]
            rows.append({"stage": "cf", "variant": name, "fold": k, "validation_fold": (k + 1) % 5, "k": r["k"], "shrink": r["shrink"],
                         "lambda": r["lambda"], "validation_mse_pair_cell": r["validation_mse"],
                         "validation_sse": float((w * (r["validation"] - data["Y"][r["va"]]) ** 2).sum()),
                         "validation_rows": int(w.sum()), "seconds": time.perf_counter() - t0})
            print(f"{time.strftime('%H:%M:%S')} cf {name} fold {k}", flush=True)
    for name in plan["dose_stage"]["variants"]:
        for k in range(5):
            t0 = time.perf_counter()
            if name == "none":
                val = data["fold"] == (k + 1) % 5
                r = {"lambda": None, "validation_sse": float((data["e"][val] ** 2).sum()), "validation_rows": int(val.sum())}
            else:
                r = dose_profile(data, k, name == "partner", plan["dose_stage"]["lambdas"])
            rows.append({"stage": "dose", "variant": name, "fold": k, "validation_fold": (k + 1) % 5, "lambda": r["lambda"],
                         "validation_sse": r["validation_sse"], "validation_rows": r["validation_rows"], "seconds": time.perf_counter() - t0})
            print(f"{time.strftime('%H:%M:%S')} dose {name} fold {k}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "study.csv", index=False)
    write_json(out / "study.json", {"config": str(config), "config_sha256": digest(config), "seconds": time.perf_counter() - started,
                                    **_provenance(data)})


def select(config, study_dir, output):
    plan, table = read_json(config), pd.read_csv(Path(study_dir) / "study.csv")
    pooled = table.groupby(["stage", "variant"], sort=False).agg(validation_sse=("validation_sse", "sum"), validation_rows=("validation_rows", "sum"),
                                                                folds=("fold", "nunique")).reset_index()
    pooled["validation_mse"] = pooled.validation_sse / pooled.validation_rows
    decision = {"config": str(config), "config_sha256": digest(config), "rule": plan["rule"]}
    for stage, key in [("cf", "cf_stage"), ("dose", "dose_stage")]:
        own = pooled[pooled.stage == stage].set_index("variant").validation_mse
        default = plan[key]["default"]
        better = [v for v in own.index if v not in (default, "none") and own[v] <= 0.99 * own[default]]
        decision[stage] = min(better, key=lambda v: own[v]) if better else default
        decision[f"{stage}_validation_mse"] = own.to_dict()
    rows = int(pooled.validation_rows.iloc[0])
    sse = pooled.set_index(["stage", "variant"]).validation_sse
    decision["validation_mse"] = {"selected": float((sse["cf", decision["cf"]] + sse["dose", decision["dose"]]) / rows),
                                  "shar_on_every_dose_row": float((sse["cf", "shar"] + sse["dose", "none"]) / rows)}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out / "study_summary.csv", index=False)
    write_json(out / "decision.json", decision)
    print(decision)


def run(config, output, cf_variant, dose_variant):
    plan, out = read_json(config), Path(output)
    if any((out / f"fold_{k}.parquet").exists() for k in range(5)):
        raise FileExistsError("Fold predictions exist; choose a new output")
    out.mkdir(parents=True, exist_ok=True)
    data, cf = load(), shar()
    spec, lambdas = plan["cf_stage"]["variants"][cf_variant], plan["cf_stage"]["ridge_lambdas"]
    started, records = time.perf_counter(), []
    for k in range(5):
        t0 = time.perf_counter()
        r = pair_cell(data, k, spec, lambdas, predict_test=True)
        if cf_variant == "shar":
            tr, va, te = cf.split(data["folds"], k)
            theirs, _, params = cf.predict_cf(data["Y"], data["H"], data["aa"], data["bb"], tr, va, te)
            if not np.array_equal(theirs, r["test"]) or (params["k"], params["shrink"]) != (r["k"], r["shrink"]):
                raise ValueError("pair_cell does not reproduce her predict_cf")
        matrix = np.full(data["Y"].shape, np.nan)
        matrix[r["te"]] = r["test"]
        rows = np.flatnonzero(data["fold"] == k)
        level = matrix[data["pi"][rows], data["cell"][rows]]
        d = dose_profile(data, k, dose_variant == "partner", plan["dose_stage"]["lambdas"], True) if dose_variant != "none" else None
        profile = d["test"] if d else np.zeros(len(rows))
        frame = data["df"].iloc[rows][COLUMNS].reset_index(drop=True).assign(
            prediction=level + profile, pair_cell_prediction=level, dose_profile=profile)
        if not np.isfinite(frame.prediction).all():
            raise ValueError(f"Fold {k} has nonfinite predictions")
        frame.to_parquet(out / f"fold_{k}.parquet", index=False)
        records.append({"fold": k, "validation_fold": (k + 1) % 5, "selected": True, "k": r["k"], "shrink": r["shrink"], "lambda": r["lambda"],
                        "validation_mse_pair_cell": r["validation_mse"], "dose_lambda": d["lambda"] if d else None,
                        "dose_validation_sse": d["validation_sse"] if d else None, "n_train_pairs": int((~np.isin(data["folds"], [k, (k + 1) % 5])).sum()),
                        "n_test_rows": len(rows), "seconds": time.perf_counter() - t0})
        print(f"{time.strftime('%H:%M:%S')} fold {k}", flush=True)
    pd.DataFrame(records).to_csv(out / "selection.csv", index=False)
    original = cf_variant == "shar" and dose_variant == "none"
    write_json(out / "run.json", {
        "method": "Neighborhood CF (Sharlene Kim), experiments/neighborhood_cf, unchanged" if original
        else f"Neighborhood CF (Sharlene Kim) with the cf_study extension: cf {cf_variant}, dose profile {dose_variant}",
        "model_family": "neighborhood_cf", "regime": "F0" if dose_variant != "none" else "identities only",
        "target": "inhibition; the CF stage predicts the pair-cell mean", "settings": {"cf": cf_variant, "dose": dose_variant, **spec,
        "ridge_lambdas": lambdas, "dose_lambdas": plan["dose_stage"]["lambdas"],
        "selection": "per fold on validation fold (k+1)%5; no refit"}, "config": str(config), "config_sha256": digest(config),
        "folds": list(range(5)), "seconds": time.perf_counter() - started, "device": "cpu", **_provenance(data),
        "prediction_sha256": {f"fold_{k}": digest(out / f"fold_{k}.parquet") for k in range(5)}})


def verify(methods, output, fold=0):
    """Coverage and pair separation for each run, then a refit of one fold with every test-pair label replaced by
    noise (her matrix and the benchmark rows): the saved predictions must come back bitwise."""
    data, report = load(), {}
    df = data["df"]
    pair_folds = df.groupby("pair_id", observed=True).fold.nunique()
    for name, directory in methods.items():
        directory, info = Path(directory), read_json(Path(directory) / "run.json")
        folds = {}
        for k in range(5):
            frame = pd.read_parquet(directory / f"fold_{k}.parquet")
            folds[k] = {"rows": len(frame), "covers_test_rows_once": bool(not frame.observation_id.duplicated().any()
                        and set(frame.observation_id) == set(df.observation_id[df.fold == k])),
                        "finite": bool(np.isfinite(frame.prediction).all())}
        sel = pd.read_csv(directory / "selection.csv")
        expected_pairs = [int((~np.isin(data["folds"], [k, (k + 1) % 5])).sum()) for k in sel.fold]
        noisy = {**data, "Y": data["Y"].copy(), "e": data["e"].copy()}
        rng = np.random.default_rng(0)
        held = data["folds"] == fold
        noisy["Y"][held] = rng.normal(50, 30, size=noisy["Y"][held].shape)
        noisy["e"][data["fold"] == fold] = rng.normal(0, 30, size=int((data["fold"] == fold).sum()))
        plan = read_json(info["config"])
        settings = info["settings"]
        r = pair_cell(noisy, fold, plan["cf_stage"]["variants"][settings["cf"]], settings["ridge_lambdas"], predict_test=True)
        matrix = np.full(data["Y"].shape, np.nan)
        matrix[r["te"]] = r["test"]
        rows = np.flatnonzero(data["fold"] == fold)
        refit = matrix[data["pi"][rows], data["cell"][rows]]
        if settings["dose"] != "none":
            refit = refit + dose_profile(noisy, fold, settings["dose"] == "partner", settings["dose_lambdas"], True)["test"]
        saved = pd.read_parquet(directory / f"fold_{fold}.parquet").set_index("observation_id").prediction.reindex(df.observation_id.iloc[rows])
        report[name] = {"directory": str(directory).replace("\\", "/"), "method": info["method"], "folds": folds,
                        "rows": sum(f["rows"] for f in folds.values()), "pairs_in_one_fold": bool((pair_folds == 1).all()),
                        "validation_fold_is_next": bool((sel.validation_fold == (sel.fold + 1) % 5).all()),
                        "training_pairs_exclude_test_and_validation": bool((sel.n_train_pairs == expected_pairs).all()),
                        f"fold_{fold}_refit_with_test_labels_replaced": {"max_abs_difference": float(np.abs(refit - saved.to_numpy()).max()),
                                                                          "identical": bool(np.array_equal(refit, saved.to_numpy()))}}
        report[name]["passed"] = bool(report[name]["rows"] == len(df) and all(f["covers_test_rows_once"] and f["finite"] for f in folds.values())
                                      and report[name]["pairs_in_one_fold"] and report[name]["validation_fold_is_next"]
                                      and report[name]["training_pairs_exclude_test_and_validation"]
                                      and report[name][f"fold_{fold}_refit_with_test_labels_replaced"]["identical"])
        print(name, report[name]["passed"], report[name][f"fold_{fold}_refit_with_test_labels_replaced"], flush=True)
    write_json(output, report)


PAIR_CELL_CONTRASTS = [("cf_selected", "selected:F0"), ("cf_selected", "selected:F2"), ("cf_selected", "bliss_reference"),
                       ("cf_selected", "cf_shar"), ("cf_selected", "gp_f2"), ("cf_shar", "selected:F0"), ("cf_shar", "selected:F2"),
                       ("cf_shar", "bliss_reference"), ("cf_shar", "additive")]


def compare(methods, output, repeats=1000, seed=2026):
    """Every model on her task: the mean of its dose-row predictions per pair and cell line against the pair-cell
    mean response, each pair-cell weighted equally. Pair-bootstrap and drug-jackknife intervals; networks average
    their three seeds' squared errors. A paired difference is called only when bootstrap, jackknife, every fold and
    every seed pairing agree in sign (the benchmark rule)."""
    from .sprint_report import CPU, DOSE, RUNS, model_paths, out_of_fold, team_predictions
    from scipy.stats import t as student
    df, _, _ = load_truth(DATASET)
    t = index(df)
    suite, protocol = RUNS / "sprint_v1", read_json("configs/neural_sprint.json")
    chosen, seeds, first = read_json(suite / "selection.json")["selected"], protocol["benchmark"]["seeds"], protocol["selection"]["seed"]
    dose_seeds = read_json(RUNS / DOSE / "protocol.json")["protocol"]["seeds"]
    vectors = {m: [out_of_fold(model_paths(m, s, chosen, suite), t) for s in seeds] for m in ["selected:F2", "selected:F0"]}
    vectors[DOSE] = [out_of_fold(model_paths(DOSE, s, chosen, suite), t) for s in dose_seeds]
    vectors["bliss_reference"] = [out_of_fold(model_paths("f2_residual", first, chosen, suite), t, "reference_offset")]
    vectors["additive"] = [out_of_fold(model_paths("additive", None, chosen, suite), t)]
    vectors.update({name: [team_predictions(d, t)] for name, d in methods.items()})
    frame = t["panel_frame"]
    truth, pair_of = frame.response.to_numpy(), pd.Series(range(len(t["pairs"])), index=t["pairs"])[frame.pair_id].to_numpy()
    fold_of = frame.fold.to_numpy()
    n_pair = np.bincount(pair_of).astype(float)
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(t["pairs"]), size=len(t["pairs"])), minlength=len(t["pairs"])) for _ in range(repeats)]).astype(float)
    jackknife = np.array([[d not in pair for pair in t["pair_drugs"]] for d in t["drugs"]], dtype=float)
    half = lambda x: student.ppf(.975, len(x) - 1) * np.sqrt((len(x) - 1) / len(x) * ((x - x.mean()) ** 2).sum())
    errors, rows = {}, []
    for name, vs in vectors.items():
        panel = [np.bincount(t["panel"], weights=v) / t["panel_n"] for v in vs]
        errors[name] = [np.bincount(pair_of, weights=(truth - p) ** 2) for p in panel]
        sse = np.mean(errors[name], 0)
        rmse_boot = np.sqrt((weights @ sse) / (weights @ n_pair))
        jack = np.sqrt((jackknife @ sse) / (jackknife @ n_pair))
        point = [regression_metrics_simple(truth, p) for p in panel]
        rmse = float(np.mean([p["rmse"] for p in point]))
        rows.append({"model": name, "seeds": len(vs), "pair_cells": len(truth), **{k: float(np.mean([p[k] for p in point])) for k in point[0]},
                     "rmse_ci_low": float(np.quantile(rmse_boot, .025)), "rmse_ci_high": float(np.quantile(rmse_boot, .975)),
                     "rmse_jackknife_low": rmse - half(jack), "rmse_jackknife_high": rmse + half(jack)})
    pairs = []
    for model, control in PAIR_CELL_CONTRASTS:
        if model not in errors or control not in errors:
            continue
        d = np.mean(errors[model], 0) - np.mean(errors[control], 0)
        point = d.sum() / n_pair.sum()
        boot = np.quantile(weights @ d / (weights @ n_pair), [.025, .975])
        jack = jackknife @ d / (jackknife @ n_pair)
        pair_fold = np.bincount(pair_of, weights=fold_of) / n_pair
        folds = [d[pair_fold == k].sum() / n_pair[pair_fold == k].sum() for k in range(5)]
        seedwise = [(x.sum() - y.sum()) / n_pair.sum() for x in errors[model] for y in errors[control]]
        evidence = [*boot, point - half(jack), point + half(jack), *folds, *seedwise]
        control_mse = np.mean(errors[control], 0).sum() / n_pair.sum()
        pairs.append({"model": model, "control": control, "delta_mse": point, "relative_mse_change": point / control_mse,
                      "ci_low": boot[0], "ci_high": boot[1], "jackknife_low": point - half(jack), "jackknife_high": point + half(jack),
                      "folds_lower": int(sum(f < 0 for f in folds)), "seed_pairs_lower": f"{sum(s < 0 for s in seedwise)}/{len(seedwise)}",
                      "verdict": "lower error" if max(evidence) < 0 else "higher error" if min(evidence) > 0 else "not established"})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pair_cell.csv", index=False, float_format="%.6g", lineterminator="\n")
    pd.DataFrame(pairs).to_csv(out / "pair_cell_comparisons.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(pairs).round(3).to_string())


def regression_metrics_simple(y, p):
    e = y - p
    return {"rmse": float(np.sqrt(np.mean(e * e))), "mse": float(np.mean(e * e)), "mae": float(np.mean(np.abs(e))),
            "r2": float(1 - (e * e).sum() / ((y - y.mean()) ** 2).sum()), "pearson": float(np.corrcoef(y, p)[0, 1])}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("compare")
    s.add_argument("--method", action="append", required=True, help="NAME=DIR")
    s.add_argument("--output", required=True)
    s = sub.add_parser("verify")
    s.add_argument("--method", action="append", required=True, help="NAME=DIR")
    s.add_argument("--output", required=True)
    s.add_argument("--fold", type=int, default=0)
    s = sub.add_parser("study")
    s.add_argument("--config", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("select")
    s.add_argument("--config", required=True)
    s.add_argument("--study", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("run")
    s.add_argument("--config", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--decision")
    s.add_argument("--cf")
    s.add_argument("--dose")
    a = p.parse_args()
    if a.command == "verify":
        verify(dict(m.split("=", 1) for m in a.method), a.output, a.fold)
    elif a.command == "compare":
        compare(dict(m.split("=", 1) for m in a.method), a.output)
    elif a.command == "study":
        study(a.config, a.output)
    elif a.command == "select":
        select(a.config, a.study, a.output)
    else:
        if a.decision:
            decision = read_json(a.decision)
            a.cf, a.dose = a.cf or decision["cf"], a.dose or decision["dose"]
        run(a.config, a.output, a.cf, a.dose)


if __name__ == "__main__":
    main()
