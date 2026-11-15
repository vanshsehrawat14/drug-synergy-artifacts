"""Data checks, a symmetric GP kernel, and reusable evaluation functions.

The notebooks show the modeling workflow. This module keeps the longer
mathematical and data-validation routines in one importable, saveable place.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

RAW_URL = "https://zenodo.org/records/18449193/files/drugcomb_data_v1.4.csv?download=1"
RENAME = {"drug_row": "drug_a", "drug_col": "drug_b", "conc_r": "dose_a",
          "conc_c": "dose_b", "cell_line_name": "cell_line",
          "conc_r_unit": "unit_a", "conc_c_unit": "unit_b"}
REQUIRED = ["block_id", "drug_a", "drug_b", "dose_a", "dose_b", "cell_line", "inhibition"]
FEATURES = ["drug_a", "dose_a", "drug_b", "dose_b"]


def demo_data(seed=42):
    """Generate synthetic inhibition data, including single-agent axes.

    These are invented drug names and responses, not DrugComb observations.
    """
    rng = np.random.default_rng(seed)
    names = [f"DEMO_DRUG_{i:02d}" for i in range(10)]
    potency = dict(zip(names, np.exp(rng.normal(0, 0.6, len(names)))))
    doses = [0.0, 0.03, 0.1, 0.3, 1.0]
    records = []
    block = 0
    for cell, sensitivity in [("DEMO_CELL_A", 1.0), ("DEMO_CELL_B", 1.5)]:
        for a, b in itertools.combinations(names, 2):
            block += 1
            interaction = rng.normal(0, 7)
            for da, db in itertools.product(doses, repeat=2):
                ea = 95 * da / (da + potency[a] * sensitivity)
                eb = 95 * db / (db + potency[b] * sensitivity)
                response = ea + eb - ea * eb / 100
                response += interaction * (da / (da + 0.2)) * (db / (db + 0.2))
                records.append([str(block), da, db, response + rng.normal(0, 2),
                                a, b, "uM", "uM", cell, "SYNTHETIC_DEMO"])
    return pd.DataFrame(records, columns=["block_id", "conc_r", "conc_c", "inhibition",
                                          "drug_row", "drug_col", "conc_r_unit",
                                          "conc_c_unit", "cell_line_name", "study_name"])


def extract_oneil(raw_path, output_path, chunksize=100_000):
    """Stream the full local CSV; retain ONEIL only, without loading 2 GB at once."""
    output_path = Path(output_path)
    temporary = output_path.with_suffix(".partial.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    offset = 0
    try:
        header = pd.read_csv(raw_path, nrows=0)
        if "study_name" not in header:
            raise ValueError("Missing study_name. Use the raw dose-level DrugComb CSV.")
        for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype={"block_id": str}):
            # Original source position makes observation IDs stable across chunk sizes.
            chunk["source_row"] = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            keep = chunk["study_name"].astype(str).str.strip().str.upper().eq("ONEIL")
            selected = chunk.loc[keep]
            if not selected.empty:
                selected.to_csv(temporary, mode="a" if count else "w", index=False, header=count == 0)
                count += len(selected)
        if count == 0:
            raise ValueError("No ONEIL rows found. Inspect study_name in the source file.")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def prepare_data(raw):
    """Validate, convert dose units to uM, and canonicalize both drug-dose arms."""
    df = raw.rename(columns=RENAME).copy()
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw data columns: {sorted(missing)}")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Required values are missing. Inspect them; do not silently impute outcomes.")
    for col in ["drug_a", "drug_b", "cell_line", "block_id"]:
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            raise ValueError(f"Empty values in {col}")
    for col in ["dose_a", "dose_b", "inhibition"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Nonfinite values in {col}")
    factors = {"m": 1e6, "mm": 1e3, "um": 1.0, "nm": 1e-3, "pm": 1e-6}
    for arm in ["a", "b"]:
        unit = f"unit_{arm}"
        if unit not in df:
            raise ValueError(f"Missing {unit}; supply measured dose units before modeling.")
        normalized = df[unit].astype(str).str.strip().str.replace("µ", "u").str.replace("μ", "u").str.lower()
        multiplier = normalized.map(factors)
        if multiplier.isna().any():
            raise ValueError(f"Unsupported units: {df.loc[multiplier.isna(), unit].unique()}")
        df[f"dose_{arm}"] *= multiplier
        df[unit] = "uM"
    if (df[["dose_a", "dose_b"]] < 0).any().any():
        raise ValueError("Negative doses found.")
    if (df.drug_a == df.drug_b).any():
        raise ValueError("Same-drug pairs found; resolve their eligibility with the team.")
    swap = df.drug_a > df.drug_b
    df.loc[swap, ["drug_a", "drug_b"]] = df.loc[swap, ["drug_b", "drug_a"]].to_numpy()
    df.loc[swap, ["dose_a", "dose_b"]] = df.loc[swap, ["dose_b", "dose_a"]].to_numpy()
    df["pair_id"] = [json.dumps([a, b], separators=(",", ":")) for a, b in zip(df.drug_a, df.drug_b)]
    if "source_row" not in df:
        df["source_row"] = np.arange(len(df))
    df["row_id"] = df.block_id + ":" + df.source_row.astype(str)
    if df.row_id.duplicated().any():
        raise ValueError("row_id is not unique.")
    # Keep block IDs globally unambiguous without deleting genuine replicates.
    df["block_key"] = df.cell_line + ":" + df.block_id
    counts = df.groupby("block_key")[["drug_a", "drug_b"]].nunique()
    if (counts > 1).any().any():
        raise ValueError("A block contains multiple drug pairs; inspect the raw block identifiers.")
    return df


def assign_folds(df, shared_path=None, seed=42):
    pairs = df[["drug_a", "drug_b", "pair_id"]].drop_duplicates().sort_values("pair_id")
    if shared_path is not None:
        mapping = pd.read_csv(shared_path)
        if not {"drug_a", "drug_b", "fold"}.issubset(mapping):
            raise ValueError("Shared folds need drug_a, drug_b, fold (integer 0 through 4).")
        mapping["pair_id"] = [json.dumps(sorted([str(a).strip(), str(b).strip()]), separators=(",", ":"))
                              for a, b in zip(mapping.drug_a, mapping.drug_b)]
        if mapping.pair_id.duplicated().any():
            raise ValueError("Duplicate unordered pairs in shared folds.")
        if not np.isfinite(pd.to_numeric(mapping.fold)).all() or not mapping.fold.isin(range(5)).all():
            raise ValueError("Shared folds must be integer IDs 0 through 4.")
        pairs = pairs.merge(mapping[["pair_id", "fold"]], on="pair_id", how="left", validate="one_to_one")
        if pairs.fold.isna().any():
            raise ValueError("The shared fold file omits some drug pairs.")
        pairs["fold"] = pairs.fold.astype(int)
        status = "TEAM_SHARED"
    else:
        if len(pairs) < 5:
            raise ValueError("At least five pairs are needed.")
        # Assign entire pairs, rather than individual dose observations.
        rng = np.random.default_rng(seed)
        ids = np.tile(np.arange(5), int(np.ceil(len(pairs) / 5)))[:len(pairs)]
        pairs["fold"] = rng.permutation(ids)
        status = "PROVISIONAL"
    result = df.merge(pairs[["pair_id", "fold"]], on="pair_id", how="left", validate="many_to_one")
    return result, pairs, status


def assert_split(train, test):
    if set(train.pair_id) & set(test.pair_id):
        raise ValueError("Drug pairs overlap between training and evaluation.")
    training_drugs = set(train.drug_a) | set(train.drug_b)
    test_drugs = set(test.drug_a) | set(test.drug_b)
    if not test_drugs.issubset(training_drugs):
        raise ValueError(f"Held-out drugs missing from training: {sorted(test_drugs-training_drugs)}")
    if train.cell_line.nunique() != 1 or test.cell_line.nunique() != 1:
        raise ValueError("This pilot fits one cell line at a time.")
    if train.cell_line.iloc[0] != test.cell_line.iloc[0]:
        raise ValueError("Training and test cell lines differ.")


def cap_training_blocks(train, max_rows=1500, seed=42):
    """Select whole experimental blocks without using inhibition or score values.

    Drug coverage is maintained. Raise if the requested budget cannot do so.
    """
    rng = np.random.default_rng(seed)
    blocks = list(train.groupby("block_key", sort=True))
    order = rng.permutation(len(blocks))
    needed = set(train.drug_a) | set(train.drug_b)
    seen, selected = set(), []
    size = 0
    # First cover all drugs; then add complete randomly ordered blocks.
    for cover in [True, False]:
        for pos in order:
            key, block = blocks[pos]
            if key in selected:
                continue
            drugs = {block.drug_a.iloc[0], block.drug_b.iloc[0]}
            if cover and not (drugs - seen):
                continue
            if size + len(block) > max_rows:
                continue
            selected.append(key)
            size += len(block)
            seen |= drugs
    if seen != needed:
        raise ValueError("Training cap is too small to retain drug coverage. Increase MAX_TRAIN_ROWS.")
    return train.loc[train.block_key.isin(selected)].copy()


def inner_split(train, seed=42):
    # Inner validation uses only outer-training pairs.
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=seed)
    for fit_idx, val_idx in splitter.split(train, groups=train.pair_id):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        try:
            assert_split(fit, val)
        except ValueError:
            continue
        return fit.copy(), val.copy()
    raise ValueError("Cannot make an inner pair split with drug coverage; increase the pilot size.")


class ArmDoseTransformer(TransformerMixin, BaseEstimator):
    """Encode IDs for equality checks; standardize pooled log doses on training only.

    Integer drug codes are labels, never ordinal distances. Both arms use the
    same transform so exchanging arms cannot change their meaning.
    """
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=FEATURES)
        names = sorted(set(X.drug_a) | set(X.drug_b))
        self.drug_codes_ = {name: i for i, name in enumerate(names)}
        dose = X[["dose_a", "dose_b"]].to_numpy(float)
        if (dose <= 0).any():
            raise ValueError("The log-dose GP uses positive combination doses only.")
        logs = np.log10(dose)
        self.mean_ = float(logs.mean())
        self.scale_ = float(logs.std()) or 1.0
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES)
        a, b = X.drug_a.map(self.drug_codes_), X.drug_b.map(self.drug_codes_)
        if a.isna().any() or b.isna().any():
            raise ValueError("An unseen drug reached the GP; this task holds out pairs, not drugs.")
        dose = X[["dose_a", "dose_b"]].to_numpy(float)
        if (dose <= 0).any():
            raise ValueError("Positive doses required. Single-agent axes are handled separately.")
        z = (np.log10(dose) - self.mean_) / self.scale_
        return np.column_stack([a, z[:, 0], b, z[:, 1]])


class SymmetricDrugDoseKernel(Kernel):
    """An additive plus symmetrized product kernel on unordered drug-dose arms.

    Arm kernel: [rho + (1-rho)*same_drug] * dose_kernel.
    rho is a fixed shared component, NOT molecular similarity.
    Pair kernel: mean of four arm similarities + weighted symmetrized product.
    Nonnegative sums/products of these kernels are positive semidefinite.
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(0.1, 10.0),
                 interaction_weight=1.0, shared_similarity=0.25, family="rbf"):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.interaction_weight = interaction_weight
        self.shared_similarity = shared_similarity
        self.family = family

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def _arm(self, u, v):
        if not 0 <= self.shared_similarity <= 1 or self.interaction_weight < 0:
            raise ValueError("Use shared_similarity in [0,1] and nonnegative interaction_weight.")
        t = np.abs(u[:, 1, None] - v[None, :, 1]) / self.length_scale
        identity = self.shared_similarity + (1-self.shared_similarity) * (u[:, 0, None] == v[None, :, 0])
        if self.family == "rbf":
            dose = np.exp(-0.5 * t*t)
            derivative = dose * t*t
        elif self.family == "matern":
            r = np.sqrt(5) * t
            dose = (1+r+r*r/3) * np.exp(-r)
            derivative = r*r*(1+r)*np.exp(-r)/3
        else:
            raise ValueError("family must be rbf or matern (nu=2.5).")
        return identity*dose, identity*derivative

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=float)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient requires Y=None.")
        Y = X if Y is None else np.asarray(Y, dtype=float)
        a, b = X[:, :2], X[:, 2:]
        c, d = Y[:, :2], Y[:, 2:]
        ac, gac = self._arm(a, c)
        ad, gad = self._arm(a, d)
        bc, gbc = self._arm(b, c)
        bd, gbd = self._arm(b, d)
        weight = self.interaction_weight
        K = (ac+ad+bc+bd)/4 + weight*(ac*bd+ad*bc)/2
        if not eval_gradient:
            return K
        if self.hyperparameter_length_scale.fixed:
            return K, np.empty(K.shape + (0,))
        gradient = (gac+gad+gbc+gbd)/4
        gradient += weight*(gac*bd+ac*gbd+gad*bc+ad*gbc)/2
        return K, gradient[..., None]

    def diag(self, X):
        X = np.asarray(X, dtype=float)
        t = np.abs(X[:, 1]-X[:, 3])/self.length_scale
        if self.family == "rbf":
            dose = np.exp(-0.5*t*t)
        elif self.family == "matern":
            r = np.sqrt(5)*t
            dose = (1+r+r*r/3)*np.exp(-r)
        else:
            raise ValueError("Unknown dose kernel family.")
        cross = (self.shared_similarity+(1-self.shared_similarity)*(X[:, 0] == X[:, 2]))*dose
        return (1+cross)/2 + self.interaction_weight*(1+cross*cross)/2

    def is_stationary(self):
        return False

    def __repr__(self):
        return (f"SymmetricDrugDoseKernel(family={self.family!r}, "
                f"length_scale={self.length_scale:.3g}, "
                f"interaction_weight={self.interaction_weight:g}, "
                f"shared_similarity={self.shared_similarity:g})")


def make_gp(family="rbf", interaction_weight=1.0, seed=42, optimize=True):
    pair_kernel = SymmetricDrugDoseKernel(family=family, interaction_weight=interaction_weight)
    kernel = ConstantKernel(1.0, (1e-3, 100)) * pair_kernel + WhiteKernel(0.05, (1e-5, 2.0))
    # WhiteKernel models measurement noise; alpha is only a small stability term.
    regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True,
                                        optimizer="fmin_l_bfgs_b" if optimize else None,
                                        n_restarts_optimizer=0, random_state=seed)
    return Pipeline([("preprocessor", ArmDoseTransformer()), ("regressor", regressor)])


def predict_batches(model, X, batch_size=256):
    means, stds = [], []
    for start in range(0, len(X), batch_size):
        mean, std = model.predict(X.iloc[start:start+batch_size], return_std=True)
        means.append(mean)
        stds.append(std)
    return np.concatenate(means), np.concatenate(stds)


def regression_metrics(actual, predicted, std=None):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mse = mean_squared_error(actual, predicted)
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if np.std(actual)>1e-12 and np.std(predicted)>1e-12 else np.nan
    out = {"mse": float(mse), "rmse": float(np.sqrt(mse)),
           "mae": float(mean_absolute_error(actual, predicted)), "pearson_r": correlation,
           "ordinary_r2": float(r2_score(actual, predicted))}
    if std is not None:
        std = np.maximum(np.asarray(std, float), 1e-12)
        out.update(coverage_95=float(np.mean(np.abs(actual-predicted) <= 1.96*std)),
                   mean_width_95=float(np.mean(2*1.96*std)),
                   mean_nlpd=float(np.mean(-norm.logpdf(actual, loc=predicted, scale=std))))
    return out


def fit_fold(data, fold, max_rows=1500, seed=42, target="inhibition"):
    """Tune kernel family inside one outer fold; fit the capped outer-training set."""
    train = cap_training_blocks(data.loc[data.fold != fold], max_rows, seed+int(fold))
    test = data.loc[data.fold == fold].copy()
    assert_split(train, test)
    fit, val = inner_split(train, seed+int(fold))
    comparisons = []
    with threadpool_limits(limits=2):
        for family in ["rbf", "matern"]:
            candidate = make_gp(family, seed=seed)
            started = time.perf_counter()
            candidate.fit(fit[FEATURES], fit[target])
            mean, std = predict_batches(candidate, val[FEATURES])
            comparisons.append({"family": family, "validation_rmse": regression_metrics(val[target], mean)["rmse"],
                                "fit_seconds": time.perf_counter()-started})
        comparison = pd.DataFrame(comparisons).sort_values("validation_rmse")
        family = comparison.iloc[0].family
        model = make_gp(family, seed=seed)
        started = time.perf_counter()
        model.fit(train[FEATURES], train[target])
        fit_seconds = time.perf_counter()-started
        mean, std = predict_batches(model, test[FEATURES])
    predictions = test[["row_id", "block_key", "pair_id", "drug_a", "drug_b", "dose_a", "dose_b", "cell_line", "fold"]].copy()
    predictions["actual"] = test[target].to_numpy()
    predictions["predicted"] = mean
    predictions["predictive_std"] = std
    predictions["lower_95"], predictions["upper_95"] = mean-1.96*std, mean+1.96*std
    metrics = regression_metrics(test[target], mean, std)
    metrics.update(fold=int(fold), n_train=len(train), n_test=len(test), family=family, fit_seconds=fit_seconds)
    return model, train, test, comparison, predictions, metrics


def dose_level_references(data):
    """Match measured single-agent axes inside each block; average axis replicates.

    This uses single-agent responses for the evaluated pair. It is a separate
    reference-only control, not an input to the primary GP. Team protocol must
    allow those measurements for held-out pairs.
    """
    combo = data.loc[(data.dose_a>0) & (data.dose_b>0)].copy()
    mono_a = data.loc[(data.dose_a>0) & (data.dose_b==0)].groupby(["block_key", "dose_a"], as_index=False).inhibition.mean().rename(columns={"inhibition": "single_a"})
    mono_b = data.loc[(data.dose_b>0) & (data.dose_a==0)].groupby(["block_key", "dose_b"], as_index=False).inhibition.mean().rename(columns={"inhibition": "single_b"})
    combo = combo.merge(mono_a, on=["block_key", "dose_a"], how="left", validate="many_to_one")
    combo = combo.merge(mono_b, on=["block_key", "dose_b"], how="left", validate="many_to_one")
    if combo[["single_a", "single_b"]].isna().any().any():
        raise ValueError("Single-agent axes are incomplete. Supply the team reference table; do not invent curves.")
    combo["bliss_reference"] = combo.single_a + combo.single_b - combo.single_a*combo.single_b/100
    combo["hsa_reference"] = combo[["single_a", "single_b"]].max(axis=1)
    combo["bliss_rebuilt"] = combo.inhibition-combo.bliss_reference
    combo["hsa_rebuilt"] = combo.inhibition-combo.hsa_reference
    return combo


def pair_jackknife(predictions, metric="mse"):
    """Delete one drug pair at a time from fixed predictions; no model refitting.

    Normal-approximation intervals quantify variation in evaluation groups,
    conditional on the fitted procedure. They are not GP predictive intervals
    or a replacement for the team's specified jackknife definition.
    """
    pairs = predictions.pair_id.unique()
    if len(pairs)<3:
        raise ValueError("At least three evaluated drug pairs required for this summary.")
    value = regression_metrics(predictions.actual, predictions.predicted)[metric]
    estimates = []
    for pair in pairs:
        subset = predictions.loc[predictions.pair_id != pair]
        estimates.append(regression_metrics(subset.actual, subset.predicted)[metric])
    estimates = np.asarray(estimates)
    se = float(np.sqrt((len(pairs)-1)/len(pairs)*np.sum((estimates-estimates.mean())**2)))
    return {"metric": metric, "estimate": value, "jackknife_se": se,
            "lower_95": value-1.96*se, "upper_95": value+1.96*se, "n_pairs": len(pairs)}
