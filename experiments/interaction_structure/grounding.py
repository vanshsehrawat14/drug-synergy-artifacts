"""Does the transferable cell-line structure track an independent biological annotation?

Compares the cross-cell correlation of pair-centred interaction targets with cell-line expression similarity
(DeepSynergy FARMS expression, independent of combination outcomes): split-half reliability, a Mantel test
with cell-label permutation, and a leave-one-cell-out predictor whose cell similarity comes only from
expression.

python experiments/interaction_structure/grounding.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --features data/processed/deepsynergy/features.npz --output results/interaction_structure
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Covariance, Data, Panel, boot_ratio, sha256  # noqa: E402
from variance import CONFIGS, loco  # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def centred_correlation(t, v):
    """Noise-corrected cross-cell correlation of pair-centred targets."""
    g = t - t.mean(1, keepdims=True)
    C = t.shape[1]
    S = g.T @ g / len(g)
    # Pair-centring scales the noise covariance by (I - J/C); subtract its expectation.
    N = np.diag(v.mean(0))
    J = np.full((C, C), 1 / C)
    S = S - (np.eye(C) - J) @ N @ (np.eye(C) - J)
    d = np.sqrt(np.clip(np.diag(S), 1e-9, None))
    return S / np.outer(d, d)


def upper(m):
    i, j = np.triu_indices(len(m), 1)
    return m[i, j]


def mantel(a, b, perms, seed):
    rng = np.random.default_rng(seed)
    obs = spearmanr(upper(a), upper(b))[0]
    null = np.empty(perms)
    for k in range(perms):
        p = rng.permutation(len(b))
        null[k] = spearmanr(upper(a), upper(b[np.ix_(p, p)]))[0]
    return float(obs), float((1 + (null >= obs).sum()) / (perms + 1)), float(np.quantile(null, .975))


def expression_loco(data, t, v, R):
    """LOCO predictor with K = tau2 J + sigma2 ((1-g) I + g R), g chosen on the validation fold."""
    C = t.shape[1]
    pred, chosen = np.zeros_like(t), {}
    w = data.w

    def predict(cov, g, x):
        diag = np.diag(cov.Csig) - cov.tau2
        s2 = np.clip(diag, 0.01 * np.diag(cov.Csig).mean(), None)
        K = cov.tau2 + np.sqrt(np.outer(s2, s2)) * ((1 - g) * np.eye(C) + g * R)
        out = np.empty_like(x)
        for c in range(C):
            o = np.delete(np.arange(C), c)
            beta = np.linalg.solve(K[np.ix_(o, o)] + np.diag(cov.N[o]), K[o, c])
            out[:, c] = x[:, o] @ beta
        return out

    for f in range(5):
        val = (f + 1) % 5
        inner = ~np.isin(data.fold, [f, val])
        cov = Covariance(t[inner], v[inner])
        vm = data.fold == val
        sse = {g: float((w[vm] * (t[vm] - predict(cov, g, t[vm])) ** 2).sum()) for g in GAMMAS}
        g = min(sse, key=sse.get)
        chosen[f] = g
        te = data.fold == f
        pred[te] = predict(Covariance(t[~te], v[~te]), g, t[te])
    return pred, chosen


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "features", "output"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    data = Data(a.observations, a.resolution)
    panel = Panel(data)
    Mt, Mref = data.pc_mean_units(data.truth_u), data.pc_mean_units(data.cache["bliss"])
    D = data.split_differences(data.split_signs(50, 97, "grid"))
    V = (panel(D) ** 2).mean(-1)
    z = np.load(a.features)
    ids = list(z["cell_ids"])
    if sorted(ids) != data.cells:
        raise ValueError("Cell lines differ between expression features and targets")
    X = z["cell_features"][[ids.index(c) for c in data.cells]]
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    Rexpr = np.corrcoef(X)
    out = {"inputs": {"features_sha256": sha256(a.features), "observations_sha256": sha256(a.observations)},
           "expression_genes": int(X.shape[1])}
    for name, t in [("bliss", panel(Mt - Mref)), ("raw", panel(Mt))]:
        Rint = centred_correlation(t, V)
        halves = [np.isin(data.fold, [0, 2, 4]), np.isin(data.fold, [1, 3])]
        Ra, Rb = (centred_correlation(t[h], V[h]) for h in halves)
        obs, p, q975 = mantel(Rint, Rexpr, a.perms, 29)
        obs_a, p_a, _ = mantel(Ra, Rexpr, 2000, 31)
        obs_b, p_b, _ = mantel(Rb, Rexpr, 2000, 37)
        rel, p_rel, _ = mantel(Ra, Rb, 2000, 41)
        p_expr, g = expression_loco(data, t, V, Rexpr)
        p1, p2, cfg = loco(data, t, V)
        w = data.w
        T = (w * t ** 2).sum(1)
        res = {"mantel_spearman": obs, "mantel_p": p, "mantel_null_q975": q975,
               "mantel_half_a": [obs_a, p_a], "mantel_half_b": [obs_b, p_b],
               "split_half_reliability_spearman": rel, "split_half_p": p_rel,
               "offdiag_mean_correlation": float(upper(Rint).mean()), "expression_gamma": g}
        for nm, pr in [("B1", p1), ("B2", p2), ("B2_expression", p_expr)]:
            E = (w * (t - pr) ** 2).sum(1)
            res[nm] = {"loco_r2": float(1 - E.sum() / T.sum()), "pair_interval": boot_ratio(E, T, 2000, 11)}
        E1, Ee, E2 = ((w * (t - pr) ** 2).sum(1) for pr in (p1, p_expr, p2))
        res["expression_minus_B1"] = {"r2": float((E1.sum() - Ee.sum()) / T.sum()),
                                      "pair_interval": boot_ratio(np.column_stack([E1, Ee]), T, 2000, 11, lambda n, d: (n[0] - n[1]) / d)}
        res["expression_share_of_B2_gain"] = float((E1.sum() - Ee.sum()) / (E1.sum() - E2.sum()))
        out[name] = res
        print(name, json.dumps({k: res[k] for k in ["mantel_spearman", "mantel_p", "split_half_reliability_spearman", "expression_share_of_B2_gain"]}), flush=True)
    Path(a.output, "grounding.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
