"""Shared paths, data loading, metrics and learners for the preliminary experiments.

Reads the DeepSynergy O'Neil labels (data/raw/deepsynergy/labels.csv, SHA-256 checked).
"""
from pathlib import Path
import copy, hashlib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/raw/deepsynergy/labels.csv'
EXPECTED_SHA = '4ef40678fff51faa84fa4de63344c5b51e2e4abe7c28a417af24ec2b50fa135a'
RAW = ROOT / 'data/processed/drugcomb_oneil'
RES = ROOT / 'results/preliminary'
RES.mkdir(parents=True, exist_ok=True)


def load_oneil(dup_rule='mean', transform=None):
    """Return Y (pairs x cells), H (pairs x drugs), folds, pair list, drugs, cells.

    dup_rule: 'mean' (default), 'first', 'second' for the 315 repeated units.
    transform: None | 'winsor1' (1/99% clip, train-agnostic global quantiles) | 'rank'
    (global rank-to-normal; label-only monotone map, documented as evaluation-scale change).
    """
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == EXPECTED_SHA
    df = pd.read_csv(SRC)
    df['a'] = df[['drug_a_name', 'drug_b_name']].min(axis=1)
    df['b'] = df[['drug_a_name', 'drug_b_name']].max(axis=1)
    if dup_rule == 'mean':
        agg = 'mean'
    elif dup_rule == 'first':
        agg = 'first'
    elif dup_rule == 'second':
        agg = 'last'
    g = df.groupby(['a', 'b', 'cell_line'], sort=True).agg(
        y=('synergy', agg), fold=('fold', 'first'), nf=('fold', 'nunique')).reset_index()
    assert g.nf.max() == 1
    drugs = sorted(set(g.a) | set(g.b)); D = {x: i for i, x in enumerate(drugs)}
    wide = g.pivot(index=['a', 'b'], columns='cell_line', values='y')
    assert not wide.isna().any().any()
    Y = wide.to_numpy().astype(float)
    if transform == 'winsor1':
        lo, hi = np.quantile(Y, [.01, .99]); Y = np.clip(Y, lo, hi)
    elif transform == 'rank':
        from scipy.stats import rankdata, norm
        r = rankdata(Y.ravel()) / (Y.size + 1); Y = norm.ppf(r).reshape(Y.shape) * Y.std()
    pairs = list(wide.index)
    aa = np.array([D[a] for a, b in pairs]); bb = np.array([D[b] for a, b in pairs])
    H = np.eye(len(D))[aa] + np.eye(len(D))[bb]
    folds = g.groupby(['a', 'b']).fold.first().reindex(wide.index).to_numpy()
    return dict(Y=Y, H=H, aa=aa, bb=bb, folds=folds, pairs=pairs, drugs=drugs,
                cells=list(wide.columns))


def proj_resid(Z, h):
    """Residual of Z after least-squares fit on design h (test-design projector P_T)."""
    return Z - h @ np.linalg.lstsq(h, Z, rcond=None)[0]


def metrics(pred, truth, h):
    pr = proj_resid(pred, h); yr = proj_resid(truth, h)
    return dict(mse=float(np.mean((pred - truth) ** 2)),
                pearson=float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
                contrast_sse=float(np.sum((pr - yr) ** 2)), contrast_ss=float(np.sum(yr ** 2)),
                contrast_r2=float(1 - np.sum((pr - yr) ** 2) / np.sum(yr ** 2)))


def ridge_additive(H, Y, tr, va, lams=(.01, .1, 1, 10, 100)):
    """Drug-in-cell additive ridge: beta (drugs x cells), lambda chosen on validation."""
    best = None
    for lam in lams:
        beta = np.linalg.solve(H[tr].T @ H[tr] + lam * np.eye(H.shape[1]), H[tr].T @ Y[tr])
        v = np.mean((H[va] @ beta - Y[va]) ** 2)
        if best is None or v < best[0]:
            best = (v, lam, beta)
    return best[1], best[2]


class Learner(torch.nn.Module):
    """Linear skip + factor or 1-hidden-layer MLP)."""
    def __init__(self, kind, nd, C, width=128, rank=8):
        super().__init__(); self.kind = kind
        self.linear = torch.nn.Linear(nd, C)
        if kind == 'factor':
            self.emb = torch.nn.Embedding(nd, rank)
            torch.nn.init.normal_(self.emb.weight, std=.15)
            self.out = torch.nn.Linear(rank, C, bias=False)
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(nd, width), torch.nn.ReLU(), torch.nn.Linear(width, C))

    def forward(self, x, a, b):
        return self.linear(x) + (self.out(self.emb(a) * self.emb(b)) if self.kind == 'factor' else self.net(x))


def fit_torch(kind, H, aa, bb, target, tr, va, te, seed, lr=.01, wd=.001, width=128, rank=8,
              max_epoch=1201, check=20, patience=10, val_target=None):
    """Adam training loop with validation early stopping. val_target lets validation use different labels than training."""
    torch.manual_seed(seed)
    X = torch.tensor(H, dtype=torch.float32); A = torch.tensor(aa); B = torch.tensor(bb)
    model = Learner(kind, H.shape[1], target.shape[1], width=width, rank=rank)
    mu = target[tr].mean(0); scale = max(float(target[tr].std()), 1)
    y = torch.tensor((target - mu) / scale, dtype=torch.float32)
    yv = y if val_target is None else torch.tensor((val_target - mu) / scale, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = np.inf; state = None; since = 0; best_epoch = 0
    for epoch in range(max_epoch):
        opt.zero_grad(); loss = ((model(X[tr], A[tr], B[tr]) - y[tr]) ** 2).mean(); loss.backward(); opt.step()
        if epoch % check == 0:
            with torch.no_grad():
                vl = float(((model(X[va], A[va], B[va]) - yv[va]) ** 2).mean())
            if vl < best - 1e-6:
                best = vl; state = copy.deepcopy(model.state_dict()); since = 0; best_epoch = epoch
            else:
                since += 1
            if since >= patience:
                break
    model.load_state_dict(state)
    with torch.no_grad():
        pred = model(X[te], A[te], B[te]).numpy() * scale + mu
    return pred, dict(epoch=best_epoch, validation_mse=best * scale ** 2)


def split(folds, f, val_shift=1):
    te = np.flatnonzero(folds == f); vf = (f + val_shift) % 5
    va = np.flatnonzero(folds == vf); tr = np.flatnonzero((folds != f) & (folds != vf))
    return tr, va, te
