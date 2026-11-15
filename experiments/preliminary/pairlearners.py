"""Identity-only symmetric pair learners (verbatim copies of the helpers first used in
r03_interaction_learners.py, moved here so later scripts can import them without
re-running R03's experiment loop)."""
import numpy as np
from common import load_oneil

_d = load_oneil()
aa, bb = _d['aa'], _d['bb']; nd = _d['H'].shape[1]; C = _d['Y'].shape[1]


def sym_kernel(Kd, P, Q, aa=aa, bb=bb):
    a1, b1, a2, b2 = aa[P][:, None], bb[P][:, None], aa[Q][None, :], bb[Q][None, :]
    return Kd[a1, a2] * Kd[b1, b2] + Kd[a1, b2] * Kd[b1, a2]


def krr(Kd, R, tr, te, lam, aa=aa, bb=bb):
    Ktt = sym_kernel(Kd, tr, tr, aa, bb); Kst = sym_kernel(Kd, te, tr, aa, bb)
    return Kst @ np.linalg.solve(Ktt + lam * np.eye(len(tr)), R[tr])


def residual_rows(R, tr, aa=aa, bb=bb, nd=nd):
    C_ = R.shape[1]
    M = np.zeros((nd, nd, C_)); W = np.zeros((nd, nd))
    for p in tr:
        M[aa[p], bb[p]] = R[p]; M[bb[p], aa[p]] = R[p]; W[aa[p], bb[p]] = W[bb[p], aa[p]] = 1
    return M, W


def drug_sim_from_resid(M, W, shrink=5.0):
    nd_ = M.shape[0]; S = np.eye(nd_)
    for i in range(nd_):
        for j in range(i + 1, nd_):
            com = (W[i] * W[j]) > 0
            com[i] = com[j] = False
            n = com.sum()
            if n < 2:
                continue
            x = M[i, com].ravel(); y = M[j, com].ravel()
            r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
            S[i, j] = S[j, i] = r * n / (n + shrink)
    return S


def knn(S, M, W, te, k, aa=aa, bb=bb):
    C_ = M.shape[2]; out = np.zeros((len(te), C_))
    for t, p in enumerate(te):
        a, b = aa[p], bb[p]; est = []
        for x, y in [(a, b), (b, a)]:
            cand = np.flatnonzero(W[x] > 0); cand = cand[cand != y]
            if len(cand) == 0:
                continue
            s = S[y, cand]; top = cand[np.argsort(-s)[:k]]; w = np.clip(S[y, top], 0, None)
            if w.sum() > 0:
                est.append((w[:, None] * M[x, top]).sum(0) / w.sum())
        out[t] = np.mean(est, 0) if est else 0
    return out
