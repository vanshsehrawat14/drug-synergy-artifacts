"""Does cross-cell transfer predict an independent re-measurement?

For re-tested pairs, leave-one-cell-out B1 and B2 use the pair's batch-1 contrasts at all other cells; each
re-tested pair-cell is scored against its batch-1 target and against its batch-2 target (cell-line batch
shift removed). Covariance and B2 settings come from pairs outside the pair's fold, chosen as in variance.py.

python experiments/interaction_gap_20261110/batch_transfer.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --output results/interaction_gap_20261110
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Covariance, Data, Panel, loco_b2, sha256  # noqa: E402
from variance import CONFIGS, eligible_retest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "output"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data = Data(a.observations, a.resolution)
    P, C = data.shape
    panel = Panel(data)
    Mt, Mref = data.pc_mean_units(data.truth_u), data.pc_mean_units(data.cache["bliss"])
    V = (panel(data.split_differences(data.split_signs(50, 97, "grid"))) ** 2).mean(-1)
    flats, _ = eligible_retest(data)
    in_el = np.isin(data.flat_row, flats)
    ref_row = data.cache["bliss"][data.unit_of_row]
    w_x = data.w.copy()
    w_x.flat[flats] = 64.0
    px = Panel(data, w=w_x)
    out = {"inputs": {"observations_sha256": sha256(a.observations)}, "pair_cells": int(len(flats))}
    for name, y, M in [("bliss", data.y - ref_row, Mt - Mref), ("raw", data.y, Mt)]:
        m = [data.pc_mean_rows(y, rows=in_el & (data.row_batch == b)) for b in (0, 1)]
        shift = pd.Series(m[1].flat[flats] - m[0].flat[flats]).groupby(flats % C).transform("mean").to_numpy()
        Ms = []
        for b in (0, 1):
            Mb = M.copy()
            Mb.flat[flats] = m[b].flat[flats] - (shift if b == 1 else 0)
            Ms.append(Mb)
        T1, T2 = px(Ms[0]), px(Ms[1])
        t_frozen = panel(M)
        pids = np.unique(flats // C)
        rows = []
        for f in range(5):
            mine = pids[data.fold[pids] == f]
            if not len(mine):
                continue
            val = (f + 1) % 5
            inner = ~np.isin(data.fold, [f, val])
            cov = Covariance(t_frozen[inner], V[inner])
            vm = data.fold == val
            sse = {cfg: float((data.w[vm] * (t_frozen[vm] - loco_b2(cov.signal(*cfg), cov.N, t_frozen[vm][..., None])[..., 0]) ** 2).sum())
                   for cfg in CONFIGS}
            cfg = min(sse, key=sse.get)
            cov = Covariance(t_frozen[data.fold != f], V[data.fold != f])
            p2 = loco_b2(cov.signal(*cfg), cov.N, T1[mine][..., None])[..., 0]
            p1 = float(cov.b1_weight(C - 1)) * (T1[mine].sum(1, keepdims=True) - T1[mine]) / (C - 1)
            for i, p in enumerate(mine):
                for c in np.flatnonzero(np.isin(p * C + np.arange(C), flats)):
                    rows.append([data.pairs[p], data.cells[c], T1[p, c], T2[p, c], p1[i, c], p2[i, c]])
        df = pd.DataFrame(rows, columns=["pair", "cell", "t1", "t2", "b1", "b2"])
        res = {}
        rng = np.random.default_rng(43)
        groups = [g.index.to_numpy() for _, g in df.groupby("pair")]
        for tgt in ["t1", "t2"]:
            e0, e1, e2 = (df[tgt] ** 2), (df[tgt] - df.b1) ** 2, (df[tgt] - df.b2) ** 2
            r = {"B1": float(1 - e1.sum() / e0.sum()), "B2": float(1 - e2.sum() / e0.sum()),
                 "B2_minus_B1": float((e1.sum() - e2.sum()) / e0.sum())}
            bs = []
            for _ in range(2000):
                ix = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
                bs.append([(e1[ix].sum() - e2[ix].sum()) / e0[ix].sum(), 1 - e1[ix].sum() / e0[ix].sum(), 1 - e2[ix].sum() / e0[ix].sum()])
            bs = np.array(bs)
            r["B2_minus_B1_pair_interval"] = [float(np.quantile(bs[:, 0], q)) for q in (.025, .975)]
            r["B1_pair_interval"] = [float(np.quantile(bs[:, 1], q)) for q in (.025, .975)]
            r["B2_pair_interval"] = [float(np.quantile(bs[:, 2], q)) for q in (.025, .975)]
            res["scored_against_" + ("batch1" if tgt == "t1" else "batch2")] = r
        res["pairs"] = int(df.pair.nunique())
        out[name] = res
        print(name, json.dumps(res), flush=True)
    Path(a.output, "batch_transfer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
