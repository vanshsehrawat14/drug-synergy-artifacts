"""Original-row target adapter, original-feature extraction, paper-protocol reproduction and
the dose-aware adaptation on the shared dose-response benchmark.

Never deserialize downloaded pickle assets. The published X.p.gz is read as raw
float64 bytes after its opcode stream is checked against an exact whitelist.

python -m cs622.deepsynergy dose --protocol configs/deepsynergy_dose.json --output results/neural/runs/deepsynergy_dose_pilot --pilot
python -m cs622.deepsynergy dose --protocol configs/deepsynergy_dose.json --output results/neural/runs/deepsynergy_dose
"""
import argparse
import gzip
import io
import json
import pickletools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, read_json, write_json
from .splits import normalize_name, pair_id, validate_folds

LABEL_SHA = "4ef40678fff51faa84fa4de63344c5b51e2e4abe7c28a417af24ec2b50fa135a"
ORIGINAL_URL = "http://www.bioinf.jku.at/software/DeepSynergy/X.p.gz"
ROWS, COLUMNS, DRUG_FEATURES, CELL_FEATURES = 46104, 12758, 4387, 3984
ANY = object()
# Exact protocol-4 opcode stream preceding the payload: numpy _reconstruct of one C-order <f8 array.
HEADER = [("PROTO", 4), ("FRAME", ANY), ("SHORT_BINUNICODE", "numpy.core.multiarray"), ("MEMOIZE", None),
          ("SHORT_BINUNICODE", "_reconstruct"), ("MEMOIZE", None), ("STACK_GLOBAL", None), ("MEMOIZE", None),
          ("SHORT_BINUNICODE", "numpy"), ("MEMOIZE", None), ("SHORT_BINUNICODE", "ndarray"), ("MEMOIZE", None),
          ("STACK_GLOBAL", None), ("MEMOIZE", None), ("BININT1", 0), ("TUPLE1", None), ("MEMOIZE", None),
          ("SHORT_BINBYTES", b"b"), ("MEMOIZE", None), ("TUPLE3", None), ("MEMOIZE", None), ("REDUCE", None),
          ("MEMOIZE", None), ("MARK", None), ("BININT1", 1), ("BININT2", ROWS), ("BININT2", COLUMNS), ("TUPLE2", None),
          ("MEMOIZE", None), ("SHORT_BINUNICODE", "numpy"), ("MEMOIZE", None), ("SHORT_BINUNICODE", "dtype"),
          ("MEMOIZE", None), ("STACK_GLOBAL", None), ("MEMOIZE", None), ("SHORT_BINUNICODE", "f8"), ("MEMOIZE", None),
          ("BININT1", 0), ("BININT1", 1), ("TUPLE3", None), ("MEMOIZE", None), ("REDUCE", None), ("MEMOIZE", None),
          ("MARK", None), ("BININT1", 3), ("SHORT_BINUNICODE", "<"), ("MEMOIZE", None), ("NONE", None),
          ("NONE", None), ("NONE", None), ("BININT", -1), ("BININT", -1), ("BININT1", 0), ("TUPLE", None),
          ("MEMOIZE", None), ("BUILD", None), ("NEWFALSE", None)]
HEADER_BYTES, PAYLOAD_BYTES = 152, ROWS * COLUMNS * 8


def _opcodes(data):
    found = []
    try:
        for op, arg, _ in pickletools.genops(io.BytesIO(data)):
            found.append((op.name, arg))
    except ValueError:  # deliberately truncated before the payload, or after STOP
        pass
    return found


def validate_original_header(header):
    found = _opcodes(header[:HEADER_BYTES])
    if len(found) != len(HEADER) or any(name != n or (arg is not ANY and value != arg)
                                         for (name, arg), (n, value) in zip(HEADER, found)):
        raise ValueError("Unexpected pickle structure; refusing to read the asset")
    if header[HEADER_BYTES] != 0x8e or int.from_bytes(header[HEADER_BYTES + 1:HEADER_BYTES + 9], "little") != PAYLOAD_BYTES:
        raise ValueError("Unexpected array payload declaration")


def extract_original(archive, labels, output, chunk_rows=256):
    """Per-drug and per-cell descriptor vectors from the published row matrix, with alignment checks.

    Rows 0..23051 are labels.csv order as drug A/drug B/cell; the next 23,052 rows swap A and B.
    Every repeated drug/cell block must be bitwise identical to its first occurrence.
    """
    archive, output = Path(archive), Path(output)
    if digest(labels) != LABEL_SHA:
        raise ValueError("Published label checksum differs from verified release")
    if (output / "features.npz").exists():
        raise FileExistsError(output / "features.npz")
    table = pd.read_csv(labels)
    names = [table[c].map(normalize_name).to_numpy() for c in ["drug_a_name", "drug_b_name", "cell_line"]]
    drugs, cells, mismatches, nonfinite = {}, {}, 0, 0
    started = time.perf_counter()
    with gzip.open(archive, "rb") as stream:
        header = stream.read(HEADER_BYTES + 9)
        validate_original_header(header)
        row = 0
        while row < ROWS:
            count = min(chunk_rows, ROWS - row)
            block = np.frombuffer(stream.read(count * COLUMNS * 8), dtype="<f8")
            if block.size != count * COLUMNS:
                raise ValueError("Truncated feature payload")
            block = block.reshape(count, COLUMNS)
            nonfinite += int((~np.isfinite(block)).sum())
            for offset, values in enumerate(block):
                j, swapped = (row + offset) % (ROWS // 2), row + offset >= ROWS // 2
                a, b, c = names[0][j], names[1][j], names[2][j]
                if swapped:
                    a, b = b, a
                for store, key, vector in [(drugs, a, values[:DRUG_FEATURES]), (drugs, b, values[DRUG_FEATURES:2 * DRUG_FEATURES]),
                                           (cells, c, values[2 * DRUG_FEATURES:])]:
                    if key not in store:
                        store[key] = vector.copy()
                    elif not np.array_equal(store[key], vector, equal_nan=True):
                        mismatches += 1
            row += count
        trailer = stream.read()
    ops = [name for name, _ in _opcodes(trailer)]
    if not ops or ops[-1] != "STOP" or set(ops) - {"MEMOIZE", "TUPLE", "BUILD", "STOP"} or len(trailer) > 16:
        raise ValueError("Unexpected content after the feature payload")
    if mismatches:
        raise ValueError(f"{mismatches} repeated drug/cell feature blocks differ; row alignment failed")
    output.mkdir(parents=True, exist_ok=True)
    drug_ids, cell_ids = sorted(drugs), sorted(cells)
    path = output / "features.npz"
    np.savez_compressed(path, drug_ids=np.array(drug_ids), drug_features=np.array([drugs[d] for d in drug_ids]),
                        cell_ids=np.array(cell_ids), cell_features=np.array([cells[c] for c in cell_ids]))
    headers = archive.with_name(archive.name + ".headers.txt")
    provenance = {
        "source": ORIGINAL_URL, "method": "DeepSynergy original descriptors and FARMS expression",
        "independent_of_combination_outcomes": True, "sha256": digest(path),
        "upstream": {"bytes": archive.stat().st_size, "sha256": digest(archive),
                     "http_headers": headers.read_text().splitlines() if headers.exists() else None,
                     "authentication": "none published; size/ETag/Last-Modified recorded, content checked for internal consistency"},
        "extraction": {"reader": "opcode whitelist then raw little-endian float64 rows; pickle.load never called",
                       "rows": ROWS, "columns": COLUMNS, "labels_sha256": LABEL_SHA,
                       "repeated_block_mismatches": 0, "nonfinite_values": nonfinite, "seconds": time.perf_counter() - started},
        "layout": {"drug_features": DRUG_FEATURES, "cell_features": CELL_FEATURES,
                   "paper_drug_blocks": {"ECFP6_counts": 1309, "physicochemical": 802, "toxicophores": 2276},
                   "cell": "FARMS-summarized E-MTAB-3610 expression after informative-call filtering",
                   "drug_id_normalization": "uppercase alphanumeric (cs622.splits.normalize_name)"},
        "drugs": len(drug_ids), "cells": len(cell_ids)}
    write_json(output / "features.provenance.json", provenance)
    print(json.dumps({k: provenance[k] for k in ["drugs", "cells", "sha256"]} | {"nonfinite": nonfinite}))
    return provenance


def prepare_published_rows(labels, output):
    if digest(labels) != LABEL_SHA:
        raise ValueError("Published label checksum differs from verified release")
    output = Path(output)
    if output.exists():
        manifest = read_json(output.with_suffix(".manifest.json"))
        if manifest["source_sha256"] != digest(labels) or manifest["output_sha256"] != digest(output):
            raise ValueError("Existing original-row labels changed")
        print(f"Verified and reused {output}")
        return
    source = pd.read_csv(labels)
    source["observation_id"] = [f"ds:source:{i}" for i in range(len(source))]
    a, b = source.drug_a_name.map(normalize_name), source.drug_b_name.map(normalize_name)
    source["drug_a"], source["drug_b"] = np.minimum(a, b), np.maximum(a, b)
    source["pair_id"] = [pair_id(x, y) for x, y in zip(a, b)]
    source["cell_line"] = source.cell_line.map(normalize_name)
    validate_folds(source)
    source.to_parquet(output, index=False)
    write_json(output.with_suffix(".manifest.json"), {"source_sha256": LABEL_SHA, "rows": len(source),
        "duplicate_pair_cell_rows_preserved": int(source.duplicated(["pair_id", "cell_line"]).sum()),
        "output_sha256": digest(output), "target": "published Combenefit Loewe, not DrugComb Loewe"})
    print(f"Preserved {len(source)} published source rows in {output}")


def verify_features(path, provenance):
    from .features import load_independent_features
    features = load_independent_features(path, provenance)
    metadata = read_json(provenance)
    if metadata.get("method") != "DeepSynergy original descriptors and FARMS expression":
        raise ValueError("Unverified reproduction feature construction")
    if len(features["drug"]) != 38 or len(features["cell"]) != 39:
        raise ValueError("Expected original 38 drugs and 39 cells")
    if {len(v) for v in features["drug"].values()} != {4387} or {len(v) for v in features["cell"].values()} != {3984}:
        raise ValueError("Expected 4387 chemical and 3984 expression features")
    print("Feature dimensions and declared provenance passed; this is not an independent reproduction audit")


PAPER = {"source": "Preuer et al. 2018, Table 2: mean over five test folds", "mse": 255.49, "mse_ci95": [239.93, 271.06],
         "rmse_mean": 15.91, "rmse_sd": 1.56, "pearson_mean": .73, "pearson_sd": .04}


def _weighted_stats(values, weights):
    w = weights / weights.sum()
    mean = w @ values
    std = np.sqrt(w @ (values - mean) ** 2)
    std[np.ptp(values[weights > 0], axis=0) == 0] = 0.
    return mean, std


def normalize_entities(table, weights, norm):
    """normalize.ipynb statistics over doubled training rows, computed from entity counts.

    Each training row repeats one entity vector per slot, so row-wise column statistics equal
    count-weighted entity statistics. Zero-variance columns are removed (feat_filt).
    """
    mean, std = _weighted_stats(table, weights)
    keep = std != 0
    z = (table[:, keep] - mean[keep]) / std[keep]
    if norm == "norm":
        return z, keep
    z = np.tanh(z)
    if norm == "tanh":
        return z, keep
    if norm != "tanh_norm":
        raise ValueError("Unknown normalization")
    mean, std = _weighted_stats(z, weights)
    return np.where(std == 0, 0., (z - mean) / np.where(std == 0, 1., std)), keep


def original_network(input_dim, layers=(8192, 4096), input_dropout=.2, dropout=.5, placement="paper"):
    """Paper: dropout on inputs. Notebook: input_dropout after the first hidden layer, width 8182."""
    import torch
    from torch import nn
    modules, width = ([nn.Dropout(input_dropout)] if placement == "paper" else []), input_dim
    for i, units in enumerate(layers):
        modules += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(input_dropout if placement == "notebook" and i == 0 else dropout)]
        width = units
    network = nn.Sequential(*modules, nn.Linear(width, 1))
    for module in network:
        if isinstance(module, nn.Linear):
            # Keras 2.0/2.1 he_normal: normal(0, sqrt(2/fan_in)) truncated at two standard deviations.
            sd = float(np.sqrt(2 / module.in_features))
            nn.init.trunc_normal_(module.weight, std=sd, a=-2 * sd, b=2 * sd)
            nn.init.zeros_(module.bias)
    return network


def original_rows(target, labels, panel):
    if target == "published":
        t = pd.read_csv(labels)
        frame = pd.DataFrame({"a": t.drug_a_name.map(normalize_name), "b": t.drug_b_name.map(normalize_name),
                              "cell": t.cell_line.map(normalize_name), "y": t.synergy, "fold": t.fold})
    else:
        t = pd.read_parquet(panel)
        frame = pd.DataFrame({"a": t.drug_a.astype(str), "b": t.drug_b.astype(str), "cell": t.cell_line.astype(str),
                              "y": t[target], "fold": t.fold})
    if not np.isfinite(frame.y.to_numpy(float)).all():
        raise ValueError("Nonfinite reproduction target")
    return frame.reset_index(drop=True)


class _Stage:
    """One SGD fit on doubled AB/BA rows, gathered from normalized entity tables on the device."""

    def __init__(self, tables, rows, settings, device, seed, path, doses=None):
        import torch
        from .training import seed_all
        seed_all(seed, settings.get("threads", 4))
        self.torch, self.device, self.settings, self.path = torch, device, settings, Path(path)
        self.drug, self.cell = (torch.as_tensor(t, dtype=torch.float32, device=device) for t in tables)
        # Dose adaptation: one standardized log10 dose per drug slot, (doubled rows, 2), on the device.
        self.rows, self.doses = rows, doses
        width = (self.drug.shape[1] + (doses is not None)) * 2 + self.cell.shape[1]
        self.model = original_network(width, settings["layers"], settings["input_dropout"],
                                      settings["dropout"], settings["placement"]).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=settings["lr"], momentum=settings["momentum"])
        self.generator = torch.Generator().manual_seed(seed)
        self.history, self.epoch = [], 0
        if self.path.exists():
            state = torch.load(self.path, map_location=device, weights_only=False)
            if state["settings"] != settings:
                raise ValueError("Checkpoint settings differ")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            # map_location moved the saved RNG states to the device; generators accept CPU byte tensors only.
            self.generator.set_state(state["generator"].cpu())
            torch.set_rng_state(state["rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            self.history, self.epoch = state["history"], state["epoch"]

    def _inputs(self, index):
        ia, ib, ic = self.rows
        if self.doses is None:
            return self.torch.cat([self.drug[ia[index]], self.drug[ib[index]], self.cell[ic[index]]], dim=1)
        dose = self.doses[index]
        return self.torch.cat([self.drug[ia[index]], dose[:, :1], self.drug[ib[index]], dose[:, 1:], self.cell[ic[index]]], dim=1)

    def predict(self, index, batch=4096):
        self.model.eval()
        with self.torch.no_grad():
            out = [self.model(self._inputs(index[i:i + batch])).squeeze(1) for i in range(0, len(index), batch)]
        return self.torch.cat(out)

    def save(self):
        torch = self.torch
        temporary = self.path.with_suffix(".tmp")
        torch.save({"settings": self.settings, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "generator": self.generator.get_state(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                    "history": self.history, "epoch": self.epoch}, temporary)
        temporary.replace(self.path)

    def stale(self):
        """Epochs since the best (first minimum) validation loss."""
        losses = [r["validation_mse"] for r in self.history]
        return len(losses) - 1 - int(np.argmin(losses)) if losses else 0

    def fit(self, train, y, epochs, validation=None, patience=None, best=None):
        """train: global doubled-row indices; y: targets for every doubled row on the device.

        validation is (index, target) or a function of this stage returning validation MSE. With
        patience, training stops after that many epochs without a strict improvement, and best
        (a path) receives the weights of each new best epoch.
        """
        torch = self.torch
        train = torch.as_tensor(train, device=self.device)
        while self.epoch < epochs:
            if patience is not None and self.stale() >= patience:
                break
            started = time.perf_counter()
            self.model.train()
            total = torch.zeros((), device=self.device)
            order = train[torch.randperm(len(train), generator=self.generator).to(self.device)]
            for index in order.split(self.settings["batch_size"]):
                loss = (self.model(self._inputs(index)).squeeze(1) - y[index]).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total += loss.detach() * len(index)
            record = {"epoch": self.epoch, "train_mse": float(total) / len(train)}
            if callable(validation):
                record["validation_mse"] = float(validation(self))
            elif validation is not None:
                index, target = validation
                record["validation_mse"] = float((self.predict(index) - target).square().mean())
            if not all(np.isfinite(v) for v in record.values()):
                raise ValueError("Nonfinite DeepSynergy loss")
            record["seconds"] = time.perf_counter() - started
            self.history.append(record)
            self.epoch += 1
            if best is not None and self.stale() == 0:
                temporary = Path(best).with_suffix(".tmp")
                torch.save({"model": self.model.state_dict(), "epoch": record["epoch"],
                            "validation_mse": record["validation_mse"]}, temporary)
                temporary.replace(best)
            if self.epoch % self.settings["checkpoint_every"] == 0 or self.epoch == epochs:
                self.save()
            if self.epoch == 1 or self.epoch % self.settings.get("print_every", 25) == 0:
                print(f"{self.path.stem} epoch={self.epoch} {json.dumps({k: round(v, 4) for k, v in record.items()})}", flush=True)
        return self


def smoothed_epoch(losses, window):
    """cv_example.ipynb: moving average, edge padding by int(window/2), argmin index."""
    losses = np.asarray(losses, dtype=float)
    window = min(window, len(losses))
    if window <= 1:
        return int(np.argmin(losses))
    moving = np.cumsum(losses)
    moving[window:] = moving[window:] - moving[:-window]
    moving = moving[window - 1:] / window
    return int(np.argmin(np.pad(moving, int(window / 2), mode="edge")))


def reproduce(features, provenance, output, folds=range(5), target="published", device="cuda", seed=2026,
              settings=None, labels="data/raw/deepsynergy/labels.csv", panel="data/processed/oneil_v1/panel.parquet"):
    import torch
    from .features import load_independent_features
    from .training import device_report, resolve_device
    settings = {"layers": [8192, 4096], "input_dropout": .2, "dropout": .5, "placement": "paper", "norm": "tanh_norm",
                "lr": 1e-5, "momentum": .5, "batch_size": 64, "epochs": 1000, "window": 25, "checkpoint_every": 50,
                "threads": 4, "allow_tf32": False, "keep_checkpoints": False, **(settings or {})}
    device = resolve_device(device)
    external = load_independent_features(features, provenance)
    drugs, cells = sorted(external["drug"]), sorted(external["cell"])
    drug_table = np.array([external["drug"][d] for d in drugs])
    cell_table = np.array([external["cell"][c] for c in cells])
    frame = original_rows(target, labels, panel)
    di, ci = {d: i for i, d in enumerate(drugs)}, {c: i for i, c in enumerate(cells)}
    a, b, c = frame.a.map(di).to_numpy(), frame.b.map(di).to_numpy(), frame.cell.map(ci).to_numpy()
    if np.isnan(np.concatenate([a, b, c]).astype(float)).any():
        raise ValueError("Rows reference drugs or cells absent from the original features")
    n = len(frame)
    # Rows 0..n-1 are A/B/cell, rows n..2n-1 the same samples as B/A/cell, as in X.p.gz.
    ia, ib, ic = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([c, c])
    y, fold = np.tile(frame.y.to_numpy(float), 2), np.tile(frame.fold.to_numpy(), 2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if settings["allow_tf32"]:
        torch.set_float32_matmul_precision("high")
    summary = []
    for k in folds:
        directory = output / f"fold_{k}"
        directory.mkdir(exist_ok=True)
        if (directory / "metrics.json").exists():
            summary.append(read_json(directory / "metrics.json"))
            continue
        validation_fold = (k + 1) % 5
        tr = np.flatnonzero((fold != k) & (fold != validation_fold))
        va, final, te = np.flatnonzero(fold == validation_fold), np.flatnonzero(fold != k), np.flatnonzero(fold == k)

        def tables(index):
            d, _ = normalize_entities(drug_table, np.bincount(ia[index], minlength=len(drugs)).astype(float), settings["norm"])
            cc, _ = normalize_entities(cell_table, np.bincount(ic[index], minlength=len(cells)).astype(float), settings["norm"])
            return d, cc
        rows = tuple(torch.as_tensor(x, device=device) for x in [ia, ib, ic])
        torch_y = torch.as_tensor(y, dtype=torch.float32, device=device)
        if (directory / "selection.json").exists():
            chosen, first_history = read_json(directory / "selection.json"), read_json(directory / "stage1_history.json")
        else:
            first = _Stage(tables(tr), rows, settings, device, seed, directory / "stage1.pt")
            first.fit(tr, torch_y, settings["epochs"], (torch.as_tensor(va, device=device), torch_y[torch.as_tensor(va, device=device)]))
            first_history = first.history
            losses = [r["validation_mse"] for r in first_history]
            chosen = {"window": settings["window"], "epoch_index": smoothed_epoch(losses, settings["window"]),
                      "notebook_window_15": smoothed_epoch(losses, 15), "raw_minimum": int(np.argmin(losses))}
            # The notebook passes the argmin index as Keras epochs; one epoch minimum avoids an untrained model.
            chosen["final_epochs"] = max(chosen["epoch_index"], 1)
            write_json(directory / "stage1_history.json", first_history)
            write_json(directory / "selection.json", chosen)
            del first
            if not settings.get("keep_checkpoints"):
                (directory / "stage1.pt").unlink(missing_ok=True)
        second = _Stage(tables(final), rows, settings, device, seed, directory / "stage2.pt")
        second.fit(final, torch_y, chosen["final_epochs"])
        prediction = second.predict(torch.as_tensor(te, device=device)).cpu().numpy().astype(float)
        half = len(te) // 2
        ab, ba = prediction[:half], prediction[half:]
        if not np.array_equal(te[:half] + n, te[half:]):
            raise ValueError("Test orderings misaligned")
        truth = y[te[:half]]
        symmetric = (ab + ba) / 2
        rows_out = frame.iloc[te[:half]].assign(prediction_ab=ab, prediction_ba=ba, prediction=symmetric)
        rows_out.to_parquet(directory / "predictions.parquet", index=False)
        metrics = {"fold": k, "validation_fold": validation_fold, "target": target, "test_samples": half,
                   "selected": chosen, "symmetrized": {"mse": float(np.mean((truth - symmetric) ** 2)),
                                                       "pearson": float(np.corrcoef(truth, symmetric)[0, 1])},
                   "doubled_rows_mse": float(np.mean((np.concatenate([truth, truth]) - prediction) ** 2)),
                   "stage_seconds": [sum(r["seconds"] for r in h) for h in [first_history, second.history]]}
        metrics["symmetrized"]["rmse"] = float(np.sqrt(metrics["symmetrized"]["mse"]))
        write_json(directory / "history.json", {"stage1": first_history, "stage2": second.history})
        write_json(directory / "metrics.json", metrics)
        if not settings.get("keep_checkpoints"):
            (directory / "stage2.pt").unlink(missing_ok=True)
        del second
        summary.append(metrics)
        print(json.dumps({"fold": k, **metrics["symmetrized"], "final_epochs": chosen["final_epochs"]}), flush=True)
    report = {"target": target, "settings": settings, "seed": seed, "device": device_report(device), "folds": summary,
              "features_sha256": digest(features), "paper": PAPER if target == "published" else None}
    if len(summary) == 5:
        mse = [m["symmetrized"]["mse"] for m in summary]
        report["mean_over_folds"] = {"mse": float(np.mean(mse)), "rmse_mean": float(np.mean(np.sqrt(mse))),
                                     "rmse_sd": float(np.std(np.sqrt(mse), ddof=1)),
                                     "pearson_mean": float(np.mean([m["symmetrized"]["pearson"] for m in summary])),
                                     "pearson_sd": float(np.std([m["symmetrized"]["pearson"] for m in summary], ddof=1))}
    write_json(output / "reproduction.json", report)
    return report



def protocol_sha(protocol):
    import hashlib
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def dose_run(protocol, output, fold, seed, device="cuda", pilot=False):
    """Dose-aware DeepSynergy on the shared benchmark rows: one test fold, one seed.

    Inputs per ordered row: drug A descriptors, its standardized log10 dose, drug B descriptors,
    its dose, cell expression. Rows, observation IDs, folds, partitions and metrics are those of
    the F0/F2 runner. Training rows enter in both drug orders; the epoch with the lowest
    validation MSE of order-averaged predictions is evaluated, as in the runner. A pilot never
    predicts test rows.
    """
    import importlib.metadata
    import platform
    import torch
    from .features import load_independent_features
    from .metrics import evaluate
    from .runner import IDENTIFIERS, git_state, load_experiment
    from .splits import partitions, validate_folds
    from .training import device_report, resolve_device
    started, output = time.perf_counter(), Path(output)
    done = output / ("pilot.json" if pilot else "metadata.json")
    if done.exists():
        return read_json(done)
    settings = {**protocol["settings"], **(protocol["pilot"]["settings"] if pilot else {})}
    device = resolve_device(device)
    config = {"dataset": protocol["dataset"], "table": "observations", "target": "inhibition", "test_fold": fold,
              "seed": seed, "model": {"kind": "deepsynergy_dose"}}
    df, _, path = load_experiment(config)
    tr, va, te = partitions(df, fold)
    external = load_independent_features(protocol["features"], protocol["provenance"])
    drugs, cells = sorted(external["drug"]), sorted(external["cell"])
    di, ci = {d: i for i, d in enumerate(drugs)}, {c: i for i, c in enumerate(cells)}
    a, b = df.drug_a.astype(str).map(di).to_numpy(), df.drug_b.astype(str).map(di).to_numpy()
    c = df.cell_line.astype(str).map(ci).to_numpy()
    if np.isnan(np.concatenate([a, b, c]).astype(float)).any():
        raise ValueError("Rows reference drugs or cells absent from the original features")
    a, b, c = a.astype(np.int64), b.astype(np.int64), c.astype(np.int64)
    # Train-only transforms: entity statistics weighted by training-row slots, dose and target means.
    drug_weights = (np.bincount(a[tr], minlength=len(drugs)) + np.bincount(b[tr], minlength=len(drugs))).astype(float)
    drug_table, drug_keep = normalize_entities(np.array([external["drug"][d] for d in drugs]), drug_weights, settings["norm"])
    cell_table, cell_keep = normalize_entities(np.array([external["cell"][x] for x in cells]),
                                               np.bincount(c[tr], minlength=len(cells)).astype(float), settings["norm"])
    log_a, log_b = np.log10(df.dose_a.to_numpy(float)), np.log10(df.dose_b.to_numpy(float))
    if not (np.isfinite(log_a).all() and np.isfinite(log_b).all()):
        raise ValueError("Combination rows require positive doses")
    pooled = np.concatenate([log_a[tr], log_b[tr]])
    dose_mean, dose_sd = float(pooled.mean()), float(pooled.std())
    za, zb = (log_a - dose_mean) / dose_sd, (log_b - dose_mean) / dose_sd
    y = df.inhibition.to_numpy(float)
    target_mean = float(y[tr].mean()) if settings["center_target"] else 0.
    n = len(df)

    def tensor(x, dtype=torch.int64):
        return torch.as_tensor(x, dtype=dtype, device=device)
    # Rows 0..n-1 are A/B/cell, rows n..2n-1 the same observations as B/A/cell.
    rows = tuple(tensor(np.concatenate(x)) for x in [(a, b), (b, a), (c, c)])
    doses = tensor(np.column_stack([np.concatenate([za, zb]), np.concatenate([zb, za])]), torch.float32)
    target = tensor(np.tile(y - target_mean, 2), torch.float32)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    stage = _Stage((drug_table, cell_table), rows, settings, device, seed, output / "checkpoints" / "latest.pt", doses)
    batch = settings.get("eval_batch_size", 8192)

    def both_orders(index):
        index = tensor(index)
        return stage.predict(index, batch).double(), stage.predict(index + n, batch).double()

    truth_va = tensor(y[va] - target_mean, torch.float64)

    def validation(_):
        ab, ba = both_orders(va)
        return float(((ab + ba) / 2 - truth_va).square().mean())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_path = output / "checkpoints" / "best.pt"
    stage.fit(np.concatenate([tr, tr + n]), target, settings["epochs"], validation, settings["patience"], best_path)
    losses = [r["validation_mse"] for r in stage.history]
    fit = {"best_epoch": int(np.argmin(losses)), "validation_mse": float(min(losses)), "epochs_completed": len(losses),
           "stopped": "patience" if stage.stale() >= settings["patience"] else "budget",
           "parameters": sum(p.numel() for p in stage.model.parameters()),
           "fit_seconds": float(sum(r["seconds"] for r in stage.history)), "device": str(device)}
    if device.type == "cuda":
        fit["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    inputs = {"drug_features_kept": int(drug_keep.sum()), "cell_features_kept": int(cell_keep.sum()),
              "input_dim": int((drug_keep.sum() + 1) * 2 + cell_keep.sum()), "dose_log10_mean": dose_mean,
              "dose_log10_sd": dose_sd, "target_mean": target_mean}
    write_json(output / "history.json", stage.history)
    record = {"protocol": protocol.get("name"), "protocol_sha256": protocol_sha(protocol), "fold": fold,
              "validation_fold": (fold + 1) % 5, "seed": seed, "pilot": pilot, "settings": settings, "git": git_state(),
              "dataset_sha256": digest(path), "features_sha256": digest(protocol["features"]), "fit": fit, "inputs": inputs,
              "partition_rows": dict(zip(["train", "validation", "test"], map(len, [tr, va, te]))),
              "device": device_report(device), "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
              "split_rule": "published test f, validation (f+1)%5, other folds train"}
    if pilot:
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(done, record)
        print(json.dumps({k: record[k] for k in ["fold", "seed", "fit"]}), flush=True)
        return record
    best = torch.load(best_path, map_location=device, weights_only=True)
    if best["epoch"] != fit["best_epoch"] or best["validation_mse"] != fit["validation_mse"]:
        raise ValueError("Best checkpoint differs from the validation history")
    stage.model.load_state_dict(best["model"])
    if not np.isclose(validation(stage), fit["validation_mse"], rtol=1e-6, atol=0):
        raise ValueError("Reloaded best checkpoint does not reproduce its validation MSE")
    # Test outcomes are read only here, after training and epoch selection.
    ab, ba = (x.cpu().numpy() + target_mean for x in both_orders(te))
    frame = df.iloc[te][IDENTIFIERS + ["block_id", "dose_a", "dose_b", "unit_a", "unit_b"]].copy()
    frame["experiment_id"], frame["dataset"], frame["study"] = f"deepsynergy_dose/seed_{seed}/fold_{fold}", "DrugComb-v1.4", "ONEIL"
    frame["target_name"], frame["feature_regime"] = "inhibition", "F1"
    frame["prediction"], frame["prediction_ab"], frame["prediction_ba"] = (ab + ba) / 2, ab, ba
    if not np.isfinite(frame[["prediction_ab", "prediction_ba"]].to_numpy()).all():
        raise ValueError("Nonfinite DeepSynergy prediction")
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = evaluate(frame.assign(inhibition=y[te]), "inhibition", protocol["bootstrap"], seed)
    write_json(output / "evaluation" / "metrics.json", metrics)
    record.update({"python": platform.python_version(), "fold_counts": validate_folds(df),
                   "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "torch", "pyarrow"]},
                   "checkpoint_sha256": digest(best_path), "elapsed_seconds": time.perf_counter() - started})
    if not settings.get("keep_checkpoints"):
        (output / "checkpoints" / "latest.pt").unlink(missing_ok=True)
    write_json(done, record)
    print(json.dumps({"fold": fold, "seed": seed, "pooled": metrics["pooled"], "fit": fit}), flush=True)
    return record


def dose_suite(protocol_path, output, seeds=None, folds=None, device="cuda", pilot=False):
    """Every (seed, fold) of the frozen dose protocol, or its validation-only pilot; resumable."""
    from .runner import git_state
    protocol, root = read_json(protocol_path), Path(output)
    manifest = root / "protocol.json"
    if manifest.exists():
        if read_json(manifest)["protocol_sha256"] != protocol_sha(protocol):
            raise ValueError("Output belongs to a different protocol")
    else:
        write_json(manifest, {"protocol": protocol, "protocol_sha256": protocol_sha(protocol), "pilot": pilot,
                              "git": git_state(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if pilot:
        runs = [(protocol["pilot"]["seed"], protocol["pilot"]["fold"])]
    else:
        runs = [(s, k) for s in (seeds or protocol["seeds"]) for k in (folds or protocol["folds"])]
    return [dose_run(protocol, root / f"seed_{s}" / f"fold_{k}", k, s, device, pilot) for s, k in runs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-labels")
    prep.add_argument("--labels", default="data/raw/deepsynergy/labels.csv")
    prep.add_argument("--output", default="data/processed/oneil_v1/published_rows.parquet")
    check = sub.add_parser("verify-features")
    check.add_argument("--features", required=True)
    check.add_argument("--provenance", required=True)
    extract = sub.add_parser("extract-original")
    extract.add_argument("--archive", default="data/raw/deepsynergy_original/X.p.gz")
    extract.add_argument("--labels", default="data/raw/deepsynergy/labels.csv")
    extract.add_argument("--output", default="data/processed/deepsynergy")
    rep = sub.add_parser("reproduce", help="paper protocol on GPU: 1000 epochs, smoothed epoch choice, refit")
    rep.add_argument("--features", default="data/processed/deepsynergy/features.npz")
    rep.add_argument("--provenance", default="data/processed/deepsynergy/features.provenance.json")
    rep.add_argument("--output", required=True)
    rep.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    rep.add_argument("--target", default="published", choices=["published", "response", "synergy_bliss", "synergy_hsa", "synergy_loewe"])
    rep.add_argument("--device", default="cuda")
    rep.add_argument("--placement", choices=["paper", "notebook"], default="paper")
    rep.add_argument("--allow-tf32", action="store_true", help="faster on Ampere+, a recorded numerical deviation")
    rep.add_argument("--smoke", action="store_true", help="tiny network and epochs; engineering check only")
    dose = sub.add_parser("dose", help="dose-aware adaptation on the shared dose-response rows")
    dose.add_argument("--protocol", default="configs/deepsynergy_dose.json")
    dose.add_argument("--output", required=True)
    dose.add_argument("--seeds", nargs="+", type=int)
    dose.add_argument("--folds", nargs="+", type=int)
    dose.add_argument("--device", default="cuda")
    dose.add_argument("--pilot", action="store_true", help="validation-only pilot declared in the protocol")
    args = p.parse_args()
    if args.command == "dose":
        dose_suite(args.protocol, args.output, args.seeds, args.folds, args.device, args.pilot)
        return
    if args.command == "reproduce":
        settings = {"placement": args.placement, "allow_tf32": args.allow_tf32}
        if args.placement == "notebook":
            settings["layers"] = [8182, 4096]
        if args.smoke:
            settings.update(layers=[16, 8], epochs=3, window=1, checkpoint_every=1, batch_size=256)
        print(json.dumps(reproduce(args.features, args.provenance, args.output, args.folds, args.target, args.device,
                                   settings=settings).get("mean_over_folds")))
        return
    if args.command == "prepare-labels":
        prepare_published_rows(args.labels, args.output)
    elif args.command == "extract-original":
        extract_original(args.archive, args.labels, args.output)
    else:
        verify_features(args.features, args.provenance)


if __name__ == "__main__":
    main()
