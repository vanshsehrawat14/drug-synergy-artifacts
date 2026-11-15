"""Prepare audited DrugComb dose rows, block scores, grid means and, for ONEIL, published labels.

python -m cs622.data --config configs/data_oneil.json
python -m cs622.data --config configs/data_almanac.json

ONEIL folds are the published DeepSynergy pair folds. Other studies get label-free generated pair
folds (cs622.splits.generated_folds) from the seed in the config.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, read_json, write_json
from .splits import generated_folds, normalize_name, pair_id, published_folds, validate_folds

RAW_COLUMNS = ["block_id", "conc_r", "conc_c", "inhibition", "drug_row", "drug_col",
               "conc_r_unit", "conc_c_unit", "cell_line_name", "drug_row_cid",
               "drug_col_cid", "cellosaurus_accession", "study_name"]
SCORES = ["synergy_bliss", "synergy_hsa", "synergy_loewe", "synergy_zip", "S", "css"]


def canonicalize(raw, merge_variants=False):
    """merge_variants: spellings that normalize to one identifier (e.g. PIRARUBICIN/pirarubicin) are one drug."""
    df = raw.copy()
    for names in [pd.concat([df.drug_row, df.drug_col]).drop_duplicates(), df.cell_line_name.drop_duplicates()]:
        normalized = names.map(normalize_name)
        if normalized.duplicated().any() and not merge_variants:
            raise ValueError("Ambiguous normalized identifiers")
    a = df.drug_row.map(normalize_name)
    b = df.drug_col.map(normalize_name)
    swap = a > b
    df["drug_a"] = a.where(~swap, b)
    df["drug_b"] = b.where(~swap, a)
    df["cell_line"] = df.cell_line_name.map(normalize_name)
    for name, r, c in [("dose", "conc_r", "conc_c"), ("unit", "conc_r_unit", "conc_c_unit"),
                       ("cid", "drug_row_cid", "drug_col_cid")]:
        df[name + "_a"] = df[r].where(~swap, df[c])
        df[name + "_b"] = df[c].where(~swap, df[r])
    keys = df[["drug_a", "drug_b"]].drop_duplicates()
    mapping = {(a, b): pair_id(a, b) for a, b in keys.itertuples(index=False, name=None)}
    df["pair_id"] = [mapping[(a, b)] for a, b in zip(df.drug_a, df.drug_b)]
    for col in ["dose_a", "dose_b", "inhibition"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid_target"] = np.isfinite(df.inhibition)
    df["valid_dose"] = (np.isfinite(df.dose_a) & np.isfinite(df.dose_b)
                        & (df.dose_a >= 0) & (df.dose_b >= 0))
    df["is_combination"] = (df.dose_a > 0) & (df.dose_b > 0)
    df["is_monotherapy"] = (df.dose_a > 0) ^ (df.dose_b > 0)
    return df


def _compact(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col != "observation_id":
            df[col] = df[col].astype("category")
    return df


def _read_raw(path, chunksize, study="ONEIL"):
    kept, total = [], 0
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=RAW_COLUMNS, low_memory=False):
        selected = chunk.loc[chunk.study_name == study].copy()
        selected["source_row"] = selected.index.to_numpy(np.int64)
        if len(selected):
            kept.append(selected)
        total += len(chunk)
    if not kept:
        raise ValueError(f"No {study} rows in raw source")
    return pd.concat(kept, ignore_index=True), total


def prepare(config):
    out = Path(config["output"])
    if (out / "manifest.json").exists():
        manifest = read_json(out / "manifest.json")
        if manifest["settings"] != config:
            raise ValueError("Preparation settings changed; use a fresh output directory")
        for name, source in manifest["sources"].items():
            if digest(config[name]) != source["sha256"]:
                raise ValueError(f"Existing prepared dataset has changed {name} source")
        for name, expected in manifest["artifacts"].items():
            if digest(out / (name + ".parquet")) != expected:
                raise ValueError(f"Prepared artifact changed: {name}")
        print(f"Verified and reused prepared dataset {out}")
        return manifest
    study = config.get("study", "ONEIL")
    published_labels = "labels" in config
    if study == "ONEIL" and not published_labels:
        raise ValueError("ONEIL uses the published DeepSynergy folds")
    raw_path, summary_path = Path(config["raw"]), Path(config["summary"])
    labels_path = Path(config["labels"]) if published_labels else None
    sources = {}
    for name, path in [("raw", raw_path), ("summary", summary_path)] + ([("labels", labels_path)] if published_labels else []):
        sources[name] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}
        expected = config.get("expected_sha256", {}).get(name)
        if expected and sources[name]["sha256"] != expected:
            raise ValueError(f"{name} checksum mismatch")
    raw, total = _read_raw(raw_path, config.get("chunksize", 500_000), study)
    df = canonicalize(raw, config.get("merge_name_variants", False))
    names = pd.concat([raw.drug_row, raw.drug_col]).drop_duplicates()
    variants = names.groupby(names.map(normalize_name)).agg(sorted)
    variants = {k: v for k, v in variants.items() if len(v) > 1}
    del raw
    if set(df.unit_a) != {"uM"} or set(df.unit_b) != {"uM"}:
        raise ValueError(f"{study} units differ from verified uM contract")
    df["observation_id"] = f"dc14:{study}:" + df.source_row.astype(str)
    excluded = _sparse_drugs(df, config.get("min_pairs_per_drug", 0))
    if excluded["drugs"]:
        df = df[~(df.drug_a.isin(excluded["drugs"]) | df.drug_b.isin(excluded["drugs"]))].copy()
    by_cell = config.get("coverage") == "drug_cell"
    if by_cell:
        excluded["drug_cells"], df = _sparse_drug_cells(df, config["min_pairs_per_drug_cell"])
    # Generated folds use pair identifiers only; no response, score or monotherapy value enters.
    folds, fold_seed = (published_folds(labels_path), None) if published_labels else _covering_folds(df, config["fold_seed"], by_cell=by_cell)
    df = df.merge(folds, on="pair_id", how="left", validate="many_to_one", sort=False)
    counts = validate_folds(df)
    block_meta = df[["block_id", "drug_row", "drug_col", "cell_line_name", "conc_r_unit", "conc_c_unit"]].drop_duplicates()
    if block_meta.block_id.duplicated().any():
        raise ValueError("Block metadata is inconsistent")
    if df.duplicated(["block_id", "conc_r", "conc_c"]).any():
        raise ValueError("Duplicate dose coordinates in block; explicit replicate IDs required")
    summary = pd.read_csv(summary_path, low_memory=False)
    summary = summary[summary.block_id.isin(block_meta.block_id)]
    matched = block_meta.merge(summary, on=list(block_meta.columns), how="left", validate="one_to_one", indicator=True)
    if (matched._merge != "both").any():
        raise ValueError("Summary block metadata mismatch or missing block")
    for col in SCORES:
        matched[col] = pd.to_numeric(matched[col], errors="coerce")
    valid = df.valid_target & df.valid_dose
    combo = df.loc[valid & df.is_combination]
    block_counts = df.groupby("block_id", sort=True).agg(rows=("inhibition", "size"))
    combo_counts = combo.groupby("block_id").size()
    if study == "ONEIL" and (not (block_counts.rows == 25).all() or not (combo_counts == 16).all()):
        raise ValueError("Incomplete ONEIL 5x5 grid; inspect invalid/missing observations")
    if combo_counts.reindex(block_counts.index, fill_value=0).eq(0).any():
        raise ValueError("Block without valid combination rows")
    # Equal block weighting first, then equal pair-cell weighting in the grid-mean track.
    blocks = combo.groupby(["block_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold"], sort=True).agg(
        response=("inhibition", "mean"), n_doses=("inhibition", "size")).reset_index()
    blocks = blocks.merge(matched[["block_id"] + SCORES], on="block_id", validate="one_to_one")
    panel = blocks.groupby(["pair_id", "drug_a", "drug_b", "cell_line", "fold"], sort=True).agg(
        **{name: (name, "mean") for name in ["response"] + SCORES}, n_blocks=("block_id", "size")).reset_index()
    panel["observation_id"] = ["dc14:panel:" + p + ":" + c for p, c in zip(panel.pair_id, panel.cell_line)]
    tables = {"observations": _compact(df), "blocks": _compact(blocks), "panel": _compact(panel), "folds": folds}
    if published_labels:
        tables["published"] = _compact(_published(labels_path))
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / (name + ".parquet"), index=False)
    manifest = {"dataset": f"DrugComb v1.4 {study}", "sources": sources, "settings": config,
                "response": "reported inhibition in percentage points; no clipping or rescaling",
                "source_rows": total, "study_rows": len(df), "combination_rows": len(combo),
                "monotherapy_rows": int(df.is_monotherapy.sum()), "blocks": len(blocks),
                "pairs": int(df.pair_id.nunique()), "drugs": len(set(df.drug_a) | set(df.drug_b)),
                "cells": int(df.cell_line.nunique()), "invalid_target": int((~df.valid_target).sum()),
                "invalid_dose": int((~df.valid_dose).sum()), "folds": counts,
                "fold_source": "published DeepSynergy labels" if published_labels else f"generated_folds(seed={fold_seed})",
                "excluded_sparse_drugs": excluded, "merged_name_variants": variants,
                "combination_rows_per_block": {str(k): int(v) for k, v in combo_counts.value_counts().sort_index().items()},
                "inhibition_range": [float(df.inhibition.min()), float(df.inhibition.max())],
                "monotherapy_provenance": "block-associated; independent acquisition not established",
                "artifacts": {name: digest(out / (name + ".parquet")) for name in tables}}
    if published_labels:
        manifest["oneil_rows"] = manifest.pop("study_rows")
        manifest["published_duplicate_rows_averaged"] = int((tables["published"].n_source_rows - 1).sum())
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(df):,} rows, {len(blocks):,} blocks, {manifest['pairs']} pairs in {out}")
    return manifest


def _sparse_drugs(df, min_pairs):
    """Drugs in fewer than min_pairs distinct pairs cannot stay seen in training under pair folds."""
    pairs = df[["pair_id", "drug_a", "drug_b"]].drop_duplicates("pair_id")
    counts = pd.concat([pairs.drug_a, pairs.drug_b]).value_counts()
    drugs = sorted(counts.index[counts < min_pairs])
    hit = pairs.drug_a.isin(drugs) | pairs.drug_b.isin(drugs)
    return {"min_pairs_per_drug": min_pairs, "drugs": drugs, "pairs": int(hit.sum()),
            "rows": int((df.drug_a.isin(drugs) | df.drug_b.isin(drugs)).sum())}


def _sparse_drug_cells(df, min_pairs):
    """Pair-cells whose drug has fewer than min_pairs distinct pairs in that cell lack training-pair monotherapy under pair folds."""
    units = df[["pair_id", "drug_a", "drug_b", "cell_line"]].drop_duplicates().astype(str)
    counts = pd.concat([units[["drug_a", "cell_line"]].set_axis(["drug", "cell"], axis=1),
                        units[["drug_b", "cell_line"]].set_axis(["drug", "cell"], axis=1)]).value_counts()
    sparse = set(counts.index[counts < min_pairs])
    hit = np.array([(a, c) in sparse or (b, c) in sparse for a, b, c in
                    zip(df.drug_a.astype(str), df.drug_b.astype(str), df.cell_line.astype(str))])
    gone = units.merge(df.loc[hit, ["pair_id", "cell_line"]].astype(str).drop_duplicates(), on=["pair_id", "cell_line"])
    return {"min_pairs_per_drug_cell": min_pairs, "drug_cells": sorted("|".join(k) for k in sparse),
            "pair_cells": len(gone), "rows": int(hit.sum())}, df[~hit].copy()


def _covering_folds(df, seed, attempts=1000, by_cell=False):
    """First seed >= seed whose generated pair folds keep every validation/test drug (or drug-cell) seen in training.

    Only pair and cell identifiers enter; the search never looks at responses."""
    pairs = df[["pair_id", "drug_a", "drug_b"]].drop_duplicates("pair_id")
    units = df[["pair_id", "drug_a", "drug_b", "cell_line"]].drop_duplicates().astype(str) if by_cell else pairs.astype(str).assign(cell_line="")

    def keys(f):
        return set(zip(f.drug_a, f.cell_line)) | set(zip(f.drug_b, f.cell_line))
    for s in range(seed, seed + attempts):
        folds = generated_folds(pairs.pair_id, seed=s)
        f = units.merge(folds.astype({"pair_id": str}), on="pair_id")
        if all(keys(f[f.fold.isin([k, (k + 1) % 5])]) <= keys(f[~f.fold.isin([k, (k + 1) % 5])]) for k in range(5)):
            return folds, s
    raise ValueError("No generated fold assignment keeps held-out drugs seen in training")


def _published(labels_path):
    published = pd.read_csv(labels_path)
    published["drug_a"] = published.drug_a_name.map(normalize_name)
    published["drug_b"] = published.drug_b_name.map(normalize_name)
    published["pair_id"] = [pair_id(a, b) for a, b in zip(published.drug_a, published.drug_b)]
    ab = np.sort(published[["drug_a", "drug_b"]].to_numpy(), axis=1)
    published[["drug_a", "drug_b"]] = ab
    published["cell_line"] = published.cell_line.map(normalize_name)
    published = published.groupby(["pair_id", "drug_a", "drug_b", "cell_line", "fold"], sort=True).agg(
        synergy=("synergy", "mean"), n_source_rows=("synergy", "size")).reset_index()
    published["observation_id"] = ["ds:panel:" + p + ":" + c for p, c in zip(published.pair_id, published.cell_line)]
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    prepare(read_json(parser.parse_args().config))


if __name__ == "__main__":
    main()
