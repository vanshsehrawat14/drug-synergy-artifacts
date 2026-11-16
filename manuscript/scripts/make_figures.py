"""Generate the manuscript and supplement figures and their provenance manifest.

python manuscript/scripts/make_figures.py \
    --observations data/processed/oneil_v1/observations.parquet \
    --targets GAP_DERIVED/pair_cell_targets.parquet --cross-batch GAP_DERIVED/cross_batch_values.csv \
    --surfaces NATIVE/surface_cases.parquet --native-panel NATIVE/panel.parquet \
    --derived AUDIT_DERIVED --resolution RES --gate2 GATE2_RERUN/results/preliminary \
    --labels data/raw/deepsynergy/labels.csv [--only fig1 fig7]

GAP_DERIVED is the --derived directory of experiments/interaction_structure/variance.py; NATIVE is
results/neural/runs/scientific_native_full of a research checkout; AUDIT_DERIVED holds the per-case tables
written by manuscript/audit/loewe_audit.py and screening_audit.py; RES is the frozen resolution cache;
GATE2_RERUN is a re-execution of experiments/preliminary r16-r21. Outputs go to manuscript/figures/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402,F401
import fig_measure  # noqa: E402
import fig_interaction  # noqa: E402
import fig_loewe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "figures"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    """Named input paths; records which inputs each figure reads."""

    def __init__(self, a):
        res, g2, der = Path(a.resolution), Path(a.gate2), Path(a.derived)
        gap = ROOT / "results" / "interaction_structure"
        aud = ROOT / "manuscript" / "audit" / "outputs"
        self.paths = {
            "observations": Path(a.observations), "targets": Path(a.targets), "cross_batch": Path(a.cross_batch),
            "surfaces": Path(a.surfaces), "native_panel": Path(a.native_panel), "labels": Path(a.labels),
            "variance_shares": res / "variance.json", "error_decomposition": res / "error_decomposition.csv",
            "resolution_inputs": res / "inputs.json", "resolution_cache": res / "cache.npz",
            "rejected_roots": der / "rejected_roots.csv", "boundary_layer": der / "boundary_layer_cases.csv",
            "pair_cell_scores": der / "pair_cell_scores.csv",
            "gate2": g2 / "r18_gate2.json", "gate2_units": g2 / "r18_units.npz",
            "interaction_audit": aud / "interaction_audit.json", "pairseen_audit": aud / "pairseen_audit.json",
            "loewe_audit": aud / "loewe_audit.json", "screening_audit": aud / "screening_audit.json",
            "variance_components": gap / "variance_components.json", "anchor_experiment": gap / "anchor_experiment.json",
            "b3_paired": gap / "b3_paired_summary.json", "batch_transfer": gap / "batch_transfer.json",
            "compare": ROOT / "results" / "compare" / "selected_models.csv",
        }
        self.used = set()
        self.cache = {}

    def __getitem__(self, k):
        self.used.add(k)
        return self.paths[k]

    def json(self, k):
        if k not in self.cache:
            self.cache[k] = json.loads(self[k].read_text(encoding="utf-8"))
        self.used.add(k)
        return self.cache[k]


FIGURES = {
    "fig1_dose_response": fig_measure.fig1, "fig2_design": fig_measure.fig2,
    "fig3_response_interaction": fig_interaction.fig3, "fig4_decomposition": fig_interaction.fig4,
    "fig5_same_pair": fig_interaction.fig5, "fig6_reference_labels": fig_loewe.fig6,
    "fig7_loewe_mechanism": fig_loewe.fig7, "fig8_screening": fig_loewe.fig8,
    "figS1_replicates": fig_measure.figS1, "figS2_anchor_robustness": fig_interaction.figS2,
    "figS3_cross_batch": fig_interaction.figS3, "figS4_loewe_populations": fig_loewe.figS4,
    "figS5_fallback_errors": fig_loewe.figS5,
}


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "targets", "cross-batch", "surfaces", "native-panel", "derived", "resolution", "gate2",
              "labels"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"figures": {}}
    here = Path(__file__).resolve().parent
    scripts = {p.name: sha(p) for p in sorted(set(here.glob("fig*.py")) | {here / "make_figures.py"})}
    hashes = {}
    for name, fn in FIGURES.items():
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        inp = Inputs(a)
        notes = fn(inp, OUT / name) or {}
        for k in inp.used:
            if k not in hashes:
                hashes[k] = sha(inp.paths[k])
        manifest["figures"][name] = {
            "outputs": {ext: sha(OUT / f"{name}.{ext}") for ext in ("pdf", "png")},
            "inputs": {k: {"file": inp.paths[k].name, "sha256": hashes[k]} for k in sorted(inp.used)},
            **({"notes": notes} if notes else {}),
        }
        print(name, "written")
    manifest["scripts"] = scripts
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
