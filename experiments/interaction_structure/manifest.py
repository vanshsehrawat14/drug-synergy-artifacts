"""Write results/interaction_structure/MANIFEST.json: SHA-256 of inputs, scripts, outputs and reports.

Text files also get an LF-normalized hash so checkouts with CRLF conversion can be compared.

python experiments/interaction_structure/manifest.py --resolution RESOLUTION_DIR --output results/interaction_structure
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from core import sha256, sha256_lf  # noqa: E402

TEXT = {".py", ".md", ".json", ".csv", ".txt"}


def entry(path):
    p = Path(path)
    out = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    if p.suffix in TEXT:
        out["sha256_lf"] = sha256_lf(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import matplotlib
    import numpy
    import pandas
    import scipy
    res, out = Path(a.resolution), Path(a.output).resolve()
    inputs = {"observations": ROOT / "data/processed/oneil_v1/observations.parquet", "cache": res / "cache.npz",
              "coordinates": res / "coordinates.parquet", "cache_inputs": res / "inputs.json",
              "pair_cell_split": ROOT / "results/pair_information/oracle/pair_cell_split.csv",
              "pairseen_audit": ROOT / "manuscript/audit/outputs/pairseen_audit.json",
              "expression_features": ROOT / "data/processed/deepsynergy/features.npz"}
    manifest = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "inputs": {k: entry(v) for k, v in inputs.items()},
        "scripts": {str(p.relative_to(ROOT)).replace("\\", "/"): entry(p) for p in sorted(HERE.glob("*.py")) + [ROOT / "tests/test_interaction_gap.py"]},
        "design": entry(HERE / "design.json"),
        "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): entry(p) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "MANIFEST.json"},
        "reports": {str(p.relative_to(ROOT)).replace("\\", "/"): entry(p)
                    for p in sorted((ROOT / "reports/interaction_structure").glob("*")) if p.is_file()},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest["outputs"]), "outputs,", len(manifest["scripts"]), "scripts")


if __name__ == "__main__":
    main()
