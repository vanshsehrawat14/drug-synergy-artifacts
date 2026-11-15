"""Third-screen validation (configs/third_screen_confirmation.json): benchmark, comparisons, Bliss bins and H1-H3.

python -m cs622.third_screen --protocol configs/third_screen_validation.json --output results/third_screen/ncats_es
python -m cs622.third_screen --protocol configs/third_screen_dipg_validation.json --output results/third_screen/ncats_dipg
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .almanac_report import analyze
from .io import read_json, write_json
from .regimes import regimes
from .runner import load_experiment
from .sprint_report import index, out_of_fold
from .stress import gain
from .suite import run_dir


def hypotheses(output, protocol_path):
    output, protocol = Path(output), read_json(protocol_path)
    runs, seeds = protocol["runs"], protocol["benchmark"]["seeds"]
    df, _, _ = load_experiment({"dataset": protocol["shared_overrides"]["dataset"], "table": "observations", "target": "inhibition",
                                "test_fold": 0, "model": {"kind": "dose"}})
    t = index(df)
    y = t["y"]
    reference = out_of_fold({k: Path(runs["baselines"]["reference_bliss"]) / f"fold_{k}" for k in range(5)}, t)
    own = {m: [out_of_fold({k: run_dir(runs["suite"], m, s, k) for k in range(5)}, t) for s in seeds] for m in ["f2_combined", "mono_only"]}
    _, level = regimes(df, reference)
    high = level == "R>=75"
    tests = {"H1": gain(t, high, y, own["f2_combined"], [reference]),
             "H2": gain(t, np.ones(len(y), bool), y, own["f2_combined"], [reference]),
             "H3": gain(t, high, y, own["f2_combined"], own["mono_only"])}
    passed, out = True, {}
    for name in ["H1", "H2", "H3"]:
        r = tests[name]
        ok = bool(r["drug_jackknife"][0] > 0 and min(r["seeds"]) > 0)
        passed = passed and ok
        out[name] = {**r, "rule_met": ok, "confirmed_in_sequence": passed}
    write_json(output / "hypotheses.json", {"declared": "configs/third_screen_confirmation.json", **out})
    print(json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.output, a.protocol)
    hypotheses(a.output, a.protocol)


if __name__ == "__main__":
    main()
