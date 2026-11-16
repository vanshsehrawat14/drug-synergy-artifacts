from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_maintained_paths_describe_the_work():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    paths = result.stdout.decode().strip("\0").split("\0")
    legacy = re.compile(r"(?:overnight|sprint|_20[0-9]{6}(?:[/.]|$))", re.IGNORECASE)
    assert not [path for path in paths if legacy.search(path)]
    assert (ROOT / "synergy_artifacts/runner.py").is_file()
    assert (ROOT / "gaussian_process/src/project_utils.py").is_file()


def test_paper_and_reproduction_are_visible_from_the_landing_page():
    lead = "\n".join((ROOT / "README.md").read_text(encoding="utf-8").splitlines()[:5])
    for target in ["manuscript/manuscript.pdf", "manuscript/supplement.pdf", "manuscript/REPRODUCIBILITY.md"]:
        assert f"]({target})" in lead
        assert (ROOT / target).is_file()
