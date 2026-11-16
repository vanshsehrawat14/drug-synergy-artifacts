"""Build manuscript.pdf and supplement.pdf from their LaTeX sources.

python manuscript/scripts/build.py              # bibliographies from references.json, pdfLaTeX three times
python manuscript/scripts/build.py supplement   # one document

Citations (\\citep{key}) are numbered; verified entries from manuscript/references.json appear in citation
order. Requires pdfLaTeX with Latin Modern, microtype, titlesec, natbib, booktabs, tabularx, caption, enumitem,
xurl and hyperref.
Prints a JSON report per document and exits non-zero on missing glyphs, undefined references, overflow or
oversized floats.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r"\\cite[pt]?\{([^}]+)\}")
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "$": r"\$"}


def esc(s):
    return "".join(SPECIAL.get(ch, ch) for ch in s)


def split_name(a):
    """'Haines BB' -> ('Haines', 'B.~B.'); corporate authors have no initials."""
    *last, ini = a.split(" ")
    if not last or not re.fullmatch(r"[^\W\d_a-z]{1,4}|[^\W\d_a-z]-[^\W\d_a-z]", ini):
        return a, ""
    return " ".join(last), "~".join(c + "." if c != "-" else "-" for c in ini).replace("~-~", "-")


def join(names):
    return "" if not names else names[0] if len(names) == 1 else " and ".join(names) if len(names) == 2 else \
        ", ".join(names[:-1]) + ", and " + names[-1]


def short_label(r):
    """Author name for textual citations: Bliss; Park and Marcotte; O'Neil et al."""
    last = [split_name(a)[0] for a in r["authors"]]
    if not last:
        return r["title"].split(" ")[0]
    return last[0] if len(last) == 1 else " and ".join(last) if len(last) == 2 else last[0] + " et~al."


def fmt_ref(r, year):
    """One entry in the plainnat layout: authors. Title. Journal, vol(issue):pages, year. doi."""
    end = lambda t: t if t.endswith((".", "?", "!")) else t + "."
    names = [(f"{ini}~{last}" if ini else last) for last, ini in map(split_name, r["authors"])]
    parts = [end(esc(join(names)).replace(r"\textasciitilde{}", "~")) if names else "",
             end(esc(r["title"].rstrip(".")))]
    cont, vol, pages = r.get("container") or "", r.get("volume"), (r.get("pages") or "").replace("-", "–")
    if cont == "arXiv":
        src = f"arXiv preprint arXiv:{pages}, {year}."
    elif cont.startswith("Proceedings of the"):
        src = r"In \emph{" + esc(cont) + "}" + (f", pages {pages}" if pages else "") + f", {year}."
    else:
        loc = (f", {vol}" + (f"({r['issue']})" if r.get("issue") else "") + (f":{pages}" if pages else "")
               if vol else (f", {pages}" if pages else ""))
        src = r"\emph{" + esc(cont) + "}" + esc(loc) + f", {year}."
    parts.append(src)
    if r.get("doi"):
        parts.append(r"doi: \href{https://doi.org/" + r["doi"] + "}{" + esc(r["doi"]) + "}.")
    elif r.get("url"):
        parts.append(r"URL \url{" + r["url"] + "}.")
    return " ".join(p for p in parts if p)


def bibliography(name, refs):
    tex = (ROOT / f"{name}.tex").read_text(encoding="utf-8")
    body = "\n".join(ln.split("%")[0] if not ln.lstrip().startswith("%") else "" for ln in tex.splitlines())
    cited = []
    for m in CITE.finditer(body):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k not in refs:
                raise SystemExit(f"{name}: unknown citation key {k}")
            if k not in cited:
                cited.append(k)
    order = cited
    year = {k: str(refs[k]["year"]) for k in order}
    groups = {}
    for k in order:
        groups.setdefault((short_label(refs[k]), year[k]), []).append(k)
    for ks in groups.values():
        if len(ks) > 1:
            for i, k in enumerate(ks):
                year[k] += "abcdefgh"[i]
    lines = [r"\begin{thebibliography}{" + str(len(order)) + "}"]
    for k in order:
        r = refs[k]
        long = join([split_name(a)[0] for a in r["authors"]]) or short_label(r)
        lines.append(r"\bibitem[" + esc(short_label(r)).replace(r"\textasciitilde{}", "~") + f"({year[k]})"
                     + esc(long) + "]{" + k + "}\n" + fmt_ref(r, year[k]))
    lines.append(r"\end{thebibliography}")
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    (out / f"{name}.bib.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return order


def build(name, refs):
    order = bibliography(name, refs)
    for _ in range(3):
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build",
                            f"{name}.tex"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0:
            print(p.stdout[-3000:])
            raise SystemExit(f"pdflatex failed for {name}")
    log = (ROOT / "build" / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    report = {"citations": len(order), "unused_references": len(set(refs) - set(order)),
              "pages": int(re.search(r"Output written on .*?\((\d+) page", log, re.S).group(1)),
              "missing_characters": len(re.findall(r"Missing character", log)),
              "overfull_hbox": len(re.findall(r"Overfull \\hbox", log)),
              "overfull_vbox": len(re.findall(r"Overfull \\vbox", log)),
              "underfull_hbox": len(re.findall(r"Underfull \\hbox", log)),
              "undefined": len(re.findall(r"(Reference|Citation) .* undefined", log)),
              "float_too_large": len(re.findall(r"Float too large", log))}
    shutil.copy(ROOT / "build" / f"{name}.pdf", ROOT / f"{name}.pdf")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["manuscript", "supplement"])
    a = ap.parse_args()
    refs = json.loads((ROOT / "references.json").read_text(encoding="utf-8"))
    bad = False
    for name in a.names:
        r = build(name, refs)
        print(name, json.dumps(r))
        bad |= any(r[key] > 0 for key in ["undefined", "missing_characters", "overfull_hbox", "overfull_vbox", "float_too_large"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
