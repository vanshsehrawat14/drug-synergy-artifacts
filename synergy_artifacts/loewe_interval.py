"""Interval-arithmetic certificates for the fixed-fit Loewe equation (exploratory, 2026-11-07 response_analysis).

For co-monotone LL.4/L.4 arms with float coefficients taken as exact binary rationals,
g(y) = sum_i x_i / D_i(y) - 1 is strictly monotone on the open joint effect interval (l, u),
so a root is unique when it exists. One end of (l, u) has an arm term tending to +inf; the
other end has a finite limit g_end. Hence a root exists iff g_end < 0.

certified_root     : [y0, y1] strictly inside (l, u), mpmath interval enclosures of g(y0), g(y1)
                     with strictly opposite signs in the monotone direction.
certified_no_root  : interval enclosure of g_end strictly positive.
unresolved         : neither (typically g_end enclosure contains 0, a root at the asymptote).
undefined_*        : opposite/flat curves or empty joint interval; no Loewe expectation by definition.
invalid_curve      : unsupported model, non-finite coefficients, b == 0, c == d, LL.4 with e <= 0, dose <= 0.

The equation matches SynergyFinder 2.0.x `eq.LL4.LL4`/`eq.L4.L4`/mixed forms: LL.4 on dose,
L.4 on natural-log dose with the arm dose entering undivided.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from mpmath import iv, mp

COLS = ["case_id", "dose_a", "dose_b", "model_a", "model_b"] + [f"{a}_{k}" for a in "ab" for k in "bcde"]


def _arms(row):
    arms = []
    for a in "ab":
        model = row[f"model_{a}"]
        p = [row[f"{a}_{k}"] for k in "bcde"]
        dose = row[f"dose_{a}"]
        if model not in ("LL.4", "L.4") or not np.all(np.isfinite(p + [dose])) or dose <= 0:
            return None
        b, c, d, e = p
        if b == 0 or c == d or (model == "LL.4" and e <= 0):
            return None
        arms.append(dict(model=model, b=b, c=c, d=d, e=e, dose=dose,
                         lo=min(c, d), hi=max(c, d), direction=np.sign(-b * (d - c))))
    return arms


class Eq:
    """Equation in exact (mp) and interval (iv) arithmetic at a fixed working precision."""

    def __init__(self, arms, dps):
        self.arms, self.dps = arms, dps
        iv.dps = dps
        mp.dps = dps
        self.iv_arm = [dict(b=iv.mpf(a["b"]), c=iv.mpf(a["c"]), d=iv.mpf(a["d"]),
                            centre=iv.log(iv.mpf(a["e"])) if a["model"] == "LL.4" else iv.mpf(a["e"]),
                            ld=iv.log(iv.mpf(a["dose"]))) for a in arms]
        self.mp_arm = [dict(b=mp.mpf(a["b"]), c=mp.mpf(a["c"]), d=mp.mpf(a["d"]),
                            centre=mp.log(mp.mpf(a["e"])) if a["model"] == "LL.4" else mp.mpf(a["e"]),
                            ld=mp.log(mp.mpf(a["dose"]))) for a in arms]

    def term_iv(self, k, y):
        a = self.iv_arm[k]
        ratio = (a["d"] - y) / (y - a["c"])
        if not ratio.a > 0:
            return None  # outside this arm's open range: not certifiable
        return iv.exp(a["ld"] - a["centre"] - iv.log(ratio) / a["b"])

    def g_iv(self, y):
        y = iv.mpf(y)
        total = iv.mpf(-1)
        for k in range(len(self.arms)):
            t = self.term_iv(k, y)
            if t is None:
                return None
            total = total + t
        return total

    def g_mp(self, y):
        s = mp.mpf(-1)
        for a in self.mp_arm:
            if y == a["c"]:
                return mp.nan
            ratio = (a["d"] - y) / (y - a["c"])
            if ratio <= 0:
                return mp.nan
            s += mp.exp(a["ld"] - a["centre"] - mp.log(ratio) / a["b"])
        return s


def _end_limit(eq, end_value, which):
    """Enclosure of lim g at an end of (l, u); returns ('inf', None) or ('finite', interval)."""
    total = iv.mpf(-1)
    for k, a in enumerate(eq.arms):
        if end_value == (a["lo"] if which == "l" else a["hi"]):
            # y -> asymptote of this arm: ratio (d-y)/(y-c) -> 0 if end==d else +inf
            ratio_to_zero = end_value == a["d"]
            log_ratio_sign = -1 if ratio_to_zero else 1
            # D = exp(centre + log(ratio)/b) -> 0 iff log(ratio)/b -> -inf
            d_to_zero = log_ratio_sign * np.sign(a["b"]) < 0
            if d_to_zero:
                return "inf", None
            continue  # term -> 0
        t = eq.term_iv(k, iv.mpf(end_value))
        if t is None:
            return "error", None
        total = total + t
    return "finite", total


def certify(row, hint=None, dps=50):
    arms = _arms(row)
    out = dict(case_id=row["case_id"])
    if arms is None:
        return {**out, "cert_status": "invalid_curve"}
    if arms[0]["direction"] != arms[1]["direction"] or arms[0]["direction"] == 0:
        return {**out, "cert_status": "undefined_opposite_or_flat"}
    lower = max(a["lo"] for a in arms)
    upper = min(a["hi"] for a in arms)
    out.update(direction=int(arms[0]["direction"]), joint_lower=lower, joint_upper=upper)
    if not lower < upper:
        return {**out, "cert_status": "undefined_empty_interval"}
    for work in (dps, 2 * dps, 4 * dps):
        eq = Eq(arms, work)
        kl, gl = _end_limit(eq, lower, "l")
        ku, gu = _end_limit(eq, upper, "u")
        if "error" in (kl, ku) or (kl == "inf") == (ku == "inf"):
            return {**out, "cert_status": "unresolved", "note": f"end structure {kl}/{ku}"}
        finite = gu if kl == "inf" else gl
        out.update(end_lo=float(mp.mpf(finite.a)), end_hi=float(mp.mpf(finite.b)), dps=work)
        if finite.a > 0:
            return {**out, "cert_status": "certified_no_root"}
        bracket = _bracket(eq, lower, upper, kl == "inf", hint)
        if bracket is not None:
            y0, y1, g0, g1, source = bracket
            mid = (y0 + y1) / 2
            return {**out, "cert_status": "certified_root", "root": float(mid),
                    # Preserve strict-interior brackets even when the root is far closer
                    # to an asymptote than 40 significant decimal digits can represent.
                    "bracket_lo": mp.nstr(y0, work + 10), "bracket_hi": mp.nstr(y1, work + 10),
                    "width": float(y1 - y0), "g_lo_enclosure": f"[{mp.nstr(mp.mpf(g0.a), 6)}, {mp.nstr(mp.mpf(g0.b), 6)}]",
                    "g_hi_enclosure": f"[{mp.nstr(mp.mpf(g1.a), 6)}, {mp.nstr(mp.mpf(g1.b), 6)}]",
                    "dist_lower": float(y0 - mp.mpf(lower)), "dist_upper": float(mp.mpf(upper) - y1),
                    "hint_source": source}
        if finite.b < 0:
            continue  # existence proved but no bracket at this precision; retry higher
    return {**out, "cert_status": "unresolved"}


def _bracket(eq, lower, upper, inf_at_lower, hint):
    """Find [y0, y1] inside (l, u) with certified sign change; g positive on the infinite side."""
    l, u = mp.mpf(lower), mp.mpf(upper)

    def ok(y0, y1):
        if not (l < y0 < y1 < u):
            return None
        g0, g1 = eq.g_iv(y0), eq.g_iv(y1)
        if g0 is None or g1 is None:
            return None
        pos, neg = (g0, g1) if inf_at_lower else (g1, g0)
        if pos.a > 0 and neg.b < 0:
            return g0, g1
        return None

    if hint is not None and np.isfinite(hint) and lower < hint < upper:
        y = mp.mpf(hint)
        scale = max(abs(y), mp.mpf(1))
        for rel in (mp.mpf(2) ** -44, mp.mpf(2) ** -36, mp.mpf(2) ** -26, mp.mpf(2) ** -16):
            delta = rel * scale
            y0 = max(y - delta, l + (y - l) / 2)
            y1 = min(y + delta, u - (u - y) / 2)
            r = ok(y0, y1)
            if r:
                return y0, y1, r[0], r[1], "hint"
    # bisection in a logit coordinate of (l, u), exact mp arithmetic
    def y_of(t):
        return l + (u - l) / (1 + mp.exp(-t))
    span = mp.mpf(0.8) * eq.dps * mp.log(10)  # keeps y_of(+-span) representable at this precision
    lo, hi = -span, span
    sign_at = lambda t: eq.g_mp(y_of(t))
    glo, ghi = sign_at(lo), sign_at(hi)
    want_lo_pos = inf_at_lower
    if mp.isnan(glo) or mp.isnan(ghi) or ((glo > 0) != want_lo_pos) or ((ghi > 0) == want_lo_pos):
        return None
    for _ in range(eq.dps * 4):
        mid = (lo + hi) / 2
        gm = sign_at(mid)
        if mp.isnan(gm):
            return None
        if (gm > 0) == want_lo_pos:
            lo = mid
        else:
            hi = mid
        if y_of(hi) - y_of(lo) < mp.mpf(10) ** (-(eq.dps // 2)) * (u - l):
            break
    y0, y1 = y_of(lo), y_of(hi)
    r = ok(y0, y1)
    return (y0, y1, r[0], r[1], "bisection") if r else None


def _work(args):
    frame, hints = args
    return [certify(r, h) for r, h in zip(frame.to_dict("records"), hints)]


def run(cases_path, verified_path, precision_path, output, workers):
    cases = pd.read_parquet(cases_path)
    verified = pd.read_parquet(verified_path).set_index("case_id").reindex(cases.case_id)
    hint = verified.independent.to_numpy(float).copy()
    native = cases.native_solution.to_numpy(float)
    accepted = ~cases.fallback.to_numpy()
    use_native = ~np.isfinite(hint) & accepted
    hint[use_native] = native[use_native]
    prec = pd.read_csv(precision_path)
    prec = prec[(prec.condition == "unchanged") & prec.certified]
    pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    hint[pos.loc[prec.case_id].to_numpy()] = prec.effect.to_numpy(float)
    frame = cases[COLS]
    chunks = np.array_split(np.arange(len(frame)), max(1, len(frame) // 2000))
    jobs = [(frame.iloc[c], hint[c]) for c in chunks]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, part in enumerate(ex.map(_work, jobs, chunksize=1)):
            rows.extend(part)
            if n % 20 == 0:
                print(f"chunk {n + 1}/{len(jobs)}", flush=True)
    table = pd.DataFrame(rows)
    table.insert(1, "surface_id", cases.surface_id.to_numpy())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    print(table.cert_status.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--verified", required=True)
    p.add_argument("--precision", default="results/interaction_analysis/precision/certificates.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    run(a.cases, a.verified, a.precision, a.output, a.workers)
