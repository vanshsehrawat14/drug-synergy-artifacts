"""Python emulation of the DrugComb/TidyComb Loewe reference (SynergyFinder 2.0.12, R/calculate_synergy_score.R),
including the numerical behaviour that decides when its fallback fires:
  * single-drug fit: drc::drm(LL.4()) = drc self-start on scaled data + R optim(BFGS) (port of R's vmmin with
    optim's central-difference gradient, ndeps 1e-3, parscale = |start|), hessian check; on failure drm(L.4) on log dose;
  * root: nleqslv(method = "Newton", global = "dbldog") 1-D port (nwsolv/nwddlg/nwtrup/nwtcvg, ftol = xtol = 1e-8,
    maxit 150), started at max(c_col, c_row) + 1; non-finite f at start or inside the finite-difference Jacobian is an R
    error, which SynergyFinder catches as termcd 999;
  * accept slv$x when termcd < 3, else fallback max(f_col(x1 + x2), f_row(x1 + x2)) (summed raw concentrations);
  * cap the reference at 100.
Sources read: SynergyFinder 2.0.12 / 1.8.0 (Bioconductor), nleqslv 3.3.2 Fortran, drc 3.0-1 R code.
"""
import math
import numpy as np

EPS = 2.220446049250313e-16
BIG = math.sqrt(1.7976931348623157e308)


class RError(Exception):
    pass


# ---------------------------------------------------------------- R optim(method = "BFGS") port (vmmin)
def _vmmin(fn, b, parscale, maxit=500, reltol=1e-7, ndeps=1e-3):
    n = len(b); p = [b[i] / parscale[i] for i in range(n)]

    def f_s(q):
        return fn([q[i] * parscale[i] for i in range(n)])

    def g_s(q):
        g = [0.0] * n
        for i in range(n):
            x = list(q); x[i] = q[i] + ndeps; v1 = f_s(x); x[i] = q[i] - ndeps; v2 = f_s(x)
            g[i] = (v1 - v2) / (2 * ndeps)
            if not math.isfinite(g[i]):
                raise RError('non-finite finite-difference value')
        return g

    f = f_s(p)
    if not math.isfinite(f):
        raise RError("initial value in 'vmmin' is not finite")
    Fmin = f; gradcount = 1; g = g_s(p); it = 1; ilast = gradcount
    B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    while True:
        if ilast == gradcount:
            B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        X = list(p); c = list(g)
        t = [-sum(B[i][j] * g[j] for j in range(n)) for i in range(n)]
        gradproj = sum(t[i] * g[i] for i in range(n))
        if gradproj < 0.0:
            steplength = 1.0; accpoint = False
            while True:
                count = 0
                for i in range(n):
                    p[i] = X[i] + steplength * t[i]
                    if 10.0 + X[i] == 10.0 + p[i]:
                        count += 1
                if count < n:
                    f = f_s(p)
                    accpoint = math.isfinite(f) and f <= Fmin + gradproj * steplength * 1e-4
                    if not accpoint:
                        steplength *= 0.2
                if count == n or accpoint:
                    break
            enough = (f > -math.inf) and abs(f - Fmin) > reltol * (abs(Fmin) + reltol)
            if not enough:
                count = n; Fmin = f
            if count < n:
                Fmin = f; g = g_s(p); gradcount += 1; it += 1
                t = [steplength * t[i] for i in range(n)]; c = [g[i] - c[i] for i in range(n)]
                D1 = sum(t[i] * c[i] for i in range(n))
                if D1 > 0:
                    Xc = [sum(B[i][j] * c[j] for j in range(n)) for i in range(n)]
                    D2 = 1.0 + sum(Xc[i] * c[i] for i in range(n)) / D1
                    for i in range(n):
                        for j in range(n):
                            B[i][j] += (D2 * t[i] * t[j] - Xc[i] * t[j] - t[i] * Xc[j]) / D1
                else:
                    ilast = gradcount
            else:
                if ilast < gradcount:
                    count = 0; ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if it >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if count == n and ilast == gradcount:
            break
    g_s(p)  # optim(hessian = TRUE): numerical Hessian needs finite gradients around the optimum
    for i in range(n):
        for s in (1, -1):
            q = list(p); q[i] += s * ndeps; g_s(q)
    return [p[i] * parscale[i] for i in range(n)], Fmin


# ---------------------------------------------------------------- drc LL.4 / L.4 fits
def ll4(x, b, c, d, e):
    if x <= 0:
        return c if b < 0 else (d if b > 0 else float('nan'))
    try:
        z = math.exp(b * (math.log(x) - math.log(e)))
    except OverflowError:
        return c
    except ValueError:
        return float('nan')
    return c + (d - c) / (1 + z)


def l4(x, b, c, d, e):  # x on log scale
    try:
        return c + (d - c) / (1 + math.exp(b * (x - e)))
    except OverflowError:
        return c


def _ols(x, y):
    mx = sum(x) / len(x); my = sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x); sxy = sum((a - mx) * (bb - my) for a, bb in zip(x, y))
    slope = sxy / sxx; return my - slope * mx, slope


def _median(v):
    s = sorted(v); n = len(s); return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_ll4(dose, resp):
    """drc::drm(response ~ dose, fct = LL.4()) as called by SynergyFinder 2 FitDoseResponse."""
    resp = list(resp)
    if np.var(resp) == 0:
        resp[-1] += 1e-10
    md = _median(dose); ds = 10 ** math.floor(math.log10(md)) if md > 0 else 1.0
    if not ds >= 1e-15:
        ds = 1.0
    mr = _median(resp)
    rs = 10 ** math.floor(math.log10(mr)) if mr > 0 else 1.0
    if not rs >= 1e-15:
        rs = 1.0
    x = [v / ds for v in dose]; y = [v / rs for v in resp]
    lo, hi = min(y), max(y); r = 0.001 * (hi - lo); c0, d0 = lo - r, hi + r
    keep = [(math.log(a), math.log((d0 - bb) / (bb - c0))) for a, bb in zip(x, y) if a > 0]
    icpt, slope = _ols([k[0] for k in keep], [k[1] for k in keep])
    b0 = slope; e0 = math.exp(-icpt / b0)
    start = [b0, c0, d0, e0]; ps = [abs(v) if abs(v) >= 1e-4 else 1.0 for v in start]

    def rss(par):
        b, c, d, e = par
        s = 0.0
        for a, bb in zip(x, y):
            if a > 0:
                try:
                    z = math.exp(b * math.log(a / e)) if e > 0 else float('nan')
                except OverflowError:
                    z = math.inf
                except ValueError:
                    return float('nan')
                v = c + (d - c) / (1 + z)
            else:
                v = c if b < 0 else (d if b > 0 else float('nan'))
            s += (bb - v) ** 2
        return s
    par, _ = _vmmin(rss, start, ps)
    b, c, d, e = par
    return ('LL.4', (b, c * rs, d * rs, e * ds))


def fit_l4(dose, resp):
    """Fallback drm(response ~ log(dose), fct = L.4()) with dose 0 -> 1e-10 (no drc scaling for L.4)."""
    resp = list(resp)
    if np.var(resp) == 0:
        resp[-1] += 1e-10
    x = [math.log(v if v > 0 else 1e-10) for v in dose]; y = resp
    lo, hi = min(y), max(y); r = 0.001 * (hi - lo); c0, d0 = lo - r, hi + r
    icpt, slope = _ols(x, [math.log((d0 - bb) / (bb - c0)) for bb in y])
    start = [slope, c0, d0, -icpt / slope]; ps = [abs(v) if abs(v) >= 1e-4 else 1.0 for v in start]

    def rss(par):
        return sum((bb - l4(a, *par)) ** 2 for a, bb in zip(x, y))
    par, _ = _vmmin(rss, start, ps)
    return ('L.4', tuple(par))


def fit_dose_response(dose, resp):
    try:
        return fit_ll4(dose, resp)
    except (RError, ZeroDivisionError, ValueError, OverflowError):
        try:
            return fit_l4(dose, resp)
        except (RError, ZeroDivisionError, ValueError, OverflowError):
            return ('FAIL', None)


def curve(model, par, x):
    """Fitted single-drug response at raw concentration x."""
    if model == 'LL.4':
        b, c, d, e = par
        return ll4(x, b, c, d, e)
    return l4(math.log(x) if x > 0 else -math.inf, *par)


# ---------------------------------------------------------------- Loewe equation and nleqslv port
def _rpow(base, ex):
    """R semantics of base^ex for doubles: NaN for negative base with non-integer exponent."""
    try:
        if base < 0 and ex != int(ex):
            return float('nan')
        return base ** ex
    except ZeroDivisionError:
        return math.inf
    except OverflowError:
        return math.inf


def _term(y, x, model, par):
    b, c, d, e = par
    if model == 'LL.4':
        den = c - y
        r = (y - d) / den if den != 0 else math.copysign(math.inf, y - d) if y != d else float('nan')
        q = _rpow(r, 1.0 / b)
        dd = e * q
        return x / dd if dd != 0 else (math.inf if x > 0 else float('nan'))
    # L.4 (eq.L4): x / exp(e + log((d - y)/(y - c)) / b)
    den = y - c
    r = (d - y) / den if den != 0 else math.copysign(math.inf, d - y)
    if r < 0 or math.isnan(r):
        return float('nan')
    try:
        lg = math.log(r) if r > 0 else -math.inf
        return x / math.exp(e + lg / b)
    except (OverflowError, ZeroDivisionError):
        return 0.0 if (e + lg / b) > 0 else math.inf


def loewe_fn(y, x1, x2, mc, pc, mr, pr):
    v = _term(y, x1, mc, pc) + _term(y, x2, mr, pr) - 1
    return v


def nleqslv_1d(fun, x0, ftol=1e-8, xtol=1e-8, maxit=150):
    """Returns (termcd, x). termcd 999 = R error (non-finite start or non-finite value inside the Jacobian)."""
    fc = fun(x0)
    if not math.isfinite(fc):
        return 999, x0
    if abs(fc) <= ftol:
        return 1, x0
    p = math.sqrt(max(10 ** (math.log10(EPS)), EPS))
    xc = x0; delta = -2.0; it = 0; xprev = fprev = fpnsav = None
    while True:
        it += 1
        h = p + p * abs(xc); xh = xc + h; h = xh - xc
        fz = fun(xh)
        if not math.isfinite(fz):
            return 999, xc
        J = (fz - fc) / h
        if J == 0 or not math.isfinite(J):
            return 6, xc
        dn = -fc / J; dnlen = abs(dn); g = J * fc
        if delta == -2.0:
            delta = dnlen
        fcnorm = 0.5 * fc * fc; retcd = 4
        while retcd > 1:
            if dnlen <= delta:
                d = dn; delta = dnlen; nwt = True
            else:
                d = dn * delta / dnlen; nwt = False
            fpred = 0.5 * (J * d + fc) ** 2
            xp = xc + d; fp = fun(xp)
            if not math.isfinite(fp):
                fp = BIG
            fpnorm = 0.5 * fp * fp
            ared = fpnorm - fcnorm; pred = fpred - fcnorm; slope = g * d
            ret3ok = (fpnorm >= fpnsav or ared > 1e-4 * slope) if retcd == 3 else False
            if retcd == 3 and ret3ok:
                retcd = 0; xp, fp, fpnorm = xprev, fprev, fpnsav; delta *= 0.5
            elif ared > 1e-4 * slope:
                if abs(d) / max(abs(xp), 1.0) < xtol:
                    retcd = 1
                else:
                    retcd = 2; dltmp = -slope * abs(d) / (2 * (ared - slope))
                    delta = 0.1 * delta if dltmp < 0.1 * delta else min(0.5 * delta, dltmp)
            elif retcd != 2 and abs(pred - ared) <= 0.1 * abs(ared) and not nwt:
                xprev, fprev, fpnsav = xp, fp, fpnorm; delta = 2 * delta; retcd = 3
            else:
                retcd = 0
                if ared >= 0.1 * pred:
                    delta *= 0.5
                elif ared <= 0.75 * pred:
                    delta = 2 * delta
        if abs(fp) <= ftol:
            return 1, xp
        if retcd == 1:
            return 3, xp
        if abs(xp - xc) / max(abs(xp), 1.0) <= xtol:
            return 2, xp
        if it >= maxit:
            return 4, xp
        xc, fc = xp, fp


def sf2_loewe_ref(x1, x2, mc, pc, mr, pr, fallback='summed', cap=True):
    """SynergyFinder 2 Loewe reference at (col dose x1, row dose x2). Returns (value, code), where code is the
    nleqslv termination code (999 = caught R error) and value is after fallback/cap."""
    start = max(pc[1], pr[1]) + 1
    code, y = nleqslv_1d(lambda v: loewe_fn(v, x1, x2, mc, pc, mr, pr), start)
    if code >= 3:
        if fallback == 'summed':
            y = max_r(curve(mc, pc, x1 + x2), curve(mr, pr, x1 + x2))
        elif fallback == 'hsa_fit':
            y = max_r(curve(mc, pc, x1), curve(mr, pr, x2))
        else:
            raise ValueError(fallback)
    if cap and y > 100:
        y = 100.0
    return y, code


def max_r(a, b):
    """R's max(): NaN-propagating."""
    if math.isnan(a) or math.isnan(b):
        return float('nan')
    return max(a, b)


def active(m, p):
    """Increasing fitted curve (inhibition rises with dose)."""
    b, c, d, e = p
    return d > c and b < 0


def _gterm(y, x, m, p):
    b, c, d, e = p
    if y >= d:
        return 0.0
    if y <= c:
        return math.inf
    return _term(y, x, m, p)


def generalized_loewe(x1, x2, mc, pc, mr, pr, iters=90):
    """Exact generalised Loewe (R15b convention): a drug's isobole term is 0 once the effect exceeds its own maximum.
    Inactive (non-increasing) curves contribute no dose equivalence: L = the active drug's fitted effect at its own
    dose; if both are inactive, L = the larger fitted effect. Bisection between max(lower) and max(upper asymptotes)."""
    ac, ar = active(mc, pc), active(mr, pr)
    if not ac and not ar:
        return max_r(curve(mc, pc, x1), curve(mr, pr, x2))
    if not ac:
        return curve(mr, pr, x2)
    if not ar:
        return curve(mc, pc, x1)
    a, bnd = max(pc[1], pr[1]), max(pc[2], pr[2])
    for _ in range(iters):
        mid = 0.5 * (a + bnd)
        if _gterm(mid, x1, mc, pc) + _gterm(mid, x2, mr, pr) - 1 > 0:
            a = mid
        else:
            bnd = mid
    return 0.5 * (a + bnd)


def exist_rule_loewe(x1, x2, mc, pc, mr, pr, iters=90):
    """Simplified SF2: the root is used when it exists inside the joint domain (both curves increasing and the isobole
    sum is < 1 just below the smaller upper asymptote); otherwise the summed-concentration fallback. Cap at 100."""
    if active(mc, pc) and active(mr, pr):
        lo, hi = max(pc[1], pr[1]), min(pc[2], pr[2])
        top = hi - 1e-9 * max(1.0, abs(hi))
        if hi > lo and _gterm(top, x1, mc, pc) + _gterm(top, x2, mr, pr) - 1 < 0:
            a, bnd = lo, hi
            for _ in range(iters):
                mid = 0.5 * (a + bnd)
                if _gterm(mid, x1, mc, pc) + _gterm(mid, x2, mr, pr) - 1 > 0:
                    a = mid
                else:
                    bnd = mid
            return min(0.5 * (a + bnd), 100.0)
    y = max_r(curve(mc, pc, x1 + x2), curve(mr, pr, x1 + x2))
    return min(y, 100.0) if not math.isnan(y) else y


def ablate(m, p, kind):
    """Curve modifications for the reference-algebra sub-ablation (applied to active curves only)."""
    if m == 'FAIL' or not active(m, p):
        return p
    b, c, d, e = p
    if kind == 'h1':
        return (-1.0, c, d, e)
    if kind == 'd100':
        return (b, c, 100.0, e)
    if kind == 'ideal':
        return (-1.0, 0.0, 100.0, e)
    return p
