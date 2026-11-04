"""Cross-fitted plug-in ML for (alpha, sigma) in M2 with symmetric alpha-stable noise (Theorem on efficient
estimation, Section 4 of Paper 1).

Folds: by rank parity of the sorted inputs.  For each point of one fold, the pilot f_hat is the median of the
k nearest (in rank) observations of the other fold, so f_hat at fold-1 points is independent of fold-1 noise.
Estimator: theta_hat = argmax over Theta of sum_i log{ sigma^{-1} p_alpha((y_i - f_hat_i)/sigma) },
theta = (alpha, log sigma).  Window k_n = smallest odd integer >= max(5, ceil(c * n^{2/3})) (any n^{1/2} << k_n << n^{3/4} is allowed).
"""
import numpy as np
from scipy.optimize import minimize
from ssm_laplace import logpdf_derivs_cubic as L

A_LO, A_HI = 0.8, 1.98          # compact parameter set for alpha inside (0, 2)


def pilot(y, k):
    """Cross-fitted running-median pilot. y must be ordered by x. Returns f_hat (same length as y)."""
    n = len(y); fh = np.empty(n)
    folds = (np.arange(0, n, 2), np.arange(1, n, 2))
    for a, b in ((0, 1), (1, 0)):
        tgt, src = folds[a], folds[b]; ys = y[src]; m = len(src)
        # position of each target point among the source points (rank order)
        pos = np.searchsorted(src, tgt)
        for j, p in enumerate(pos):
            lo = max(0, min(p - k // 2, m - k)); fh[tgt[j]] = np.median(ys[lo:lo + k])
    return fh


def pilot_ll(x, y, k, iters=30):
    """Cross-fitted local-linear median (least absolute deviations) pilot, by IRLS, on the k nearest
    other-fold points (in rank). x, y ordered by x."""
    n = len(y); fh = np.empty(n)
    folds = (np.arange(0, n, 2), np.arange(1, n, 2))
    for a, b in ((0, 1), (1, 0)):
        tgt, src = folds[a], folds[b]; m = len(src)
        pos = np.searchsorted(src, tgt)
        lo = np.clip(pos - k // 2, 0, m - k)
        idx = src[lo[:, None] + np.arange(k)[None, :]]            # (T, k) window indices
        X = x[idx] - x[tgt][:, None]; Y = y[idx]
        hsc = np.abs(X).max(1, keepdims=True) + 1e-12; X = X / hsc
        b0 = np.median(Y, 1); b1 = np.zeros(len(tgt))
        for _ in range(iters):
            r = Y - b0[:, None] - b1[:, None] * X
            w = 1.0 / np.maximum(np.abs(r), 1e-6 * (np.median(np.abs(r), 1, keepdims=True) + 1e-12))
            S0 = w.sum(1); S1 = (w * X).sum(1); S2 = (w * X * X).sum(1); T0 = (w * Y).sum(1); T1 = (w * X * Y).sum(1)
            det = S0 * S2 - S1 * S1
            b0 = (S2 * T0 - S1 * T1) / det; b1 = (S0 * T1 - S1 * T0) / det
        fh[tgt] = b0
    return fh


def robust_scale(y):
    """Centre and scale used for the GP fits: median and normalised MAD. Unlike the sample mean and sd, both
    converge to finite non-zero limits under infill with alpha < 2 noise (the sample sd diverges like
    n^{1/alpha - 1/2}), so the box constraints below define a fixed compact hyperparameter set in original units,
    as the GP-pilot theorem requires."""
    m = np.median(y); s = 1.4826 * np.median(np.abs(y - m))
    return m, (s if s > 0 else np.std(y) + 1e-12)


A_GP_HI = 1.98   # working alpha of the GP pilot kept inside (0, 2): at alpha = 2 the loss is quadratic, not Lipschitz


def _gp_fit(S, x, y, start, maxfev=600):
    """Laplace ML of the GP with SaS noise on robustly standardised y (as ml_rate_study.fit, but median/MAD
    standardisation, alpha <= 1.98 and a smaller evaluation budget: the pilot only needs a good posterior mode)."""
    m, sc = robust_scale(y); ys = (y - m) / sc; span = np.ptp(x)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (0.8, A_GP_HI)]
    lo, hi = np.array([u for u, _ in b]), np.array([v for _, v in b]); warm = [None]
    def negL(p):
        p = np.clip(p, lo, hi)
        try:
            l, fh, av, sg = S.laplace(x, ys, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], warm=warm[0])
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e10
        if sg > 0 and np.isfinite(l): warm[0] = (fh, av); return -l
        return 1e10
    p0 = np.clip(np.array(start, float), lo, hi)
    r1 = minimize(lambda q: negL(np.r_[q, p0[2:]]), p0[:2], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxfev=150))
    r = minimize(negL, np.r_[r1.x, p0[2:]], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxfev=maxfev))
    return np.clip(r.x, lo, hi)


def pilot_gp(x, y, info=None, ftrue=None):
    """Cross-fitted GP pilot: for each fold, Laplace ML of the GP with SaS noise (O(n) state-space code,
    started from the difference estimator) on the OTHER fold, and its posterior mode linearly interpolated
    to this fold's inputs. Covered by the GP-pilot theorem (Appendix C) when the computed mode is the global
    one. If info is a dict, per-fold diagnostics are stored: hyperparameters, sign of det(I + K W) at the mode
    (fold if <= 0) and, if ftrue (array of f at x) is given, the sup-distance between the computed mode and
    the mode reached by Newton started from the truth (a check that the computed mode is the relevant one)."""
    import os, sys, importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("mlr", os.path.join(here, "..", "scripts", "03_stable_noise", "ml_rate_study.py"))
    cwd = os.getcwd(); MR = importlib.util.module_from_spec(spec); spec.loader.exec_module(MR); os.chdir(cwd)
    import ssm_laplace as S
    n = len(y); fh = np.empty(n); folds = (np.arange(0, n, 2), np.arange(1, n, 2))
    for a, b in ((0, 1), (1, 0)):
        tgt, src = folds[a], folds[b]; xs, ys = x[src], y[src]
        m, sc = robust_scale(ys)
        _, ac, st = MR.ecf_start(ys)
        p = _gp_fit(S, xs, ys, [0.0, np.log(0.25 * np.ptp(xs)), np.log(st / sc), min(ac, A_GP_HI)])
        z = (ys - m) / sc
        _, fm, _, sg = S.laplace(xs, z, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
        fh[tgt] = np.interp(x[tgt], xs, m + sc * fm)
        if info is not None:
            d = dict(p=[float(v) for v in p], sign=int(sg))
            if ftrue is not None:
                ft = (ftrue[src] - m) / sc
                _, fm2, _, _ = S.laplace(xs, z, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], warm=(ft, None))
                d["mode_vs_truth_start"] = float(sc * np.max(np.abs(fm - fm2)))
            info.setdefault("folds", []).append(d)
    return fh


def window(n, c=0.5, power=2 / 3):
    """Running-median window: the smallest odd integer >= max(5, ceil(c n^power)). Odd windows make the window
    median a single order statistic, as the Lemma (running-median pilot) requires. (Before 4 Nov 2026 the code used
    max(5, ceil(c n^power)) without the odd rounding, i.e. k = 5, 10, 24, 59 at n = 120, 480, 1920, 7680, and np.median
    averaged the two middle order statistics for even k; the median-pilot column of Table 2 was rerun with odd k.)"""
    k = int(max(5, np.ceil(c * n ** power)))
    return k if k % 2 == 1 else k + 1


def fit(res, start=(1.5, None)):
    """ML for (alpha, log sigma) from residuals res (known-location SaS model)."""
    s0 = np.log(np.median(np.abs(res)) + 1e-12) if start[1] is None else start[1]
    def nll(p):
        a = float(np.clip(p[0], A_LO, A_HI)); s = np.exp(p[1])
        return -(L(res / s, a)[0].sum() - len(res) * p[1])
    best = None
    for a0 in (start[0], 1.2, 1.9):
        r = minimize(nll, [a0, s0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    a = float(np.clip(best.x[0], A_LO, A_HI))
    return a, float(np.exp(best.x[1])), -best.fun


def crossfit_alpha(y, c=0.5, x=None, kind="median", power=2/3, info=None, ftrue=None):
    """y ordered by x. kind = 'median' (running median; Lemma) or 'll' (local-linear median; needs x) or 'gp'.
    Returns (alpha_hat, sigma_hat, k). If info is a dict, the pilot values are stored in info['fhat']."""
    y = np.asarray(y, float); n = len(y)
    k = window(n, c, power)
    if kind == "median": fh = pilot(y, k)
    elif kind == "ll": fh = pilot_ll(np.asarray(x, float), y, k)
    else: fh = pilot_gp(np.asarray(x, float), y, info=info, ftrue=ftrue); k = 0
    if info is not None: info["fhat"] = fh
    a, s, _ = fit(y - fh)
    return a, s, k
