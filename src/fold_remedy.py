"""Laplace variants for M2 with SaS noise and fold diagnostics.
standard : psi(f_hat) - 1/2 log|I + K W|        (W = -d2 log p, can be negative)
fisher   : psi(f_hat) - 1/2 log|I + K W_em|     (W_em = EM weights psi(z)/z/sig^2 > 0; never folds)
lam_min  : smallest eigenvalue of I + K^{1/2} W K^{1/2}; -> 0 at a fold (saddle-node) of the posterior mode.
At a fold (lam_min ~ 0) the mode, and hence the standard Laplace value, is machine-dependent; see README note 6."""
import sys, numpy as np
import exact_m2 as X
from stable_noise_gp import lik
from scipy.optimize import minimize
TAU = 0.1   # fold guard: reject hyperparameters with lam_min(I + K^1/2 W K^1/2) < TAU


LAM_MACHINE = 1e-6   # below this, lam_min is zero to working precision: the mode sits at a fold and the
                     # Laplace value there depends on where the Newton iterate stops (machine-dependent)


def evaluate(E, p):
    """Laplace variants at hyperparameters p = (log sf, log ell, log sig, alpha) on E's standardised data.
    standard is NaN (not the -1e10 sentinel of StableNoiseGP._laplace) when det(I + K W) <= 0, i.e. at or past a
    fold; the Fisher-type value only needs psi at the mode and the positive EM weights, so it is always defined."""
    sf, ell, sig, a = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]
    K, fm, lml, _ = E.laplace(sf, ell, sig, a); n = len(E.y)
    _, g, W, w = lik(E.y, fm, sig, a)
    sgn, ld = np.linalg.slogdet(np.eye(n) + K*W[None, :])
    if sgn > 0:
        psi_val = lml + 0.5*ld                      # identical to the earlier code whenever Laplace is defined
        standard = float(lml)
    else:                                           # psi(f_hat) computed as in StableNoiseGP._laplace
        psi_val = lik(E.y, fm, sig, a)[0].sum() - 0.5*fm @ np.linalg.inv(K) @ fm
        standard = np.nan
    fisher = psi_val - 0.5*np.linalg.slogdet(np.eye(n) + K*w[None, :])[1]
    Ls = np.linalg.cholesky(K); B = np.eye(n) + Ls.T @ (W[:, None]*Ls)
    lam = float(np.linalg.eigvalsh((B + B.T)/2).min())
    return dict(standard=standard, fisher=float(fisher), lam_min=lam, n_neg=int((W < 0).sum()),
                guarded=standard if (lam >= TAU and np.isfinite(standard)) else -np.inf,
                machine_dependent=bool(lam < LAM_MACHINE))


def fit(E, objective="standard", starts=None, maxfev=600):
    span = np.ptp(E.x); n = len(E.y)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02*span), np.log(3*span)), (np.log(1e-3), np.log(2.0)), (0.8, 2.0)]
    lo, hi = np.array([l for l, _ in b]), np.array([h for _, h in b])
    def neg(p):
        try:
            v = evaluate(E, p)[objective]; return -v if np.isfinite(v) else 1e10
        except (np.linalg.LinAlgError, ValueError): return 1e10
    starts = starts or [np.array([0.0, np.log(0.25*span), np.log(0.1), a]) for a in (1.3, 1.7, 1.95)]
    best = None
    for p0 in starts:
        r = minimize(neg, np.clip(p0, lo, hi), method="Nelder-Mead", bounds=b, options=dict(xatol=2e-3, fatol=1e-3, maxfev=maxfev))
        if best is None or r.fun < best.fun: best = r
    return best.x, -best.fun - n*np.log(E.ys)
