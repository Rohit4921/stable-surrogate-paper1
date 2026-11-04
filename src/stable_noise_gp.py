"""
M2: GP regression with symmetric alpha-stable observation noise.

    f ~ GP(0, sf^2 R_ell),   y_i = f(x_i) + sigma * eps_i,   eps_i iid S_alpha(1, 0, 0)  (ch.f. exp(-|t|^alpha))

The S alpha S density has no closed form.  We tabulate log p_alpha(z) by direct
Fourier inversion  p(z) = (1/pi) int_0^inf cos(tz) exp(-t^alpha) dt  on a fine grid,
switch to the tail expansion  p(z) ~ Gamma(1+alpha) sin(pi alpha/2) / (pi |z|^{1+alpha})
far out, and interpolate with a cubic spline in (log p).  alpha = 2 is N(0, 2).

Posterior of f: Laplace approximation (Newton on the non-log-concave posterior, started
from an IRLS / EM fit).  Because S alpha S is a normal scale mixture, the EM weights are
    w(z) = psi(z)/z,   psi = -(log p)'   (>= 0),
and they are used for the positive-definite predictive covariance, as for GP-t.
Hyperparameters (sf, ell, sigma, alpha) maximise the Laplace marginal likelihood.
"""
import numpy as np
from functools import lru_cache
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import gamma as G
import sys
from surrogates import matern52

ZMAX = 60.0
_ZG = np.r_[np.linspace(0, 10, 1001), np.linspace(10.1, ZMAX, 500)]


@lru_cache(maxsize=4096)
def _table(a_key):
    a = a_key / 1000.0
    # integration grid in t: exp(-t^a) negligible beyond T
    T = (40.0) ** (1 / a)
    t = np.linspace(0, T, 12001)
    w = np.full(t.size, t[1] - t[0]); w[0] = w[-1] = w[0] / 2
    ker = np.exp(-t ** a) * w
    dens = (np.cos(np.outer(_ZG, t)) @ ker) / np.pi
    # tail expansion (two terms) for |z| large; blend where the quadrature loses accuracy
    c1 = G(1 + a) * np.sin(np.pi * a / 2) / np.pi
    c2 = -G(1 + 2 * a) * np.sin(np.pi * a) / (2 * np.pi)
    zz = np.maximum(_ZG, 1e-9)
    tail = c1 * zz ** (-1 - a) + c2 * zz ** (-1 - 2 * a)
    if a < 1.999:
        use_tail = (_ZG > 8) | (dens < 1e-9)
        dens = np.where(use_tail & (tail > 0), tail, dens)
    dens = np.maximum(dens, 1e-300)
    return CubicSpline(_ZG, np.log(dens), bc_type=((1, 0.0), "not-a-knot")), a, c1


# v2 tables (26 Sep 2026): the Fourier table above switched to a two-term tail expansion at |z| = 8,
# leaving a jump of up to 0.04 nats (alpha near 2) and a spike in the first two derivatives there.
# We now use log densities precomputed with scipy.stats.levy_stable (build_tables.py), splined on
# [0, 60], with a continuous power-law extension beyond 60.
import os
import _paths
_NPZ = os.path.join(_paths.DATA, "sas_tables.npz")
_T = np.load(_NPZ) if os.path.exists(_NPZ) else None


@lru_cache(maxsize=512)
def _table_v2(k):
    """k = round(100 * alpha), 50 <= k <= 199."""
    a = k / 100.0
    lp = _T["logp"][k - 50]
    sp = CubicSpline(_T["z"], lp, bc_type=((1, 0.0), "not-a-knot"))
    return sp, a, float(sp(ZMAX)), float(sp(ZMAX, 1))


def _single(z, a_key):
    if _T is None:
        return _single_fourier(z, a_key)
    sp, a, l60, _ = _table_v2(int(round(a_key / 10)))
    s = np.sign(z); az = np.abs(z)
    inside = az <= ZMAX
    lp = np.empty_like(az); d1 = np.empty_like(az); d2 = np.empty_like(az)
    lp[inside] = sp(az[inside]); d1[inside] = sp(az[inside], 1); d2[inside] = sp(az[inside], 2)
    o = ~inside
    if o.any():
        lp[o] = l60 - (1 + a) * np.log(az[o] / ZMAX); d1[o] = -(1 + a) / az[o]; d2[o] = (1 + a) / az[o] ** 2
    return lp, s * d1, d2


def _single_fourier(z, a_key):
    sp, a, c1 = _table(a_key)
    s = np.sign(z); az = np.abs(z)
    inside = az <= ZMAX
    lp = np.empty_like(az); d1 = np.empty_like(az); d2 = np.empty_like(az)
    lp[inside] = sp(az[inside]); d1[inside] = sp(az[inside], 1); d2[inside] = sp(az[inside], 2)
    o = ~inside
    if o.any():
        lp[o] = np.log(c1) - (1 + a) * np.log(az[o]); d1[o] = -(1 + a) / az[o]; d2[o] = (1 + a) / az[o] ** 2
    return lp, s * d1, d2


def _gauss(z):
    return -z ** 2 / 4 - 0.5 * np.log(4 * np.pi), -z / 2, np.full_like(z, -0.5)


def logpdf_derivs(z, a):
    """log p_a(z) and its first two z-derivatives for standard SaS (ch.f. exp(-|t|^a)).
    Tables on an alpha grid of step 0.01, linear interpolation in alpha (so the
    likelihood is continuous and piecewise smooth in alpha); exact Gaussian at a = 2."""
    z = np.asarray(z, dtype=float)
    a = float(min(max(a, 0.5), 2.0))
    lo = int(np.floor(a * 100 + 1e-9)); hi = min(lo + 1, 200); t = a * 100 - lo
    def at(k):
        return _gauss(z) if k >= 200 else _single(z, k * 10)
    A = at(lo)
    if t < 1e-12 or lo >= 200:
        return A
    B = at(hi)
    return tuple((1 - t) * x + t * y for x, y in zip(A, B))


def lik(y, f, sig, a):
    z = (y - f) / sig
    lp, d1, d2 = logpdf_derivs(z, a)
    g = -d1 / sig                      # d/df log p(y|f)
    W = -d2 / sig ** 2                 # -d2/df2
    zs = np.where(np.abs(z) < 1e-8, 1e-8, z)
    w_em = np.maximum(-d1 / zs, 1e-12) / sig ** 2
    return lp - np.log(sig), g, W, w_em


class StableNoiseGP:
    def __init__(self, a_min=0.8, a_max=2.0, fixed_alpha=None):
        self.a_min, self.a_max, self.fixed_alpha = a_min, a_max, fixed_alpha

    def _laplace(self, K, y, sig, a, f0=None):
        n = len(y)
        Kinv = np.linalg.inv(K)
        psi = lambda f: lik(y, f, sig, a)[0].sum() - 0.5 * f @ Kinv @ f
        f = K @ np.linalg.solve(K + sig ** 2 * np.eye(n), y) if f0 is None else f0.copy()
        for _ in range(50 if f0 is None else 10):            # IRLS / EM (fewer when warm-started)
            w = lik(y, f, sig, a)[3]
            f = K @ np.linalg.solve(K + np.diag(1 / w), y)
        cur = psi(f)
        for _ in range(100):                                 # Newton with line search
            _, g, W, _ = lik(y, f, sig, a)
            fn = K @ np.linalg.solve(np.eye(n) + W[:, None] * K, W * f + g)
            step = 1.0
            while True:
                ft = f + step * (fn - f); new = psi(ft)
                if new >= cur - 1e-10 or step < 1e-4: break
                step *= 0.5
            done = abs(new - cur) < 1e-9
            if new >= cur: f, cur = ft, new
            else: break        # no improvement: the state is unchanged, so further iterations would repeat this one
            if done: break
        _, g, W, w_em = lik(y, f, sig, a)
        sign, ld = np.linalg.slogdet(np.eye(n) + K * W[None, :])
        lml = cur - 0.5 * ld if sign > 0 else -1e10
        return f, lml, g, W, w_em

    def _unpack(self, p):
        a = self.fixed_alpha if self.fixed_alpha is not None else p[3]
        return np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), a

    def _neg(self, p, x, y):
        sf, ell, sig, a = self._unpack(p)
        K = matern52(x, x, sf, ell) + 1e-8 * np.eye(len(y))
        f, lml, *_ = self._laplace(K, y, sig, a, getattr(self, "_fw", None))
        self._fw = f
        return -lml

    def fit(self, x, y, restarts=5, seed=0):
        self.x = np.asarray(x, float); self.ym, self.ys = y.mean(), y.std()
        self.y = (y - self.ym) / self.ys
        rng = np.random.default_rng(seed); span = np.ptp(self.x)
        b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0))]
        if self.fixed_alpha is None:
            b.append((self.a_min, self.a_max))
        best = None
        for k in range(restarts):
            self._fw = None
            p0 = np.array([rng.uniform(lo, hi) for lo, hi in b])
            r = minimize(self._neg, p0, args=(self.x, self.y), method="L-BFGS-B", bounds=b,
                         options=dict(eps=1e-3))
            if best is None or r.fun < best.fun: best = r
        self.sf, self.ell, self.sig, self.alpha = self._unpack(best.x)
        n = len(self.y)
        self.K = matern52(self.x, self.x, self.sf, self.ell) + 1e-8 * np.eye(n)
        self._fw = None
        self.f, lml, self.g, self.W, self.w_em = self._laplace(self.K, self.y, self.sig, self.alpha)
        self.lml = lml - n * np.log(self.ys)
        return self

    def predict(self, xs):
        ks = matern52(self.x, xs, self.sf, self.ell)
        mu = ks.T @ self.g
        A = np.linalg.solve(np.eye(len(self.y)) + self.w_em[:, None] * self.K, self.w_em[:, None] * ks)
        var = np.maximum(self.sf ** 2 - (ks * A).sum(0), 1e-12)
        return self.ym + self.ys * mu, self.ys * np.sqrt(var)


def rsas(a, size, rng):
    """Chambers-Mallows-Stuck sampler for standard symmetric alpha-stable (ch.f. exp(-|t|^a))."""
    V = rng.uniform(-np.pi / 2, np.pi / 2, size); E = rng.exponential(size=size)
    if abs(a - 1) < 1e-9:
        return np.tan(V)
    return np.sin(a * V) / np.cos(V) ** (1 / a) * (np.cos(V - a * V) / E) ** ((1 - a) / a)
