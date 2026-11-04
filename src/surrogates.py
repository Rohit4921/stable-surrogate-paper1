"""
GP vs Student-t process surrogate: a from-scratch implementation for the
surrogate-selection demonstration (ERP / Stable Surrogates).

Conventions
-----------
* Inputs x are 1-D arrays; outputs y are standardised (mean 0, sd 1) before
  fitting, and everything is reported back on the original scale.
* Kernel: Matern-5/2 with signal sd sigma_f, length-scale ell; the noise
  variance sigma_n^2 is added on the diagonal, so K_y = sigma_f^2 R + sigma_n^2 I.
* GP:  y ~ N_n(0, K_y).
* TP (Shah, Wilson & Ghahramani 2014, noise inside the kernel):
       y ~ MVT_n(nu, 0, K_y),  with Cov(y) = K_y  (nu > 2).
  Density:
    log p = lgamma((nu+n)/2) - lgamma(nu/2) - n/2 log((nu-2) pi)
            - 1/2 log|K_y| - (nu+n)/2 log(1 + beta/(nu-2)),  beta = y' K_y^{-1} y.
  Conditional (Shah et al., Lemma 3): y2 | y1 ~ MVT(nu + n1, mu_GP, (nu+beta-2)/(nu+n1-2) Sigma_GP),
  i.e. a Student-t with d = nu+n1 degrees of freedom, the GP mean, and
  scale^2 = (nu+beta-2)/(nu+n1) * Sigma_GP.
"""
import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln, betaln
from scipy import stats


# ----------------------------------------------------------------- kernel
def matern52(x1, x2, sf, ell):
    """Isotropic Matern-5/2. 1-D arrays (n,) as before; 2-D arrays (n, d) use Euclidean distance."""
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    if x1.ndim == 2:
        r = np.sqrt(np.maximum(((x1[:, None, :] - x2[None, :, :]) ** 2).sum(-1), 0.0)) / ell
    else:
        r = np.abs(x1[:, None] - x2[None, :]) / ell
    s5 = np.sqrt(5.0) * r
    return sf**2 * (1 + s5 + 5.0 / 3.0 * r**2) * np.exp(-s5)


# ----------------------------------------------------------------- scores
def t_logpdf(y, mu, s, d):
    return stats.t.logpdf(y, df=d, loc=mu, scale=s)


def crps_t(y, mu, s, d):
    """Closed-form CRPS of a location-scale Student-t (Jordan, Krueger & Lerch 2019)."""
    z = (y - mu) / s
    if np.isinf(d) or d > 1e6:
        return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    F, f = stats.t.cdf(z, d), stats.t.pdf(z, d)
    c = 2 * np.sqrt(d) * np.exp(betaln(0.5, d - 0.5) - 2 * betaln(0.5, d / 2)) / (d - 1)
    return s * (z * (2 * F - 1) + 2 * f * (d + z**2) / (d - 1) - c)


def ei_t(mu, s, d, best):
    """Expected improvement (maximisation) under a Student-t predictive; d=inf gives Gaussian EI."""
    s = np.maximum(s, 1e-12)
    z = (mu - best) / s
    if np.isinf(d):
        return (mu - best) * stats.norm.cdf(z) + s * stats.norm.pdf(z)
    return (mu - best) * stats.t.cdf(z, d) + s * (d + z**2) / (d - 1) * stats.t.pdf(z, d)


# ----------------------------------------------------------------- model
class Surrogate:
    """kind = 'gp' or 'tp'.  If kind == 'tp' and nu_fixed is given, nu is not learned."""

    def __init__(self, kind="gp", nu_fixed=None, nu_max=1000.0):
        self.kind, self.nu_fixed, self.nu_max = kind, nu_fixed, nu_max

    # ---- parameter packing (log scale)
    def _unpack(self, p):
        sf, ell, sn = np.exp(p[:3])
        if self.kind == "gp":
            nu = np.inf
        elif self.nu_fixed is not None:
            nu = self.nu_fixed
        else:
            nu = 2.0 + np.exp(p[3])
        return sf, ell, sn, nu

    def _nlml(self, p, x, y):
        sf, ell, sn, nu = self._unpack(p)
        n = len(y)
        K = matern52(x, x, sf, ell) + (sn**2 + 1e-8) * np.eye(n)
        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve((L, True), y)
        beta = y @ a
        logdet = 2 * np.log(np.diag(L)).sum()
        if np.isinf(nu):
            return 0.5 * beta + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        return -(gammaln((nu + n) / 2) - gammaln(nu / 2) - 0.5 * n * np.log((nu - 2) * np.pi)
                 - 0.5 * logdet - 0.5 * (nu + n) * np.log1p(beta / (nu - 2)))

    def fit(self, x, y, restarts=12, seed=0):
        self.x = np.asarray(x, float)
        self.ym, self.ys = y.mean(), y.std()
        self.y = (y - self.ym) / self.ys
        rng = np.random.default_rng(seed)
        span = np.ptp(self.x)
        b = [(np.log(0.05), np.log(10.0)),            # sigma_f
             (np.log(0.01 * span), np.log(5 * span)),  # ell
             (np.log(1e-3), np.log(2.0))]              # sigma_n
        learn_nu = self.kind == "tp" and self.nu_fixed is None
        if learn_nu:
            b.append((np.log(0.05), np.log(self.nu_max - 2)))
        best = None
        for _ in range(restarts):
            p0 = np.array([rng.uniform(lo, hi) for lo, hi in b])
            r = minimize(self._nlml, p0, args=(self.x, self.y), method="L-BFGS-B", bounds=b)
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x
        self.sf, self.ell, self.sn, self.nu = self._unpack(self.p)
        n = len(self.y)
        # log marginal likelihood on the ORIGINAL y scale (Jacobian of standardisation)
        self.lml = -best.fun - n * np.log(self.ys)
        K = matern52(self.x, self.x, self.sf, self.ell) + (self.sn**2 + 1e-8) * np.eye(n)
        self.L = cholesky(K)
        self.Kinv = cho_solve((self.L, True), np.eye(n))
        self.alpha = self.Kinv @ self.y
        self.beta = self.y @ self.alpha
        return self

    # ---- predictive for new points: returns (mu, scale, dof) on original scale
    def predict(self, xs, noisy=False):
        ks = matern52(self.x, xs, self.sf, self.ell)
        mu = ks.T @ self.alpha
        v = solve_triangular(self.L, ks, lower=True)
        var = self.sf**2 - (v**2).sum(0) + (self.sn**2 if noisy else 0.0)
        var = np.maximum(var, 1e-12)
        n = len(self.y)
        if np.isinf(self.nu):
            s, d = np.sqrt(var), np.inf
        else:
            d = self.nu + n
            s = np.sqrt((self.nu + self.beta - 2) / (self.nu + n) * var)
        return self.ym + self.ys * mu, self.ys * s, d

    # ---- closed-form leave-one-out predictive for each training y_i
    def loo(self):
        A, a, n = self.Kinv, self.alpha, len(self.y)
        Aii = np.diag(A)
        mu = self.y - a / Aii
        var = 1.0 / Aii
        if np.isinf(self.nu):
            s, d = np.sqrt(var), np.inf
        else:
            beta_i = self.beta - a**2 / Aii                 # y_{-i}' K_{-i}^{-1} y_{-i}
            d = self.nu + n - 1
            s = np.sqrt((self.nu + beta_i - 2) / (self.nu + n - 1) * var)
        return self.ym + self.ys * mu, self.ys * s, d

    def loo_scores(self, y):
        mu, s, d = self.loo()
        if np.isinf(d):
            lpd = stats.norm.logpdf(y, mu, s)
        else:
            lpd = t_logpdf(y, mu, s, d)
        return lpd, crps_t(y, mu, s, d), mu, s, d


def profile_nu(x, y, nus, restarts=4):
    """Profile log marginal likelihood of the TP over a grid of nu."""
    out = []
    for nu in nus:
        m = Surrogate("tp", nu_fixed=nu).fit(x, y, restarts=restarts)
        out.append(m.lml)
    return np.array(out)


# ======================================================================
# GP with Student-t likelihood (independent heavy-tailed noise), Laplace approximation
#   f ~ GP(0, sigma_f^2 R),  y_i | f_i ~ t_nu(f_i, sigma^2)  (scale sigma)
# Newton iterations on Psi(f) = log p(y|f) - 1/2 f' K^{-1} f  (W may be negative:
# the t likelihood is not log-concave, so plain dense solves with a line search are used).
# ======================================================================
from numpy.polynomial.hermite_e import hermegauss
_GH_X, _GH_W = hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()


def _tlik(y, f, sig, nu):
    r = y - f
    q = nu * sig**2 + r**2
    lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * sig**2)
          - (nu + 1) / 2 * np.log1p(r**2 / (nu * sig**2)))
    g = (nu + 1) * r / q
    W = (nu + 1) * (nu * sig**2 - r**2) / q**2
    return lp, g, W


def _em_weights(y, f, sig, nu):
    """Scale-mixture (EM) precisions at the mode: w_i / sig^2, w_i = (nu+1)/(nu + r_i^2/sig^2) > 0.
    The Laplace mode satisfies f = K (K + diag(sig^2/w))^{-1} y, so these weights give a
    coherent, positive-definite Gaussian approximation (heteroscedastic GP)."""
    return (nu + 1) / (nu + (y - f)**2 / sig**2) / sig**2


class TLikGP:
    kind = "gp_tlik"

    def __init__(self, nu_max=1000.0):
        self.nu_max = nu_max

    def _laplace(self, K, y, sig, nu, f0=None):
        n = len(y)
        Kinv = np.linalg.inv(K)
        def psi(f):
            return _tlik(y, f, sig, nu)[0].sum() - 0.5 * f @ Kinv @ f
        # start from the Gaussian-noise GP mean (avoids spurious modes of the non-log-concave
        # t posterior); keep a warm start only if it is better
        # candidate starts: IRLS / EM for the normal scale mixture (down-weights outliers),
        # a plain Gaussian-noise GP mean, and the warm start; keep the best
        cands = []
        f = K @ np.linalg.solve(K + sig**2 * np.eye(n), y)
        for _ in range(60):
            w = (nu + 1) / (nu + (y - f)**2 / sig**2)
            f = K @ np.linalg.solve(K + np.diag(sig**2 / w), y)
        cands.append(f)
        cands.append(K @ np.linalg.solve(K + sig**2 * (nu / (nu - 2) if nu > 2.5 else 5.0) * np.eye(n), y))
        if f0 is not None:
            cands.append(f0.copy())
        f = max(cands, key=psi)
        cur = psi(f)
        for _ in range(100):
            lp, g, W = _tlik(y, f, sig, nu)
            b = W * f + g
            fn = K @ np.linalg.solve(np.eye(n) + W[:, None] * K, b)
            step = 1.0
            while True:
                ft = f + step * (fn - f)
                new = psi(ft)
                if new >= cur - 1e-10 or step < 1e-4:
                    break
                step *= 0.5
            if abs(new - cur) < 1e-9:
                f, cur = ft, new
                break
            f, cur = ft, new
        lp, g, W = _tlik(y, f, sig, nu)
        B = np.eye(n) + K * W[None, :]          # I + K W
        sign, logdetB = np.linalg.slogdet(B)
        if sign <= 0:
            return f, -1e10, g, W
        return f, cur - 0.5 * logdetB, g, W

    def _unpack(self, p):
        return np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), 1.0 + np.exp(p[3])

    def _neg(self, p, x, y):
        sf, ell, sig, nu = self._unpack(p)
        K = matern52(x, x, sf, ell) + 1e-8 * np.eye(len(y))
        f, lml, _, _ = self._laplace(K, y, sig, nu, getattr(self, "_fwarm", None))
        self._fwarm = f
        return -lml

    def fit(self, x, y, restarts=12, seed=0, standardise="sd"):
        self.x = np.asarray(x, float)
        self.ym, self.ys = robust_scale(y) if standardise == "robust" else (y.mean(), y.std())
        self.y = (y - self.ym) / self.ys
        rng = np.random.default_rng(seed)
        span = np.ptp(self.x)
        b = [(np.log(0.05), np.log(10.0)), (np.log(0.01 * span), np.log(5 * span)),
             (np.log(1e-3), np.log(2.0)), (np.log(0.2), np.log(self.nu_max - 1))]
        best = None
        for _ in range(restarts):
            self._fwarm = None
            p0 = np.array([rng.uniform(lo, hi) for lo, hi in b])
            r = minimize(self._neg, p0, args=(self.x, self.y), method="L-BFGS-B", bounds=b)
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x
        self.sf, self.ell, self.sig, self.nu = self._unpack(best.x)
        self.sn = self.sig
        n = len(self.y)
        self.K = matern52(self.x, self.x, self.sf, self.ell) + 1e-8 * np.eye(n)
        self._fwarm = None
        self.f, lml, self.g, self.W = self._laplace(self.K, self.y, self.sig, self.nu)
        self.lml = lml - n * np.log(self.ys)
        self.Weff = _em_weights(self.y, self.f, self.sig, self.nu)
        return self

    def predict(self, xs, noisy=False):
        ks = matern52(self.x, xs, self.sf, self.ell)
        mu = ks.T @ self.g
        n = len(self.y)
        A = np.linalg.solve(np.eye(n) + self.Weff[:, None] * self.K, self.Weff[:, None] * ks)
        var = np.maximum(self.sf**2 - (ks * A).sum(0), 1e-12)
        return self.ym + self.ys * mu, self.ys * np.sqrt(var), np.inf   # latent f is Gaussian

    def loo_scores(self, y, nsamp=20000, seed=1):
        """Exact-refit LOO with hyperparameters fixed (as for the GP/TP closed forms):
        re-run the Laplace approximation without point i, then integrate the t noise
        against the Gaussian latent predictive by Gauss-Hermite quadrature."""
        n = len(y)
        ys = (y - self.ym) / self.ys
        lpd = np.empty(n); crps = np.empty(n); mus = np.empty(n); sds = np.empty(n)
        rng = np.random.default_rng(seed)
        for i in range(n):
            keep = np.arange(n) != i
            Ki = self.K[np.ix_(keep, keep)]
            fi, _, gi, Wi = self._laplace(Ki, self.y[keep], self.sig, self.nu, self.f[keep])
            ks = self.K[keep, i]
            m = ks @ gi
            We = _em_weights(self.y[keep], fi, self.sig, self.nu)
            A = np.linalg.solve(np.eye(n - 1) + We[:, None] * Ki, We * ks)
            v = max(self.K[i, i] - ks @ A, 1e-12)
            fq = m + np.sqrt(v) * _GH_X
            dens = np.exp(_tlik(ys[i], fq, self.sig, self.nu)[0])
            lpd[i] = np.log((_GH_W * dens).sum()) - np.log(self.ys)
            smp = m + np.sqrt(v) * rng.standard_normal(nsamp) + self.sig * rng.standard_t(self.nu, nsamp)
            a, b2 = smp[: nsamp // 2], smp[nsamp // 2:]
            crps[i] = self.ys * (np.abs(smp - ys[i]).mean() - 0.5 * np.abs(a - b2).mean())
            mus[i], sds[i] = m, np.sqrt(v)
        return lpd, crps, self.ym + self.ys * mus, self.ys * sds, self.nu


def robust_scale(y):
    """Median and normalised median absolute deviation (falls back to the sd if the MAD is zero)."""
    y = np.asarray(y, float); m = np.median(y); s = 1.4826 * np.median(np.abs(y - m))
    return m, (s if s > 0 else y.std() + 1e-12)


# ======================================================================
# RCGP-type robust GP (after Altamirano, Briol and Knoblauch, robust and conjugate GP regression):
#   generalised-Bayes posterior with inverse-multiquadric weights
#     w(x, y) = beta (1 + (y - m(x))^2 / c^2)^{-1/2},  beta = sigma / sqrt(2),
#   giving the conjugate update with heteroscedastic noise sigma^2 J_w, J_w = diag(sigma^2 w^{-2} / 2)
#   = diag(1 + r^2/c^2), and shifted targets m_w = m + sigma^2 d/dy log w^2 = m - 2 sigma^2 r / (c^2 + r^2).
#   Prior mean m: the median of y (after robust standardisation, m = 0); c: the 95% quantile of |y - m|.
#   Hyperparameters (sigma_f, ell, sigma) by the closed-form leave-one-out predictive log density.
#   Our implementation; constants follow our reading of the method and should be checked against the paper.
# ======================================================================
class RCGP:
    """RCGP-type robust conjugate GP (Altamirano, Briol and Knoblauch, ICML 2024) with IMQ weights
    w_i = beta {1 + (y_i - m)^2 / c^2}^(-1/2), constant prior mean m = 0 on the median/MAD-standardised scale,
    beta = sigma / sqrt(2) (sigma = Gaussian noise sd; their recommended choice, so c -> infinity gives the GP),
    and c the empirical (1 - eps) quantile of |y_i - m|, eps = 0.05. With this beta the robust posterior is a GP
    posterior with noise covariance sigma^2 diag(J), J_i = 1 + r_i^2 / c^2, and data z = y - m_w with
    m_w = m - 2 sigma^2 r / (c^2 + r^2), r = y - m (see _parts). Kernel and noise hyperparameters by LOO
    predictive density; c by the quantile rule, not by LOO."""
    kind = "rcgp"

    def __init__(self, eps=0.05):
        self.eps = eps

    def _parts(self, p, x, y):
        sf, ell, sn = np.exp(p)
        n = len(y); r = y.copy()                      # prior mean 0 on the robustly standardised scale
        c = max(np.quantile(np.abs(r), 1 - self.eps), 1e-6)
        J = 1 + r ** 2 / c ** 2
        z = y + 2 * sn ** 2 * r / (c ** 2 + r ** 2)    # y - m_w
        A = matern52(x, x, sf, ell) + sn ** 2 * np.diag(J) + 1e-8 * np.eye(n)
        return sf, ell, sn, A, z

    def _neg_loo(self, p, x, y):
        try:
            sf, ell, sn, A, z = self._parts(p, x, y)
            Ai = np.linalg.inv(A); d = np.diag(Ai)
            if np.any(d <= 0): return 1e10
            mu = z - (Ai @ z) / d; v = 1 / d
            return -np.sum(-0.5 * np.log(2 * np.pi * v) - 0.5 * (z - mu) ** 2 / v)
        except np.linalg.LinAlgError:
            return 1e10

    def fit(self, x, y, restarts=4, seed=0):
        self.x = np.asarray(x, float); self.ym, self.ys = robust_scale(y); self.y = (y - self.ym) / self.ys
        rng = np.random.default_rng(seed); span = np.ptp(self.x)
        b = [(np.log(0.05), np.log(10.0)), (np.log(0.01 * span), np.log(5 * span)), (np.log(1e-3), np.log(2.0))]
        best = None
        for _ in range(restarts):
            p0 = np.array([rng.uniform(lo, hi) for lo, hi in b])
            r = minimize(self._neg_loo, p0, args=(self.x, self.y), method="L-BFGS-B", bounds=b)
            if best is None or r.fun < best.fun: best = r
        self.p = best.x
        self.sf, self.ell, self.sn, A, z = self._parts(best.x, self.x, self.y)
        self.L = cholesky(A); self.a = cho_solve((self.L, True), z)
        return self

    def predict(self, xs, noisy=False):
        ks = matern52(self.x, xs, self.sf, self.ell)
        mu = ks.T @ self.a
        v = solve_triangular(self.L, ks, lower=True)
        var = np.maximum(self.sf ** 2 - (v ** 2).sum(0), 1e-12)
        return self.ym + self.ys * mu, self.ys * np.sqrt(var), np.inf
