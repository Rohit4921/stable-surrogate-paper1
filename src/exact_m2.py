"""
Exact (Monte Carlo) log marginal likelihood for M2: GP with symmetric alpha-stable noise.

    log p(y | sf, ell, sig, a) = log  int N(f; 0, K) prod_i p_a((y_i - f_i)/sig)/sig  df

Two estimators, both on the standardised scale used by StableNoiseGP:
  * IS  : importance sampling from a multivariate t_4 proposal centred at the Laplace mode,
          with the EM-weight covariance (K^{-1} + diag w)^{-1}.  Common random numbers make the
          estimate a smooth function of the hyperparameters, so it can be optimised.
  * AIS : annealed importance sampling from the prior N(0, K) to the posterior with
          elliptical slice sampling (Murray, Adams & MacKay 2010) at each temperature.
          Used to validate IS (it does not rely on the Laplace mode, so it sees other modes).
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from stable_noise_gp import StableNoiseGP, logpdf_derivs
from surrogates import matern52

NU_Q = 4.0


def loglik_f(y, F, sig, a):
    """sum_i log p(y_i | f_i) for each row of F (S x n)."""
    lp = logpdf_derivs(((y[None, :] - F) / sig).ravel(), a)[0].reshape(F.shape)
    return lp.sum(1) - y.size * np.log(sig)


def logprior(F, Lk):
    n = Lk.shape[0]
    Z = np.linalg.solve(Lk, F.T)
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(Lk)).sum() - 0.5 * n * np.log(2 * np.pi)


class ExactM2:
    def __init__(self, x, y, S=4000, seed=0, standardise="sd"):
        """standardise='sd' (as in the stored Table 3 runs) or 'robust' (median and normalised MAD; phase C)."""
        self.x = np.asarray(x, float)
        if standardise == "robust":
            m = np.median(y); s = 1.4826 * np.median(np.abs(y - m)); self.ym, self.ys = m, (s if s > 0 else y.std())
        else:
            self.ym, self.ys = y.mean(), y.std()
        self.y = (y - self.ym) / self.ys
        n = len(y); r = np.random.default_rng(seed)
        self.Z = r.standard_normal((S, n)); self.C = r.chisquare(NU_Q, S)
        self.m = StableNoiseGP()

    def K(self, sf, ell):
        return matern52(self.x, self.x, sf, ell) + 1e-8 * np.eye(len(self.y))

    def laplace(self, sf, ell, sig, a):
        K = self.K(sf, ell)
        f, lml, g, W, w = self.m._laplace(K, self.y, sig, a)
        return K, f, lml, w

    def is_logml(self, sf, ell, sig, a, return_ess=False):
        K, f, lml, w = self.laplace(sf, ell, sig, a)
        n = len(self.y)
        Kinv = np.linalg.inv(K)
        Sig = np.linalg.inv(Kinv + np.diag(w)); Sig = (Sig + Sig.T) / 2
        Ls = np.linalg.cholesky(Sig)
        scale = np.sqrt(NU_Q / self.C)[:, None]
        F = f[None, :] + (self.Z @ Ls.T) * scale
        # log density of multivariate t_nu(f, Sig)
        q2 = ((self.Z * scale) ** 2).sum(1)
        logq = (gammaln((NU_Q + n) / 2) - gammaln(NU_Q / 2) - 0.5 * n * np.log(NU_Q * np.pi)
                - np.log(np.diag(Ls)).sum() - 0.5 * (NU_Q + n) * np.log1p(q2 / NU_Q))
        Lk = np.linalg.cholesky(K)
        lw = loglik_f(self.y, F, sig, a) + logprior(F, Lk) - logq
        est = logsumexp(lw) - np.log(len(lw))
        if return_ess:
            wn = np.exp(lw - lw.max()); ess = wn.sum() ** 2 / (wn ** 2).sum()
            return est, lml, ess
        return est

    def vb(self, sf, ell, sig, a, iters=200, tol=1e-9):
        """Variational (tangent-quadratic) lower bound on log p(y).  For any normal scale mixture,
        u -> log p(sqrt u) is convex, so  log p(z) >= log p(xi) - w(xi)/2 (z^2 - xi^2),  w = psi(xi)/xi.
        Integrating the Gaussian lower bound over f is closed form; xi is updated by EM
        (xi_i^2 = E_q z_i^2), which increases the bound monotonically.  Exact at a = 2."""
        K = self.K(sf, ell); y = self.y; n = len(y)
        A = K + sig ** 2 * np.eye(n)
        m = K @ np.linalg.solve(A, y); v = np.diag(K - K @ np.linalg.solve(A, K))
        prev = -np.inf
        for _ in range(iters):
            xi = np.maximum(np.sqrt((y - m) ** 2 + np.maximum(v, 0)) / sig, 1e-6)
            lp, d1, _ = logpdf_derivs(xi, a)
            w = np.maximum(-d1 / xi, 1e-12)
            D = sig ** 2 / w
            A = K + np.diag(D); L = np.linalg.cholesky(A)
            al = np.linalg.solve(L.T, np.linalg.solve(L, y))
            logN = -0.5 * y @ al - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)
            bound = (lp - np.log(sig) + w * xi ** 2 / 2 + 0.5 * np.log(2 * np.pi * D)).sum() + logN
            B = np.linalg.solve(L, K)
            m = K @ al; v = np.diag(K) - (B ** 2).sum(0)
            if abs(bound - prev) < tol: break
            prev = bound
        S = K - B.T @ B
        return bound, m, (S + S.T) / 2

    def is_mix(self, sf, ell, sig, a, return_ess=False):
        """IS with a defensive mixture proposal (t_4 at the Laplace mode with EM covariance,
        t_4 at the VB posterior, t_4 at the VB posterior with 4x covariance; weights .4/.4/.2)
        and truncated weights (Ionides 2008: cap at mean * sqrt(S)).  Common random numbers."""
        K, f, lml, w = self.laplace(sf, ell, sig, a)
        vbb, mv, Sv = self.vb(sf, ell, sig, a)
        n = len(self.y); S = self.Z.shape[0]
        Sl = np.linalg.inv(np.linalg.inv(K) + np.diag(w)); Sl = (Sl + Sl.T) / 2
        comps = [(0.4, f, Sl), (0.4, mv, Sv + 1e-10 * np.eye(n)), (0.2, mv, 4 * Sv + 1e-10 * np.eye(n))]
        if not hasattr(self, "U"):
            self.U = np.random.default_rng(12345).uniform(size=S)
        cut = np.cumsum([c[0] for c in comps]); k = np.searchsorted(cut, self.U * cut[-1])
        Ls = [np.linalg.cholesky(c[2]) for c in comps]
        scale = np.sqrt(NU_Q / self.C)[:, None]
        F = np.empty((S, n))
        for j, (p, mu, _) in enumerate(comps):
            F[k == j] = mu + (self.Z[k == j] @ Ls[j].T) * scale[k == j]
        cst = gammaln((NU_Q + n) / 2) - gammaln(NU_Q / 2) - 0.5 * n * np.log(NU_Q * np.pi)
        lq = []
        for (p, mu, _), L in zip(comps, Ls):
            u = np.linalg.solve(L, (F - mu).T); q2 = (u ** 2).sum(0)
            lq.append(np.log(p) + cst - np.log(np.diag(L)).sum() - 0.5 * (NU_Q + n) * np.log1p(q2 / NU_Q))
        logq = logsumexp(np.array(lq), axis=0)
        lw = loglik_f(self.y, F, sig, a) + logprior(F, np.linalg.cholesky(K)) - logq
        raw = logsumexp(lw) - np.log(S)
        cap = raw + 0.5 * np.log(S)
        lt = np.minimum(lw, cap); est = logsumexp(lt) - np.log(S)
        if return_ess:
            wn = np.exp(lw - lw.max()); ess = wn.sum() ** 2 / (wn ** 2).sum()
            wt = np.exp(lt - lt.max()); se = float(wt.std() / (wt.mean() * np.sqrt(S)))   # delta-method s.e. of est
            return est, dict(laplace=lml, vb=vbb, raw=raw, ess=ess, se=se)
        return est

    def ais_logml(self, sf, ell, sig, a, chains=64, T=400, seed=1):
        """AIS with elliptical slice sampling; returns (estimate, s.e. of log estimate)."""
        r = np.random.default_rng(seed)
        K = self.K(sf, ell); Lk = np.linalg.cholesky(K); n = len(self.y)
        betas = np.r_[0, np.geomspace(1e-4, 1, T)]
        F = (r.standard_normal((chains, n)) @ Lk.T)
        ll = loglik_f(self.y, F, sig, a)
        lw = np.zeros(chains)
        for b0, b1 in zip(betas[:-1], betas[1:]):
            lw += (b1 - b0) * ll
            # one elliptical slice sweep at temperature b1, vectorised over chains
            nu = r.standard_normal((chains, n)) @ Lk.T
            logu = b1 * ll + np.log(r.uniform(size=chains))
            th = r.uniform(0, 2 * np.pi, chains); lo, hi = th - 2 * np.pi, th.copy()
            active = np.ones(chains, bool); Fn = F.copy(); lln = ll.copy()
            for _ in range(60):
                idx = np.where(active)[0]
                if idx.size == 0: break
                Fp = F[idx] * np.cos(th[idx])[:, None] + nu[idx] * np.sin(th[idx])[:, None]
                lp = loglik_f(self.y, Fp, sig, a)
                ok = b1 * lp > logu[idx]
                Fn[idx[ok]] = Fp[ok]; lln[idx[ok]] = lp[ok]; active[idx[ok]] = False
                bad = idx[~ok]
                neg = th[bad] < 0
                lo[bad[neg]] = th[bad[neg]]; hi[bad[~neg]] = th[bad[~neg]]
                th[bad] = r.uniform(lo[bad], hi[bad])
            F, ll = Fn, lln
        est = logsumexp(lw) - np.log(chains)
        wn = np.exp(lw - lw.max()); wn /= wn.sum()
        se = np.sqrt(((wn - 1 / chains) ** 2).sum()) if chains > 1 else np.nan  # rough relative s.e.
        return est, se


def ais_bridge(E, sf, ell, sig, a, chains=64, T=1000, infl=4.0, sweeps=1, seed=1):
    """AIS from a Gaussian-noise GP posterior (noise var s0^2 = infl * 2 sig^2) to the SaS posterior.
    pi_b(f) ∝ N(f;0,K) N(y;f,s0^2 I)^(1-b) p(y|f)^b.  The Gaussian part is N(m_b, S_b) (GP posterior with
    noise var s0^2/(1-b)); elliptical slice sampling handles the p(y|f)^b factor.  Independent of the
    Laplace mode, so it can find modes the IS proposal misses.  Returns (log Z estimate, ESS of AIS weights)."""
    r = np.random.default_rng(seed)
    y = E.y; n = len(y); K = E.K(sf, ell); s02 = infl * 2 * sig ** 2
    def gpost(b):
        if b >= 1: return np.zeros(n), np.linalg.cholesky(K)
        A = K + s02 / (1 - b) * np.eye(n); m = K @ np.linalg.solve(A, y)
        S = K - K @ np.linalg.solve(A, K); S = (S + S.T) / 2 + 1e-10 * np.eye(n)
        return m, np.linalg.cholesky(S)
    lgauss = lambda F: (-0.5 * ((y[None, :] - F) ** 2).sum(1) / s02 - 0.5 * n * np.log(2 * np.pi * s02))
    from scipy.stats import multivariate_normal as mvn
    logZ0 = mvn(np.zeros(n), K + s02 * np.eye(n)).logpdf(y)
    betas = np.linspace(0, 1, T + 1) ** 4
    m, L = gpost(0.0)
    F = m[None, :] + r.standard_normal((chains, n)) @ L.T
    ll = loglik_f(y, F, sig, a); lg = lgauss(F)
    lw = np.zeros(chains)
    for b0, b1 in zip(betas[:-1], betas[1:]):
        lw += (b1 - b0) * (ll - lg)
        m, L = gpost(b1)
        for _ in range(sweeps):
            nu = r.standard_normal((chains, n)) @ L.T
            G = F - m[None, :]
            logu = b1 * ll + np.log(r.uniform(size=chains))
            th = r.uniform(0, 2 * np.pi, chains); lo, hi = th - 2 * np.pi, th.copy()
            active = np.ones(chains, bool); Fn = F.copy(); lln = ll.copy()
            for _ in range(80):
                idx = np.where(active)[0]
                if idx.size == 0: break
                Fp = m[None, :] + G[idx] * np.cos(th[idx])[:, None] + nu[idx] * np.sin(th[idx])[:, None]
                lp = loglik_f(y, Fp, sig, a)
                ok = b1 * lp > logu[idx]
                Fn[idx[ok]] = Fp[ok]; lln[idx[ok]] = lp[ok]; active[idx[ok]] = False
                bad = idx[~ok]; neg = th[bad] < 0
                lo[bad[neg]] = th[bad[neg]]; hi[bad[~neg]] = th[bad[~neg]]
                th[bad] = r.uniform(lo[bad], hi[bad])
            F, ll = Fn, lln
        lg = lgauss(F)
    est = logZ0 + logsumexp(lw) - np.log(chains)
    wn = np.exp(lw - lw.max())
    return est, wn.sum() ** 2 / (wn ** 2).sum()


def fit_both(x, y, S=10000, seed=0, a_min=0.8, a_max=2.0):
    """ML over (log sf, log ell, log sig, a) for the Laplace and the exact (IS) objective.
    Laplace: Nelder-Mead from 3 starts.  Exact: Nelder-Mead from the Laplace optimum and from the
    Laplace optimum with alpha moved 0.35 lower (to catch a lower-alpha optimum)."""
    E = ExactM2(x, y, S=S, seed=seed); span = np.ptp(E.x); n = len(y)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (a_min, a_max)]
    lo, hi = np.array([l for l, _ in b]), np.array([h for _, h in b])
    un = lambda p: (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
    def negL(p):
        try: return -E.laplace(*un(p))[2]
        except np.linalg.LinAlgError: return 1e10
    def negE(p):
        try:
            v = E.is_logml(*un(p)); return -v if np.isfinite(v) else 1e10
        except np.linalg.LinAlgError: return 1e10
    def nm(obj, starts, maxfev):
        best = None
        for p0 in starts:
            rr = minimize(obj, np.clip(p0, lo, hi), method="Nelder-Mead", bounds=b,
                          options=dict(xatol=2e-3, fatol=1e-3, maxfev=maxfev, initial_simplex=None))
            if best is None or rr.fun < best.fun: best = rr
        return best
    st = [np.array([0.0, np.log(0.25 * span), np.log(0.1), a]) for a in (1.3, 1.7, 1.95)]
    rl = nm(negL, st, 1500)
    pl = rl.x
    re_ = nm(negE, [pl, np.r_[pl[:3], max(a_min + 0.1, pl[3] - 0.35)]], 500)
    pe = re_.x
    c = -n * np.log(E.ys)
    lm, _, ess = E.is_logml(*un(pe), return_ess=True)
    return dict(p_laplace=pl.tolist(), p_exact=pe.tolist(), a_laplace=float(pl[3]), a_exact=float(pe[3]),
                laplace_logml=float(-rl.fun + c), exact_logml=float(-re_.fun + c),
                exact_at_laplace=float(-negE(pl) + c), laplace_at_exact=float(-negL(pe) + c),
                ess_at_exact=float(ess), nfev_exact=int(re_.nfev))


def fit_three(x, y, S=6000, S_final=40000, seed=0, a_min=0.8, a_max=2.0, standardise="sd"):
    """ML for (log sf, log ell, log sig, alpha) under three objectives on the same data:
    Laplace, the VB lower bound, and the exact likelihood (mixture IS with CRN).
    Every optimum is then re-evaluated with an independent, larger IS run."""
    E = ExactM2(x, y, S=S, seed=seed, standardise=standardise); span = np.ptp(E.x); n = len(y); c = -n * np.log(E.ys)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (a_min, a_max)]
    lo, hi = np.array([l for l, _ in b]), np.array([h for _, h in b])
    un = lambda p: (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
    def safe(fn):
        def g(p):
            try:
                v = fn(p); return -v if np.isfinite(v) else 1e10
            except (np.linalg.LinAlgError, ValueError): return 1e10
        return g
    obj = {"laplace": safe(lambda p: E.laplace(*un(p))[2]),
           "vb": safe(lambda p: E.vb(*un(p))[0]),
           "exact": safe(lambda p: E.is_mix(*un(p)))}
    def nm(o, starts, maxfev):
        best = None
        for p0 in starts:
            rr = minimize(o, np.clip(p0, lo, hi), method="Nelder-Mead", bounds=b,
                          options=dict(xatol=2e-3, fatol=1e-3, maxfev=maxfev))
            if best is None or rr.fun < best.fun: best = rr
        return best
    st = [np.array([0.0, np.log(0.25 * span), np.log(0.1), a]) for a in (1.3, 1.7, 1.95)]
    res = {"laplace": nm(obj["laplace"], st, 1500)}
    res["vb"] = nm(obj["vb"], st + [res["laplace"].x], 1500)
    cand = [res["laplace"].x, res["vb"].x]
    s0 = min(cand, key=lambda p: obj["exact"](p))
    res["exact"] = nm(obj["exact"], [s0, np.r_[res["vb"].x[:3], max(a_min + 0.1, res["vb"].x[3] - 0.3)]], 400)
    F = ExactM2(x, y, S=S_final, seed=seed + 101, standardise=standardise)
    out = {}
    for k, rr in res.items():
        est, info = F.is_mix(*un(rr.x), return_ess=True)
        out[k] = dict(p=rr.x.tolist(), alpha=float(rr.x[3]), own=float(-rr.fun + c),
                      exact=float(est + c), exact_raw=float(info["raw"] + c), ess=float(info["ess"]),
                      laplace=float(info["laplace"] + c), vb=float(info["vb"] + c), se=info["se"])
    return out
