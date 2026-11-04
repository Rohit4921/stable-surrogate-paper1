"""
Phase C repeated sequential BO study (replaces bo_study.py's 20-repetition runs).

Surrogates (each runs its own BO loop from the same initial design, with common random numbers for the noise):
  GP     Gaussian noise, ML hyperparameters (mean/sd standardisation)
  TP     Student-t process (mean/sd standardisation)
  GP-t   GP with Student-t noise, Laplace ML (median/MAD standardisation)
  RCGP   RCGP-type robust conjugate GP (surrogates.RCGP; median/MAD standardisation)
  SaS    GP with SaS noise: Laplace ML (median/MAD standardisation, alpha in [0.8, 2], alpha = 2 being the GP),
         warm start + 1 random start, then the IS safeguard (refit on the IS likelihood if Laplace exceeds it by
         more than 1 nat). In 1-D the Laplace objective uses the O(n) state-space code.
Problems (maximisation; f* known):
  gauss     f(x) = x sin(10x) + cos(2x) on [0, 1.2], noise 0.05 N(0,1)
  outliers  same f, 0.05 N(0,1) plus, w.p. 0.15, a gross error +-U(0.5, 1.0)
  sas13     same f, 0.05 SaS(1.3)
  nifty     NIFTY 50 moving-average look-back (as in bo_study.py)
  t2        same f, 0.05 t_2 noise                         (heavy-tailed, symmetric, not stable: misspecified for SaS)
  skewout   same f, 0.05 N(0,1) plus, w.p. 0.15, +U(0.5, 1.0) (one-sided outliers: asymmetric, misspecified)
  branin2   2-D Branin on the unit square, negated and standardised (sd 1 under uniform inputs), noise 0.1 SaS(1.3)
  hart6     6-D Hartmann, negated and standardised, noise 0.1 SaS(1.3)
  gardner1  constrained 2-D problem 1 of Gardner et al. (2014) on [0, 6]^2: minimise cos(2 x1) cos(x2) + sin(x1)
            subject to cos(x1) cos(x2) - sin(x1) sin(x2) <= 0.5; objective negated and standardised (mean and sd
            under uniform inputs), noise 0.1 SaS(1.3) on the objective, constraint observed without noise
  gardner2  constrained 2-D problem 2 of Gardner et al. (2014) on [0, 6]^2: minimise sin(x1) + x2 subject to
            sin(x1) sin(x2) <= -0.95; as gardner1 (feasible set about 1.8% of the domain)
  (The Gardner definitions are as commonly used in the constrained-BO literature; check against the paper.)
Constrained problems: the constraint is modelled by a separate GP (Gaussian likelihood, ML hyperparameters, the same
for all surrogates) and the acquisition is constrained EI, EI(x) * P(c(x) <= 0) (Gardner et al. 2014), with the
incumbent the largest posterior mean of f at the feasible data; before any feasible point is found the acquisition
is P(c(x) <= 0). Recommendation: the evaluated feasible point with the largest posterior mean of f (the constraint is
observed exactly); before any feasible point has been evaluated, the regret is that of the worst feasible point.
Budgets: 1-D n0 = 8 + 22 EI steps; branin2, gardner1, gardner2 n0 = 10 + 30; hart6 n0 = 20 + 40.
Acquisition: EI on the latent f (Student-t predictive for the TP). 1-D: 401-point grid. d > 1: 1,024 scrambled-Sobol
points plus 256 Gaussian perturbations (sd 0.05) of the 4 data points with the largest posterior mean, redrawn at
every step from a seeded generator. Recommendation: maximiser of the posterior mean over the candidates and the data;
simple regret f* - f(recommendation).
Seeds: initial design default_rng(1000 rep + 7), noise default_rng(1000 rep + 11), candidates default_rng(1000 rep + 13 + it).
Usage: python bo_study_c4.py <problem> <rep range a-b> <tag> [kinds]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from surrogates import Surrogate, TLikGP, RCGP, ei_t, matern52, robust_scale
from stable_noise_gp import StableNoiseGP, rsas
from exact_m2 import ExactM2
import ssm_laplace as SSM
import nifty_objective as no
no.WIN = 504
RESTARTS = 4
f_syn = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
_HA = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_HP = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_HAL = np.array([1.0, 1.2, 3.0, 3.2])
hart6 = lambda X: (_HAL * np.exp(-((np.atleast_2d(X)[:, None, :] - _HP[None]) ** 2 * _HA[None]).sum(-1))).sum(-1)   # = -Hartmann6


def gardner(X, k):
    """Objective (to be minimised) and constraint value minus threshold (feasible iff <= 0); X in [0, 1]^2."""
    X = np.atleast_2d(X); x1, x2 = 6 * X[:, 0], 6 * X[:, 1]
    if k == 1: return np.cos(2 * x1) * np.cos(x2) + np.sin(x1), np.cos(x1) * np.cos(x2) - np.sin(x1) * np.sin(x2) - 0.5
    return np.sin(x1) + x2, np.sin(x1) * np.sin(x2) + 0.95


# mean and sd of the objective under uniform inputs, its constrained minimum, and the worst feasible value
# (3001^2 grid; minimum refined by SLSQP): see the docstring
_GC = {1: dict(mu=0.0086416, sd=0.8668951, fmin=-2.0, fworst=1.9999983),
       2: dict(mu=3.0065895, sd=1.8772727, fmin=0.2532359, fworst=6.0279997)}


def branin_neg(X):
    X = np.atleast_2d(X); x1 = -5 + 15 * X[:, 0]; x2 = 15 * X[:, 1]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return -((x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10)


class Problem:
    def __init__(self, name):
        self.name = name; self.d = 1; self.constrained = False
        if name == "nifty":
            self.lo, self.hi = np.log(5), np.log(250); self.grid = np.linspace(self.lo, self.hi, 401)
            self.Ls = np.arange(5, 251); self.fL = np.array([no.true_objective(L) for L in self.Ls]); self.fstar = self.fL.max()
            self.n0, self.nit = 8, 22
        elif name == "branin2":
            self.d, self.n0, self.nit = 2, 10, 30
            self.mu, self.sd = -54.2713, 51.2086; self.fstar = (-0.397887 - self.mu) / self.sd
        elif name in ("gardner1", "gardner2"):
            self.d, self.n0, self.nit, self.constrained = 2, 10, 30, True; self.k = int(name[-1]); g = _GC[self.k]
            self.mu, self.sd = g["mu"], g["sd"]; self.fstar = -(g["fmin"] - g["mu"]) / g["sd"]
            self.rworst = (g["fworst"] - g["fmin"]) / g["sd"]
        elif name == "hart6":
            self.d, self.n0, self.nit = 6, 20, 40
            self.mu, self.sd = 0.25746, 0.38298; self.fstar = (3.32237 - self.mu) / self.sd
        else:
            self.lo, self.hi = 0.0, 1.2; self.grid = np.linspace(0, 1.2, 401)
            self.fstar = f_syn(np.linspace(0, 1.2, 20001)).max(); self.n0, self.nit = 8, 22

    def f(self, x):
        if self.name == "nifty": return float(self.fL[int(np.clip(round(np.exp(x)), 5, 250)) - 5])
        if self.name == "branin2": return float((branin_neg(x)[0] - self.mu) / self.sd)
        if self.name == "hart6": return float((hart6(x)[0] - self.mu) / self.sd)
        if self.constrained: return float(-(gardner(x, self.k)[0][0] - self.mu) / self.sd)
        return float(f_syn(x))

    def observe(self, x, rng):
        """One noisy evaluation; rng advances by the same number of draws whatever x is (CRN)."""
        if self.name == "nifty": return float(no.evaluate(np.exp(x), rng))
        e = rng.standard_normal(); u, s, m = rng.uniform(), rng.choice([-1.0, 1.0]), rng.uniform(0.5, 1.0)
        a = rsas(1.3, 1, rng)[0]; t2 = rng.standard_t(2)
        fx = self.f(x)
        if self.name == "gauss": return fx + 0.05 * e
        if self.name == "outliers": return fx + 0.05 * e + (s * m if u < 0.15 else 0.0)
        if self.name == "sas13": return fx + 0.05 * a
        if self.name == "t2": return fx + 0.05 * t2
        if self.name == "skewout": return fx + 0.05 * e + (m if u < 0.15 else 0.0)
        if self.name in ("branin2", "hart6") or self.constrained: return fx + 0.1 * a
        raise ValueError(self.name)

    def initial(self, rng):
        if self.d == 1:
            return self.lo + (np.arange(self.n0) + rng.uniform(size=self.n0)) / self.n0 * (self.hi - self.lo)
        return qmc.LatinHypercube(d=self.d, seed=rng).random(self.n0)

    def cons(self, x):
        return float(gardner(x, self.k)[1][0])

    def candidates(self, model, x, rng, feas=None):
        if self.d == 1: return self.grid
        C = qmc.Sobol(d=self.d, scramble=True, seed=rng).random(1024)
        mux = model.predict(x)[0]
        if feas is not None and feas.any(): mux = np.where(feas, mux, -np.inf)
        top = x[np.argsort(mux)[-4:]]
        L = np.clip(np.repeat(top, 64, axis=0) + 0.05 * rng.standard_normal((256, self.d)), 0, 1)
        return np.vstack([C, L, x])


class SaSSurrogate:
    """GP with SaS noise: Laplace ML (warm start + random start) and the IS safeguard; median/MAD standardisation."""
    def __init__(self):
        self.p = None; self.log = []

    def fit(self, x, y, seed):
        x = np.asarray(x, float); d1 = x.ndim == 1
        E = ExactM2(x, y, S=3000, seed=seed, standardise="robust"); span = np.ptp(E.x); n = len(y)
        b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (0.8, 2.0)]
        lo, hi = np.array([l for l, _ in b]), np.array([h for _, h in b])
        un = lambda p: (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
        def safe(fn):
            def g(p):
                try:
                    v = fn(p); return -v if np.isfinite(v) else 1e10
                except (np.linalg.LinAlgError, ValueError, FloatingPointError): return 1e10
            return g
        if d1:   # O(n) state-space Laplace on the sorted inputs
            o = np.argsort(x); xs_, ys_ = x[o], E.y[o]
            if np.any(np.diff(xs_) <= 0): xs_ = xs_ + 1e-9 * np.arange(n)
            def lapv(p):
                l, _, _, sg = SSM.laplace(xs_, ys_, *un(np.clip(p, lo, hi))); return l if sg > 0 else -np.inf
            negL = safe(lapv)
        else:
            negL = safe(lambda p: E.laplace(*un(p))[2])
        negX = safe(lambda p: E.is_mix(*un(p)))
        rng = np.random.default_rng(seed)
        starts = [np.array([rng.uniform(l, h) for l, h in b])]
        starts += [self.p] if self.p is not None else [np.array([0.0, np.log(0.25 * span), np.log(0.1), a]) for a in (1.3, 1.95)]
        best = None
        for p0 in starts:
            r = minimize(negL, np.clip(p0, lo, hi), method="Nelder-Mead", bounds=b, options=dict(xatol=2e-3, fatol=1e-3, maxfev=400))
            if best is None or r.fun < best.fun: best = r
        p = np.clip(best.x, lo, hi); lap = -best.fun; ex = -negX(p); refit = False
        if lap - ex > 1.0:
            r = minimize(negX, p, method="Nelder-Mead", bounds=b, options=dict(xatol=2e-3, fatol=1e-3, maxfev=150))
            if r.fun < negX(p): p = np.clip(r.x, lo, hi)
            refit = True
        self.p = p
        m = StableNoiseGP(); m.x = x; m.ym, m.ys = E.ym, E.ys; m.y = E.y
        m.sf, m.ell, m.sig, m.alpha = un(p)
        m.K = matern52(m.x, m.x, m.sf, m.ell) + 1e-8 * np.eye(n); m._fw = None
        m.f, _, m.g, m.W, m.w_em = m._laplace(m.K, m.y, m.sig, m.alpha)
        self.m = m; self.alpha = float(p[3]); self.log.append(dict(alpha=self.alpha, gap=float(lap - ex), refit=refit))
        return self

    def predict(self, xs):
        mu, s = self.m.predict(xs); return mu, s, np.inf


def fit_model(kind, x, y, state, seed):
    if kind == "GP": return Surrogate("gp").fit(x, y, RESTARTS, seed)
    if kind == "TP": return Surrogate("tp").fit(x, y, RESTARTS, seed)
    if kind == "GP-t": return TLikGP().fit(x, y, RESTARTS, seed, standardise="robust")
    if kind == "RCGP": return RCGP().fit(x, y, RESTARTS, seed)
    if kind == "SaS": return state.fit(x, y, seed)


def run(problem, kind, rep):
    P = Problem(problem)
    if os.environ.get("BO_NIT"): P.nit = int(os.environ["BO_NIT"])          # shorter runs for testing only
    x = P.initial(np.random.default_rng(1000 * rep + 7))
    noise = np.random.default_rng(1000 * rep + 11)
    y = np.array([P.observe(xi, noise) for xi in x])
    state = SaSSurrogate() if kind == "SaS" else None
    cv = np.array([P.cons(xi) for xi in x]) if P.constrained else None
    regret, tail = [], []; t0 = time.time()
    for it in range(P.nit + 1):
        m = fit_model(kind, x, y, state, seed=rep * 100 + it)
        feas = cv <= 0 if P.constrained else None
        C = P.candidates(m, x, np.random.default_rng(1000 * rep + 13 + it), feas)
        mu, s, d = m.predict(C); mux = m.predict(x)[0]
        if P.constrained:
            if feas.any():
                xr = x[np.flatnonzero(feas)[np.argmax(mux[feas])]]; regret.append(P.fstar - P.f(xr))
            else:
                regret.append(P.rworst)
        else:
            xr = C[np.argmax(mu)]; regret.append(P.fstar - P.f(xr))
        tail.append(float(m.alpha) if kind == "SaS" else (float(m.nu) if kind in ("TP", "GP-t") else np.nan))
        if it == P.nit: break
        if P.constrained:
            mc = Surrogate("gp").fit(x, cv, RESTARTS, rep * 100 + it + 50)
            mcu, scu, _ = mc.predict(C); pof = norm.cdf(-mcu / np.maximum(scu, 1e-12))
            e = ei_t(mu, s, d, mux[feas].max()) * pof if feas.any() else pof
        else:
            e = ei_t(mu, s, d, mux.max())
        xn = C[np.argmax(e)]
        x = np.r_[x, xn] if P.d == 1 else np.vstack([x, xn]); y = np.r_[y, P.observe(xn, noise)]
        if P.constrained: cv = np.r_[cv, P.cons(xn)]
    out = dict(problem=problem, kind=kind, rep=rep, regret=[float(v) for v in regret], tail=tail, secs=time.time() - t0)
    if P.constrained: out["n_feasible"] = int((cv <= 0).sum())
    if state is not None: out["sas_log"] = state.log
    return out


if __name__ == "__main__":
    problem, (a, b), tag = sys.argv[1], [int(v) for v in sys.argv[2].split("-")], sys.argv[3]
    kinds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["GP", "TP", "GP-t", "RCGP", "SaS"]
    fn = f"bo_{problem}_{tag}.json"
    done = json.load(open(fn)) if os.path.exists(fn) else []
    have = {(o["kind"], o["rep"]) for o in done}
    for rep in range(a, b):
        for kind in kinds:
            if (kind, rep) in have: continue
            o = run(problem, kind, rep); done.append(o)
            json.dump(done, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
            print(f"{problem} rep {rep} {kind:5s} final regret {o['regret'][-1]:.4f}  tail {o['tail'][-1]:.2f}  {o['secs']:.0f}s", flush=True)
