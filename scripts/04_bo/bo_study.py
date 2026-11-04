"""
Repeated sequential BO study: GP, Student-t process (TP), GP with Student-t noise (GP-t) and
GP with symmetric alpha-stable noise (SaS), each running its own BO loop from the same
initial design with common random numbers for the noise.

Problems (maximisation):
  gauss    f(x) = x sin(10x) + cos(2x) on [0, 1.2], noise 0.05 N(0,1)
  outliers same f, noise 0.05 N(0,1) plus, with prob. 0.15, a gross error +-U(0.5, 1.0)
  sas13    same f, noise 0.05 SaS(alpha = 1.3)
  nifty    NIFTY 50 moving-average look-back L in [5, 250] (x = log L); one evaluation is the
           annualised excess return over buy-and-hold on a random two-year window.
Budget: n0 = 8 stratified initial points + 22 EI iterations (n = 30).
EI is on the latent f for every surrogate (Gaussian latent posterior for GP, GP-t, SaS;
Student-t predictive for the TP); the incumbent is the largest posterior mean at the data.
Recommendation after each step: maximiser of the posterior mean on the grid; simple regret
f* - f(recommendation).
SaS fitting: Laplace ML by Nelder-Mead (warm start from the previous step + 1 random start),
then the safeguard of Section 4.4: one exact-likelihood evaluation (mixture IS, S = 3000) at the
Laplace optimum; if Laplace exceeds it by more than 1 nat, refit on the exact objective.
Usage: python bo_study.py <problem> <reps comma list> <tag>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))   # result files are read and written here
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from surrogates import Surrogate, TLikGP, ei_t, matern52
from stable_noise_gp import StableNoiseGP, rsas
from exact_m2 import ExactM2
import nifty_objective as no
no.WIN = 504

N0, NIT, RESTARTS = 8, int(os.environ.get('BO_NIT', 22)), 4   # BO_NIT: shorter runs for testing
f_syn = lambda x: x * np.sin(10 * x) + np.cos(2 * x)


class Problem:
    def __init__(self, name):
        self.name = name
        if name == "nifty":
            self.lo, self.hi = np.log(5), np.log(250)
            self.grid = np.linspace(self.lo, self.hi, 401)
            self.Ls = np.arange(5, 251); self.fL = np.array([no.true_objective(L) for L in self.Ls])
            self.fstar = self.fL.max()
        else:
            self.lo, self.hi = 0.0, 1.2
            self.grid = np.linspace(0, 1.2, 401); self.fstar = f_syn(np.linspace(0, 1.2, 20001)).max()

    def f(self, x):
        if self.name == "nifty":
            return float(self.fL[int(np.clip(round(np.exp(x)), 5, 250)) - 5])
        return float(f_syn(x))

    def observe(self, x, rng):
        """One noisy evaluation; rng is advanced by the same number of draws whatever x is (CRN)."""
        if self.name == "nifty":
            return float(no.evaluate(np.exp(x), rng))
        e = 0.05 * rng.standard_normal()
        u, s, m = rng.uniform(), rng.choice([-1.0, 1.0]), rng.uniform(0.5, 1.0)
        a = rsas(1.3, 1, rng)[0]
        if self.name == "gauss": return float(f_syn(x) + e)
        if self.name == "outliers": return float(f_syn(x) + e + (s * m if u < 0.15 else 0.0))
        if self.name == "sas13": return float(f_syn(x) + 0.05 * a)
        raise ValueError(self.name)


class SaSSurrogate:
    """GP with SaS noise: warm-started Laplace ML + exact-likelihood safeguard."""
    def __init__(self):
        self.p = None; self.log = []

    def fit(self, x, y, seed):
        E = ExactM2(x, y, S=3000, seed=seed); span = np.ptp(E.x); n = len(y)
        b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (0.8, 2.0)]
        lo, hi = np.array([l for l, _ in b]), np.array([h for _, h in b])
        un = lambda p: (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
        def safe(fn):
            def g(p):
                try:
                    v = fn(p); return -v if np.isfinite(v) else 1e10
                except (np.linalg.LinAlgError, ValueError): return 1e10
            return g
        negL = safe(lambda p: E.laplace(*un(p))[2]); negX = safe(lambda p: E.is_mix(*un(p)))
        rng = np.random.default_rng(seed)
        starts = [np.array([rng.uniform(l, h) for l, h in b])]
        starts += [self.p] if self.p is not None else [np.array([0.0, np.log(0.25 * span), np.log(0.1), a]) for a in (1.3, 1.95)]
        best = None
        for p0 in starts:
            r = minimize(negL, np.clip(p0, lo, hi), method="Nelder-Mead", bounds=b, options=dict(xatol=2e-3, fatol=1e-3, maxfev=400))
            if best is None or r.fun < best.fun: best = r
        p = best.x; lap = -best.fun; ex = -negX(p); refit = False
        if lap - ex > 1.0:
            r = minimize(negX, p, method="Nelder-Mead", bounds=b, options=dict(xatol=2e-3, fatol=1e-3, maxfev=150))
            if r.fun < negX(p): p = r.x
            refit = True
        self.p = p
        m = StableNoiseGP(); m.x = np.asarray(x, float); m.ym, m.ys = y.mean(), y.std(); m.y = (y - m.ym) / m.ys
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
    if kind == "GP-t": return TLikGP().fit(x, y, RESTARTS, seed)
    if kind == "SaS": return state.fit(x, y, seed)


def run(problem, kind, rep):
    P = Problem(problem)
    r0 = np.random.default_rng(1000 * rep + 7)
    x = P.lo + (np.arange(N0) + r0.uniform(size=N0)) / N0 * (P.hi - P.lo)
    noise = np.random.default_rng(1000 * rep + 11)
    y = np.array([P.observe(xi, noise) for xi in x])
    state = SaSSurrogate() if kind == "SaS" else None
    regret, tail, xs = [], [], []
    t0 = time.time()
    for it in range(NIT + 1):
        m = fit_model(kind, x, y, state, seed=rep * 100 + it)
        mu, s, d = m.predict(P.grid)
        mux = m.predict(x)[0]
        xr = P.grid[np.argmax(mu)]; regret.append(P.fstar - P.f(xr))
        tail.append(float(m.alpha) if kind == "SaS" else (float(m.nu) if kind in ("TP", "GP-t") else np.nan))
        if it == NIT: break
        e = ei_t(mu, s, d, mux.max())
        xn = P.grid[np.argmax(e)]
        x = np.r_[x, xn]; y = np.r_[y, P.observe(xn, noise)]; xs.append(float(xn))
    out = dict(problem=problem, kind=kind, rep=rep, regret=[float(v) for v in regret], tail=tail, queries=xs,
               secs=time.time() - t0)
    if state is not None: out["sas_log"] = state.log
    return out


if __name__ == "__main__":
    problem, reps, tag = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
    kinds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["GP", "TP", "GP-t", "SaS"]
    fn = f"bo_{problem}_{tag}.json"
    done = json.load(open(fn)) if os.path.exists(fn) else []
    have = {(o["kind"], o["rep"]) for o in done}
    for rep in reps:
        for kind in kinds:
            if (kind, rep) in have: continue
            o = run(problem, kind, rep); done.append(o)
            json.dump(done, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
            print(f"{problem} rep {rep} {kind:5s} final regret {o['regret'][-1]:.4f}  tail {o['tail'][-1]:.2f}  {o['secs']:.0f}s", flush=True)
