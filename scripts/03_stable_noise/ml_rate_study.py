"""Rate of the Laplace ML estimate of alpha in M2 for n = 120 ... 7680, started from the difference-ECF estimate.

Data (same generator as Table 1): x stratified on [0, 1.2], y = f(x) + 0.05 eps, eps iid SaS(alpha0),
rng = default_rng(int(1000 alpha0) + 10 n + rep); reps 0-9 at n = 120, 240 are the Table 1 data sets.
Fit: Laplace ML over (sf, ell, sigma, alpha) on standardised y with the O(n) state-space Laplace
(ssm_laplace.py, cubic interpolation of the density in alpha). Start: alpha_tilde (difference-ECF,
clipped to [0.85, 1.97]) and its scale estimate sigma_tilde; (sf, ell) first optimised with
(sigma, alpha) fixed at the start, then all four by Nelder-Mead.
Check of the start (reps 0-19): a second fit from the fixed start (sf, ell, sigma, alpha) = (1, span/4, 0.1, 1.95).
Usage: python ml_rate_study.py <alpha0> <n list> <rep list> <tag>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))   # result files are read and written here
import sys, os, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
HERE = os.getcwd()
from scipy.optimize import minimize
import ssm_laplace as S
from ecf_alpha import differences
from stable_noise_gp import rsas
f_true = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
C = (0.2, 1.0)


def data(a0, n, rep):
    r = np.random.default_rng(int(a0 * 1000) + 10 * n + rep)
    x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2)
    return x, f_true(x) + 0.05 * rsas(a0, n, r)


def ecf_start(y):
    """alpha_tilde (unclipped and clipped) and the matching scale estimate sigma_tilde."""
    D = differences(y, 1); s = np.median(np.abs(D)); t1, t2 = C[0] / s, C[1] / s
    p1, p2 = np.mean(np.cos(t1 * D)), np.mean(np.cos(t2 * D))
    a = np.log(np.log(p1) / np.log(p2)) / np.log(t1 / t2)
    ac = float(np.clip(a, 0.85, 1.97))
    sig = (-np.log(p2) / (2 * t2 ** ac)) ** (1 / ac)
    return float(a), ac, float(sig)


def fit(x, y, start):
    ys = (y - y.mean()) / y.std(); span = np.ptp(x)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (0.8, 2.0)]
    lo, hi = np.array([u for u, _ in b]), np.array([v for _, v in b])
    warm = [None]; nev = [0]
    def negL(p):
        p = np.clip(p, lo, hi); nev[0] += 1
        try:
            l, fh, av, sg = S.laplace(x, ys, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], warm=warm[0])
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e10
        if sg > 0 and np.isfinite(l): warm[0] = (fh, av); return -l
        return 1e10
    p0 = np.clip(np.array(start, float), lo, hi)
    r1 = minimize(lambda q: negL(np.r_[q, p0[2:]]), p0[:2], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxfev=300))
    p1 = np.r_[r1.x, p0[2:]]
    r = minimize(negL, p1, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5 + 1e-8 * len(y), maxfev=3000))
    p = np.clip(r.x, lo, hi)
    return dict(p=[float(v) for v in p], alpha=float(p[3]), sigma=float(np.exp(p[2]) * y.std()), ell=float(np.exp(p[1])),
                lml=float(-r.fun - len(y) * np.log(y.std())), nfev=nev[0], converged=bool(r.success))


if __name__ == "__main__":
    a0 = float(sys.argv[1]); ns = [int(v) for v in sys.argv[2].split(",")]
    reps = [int(v) for v in sys.argv[3].split(",")] if "-" not in sys.argv[3] else list(range(*[int(v) for v in sys.argv[3].split("-")]))
    fn = os.path.join(HERE, f"ml_rate_a{a0}_{sys.argv[4]}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else []
    have = {(o["n"], o["rep"]) for o in out}
    for n in ns:
        for rep in reps:
            if (n, rep) in have: continue
            x, y = data(a0, n, rep); t0 = time.time()
            a_raw, a_c, sig_t = ecf_start(y)
            ysd = y.std()
            o = dict(a0=a0, n=n, rep=rep, alpha_tilde=a_raw, alpha_tilde_clipped=float(min(a_raw, 2.0)), sigma_tilde=sig_t)
            o["ml"] = fit(x, y, [0.0, np.log(0.25 * np.ptp(x)), np.log(sig_t / ysd), a_c])
            if rep < 20:
                o["ml_fixed_start"] = fit(x, y, [0.0, np.log(0.25 * np.ptp(x)), np.log(0.1), 1.95])
            o["secs"] = time.time() - t0
            out.append(o)
            json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
            extra = ""
            if "ml_fixed_start" in o:
                extra = f" | fixed start: alpha {o['ml_fixed_start']['alpha']:.3f}, dlml {o['ml_fixed_start']['lml'] - o['ml']['lml']:+.3f}"
            print(f"a0={a0} n={n} rep={rep}: tilde {a_raw:.3f}  ML {o['ml']['alpha']:.3f} (sig {o['ml']['sigma']:.4f}, ell {o['ml']['ell']:.3f}, "
                  f"{o['ml']['nfev']} evals){extra}  {o['secs']:.0f}s", flush=True)
