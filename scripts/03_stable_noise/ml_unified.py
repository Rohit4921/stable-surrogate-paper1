"""Phase C unified Laplace-ML study (Table 1, Figure 2(c), Laplace column of Table 2).

Replaces consistency.py (dense, 2 restarts, 10-15 reps) and the sd-standardised ml_rate_study.py runs with one
protocol that meets the assumptions of the Laplace-efficiency theorem (Theorem 28, (A7)):
  * data: as before, x stratified on [0, 1.2], y = f(x) + 0.05 eps, eps iid SaS(alpha0),
    rng = default_rng(int(1000 alpha0) + 10 n + rep)   (so reps 0-9 at n = 120, 240 are the old Table 1 data);
  * y centred by its median and scaled by its normalised MAD (robust_scale), not by mean/sd;
  * SaS working alpha in [0.8, 1.98]; the Gaussian model (alpha = 2) is fitted separately as the GP;
  * two starts for every data set: the difference-ECF start (alpha_tilde, sigma_tilde) and the fixed start
    (sf, ell, sigma, alpha) = (1, span/4, 0.1, 1.95); the fit with the larger Laplace log-likelihood is kept;
  * recorded: alpha_hat, sigma_hat, ell, Laplace lml, GP lml (LR = 2 (lml_SaS - lml_GP)), sign of
    det(I + K W) at the mode (fold if <= 0), ||f_hat - f||_n and max|f_hat - f| at the inputs.
Usage: python ml_unified.py <alpha0> <n list> <reps> <tag>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))
import sys, os, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
import ssm_laplace as S
from ecf_alpha import differences
from stable_noise_gp import rsas
from crossfit_alpha import robust_scale
f_true = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
C = (0.2, 1.0)
A_HI = 1.98


def data(a0, n, rep):
    r = np.random.default_rng(int(a0 * 1000) + 10 * n + rep)
    x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2)
    return x, f_true(x) + 0.05 * rsas(a0, n, r)


def ecf_start(y):
    D = differences(y, 1); s = np.median(np.abs(D)); t1, t2 = C[0] / s, C[1] / s
    p1, p2 = np.mean(np.cos(t1 * D)), np.mean(np.cos(t2 * D))
    a = np.log(np.log(p1) / np.log(p2)) / np.log(t1 / t2)
    ac = float(np.clip(a, 0.85, 1.97)) if np.isfinite(a) else 1.5
    sig = (-np.log(p2) / (2 * t2 ** ac)) ** (1 / ac)
    return float(a), ac, float(sig)


def fit(x, y, start, gaussian=False):
    """Laplace ML on robustly standardised y. gaussian=True fits the GP (alpha fixed at 2; Laplace is exact)."""
    m, sc = robust_scale(y); ys = (y - m) / sc; span = np.ptp(x)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0)), (0.8, A_HI)]
    k = 3 if gaussian else 4
    lo, hi = np.array([u for u, _ in b])[:k], np.array([v for _, v in b])[:k]
    warm = [None]; nev = [0]
    def negL(p):
        p = np.clip(p, lo, hi); nev[0] += 1; a = 2.0 if gaussian else p[3]
        try:
            l, fh, av, sg = S.laplace(x, ys, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), a, warm=warm[0])
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e10
        if sg > 0 and np.isfinite(l): warm[0] = (fh, av); return -l
        return 1e10
    p0 = np.clip(np.array(start, float)[:k], lo, hi)
    r1 = minimize(lambda q: negL(np.r_[q, p0[2:]]), p0[:2], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxfev=300))
    r = minimize(negL, np.r_[r1.x, p0[2:]], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5 + 1e-8 * len(y), maxfev=3000))
    p = np.clip(r.x, lo, hi); a = 2.0 if gaussian else float(p[3])
    _, fm, _, sg = S.laplace(x, ys, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), a)
    D = m + sc * fm - f_true(x)
    return dict(p=[float(v) for v in p], alpha=a, sigma=float(np.exp(p[2]) * sc), ell=float(np.exp(p[1])),
                lml=float(-r.fun - len(y) * np.log(sc)), nfev=nev[0], converged=bool(r.success), sign=int(sg),
                dn=float(np.sqrt(np.mean(D ** 2))), dinf=float(np.abs(D).max()))


if __name__ == "__main__":
    a0 = float(sys.argv[1]); ns = [int(v) for v in sys.argv[2].split(",")]; REPS = int(sys.argv[3])
    fn = f"ml_unified_a{a0}_{sys.argv[4]}.json"
    out = json.load(open(fn)) if os.path.exists(fn) else []
    have = {(o["n"], o["rep"]) for o in out}
    for n in ns:
        for rep in range(REPS):
            if (n, rep) in have: continue
            x, y = data(a0, n, rep); t0 = time.time(); m, sc = robust_scale(y); span = np.ptp(x)
            a_raw, a_c, sig_t = ecf_start(y)
            A = fit(x, y, [0.0, np.log(0.25 * span), np.log(sig_t / sc), a_c])
            B = fit(x, y, [0.0, np.log(0.25 * span), np.log(0.1), 1.95])
            best = A if A["lml"] >= B["lml"] else B
            G = fit(x, y, [0.0, np.log(0.25 * span), np.log(sig_t / sc)], gaussian=True)
            o = dict(a0=a0, n=n, rep=rep, alpha_tilde=a_raw, ml=best, start_ecf=A, start_fixed=dict(alpha=B["alpha"], lml=B["lml"]),
                     gp=dict(lml=G["lml"], sigma=G["sigma"], ell=G["ell"], dn=G["dn"]), LR=float(2 * (best["lml"] - G["lml"])),
                     secs=time.time() - t0)
            out.append(o); json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
            print(f"a0={a0} n={n} rep={rep}: tilde {a_raw:.3f} ML {best['alpha']:.3f} (starts {A['alpha']:.3f}/{B['alpha']:.3f}, "
                  f"dlml {B['lml'] - A['lml']:+.2f}) LR {o['LR']:.2f} dn {best['dn']:.4f} sign {best['sign']} {o['secs']:.0f}s", flush=True)
    print("done")
