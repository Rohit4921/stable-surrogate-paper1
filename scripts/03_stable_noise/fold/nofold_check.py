"""c_alpha = sup_r l''_alpha(r) (convex-tail curvature of log p_alpha) and the no-fold ratio
rho = (c_alpha/sigma^2) * lambda_max(Sigma_NN), Sigma = (K^{-1} + W_+)^{-1}, N = {i : W_i < 0}. rho < 1 => no fold."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import sys, json, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
from stable_noise_gp import logpdf_derivs, lik, rsas
import exact_m2 as X, fold_remedy as remedy
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
r = np.linspace(0, 60, 600001)
c = {}
for a in np.round(np.arange(0.8, 2.0, 0.1), 2).tolist() + [1.95, 1.99]:
    d2 = logpdf_derivs(r, a)[2]; c[a] = float(d2.max())
print("c_alpha:", {k: round(v, 4) for k, v in c.items()})
def rho_at(x, y, p):
    E = X.ExactM2(x, y, S=10); sf, ell, sig, a = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]
    K, fm, _, _ = E.laplace(sf, ell, sig, a); _, g, W, _ = lik(E.y, fm, sig, a)
    N = np.where(W < 0)[0]
    if len(N) == 0: return 0.0, remedy.evaluate(E, p)["lam_min"]
    S = np.linalg.inv(np.linalg.inv(K) + np.diag(np.maximum(W, 0)))
    ca = float(logpdf_derivs(r, a)[2].max())
    return float(ca / sig**2 * np.linalg.eigvalsh(S[np.ix_(N, N)]).max()), remedy.evaluate(E, p)["lam_min"]
rows = []
for fn in sorted(glob.glob("../exact3_a*_main.json")):
    for o in json.load(open(fn)):
        a0, n, rep = o["a0"], o["n"], o["rep"]; g = np.random.default_rng(int(a0*1000)+10*n+rep)
        x = np.sort((np.arange(n)+g.uniform(size=n))/n*1.2); y = f(x)+0.05*rsas(a0, n, g)
        rho, lam = rho_at(x, y, o["laplace"]["p"])
        rows.append(dict(a0=a0, n=n, rep=rep, rho=rho, lam=lam, gap=o["laplace"]["own"]-o["laplace"]["exact"],
                         machine_dependent=bool(lam < remedy.LAM_MACHINE)))   # at a fold the mode (hence rho) is not portable
for fn in ("fold_large_120.json", "fold_large_240.json"):
    try: L = json.load(open(fn))
    except FileNotFoundError: continue
    for o in L:
        a0, n, rep = o["a0"], o["n"], o["rep"]; g = np.random.default_rng(int(a0*1000)+10*n+rep)
        x = np.sort((np.arange(n)+g.uniform(size=n))/n*1.2); y = f(x)+0.05*rsas(a0, n, g)
        rho, lam = rho_at(x, y, o["standard"]["p"]); rows.append(dict(a0=a0, n=n, rep=rep, rho=rho, lam=lam, gap=None))
json.dump(dict(c_alpha={str(k): v for k, v in c.items()}, rows=rows), open("nofold_check.json", "w"), indent=1)
for n in (30, 60, 120, 240):
    R = [q for q in rows if q["n"] == n]
    if not R: continue
    rh = np.array([q["rho"] for q in R]); lm = np.array([q["lam"] for q in R])
    print(f"n={n}: fits {len(R)}, rho>=1: {int((rh >= 1).sum())}, lam<1e-3: {int((lm < 1e-3).sum())}, median rho {np.median(rh):.3f}, max rho among non-folds {rh[lm >= 1e-3].max():.3f}")
print("folds:", [(q["a0"], q["n"], q["rep"], round(q["rho"], 3)) for q in rows if q["lam"] < 1e-3])
