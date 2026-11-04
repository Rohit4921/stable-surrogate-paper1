"""Phase C rerun of the Laplace / VB / IS comparison (Table 3, Figure 3) under the protocol of Theorem 28:
median/MAD standardisation and SaS working alpha in [0.8, 1.98]; the Gaussian model is fitted separately (GP null).
Adds Monte Carlo diagnostics of the IS likelihood at every optimum:
  * se:        delta-method standard error of the truncated IS estimate (S_final = 40,000 draws);
  * exact_big: an independent IS estimate with 4x the draws (160,000, new seed), for S-sensitivity;
  * ais:       AIS from the Gaussian-noise GP posterior (ais_bridge, 64 chains, 1,000 temperatures), which does
               not use the Laplace mode, at the IS and Laplace optima.
Data: as exact_study.py (rng = default_rng(int(1000 alpha0) + 10 n + rep)).
Usage: python exact_c2_study.py <alpha0> <n> <rep list a-b> <tag>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
import exact_m2 as X
from stable_noise_gp import rsas
f = lambda x: x * np.sin(10 * x) + np.cos(2 * x)
A_HI = 1.98
a0 = float(sys.argv[1]); n = int(sys.argv[2]); lo_, hi_ = [int(v) for v in sys.argv[3].split("-")]; tag = sys.argv[4]
fn = f"exact_c2_a{a0}_n{n}_{tag}.json"
out = json.load(open(fn)) if os.path.exists(fn) else []
done = {o["rep"] for o in out}
un = lambda p: (np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3])
for rep in range(lo_, hi_):
    if rep in done: continue
    r = np.random.default_rng(int(a0 * 1000) + 10 * n + rep)
    x = np.sort((np.arange(n) + r.uniform(size=n)) / n * 1.2); y = f(x) + 0.05 * rsas(a0, n, r)
    t = time.time()
    res = X.fit_three(x, y, a_max=A_HI, standardise="robust")
    E = X.ExactM2(x, y, S=10, standardise="robust"); c = -n * np.log(E.ys); span = np.ptp(E.x)
    # GP null (alpha = 2, Gaussian noise: Laplace is exact)
    b = [(np.log(0.1), np.log(10)), (np.log(0.02 * span), np.log(3 * span)), (np.log(1e-3), np.log(2.0))]
    obj = lambda p: -E.laplace(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), 2.0)[2]
    best = min((minimize(obj, p0, method="Nelder-Mead", bounds=b, options=dict(xatol=1e-4, fatol=1e-6, maxfev=3000))
                for p0 in ([0, np.log(0.25 * span), np.log(0.1)], [0, np.log(0.1 * span), np.log(0.3)], res["exact"]["p"][:3])),
               key=lambda rr: rr.fun)
    res["gp_null"] = float(-best.fun + c)
    # S-sensitivity: independent IS with 4x draws
    B = X.ExactM2(x, y, S=160000, seed=303, standardise="robust")
    for k in ("laplace", "vb", "exact"):
        est, info = B.is_mix(*un(res[k]["p"]), return_ess=True)
        res[k]["exact_big"] = float(est + c); res[k]["se_big"] = info["se"]
    del B
    # AIS (independent of the Laplace mode) at the IS and Laplace optima
    for k in ("exact", "laplace"):
        try:
            v, ess = X.ais_bridge(E, *un(res[k]["p"]), chains=64, T=1000, seed=1)
            res[k]["ais"] = float(v + c); res[k]["ais_ess"] = float(ess)
        except np.linalg.LinAlgError:
            res[k]["ais"] = None
    res.update(a0=a0, n=n, rep=rep, secs=time.time() - t)
    out.append(res); json.dump(out, open(fn + ".tmp", "w"), indent=0); os.replace(fn + ".tmp", fn)
    L, V, Xx = res["laplace"], res["vb"], res["exact"]
    print(f"{n} {rep}: alpha L {L['alpha']:.3f} VB {V['alpha']:.3f} IS {Xx['alpha']:.3f} | IS at IS-opt {Xx['exact']:.2f} "
          f"(se {Xx['se']:.3f}, 4S {Xx['exact_big']:.2f}, AIS {Xx['ais']}) | Laplace gap {L['own'] - L['exact']:+.2f} | "
          f"LR(IS) {2 * (Xx['own'] - res['gp_null']):.2f} | {res['secs']:.0f}s", flush=True)
print("done")
