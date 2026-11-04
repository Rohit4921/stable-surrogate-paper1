import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import fold_remedy as remedy, exact_m2 as X
from stable_noise_gp import rsas
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
D = {(o["a0"], o["n"], o["rep"]): o for fn in ("1.3_n30", "1.7_n30", "1.7_n60") for o in json.load(open(f"../exact3_a{fn}_main.json"))}
out = []
for a0, n, rep in ((1.3, 30, 1), (1.7, 30, 1), (1.7, 60, 2)):
    o = D[(a0, n, rep)]; r = np.random.default_rng(int(a0*1000)+10*n+rep)
    x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x)+0.05*rsas(a0, n, r)
    E = X.ExactM2(x, y, S=10); p, val = remedy.fit(E, "guarded")
    F = X.ExactM2(x, y, S=40000, seed=101); ex = F.is_mix(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]) - n*np.log(E.ys)
    ev = remedy.evaluate(E, p)
    row = dict(a0=a0, n=n, rep=rep, guarded_alpha=float(p[3]), guarded_own=val, guarded_exact=float(ex), lam=ev["lam_min"],
               laplace_alpha=o["laplace"]["alpha"], laplace_own=o["laplace"]["own"], laplace_exact=o["laplace"]["exact"],
               exact_alpha=o["exact"]["alpha"], exact_opt=o["exact"]["exact"])
    out.append(row); print(row, flush=True)
json.dump(out, open("guard_small.json", "w"), indent=1)
