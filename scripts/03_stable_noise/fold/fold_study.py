"""Fold study. mode 'small': Fisher-type refits of the 64 Section 4.3 data sets (n = 30, 60) with exact
check; mode 'large': standard and Fisher fits at n = 120, 240 on the Table 1 data sets."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import sys, os, json, time, glob, numpy as np, warnings
warnings.filterwarnings("ignore")
import fold_remedy as remedy, exact_m2 as X
from stable_noise_gp import rsas
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
mode = sys.argv[1]; fn = f"fold_{mode}{sys.argv[3] if len(sys.argv) > 3 else ''}.json"
out = json.load(open(fn)) if os.path.exists(fn) else []
done = {(o["a0"], o["n"], o["rep"]) for o in out}
def data(a0, n, rep):
    r = np.random.default_rng(int(a0*1000)+10*n+rep)
    x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); return x, f(x)+0.05*rsas(a0, n, r)
if mode == "small":
    jobs = [(o["a0"], o["n"], o["rep"], o) for fn2 in sorted(glob.glob("../exact3_a*_main.json")) for o in json.load(open(fn2))]
else:
    ns = [int(v) for v in sys.argv[2].split(",")]
    jobs = [(a0, n, rep, None) for n in ns for a0 in (1.3, 1.7) for rep in range(10)]
for a0, n, rep, o in jobs:
    if (a0, n, rep) in done: continue
    x, y = data(a0, n, rep); E = X.ExactM2(x, y, S=10); t = time.time()
    res = dict(a0=a0, n=n, rep=rep)
    kinds = ("fisher",) if mode == "small" else ("standard", "fisher")
    for k in kinds:
        p, val = remedy.fit(E, k)
        ev = remedy.evaluate(E, p); c = -n*np.log(E.ys)
        res[k] = dict(p=p.tolist(), alpha=float(p[3]), own=val, lam_min=ev["lam_min"], n_neg=ev["n_neg"],
                      standard_at=ev["standard"] + c, fisher_at=ev["fisher"] + c)
    if mode == "small":
        F = X.ExactM2(x, y, S=40000, seed=101)
        est, info = F.is_mix(*(np.exp(res["fisher"]["p"][0]), np.exp(res["fisher"]["p"][1]), np.exp(res["fisher"]["p"][2]), res["fisher"]["p"][3]), return_ess=True)
        res["fisher"]["exact"] = float(est - n*np.log(E.ys)); res["fisher"]["ess"] = float(info["ess"])
        res["laplace_alpha"] = o["laplace"]["alpha"]; res["exact_alpha"] = o["exact"]["alpha"]
        res["exact_opt"] = o["exact"]["exact"]; res["exact_at_laplace"] = o["laplace"]["exact"]
    res["secs"] = time.time() - t; out.append(res)
    json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    msg = " | ".join(f"{k}: a={res[k]['alpha']:.3f} lam={res[k]['lam_min']:.1e}" for k in kinds)
    print(f"a0={a0} n={n} rep={rep}: {msg}" + (f" | exact a={res['exact_alpha']:.3f}" if mode == "small" else "") + f" ({res['secs']:.0f}s)", flush=True)
print("done")
