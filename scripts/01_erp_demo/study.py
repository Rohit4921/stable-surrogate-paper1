"""Operating characteristics of the selection rule (synthetic) and replicate study on NIFTY 50."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('erp_demo'))   # result files are read and written here
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
import nifty_objective as no
from procedure import run
no.WIN = 504
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
XSTAR, FSTAR = 0.175, None
gridS = np.linspace(0, 1.2, 401); FSTAR = f(gridS).max()

def record(R, x_to_f, grid, truth_grid):
    out = {"choice": R["choice"]}
    for k in ("GP", "TP", "GP-t"):
        p = R["pred"][k]
        out[k] = dict(nu=float(R["models"][k].nu) if k != "GP" else None,
                      f_next=float(x_to_f(p["x_next"])),
                      f_argmax_mean=float(x_to_f(grid[np.argmax(p["mu"])])),
                      rmse=float(np.sqrt(np.mean((p["mu"] - truth_grid)**2))))
    for k in ("TP", "GP-t"):
        c = R["cand"][k]
        out[k].update(logbf=float(c["logbf"]), dlpd=float(c["dlpd"]), a=bool(c["a"]), b=bool(c["b"]), c=bool(c["c"]))
    return out

which = sys.argv[1]
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
res = []
if which == "synthetic":
    for scen in ("clean", "contaminated", "t-noise"):
        for rep in range(30):
            rng = np.random.default_rng(SEED0 + rep); n = 25
            x = np.sort((np.arange(n)+rng.uniform(size=n))/n*1.2)
            y = f(x) + 0.05*rng.standard_normal(n)
            if scen == "contaminated":
                k = rng.choice(n, 4, replace=False)
                y[k] += rng.choice([-1, 1], 4) * rng.uniform(0.6, 1.0, 4)
            elif scen == "t-noise":
                y = f(x) + 0.05*rng.standard_t(1.5, n)
            R = run(x, y, gridS, restarts=6, seed=rep)
            r = record(R, f, gridS, f(gridS)); r.update(scen=scen, rep=rep); res.append(r)
            print(scen, rep, r["choice"], round(r["GP-t"]["nu"], 1), flush=True)
else:
    gridN = np.linspace(np.log(5), np.log(250), 401)
    Lg = np.round(np.exp(gridN)).astype(int)
    cache = {L: no.true_objective(L) for L in np.unique(Lg)}
    truth = np.array([cache[L] for L in Lg])
    x_to_f = lambda xv: cache[int(round(np.exp(xv)))] if int(round(np.exp(xv))) in cache else no.true_objective(int(round(np.exp(xv))))
    for n in (15, 30, 60):
        for rep in range(20):
            rng = np.random.default_rng(5000 + 100*n + rep)
            u = (np.arange(n)+rng.uniform(size=n))/n
            L = np.exp(np.log(5)+u*(np.log(250)-np.log(5)))
            x = np.log(L); y = np.array([no.evaluate(l, rng) for l in L])
            R = run(x, y, gridN, restarts=6, seed=rep)
            r = record(R, x_to_f, gridN, truth); r.update(n=n, rep=rep); res.append(r)
            print(n, rep, r["choice"], round(r["GP-t"]["nu"], 1), flush=True)
json.dump(res, open(f"results_{which}{TAG}.json", "w"), indent=1)
print("done")
