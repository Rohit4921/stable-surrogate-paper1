import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings("ignore")
from exact_m2 import fit_three
from stable_noise_gp import rsas
a0 = float(sys.argv[1]); n = int(sys.argv[2]); reps = [int(v) for v in sys.argv[3].split(",")]; tag = sys.argv[4]
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
fn = f"exact3_a{a0}_n{n}_{tag}.json"
out = json.load(open(fn)) if os.path.exists(fn) else []
done = {o["rep"] for o in out}
for rep in reps:
    if rep in done: continue
    r = np.random.default_rng(int(a0*1000) + 10*n + rep)          # same data as consistency.py
    x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x) + 0.05*rsas(a0, n, r)
    t = time.time(); res = fit_three(x, y); res.update(a0=a0, n=n, rep=rep, secs=time.time()-t)
    out.append(res); json.dump(out, open(fn, "w"), indent=0)
    L, V, X = res["laplace"], res["vb"], res["exact"]
    print(f"{n} {rep}: alpha L {L['alpha']:.3f} VB {V['alpha']:.3f} X {X['alpha']:.3f} | exact logml at L {L['exact']:.2f} (Laplace says {L['own']:.2f}), "
          f"at VB {V['exact']:.2f}, at X {X['exact']:.2f} (ESS {X['ess']:.0f}) | {res['secs']:.0f}s", flush=True)
print("done")
