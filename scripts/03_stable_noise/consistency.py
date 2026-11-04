import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import numpy as np, json, sys, warnings, time
warnings.filterwarnings("ignore")
from stable_noise_gp import StableNoiseGP, rsas
a0 = float(sys.argv[1]); NS = [int(v) for v in sys.argv[2].split(",")]; REPS = int(sys.argv[3]); TAG = sys.argv[4]; f = lambda x: x*np.sin(10*x)+np.cos(2*x)
g = np.linspace(0, 1.2, 301); out = []
for n in NS:
    for rep in range(int(__import__("os").environ.get("START", 0)), REPS):
        r = np.random.default_rng(int(a0*1000) + 10*n + rep)
        x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x) + 0.05*rsas(a0, n, r)
        t = time.time(); m = StableNoiseGP().fit(x, y, restarts=2, seed=rep)
        mu, _ = m.predict(g)
        out.append(dict(a0=a0, n=n, rep=rep, a_hat=float(m.alpha), sig=float(m.sig*m.ys), ell=float(m.ell),
                        rmse=float(np.sqrt(np.mean((mu-f(g))**2))), secs=time.time()-t))
        print(n, rep, round(m.alpha, 3), flush=True)
        json.dump(out, open(f"consistency_a{a0}_{TAG}.json", "w"))
print("done")
