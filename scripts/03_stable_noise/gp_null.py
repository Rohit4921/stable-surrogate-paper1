"""Max log ML at alpha = 2 (Gaussian noise; Laplace is exact there) for every data set of the exact study."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('stable_noise'))   # result files are read and written here
import json, glob, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from exact_m2 import ExactM2
from stable_noise_gp import rsas
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
out = {}
for fn in sorted(glob.glob("exact3_a*_main.json")):
    for o in json.load(open(fn)):
        a0, n, rep = o["a0"], o["n"], o["rep"]
        r = np.random.default_rng(int(a0*1000) + 10*n + rep)
        x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x) + 0.05*rsas(a0, n, r)
        E = ExactM2(x, y, S=10); span = np.ptp(E.x)
        b = [(np.log(0.1), np.log(10)), (np.log(0.02*span), np.log(3*span)), (np.log(1e-3), np.log(2.0))]
        obj = lambda p: -E.laplace(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), 2.0)[2]
        best = min((minimize(obj, p0, method="Nelder-Mead", bounds=b, options=dict(xatol=1e-4, fatol=1e-6, maxfev=3000))
                    for p0 in ([0, np.log(0.25*span), np.log(0.1)], [0, np.log(0.1*span), np.log(0.3)], o["exact"]["p"][:3])),
                   key=lambda rr: rr.fun)
        out[f"{a0}_{n}_{rep}"] = float(-best.fun - n*np.log(E.ys))
json.dump(out, open("gp_null.json", "w"), indent=0); print(len(out))
