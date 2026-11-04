"""One observation y = f + eps, f ~ N(0, v), eps ~ SaS(alpha = 1.3), sigma = 1, y = 10.
Folds (saddle-nodes) of the posterior mode satisfy l''(r) = 1/v and -l'(r) = (y - r)/v with r = y - f.
Near a fold the Laplace value at the vanishing mode diverges like -1/2 log H; the exact evidence is smooth."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import sys, json, numpy as np; from stable_noise_gp import logpdf_derivs
from scipy.optimize import brentq
a, y = 1.3, 10.0
L = lambda r: logpdf_derivs(np.atleast_1d(np.asarray(r, float)), a)
h = lambda r: -L(r)[1][0] - (y - r) * L(r)[2][0]
rs = np.linspace(1.75, y - 0.01, 4000); hv = np.array([h(r) for r in rs]); k = np.where(np.diff(np.sign(hv)))[0]
folds = [brentq(h, rs[i], rs[i+1]) for i in k]
fg = np.linspace(-25, 35, 600001); dfg = fg[1]-fg[0]; lpl = L(y - fg)[0]
def exact(v):
    lp = lpl - fg**2/(2*v) - 0.5*np.log(2*np.pi*v); return float(np.log(np.sum(np.exp(lp - lp.max()))*dfg) + lp.max())
out = dict(alpha=a, y=y, folds=[])
for r_star in folds:
    v_star = 1 / L(r_star)[2][0]; f_star = y - r_star
    # side on which the mode near f_star exists: test v_star*(1 +/- 1e-3)
    rows = []
    for sgn in (+1, -1):
        for d in np.geomspace(1e-1, 1e-10, 20):
            v = v_star * (1 + sgn*d); sc = lambda f: -L(y - f)[1][0] - f / v
            g = np.linspace(f_star - 3*np.sqrt(d) - 1e-3, f_star + 3*np.sqrt(d) + 1e-3, 4001); s = np.array([sc(q) for q in g])
            roots = [brentq(sc, g[i], g[i+1]) for i in np.where(np.diff(np.sign(s)))[0]]
            for fm in roots:
                H = -L(y - fm)[2][0] + 1/v
                if H <= 0: continue
                lap = float(L(y - fm)[0][0] - fm**2/(2*v) - 0.5*np.log(2*np.pi*v) + 0.5*np.log(2*np.pi/H))
                rows.append(dict(side=sgn, delta=float(d), v=float(v), f_mode=float(fm), H=float(H), laplace=lap, exact=exact(v)))
    out["folds"].append(dict(r_star=float(r_star), v_star=float(v_star), f_star=float(f_star), rows=rows))
    print(f"fold at r* = {r_star:.4f} (mode f* = {f_star:.3f}), v* = {v_star:.4f}")
    for q in rows[::3]: print(f"   v = v*(1{'+' if q['side'] > 0 else '-'}{q['delta']:.0e}): H = {q['H']:.2e}, Laplace - exact = {q['laplace'] - q['exact']:6.2f}, -1/2 log H = {-0.5*np.log(q['H']):6.2f}")
json.dump(out, open("toy_exact.json", "w"), indent=1)
