"""Fit the GP with SaS noise (M2) to the two demonstration data sets of the 23 Sep note
(synthetic with 4 outliers; NIFTY 50, n = 30) and compare next query / recommendation with GP and GP-t."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('bo'))   # result files are read and written here
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from exact_m2 import fit_three
from stable_noise_gp import StableNoiseGP
from surrogates import matern52
import nifty_objective as no
no.WIN = 504
f = lambda x: x*np.sin(10*x)+np.cos(2*x)

def model_at(x, y, p):
    m = StableNoiseGP(); m.x = np.asarray(x, float); m.ym, m.ys = y.mean(), y.std(); m.y = (y - m.ym)/m.ys
    m.sf, m.ell, m.sig, m.alpha = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]
    n = len(y); m.K = matern52(m.x, m.x, m.sf, m.ell) + 1e-8*np.eye(n); m._fw = None
    m.f, _, m.g, m.W, m.w_em = m._laplace(m.K, m.y, m.sig, m.alpha)
    return m

def ei(mu, s, best):
    z = (mu - best)/s; return s*(z*stats.norm.cdf(z) + stats.norm.pdf(z))

out = {}
# synthetic data exactly as in demo/figures.py
rng = np.random.default_rng(7); n = 25
x = np.sort((np.arange(n)+rng.uniform(size=n))/n*1.2)
yA = f(x) + 0.05*rng.standard_normal(n); yB = yA.copy(); yB[[4, 11, 16, 21]] += [-0.8, 0.8, -0.9, 0.9]
g = np.linspace(0, 1.2, 401); fstar = f(g).max()
for lab, y in (("A", yA), ("B", yB)):
    r = fit_three(x, y)
    res = {}
    for k in ("laplace", "exact"):
        m = model_at(x, y, r[k]["p"]); mu, s = m.predict(g); best = m.predict(x)[0].max()
        e = ei(mu, s, best); xr = g[np.argmax(mu)]
        res[k] = dict(alpha=r[k]["alpha"], x_next=float(g[e.argmax()]), x_rec=float(xr), regret=float(fstar - f(xr)),
                      exact_logml=r[k]["exact"], own=r[k]["own"])
    out[lab] = res; print(lab, json.dumps(res, indent=0), flush=True)
# NIFTY data set exactly as in demo/figures.py (rngE draws come from a separate generator)
rngD = np.random.default_rng(2026); nN = 30
u = (np.arange(nN)+rngD.uniform(size=nN))/nN
L = np.exp(np.log(5)+u*(np.log(250)-np.log(5))); xN = np.log(L); yN = np.array([no.evaluate(l, rngD) for l in L])
gN = np.linspace(np.log(5), np.log(250), 401)
r = fit_three(xN, yN); res = {}
Lg = np.exp(gN)
for k in ("laplace", "exact"):
    m = model_at(xN, yN, r[k]["p"]); mu, s = m.predict(gN); best = m.predict(xN)[0].max()
    e = ei(mu, s, best); Lr = int(round(Lg[np.argmax(mu)]))
    res[k] = dict(alpha=r[k]["alpha"], L_next=float(Lg[e.argmax()]), L_rec=Lr, f_rec=float(no.true_objective(Lr)),
                  exact_logml=r[k]["exact"], own=r[k]["own"])
out["NIFTY"] = res; print("NIFTY", json.dumps(res, indent=0))
json.dump(out, open("bo_demo_stable.json", "w"), indent=1)
