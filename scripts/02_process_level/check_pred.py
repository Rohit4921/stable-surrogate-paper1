"""Numerical checks for Theorem (predictive) and the uniqueness of s-hat."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from surrogates import Surrogate
from scipy.special import logsumexp
from scipy import stats
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
r = np.random.default_rng(7); n = 25
x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); g = np.linspace(0, 1.2, 7)
print("(1) TP with fixed nu vs GP: kernel, noise ratio, predictive scale, dof")
for lab, y in (("gaussian", f(x)+0.05*r.standard_normal(n)), ("t1.5 noise", f(x)+0.05*r.standard_t(1.5, n))):
    gp = Surrogate("gp").fit(x, y, restarts=12)
    mg, sg, _ = gp.predict(g)
    for nu in (3.0, 5.0, 20.0):
        tp = Surrogate("tp", nu_fixed=nu).fit(x, y, restarts=12)
        mt, st, dt = tp.predict(g)
        print(f"  {lab:10s} nu={nu:4.0f}: ell TP/GP={tp.ell/gp.ell:.4f}  (sn/sf) TP/GP={(tp.sn/tp.sf)/(gp.sn/gp.sf):.4f}  "
              f"max|mu diff|={np.abs(mt-mg).max():.1e}  scale TP/GP: {np.round(st/sg,4).min()}-{np.round(st/sg,4).max()}  dof={dt:.0f}  "
              f"beta_TP={tp.beta:.3f} vs n(nu-2)/nu={n*(nu-2)/nu:.3f}")

print("\n(2) s -> M_n(alpha, s) for positive (alpha/2)-stable mixing: number of local maxima on a fine log-s grid")
NS = 400_000; rr = np.random.default_rng(0); U = rr.uniform(0, np.pi, NS); E = rr.exponential(size=NS)
ps = lambda a: np.sin(a*U)/np.sin(U)**(1/a) * (np.sin((1-a)*U)/E)**((1-a)/a)
for nn in (10, 25, 100):
    row = []
    for al in (0.6, 1.0, 1.3, 1.6, 1.9):
        lS = np.log(ps(al/2)); t = np.linspace(-15, 6, 841)
        m = np.array([logsumexp(-nn/2*(tt+lS) - 0.5*np.exp(-(tt+lS))) for tt in t])
        d = np.diff(m); nmax = int(((d[:-1] > 0) & (d[1:] <= 0)).sum())
        row.append(f"a={al}: {nmax} max at log s={t[m.argmax()]:.2f}")
    print(f"  n={nn:3d}  " + "; ".join(row))

print("\n(3) unimodality of the density of log S, S positive beta-stable (beta = alpha/2)")
for be in (0.3, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95):
    lS = np.log(ps(be)); h, e = np.histogram(lS, bins=np.linspace(np.quantile(lS, .001), np.quantile(lS, .999), 120))
    hs = np.convolve(h, np.ones(5)/5, "same")[3:-3]; d = np.diff(hs)
    print(f"  beta={be}: sign changes of smoothed histogram slope = {int((np.diff(np.sign(d[np.abs(d)>np.abs(d).max()*0.02]))!=0).sum())}")
