"""Predictive shape under M1 at the ML: xi = sqrt(n s_hat V~) Z, V~ ∝ g_n(s_hat v) dF(v).
Report the 97.5% quantile (GP: 1.960) and the EI shape e_G(z) = E(z+xi)^+ relative to the Gaussian at z = -1, -2."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
import numpy as np
from scipy.special import logsumexp
from scipy import stats
from scipy.optimize import brentq
xu, wu = np.polynomial.legendre.leggauss(300); U = (xu+1)*np.pi/2; WU = wu/2
w = np.linspace(-25, 4, 1500); dw = w[1]-w[0]; lwE = w - np.exp(w) + np.log(dw)
def eG(z, sd, p):   # E (z + sd Z)^+ mixed over sd with probs p
    return (p*(z*stats.norm.cdf(z/sd) + sd*stats.norm.pdf(z/sd))).sum()
rows = []
for n in (10, 25, 100):
    for al in (1.0, 1.3, 1.6, 1.9):
        b = al/2; gam = (1-b)/b
        lA = b/(1-b)*np.log(np.sin(b*U)) + np.log(np.sin((1-b)*U)) - np.log(np.sin(U))/(1-b)
        lS = (gam*(lA[:, None] - w[None, :])).ravel(); base = (np.log(WU)[:, None] + lwE[None, :]).ravel()
        t = np.linspace(-9, 1, 2001)
        M = np.array([logsumexp(base - n/2*(tt+lS) - 0.5*np.exp(-(tt+lS))) for tt in t]); ls = t[M.argmax()]
        lp = base - n/2*(ls+lS) - 0.5*np.exp(-(ls+lS)); p = np.exp(lp - logsumexp(lp))
        keep = p > 1e-14; p = p[keep]/p[keep].sum(); sd = np.sqrt(n*np.exp(ls + lS[keep]))
        q = brentq(lambda q: (p*stats.norm.cdf(q/sd)).sum() - 0.975, 0.5, 50)
        r1, r2 = eG(-1, sd, p)/eG(-1, 1, 1.0), eG(-2, sd, p)/eG(-2, 1, 1.0)
        rows.append((n, al, q, r1, r2))
        print(f"n={n:3d} alpha={al}: q97.5={q:.3f} (GP 1.960); EI shape ratio vs GP at z=-1: {r1:.3f}, z=-2: {r2:.3f}", flush=True)
for n in (10, 25, 100):
    for nu in (3, 5):
        d = nu+n; T = stats.t(d)
        eT = lambda z: z*T.cdf(z) + (d+z**2)/(d-1)*T.pdf(z)
        print(f"TP n={n} nu={nu}: q97.5={T.ppf(.975):.3f}; EI ratio z=-1: {eT(-1)/eG(-1,1,1.0):.3f}, z=-2: {eT(-2)/eG(-2,1,1.0):.3f}")
