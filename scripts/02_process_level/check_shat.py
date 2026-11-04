"""Deterministic check that s -> M_n(alpha, s) = E g_n(sS) has a unique maximum, S positive (alpha/2)-stable.
Kanter: S = (A(U)/E)^{(1-b)/b}, U ~ U(0,pi), E ~ Exp(1), A(u) = sin(bu)^{b/(1-b)} sin((1-b)u) / sin(u)^{1/(1-b)}.
Integrate over U (Gauss-Legendre) and over w = log E (trapezoid), all in log space."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
import numpy as np
from scipy.special import logsumexp
xu, wu = np.polynomial.legendre.leggauss(300); U = (xu+1)*np.pi/2; WU = wu*np.pi/2/np.pi   # density 1/pi
w = np.linspace(-25, 4, 1500); dw = w[1]-w[0]; lwE = w - np.exp(w) + np.log(dw)            # log(e^{-E} dE) with E = e^w
out = {}
for n in (10, 25, 100):
    for al in (0.6, 1.0, 1.3, 1.6, 1.9, 1.99):
        b = al/2; gam = (1-b)/b
        lA = b/(1-b)*np.log(np.sin(b*U)) + np.log(np.sin((1-b)*U)) - np.log(np.sin(U))/(1-b)
        lS = gam*(lA[:, None] - w[None, :])                     # 300 x 1500
        base = np.log(WU)[:, None] + lwE[None, :]
        t = np.linspace(-9, 1, 501)
        M = np.array([logsumexp(base - n/2*(tt+lS) - 0.5*np.exp(-(tt+lS))) for tt in t])
        d = np.diff(M); nmax = int(((d[:-1] > 0) & (d[1:] <= 0)).sum())
        # log H_n(alpha) - log H_n(2)
        h2 = -n/2*np.log(1/n) - n/2
        out[(n, al)] = (nmax, t[M.argmax()], M.max()-h2)
        print(f"n={n:3d} alpha={al}: local maxima={nmax}, log s_hat={t[M.argmax()]:.3f} (Gaussian: {np.log(1/n):.3f}), "
              f"log H - log H(2) = {M.max()-h2:+.3f}", flush=True)
