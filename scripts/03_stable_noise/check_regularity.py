# Numerical check of assumption (R): boundedness of r-derivatives of the (alpha, log sigma) scores of
# l_theta(r) = -log sigma + log p_alpha(r/sigma), on a neighbourhood of theta0, using the package's density tables.
import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from ssm_laplace import logpdf_derivs_cubic as L
def ell(r, a, ls):           # log density of residual r under SaS(alpha, sigma = e^ls)
    s = np.exp(ls); lp, d1, d2 = L(r / s, a); return lp - ls
r = np.concatenate([-np.geomspace(1e4, 1e-3, 3000), [0.0], np.geomspace(1e-3, 1e4, 3000)])
h = 1e-3
for a0 in (1.0, 1.3, 1.7, 1.9):
    out = {}
    for a in (a0 - 0.05, a0, a0 + 0.05):
        ls = 0.0
        # alpha- and log-sigma scores and second derivatives by central differences
        f = lambda aa, ll: ell(r, aa, ll)
        sa = (f(a + h, ls) - f(a - h, ls)) / (2 * h); ss = (f(a, ls + h) - f(a, ls - h)) / (2 * h)
        saa = (f(a + h, ls) - 2 * f(a, ls) + f(a - h, ls)) / h**2
        sas = (f(a + h, ls + h) - f(a + h, ls - h) - f(a - h, ls + h) + f(a - h, ls - h)) / (4 * h * h)
        sss = (f(a, ls + h) - 2 * f(a, ls) + f(a, ls - h)) / h**2
        for name, g in (("s_a", sa), ("s_s", ss), ("s_aa", saa), ("s_as", sas), ("s_ss", sss)):
            g1 = np.gradient(g, r); g2 = np.gradient(g1, r)
            out.setdefault(name, []).append((np.abs(g1).max(), np.abs(g2).max(), np.abs(g1[np.abs(r) > 1e3]).max()))
        out.setdefault("l_r", []).append((np.abs(np.gradient(f(a, ls), r)).max(),))
    print(f"alpha0={a0}: " + "; ".join(f"{k}: sup|d_r|={max(v[0] for v in vals):.2f}" + (f", sup|d_r^2|={max(v[1] for v in vals):.2f}, tail|d_r|={max(v[2] for v in vals):.1e}" if len(vals[0]) > 1 else "") for k, vals in out.items()))
