"""
WP1 check: process-level scale mixture Y = sqrt(A) G, A = c*S, S ~ positive (alpha/2)-stable
(Laplace transform E exp(-l S) = exp(-l^{alpha/2})), so Y is sub-Gaussian alpha-stable.
Claim: log L(alpha, c, kappa) = log C(y, kappa) + log M_n(alpha, c/beta0), so the profile
likelihood in alpha, H_n(alpha) = sup_s M_n(alpha, s), does not depend on the data.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
import numpy as np
from scipy.special import logsumexp
from surrogates import matern52, Surrogate

rng = np.random.default_rng(0)
NS = 400_000
U = rng.uniform(0, np.pi, NS); E = rng.exponential(size=NS)

def pos_stable(a):
    # Kanter/CMS representation, 0<a<1, Laplace transform exp(-l^a)
    return np.sin(a*U)/np.sin(U)**(1/a) * (np.sin((1-a)*U)/E)**((1-a)/a)

def log_Mn(alpha, logs, n):
    """log E[ (sS)^{-n/2} exp(-1/(2 s S)) ] for each log s in logs (common random numbers)."""
    if alpha >= 2: 
        s = np.exp(logs); return -n/2*logs - 1/(2*s)
    S = pos_stable(alpha/2)
    out = []
    for ls in logs:
        v = np.exp(ls)*S
        out.append(logsumexp(-n/2*np.log(v) - 1/(2*v)) - np.log(NS))
    return np.array(out)

def log_Hn(alpha, n):
    logs = np.linspace(-12, 4, 161)
    m = log_Mn(alpha, logs, n)
    i = m.argmax(); fine = np.linspace(logs[max(i-2,0)], logs[min(i+2,160)], 81)
    return log_Mn(alpha, fine, n).max()

alphas = [1.1, 1.3, 1.5, 1.7, 1.8, 1.9, 1.95, 1.99, 2.0]
print("log H_n(alpha) - log H_n(2):")
for n in (10, 25, 50):
    h2 = log_Hn(2.0, n)
    print(f" n={n:3d}:", "  ".join(f"a={a}:{log_Hn(a,n)-h2:+.3f}" for a in alphas))

# direct check on real data sets: full likelihood profiled over c, kernel fixed at GP ML fit
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
r = np.random.default_rng(7); n = 25
x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2)
ys = {"gaussian": f(x)+0.05*r.standard_normal(n)}
yb = ys["gaussian"].copy(); yb[[4,11,16,21]] += [-0.8,0.8,-0.9,0.9]; ys["outliers"] = yb
ys["stable-noise a=1.3"] = f(x) + 0.05*np.sign(r.standard_normal(n))*np.abs(r.standard_normal(n))*np.sqrt(2*pos_stable(0.65)[:n])
print("\nDirect profile log L(alpha) - log L(GP), kernel fixed at each data set's GP fit:")
for lab, y in ys.items():
    g = Surrogate("gp").fit(x, y)
    y0 = (y - y.mean())/y.std()
    K0 = matern52(x, x, 1.0, g.ell) + (g.sn/g.sf)**2*np.eye(n) + 1e-10*np.eye(n)
    beta0 = y0 @ np.linalg.solve(K0, y0)
    # log L = -n/2 log(2 pi) - 1/2 log|K0| + log E[ A^{-n/2} exp(-beta0/(2A)) ],  A = c S
    def prof(alpha):
        logs = np.linspace(-10, 8, 181) + np.log(beta0)   # log c
        if alpha >= 2:
            c = np.exp(logs); vals = -n/2*logs - beta0/(2*c)
        else:
            S = pos_stable(alpha/2)
            vals = np.array([logsumexp(-n/2*np.log(np.exp(lc)*S) - beta0/(2*np.exp(lc)*S)) - np.log(NS) for lc in logs])
        return vals.max()
    base = prof(2.0)
    print(f" {lab:20s}", "  ".join(f"a={a}:{prof(a)-base:+.3f}" for a in [1.1,1.5,1.9,1.99]))
