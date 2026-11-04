"""
Difference-based empirical-characteristic-function (ECF) estimator of the tail index alpha
under M2:  y_i = f(x_i) + sigma * eps_i,  eps_i iid SaS with ch.f. exp(-|t|^alpha).

order 1 (any design):  D_j = y_{2j} - y_{2j-1},          noise part ~ ch.f. exp(-2 sigma^a |t|^a)
order 2 (equispaced):  D_j = y_{3j} - 2 y_{3j-1} + y_{3j-2}, noise part ~ ch.f. exp(-(2+2^a) sigma^a |t|^a)
In both cases log phi(t1) / log phi(t2) = (t1/t2)^alpha, so
    alpha_hat = log[ log phi_hat(t1) / log phi_hat(t2) ] / log(t1/t2),
with t_k = c_k / s and s = median |D_j| (scale-free; estimating s has no first-order effect).
"""
import numpy as np
from scipy import stats
from scipy.stats import levy_stable


def differences(y, order=1):
    y = np.asarray(y, float)
    if order == 1:
        m = len(y) // 2; return y[1:2*m:2] - y[0:2*m:2]
    m = len(y) // 3; y3 = y[:3*m].reshape(m, 3); return y3[:, 2] - 2*y3[:, 1] + y3[:, 0]


def alpha_hat(y, order=1, c=(0.2, 1.0), clip=True):
    D = differences(y, order); s = np.median(np.abs(D))
    t1, t2 = c[0] / s, c[1] / s
    p1, p2 = np.mean(np.cos(t1 * D)), np.mean(np.cos(t2 * D))
    if not (0 < p2 < 1 and 0 < p1 < 1): return np.nan
    a = np.log(np.log(p1) / np.log(p2)) / np.log(t1 / t2)
    return float(np.clip(a, 1e-3, 2.0)) if clip else float(a)


def asym_sd(alpha, order=1, c=(0.2, 1.0)):
    """Delta-method asymptotic sd of sqrt(m)(alpha_hat - alpha) at t_k = c_k / s_inf (no bias)."""
    k = 2.0 if order == 1 else 2.0 + 2.0**alpha          # noise part = k^{1/alpha} * SaS(1) (unit sigma)
    scale = k ** (1 / alpha)
    s_inf = scale * levy_stable.ppf(0.75, alpha, 0) if alpha < 2 else scale * np.sqrt(2) * stats.norm.ppf(0.75)
    t = np.array(c) / s_inf
    psi = lambda u: np.exp(-k * np.abs(u) ** alpha)
    S = np.array([[0.5 * (psi(a - b) + psi(a + b)) - psi(a) * psi(b) for b in t] for a in t])
    u = psi(t); L = np.log(t[0] / t[1])
    grad = np.array([1 / (u[0] * np.log(u[0]) * L), -1 / (u[1] * np.log(u[1]) * L)])
    return float(np.sqrt(grad @ S @ grad))
