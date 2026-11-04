"""O(n) Laplace approximation for M2 (GP with Matern-5/2 kernel + symmetric alpha-stable noise) on 1-D inputs.

The GP is written as the 3-dimensional state-space model of the Matern-5/2 process, so every linear solve
(K^{-1} + diag(c))^{-1} b and log|I + K diag(c)| costs O(n) (C kernel ssm_kernel.c, compiled with gcc on first
import). If no C compiler is available, or SSM_PURE_PYTHON=1 is set, the same algorithm runs in pure Python
(_solve_py: the same floating-point operations in the same order, so the same results; it is slower).
The Laplace value is the same as StableNoiseGP._laplace (dense, O(n^3)) up to the 1e-8 jitter used there.

interp='linear' reproduces logpdf_derivs of stable_noise_gp (linear in alpha between table nodes);
interp='cubic' uses Catmull-Rom interpolation in alpha, so the likelihood is C^1 in alpha (used in the ML rate study).
"""
import os, sys, ctypes, subprocess, warnings
from math import exp as _exp, log as _log, sqrt as _sqrt, isfinite as _isfinite
import numpy as np
from numpy.ctypeslib import ndpointer
HERE = os.path.dirname(os.path.abspath(__file__))
from stable_noise_gp import _single, _gauss, lik as lik_linear

_SRC = os.path.join(HERE, "ssm_kernel.c")
_SO = os.path.join(HERE, "ssm_kernel" + (".dll" if os.name == "nt" else ".so"))


def _load_c():
    """Compile (if needed) and load the C kernel; None if that is not possible."""
    if os.environ.get("SSM_PURE_PYTHON") == "1":
        return None
    try:
        if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
            flags = ["-O2", "-shared"] + ([] if os.name == "nt" else ["-fPIC"])
            subprocess.check_call(["gcc", *flags, "-o", _SO, _SRC, "-lm"])
        lib = ctypes.CDLL(_SO)
    except (OSError, subprocess.CalledProcessError) as e:     # no gcc, failed build, or unloadable library
        warnings.warn(f"ssm_laplace: C kernel unavailable ({type(e).__name__}: {e}); using the pure-Python "
                      "kernel (same results, slower). Install gcc for speed; see README.", RuntimeWarning)
        return None
    D = ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.ssm_solve.argtypes = [ctypes.c_int, D, ctypes.c_double, ctypes.c_double, D, D, D,
                              ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    lib.ssm_solve.restype = ctypes.c_int
    return lib


_lib = _load_c()
BACKEND = "C" if _lib is not None else "python"


def _solve_py(x, sf2, ell, c, b):
    """Line-by-line port of ssm_solve in ssm_kernel.c (same operations, same order). Returns (status, f, logdet, sign)."""
    n = len(x)
    lam = _sqrt(5.0) / ell; l2 = lam * lam
    Pinf = [sf2, 0.0, -sf2 * l2 / 3, 0.0, sf2 * l2 / 3, 0.0, -sf2 * l2 / 3, 0.0, sf2 * l2 * l2]
    N = [lam, 1.0, 0.0, 0.0, lam, 1.0, -lam * lam * lam, -3 * lam * lam, -2 * lam]
    N2 = [0.0] * 9
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3): s += N[3 * i + k] * N[3 * k + j]
            N2[3 * i + j] = s
    mp, mf, Pp, Pf, At = [None] * n, [None] * n, [None] * n, [None] * n, [None] * n
    status, ld, sg = 0, 0.0, 1
    for t in range(n):
        if t == 0:
            m_ = [0.0, 0.0, 0.0]; P_ = list(Pinf)
        else:
            d = x[t] - x[t - 1]; e = _exp(-lam * d)
            A = [e * ((1.0 if i % 4 == 0 else 0.0) + N[i] * d + 0.5 * N2[i] * d * d) for i in range(9)]
            At[t] = A
            mprev, Pprev = mf[t - 1], Pf[t - 1]
            m_ = [A[3 * i] * mprev[0] + A[3 * i + 1] * mprev[1] + A[3 * i + 2] * mprev[2] for i in range(3)]
            D = [Pprev[i] - Pinf[i] for i in range(9)]
            T1 = [0.0] * 9
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for k in range(3): s += A[3 * i + k] * D[3 * k + j]
                    T1[3 * i + j] = s
            T2 = [0.0] * 9
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for k in range(3): s += T1[3 * i + k] * A[3 * j + k]
                    T2[3 * i + j] = s
            P_ = [Pinf[i] + T2[i] for i in range(9)]
            for i in range(3):
                for j in range(i + 1, 3):
                    s = 0.5 * (P_[3 * i + j] + P_[3 * j + i]); P_[3 * i + j] = P_[3 * j + i] = s
        mp[t], Pp[t] = m_, P_
        s = P_[0]; den = 1.0 + c[t] * s
        if den == 0.0 or not _isfinite(den):
            status = 1; break
        ld += _log(abs(den))
        if den < 0: sg = -sg
        kv = [P_[0], P_[3], P_[6]]; inn = (b[t] - c[t] * m_[0]) / den
        mf[t] = [m_[i] + kv[i] * inn for i in range(3)]
        Pf[t] = [P_[3 * i + j] - c[t] * kv[i] * kv[j] / den for i in range(3) for j in range(3)]
    fout = [0.0] * n
    if status == 0:
        ms = list(mf[n - 1]); fout[n - 1] = ms[0]
        for t in range(n - 2, -1, -1):
            A, Pn, mn, P, m = At[t + 1], Pp[t + 1], mp[t + 1], Pf[t], mf[t]
            r = [ms[i] - mn[i] for i in range(3)]
            a_, b_, c_, d_, e_, f_, g_, h_, k_ = Pn                     # solve3 (cofactors), as in the C code
            C00, C01, C02 = e_ * k_ - f_ * h_, -(d_ * k_ - f_ * g_), d_ * h_ - e_ * g_
            det = a_ * C00 + b_ * C01 + c_ * C02
            if det == 0.0 or not _isfinite(det):
                status = 1; break
            inv = [C00, -(b_ * k_ - c_ * h_), b_ * f_ - c_ * e_,
                   C01, a_ * k_ - c_ * g_, -(a_ * f_ - c_ * d_),
                   C02, -(a_ * h_ - b_ * g_), a_ * e_ - b_ * d_]
            u = [(inv[3 * i] * r[0] + inv[3 * i + 1] * r[1] + inv[3 * i + 2] * r[2]) / det for i in range(3)]
            v = [A[i] * u[0] + A[3 + i] * u[1] + A[6 + i] * u[2] for i in range(3)]
            ms = [m[i] + P[3 * i] * v[0] + P[3 * i + 1] * v[1] + P[3 * i + 2] * v[2] for i in range(3)]
            fout[t] = ms[0]
    return status, fout, ld, sg


def solve(x, sf, ell, c, b):
    """f = (K^{-1} + diag c)^{-1} b and log|I + K diag c| (with sign). x must be sorted."""
    if _lib is None:
        st, out, ld, sg = _solve_py(np.asarray(x, float).tolist(), sf * sf, ell,
                                    np.asarray(c, float).tolist(), np.asarray(b, float).tolist())
        if st != 0:
            raise np.linalg.LinAlgError("zero pivot")
        return np.array(out), ld, sg
    n = len(x); out = np.empty(n); ld = ctypes.c_double(); sg = ctypes.c_int()
    st = _lib.ssm_solve(n, np.ascontiguousarray(x, float), sf * sf, ell, np.ascontiguousarray(c, float),
                        np.ascontiguousarray(b, float), out, ctypes.byref(ld), ctypes.byref(sg))
    if st != 0:
        raise np.linalg.LinAlgError("zero pivot")
    return out, ld.value, sg.value


def _node(z, k):
    return _gauss(z) if k >= 200 else _single(z, k * 10)


def logpdf_derivs_cubic(z, a):
    """log p_a(z) and two z-derivatives; Catmull-Rom in alpha over nodes k-1..k+2 (step 0.01); linear for alpha >= 1.98 (as in stable_noise_gp)."""
    z = np.asarray(z, float); a = float(min(max(a, 0.51), 2.0))
    k = int(np.floor(a * 100 + 1e-9)); t = a * 100 - k
    if k >= 200 or t < 1e-12:
        return _node(z, min(k, 200))
    if k + 2 > 199 or k - 1 < 50:                       # never use the Gaussian node inside a cubic stencil
        A, B = _node(z, k), _node(z, k + 1)
        return tuple((1 - t) * u + t * v for u, v in zip(A, B))
    P = [_node(z, j) for j in (k - 1, k, k + 1, k + 2)]
    w = (-0.5 * t ** 3 + t ** 2 - 0.5 * t, 1.5 * t ** 3 - 2.5 * t ** 2 + 1, -1.5 * t ** 3 + 2 * t ** 2 + 0.5 * t, 0.5 * t ** 3 - 0.5 * t ** 2)
    return tuple(sum(wi * Pi[q] for wi, Pi in zip(w, P)) for q in range(3))


def lik_cubic(y, f, sig, a):
    z = (y - f) / sig
    lp, d1, d2 = logpdf_derivs_cubic(z, a)
    zs = np.where(np.abs(z) < 1e-8, 1e-8, z)
    return lp - np.log(sig), -d1 / sig, -d2 / sig ** 2, np.maximum(-d1 / zs, 1e-12) / sig ** 2


def laplace(x, y, sf, ell, sig, a, warm=None, interp="cubic", tol=None):
    """Laplace log marginal likelihood of standardised y. Returns (lml, f_hat, a_hat, sign) where a_hat = K^{-1} f_hat.
    warm = (f, K^{-1} f) from a previous call (any hyperparameters) is used only as a starting point for f."""
    L = lik_cubic if interp == "cubic" else lik_linear
    if tol is None:
        tol = 1e-9 + 1e-10 * len(y)       # psi is evaluated with rounding noise of roughly this size (ill-conditioned K)
    psi = lambda f, av: L(y, f, sig, a)[0].sum() - 0.5 * f @ av
    if warm is None:
        c = np.full(len(y), 1 / sig ** 2); f, _, _ = solve(x, sf, ell, c, c * y); n_em = 50
    else:
        f = warm[0].copy(); n_em = 10
    # EM / IRLS steps: f = (K^-1 + W_em)^-1 W_em y; (f, K^-1 f) stay consistent
    for _ in range(n_em):
        w = L(y, f, sig, a)[3]
        fn, _, _ = solve(x, sf, ell, w, w * y)
        done = np.max(np.abs(fn - f)) < 1e-7; f = fn
        if done: break
    av = w * y - w * f
    cur = psi(f, av)
    for _ in range(100):                                   # Newton with line search, EM step as fallback
        _, g, W, w = L(y, f, sig, a)
        bb = W * f + g
        try:
            fn, _, _ = solve(x, sf, ell, W, bb); an = bb - W * fn
        except np.linalg.LinAlgError:
            fn = None
        ok = False
        if fn is not None and np.all(np.isfinite(fn)):
            step = 1.0
            while step >= 1e-4:
                ft, at = f + step * (fn - f), av + step * (an - av); new = psi(ft, at)
                if new >= cur - 1e-12: ok = True; break
                step *= 0.5
        if not ok:                                         # EM step (monotone)
            ft, _, _ = solve(x, sf, ell, w, w * y); at = w * y - w * ft; new = psi(ft, at)
        done = abs(new - cur) < tol or np.max(np.abs(g - av)) < 1e-9
        if new >= cur: f, av, cur = ft, at, new
        elif not ok:
            break      # neither Newton nor EM improved: the state is unchanged, so further iterations would repeat this one
        if done: break
    W = L(y, f, sig, a)[2]
    _, ld, sg = solve(x, sf, ell, W, np.zeros(len(y)))
    lml = cur - 0.5 * ld if sg > 0 else -1e10
    return lml, f, av, sg

