"""(1) Slice from the exact optimum to the (failing) Laplace optimum for the three fold cases.
    At t = 1 (the Laplace optimum, a fold) 'laplace' is the stored value of the exact-likelihood study and
    'laplace_recomputed' the machine-dependent cold-start value (NaN if det(I + K W) <= 0).
(2) One-observation toy: f ~ N(0, v), y | f ~ SaS(alpha) located at f, y fixed far out; Laplace at the
outlier-side local mode versus exact evidence as v varies, and the inflection point z* of log p_alpha."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise/fold"))   # result files are read and written here
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import fold_remedy as remedy, exact_m2 as X
from stable_noise_gp import rsas, logpdf_derivs
from scipy.optimize import brentq
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
out = {"slices": [], "toy": {}, "inflection": {}}
D = {(o["a0"], o["n"], o["rep"]): o for fn in ("1.3_n30", "1.7_n30", "1.7_n60") for o in json.load(open(f"../exact3_a{fn}_main.json"))}
for a0, n, rep in ((1.3, 30, 1), (1.7, 30, 1), (1.7, 60, 2)):
    o = D[(a0, n, rep)]; r = np.random.default_rng(int(a0*1000)+10*n+rep)
    x = np.sort((np.arange(n)+r.uniform(size=n))/n*1.2); y = f(x)+0.05*rsas(a0, n, r)
    E = X.ExactM2(x, y, S=10); F = X.ExactM2(x, y, S=20000, seed=7); c = -n*np.log(E.ys)
    pX, pL = np.array(o["exact"]["p"]), np.array(o["laplace"]["p"])
    ts = np.r_[np.linspace(0, 0.9, 10), 1 - np.geomspace(0.1, 1e-6, 12), 1.0]
    rows = []
    for t in ts:
        p = (1-t)*pX + t*pL; ev = remedy.evaluate(E, p)
        ex = F.is_mix(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]) + c
        row = dict(t=float(t), lam=ev["lam_min"], laplace=float(ev["standard"]+c), fisher=ev["fisher"]+c, exact=float(ex),
                   machine_dependent=ev["machine_dependent"])
        if t == 1.0:
            # at the Laplace optimum lam_min = 0 to working precision: the recomputed (cold-start) value is
            # machine-dependent or undefined, so report the study's own stored value (README note 6)
            row["laplace_recomputed"] = row["laplace"]; row["laplace"] = float(o["laplace"]["own"])
        rows.append(row)
    out["slices"].append(dict(a0=a0, n=n, rep=rep, rows=rows))
    print(a0, n, rep); [print(f"   t={q['t']:.6f} lam={q['lam']:.2e} laplace={q['laplace']:7.2f} exact={q['exact']:7.2f}" + (" (stored; lam at working precision)" if q['t'] == 1.0 else "")) for q in rows[-6:]]
# inflection points of log p_alpha: d2 log p(z*) = 0
for a in (1.0, 1.3, 1.5, 1.7, 1.9):
    d2 = lambda z: logpdf_derivs(np.array([z]), a)[2][0]
    out["inflection"][str(a)] = float(brentq(d2, 0.3, 20)); print("alpha", a, "inflection z* =", round(out["inflection"][str(a)], 3))
# toy: alpha = 1.3, sigma = 1, y = 6; f ~ N(0, v)
a, y0 = 1.3, 6.0; fg = np.linspace(-15, 25, 200001); dfg = fg[1]-fg[0]
lp_like = logpdf_derivs(y0 - fg, a)[0]
toy = []
for v in np.geomspace(0.3, 30, 400):
    lpost = lp_like - fg**2/(2*v) - 0.5*np.log(2*np.pi*v)
    exact = np.log(np.sum(np.exp(lpost - lpost.max()))*dfg) + lpost.max()
    # local maxima of lpost on the grid
    i = np.where((lpost[1:-1] > lpost[:-2]) & (lpost[1:-1] >= lpost[2:]))[0] + 1
    modes = []
    for k in i:
        fk = fg[k]; _, d1, d2 = logpdf_derivs(np.array([y0 - fk]), a); H = -d2[0] + 1/v   # -(d2/df2) lpost
        modes.append(dict(f=float(fk), H=float(H), laplace=float(lpost[k] + 0.5*np.log(2*np.pi/H)) if H > 0 else None))
    toy.append(dict(v=float(v), exact=float(exact), modes=modes))
out["toy"] = dict(alpha=a, y=y0, rows=toy)
outl = [(t["v"], m) for t in toy for m in t["modes"] if m["f"] > 3]
if outl:
    print("toy: outlier-side mode exists for v >=", round(min(v for v, _ in outl), 3), "; at that v, H =", f"{min(outl)[1]['H']:.2e}",
          " Laplace - exact there:", round(min(outl)[1]["laplace"] - [t["exact"] for t in toy if t["v"] == min(outl)[0]][0], 2))
json.dump(out, open("slice_toy.json", "w"), indent=1)
