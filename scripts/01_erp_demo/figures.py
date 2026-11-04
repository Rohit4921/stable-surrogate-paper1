import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths
_os.chdir(_paths.results('erp_demo'))   # result files are read and written here
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
import nifty_objective as no
from surrogates import Surrogate, TLikGP, profile_nu
from procedure import run, summary

no.WIN = 504
C_GP, C_TP, C_T, INK, INK2, GRID_C = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "DejaVu Sans", "lines.linewidth": 1.6})
f = lambda x: x*np.sin(10*x)+np.cos(2*x)
OUT = {}


def band(ax, g, mu, s, d, color, label, ls="-"):
    q = stats.norm.ppf(0.975) if np.isinf(d) else stats.t.ppf(0.975, d)
    ax.fill_between(g, mu - q*s, mu + q*s, color=color, alpha=0.15, lw=0)
    ax.plot(g, mu, color=color, ls=ls, label=label)


# ------------------------------------------------------------------ synthetic cases
rng = np.random.default_rng(7); n = 25
x = np.sort((np.arange(n)+rng.uniform(size=n))/n*1.2)
yA = f(x) + 0.05*rng.standard_normal(n)
yB = yA.copy(); OUTL = [4, 11, 16, 21]; yB[OUTL] += [-0.8, 0.8, -0.9, 0.9]
g = np.linspace(0, 1.2, 401)
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), constrained_layout=True)
for row, (lab, y) in enumerate([("A. Gaussian noise", yA), ("B. 4 isolated outliers", yB)]):
    R = run(x, y, g); OUT[lab] = R
    print(lab); print(summary(R))
    ax = axs[row, 0]
    ax.plot(g, f(g), color=INK, lw=0.9, label="true f")
    for k, c, ls in (("GP", C_GP, "-"), ("TP", C_TP, "--"), ("GP-t", C_T, "-")):
        p = R["pred"][k]; band(ax, g, p["mu"], p["s"], p["d"], c, k, ls)
    clean = np.setdiff1d(np.arange(n), OUTL) if row == 1 else np.arange(n)
    ax.plot(x[clean], y[clean], "o", ms=4, color=INK2, zorder=5, label="observations")
    if row == 1:
        ax.plot(x[OUTL], y[OUTL], "o", ms=6, mfc="white", mec=INK, mew=1.2, zorder=6, label="planted outliers")
    ax.set_title(lab, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if row == 0: ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    if row == 1: ax.legend(handles=ax.get_legend_handles_labels()[0][-1:], fontsize=7.5, loc="lower left")
    # pointwise LOO gain
    ax = axs[row, 1]
    for k, c, off in (("TP", C_TP, -0.004), ("GP-t", C_T, 0.004)):
        d = R["loo"][k][0] - R["loo"]["GP"][0]
        ax.vlines(x + off, 0, d, color=c, lw=2)
        ax.plot(x + off, d, "o", ms=3, color=c, label=f"{k} − GP (sum {d.sum():+.1f})")
    ax.axhline(0, color=INK2, lw=0.8)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45*(hi - lo))
    ax.set_xlabel("x"); ax.set_ylabel("gain (nats per point)")
    ax.set_title("LOO log-density gain over GP", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, loc="upper left")
    # EI
    ax = axs[row, 2]
    for k, c, ls in (("GP", C_GP, "-"), ("TP", C_TP, "--"), ("GP-t", C_T, "-")):
        e = R["pred"][k]["ei"]; ax.plot(g, e/e.max(), color=c, ls=ls, label=f"{k}: next x = {R['pred'][k]['x_next']:.3f}")
    ax.axvline(0.175, color=INK, lw=0.8, ls=":"); ax.text(0.18, 1.02, "true optimum", fontsize=7.5, color=INK2)
    ax.set_xlabel("x"); ax.set_ylabel("EI (scaled to max 1)"); ax.set_ylim(0, 1.12)
    ax.set_title("Expected improvement: next query", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, loc="upper right")
fig.savefig(_paths.fig("fig1_synthetic.pdf")); fig.savefig(_paths.fig("fig1_synthetic.png"), dpi=150)

# ------------------------------------------------------------------ TP profile likelihood result
fig, ax = plt.subplots(figsize=(5.2, 3.3), constrained_layout=True)
nus = np.exp(np.linspace(np.log(2.2), np.log(1000), 25))
h = lambda nu, n: gammaln((nu+n)/2)-gammaln(nu/2)-n/2*np.log(nu)-(nu+n)/2*np.log1p(n/nu)
hinf = lambda n: -n/2*np.log(2)-n/2
ax.plot(nus, h(nus, 25) - hinf(25), color=INK, lw=2.4, alpha=0.35, label="analytic $h_n(\\nu)-h_n(\\infty)$, n = 25")
yC = f(x) + 0.05*np.random.default_rng(3).standard_t(1.5, n)
marks = {"Gaussian noise": ("o", yA), "4 outliers": ("s", yB), "$t_{1.5}$ noise": ("^", yC)}
for lab, (mk, y) in marks.items():
    gp = Surrogate("gp").fit(x, y)
    prof = profile_nu(x, y, nus[::3]) - gp.lml
    ax.plot(nus[::3], prof, mk, ms=5, mfc="none", mec=INK2, label=f"numerical, {lab}")
ax.set_xscale("log"); ax.set_xlabel("degrees of freedom ν (fixed)"); ax.set_ylabel("log ML(TP, ν) − log ML(GP)")
ax.set_title("TP evidence for ν is the same for every data set", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(_paths.fig("fig2_tp_profile.pdf")); fig.savefig(_paths.fig("fig2_tp_profile.png"), dpi=150)

# ------------------------------------------------------------------ NIFTY
fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), constrained_layout=True)
r = np.log(no.C[1:]/no.C[:-1]); z = (r - r.mean())/r.std()
qq = stats.probplot(z, dist="norm")
ax = axs[0, 0]
ax.plot(qq[0][0], qq[0][1], ".", ms=2.5, color=INK2)
ax.plot([-4, 4], [-4, 4], color=C_GP, lw=1)
ax.set_xlabel("normal quantile"); ax.set_ylabel("standardised daily log return")
ax.set_title(f"(a) NIFTY 50 daily returns: heavy-tailed (excess kurtosis {stats.kurtosis(r):.0f})", loc="left", fontsize=9.5, color=INK)
OUT["nifty_kurt"] = float(stats.kurtosis(r)); OUT["nifty_skew"] = float(stats.skew(r))

ax = axs[0, 1]
rngE = np.random.default_rng(11)
for L, c in ((10, C_GP), (60, C_T), (150, C_TP)):
    ys = np.array([no.evaluate(L, rngE) for _ in range(3000)])
    zz = (ys - ys.mean())/ys.std(); q = stats.probplot(zz, dist="norm")[0]
    ax.plot(q[0], q[1], ".", ms=2, color=c, label=f"L = {L}: excess kurtosis {stats.kurtosis(ys):.1f}")
    OUT[f"noise_kurt_L{L}"] = float(stats.kurtosis(ys)); OUT[f"noise_sd_L{L}"] = float(ys.std())
ax.plot([-4, 4], [-4, 4], color=INK2, lw=0.8)
ax.set_xlabel("normal quantile"); ax.set_ylabel("standardised evaluation noise")
ax.set_title("(b) Noise of one BO evaluation y(L) at fixed L", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, loc="upper left", markerscale=3)

rngD = np.random.default_rng(2026); nN = 30
u = (np.arange(nN)+rngD.uniform(size=nN))/nN
L = np.exp(np.log(5)+u*(np.log(250)-np.log(5)))
xN = np.log(L); yN = np.array([no.evaluate(l, rngD) for l in L])
gN = np.linspace(np.log(5), np.log(250), 401)
RN = run(xN, yN, gN); OUT["NIFTY"] = RN
print("NIFTY"); print(summary(RN, lambda v: f"L={np.exp(v):.0f}"))
Lg = np.unique(np.round(np.exp(gN)).astype(int)); fL = np.array([no.true_objective(l) for l in Lg])
OUT["nifty_true_argmax"] = int(Lg[fL.argmax()]); OUT["nifty_true_max"] = float(fL.max())
ax = axs[1, 0]
ax.plot(Lg, 100*fL, color=INK, lw=0.9, label="f(L): average over all 2-year windows")
for k, c, ls in (("GP", C_GP, "-"), ("TP", C_TP, "--"), ("GP-t", C_T, "-")):
    p = RN["pred"][k]; band(ax, np.exp(gN), 100*p["mu"], 100*p["s"], p["d"], c, k, ls)
ax.plot(L, 100*yN, "o", ms=4, color=INK2, label="30 noisy evaluations")
ax.set_xscale("log"); ax.set_xlabel("moving-average look-back L (trading days)")
ax.set_ylabel("excess return over buy-and-hold (% p.a.)")
ax.set_title("(c) One BO data set: n = 30 evaluations", loc="left", fontsize=9.5, color=INK)
lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.35*(hi - lo), hi)
ax.legend(fontsize=7, loc="lower left", ncol=3)
ax = axs[1, 1]
for k, c, ls in (("GP", C_GP, "-"), ("TP", C_TP, "--"), ("GP-t", C_T, ":")):
    e = RN["pred"][k]["ei"]; ax.plot(np.exp(gN), e/e.max(), color=c, ls=ls, lw=1.8 if k != "GP-t" else 2.2,
                                      label=f"{k}: next L = {np.exp(RN['pred'][k]['x_next']):.0f}")
ax.axvline(OUT["nifty_true_argmax"], color=INK, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("look-back L"); ax.set_ylabel("EI (scaled to max 1)"); ax.set_ylim(0, 1.12)
ax.set_title("(d) Next evaluation proposed by each surrogate", loc="left", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, loc="upper left")
fig.savefig(_paths.fig("fig3_nifty.pdf")); fig.savefig(_paths.fig("fig3_nifty.png"), dpi=150)


def slim(R):
    out = {"choice": R["choice"], "note": R["note"], "sw_p": float(R["sw_p"]), "exkurt": float(R["exkurt"]),
           "maxabs": float(R["maxabs"]), "gp_ell": float(R["models"]["GP"].ell),
           "x_next": {k: float(R["pred"][k]["x_next"]) for k in R["pred"]},
           "loo_sum": {k: float(R["loo"][k][0].sum()) for k in R["loo"]}}
    for k, v in R["cand"].items():
        out[k] = {kk: (float(vv) if not isinstance(vv, (bool, np.bool_)) else bool(vv))
                  for kk, vv in v.items() if kk != "top"}
    return out


json.dump({k: (slim(v) if isinstance(v, dict) and "cand" in v else v) for k, v in OUT.items()},
          open("results_detail.json", "w"), indent=1)
print("saved")
