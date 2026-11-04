"""
Surrogate selection for a given data set: GP  vs  Student-t process (TP)  vs  GP with
Student-t likelihood (GP-t).  All three use the same Matern-5/2 kernel.

  TP   : heavy tails in the WHOLE process (one scale-mixing variable shared by all points).
  GP-t : heavy tails in the individual observation noise (one mixing variable per point).

Step 1  Fit all three by type-II maximum likelihood (TP and GP-t also learn nu).
Step 2  Diagnose on GP leave-one-out standardised residuals (not on raw y).
Step 3  Overall predictive fit: LOO log predictive density (LPD) and CRPS, paired vs GP.
Step 4  Fit where decisions are made: the same scores on the decision region = training
        points whose posterior mean (GP or candidate) is in its top quartile.
Step 5  Does the choice change the decision?  Compare EI surfaces / proposed next points.
Step 6  Pre-specified rule, for each heavy-tailed candidate k in {TP, GP-t}:
          (a) tail evidence:        nu_hat_k < 30  and  log Bayes factor vs GP > 1
          (b) predictive support:   LOO LPD gain over GP > 0 overall and >= 0 in the decision region
          (c) decision impact:      its EI proposal differs from the GP's by > 1/4 GP length-scale
        (a)+(b)+(c)  -> use candidate k (if both pass, the larger decision-region gain)
        (a)+(b) only -> GP for now: same next query; re-check as data accrue
        otherwise    -> GP
"""
import numpy as np
from scipy import stats
from surrogates import Surrogate, TLikGP, ei_t

NU_GAUSS, LOGBF_MIN = 30.0, 1.0


def paired(d):
    d = np.asarray(d, float)
    return d.sum(), (np.sqrt(len(d)) * d.std(ddof=1) if len(d) > 1 else np.nan)


def _ei(m, x, grid):
    mu, s, d = m.predict(grid)
    best = m.predict(x)[0].max()
    ei = ei_t(mu, s, d, best)
    return dict(mu=mu, s=s, d=d, ei=ei, x_next=grid[np.argmax(ei)])


def run(x, y, grid, restarts=12, seed=0):
    models = {"GP": Surrogate("gp").fit(x, y, restarts, seed),
              "TP": Surrogate("tp").fit(x, y, restarts, seed),
              "GP-t": TLikGP().fit(x, y, restarts, seed)}
    loo = {k: m.loo_scores(y) for k, m in models.items()}
    pred = {k: _ei(m, x, grid) for k, m in models.items()}
    gp = models["GP"]
    lpd_g, crps_g, mu_g, s_g, _ = loo["GP"]
    r = (y - mu_g) / s_g
    res = dict(models=models, loo=loo, pred=pred, r=r,
               sw_p=stats.shapiro(r).pvalue, exkurt=stats.kurtosis(r), maxabs=np.abs(r).max(),
               cand={})
    top_g = gp.predict(x)[0] >= np.quantile(gp.predict(x)[0], 0.75)
    for k in ("TP", "GP-t"):
        m = models[k]
        lpd_k, crps_k = loo[k][0], loo[k][1]
        mk = m.predict(x)[0]
        top = top_g | (mk >= np.quantile(mk, 0.75))
        dl, se = paired(lpd_k - lpd_g)
        dlt, set_ = paired((lpd_k - lpd_g)[top])
        dc, sec = paired(crps_g - crps_k)
        dct, sect = paired((crps_g - crps_k)[top])
        logbf = m.lml - gp.lml
        dx = abs(pred[k]["x_next"] - pred["GP"]["x_next"])
        rho = stats.spearmanr(pred[k]["ei"], pred["GP"]["ei"]).statistic
        a = (m.nu < NU_GAUSS) and (logbf > LOGBF_MIN)
        b = (dl > 0) and (dlt >= 0)
        c = dx > 0.25 * gp.ell
        res["cand"][k] = dict(nu=m.nu, logbf=logbf, dlpd=dl, se=se, dlpd_top=dlt, se_top=set_,
                              dcrps=dc, se_c=sec, dcrps_top=dct, se_ct=sect, n_top=int(top.sum()),
                              dx=dx, rho=rho, a=a, b=b, c=c, top=top)
    passing = [k for k, v in res["cand"].items() if v["a"] and v["b"] and v["c"]]
    fit_only = [k for k, v in res["cand"].items() if v["a"] and v["b"] and not v["c"]]
    if passing:
        choice = max(passing, key=lambda k: res["cand"][k]["dlpd_top"])
        note = "heavy tails supported and they change the next query"
    elif fit_only:
        choice, note = "GP", f"{'/'.join(fit_only)} fits better but proposes the same next point"
    else:
        choice, note = "GP", "no supported evidence of heavy tails"
    res["choice"], res["note"] = choice, note
    return res


def summary(res, xfmt=lambda v: f"{v:.3f}"):
    lines = [f"  GP LOO residuals: Shapiro-Wilk p = {res['sw_p']:.3f}, excess kurtosis = {res['exkurt']:.2f}, max|r| = {res['maxabs']:.2f}",
             f"  GP proposes x_next = {xfmt(res['pred']['GP']['x_next'])}"]
    for k, v in res["cand"].items():
        lines.append(f"  {k:4s}: nu = {v['nu']:7.2f}  logBF = {v['logbf']:6.2f}  dLPD = {v['dlpd']:6.2f} ({v['se']:.2f})  "
                     f"dLPD_top = {v['dlpd_top']:6.2f} ({v['se_top']:.2f}, n={v['n_top']})  dCRPS = {v['dcrps']:.4f} ({v['se_c']:.4f})  "
                     f"x_next = {xfmt(res['pred'][k]['x_next'])}  rho_EI = {v['rho']:.2f}  [a={v['a']}, b={v['b']}, c={v['c']}]")
    lines.append(f"  => choose {res['choice']}  ({res['note']})")
    return "\n".join(lines)
