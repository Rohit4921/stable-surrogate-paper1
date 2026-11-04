"""Summarise crossfit_a*_v2.json (Table 2): sqrt(n) x RMSE of alpha-hat for the running-median and GP pilots and the
oracle (f known), Wald coverage, Laplace ML from ml_rate_summary.json, and the pilot diagnostics: median ||Delta||_n and
max|Delta_i| per pilot with fitted log-log slopes in n (Lemma: n^{-1/3} for k ~ n^{2/3}; Theorem C: n^{-2/5} up to logs),
the number of GP-pilot fits whose mode has det(I + K W) <= 0 (folds), the largest sup-distance between the computed mode
and the mode reached from the truth, and the range of the pilot's working alpha. Writes crossfit_summary.json.
(crossfit_a*_main.json are the superseded v1 runs: GP pilot standardised by sample mean/sd, alpha <= 2.)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths
_os.chdir(_paths.results("stable_noise"))
import json, numpy as np
ML = {(o["a0"], o["n"]): o for o in json.load(open("ml_unified_summary.json"))}   # phase C Laplace ML
OR = json.load(open("oracle_fisher.json"))
MLR = {a0: json.load(open(f"ml_unified_a{a0}_c1.json")) for a0 in (1.3, 1.7)}             # per-replicate Laplace ML
rows = []
for a0 in (1.3, 1.7):
    fn = f"crossfit_a{a0}_v2.json"
    if not _os.path.exists(fn): continue
    D = json.load(open(fn))
    for n in sorted({o["n"] for o in D}):
        R = [o for o in D if o["n"] == n]; row = dict(a0=a0, n=n, reps=len(R), bound=OR[str(a0)]["sd_oracle"])
        # 4 Nov 2026: the running-median pilot uses odd windows (crossfit_median_odd.py, same data sets); the even-window
        # values of the original run are kept as med_even_* for the record.
        odd = f"crossfit_median_odd_a{a0}.json"
        if _os.path.exists(odd):
            Od = {(o["n"], o["rep"]): o for o in json.load(open(odd))["rows"]}
            a_e = np.array([o["alpha_med"] for o in R])
            row["med_even_rmse_sqrtn"] = float(np.sqrt(n * np.mean((a_e - a0) ** 2))); row["med_even_cover"] = float(np.mean([o["cover_med"] for o in R]))
            row["k_even"] = R[0]["k"]
            R = [dict(o, alpha_med=Od[(o["n"], o["rep"])]["alpha_med"], cover_med=Od[(o["n"], o["rep"])]["cover_med"], k=Od[(o["n"], o["rep"])]["k"],
                      dn_med=Od[(o["n"], o["rep"])]["dn_med"], dinf_med=Od[(o["n"], o["rep"])]["dinf_med"]) for o in R]
        bsi = np.random.default_rng(int(1000 * a0) + n).integers(0, len(R), (2000, len(R)))   # Monte Carlo SEs (bootstrap over replicates)
        for key in ("med", "gp", "oracle"):
            a = np.array([o[f"alpha_{key}"] for o in R])
            row[f"{key}_rmse_sqrtn_se"] = float(np.std(np.sqrt(n * np.mean((a[bsi] - a0) ** 2, 1))))
            cv = float(np.mean([o[f"cover_{key}"] for o in R])); row[f"{key}_cover_se"] = float(np.sqrt(cv * (1 - cv) / len(R)))
            row[f"{key}_bias"] = float(a.mean() - a0); row[f"{key}_rmse_sqrtn"] = float(np.sqrt(n * np.mean((a - a0) ** 2)))
            row[f"{key}_cover"] = float(np.mean([o[f"cover_{key}"] for o in R]))
        if (a0, n) in ML:
            row["laplace_rmse_sqrtn"] = ML[(a0, n)]["ml_sqrt_n_rmse"]
            Lr = [o for o in MLR[a0] if o["n"] == n]; la = np.array([o["ml"]["alpha"] for o in Lr])
            bl = np.random.default_rng(int(1000 * a0) + n + 1).integers(0, len(la), (2000, len(la)))
            row["laplace_rmse_sqrtn_se"] = float(np.std(np.sqrt(n * np.mean((la[bl] - a0) ** 2, 1))))
        row["k"] = R[0]["k"]
        for key in ("med", "gp"):
            row[f"{key}_dn_median"] = float(np.median([o[f"dn_{key}"] for o in R])); row[f"{key}_dinf_median"] = float(np.median([o[f"dinf_{key}"] for o in R]))
        F = [d for o in R for d in o["gp_folds"]]
        row["gp_fold_count"] = int(sum(d["sign"] <= 0 for d in F)); row["gp_fits"] = len(F)
        row["gp_modecheck_max"] = float(max(d["mode_vs_truth_start"] for d in F))
        row["gp_modecheck_differ"] = int(sum(d["mode_vs_truth_start"] > 1e-4 for d in F))   # truth start reached another mode
        row["gp_alpha_work_min"] = float(min(d["p"][3] for d in F)); row["gp_alpha_work_max"] = float(max(d["p"][3] for d in F))
        rows.append(row)
        print(f"a0={a0} n={n:5d} reps={len(R)} k={row['k']:3d} | sqrt(n)RMSE: median {row['med_rmse_sqrtn']:.2f} (bias {row['med_bias']:+.3f}, cover {row['med_cover']:.2f}) "
              f"GP {row['gp_rmse_sqrtn']:.2f} (bias {row['gp_bias']:+.3f}, cover {row['gp_cover']:.2f}) oracle {row['oracle_rmse_sqrtn']:.2f} (cover {row['oracle_cover']:.2f}) "
              f"Laplace {row.get('laplace_rmse_sqrtn', float('nan')):.2f} | bound {row['bound']:.2f}")
slopes = {}
for a0 in (1.3, 1.7):
    Rw = [r for r in rows if r["a0"] == a0]
    if len(Rw) < 2: continue
    ln = np.log([r["n"] for r in Rw])
    for key in ("med_dn_median", "gp_dn_median", "gp_dinf_median"):
        slopes[f"{a0}_{key}"] = float(np.polyfit(ln, np.log([r[key] for r in Rw]), 1)[0])
for r in rows:
    print(f"  a0={r['a0']} n={r['n']:5d}: median ||Delta||_n  median-pilot {r['med_dn_median']:.4f}  GP {r['gp_dn_median']:.4f}  "
          f"(GP sup {r['gp_dinf_median']:.4f}); GP folds {r['gp_fold_count']}/{r['gp_fits']}; mode check max {r['gp_modecheck_max']:.1e} (differ>1e-4: {r['gp_modecheck_differ']}); "
          f"working alpha [{r['gp_alpha_work_min']:.2f}, {r['gp_alpha_work_max']:.2f}]")
print("log-log slopes in n:", {k: round(v, 3) for k, v in slopes.items()})
json.dump(dict(rows=rows, slopes=slopes), open("crossfit_summary.json", "w"), indent=1)
