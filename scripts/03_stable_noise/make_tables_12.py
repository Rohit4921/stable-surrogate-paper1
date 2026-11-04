"""Write the bodies of Tables 1 and 2 of the paper (paper/tab_cons.tex, paper/tab_cf.tex) directly from the stored
summaries, so that the printed numbers cannot drift from the results (added 4 Nov 2026 after the pre-submission audit
found two transcription errors in Table 2). Table 1: ml_unified_summary.json. Table 2: crossfit_summary.json
(running-median pilot with odd windows) and ml_unified_summary.json (Laplace ML column)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import _paths, json
R = _paths.results("stable_noise"); P = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "paper")
fmtn = lambda n: f"{n:,}".replace(",", "{,}")
ML = json.load(open(_os.path.join(R, "ml_unified_summary.json")))
lines = []
for a0 in (1.3, 1.7, 2.0):
    rows = [o for o in ML if o["a0"] == a0 and o["n"] in (30, 60, 120, 240, 1920)]
    for i, o in enumerate(sorted(rows, key=lambda o: o["n"])):
        end = "\\\\\\addlinespace" if (i == len(rows) - 1 and a0 != 2.0) else ("\\\\\\bottomrule" if i == len(rows) - 1 else "\\\\")
        lines.append(f"{a0:.1f} & {fmtn(o['n'])} & {o['ml_median']:.2f} & {o['ml_mean']:.2f} & {o['ml_rmse']:.3f} & {o['ml_at_bound']:.2f} & {o['dn_median']:.4f}{end}")
open(_os.path.join(P, "tab_cons.tex"), "w").write("\n".join(lines) + "\n")
CF = json.load(open(_os.path.join(R, "crossfit_summary.json")))["rows"]
lines = []
for a0 in (1.3, 1.7):
    rows = sorted([r for r in CF if r["a0"] == a0], key=lambda r: r["n"])
    for i, r in enumerate(rows):
        lines.append(f"{a0:.1f} & {fmtn(r['n'])} & {r['k']} & {r['med_rmse_sqrtn']:.2f} & {r['gp_rmse_sqrtn']:.2f} & {r['oracle_rmse_sqrtn']:.2f} & "
                     f"{r['laplace_rmse_sqrtn']:.2f} & {r['med_cover']:.2f} & {r['gp_cover']:.2f} & {r['oracle_cover']:.2f}\\\\")
    if a0 == 1.3: lines.append("\\addlinespace")
open(_os.path.join(P, "tab_cf.tex"), "w").write("\n".join(lines) + "\n")
print(open(_os.path.join(P, "tab_cons.tex")).read()); print(open(_os.path.join(P, "tab_cf.tex")).read())
