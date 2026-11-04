"""Write panel (a) of the BO-study table (paper/tab_bo_a.tex) from results/bo/bo_summary_c4.json: median final
regret and CVaR_0.9 (mean of the worst 10% of runs) per problem and surrogate; bold marks the smallest median as printed
(ties bold); the numbers of repetitions are given in the caption. Added 4 Nov 2026 so that the printed numbers are generated, not transcribed."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src'))
import _paths, json
S = json.load(open(_os.path.join(_paths.results("bo"), "bo_summary_c4.json")))["problems"]
P = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "paper")
ROWS = [("gauss", r"Gaussian noise"), ("outliers", r"15\% gross outliers"), ("skewout", r"one-sided outliers$^\dagger$"),
        ("sas13", r"S$\alpha$S($1.3$) noise"), ("t2", r"$t_2$ noise$^\dagger$"), ("branin2", r"Branin, S$\alpha$S($1.3$)"),
        ("gardner1", r"Gardner~1, S$\alpha$S($1.3$)"), ("gardner2", r"Gardner~2, S$\alpha$S($1.3$)"), ("hart6", r"Hartmann-6, S$\alpha$S($1.3$)")]
K = ["GP", "TP", "GP-t", "RCGP", "SaS"]
out = []
for i, (key, lab) in enumerate(ROWS):
    v = S[key]["surrogates"]; dm, dc = (2, 2) if key == "hart6" else (4, 3)
    med = {k: f"{v[k]['median_final']:.{dm}f}" for k in K}; best = min(float(m) for m in med.values())
    cells = [(f"\\textbf{{{med[k]}}}" if float(med[k]) == best else med[k]) + f" ({v[k]['cvar90']:.{dc}f})" for k in K]
    out.append(f"{lab} & " + " & ".join(cells) + ("\\\\\\bottomrule" if i == len(ROWS) - 1 else "\\\\"))
open(_os.path.join(P, "tab_bo_a.tex"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
