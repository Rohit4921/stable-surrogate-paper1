import matplotlib.pyplot as plt
C_GP, C_TP, C_T, INK, INK2, GRID_C = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "DejaVu Sans", "lines.linewidth": 1.6})

# Embed TrueType (Type 42) rather than Type 3 fonts in PDF output (journal production requirement, 8 Oct 2026).
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42
