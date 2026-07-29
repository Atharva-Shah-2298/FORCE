import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

BAR = {1: "#b0a4e3", 2: "#2b2d72", 3: "#f2921e"}
LINE = {
    "ODF-FP":       ("#2ba3e0", "o", "-"),
    "CSD":          ("#28348f", "X", "-"),
    "CSA":          ("#e8781e", "^", "-"),
    "GQI":          ("#7d2b8f", "D", "-"),
    "FORCE_a1e-05": ("#e5449a", "s", ":"),
    "FORCE_a1e-04": ("#d9b40f", "D", ":"),
    "FORCE_a1e-03": ("#178a7d", "^", ":"),
}
BARS_ORDER = ["CSA", "CSD", "GQI", "ODF-FP", "FORCE_a1e-03", "FORCE_a1e-04", "FORCE_a1e-05"]
BARS_LAB = ["CSA", "CSD", "GQI", "ODFFP", "FORCE\n(α=1e-3)", "FORCE\n(α=1e-4)", "FORCE\n(α=1e-5)"]
LINE_ORDER = ["ODF-FP", "CSD", "CSA", "GQI", "FORCE_a1e-05", "FORCE_a1e-04", "FORCE_a1e-03"]
LINE_LAB = {"ODF-FP": "ODFFP", "CSD": "CSD", "CSA": "CSA", "GQI": "GQI",
            "FORCE_a1e-05": "FORCE α=1e-05", "FORCE_a1e-04": "FORCE α=1e-04",
            "FORCE_a1e-03": "FORCE α=1e-03"}
BUCK = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
BUCK_LAB = [f"{lo}-{hi}°" for lo, hi in BUCK]
SNRS = [50, 20, 10]

nufo = pd.read_csv(f"{HERE}/out/nufo_accuracy.csv")

d = np.load(f"{HERE}/data/crossings.npz")
a1, a2, snr, ta = d["axis1"], d["axis2"], d["snr"], d["true_angle"]
Nc = a1.shape[0]
bkt = np.array([min(7, int((x - 10) // 10)) for x in ta])

def ang(u, v):
    return np.rad2deg(np.arccos(np.clip(abs(float(u @ v)), 0, 1)))

def prune(dirs, vals):
    dirs = np.asarray(dirs, float).reshape(-1, 3); vals = np.asarray(vals, float).reshape(-1)
    ok = np.isfinite(dirs).all(1) & np.isfinite(vals) & (np.linalg.norm(np.nan_to_num(dirs), axis=1) > 1e-6)
    dirs, vals = dirs[ok], vals[ok]
    if len(dirs) == 0 or vals.max() <= 0:
        return dirs
    return dirs[vals >= 0.5 * vals.max()]

def recall(dirs, x1, x2, tol=20.0):
    if len(dirs) == 0:
        return 0.0
    used, c = set(), 0
    for t in (x1, x2):
        best, ba = -1, tol
        for i, dd in enumerate(dirs):
            if i in used:
                continue
            an = ang(dd, t)
            if an <= ba:
                ba, best = an, i
        if best >= 0:
            used.add(best); c += 1
    return c / 2.0

PDR = {}
for m in LINE_ORDER:
    z = np.load(f"{HERE}/out/peaks_{m}.npz")
    pdirs, pvals = z["peak_dirs"], z["peak_vals"]
    rr = np.array([recall(prune(pdirs[i], pvals[i]), a1[i], a2[i]) for i in range(Nc)])
    PDR[m] = {sv: [100 * rr[(snr == sv) & (bkt == b)].mean() if ((snr == sv) & (bkt == b)).any()
                   else np.nan for b in range(8)] for sv in SNRS}

fig = plt.figure(figsize=(14.5, 14.2))
gs = fig.add_gridspec(3, 2, left=0.075, right=0.975, top=0.895, bottom=0.058,
                      hspace=0.40, wspace=0.165)
axL = [fig.add_subplot(gs[r, 0]) for r in range(3)]
axR = [fig.add_subplot(gs[r, 1]) for r in range(3)]

x = np.arange(len(BARS_ORDER)); w = 0.27
for r, sv in enumerate(SNRS):

    ax = axL[r]; xb = np.arange(8)
    for m in LINE_ORDER:
        c, mk, ls = LINE[m]
        ax.plot(xb, PDR[m][sv], ls, color=c, marker=mk, ms=7, lw=1.8)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Peak Detection Rate", fontsize=12)
    ax.set_xticks(xb); ax.set_xticklabels(BUCK_LAB, fontsize=10)
    ax.grid(color="0.9", lw=0.6); ax.set_axisbelow(True)

    ax = axR[r]
    for k, nn in enumerate((1, 2, 3)):
        vals = [nufo[(nufo.snr == sv) & (nufo.method == m) & (nufo.true_N == nn)].accuracy.values[0] * 100
                for m in BARS_ORDER]
        ax.bar(x + (k - 1) * w, vals, w, color=BAR[nn], edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(BARS_LAB, fontsize=10)
    ax.set_xlim(-0.55, len(BARS_ORDER) - 0.45)
    ax.grid(axis="y", color="0.85", lw=0.7); ax.set_axisbelow(True)

fig.canvas.draw()
for r, sv in enumerate(SNRS):
    p = axL[r].get_position()
    fig.text(0.5, p.y1 + 0.011, f"SNR={sv}", ha="center", va="bottom",
             fontsize=16, fontweight="bold")
for r in range(2):
    y = (axL[r].get_position().y0 + axL[r + 1].get_position().y1) / 2
    fig.add_artist(Line2D([0.03, 0.97], [y, y], color="0.55", lw=1.2, transform=fig.transFigure))

fig.text(axL[2].get_position().x0 + axL[2].get_position().width / 2, 0.016,
         "Crossing Angles Buckets", ha="center", fontsize=14)
fig.text(axR[2].get_position().x0 + axR[2].get_position().width / 2, 0.016,
         "Methods", ha="center", fontsize=14)

def _h(m):
    return Line2D([0], [0], color=LINE[m][0], marker=LINE[m][1], linestyle=LINE[m][2],
                 markersize=8, lw=1.8, label=LINE_LAB[m])

inv = ["ODF-FP", "CSD", "CSA", "GQI"]
frc = ["FORCE_a1e-05", "FORCE_a1e-04", "FORCE_a1e-03"]
xl = axL[0].get_position().x0
leg_inv = fig.legend(handles=[_h(m) for m in inv], loc="upper left",
                     bbox_to_anchor=(xl, 0.985), ncol=4, frameon=False,
                     fontsize=10.5, handletextpad=0.4, columnspacing=1.4)
fig.add_artist(leg_inv)
fig.legend(handles=[_h(m) for m in frc], loc="upper left",
           bbox_to_anchor=(xl, 0.958), ncol=3, frameon=False,
           fontsize=10.5, handletextpad=0.4, columnspacing=1.4)

nh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BAR[n], markersize=11,
             label=f"N={n}") for n in (1, 2, 3)]
fig.legend(handles=nh, loc="upper right", bbox_to_anchor=(0.975, 0.978),
           ncol=3, frameon=False, fontsize=12, handletextpad=0.3, columnspacing=1.5)

fig.savefig(f"{HERE}/figures/reference_style_figure.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("saved figures/reference_style_figure.png")
