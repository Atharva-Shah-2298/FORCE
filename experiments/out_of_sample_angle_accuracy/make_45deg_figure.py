import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

d = np.load(f"{HERE}/data/crossings.npz")
a1, a2, snr, ta = d["axis1"], d["axis2"], d["snr"], d["true_angle"]
N = a1.shape[0]
bkt = np.array([min(7, int((x - 10) // 10)) for x in ta])
BUCK = [f"{lo}-{hi}°" for lo, hi in [(10, 20), (20, 30), (30, 40), (40, 50),
                                     (50, 60), (60, 70), (70, 80), (80, 90)]]
SNRS = [50, 20, 10]

METH = {
    "FORCE":  ("peaks_FORCE_a1e-05", "#2ca02c", "o", 2.7),
    "ODF-FP": ("peaks_ODF-FP",       "#2ba3e0", "s", 1.9),
    "CSD":    ("peaks_CSD_45",       "#28348f", "X", 1.9),
    "CSA":    ("peaks_CSA_45",       "#e8781e", "^", 1.9),
    "GQI":    ("peaks_GQI_45",       "#7d2b8f", "D", 1.9),
}

def ang(u, v):
    return np.rad2deg(np.arccos(np.clip(abs(float(u @ v)), 0, 1)))

def prune(dr, vl):
    dr = np.asarray(dr, float).reshape(-1, 3); vl = np.asarray(vl, float).reshape(-1)
    ok = np.isfinite(dr).all(1) & np.isfinite(vl) & (np.linalg.norm(np.nan_to_num(dr), axis=1) > 1e-6)
    dr, vl = dr[ok], vl[ok]
    if len(dr) == 0 or vl.max() <= 0:
        return dr
    return dr[vl >= 0.5 * vl.max()]

def recall(dr, x1, x2, tol=20.0):
    if len(dr) == 0:
        return 0.0
    used, c = set(), 0
    for t in (x1, x2):
        best, ba = -1, tol
        for i, dd in enumerate(dr):
            if i in used:
                continue
            an = ang(dd, t)
            if an <= ba:
                ba, best = an, i
        if best >= 0:
            used.add(best); c += 1
    return c / 2.0

PDR = {}
for m, (f, *_ ) in METH.items():
    z = np.load(f"{HERE}/out/{f}.npz")
    pdd, pvv = z["peak_dirs"], z["peak_vals"]
    rr = np.array([recall(prune(pdd[i], pvv[i]), a1[i], a2[i]) for i in range(N)])
    PDR[m] = {sv: [100 * rr[(snr == sv) & (bkt == b)].mean() if ((snr == sv) & (bkt == b)).any()
                   else np.nan for b in range(8)] for sv in SNRS}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
xb = np.arange(8)
for ax, sv in zip(axes, SNRS):
    ax.axhspan(0, 50, color="0.9", alpha=0.5, zorder=0)
    for m, (f, c, mk, lw) in METH.items():
        ax.plot(xb, PDR[m][sv], "-", color=c, marker=mk, ms=7, lw=lw,
                label=m, zorder=5 if m == "FORCE" else 3)
    ax.axhline(50, color="0.6", ls=":", lw=1, zorder=1)
    ax.set_title(f"SNR {sv}", fontsize=13, fontweight="bold")
    ax.set_xticks(xb); ax.set_xticklabels(BUCK, fontsize=9.5, rotation=35, ha="right")
    ax.set_ylim(40, 102); ax.set_yticks(range(40, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(40, 101, 10)])
    ax.set_xlabel("crossing angle", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("peak detection rate", fontsize=12)
axes[0].legend(fontsize=10.5, loc="upper left", framealpha=0.95)
axes[0].text(0.02, 0.12, "50% = single fibre detected", transform=axes[0].transAxes,
             fontsize=8.5, color="0.4", style="italic")
fig.suptitle("Peak detection on out-of-sample Monte Carlo crossings "
             "(45° minimum peak separation for the inverse methods)",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{HERE}/figures/peak_detection_45deg.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("saved figures/peak_detection_45deg.png")
