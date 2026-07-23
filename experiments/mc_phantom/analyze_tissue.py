"""Per-tissue-type FORCE recovery: fiber count, WM/GM/CSF fractions, NDI/ODI."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORCE_OUT = os.path.join(HERE, "force_out")
FIG = os.path.join(HERE, "figures")

gt = np.load(os.path.join(DATA, "ground_truth_tissue.npz"), allow_pickle=True)
f = dict(np.load(os.path.join(FORCE_OUT, "force_tissue.npz")))
typ = gt["type"].astype(str)
wm, gm, csf = f["wm_fraction"], f["gm_fraction"], f["csf_fraction"]
N = len(typ)
labels = [f"{typ[i]}\n" + (f"{int(gt['cross_deg'][i])}°" if np.isfinite(gt['cross_deg'][i])
          else f"wm{gt['f_wm'][i]:.1f}/gm{gt['f_gm'][i]:.1f}/csf{gt['f_fw'][i]:.1f}")
          for i in range(N)]

fig, axes = plt.subplots(2, 1, figsize=(15, 9), height_ratios=[2, 1])

# --- fractions: true (hatched) vs FORCE (solid), 3 compartments ---
ax = axes[0]
x = np.arange(N); w = 0.13
for k, (tk, ek, col, lab) in enumerate([
        ("f_wm", wm, "C0", "WM"), ("f_gm", gm, "C2", "GM"), ("f_fw", csf, "C3", "CSF")]):
    ax.bar(x + (k - 1) * 2 * w - w / 2, gt[tk].astype(float), w, color=col,
           alpha=0.45, hatch="//", label=f"{lab} true" if k == 0 else None)
    ax.bar(x + (k - 1) * 2 * w + w / 2, ek, w, color=col,
           label=f"{lab} FORCE" if k == 0 else None)
ax.bar(0, 0, color="grey", alpha=0.45, hatch="//", label="true (hatched)")
ax.bar(0, 0, color="grey", label="FORCE (solid)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("compartment fraction"); ax.set_ylim(0, 1.05)
ax.legend(ncol=4, fontsize=8); ax.set_title(
    "FORCE compartment-fraction recovery by tissue type "
    "(WM=blue GM=green CSF=red; hatched=true, solid=FORCE)")
for i in range(N):
    ax.axvline(i + 0.5, color="grey", lw=0.3, alpha=0.3)

# --- fiber count: true vs ODF peaks ---
ax = axes[1]
ax.bar(x - 0.18, gt["num_fibers"].astype(float), 0.36, color="k", alpha=0.4,
       hatch="//", label="true # fibers")
ax.bar(x + 0.18, f["n_peaks"], 0.36, color="C1", label="FORCE ODF peaks")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("# fibers"); ax.set_yticks([0, 1, 2, 3])
ax.legend(fontsize=9); ax.set_title("Fiber-count recovery (ODF peaks)")
for i in range(N):
    cr = f["cross_deg"][i]
    if np.isfinite(cr):
        ax.text(i + 0.18, f["n_peaks"][i] + 0.05, f"{cr:.0f}°", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "tissue_recovery.png"), dpi=150, bbox_inches="tight")
print("saved figures/tissue_recovery.png")

# text table
ndi_e = np.clip(f["nd"] / np.maximum(wm, 1e-6), 0, 1)
odi_e = np.clip(1 - (1 - f["dispersion"]) / np.maximum(wm, 1e-6), 0, 1)
lines = ["Per-tissue-type FORCE recovery (clean MC substrates)\n" + "=" * 70]
hdr = f"{'type':<10}{'nf_t':>4}{'peaks':>6}{'WMt/e':>10}{'GMt/e':>10}{'CSFt/e':>10}{'NDIt/e':>10}{'ODIt/e':>10}"
lines.append(hdr)
for i in range(N):
    lines.append(
        f"{typ[i]:<10}{gt['num_fibers'][i]:>4.0f}{f['n_peaks'][i]:>6d}"
        f"{gt['f_wm'][i]:>5.1f}/{wm[i]:<4.2f}{gt['f_gm'][i]:>5.1f}/{gm[i]:<4.2f}"
        f"{gt['f_fw'][i]:>5.1f}/{csf[i]:<4.2f}"
        f"{gt['ndi'][i]:>5.2f}/{ndi_e[i]:<4.2f}{gt['odi'][i]:>5.2f}/{odi_e[i]:<4.2f}")
txt = "\n".join(lines)
print("\n" + txt)
with open(os.path.join(FIG, "tissue_summary.txt"), "w") as fp:
    fp.write(txt)
