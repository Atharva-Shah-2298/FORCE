"""Compare FORCE recovery against the Monte Carlo substrate ground truth.

Reports bias/MAE for ND, ODI, FW, FA, MD, RD at each noise level, plus a
breakdown by axon radius (the radius is invisible to the substrate's ICVF
but drives the restricted-diffusion model mismatch FORCE's stick dictionary
cannot represent — so it is the interesting axis). Saves calibration plots.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORCE_OUT = os.path.join(HERE, "force_out")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
TAGS = ["clean", "snr50", "snr20"]


def bias_mae(est, tru, m=None):
    e = np.asarray(est, float); t = np.asarray(tru, float)
    ok = np.isfinite(e) & np.isfinite(t)
    if m is not None:
        ok &= m
    if ok.sum() == 0:
        return np.nan, np.nan, 0
    dd = e[ok] - t[ok]
    return float(dd.mean()), float(np.abs(dd).mean()), int(ok.sum())


gt = np.load(os.path.join(DATA, "ground_truth.npz"))
radius = gt["radius_um"]; icvf = gt["icvf"]; f_fw = gt["f_fw"]
N = gt["nd"].shape[0]
# Truth. ODI must be compared on the ANISOTROPIC compartment only, like NODDI:
#   FORCE.dispersion is a 3-compartment voxel composite
#     odi_voxel = 1 - f_wm*(1 - odi_wm), with GM & CSF each contributing 1.
#   The substrate bundle is coherent => true within-WM fiber dispersion = 0.
# Isotropic fraction truth = free water only (no GM in the substrate).
truth = {"nd": gt["nd"], "odi_comp": gt["odi"], "odi_wm": np.zeros(N),
         "fw": gt["fw"], "iso": f_fw, "fa": gt["fa"], "md": gt["md"],
         "rd": gt["rd"], "ndi_within_wm": gt["ndi_within_wm"]}

force = {}
for tag in TAGS:
    p = os.path.join(FORCE_OUT, f"force_mc_{tag}.npz")
    if not os.path.exists(p):
        continue
    d = dict(np.load(p))
    wmf = np.maximum(d["wm_fraction"], 1e-6)
    odi_wm = np.clip(1.0 - (1.0 - d["dispersion"]) / wmf, 0.0, 1.0)  # back out anisotropic ODI
    iso = d["gm_fraction"] + d["csf_fraction"]                       # total isotropic frac
    force[tag] = {"nd": d["nd"], "odi_comp": d["dispersion"], "odi_wm": odi_wm,
                  "fw": d["csf_fraction"], "iso": iso,
                  "fa": d["fa"], "md": d["md"], "rd": d["rd"]}

ROWS = [("ND (voxel)", "nd", 1.0),
        ("ODI anisotrop", "odi_wm", 1.0),   # NODDI-comparable orientation dispersion
        ("ODI composite", "odi_comp", 1.0),  # FORCE 3-compartment voxel composite (ref)
        ("iso frac GM+CSF", "iso", 1.0),     # isotropic-fraction leakage (truth = f_fw)
        ("FW (CSF only)", "fw", 1.0), ("FA", "fa", 1.0),
        ("MD x1e3", "md", 1e3), ("RD x1e3", "rd", 1e3)]
radii = sorted(set(radius.tolist()))

lines = []
def W(s=""):
    print(s); lines.append(s)

W("=" * 74)
W("Exp 10 — FORCE recovery on a Monte Carlo (true restricted-diffusion) phantom")
W("=" * 74)
W(f"N voxels = {N};  substrates = ICVF{sorted(set(icvf.round(2).tolist()))} "
  f"x radius{radii}um  x FW{sorted(set(f_fw.tolist()))}")
W("Truth: ndi_within_wm=ICVF (geometry), coherent bundle (ODI~0), "
  "FW added as CSF compartment.")
W("MD, RD x1e3 mm^2/s. The dictionary uses a zero-radius STICK intra model,")
W("so finite axon radius is a genuine forward-model mismatch.\n")

summary = {}
for tag in force:
    W(f"--- {tag} ---")
    W(f"{'metric':<14}{'bias':>10}{'mae':>10}{'  | by radius (MAE)':<24}")
    summary[tag] = {}
    for label, key, sc in ROWS:
        b, m, n = bias_mae(force[tag][key], truth[key])
        by_r = []
        for r in radii:
            _, mr, _ = bias_mae(force[tag][key], truth[key], m=(radius == r))
            by_r.append(f"r{r:g}:{mr*sc:.3f}")
        W(f"{label:<14}{b*sc:>10.4f}{m*sc:>10.4f}   {'  '.join(by_r)}")
        summary[tag][label] = {"bias": b*sc, "mae": m*sc, "n": n}
    W("")

with open(os.path.join(FIG, "mc_summary.txt"), "w") as f:
    f.write("\n".join(lines))
with open(os.path.join(FIG, "mc_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---- calibration: recovered NDI(=ND/f_wm) and ND vs true, by radius ----
colors = {radii[0]: "C0"}
if len(radii) > 1:
    colors[radii[1]] = "C3"
fig, axes = plt.subplots(len(force), 3, figsize=(14, 4 * len(force)),
                         squeeze=False)
for r_i, tag in enumerate(force):
    for c, (label, key, lim) in enumerate(
            [("ND (voxel)", "nd", (0, 0.8)),
             ("ODI anisotropic (NODDI-like)", "odi_wm", (0, 0.3)),
             ("iso frac GM+CSF (truth=FW)", "iso", (0, 0.6))]):
        ax = axes[r_i][c]
        for r in radii:
            m = radius == r
            ax.scatter(truth[key][m], force[tag][key][m], s=45, c=colors[r],
                       alpha=0.7, label=f"r={r:g}um", edgecolors="k", linewidths=0.3)
        lo, hi = lim
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"true {label}"); ax.set_ylabel(f"FORCE {label}")
        ax.set_title(f"{tag} — {label}", fontsize=10)
        if r_i == 0 and c == 0:
            ax.legend(fontsize=8)
plt.suptitle("FORCE recovery vs Monte Carlo ground truth (red = identity)",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(FIG, "mc_calibration.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved figures/mc_calibration.png")

# ---- headline: recovered NDI-within-WM vs true ICVF (noise-free) ----
tag0 = "clean" if "clean" in force else list(force)[0]
fig, ax = plt.subplots(figsize=(6, 6))
ndi_est = force[tag0]["nd"] / np.maximum(1 - f_fw, 1e-6)   # back out within-WM NDI
for r in radii:
    m = radius == r
    ax.scatter(icvf[m], ndi_est[m], s=70, c=colors[r], alpha=0.8,
               edgecolors="k", linewidths=0.4, label=f"r={r:g}um")
ax.plot([0, 0.8], [0, 0.8], "r--", lw=1)
ax.set_xlabel("true ICVF (substrate geometry)")
ax.set_ylabel("FORCE NDI within-WM (= nd / f_wm)")
ax.set_title(f"FORCE neurite density vs MC intra-axonal volume fraction ({tag0})")
ax.legend(); ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "mc_ndi_vs_icvf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved figures/mc_ndi_vs_icvf.png")
print("\nOutputs under", FIG)
