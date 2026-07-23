"""Joint recovery: NDI, ODI, FW, FA, MD, RD all from the SAME 60-voxel MC
phantom where ICVF, ODI and FW vary together. Reports bias/MAE per metric and
the key cross-interactions (does FW bias NDI/ODI? does ODI bias FW?).
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
TAGS = ["clean", "snr20"]


def bm(e, t, m=None):
    e = np.asarray(e, float); t = np.asarray(t, float)
    ok = np.isfinite(e) & np.isfinite(t)
    if m is not None:
        ok &= m
    d = e[ok] - t[ok]
    return float(d.mean()), float(np.abs(d).mean()), int(ok.sum())


gt = np.load(os.path.join(DATA, "ground_truth_joint.npz"))
icvf, odi_t, fw_t = gt["icvf"], gt["odi_wm"], gt["fw"]
truth = {"nd": gt["nd"], "ndi_within_wm": gt["ndi_within_wm"], "odi_wm": odi_t,
         "fw": fw_t, "fa": gt["fa"], "md": gt["md"], "rd": gt["rd"]}
N = truth["nd"].shape[0]

force = {}
for tag in TAGS:
    d = dict(np.load(os.path.join(FORCE_OUT, f"force_joint_{tag}.npz")))
    wmf = np.maximum(d["wm_fraction"], 1e-6)
    force[tag] = {
        "nd": d["nd"],
        "ndi_within_wm": d["nd"] / wmf,                       # within-WM NDI
        "odi_wm": np.clip(1 - (1 - d["dispersion"]) / wmf, 0, 1),
        "fw": d["csf_fraction"],
        "gm": d["gm_fraction"],
        "iso": d["gm_fraction"] + d["csf_fraction"],
        "fa": d["fa"], "md": d["md"], "rd": d["rd"],
    }

# truth has NO GM compartment (substrate is WM + free-water@3.0e-3 = CSF only),
# so true GM=0 and true total-isotropic = FW.
truth["gm"] = np.zeros(N)
truth["iso"] = fw_t
ROWS = [("NDI within-WM", "ndi_within_wm", 1.0),
        ("ND (voxel)", "nd", 1.0),
        ("ODI (anisotropic)", "odi_wm", 1.0),
        ("CSF frac (=FW)", "fw", 1.0),
        ("GM frac (truth=0)", "gm", 1.0),
        ("iso frac GM+CSF", "iso", 1.0),
        ("FA", "fa", 1.0), ("MD x1e3", "md", 1e3), ("RD x1e3", "rd", 1e3)]

lines = []
def W(s=""):
    print(s); lines.append(s)

W("=" * 76)
W("Exp 10 JOINT — NDI, ODI, FW recovered together on one MC phantom")
W("=" * 76)
W(f"N = {N} voxels = ICVF{sorted(set(icvf.round(2).tolist()))} x "
  f"ODI{sorted(set(odi_t.round(2).tolist()))} x FW{sorted(set(fw_t.round(2).tolist()))}")
W("All metrics from the SAME voxels. ODI on anisotropic compartment "
  "(=1-(1-disp)/f_wm); true ODI<=0.30 = dict-representable.\n")

summary = {}
for tag in TAGS:
    W(f"--- {tag} ---")
    W(f"{'metric':<20}{'bias':>9}{'mae':>9}{'  (MAE by FW level)':<28}")
    summary[tag] = {}
    for label, key, sc in ROWS:
        b, m, n = bm(force[tag][key], truth[key])
        byfw = "  ".join(
            f"fw{f:g}:{bm(force[tag][key], truth[key], m=np.isclose(fw_t, f))[1]*sc:.3f}"
            for f in sorted(set(fw_t.tolist())))
        W(f"{label:<20}{b*sc:>9.4f}{m*sc:>9.4f}   {byfw}")
        summary[tag][label] = {"bias": b*sc, "mae": m*sc, "n": n}
    W("")

# cross-interactions (clean)
f = force["clean"]
W("Cross-interactions (clean):")
for label, key in [("NDI within-WM", "ndi_within_wm"), ("ODI", "odi_wm")]:
    rho = np.corrcoef(f[key] - truth[key], fw_t)[0, 1]
    W(f"  corr( {label} error , true FW )      = {rho:+.2f}")
rho = np.corrcoef(f["fw"] - truth["fw"], odi_t)[0, 1]
W(f"  corr( FW error , true ODI )           = {rho:+.2f}")
rho = np.corrcoef(f["ndi_within_wm"] - truth["ndi_within_wm"], odi_t)[0, 1]
W(f"  corr( NDI error , true ODI )          = {rho:+.2f}")

with open(os.path.join(FIG, "joint_summary.txt"), "w") as fp:
    fp.write("\n".join(lines))
with open(os.path.join(FIG, "joint_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)

# calibration grid: rows = metrics, color = FW level
PLOT = [("NDI within-WM", "ndi_within_wm", (0, 0.8)),
        ("ODI (anisotropic)", "odi_wm", (0, 0.4)),
        ("FW", "fw", (0, 0.4)), ("FA", "fa", (0, 0.9))]
fwlev = sorted(set(fw_t.tolist()))
cmap = {fwlev[0]: "C0", fwlev[1]: "C1", fwlev[2]: "C3"}
fig, axes = plt.subplots(2, 4, figsize=(19, 9))
for r, tag in enumerate(TAGS):
    for c, (label, key, lim) in enumerate(PLOT):
        ax = axes[r, c]
        for fl in fwlev:
            m = np.isclose(fw_t, fl)
            ax.scatter(truth[key][m], force[tag][key][m], s=42, c=cmap[fl],
                       alpha=0.75, edgecolors="k", linewidths=0.3,
                       label=f"FW={fl:g}")
        lo, hi = lim
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        if key == "odi_wm":
            ax.axvspan(0.30, hi, color="grey", alpha=0.12)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"true {label}"); ax.set_ylabel(f"FORCE {label}")
        ax.set_title(f"{tag} — {label}", fontsize=10)
        if r == 0 and c == 0:
            ax.legend(fontsize=8)
plt.suptitle("FORCE joint recovery (NDI, ODI, FW vary together) — color = FW level",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(FIG, "joint_calibration.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved figures/joint_calibration.png")
print("Outputs under", FIG)
