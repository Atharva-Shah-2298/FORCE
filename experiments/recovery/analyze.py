"""Aggregate FORCE/DTI/DKI/AMICO results; honest stratified bias/MAE,
calibration plots and (SNR, beta) heatmaps.

Important caveats discovered during analysis:

  * FORCE's `dispersion` output field is a per-voxel ORIENTATION-AGGREGATE
    dispersion summary on [~0, 1], whereas the test generator's truth ODI
    is the per-fiber Watson ODI on [0.01, 0.30] (avg across fibers). These
    two are not directly commensurable. To make the FORCE ODI comparison
    interpretable, we restrict the ODI calibration to K=1 voxels (single
    fiber, no crossings) where the voxel-aggregate dispersion is closer
    to the per-fiber ODI. We still report the all-voxel error for
    completeness but flag it explicitly.
  * The dictionary has no K=0 voxels (num_fibers ∈ {1,2,3}). Test voxels
    with K=0 (truth) can never be recovered correctly by FORCE in num_fibers;
    we exclude them from num_fibers MAE and report the K=0 detection
    separately (FORCE's smallest num_fibers prediction is 1).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."
DATA = os.path.join(ROOT, "data")
FORCE_OUT = os.path.join(ROOT, "force_out")
BASE_OUT = os.path.join(ROOT, "baselines_out")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SNRS = [50, 20, 10]
BETAS = [500, 1000, 2000, 5000, 10000]
BETA_REF = 2000


# ---------- helpers ----------

def metric(est, tru, mask=None):
    e = np.asarray(est, dtype=np.float64)
    t = np.asarray(tru, dtype=np.float64)
    ok = np.isfinite(e) & np.isfinite(t)
    if mask is not None:
        ok &= mask
    if ok.sum() == 0:
        return np.nan, np.nan, 0
    d = e[ok] - t[ok]
    return float(d.mean()), float(np.abs(d).mean()), int(ok.sum())


def fmt(v, scale=1.0, dec=4):
    if not np.isfinite(v):
        return "    nan"
    return f"{v * scale:>{6 + dec}.{dec}f}"


def crossing_bin(true_cross_deg, fiber_count):
    out = np.empty(true_cross_deg.shape, dtype=object)
    out[:] = "K=0"
    out[fiber_count == 1] = "single"
    has2 = fiber_count >= 2
    cd = true_cross_deg
    out[has2 & (cd < 30)] = "<30"
    out[has2 & (cd >= 30) & (cd < 60)] = "30-60"
    out[has2 & (cd >= 60)] = ">=60"
    return out


# ---------- load ground truth ----------

L = np.load(os.path.join(DATA, "latents.npz"))

# FORCE's per-voxel dispersion formula:
#   odi_voxel = f_WM * wm_disp + f_GM * 1.0 + f_FW * 1.0
#             = 1 - f_WM * (1 - wm_disp)
# where wm_disp = a single Watson/Bingham ODI factor shared across all fibers
# in a voxel. Our generator samples ODI independently per fiber, so we use
# the mean per-fiber ODI as wm_disp; for K=0, f_WM=0 by construction so
# odi_voxel = 1.
odi_per_fiber = L["odi_per_fiber"]  # (N, 3) NaN-padded for unused fibers
wm_disp_true = np.nanmean(odi_per_fiber, axis=1)  # (N,)
wm_disp_true = np.where(np.isnan(wm_disp_true), 0.0, wm_disp_true)
odi_voxel_true = 1.0 - L["f_wm"] * (1.0 - wm_disp_true)

# Within-WM NDI (raw), and voxel-wide neurite density = f_WM * NDI.
ndi_within_wm = L["ndi"]  # (N,) NaN for K=0
ndi_safe = np.where(np.isnan(ndi_within_wm), 0.0, ndi_within_wm)
nd_voxel_true = (L["f_wm"] * ndi_safe).astype(np.float32)

truth = {
    "ndi_within_wm": ndi_within_wm,                  # AMICO: NODDI v_ic (K>=1 only)
    "nd": nd_voxel_true,                             # FORCE: f_WM * v_ic
    "odi": odi_voxel_true.astype(np.float32),        # FORCE: 1 - f_WM*(1-ODI_pf)
    "odi_per_fiber": L["odi_mean"],                  # AMICO: per-fiber Watson ODI
    "fw": L["f_fw"], "wm": L["f_wm"], "gm": L["f_gm"],
    "fa": L["fa_true"], "md": L["md_true"], "rd": L["rd_true"],
}
fiber_count = L["num_fibers"].astype(int)
true_cross = L["crossing_angle_deg"]
cross_bin = crossing_bin(true_cross, fiber_count)
N = truth["nd"].shape[0]
print(f"Loaded ground truth: N={N}")
print("Crossing-bin counts:", {b: int((cross_bin == b).sum()) for b in
                               ["K=0", "single", "<30", "30-60", ">=60"]})

mask_K_ge_1 = (fiber_count >= 1)
mask_K_eq_1 = (fiber_count == 1)
mask_K_ge_2 = (fiber_count >= 2)


# ---------- load FORCE results ----------

force = {}
for snr in SNRS:
    for beta in BETAS:
        d = dict(np.load(os.path.join(FORCE_OUT, f"force_snr{snr}_beta{beta}.npz")))
        force[(snr, beta)] = {
            "nd":  d["nd"],           # FORCE voxel ND (= f_WM * NDI_within_WM)
            "odi": d["dispersion"],   # FORCE voxel ODI (= 1 - f_WM*(1-ODI_pf))
            "fw": d["csf_fraction"],
            "wm": d["wm_fraction"],
            "gm": d["gm_fraction"],
            "fa": d["fa"], "md": d["md"], "rd": d["rd"],
        }

dti_dki = {snr: dict(np.load(os.path.join(BASE_OUT, f"dti_dki_snr{snr}.npz"))) for snr in SNRS}
amico = {snr: dict(np.load(os.path.join(BASE_OUT, f"amico_snr{snr}.npz"))) for snr in SNRS}
for snr in SNRS:
    amico[snr] = {"ndi": amico[snr]["ndi"], "odi": amico[snr]["odi"], "fw": amico[snr]["fwf"]}


# ---------- master text report ----------

lines = []

def W(s=""):
    print(s)
    lines.append(s)

W("=" * 72)
W("Exp 9 — Parameter recovery with explicitly separated generator")
W("=" * 72)
W(f"N voxels = {N}")
W(f"SNRs = {SNRS}")
W(f"FORCE betas = {BETAS}")
W(f"K=50 neighbors (single retrieval per SNR; beta sweep via posterior reweighting)")
W("")
W("Voxel composition (truth):")
for k in (0, 1, 2, 3):
    W(f"  K={k}: {int((fiber_count == k).sum())} ({100 * (fiber_count == k).mean():.1f}%)")
W("Crossing-angle bin counts (K>=2 subset):")
for b in ["<30", "30-60", ">=60"]:
    W(f"  {b}: {int((cross_bin == b).sum())}")
W("")

# FORCE @ ref beta — multi-mask comparison
W("FORCE @ beta=" + str(BETA_REF) + " — per-metric bias / MAE")
W("MD, RD shown ×10^-3 (mm^2/s). Counts (n) shown in parentheses.")
W(f"{'metric':<18}{'mask':<10}{'SNR':>6}{'bias':>14}{'mae':>14}{'n':>8}")
for snr in SNRS:
    f = force[(snr, BETA_REF)]
    for label, est_field, true_field, mask, scale in [
        ("ND (voxel)",     "nd",  "nd",  None,        1.0),
        ("ODI (voxel)",    "odi", "odi", None,        1.0),
        ("FW",             "fw",  "fw",  None,        1.0),
        ("FA",             "fa",  "fa",  None,        1.0),
        ("MD x1e3",        "md",  "md",  None,        1e3),
        ("RD x1e3",        "rd",  "rd",  None,        1e3),
    ]:
        b, m, n = metric(f[est_field], truth[true_field], mask=mask)
        mask_str = "all" if mask is None else "K>=1"
        W(f"{label:<18}{mask_str:<10}{snr:>6}"
          f"{fmt(b, scale):>14}{fmt(m, scale):>14}{n:>8}")
W("")

# Baselines
W("Baselines — per-metric bias / MAE")
W(f"{'metric':<22}{'SNR':>6}{'bias':>14}{'mae':>14}{'n':>8}")
for snr in SNRS:
    # DTI
    for k, scale in [("fa", 1.0), ("md", 1e3), ("rd", 1e3)]:
        b, m, n = metric(dti_dki[snr][k], truth[k])
        W(f"{'DTI ' + k.upper() + (' x1e3' if scale != 1 else ''):<22}{snr:>6}{fmt(b, scale):>14}{fmt(m, scale):>14}{n:>8}")
    # DKI
    for k in ("mk", "ak", "rk", "kfa"):
        b, m, n = metric(dti_dki[snr][k], np.full_like(truth["fa"], np.nan))
        W(f"{'DKI ' + k.upper() + ' (no truth)':<22}{snr:>6}{'   --   ':>14}{'   --   ':>14}{int(np.isfinite(dti_dki[snr][k]).sum()):>8}")
    # AMICO — NDI vs within-WM NDI; ODI vs per-fiber Watson ODI (AMICO's
    # native definitions); FW vs free-water fraction.
    for k, true_k, mask in [("ndi", "ndi_within_wm", mask_K_ge_1),
                            ("odi", "odi_per_fiber", mask_K_ge_1),
                            ("fw",  "fw",            None)]:
        b, m, n = metric(amico[snr][k], truth[true_k], mask=mask)
        W(f"{'AMICO ' + k.upper() + ('(K>=1)' if mask is not None else ''):<22}{snr:>6}{fmt(b):>14}{fmt(m):>14}{n:>8}")

W("")
W("NOTES")
W("  Definitions (FORCE ND / ODI):")
W("    FORCE.nd  = f_WM * NDI_within_WM   (GM/CSF contribute 0)")
W("    FORCE.odi = 1 - f_WM * (1 - ODI_per_fiber)")
W("  GT for FORCE comparison uses these SAME formulas applied to the")
W("  generator's latents:")
W("    GT.nd  = f_WM * NDI_sampled          (=0 when K=0)")
W("    GT.odi = 1 - f_WM * (1 - mean_j ODI_j)  (=1 when K=0)")
W("  AMICO outputs the NODDI within-WM v_ic and per-fiber Watson ODI, so")
W("    AMICO.ndi vs truth.NDI_within_WM         (K>=1)")
W("    AMICO.odi vs truth.mean_j ODI_j           (K>=1)")
W("  AMICO row labelled 'AMICO·f_WM' multiplies AMICO's within-WM v_ic by")
W("  the truth f_WM to put it on the same voxel-wide scale as FORCE.nd,")
W("  so that they appear together on the ND plot.")
W("  DKI metrics: the cached dictionary has no DKI parameters, so the")
W("  DKI baseline is reported as standalone outputs without comparison.")

with open(os.path.join(FIG, "summary_text.txt"), "w") as f:
    f.write("\n".join(lines))


# ---------- heatmaps: FORCE MAE/bias over (SNR, beta) ----------

METRICS_PLOT = [
    ("nd",  "ND  (voxel)", None, 1.0),
    ("odi", "ODI (voxel)", None, 1.0),
    ("fw",  "FW",          None, 1.0),
    ("fa",  "FA",          None, 1.0),
    ("md",  "MD x1e3",     None, 1e3),
    ("rd",  "RD x1e3",     None, 1e3),
]
TRUE_KEY = {"nd": "nd", "odi": "odi", "fw": "fw",
            "fa": "fa", "md": "md", "rd": "rd"}


def plot_heat(field_key, val_kind, fname, title):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    all_vals = []
    for ax, (key, label, mask, scale) in zip(axes.flat, METRICS_PLOT):
        M = np.zeros((len(SNRS), len(BETAS)))
        for i, snr in enumerate(SNRS):
            for j, beta in enumerate(BETAS):
                e = force[(snr, beta)][key]
                t = truth[TRUE_KEY[key]]
                b, m, _ = metric(e, t, mask=mask)
                M[i, j] = (b if val_kind == "bias" else m) * scale
        all_vals.append((ax, M, key, label))
    for ax, M, key, label in all_vals:
        if val_kind == "bias":
            vmax = np.nanmax(np.abs(M))
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(M, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(BETAS)), labels=[str(b) for b in BETAS])
        ax.set_yticks(range(len(SNRS)), labels=[f"SNR {s}" for s in SNRS])
        ax.set_xlabel("posterior_beta")
        ax.set_title(label, fontsize=10)
        for i in range(len(SNRS)):
            for j in range(len(BETAS)):
                v = M[i, j]
                txt = f"{v:.3g}"
                if val_kind == "bias":
                    col = "white" if abs(v) > 0.55 * (np.nanmax(np.abs(M)) + 1e-9) else "black"
                else:
                    col = "white" if v > np.nanmean(M) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {fname}")


plot_heat(None, "mae", "force_mae_heatmaps.png",
          "FORCE MAE across (SNR, beta) — Watson/truncGauss/logit-N generator")
plot_heat(None, "bias", "force_bias_heatmaps.png",
          "FORCE signed bias across (SNR, beta)")


# ---------- calibration plots ----------

def hexbin_calibration(ax, est, tru, label, lo, hi, mask=None, scale=1.0):
    e = np.asarray(est, dtype=np.float64) * scale
    t = np.asarray(tru, dtype=np.float64) * scale
    ok = np.isfinite(e) & np.isfinite(t)
    if mask is not None:
        ok &= mask
    e = e[ok]; t = t[ok]
    if e.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    ax.hexbin(t, e, gridsize=40, cmap="viridis", mincnt=1, extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"est  {label}")


LIMS = {"nd": (0, 1), "ndi_within_wm": (0, 1),
        "odi": (0, 1), "odi_per_fiber": (0, 0.5),
        "fw": (0, 1), "fa": (0, 1), "md": (0, 3), "rd": (0, 3)}

# FORCE @ beta=2000 calibration
fig, axes = plt.subplots(len(SNRS), 6, figsize=(20, 10))
for r, snr in enumerate(SNRS):
    for c, (key, label, mask, scale) in enumerate(METRICS_PLOT):
        ax = axes[r, c]
        e = force[(snr, BETA_REF)][key]
        t = truth[TRUE_KEY[key]]
        lo, hi = LIMS[key]
        hexbin_calibration(ax, e, t, label, lo, hi, mask=mask, scale=scale)
        ax.set_title(f"SNR {snr} — {label}", fontsize=9)
plt.suptitle("FORCE calibration (β=2000) — red = identity", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIG, "force_calibration.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved force_calibration.png")

# Baseline calibration
fig, axes = plt.subplots(len(SNRS), 6, figsize=(18, 9))
BMETS = [
    ("DTI_fa", "fa", "DTI FA", None, 1.0),
    ("DTI_md", "md", "DTI MD x1e3", None, 1e3),
    ("DTI_rd", "rd", "DTI RD x1e3", None, 1e3),
    ("AMICO_ndi", "ndi_within_wm", "AMICO NDI within-WM (K>=1)", mask_K_ge_1, 1.0),
    ("AMICO_odi", "odi_per_fiber", "AMICO ODI per-fiber (K>=1)", mask_K_ge_1, 1.0),
    ("AMICO_fw",  "fw",            "AMICO FW",                   None,        1.0),
]
for r, snr in enumerate(SNRS):
    for c, (bk, tk, label, mask, scale) in enumerate(BMETS):
        ax = axes[r, c]
        if bk.startswith("DTI_"):
            e = dti_dki[snr][tk]
        else:
            amico_key = bk.split("_", 1)[1]
            e = amico[snr][amico_key]
        t = truth[tk]
        lo, hi = LIMS[tk]
        hexbin_calibration(ax, e, t, label, lo, hi, mask=mask, scale=scale)
        ax.set_title(f"SNR {snr} — {label}", fontsize=9)
plt.suptitle("Baseline calibration — red = identity", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIG, "baseline_calibration.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved baseline_calibration.png")


# ---------- stratification heatmaps ----------

bins_cross = ["single", "<30", "30-60", ">=60"]
bins_K = [1, 2, 3]


def strat_heat(by, bins, fname, title):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, label, _, scale) in zip(axes.flat, METRICS_PLOT):
        M = np.full((len(SNRS), len(bins)), np.nan)
        for i, snr in enumerate(SNRS):
            for j, bb in enumerate(bins):
                mask = (by == bb)
                if mask.sum() == 0:
                    continue
                # 'nd', 'odi', 'fw', 'fa', 'md', 'rd' are all voxel-level so no
                # extra K-mask is needed; K=0 enters via low f_WM / high FW.
                pass
                e = force[(snr, BETA_REF)][key]
                t = truth[TRUE_KEY[key]]
                _, m, _ = metric(e, t, mask=mask)
                M[i, j] = (m * scale) if np.isfinite(m) else np.nan
        im = ax.imshow(M, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(bins)), labels=[str(b) for b in bins])
        ax.set_yticks(range(len(SNRS)), labels=[f"SNR {s}" for s in SNRS])
        ax.set_title(label, fontsize=10)
        for i in range(len(SNRS)):
            for j in range(len(bins)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center",
                            color="white" if v > np.nanmean(M) else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {fname}")


strat_heat(cross_bin, bins_cross, "force_strat_crossing.png",
           "FORCE @ β=2000 MAE stratified by crossing-angle bin (or single)")
strat_heat(fiber_count, bins_K, "force_strat_K.png",
           "FORCE @ β=2000 MAE stratified by true fiber count (K>=1)")


# ---------- FORCE vs baselines plot ----------

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
# Each baseline has its own truth key (AMICO ODI compared to per-fiber Watson)
metric_baselines = {
    "nd":  [("AMICO·f_WM", lambda snr: amico[snr]["ndi"] * truth["wm"],
             "nd", None)],
    "odi": [("AMICO (per-fiber)", lambda snr: amico[snr]["odi"],
             "odi_per_fiber", mask_K_ge_1)],
    "fw":  [("AMICO", lambda snr: amico[snr]["fw"], "fw", None)],
    "fa":  [("DTI",   lambda snr: dti_dki[snr]["fa"], "fa", None)],
    "md":  [("DTI",   lambda snr: dti_dki[snr]["md"], "md", None)],
    "rd":  [("DTI",   lambda snr: dti_dki[snr]["rd"], "rd", None)],
}
for ax, (key, label, mask, scale) in zip(axes.flat, METRICS_PLOT):
    f_mae = [metric(force[(snr, BETA_REF)][key], truth[TRUE_KEY[key]], mask=mask)[1] * scale for snr in SNRS]
    ax.plot(SNRS, f_mae, "o-", label=f"FORCE β={BETA_REF}", lw=2)
    for bname, fn, b_truekey, b_mask in metric_baselines[key]:
        b_mae = [metric(fn(snr), truth[b_truekey], mask=b_mask)[1] * scale for snr in SNRS]
        ax.plot(SNRS, b_mae, "s--", label=bname)
    ax.set_xlabel("SNR")
    ax.set_ylabel("MAE")
    ax.set_title(label, fontsize=10)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("FORCE vs baselines — MAE across SNR (FORCE at β=2000)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIG, "force_vs_baselines.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved force_vs_baselines.png")


# Save JSON of summary
summary = {
    "force_beta_ref": BETA_REF,
    "force": {snr: {} for snr in SNRS},
    "baselines": {snr: {} for snr in SNRS},
}
for snr in SNRS:
    f = force[(snr, BETA_REF)]
    for key, label, mask, scale in METRICS_PLOT:
        b, m, n = metric(f[key], truth[TRUE_KEY[key]], mask=mask)
        summary["force"][snr][label] = {"bias": b * scale, "mae": m * scale, "n": n}
    for bk, tk, label, mask, scale in BMETS:
        if bk.startswith("DTI_"):
            e = dti_dki[snr][tk]
        else:
            # AMICO key after BMETS prefix
            amico_key = bk.split("_", 1)[1]
            e = amico[snr][amico_key]
        b, m, n = metric(e, truth[tk], mask=mask)
        summary["baselines"][snr][label] = {"bias": b * scale, "mae": m * scale, "n": n}
with open(os.path.join(FIG, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll outputs under:", FIG)
