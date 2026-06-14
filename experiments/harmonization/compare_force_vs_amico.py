"""Paired t-tests and comparison figure: FORCE vs AMICO-NODDI on matched metrics.

Mapping: NDI↔ND, ODI↔Dispersion, FWF↔CSF Fraction
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("HARMONIZATION_ROOT", Path(__file__).resolve().parent))
ALIGNED_ROOT = Path(os.environ.get("HARMONIZATION_ALIGNED", ROOT / "output_aligned"))
RESULTS = Path(os.environ.get("HARMONIZATION_RESULTS_ALIGNED",
                              ROOT / "harmonization_results_aligned"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
UNIQUE = ["ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
          "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1"]
RETEST = ["ses-c10r1", "ses-c10r2", "ses-c10r3"]

PAIRS = [
    ("Neurite density",        "FORCE", "force", "nd.nii.gz",
                                "AMICO", "AMICO/NODDI", "fit_NDI.nii.gz"),
    ("Orientation dispersion", "FORCE", "force", "dispersion.nii.gz",
                                "AMICO", "AMICO/NODDI", "fit_ODI.nii.gz"),
    ("Free-water / CSF",       "FORCE", "force", "csf_fraction.nii.gz",
                                "AMICO", "AMICO/NODDI", "fit_FWF.nii.gz"),
]


def load_vals(sub, ses, method_dir, fname, wm):
    p = ALIGNED_ROOT / sub / ses / method_dir / fname
    if not p.exists():
        return None
    v = nib.load(str(p)).get_fdata()
    if v.ndim == 4:
        v = v[..., 0]
    if v.shape != wm.shape:
        return None
    out = v[wm].astype(np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def cov(stack):
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1)
    valid = np.abs(mean) > 1e-10
    return float(np.nanmedian(std[valid] / np.abs(mean[valid])) * 100)


def icc31(stack):
    data = stack.T
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] > 50000:
        rng = np.random.default_rng(42)
        data = data[rng.choice(data.shape[0], 50000, replace=False)]
    n, k = data.shape
    grand = data.mean()
    ss_r = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_t = ((data - grand) ** 2).sum()
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    return float(np.clip((ms_r - ms_e) / (ms_r + (k - 1) * ms_e), 0, 1))


# Load WM masks
wm = {}
for sub in SUBJECTS:
    m = ALIGNED_ROOT / sub / "common_wm_mask.nii.gz"
    if m.exists():
        wm[sub] = nib.load(str(m)).get_fdata() > 0.5

rows = []
for pair_name, f_label, f_dir, f_name, a_label, a_dir, a_name in PAIRS:
    f_covs = []
    a_covs = []
    f_retest = []
    a_retest = []
    f_iccs = []
    a_iccs = []

    for sub in SUBJECTS:
        if sub not in wm:
            continue
        # Cross-scanner
        fs = [load_vals(sub, ses, f_dir, f_name, wm[sub]) for ses in UNIQUE]
        as_ = [load_vals(sub, ses, a_dir, a_name, wm[sub]) for ses in UNIQUE]
        fs = [x for x in fs if x is not None]
        as_ = [x for x in as_ if x is not None]
        if len(fs) >= 3 and len(as_) >= 3:
            f_covs.append(cov(np.array(fs)))
            a_covs.append(cov(np.array(as_)))
            f_iccs.append(icc31(np.array(fs)))
            a_iccs.append(icc31(np.array(as_)))
        # Test-retest
        frt = [load_vals(sub, ses, f_dir, f_name, wm[sub]) for ses in RETEST]
        art = [load_vals(sub, ses, a_dir, a_name, wm[sub]) for ses in RETEST]
        frt = [x for x in frt if x is not None]
        art = [x for x in art if x is not None]
        if len(frt) >= 2 and len(art) >= 2:
            f_retest.append(cov(np.array(frt)))
            a_retest.append(cov(np.array(art)))

    t_cov, p_cov = sstats.ttest_rel(f_covs, a_covs) if len(f_covs) >= 2 else (np.nan, np.nan)
    t_icc, p_icc = sstats.ttest_rel(f_iccs, a_iccs) if len(f_iccs) >= 2 else (np.nan, np.nan)

    rows.append({
        "pair":            pair_name,
        "force_metric":    f_name.replace(".nii.gz", ""),
        "amico_metric":    a_name.replace(".nii.gz", ""),
        "force_cov_mean":  float(np.mean(f_covs)),
        "force_cov_std":   float(np.std(f_covs)),
        "amico_cov_mean":  float(np.mean(a_covs)),
        "amico_cov_std":   float(np.std(a_covs)),
        "force_retest":    float(np.mean(f_retest)) if f_retest else np.nan,
        "amico_retest":    float(np.mean(a_retest)) if a_retest else np.nan,
        "force_icc":       float(np.mean(f_iccs)),
        "amico_icc":       float(np.mean(a_iccs)),
        "t_cov_paired":    float(t_cov),
        "p_cov_paired":    float(p_cov),
        "t_icc_paired":    float(t_icc),
        "p_icc_paired":    float(p_icc),
    })

df = pd.DataFrame(rows)
(RESULTS / "statistics").mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS / "statistics" / "force_vs_amico.csv", index=False)

# Print table
print("\n" + "=" * 100)
print("FORCE vs AMICO-NODDI — matched metrics, anatomy-aligned, white matter")
print("=" * 100)
print(f"{'Pair':<26} {'FORCE CoV':>10} {'AMICO CoV':>10} {'FORCE ICC':>10} "
      f"{'AMICO ICC':>10} {'p(CoV)':>9} {'p(ICC)':>9}")
print("-" * 100)
for _, r in df.iterrows():
    print(f"{r['pair']:<26} "
          f"{r['force_cov_mean']:>8.2f}%  "
          f"{r['amico_cov_mean']:>8.2f}%  "
          f"{r['force_icc']:>10.3f} "
          f"{r['amico_icc']:>10.3f} "
          f"{r['p_cov_paired']:>9.4f} {r['p_icc_paired']:>9.4f}")
print("=" * 100)

# Figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("FORCE vs AMICO-NODDI: cross-scanner CoV and ICC (anatomy-aligned, WM)",
             fontweight="bold")
pairs = df["pair"].tolist()
x = np.arange(len(pairs))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, df["force_cov_mean"], w, yerr=df["force_cov_std"],
       color="#2ca02c", edgecolor="black", label="FORCE")
ax.bar(x + w/2, df["amico_cov_mean"], w, yerr=df["amico_cov_std"],
       color="#d62728", edgecolor="black", label="AMICO-NODDI")
ax.set_xticks(x)
ax.set_xticklabels([p.replace(" / ", "\n") for p in pairs], rotation=0, fontsize=9)
ax.set_ylabel("Cross-scanner CoV (%)")
ax.set_title("Cross-scanner variability (lower = better)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(x - w/2, df["force_retest"], w, color="#2ca02c", edgecolor="black",
       label="FORCE", hatch="///", alpha=0.7)
ax.bar(x + w/2, df["amico_retest"], w, color="#d62728", edgecolor="black",
       label="AMICO-NODDI", hatch="///", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([p.replace(" / ", "\n") for p in pairs], rotation=0, fontsize=9)
ax.set_ylabel("Test-retest CoV (%)")
ax.set_title("Irreducible floor (same scanner)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
ax.bar(x - w/2, df["force_icc"], w, color="#2ca02c", edgecolor="black", label="FORCE")
ax.bar(x + w/2, df["amico_icc"], w, color="#d62728", edgecolor="black",
       label="AMICO-NODDI")
ax.set_xticks(x)
ax.set_xticklabels([p.replace(" / ", "\n") for p in pairs], rotation=0, fontsize=9)
ax.set_ylabel("ICC(3,1)")
ax.set_title("Cross-scanner reliability (higher = better)")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
(RESULTS / "publication_figures").mkdir(parents=True, exist_ok=True)
fig.savefig(str(RESULTS / "publication_figures" / "force_vs_amico.png"),
            dpi=200, bbox_inches="tight")
print(f"\nSaved: {RESULTS / 'publication_figures' / 'force_vs_amico.png'}")
print(f"CSV:   {RESULTS / 'statistics' / 'force_vs_amico.csv'}")
