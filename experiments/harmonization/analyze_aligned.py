"""
Recompute cross-scanner CoV and ICC on the aligned volumes within the
per-subject common WM mask.

This replaces the pathological voxel-index-aligned stats in
analyze_harmonization.py with anatomy-aligned ones.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("analyze_aligned")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HARMONIZATION_ROOT", SCRIPT_DIR))
ALIGNED_ROOT = Path(os.environ.get("HARMONIZATION_ALIGNED",
                                   ROOT / "output_aligned"))
RESULTS_DIR = Path(os.environ.get("HARMONIZATION_RESULTS_ALIGNED",
                                  ROOT / "harmonization_results_aligned"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
UNIQUE_SCANNER_SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
]
RETEST_SESSIONS = ["ses-c10r1", "ses-c10r2", "ses-c10r3"]

METHODS = {
    "DTI": {"dir": "dti", "metrics": {
        "FA": "fa.nii.gz", "MD": "md.nii.gz",
        "AD": "ad.nii.gz", "RD": "rd.nii.gz"}},
    "DKI": {"dir": "dki", "metrics": {
        "FA": "fa.nii.gz", "MD": "md.nii.gz",
        "AD": "ad.nii.gz", "RD": "rd.nii.gz",
        "MK": "mk.nii.gz", "AK": "ak.nii.gz", "RK": "rk.nii.gz"}},
    "FORCE": {"dir": "force", "metrics": {
        "FA": "fa.nii.gz", "MD": "md.nii.gz", "RD": "rd.nii.gz",
        "ND": "nd.nii.gz", "Dispersion": "dispersion.nii.gz",
        "WM Fraction": "wm_fraction.nii.gz",
        "GM Fraction": "gm_fraction.nii.gz",
        "CSF Fraction": "csf_fraction.nii.gz",
        "uFA": "ufa.nii.gz",
        "MK": "mk.nii.gz", "AK": "ak.nii.gz", "RK": "rk.nii.gz"}},
    "AMICO-NODDI": {"dir": "AMICO/NODDI", "metrics": {
        "NDI": "fit_NDI.nii.gz",
        "ODI": "fit_ODI.nii.gz",
        "FWF": "fit_FWF.nii.gz"}},
}


def load_metric(subject, session, method, filename, wm_mask):
    """Load a metric volume, return WM-masked 1D vector or None."""
    p = ALIGNED_ROOT / subject / session / METHODS[method]["dir"] / filename
    if not p.exists():
        return None
    vol = nib.load(str(p)).get_fdata().astype(np.float64)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.shape != wm_mask.shape:
        return None
    vals = vol[wm_mask]
    vals[~np.isfinite(vals)] = np.nan
    return vals


def compute_cov(stack):
    """stack: (n_sessions, n_voxels). Returns median per-voxel CoV (%)."""
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1)
    valid = np.abs(mean) > 1e-10
    cov = std[valid] / np.abs(mean[valid])
    return float(np.nanmedian(cov) * 100)


def compute_icc31(stack):
    """ICC(3,1) two-way mixed, consistency, single measures. stack: (n_ses, n_vox)."""
    data = stack.T  # (n_vox, n_ses) — "targets" × "raters"
    # Drop rows with NaN
    valid = np.all(np.isfinite(data), axis=1)
    data = data[valid]
    if data.shape[0] < 10 or data.shape[1] < 2:
        return float("nan")

    # Subsample for speed
    if data.shape[0] > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(data.shape[0], 50000, replace=False)
        data = data[idx]

    n, k = data.shape
    grand = data.mean()
    row_m = data.mean(axis=1)
    col_m = data.mean(axis=0)
    ss_t = ((data - grand) ** 2).sum()
    ss_r = k * ((row_m - grand) ** 2).sum()
    ss_c = n * ((col_m - grand) ** 2).sum()
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
    return float(np.clip(icc, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_dir = RESULTS_DIR / "statistics"
    figs_dir = RESULTS_DIR / "publication_figures"
    stats_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    # Load per-subject WM masks
    wm_masks = {}
    for sub in SUBJECTS:
        p = ALIGNED_ROOT / sub / "common_wm_mask.nii.gz"
        if not p.exists():
            log.error(f"Missing common WM mask for {sub}: {p}")
            continue
        wm_masks[sub] = nib.load(str(p)).get_fdata() > 0.5
        log.info(f"{sub}: WM mask has {wm_masks[sub].sum()} voxels")

    # Cross-scanner CoV + ICC (unique scanners)
    cov_rows = []
    icc_rows = []
    retest_rows = []

    for method, info in METHODS.items():
        for metric, fname in info["metrics"].items():
            cov_xsubj = []
            icc_xsubj = []
            retest_cov_xsubj = []
            for sub in SUBJECTS:
                if sub not in wm_masks:
                    continue
                wm = wm_masks[sub]

                # Cross-scanner (10 unique scanners)
                stack = []
                for ses in UNIQUE_SCANNER_SESSIONS:
                    v = load_metric(sub, ses, method, fname, wm)
                    if v is not None:
                        stack.append(v)
                if len(stack) >= 3:
                    S = np.array(stack)
                    cov_xsubj.append(compute_cov(S))
                    icc_xsubj.append(compute_icc31(S))

                # Test-retest (c10 r1/r2/r3)
                stack_rt = []
                for ses in RETEST_SESSIONS:
                    v = load_metric(sub, ses, method, fname, wm)
                    if v is not None:
                        stack_rt.append(v)
                if len(stack_rt) >= 2:
                    S_rt = np.array(stack_rt)
                    retest_cov_xsubj.append(compute_cov(S_rt))

            if cov_xsubj:
                cov_rows.append({
                    "method": method, "metric": metric,
                    "cov_mean": float(np.mean(cov_xsubj)),
                    "cov_std": float(np.std(cov_xsubj)),
                    "n_subjects": len(cov_xsubj),
                })
            if icc_xsubj:
                icc_rows.append({
                    "method": method, "metric": metric,
                    "icc_mean": float(np.mean(icc_xsubj)),
                    "icc_std": float(np.std(icc_xsubj)),
                    "n_subjects": len(icc_xsubj),
                })
            if retest_cov_xsubj:
                retest_rows.append({
                    "method": method, "metric": metric,
                    "cov_mean": float(np.mean(retest_cov_xsubj)),
                    "cov_std": float(np.std(retest_cov_xsubj)),
                    "n_subjects": len(retest_cov_xsubj),
                })
            log.info(f"{method:12s} {metric:12s}  "
                     f"xscan CoV={np.mean(cov_xsubj) if cov_xsubj else 'NA':>6}  "
                     f"ICC={np.mean(icc_xsubj) if icc_xsubj else 'NA':>6}  "
                     f"retest CoV={np.mean(retest_cov_xsubj) if retest_cov_xsubj else 'NA':>6}")

    cov_df = pd.DataFrame(cov_rows)
    icc_df = pd.DataFrame(icc_rows)
    retest_df = pd.DataFrame(retest_rows)
    cov_df.to_csv(stats_dir / "cross_scanner_cov_aligned.csv", index=False)
    icc_df.to_csv(stats_dir / "icc_aligned.csv", index=False)
    retest_df.to_csv(stats_dir / "retest_cov_aligned.csv", index=False)

    # Between-subject CoV (median per session, std across subjects)
    between_rows = []
    for method, info in METHODS.items():
        for metric, fname in info["metrics"].items():
            per_session = []
            for ses in UNIQUE_SCANNER_SESSIONS:
                medians = []
                for sub in SUBJECTS:
                    if sub not in wm_masks:
                        continue
                    v = load_metric(sub, ses, method, fname, wm_masks[sub])
                    if v is not None:
                        medians.append(float(np.nanmedian(v)))
                if len(medians) >= 2:
                    m = np.mean(medians)
                    if abs(m) > 1e-10:
                        per_session.append(np.std(medians, ddof=1) / abs(m) * 100)
            if per_session:
                between_rows.append({
                    "method": method, "metric": metric,
                    "cov_mean": float(np.mean(per_session)),
                    "cov_std": float(np.std(per_session)),
                })
    between_df = pd.DataFrame(between_rows)
    between_df.to_csv(stats_dir / "between_subject_cov_aligned.csv", index=False)

    # Paired t-tests on CoV (FORCE vs DTI, FORCE vs DKI) for shared metrics
    from scipy import stats as sstats
    stat_rows = []
    shared = ["FA", "MD", "RD"]
    for metric in shared:
        m_covs = {}
        for method in ["DTI", "DKI", "FORCE"]:
            covs = []
            for sub in SUBJECTS:
                if sub not in wm_masks:
                    continue
                wm = wm_masks[sub]
                stack = [load_metric(sub, s, method,
                                     METHODS[method]["metrics"][metric], wm)
                         for s in UNIQUE_SCANNER_SESSIONS]
                stack = [s for s in stack if s is not None]
                if len(stack) >= 3:
                    covs.append(compute_cov(np.array(stack)))
            m_covs[method] = covs

        for other in ["DTI", "DKI"]:
            f = m_covs.get("FORCE", [])
            o = m_covs.get(other, [])
            # Paired t-test across the same subjects (each subject provides one
            # CoV per method, paired across methods).
            if len(f) >= 2 and len(f) == len(o):
                t, p = sstats.ttest_rel(f, o)
                stat_rows.append({
                    "metric": metric, "comparison": f"FORCE vs {other}",
                    "t": float(t), "p": float(p),
                    "force_cov": float(np.mean(f)),
                    "other_cov": float(np.mean(o)),
                    "significant_05": bool(p < 0.05),
                })
    pd.DataFrame(stat_rows).to_csv(stats_dir / "stat_tests_aligned.csv", index=False)

    # ─── Figures ────────────────────────────────────────────────────────────
    if not args.skip_figures:
        _plot_shared_cov(cov_df, retest_df, figs_dir)
        _plot_within_vs_between(cov_df, between_df, figs_dir)
        _plot_icc_heatmap(icc_df, figs_dir)

    # Console summary
    print("\n" + "=" * 72)
    print("ALIGNED ANALYSIS — shared metrics")
    print("=" * 72)
    print(f"{'Method':<6} {'Metric':<6} {'x-scan CoV':>11} {'Retest CoV':>11} {'ICC':>7}")
    print("-" * 72)
    for method in ["DTI", "DKI", "FORCE"]:
        for metric in ["FA", "MD", "RD"]:
            xc = cov_df[(cov_df.method == method) & (cov_df.metric == metric)]
            rc = retest_df[(retest_df.method == method) & (retest_df.metric == metric)]
            ic = icc_df[(icc_df.method == method) & (icc_df.metric == metric)]
            xc_str = f"{xc.cov_mean.values[0]:5.2f}%" if not xc.empty else "N/A"
            rc_str = f"{rc.cov_mean.values[0]:5.2f}%" if not rc.empty else "N/A"
            ic_str = f"{ic.icc_mean.values[0]:.3f}" if not ic.empty else "N/A"
            print(f"{method:<6} {metric:<6} {xc_str:>11} {rc_str:>11} {ic_str:>7}")
    print("=" * 72)


def _plot_shared_cov(cov_df, retest_df, out_dir):
    shared = ["FA", "MD", "RD"]
    methods = ["DTI", "DKI", "FORCE"]
    colors = {"DTI": "#1f77b4", "DKI": "#ff7f0e", "FORCE": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Cross-Scanner CoV (aligned to subject reference)\n"
                 "Solid = cross-scanner, hatched = test-retest floor", fontweight="bold")
    x = np.arange(len(methods))
    w = 0.35
    for ax, metric in zip(axes, shared):
        xc_vals = []
        rc_vals = []
        xc_err = []
        rc_err = []
        for m in methods:
            xc = cov_df[(cov_df.method == m) & (cov_df.metric == metric)]
            rc = retest_df[(retest_df.method == m) & (retest_df.metric == metric)]
            xc_vals.append(xc.cov_mean.values[0] if not xc.empty else 0)
            xc_err.append(xc.cov_std.values[0] if not xc.empty else 0)
            rc_vals.append(rc.cov_mean.values[0] if not rc.empty else 0)
            rc_err.append(rc.cov_std.values[0] if not rc.empty else 0)
        ax.bar(x - w/2, xc_vals, w, yerr=xc_err, color=[colors[m] for m in methods],
               edgecolor="black", label="Cross-scanner")
        ax.bar(x + w/2, rc_vals, w, yerr=rc_err,
               color=[colors[m] for m in methods], hatch="///",
               edgecolor="black", alpha=0.6, label="Test-retest")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("CoV (%)")
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "cov_aligned_shared_metrics.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def _plot_within_vs_between(within_df, between_df, out_dir):
    shared = ["FA", "MD", "RD"]
    methods = ["DTI", "DKI", "FORCE"]
    colors = {"DTI": "#1f77b4", "DKI": "#ff7f0e", "FORCE": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Within-Subject Cross-Scanner vs Between-Subject CoV (aligned)",
                 fontweight="bold")
    x = np.arange(len(methods))
    w = 0.35
    for ax, metric in zip(axes, shared):
        w_vals = [within_df[(within_df.method == m) & (within_df.metric == metric)]
                  .cov_mean.values[0]
                  if not within_df[(within_df.method == m) & (within_df.metric == metric)].empty
                  else 0 for m in methods]
        b_vals = [between_df[(between_df.method == m) & (between_df.metric == metric)]
                  .cov_mean.values[0]
                  if not between_df[(between_df.method == m) & (between_df.metric == metric)].empty
                  else 0 for m in methods]
        ax.bar(x - w/2, w_vals, w, color=[colors[m] for m in methods],
               edgecolor="black", label="Within-subject (scanner)")
        ax.bar(x + w/2, b_vals, w, color=[colors[m] for m in methods], hatch="///",
               edgecolor="black", alpha=0.6, label="Between-subject (biology)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("CoV (%)")
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "within_vs_between_aligned.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def _plot_icc_heatmap(icc_df, out_dir):
    if icc_df.empty:
        return
    pivot = icc_df.pivot(index="method", columns="metric", values="icc_mean")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.3 < v < 0.75 else "white",
                        fontsize=8)
    ax.set_title("ICC(3,1) — cross-scanner (aligned)")
    plt.colorbar(im, ax=ax, label="ICC")
    plt.tight_layout()
    fig.savefig(str(out_dir / "icc_heatmap_aligned.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
