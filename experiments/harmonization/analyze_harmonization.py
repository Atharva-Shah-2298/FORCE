"""
Publication-ready harmonization analysis.

Generates:
1. Cross-scanner harmonization: same subject across scanners (CoV, ICC)
2. Cross-subject variability: different subjects at same scanner
3. Comparison map plots (A vs B vs |B-A|) for all methods
4. Bar plots / violin plots with statistical tests
5. Tract profile comparisons

Run with skyline env (for nibabel, numpy, scipy, matplotlib):
    C:\\Users\\athu2\\miniconda3\\envs\\skyline\\python.exe analyze_harmonization.py
"""
from __future__ import annotations

import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Configuration ───────────────────────────────────────────────────────────

import os as _os
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(_os.environ.get("HARMONIZATION_ROOT", SCRIPT_DIR))
OUTPUT_ROOT = Path(_os.environ.get("HARMONIZATION_OUTPUT", ROOT / "output"))
RESULTS_DIR = Path(_os.environ.get("HARMONIZATION_RESULTS", ROOT / "harmonization_results"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
# 10 different scanners + 2 repeat scans on scanner 10
SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
    "ses-c10r2", "ses-c10r3",
]
# For cross-scanner analysis, use only unique scanners (one run per scanner)
UNIQUE_SCANNER_SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
]
# For test-retest (same scanner repeated)
RETEST_SESSIONS = ["ses-c10r1", "ses-c10r2", "ses-c10r3"]

# Microstructure metrics to compare per method
METHODS = {
    "DTI": {
        "dir": "dti",
        "metrics": {
            "FA": "fa.nii.gz",
            "MD": "md.nii.gz",
            "AD": "ad.nii.gz",
            "RD": "rd.nii.gz",
        },
    },
    "DKI": {
        "dir": "dki",
        "metrics": {
            "FA": "fa.nii.gz",
            "MD": "md.nii.gz",
            "AD": "ad.nii.gz",
            "RD": "rd.nii.gz",
            "MK": "mk.nii.gz",
            "AK": "ak.nii.gz",
            "RK": "rk.nii.gz",
        },
    },
    "FORCE": {
        "dir": "force",
        "metrics": {
            "FA": "fa.nii.gz",
            "MD": "md.nii.gz",
            "RD": "rd.nii.gz",
            "ND": "nd.nii.gz",
            "Dispersion": "dispersion.nii.gz",
            "WM Fraction": "wm_fraction.nii.gz",
            "GM Fraction": "gm_fraction.nii.gz",
            "CSF Fraction": "csf_fraction.nii.gz",
            "uFA": "ufa.nii.gz",
            "MK": "mk.nii.gz",
            "AK": "ak.nii.gz",
            "RK": "rk.nii.gz",
        },
    },
    "AMICO-NODDI": {
        "dir": "AMICO/NODDI",
        "metrics": {
            "NDI": "fit_NDI.nii.gz",
            "ODI": "fit_ODI.nii.gz",
            "FWF": "fit_FWF.nii.gz",
        },
    },
}

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette for methods
METHOD_COLORS = {
    "DTI": "#1f77b4",
    "DKI": "#ff7f0e",
    "FORCE": "#2ca02c",
    "AMICO-NODDI": "#d62728",
}
SUBJECT_COLORS = {
    "sub-1": "#4C72B0",
    "sub-2": "#DD8452",
    "sub-3": "#55A868",
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_masked_volume(nii_path: Path, mask_path: Path) -> np.ndarray | None:
    """Load a NIfTI volume and return masked voxel values (1D)."""
    if not nii_path.exists() or not mask_path.exists():
        return None
    vol = nib.load(str(nii_path)).get_fdata().astype(np.float64)
    mask = nib.load(str(mask_path)).get_fdata() > 0.5
    if vol.ndim == 4:
        # Take first volume for 4D data
        vol = vol[..., 0]
    if mask.shape != vol.shape:
        return None
    vals = vol[mask]
    # Remove NaN/Inf
    vals = vals[np.isfinite(vals)]
    return vals


def load_volume_3d(nii_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a NIfTI volume and mask as 3D arrays."""
    if not nii_path.exists() or not mask_path.exists():
        return None
    vol = nib.load(str(nii_path)).get_fdata().astype(np.float64)
    mask = nib.load(str(mask_path)).get_fdata() > 0.5
    if vol.ndim == 4:
        vol = vol[..., 0]
    return vol, mask


def get_metric_path(subject: str, session: str, method: str, metric_file: str) -> Path:
    """Get path to a metric NIfTI file."""
    method_dir = METHODS[method]["dir"]
    return OUTPUT_ROOT / subject / session / "dwi" / method_dir / metric_file


def get_mask_path(subject: str, session: str) -> Path:
    return OUTPUT_ROOT / subject / session / "dwi" / "brain_mask.nii.gz"


# ─── Statistics ──────────────────────────────────────────────────────────────

def compute_cov(values_list: list[np.ndarray]) -> float:
    """Compute voxel-wise coefficient of variation across sessions."""
    # Stack: (n_sessions, n_voxels) - only use common voxels
    min_len = min(len(v) for v in values_list)
    stacked = np.array([v[:min_len] for v in values_list])
    mean_per_voxel = np.mean(stacked, axis=0)
    std_per_voxel = np.std(stacked, axis=0, ddof=1)
    # Avoid division by zero
    valid = mean_per_voxel > 1e-10
    cov_values = std_per_voxel[valid] / mean_per_voxel[valid]
    return np.median(cov_values) * 100  # as percentage


def compute_icc(values_list: list[np.ndarray], icc_type: str = "ICC(3,1)") -> float:
    """Compute Intraclass Correlation Coefficient.
    ICC(3,1) - two-way mixed, consistency, single measures.
    """
    min_len = min(len(v) for v in values_list)
    # Subsample for computational efficiency
    n_sample = min(min_len, 50000)
    idx = np.random.choice(min_len, n_sample, replace=False) if min_len > n_sample else np.arange(min_len)
    data = np.array([v[idx] for v in values_list]).T  # (n_voxels, n_sessions)

    n, k = data.shape
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_row = k * np.sum((row_means - grand_mean) ** 2)
    ss_col = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_row - ss_col

    ms_row = ss_row / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_col = ss_col / (k - 1)

    # ICC(3,1)
    icc = (ms_row - ms_error) / (ms_row + (k - 1) * ms_error)
    return np.clip(icc, 0, 1)


def compute_mean_abs_diff(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """Mean absolute difference between two sets of voxel values."""
    min_len = min(len(vals_a), len(vals_b))
    return np.mean(np.abs(vals_a[:min_len] - vals_b[:min_len]))


# ─── Plot: Comparison Maps (A vs B vs |B-A|) ────────────────────────────────

def plot_comparison_maps(
    subject: str,
    session_a: str,
    session_b: str,
    method: str,
    metric_name: str,
    metric_file: str,
    out_dir: Path,
):
    """3×3 panel: Sag/Cor/Ax × A/B/|B-A| like existing comparison_results."""
    path_a = get_metric_path(subject, session_a, method, metric_file)
    path_b = get_metric_path(subject, session_b, method, metric_file)
    mask_a_path = get_mask_path(subject, session_a)
    mask_b_path = get_mask_path(subject, session_b)

    result_a = load_volume_3d(path_a, mask_a_path)
    result_b = load_volume_3d(path_b, mask_b_path)
    if result_a is None or result_b is None:
        return

    va, ma = result_a
    vb, mb = result_b

    if va.shape != vb.shape:
        return

    # Auto-detect value range
    all_vals = np.concatenate([va[ma], vb[mb]])
    vmin = np.percentile(all_vals[np.isfinite(all_vals)], 1)
    vmax = np.percentile(all_vals[np.isfinite(all_vals)], 99)

    vres = np.abs(vb - va)
    nx, ny, nz = va.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), facecolor="black", layout="constrained")
    fig.suptitle(f"{method} - {metric_name}\n{subject}: {session_a} vs {session_b}",
                 color="white", fontsize=12)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["inferno"]

    planes = [("sag", ix, lambda v, m, i: (v[i, :, :], m[i, :, :])),
              ("cor", iy, lambda v, m, i: (v[:, i, :], m[:, i, :])),
              ("ax",  iz, lambda v, m, i: (v[:, :, i], m[:, :, i]))]
    plane_labels = ["Sag", "Cor", "Ax"]

    for r, (pname, pidx, slicer) in enumerate(planes):
        for c, (data3d, mask3d, col_prefix) in enumerate([
            (va, ma, "A"), (vb, mb, "B"), (vres, ma | mb, "|B-A|")
        ]):
            ax = axes[r, c]
            ax.set_facecolor("black")
            sl, m_sl = slicer(data3d, mask3d, pidx)
            sl_masked = np.where(m_sl, sl, np.nan)
            sl_masked = np.rot90(sl_masked, k=1)

            im = ax.imshow(sl_masked, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(f"{col_prefix} {plane_labels[r]}", color="white", fontsize=10)
            ax.axis("off")

        cbar = fig.colorbar(im, ax=axes[r, :], orientation="vertical",
                           fraction=0.03, pad=0.02, shrink=0.85)
        cbar.ax.tick_params(colors="white")

    safe_name = f"{method}_{metric_name}_{subject}_{session_a}_vs_{session_b}".replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{safe_name}.png", dpi=150, facecolor="black")
    plt.close(fig)


# ─── Plot: Cross-Scanner CoV Bar Plot ───────────────────────────────────────

def plot_cross_scanner_cov(cov_data: pd.DataFrame, out_path: Path):
    """
    Bar plot: CoV for each metric, grouped by method.
    Shows that FORCE has lower CoV (better cross-scanner reproducibility).
    """
    plt.style.use("default")

    # Get unique metrics that appear across methods
    shared_metrics = ["FA", "MD", "RD"]
    extra_metrics = {
        "DTI": ["AD"],
        "DKI": ["AD", "MK", "AK", "RK"],
        "FORCE": ["ND", "Dispersion", "uFA", "MK", "AK", "RK"],
        "AMICO-NODDI": ["NDI", "ODI", "FWF"],
    }

    # Figure 1: Shared metrics (FA, MD, RD) across all methods
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle("Cross-Scanner Coefficient of Variation (CoV%)\nLower = Better Harmonization",
                 fontsize=12, fontweight="bold")

    for ax_idx, metric in enumerate(shared_metrics):
        ax = axes[ax_idx]
        methods_with_metric = []
        cov_values = []
        cov_errors = []
        colors = []

        for method in ["DTI", "DKI", "FORCE"]:
            subset = cov_data[
                (cov_data["method"] == method) & (cov_data["metric"] == metric)
            ]
            if not subset.empty:
                methods_with_metric.append(method)
                cov_values.append(subset["cov_mean"].values[0])
                cov_errors.append(subset["cov_std"].values[0])
                colors.append(METHOD_COLORS[method])

        bars = ax.bar(methods_with_metric, cov_values, yerr=cov_errors,
                      color=colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("CoV (%)" if ax_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, cov_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path.parent / "cross_scanner_cov_shared_metrics.png")
    plt.close(fig)

    # Figure 2: All metrics per method (separate subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Scanner Coefficient of Variation by Method\n(Averaged across subjects)",
                 fontsize=13, fontweight="bold")

    for idx, (method, ax) in enumerate(zip(["DTI", "DKI", "FORCE", "AMICO-NODDI"],
                                           axes.flat)):
        subset = cov_data[cov_data["method"] == method].sort_values("cov_mean")
        if subset.empty:
            ax.set_visible(False)
            continue

        bars = ax.barh(subset["metric"], subset["cov_mean"],
                       xerr=subset["cov_std"],
                       color=METHOD_COLORS[method], capsize=3,
                       edgecolor="black", linewidth=0.5)
        ax.set_xlabel("CoV (%)")
        ax.set_title(method, fontweight="bold", color=METHOD_COLORS[method])
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, subset["cov_mean"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path.parent / "cross_scanner_cov_all_methods.png")
    plt.close(fig)


# ─── Plot: Cross-Subject Variability vs Cross-Scanner Variability ────────────

def plot_variability_comparison(within_subject_cov: pd.DataFrame,
                               between_subject_cov: pd.DataFrame,
                               out_path: Path):
    """
    Grouped bar plot showing within-subject CoV (cross-scanner) vs
    between-subject CoV for each method.
    FORCE should show: low within-subject (good harmonization) and
    distinguishable between-subject (maintains biological differences).
    """
    plt.style.use("default")
    shared_metrics = ["FA", "MD", "RD"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Cross-Scanner Reproducibility vs Cross-Subject Sensitivity\n"
                 "Good method: low within-subject CoV, high between-subject CoV",
                 fontsize=12, fontweight="bold")

    width = 0.35
    methods_list = ["DTI", "DKI", "FORCE"]

    for ax_idx, metric in enumerate(shared_metrics):
        ax = axes[ax_idx]
        x = np.arange(len(methods_list))

        within_vals = []
        between_vals = []
        for method in methods_list:
            w = within_subject_cov[
                (within_subject_cov["method"] == method) &
                (within_subject_cov["metric"] == metric)
            ]
            b = between_subject_cov[
                (between_subject_cov["method"] == method) &
                (between_subject_cov["metric"] == metric)
            ]
            within_vals.append(w["cov_mean"].values[0] if not w.empty else 0)
            between_vals.append(b["cov_mean"].values[0] if not b.empty else 0)

        bars1 = ax.bar(x - width/2, within_vals, width, label="Within-Subject\n(Cross-Scanner)",
                       color=[METHOD_COLORS[m] for m in methods_list], alpha=0.7,
                       edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, between_vals, width, label="Between-Subject",
                       color=[METHOD_COLORS[m] for m in methods_list], alpha=1.0,
                       edgecolor="black", linewidth=0.5, hatch="//")

        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_list)
        ax.set_ylabel("CoV (%)" if ax_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 2:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─── Plot: Violin/Box Plots per Subject across Scanners ─────────────────────

def plot_violin_per_subject(all_data: dict, method: str, metric: str,
                           metric_file: str, out_dir: Path):
    """
    Violin plot: distribution of metric values across scanners for each subject.
    Shows FORCE gives tight distributions (consistent across scanners) but
    different medians per subject (captures biological variability).
    """
    plt.style.use("default")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"{method} - {metric}: Distribution Across Scanners",
                 fontsize=12, fontweight="bold")

    for sub_idx, subject in enumerate(SUBJECTS):
        ax = axes[sub_idx]
        session_data = []
        session_labels = []

        for ses in UNIQUE_SCANNER_SESSIONS:
            key = (subject, ses, method, metric)
            if key in all_data and all_data[key] is not None:
                # Subsample for plotting
                vals = all_data[key]
                if len(vals) > 5000:
                    vals = np.random.choice(vals, 5000, replace=False)
                session_data.append(vals)
                # Short label: just scanner number
                session_labels.append(ses.replace("ses-c", "C").replace("r1", ""))

        if not session_data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(session_data, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(SUBJECT_COLORS[subject])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')

        ax.set_xticks(range(1, len(session_labels) + 1))
        ax.set_xticklabels(session_labels, rotation=45, fontsize=7)
        ax.set_title(subject, fontweight="bold", color=SUBJECT_COLORS[subject])
        ax.set_xlabel("Scanner")
        if sub_idx == 0:
            ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe = f"violin_{method}_{metric}".replace(" ", "_")
    fig.savefig(out_dir / f"{safe}.png")
    plt.close(fig)


# ─── Plot: ICC Heatmap ──────────────────────────────────────────────────────

def plot_icc_heatmap(icc_data: pd.DataFrame, out_path: Path):
    """Heatmap of ICC values: methods × metrics."""
    plt.style.use("default")

    # Pivot to methods × metrics
    pivot = icc_data.pivot_table(index="method", columns="metric", values="icc_mean")

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="ICC", shrink=0.8)
    ax.set_title("Intraclass Correlation Coefficient (ICC)\nHigher = Better Cross-Scanner Reproducibility",
                fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─── Plot: Pairwise Scatter Correlation ──────────────────────────────────────

def plot_pairwise_correlation(all_data: dict, subject: str,
                             session_a: str, session_b: str,
                             method: str, metrics: list[str],
                             out_dir: Path):
    """Scatter plot of session A vs session B for key metrics."""
    plt.style.use("default")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(f"{method}: {session_a} vs {session_b} ({subject})",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        key_a = (subject, session_a, method, metric)
        key_b = (subject, session_b, method, metric)

        if key_a not in all_data or key_b not in all_data:
            ax.set_visible(False)
            continue

        va, vb = all_data[key_a], all_data[key_b]
        if va is None or vb is None:
            ax.set_visible(False)
            continue

        min_len = min(len(va), len(vb))
        # Subsample for scatter
        n_sample = min(min_len, 10000)
        idx = np.random.choice(min_len, n_sample, replace=False)
        x, y = va[idx], vb[idx]

        ax.scatter(x, y, s=1, alpha=0.1, color=METHOD_COLORS[method])
        # Identity line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.8)

        r, _ = stats.pearsonr(x, y)
        ax.set_title(f"{metric}\nr = {r:.4f}", fontweight="bold")
        ax.set_xlabel(session_a)
        ax.set_ylabel(session_b)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    safe = f"scatter_{method}_{subject}_{session_a}_vs_{session_b}".replace(" ", "_")
    fig.savefig(out_dir / f"{safe}.png")
    plt.close(fig)


# ─── Plot: Summary Radar Chart ──────────────────────────────────────────────

def plot_radar_summary(cov_data: pd.DataFrame, icc_data: pd.DataFrame, out_path: Path):
    """Radar chart comparing methods on key performance indicators."""
    plt.style.use("default")

    shared_metrics = ["FA", "MD", "RD"]
    methods = ["DTI", "DKI", "FORCE"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(polar=True))

    # Radar 1: CoV (lower is better → invert for radar)
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, len(shared_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for method in methods:
        vals = []
        for metric in shared_metrics:
            subset = cov_data[
                (cov_data["method"] == method) & (cov_data["metric"] == metric)
            ]
            vals.append(subset["cov_mean"].values[0] if not subset.empty else 0)
        vals += vals[:1]
        ax.plot(angles, vals, '-o', linewidth=2, label=method,
                color=METHOD_COLORS[method], markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=METHOD_COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(shared_metrics)
    ax.set_title("Cross-Scanner CoV (%)\n(Lower = Better)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Radar 2: ICC (higher is better)
    ax = axes[1]
    for method in methods:
        vals = []
        for metric in shared_metrics:
            subset = icc_data[
                (icc_data["method"] == method) & (icc_data["metric"] == metric)
            ]
            vals.append(subset["icc_mean"].values[0] if not subset.empty else 0)
        vals += vals[:1]
        ax.plot(angles, vals, '-o', linewidth=2, label=method,
                color=METHOD_COLORS[method], markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=METHOD_COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(shared_metrics)
    ax.set_ylim(0, 1)
    ax.set_title("ICC (3,1)\n(Higher = Better)", fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─── Plot: Tract Count Comparison ───────────────────────────────────────────

def collect_tract_metrics(subject: str, sessions: list[str],
                          tracking_type: str = "force") -> pd.DataFrame:
    """Collect tract bundle metrics across sessions."""
    bundle_dir_name = f"org_bundles_{tracking_type}"
    rows = []

    for ses in sessions:
        bundle_dir = OUTPUT_ROOT / subject / ses / "dwi" / bundle_dir_name
        if not bundle_dir.exists():
            continue

        trx_files = list(bundle_dir.glob("*.trx"))
        for trx_file in trx_files:
            # Extract bundle name from filename
            name = trx_file.stem
            # Get file size as proxy for streamline count
            size_mb = trx_file.stat().st_size / (1024 * 1024)
            rows.append({
                "subject": subject,
                "session": ses,
                "tracking": tracking_type,
                "bundle": name,
                "size_mb": size_mb,
            })

    return pd.DataFrame(rows)


def plot_tract_comparison(out_dir: Path):
    """Compare FORCE vs CSD tracking: bundle sizes across scanners."""
    plt.style.use("default")

    all_tracts = []
    for sub in SUBJECTS:
        for track_type in ["force", "csd"]:
            df = collect_tract_metrics(sub, UNIQUE_SCANNER_SESSIONS, track_type)
            all_tracts.append(df)

    if not all_tracts:
        return

    all_df = pd.concat(all_tracts, ignore_index=True)
    if all_df.empty:
        return

    # Get top 10 most common bundles
    top_bundles = all_df.groupby("bundle")["size_mb"].count().nlargest(10).index

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Bundle Consistency Across Scanners\nFORCE vs CSD Tracking",
                 fontsize=13, fontweight="bold")

    for ax_idx, track_type in enumerate(["force", "csd"]):
        ax = axes[ax_idx]
        subset = all_df[
            (all_df["tracking"] == track_type) & (all_df["bundle"].isin(top_bundles))
        ]
        if subset.empty:
            continue

        # CoV of bundle size per subject
        cov_data = []
        for sub in SUBJECTS:
            for bundle in top_bundles:
                bdata = subset[
                    (subset["subject"] == sub) & (subset["bundle"] == bundle)
                ]["size_mb"]
                if len(bdata) > 1:
                    cov_data.append({
                        "subject": sub,
                        "bundle": bundle.split("_")[-2] if "_" in bundle else bundle[:15],
                        "cov": (bdata.std() / bdata.mean() * 100) if bdata.mean() > 0 else 0,
                    })

        if cov_data:
            cov_df = pd.DataFrame(cov_data)
            # Grouped bar by subject
            bundles = cov_df["bundle"].unique()
            x = np.arange(len(bundles))
            w = 0.25
            for i, sub in enumerate(SUBJECTS):
                sub_data = cov_df[cov_df["subject"] == sub]
                vals = [sub_data[sub_data["bundle"] == b]["cov"].mean()
                        for b in bundles]
                ax.bar(x + i*w, vals, w, label=sub,
                       color=SUBJECT_COLORS[sub], edgecolor="black", linewidth=0.5)

            ax.set_xticks(x + w)
            ax.set_xticklabels(bundles, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("CoV (%)")
            ax.set_title(f"{track_type.upper()} Tracking", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "tract_bundle_cov_comparison.png")
    plt.close(fig)


# ─── Main Analysis ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Harmonization analysis")
    parser.add_argument("--maps-only", action="store_true",
                        help="Only generate comparison map plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    maps_dir = RESULTS_DIR / "comparison_maps"
    maps_dir.mkdir(exist_ok=True)
    stats_dir = RESULTS_DIR / "statistics"
    stats_dir.mkdir(exist_ok=True)
    plots_dir = RESULTS_DIR / "publication_figures"
    plots_dir.mkdir(exist_ok=True)

    np.random.seed(42)

    # ── Step 1: Load all data ────────────────────────────────────────────────
    print("Loading metric data across all subjects/sessions/methods...")
    all_data = {}  # (subject, session, method, metric) -> masked values
    available = []

    for sub in SUBJECTS:
        for ses in SESSIONS:
            mask_path = get_mask_path(sub, ses)
            for method, method_info in METHODS.items():
                for metric_name, metric_file in method_info["metrics"].items():
                    nii_path = get_metric_path(sub, ses, method, metric_file)
                    key = (sub, ses, method, metric_name)
                    vals = load_masked_volume(nii_path, mask_path)
                    all_data[key] = vals
                    if vals is not None:
                        available.append(key)

    print(f"  Loaded {len(available)} metric volumes")

    # ── Step 2: Comparison map plots ─────────────────────────────────────────
    print("\nGenerating comparison map plots...")
    # For each subject, compare first two sessions
    for sub in SUBJECTS:
        ses_pairs = [("ses-c01r1", "ses-c02r1"), ("ses-c01r1", "ses-c05r1"),
                     ("ses-c01r1", "ses-c10r1")]
        for ses_a, ses_b in ses_pairs:
            for method, method_info in METHODS.items():
                for metric_name, metric_file in method_info["metrics"].items():
                    plot_comparison_maps(sub, ses_a, ses_b, method,
                                        metric_name, metric_file,
                                        maps_dir / method)
    print("  Done.")

    if args.maps_only:
        print("Maps-only mode. Exiting.")
        return

    # ── Step 3: Cross-Scanner CoV ────────────────────────────────────────────
    print("\nComputing cross-scanner CoV...")
    cov_rows = []
    for method, method_info in METHODS.items():
        for metric_name in method_info["metrics"]:
            cov_per_subject = []
            for sub in SUBJECTS:
                session_vals = []
                for ses in UNIQUE_SCANNER_SESSIONS:
                    key = (sub, ses, method, metric_name)
                    if key in all_data and all_data[key] is not None:
                        session_vals.append(all_data[key])

                if len(session_vals) >= 3:
                    cov = compute_cov(session_vals)
                    cov_per_subject.append(cov)

            if cov_per_subject:
                cov_rows.append({
                    "method": method,
                    "metric": metric_name,
                    "cov_mean": np.mean(cov_per_subject),
                    "cov_std": np.std(cov_per_subject),
                    "n_subjects": len(cov_per_subject),
                })

    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(stats_dir / "cross_scanner_cov.csv", index=False)
    print(f"  CoV computed for {len(cov_rows)} method-metric combinations")

    # ── Step 4: ICC ──────────────────────────────────────────────────────────
    print("\nComputing ICC...")
    icc_rows = []
    for method, method_info in METHODS.items():
        for metric_name in method_info["metrics"]:
            icc_per_subject = []
            for sub in SUBJECTS:
                session_vals = []
                for ses in UNIQUE_SCANNER_SESSIONS:
                    key = (sub, ses, method, metric_name)
                    if key in all_data and all_data[key] is not None:
                        session_vals.append(all_data[key])

                if len(session_vals) >= 3:
                    icc = compute_icc(session_vals)
                    icc_per_subject.append(icc)

            if icc_per_subject:
                icc_rows.append({
                    "method": method,
                    "metric": metric_name,
                    "icc_mean": np.mean(icc_per_subject),
                    "icc_std": np.std(icc_per_subject),
                    "n_subjects": len(icc_per_subject),
                })

    icc_df = pd.DataFrame(icc_rows)
    icc_df.to_csv(stats_dir / "icc_results.csv", index=False)
    print(f"  ICC computed for {len(icc_rows)} method-metric combinations")

    # ── Step 5: Between-Subject CoV ──────────────────────────────────────────
    print("\nComputing between-subject CoV...")
    between_cov_rows = []
    for method, method_info in METHODS.items():
        for metric_name in method_info["metrics"]:
            cov_per_session = []
            for ses in UNIQUE_SCANNER_SESSIONS:
                subject_vals = []
                for sub in SUBJECTS:
                    key = (sub, ses, method, metric_name)
                    if key in all_data and all_data[key] is not None:
                        subject_vals.append(np.median(all_data[key]))

                if len(subject_vals) >= 2:
                    mean_val = np.mean(subject_vals)
                    if mean_val > 1e-10:
                        cov = np.std(subject_vals, ddof=1) / mean_val * 100
                        cov_per_session.append(cov)

            if cov_per_session:
                between_cov_rows.append({
                    "method": method,
                    "metric": metric_name,
                    "cov_mean": np.mean(cov_per_session),
                    "cov_std": np.std(cov_per_session),
                })

    between_cov_df = pd.DataFrame(between_cov_rows)
    between_cov_df.to_csv(stats_dir / "between_subject_cov.csv", index=False)

    # ── Step 6: Statistical Tests ────────────────────────────────────────────
    print("\nRunning statistical tests...")
    stat_rows = []
    shared_metrics = ["FA", "MD", "RD"]
    for metric in shared_metrics:
        # Collect CoV values per method across subjects
        method_covs = {}
        for method in ["DTI", "DKI", "FORCE"]:
            covs = []
            for sub in SUBJECTS:
                session_vals = []
                for ses in UNIQUE_SCANNER_SESSIONS:
                    key = (sub, ses, method, metric)
                    if key in all_data and all_data[key] is not None:
                        session_vals.append(all_data[key])
                if len(session_vals) >= 3:
                    covs.append(compute_cov(session_vals))
            method_covs[method] = covs

        # Pairwise comparisons (FORCE vs others)
        for other in ["DTI", "DKI"]:
            if method_covs.get("FORCE") and method_covs.get(other):
                if len(method_covs["FORCE"]) >= 2 and len(method_covs[other]) >= 2:
                    t_stat, p_val = stats.ttest_ind(
                        method_covs["FORCE"], method_covs[other],
                        equal_var=False
                    )
                    stat_rows.append({
                        "metric": metric,
                        "comparison": f"FORCE vs {other}",
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "force_mean_cov": np.mean(method_covs["FORCE"]),
                        "other_mean_cov": np.mean(method_covs[other]),
                        "significant": p_val < 0.05,
                    })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(stats_dir / "statistical_tests.csv", index=False)
    print(f"  {len(stat_rows)} comparisons performed")

    # ── Step 7: Generate Publication Figures ──────────────────────────────────
    print("\nGenerating publication figures...")

    if not cov_df.empty:
        plot_cross_scanner_cov(cov_df, plots_dir / "cross_scanner_cov.png")
        print("  - Cross-scanner CoV bar plots")

    if not cov_df.empty and not between_cov_df.empty:
        plot_variability_comparison(cov_df, between_cov_df,
                                   plots_dir / "variability_comparison.png")
        print("  - Within vs between subject variability")

    if not icc_df.empty:
        plot_icc_heatmap(icc_df, plots_dir / "icc_heatmap.png")
        print("  - ICC heatmap")

    if not cov_df.empty and not icc_df.empty:
        plot_radar_summary(cov_df, icc_df, plots_dir / "radar_summary.png")
        print("  - Radar summary chart")

    # Violin plots for key metrics
    for method in ["DTI", "DKI", "FORCE"]:
        for metric in ["FA", "MD"]:
            if (SUBJECTS[0], UNIQUE_SCANNER_SESSIONS[0], method, metric) in all_data:
                metric_file = METHODS[method]["metrics"].get(metric)
                if metric_file:
                    plot_violin_per_subject(all_data, method, metric,
                                           metric_file, plots_dir)
    print("  - Violin plots")

    # Pairwise scatter plots
    for sub in SUBJECTS:
        for method in ["FORCE", "DTI"]:
            metrics_list = list(METHODS[method]["metrics"].keys())[:4]
            plot_pairwise_correlation(
                all_data, sub, "ses-c01r1", "ses-c05r1",
                method, metrics_list, plots_dir
            )
    print("  - Scatter correlation plots")

    # Tract comparison
    plot_tract_comparison(plots_dir)
    print("  - Tract bundle comparison")

    # ── Summary Report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Statistics:  {stats_dir}")
    print(f"  Map plots:   {maps_dir}")
    print(f"  Pub figures: {plots_dir}")

    if not cov_df.empty:
        print(f"\n  Cross-Scanner CoV Summary (shared metrics):")
        for method in ["DTI", "DKI", "FORCE"]:
            for metric in ["FA", "MD", "RD"]:
                row = cov_df[(cov_df["method"] == method) & (cov_df["metric"] == metric)]
                if not row.empty:
                    print(f"    {method:6s} {metric:3s}: {row['cov_mean'].values[0]:6.2f}% "
                          f"(± {row['cov_std'].values[0]:.2f}%)")

    if not icc_df.empty:
        print(f"\n  ICC Summary (shared metrics):")
        for method in ["DTI", "DKI", "FORCE"]:
            for metric in ["FA", "MD", "RD"]:
                row = icc_df[(icc_df["method"] == method) & (icc_df["metric"] == metric)]
                if not row.empty:
                    print(f"    {method:6s} {metric:3s}: {row['icc_mean'].values[0]:.3f} "
                          f"(± {row['icc_std'].values[0]:.3f})")


if __name__ == "__main__":
    main()
