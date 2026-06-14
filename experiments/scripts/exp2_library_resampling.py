#!/usr/bin/env python
"""Exp 2 — Library resampling stability.

Run FORCE N_TRIALS times on the same DWI with independently generated
libraries (use_cache=False). Collect per-trial FA, MD, ND, dispersion (ODI),
num_fibers maps. Compute aggregate instability metrics over WM.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

METRICS = ("fa", "md", "nd", "dispersion", "rd", "wm_fraction", "gm_fraction",
           "csf_fraction", "num_fibers", "ufa_voxel")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_trial(trial_idx, cfg, out_dir, force_flag=False):
    """Run a single FORCE trial and save metric maps."""
    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.reconst.force import FORCEModel

    trial_dir = out_dir / f"trial_{trial_idx:02d}"
    fa_path = trial_dir / "fa.nii.gz"
    if fa_path.exists() and not force_flag:
        log.info(f"Trial {trial_idx} already done, skipping (use --force to rerun)")
        return

    trial_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"--- Trial {trial_idx} ---")
    data, affine = load_nifti(cfg["dwi"])
    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                          atol=cfg["bvecs_tol"])
    mask = load_nifti_data(cfg["mask"]).astype(bool)

    model = FORCEModel(
        gtab,
        penalty=cfg["penalty"],
        n_neighbors=cfg["n_neighbors"],
        use_posterior=True,
        posterior_beta=cfg["posterior_beta"],
        verbose=True,
    )

    t0 = time.time()
    log.info(f"Generating library (use_cache=False) ...")
    model.generate(
        num_simulations=cfg["num_simulations"],
        num_cpus=-1,
        use_cache=False,
        verbose=True,
    )
    t_gen = time.time() - t0
    log.info(f"Library generated in {t_gen:.1f}s")

    t0 = time.time()
    log.info("Fitting ...")
    force_fit = model.fit(data, mask=mask, engine="serial", verbose=True)
    t_fit = time.time() - t0
    log.info(f"Fit done in {t_fit:.1f}s")

    # Save metric maps
    for m in METRICS:
        arr = getattr(force_fit, m, None)
        if arr is not None:
            nib.save(nib.Nifti1Image(arr.astype(np.float32), affine),
                     str(trial_dir / f"{m}.nii.gz"))

    # Save timing
    with open(trial_dir / "timing.json", "w") as f:
        json.dump({"generate_s": t_gen, "fit_s": t_fit}, f, indent=2)

    log.info(f"Trial {trial_idx} saved to {trial_dir}")


def compute_summary(cfg, out_dir, n_trials):
    """Compute instability metrics across trials."""
    import nibabel as nib

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Load WM mask
    wm_mask = nib.load(cfg["wm_mask_path"]).get_fdata().astype(bool)
    # Handle potential shape mismatch (single-slice vs 3D)
    ref_shape = nib.load(cfg["mask"]).shape
    if wm_mask.shape != ref_shape:
        log.warning(f"WM mask shape {wm_mask.shape} != brain mask shape {ref_shape}. "
                    "Creating WM mask from DTI FA > 0.2.")
        fa_dti = nib.load(str(Path(cfg["dwi"]).parent / "fa_dti.nii.gz")).get_fdata()
        brain = nib.load(cfg["mask"]).get_fdata().astype(bool)
        wm_mask = (fa_dti > cfg["wm_fa_threshold"]) & brain

    affine = nib.load(cfg["mask"]).affine
    n_wm = int(np.sum(wm_mask))
    log.info(f"WM mask: {n_wm} voxels")

    # Load ambiguity maps (Aₘ) from existing FORCE run
    amb_dir = Path(cfg["dwi"]).parent / "FORCE_latest" / "microstructure_ambiguity"

    results = {}
    for m in METRICS:
        # Stack trials
        stacks = []
        for t in range(n_trials):
            p = out_dir / f"trial_{t:02d}" / f"{m}.nii.gz"
            if not p.exists():
                log.warning(f"Missing {p}, skipping metric {m}")
                break
            stacks.append(nib.load(str(p)).get_fdata())
        else:
            stack = np.array(stacks)  # (n_trials, X, Y, Z)
            wm_stack = stack[:, wm_mask]  # (n_trials, n_wm)

            # Aggregate MAD across trials (mean abs deviation from trial mean)
            trial_mean = np.mean(wm_stack, axis=0)
            mad = np.mean(np.abs(wm_stack - trial_mean[None, :]), axis=0)

            # Prior range: use range across library (approximate from data range)
            prior_range = float(np.max(wm_stack) - np.min(wm_stack))
            # Better: use the full-volume range across all trials
            all_vals = stack[:, nib.load(cfg["mask"]).get_fdata().astype(bool)]
            prior_range = float(np.max(all_vals) - np.min(all_vals))

            mad_norm = mad / (prior_range + 1e-12)

            # Load Aₘ if available
            am_name = m if m != "ufa_voxel" else "ufa_voxel"
            am_path = amb_dir / f"{am_name}.nii.gz"
            am_wm = None
            if am_path.exists():
                am_data = nib.load(str(am_path)).get_fdata()
                if am_data.shape == ref_shape:
                    am_wm = am_data[wm_mask]

            results[m] = {
                "mad_mean": float(np.mean(mad_norm)),
                "mad_median": float(np.median(mad_norm)),
                "mad_p95": float(np.percentile(mad_norm, 95)),
                "prior_range": prior_range,
            }
            if am_wm is not None:
                results[m]["am_median"] = float(np.median(am_wm))
                results[m]["am_p95"] = float(np.percentile(am_wm, 95))
                ratio = float(np.median(mad_norm)) / (float(np.median(am_wm)) + 1e-12)
                results[m]["ratio_median"] = ratio

            # Save MAD map
            mad_vol = np.zeros(ref_shape, dtype=np.float32)
            mad_vol[wm_mask] = mad_norm.astype(np.float32)
            nib.save(nib.Nifti1Image(mad_vol, affine),
                     str(summary_dir / f"mad_norm_{m}.nii.gz"))

            log.info(f"{m}: MAD median={results[m]['mad_median']:.4f}, "
                     f"p95={results[m]['mad_p95']:.4f}"
                     + (f", Aₘ median={results[m].get('am_median', 'N/A')}" if am_wm is not None else ""))

    # Save summary JSON
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Figures ---
    _make_figures(cfg, out_dir, results, wm_mask, affine, n_trials, amb_dir)

    return results


def _make_figures(cfg, out_dir, results, wm_mask, affine, n_trials, amb_dir):
    """Generate figures for Exp 2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    ref_shape = wm_mask.shape

    # 1. Slice-by-slice MAD maps
    mid_z = ref_shape[2] // 2
    key_metrics = [m for m in ("fa", "md", "nd", "dispersion") if m in results]
    if key_metrics:
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(4 * len(key_metrics), 4))
        if len(key_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, key_metrics):
            mad_map = nib.load(str(summary_dir / f"mad_norm_{m}.nii.gz")).get_fdata()
            im = ax.imshow(mad_map[:, :, mid_z].T, origin="lower", cmap="hot",
                           vmin=0, vmax=np.percentile(mad_map[wm_mask], 99))
            ax.set_title(f"MAD {m.upper()}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(str(fig_dir / "exp2_mad_slices.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 2. Scatter: per-voxel aggregate MAD vs Aₘ
    for m in key_metrics:
        am_name = m if m != "ufa_voxel" else "ufa_voxel"
        am_path = amb_dir / f"{am_name}.nii.gz"
        if not am_path.exists():
            continue
        mad_map = nib.load(str(summary_dir / f"mad_norm_{m}.nii.gz")).get_fdata()
        am_map = nib.load(str(am_path)).get_fdata()
        if am_map.shape != ref_shape:
            continue

        mad_wm = mad_map[wm_mask]
        am_wm = am_map[wm_mask]

        # Subsample for plotting
        rng = np.random.default_rng(42)
        n_pts = min(10000, len(mad_wm))
        idx = rng.choice(len(mad_wm), n_pts, replace=False)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(am_wm[idx], mad_wm[idx], s=1, alpha=0.3, c="steelblue")
        ax.set_xlabel(f"Aₘ ({m.upper()})")
        ax.set_ylabel(f"Aggregate MAD ({m.upper()})")
        ax.set_title(f"Library-resampling stability vs concentration")

        # Correlation
        corr = np.corrcoef(am_wm, mad_wm)[0, 1]
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=10, va="top")
        plt.tight_layout()
        plt.savefig(str(fig_dir / f"exp2_scatter_mad_vs_am_{m}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"Scatter {m}: r = {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Library resampling stability")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip fitting, just recompute summary from existing trials")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp2_library_resampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials = cfg["exp2_n_trials"]

    # Add file handler
    fh = logging.FileHandler(out_dir / "exp2.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    if not args.summary_only:
        for t in range(n_trials):
            run_trial(t, cfg, out_dir, force_flag=args.force)

    results = compute_summary(cfg, out_dir, n_trials)

    # Print summary table
    print("\n" + "=" * 80)
    print("Exp 2 — Library Resampling Stability Summary")
    print("=" * 80)
    header = f"{'Metric':<18} {'Median Aₘ':>10} {'Median MAD':>12} {'P95 MAD':>10} {'Ratio':>8}"
    print(header)
    print("-" * len(header))
    for m in METRICS:
        if m not in results:
            continue
        r = results[m]
        am = r.get("am_median", None)
        ratio = r.get("ratio_median", None)
        print(f"{m:<18} {am if am is not None else 'N/A':>10.4f}"
              f" {r['mad_median']:>12.4f} {r['mad_p95']:>10.4f}"
              f" {ratio if ratio is not None else 'N/A':>8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
