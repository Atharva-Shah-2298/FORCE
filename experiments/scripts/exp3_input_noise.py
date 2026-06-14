#!/usr/bin/env python
"""Exp 3 — Input-noise perturbation stability.

Fix the library. Add independent Rician noise to the DWI at the subject's
estimated SNR. Run FORCE N_NOISE times with the same library. Compute CV
of aggregate outputs over WM.
"""
import argparse
import json
import logging
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

METRICS = ("fa", "md", "nd", "dispersion", "rd", "wm_fraction",
           "gm_fraction", "csf_fraction", "num_fibers", "ufa_voxel")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_wm_mask(cfg):
    import nibabel as nib
    ref_shape = nib.load(cfg["mask"]).shape
    wm_path = cfg.get("wm_mask_path")
    if wm_path and Path(wm_path).exists():
        wm = nib.load(wm_path).get_fdata().astype(bool)
        if wm.shape == ref_shape:
            return wm
    fa_path = Path(cfg["dwi"]).parent / "fa_dti.nii.gz"
    fa = nib.load(str(fa_path)).get_fdata()
    brain = nib.load(cfg["mask"]).get_fdata().astype(bool)
    return (fa > cfg["wm_fa_threshold"]) & brain


def estimate_snr(data, mask):
    """Estimate SNR from b=0 volumes using the background method.

    SNR = mean(signal in mask) / std(signal outside mask).
    Uses the first b=0 volume.
    """
    if data.ndim == 4:
        b0 = data[..., 0]
    else:
        b0 = data
    signal = b0[mask]
    # Background: outside mask
    bg = b0[~mask]
    bg = bg[bg > 0]  # exclude zeros
    if len(bg) == 0:
        return 20.0  # fallback
    noise_std = np.std(bg)
    if noise_std < 1e-12:
        return 20.0
    snr = np.mean(signal) / noise_std
    return float(snr)


def add_rician_noise(data, snr, rng):
    """Add Rician noise to DWI data.

    Parameters
    ----------
    data : ndarray
        Clean DWI data.
    snr : float
        Target SNR (mean_signal / sigma).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    noisy : ndarray
        Noisy data with Rician distribution.
    """
    sigma = np.mean(data[data > 0]) / snr
    real = data + rng.normal(0, sigma, data.shape)
    imag = rng.normal(0, sigma, data.shape)
    noisy = np.sqrt(real ** 2 + imag ** 2)
    return noisy.astype(np.float32)


def run_noise_trial(trial_idx, cfg, model, data, mask, affine, out_dir, snr,
                    force_flag=False):
    """Fit one noise realization."""
    import nibabel as nib

    trial_dir = out_dir / f"noise_{trial_idx:02d}"
    fa_path = trial_dir / "fa.nii.gz"
    if fa_path.exists() and not force_flag:
        log.info(f"Noise trial {trial_idx} already done, skipping")
        return

    trial_dir.mkdir(parents=True, exist_ok=True)

    seed = 10000 + trial_idx
    rng = np.random.default_rng(seed)
    log.info(f"--- Noise trial {trial_idx} (seed={seed}, SNR={snr:.1f}) ---")

    noisy_data = add_rician_noise(data, snr, rng)

    t0 = time.time()
    force_fit = model.fit(noisy_data, mask=mask, engine="serial", verbose=True)
    t_fit = time.time() - t0
    log.info(f"Fit done in {t_fit:.1f}s")

    for m in METRICS:
        arr = getattr(force_fit, m, None)
        if arr is not None:
            nib.save(nib.Nifti1Image(arr.astype(np.float32), affine),
                     str(trial_dir / f"{m}.nii.gz"))

    with open(trial_dir / "meta.json", "w") as f:
        json.dump({"seed": seed, "snr": snr, "fit_s": t_fit}, f, indent=2)


def compute_summary(cfg, out_dir, n_noise):
    """Compute CV metrics across noise trials."""
    import nibabel as nib

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    wm_mask = get_wm_mask(cfg)
    ref_shape = nib.load(cfg["mask"]).shape
    affine = nib.load(cfg["mask"]).affine
    n_wm = int(np.sum(wm_mask))

    amb_dir = Path(cfg["dwi"]).parent / "FORCE_latest" / "microstructure_ambiguity"

    results = {}
    for m in METRICS:
        stacks = []
        for t in range(n_noise):
            p = out_dir / f"noise_{t:02d}" / f"{m}.nii.gz"
            if not p.exists():
                break
            stacks.append(nib.load(str(p)).get_fdata())
        else:
            stack = np.array(stacks)
            wm_stack = stack[:, wm_mask]

            mean_val = np.mean(wm_stack, axis=0)
            std_val = np.std(wm_stack, axis=0)
            cv = std_val / (np.abs(mean_val) + 1e-12)

            results[m] = {
                "cv_mean": float(np.mean(cv)),
                "cv_median": float(np.median(cv)),
                "cv_p95": float(np.percentile(cv, 95)),
            }

            # Save CV map
            cv_vol = np.zeros(ref_shape, dtype=np.float32)
            cv_vol[wm_mask] = cv.astype(np.float32)
            nib.save(nib.Nifti1Image(cv_vol, affine),
                     str(summary_dir / f"cv_{m}.nii.gz"))

            # Load Aₘ for correlation
            am_path = amb_dir / f"{m}.nii.gz"
            if am_path.exists():
                am = nib.load(str(am_path)).get_fdata()
                if am.shape == ref_shape:
                    am_wm = am[wm_mask]
                    corr = float(np.corrcoef(cv, am_wm)[0, 1])
                    results[m]["am_corr"] = corr

            log.info(f"{m}: CV median={results[m]['cv_median']:.4f}, "
                     f"p95={results[m]['cv_p95']:.4f}")

    with open(summary_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    _make_figures(cfg, out_dir, results, wm_mask, n_noise, amb_dir)
    return results


def _make_figures(cfg, out_dir, results, wm_mask, n_noise, amb_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    ref_shape = wm_mask.shape
    mid_z = ref_shape[2] // 2

    key_metrics = [m for m in ("fa", "md", "nd", "dispersion") if m in results]

    # CV slice maps
    if key_metrics:
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(4 * len(key_metrics), 4))
        if len(key_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, key_metrics):
            cv_map = nib.load(str(summary_dir / f"cv_{m}.nii.gz")).get_fdata()
            im = ax.imshow(cv_map[:, :, mid_z].T, origin="lower", cmap="hot",
                           vmin=0, vmax=np.percentile(cv_map[wm_mask], 99))
            ax.set_title(f"CV {m.upper()}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(str(fig_dir / "exp3_cv_slices.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # Scatter CV vs Aₘ
    for m in key_metrics:
        am_path = amb_dir / f"{m}.nii.gz"
        if not am_path.exists():
            continue
        cv_map = nib.load(str(summary_dir / f"cv_{m}.nii.gz")).get_fdata()
        am_map = nib.load(str(am_path)).get_fdata()
        if am_map.shape != ref_shape:
            continue
        cv_wm = cv_map[wm_mask]
        am_wm = am_map[wm_mask]

        rng = np.random.default_rng(42)
        n_pts = min(10000, len(cv_wm))
        idx = rng.choice(len(cv_wm), n_pts, replace=False)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(am_wm[idx], cv_wm[idx], s=1, alpha=0.3, c="steelblue")
        ax.set_xlabel(f"Aₘ ({m.upper()})")
        ax.set_ylabel(f"CV ({m.upper()})")
        ax.set_title("Input-noise CV vs Aₘ")
        corr = results[m].get("am_corr", np.nan)
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=10, va="top")
        plt.tight_layout()
        plt.savefig(str(fig_dir / f"exp3_scatter_cv_vs_am_{m}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Input-noise stability")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp3_input_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_noise = cfg["exp3_n_noise"]

    fh = logging.FileHandler(out_dir / "exp3.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    if not args.summary_only:
        import nibabel as nib
        from dipy.core.gradients import gradient_table
        from dipy.io import read_bvals_bvecs
        from dipy.io.image import load_nifti, load_nifti_data
        from dipy.reconst.force import FORCEModel

        data, affine = load_nifti(cfg["dwi"])
        bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                              atol=cfg["bvecs_tol"])
        mask = load_nifti_data(cfg["mask"]).astype(bool)

        # Estimate SNR
        snr = estimate_snr(data, mask)
        log.info(f"Estimated SNR: {snr:.1f}")

        # Build model with FIXED library (use_cache=True)
        model = FORCEModel(
            gtab,
            penalty=cfg["penalty"],
            n_neighbors=cfg["n_neighbors"],
            use_posterior=True,
            posterior_beta=cfg["posterior_beta"],
            verbose=True,
        )
        model.generate(
            num_simulations=cfg["num_simulations"],
            num_cpus=-1,
            use_cache=True,
            verbose=True,
        )

        for t in range(n_noise):
            run_noise_trial(t, cfg, model, data, mask, affine, out_dir, snr,
                            force_flag=args.force)

    results = compute_summary(cfg, out_dir, n_noise)

    print("\n" + "=" * 80)
    print("Exp 3 — Input-Noise Perturbation Stability")
    print("=" * 80)
    header = f"{'Metric':<18} {'CV Median':>10} {'CV P95':>10} {'Aₘ Corr':>10}"
    print(header)
    print("-" * len(header))
    for m in METRICS:
        if m not in results:
            continue
        r = results[m]
        corr = r.get("am_corr", None)
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        print(f"{m:<18} {r['cv_median']:>10.4f} {r['cv_p95']:>10.4f} {corr_str:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
