#!/usr/bin/env python
"""Exp 7 — DTI reference comparison.

Run DTI on the same noise-perturbed DWIs from Exp 3. Compare FORCE-aggregate
CV vs DTI CV for FA and MD.
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
    b0 = data[..., 0] if data.ndim == 4 else data
    signal = b0[mask]
    bg = b0[~mask]
    bg = bg[bg > 0]
    if len(bg) == 0:
        return 20.0
    noise_std = np.std(bg)
    if noise_std < 1e-12:
        return 20.0
    return float(np.mean(signal) / noise_std)


def add_rician_noise(data, snr, rng):
    sigma = np.mean(data[data > 0]) / snr
    real = data + rng.normal(0, sigma, data.shape)
    imag = rng.normal(0, sigma, data.shape)
    return np.sqrt(real ** 2 + imag ** 2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Exp 7: DTI reference comparison")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp7_dti_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp7.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        log.info("Exp 7 already done. Use --force to rerun.")
        return

    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.reconst.dti import TensorModel

    data, affine = load_nifti(cfg["dwi"])
    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                          atol=cfg["bvecs_tol"])
    brain_mask = load_nifti_data(cfg["mask"]).astype(bool)
    wm_mask = get_wm_mask(cfg)
    n_wm = int(np.sum(wm_mask))

    snr = estimate_snr(data, brain_mask)
    log.info(f"SNR: {snr:.1f}, WM voxels: {n_wm}")

    n_noise = cfg["exp3_n_noise"]  # reuse same count as Exp 3

    fa_stacks = []
    md_stacks = []

    for t in range(n_noise):
        trial_dir = out_dir / f"noise_{t:02d}"
        fa_path = trial_dir / "dti_fa.nii.gz"
        md_path = trial_dir / "dti_md.nii.gz"

        if fa_path.exists() and md_path.exists() and not args.force:
            log.info(f"DTI trial {t} already done, loading")
            fa_stacks.append(nib.load(str(fa_path)).get_fdata()[wm_mask])
            md_stacks.append(nib.load(str(md_path)).get_fdata()[wm_mask])
            continue

        trial_dir.mkdir(parents=True, exist_ok=True)
        seed = 10000 + t  # same seeds as Exp 3
        rng = np.random.default_rng(seed)
        noisy = add_rician_noise(data, snr, rng)

        log.info(f"DTI trial {t} (seed={seed}) ...")
        t0 = time.time()
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(noisy, mask=brain_mask)
        log.info(f"DTI fit done in {time.time() - t0:.1f}s")

        fa = tenfit.fa
        md = tenfit.md

        nib.save(nib.Nifti1Image(fa.astype(np.float32), affine), str(fa_path))
        nib.save(nib.Nifti1Image(md.astype(np.float32), affine), str(md_path))

        fa_stacks.append(fa[wm_mask])
        md_stacks.append(md[wm_mask])

    # Compute CV
    fa_stack = np.array(fa_stacks)  # (n_noise, n_wm)
    md_stack = np.array(md_stacks)

    fa_cv = np.std(fa_stack, axis=0) / (np.abs(np.mean(fa_stack, axis=0)) + 1e-12)
    md_cv = np.std(md_stack, axis=0) / (np.abs(np.mean(md_stack, axis=0)) + 1e-12)

    # Load FORCE CV from Exp 3
    exp3_dir = Path(cfg["output_root"]) / "exp3_input_noise" / "summary"
    force_fa_cv = None
    force_md_cv = None
    if (exp3_dir / "cv_fa.nii.gz").exists():
        force_fa_cv = nib.load(str(exp3_dir / "cv_fa.nii.gz")).get_fdata()[wm_mask]
    if (exp3_dir / "cv_md.nii.gz").exists():
        force_md_cv = nib.load(str(exp3_dir / "cv_md.nii.gz")).get_fdata()[wm_mask]

    results = {
        "dti_fa_cv_median": float(np.median(fa_cv)),
        "dti_fa_cv_p95": float(np.percentile(fa_cv, 95)),
        "dti_md_cv_median": float(np.median(md_cv)),
        "dti_md_cv_p95": float(np.percentile(md_cv, 95)),
    }
    if force_fa_cv is not None:
        results["force_fa_cv_median"] = float(np.median(force_fa_cv))
        results["ratio_fa_cv"] = float(np.median(force_fa_cv) / (np.median(fa_cv) + 1e-12))
    if force_md_cv is not None:
        results["force_md_cv_median"] = float(np.median(force_md_cv))
        results["ratio_md_cv"] = float(np.median(force_md_cv) / (np.median(md_cv) + 1e-12))

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    _make_figures(cfg, out_dir, fa_cv, md_cv, force_fa_cv, force_md_cv, wm_mask)

    print("\n" + "=" * 80)
    print("Exp 7 — DTI Reference Comparison")
    print("=" * 80)
    print(f"DTI FA CV: median={results['dti_fa_cv_median']:.4f}, "
          f"p95={results['dti_fa_cv_p95']:.4f}")
    print(f"DTI MD CV: median={results['dti_md_cv_median']:.4f}, "
          f"p95={results['dti_md_cv_p95']:.4f}")
    if "force_fa_cv_median" in results:
        print(f"FORCE FA CV: median={results['force_fa_cv_median']:.4f}, "
              f"ratio={results['ratio_fa_cv']:.2f}x DTI")
    if "force_md_cv_median" in results:
        print(f"FORCE MD CV: median={results['force_md_cv_median']:.4f}, "
              f"ratio={results['ratio_md_cv']:.2f}x DTI")
    print("=" * 80)


def _make_figures(cfg, out_dir, fa_cv, md_cv, force_fa_cv, force_md_cv, wm_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric, dti_cv, force_cv in [("FA", fa_cv, force_fa_cv),
                                      ("MD", md_cv, force_md_cv)]:
        if force_cv is None:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Histograms
        bins = np.linspace(0, np.percentile(np.concatenate([dti_cv, force_cv]), 99), 80)
        ax1.hist(dti_cv, bins=bins, density=True, alpha=0.6, label=f"DTI {metric}",
                 color="steelblue")
        ax1.hist(force_cv, bins=bins, density=True, alpha=0.6, label=f"FORCE {metric}",
                 color="coral")
        ax1.set_xlabel(f"CV ({metric})")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.set_title(f"Noise CV: FORCE vs DTI — {metric}")

        # Scatter
        rng = np.random.default_rng(42)
        n_pts = min(10000, len(dti_cv))
        idx = rng.choice(len(dti_cv), n_pts, replace=False)
        ax2.scatter(dti_cv[idx], force_cv[idx], s=1, alpha=0.3, c="steelblue")
        lim = max(np.percentile(dti_cv, 99), np.percentile(force_cv, 99))
        ax2.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
        ax2.set_xlabel(f"DTI {metric} CV")
        ax2.set_ylabel(f"FORCE {metric} CV")
        ax2.legend()
        ax2.set_title(f"Per-voxel CV: FORCE vs DTI")

        plt.tight_layout()
        plt.savefig(str(fig_dir / f"exp7_dti_vs_force_{metric.lower()}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
