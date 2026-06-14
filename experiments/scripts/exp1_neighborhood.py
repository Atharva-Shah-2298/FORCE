#!/usr/bin/env python
"""Exp 1 — Neighborhood concentration analysis (postprocess, no re-fit).

Run FORCE once (or reuse cached library). For every WM voxel, extract the
top-K neighborhood and compute:
  - Per-parameter unweighted std inside top-K (upper bound on movement)
  - Per-parameter softmax-weighted std (what the aggregate actually sees)
  - Aₘ(v) from the already-saved ambiguity maps

This requires direct access to the search index and neighbor indices,
so we use the Python API rather than the CLI.
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

SCALAR_PARAMS = ("fa", "md", "nd", "dispersion", "rd", "wm_fraction",
                 "gm_fraction", "csf_fraction", "num_fibers", "ufa_voxel")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_wm_mask(cfg):
    """Load or create WM mask."""
    import nibabel as nib
    ref_shape = nib.load(cfg["mask"]).shape
    wm_path = cfg.get("wm_mask_path")
    if wm_path and Path(wm_path).exists():
        wm = nib.load(wm_path).get_fdata().astype(bool)
        if wm.shape == ref_shape:
            return wm
    # Fallback: FA > threshold
    fa_path = Path(cfg["dwi"]).parent / "fa_dti.nii.gz"
    fa = nib.load(str(fa_path)).get_fdata()
    brain = nib.load(cfg["mask"]).get_fdata().astype(bool)
    return (fa > cfg["wm_fa_threshold"]) & brain


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Neighborhood concentration")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp1_neighborhood"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp1.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        log.info("Exp 1 summary already exists. Use --force to rerun.")
        return

    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.reconst.force import (
        FORCEModel, MICRO_PARAMS, normalize_signals, softmax_stable
    )

    # Load data
    log.info("Loading data ...")
    data, affine = load_nifti(cfg["dwi"])
    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                          atol=cfg["bvecs_tol"])
    brain_mask = load_nifti_data(cfg["mask"]).astype(bool)
    wm_mask = get_wm_mask(cfg)
    n_wm = int(np.sum(wm_mask))
    log.info(f"WM mask: {n_wm} voxels")

    # Build model + library (use cache)
    model = FORCEModel(
        gtab,
        penalty=cfg["penalty"],
        n_neighbors=cfg["n_neighbors"],
        use_posterior=True,
        posterior_beta=cfg["posterior_beta"],
        verbose=True,
    )
    log.info("Generating/loading library ...")
    model.generate(
        num_simulations=cfg["num_simulations"],
        num_cpus=-1,
        use_cache=True,
        verbose=True,
    )

    # Extract WM voxel signals
    log.info("Extracting WM voxel signals ...")
    wm_data = data[wm_mask]  # (n_wm, n_grad)
    wm_data = np.ascontiguousarray(wm_data, dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(wm_data, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    query_norm = np.ascontiguousarray(wm_data / norms)

    # Search
    K = cfg["n_neighbors"]
    log.info(f"Searching top-{K} neighbors ...")
    t0 = time.time()
    D, neighbors = model._index.search(query_norm, k=K)
    log.info(f"Search done in {time.time() - t0:.1f}s")

    # Penalized scores
    S = D - model._penalty_array[neighbors]

    # Softmax weights
    beta = cfg["posterior_beta"]
    W = softmax_stable(beta * S, axis=1)  # (n_wm, K)

    # For each scalar param, compute neighborhood stats
    d = model.simulations
    results = {}
    csv_data = {"voxel_idx": np.arange(n_wm)}

    for param in SCALAR_PARAMS:
        if param not in d:
            log.warning(f"Param {param} not in simulations, skipping")
            continue

        lib_vals = d[param]  # (n_lib,)
        prior_range = float(lib_vals.max() - lib_vals.min())

        # Neighborhood values: (n_wm, K)
        neigh_vals = lib_vals[neighbors]

        # Unweighted std
        std_unw = np.std(neigh_vals, axis=1)
        std_unw_norm = std_unw / (prior_range + 1e-12)

        # Weighted std
        wmean = np.sum(W * neigh_vals, axis=1)
        wvar = np.sum(W * (neigh_vals - wmean[:, None]) ** 2, axis=1)
        std_w = np.sqrt(np.maximum(wvar, 0))
        std_w_norm = std_w / (prior_range + 1e-12)

        results[param] = {
            "prior_range": prior_range,
            "unweighted_std_median": float(np.median(std_unw_norm)),
            "unweighted_std_p95": float(np.percentile(std_unw_norm, 95)),
            "weighted_std_median": float(np.median(std_w_norm)),
            "weighted_std_p95": float(np.percentile(std_w_norm, 95)),
        }

        csv_data[f"{param}_std_unw"] = std_unw_norm
        csv_data[f"{param}_std_w"] = std_w_norm

        log.info(f"{param}: unw_std median={results[param]['unweighted_std_median']:.4f}, "
                 f"w_std median={results[param]['weighted_std_median']:.4f}")

    # Load existing Aₘ maps and add to results
    amb_dir = Path(cfg["dwi"]).parent / "FORCE_latest" / "microstructure_ambiguity"
    for param in SCALAR_PARAMS:
        if param not in results:
            continue
        am_name = param
        am_path = amb_dir / f"{am_name}.nii.gz"
        if am_path.exists():
            am_data = nib.load(str(am_path)).get_fdata()
            if am_data.shape == brain_mask.shape:
                am_wm = am_data[wm_mask]
                results[param]["am_median"] = float(np.median(am_wm))
                results[param]["am_p95"] = float(np.percentile(am_wm, 95))
                csv_data[f"{param}_am"] = am_wm

    # Save summary JSON
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV (subsampled if too large)
    n_save = min(n_wm, 50000)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_wm, n_save, replace=False) if n_save < n_wm else np.arange(n_wm)
    import csv
    csv_path = out_dir / "neighborhood_stats.csv"
    fields = list(csv_data.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in idx:
            writer.writerow([csv_data[k][i] if hasattr(csv_data[k], '__getitem__') else csv_data[k]
                             for k in fields])
    log.info(f"CSV saved: {csv_path} ({n_save} rows)")

    # --- Figures ---
    _make_figures(cfg, out_dir, results, csv_data, wm_mask, affine)

    # Print summary table
    print("\n" + "=" * 80)
    print("Exp 1 — Neighborhood Concentration Summary")
    print("=" * 80)
    header = (f"{'Param':<18} {'Unw Std Med':>12} {'Unw Std P95':>12} "
              f"{'W Std Med':>12} {'W Std P95':>12} {'Aₘ Med':>10}")
    print(header)
    print("-" * len(header))
    for param in SCALAR_PARAMS:
        if param not in results:
            continue
        r = results[param]
        am = r.get("am_median", "N/A")
        am_str = f"{am:.4f}" if isinstance(am, float) else am
        print(f"{param:<18} {r['unweighted_std_median']:>12.4f} "
              f"{r['unweighted_std_p95']:>12.4f} "
              f"{r['weighted_std_median']:>12.4f} {r['weighted_std_p95']:>12.4f} "
              f"{am_str:>10}")
    print("=" * 80)


def _make_figures(cfg, out_dir, results, csv_data, wm_mask, affine):
    """Generate Exp 1 figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key_params = [p for p in ("fa", "md", "nd", "dispersion", "num_fibers")
                  if p in results]

    # Histograms of Aₘ
    has_am = [p for p in key_params if f"{p}_am" in csv_data]
    if has_am:
        fig, axes = plt.subplots(1, len(has_am), figsize=(4 * len(has_am), 3.5))
        if len(has_am) == 1:
            axes = [axes]
        for ax, p in zip(axes, has_am):
            am = csv_data[f"{p}_am"]
            ax.hist(am, bins=100, density=True, alpha=0.7, color="steelblue")
            ax.axvline(np.median(am), color="red", ls="--", label=f"median={np.median(am):.3f}")
            ax.set_xlabel(f"Aₘ ({p.upper()})")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(str(fig_dir / "exp1_am_histograms.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # Histograms of weighted std
    fig, axes = plt.subplots(1, len(key_params), figsize=(4 * len(key_params), 3.5))
    if len(key_params) == 1:
        axes = [axes]
    for ax, p in zip(axes, key_params):
        w_std = csv_data.get(f"{p}_std_w")
        unw_std = csv_data.get(f"{p}_std_unw")
        if w_std is not None:
            ax.hist(w_std, bins=100, density=True, alpha=0.6, color="steelblue",
                    label="Weighted")
        if unw_std is not None:
            ax.hist(unw_std, bins=100, density=True, alpha=0.4, color="orange",
                    label="Unweighted")
        ax.set_xlabel(f"Std / prior range ({p.upper()})")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "exp1_neighborhood_std_hist.png"), dpi=200, bbox_inches="tight")
    plt.close()

    log.info("Figures saved to " + str(fig_dir))


if __name__ == "__main__":
    main()
