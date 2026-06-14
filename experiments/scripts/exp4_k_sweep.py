#!/usr/bin/env python
"""Exp 4 — K sweep.

Run search once with max K, then postprocess with different K values.
K is a postprocessing parameter over the already-ranked cosine similarities.
No library regeneration needed.
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
                 "num_fibers", "ufa_voxel")


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


def main():
    parser = argparse.ArgumentParser(description="Exp 4: K sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp4_k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp4.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        log.info("Exp 4 already done. Use --force to rerun.")
        return

    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.reconst.force import FORCEModel, softmax_stable

    data, affine = load_nifti(cfg["dwi"])
    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                          atol=cfg["bvecs_tol"])
    brain_mask = load_nifti_data(cfg["mask"]).astype(bool)
    wm_mask = get_wm_mask(cfg)
    n_wm = int(np.sum(wm_mask))
    log.info(f"WM voxels: {n_wm}")

    K_values = cfg["exp4_k_values"]
    K_max = max(K_values)
    beta = cfg["posterior_beta"]
    K_ref = cfg["n_neighbors"]  # reference K=50

    # Build model with max K
    model = FORCEModel(
        gtab,
        penalty=cfg["penalty"],
        n_neighbors=K_max,
        use_posterior=True,
        posterior_beta=beta,
        verbose=True,
    )
    model.generate(num_simulations=cfg["num_simulations"], num_cpus=-1,
                   use_cache=True, verbose=True)

    # Extract WM signals
    wm_data = np.ascontiguousarray(data[wm_mask], dtype=np.float32)
    norms = np.linalg.norm(wm_data, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    query_norm = np.ascontiguousarray(wm_data / norms)

    # Single search with K_max
    log.info(f"Searching top-{K_max} neighbors ...")
    t0 = time.time()
    D, neighbors = model._index.search(query_norm, k=K_max)
    log.info(f"Search done in {time.time() - t0:.1f}s")

    S_full = D - model._penalty_array[neighbors]  # (n_wm, K_max)

    d = model.simulations
    # Compute prior ranges
    prior_ranges = {}
    for p in SCALAR_PARAMS:
        if p in d:
            prior_ranges[p] = float(d[p].max() - d[p].min())

    # Reference: posterior-averaged values at K_ref
    ref_vals = {}
    K_ref_idx = min(K_ref, K_max)
    S_ref = S_full[:, :K_ref_idx]
    W_ref = softmax_stable(beta * S_ref, axis=1)
    for p in SCALAR_PARAMS:
        if p in d:
            neigh_vals = d[p][neighbors[:, :K_ref_idx]]
            ref_vals[p] = np.sum(W_ref * neigh_vals, axis=1)

    # Sweep K
    results = {p: {} for p in SCALAR_PARAMS if p in d}
    for K in K_values:
        log.info(f"Processing K={K} ...")
        S_k = S_full[:, :K]
        W_k = softmax_stable(beta * S_k, axis=1)

        for p in SCALAR_PARAMS:
            if p not in d:
                continue
            neigh_vals = d[p][neighbors[:, :K]]
            agg = np.sum(W_k * neigh_vals, axis=1)

            # Change relative to K_ref, normalized by prior range
            delta = np.abs(agg - ref_vals[p]) / (prior_ranges[p] + 1e-12)
            results[p][K] = {
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "p95_delta": float(np.percentile(delta, 95)),
            }

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Figures ---
    _make_figures(cfg, out_dir, results, K_values, K_ref)

    # Print
    print("\n" + "=" * 80)
    print(f"Exp 4 — K Sweep (reference K={K_ref})")
    print("=" * 80)
    header = "K     " + "  ".join(f"{p:>12}" for p in SCALAR_PARAMS if p in results)
    print(header)
    print("-" * len(header))
    for K in K_values:
        vals = "  ".join(
            f"{results[p][K]['median_delta']:>12.5f}" if K in results.get(p, {}) else f"{'N/A':>12}"
            for p in SCALAR_PARAMS if p in results
        )
        print(f"{K:<6}{vals}")
    print("=" * 80)


def _make_figures(cfg, out_dir, results, K_values, K_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key_params = [p for p in ("fa", "md", "nd", "dispersion") if p in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in key_params:
        medians = [results[p][K]["median_delta"] for K in K_values]
        ax.plot(K_values, medians, "o-", label=p.upper(), markersize=4)
    ax.axvline(K_ref, color="gray", ls="--", alpha=0.5, label=f"K={K_ref} (default)")
    ax.set_xlabel("K (number of neighbors)")
    ax.set_ylabel("Median |Δ| / prior range (vs K=50)")
    ax.set_xscale("log")
    ax.set_title("K Sweep — Aggregate Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "exp4_k_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
