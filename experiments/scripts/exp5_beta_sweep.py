#!/usr/bin/env python
"""Exp 5 — β (softmax temperature) sweep.

Single search pass, then reweight with different β values.
Expect a plateau in the moderate-β zone, collapsing to argmax at high β.
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
    parser = argparse.ArgumentParser(description="Exp 5: Beta sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp5_beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp5.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        log.info("Exp 5 already done. Use --force to rerun.")
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

    K = cfg["n_neighbors"]
    beta_values = cfg["exp5_beta_values"]
    beta_ref = cfg["posterior_beta"]

    model = FORCEModel(
        gtab,
        penalty=cfg["penalty"],
        n_neighbors=K,
        use_posterior=True,
        posterior_beta=beta_ref,
        verbose=True,
    )
    model.generate(num_simulations=cfg["num_simulations"], num_cpus=-1,
                   use_cache=True, verbose=True)

    # WM signals
    wm_data = np.ascontiguousarray(data[wm_mask], dtype=np.float32)
    norms = np.linalg.norm(wm_data, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    query_norm = np.ascontiguousarray(wm_data / norms)

    log.info(f"Searching top-{K} neighbors ...")
    D, neighbors = model._index.search(query_norm, k=K)
    S = D - model._penalty_array[neighbors]

    d = model.simulations
    prior_ranges = {}
    for p in SCALAR_PARAMS:
        if p in d:
            prior_ranges[p] = float(d[p].max() - d[p].min())

    # Reference at default beta
    W_ref = softmax_stable(beta_ref * S, axis=1)
    ref_vals = {}
    for p in SCALAR_PARAMS:
        if p in d:
            ref_vals[p] = np.sum(W_ref * d[p][neighbors], axis=1)

    # Also compute argmax values for comparison
    best = np.argmax(S, axis=1)
    lib_idx_best = neighbors[np.arange(n_wm), best]
    argmax_vals = {}
    for p in SCALAR_PARAMS:
        if p in d:
            argmax_vals[p] = d[p][lib_idx_best]

    # Sweep beta
    results = {p: {} for p in SCALAR_PARAMS if p in d}
    for beta in beta_values:
        log.info(f"Processing β={beta} ...")
        W_b = softmax_stable(beta * S, axis=1)

        # Effective number of neighbors (entropy-based)
        eff_k = np.exp(-np.sum(W_b * np.log(W_b + 1e-12), axis=1))

        for p in SCALAR_PARAMS:
            if p not in d:
                continue
            agg = np.sum(W_b * d[p][neighbors], axis=1)
            delta = np.abs(agg - ref_vals[p]) / (prior_ranges[p] + 1e-12)
            delta_argmax = np.abs(agg - argmax_vals[p]) / (prior_ranges[p] + 1e-12)

            results[p][beta] = {
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "p95_delta": float(np.percentile(delta, 95)),
                "median_delta_from_argmax": float(np.median(delta_argmax)),
                "median_eff_k": float(np.median(eff_k)),
            }

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    _make_figures(cfg, out_dir, results, beta_values, beta_ref)

    print("\n" + "=" * 80)
    print(f"Exp 5 — β Sweep (reference β={beta_ref})")
    print("=" * 80)
    header = "β         eff_K  " + "  ".join(f"{p:>12}" for p in SCALAR_PARAMS if p in results)
    print(header)
    print("-" * len(header))
    for beta in beta_values:
        eff_k = results[next(iter(results))][beta]["median_eff_k"]
        vals = "  ".join(
            f"{results[p][beta]['median_delta']:>12.5f}" if beta in results.get(p, {}) else f"{'N/A':>12}"
            for p in SCALAR_PARAMS if p in results
        )
        print(f"{beta:<10.0f}{eff_k:<7.1f}{vals}")
    print("=" * 80)


def _make_figures(cfg, out_dir, results, beta_values, beta_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key_params = [p for p in ("fa", "md", "nd", "dispersion") if p in results]

    # Plot delta vs beta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: delta from reference
    for p in key_params:
        medians = [results[p][b]["median_delta"] for b in beta_values]
        ax1.plot(beta_values, medians, "o-", label=p.upper(), markersize=4)
    ax1.axvline(beta_ref, color="gray", ls="--", alpha=0.5, label=f"β={beta_ref}")
    ax1.set_xlabel("β")
    ax1.set_ylabel("Median |Δ| / prior range (vs β=2000)")
    ax1.set_xscale("log")
    ax1.set_title("β Sweep — Deviation from Default")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: delta from argmax
    for p in key_params:
        medians = [results[p][b]["median_delta_from_argmax"] for b in beta_values]
        ax2.plot(beta_values, medians, "o-", label=p.upper(), markersize=4)
    ax2.set_xlabel("β")
    ax2.set_ylabel("Median |Δ| / prior range (vs argmax)")
    ax2.set_xscale("log")
    ax2.set_title("β Sweep — Distance from Argmax")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(fig_dir / "exp5_beta_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Effective K plot
    fig, ax = plt.subplots(figsize=(6, 4))
    first_param = next(iter(results))
    eff_ks = [results[first_param][b]["median_eff_k"] for b in beta_values]
    ax.plot(beta_values, eff_ks, "ko-", markersize=5)
    ax.set_xlabel("β")
    ax.set_ylabel("Median effective K")
    ax.set_xscale("log")
    ax.set_title("Effective Neighborhood Size vs β")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "exp5_effective_k.png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
