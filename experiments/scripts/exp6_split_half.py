#!/usr/bin/env python
"""Exp 6 — Split-half library consistency.

Split the 500K library into two disjoint 250K halves, re-match on each half,
aggregate, compare. Repeat N_SPLITS times. Cheaper alternative to Exp 2.
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


def search_and_aggregate(query_norm, signals_norm, penalty_array, params_dict,
                         K, beta, scalar_params):
    """Build index from signals, search, aggregate."""
    from dipy.reconst.force import create_signal_index, softmax_stable

    index = create_signal_index(signals_norm)
    D, neighbors = index.search(query_norm, k=K)
    S = D - penalty_array[neighbors]
    W = softmax_stable(beta * S, axis=1)

    agg = {}
    for p in scalar_params:
        if p in params_dict:
            agg[p] = np.sum(W * params_dict[p][neighbors], axis=1)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Exp 6: Split-half library")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_root"]) / "exp6_split_half"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp6.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        log.info("Exp 6 already done. Use --force to rerun.")
        return

    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, load_nifti_data
    from dipy.reconst.force import FORCEModel

    data, affine = load_nifti(cfg["dwi"])
    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=cfg["b0_threshold"],
                          atol=cfg["bvecs_tol"])
    brain_mask = load_nifti_data(cfg["mask"]).astype(bool)
    wm_mask = get_wm_mask(cfg)
    n_wm = int(np.sum(wm_mask))
    log.info(f"WM voxels: {n_wm}")

    K = cfg["n_neighbors"]
    beta = cfg["posterior_beta"]
    penalty = cfg["penalty"]
    n_splits = cfg["exp6_n_splits"]

    # Load library (use cache)
    model = FORCEModel(
        gtab, penalty=penalty, n_neighbors=K,
        use_posterior=True, posterior_beta=beta, verbose=True,
    )
    model.generate(num_simulations=cfg["num_simulations"], num_cpus=-1,
                   use_cache=True, verbose=True)

    d = model.simulations
    n_lib = d["signals"].shape[0]
    log.info(f"Library size: {n_lib}")

    # Normalize library signals
    lib_signals = d["signals"].astype(np.float32)
    lib_norms = np.linalg.norm(lib_signals, axis=1, keepdims=True)
    lib_norms[lib_norms == 0] = 1.0
    lib_signals_norm = np.ascontiguousarray(lib_signals / lib_norms)

    # Penalty array for full library
    num_fibers = d.get("num_fibers", np.zeros(n_lib, dtype=np.float32))
    penalty_full = (penalty * num_fibers).astype(np.float32)

    # WM signals
    wm_data = np.ascontiguousarray(data[wm_mask], dtype=np.float32)
    norms = np.linalg.norm(wm_data, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    query_norm = np.ascontiguousarray(wm_data / norms)

    # Prior ranges
    prior_ranges = {}
    for p in SCALAR_PARAMS:
        if p in d:
            prior_ranges[p] = float(d[p].max() - d[p].min())

    # Split-half trials
    # For each split, compute |half_A - half_B| per voxel
    all_diffs = {p: [] for p in SCALAR_PARAMS if p in d}

    for s in range(n_splits):
        rng = np.random.default_rng(seed=s)
        perm = rng.permutation(n_lib)
        half = n_lib // 2
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        log.info(f"Split {s}: half_A={len(idx_a)}, half_B={len(idx_b)}")

        for label, idx in [("A", idx_a), ("B", idx_b)]:
            # Build sub-library
            sub_signals = lib_signals_norm[idx]
            sub_penalty = penalty_full[idx]
            sub_params = {}
            for p in SCALAR_PARAMS:
                if p in d:
                    sub_params[p] = d[p][idx]

            t0 = time.time()
            agg = search_and_aggregate(query_norm, sub_signals, sub_penalty,
                                       sub_params, K, beta, SCALAR_PARAMS)
            log.info(f"  Half {label} done in {time.time() - t0:.1f}s")

            if label == "A":
                agg_a = agg
            else:
                agg_b = agg

        for p in SCALAR_PARAMS:
            if p in agg_a and p in agg_b:
                diff = np.abs(agg_a[p] - agg_b[p]) / (prior_ranges[p] + 1e-12)
                all_diffs[p].append(diff)

    # Aggregate over splits
    results = {}
    for p in SCALAR_PARAMS:
        if p not in all_diffs or not all_diffs[p]:
            continue
        stack = np.array(all_diffs[p])  # (n_splits, n_wm)
        mean_diff = np.mean(stack, axis=0)
        results[p] = {
            "half_diff_mean": float(np.mean(mean_diff)),
            "half_diff_median": float(np.median(mean_diff)),
            "half_diff_p95": float(np.percentile(mean_diff, 95)),
        }
        log.info(f"{p}: half-diff median={results[p]['half_diff_median']:.5f}, "
                 f"p95={results[p]['half_diff_p95']:.5f}")

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Exp 6 — Split-Half Library Consistency")
    print("=" * 80)
    header = f"{'Param':<18} {'Half-Diff Med':>14} {'Half-Diff P95':>14}"
    print(header)
    print("-" * len(header))
    for p in SCALAR_PARAMS:
        if p not in results:
            continue
        r = results[p]
        print(f"{p:<18} {r['half_diff_median']:>14.5f} {r['half_diff_p95']:>14.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
