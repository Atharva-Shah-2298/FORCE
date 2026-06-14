#!/usr/bin/env python
"""Exp 8 - FORCE dictionary adequacy controls.

This experiment treats dictionary dependence as a testable model-adequacy
question. It generates several FORCE dictionaries, including deliberately
restricted controls, then asks whether each dictionary can predict held-out
diffusion measurements and whether posterior scalar estimates remain stable.
"""
import argparse
import csv
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

DEFAULT_DIFFUSIVITY = {
    "wm_d_par_range": (2.0e-3, 3.0e-3),
    "wm_d_perp_range": (0.3e-3, 1.5e-3),
    "gm_d_iso_range": (0.7e-3, 1.2e-3),
    "csf_d": 3.0e-3,
}

VARIANTS = {
    "default": {
        "label": "Default",
        "diffusivity_config": DEFAULT_DIFFUSIVITY,
        "post_filter": None,
    },
    "widened_diffusivity": {
        "label": "Widened diffusivity",
        "diffusivity_config": {
            "wm_d_par_range": (1.5e-3, 3.5e-3),
            "wm_d_perp_range": (0.1e-3, 2.0e-3),
            "gm_d_iso_range": (0.5e-3, 1.5e-3),
            "csf_d": 3.0e-3,
        },
        "post_filter": None,
    },
    "broad_diffusivity": {
        "label": "Broad diffusivity",
        "diffusivity_config": {
            "wm_d_par_range": (1.0e-3, 4.0e-3),
            "wm_d_perp_range": (0.05e-3, 2.5e-3),
            "gm_d_iso_range": (0.3e-3, 2.0e-3),
            "csf_d": 3.0e-3,
        },
        "post_filter": None,
    },
    "very_broad_diffusivity": {
        "label": "Very broad diffusivity",
        "diffusivity_config": {
            "wm_d_par_range": (0.5e-3, 4.5e-3),
            "wm_d_perp_range": (0.02e-3, 3.0e-3),
            "gm_d_iso_range": (0.2e-3, 2.5e-3),
            "csf_d": 3.0e-3,
        },
        "post_filter": None,
    },
    "narrowed_diffusivity": {
        "label": "Narrowed diffusivity",
        "diffusivity_config": {
            "wm_d_par_range": (2.2e-3, 2.8e-3),
            "wm_d_perp_range": (0.5e-3, 1.1e-3),
            "gm_d_iso_range": (0.8e-3, 1.0e-3),
            "csf_d": 3.0e-3,
        },
        "post_filter": None,
    },
    "limited_csf": {
        "label": "Limited CSF fraction",
        "diffusivity_config": DEFAULT_DIFFUSIVITY,
        "post_filter": {"key": "csf_fraction", "op": "<=", "value": 0.05},
    },
    "no_crossing": {
        "label": "No crossing fibers",
        "diffusivity_config": DEFAULT_DIFFUSIVITY,
        "post_filter": {"key": "num_fibers", "op": "<=", "value": 1.0},
    },
}

SCALAR_PARAMS = (
    "fa",
    "md",
    "rd",
    "nd",
    "dispersion",
    "wm_fraction",
    "gm_fraction",
    "csf_fraction",
    "num_fibers",
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def as_jsonable(obj):
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def b0_normalize(signals, bvals, b0_threshold):
    b0 = bvals <= b0_threshold
    s0 = signals[..., b0].mean(axis=-1, keepdims=True)
    s0[s0 == 0] = 1.0
    return signals / s0


def l2_normalize(signals):
    norm = np.linalg.norm(signals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.ascontiguousarray(signals / norm, dtype=np.float32)


def stable_softmax(scores, beta):
    scaled = beta * scores
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / (np.sum(weights, axis=1, keepdims=True) + 1e-30)


def finite_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": np.nan, "mean": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def select_eval_voxels(cfg, data_shape, max_voxels, seed, z_slice=None):
    import nibabel as nib

    brain = nib.load(cfg["mask"]).get_fdata().astype(bool)
    if brain.shape != tuple(data_shape[:3]):
        raise ValueError(f"Mask shape {brain.shape} does not match DWI shape {data_shape[:3]}")

    if z_slice is not None:
        slab = np.zeros_like(brain, dtype=bool)
        slab[:, :, z_slice] = True
        brain &= slab

    candidates = np.argwhere(brain)
    if candidates.size == 0:
        raise ValueError("No evaluation voxels found in mask")

    rng = np.random.default_rng(seed)
    n = min(max_voxels, candidates.shape[0])
    chosen = rng.choice(candidates.shape[0], size=n, replace=False)
    coords = candidates[chosen]

    tissue = np.full(n, "brain", dtype=object)
    for name, key in (("WM", "wm_mask_path"), ("GM", "gm_mask_path"), ("CSF", "csf_mask_path")):
        path = cfg.get(key)
        if not path or not Path(path).exists():
            continue
        mask = nib.load(path).get_fdata().astype(bool)
        if mask.shape == brain.shape:
            tissue[mask[coords[:, 0], coords[:, 1], coords[:, 2]]] = name

    return coords, tissue


def make_holdout_split(bvals, b0_threshold, holdout_fraction, seed):
    rng = np.random.default_rng(seed)
    b0 = bvals <= b0_threshold
    train = b0.copy()
    test = np.zeros_like(b0, dtype=bool)

    nonzero_shells = np.unique(np.round(bvals[~b0] / 100.0) * 100.0)
    for shell in nonzero_shells:
        shell_idx = np.where((~b0) & (np.abs(bvals - shell) <= 100))[0]
        if shell_idx.size == 0:
            continue
        n_test = max(1, int(round(shell_idx.size * holdout_fraction)))
        held = rng.choice(shell_idx, size=n_test, replace=False)
        test[held] = True

    train |= (~b0) & (~test)
    if not np.any(test):
        raise ValueError("No held-out volumes selected")
    return train, test


def apply_post_filter(simulations, post_filter):
    if post_filter is None:
        return simulations, int(simulations["signals"].shape[0]), int(simulations["signals"].shape[0])

    key = post_filter["key"]
    value = float(post_filter["value"])
    if post_filter["op"] == "<=":
        keep = np.asarray(simulations[key]) <= value
    else:
        raise ValueError(f"Unsupported filter operation: {post_filter['op']}")

    before = int(keep.size)
    after = int(np.sum(keep))
    if after == 0:
        raise ValueError(f"Filter {post_filter} removed all dictionary entries")

    filtered = {}
    for k, v in simulations.items():
        arr = np.asarray(v)
        if arr.shape and arr.shape[0] == before:
            filtered[k] = arr[keep]
        else:
            filtered[k] = arr
    return filtered, before, after


def subset_simulations(simulations, indices):
    indices = np.asarray(indices)
    n_entries = simulations["signals"].shape[0]
    subset = {}
    for k, v in simulations.items():
        arr = np.asarray(v)
        if arr.shape and arr.shape[0] == n_entries:
            subset[k] = arr[indices]
        else:
            subset[k] = arr
    return subset


def load_or_generate_variant(name, spec, cfg, out_dir, force, num_simulations, num_cpus):
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.reconst.force import FORCEModel

    variant_dir = out_dir / "libraries" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    path = variant_dir / "simulations.npz"
    meta_path = variant_dir / "metadata.json"

    if path.exists() and not force:
        log.info("Loading cached dictionary %s", path)
        loaded = dict(np.load(path))
        with open(meta_path) as f:
            meta = json.load(f)
        return loaded, meta

    bvals, bvecs = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    gtab = gradient_table(
        bvals,
        bvecs=bvecs,
        b0_threshold=cfg["b0_threshold"],
        atol=cfg["bvecs_tol"],
    )
    model = FORCEModel(
        gtab,
        penalty=cfg["penalty"],
        n_neighbors=cfg["n_neighbors"],
        use_posterior=True,
        posterior_beta=cfg["posterior_beta"],
        verbose=True,
    )

    t0 = time.time()
    log.info("Generating %s dictionary with %d simulations", name, num_simulations)
    model.generate(
        num_simulations=num_simulations,
        num_cpus=num_cpus,
        use_cache=False,
        diffusivity_config=spec["diffusivity_config"],
        compute_dti=True,
        compute_dki=False,
        verbose=True,
    )
    simulations = model.simulations
    generated_s = time.time() - t0
    simulations, before, after = apply_post_filter(simulations, spec["post_filter"])

    np.savez_compressed(path, **simulations)
    meta = {
        "name": name,
        "label": spec["label"],
        "diffusivity_config": as_jsonable(spec["diffusivity_config"]),
        "post_filter": as_jsonable(spec["post_filter"]),
        "generated_entries": before,
        "effective_entries": after,
        "generated_s": generated_s,
        "path": str(path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return simulations, meta


def posterior_predict_variant(
    simulations,
    signals_norm,
    bvals,
    b0_threshold,
    train_idx,
    test_idx,
    penalty,
    beta,
    n_neighbors,
):
    from dipy.reconst.force import create_signal_index

    lib = b0_normalize(
        simulations["signals"].astype(np.float32),
        bvals,
        b0_threshold,
    ).astype(np.float32)
    lib_train = lib[:, train_idx]
    lib_test = lib[:, test_idx]
    lib_train_norm = l2_normalize(lib_train)
    index = create_signal_index(lib_train_norm)

    query = l2_normalize(signals_norm[:, train_idx])
    k = min(n_neighbors, lib.shape[0])
    distances, neighbors = index.search(query, k=k)
    penalty_array = penalty * np.asarray(simulations["num_fibers"], dtype=np.float32)
    scores = distances - penalty_array[neighbors]
    weights = stable_softmax(scores, beta)

    pred_test = np.einsum("nk,nkm->nm", weights, lib_test[neighbors])
    obs_test = signals_norm[:, test_idx]
    residual = pred_test - obs_test
    rmse_vox = np.sqrt(np.mean(residual * residual, axis=1))
    mae_vox = np.mean(np.abs(residual), axis=1)
    bias_vox = np.mean(residual, axis=1)
    residual_mad_vox = np.median(np.abs(residual - np.median(residual, axis=1, keepdims=True)), axis=1)
    denom = np.sqrt(np.mean(obs_test * obs_test, axis=1)) + 1e-12
    nrmse_vox = rmse_vox / denom

    pred_norm = np.linalg.norm(pred_test, axis=1)
    obs_norm = np.linalg.norm(obs_test, axis=1)
    cosine_vox = np.sum(pred_test * obs_test, axis=1) / (pred_norm * obs_norm + 1e-12)
    ss_res = np.sum(residual * residual, axis=1)
    obs_centered = obs_test - np.mean(obs_test, axis=1, keepdims=True)
    ss_tot = np.sum(obs_centered * obs_centered, axis=1)
    r2_vox = 1.0 - ss_res / (ss_tot + 1e-12)
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1)
    eff_k = np.exp(entropy)

    shell_metrics = {}
    test_bvals = bvals[test_idx]
    for shell in np.unique(np.round(test_bvals / 100.0) * 100.0):
        if shell <= b0_threshold:
            continue
        shell_mask = np.abs(test_bvals - shell) <= 100
        if not np.any(shell_mask):
            continue
        shell_obs = obs_test[:, shell_mask]
        shell_pred = pred_test[:, shell_mask]
        shell_res = shell_pred - shell_obs
        shell_rmse = np.sqrt(np.mean(shell_res * shell_res, axis=1))
        shell_denom = np.sqrt(np.mean(shell_obs * shell_obs, axis=1)) + 1e-12
        shell_pred_norm = np.linalg.norm(shell_pred, axis=1)
        shell_obs_norm = np.linalg.norm(shell_obs, axis=1)
        shell_metrics[int(shell)] = {
            "nrmse": shell_rmse / shell_denom,
            "mae": np.mean(np.abs(shell_res), axis=1),
            "bias": np.mean(shell_res, axis=1),
            "cosine": np.sum(shell_pred * shell_obs, axis=1) / (shell_pred_norm * shell_obs_norm + 1e-12),
        }

    scalar_maps = {}
    for param in SCALAR_PARAMS:
        if param in simulations:
            vals = np.asarray(simulations[param], dtype=np.float32)
            scalar_maps[param] = np.sum(weights * vals[neighbors], axis=1)

    return {
        "neighbors": neighbors,
        "weights": weights,
        "rmse": rmse_vox,
        "mae": mae_vox,
        "bias": bias_vox,
        "residual_mad": residual_mad_vox,
        "nrmse": nrmse_vox,
        "cosine": cosine_vox,
        "r2": r2_vox,
        "entropy": entropy,
        "eff_k": eff_k,
        "shell_metrics": shell_metrics,
        "scalars": scalar_maps,
    }


def summarize_variant(name, label, meta, pred, tissue, default_scalars=None):
    rows = []
    groups = [("brain", np.ones(tissue.shape[0], dtype=bool))]
    for t in ("WM", "GM", "CSF"):
        mask = tissue == t
        if np.any(mask):
            groups.append((t, mask))

    for group, mask in groups:
        row = {
            "variant": name,
            "label": label,
            "group": group,
            "generated_entries": meta["generated_entries"],
            "effective_entries": meta["effective_entries"],
            "heldout_rmse_median": finite_stats(pred["rmse"][mask])["median"],
            "heldout_nrmse_median": finite_stats(pred["nrmse"][mask])["median"],
            "heldout_mae_median": finite_stats(pred["mae"][mask])["median"],
            "heldout_bias_median": finite_stats(pred["bias"][mask])["median"],
            "heldout_abs_bias_median": finite_stats(np.abs(pred["bias"][mask]))["median"],
            "heldout_residual_mad_median": finite_stats(pred["residual_mad"][mask])["median"],
            "heldout_cosine_median": finite_stats(pred["cosine"][mask])["median"],
            "heldout_r2_median": finite_stats(pred["r2"][mask])["median"],
            "posterior_entropy_median": finite_stats(pred["entropy"][mask])["median"],
            "effective_k_median": finite_stats(pred["eff_k"][mask])["median"],
        }
        for shell, shell_data in pred["shell_metrics"].items():
            row[f"b{shell}_nrmse_median"] = finite_stats(shell_data["nrmse"][mask])["median"]
            row[f"b{shell}_mae_median"] = finite_stats(shell_data["mae"][mask])["median"]
            row[f"b{shell}_bias_median"] = finite_stats(shell_data["bias"][mask])["median"]
            row[f"b{shell}_cosine_median"] = finite_stats(shell_data["cosine"][mask])["median"]
        if default_scalars is not None:
            for param, vals in pred["scalars"].items():
                if param not in default_scalars:
                    continue
                ref = default_scalars[param]
                prior_range = np.nanmax(ref) - np.nanmin(ref)
                prior_range = prior_range if prior_range > 0 else 1.0
                delta = np.abs(vals - ref) / prior_range
                row[f"{param}_abs_delta_vs_default_median_norm"] = finite_stats(delta[mask])["median"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(out_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brain_rows = [r for r in rows if r["group"] == "brain"]
    labels = [r["label"] for r in brain_rows]
    x = np.arange(len(brain_rows))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    metrics = [
        ("heldout_nrmse_median", "Held-out NRMSE"),
        ("heldout_mae_median", "Held-out MAE"),
        ("heldout_r2_median", "Held-out R2"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in brain_rows]
        ax.bar(x, vals, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "dictionary_adequacy_predictive_metrics.png", dpi=250)
    plt.close(fig)

    delta_keys = [
        "fa_abs_delta_vs_default_median_norm",
        "md_abs_delta_vs_default_median_norm",
        "nd_abs_delta_vs_default_median_norm",
        "dispersion_abs_delta_vs_default_median_norm",
        "csf_fraction_abs_delta_vs_default_median_norm",
        "num_fibers_abs_delta_vs_default_median_norm",
    ]
    available = [k for k in delta_keys if any(k in r for r in brain_rows)]
    if available:
        fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
        width = 0.8 / len(available)
        for i, key in enumerate(available):
            vals = [r.get(key, np.nan) for r in brain_rows]
            offset = (i - (len(available) - 1) / 2) * width
            label = key.replace("_abs_delta_vs_default_median_norm", "").upper()
            ax.bar(x + offset, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Median |delta| vs default / observed default range")
        ax.set_title("Posterior Scalar Sensitivity to Dictionary Variant")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.savefig(out_dir / "dictionary_adequacy_scalar_sensitivity.png", dpi=250)
        plt.close(fig)

    shell_keys = sorted(
        {k for r in brain_rows for k in r if k.startswith("b") and k.endswith("_nrmse_median")}
    )
    if shell_keys:
        fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
        width = 0.8 / len(shell_keys)
        for i, key in enumerate(shell_keys):
            vals = [r.get(key, np.nan) for r in brain_rows]
            offset = (i - (len(shell_keys) - 1) / 2) * width
            label = key.replace("_nrmse_median", "").upper()
            ax.bar(x + offset, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Median held-out NRMSE")
        ax.set_title("Held-out Prediction by b-value Shell")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=min(3, len(shell_keys)), fontsize=8)
        fig.savefig(out_dir / "dictionary_adequacy_shell_nrmse.png", dpi=250)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Exp 8: FORCE dictionary adequacy")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-simulations", type=int, default=120000)
    parser.add_argument("--num-cpus", type=int, default=-1)
    parser.add_argument("--max-voxels", type=int, default=20000)
    parser.add_argument("--z-slice", type=int, default=None, help="Optional axial slice for faster runs")
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("csf_mask_path", "/path/to/subject/csf_mask.nii.gz")
    cfg.setdefault("gm_mask_path", "/path/to/subject/gm_mask.nii.gz")

    out_dir = Path(cfg["output_root"]) / "exp8_dictionary_adequacy"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "exp8.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    import nibabel as nib
    from dipy.io import read_bvals_bvecs

    log.info("Loading DWI")
    img = nib.load(cfg["dwi"])
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    bvals, _ = read_bvals_bvecs(cfg["bvals"], cfg["bvecs"])
    coords, tissue = select_eval_voxels(
        cfg,
        data.shape,
        args.max_voxels,
        args.seed,
        z_slice=args.z_slice,
    )
    signals = data[coords[:, 0], coords[:, 1], coords[:, 2], :]
    signals_norm = b0_normalize(signals, bvals, cfg["b0_threshold"]).astype(np.float32)
    train_idx, test_idx = make_holdout_split(
        bvals,
        cfg["b0_threshold"],
        args.holdout_fraction,
        args.seed,
    )
    del data

    run_meta = {
        "config": args.config,
        "num_simulations": args.num_simulations,
        "max_voxels": args.max_voxels,
        "n_eval_voxels": int(coords.shape[0]),
        "z_slice": args.z_slice,
        "holdout_fraction": args.holdout_fraction,
        "n_train_volumes": int(np.sum(train_idx)),
        "n_test_volumes": int(np.sum(test_idx)),
        "seed": args.seed,
        "variants": as_jsonable(VARIANTS),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    all_rows = []
    default_scalars = None
    default_simulations = None
    variant_meta = {}
    compact_outputs = {}
    for name, spec in VARIANTS.items():
        log.info("=== Variant: %s ===", name)
        simulations, meta = load_or_generate_variant(
            name,
            spec,
            cfg,
            out_dir,
            args.force,
            args.num_simulations,
            args.num_cpus,
        )
        variant_meta[name] = meta
        if name == "default":
            default_simulations = simulations
        pred = posterior_predict_variant(
            simulations,
            signals_norm,
            bvals,
            cfg["b0_threshold"],
            train_idx,
            test_idx,
            cfg["penalty"],
            cfg["posterior_beta"],
            cfg["n_neighbors"],
        )
        if name == "default":
            default_scalars = pred["scalars"]
        rows = summarize_variant(
            name,
            spec["label"],
            meta,
            pred,
            tissue,
            default_scalars=default_scalars,
        )
        all_rows.extend(rows)
        compact_outputs[name] = {
            "rmse": pred["rmse"].astype(np.float32),
            "nrmse": pred["nrmse"].astype(np.float32),
            "mae": pred["mae"].astype(np.float32),
            "bias": pred["bias"].astype(np.float32),
            "cosine": pred["cosine"].astype(np.float32),
            "r2": pred["r2"].astype(np.float32),
            "eff_k": pred["eff_k"].astype(np.float32),
            "scalars": {k: v.astype(np.float32) for k, v in pred["scalars"].items()},
        }
        log.info(
            "%s: median held-out NRMSE %.4f, cosine %.4f, effK %.2f",
            name,
            rows[0]["heldout_nrmse_median"],
            rows[0]["heldout_cosine_median"],
            rows[0]["effective_k_median"],
        )

    if default_simulations is not None:
        rng = np.random.default_rng(args.seed + 1000)
        for source_name in ("limited_csf", "no_crossing"):
            n_match = int(variant_meta[source_name]["effective_entries"])
            if n_match >= default_simulations["signals"].shape[0]:
                continue
            name = f"default_size_matched_to_{source_name}"
            label = f"Default size-matched to {VARIANTS[source_name]['label']}"
            log.info("=== Variant: %s ===", name)
            indices = rng.choice(default_simulations["signals"].shape[0], size=n_match, replace=False)
            simulations = subset_simulations(default_simulations, indices)
            pred = posterior_predict_variant(
                simulations,
                signals_norm,
                bvals,
                cfg["b0_threshold"],
                train_idx,
                test_idx,
                cfg["penalty"],
                cfg["posterior_beta"],
                cfg["n_neighbors"],
            )
            meta = {
                "generated_entries": int(default_simulations["signals"].shape[0]),
                "effective_entries": n_match,
            }
            rows = summarize_variant(
                name,
                label,
                meta,
                pred,
                tissue,
                default_scalars=default_scalars,
            )
            all_rows.extend(rows)
            compact_outputs[name] = {
                "rmse": pred["rmse"].astype(np.float32),
                "nrmse": pred["nrmse"].astype(np.float32),
                "cosine": pred["cosine"].astype(np.float32),
                "eff_k": pred["eff_k"].astype(np.float32),
                "scalars": {k: v.astype(np.float32) for k, v in pred["scalars"].items()},
            }
            log.info(
                "%s: median held-out NRMSE %.4f, cosine %.4f, effK %.2f",
                name,
                rows[0]["heldout_nrmse_median"],
                rows[0]["heldout_cosine_median"],
                rows[0]["effective_k_median"],
            )

    write_csv(out_dir / "dictionary_adequacy_summary.csv", all_rows)
    np.savez_compressed(
        out_dir / "dictionary_adequacy_voxel_metrics.npz",
        coords=coords,
        tissue=tissue,
        train_idx=train_idx,
        test_idx=test_idx,
        **{f"{name}_{key}": value for name, d in compact_outputs.items() for key, value in d.items() if key != "scalars"},
    )
    make_figures(out_dir, all_rows)

    with open(out_dir / "paper_summary.md", "w") as f:
        f.write("# Dictionary Adequacy Summary\n\n")
        f.write(
            "FORCE was evaluated as a dictionary-conditioned posterior by testing "
            "whether each dictionary variant could predict held-out diffusion "
            "measurements. Deliberately restricted dictionaries serve as model-"
            "misspecification controls.\n\n"
        )
        for row in [r for r in all_rows if r["group"] == "brain"]:
            f.write(
                f"- **{row['label']}**: median held-out NRMSE "
                f"{row['heldout_nrmse_median']:.4f}, cosine "
                f"{row['heldout_cosine_median']:.4f}, R2 "
                f"{row['heldout_r2_median']:.3f}, MAE "
                f"{row['heldout_mae_median']:.4f}, effective K "
                f"{row['effective_k_median']:.1f}, effective library entries "
                f"{row['effective_entries']:,}.\n"
            )

    print("\nExp 8 — Dictionary Adequacy")
    print(f"Saved outputs to {out_dir}")
    for row in [r for r in all_rows if r["group"] == "brain"]:
        print(
            f"{row['label']:<22} NRMSE={row['heldout_nrmse_median']:.4f} "
            f"cos={row['heldout_cosine_median']:.4f} effK={row['effective_k_median']:.1f}"
        )


if __name__ == "__main__":
    main()
