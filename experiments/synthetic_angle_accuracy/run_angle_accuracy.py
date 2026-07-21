#!/usr/bin/env python
"""Synthetic angular-resolution and fiber-count (NUFO) experiment (paper Sec. 2.4).

Two benchmarks, both run entirely through DIPY:

* angular resolution - two-fiber crossings binned by crossing angle, scored as
  the fraction of true peaks recovered within 20 degrees.
* fiber count (NUFO) - one-, two- and three-fiber voxels, scored as the
  fraction whose number of detected peaks equals the true count.

FORCE peaks come from `force_peaks`, i.e. the fiber directions of the matched
simulation (`FORCEFit.label`), not from peak-finding on the posterior ODF. The
matched simulation is a known configuration, so its orientations are exact.
Pass `--peaks posterior_odf` to reproduce the ODF alternative.

Baselines are CSA, CSD and GQI. The library is built with
`two_fiber_min_angle=0` so crossings below DIPY's default 30 degree limit are
representable; that needs DIPY >= 1.13.

    python run_angle_accuracy.py --out-dir results --num-cpus -1
"""

import argparse
import json
import os
import time

import numpy as np

from dipy.data import default_sphere
from dipy.direction import peak_directions, peaks_from_model
from dipy.reconst import dti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.shm import CsaOdfModel
from dipy.sims.voxel import add_noise

from metrics import evaluate, valid_peak_dirs
from simulate_testset import (
    BUCKETS,
    VoxelSimulator,
    build_gtab,
    make_angular_testset,
    make_nufo_testset,
)

RELATIVE_PEAK_THRESHOLD = 0.8
MIN_SEPARATION_ANGLE = 10.0
ANGULAR_THRESHOLD = 20.0


def format_penalty(alpha):
    """Render a penalty as a power of ten, e.g. 1e-4."""
    if alpha == 0:
        return "0"
    mantissa, exponent = f"{alpha:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def add_rician_noise(signals, snr, seed):
    """Rician noise at a given SNR; `snr=None` returns the clean signals."""
    if snr is None:
        return signals.astype(np.float32)
    np.random.seed(seed)  # dipy.sims.voxel.add_noise draws from the global RNG
    return add_noise(signals, snr, S0=100.0, noise_type="rician").astype(np.float32)


def estimate_response(gtab, signals, fa_threshold=0.8):
    """Single-fiber CSD response from the highest-FA voxels of the test set."""
    fit = dti.TensorModel(gtab).fit(np.asarray(signals))
    fa = fit.fa
    selected = np.flatnonzero(fa > fa_threshold)
    if len(selected) < 10:
        selected = np.argsort(fa)[-max(10, len(fa) // 100):]
        print(
            f"  only {int((fa > fa_threshold).sum())} voxels with FA > {fa_threshold}; "
            f"using the top {len(selected)} instead",
            flush=True,
        )
    evals = np.mean(fit.evals[selected], axis=0)
    return np.asarray(evals, dtype=float), float(1 / evals.sum())


def force_predictions(model, signals, peaks_source, sphere):
    """Peak directions and fiber counts from a FORCE fit."""
    fit = model.fit(signals)

    if peaks_source == "simulation":
        peaks = force_peaks(fit)
        return valid_peak_dirs(peaks.peak_dirs)

    odfs = np.asarray(fit.odf, dtype=np.float64)
    dirs_list, counts = [], np.zeros(len(odfs), dtype=np.int32)
    for i, odf in enumerate(odfs):
        dirs, _, _ = peak_directions(
            odf,
            sphere,
            relative_peak_threshold=RELATIVE_PEAK_THRESHOLD,
            min_separation_angle=MIN_SEPARATION_ANGLE,
        )
        dirs_list.append(np.asarray(dirs, dtype=np.float32))
        counts[i] = len(dirs)
    return dirs_list, counts


def baseline_predictions(model, signals, sphere, num_cpus):
    peaks = peaks_from_model(
        model,
        signals,
        sphere,
        relative_peak_threshold=RELATIVE_PEAK_THRESHOLD,
        min_separation_angle=MIN_SEPARATION_ANGLE,
        mask=None,
        parallel=num_cpus != 1,
        num_processes=None if num_cpus in (-1, 0) else num_cpus,
    )
    return valid_peak_dirs(peaks.peak_dirs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--library-size", type=int, default=1_000_000,
                        help="number of FORCE simulations in the matching library")
    parser.add_argument("--n-per-bucket", type=int, default=1000,
                        help="two-fiber test voxels per crossing-angle bucket")
    parser.add_argument("--n-per-class", type=int, default=10_000,
                        help="test voxels per fiber-count class (1, 2 and 3 fibers)")
    parser.add_argument("--snr", type=float, nargs="+", default=[10, 20, 50])
    parser.add_argument("--penalties", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5],
                        help="FORCE fiber-complexity penalties (alpha) to sweep")
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--two-fiber-min-angle", type=float, default=0.0,
                        help="minimum crossing angle in the library, in degrees. DIPY's "
                             "default of 30 leaves the 10-30 degree bins unreachable, so "
                             "this experiment relaxes it to 0")
    parser.add_argument("--three-fiber-min-angle", type=float, default=60.0)
    parser.add_argument("--peaks", choices=["simulation", "posterior_odf"],
                        default="simulation",
                        help="where FORCE peaks come from (see module docstring)")
    parser.add_argument("--baselines", nargs="*", default=["CSA", "CSD", "GQI"],
                        choices=["CSA", "CSD", "GQI"])
    parser.add_argument("--num-cpus", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true",
                        help="regenerate the FORCE library instead of reusing the DIPY cache")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    snr_levels = [None if s <= 0 else float(s) for s in args.snr]
    t_start = time.time()

    print("[1/6] gradient table and simulator", flush=True)
    gtab = build_gtab(seed=args.seed)
    sim = VoxelSimulator(gtab)
    sphere = sim.sphere
    target_dirs = sim.target_sphere
    print(f"  {len(gtab.bvals)} volumes, {sim.n_dirs} sphere directions", flush=True)

    print(f"[2/6] FORCE library ({args.library_size:,} simulations)", flush=True)
    t0 = time.time()
    library = FORCEModel(gtab, n_neighbors=args.n_neighbors)
    library.generate(
        num_simulations=args.library_size,
        num_cpus=args.num_cpus,
        two_fiber_min_angle=args.two_fiber_min_angle,
        three_fiber_min_angle=args.three_fiber_min_angle,
        use_cache=not args.no_cache,
        verbose=True,
    )
    simulations = library.simulations
    print(f"  done in {time.time() - t0:.0f} s", flush=True)

    print("[3/6] ground-truth test sets", flush=True)
    t0 = time.time()
    angular = make_angular_testset(sim, args.n_per_bucket, seed=args.seed + 1)
    nufo = make_nufo_testset(sim, args.n_per_class, seed=args.seed + 2)
    print(f"  {len(angular['signals'])} angular + {len(nufo['signals'])} NUFO voxels "
          f"in {time.time() - t0:.0f} s", flush=True)

    bucket_indices = {
        f"({lo}, {hi})": np.flatnonzero(angular["bucket_index"] == b)
        for b, (lo, hi) in enumerate(BUCKETS)
    }
    class_indices = {
        f"N={n}": np.flatnonzero(nufo["num_fibers"] == n) for n in (1, 2, 3)
    }

    print("[4/6] CSD response function", flush=True)
    response = estimate_response(gtab, angular["signals"])
    print(f"  evals {response[0]}, ratio {response[1]:.1f}", flush=True)

    models = {}
    if "CSA" in args.baselines:
        models["CSA"] = CsaOdfModel(gtab, sh_order_max=8)
    if "CSD" in args.baselines:
        models["CSD"] = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    if "GQI" in args.baselines:
        models["GQI"] = GeneralizedQSamplingModel(gtab, sampling_length=1.25)

    force_models = {}
    for alpha in args.penalties:
        force_models[f"FORCE (α={format_penalty(alpha)})"] = FORCEModel(
            gtab,
            simulations=simulations,
            penalty=alpha,
            n_neighbors=args.n_neighbors,
            use_posterior=args.peaks == "posterior_odf",
        )

    results = {"angular": {}, "nufo": {}}

    for stage, testset, groups in (
        ("angular", angular, bucket_indices),
        ("nufo", nufo, class_indices),
    ):
        step = 5 if stage == "angular" else 6
        print(f"[{step}/6] {stage} benchmark", flush=True)
        for snr in snr_levels:
            key = "clean" if snr is None else f"{snr:g}"
            noisy = add_rician_noise(testset["signals"], snr, seed=args.seed + 7)
            print(f"  SNR {key}", flush=True)

            predictions = {}
            for name, model in force_models.items():
                t0 = time.time()
                predictions[name] = force_predictions(model, noisy, args.peaks, sphere)
                print(f"    {name}: {time.time() - t0:.0f} s", flush=True)
            for name, model in models.items():
                t0 = time.time()
                predictions[name] = baseline_predictions(
                    model, noisy, sphere, args.num_cpus
                )
                print(f"    {name}: {time.time() - t0:.0f} s", flush=True)

            for name, (dirs_list, counts) in predictions.items():
                bucket = results[stage].setdefault(name, {}).setdefault(key, {})
                for group, indices in groups.items():
                    bucket[group] = evaluate(
                        indices,
                        dirs_list,
                        counts,
                        testset["labels"],
                        testset["num_fibers"],
                        target_dirs,
                        threshold_deg=ANGULAR_THRESHOLD,
                    )

    payload = {
        "config": {
            "library_size": args.library_size,
            "n_per_bucket": args.n_per_bucket,
            "n_per_class": args.n_per_class,
            "snr": [None if s is None else float(s) for s in snr_levels],
            "penalties": args.penalties,
            "n_neighbors": args.n_neighbors,
            "two_fiber_min_angle": args.two_fiber_min_angle,
            "three_fiber_min_angle": args.three_fiber_min_angle,
            "peaks": args.peaks,
            "seed": args.seed,
            "n_volumes": int(len(gtab.bvals)),
            "n_sphere_dirs": int(sim.n_dirs),
            "angular_threshold_deg": ANGULAR_THRESHOLD,
            "relative_peak_threshold": RELATIVE_PEAK_THRESHOLD,
            "min_separation_angle": MIN_SEPARATION_ANGLE,
            "buckets": [list(b) for b in BUCKETS],
            "runtime_s": round(time.time() - t_start, 1),
        },
        "results": results,
    }
    out = os.path.join(args.out_dir, "angle_nufo_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out} ({time.time() - t_start:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
