# Synthetic angular-resolution and fiber-count experiment

Paper Section 2.4. Two-fiber crossings binned by crossing angle (10°–90°) and
one/two/three-fiber voxels are simulated with intra (stick) + extra (zeppelin)
compartments, Bingham dispersion and GM/CSF partial volume, then corrupted with
Rician noise at SNR 10/20/50. Each method is scored on how many true peaks it
recovers within 20° and on how often it gets the fiber count exactly right.

Everything runs on DIPY — no FAISS, no Ray, no Cython extension to build.

## Run

```bash
python run_angle_accuracy.py --out-dir results --num-cpus -1
python plot_results.py --results results --out-dir results
```

That writes `angle_nufo_results.json`, `angle_accuracy.png`,
`nufo_accuracy.png` and the matching CSV tables. A quick smoke run:

```bash
python run_angle_accuracy.py --out-dir smoke --library-size 100000 \
    --n-per-bucket 100 --n-per-class 100 --snr 50 --penalties 1e-4
```

## Files

| File | Role |
| --- | --- |
| `run_angle_accuracy.py` | builds the library, simulates the test sets, runs every method, writes metrics |
| `simulate_testset.py` | ground-truth voxel simulation (crossing-angle bins and fiber-count classes) |
| `metrics.py` | Hungarian one-to-one peak matching and the per-group metrics |
| `plot_results.py` | renders both figures and the CSV tables |
| `odffp_reference.json` | ODF-fingerprinting curves reused from the original run (see below) |

## Where FORCE's peaks come from

FORCE peaks are read from `dipy.reconst.force.force_peaks`, which returns the
fiber directions **of the matched simulation** (`FORCEFit.label`). The matched
simulation is a known configuration, so its orientations are exact. Running a
peak finder over the posterior ODF instead re-introduces the angular blurring
that forward modeling avoids, and it is not what the model reports. Pass
`--peaks posterior_odf` to reproduce that alternative for comparison.

## Library crossing-angle limit

DIPY's simulator resamples any two-fiber configuration closer than 30° and any
three-fiber configuration with a pair closer than 60°. A library built that way
contains no shallow crossing, so FORCE can never report one and the 10–30° bins
are unreachable by construction.

This experiment therefore passes `--two-fiber-min-angle 0`, which needs the
`two_fiber_min_angle` / `three_fiber_min_angle` arguments on
`FORCEModel.generate` (DIPY ≥ 1.13). The limits are part of the simulation cache
key, so a relaxed library never collides with a default one.

## Baselines

CSA, CSD and GQI are run through `dipy.direction.peaks_from_model` with
`relative_peak_threshold=0.8` and `min_separation_angle=10`, matching the
original experiment. The CSD response function is estimated from the highest-FA
voxels of the test set itself.

ODF fingerprinting is not part of DIPY, so its curve is **not** recomputed here:
`odffp_reference.json` holds the per-bucket peak-recall values from the original
paper run, measured with the same 20° tolerance and the same 1000 voxels per
bucket. Pass `--odffp-reference ''` to `plot_results.py` to drop that curve.

## Metric

`peak_detection_rate` is recall under an optimal one-to-one assignment: matched
true peaks / total true peaks, with a 20° tolerance and antipodal symmetry. A
method that reports a single peak in a two-fiber voxel therefore scores 50%, and
extra spurious peaks cannot raise the score (they lower `peak_precision`, which
is reported alongside). `fiber_count_accuracy` is the fraction of voxels whose
number of detected peaks equals the true count.
