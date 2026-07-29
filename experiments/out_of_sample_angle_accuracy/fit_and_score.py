import os
import sys
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, force_peaks

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
SMOKE = os.environ.get("SMOKE") == "1"

def main(min_angle):
    d = np.load(f"{HERE}/data/crossings.npz")
    signals = d["signals"].astype(np.float32)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if SMOKE:
        signals = signals[::250]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    N = signals.shape[0]
    data4d = signals.reshape(N, 1, 1, -1)
    mask = np.ones((N, 1, 1), dtype=bool)

    model = FORCEModel(gtab, n_neighbors=50, use_posterior=True,
                       posterior_beta=2000.0, compute_odf=True, verbose=True)
    print(f"[min_angle={min_angle}] generating library ...", flush=True)
    model.generate(num_simulations=2000 if SMOKE else 500000, use_cache=True,
                   num_cpus=-1, two_fiber_min_angle=float(min_angle),
                   compute_dti=True, compute_dki=False)
    print(f"[min_angle={min_angle}] fitting {N} voxels ...", flush=True)
    fit = model.fit(data4d, mask=mask, engine="serial")

    pk = force_peaks(fit, mask=mask)
    peak_dirs = np.asarray(pk.peak_dirs).reshape(N, 5, 3).astype(np.float32)
    peak_vals = np.asarray(pk.peak_values).reshape(N, 5).astype(np.float32)

    odf = np.asarray(fit.odf).reshape(N, -1).astype(np.float32)
    nf = np.asarray(fit.num_fibers).reshape(N).astype(np.float32)

    tag = "smoke_" if SMOKE else ""
    np.savez(f"{HERE}/out/{tag}fit_min{int(min_angle)}.npz",
             peak_dirs=peak_dirs, peak_vals=peak_vals, odf=odf, num_fibers=nf)
    npk = (np.linalg.norm(peak_dirs, axis=2) > 0).sum(1)
    print(f"  native peaks/voxel: mean {npk.mean():.2f}  "
          f"(1:{(npk==1).mean():.2f} 2:{(npk==2).mean():.2f} 3:{(npk>=3).mean():.2f})")
    print(f"  saved out/{tag}fit_min{int(min_angle)}.npz")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
