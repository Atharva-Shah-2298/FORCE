import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, force_peaks

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"

def main():
    d = np.load(f"{HERE}/data/crossings.npz")
    sig = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    N = sig.shape[0]
    data4d = sig.reshape(N, 1, 1, -1); mask = np.ones((N, 1, 1), bool)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    for a in (1e-3, 1e-4):
        m = FORCEModel(gtab, n_neighbors=50, penalty=a, use_posterior=True,
                       posterior_beta=2000.0, compute_odf=False)
        m.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   two_fiber_min_angle=5.0, compute_dti=True, compute_dki=False)
        fit = m.fit(data4d, mask=mask, engine="serial")
        pk = force_peaks(fit, mask=mask)
        np.savez(f"{HERE}/out/peaks_FORCE_a{a:.0e}.npz",
                 peak_dirs=np.asarray(pk.peak_dirs).reshape(N, -1, 3).astype(np.float32),
                 peak_vals=np.asarray(pk.peak_values).reshape(N, -1).astype(np.float32))
        print(f"saved peaks_FORCE_a{a:.0e}.npz")

if __name__ == "__main__":
    main()
