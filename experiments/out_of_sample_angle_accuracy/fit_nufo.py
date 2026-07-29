import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.gqi import GeneralizedQSamplingModel

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
SPH = get_sphere(name="repulsion724")
REL, MINSEP = 0.5, 15.0
ALPHAS = [1e-3, 1e-4, 1e-5]

def main():
    d = np.load(f"{HERE}/data/nufo.npz")
    sig = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    N = sig.shape[0]
    data4d = sig.reshape(N, 1, 1, -1); mask = np.ones((N, 1, 1), bool)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    keep2k = (bvals <= 50) | ((bvals >= 1500) & (bvals <= 2500))
    gtab2k = gradient_table(bvals[keep2k], bvecs=bvecs[keep2k], b0_threshold=50)
    data2k = data4d[..., keep2k]

    counts = {}

    for a in ALPHAS:
        m = FORCEModel(gtab, n_neighbors=50, penalty=a, use_posterior=True,
                       posterior_beta=2000.0, compute_odf=False, verbose=False)
        m.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   two_fiber_min_angle=5.0,
                   three_fiber_min_angle=60.0,
                   compute_dti=True, compute_dki=False)
        fit = m.fit(data4d, mask=mask, engine="serial")
        pk = force_peaks(fit, mask=mask)
        pdirs = np.asarray(pk.peak_dirs).reshape(N, -1, 3)
        pvals = np.asarray(pk.peak_values).reshape(N, -1)
        valid = np.linalg.norm(pdirs, axis=2) > 0
        pvals = np.where(valid, pvals, 0.0)

        mx = pvals.max(1, keepdims=True)
        cnt = ((pvals >= 0.5 * mx) & valid & (mx > 0)).sum(1)
        counts[f"FORCE_a{a:.0e}"] = cnt.astype(np.int8)
        print(f"FORCE alpha={a:.0e} done; mean peaks {cnt.mean():.2f}", flush=True)

    resp = (np.array([2.108e-3, 0.239e-3, 0.227e-3]), 1.0)
    inv = {
        "CSA": (CsaOdfModel(gtab2k, sh_order_max=8), data2k),
        "CSD": (ConstrainedSphericalDeconvModel(gtab2k, resp, sh_order_max=8), data2k),
        "GQI": (GeneralizedQSamplingModel(gtab, sampling_length=1.2), data4d),
    }
    for name, (model, dat) in inv.items():
        pam = peaks_from_model(model, dat, SPH, relative_peak_threshold=REL,
                               min_separation_angle=25.0, mask=mask, npeaks=3,
                               return_odf=False, parallel=False)
        pv = np.asarray(pam.peak_values).reshape(N, 3)
        counts[name] = (pv > 0).sum(1).astype(np.int8)
        print(f"{name} done; mean peaks {counts[name].mean():.2f}", flush=True)

    np.savez(f"{HERE}/out/nufo_counts_dev.npz", true_N=d["true_N"], snr=d["snr"], **counts)
    print("saved out/nufo_counts_dev.npz")

if __name__ == "__main__":
    main()
