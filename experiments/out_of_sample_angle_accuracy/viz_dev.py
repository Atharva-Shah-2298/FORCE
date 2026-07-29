import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.force import FORCEModel, force_peaks

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
common = get_sphere(name="symmetric362")
CV, CF = common.vertices, common.faces

def resample(odf_src, src_verts, dst=CV):
    sim = np.abs(dst @ src_verts.T)
    return odf_src[np.argmax(sim, axis=1)]

def main():
    d = np.load(f"{HERE}/data/crossings.npz")
    sig = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    a1, a2, snr, ta = d["axis1"], d["axis2"], d["snr"], d["true_angle"]

    pool = np.where((snr == 50) & (ta < 40))[0]
    sel = np.random.default_rng(7).choice(pool, 10, replace=False)
    sel = sel[np.argsort(ta[sel])]
    np.save(f"{HERE}/out/viz_sel.npy", sel)
    print("selected voxels (angle):", list(zip(sel.tolist(), ta[sel].astype(int).tolist())))

    s10 = sig[sel]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    keep2k = (bvals <= 50) | ((bvals >= 1500) & (bvals <= 2500))
    g2k = gradient_table(bvals[keep2k], bvecs=bvecs[keep2k], b0_threshold=50)
    resp = (np.array([2.108e-3, 0.239e-3, 0.227e-3]), 1.0)

    odfs, peaks = {}, {}

    odfs["CSA"] = CsaOdfModel(g2k, sh_order_max=8).fit(s10[:, keep2k]).odf(common)
    odfs["CSD"] = ConstrainedSphericalDeconvModel(g2k, resp, sh_order_max=8).fit(s10[:, keep2k]).odf(common)
    odfs["GQI"] = GeneralizedQSamplingModel(gtab, sampling_length=1.2).fit(s10).odf(common)

    fm = FORCEModel(gtab, n_neighbors=50, penalty=1e-5, use_posterior=True,
                    posterior_beta=2000.0, compute_odf=True)
    fm.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                two_fiber_min_angle=5.0, compute_dti=True, compute_dki=False)
    from dipy.sims.force import default_sphere as fsph
    ffit = fm.fit(s10.reshape(10, 1, 1, -1), mask=np.ones((10, 1, 1), bool), engine="serial")
    force_odf_src = np.asarray(ffit.odf).reshape(10, -1)
    odfs["FORCE"] = np.array([resample(force_odf_src[i], fsph.vertices) for i in range(10)])
    fpk = force_peaks(ffit, mask=np.ones((10, 1, 1), bool))
    peaks["FORCE"] = np.asarray(fpk.peak_dirs).reshape(10, -1, 3)

    for m in ("CSA", "CSD", "GQI"):
        peaks[m] = np.load(f"{HERE}/out/peaks_{m}.npz")["peak_dirs"][sel]

    np.savez(f"{HERE}/out/viz_dev.npz",
             verts=CV.astype(np.float32), faces=CF.astype(np.int32),
             a1=a1[sel].astype(np.float32), a2=a2[sel].astype(np.float32),
             true_angle=ta[sel].astype(np.float32),
             **{f"odf_{k}": v.astype(np.float32) for k, v in odfs.items()},
             **{f"pk_{k}": v.astype(np.float32) for k, v in peaks.items()})
    print("saved out/viz_dev.npz  odf keys:", list(odfs.keys()))

if __name__ == "__main__":
    main()
