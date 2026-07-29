import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.gqi import GeneralizedQSamplingModel

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
TOL_MATCH = 20.0
REL, MINSEP = 0.5, 25.0
SPH = get_sphere(name="repulsion724")

def ang(u, v):
    return np.rad2deg(np.arccos(np.clip(abs(float(np.dot(u, v))), 0, 1)))

def score_pair(p1, p2, a1, a2):
    d_aa = ang(p1, a1) + ang(p2, a2)
    d_ab = ang(p1, a2) + ang(p2, a1)
    e1, e2 = (ang(p1, a1), ang(p2, a2)) if d_aa <= d_ab else (ang(p1, a2), ang(p2, a1))
    return 0.5 * (e1 + e2), (e1 <= TOL_MATCH and e2 <= TOL_MATCH), ang(p1, p2)

def score_peaks(peak_dirs, peak_vals, a1, a2):
    nz = np.linalg.norm(peak_dirs, axis=1) > 0
    dirs, vals = peak_dirs[nz], peak_vals[nz]
    if dirs.shape[0] < 2:
        return dirs.shape[0], np.nan, False, np.nan
    o = np.argsort(-vals)[:2]
    merr, resolved, cross = score_pair(dirs[o[0]], dirs[o[1]], a1, a2)
    return int(nz.sum()), merr, resolved, cross

def shell(gtab_bvals, bvecs, sig, keep):
    m = keep
    return gtab_bvals[m], bvecs[m], sig[:, m]

def main():
    d = np.load(f"{HERE}/data/crossings.npz")
    sig = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    a1, a2, snr, true_ang = d["axis1"], d["axis2"], d["snr"], d["true_angle"]
    N = sig.shape[0]
    data4d = sig.reshape(N, 1, 1, -1)
    mask = np.ones((N, 1, 1), bool)

    gtab_full = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    keep2k = (bvals <= 50) | ((bvals >= 1500) & (bvals <= 2500))
    gtab_2k = gradient_table(bvals[keep2k], bvecs=bvecs[keep2k], b0_threshold=50)
    data2k = data4d[..., keep2k]

    response = (np.array([2.108e-3, 0.239e-3, 0.227e-3]), 1.0)
    models = {
        "CSA": (CsaOdfModel(gtab_2k, sh_order_max=8), data2k),
        "CSD": (ConstrainedSphericalDeconvModel(gtab_2k, response, sh_order_max=8), data2k),
        "GQI": (GeneralizedQSamplingModel(gtab_full, sampling_length=1.2), data4d),
    }

    rows = []
    angles = sorted(set(true_ang.tolist())); snrs = sorted(set(snr.tolist()), reverse=True)
    for name, (model, dat) in models.items():
        print(f"[{name}] peaks_from_model ...", flush=True)
        pam = peaks_from_model(model, dat, SPH, relative_peak_threshold=REL,
                               min_separation_angle=MINSEP, mask=mask, npeaks=5,
                               return_odf=False, parallel=False)
        pdirs = np.asarray(pam.peak_dirs).reshape(N, 5, 3)
        pvals = np.asarray(pam.peak_values).reshape(N, 5)
        np.savez(f"{HERE}/out/peaks_{name}.npz",
                 peak_dirs=pdirs.astype(np.float32), peak_vals=pvals.astype(np.float32))
        res = np.array([score_peaks(pdirs[i], pvals[i], a1[i], a2[i]) for i in range(N)], object)
        npk = res[:, 0].astype(float); merr = res[:, 1].astype(float)
        resolved = res[:, 2].astype(bool); cross = res[:, 3].astype(float)
        for sv in snrs:
            for aa in angles:
                m = (snr == sv) & (true_ang == aa)
                rows.append(dict(method=name, snr=int(sv), true_angle=int(aa),
                                 n=int(m.sum()), resolved_rate=float(np.mean(resolved[m])),
                                 found2_rate=float(np.mean(npk[m] >= 2)),
                                 ang_err=float(np.nanmean(merr[m]))))
        r = pd.DataFrame(rows); r = r[r.method == name]
        print(f"  {name} resolved @SNR50: " +
              " ".join(f"{int(aa)}:{r[(r.snr==50)&(r.true_angle==int(aa))].resolved_rate.values[0]:.2f}"
                       for aa in angles))
    pd.DataFrame(rows).to_csv(f"{HERE}/out/angular_accuracy_inverse.csv", index=False)
    print("\nsaved out/angular_accuracy_inverse.csv")

if __name__ == "__main__":
    main()
