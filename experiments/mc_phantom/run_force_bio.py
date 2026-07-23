"""FORCE on the biological phantom at each SNR. Reads the same NIfTI files
AMICO uses, so both fit identical noisy data."""
import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
OUT = os.path.join(HERE, "bio_out")
SNRS = ["clean", 50, 20, 10]
K, BETA = 50, 2000.0


def main():
    os.makedirs(OUT, exist_ok=True)
    bvals = np.loadtxt(os.path.join(DATA, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(DATA, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=BETA,
                       compute_odf=False, verbose=True)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    for snr in SNRS:
        vol = nib.load(os.path.join(DATA, f"dwi_{snr}.nii.gz")).get_fdata()
        Q = vol.reshape(-1, vol.shape[-1]).astype(np.float64)
        qn = np.linalg.norm(Q, axis=1, keepdims=True); qn[qn == 0] = 1
        Dist, neigh = model._index.search(
            np.ascontiguousarray((Q / qn).astype(np.float32)), k=K)
        W = softmax_stable(BETA * (Dist - model._penalty_array[neigh]), axis=1).astype(np.float32)
        d = {f: np.einsum('nk,nk->n', W, sims[f][neigh]).astype(np.float32)
             for f in ("nd", "dispersion", "wm_fraction", "gm_fraction", "csf_fraction")}
        wmf = np.maximum(d["wm_fraction"], 1e-6)
        out = {
            "ndi": np.clip(d["nd"] / wmf, 0, 1),                       # within-WM NDI
            "odi": np.clip(1 - (1 - d["dispersion"]) / wmf, 0, 1),    # anisotropic ODI
            "fw": d["csf_fraction"],                                   # free water (cf AMICO ISOVF)
            "iso": d["gm_fraction"] + d["csf_fraction"],               # total isotropic
            "gm": d["gm_fraction"],
            "voxel_nd": d["nd"],
        }
        np.savez(os.path.join(OUT, f"force_{snr}.npz"), **out)
        print(f"  [{snr}] NDI {out['ndi'].mean():.3f} ODI {out['odi'].mean():.3f} "
              f"FW {out['fw'].mean():.3f}")


if __name__ == "__main__":
    main()
