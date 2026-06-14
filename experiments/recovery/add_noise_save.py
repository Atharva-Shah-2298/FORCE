"""Add Rician noise to clean signals at SNR={50,20,10} and save as 4D NIfTI."""
import os
import numpy as np
import nibabel as nib
from numpy.random import default_rng

DATA = "data"
SNRS = [50, 20, 10]
SEED = 20260601


def main():
    d = np.load(os.path.join(DATA, "signals_clean.npz"))
    sig = d["signals"]  # (N, T), S0 = 1
    bvals = d["bvals"]
    bvecs = d["bvecs"]
    N, T = sig.shape
    print(f"N={N}, T={T}, S0~1 (normalized)")
    rng = default_rng(SEED)

    # 4D NIfTI shape (N, 1, 1, T); intensity ~ S0 * sig. To keep AMICO and
    # standard fit tools happy, use S0=1000 and Rician noise sigma = S0/SNR.
    S0 = 1000.0
    affine = np.eye(4)

    # Build a binary mask (N,1,1)
    mask = np.ones((N, 1, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(DATA, "mask.nii.gz"))

    for snr in SNRS:
        sigma = S0 / snr
        # complex noise on real signal then magnitude
        re = S0 * sig + rng.normal(0.0, sigma, size=sig.shape)
        im = rng.normal(0.0, sigma, size=sig.shape)
        noisy = np.sqrt(re ** 2 + im ** 2).astype(np.float32)
        vol = noisy.reshape(N, 1, 1, T)
        out_p = os.path.join(DATA, f"dwi_snr{snr}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine), out_p)
        # report stats
        b0_idx = np.where(bvals <= 50)[0]
        print(f"SNR={snr}: saved {out_p}  shape={vol.shape}  "
              f"mean_b0={noisy[:, b0_idx].mean():.1f}  sigma={sigma:.1f}")

    # also save clean nifti for sanity
    clean = (S0 * sig).reshape(N, 1, 1, T).astype(np.float32)
    nib.save(nib.Nifti1Image(clean, affine), os.path.join(DATA, "dwi_clean.nii.gz"))
    print("clean saved")

    # Save bvals/bvecs in canonical FSL layout
    np.savetxt(os.path.join(DATA, "bvals"), bvals[None, :], fmt="%.6g")
    bv = bvecs if bvecs.shape[0] == 3 else bvecs.T
    np.savetxt(os.path.join(DATA, "bvecs"), bv, fmt="%.8f")
    print("bvals/bvecs saved")


if __name__ == "__main__":
    main()
