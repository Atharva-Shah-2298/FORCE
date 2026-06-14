"""DTI (FA/MD/RD) and DKI (MK/AK/RK/KFA) baselines per SNR — skyline env."""
import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.dki import DiffusionKurtosisModel

DATA = "data"
OUT = "baselines_out"
SNRS = [50, 20, 10]
os.makedirs(OUT, exist_ok=True)


def main():
    bvals = np.loadtxt(os.path.join(DATA, "bvals")).astype(np.float64).flatten()
    bvecs = np.loadtxt(os.path.join(DATA, "bvecs")).astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    # DTI: b <= 1000 + b0
    dti_mask = (bvals <= 1050)
    gtab_dti = gradient_table(bvals[dti_mask], bvecs=bvecs[dti_mask], b0_threshold=50)
    dti_model = TensorModel(gtab_dti)

    # DKI: full multi-shell
    gtab_full = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    dki_model = DiffusionKurtosisModel(gtab_full)

    for snr in SNRS:
        vol = nib.load(os.path.join(DATA, f"dwi_snr{snr}.nii.gz")).get_fdata().astype(np.float32)
        Q = vol.reshape(-1, vol.shape[-1])
        print(f"\n=== SNR={snr} ({Q.shape[0]} voxels) ===")

        print("  DTI fitting ...")
        dti_fit = dti_model.fit(Q[:, dti_mask])
        fa = np.asarray(dti_fit.fa, dtype=np.float32)
        md = np.asarray(dti_fit.md, dtype=np.float32)
        rd = np.asarray(dti_fit.rd, dtype=np.float32)

        print("  DKI fitting ...")
        dki_fit = dki_model.fit(Q)
        mk = np.asarray(dki_fit.mk(), dtype=np.float32)
        ak = np.asarray(dki_fit.ak(), dtype=np.float32)
        rk = np.asarray(dki_fit.rk(), dtype=np.float32)
        kfa = np.asarray(dki_fit.kfa, dtype=np.float32)

        np.savez(os.path.join(OUT, f"dti_dki_snr{snr}.npz"),
                 fa=fa, md=md, rd=rd, mk=mk, ak=ak, rk=rk, kfa=kfa)
        print(f"  FA mean {fa.mean():.3f}  MD mean {md.mean()*1e3:.3f}e-3  "
              f"MK mean {np.nanmean(mk):.3f}")


if __name__ == "__main__":
    main()
