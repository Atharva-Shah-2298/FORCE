"""AMICO-NODDI on the biological phantom (base env, amico 2.1.0).
Outputs within-WM NDI (v_ic), Watson ODI, free-water ISOVF per SNR."""
import os
import shutil
import numpy as np
import nibabel as nib
import amico

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
OUT = os.path.join(HERE, "bio_out")
SNRS = ["clean", 50, 20, 10]


def fit(snr, work):
    subj = os.path.join(work, f"snr{snr}")
    os.makedirs(subj, exist_ok=True)
    shutil.copy(os.path.join(DATA, f"dwi_{snr}.nii.gz"), os.path.join(subj, "dwi.nii.gz"))
    shutil.copy(os.path.join(DATA, "mask.nii.gz"), os.path.join(subj, "mask.nii.gz"))
    shutil.copy(os.path.join(DATA, "dwi.scheme"), os.path.join(subj, "dwi.scheme"))
    amico.setup()
    ae = amico.Evaluation(work, f"snr{snr}")
    ae.load_data("dwi.nii.gz", "dwi.scheme", mask_filename="mask.nii.gz", b0_thr=50)
    ae.set_model("NODDI")
    ae.generate_kernels(regenerate=False)
    ae.load_kernels()
    ae.fit()
    ae.save_results(path_suffix="results")
    rdir = None
    for r, _, files in os.walk(subj):
        if any(f.startswith("fit_") and f.endswith(".nii.gz") for f in files):
            rdir = r; break

    def _l(n):
        p = os.path.join(rdir, n)
        return nib.load(p).get_fdata().squeeze().astype(np.float32).flatten() if os.path.exists(p) else None
    return _l("fit_NDI.nii.gz"), _l("fit_ODI.nii.gz"), _l("fit_FWF.nii.gz")


def main():
    os.makedirs(OUT, exist_ok=True)
    work = os.path.join(OUT, "amico_work")
    os.makedirs(work, exist_ok=True)
    for snr in SNRS:
        print(f"=== AMICO-NODDI SNR={snr} ===")
        ndi, odi, fwf = fit(snr, work)
        np.savez(os.path.join(OUT, f"amico_{snr}.npz"), ndi=ndi, odi=odi, fw=fwf)
        print(f"  NDI {np.nanmean(ndi):.3f} ODI {np.nanmean(odi):.3f} FW {np.nanmean(fwf):.3f}")


if __name__ == "__main__":
    main()
