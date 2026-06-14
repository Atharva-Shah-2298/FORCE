"""AMICO-NODDI baseline per SNR — base env (amico 2.1.0).

Uses HCP 165840's existing bvals.scheme file (matches our gtab).
Builds the NODDI kernel once, then fits each SNR.
"""
import os
import sys
import shutil
import numpy as np
import nibabel as nib

import amico

DATA = "data"
OUT = "baselines_out"
SCHEME = "/path/to/subject/bvals.scheme"
SNRS = [50, 20, 10]

os.makedirs(OUT, exist_ok=True)


def fit_one_snr(snr, work_dir):
    """Run AMICO-NODDI on dwi_snr{snr}.nii.gz."""
    # AMICO expects a standard subject-folder layout. Build it.
    subj_dir = os.path.join(work_dir, f"snr{snr}")
    os.makedirs(subj_dir, exist_ok=True)

    src_dwi = os.path.join(DATA, f"dwi_snr{snr}.nii.gz")
    src_mask = os.path.join(DATA, "mask.nii.gz")
    dst_dwi = os.path.join(subj_dir, "dwi.nii.gz")
    dst_mask = os.path.join(subj_dir, "mask.nii.gz")
    dst_scheme = os.path.join(subj_dir, "dwi.scheme")
    if not os.path.exists(dst_dwi):
        shutil.copy(src_dwi, dst_dwi)
    if not os.path.exists(dst_mask):
        shutil.copy(src_mask, dst_mask)
    if not os.path.exists(dst_scheme):
        shutil.copy(SCHEME, dst_scheme)

    amico.setup()
    ae = amico.Evaluation(work_dir, f"snr{snr}")
    ae.load_data("dwi.nii.gz", "dwi.scheme", mask_filename="mask.nii.gz",
                 b0_thr=50)
    ae.set_model("NODDI")
    # Default NODDI params; kernel resolution typical
    ae.generate_kernels(regenerate=False)
    ae.load_kernels()
    ae.fit()
    out_dir_amico = os.path.join(subj_dir, "AMICO", "NODDI")
    ae.save_results(path_suffix="results")
    # Standard outputs: fit_NDI.nii.gz, fit_ODI.nii.gz, fit_FWF.nii.gz
    # Locate them — AMICO 2.x writes under subject_dir/AMICO/<model>/results/
    results_dir = None
    for r, dirs, files in os.walk(subj_dir):
        if any(f.startswith("fit_") and f.endswith(".nii.gz") for f in files):
            results_dir = r
            break
    if results_dir is None:
        # try newer scheme
        results_dir = os.path.join(subj_dir, "AMICO", "NODDI", "results")

    def _load(name):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            return nib.load(p).get_fdata().squeeze().astype(np.float32)
        return None

    ndi = _load("fit_NDI.nii.gz")
    odi = _load("fit_ODI.nii.gz")
    fwf = _load("fit_FWF.nii.gz")
    if ndi is None or odi is None or fwf is None:
        # try alternative names
        for n in os.listdir(results_dir):
            print("  amico out:", n)
        raise RuntimeError("AMICO NODDI outputs not found")
    return ndi.flatten(), odi.flatten(), fwf.flatten(), results_dir


def main():
    work_dir = os.path.join(OUT, "amico_work")
    os.makedirs(work_dir, exist_ok=True)
    results = {}
    for snr in SNRS:
        print(f"\n=== AMICO-NODDI SNR={snr} ===")
        ndi, odi, fwf, rdir = fit_one_snr(snr, work_dir)
        print(f"  results: {rdir}")
        print(f"  NDI mean {np.nanmean(ndi):.3f}  ODI mean {np.nanmean(odi):.3f}  "
              f"FWF mean {np.nanmean(fwf):.3f}")
        np.savez(os.path.join(OUT, f"amico_snr{snr}.npz"),
                 ndi=ndi, odi=odi, fwf=fwf)
        results[snr] = (ndi, odi, fwf)
    print("AMICO done.")


if __name__ == "__main__":
    main()
