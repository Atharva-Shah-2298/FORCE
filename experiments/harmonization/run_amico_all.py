"""
Run AMICO (NODDI) on all subjects and sessions.

Must be run with the base conda environment (which has amico installed):
    C:\\Users\\athu2\\miniconda3\\python.exe run_amico_all.py

Expects dipy_auto pipeline to have already been run (needs dwi_patch2self.nii.gz
and brain_mask.nii.gz from preprocessing).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HARMONIZATION_ROOT", SCRIPT_DIR))
OUTPUT_ROOT = Path(os.environ.get("HARMONIZATION_OUTPUT", ROOT / "output"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
    "ses-c10r2", "ses-c10r3",
]


def bval_bvec_to_scheme(bval_path: Path, bvec_path: Path, b0_thr: float = 0.0) -> np.ndarray:
    """Build Nx4 AMICO scheme: gx, gy, gz, b (s/mm^2)."""
    bvals = np.atleast_1d(np.loadtxt(str(bval_path)))
    bvecs = np.loadtxt(str(bvec_path))
    if bvecs.shape[0] == 3:
        g = bvecs.T.copy()
    else:
        g = bvecs.copy()
    n = bvals.shape[0]
    for i in range(n):
        norm = np.linalg.norm(g[i])
        if bvals[i] <= b0_thr:
            g[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif norm > 1e-8:
            g[i] = g[i] / norm
        else:
            g[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.column_stack([g, bvals.astype(np.float64)])


def write_scheme(dwi_dir: Path, stem: str) -> Path:
    bval = dwi_dir / f"{stem}.bval"
    bvec = dwi_dir / f"{stem}.bvec"
    out = dwi_dir / "DWI.scheme"
    tab = bval_bvec_to_scheme(bval, bvec)
    np.savetxt(out, tab, fmt="%.8e")
    return out


def run_amico_session(subject: str, session: str, regenerate_kernels: bool = True) -> Path | None:
    import amico

    dwi_dir = OUTPUT_ROOT / subject / session / "dwi"

    # Check prerequisites
    denoised = dwi_dir / "dwi_patch2self.nii.gz"
    mask = dwi_dir / "brain_mask.nii.gz"
    if not denoised.exists() or not mask.exists():
        print(f"  Skipping {subject}/{session}: missing preprocessed files")
        return None

    # Check if already done
    noddi_dir = dwi_dir / "AMICO" / "NODDI"
    if noddi_dir.exists() and (noddi_dir / "fit_NDI.nii.gz").exists():
        print(f"  {subject}/{session}: AMICO already complete, skipping")
        return noddi_dir

    stem = f"{subject}_{session}_dwi"
    write_scheme(dwi_dir, stem)

    # AMICO study dir is parent of the relative path
    study_dir = OUTPUT_ROOT / subject
    rel_subject = f"{session}/dwi"

    amico.set_verbose(2)
    ae = amico.Evaluation(str(study_dir), rel_subject)
    ae.load_data(
        dwi_filename="dwi_patch2self.nii.gz",
        scheme_filename="DWI.scheme",
        mask_filename="brain_mask.nii.gz",
        b0_thr=0,
    )
    ae.set_model("NODDI")
    ae.generate_kernels(regenerate=regenerate_kernels)
    ae.load_kernels()
    ae.fit()
    ae.save_results()
    return noddi_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run AMICO NODDI on all subjects/sessions")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS
    sessions = [args.session] if args.session else SESSIONS

    import amico
    print("AMICO setup...")
    amico.setup()

    results = {}
    for sub in subjects:
        for ses in sessions:
            print(f"\n=== {sub} / {ses} ===")
            try:
                result = run_amico_session(sub, ses, regenerate_kernels=True)
                results[(sub, ses)] = "SUCCESS" if result else "SKIPPED"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[(sub, ses)] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("AMICO SUMMARY")
    print(f"{'='*60}")
    for (sub, ses), status in results.items():
        print(f"  {sub}/{ses}: {status}")


if __name__ == "__main__":
    main()
