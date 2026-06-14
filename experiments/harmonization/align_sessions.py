"""
Post-hoc registration of all sessions to each subject's reference session.

For each (subject, session) pair where session != reference:
  1. Register session's DTI-FA to reference's DTI-FA (affine)
  2. Save affine matrix
  3. Apply affine to every metric map (DTI/DKI/FORCE/AMICO) and brain mask
  4. Write aligned NIfTIs under output_aligned/

Resumable: skips sessions whose affine matrix already exists, unless --force.

Usage (on remote server / wherever the mount is):
    python align_sessions.py                 # align all
    python align_sessions.py --subject sub-1
    python align_sessions.py --session ses-c02r1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("align")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HARMONIZATION_ROOT", SCRIPT_DIR))
OUTPUT_ROOT = Path(os.environ.get("HARMONIZATION_OUTPUT", ROOT / "output"))
ALIGNED_ROOT = Path(os.environ.get("HARMONIZATION_ALIGNED",
                                   ROOT / "output_aligned"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
    "ses-c10r2", "ses-c10r3",
]
REFERENCE_SESSION = "ses-c01r1"

# Every metric map to transform per session. (method_dir, filename) pairs.
METRIC_MAPS = [
    ("dti", "fa.nii.gz"),
    ("dti", "md.nii.gz"),
    ("dti", "ad.nii.gz"),
    ("dti", "rd.nii.gz"),
    ("dki", "fa.nii.gz"),
    ("dki", "md.nii.gz"),
    ("dki", "ad.nii.gz"),
    ("dki", "rd.nii.gz"),
    ("dki", "mk.nii.gz"),
    ("dki", "ak.nii.gz"),
    ("dki", "rk.nii.gz"),
    ("force", "fa.nii.gz"),
    ("force", "md.nii.gz"),
    ("force", "rd.nii.gz"),
    ("force", "nd.nii.gz"),
    ("force", "dispersion.nii.gz"),
    ("force", "wm_fraction.nii.gz"),
    ("force", "gm_fraction.nii.gz"),
    ("force", "csf_fraction.nii.gz"),
    ("force", "ufa.nii.gz"),
    ("force", "mk.nii.gz"),
    ("force", "ak.nii.gz"),
    ("force", "rk.nii.gz"),
    ("AMICO/NODDI", "fit_NDI.nii.gz"),
    ("AMICO/NODDI", "fit_ODI.nii.gz"),
    ("AMICO/NODDI", "fit_FWF.nii.gz"),
]


def session_dir(subject: str, session: str) -> Path:
    return OUTPUT_ROOT / subject / session / "dwi"


def aligned_dir(subject: str, session: str) -> Path:
    return ALIGNED_ROOT / subject / session


def dti_fa_path(subject: str, session: str) -> Path:
    return session_dir(subject, session) / "dti" / "fa.nii.gz"


def mask_path(subject: str, session: str) -> Path:
    return session_dir(subject, session) / "brain_mask.nii.gz"


def register_affine(static_img: nib.Nifti1Image,
                    moving_img: nib.Nifti1Image):
    """Register moving to static via center-of-mass → translation → rigid → affine.

    Returns an AffineMap whose .transform() maps moving-space voxels to static space.
    """
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import (
        AffineTransform3D,
        RigidTransform3D,
        TranslationTransform3D,
    )

    static = np.asarray(static_img.get_fdata(), dtype=np.float32)
    static_aff = static_img.affine
    moving = np.asarray(moving_img.get_fdata(), dtype=np.float32)
    moving_aff = moving_img.affine

    nbins = 32
    metric = MutualInformationMetric(nbins, sampling_proportion=None)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        verbosity=0,
    )

    # 1) Center-of-mass alignment
    c_of_mass = transform_centers_of_mass(static, static_aff, moving, moving_aff)

    # 2) Translation
    starting_affine = c_of_mass.affine
    transl = affreg.optimize(static, moving, TranslationTransform3D(), None,
                             static_aff, moving_aff,
                             starting_affine=starting_affine)

    # 3) Rigid
    rigid = affreg.optimize(static, moving, RigidTransform3D(), None,
                            static_aff, moving_aff,
                            starting_affine=transl.affine)

    # 4) Affine
    affine = affreg.optimize(static, moving, AffineTransform3D(), None,
                             static_aff, moving_aff,
                             starting_affine=rigid.affine)
    return affine


def apply_affine_to_volume(src_img: nib.Nifti1Image,
                           affine_map,
                           ref_img: nib.Nifti1Image,
                           interp: str = "linear") -> nib.Nifti1Image:
    """Resample src_img into ref_img space using a fitted AffineMap."""
    moving = np.asarray(src_img.get_fdata(), dtype=np.float32)
    moving_aff = src_img.affine

    if interp == "nearest":
        warped = affine_map.transform(
            moving, interpolation="nearest",
            sampling_grid_shape=ref_img.shape[:3],
        )
    else:
        warped = affine_map.transform(
            moving,
            sampling_grid_shape=ref_img.shape[:3],
        )
    return nib.Nifti1Image(warped.astype(np.float32), ref_img.affine)


def process_session(subject: str, session: str, force: bool) -> dict:
    """Register session to reference and resample all metric maps. Returns status dict."""
    is_ref = session == REFERENCE_SESSION
    out_dir = aligned_dir(subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)

    affine_txt = out_dir / "affine_to_ref.txt"

    # Reference session: just copy (symlink) all maps into the aligned tree
    if is_ref:
        if (out_dir / "brain_mask.nii.gz").exists() and not force:
            return {"subject": subject, "session": session, "status": "skipped_ref"}

        log.info(f"[{subject}/{session}] Reference session — copying to aligned tree")

        src_mask = mask_path(subject, session)
        if src_mask.exists():
            _symlink_or_copy(src_mask, out_dir / "brain_mask.nii.gz")

        for method_dir, fname in METRIC_MAPS:
            src = session_dir(subject, session) / method_dir / fname
            if not src.exists():
                continue
            dst = out_dir / method_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            _symlink_or_copy(src, dst)

        # Write identity affine
        np.savetxt(affine_txt, np.eye(4), fmt="%.8f")
        return {"subject": subject, "session": session, "status": "ref_copied"}

    # Non-reference session
    ref_fa = dti_fa_path(subject, REFERENCE_SESSION)
    mov_fa = dti_fa_path(subject, session)

    if not ref_fa.exists():
        return {"subject": subject, "session": session, "status": "missing_ref_fa"}
    if not mov_fa.exists():
        return {"subject": subject, "session": session, "status": "missing_mov_fa"}

    static_img = nib.load(str(ref_fa))
    moving_img = nib.load(str(mov_fa))

    if affine_txt.exists() and not force:
        log.info(f"[{subject}/{session}] Loading cached affine")
        M = np.loadtxt(affine_txt)
        from dipy.align.imaffine import AffineMap
        affine_map = AffineMap(
            M,
            domain_grid_shape=static_img.shape[:3],
            domain_grid2world=static_img.affine,
            codomain_grid_shape=moving_img.shape[:3],
            codomain_grid2world=moving_img.affine,
        )
    else:
        log.info(f"[{subject}/{session}] Registering DTI-FA → reference")
        t0 = time.time()
        affine_map = register_affine(static_img, moving_img)
        log.info(f"[{subject}/{session}] Registration done in {time.time() - t0:.1f}s")
        np.savetxt(affine_txt, affine_map.affine, fmt="%.8f")

    # Apply to mask (nearest) and every metric map (linear)
    src_mask = mask_path(subject, session)
    if src_mask.exists():
        warped = apply_affine_to_volume(nib.load(str(src_mask)), affine_map,
                                        static_img, interp="nearest")
        # Binarize
        warped_data = (warped.get_fdata() > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(warped_data, static_img.affine),
                 str(out_dir / "brain_mask.nii.gz"))

    n_applied = 0
    n_missing = 0
    for method_dir, fname in METRIC_MAPS:
        src = session_dir(subject, session) / method_dir / fname
        if not src.exists():
            n_missing += 1
            continue
        dst = out_dir / method_dir / fname
        if dst.exists() and not force:
            n_applied += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        warped = apply_affine_to_volume(nib.load(str(src)), affine_map,
                                        static_img, interp="linear")
        nib.save(warped, str(dst))
        n_applied += 1

    log.info(f"[{subject}/{session}] Applied affine to {n_applied} metric maps "
             f"({n_missing} missing)")
    return {
        "subject": subject,
        "session": session,
        "status": "aligned",
        "n_applied": n_applied,
        "n_missing": n_missing,
    }


def _symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS
    sessions = [args.session] if args.session else SESSIONS

    ALIGNED_ROOT.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(ALIGNED_ROOT / "align.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    log.info(f"OUTPUT_ROOT  = {OUTPUT_ROOT}")
    log.info(f"ALIGNED_ROOT = {ALIGNED_ROOT}")

    results = []
    for sub in subjects:
        for ses in sessions:
            if not session_dir(sub, ses).exists():
                log.warning(f"No data for {sub}/{ses}, skipping")
                continue
            try:
                res = process_session(sub, ses, force=args.force)
            except Exception as e:
                log.exception(f"[{sub}/{ses}] FAILED: {e}")
                res = {"subject": sub, "session": ses, "status": "error",
                       "error": str(e)}
            results.append(res)

    # Summary
    log.info("=" * 60)
    log.info("ALIGNMENT SUMMARY")
    for r in results:
        log.info(f"  {r['subject']}/{r['session']}: {r['status']}")

    import json
    with open(ALIGNED_ROOT / "alignment_status.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
