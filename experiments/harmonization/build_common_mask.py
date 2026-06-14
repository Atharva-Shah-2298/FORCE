"""
Build a per-subject common WM mask from the aligned outputs.

Per subject:
  common_brain = intersection of all aligned brain_mask.nii.gz
  common_wm    = common_brain ∩ (reference-session DTI FA > WM_FA_THRESHOLD)

Saves:
  output_aligned/<subject>/common_brain_mask.nii.gz
  output_aligned/<subject>/common_wm_mask.nii.gz
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("common_mask")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HARMONIZATION_ROOT", SCRIPT_DIR))
ALIGNED_ROOT = Path(os.environ.get("HARMONIZATION_ALIGNED",
                                   ROOT / "output_aligned"))

SUBJECTS = ["sub-1", "sub-2", "sub-3"]
SESSIONS = [
    "ses-c01r1", "ses-c02r1", "ses-c03r1", "ses-c04r1", "ses-c05r1",
    "ses-c06r1", "ses-c07r1", "ses-c08r1", "ses-c09r1", "ses-c10r1",
    "ses-c10r2", "ses-c10r3",
]
REFERENCE_SESSION = "ses-c01r1"
WM_FA_THRESHOLD = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--wm-threshold", type=float, default=WM_FA_THRESHOLD)
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS

    summary = {}
    for sub in subjects:
        sub_dir = ALIGNED_ROOT / sub
        if not sub_dir.exists():
            log.warning(f"Missing aligned tree for {sub}")
            continue

        # Load all brain masks
        masks = []
        ref_affine = None
        for ses in SESSIONS:
            p = sub_dir / ses / "brain_mask.nii.gz"
            if not p.exists():
                log.warning(f"Missing {p}")
                continue
            img = nib.load(str(p))
            masks.append(img.get_fdata() > 0.5)
            if ref_affine is None:
                ref_affine = img.affine
                ref_shape = img.shape[:3]

        if not masks:
            log.error(f"No masks for {sub}")
            continue

        # Intersection
        common_brain = masks[0].copy()
        for m in masks[1:]:
            if m.shape != common_brain.shape:
                log.warning(f"Shape mismatch in {sub}: {m.shape} vs {common_brain.shape}")
                continue
            common_brain &= m

        # Reference DTI FA → WM mask
        ref_fa_path = sub_dir / REFERENCE_SESSION / "dti" / "fa.nii.gz"
        if not ref_fa_path.exists():
            log.error(f"No reference FA for {sub}: {ref_fa_path}")
            continue
        fa = nib.load(str(ref_fa_path)).get_fdata()
        wm = (fa > args.wm_threshold) & common_brain

        n_brain = int(common_brain.sum())
        n_wm = int(wm.sum())

        nib.save(nib.Nifti1Image(common_brain.astype(np.uint8), ref_affine),
                 str(sub_dir / "common_brain_mask.nii.gz"))
        nib.save(nib.Nifti1Image(wm.astype(np.uint8), ref_affine),
                 str(sub_dir / "common_wm_mask.nii.gz"))

        summary[sub] = {
            "n_sessions_masks": len(masks),
            "n_common_brain": n_brain,
            "n_common_wm": n_wm,
            "wm_threshold": args.wm_threshold,
        }
        log.info(f"{sub}: brain={n_brain} voxels, WM={n_wm} voxels "
                 f"(from {len(masks)} sessions)")

    with open(ALIGNED_ROOT / "common_mask_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
