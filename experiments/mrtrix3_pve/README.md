# MRtrix3 partial-volume (tissue volume-fraction) maps

The tissue partial-volume maps used as a reference for FORCE's WM/GM/CSF fraction
maps were produced with [MRtrix3](https://www.mrtrix.org) via multi-shell
multi-tissue constrained spherical deconvolution (MSMT-CSD). The exact CLI
pipeline (run on the multi-shell HCP data) is below.

```bash
# 1. Convert the preprocessed DWI and brain mask to MRtrix format,
#    embedding the gradient scheme.
mrconvert dwi.nii.gz dwi.mif -fslgrad bvecs bvals
mrconvert brain_mask.nii.gz mask.mif

# 2. Estimate WM / GM / CSF response functions (unsupervised, Dhollander).
dwi2response dhollander dwi.mif \
    wm_response.txt gm_response.txt csf_response.txt \
    -mask mask.mif

# 3. Multi-shell multi-tissue CSD: WM FOD + GM and CSF compartments.
dwi2fod msmt_csd dwi.mif \
    wm_response.txt wmfod.mif \
    gm_response.txt  gm.mif \
    csf_response.txt csf.mif \
    -mask mask.mif

# 4. Build the 3-tissue partial-volume map: take the l=0 (DC) term of the
#    WM FOD as the WM volume fraction and stack CSF, GM, WM into one image.
mrconvert -coord 3 0 wmfod.mif - | mrcat csf.mif gm.mif - vf.mif
mrconvert vf.mif vf.nii.gz
```

`vf.nii.gz` is a 4-D image with three volumes — **[CSF, GM, WM]** in that order —
giving the per-voxel tissue partial-volume fractions that FORCE's
`csf_fraction` / `gm_fraction` / `wm_fraction` outputs are compared against.

> Note: this multi-tissue pipeline needs multi-shell data. For the single-shell
> Stanford HARDI CSD/tractography comparison the response was instead estimated
> with `dwi2response tournier csd_dwi.mif response_csd.txt -fslgrad HARDI150.bval HARDI150.bvec`.
