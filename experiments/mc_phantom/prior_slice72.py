"""Compare FORCE on HCP 165840 slice z=72 with the default within-WM NDI prior
f_intra in [0.6,0.9] vs a widened [0.3,0.9] prior (so FORCE can represent
lower-density WM directly instead of leaking it into GM/CSF).

The widened prior is injected by intercepting the Python-level
np.random.uniform(0.6,0.9) draw used for f_intra (the only place that exact
range is requested) and remapping it to (0.3,0.9). Verified to change the
dictionary's within-WM NDI to [0.30,0.90] with no recompile. Linux fork lets
the patch propagate to the parallel generation workers.

Outputs per prior (in force_slice72_<tag>/): PVE maps (wm/gm/csf fraction,
nd, dispersion) as full-volume NIfTI with only z=72 filled, and a .pam5
(posterior ODF -> SH order 8 + up to 5 peaks) for ODF inspection at z=72.
"""
import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.shm import sf_to_sh
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.peaks import save_pam
from dipy.reconst.force import FORCEModel, softmax_stable

P = os.environ.get("FORCE_SUBJECT", "hcp_subject")  # HCP-like subject: DWI + bvals/bvecs
OUTROOT = os.path.dirname(os.path.abspath(__file__))
Z = 72
K, BETA, NSIM = 50, 2000.0, 500000


def patched_uniform():
    _orig = np.random.uniform
    def patched(low=0.0, high=1.0, size=None):
        if np.isscalar(low) and np.isscalar(high) and abs(low - 0.6) < 1e-9 and abs(high - 0.9) < 1e-9:
            low = 0.3
        return _orig(low, high, size)
    return _orig, patched


def sphere_nn(sphere, k=18):
    V = sphere.vertices
    s = V @ V.T
    np.fill_diagonal(s, -1.0)
    return np.argpartition(-s, k, axis=1)[:, :k]


def extract_peaks(odf, sphere, nn, min_sep=20.0, top=5, rel=0.3):
    local = np.all(odf[:, None] >= odf[nn] - 1e-9, axis=1) & (odf > rel * odf.max())
    cand = np.where(local)[0]
    if cand.size == 0:
        return np.zeros((top, 3)), np.zeros(top), np.full(top, -1, int)
    order = cand[np.argsort(-odf[cand])]
    cthr = np.cos(np.deg2rad(min_sep))
    dirs, vals, idxs = [], [], []
    for i in order:
        d = sphere.vertices[i]
        if all(abs(float(d @ p)) <= cthr for p in dirs):
            dirs.append(d); vals.append(float(odf[i])); idxs.append(int(i))
            if len(dirs) >= top:
                break
    pd = np.zeros((top, 3)); pv = np.zeros(top); pi = np.full(top, -1, int)
    for j in range(len(dirs)):
        pd[j] = dirs[j]; pv[j] = vals[j]; pi[j] = idxs[j]
    return pd, pv, pi


def run_prior(tag, patched, gtab, dwi_slice, mask_slice, affine):
    out = os.path.join(OUTROOT, f"force_slice72_{tag}")
    os.makedirs(out, exist_ok=True)
    print(f"\n===== prior '{tag}' =====")
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=BETA,
                       compute_odf=True, verbose=True)
    if patched:
        _orig, patch = patched_uniform()
        np.random.uniform = patch
        try:
            model.generate(num_simulations=NSIM, use_cache=False, num_cpus=-1,
                           compute_dti=True, compute_dki=False)
        finally:
            np.random.uniform = _orig
    else:
        model.generate(num_simulations=NSIM, use_cache=True, num_cpus=-1,
                       compute_dti=True, compute_dki=False)
    sims = model.simulations
    wmnd = np.asarray(sims["nd"]) / np.maximum(np.asarray(sims["wm_fraction"]), 1e-9)
    print(f"  dict within-WM NDI range [{wmnd.min():.2f},{wmnd.max():.2f}] "
          f"%<0.6={100*(wmnd<0.6).mean():.0f}; odfs {sims['odfs'].shape}")

    # slice-72 voxels
    yx = np.where(mask_slice > 0)
    Q = dwi_slice[yx].astype(np.float64)              # (Nvox, 288)
    qn = np.linalg.norm(Q, axis=1, keepdims=True); qn[qn == 0] = 1
    Dist, neigh = model._index.search(np.ascontiguousarray((Q / qn).astype(np.float32)), k=K)
    W = softmax_stable(BETA * (Dist - model._penalty_array[neigh]), axis=1).astype(np.float32)

    XS, YS, ZS = mask_slice.shape[0], mask_slice.shape[1], 145
    maps = {}
    for fld in ("wm_fraction", "gm_fraction", "csf_fraction", "nd", "dispersion", "fa", "md"):
        vals = np.einsum('nk,nk->n', W, sims[fld][neigh]).astype(np.float32)
        vol = np.zeros((XS, YS, ZS), np.float32)
        vol[yx[0], yx[1], Z] = vals
        maps[fld] = vol
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(out, f"{fld}.nii.gz"))

    # posterior ODF + SH + peaks
    sph = default_sphere
    if sims["odfs"].shape[1] != sph.vertices.shape[0]:
        from dipy.core.sphere import Sphere
        # FORCE odfs may be on a denser sphere; rebuild matching sphere not stored,
        # fall back: use the model's odf sphere if available
        sph = getattr(model, "sphere", default_sphere)
    nv = sims["odfs"].shape[1]
    nn = sphere_nn(sph)
    odf_vol = np.zeros((XS, YS, ZS, nv), np.float32)
    sh_vol = np.zeros((XS, YS, ZS, 45), np.float32)
    pdv = np.zeros((XS, YS, ZS, 5, 3), np.float32)
    pvv = np.zeros((XS, YS, ZS, 5), np.float32)
    piv = np.full((XS, YS, ZS, 5), -1, np.int32)
    Nvox = Q.shape[0]
    B = 1000
    for s0 in range(0, Nvox, B):
        e = min(s0 + B, Nvox)
        o = sims["odfs"][neigh[s0:e]].astype(np.float32)         # (b,K,nv)
        o /= (o.max(axis=2, keepdims=True) + 1e-8)
        op = np.einsum('nk,nkv->nv', W[s0:e], o)
        op /= (op.max(axis=1, keepdims=True) + 1e-8)
        for j in range(e - s0):
            xi, yi = yx[0][s0 + j], yx[1][s0 + j]
            odf_vol[xi, yi, Z] = op[j]
            sh_vol[xi, yi, Z] = sf_to_sh(op[j], sphere=sph, sh_order_max=8)
            pd, pv, pi = extract_peaks(op[j], sph, nn)
            pdv[xi, yi, Z] = pd; pvv[xi, yi, Z] = pv; piv[xi, yi, Z] = pi

    pam = PeaksAndMetrics()
    pam.sphere = sph
    pam.peak_dirs = pdv
    pam.peak_values = pvv
    pam.peak_indices = piv
    pam.shm_coeff = sh_vol
    pam.odf = odf_vol
    pam.affine = affine
    pam.B = None
    pam.total_weight = 0.5
    pam.ang_thr = 60.0
    pam.gfa = np.zeros((XS, YS, ZS), np.float32)
    pam.qa = pvv.copy()
    save_pam(os.path.join(out, "peaks_z72.pam5"), pam, affine=affine)
    print(f"  saved {out}/peaks_z72.pam5 + PVE maps  ({Nvox} voxels at z={Z})")
    del model, sims
    import gc; gc.collect()
    return maps


def main():
    bvals = np.loadtxt(os.path.join(P, "bvals")).ravel()
    bvecs = np.loadtxt(os.path.join(P, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    dwi_img = nib.load(os.path.join(P, "denoised_arr_p2s.nii.gz"))
    affine = dwi_img.affine
    dwi = dwi_img.get_fdata(dtype=np.float32)
    mask = nib.load(os.path.join(P, "nodif_brain_mask.nii.gz")).get_fdata()
    dwi_slice = dwi[:, :, Z, :]
    mask_slice = mask[:, :, Z]
    print(f"slice z={Z}: {int((mask_slice>0).sum())} voxels")
    run_prior("oldprior", False, gtab, dwi_slice, mask_slice, affine)
    run_prior("newprior", True, gtab, dwi_slice, mask_slice, affine)


if __name__ == "__main__":
    main()
