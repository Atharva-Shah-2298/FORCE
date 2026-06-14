"""
Exp 9 — Generator with explicit Watson / truncated-Gaussian / logit-normal
separation from the FORCE dictionary.

Outputs N voxels with full latent ground truth, on the *identical*
HCP 165840 bvals/bvecs that the cached 500K dictionary was built against.

Separation from dictionary (auditable):
  - Orientation distribution: Watson(mu, kappa), Monte-Carlo sampled M=64
    sticks per fiber. Dictionary uses Bingham LUT on default_sphere.
  - Diffusivity sampling: truncnorm within the dictionary's prior ranges,
    independent across compartments. Dictionary uses Uniform priors over
    the same ranges (no tortuosity coupling on either side).
  - Tissue-fraction prior: softmax of independent Gaussian latents
    (logit-normal). Dictionary uses Dirichlet(2,1,1).
  - WM intra-axonal fraction (NDI within WM): truncnorm on [0.1, 0.9].
  - csf_d fixed at 3.0e-3 (same as dictionary).
  - No soma / sphere compartment, no inter-compartment exchange — same
    compartmental family as the dictionary.
"""

import argparse
import json
import os
import numpy as np
from numpy.random import default_rng
from scipy.stats import truncnorm
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel


# --- Dictionary's diffusivity priors (uniform; we match the ranges
# --- but draw from truncnorm instead) ---
WM_DPAR_LO, WM_DPAR_HI = 2.0e-3, 3.0e-3
WM_DPERP_LO, WM_DPERP_HI = 0.3e-3, 1.5e-3
GM_DISO_LO, GM_DISO_HI = 0.7e-3, 1.2e-3
CSF_D = 3.0e-3

# Truncated-Gaussian hyperparameters (mean = midpoint, sd = quarter-range)
WM_DPAR_MU, WM_DPAR_SD = 0.5 * (WM_DPAR_LO + WM_DPAR_HI), 0.25 * (WM_DPAR_HI - WM_DPAR_LO)
WM_DPERP_MU, WM_DPERP_SD = 0.5 * (WM_DPERP_LO + WM_DPERP_HI), 0.30 * (WM_DPERP_HI - WM_DPERP_LO)
GM_DISO_MU, GM_DISO_SD = 0.5 * (GM_DISO_LO + GM_DISO_HI), 0.30 * (GM_DISO_HI - GM_DISO_LO)

# ODI range that the dictionary covers (matches sims/force.py odi_range)
ODI_LO, ODI_HI = 0.01, 0.30
ODI_MU, ODI_SD = 0.15, 0.12

# NDI sampling
NDI_LO, NDI_HI = 0.10, 0.90
NDI_MU, NDI_SD = 0.55, 0.20

# Tissue-fraction logit-normal hyperparameters (favor WM-dominated voxels
# while keeping substantial GM and FW variability — different shape from
# the dictionary's Dirichlet(2,1,1).)
LOGIT_WM_MU, LOGIT_WM_SD = 1.0, 1.2
LOGIT_GM_MU, LOGIT_GM_SD = -0.5, 1.0
LOGIT_FW_MU, LOGIT_FW_SD = -0.5, 1.0


def _trunc_norm(rng, mu, sd, lo, hi, size):
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size, random_state=rng)


def watson_sample(mu, kappa, M, rng):
    """Draw M unit vectors from Watson(mu, kappa) by rejection on cos(theta).

    Watson density on the sphere: f(n) prop exp(kappa * (mu.n)^2).
    Marginal of t = mu.n on [-1,1] has density prop exp(kappa t^2).
    Azimuth uniform.
    """
    # rejection sample t = cos(theta)
    accepted = np.empty(M)
    n = 0
    max_log = kappa  # max of kappa*t^2 over [-1,1] is at |t|=1
    while n < M:
        batch = M - n
        t = rng.uniform(-1.0, 1.0, size=batch)
        log_f = kappa * t * t
        u = rng.uniform(size=batch)
        keep = np.log(u) < (log_f - max_log)
        nk = int(keep.sum())
        accepted[n:n + nk] = t[keep][:nk]
        n += nk
    phi = rng.uniform(0.0, 2.0 * np.pi, size=M)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - accepted ** 2))
    # local-frame unit vector with mu = (0,0,1)
    local = np.column_stack([sin_th * np.cos(phi),
                             sin_th * np.sin(phi),
                             accepted])
    # rotate local frame so that (0,0,1) -> mu
    R = _rot_z_to(mu)
    return local @ R.T  # (M, 3) in lab frame


def _rot_z_to(v):
    """Rotation matrix R s.t. R @ [0,0,1] = v (unit)."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg around x
        return np.diag([1.0, -1.0, -1.0])
    k = np.cross(z, v)
    s = np.linalg.norm(k)
    k = k / s
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def odi_to_kappa(odi):
    return 1.0 / np.tan(np.pi * odi / 2.0)


def _sample_random_unit(rng):
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * np.cos(phi), s * np.sin(phi), z])


def _sample_second_direction(primary, angle_rad, rng):
    """Sample a unit vector at exactly angle_rad from `primary`."""
    # build any vector perp to primary
    other = _sample_random_unit(rng)
    perp = other - np.dot(other, primary) * primary
    nperp = np.linalg.norm(perp)
    if nperp < 1e-6:
        other = np.array([1.0, 0.0, 0.0]) if abs(primary[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = other - np.dot(other, primary) * primary
        nperp = np.linalg.norm(perp)
    perp = perp / nperp
    return np.cos(angle_rad) * primary + np.sin(angle_rad) * perp


def compute_signal(bvals, bvecs, components):
    """Signal from compartment list.

    components: list of dicts with keys:
        kind: 'aniso'  -> {direction, d_par, d_perp, weight}
        kind: 'iso'    -> {d, weight}
    Returns S (n_bvals,), normalized so that S(b=0) = sum(weights).
    """
    sig = np.zeros(bvals.shape, dtype=np.float64)
    for c in components:
        w = c["weight"]
        if c["kind"] == "iso":
            sig += w * np.exp(-bvals * c["d"])
        else:
            n = c["direction"]
            d_par = c["d_par"]
            d_perp = c["d_perp"]
            dot = bvecs @ n  # (n_bvals,)
            # signal = exp(-b*[d_perp + (d_par-d_perp)*(g.n)^2])
            sig += w * np.exp(-bvals * (d_perp + (d_par - d_perp) * dot * dot))
    return sig


def make_voxel(rng, bvals, bvecs, M_sticks=64):
    """Sample one voxel; return (signal_clean, latents_dict)."""
    # --- choose number of fibers ---
    p_K = np.array([0.10, 0.40, 0.35, 0.15])  # K = 0,1,2,3
    K = int(rng.choice([0, 1, 2, 3], p=p_K))

    # --- tissue fractions via logit-normal (non-Dirichlet) ---
    u_wm = rng.normal(LOGIT_WM_MU, LOGIT_WM_SD)
    u_gm = rng.normal(LOGIT_GM_MU, LOGIT_GM_SD)
    u_fw = rng.normal(LOGIT_FW_MU, LOGIT_FW_SD)
    u = np.array([u_wm, u_gm, u_fw])
    e = np.exp(u - u.max())
    f_wm, f_gm, f_fw = (e / e.sum()).tolist()

    # If K=0, force WM fraction to 0 (no fibers means no anisotropic WM)
    if K == 0:
        # rescale: send WM mass into GM/FW preserving ratio
        rest = f_gm + f_fw
        if rest < 1e-6:
            f_gm = 0.5; f_fw = 0.5
        else:
            f_gm = f_gm / rest
            f_fw = f_fw / rest
        f_wm = 0.0

    # --- crossing angle ---
    if K >= 2:
        # Truncated Gaussian on [10°, 90°] mean 55°, sd 25°
        ang_lo, ang_hi = 10.0, 90.0
        ang_mu, ang_sd = 55.0, 25.0
        a, b = (ang_lo - ang_mu) / ang_sd, (ang_hi - ang_mu) / ang_sd
        cross_deg = float(truncnorm.rvs(a, b, loc=ang_mu, scale=ang_sd, random_state=rng))
    else:
        cross_deg = np.nan

    # --- fiber directions ---
    dirs = np.full((3, 3), np.nan)
    if K >= 1:
        dirs[0] = _sample_random_unit(rng)
    if K >= 2:
        dirs[1] = _sample_second_direction(dirs[0], np.deg2rad(cross_deg), rng)
    if K == 3:
        # third fiber: angle from primary uniform on [30, 90], independent rotation
        ang3 = float(truncnorm.rvs(-1.0, 1.4, loc=60.0, scale=25.0, random_state=rng))
        # rotate around primary axis from dirs[1]
        # construct an orthonormal frame around primary
        z = dirs[0]
        x = dirs[1] - np.dot(dirs[1], z) * z
        x = x / max(np.linalg.norm(x), 1e-8)
        y = np.cross(z, x)
        phi3 = rng.uniform(0, 2 * np.pi)
        d3_in_plane = np.cos(phi3) * x + np.sin(phi3) * y
        dirs[2] = np.cos(np.deg2rad(ang3)) * z + np.sin(np.deg2rad(ang3)) * d3_in_plane
        dirs[2] = dirs[2] / max(np.linalg.norm(dirs[2]), 1e-8)

    # --- per-fiber diffusivities (truncated Gaussian) and ODI ---
    odi = np.full(3, np.nan)
    d_par = np.full(3, np.nan)
    d_perp = np.full(3, np.nan)
    if K >= 1:
        odi[:K] = _trunc_norm(rng, ODI_MU, ODI_SD, ODI_LO, ODI_HI, K)
        d_par[:K] = _trunc_norm(rng, WM_DPAR_MU, WM_DPAR_SD, WM_DPAR_LO, WM_DPAR_HI, K)
        d_perp[:K] = _trunc_norm(rng, WM_DPERP_MU, WM_DPERP_SD, WM_DPERP_LO, WM_DPERP_HI, K)

    # --- NDI ---
    if K >= 1:
        ndi = float(_trunc_norm(rng, NDI_MU, NDI_SD, NDI_LO, NDI_HI, 1)[0])
    else:
        ndi = np.nan

    # --- GM diffusivity ---
    d_gm = float(_trunc_norm(rng, GM_DISO_MU, GM_DISO_SD, GM_DISO_LO, GM_DISO_HI, 1)[0])

    # --- assemble components ---
    components = []
    if f_gm > 0:
        components.append({"kind": "iso", "d": d_gm, "weight": f_gm})
    if f_fw > 0:
        components.append({"kind": "iso", "d": CSF_D, "weight": f_fw})
    if K >= 1 and f_wm > 0:
        per_fiber_wm = f_wm / K
        for j in range(K):
            mu = dirs[j]
            kappa = odi_to_kappa(odi[j])
            sticks = watson_sample(mu, kappa, M_sticks, rng)
            w_intra = per_fiber_wm * ndi / M_sticks
            w_extra = per_fiber_wm * (1.0 - ndi) / M_sticks
            for s in sticks:
                components.append({"kind": "aniso", "direction": s,
                                   "d_par": d_par[j], "d_perp": 0.0,
                                   "weight": w_intra})
                components.append({"kind": "aniso", "direction": s,
                                   "d_par": d_par[j], "d_perp": d_perp[j],
                                   "weight": w_extra})

    sig = compute_signal(bvals, bvecs, components)

    latents = {
        "num_fibers": K,
        "crossing_angle_deg": cross_deg,
        "f_wm": f_wm, "f_gm": f_gm, "f_fw": f_fw,
        "ndi": ndi,
        "odi_mean": float(np.nanmean(odi[:K])) if K >= 1 else np.nan,
        "odi_per_fiber": odi.tolist(),
        "d_par_per_fiber": d_par.tolist(),
        "d_perp_per_fiber": d_perp.tolist(),
        "d_gm": d_gm,
        "dirs": dirs.tolist(),
    }
    return sig.astype(np.float32), latents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bvals", default="/path/to/subject/bvals")
    p.add_argument("--bvecs", default="/path/to/subject/bvecs")
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--M_sticks", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bvals = np.loadtxt(args.bvals).astype(np.float64)
    bvecs = np.loadtxt(args.bvecs).astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    # FORCE expects b-values in s/mm^2 with reasonable scale already.
    # dipy multi_tensor and our compute_signal assume b in s/mm^2, d in mm^2/s.
    b_for_signal = bvals.copy()

    print(f"gtab: n_vols={bvals.size} shells={np.unique(np.round(bvals/100)*100).astype(int)}")

    rng = default_rng(args.seed)
    signals = np.empty((args.N, bvals.size), dtype=np.float32)
    latents_list = []
    from tqdm import tqdm
    for i in tqdm(range(args.N), desc="generating"):
        s, lat = make_voxel(rng, b_for_signal, bvecs, M_sticks=args.M_sticks)
        signals[i] = s
        latents_list.append(lat)

    # Derive clean-signal DTI ground truth (FA/MD/RD) using small-b shells
    # (b <= 1000) to stay in the monoexponential DTI regime
    gtab_full = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    dti_mask = (bvals <= 1050)
    dti_gtab = gradient_table(bvals[dti_mask], bvecs=bvecs[dti_mask], b0_threshold=50)
    print(f"DTI ground truth uses {int(dti_mask.sum())} volumes (b<=1000)")
    dti_model = TensorModel(dti_gtab)
    dti_fit = dti_model.fit(signals[:, dti_mask])
    fa_true = np.asarray(dti_fit.fa, dtype=np.float32)
    md_true = np.asarray(dti_fit.md, dtype=np.float32)
    rd_true = np.asarray(dti_fit.rd, dtype=np.float32)

    # Pack ground truth arrays
    N = args.N
    K = np.array([l["num_fibers"] for l in latents_list], dtype=np.int8)
    crossing = np.array([l["crossing_angle_deg"] for l in latents_list], dtype=np.float32)
    f_wm = np.array([l["f_wm"] for l in latents_list], dtype=np.float32)
    f_gm = np.array([l["f_gm"] for l in latents_list], dtype=np.float32)
    f_fw = np.array([l["f_fw"] for l in latents_list], dtype=np.float32)
    ndi = np.array([l["ndi"] for l in latents_list], dtype=np.float32)
    odi_mean = np.array([l["odi_mean"] for l in latents_list], dtype=np.float32)
    dirs = np.array([l["dirs"] for l in latents_list], dtype=np.float32)
    odi_pf = np.array([l["odi_per_fiber"] for l in latents_list], dtype=np.float32)
    dpar_pf = np.array([l["d_par_per_fiber"] for l in latents_list], dtype=np.float32)
    dperp_pf = np.array([l["d_perp_per_fiber"] for l in latents_list], dtype=np.float32)
    d_gm = np.array([l["d_gm"] for l in latents_list], dtype=np.float32)

    np.savez(os.path.join(args.out_dir, "signals_clean.npz"),
             signals=signals, bvals=bvals.astype(np.float32),
             bvecs=bvecs.astype(np.float32))
    np.savez(os.path.join(args.out_dir, "latents.npz"),
             num_fibers=K, crossing_angle_deg=crossing,
             f_wm=f_wm, f_gm=f_gm, f_fw=f_fw,
             ndi=ndi, odi_mean=odi_mean, dirs=dirs,
             odi_per_fiber=odi_pf,
             d_par_per_fiber=dpar_pf,
             d_perp_per_fiber=dperp_pf,
             d_gm=d_gm,
             fa_true=fa_true, md_true=md_true, rd_true=rd_true)

    # Save bvals/bvecs as plain text alongside (for downstream tools)
    np.savetxt(os.path.join(args.out_dir, "bvals"), bvals[None, :], fmt="%.6g")
    # bvecs as 3xN
    bv_save = bvecs.T if bvecs.shape[0] == bvecs.size // 3 else bvecs
    np.savetxt(os.path.join(args.out_dir, "bvecs"), bv_save, fmt="%.8f")

    # Print summary stats
    print(f"\nVoxel-count summary (N={N}):")
    for k in range(4):
        n = int((K == k).sum())
        print(f"  K={k}: {n}  ({100*n/N:.1f}%)")
    print(f"FA range: [{fa_true.min():.3f}, {fa_true.max():.3f}] mean {fa_true.mean():.3f}")
    print(f"MD range: [{md_true.min()*1e3:.3f}, {md_true.max()*1e3:.3f}] e-3 mean {md_true.mean()*1e3:.3f}")
    print(f"f_wm   mean {f_wm.mean():.3f} sd {f_wm.std():.3f}")
    print(f"f_gm   mean {f_gm.mean():.3f} sd {f_gm.std():.3f}")
    print(f"f_fw   mean {f_fw.mean():.3f} sd {f_fw.std():.3f}")
    print(f"NDI    mean {np.nanmean(ndi):.3f} sd {np.nanstd(ndi):.3f}")
    print(f"ODI    mean {np.nanmean(odi_mean):.3f} sd {np.nanstd(odi_mean):.3f}")
    print(f"signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
