"""Biologically-ranged Monte Carlo phantom for FORCE vs AMICO-NODDI validation.

N single-fiber WM voxels sampled across HEALTHY adult-brain ranges:
    NDI (intra-neurite volume fraction / ICVF) ~ U[0.40, 0.72]   (NODDI WM)
    ODI (Watson orientation dispersion)        ~ U[0.03, 0.30]   (within dict)
    FW  (isotropic free water / ISOVF)         ~ U[0.00, 0.20]   (WM, partial CSF)
    fiber orientation                          random on the sphere
    axon radius 1.0 um (biological; ~stick at clinical b)

Signal = true MC random-walk physics: a coherent packed-cylinder kernel
S_b(theta) (restricted intra + hindered extra) is MC-simulated at a grid of
ICVF, linearly interpolated to each voxel's ICVF, Watson-convolved to its ODI
about its fiber direction, then mixed with free water. Saved clean + Rician
noise at SNR {50,20,10} as NIfTI (S0=1000) with a scheme file, for both FORCE
(skyline) and AMICO-NODDI (base env).
"""
import argparse
import os
import time
import numpy as np
import np2_shim  # noqa: F401
from disimpy import gradients, simulations, substrates
from packed_cylinders import build_substrate_mesh
from generate_dispersed_phantom import (watson_sample_z, odi_to_kappa,
                                        kernel_directions, SHELLS)


def rot_z_to(v):
    """Rotation matrix taking +z onto unit vector ``v`` (Rodrigues)."""
    v = np.asarray(v, float); v = v / np.linalg.norm(v)
    z = np.array([0., 0., 1.]); c = float(z @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1., -1., -1.])
    k = np.cross(z, v); s = np.linalg.norm(k); k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
PROTO = os.environ.get("FORCE_PROTO", "hcp_subject")  # HCP-like subject dir with bvals/bvecs
D0 = 2.2e-9
CSF_D = 3.0e-9
DELTA, delta = 30e-3, 10e-3
S0 = 1000.0


def sim_kernel_grid(icvf, radius_um, n_walkers, n_t, n_side, n_theta, n_phi, seed):
    """MC kernel reduced to axially-symmetric S[shell, theta_grid]."""
    s = build_substrate_mesh(icvf, radius_um * 1e-6, n_side=n_side, n_theta=24)
    sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                          padding=s["padding"], init_pos="uniform",
                          n_sv=np.array([20, 20, 20]), quiet=True)
    dirs, tt, thetas, nphi = kernel_directions(n_theta, n_phi)
    bvals = [0.0]; bvecs = [[1.0, 0, 0]]
    for b in SHELLS:
        for d in dirs:
            bvals.append(b); bvecs.append(d)
    bvals = np.array(bvals); bvecs = np.array(bvecs)
    grad, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
    sig = simulations.simulation(n_walkers, D0, grad, dt, sub, seed=seed, quiet=True)
    S = sig / sig[0]
    npts = len(dirs)
    Ksh = np.zeros((len(SHELLS), len(thetas)))
    idx = 1
    for si in range(len(SHELLS)):
        Ksh[si] = S[idx:idx + npts].reshape(len(thetas), nphi).mean(axis=1)
        idx += npts
    return Ksh, thetas, s["icvf"]


def voxel_signal(Ksh_icvf, thetas, mu, ODI, bvals, bvecs, M, rng):
    """Watson(mu,ODI)-convolve interpolated kernel onto the protocol."""
    if ODI <= 1e-4:
        nm = mu[None, :]
    else:
        nm = watson_sample_z(odi_to_kappa(ODI), M, rng) @ rot_z_to(mu).T
    cosang = np.abs(bvecs @ nm.T)
    theta = np.arccos(np.clip(cosang, 0, 1))
    S = np.ones(bvals.shape)
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        sh = int(np.argmin([abs(b - x) for x in SHELLS]))
        S[i] = np.interp(theta[i], thetas, Ksh_icvf[sh]).mean()
    return S


def rand_unit(rng):
    z = rng.uniform(-1, 1); phi = rng.uniform(0, 2 * np.pi)
    r = np.sqrt(max(0, 1 - z * z))
    return np.array([r * np.cos(phi), r * np.sin(phi), z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--icvf_grid", type=float, nargs="+",
                    default=[0.40, 0.50, 0.58, 0.66, 0.72])
    ap.add_argument("--radius_um", type=float, default=1.0)
    ap.add_argument("--n_walkers", type=int, default=int(1e5))
    ap.add_argument("--n_t", type=int, default=4000)
    ap.add_argument("--n_side", type=int, default=6)
    ap.add_argument("--n_theta", type=int, default=31)
    ap.add_argument("--n_phi", type=int, default=6)
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    import nibabel as nib

    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    csf = np.exp(-bvals * 1e6 * CSF_D)

    print("Simulating MC kernels across ICVF grid ...")
    grid_icvf, kernels = [], []
    for gi, icvf in enumerate(args.icvf_grid):
        t0 = time.time()
        Ksh, thetas, icvf_geom = sim_kernel_grid(
            icvf, args.radius_um, args.n_walkers, args.n_t, args.n_side,
            args.n_theta, args.n_phi, args.seed + gi)
        grid_icvf.append(icvf_geom); kernels.append(Ksh)
        print(f"  ICVF={icvf_geom:.3f}  {time.time()-t0:.0f}s")
    grid_icvf = np.array(grid_icvf)
    kernels = np.array(kernels)        # (G, shell, theta)
    lo, hi = grid_icvf.min(), grid_icvf.max()

    rng = np.random.default_rng(args.seed + 999)
    N = args.N
    signals = np.empty((N, bvals.size), np.float32)
    gt = {k: np.empty(N, np.float32) for k in ("ndi", "odi", "fw", "voxel_nd")}
    for i in range(N):
        icvf = rng.uniform(lo, hi)
        # linear interp of kernel over ICVF grid (per shell, per theta)
        Ksh_i = np.empty_like(kernels[0])
        for sh in range(kernels.shape[1]):
            for tj in range(kernels.shape[2]):
                Ksh_i[sh, tj] = np.interp(icvf, grid_icvf, kernels[:, sh, tj])
        odi = rng.uniform(0.03, 0.30)
        fw = rng.uniform(0.0, 0.20)
        mu = rand_unit(rng)
        S_wm = voxel_signal(Ksh_i, thetas, mu, odi, bvals, bvecs, args.M, rng)
        S = (1 - fw) * S_wm + fw * csf
        signals[i] = S
        gt["ndi"][i] = icvf; gt["odi"][i] = odi; gt["fw"][i] = fw
        gt["voxel_nd"][i] = (1 - fw) * icvf
    print(f"\n{N} biological voxels: NDI[{gt['ndi'].min():.2f},{gt['ndi'].max():.2f}] "
          f"ODI[{gt['odi'].min():.2f},{gt['odi'].max():.2f}] "
          f"FW[{gt['fw'].min():.2f},{gt['fw'].max():.2f}]")

    # save clean signals + GT
    np.savez(os.path.join(DATA, "signals_bio_clean.npz"), signals=signals,
             bvals=bvals.astype(np.float32), bvecs=bvecs.astype(np.float32))
    np.savez(os.path.join(DATA, "ground_truth_bio.npz"), **gt)

    # NIfTI (S0=1000) + Rician noise, + mask + scheme
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones((N, 1, 1), np.uint8), aff),
             os.path.join(DATA, "mask.nii.gz"))
    nrng = np.random.default_rng(args.seed + 4242)
    for snr in ("clean", 50, 20, 10):
        if snr == "clean":
            noisy = S0 * signals
        else:
            sig = S0 / snr
            re = S0 * signals + nrng.normal(0, sig, signals.shape)
            im = nrng.normal(0, sig, signals.shape)
            noisy = np.sqrt(re ** 2 + im ** 2)
        vol = noisy.reshape(N, 1, 1, bvals.size).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, aff), os.path.join(DATA, f"dwi_{snr}.nii.gz"))
    import shutil
    shutil.copy(os.path.join(PROTO, "bvals.scheme"), os.path.join(DATA, "dwi.scheme"))
    np.savetxt(os.path.join(DATA, "bvals"), bvals[None, :], fmt="%.6g")
    np.savetxt(os.path.join(DATA, "bvecs"), bvecs.T if bvecs.shape[0] != 3 else bvecs,
               fmt="%.8f")
    print(f"saved phantom -> {DATA}")


if __name__ == "__main__":
    main()
