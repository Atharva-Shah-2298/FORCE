"""Model-free (fitting-free) diffusion-kurtosis ground truth (MK/AK/RK) for the
biological MC phantom, computed directly from Monte-Carlo molecular
displacements. See validate_kurtosis.py for the correctness checks.

Rationale (answers "how do you know the non-Gaussianity?"):
The apparent diffusional kurtosis DKI estimates is, in the narrow-pulse q-space
picture, the excess kurtosis of the molecular displacement projected along n:
        K(n) = <x^4>_c / <x^2>^2 ,     x = displacement . n
which for zero-mean displacements is  <x^4>/<x^2>^2 - 3.  We MEASURE it from the
actual random-walk displacements of the same disimpy substrates that generated
the phantom -- no DKI fit, no cumulant regression.

Per grid ICVF we record the coherent kernel's full raw 2nd/4th displacement
moment TENSORS (fiber || z). Per voxel we then reconstruct the true displacement
distribution exactly as the signal was built:
  * interpolate the kernel moment tensors to the voxel ICVF,
  * Watson(ODI) dispersion  = orientation-average of the rotated tensors,
  * free water (fraction FW) = isotropic Gaussian tensors added into the mixture
    (raw moments of a mixture are the weighted sum of component raw moments),
then evaluate K(n) analytically from the mixture tensors and take
  MK = mean over the sphere,  AK = K along mean fiber,  RK = mean over n _|_ fiber.

Run in the `skyline` env (needs disimpy + a CUDA GPU).
"""
import argparse
import os
import numpy as np
import np2_shim  # noqa: F401  (restores np.product etc. for disimpy)
from disimpy import substrates, simulations
from packed_cylinders import build_substrate_mesh
from generate_dispersed_phantom import watson_sample_z, odi_to_kappa

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
OUT = os.path.join(HERE, "bio_out")
D0 = 2.2e-9          # intrinsic diffusivity (matches bio_phantom)
CSF_D = 3.0e-9       # free-water diffusivity (matches bio_phantom)
DELTA, delta = 30e-3, 10e-3
T_EFF = DELTA - delta / 3.0     # effective (narrow-pulse) diffusion time
_I = np.eye(3)


def kernel_tensors(icvf, radius_um, n_walkers, n_t, n_side, seed):
    """Full raw 2nd/4th displacement moment tensors of a coherent packed-cylinder
    kernel (fiber || z). Returns realized ICVF, D2 (3,3), T4 (3,3,3,3)."""
    s = build_substrate_mesh(icvf, radius_um * 1e-6, n_side=n_side, n_theta=24)
    mk_sub = lambda ip: substrates.mesh(  # noqa: E731
        s["vertices"], s["faces"], periodic=True, padding=s["padding"],
        init_pos=ip, n_sv=np.array([20, 20, 20]), quiet=True)
    vs = mk_sub("uniform").voxel_size
    rng = np.random.default_rng(seed)
    init = rng.random((n_walkers, 3)) * vs
    _, fin = simulations.simulation(int(n_walkers), D0, np.zeros((1, n_t, 3)),
                                    float(T_EFF / n_t), mk_sub(init), seed=seed,
                                    final_pos=True, quiet=True)
    d = fin - init
    d -= d.mean(0, keepdims=True)
    D2 = np.einsum('ni,nj->ij', d, d) / d.shape[0]
    T4 = np.einsum('ni,nj,nk,nl->ijkl', d, d, d, d) / d.shape[0]
    return s["icvf"], D2, T4


def batch_rot_z_to(u):
    """Vectorised rotation mapping +z to each unit vector in u (W,3).
    Fibers are antipodally symmetric for diffusion, so flip to the z>=0 hemisphere
    to keep the rotation well conditioned (avoids the u=-z singularity)."""
    u = u * np.sign(u[:, 2:3] + 1e-12)
    c = u[:, 2]
    vx, vy = -u[:, 1], u[:, 0]            # v = z x u  (vz = 0)
    W = u.shape[0]
    Vx = np.zeros((W, 3, 3))
    Vx[:, 0, 1] = 0.0;  Vx[:, 0, 2] = vy
    Vx[:, 1, 2] = -vx
    Vx[:, 1, 0] = 0.0
    Vx[:, 2, 0] = -vy;  Vx[:, 2, 1] = vx
    return _I[None] + Vx + np.einsum('wij,wjk->wik', Vx, Vx) / (1 + c)[:, None, None]


def disperse_and_mix(D2, T4, fw, R):
    """Watson-average the kernel tensors over pre-sampled fiber rotations R
    (mean fiber = z), then add the free-water isotropic Gaussian compartment."""
    W = R.shape[0]
    D2d = np.einsum('wia,wjb,ab->ij', R, R, D2) / W
    T4d = np.einsum('wia,wjb,wkc,wld,abcd->ijkl', R, R, R, R, T4) / W
    sig2 = 2 * CSF_D * T_EFF
    T4fw = sig2**2 * (np.einsum('ij,kl->ijkl', _I, _I) +
                      np.einsum('ik,jl->ijkl', _I, _I) +
                      np.einsum('il,jk->ijkl', _I, _I))
    D2m = (1 - fw) * D2d + fw * sig2 * _I
    T4m = (1 - fw) * T4d + fw * T4fw
    return D2m, T4m


def K_of(D2, T4, dirs):
    m2 = np.einsum('di,dj,ij->d', dirs, dirs, D2)
    m4 = np.einsum('di,dj,dk,dl,ijkl->d', dirs, dirs, dirs, dirs, T4)
    return m4 / m2**2 - 3


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icvf_grid", type=float, nargs="+",
                    default=[0.40, 0.50, 0.58, 0.66, 0.72])
    ap.add_argument("--radius_um", type=float, default=1.0)
    ap.add_argument("--n_walkers", type=int, default=300000)
    ap.add_argument("--n_t", type=int, default=2000)
    ap.add_argument("--n_side", type=int, default=6)
    ap.add_argument("--nwatson", type=int, default=3000)
    ap.add_argument("--ndir_meas", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()

    print("Simulating MC displacement kernels across the ICVF grid ...")
    grid_icvf, grid_D2, grid_T4 = [], [], []
    for gi, icvf in enumerate(args.icvf_grid):
        ric, D2, T4 = kernel_tensors(icvf, args.radius_um, args.n_walkers,
                                     args.n_t, args.n_side, args.seed + gi)
        grid_icvf.append(ric); grid_D2.append(D2); grid_T4.append(T4)
        Dpar, Dperp = D2[2, 2] / (2 * T_EFF), 0.5 * (D2[0, 0] + D2[1, 1]) / (2 * T_EFF)
        Kperp = T4[0, 0, 0, 0] / D2[0, 0]**2 - 3
        print(f"  ICVF={ric:.3f}  perp-kurt={Kperp:.2f}  "
              f"D_par={Dpar*1e9:.3f} D_perp={Dperp*1e9:.3f} um2/ms")
    order = np.argsort(grid_icvf)
    grid_icvf = np.array(grid_icvf)[order]
    grid_D2 = np.array(grid_D2)[order]        # (G,3,3)
    grid_T4 = np.array(grid_T4)[order]        # (G,3,3,3,3)

    gt = np.load(os.path.join(DATA, "ground_truth_bio.npz"))
    ndi, odi, fw = gt["ndi"], gt["odi"], gt["fw"]
    N = ndi.size
    sphere = fibonacci_sphere(args.ndir_meas)
    ez = np.array([[0.0, 0.0, 1.0]])
    phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    perp = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])  # n _|_ z

    mk = np.empty(N); ak = np.empty(N); rk = np.empty(N)
    rng = np.random.default_rng(args.seed + 12345)
    for i in range(N):
        D2_i = np.stack([np.interp(ndi[i], grid_icvf, grid_D2[:, a, b])
                         for a in range(3) for b in range(3)]).reshape(3, 3)
        T4_i = np.array([np.interp(ndi[i], grid_icvf, grid_T4[:, a, b, c, d])
                         for a in range(3) for b in range(3)
                         for c in range(3) for d in range(3)]).reshape(3, 3, 3, 3)
        R = batch_rot_z_to(watson_sample_z(odi_to_kappa(float(odi[i])),
                                           args.nwatson, rng))
        D2m, T4m = disperse_and_mix(D2_i, T4_i, float(fw[i]), R)
        mk[i] = K_of(D2m, T4m, sphere).mean()
        ak[i] = K_of(D2m, T4m, ez)[0]
        rk[i] = K_of(D2m, T4m, perp).mean()

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "gt_kurtosis_mc.npz"),
             mk=mk.astype(np.float32), ak=ak.astype(np.float32),
             rk=rk.astype(np.float32), grid_icvf=grid_icvf.astype(np.float32),
             t_eff=np.float32(T_EFF))
    print(f"\nTrue MC kurtosis over {N} voxels (narrow-pulse, fitting-free):")
    for nm, arr in [("MK", mk), ("AK", ak), ("RK", rk)]:
        print(f"  {nm} {arr.mean():.3f} [{arr.min():.3f}, {arr.max():.3f}]")
    print("saved bio_out/gt_kurtosis_mc.npz")


if __name__ == "__main__":
    main()
