"""Correctness checks for the model-free MC kurtosis ground truth.

Checks, on one MC substrate:
  (1) in-plane isotropy of the square-lattice kernel (is <dx^4> ~ 3<dx^2 dy^2>?),
  (2) coherent-kernel MK/AK/RK from the full 2nd/4th displacement moment tensors
      vs a DIRECT, assumption-free K(n)=<x^4>/<x^2>^2-3 on the raw displacements,
  (3) reduced 5-scalar (axial-symmetry) formula vs the full-tensor result,
  (4) free-water limit MK->0.
"""
import numpy as np
import np2_shim  # noqa: F401
from disimpy import substrates, simulations
from packed_cylinders import build_substrate_mesh
from generate_tissue_substrates import rot_z_to
from generate_dispersed_phantom import watson_sample_z, odi_to_kappa

D0 = 2.2e-9; CSF_D = 3.0e-9
DELTA, delta = 30e-3, 10e-3; T_EFF = DELTA - delta / 3.0


def simulate_disp(icvf, n=200000, n_t=2000, n_side=6, seed=1):
    s = build_substrate_mesh(icvf, 1.0e-6, n_side=n_side, n_theta=24)
    def mk(ip):
        return substrates.mesh(s["vertices"], s["faces"], periodic=True,
                               padding=s["padding"], init_pos=ip,
                               n_sv=np.array([20, 20, 20]), quiet=True)
    vs = mk("uniform").voxel_size
    rng = np.random.default_rng(seed)
    init = rng.random((n, 3)) * vs
    grad = np.zeros((1, n_t, 3))
    _, fin = simulations.simulation(int(n), D0, grad, float(T_EFF / n_t),
                                    mk(init), seed=seed, final_pos=True, quiet=True)
    d = fin - init
    return d - d.mean(0, keepdims=True)


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); gold = np.pi * (1 + 5 ** 0.5)
    return np.column_stack([np.sin(phi)*np.cos(gold*i), np.sin(phi)*np.sin(gold*i),
                            np.cos(phi)])


def K_direct(d, dirs):
    """K(n) straight from displacement sample, no model assumptions."""
    x = d @ dirs.T                    # (N, Ndir)
    m2 = (x**2).mean(0); m4 = (x**4).mean(0)
    return m4 / m2**2 - 3


def tensors(d):
    D2 = np.einsum('ni,nj->ij', d, d) / d.shape[0]
    T4 = np.einsum('ni,nj,nk,nl->ijkl', d, d, d, d) / d.shape[0]
    return D2, T4


def K_from_tensors(D2, T4, dirs):
    m2 = np.einsum('di,dj,ij->d', dirs, dirs, D2)
    m4 = np.einsum('di,dj,dk,dl,ijkl->d', dirs, dirs, dirs, dirs, T4)
    return m4 / m2**2 - 3


def disperse_and_mix(D2, T4, odi, fw, nwatson=4000, seed=0):
    """Watson(odi) orientation-average the kernel tensors (mean fiber = z), then
    add a free-water isotropic Gaussian compartment (fraction fw)."""
    rng = np.random.default_rng(seed)
    u = watson_sample_z(odi_to_kappa(odi), nwatson, rng)     # (W,3)
    R = np.stack([rot_z_to(ui) for ui in u])                 # (W,3,3)
    D2d = np.einsum('wia,wjb,ab->ij', R, R, D2) / nwatson
    T4d = np.einsum('wia,wjb,wkc,wld,abcd->ijkl', R, R, R, R, T4) / nwatson
    sig2 = 2 * CSF_D * T_EFF
    I = np.eye(3)
    D2fw = sig2 * I
    T4fw = sig2**2 * (np.einsum('ij,kl->ijkl', I, I) +
                      np.einsum('ik,jl->ijkl', I, I) +
                      np.einsum('il,jk->ijkl', I, I))
    D2m = (1-fw)*D2d + fw*D2fw
    T4m = (1-fw)*T4d + fw*T4fw
    return D2m, T4m


if __name__ == "__main__":
    icvf = 0.58
    d = simulate_disp(icvf)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    print(f"=== substrate icvf~{icvf} (N={d.shape[0]}) ===")
    print("(1) in-plane isotropy of 4th moments:")
    print(f"    <dx^4>={np.mean(dx**4):.3e}  <dy^4>={np.mean(dy**4):.3e}  "
          f"3<dx^2 dy^2>={3*np.mean(dx**2*dy**2):.3e}")
    iso = np.mean(dx**4) / (3*np.mean(dx**2*dy**2))
    print(f"    ratio <dx^4>/(3<dx^2dy^2>) = {iso:.3f}  (1.0 => isotropic in-plane)")

    dirs = fib(2000)
    D2, T4 = tensors(d)

    # (2) coherent (odi tiny, fw=0): full-tensor vs direct-sample
    Kt = K_from_tensors(D2, T4, dirs)
    Kd = K_direct(d, dirs)
    print("\n(2) coherent kernel, full-tensor vs direct-sample K(n):")
    print(f"    MK  tensor={Kt.mean():.4f}  direct={Kd.mean():.4f}")
    zc = np.abs(dirs[:, 2])
    ax = zc > 0.999; rad = zc < 0.03
    print(f"    AK  tensor={Kt[ax].mean():.4f}  direct={Kd[ax].mean():.4f}  (n~z)")
    print(f"    RK  tensor={Kt[rad].mean():.4f} direct={Kd[rad].mean():.4f} (n_|_z)")

    # AK/RK directly along exact axes
    ez = np.array([[0, 0, 1.0]]); exy = fib(2000); exy = exy[np.abs(exy[:,2])<0.03]
    print(f"    AK(exact n=z) direct = {K_direct(d, ez)[0]:.4f}")
    print(f"    RK(exact n_|_) direct = {K_direct(d, exy).mean():.4f}")

    # (3) reduced 5-scalar axial-symmetry formula
    M2p = 0.5*(np.mean(dx**2)+np.mean(dy**2)); M2z = np.mean(dz**2)
    M4perp = 0.5*(np.mean(dx**4)+np.mean(dy**4)); M4z = np.mean(dz**4)
    M4perpz = 0.5*(np.mean(dx**2*dz**2)+np.mean(dy**2*dz**2))
    c = dirs[:, 2]; c2 = c*c; s2 = 1-c2       # coherent: fiber=z so cos=n_z
    m2 = s2*M2p + c2*M2z
    m4 = s2**2*M4perp + 6*s2*c2*M4perpz + c2**2*M4z
    Kr = m4/m2**2 - 3
    print("\n(3) reduced 5-scalar formula (coherent):")
    print(f"    MK reduced={Kr.mean():.4f}  vs full-tensor={Kt.mean():.4f}")

    # (4) dispersion + free-water via full tensors; free-water limit
    for odi, fw in [(0.001, 0.0), (0.15, 0.0), (0.15, 0.2), (0.0, 1.0)]:
        D2m, T4m = disperse_and_mix(D2, T4, max(odi, 1e-4), fw)
        Km = K_from_tensors(D2m, T4m, dirs)
        print(f"\n(4) odi={odi} fw={fw}:  MK={Km.mean():.4f}  "
              f"AK={Km[ax].mean():.4f}  RK={Km[rad].mean():.4f}")
