"""Five tissue-configuration substrates for FORCE, all from true MC physics:

  1. WM 1 fiber          (single coherent bundle)
  2. WM 2 fibers         (crossing)
  3. WM 3 fibers         (crossing)
  4. WM + GM             (anisotropic bundle + isotropic restricted soma)
  5. GM + CSF            (isotropic restricted soma + free water)

WM response = MC random walk in packed cylinders, reduced to the axially-
symmetric kernel S_b(theta) and composed over fiber directions (each a Watson
population). GM response = MC random walk in packed spheres (soma), isotropic
restricted S_gm(b). CSF = free water exp(-b*3e-3) (exact free diffusion).
Multi-fiber / multi-compartment voxels are volume-weighted sums of these MC
responses (intra-axonal restriction is exactly additive; the standard model).

Ground truth per voxel: tissue type, num_fibers, f_wm/f_gm/f_fw, NDI(=ICVF),
ODI, crossing angle.
"""
import argparse
import os
import time
import numpy as np
import np2_shim  # noqa: F401
from disimpy import gradients, simulations, substrates
from generate_dispersed_phantom import (simulate_kernel, watson_sample_z,
                                         odi_to_kappa, SHELLS)
from packed_spheres import build_gm_substrate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROTO = "/home/athshah/Phi/165840"
D0 = 2.0e-9
CSF_D = 3.0e-9
DELTA, delta = 30e-3, 10e-3


def rot_z_to(v):
    v = np.asarray(v, float); v = v / np.linalg.norm(v)
    z = np.array([0., 0., 1.]); c = float(z @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1., -1., -1.])
    k = np.cross(z, v); s = np.linalg.norm(k); k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def fiber_signal(interps, mu, ODI, bvals, bvecs, M, rng):
    """Watson(mu, ODI) dispersed WM kernel signal on the protocol (S0=1)."""
    if ODI <= 1e-4:
        nm = np.asarray(mu, float)[None, :]
    else:
        nm = watson_sample_z(odi_to_kappa(ODI), M, rng) @ rot_z_to(mu).T
    cosang = np.abs(bvecs @ nm.T)
    theta = np.arccos(np.clip(cosang, 0, 1))
    S = np.ones(bvals.shape)
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        shell = SHELLS[int(np.argmin([abs(b - sh) for sh in SHELLS]))]
        S[i] = interps[shell](theta[i]).mean()
    return S


def wm_signal(interps, dirs, fracs, ODI, bvals, bvecs, M, rng):
    """Volume-weighted multi-fiber WM signal."""
    S = np.zeros(bvals.shape)
    for mu, fr in zip(dirs, fracs):
        S += fr * fiber_signal(interps, mu, ODI, bvals, bvecs, M, rng)
    return S


def simulate_gm_kernel(f_soma, radius_um, n_walkers, n_t, n_side, seed):
    """Isotropic GM (soma) response S_gm(b) per shell."""
    s = build_gm_substrate(f_soma, radius_um * 1e-6, n_side=n_side)
    sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                          padding=s["padding"], init_pos="uniform",
                          n_sv=np.array([16, 16, 16]), quiet=True)
    # ~12 directions per shell, averaged (isotropic)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((12, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals = [0.]; bvecs = [[1, 0, 0]]
    for b in SHELLS:
        for d in dirs:
            bvals.append(b); bvecs.append(d)
    bvals = np.array(bvals); bvecs = np.array(bvecs)
    g, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
    t0 = time.time()
    sig = simulations.simulation(n_walkers, D0, g, dt, sub, seed=seed, quiet=True)
    S = sig / sig[0]
    gm = {}
    idx = 1
    for b in SHELLS:
        gm[b] = float(S[idx:idx + 12].mean()); idx += 12
    return gm, s["f_soma"], time.time() - t0


def gm_vec(gm, bvals):
    v = np.ones(bvals.shape)
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        shell = SHELLS[int(np.argmin([abs(b - sh) for sh in SHELLS]))]
        v[i] = gm[shell]
    return v


def ortho_dirs(n, rng):
    """n roughly-orthogonal unit directions for crossings."""
    if n == 1:
        return [np.array([0, 0, 1.])]
    if n == 2:
        return [np.array([0, 0, 1.]), np.array([1., 0, 0])]      # 90 deg
    return [np.array([0, 0, 1.]), np.array([1., 0, 0]), np.array([0, 1., 0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_icvf", type=float, default=0.6)
    ap.add_argument("--wm_radius_um", type=float, default=2.0)
    ap.add_argument("--gm_fsoma", type=float, default=0.40)
    ap.add_argument("--gm_radius_um", type=float, default=8.0)
    ap.add_argument("--n_walkers", type=int, default=int(2e5))
    ap.add_argument("--n_t", type=int, default=3000)
    ap.add_argument("--M", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    csf = np.exp(-bvals * 1e6 * CSF_D)
    rng = np.random.default_rng(args.seed)

    print("Simulating WM kernel (packed cylinders) ...")
    wm, wm_icvf, dur, _ = simulate_kernel(args.wm_icvf, args.wm_radius_um,
                                          args.n_walkers, args.n_t, 6, 31, 6,
                                          args.seed)
    print(f"  WM ICVF={wm_icvf:.3f}  {dur:.1f}s")
    print("Simulating GM kernel (packed spheres) ...")
    gm, f_soma, dur = simulate_gm_kernel(args.gm_fsoma, args.gm_radius_um,
                                         args.n_walkers, args.n_t, 4, args.seed + 1)
    gmv = gm_vec(gm, bvals)
    print(f"  GM f_soma={f_soma:.3f}  S_gm(b1k)={gm[1000.]:.3f}  {dur:.1f}s")

    ODI_WM = 0.10
    voxels = []   # (type, num_fibers, f_wm, f_gm, f_fw, ndi, odi, cross_deg, signal)

    def add(typ, nf, fwm, fgm, ffw, ndi, odi, cross, S):
        Sn = (fwm > 0) * 0  # placeholder
        voxels.append((typ, nf, fwm, fgm, ffw, ndi, odi, cross, S.astype(np.float32)))

    # 1. WM 1 fiber (two dispersions)
    for odi in (0.05, 0.15):
        S = wm_signal(wm, ortho_dirs(1, rng), [1.0], odi, bvals, bvecs, args.M, rng)
        add("WM_1fiber", 1, 1.0, 0.0, 0.0, wm_icvf, odi, np.nan, S)
    # 2. WM 2 fibers (90 and 45 deg)
    for cross in (90.0, 45.0):
        d1 = np.array([0, 0, 1.])
        d2 = np.array([np.sin(np.deg2rad(cross)), 0, np.cos(np.deg2rad(cross))])
        S = wm_signal(wm, [d1, d2], [0.5, 0.5], ODI_WM, bvals, bvecs, args.M, rng)
        add("WM_2fiber", 2, 1.0, 0.0, 0.0, wm_icvf, ODI_WM, cross, S)
    # 3. WM 3 fibers (orthogonal)
    S = wm_signal(wm, ortho_dirs(3, rng), [1/3.]*3, ODI_WM, bvals, bvecs, args.M, rng)
    add("WM_3fiber", 3, 1.0, 0.0, 0.0, wm_icvf, ODI_WM, 90.0, S)
    # 4. WM + GM (two splits), single fiber
    for fwm, fgm in ((0.6, 0.4), (0.4, 0.6)):
        Sw = wm_signal(wm, ortho_dirs(1, rng), [1.0], 0.10, bvals, bvecs, args.M, rng)
        S = fwm * Sw + fgm * gmv
        add("WM+GM", 1, fwm, fgm, 0.0, wm_icvf, 0.10, np.nan, S)
    # 5. GM + CSF (two splits)
    for fgm, ffw in ((0.5, 0.5), (0.3, 0.7)):
        S = fgm * gmv + ffw * csf
        add("GM+CSF", 0, 0.0, fgm, ffw, np.nan, np.nan, np.nan, S)

    signals = np.array([v[8] for v in voxels])
    keys = ("type", "num_fibers", "f_wm", "f_gm", "f_fw", "ndi", "odi", "cross_deg")
    gt = {k: np.array([v[i] for v in voxels],
                      dtype=object if k == "type" else np.float32)
          for i, k in enumerate(keys)}
    N = signals.shape[0]
    print(f"\n{N} tissue voxels:")
    for v in voxels:
        print(f"  {v[0]:<10} nf={v[1]} f_wm/gm/fw={v[2]:.1f}/{v[3]:.1f}/{v[4]:.1f} "
              f"cross={v[7]}")

    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    dmask = bvals <= 1050
    gtab = gradient_table(bvals[dmask], bvecs=bvecs[dmask], b0_threshold=50)
    dfit = TensorModel(gtab).fit(signals[:, dmask])
    gt["fa"] = np.asarray(dfit.fa, np.float32)
    gt["md"] = np.asarray(dfit.md, np.float32)

    np.savez(os.path.join(DATA, "signals_tissue.npz"), signals=signals,
             bvals=bvals.astype(np.float32), bvecs=bvecs.astype(np.float32))
    np.savez(os.path.join(DATA, "ground_truth_tissue.npz"),
             **{k: (v if k != "type" else v.astype(str)) for k, v in gt.items()})
    print(f"FA per voxel: {np.round(gt['fa'],2).tolist()}")
    print(f"saved -> {DATA}/signals_tissue.npz, ground_truth_tissue.npz")


if __name__ == "__main__":
    main()
