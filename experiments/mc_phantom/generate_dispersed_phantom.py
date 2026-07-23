"""Build a Monte Carlo phantom WITH orientation dispersion, to test FORCE's
dispersion (ODI) recovery across a range of true ODI.

Method (rigorous kernel convolution):
  1. MC-simulate the coherent packed-cylinder substrate on a dense direction
     scheme and reduce it to an axially-symmetric kernel S_b(theta) = the true
     restricted+hindered response as a function of angle to the fiber axis.
     This kernel carries the full Monte Carlo diffusion physics.
  2. For a target Watson dispersion ODI (kappa = 1/tan(pi*ODI/2)), draw M fiber
     orientations ~ Watson(mu=z, kappa) and orientationally average the kernel:
         S_disp(g_i) = (1/M) sum_m S_{b_i}( angle(g_i, n_m) )
     This is the physically exact construction of a dispersed bundle from
     identical, locally-straight axon segments — the standard dispersion model
     (also what FORCE/NODDI assume). The intra-axonal restriction and
     extra-axonal hindrance come entirely from the MC kernel; only the
     orientation averaging is analytic.

Ground truth: odi_wm = ODI (within-WM Watson dispersion), ICVF (= NDI), fw.
Limitation: the extra-axonal hindrance geometry is taken from the coherent
packing and rotated; a fully dispersed-cylinder mesh would also change the
extra-axonal tortuosity. This is the same approximation FORCE itself makes,
so the comparison is fair.
"""
import argparse
import os
import time
import numpy as np
import np2_shim  # noqa: F401
from disimpy import gradients, simulations, substrates
from packed_cylinders import build_substrate_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROTO = "/home/athshah/Phi/165840"
D0 = 2.0e-9
CSF_D = 3.0e-9
DELTA, delta = 30e-3, 10e-3
SHELLS = [1000.0, 2000.0, 3000.0]


def odi_to_kappa(odi):
    return 1.0 / np.tan(np.pi * np.asarray(odi) / 2.0)


def watson_sample_z(kappa, M, rng):
    """M unit vectors ~ Watson(mu=z, kappa): t=cos(theta) density ∝ exp(k t^2)."""
    out = np.empty((M, 3))
    n = 0
    while n < M:
        batch = 2 * (M - n) + 16
        t = rng.uniform(-1.0, 1.0, batch)
        u = rng.uniform(0.0, 1.0, batch)
        keep = u < np.exp(kappa * (t * t - 1.0))      # accept ∝ exp(k(t^2-1))
        tk = t[keep]
        take = min(tk.size, M - n)
        phi = rng.uniform(0, 2 * np.pi, take)
        st = np.sqrt(np.maximum(0.0, 1.0 - tk[:take] ** 2))
        out[n:n + take, 0] = st * np.cos(phi)
        out[n:n + take, 1] = st * np.sin(phi)
        out[n:n + take, 2] = tk[:take]
        n += take
    return out


def kernel_directions(n_theta=31, n_phi=6):
    """Directions sampling theta in [0,90] x phi, per shell (for axial-sym kernel)."""
    thetas = np.linspace(0.0, np.pi / 2, n_theta)
    # offset azimuths so no direction lands on +-x/+-y axes (disimpy.pgse rejects
    # the exact antipode of its internal base axis as an improper rotation)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False) + 0.17
    dirs, tt = [], []
    for th in thetas:
        for ph in phis:
            dirs.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            tt.append(th)
    return np.array(dirs), np.array(tt), thetas, n_phi


def simulate_kernel(icvf, radius_um, n_walkers, n_t, n_side, n_theta, n_phi, seed):
    """Return interpolators S_b(theta) for each shell, plus realized ICVF."""
    s = build_substrate_mesh(icvf, radius_um * 1e-6, n_side=n_side, n_theta=24)
    sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                          padding=s["padding"], init_pos="uniform",
                          n_sv=np.array([20, 20, 20]), quiet=True)
    dirs, tt, thetas, nphi = kernel_directions(n_theta, n_phi)
    # gradient scheme: b0 + each shell x all dirs
    bvals = [0.0]
    bvecs = [[1.0, 0, 0]]
    for b in SHELLS:
        for d in dirs:
            bvals.append(b)
            bvecs.append(d)
    bvals = np.array(bvals)
    bvecs = np.array(bvecs)
    grad, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
    t0 = time.time()
    sig = simulations.simulation(n_walkers, D0, grad, dt, sub, seed=seed, quiet=True)
    dur = time.time() - t0
    S = sig / sig[0]
    # reduce to axially-symmetric kernel S_b(theta): average over phi
    from scipy.interpolate import PchipInterpolator
    interps = {}
    idx = 1
    npts = len(dirs)
    for b in SHELLS:
        block = S[idx:idx + npts].reshape(len(thetas), nphi)
        s_theta = block.mean(axis=1)               # average azimuth -> axial sym
        interps[b] = PchipInterpolator(thetas, s_theta, extrapolate=True)
        idx += npts
    return interps, s["icvf"], dur, dt


def convolve(interps, ODI, bvals, bvecs, M, rng):
    """Watson-convolve kernel -> dispersed signal on the measured protocol."""
    if ODI <= 1e-4:
        nm = np.array([[0.0, 0.0, 1.0]])
    else:
        nm = watson_sample_z(odi_to_kappa(ODI), M, rng)        # (M,3)
    S = np.ones(bvals.shape, dtype=np.float64)
    cosang = np.abs(bvecs @ nm.T)                              # (T, M)
    theta = np.arccos(np.clip(cosang, 0.0, 1.0))              # angle to each fiber
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        shell = SHELLS[int(np.argmin([abs(b - sh) for sh in SHELLS]))]
        S[i] = interps[shell](theta[i]).mean()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icvfs", type=float, nargs="+", default=[0.4, 0.6])
    ap.add_argument("--radius_um", type=float, default=2.0)
    ap.add_argument("--odis", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    ap.add_argument("--fw", type=float, default=0.0)
    ap.add_argument("--n_walkers", type=int, default=int(2e5))
    ap.add_argument("--n_t", type=int, default=3000)
    ap.add_argument("--n_side", type=int, default=6)
    ap.add_argument("--n_theta", type=int, default=31)
    ap.add_argument("--n_phi", type=int, default=6)
    ap.add_argument("--M", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    csf_sig = np.exp(-bvals * 1e6 * CSF_D)
    rng = np.random.default_rng(args.seed)

    signals, gt = [], {k: [] for k in ("icvf", "radius_um", "odi_wm", "f_fw",
                                       "f_wm", "nd", "fw")}
    for si, icvf in enumerate(args.icvfs):
        interps, icvf_geom, dur, dt = simulate_kernel(
            icvf, args.radius_um, args.n_walkers, args.n_t, args.n_side,
            args.n_theta, args.n_phi, args.seed + si)
        print(f"kernel ICVF={icvf_geom:.3f} r={args.radius_um}um  {dur:.1f}s "
              f"dt={dt*1e6:.1f}us  S1k(theta=90)={interps[1000.](np.pi/2):.3f} "
              f"S1k(theta=0)={interps[1000.](0.0):.3f}")
        for ODI in args.odis:
            S = convolve(interps, ODI, bvals, bvecs, args.M, rng)
            f_wm = 1.0 - args.fw
            Sv = f_wm * S + args.fw * csf_sig
            signals.append(Sv.astype(np.float32))
            gt["icvf"].append(icvf_geom)
            gt["radius_um"].append(args.radius_um)
            gt["odi_wm"].append(ODI)
            gt["f_fw"].append(args.fw)
            gt["f_wm"].append(f_wm)
            gt["nd"].append(f_wm * icvf_geom)
            gt["fw"].append(args.fw)
    signals = np.array(signals)
    for k in gt:
        gt[k] = np.array(gt[k], np.float32)
    N = signals.shape[0]
    print(f"\n{N} dispersed voxels ({len(args.icvfs)} ICVF x {len(args.odis)} ODI)")

    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    dmask = bvals <= 1050
    gtab = gradient_table(bvals[dmask], bvecs=bvecs[dmask], b0_threshold=50)
    dfit = TensorModel(gtab).fit(signals[:, dmask])
    gt["fa"] = np.asarray(dfit.fa, np.float32)
    gt["md"] = np.asarray(dfit.md, np.float32)
    gt["rd"] = np.asarray(dfit.rd, np.float32)

    np.savez(os.path.join(DATA, "signals_disp.npz"), signals=signals,
             bvals=bvals.astype(np.float32), bvecs=bvecs.astype(np.float32))
    np.savez(os.path.join(DATA, "ground_truth_disp.npz"), **gt)
    print(f"true ODI sweep: {sorted(set(gt['odi_wm'].round(3).tolist()))}")
    print(f"FA range {gt['fa'].min():.3f}..{gt['fa'].max():.3f}  "
          f"(should DECREASE as ODI rises)")
    print(f"saved -> {DATA}/signals_disp.npz, ground_truth_disp.npz")


if __name__ == "__main__":
    main()
