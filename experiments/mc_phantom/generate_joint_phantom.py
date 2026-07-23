"""Joint Monte Carlo phantom: NDI (ICVF), ODI (dispersion) and FW all vary
together, so FORCE recovery of every microstructure metric is tested on the
SAME voxel set (and their interactions are exposed).

Each voxel:
    S = (1 - FW) * Watson_conv( MC_kernel[ICVF], ODI ) + FW * exp(-b * D_csf)
where MC_kernel[ICVF] is the axially-symmetric restricted+hindered response
from a GPU random walk in a packed-cylinder substrate (true diffusion physics),
and the Watson convolution applies orientation dispersion ODI (exact dispersion
model). Full factorial over ICVF x ODI x FW.

Ground truth (all simultaneously):
    ndi_within_wm = ICVF ;  nd_voxel = (1-FW)*ICVF ;  odi_wm = ODI ;  fw = FW
    FA/MD/RD = DTI(b<=1000) on the clean signal.
"""
import argparse
import os
import numpy as np
import np2_shim  # noqa: F401
from generate_dispersed_phantom import simulate_kernel, convolve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROTO = os.environ.get("FORCE_PROTO", "hcp_subject")  # HCP-like subject dir with bvals/bvecs
CSF_D = 3.0e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icvfs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--odis", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--fws", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    ap.add_argument("--radius_um", type=float, default=2.0)
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

    # one MC kernel per ICVF (true physics); reused across ODI/FW
    kernels = {}
    for si, icvf in enumerate(args.icvfs):
        interps, icvf_geom, dur, dt = simulate_kernel(
            icvf, args.radius_um, args.n_walkers, args.n_t, args.n_side,
            args.n_theta, args.n_phi, args.seed + si)
        kernels[icvf] = (interps, icvf_geom)
        print(f"kernel ICVF={icvf_geom:.3f}  {dur:.1f}s")

    signals = []
    gt = {k: [] for k in ("icvf", "odi_wm", "f_fw", "f_wm", "ndi_within_wm",
                          "nd", "fw", "radius_um")}
    for icvf in args.icvfs:
        interps, icvf_geom = kernels[icvf]
        for ODI in args.odis:
            S_disp = convolve(interps, ODI, bvals, bvecs, args.M, rng)
            for FW in args.fws:
                f_wm = 1.0 - FW
                Sv = f_wm * S_disp + FW * csf_sig
                signals.append(Sv.astype(np.float32))
                gt["icvf"].append(icvf_geom)
                gt["odi_wm"].append(ODI)
                gt["f_fw"].append(FW)
                gt["f_wm"].append(f_wm)
                gt["ndi_within_wm"].append(icvf_geom)
                gt["nd"].append(f_wm * icvf_geom)
                gt["fw"].append(FW)
                gt["radius_um"].append(args.radius_um)
    signals = np.array(signals)
    for k in gt:
        gt[k] = np.array(gt[k], np.float32)
    N = signals.shape[0]
    print(f"\n{N} joint voxels = {len(args.icvfs)} ICVF x {len(args.odis)} ODI "
          f"x {len(args.fws)} FW")

    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    dmask = bvals <= 1050
    gtab = gradient_table(bvals[dmask], bvecs=bvecs[dmask], b0_threshold=50)
    dfit = TensorModel(gtab).fit(signals[:, dmask])
    gt["fa"] = np.asarray(dfit.fa, np.float32)
    gt["md"] = np.asarray(dfit.md, np.float32)
    gt["rd"] = np.asarray(dfit.rd, np.float32)

    np.savez(os.path.join(DATA, "signals_joint.npz"), signals=signals,
             bvals=bvals.astype(np.float32), bvecs=bvecs.astype(np.float32))
    np.savez(os.path.join(DATA, "ground_truth_joint.npz"), **gt)
    print(f"ICVF {sorted(set(gt['ndi_within_wm'].round(2).tolist()))}  "
          f"ODI {sorted(set(gt['odi_wm'].round(2).tolist()))}  "
          f"FW {sorted(set(gt['fw'].round(2).tolist()))}")
    print(f"FA {gt['fa'].min():.3f}..{gt['fa'].max():.3f}  "
          f"MDx1e3 {gt['md'].min()*1e3:.3f}..{gt['md'].max()*1e3:.3f}")
    print(f"saved -> {DATA}/signals_joint.npz, ground_truth_joint.npz")


if __name__ == "__main__":
    main()
