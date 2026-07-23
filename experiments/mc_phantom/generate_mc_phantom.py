"""Generate a Monte Carlo diffusion phantom for FORCE microstructure-recovery
testing.

Each substrate is a periodic packing of impermeable parallel cylinders
(axons along +z). A GPU random walk (disimpy) produces the diffusion signal
on subject 165840's *actual* FORCE acquisition (288 vols, shells 0/1/2/3k),
with the fiber coherent along z. The signal carries TRUE restricted
(intra-axonal) and hindered (extra-axonal) diffusion physics — it is NOT
produced by any analytic compartment model, so recovering microstructure
from it is an honest test.

Ground truth (in FORCE's definitions; see exp9 analyze.py):
    ndi_within_wm = ICVF  (intra-axonal volume fraction, set by geometry)
    f_wm = 1 - f_fw,  f_gm = 0,  f_fw = free-water fraction (added in signal
           domain as a CSF compartment, D=3.0e-3 mm^2/s)
    nd_voxel  = f_wm * ICVF
    odi_voxel = 1 - f_wm * (1 - ODI_pf);  coherent bundle => ODI_pf≈0 =>
                odi_voxel ≈ f_fw
FA/MD/RD truth from DTI (b<=1000) on the clean (noise-free) signal, mirroring
generate_test_voxels.py.

Grid: ICVF x axon-radius substrates, each expanded to free-water levels.
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
D0 = 2.0e-9        # intrinsic diffusivity m^2/s (=2.0e-3 mm^2/s, within dict d_par range)
CSF_D = 3.0e-9     # free-water diffusivity m^2/s
DELTA, delta = 30e-3, 10e-3


def load_protocol():
    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    return bvals, bvecs


def simulate_substrate(icvf, radius_um, bvals, bvecs, n_walkers, n_t,
                       n_side, n_theta, seed):
    s = build_substrate_mesh(icvf, radius_um * 1e-6, n_side=n_side, n_theta=n_theta)
    sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                          padding=s["padding"], init_pos="uniform",
                          n_sv=np.array([20, 20, 20]), quiet=True)
    grad, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
    t0 = time.time()
    sig = simulations.simulation(n_walkers, D0, grad, dt, sub, seed=seed, quiet=True)
    dur = time.time() - t0
    b0 = sig[bvals <= 50].mean()
    return (sig / b0).astype(np.float64), s["icvf"], dur, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icvfs", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--radii_um", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--fw_levels", type=float, nargs="+", default=[0.0, 0.2])
    ap.add_argument("--n_walkers", type=int, default=int(1e5))
    ap.add_argument("--n_t", type=int, default=3000)
    ap.add_argument("--n_side", type=int, default=6)
    ap.add_argument("--n_theta", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    bvals, bvecs = load_protocol()
    print(f"protocol: {bvals.size} vols, shells "
          f"{np.unique(np.round(bvals/100)*100).astype(int)}")
    csf_sig = np.exp(-bvals * 1e6 * CSF_D)            # free-water signal, S0=1

    sub_sigs = []      # raw cylinder signal per substrate
    sub_meta = []      # (icvf_geom, radius_um)
    sidx = 0
    for r in args.radii_um:
        for icvf in args.icvfs:
            S, icvf_geom, dur, dt = simulate_substrate(
                icvf, r, bvals, bvecs, args.n_walkers, args.n_t,
                args.n_side, args.n_theta, args.seed + sidx)
            sub_sigs.append(S)
            sub_meta.append((icvf_geom, r))
            print(f"  [{sidx:2d}] ICVF={icvf_geom:.3f} r={r}um  {dur:.1f}s  "
                  f"dt={dt*1e6:.1f}us  S(b1k)perp~{S[bvals>900].min():.3f}")
            sidx += 1

    # Expand substrates x free-water levels into voxels
    signals = []
    gt = {k: [] for k in ("icvf", "radius_um", "f_fw", "f_wm", "ndi_within_wm",
                          "nd", "odi", "fw")}
    for (icvf_geom, r), S in zip(sub_meta, sub_sigs):
        for f_fw in args.fw_levels:
            f_wm = 1.0 - f_fw
            Sv = f_wm * S + f_fw * csf_sig
            signals.append(Sv.astype(np.float32))
            gt["icvf"].append(icvf_geom)
            gt["radius_um"].append(r)
            gt["f_fw"].append(f_fw)
            gt["f_wm"].append(f_wm)
            gt["ndi_within_wm"].append(icvf_geom)
            gt["nd"].append(f_wm * icvf_geom)
            gt["odi"].append(1.0 - f_wm * (1.0 - 0.0))   # coherent: ODI_pf≈0
            gt["fw"].append(f_fw)
    signals = np.array(signals)                            # (Nvox, 288)
    for k in gt:
        gt[k] = np.array(gt[k], dtype=np.float32)
    N = signals.shape[0]
    print(f"\n{N} phantom voxels "
          f"({len(sub_meta)} substrates x {len(args.fw_levels)} FW levels)")

    # FA/MD/RD truth via DTI on clean signal (b<=1000)
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    dti_mask = bvals <= 1050
    gt_dti = gradient_table(bvals[dti_mask], bvecs=bvecs[dti_mask], b0_threshold=50)
    dfit = TensorModel(gt_dti).fit(signals[:, dti_mask])
    gt["fa"] = np.asarray(dfit.fa, np.float32)
    gt["md"] = np.asarray(dfit.md, np.float32)
    gt["rd"] = np.asarray(dfit.rd, np.float32)

    np.savez(os.path.join(DATA, "signals_mc.npz"),
             signals=signals, bvals=bvals.astype(np.float32),
             bvecs=bvecs.astype(np.float32),
             sub_signals=np.array(sub_sigs, np.float32),
             sub_icvf=np.array([m[0] for m in sub_meta], np.float32),
             sub_radius=np.array([m[1] for m in sub_meta], np.float32))
    np.savez(os.path.join(DATA, "ground_truth.npz"), **gt)
    np.savetxt(os.path.join(DATA, "bvals"), bvals[None, :], fmt="%.6g")
    np.savetxt(os.path.join(DATA, "bvecs"),
               (bvecs.T if bvecs.shape[0] != 3 else bvecs), fmt="%.8f")
    print("Ground-truth ranges:")
    print(f"  ICVF(=NDI within WM): {gt['ndi_within_wm'].min():.2f}..{gt['ndi_within_wm'].max():.2f}")
    print(f"  nd voxel:            {gt['nd'].min():.2f}..{gt['nd'].max():.2f}")
    print(f"  FW:                  {sorted(set(gt['fw'].tolist()))}")
    print(f"  FA:                  {gt['fa'].min():.3f}..{gt['fa'].max():.3f}")
    print(f"  MD x1e3:             {gt['md'].min()*1e3:.3f}..{gt['md'].max()*1e3:.3f}")
    print(f"saved -> {DATA}/signals_mc.npz, ground_truth.npz")


if __name__ == "__main__":
    main()
