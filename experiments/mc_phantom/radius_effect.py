"""Does biological realism move FORCE's estimates closer to truth?

Healthy adult human WM axons are mostly sub-micron to ~1 um (corpus-callosum
histology). At clinical b<=3000 / Delta~30 ms such axons are nearly STICKS,
which is exactly FORCE's zero-radius dictionary assumption. My earlier 2-4 um
radii exaggerated the restriction and manufactured the GM/CSF "iso-leak".

This sweeps axon radius {0.5, 1.0, 2.0, 4.0} um at fixed HEALTHY WM values
(ICVF=0.6, intrinsic D0=2.2e-3, coherent bundle) and measures how FORCE's
compartment partition (WM vs GM vs CSF) and NDI move as the substrate becomes
biologically realistic. The MC timestep is scaled per radius so the rms step
stays ~r/4 (restriction resolved).
"""
import os
import time
import numpy as np
import np2_shim  # noqa: F401
from disimpy import gradients, simulations, substrates
from packed_cylinders import build_substrate_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROTO = "/home/athshah/Phi/165840"
D0 = 2.2e-9                 # intra/extra intrinsic diffusivity (2.2 um^2/ms)
DELTA, delta = 30e-3, 10e-3
ICVF = 0.60


def n_t_for_radius(radius_m, frac=0.25, t_total=DELTA + delta, n_min=3000, n_max=32000):
    """timesteps so rms 3D step ~= frac*radius (resolve restriction)."""
    step = frac * radius_m
    dt = step ** 2 / (6 * D0)
    return int(np.clip(np.ceil(t_total / dt), n_min, n_max))


def main():
    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    csf = np.exp(-bvals * 1e6 * 3.0e-9)
    radii = [0.5, 1.0, 2.0, 4.0]
    n_walkers = int(1e5)

    sub_sigs = {}
    for r in radii:
        s = build_substrate_mesh(ICVF, r * 1e-6, n_side=6, n_theta=24)
        sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                              padding=s["padding"], init_pos="uniform",
                              n_sv=np.array([20, 20, 20]), quiet=True)
        n_t = n_t_for_radius(r * 1e-6)
        grad, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
        t0 = time.time()
        sig = simulations.simulation(n_walkers, D0, grad, dt, sub, seed=7, quiet=True)
        S = sig / sig[bvals <= 50].mean()
        sub_sigs[r] = S.astype(np.float32)
        step = np.sqrt(6 * D0 * dt) * 1e6
        print(f"r={r}um  n_t={n_t}  dt={dt*1e6:.2f}us  step={step:.3f}um "
              f"(r/step={r/step:.1f})  {time.time()-t0:.0f}s  "
              f"S_perp(b3k)={S[bvals>2900].max():.3f}")

    np.savez(os.path.join(DATA, "radius_sweep_signals.npz"),
             radii=np.array(radii), bvals=bvals.astype(np.float32),
             bvecs=bvecs.astype(np.float32),
             **{f"r{r}": sub_sigs[r] for r in radii},
             csf=csf.astype(np.float32))
    print("saved radius_sweep_signals.npz")


if __name__ == "__main__":
    main()
