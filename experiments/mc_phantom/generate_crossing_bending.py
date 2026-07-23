"""Simulate true-MC crossing and bending substrates on the 165840 protocol."""
import os, time
import numpy as np
import np2_shim  # noqa
from disimpy import gradients, simulations, substrates
from crossing_bending import build_crossing, build_undulating

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_cb")
PROTO = os.environ.get("FORCE_PROTO", "hcp_subject")  # HCP-like subject dir with bvals/bvecs
D0 = 2.2e-9
DELTA, delta = 30e-3, 10e-3


def simulate(mesh, bvals, bvecs, n_walkers, n_t, seed):
    sub = substrates.mesh(mesh["vertices"], mesh["faces"], periodic=True,
                          padding=mesh["padding"], init_pos="uniform",
                          n_sv=np.array([24, 24, 24]), quiet=True)
    g, dt = gradients.pgse(delta, DELTA, n_t, bvals * 1e6, bvecs)
    t0 = time.time()
    sig = simulations.simulation(n_walkers, D0, g, dt, sub, seed=seed, quiet=True)
    return (sig / sig[bvals <= 50].mean()).astype(np.float32), time.time() - t0


def main():
    os.makedirs(DATA, exist_ok=True)
    bvals = np.loadtxt(os.path.join(PROTO, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(PROTO, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    rows = []   # (name, kind, num_fibers, cross_deg, max_tilt_deg, icvf, signal)

    # --- CROSSING 90 deg (true interpenetrating) ---
    cr = build_crossing(radius=1.2e-6, L=30e-6, layer_t=3.6e-6, spacing=3.6e-6, n_theta=20)
    S, dur = simulate(cr, bvals, bvecs, int(1e5), 5000, 11)
    print(f"CROSSING90 ICVF={cr['icvf']:.3f} {dur:.0f}s")
    rows.append(("crossing90", "crossing", 2, 90.0, np.nan, cr["icvf"], S))

    # --- BENDING sweep (undulation amplitude -> curvature) ---
    for A in (0.0, 1.5e-6, 3.0e-6, 4.5e-6):
        un = build_undulating(radius=1.0e-6, L=22e-6, amplitude=A, wavelength=22e-6,
                              spacing_y=3.2e-6, spacing_z=3.2e-6, n_theta=16, n_seg=36)
        S, dur = simulate(un, bvals, bvecs, int(8e4), 5000, 21)
        print(f"BEND A={A*1e6:.1f}um tilt={un['max_tilt_deg']:.0f}deg "
              f"ICVF={un['icvf']:.3f} faces={un['faces'].shape[0]} {dur:.0f}s")
        rows.append((f"bend_A{A*1e6:.1f}", "bending", 1, np.nan,
                     un["max_tilt_deg"], un["icvf"], S))

    signals = np.array([r[6] for r in rows])
    np.savez(os.path.join(DATA, "signals_cb.npz"),
             signals=signals, bvals=bvals.astype(np.float32),
             bvecs=bvecs.astype(np.float32),
             names=np.array([r[0] for r in rows]),
             kind=np.array([r[1] for r in rows]),
             num_fibers=np.array([r[2] for r in rows], np.float32),
             cross_deg=np.array([r[3] for r in rows], np.float32),
             max_tilt_deg=np.array([r[4] for r in rows], np.float32),
             icvf=np.array([r[5] for r in rows], np.float32))
    print(f"saved {signals.shape[0]} substrates -> {DATA}/signals_cb.npz")


if __name__ == "__main__":
    main()
