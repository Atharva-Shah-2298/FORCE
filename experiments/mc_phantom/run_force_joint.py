"""Run FORCE on the dispersed MC phantom and record its dispersion output."""
import os
import numpy as np
from numpy.random import default_rng
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "force_out")
K, BETA = 50, 2000.0


def add_rician(sig, snr, rng):
    s = 1.0 / snr
    return np.sqrt((sig + rng.normal(0, s, sig.shape)) ** 2 + rng.normal(0, s, sig.shape) ** 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(DATA, "signals_joint.npz"))
    signals = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    print(f"{signals.shape[0]} joint voxels")
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=BETA,
                       compute_odf=False, verbose=True)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    rng = default_rng(20260623)
    for snr in (None, 20):
        tag = "clean" if snr is None else f"snr{snr}"
        Q = signals.copy() if snr is None else add_rician(signals, snr, rng)
        qn = np.linalg.norm(Q, axis=1, keepdims=True); qn[qn == 0] = 1
        Dist, neigh = model._index.search(np.ascontiguousarray((Q / qn).astype(np.float32)), k=K)
        S = Dist - model._penalty_array[neigh]
        W = softmax_stable(BETA * S, axis=1).astype(np.float32)
        m = {f: np.einsum('nk,nk->n', W, sims[f][neigh]).astype(np.float32)
             for f in ("fa", "md", "rd", "nd", "dispersion", "csf_fraction",
                       "wm_fraction", "gm_fraction") if f in sims}
        np.savez(os.path.join(OUT, f"force_joint_{tag}.npz"), **m)
        print(f"  [{tag}] saved.")


if __name__ == "__main__":
    main()
