"""Run FORCE on the Monte Carlo phantom signals and recover microstructure.

Single top-K=50 retrieval against the cached 500K FORCE dictionary
(subject 165840 protocol), posterior softmax at beta=2000 — the same
configuration exp9 used. Optionally adds Rician noise at given SNRs.
Outputs FORCE's voxel metrics for comparison against the substrate
ground truth.
"""
import os
import numpy as np
import nibabel as nib  # noqa: F401
from numpy.random import default_rng

from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "force_out")
K = 50
BETA = 2000.0
SNRS = [None, 50, 20]      # None = noise-free
S0 = 1000.0


def add_rician(sig, snr, rng):
    sigma = 1.0 / snr      # signals are S0=1 normalized -> sigma = 1/snr
    re = sig + rng.normal(0, sigma, sig.shape)
    im = rng.normal(0, sigma, sig.shape)
    return np.sqrt(re ** 2 + im ** 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(DATA, "signals_mc.npz"))
    signals = d["signals"].astype(np.float64)        # (N,288) S0=1
    bvals = d["bvals"].astype(np.float64)
    bvecs = d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    N = signals.shape[0]
    print(f"{N} MC phantom voxels")

    print("Loading cached 500K FORCE dictionary ...")
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True,
                       posterior_beta=BETA, compute_odf=False, verbose=True)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    print(f"dictionary signals {sims['signals'].shape}; fields {sorted(sims.keys())}")

    rng = default_rng(20260623)
    for snr in SNRS:
        tag = "clean" if snr is None else f"snr{snr}"
        Q = signals.copy() if snr is None else add_rician(signals, snr, rng)
        qn = np.linalg.norm(Q, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        Qn = np.ascontiguousarray((Q / qn).astype(np.float32))
        Dist, neigh = model._index.search(Qn, k=K)
        S = Dist - model._penalty_array[neigh]
        W = softmax_stable(BETA * S, axis=1).astype(np.float32)
        metrics = {}
        for fld in ("fa", "md", "rd", "nd", "dispersion", "csf_fraction",
                    "wm_fraction", "gm_fraction", "num_fibers"):
            if fld in sims:
                metrics[fld] = np.einsum('nk,nk->n', W, sims[fld][neigh]).astype(np.float32)
        np.savez(os.path.join(OUT, f"force_mc_{tag}.npz"), **metrics)
        print(f"  [{tag}] saved force_mc_{tag}.npz  "
              f"nd={metrics['nd'].mean():.3f} disp={metrics['dispersion'].mean():.3f} "
              f"csf={metrics['csf_fraction'].mean():.3f}")


if __name__ == "__main__":
    main()
