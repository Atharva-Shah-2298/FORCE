"""Run FORCE matching with single retrieval per SNR + beta reweighting.

For each SNR ∈ {50,20,10}: search top-K=50 neighbors once. Then for each
beta in BETAS: posterior-weighted estimate of NDI, ODI, FW, FA, MD, RD,
num_fibers, and crossing-angle (from peak-1/peak-2 of posterior ODF).
"""
import os
import sys
import numpy as np
import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.reconst.force import (
    FORCEModel, softmax_stable, normalize_signals, create_signal_index,
    compute_uncertainty_ambiguity,
)
from dipy.data import default_sphere

DATA = "data"
OUT = "force_out"
SNRS = [50, 20, 10]
BETAS = [500, 1000, 2000, 5000, 10000]
K = 50

os.makedirs(OUT, exist_ok=True)


def build_sphere_neighbor_list(sphere, k_top=20):
    """For each vertex, list its k_top nearest neighbor vertices (by angle)."""
    V = sphere.vertices
    sims = V @ V.T
    np.fill_diagonal(sims, -1.0)
    nn = np.argpartition(-sims, k_top, axis=1)[:, :k_top]
    return nn


def extract_top_peaks(odf, sphere, sphere_nn, min_sep_deg=20.0, top_n=3, rel_thr=0.5):
    """Return up to top_n peak directions and their amplitudes.

    odf: (n_vert,) ODF on the sphere (positive values).
    """
    n_vert = odf.shape[0]
    # local max: greater than all neighbors
    local = np.all(odf[:, None] >= odf[sphere_nn] - 1e-9, axis=1)
    # remove zeros
    local &= (odf > rel_thr * odf.max())
    cand = np.where(local)[0]
    if cand.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    # sort by amplitude desc
    order = cand[np.argsort(-odf[cand])]
    peaks = []
    amps = []
    cos_thr = np.cos(np.deg2rad(min_sep_deg))
    for idx in order:
        d = sphere.vertices[idx]
        keep = True
        for p in peaks:
            # antipodal symmetry: consider |d.p|
            if abs(float(d @ p)) > cos_thr:
                keep = False
                break
        if keep:
            peaks.append(d)
            amps.append(float(odf[idx]))
            if len(peaks) >= top_n:
                break
    if not peaks:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(peaks), np.array(amps)


def main():
    bvals = np.loadtxt(os.path.join(DATA, "bvals")).astype(np.float64).flatten()
    bvecs = np.loadtxt(os.path.join(DATA, "bvecs")).astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    print("Building FORCE model + loading cached 500K dictionary ...")
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=2000.0,
                       compute_odf=True, verbose=True)
    # use_cache=True will pick a matching cached library
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    print(f"Loaded library: signals shape {sims['signals'].shape}")
    print(f"  fields: {sorted(sims.keys())}")

    has_kurt = ("mk" in sims)
    if not has_kurt:
        print("  NOTE: cached dictionary lacks DKI metrics (mk/ak/rk/kfa). "
              "FORCE will not produce kurtosis estimates; the DKI baseline "
              "still provides them for comparison.")

    # Sphere for peak extraction
    sphere = default_sphere
    sphere_nn = build_sphere_neighbor_list(sphere, k_top=18)

    # Loop SNR
    for snr in SNRS:
        print(f"\n=== SNR = {snr} ===")
        vol = nib.load(os.path.join(DATA, f"dwi_snr{snr}.nii.gz")).get_fdata().astype(np.float32)
        # shape (N, 1, 1, T) -> (N, T)
        Q = vol.reshape(-1, vol.shape[-1])
        Nv = Q.shape[0]
        # Normalize as FORCE does internally
        qn = np.linalg.norm(Q, axis=1, keepdims=True).astype(np.float32)
        qn[qn == 0] = 1.0
        Qn = np.ascontiguousarray(Q / qn)

        print(f"  searching top-K={K} neighbors over {sims['signals'].shape[0]} library ...")
        D, neigh = model._index.search(Qn, k=K)
        S = D - model._penalty_array[neigh]  # (Nv, K)
        # Save cosine sims and neighbors so we can sweep beta downstream cheaply
        np.save(os.path.join(OUT, f"neighbors_snr{snr}.npy"), neigh.astype(np.int32))
        np.save(os.path.join(OUT, f"scores_snr{snr}.npy"), S.astype(np.float32))

        # For each beta compute weighted estimates
        for beta in BETAS:
            print(f"  beta={beta} -> posterior weighting + param recovery ...")
            W = softmax_stable(beta * S, axis=1).astype(np.float32)  # (Nv, K)
            # per-metric weighted averages
            metrics = {}
            for fld in ("fa", "md", "rd", "nd", "dispersion",
                        "csf_fraction", "wm_fraction", "gm_fraction",
                        "num_fibers"):
                if fld in sims:
                    metrics[fld] = np.einsum('nk,nk->n', W, sims[fld][neigh]).astype(np.float32)
            if has_kurt:
                for fld in ("mk", "ak", "rk", "kfa"):
                    metrics[fld] = np.einsum('nk,nk->n', W, sims[fld][neigh]).astype(np.float32)

            # Posterior ODF + peak extraction
            # odfs: (Nlib, n_vert) float16; cast on slice
            n_vert = sims["odfs"].shape[1]
            odf_post = np.zeros((Nv, n_vert), dtype=np.float32)
            # batched to limit memory
            BATCH = 500
            for s in range(0, Nv, BATCH):
                e = min(s + BATCH, Nv)
                ng = neigh[s:e]  # (b, K)
                # (b, K, n_vert)
                o = sims["odfs"][ng].astype(np.float32)
                # per-neighbor normalize
                o /= (o.max(axis=2, keepdims=True) + 1e-8)
                # weighted sum
                op = np.einsum('nk,nkv->nv', W[s:e], o)
                op /= (op.max(axis=1, keepdims=True) + 1e-8)
                odf_post[s:e] = op

            peak_dirs = np.full((Nv, 3, 3), np.nan, dtype=np.float32)
            peak_amps = np.zeros((Nv, 3), dtype=np.float32)
            cross_deg_est = np.full(Nv, np.nan, dtype=np.float32)
            n_peaks_est = np.zeros(Nv, dtype=np.int8)
            for i in range(Nv):
                pks, amps = extract_top_peaks(odf_post[i], sphere, sphere_nn,
                                              min_sep_deg=20.0, top_n=3,
                                              rel_thr=0.5)
                n_peaks_est[i] = pks.shape[0]
                if pks.shape[0] >= 1:
                    peak_dirs[i, :pks.shape[0]] = pks
                    peak_amps[i, :pks.shape[0]] = amps
                if pks.shape[0] >= 2:
                    d0, d1 = pks[0], pks[1]
                    c = abs(float(d0 @ d1))
                    cross_deg_est[i] = float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))

            metrics["crossing_angle_deg"] = cross_deg_est
            metrics["n_peaks"] = n_peaks_est
            metrics["peak_dirs"] = peak_dirs
            metrics["peak_amps"] = peak_amps

            np.savez(os.path.join(OUT, f"force_snr{snr}_beta{beta}.npz"), **metrics)
            print(f"    saved force_snr{snr}_beta{beta}.npz "
                  f"(fields: {sorted(metrics.keys())})")


if __name__ == "__main__":
    main()
