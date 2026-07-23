"""Run FORCE on the 5 tissue substrate types; recover fiber count (ODF peaks +
posterior num_fibers), WM/GM/CSF fractions, NDI, ODI, crossing angle."""
import os
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable
from dipy.data import default_sphere

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE
OUT = os.path.join(HERE, "force_out")
K, BETA = 50, 2000.0


def sphere_nn(sphere, k=18):
    V = sphere.vertices; s = V @ V.T; np.fill_diagonal(s, -1)
    return np.argpartition(-s, k, axis=1)[:, :k]


def peaks(odf, sphere, nn, min_sep=25.0, top=3, rel=0.5):
    local = np.all(odf[:, None] >= odf[nn] - 1e-9, axis=1) & (odf > rel * odf.max())
    cand = np.where(local)[0]
    if cand.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    order = cand[np.argsort(-odf[cand])]
    cthr = np.cos(np.deg2rad(min_sep)); ps, am = [], []
    for i in order:
        d = sphere.vertices[i]
        if all(abs(float(d @ p)) <= cthr for p in ps):
            ps.append(d); am.append(float(odf[i]))
            if len(ps) >= top:
                break
    return np.array(ps), np.array(am)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(DATA, "data_cb/signals_cb.npz"))
    signals = d["signals"].astype(np.float64)
    bvals, bvecs = d["bvals"].astype(np.float64), d["bvecs"].astype(np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=BETA,
                       compute_odf=True, verbose=True)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    qn = np.linalg.norm(signals, axis=1, keepdims=True); qn[qn == 0] = 1
    Dist, neigh = model._index.search(np.ascontiguousarray((signals/qn).astype(np.float32)), k=K)
    S = Dist - model._penalty_array[neigh]
    W = softmax_stable(BETA * S, axis=1).astype(np.float32)

    m = {f: np.einsum('nk,nk->n', W, sims[f][neigh]).astype(np.float32)
         for f in ("nd", "dispersion", "wm_fraction", "gm_fraction",
                   "csf_fraction", "num_fibers", "fa", "md") if f in sims}

    # ODF peaks -> detected fiber count + crossing angle
    sph = default_sphere; nn = sphere_nn(sph)
    nv = sims["odfs"].shape[1]
    Nv = signals.shape[0]
    npk = np.zeros(Nv, np.int8); cross = np.full(Nv, np.nan, np.float32)
    for i in range(Nv):
        o = sims["odfs"][neigh[i]].astype(np.float32)
        o /= (o.max(axis=1, keepdims=True) + 1e-8)
        op = np.einsum('k,kv->v', W[i], o); op /= op.max() + 1e-8
        pk, am = peaks(op, sph, nn)
        npk[i] = pk.shape[0]
        if pk.shape[0] >= 2:
            cross[i] = np.rad2deg(np.arccos(np.clip(abs(float(pk[0] @ pk[1])), 0, 1)))
    m["n_peaks"] = npk; m["cross_deg"] = cross
    np.savez(os.path.join(OUT, "force_cb.npz"), **m)
    print("saved force_cb.npz")


if __name__ == "__main__":
    main()
