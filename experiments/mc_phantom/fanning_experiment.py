"""True-MC fanning (Watson-oriented cylinders, 1 fiber) -> FORCE ODI recovery.
The proper test the undulation was NOT: a unimodal fan should be recovered well."""
import os, time
import numpy as np
import np2_shim  # noqa
from disimpy import gradients, simulations, substrates
from crossing_bending import build_fanning
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.force import FORCEModel, softmax_stable

HERE = os.path.dirname(os.path.abspath(__file__))
P = "/home/athshah/Phi/165840"
D0 = 2.2e-9


def odi_to_kappa(o): return 1.0 / np.tan(np.pi * o / 2.0)


def sphere_nn(sph, k=18):
    s = sph.vertices @ sph.vertices.T; np.fill_diagonal(s, -1)
    return np.argpartition(-s, k, axis=1)[:, :k]


def npeaks(odf, sph, nn, rel=0.5, sep=25.0):
    loc = np.all(odf[:, None] >= odf[nn] - 1e-9, axis=1) & (odf > rel * odf.max())
    cand = np.where(loc)[0]
    if cand.size == 0:
        return 0
    order = cand[np.argsort(-odf[cand])]; cth = np.cos(np.deg2rad(sep)); ps = []
    for i in order:
        d = sph.vertices[i]
        if all(abs(float(d @ p)) <= cth for p in ps):
            ps.append(d)
    return len(ps)


def main():
    bvals = np.loadtxt(f"{P}/bvals").ravel(); bvecs = np.loadtxt(f"{P}/bvecs")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    odis = [0.03, 0.08, 0.13, 0.18, 0.23, 0.28]
    sigs, truth = [], []
    for od in odis:
        m = build_fanning(od, 0.9e-6, 30e-6, 80, n_theta=16, n_seg=22, seed=3)
        sub = substrates.mesh(m["vertices"], m["faces"], periodic=False,
                              padding=m["padding"], init_pos="uniform",
                              n_sv=np.array([22, 22, 22]), quiet=True)
        g, dt = gradients.pgse(10e-3, 30e-3, 5000, bvals * 1e6, bvecs)
        t0 = time.time()
        s = simulations.simulation(int(8e4), D0, g, dt, sub, seed=4, quiet=True)
        sigs.append((s / s[bvals <= 50].mean()).astype(np.float64))
        truth.append((od, m["icvf"], m["n_cyl"], m["disp_deg"]))
        print(f"ODI={od:.2f} ICVF={m['icvf']:.3f} n_cyl={m['n_cyl']} "
              f"disp={m['disp_deg']:.0f}deg {time.time()-t0:.0f}s")

    model = FORCEModel(gtab, n_neighbors=50, use_posterior=True, posterior_beta=2000.,
                       compute_odf=True, verbose=False)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    Q = np.array(sigs); qn = np.linalg.norm(Q, axis=1, keepdims=True)
    D, nb = model._index.search(np.ascontiguousarray((Q / qn).astype(np.float32)), k=50)
    W = softmax_stable(2000. * (D - model._penalty_array[nb]), axis=1)
    wm = np.einsum('nk,nk->n', W, sims["wm_fraction"][nb])
    disp = np.einsum('nk,nk->n', W, sims["dispersion"][nb])
    nd = np.einsum('nk,nk->n', W, sims["nd"][nb])
    odi_rec = np.clip(1 - (1 - disp) / np.maximum(wm, 1e-6), 0, 1)
    sph = default_sphere; nn = sphere_nn(sph)
    npk = []
    for i in range(len(sigs)):
        o = sims["odfs"][nb[i]].astype(np.float32); o /= o.max(1, keepdims=True) + 1e-8
        op = (W[i][:, None] * o).sum(0)
        npk.append(npeaks(op, sph, nn))

    print(f"\n{'trueODI':>8}{'ICVF':>6}{'recODI':>8}{'#peaks':>7}{'nd':>6}{'NDIwm':>7}")
    for i, (od, ic, nc, dd) in enumerate(truth):
        print(f"{od:>8.2f}{ic:>6.2f}{odi_rec[i]:>8.3f}{npk[i]:>7d}{nd[i]:>6.2f}"
              f"{nd[i]/max(wm[i],1e-6):>7.2f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tod = np.array([t[0] for t in truth])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(tod, odi_rec, "o-", lw=2, label="FORCE recovered ODI")
    ax.plot([0, 0.3], [0, 0.3], "r--", label="identity")
    for x, y, p in zip(tod, odi_rec, npk):
        ax.annotate(f"{p}pk", (x, y), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("true Watson ODI (fanning)"); ax.set_ylabel("FORCE recovered ODI")
    ax.set_title("True-MC FANNING (1 fiber): FORCE ODI recovery")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.32)
    plt.tight_layout(); plt.savefig(f"{HERE}/figures/fanning_recovery.png", dpi=140, bbox_inches="tight")
    print("saved figures/fanning_recovery.png")


if __name__ == "__main__":
    main()
