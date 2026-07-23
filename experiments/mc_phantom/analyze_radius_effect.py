"""Run FORCE on the radius-sweep substrates and show compartment partition +
NDI moving toward truth as axon radius becomes biologically realistic.
Truth (all radii): WM=1, GM=0, CSF=0, NDI(within-WM)=ICVF=0.60, coherent.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
ICVF = 0.60
FW_MIX = [0.0, 0.10]            # also test a realistic small free-water level


def main():
    d = np.load(os.path.join(DATA, "radius_sweep_signals.npz"))
    radii = d["radii"].tolist()
    bvals, bvecs = d["bvals"].astype(float), d["bvecs"].astype(float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    csf = d["csf"].astype(float)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    # build voxels: each radius x FW level
    sig, meta = [], []
    for r in radii:
        S = d[f"r{r}"].astype(float)
        for fw in FW_MIX:
            sig.append((1 - fw) * S + fw * csf)
            meta.append((r, fw))
    sig = np.array(sig)

    model = FORCEModel(gtab, n_neighbors=50, use_posterior=True,
                       posterior_beta=2000.0, compute_odf=False, verbose=True)
    model.generate(num_simulations=500000, use_cache=True, num_cpus=-1,
                   compute_dti=True, compute_dki=False)
    sims = model.simulations
    qn = np.linalg.norm(sig, axis=1, keepdims=True); qn[qn == 0] = 1
    Dist, neigh = model._index.search(np.ascontiguousarray((sig/qn).astype(np.float32)), k=50)
    Sc = Dist - model._penalty_array[neigh]
    W = softmax_stable(2000.0 * Sc, axis=1).astype(np.float32)
    out = {f: np.einsum('nk,nk->n', W, sims[f][neigh]) for f in
           ("nd", "dispersion", "wm_fraction", "gm_fraction", "csf_fraction")}

    print(f"\n{'r(um)':>6}{'FW':>5}{'WM':>7}{'GM':>7}{'CSF':>7}{'iso(GM+CSF)':>13}"
          f"{'NDIwm':>8}{'NDtrue':>8}{'ND':>7}")
    rows = {}
    for i, (r, fw) in enumerate(meta):
        wm = out["wm_fraction"][i]; gm = out["gm_fraction"][i]; cf = out["csf_fraction"][i]
        ndi = out["nd"][i] / max(wm, 1e-6)
        ndt = (1 - fw) * ICVF
        print(f"{r:>6.1f}{fw:>5.2f}{wm:>7.2f}{gm:>7.2f}{cf:>7.2f}{gm+cf:>13.2f}"
              f"{ndi:>8.2f}{ndt:>8.2f}{out['nd'][i]:>7.2f}")
        rows.setdefault(fw, []).append((r, wm, gm, cf, gm+cf, ndi, out["nd"][i], ndt))

    # plot: iso-leak (GM+CSF, truth=FW) and NDI-within-WM vs radius
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for fw in FW_MIX:
        arr = np.array(rows[fw])
        rr = arr[:, 0]
        axes[0].plot(rr, arr[:, 4], "o-", label=f"FORCE iso (FW={fw:g})")
        axes[0].axhline(fw, ls=":", color="grey")
        axes[1].plot(rr, arr[:, 5], "o-", label=f"FORCE NDI within-WM (FW={fw:g})")
    axes[0].axhspan(0.4, 1.0, color="green", alpha=0.06)
    axes[0].text(2.8, 0.02, "biological\n(<~1um)", color="green", fontsize=8)
    axes[0].set_xlabel("axon radius (um)"); axes[0].set_ylabel("isotropic frac GM+CSF")
    axes[0].set_title("Iso-leak vs axon radius (dotted = true FW)")
    axes[0].legend(fontsize=8); axes[0].axvspan(0, 1.0, color="green", alpha=0.06)
    axes[1].axhline(ICVF, ls="--", color="r", label="true NDI=0.60")
    axes[1].set_xlabel("axon radius (um)"); axes[1].set_ylabel("NDI within-WM")
    axes[1].set_title("NDI vs axon radius"); axes[1].legend(fontsize=8)
    axes[1].axvspan(0, 1.0, color="green", alpha=0.06)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "radius_effect.png"), dpi=150, bbox_inches="tight")
    print("\nsaved figures/radius_effect.png")


if __name__ == "__main__":
    main()
