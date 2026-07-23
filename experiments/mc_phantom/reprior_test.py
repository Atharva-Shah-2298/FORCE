"""Does changing the dictionary tissue-fraction prior move FORCE's partition
closer to truth on the biologically-realistic phantom?

The 500K dictionary draws fractions from Dirichlet(2,1,1) (WM~0.5 mean; only
0.7% of atoms near-pure-WM). Re-sampling the existing atoms by their WM fraction
is equivalent to re-generating with a different fraction prior (microstructure
is sampled independently of fraction). We compare matching under:
  - original  : the native Dirichlet(2,1,1) density
  - flat_wm   : WM fraction uniform on [0,1]
  - wm_heavy  : prior favouring dense-WM voxels (more near-pure-WM atoms)
on the biological substrate (sub-micron axons, ICVF=0.60, true wm=1/iso=0,
plus a realistic FW=0.07 variant — even corpus callosum has some free water).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = os.path.expanduser("~/.dipy/force_simulations")
ICVF = 0.60
K, BETA = 50, 2000.0


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    z = np.load(os.path.join(BASE, "force_sim_100.npz"), mmap_mode="r")
    wm = np.asarray(z["wm_fraction"]); gm = np.asarray(z["gm_fraction"])
    csf = np.asarray(z["csf_fraction"]); nd = np.asarray(z["nd"])
    disp = np.asarray(z["dispersion"])
    N = wm.shape[0]
    print(f"dictionary {N} atoms; normalizing signals ...")
    sig = np.empty((N, z["signals"].shape[1]), np.float32)
    CH = 50000
    for s in range(0, N, CH):
        blk = np.asarray(z["signals"][s:s+CH], np.float32)
        blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
        sig[s:s+CH] = blk

    rng = np.random.default_rng(0)
    # --- build re-prior'd index sets (atom indices) ---
    bins = np.linspace(0, 1, 21)
    binid = np.clip(np.digitize(wm, bins) - 1, 0, 19)
    priors = {}
    priors["original"] = np.arange(N)
    # flat_wm: equal atoms per wm-bin
    per = 6000
    idx = []
    for b in range(20):
        m = np.where(binid == b)[0]
        if m.size:
            idx.append(rng.choice(m, size=min(per, m.size), replace=False))
    priors["flat_wm"] = np.concatenate(idx)
    # wm_heavy: linearly more atoms toward high wm
    idx = []
    for b in range(20):
        m = np.where(binid == b)[0]
        if m.size:
            cap = int(800 + 9000 * (b / 19.0))     # few low-wm, many high-wm
            idx.append(rng.choice(m, size=min(cap, m.size), replace=False))
    priors["wm_heavy"] = np.concatenate(idx)
    for k, v in priors.items():
        print(f"  prior '{k}': {v.size} atoms, %near-pure-WM(iso<0.05)="
              f"{100*((gm+csf)[v]<0.05).mean():.2f}%")

    # --- biological test voxels ---
    d = np.load(os.path.join(DATA, "radius_sweep_signals.npz"))
    csf_sig = d["csf"].astype(np.float64)
    voxels = []
    for r in (0.5, 1.0):
        S = d[f"r{r}"].astype(np.float64)
        for fw in (0.0, 0.07):
            Sv = (1 - fw) * S + fw * csf_sig
            voxels.append((f"r{r}_FW{fw:.2f}", fw, Sv / np.linalg.norm(Sv)))

    print(f"\nTruth: pure WM substrate (iso = FW), within-WM NDI = {ICVF:.2f}")
    print(f"{'voxel':<12}{'prior':<10}{'WM':>6}{'GM':>6}{'CSF':>6}{'iso':>6}"
          f"{'NDIwm':>7}{'ND':>6}{'isoTrue':>8}")
    for name, fw, Sv in voxels:
        Sv = Sv.astype(np.float32)
        for pk, pidx in priors.items():
            cs = sig[pidx] @ Sv
            top = np.argpartition(-cs, K)[:K]
            W = softmax(BETA * cs[top])
            ai = pidx[top]
            WM = float((W * wm[ai]).sum()); GM = float((W * gm[ai]).sum())
            CF = float((W * csf[ai]).sum()); ND = float((W * nd[ai]).sum())
            ndi = ND / max(WM, 1e-6)
            print(f"{name:<12}{pk:<10}{WM:>6.2f}{GM:>6.2f}{CF:>6.2f}{GM+CF:>6.2f}"
                  f"{ndi:>7.2f}{ND:>6.2f}{fw:>8.2f}")
        print()


if __name__ == "__main__":
    main()
