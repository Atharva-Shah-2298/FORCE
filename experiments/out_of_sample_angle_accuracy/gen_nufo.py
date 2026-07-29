import os, sys
import numpy as np

EXP10 = "/home/athshah/force_paper_experiments/exp10_mc_phantom"
sys.path.insert(0, EXP10); os.chdir(EXP10)
import np2_shim
from generate_dispersed_phantom import simulate_kernel, SHELLS

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
PROTO = "/home/athshah/Phi/165840"
NPER, SNRS = 150, [50, 20, 10]
ANG_LO, ANG_HI, MIN_SEP = 15.0, 90.0, 15.0
ICVF, RADIUS_UM, SEED = 0.6, 2.0, 20260729

def _rand_unit(rng):
    v = rng.normal(size=3); return v / np.linalg.norm(v)

def _axis_at(a_ref, theta_deg, rng):
    th = np.deg2rad(theta_deg)
    tmp = np.array([1.0, 0, 0]) if abs(a_ref[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a_ref, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a_ref, e1)
    phi = rng.uniform(0, 2 * np.pi)
    return np.cos(th) * a_ref + np.sin(th) * (np.cos(phi) * e1 + np.sin(phi) * e2)

def rand_axes(n, rng):
    a1 = _rand_unit(rng)
    if n == 1:
        return a1[None]
    if n == 2:
        return np.stack([a1, _axis_at(a1, rng.uniform(ANG_LO, ANG_HI), rng)])
    while True:
        a2 = _axis_at(a1, rng.uniform(60.0, ANG_HI), rng)
        a3 = _axis_at(a1, rng.uniform(60.0, ANG_HI), rng)
        if np.rad2deg(np.arccos(abs(np.clip(a2 @ a3, -1, 1)))) >= 60.0:
            return np.stack([a1, a2, a3])

def multi_signal(interps, axes, bvals, bvecs):
    S = np.ones(bvals.shape)
    cos = np.abs(bvecs @ axes.T)
    th = np.arccos(np.clip(cos, 0, 1))
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        shell = SHELLS[int(np.argmin([abs(b - s) for s in SHELLS]))]
        S[i] = np.mean([interps[shell](th[i, k]) for k in range(axes.shape[0])])
    return S

def rician(s, sig, rng):
    return np.sqrt((s + rng.normal(0, sig, s.shape)) ** 2 + rng.normal(0, sig, s.shape) ** 2)

def main():
    bvals = np.loadtxt(f"{PROTO}/bvals").ravel().astype(float)
    bvecs = np.loadtxt(f"{PROTO}/bvecs")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    rng = np.random.default_rng(SEED)
    print("simulating MC kernel ...", flush=True)
    interps, icvf_geom, dur, _ = simulate_kernel(ICVF, RADIUS_UM, int(2e5), 3000, 6, 31, 6, SEED)
    print(f"  kernel done {dur:.0f}s icvf={icvf_geom:.3f}")

    sig, gN, gsnr = [], [], []
    for N in (1, 2, 3):
        for _ in range(NPER):
            ax = rand_axes(N, rng)
            Sc = multi_signal(interps, ax, bvals, bvecs)
            for snr in SNRS:
                sig.append(rician(Sc, 1.0 / snr, rng).astype(np.float32))
                gN.append(N); gsnr.append(snr)
    sig = np.asarray(sig, np.float32)
    np.savez(f"{HERE}/data/nufo.npz", signals=sig,
             bvals=bvals.astype(np.float32), bvecs=bvecs.astype(np.float32),
             true_N=np.asarray(gN, np.int8), snr=np.asarray(gsnr, np.float32))
    print(f"saved {sig.shape[0]} voxels (N=1/2/3 x {SNRS} x {NPER}) -> data/nufo.npz")

if __name__ == "__main__":
    main()
