import os
import sys
import numpy as np

EXP10 = "/home/athshah/force_paper_experiments/exp10_mc_phantom"
sys.path.insert(0, EXP10)
os.chdir(EXP10)
import np2_shim
from generate_dispersed_phantom import simulate_kernel, SHELLS

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
PROTO = "/home/athshah/Phi/165840"

ANGLES = [15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
SNRS = [50, 20, 10]
N_PER = 50
ICVF, RADIUS_UM = 0.6, 2.0
SEED = 20260728

def random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def crossing_signal(interps, a1, a2, bvals, bvecs):
    S = np.ones(bvals.shape, dtype=np.float64)
    cos1 = np.abs(bvecs @ a1)
    cos2 = np.abs(bvecs @ a2)
    th1 = np.arccos(np.clip(cos1, 0.0, 1.0))
    th2 = np.arccos(np.clip(cos2, 0.0, 1.0))
    for i, b in enumerate(bvals):
        if b <= 50:
            continue
        shell = SHELLS[int(np.argmin([abs(b - sh) for sh in SHELLS]))]
        S[i] = 0.5 * (interps[shell](th1[i]) + interps[shell](th2[i]))
    return S

def rician(s, sigma, rng):
    return np.sqrt((s + rng.normal(0, sigma, s.shape)) ** 2 + rng.normal(0, sigma, s.shape) ** 2)

def main():
    bvals = np.loadtxt(f"{PROTO}/bvals").ravel().astype(float)
    bvecs = np.loadtxt(f"{PROTO}/bvecs")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    rng = np.random.default_rng(SEED)

    print(f"Simulating MC kernel (ICVF={ICVF}, r={RADIUS_UM}um) ...", flush=True)
    interps, icvf_geom, dur, dt = simulate_kernel(
        ICVF, RADIUS_UM, n_walkers=int(2e5), n_t=3000, n_side=6,
        n_theta=31, n_phi=6, seed=SEED)
    print(f"  kernel done in {dur:.1f}s  icvf_geom={icvf_geom:.3f}  dt={dt*1e6:.1f}us")
    print(f"  S1000(0deg)={interps[1000.](0.0):.3f}  S1000(90deg)={interps[1000.](np.pi/2):.3f}")

    signals, g_ang, g_a1, g_a2, g_snr = [], [], [], [], []
    for ang in ANGLES:
        th = np.deg2rad(ang)
        base1 = np.array([0.0, 0.0, 1.0])
        base2 = np.array([np.sin(th), 0.0, np.cos(th)])
        for snr in SNRS:
            sigma = 1.0 / snr
            for _ in range(N_PER):
                R = random_rotation(rng)
                a1 = R @ base1
                a2 = R @ base2
                Sc = crossing_signal(interps, a1, a2, bvals, bvecs)
                Sn = rician(Sc, sigma, rng).astype(np.float32)
                signals.append(Sn)
                g_ang.append(ang); g_a1.append(a1); g_a2.append(a2); g_snr.append(snr)

    signals = np.asarray(signals, np.float32)
    np.savez(f"{HERE}/data/crossings.npz",
             signals=signals,
             bvals=bvals.astype(np.float32),
             bvecs=bvecs.astype(np.float32),
             true_angle=np.asarray(g_ang, np.float32),
             axis1=np.asarray(g_a1, np.float32),
             axis2=np.asarray(g_a2, np.float32),
             snr=np.asarray(g_snr, np.float32),
             icvf=np.float32(icvf_geom), radius_um=np.float32(RADIUS_UM))
    print(f"\nSaved {signals.shape[0]} voxels "
          f"({len(ANGLES)} angles x {len(SNRS)} SNR x {N_PER}) -> data/crossings.npz")

if __name__ == "__main__":
    main()
