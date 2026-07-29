import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
TOL = 20.0
AMP_FRAC = 0.5
FILES = {
    "FORCE":  (f"{HERE}/out/fit_min5.npz",   "#2ca02c", "o", 2.4),
    "ODF-FP": (f"{HERE}/out/peaks_ODF-FP.npz", "#9467bd", "s", 1.7),
    "CSD":    (f"{HERE}/out/peaks_CSD.npz",  "#1f77b4", "^", 1.7),
    "CSA":    (f"{HERE}/out/peaks_CSA.npz",  "#ff7f0e", "v", 1.7),
    "GQI":    (f"{HERE}/out/peaks_GQI.npz",  "#8c564b", "D", 1.7),
}

def ang(u, v):
    return np.rad2deg(np.arccos(np.clip(abs(float(np.dot(u, v))), 0, 1)))

def score(dirs, vals, a1, a2):
    dirs = np.asarray(dirs, float).reshape(-1, 3)
    vals = np.asarray(vals, float).reshape(-1)
    ok = np.isfinite(dirs).all(1) & np.isfinite(vals) & (np.linalg.norm(np.nan_to_num(dirs), axis=1) > 1e-6)
    dirs, vals = dirs[ok], vals[ok]
    if dirs.shape[0] == 0 or vals.max() <= 0:
        return 0, np.nan, False, np.nan
    keep = vals >= AMP_FRAC * vals.max()
    dirs, vals = dirs[keep], vals[keep]
    if dirs.shape[0] < 2:
        return int(dirs.shape[0]), np.nan, False, np.nan
    o = np.argsort(-vals)[:2]
    p1, p2 = dirs[o[0]], dirs[o[1]]
    d_aa = ang(p1, a1) + ang(p2, a2); d_ab = ang(p1, a2) + ang(p2, a1)
    e1, e2 = (ang(p1, a1), ang(p2, a2)) if d_aa <= d_ab else (ang(p1, a2), ang(p2, a1))
    return int(dirs.shape[0]), 0.5 * (e1 + e2), (e1 <= TOL and e2 <= TOL), ang(p1, p2)

def main():
    d = np.load(f"{HERE}/data/crossings.npz")
    a1, a2, snr, true_ang = d["axis1"], d["axis2"], d["snr"], d["true_angle"]
    angles = sorted(set(true_ang.tolist())); snrs = sorted(set(snr.tolist()), reverse=True)
    N = a1.shape[0]

    rows = []
    for method, (path, *_ ) in FILES.items():
        z = np.load(path)
        pdirs, pvals = z["peak_dirs"], z["peak_vals"]
        res = np.array([score(pdirs[i], pvals[i], a1[i], a2[i]) for i in range(N)], object)
        npk = res[:, 0].astype(float); merr = res[:, 1].astype(float)
        resolved = res[:, 2].astype(bool); cross = res[:, 3].astype(float)
        for sv in snrs:
            for aa in angles:
                m = (snr == sv) & (true_ang == aa)
                rows.append(dict(method=method, snr=int(sv), true_angle=int(aa), n=int(m.sum()),
                                 resolved_rate=float(np.mean(resolved[m])),
                                 mean_npeaks=float(np.mean(npk[m])),
                                 ang_err=float(np.nanmean(merr[m])),
                                 cross_est=float(np.nanmean(cross[m]))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/out/angular_accuracy_all.csv", index=False)

    print("=" * 90)
    print("UNIFORM amplitude-pruned scoring (0.5x max amp, top-2, <20 deg) on OUT-OF-MODEL MC crossings")
    print("=" * 90)
    for sv in snrs:
        print(f"\n--- SNR {sv}: correctly-resolved rate ---")
        print("  ang   " + "".join(f"{m:>8}" for m in FILES))
        for aa in angles:
            print(f"  {int(aa):3d}  " + "".join(
                f"{df[(df.method==m)&(df.snr==sv)&(df.true_angle==aa)].resolved_rate.values[0]:8.2f}"
                for m in FILES))

    print("\n--- recovered crossing angle (deg) at SNR50, shallow bins ---")
    print("  ang   " + "".join(f"{m:>8}" for m in FILES))
    for aa in [15, 20, 30, 40]:
        print(f"  {int(aa):3d}  " + "".join(
            f"{df[(df.method==m)&(df.snr==50)&(df.true_angle==aa)].cross_est.values[0]:8.1f}"
            for m in FILES))

    fig, axes = plt.subplots(2, len(snrs), figsize=(15.5, 8.2), sharex=True)
    for j, sv in enumerate(snrs):
        ax0, ax1 = axes[0, j], axes[1, j]
        for m, (path, c, mk, lw) in FILES.items():
            sub = df[(df.method == m) & (df.snr == sv)].sort_values("true_angle")
            ax0.plot(sub.true_angle, sub.resolved_rate, "-", color=c, marker=mk, ms=5, lw=lw,
                     label=m, zorder=3 if m == "FORCE" else 2)
            stable = sub[sub.resolved_rate * sub.n >= 5]
            ax1.plot(stable.true_angle, stable.ang_err, "-", color=c, marker=mk, ms=4, lw=lw)
        ax0.set_title(f"SNR {sv}", fontsize=12, fontweight="bold")
        ax0.set_ylim(-0.03, 1.03); ax1.set_ylim(0, 20)
        ax1.set_xlabel("true crossing angle (deg)", fontsize=11)
        for ax in (ax0, ax1):
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("correctly resolved\n(both fibres < 20 deg)", fontsize=11)
    axes[1, 0].set_ylabel("angular error of\nresolved crossings (deg)", fontsize=11)
    axes[0, 0].legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Angular accuracy on out-of-model Monte Carlo crossings: "
                 "FORCE vs CSD / CSA / GQI / ODF-FP (uniform amplitude-pruned scoring)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{HERE}/figures/angular_accuracy_all.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("\nsaved figures/angular_accuracy_all.png and out/angular_accuracy_all.csv")

if __name__ == "__main__":
    main()
