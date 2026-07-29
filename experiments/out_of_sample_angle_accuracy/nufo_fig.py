import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
dev = np.load(f"{HERE}/out/nufo_counts_dev.npz")
odf = np.load(f"{HERE}/out/nufo_counts_odffp.npz")
true_N, snr = dev["true_N"], dev["snr"]

COUNTS = {"CSA": dev["CSA"], "CSD": dev["CSD"], "GQI": dev["GQI"],
          "ODF-FP": odf["counts"],
          "FORCE_a1e-03": dev["FORCE_a1e-03"],
          "FORCE_a1e-04": dev["FORCE_a1e-04"],
          "FORCE_a1e-05": dev["FORCE_a1e-05"]}
ORDER = ["CSA", "CSD", "GQI", "ODF-FP", "FORCE_a1e-03", "FORCE_a1e-04", "FORCE_a1e-05"]
LABELS = ["CSA", "CSD", "GQI", "ODFFP", "FORCE\n(α=1e-3)", "FORCE\n(α=1e-4)", "FORCE\n(α=1e-5)"]
NCOL = {1: "#b3a7e6", 2: "#2b2d6e", 3: "#f08a24"}
SNRS = [50, 20, 10]

rows = []
for sv in SNRS:
    for m in ORDER:
        c = COUNTS[m]
        for nn in (1, 2, 3):
            sel = (snr == sv) & (true_N == nn)
            rows.append(dict(snr=sv, method=m, true_N=nn,
                             accuracy=float(np.mean(c[sel] == nn))))
df = pd.DataFrame(rows)
df.to_csv(f"{HERE}/out/nufo_accuracy.csv", index=False)

print("=== NUFO accuracy (fraction reporting exactly N peaks) ===")
for sv in SNRS:
    print(f"\n-- SNR {sv} --   N=1 / N=2 / N=3")
    for m, lab in zip(ORDER, LABELS):
        vals = [df[(df.snr == sv) & (df.method == m) & (df.true_N == nn)].accuracy.values[0] for nn in (1, 2, 3)]
        print(f"  {lab.replace(chr(10),' '):16s} {vals[0]:.2f} / {vals[1]:.2f} / {vals[2]:.2f}")

fig, axes = plt.subplots(len(SNRS), 1, figsize=(11, 12))
x = np.arange(len(ORDER)); w = 0.26
for ax, sv in zip(axes, SNRS):
    for k, nn in enumerate((1, 2, 3)):
        vals = [df[(df.snr == sv) & (df.method == m) & (df.true_N == nn)].accuracy.values[0] for m in ORDER]
        ax.bar(x + (k - 1) * w, np.array(vals) * 100, w, color=NCOL[nn],
               edgecolor="white", linewidth=0.5, label=f"N={nn}")
    ax.set_title(f"SNR={sv}", fontsize=14, fontweight="bold")
    ax.set_ylabel("NUFO accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105); ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if sv == SNRS[0]:
        ax.legend(ncol=3, fontsize=11, loc="upper left")
axes[-1].set_xlabel("Methods", fontsize=12)
fig.suptitle("Number-of-fibres accuracy on out-of-model Monte Carlo voxels\n"
             "(fraction of true N-fibre voxels where the method reports exactly N peaks)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{HERE}/figures/nufo_accuracy.png", dpi=200, bbox_inches="tight", facecolor="white")
print("\nsaved figures/nufo_accuracy.png and out/nufo_accuracy.csv")
