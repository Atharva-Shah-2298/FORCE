#!/usr/bin/env python
"""Render the angular-resolution and fiber-count figures from a results JSON.

Writes angle_accuracy.png / nufo_accuracy.png plus the plotted numbers as CSV.

    python plot_results.py --results results --out-dir results
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical hues in fixed slot order. FORCE takes the first three slots and
# is drawn solid; baselines are dashed, so method family reads without colour.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

# One-hue ordinal ramp for the fiber-count bars (N=1 -> N=3 is ordered).
COUNT_RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


def bucket_label(key):
    lo, hi = key.strip("()").split(",")
    return f"{int(lo)}–{int(hi)}°"


def order_methods(names):
    """FORCE variants first (finest penalty first), then the baselines."""
    force = sorted(
        (n for n in names if n.startswith("FORCE")),
        key=lambda n: -float(n.split("=")[-1].rstrip(")")),
    )
    rest = [n for n in ("CSA", "CSD", "GQI", "ODFFP") if n in names]
    return force + rest + [n for n in names if n not in force and n not in rest]


def style_for(name, index):
    is_force = name.startswith("FORCE")
    return {
        "color": SLOTS[index % len(SLOTS)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": "-" if is_force else "--",
        "linewidth": 2.0 if is_force else 1.6,
        "markersize": 6.5,
        "markerfacecolor": SLOTS[index % len(SLOTS)] if is_force else "white",
        "markeredgecolor": SLOTS[index % len(SLOTS)],
        "markeredgewidth": 1.6,
        "zorder": 3 if is_force else 2,
    }


def load_odffp(path, snr_keys, buckets):
    """ODFFP reference curves from the original paper run (not recomputed)."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        ref = json.load(fh)
    results = ref["variants"]["ODFFP"]["results"]
    out = {}
    for snr in snr_keys:
        if snr not in results:
            return None
        out[snr] = {
            b: results[snr][b]["peak_detection_rate"] for b in buckets if b in results[snr]
        }
    return out


def plot_angular(results, snr_keys, out_path, odffp=None):
    methods = order_methods(list(results))
    buckets = list(results[methods[0]][snr_keys[0]])
    if odffp is not None:
        methods = methods + ["ODFFP"]
    styles = {m: style_for(m, i) for i, m in enumerate(methods)}
    x = np.arange(len(buckets))

    fig, axes = plt.subplots(
        len(snr_keys), 1, figsize=(8.0, 1.95 * len(snr_keys) + 1.15),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, snr in zip(axes, snr_keys):
        for method in methods:
            if method == "ODFFP" and odffp is not None:
                values = [100 * odffp[snr][b] for b in buckets]
            else:
                values = [100 * results[method][snr][b]["peak_detection_rate"] for b in buckets]
            ax.plot(x, values, label=method, **styles[method])
        ax.set_ylabel("Peak detection rate (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        label = "no noise" if snr == "clean" else f"SNR = {snr}"
        ax.text(0.01, 0.93, label, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold", color=TEXT_PRIMARY)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([bucket_label(b) for b in buckets])
    axes[-1].set_xlabel("Crossing angle")

    n_rows = 2 if len(methods) > 4 else 1
    height = 1.95 * len(snr_keys) + 1.15
    top = 1.0 - (0.30 if n_rows == 2 else 0.22) / height
    fig.tight_layout(rect=(0, 0, 1, top - 0.06))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=int(np.ceil(len(labels) / n_rows)),
               bbox_to_anchor=(0.5, top), columnspacing=1.5, handlelength=2.4)
    fig.suptitle("Two-fiber crossings: peaks recovered within 20°",
                 y=0.995, fontsize=12, color=TEXT_PRIMARY)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_nufo(results, snr_keys, out_path):
    methods = order_methods(list(results))
    classes = list(results[methods[0]][snr_keys[0]])
    width = 0.8 / len(classes)
    x = np.arange(len(methods))

    fig, axes = plt.subplots(
        len(snr_keys), 1, figsize=(7.6, 2.05 * len(snr_keys) + 0.95),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, snr in zip(axes, snr_keys):
        for c_idx, cls in enumerate(classes):
            values = [100 * results[m][snr][cls]["fiber_count_accuracy"] for m in methods]
            offset = (c_idx - (len(classes) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.92, label=cls,
                          color=COUNT_RAMP[c_idx % len(COUNT_RAMP)],
                          edgecolor="white", linewidth=1.2, zorder=2)
            ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2, color=TEXT_SECONDARY)
        ax.set_ylabel("Fiber-count accuracy (%)")
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        label = "no noise" if snr == "clean" else f"SNR = {snr}"
        ax.text(0.01, 0.93, label, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold", color=TEXT_PRIMARY)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(methods)
    axes[-1].set_xlabel("Method")

    top = 1.0 - 0.30 / (2.05 * len(snr_keys) + 0.95)
    fig.tight_layout(rect=(0, 0, 1, top - 0.05))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, top), title="True fiber count",
               title_fontsize=9, columnspacing=1.6)
    fig.suptitle("Voxels whose fiber count is recovered exactly",
                 y=0.995, fontsize=12, color=TEXT_PRIMARY)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_csv(results, out_path, group_name):
    fields = ["method", "snr", group_name, "peak_detection_rate", "peak_precision",
              "peak_f1", "fiber_count_accuracy", "mean_angular_error", "num_voxels"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for method, by_snr in results.items():
            for snr, by_group in by_snr.items():
                for group, metrics in by_group.items():
                    writer.writerow({"method": method, "snr": snr, group_name: group, **metrics})
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="directory holding angle_nufo_results.json")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--odffp-reference", default="odffp_reference.json",
                        help="ODFFP curves from the original run; pass '' to omit")
    args = parser.parse_args()

    with open(os.path.join(args.results, "angle_nufo_results.json")) as fh:
        payload = json.load(fh)
    results = payload["results"]
    os.makedirs(args.out_dir, exist_ok=True)

    any_method = next(iter(results["angular"]))
    snr_keys = list(results["angular"][any_method])
    buckets = list(results["angular"][any_method][snr_keys[0]])
    odffp = load_odffp(args.odffp_reference, snr_keys, buckets)
    if args.odffp_reference and odffp is None:
        print("ODFFP reference not usable for these SNRs/buckets; omitting that curve")

    plot_angular(results["angular"], snr_keys,
                 os.path.join(args.out_dir, "angle_accuracy.png"), odffp=odffp)
    plot_nufo(results["nufo"], snr_keys,
              os.path.join(args.out_dir, "nufo_accuracy.png"))
    write_csv(results["angular"], os.path.join(args.out_dir, "angle_accuracy.csv"), "bucket")
    write_csv(results["nufo"], os.path.join(args.out_dir, "nufo_accuracy.csv"), "fiber_class")


if __name__ == "__main__":
    main()
