"""Scatter: phantom ground truth vs FORCE and vs AMICO-NODDI, for NDI / ODI /
FW across SNRs. One figure per method (GT on x, estimate on y, identity line),
plus an overlaid comparison and a bias/MAE table."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
OUT = os.path.join(HERE, "bio_out")
FIG = os.path.join(HERE, "figures")
SNRS = ["clean", 50, 20, 10]
# Neurite density compared on the VOXEL scale (intra-neurite fraction of the
# whole voxel) — the only fair common basis. FORCE.nd is already voxel-scale;
# AMICO's NODDI v_ic is within-TISSUE, so its voxel-scale neurite density is
# v_ic*(1-ISOVF) (NODDI's own definition). No division of FORCE.
METRICS = [("nd", "Neurite density (voxel)", (0, 0.7)),
           ("odi", "ODI (dispersion)", (0, 0.4)),
           ("fw", "FW (free water)", (0, 0.35))]
# DTI/DKI-native metrics: (key, label, display_limits, scale applied to values).
# Ground truth = clean-signal DTI(FA/MD/RD)/DKI(MK) fit (see run_dti_dki_bio.py).
NATIVE = [("fa", "FA", (0, 1.0), 1.0),
          ("md", "MD (um^2/ms)", (0, 1.6), 1e3),
          ("rd", "RD (um^2/ms)", (0, 1.2), 1e3),
          ("mk", "MK (mean kurtosis)", (0, 2.8), 1.0),
          ("ak", "AK (axial kurtosis)", (0, 1.4), 1.0),
          ("rk", "RK (radial kurtosis)", (0, 5.0), 1.0)]
# Kurtosis metrics whose ground truth is the model-free MC displacement kurtosis.
KURT = {"mk", "ak", "rk"}
# Colourblind-safe (Okabe-Ito) publication palette.
PALETTE = {"FORCE": "#0072B2",   # blue
           "AMICO": "#E69F00",   # orange
           "DTI":   "#D55E00",   # vermillion
           "DKI":   "#009E73"}   # green


def load():
    gt = np.load(os.path.join(DATA, "ground_truth_bio.npz"))
    truth = {"nd": (1 - gt["fw"]) * gt["ndi"],     # voxel neurite density
             "odi": gt["odi"], "fw": gt["fw"]}
    force = {}
    for s in SNRS:
        d = dict(np.load(os.path.join(OUT, f"force_{s}.npz")))
        force[s] = {"nd": d["voxel_nd"], "odi": d["odi"], "fw": d["fw"]}
    amico = {}
    for s in SNRS:
        p = os.path.join(OUT, f"amico_{s}.npz")
        if os.path.exists(p):
            a = dict(np.load(p))
            amico[s] = {"nd": a["ndi"] * (1 - a["fw"]), "odi": a["odi"], "fw": a["fw"]}
    return truth, force, amico


def load_native():
    """DTI/DKI-native metrics for FORCE / DTI / DKI vs ground truth.

    FA/MD/RD truth = clean-signal DTI(b<=1000) fit. MK truth = the model-free
    Monte-Carlo displacement kurtosis (mc_kurtosis_truth.py) when available,
    otherwise falls back to the clean-signal DKI fit."""
    gtp = os.path.join(OUT, "gt_dtidki.npz")
    if not os.path.exists(gtp):
        return None
    gt = dict(np.load(gtp))
    mk_src = "clean-signal DKI fit"
    mcp = os.path.join(OUT, "gt_kurtosis_mc.npz")
    if os.path.exists(mcp):
        mc = np.load(mcp)
        for kk in ("mk", "ak", "rk"):
            gt[kk] = mc[kk]
        mk_src = "MC displacement kurtosis (narrow-pulse, fitting-free)"
    methods = {}  # name -> {snr -> {metric: array}}
    for name, fn in [("DTI", "dti"), ("DKI", "dki"), ("FORCE", "force_dtidki")]:
        d = {}
        for s in SNRS:
            p = os.path.join(OUT, f"{fn}_{s}.npz")
            if os.path.exists(p):
                d[s] = dict(np.load(p))
        if d:
            methods[name] = d
    return gt, methods, mk_src


def bm(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    ok = np.isfinite(e) & np.isfinite(t)
    d = e[ok] - t[ok]
    return d.mean(), np.abs(d).mean()


def scatter_fig(truth, est, name, color):
    fig, axes = plt.subplots(len(METRICS), len(SNRS), figsize=(4*len(SNRS), 11))
    for r, (k, lab, lim) in enumerate(METRICS):
        for c, s in enumerate(SNRS):
            ax = axes[r, c]
            if s not in est:
                ax.axis("off"); continue
            t = truth[k]; e = est[s][k]
            ax.scatter(t, e, s=14, c=color, alpha=0.5, edgecolors="none")
            lo, hi = lim
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            b, m = bm(e, t)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"SNR {s} — {lab}\nbias {b:+.3f}  MAE {m:.3f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name} estimate")
            if r == len(METRICS)-1:
                ax.set_xlabel("ground truth")
    plt.suptitle(f"{name} recovery on biological MC phantom (red = identity)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fn = os.path.join(FIG, f"bio_scatter_{name.lower()}.png")
    plt.savefig(fn, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", os.path.relpath(fn, HERE))


def overlay(truth, force, amico):
    fig, axes = plt.subplots(len(METRICS), len(SNRS), figsize=(4*len(SNRS), 11))
    for r, (k, lab, lim) in enumerate(METRICS):
        for c, s in enumerate(SNRS):
            ax = axes[r, c]
            t = truth[k]
            ax.scatter(t, force[s][k], s=12, c=PALETTE["FORCE"], alpha=0.45, label="FORCE")
            if s in amico:
                ax.scatter(t, amico[s][k], s=12, c=PALETTE["AMICO"], alpha=0.45, label="AMICO")
            lo, hi = lim
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            fb, fm = bm(force[s][k], t)
            txt = f"FORCE MAE {fm:.3f}"
            if s in amico:
                _, am = bm(amico[s][k], t); txt += f"\nAMICO MAE {am:.3f}"
            ax.set_title(f"SNR {s} — {lab}", fontsize=9)
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
            if c == 0:
                ax.set_ylabel("estimate")
            if r == len(METRICS)-1:
                ax.set_xlabel("ground truth")
    plt.suptitle("FORCE vs AMICO-NODDI — biological MC phantom", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fn = os.path.join(FIG, "bio_scatter_overlay.png")
    plt.savefig(fn, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", os.path.relpath(fn, HERE))


def native_fig(gt, methods, mk_src="clean-signal DKI fit"):
    """FORCE vs DTI vs DKI on their native scalars (FA/MD/RD/MK)."""
    # FORCE and DTI are both very accurate and land on top of each other, so use
    # distinct shapes + hollow (outline-only) markers: overlapping points then
    # show both outlines instead of one method hiding the other.
    order = [n for n in ("DKI", "DTI", "FORCE") if n in methods]
    markers = {"FORCE": "o", "DTI": "^", "DKI": "s"}
    fig, axes = plt.subplots(len(NATIVE), len(SNRS),
                             figsize=(4*len(SNRS), 3.3*len(NATIVE)))
    for r, (k, lab, lim, sc) in enumerate(NATIVE):
        t = np.asarray(gt[k], float) * sc
        for c, s in enumerate(SNRS):
            ax = axes[r, c]
            lo, hi = lim
            lines = []
            for name in order:
                d = methods[name]
                if s not in d or k not in d[s]:
                    continue
                e = np.asarray(d[s][k], float) * sc
                ax.scatter(t, e, s=26, marker=markers[name], facecolors="none",
                           edgecolors=PALETTE[name], linewidths=0.7, alpha=0.8,
                           label=name)
                _, m = bm(e, t); lines.append(f"{name} MAE {m:.3f}")
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            title = f"SNR {s} — {lab}"
            if k in KURT:
                title += "\ntruth = MC displacement kurtosis"
            ax.set_title(title, fontsize=9)
            ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
                    va="top", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
            if c == 0:
                ax.set_ylabel("estimate")
            if r == len(NATIVE)-1:
                ax.set_xlabel("ground truth")
    plt.suptitle("Signal-representation metrics on the biological MC phantom — "
                 "FORCE (blue) vs DTI (vermillion) vs DKI (green); black = identity."
                 f"  FA/MD/RD truth = clean DTI; MK truth = {mk_src}",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fn = os.path.join(FIG, "bio_scatter_dtidki.png")
    plt.savefig(fn, dpi=150, bbox_inches="tight"); plt.close()
    print("saved", os.path.relpath(fn, HERE))


def main():
    truth, force, amico = load()
    print(f"N = {truth['nd'].size} voxels; AMICO SNRs: {list(amico)}")
    scatter_fig(truth, force, "FORCE", PALETTE["FORCE"])
    if amico:
        scatter_fig(truth, amico, "AMICO", PALETTE["AMICO"])
        overlay(truth, force, amico)
    print(f"\n{'metric':<8}{'SNR':>6}{'FORCE bias/MAE':>20}{'AMICO bias/MAE':>20}")
    for k, lab, _ in METRICS:
        for s in SNRS:
            fb, fm = bm(force[s][k], truth[k])
            row = f"{k:<8}{str(s):>6}{f'{fb:+.3f}/{fm:.3f}':>20}"
            if s in amico:
                ab, am = bm(amico[s][k], truth[k]); row += f"{f'{ab:+.3f}/{am:.3f}':>20}"
            print(row)

    nat = load_native()
    if nat:
        gt, methods, mk_src = nat
        native_fig(gt, methods, mk_src)
        print(f"\nDTI/DKI-native metrics ({', '.join(methods)}); MK truth: {mk_src}")
        print(f"{'metric':<8}{'SNR':>6}" + "".join(f"{n+' MAE':>14}" for n in methods))
        for k, lab, _, sc in NATIVE:
            for s in SNRS:
                row = f"{k:<8}{str(s):>6}"
                for n, d in methods.items():
                    if s in d and k in d[s]:
                        _, m = bm(np.asarray(d[s][k])*sc, np.asarray(gt[k])*sc)
                        row += f"{m:>14.3f}"
                    else:
                        row += f"{'-':>14}"
                print(row)


if __name__ == "__main__":
    main()
