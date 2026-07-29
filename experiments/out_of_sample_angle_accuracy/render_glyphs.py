import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
dev = np.load(f"{HERE}/out/viz_dev.npz")
odf_fp = np.load(f"{HERE}/out/viz_odffp.npz")
V, F = dev["verts"].astype(float), dev["faces"]
a1, a2, tang = dev["a1"], dev["a2"], dev["true_angle"]
NV = a1.shape[0]

ODF = {"CSA": dev["odf_CSA"], "CSD": dev["odf_CSD"], "GQI": dev["odf_GQI"],
       "ODFFP": odf_fp["odf_ODFFP"], "FORCE": dev["odf_FORCE"]}
PK = {"CSA": dev["pk_CSA"], "CSD": dev["pk_CSD"], "GQI": dev["pk_GQI"],
      "ODFFP": odf_fp["pk_ODFFP"], "FORCE": dev["pk_FORCE"]}
ROWS = ["CSA", "CSD", "GQI", "ODFFP", "FORCE"]
RCOL = {"CSA": "#ff7f0e", "CSD": "#1f77b4", "GQI": "#8c564b",
        "ODFFP": "#9467bd", "FORCE": "#2ca02c"}

def align_rot(u, v):
    n = np.cross(u, v); n = n / (np.linalg.norm(n) + 1e-9)
    b = u + v; b = b / (np.linalg.norm(b) + 1e-9)
    e3 = n
    e1 = b - (b @ e3) * e3; e1 /= (np.linalg.norm(e1) + 1e-9)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3])

def draw_glyph(ax, odf, R):
    o = odf.astype(float)
    o = (o - o.min()) / (o.max() - o.min() + 1e-9)
    pts = (R @ (o[:, None] * V).T).T
    polys = pts[F]
    fa = o[F].mean(1)
    pc = Poly3DCollection(polys, facecolors=plt.cm.viridis(fa),
                          edgecolors="none", linewidths=0, shade=False)
    ax.add_collection3d(pc)

def draw_lines(ax, dirs, R, color, ls, lw, L=1.25):
    for p in dirs:
        if np.linalg.norm(p) < 1e-6:
            continue
        q = R @ (p / np.linalg.norm(p))
        ax.plot([-L*q[0], L*q[0]], [-L*q[1], L*q[1]], [-L*q[2], L*q[2]],
                color=color, ls=ls, lw=lw, zorder=10)

fig = plt.figure(figsize=(26, 14))
for r, meth in enumerate(ROWS):
    for c in range(NV):
        ax = fig.add_subplot(len(ROWS), NV, r * NV + c + 1, projection="3d")
        R = align_rot(a1[c], a2[c])
        draw_glyph(ax, ODF[meth][c], R)

        pk = PK[meth][c].reshape(-1, 3)
        nz = np.linalg.norm(pk, axis=1) > 1e-6
        draw_lines(ax, pk[nz], R, "crimson", "-", 2.4)
        draw_lines(ax, np.stack([a1[c], a2[c]]), R, "0.35", "--", 1.3, L=1.4)
        ax.view_init(elev=90, azim=-90)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        if r == 0:
            ax.set_title(f"{int(tang[c])}°", fontsize=13, fontweight="bold", pad=0)

    ypos = 1 - (r + 0.5) / len(ROWS)
    fig.text(0.055, ypos, meth, ha="right", va="center", fontsize=15,
             fontweight="bold", color=RCOL[meth])

fig.suptitle("ODF glyphs + detected peaks (crimson) vs ground-truth fibres (dashed) "
             "for 10 crossings below 40°, SNR 50  —  top-down view of the crossing plane",
             fontsize=15, fontweight="bold", y=0.98)
fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.01, wspace=0.02, hspace=0.05)
fig.savefig(f"{HERE}/figures/glyphs_below40.png", dpi=150, facecolor="white")
print("saved figures/glyphs_below40.png")
