# Synthetic angular-resolution and fiber-count experiment

`simulation_with_dispersion.ipynb` reproduces the synthetic experiment of paper
Section 2.4 (Fig. 6f,g) using the DIPY FORCE API.

It simulates, on the Stanford HARDI single-shell scheme (b = 2000 s/mm², 150
directions):

* **two-fiber crossings** across crossing-angle bins (10°–80°), and
* **single / two / three-fiber** voxels,

each with intra (stick) + extra (zeppelin) compartments, Watson dispersion, and
Rician noise at SNR 10/20/50. FORCE peaks (`dipy.reconst.force`, best match) are
compared with CSA, GQI and CSD for angular resolution (fraction of crossings
resolved within 20°) and fiber-count (NUFO) accuracy.

The committed outputs come from a fast reduced run (`N_PER_BIN = 120`, 150k-entry
library). Increase `N_PER_BIN` and `num_simulations` (the paper used ~2000 voxels
per bin and a 500k library) for smoother curves; the qualitative result —
FORCE leading at shallow crossings and counting multi-fiber voxels most
accurately — is already visible.
