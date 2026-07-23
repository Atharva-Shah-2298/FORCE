"""FORCE recovery on true-MC crossing and bending substrates."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
g = np.load(os.path.join(HERE, "data_cb", "signals_cb.npz"))
f = dict(np.load(os.path.join(HERE, "force_out", "force_cb.npz")))
names = g["names"].astype(str); kind = g["kind"].astype(str)
wm = f["wm_fraction"]
ndi = np.clip(f["nd"] / np.maximum(wm, 1e-6), 0, 1)
odi = np.clip(1 - (1 - f["dispersion"]) / np.maximum(wm, 1e-6), 0, 1)

print("=" * 78)
print("FORCE on true-MC CROSSING & BENDING substrates")
print("=" * 78)
print("\n--- CROSSING (true interpenetrating, GT: 2 fibers @ 90deg) ---")
print(f"{'name':<12}{'GTfib':>6}{'peaks':>6}{'cross_est':>10}{'ICVF':>6}"
      f"{'nd':>6}{'NDIwm':>7}{'ODIwm':>7}")
for i in np.where(kind == "crossing")[0]:
    print(f"{names[i]:<12}{int(g['num_fibers'][i]):>6}{int(f['n_peaks'][i]):>6}"
          f"{f['cross_deg'][i]:>10.0f}{g['icvf'][i]:>6.2f}{f['nd'][i]:>6.2f}"
          f"{ndi[i]:>7.2f}{odi[i]:>7.2f}")

print("\n--- BENDING (undulation; GT: 1 fiber, curvature -> orientation spread) ---")
print(f"{'name':<12}{'maxTilt':>8}{'peaks':>6}{'ICVF':>6}{'nd':>6}"
      f"{'NDIwm':>7}{'ODIwm(recov)':>13}")
for i in np.where(kind == "bending")[0]:
    print(f"{names[i]:<12}{g['max_tilt_deg'][i]:>8.0f}{int(f['n_peaks'][i]):>6}"
          f"{g['icvf'][i]:>6.2f}{f['nd'][i]:>6.2f}{ndi[i]:>7.2f}{odi[i]:>13.3f}")
print("\nKey checks: crossing -> 2 peaks near 90deg; bending -> ODI rises with "
      "curvature (max tilt) while staying 1 peak.")
