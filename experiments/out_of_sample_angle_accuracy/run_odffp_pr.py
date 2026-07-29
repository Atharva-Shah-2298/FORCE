import sys, os, importlib.util
import numpy as np

WT = "/home/athshah/force_paper_experiments/angular_accuracy_mc/dipy_pr2962"
HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
TOL_MATCH = 20.0

import dipy.data
if not hasattr(dipy.data, "Sphere"):
    from dipy.core.sphere import Sphere as _S
    dipy.data.Sphere = _S

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

def ang(u, v):
    return np.rad2deg(np.arccos(np.clip(abs(float(np.dot(u, v))), 0, 1)))

def score(peaks, a1, a2):
    p = np.asarray(peaks, float).reshape(-1, 3)
    ok = np.isfinite(p).all(1) & (np.linalg.norm(np.nan_to_num(p), axis=1) > 1e-6)
    p = p[ok]
    if p.shape[0] < 2:
        return p.shape[0], np.nan, False, np.nan
    a1e = np.array([ang(q, a1) for q in p]); a2e = np.array([ang(q, a2) for q in p])
    i1, i2 = int(np.argmin(a1e)), int(np.argmin(a2e))
    if i1 == i2:
        order = np.argsort(a2e)
        i2 = int(order[1]) if len(order) > 1 else i2
    e1, e2 = a1e[i1], a2e[i2]
    resolved = (e1 <= TOL_MATCH) and (e2 <= TOL_MATCH) and (i1 != i2)
    return int(p.shape[0]), 0.5 * (e1 + e2), bool(resolved), ang(p[i1], p[i2])

def main():
    load("dipy.core.dsi_sphere", f"{WT}/dipy/core/dsi_sphere.py")
    odffp = load("odffp_pr", f"{WT}/dipy/reconst/odffp.py")

    from dipy.core.gradients import gradient_table
    d = np.load(f"{HERE}/data/crossings.npz")
    sig = d["signals"].astype(float)
    bvals, bvecs = d["bvals"].astype(float), d["bvecs"].astype(float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    a1, a2, snr, true_ang = d["axis1"], d["axis2"], d["snr"], d["true_angle"]
    N = sig.shape[0]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    print("generating ODF-FP dictionary (500k) ...", flush=True)
    od = odffp.OdffpDictionary(gtab)
    od.generate(dict_size=500000, max_peaks_num=3)
    print(f"  dict odf {np.asarray(od.odf).shape}", flush=True)

    print("fitting ODF-FP (original PR) ...", flush=True)
    model = odffp.OdffpModel(gtab, od)
    fit = model.fit(sig.reshape(N, 1, 1, -1), mask=np.ones((N, 1, 1), bool), penalty=1e-5)
    peaks = np.asarray(fit._peak_dirs).reshape(N, -1, 3)
    dict_idx = np.asarray(fit._dict_idx).reshape(N)

    ratio = np.asarray(od.ratio)
    amp = ratio[1:, dict_idx].T
    ppv = np.asarray(od.peaks_per_voxel)[dict_idx]

    np.savez(f"{HERE}/out/peaks_ODF-FP.npz",
             peak_dirs=peaks.astype(np.float32),
             peak_vals=np.nan_to_num(amp).astype(np.float32),
             peaks_per_voxel=ppv.astype(np.int16))
    print(f"  saved out/peaks_ODF-FP.npz  peaks_per_voxel dist: "
          f"1:{np.mean(ppv==1):.2f} 2:{np.mean(ppv==2):.2f} 3:{np.mean(ppv>=3):.2f}")

if __name__ == "__main__":
    main()
