import sys, importlib.util
import numpy as np

WT = "/home/athshah/force_paper_experiments/angular_accuracy_mc/dipy_pr2962"
HERE = "/home/athshah/force_paper_experiments/angular_accuracy_mc"
import dipy.data
if not hasattr(dipy.data, "Sphere"):
    from dipy.core.sphere import Sphere as _S
    dipy.data.Sphere = _S

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

def main():
    load("dipy.core.dsi_sphere", f"{WT}/dipy/core/dsi_sphere.py")
    odffp = load("odffp_pr", f"{WT}/dipy/reconst/odffp.py")
    from dipy.core.gradients import gradient_table
    d = np.load(f"{HERE}/data/nufo.npz")
    sig = d["signals"].astype(float)
    bvals, bvecs = d["bvals"].astype(float), d["bvecs"].astype(float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    N = sig.shape[0]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    print("generating ODF-FP dict (500k) ...", flush=True)
    od = odffp.OdffpDictionary(gtab); od.generate(dict_size=500000, max_peaks_num=3)
    print("fitting ...", flush=True)
    fit = odffp.OdffpModel(gtab, od).fit(sig.reshape(N, 1, 1, -1),
                                         mask=np.ones((N, 1, 1), bool), penalty=1e-5)
    dict_idx = np.asarray(fit._dict_idx).reshape(N)
    amp = np.asarray(od.ratio)[1:, dict_idx].T

    cnt = np.zeros(N, np.int8)
    for i in range(N):
        v = np.nan_to_num(amp[i]); mx = v.max()
        cnt[i] = int((v >= 0.5 * mx).sum()) if mx > 0 else 0
    np.savez(f"{HERE}/out/nufo_counts_odffp.npz", counts=cnt,
             true_N=d["true_N"], snr=d["snr"])
    for nn in (1, 2, 3):
        print(f"  true N={nn}: mean reported {cnt[d['true_N']==nn].mean():.2f}")
    print("saved out/nufo_counts_odffp.npz")

if __name__ == "__main__":
    main()
