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
    from dipy.data import get_sphere
    from dipy.core.gradients import gradient_table
    load("dipy.core.dsi_sphere", f"{WT}/dipy/core/dsi_sphere.py")
    odffp = load("odffp_pr", f"{WT}/dipy/reconst/odffp.py")

    CV = get_sphere(name="symmetric362").vertices
    sel = np.load(f"{HERE}/out/viz_sel.npy")
    d = np.load(f"{HERE}/data/crossings.npz")
    s10 = d["signals"][sel].astype(float)
    bvals, bvecs = d["bvals"].astype(float), d["bvecs"].astype(float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    print("generating ODF-FP dict (500k) ...", flush=True)
    od = odffp.OdffpDictionary(gtab); od.generate(dict_size=500000, max_peaks_num=3)
    fit = odffp.OdffpModel(gtab, od).fit(s10.reshape(10, 1, 1, -1),
                                         mask=np.ones((10, 1, 1), bool), penalty=1e-5)
    odf_half = np.asarray(fit._odf).reshape(10, -1)
    tess = np.asarray(od.tessellation.vertices)[:odf_half.shape[1]]
    sim = np.abs(CV @ tess.T)
    idx = np.argmax(sim, axis=1)
    odf = odf_half[:, idx]
    peaks = np.asarray(fit._peak_dirs).reshape(10, -1, 3)
    np.savez(f"{HERE}/out/viz_odffp.npz",
             odf_ODFFP=odf.astype(np.float32), pk_ODFFP=peaks.astype(np.float32))
    print("saved out/viz_odffp.npz")

if __name__ == "__main__":
    main()
