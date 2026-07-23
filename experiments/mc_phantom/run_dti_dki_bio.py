"""DTI and DKI on the biological MC phantom, plus FORCE's own tensor/kurtosis
scalars, for a fair signal-representation comparison.

DTI and DKI do NOT estimate NODDI parameters (NDI/ODI/FW); they yield tensor
scalars (FA/MD/RD) and kurtosis (MK). So we compare on their NATIVE metrics:
    ground truth  = DTI(b<=1000) / DKI(full) fit on the CLEAN signal
                    (same convention used elsewhere in this experiment)
    estimate(SNR) = the same fit on the noisy volume at that SNR

Reads the identical NIfTI files AMICO/FORCE use, so every method fits the same
noisy data."""
import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.dki import DiffusionKurtosisModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_bio")
OUT = os.path.join(HERE, "bio_out")
SNRS = ["clean", 50, 20, 10]
DTI_BMAX = 1050.0          # single-tensor regime (mirrors phantom FA/MD/RD truth)
K, BETA = 50, 2000.0       # FORCE settings, identical to run_force_bio.py


def load_dwi(snr):
    return nib.load(os.path.join(DATA, f"dwi_{snr}.nii.gz")).get_fdata().reshape(
        -1, len(np.loadtxt(os.path.join(DATA, "bvals")).ravel())).astype(np.float64)


def fit_dti(sig, gtab_dti, dti_cols):
    fit = TensorModel(gtab_dti).fit(sig[:, dti_cols])
    return {"fa": np.asarray(fit.fa, np.float32),
            "md": np.asarray(fit.md, np.float32),
            "rd": np.asarray(fit.rd, np.float32)}


def fit_dki(sig, gtab):
    fit = DiffusionKurtosisModel(gtab).fit(sig)
    return {"fa": np.asarray(fit.fa, np.float32),
            "md": np.asarray(fit.md, np.float32),
            "rd": np.asarray(fit.rd, np.float32),
            "mk": np.asarray(fit.mk(min_kurtosis=0, max_kurtosis=3), np.float32),
            "ak": np.asarray(fit.ak(min_kurtosis=0, max_kurtosis=3), np.float32),
            "rk": np.asarray(fit.rk(min_kurtosis=0, max_kurtosis=3), np.float32)}


def _find_dki_library(gtab):
    """Locate a cached FORCE library that exactly matches this gtab AND already
    contains DKI (mk). The plain cache lookup ignores compute_dki and may return
    a DTI-only library, so we search the registry ourselves."""
    import glob
    import json
    b = gtab.bvals.astype(float); v = gtab.bvecs.astype(float)
    cdir = os.path.expanduser("~/.dipy/force_simulations")
    reg_path = os.path.join(cdir, "cache_registry.json")
    if not os.path.exists(reg_path):
        return None
    reg = json.load(open(reg_path))
    for e in reg:
        rb = np.array(e.get("bvals", []))
        rv = np.array(e.get("bvecs", []))
        if rb.shape != b.shape or rv.shape != v.shape:
            continue
        if not (np.allclose(rb, b, atol=1e-3) and np.allclose(rv, v, atol=1e-4)):
            continue
        p = os.path.join(cdir, e["filename"])
        if os.path.exists(p) and "mk" in np.load(p).files:
            return p
    return None


def force_scalars(gtab):
    """Weighted FORCE tensor+kurtosis scalars per voxel, same neighbour search
    and softmax weighting as run_force_bio.py, but for fa/md/rd/mk."""
    from dipy.reconst.force import FORCEModel, softmax_stable
    model = FORCEModel(gtab, n_neighbors=K, use_posterior=True, posterior_beta=BETA,
                       compute_odf=False, verbose=True)
    dki_lib = _find_dki_library(gtab)
    if dki_lib is not None:
        print(f"[FORCE] loading DKI library {os.path.basename(dki_lib)}")
        model.load(dki_lib)
    else:
        print("[FORCE] no cached DKI library matched; generating (compute_dki=True)")
        model.generate(num_simulations=500000, use_cache=False, num_cpus=-1,
                       compute_dti=True, compute_dki=True)
    sims = model.simulations
    have_mk = "mk" in sims
    fields = ("fa", "md", "rd") + (("mk", "ak", "rk") if have_mk else ())

    def per_snr(snr):
        vol = nib.load(os.path.join(DATA, f"dwi_{snr}.nii.gz")).get_fdata()
        Q = vol.reshape(-1, vol.shape[-1]).astype(np.float64)
        qn = np.linalg.norm(Q, axis=1, keepdims=True); qn[qn == 0] = 1
        Dist, neigh = model._index.search(
            np.ascontiguousarray((Q / qn).astype(np.float32)), k=K)
        W = softmax_stable(BETA * (Dist - model._penalty_array[neigh]), axis=1).astype(np.float32)
        return {f: np.einsum('nk,nk->n', W, sims[f][neigh]).astype(np.float32) for f in fields}

    return per_snr, have_mk


def main():
    os.makedirs(OUT, exist_ok=True)
    bvals = np.loadtxt(os.path.join(DATA, "bvals")).ravel().astype(float)
    bvecs = np.loadtxt(os.path.join(DATA, "bvecs"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    dti_cols = np.where(bvals <= DTI_BMAX)[0]
    gtab_dti = gradient_table(bvals[dti_cols], bvecs=bvecs[dti_cols], b0_threshold=50)

    # --- classical models on every SNR ---
    dti, dki = {}, {}
    for snr in SNRS:
        sig = load_dwi(snr)
        dti[snr] = fit_dti(sig, gtab_dti, dti_cols)
        dki[snr] = fit_dki(sig, gtab)
        np.savez(os.path.join(OUT, f"dti_{snr}.npz"), **dti[snr])
        np.savez(os.path.join(OUT, f"dki_{snr}.npz"), **dki[snr])
        print(f"[DTI/DKI {snr}] FA {np.nanmean(dti[snr]['fa']):.3f} "
              f"MD {np.nanmean(dti[snr]['md'])*1e3:.3f} "
              f"MK {np.nanmean(dki[snr]['mk']):.3f}")

    # --- native ground truth = clean-signal fits (FA/MD/RD via DTI, MK via DKI) ---
    gt = {"fa": dti["clean"]["fa"], "md": dti["clean"]["md"],
          "rd": dti["clean"]["rd"], "mk": dki["clean"]["mk"]}
    np.savez(os.path.join(OUT, "gt_dtidki.npz"), **gt)

    # --- FORCE tensor/kurtosis scalars (same weighting as the NODDI figure) ---
    try:
        per_snr, have_mk = force_scalars(gtab)
        for snr in SNRS:
            fs = per_snr(snr)
            np.savez(os.path.join(OUT, f"force_dtidki_{snr}.npz"), **fs)
            print(f"[FORCE scalars {snr}] FA {np.nanmean(fs['fa']):.3f} "
                  f"MD {np.nanmean(fs['md'])*1e3:.3f}"
                  + (f" MK {np.nanmean(fs['mk']):.3f}" if have_mk else " (no MK)"))
        if not have_mk:
            print("WARNING: FORCE library lacks DKI (mk); MK row will omit FORCE.")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: FORCE scalar extraction failed ({e}); native figure will "
              f"show DTI/DKI only.")


if __name__ == "__main__":
    main()
