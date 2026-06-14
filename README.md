# FORCE: FORward modeling for Complex microstructure Estimation

<p align="center">
  <img src="docs/img/capabilities.png" alt="FORCE capabilities" width="760"/>
</p>

FORCE is a forward-modeling framework for diffusion MRI. Instead of inverting the
measured signal into a narrow parametric model, FORCE simulates a large library of
biologically plausible intra-voxel fiber configurations and tissue compositions,
then matches the measured signal of each voxel to its nearest simulations in
signal space. From that single fit it recovers the **full** spectrum of voxel-wise
diffusion metrics — diffusivity (FA, MD, RD), kurtosis (MK, AK, RK, KFA),
partial-volume fractions (WM, GM, CSF), dispersion (ODI), neurite density (NDI),
fiber orientations (peaks/ODFs), and a per-voxel uncertainty — without specialized
acquisition or extra scan time.

> **FORCE: FORward modeling for Complex microstructure Estimation.**
> Atharva Jaydeep Shah, Rafael Neto Henriques, Alonso Ramirez-Manzanares, Patryk
> Filipiak, Steven Baete, Kaustav Deka, Maharshi Gor, Serge Koudoro, Eleftherios
> Garyfallidis.
> Preprint: https://www.researchsquare.com/article/rs-8151109/v1

This repository contains the **experiment code and results for the paper**. FORCE
itself is part of **DIPY** and can be run from the Python API or the command line —
see below.

---

## FORCE is in DIPY

FORCE ships with [DIPY](https://dipy.org) (≥ 1.13) as `dipy.reconst.force`, with a
full tutorial here:
**https://docs.dipy.org/dev/examples_built/reconstruction/reconst_force.html**

```bash
pip install "dipy>=1.13"
```

### Python API

```python
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.io.peaks import save_pam

data, affine = load_nifti("dwi.nii.gz")
bvals, bvecs = read_bvals_bvecs("bvals", "bvecs")
gtab = gradient_table(bvals, bvecs=bvecs)
mask, _ = load_nifti("brain_mask.nii.gz")

# Build the model and its simulation library (cached after first build)
model = FORCEModel(gtab, n_neighbors=50, use_posterior=True, posterior_beta=2000.0)
model.generate(num_simulations=500000, num_cpus=-1, use_cache=True)

# Fit and read off the full metric suite from a single fit
fit = model.fit(data, mask=mask.astype(bool))
save_nifti("force_fa.nii.gz", fit.fa.astype(np.float32), affine)
save_nifti("force_md.nii.gz", fit.md.astype(np.float32), affine)
save_nifti("force_nd.nii.gz", fit.nd.astype(np.float32), affine)
save_nifti("force_dispersion.nii.gz", fit.dispersion.astype(np.float32), affine)
save_nifti("force_uncertainty.nii.gz", fit.uncertainty.astype(np.float32), affine)

# Fiber orientations for tractography
save_pam("force_peaks.pam5", force_peaks(fit, mask=mask.astype(bool)), affine=affine)
```

`fit` exposes `fa, md, rd, wm_fraction, gm_fraction, csf_fraction, num_fibers,
dispersion, nd, uncertainty, ambiguity` (and kurtosis maps when the library is
built with `compute_dki=True`).

### Command line

```bash
dipy_fit_force dwi.nii.gz bvals bvecs brain_mask.nii.gz \
    --out_dir force_out --num_simulations 500000 --num_cpus -1 \
    --save_metrics fa md rd nd dispersion wm_fraction gm_fraction csf_fraction \
    --out_peaks force_peaks.pam5 --verbose
```

Run `dipy_fit_force --help` for the full list of metrics and options.

---

## Results from the paper

### Microstructure on the Human Connectome Project (3T)

From one fit, FORCE reproduces standard DTI/DKI contrasts and yields stable,
anatomically consistent kurtosis. Notably it produced **zero** degenerate or
non-physical kurtosis estimates across white matter, where conventional weighted-
linear DKI fitting still yielded thousands of degenerate MK/RK voxels.

| DKI vs FORCE | NODDI: AMICO vs FORCE |
| --- | --- |
| <img src="docs/img/hcp_dki_vs_force.png" width="380"/> | <img src="docs/img/hcp_noddi_amico_vs_force.png" width="380"/> |

For NODDI parameters, FORCE gives sharper, less noisy orientation-dispersion (ODI)
and neurite-density (NDI) maps than AMICO (ODI correlation with inverted T1w
*r* = 0.93 vs 0.82; NDI *r* = 0.90 vs 0.88) and a cleaner free-water map.

### Single-shell Stanford HARDI and tractography

FORCE recovers plausible NODDI maps even from single-shell data, and produces
tractography peaks comparable to CSD/MSMT-CSD.

| NODDI from single-shell HARDI | Recobundles tracts (FORCE) |
| --- | --- |
| <img src="docs/img/stanford_hardi_noddi.png" width="380"/> | <img src="docs/img/bundles.png" width="300"/> |

### Stability, noise robustness and cross-scanner harmonization

FORCE estimates are stable to library resampling, input noise and its two
hyper-parameters (*K*, β), and it harmonizes better across scanners than DTI, DKI
and AMICO-NODDI (e.g. neurite-density ICC 0.877 vs 0.567 for AMICO; free-water
cross-scanner CoV 18.9% vs 36.2%).

| Results summary (paper Fig. 6) | Stability | Harmonization |
| --- | --- | --- |
| <img src="docs/img/results_summary.png" width="260"/> | <img src="docs/img/stability.png" width="260"/> | <img src="docs/img/harmonization.png" width="260"/> |

### Synthetic angular resolution and fiber-count accuracy

On simulated crossings, FORCE achieves the highest and most uniform peak-detection
rates across angle bins, with a clear advantage at **shallow crossings (10–40°)**
where CSA/GQI/CSD largely fail; it is also the most accurate at counting fibers in
multi-fiber voxels. These plots are reproduced by
[`experiments/synthetic_angle_accuracy/`](experiments/synthetic_angle_accuracy/).

| Angular resolution | Fiber-count (NUFO) accuracy |
| --- | --- |
| <img src="docs/img/angle_accuracy.png" width="380"/> | <img src="docs/img/nufo_accuracy.png" width="300"/> |

### Phantom, ex vivo and clinical data

| DiSCo phantom | Ex vivo mouse | Tumor (glioma) | Parkinson's (BUAN) |
| --- | --- | --- | --- |
| <img src="docs/img/disco_phantom.png" width="220"/> | <img src="docs/img/exvivo_mouse_noddi.png" width="170"/> | <img src="docs/img/tumor_noddi.png" width="170"/> | <img src="docs/img/buan_parkinsons.png" width="190"/> |

On the DiSCo phantom FORCE matches or exceeds CSD-family methods at low SNR
(connectivity correlation 0.868 vs 0.858 for MSMT-CSD at SNR 10). It also produces
clean DTI/NODDI maps on ex-vivo mouse and human brain and on clinical glioma and
PPMI Parkinson's scans.

<p align="center">
  <img src="docs/img/concept.png" alt="FORCE pipeline" width="700"/>
</p>

---

## Paper experiments

All experiments live under [`experiments/`](experiments/) and run FORCE through the
DIPY API. See [`experiments/README.md`](experiments/README.md) for the full index.

| Experiment | What it shows | Paper |
| --- | --- | --- |
| [`scripts/exp1_neighborhood.py`](experiments/scripts/exp1_neighborhood.py) | Posterior neighborhood concentration | Fig. 6 / stability |
| [`scripts/exp2_library_resampling.py`](experiments/scripts/exp2_library_resampling.py) | Stability to library resampling | stability |
| [`scripts/exp3_input_noise.py`](experiments/scripts/exp3_input_noise.py) | Robustness to input noise | Fig. 6a |
| [`scripts/exp4_k_sweep.py`](experiments/scripts/exp4_k_sweep.py) | Sensitivity to *K* | stability |
| [`scripts/exp5_beta_sweep.py`](experiments/scripts/exp5_beta_sweep.py) | Sensitivity to softmax β | stability |
| [`scripts/exp6_split_half.py`](experiments/scripts/exp6_split_half.py) | Split-half library consistency | stability |
| [`scripts/exp7_dti_reference.py`](experiments/scripts/exp7_dti_reference.py) | FORCE vs DTI noise CV | Fig. 6a |
| [`scripts/exp8_dictionary_adequacy.py`](experiments/scripts/exp8_dictionary_adequacy.py) | Held-out predictive dictionary adequacy | §2.1 |
| [`recovery/`](experiments/recovery/) | Parameter recovery vs a separated generator | validation |
| [`harmonization/`](experiments/harmonization/) | Cross-scanner reproducibility (ICC, CoV) | Fig. 6b,c |
| [`synthetic_angle_accuracy/`](experiments/synthetic_angle_accuracy/) | Angular resolution + fiber count | Fig. 6f,g |

```
.
├── README.md
├── requirements.txt
├── docs/img/                  # figures used in this README
└── experiments/
    ├── README.md              # experiment index
    ├── config.yaml            # data paths + parameters (edit for your data)
    ├── run_all.sh             # run the stability experiments in order
    ├── scripts/               # exp1–exp8
    ├── recovery/              # parameter-recovery experiment
    ├── harmonization/         # cross-scanner reproducibility
    └── synthetic_angle_accuracy/   # angular-resolution / NUFO notebook
```

---

## Citation

```bibtex
@article{shah2025force,
  title   = {FORCE: FORward modeling for Complex microstructure Estimation},
  author  = {Shah, Atharva Jaydeep and Neto Henriques, Rafael and
             Ramirez-Manzanares, Alonso and Filipiak, Patryk and Baete, Steven and
             Deka, Kaustav and Gor, Maharshi and Koudoro, Serge and
             Garyfallidis, Eleftherios},
  year    = {2025},
  note    = {Preprint, Research Square, rs-8151109/v1},
  url     = {https://www.researchsquare.com/article/rs-8151109/v1}
}
```
