# Microstructure recovery on a Monte Carlo (true-physics) phantom

FORCE microstructure recovery compared against **DTI, DKI and AMICO-NODDI** on a
phantom whose signal comes from an actual GPU random-walk diffusion simulation
([disimpy](https://github.com/kerkelae/disimpy)) inside packed impermeable
cylinders — not from any analytic compartment model. Because the signal carries
true restricted (intra-axonal) and hindered (extra-axonal) physics that no
forward model produced, recovering microstructure from it is an honest,
out-of-model test rather than a self-consistent one.

**Phantom.** N = 400 single-fiber white-matter voxels sampled across healthy
adult ranges (neurite density / ICVF ∈ [0.40, 0.72], ODI ∈ [0.03, 0.30], free
water ∈ [0, 0.20], random fiber orientation, axon radius 1 µm). The signal is a
Monte-Carlo cylinder kernel interpolated over ICVF, Watson-convolved for
dispersion, plus a free-water compartment, then corrupted with Rician noise at
SNR {clean, 50, 20, 10}. The same NIfTI is fitted by every method.

## Requirements

This experiment needs a machine the rest of the repo does not:

- **An NVIDIA GPU with CUDA** — the Monte-Carlo walk runs on the GPU via
  `numba.cuda`; there is no CPU fallback.
- **disimpy** (0.3.0), which pulls in `numba` and the CUDA toolchain.
- **DIPY with FORCE** (`dipy.reconst.force`, DIPY ≥ 1.13).
- **AMICO** (`dmri-amico`) for the NODDI baseline.

No simulation output is committed — the phantom signals are yours to regenerate.

## Installation

disimpy is GPU-specific, so install it into an environment that already sees
your CUDA toolkit:

```bash
pip install disimpy            # or: pip install git+https://github.com/kerkelae/disimpy
python -c "from numba import cuda; print(cuda.detect())"   # confirm the GPU is visible
```

If `cuda.detect()` finds no device, fix that first — the scripts will not fall
back to CPU.

**NumPy ≥ 2 note.** disimpy 0.3.0 still calls a few removed NumPy aliases
(`np.trapz`, `np.product`, …). [`np2_shim.py`](np2_shim.py) restores them and is
imported before disimpy, so keep the shim next to the scripts.

DIPY (with FORCE) and AMICO:

```bash
pip install "dipy>=1.13" dmri-amico
```

## What you must supply

The phantom is simulated on a **real acquisition protocol**, so supply an
**HCP-like subject** — a directory holding two FSL-format text files, `bvals`
and `bvecs` (the paper used a 288-volume, b = 0/1/2/3k scheme; any multi-shell
HARDI protocol works). The scripts read it from the `FORCE_PROTO` environment
variable, defaulting to `hcp_subject/` next to the scripts:

```bash
export FORCE_PROTO=/path/to/your/hcp_subject
```

## Running it

```bash
python bio_phantom.py       # GPU: generate the N=400 phantom + noise -> NIfTI (data_bio/)
python run_force_bio.py     # FORCE   (K=50, beta=2000)
python run_amico_bio.py     # AMICO-NODDI baseline
python run_dti_dki_bio.py   # DTI + DKI baseline
python analyze_bio.py       # bias / MAE tables + ground-truth-vs-estimate scatter
```

Neurite density is compared on the **voxel scale** — the intra-neurite fraction
of the whole voxel, the only fair common basis across methods. `FORCE.nd` is
already voxel-scale; NODDI's `v_ic` is within-tissue, so its voxel neurite
density is `v_ic * (1 - ISOVF)`.

## Result (from the paper)

Bias / MAE against the substrate ground truth:

| metric | SNR | FORCE | AMICO-NODDI |
| --- | --- | --- | --- |
| neurite density | clean | −0.012 / **0.018** | −0.094 / 0.094 |
| neurite density | 20 | +0.011 / **0.020** | −0.070 / 0.070 |
| ODI | clean | −0.002 / **0.010** | −0.036 / 0.036 |
| ODI | 20 | +0.010 / **0.020** | −0.026 / 0.027 |
| free water | clean | +0.110 / **0.114** | +0.206 / 0.206 |
| free water | 20 | +0.105 / **0.110** | +0.222 / 0.222 |

FORCE is more accurate on all three at usable SNR — roughly 5× lower neurite-
density error than AMICO-NODDI (MAE 0.018 vs 0.094), because AMICO's fixed
diffusivities and tortuosity constraint push the model mismatch into an
over-estimated free-water fraction.

All scripts read and write inside this directory (`data_bio/`, `bio_out/`);
those output directories are git-ignored and recreated on first run.
