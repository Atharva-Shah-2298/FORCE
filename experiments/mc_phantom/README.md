# Monte Carlo (true-physics) diffusion phantom

FORCE recovery tested against a phantom whose signal comes from an **actual
random-walk diffusion simulation** (GPU, [disimpy](https://github.com/kerkelae/disimpy))
inside packed impermeable cylinders — not from any analytic compartment model.
The signal carries true restricted (intra-axonal) and hindered (extra-axonal)
diffusion physics, so recovering microstructure with FORCE is an honest,
out-of-model test rather than a self-consistent forward-model check.

The full experiment catalogue, ground-truth definitions, and every result table
are in [`SUMMARY.md`](SUMMARY.md). This file covers only how to install the
dependencies and run it.

## Requirements

This experiment needs a machine the rest of the repo does not:

- **An NVIDIA GPU with CUDA.** The Monte Carlo walk runs on the GPU via
  `numba.cuda`; there is no CPU fallback. The phantoms in the paper were run on
  an RTX A6000 (~7 min per coherent grid).
- **disimpy** (0.3.0), which pulls in `numba` and the CUDA toolchain.
- **DIPY with FORCE** (`dipy.reconst.force`, DIPY ≥ 1.13) for the matching step.
- **AMICO** (`dmri-amico`), optional — only for the FORCE-vs-NODDI baseline
  (`run_amico_bio.py`).

Nothing here runs in CI, and no simulation output is committed — the phantom
substrates and signals are yours to regenerate.

## Installation

disimpy is GPU-specific, so install it into an environment that already sees
your CUDA toolkit:

```bash
pip install disimpy            # or: pip install git+https://github.com/kerkelae/disimpy
python -c "from numba import cuda; print(cuda.detect())"   # confirm the GPU is visible
```

If `cuda.detect()` finds no device, fix that before going further — the scripts
will not fall back to CPU.

**NumPy ≥ 2 note.** disimpy 0.3.0 still calls a few removed NumPy aliases
(`np.trapz`, `np.product`, …). [`np2_shim.py`](np2_shim.py) restores them and is
imported before disimpy in every generator, so no disimpy patching is needed —
just keep the shim next to the scripts.

DIPY (with the FORCE reconstruction) and, optionally, AMICO:

```bash
pip install "dipy>=1.13"
pip install dmri-amico          # optional, baseline only
```

## What you must supply

The generators simulate on a **real acquisition protocol** rather than a toy
scheme. Each script points at a protocol directory:

```python
PROTO = "/home/athshah/Phi/165840"   # edit this
```

That directory must hold two FSL-format text files, `bvals` and `bvecs` (the
paper used subject 165840's 288-volume, b = 0/1/2/3k protocol). Point `PROTO` at
your own `bvals`/`bvecs` and the phantom is simulated on your scheme. The
HCP-slice scripts (`prior_slice72.py`) additionally expect that subject's DWI
volume; adapt the paths at the top of those files to your data.

## Running it

The core coherent-phantom pipeline (experiment 1 in `SUMMARY.md`):

```bash
python validate_substrate.py     # optional: proves the substrate is truly restricted
python generate_mc_phantom.py    # GPU random walk over an ICVF x radius x free-water grid
python run_force_mc.py           # FORCE retrieval (K=50, beta=2000) on the MC signals
python analyze_mc.py             # bias / MAE + calibration vs substrate ground truth
```

Every other experiment follows the same `generate_* -> run_force_* -> analyze_*`
shape. The mapping from script to experiment and expected result is the table in
[`SUMMARY.md`](SUMMARY.md); in brief:

| substrate generator | FORCE runner | analysis |
| --- | --- | --- |
| `packed_cylinders.py` (single fiber) | `run_force_mc.py` | `analyze_mc.py` |
| `generate_dispersed_phantom.py` (Watson fanning) | `run_force_disp.py` | in-script |
| `generate_joint_phantom.py` (NDI×ODI×FW) | `run_force_joint.py` | `analyze_joint.py` |
| `generate_tissue_substrates.py` (WM/GM/CSF, `packed_spheres.py` soma) | `run_force_tissue.py` | `analyze_tissue.py` |
| `generate_crossing_bending.py` (`crossing_bending.py`) | `run_force_cb.py` | `analyze_cb.py` |
| `bio_phantom.py` (N=400 biological range) | `run_force_bio.py`, `run_amico_bio.py` | `analyze_bio.py` |
| `radius_effect.py` (axon-radius sweep) | — | `analyze_radius_effect.py` |

All scripts read and write inside this directory (`data/`, `force_out/`,
`bio_out/`, `figures/`, …). Those output directories are git-ignored and are not
part of the repository — they are recreated on first run.
