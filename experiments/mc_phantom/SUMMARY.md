# FORCE validation on true Monte-Carlo diffusion phantoms — full summary

A suite of **Monte-Carlo random-walk diffusion phantoms** (GPU, disimpy) used to
test FORCE's microstructure recovery against ground-truth substrate geometry.
The signal comes from spins diffusing in geometric substrates with membrane
reflection (PGSE phase accumulation), so it carries **true restricted +
hindered physics that no analytic model produced** — an honest, out-of-model
test (unlike a forward-model phantom, which is circular). All runs use subject
165840's real 288-direction protocol (b = 0/1/2/3k).

Engine: disimpy 0.3.0 (numba.cuda) on RTX A6000; conda env `skyline`
(`/home/athshah/miniconda3/envs/skyline/bin/python`). SI units; `np2_shim.py`
restores numpy<2 aliases disimpy needs. AMICO baseline runs in `base` env.

---

## Validation of the engine / substrates
* Free diffusion recovers exp(-bD) to **<0.2%** (`validate_substrate.py`).
* Packed cylinders give **true restriction**: perpendicular ADC collapses
  0.40->0.07 (x1e-3) with b, signal above the free-Gaussian curve.
* GM soma (packed spheres): isotropic, ADC ~0.84e-3 (validated).
* ICVF kernel interpolation validated vs direct simulation to **0.001** signal err.
* Crossing: S_x≈S_y, S_z (⊥ both) highest. Fanning: S_x stays parallel,
  anisotropy drops with ODI (unimodal). Undulation: bimodal (≠ fanning).

---

## Experiments and key results

| # | Experiment | Substrate (true MC) | Headline result | Figure |
|---|---|---|---|---|
| 1 | Coherent NDI/ODI/FW | packed cylinders, ICVF 0.3-0.7 x r{2,4}um x FW{0,0.2} | ND MAE **0.029**; ODI(anisotropic) MAE 0.035; residual = isotropic-fraction leak +0.30 | mc_ndi_vs_icvf, mc_calibration |
| 2 | Dispersion sweep | coherent kernel + Watson convolution, ICVF{0.4,0.6} | ODI recovery MAE **0.017** (clean); saturates at dict ceiling 0.30 | disp_recovery |
| 3 | Joint NDI×ODI×FW | factorial, all vary together | metrics recovered **independently** (cross-talk ~0); FW/iso over-assigned | joint_calibration |
| 4 | Five tissue types | WM 1/2/3-fiber, WM+GM, GM+CSF (cylinders + soma spheres) | fiber count ok (fails 45° crossing); CSF ok; **GM soma badly under-recovered** (absorbed into WM) | tissue_recovery |
| 5 | Axon-radius sweep | r{0.5,1,2,4}um, ICVF 0.6 | iso-leak **flat** across radius -> not a radius/restriction effect | radius_effect |
| 6 | Dictionary-prior probe | reprior 500K dict to flat/wm-heavy | removes spurious GM (0.13->0.01) but total iso unchanged -> prior not the lever | (reprior_test stdout) |
| 7 | Phantom vs real HCP CC | match to real corpus callosum | leak was a **phantom-density artifact**: at real CC density ICVF 0.70 phantom matches real CC (WM 0.88 vs 0.89) | phantom_vs_hcp_cc |
| 8 | FORCE vs AMICO-NODDI | N=400 biological-range voxels, SNR{clean,50,20,10} | **FORCE wins on all 3**: ND MAE 0.018 vs 0.094; ODI 0.010 vs 0.036; FW +0.11 vs +0.21 | bio_scatter_overlay |
| 9 | Prior widening on HCP | [0.6,0.9]->[0.3,0.9] f_intra, slice z=72 | barely changes anything (mean PVE shift ≤0.02, ODF r=0.978) | prior_pve_slice72 |
| 10 | True crossing | interpenetrating orthogonal cylinders | recovers **2 peaks at 84°**, ND 0.31 vs ICVF 0.32 | crossing_substrate |
| 11 | Bending (undulation) | sinusoidal cylinders (bimodal) | partly read as dispersion but **under-estimated**; peaks->0, ND under-estimated | bending_recovery |
| 12 | True fanning, biological | Watson-kernel, ICVF{0.4,0.6}, ODI≤0.25 | **on identity, MAE 0.011** — FORCE recovers fanning across the real WM range | fanning_biological |

---

## Headline findings

**What FORCE recovers well (biological regime, usable SNR):**
* **Neurite density** — voxel ND tracks true ICVF, MAE ~0.02, robust to axon
  radius and SNR. (`FORCE.nd = f_wm × within-WM NDI`, confirmed at source
  `sim_core.pyx:517` — compare it directly, do NOT divide by the leaky
  wm_fraction.) Beats AMICO‑NODDI (0.018 vs 0.094).
* **Orientation dispersion (fanning)** — MAE **0.011** across the biological WM
  range (ODI ≤ 0.25). The 0.30 dictionary ceiling never bites: real WM fiber
  ODI is low (median 0.19, 0% of WM > 0.30).
* **Crossings** — true interpenetrating 90° crossing -> 2 peaks at 84°, accurate
  ND. Fiber count correct for 1/2/3 fibers above the ~45° angular-resolution limit.
* **Free water / DTI metrics** — recovered; FORCE's FW less biased than AMICO's.

**Genuine limitations exposed:**
* **GM/soma compartment** — FORCE's isotropic-Gaussian GM ≠ true restricted soma,
  so soma signal is absorbed into WM (GM badly under-recovered).
* **Isotropic-fraction leak / within-WM NDI** — FORCE under-estimates WM fraction
  for lower-density WM, so `nd/wm_fraction` inflates. Largely a phantom-density
  artifact (vanishes at real CC density) + partly biophysical (extra-axonal vs
  zeppelin). Widening the f_intra prior barely helps on real data.
* **Bending (undulation)** — bimodal sub-resolution case; dispersion
  under-estimated, peaks lost, ND under-estimated.
* **Composite `dispersion` ≠ fiber ODI** — GM & CSF each contribute ODI=1, so the
  raw map reads high (WM median 0.30) vs true fiber ODI (0.19). Back out
  `odi_wm = 1-(1-dispersion)/f_wm`.

---

## Scope / caveats
Single straight fibers (coherent / fanned), true crossings (90°), bending
(undulation), and WM/GM/CSF mixtures. Fanning at realistic ICVF uses the
kernel-convolution method (exact for dispersion); a direct-MC fan packs only to
ICVF~0.10. Dictionary is 10/20/70% 1/2/3-fiber, f_intra∈[0.6,0.9],
ODI grid 0.01–0.30, GM iso 0.7–1.2e-3, CSF 3.0e-3, 500K atoms, K=50, β=2000.

---

## Code (all in this folder; env = skyline python)

**Substrate generators (mesh geometry):**
* `packed_cylinders.py` — square-packed parallel cylinders (single fiber).
* `packed_spheres.py` — packed soma spheres (GM compartment).
* `crossing_bending.py` — `build_crossing` (interpenetrating crossing),
  `build_undulating` (bending), `build_fanning` (Watson-oriented fan).
* `np2_shim.py` — numpy>=2 alias shim for disimpy.

**Phantom signal generation (MC sim on the protocol):**
* `generate_mc_phantom.py` — coherent ICVF×radius×FW grid (exp 1).
* `generate_dispersed_phantom.py` — kernel + Watson convolution, ODI sweep (exp 2,12).
* `generate_joint_phantom.py` — factorial NDI×ODI×FW (exp 3).
* `generate_tissue_substrates.py` — 5 tissue types (exp 4).
* `radius_effect.py` — axon-radius sweep (exp 5).
* `icvf_sweep.py` — ICVF sweep vs real CC (exp 7).
* `bio_phantom.py` — N=400 biological-range voxels + noise + NIfTI (exp 8).
* `generate_crossing_bending.py` — crossing + bending substrates (exp 10,11).
* `fanning_experiment.py` — direct-MC fanning (exp 12 cross-check).

**FORCE / AMICO runners:**
* `run_force_mc.py`, `run_force_disp.py`, `run_force_joint.py`,
  `run_force_tissue.py`, `run_force_bio.py`, `run_force_cb.py` — FORCE matching
  (K=50, β=2000) per experiment.
* `run_amico_bio.py` — AMICO-NODDI baseline (base env).
* `prior_slice72.py` — old vs widened-prior FORCE on HCP slice z=72 (exp 9);
  injects f_intra [0.3,0.9] by intercepting np.random.uniform(0.6,0.9).
* `reprior_test.py` — dictionary re-prior probe (exp 6).

**Analysis / figures:**
* `validate_substrate.py` — restriction physics check.
* `analyze_mc.py`, `analyze_joint.py`, `analyze_tissue.py`,
  `analyze_radius_effect.py`, `analyze_bio.py`, `analyze_cb.py` — bias/MAE
  tables + calibration/scatter figures per experiment.

**Outputs:** `data*/` (signals + ground truth), `force_out/`, `bio_out/`,
`force_slice72_*/` (PVE + peaks.pam5 at z=72), `figures/` (15 PNGs).
~2,800 lines total.
