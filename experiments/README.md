# FORCE paper experiments

Code for the experiments in *FORCE: FORward modeling for Complex microstructure
Estimation*. Every experiment runs FORCE through its public DIPY API
(`dipy.reconst.force`); see the [repository README](../README.md) for installation
and the FORCE API. Plotting/figure-assembly code from the paper is intentionally
not included — these scripts compute the reported quantities and write small JSON/
CSV summaries.

## Setup

```bash
pip install -r ../requirements.txt
```

Edit [`config.yaml`](config.yaml) to point at one preprocessed subject (the paper
used HCP 165840: shells 0/1000/2000/3000, 288 volumes) and an output directory.

## Stability and adequacy (`scripts/`, `results/`)

Run individually or all at once with [`run_all.sh`](run_all.sh):

```bash
python scripts/exp2_library_resampling.py --config config.yaml
./run_all.sh
```

| Script | Question | Paper |
| --- | --- | --- |
| `exp1_neighborhood.py` | How concentrated is the top-K posterior neighborhood per voxel? | stability |
| `exp2_library_resampling.py` | Do estimates move when the simulation library is regenerated? | stability |
| `exp3_input_noise.py` | Coefficient of variation under repeated Rician input noise. | Fig. 6a |
| `exp4_k_sweep.py` | Sensitivity to the number of retrieved neighbours *K*. | stability |
| `exp5_beta_sweep.py` | Sensitivity to the softmax temperature β. | stability |
| `exp6_split_half.py` | Consistency across disjoint library halves. | stability |
| `exp7_dti_reference.py` | FORCE vs DTI noise CV on the same perturbed inputs. | Fig. 6a |
| `exp8_dictionary_adequacy.py` | Held-out predictive adequacy of the dictionary and restricted controls. | §2.1 |

Small numeric summaries from the paper runs are in [`results/`](results/).

## Parameter recovery (`recovery/`)

Forward-model validation against an explicitly *different* generator (Watson /
truncated-Gaussian / logit-normal latents), compared with AMICO-NODDI, DTI and DKI.
See [`recovery/README.md`](recovery/README.md).

## Cross-scanner harmonization (`harmonization/`)

Multi-scanner reproducibility on the Tong et al. cohort: per-session alignment, a
common white-matter mask, intraclass correlation (ICC[3,1]) and cross-scanner CoV
for FORCE vs DTI/DKI/AMICO-NODDI, with paired t-tests (paper Fig. 6b,c). AMICO
comparisons require the `dmri-amico` package.

| Script | Role |
| --- | --- |
| `align_sessions.py` | Register every session to each subject's reference session |
| `build_common_mask.py` | Build the shared white-matter mask used for ICC/CoV |
| `analyze_aligned.py` | Per-metric ICC and cross-scanner CoV from aligned maps |
| `analyze_harmonization.py` | Aggregate ICC/CoV across subjects and methods |
| `compare_force_vs_amico.py` | Paired t-tests, FORCE vs AMICO-NODDI |
| `run_amico_all.py` | AMICO-NODDI baseline across sessions |

## Synthetic angular resolution and fiber count (`synthetic_angle_accuracy/`)

[`run_angle_accuracy.py`](synthetic_angle_accuracy/run_angle_accuracy.py) simulates
two-fiber crossings across angle bins and single/two/three-fiber voxels with Rician
noise, then measures angular resolution and fiber-count accuracy for FORCE against
CSA, GQI and CSD (paper Fig. 6f,g); [`plot_results.py`](synthetic_angle_accuracy/plot_results.py)
renders the figures. FORCE peaks are taken from the matched simulation via
`force_peaks`, and the library is built with the crossing-angle limit relaxed so
shallow crossings are representable — see
[`synthetic_angle_accuracy/README.md`](synthetic_angle_accuracy/README.md).

## Monte Carlo true-physics phantom (`mc_phantom/`)

FORCE recovery tested against phantoms whose signal comes from an actual
GPU random-walk diffusion simulation (disimpy) in packed impermeable cylinders,
rather than any analytic compartment model — an out-of-model validation of
neurite density, orientation dispersion, fiber count, crossings and free water.
Requires an NVIDIA GPU and disimpy; no simulation output is committed. Setup and
the full experiment catalogue are in [`mc_phantom/README.md`](mc_phantom/README.md)
and [`mc_phantom/SUMMARY.md`](mc_phantom/SUMMARY.md).

## Reference partial-volume maps (MRtrix3)

The tissue partial-volume maps that FORCE's WM/GM/CSF fractions are compared
against were generated with MRtrix3 (Dhollander response → MSMT-CSD → stacked
volume-fraction map). The exact CLI pipeline is in
[`mrtrix3_pve/README.md`](mrtrix3_pve/README.md).
