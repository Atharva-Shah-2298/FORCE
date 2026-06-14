# Synthetic angular-resolution and fiber-count experiment

`simulation_with_dispersion.ipynb` is the synthetic experiment from paper
Section 2.4 (Fig. 6f,g). It simulates two-fiber crossings across crossing-angle
bins (10°–80°) and single/two/three-fiber voxels with intra (stick) + extra
(zeppelin) compartments, Bingham dispersion and Rician noise (SNR 10/20/50), then
measures angular resolution and fiber-count (NUFO) accuracy for FORCE against
CSA, GQI and CSD.

## Support code

The notebook simulates signals with the small `faster_multitensor` Cython
extension and the helpers in `utils/`. Build the extension once before running:

```bash
python setup.py build_ext --inplace
```

Requires `cython` and a C compiler with OpenMP, plus `faiss-cpu`, `ray`, `psutil`,
`pandas` in addition to the repository requirements:

```bash
pip install cython faiss-cpu ray psutil pandas
```

## Run

```bash
jupyter notebook simulation_with_dispersion.ipynb
```

The notebook generates the simulation dictionary, performs nearest-neighbour
matching, and computes per-angle-bin peak-detection accuracy and NUFO accuracy,
reproducing the angular-resolution and fiber-count results of the paper.
