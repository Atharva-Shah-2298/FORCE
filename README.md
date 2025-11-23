{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # FORCE: FORward modeling for Complex microstructure Estimation\
\
This repository contains code for the FORCE pipeline, which builds a large library of simulated diffusion MRI signals and then matches real data to this library to obtain fiber orientations and microstructural maps.\
\
The main components provided here are:\
\
- `simulation.py`  \
  Generates a large simulated signal library and saves it to `simulated_data.npz`.\
\
- `matching.py`  \
  Uses FAISS and Ray to match real diffusion MRI data to the simulated library and produces ODFs, peaks, and microstructural maps.\
\
- `setup_fast.py`  \
  Builds the `faster_multitensor` Cython extension used in the simulator.\
\
Additional code (expected in the repository) includes:\
\
- `faster_multitensor.pyx` and its compiled extension module  \
- `utils/geometry.py`, `utils/distribution.py`, `utils/analytical.py`  \
\
The FORCE method is described in the associated preprint:\
\
> FORCE: FORward modeling for Complex microstructure Estimation  \
> https://www.researchsquare.com/article/rs-8151109/v1\
\
---\
\
## 1. Requirements\
\
### 1.1 Python and OS\
\
- Python 3.8 or newer\
- A Unix like environment is strongly recommended (Linux or macOS)\
- A C or C++ compiler for building the Cython extension (for example `gcc` or `clang`)\
\
### 1.2 Python packages\
\
The scripts in this repository use the following Python packages:\
\
**Core scientific stack**\
\
- `numpy`\
- `scipy`\
- `matplotlib`\
\
**Diffusion MRI and microstructure modeling**\
\
- `dipy`  \
- `nibabel` (usually installed automatically with `dipy`, but you can also install it explicitly)\
\
**Parallel and distributed computing**\
\
- `ray`\
- `tqdm`\
- `psutil`\
\
**Similarity search backend**\
\
- FAISS Python bindings, imported as:\
\
  ```python\
  import faiss\
}