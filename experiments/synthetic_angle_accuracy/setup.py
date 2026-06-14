"""Build the faster_multitensor Cython extension used by the notebook.

    python setup.py build_ext --inplace
"""
import platform

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

omp = [] if platform.system() == "Darwin" else ["-fopenmp"]

setup(
    ext_modules=cythonize(
        [Extension("faster_multitensor", ["faster_multitensor.pyx"],
                   include_dirs=[np.get_include()],
                   extra_compile_args=["-O3"] + omp,
                   extra_link_args=omp)],
        compiler_directives={"language_level": "3", "boundscheck": False,
                             "wraparound": False, "initializedcheck": False},
    ),
)
