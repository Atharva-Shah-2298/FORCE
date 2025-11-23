from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "faster_multitensor",
        sources=["faster_multitensor.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native', '-ffast-math'],
        extra_link_args=['-O3', '-march=native', '-ffast-math']
    ),
    Extension(
        "vector_search",
        sources=["vector_search.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native', '-ffast-math'],
        extra_link_args=['-O3', '-march=native', '-ffast-math']
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False
        }
    )
)