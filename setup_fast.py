from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

print("Building Cython extensions with OpenMP support")


def get_openmp_flags():
    """Get OpenMP compiler and linker flags based on platform.

    Returns
    -------
    compile_args : list
        Compiler arguments for OpenMP.
    link_args : list
        Linker arguments for OpenMP.

    Notes
    -----
    OpenMP is always enabled for matching.py (uses Cython+OpenMP).
    simulation.py may still use Ray, but Ray and OpenMP don't conflict
    because they're used in different scripts.
    """
    system = platform.system()

    if system == 'Darwin':  # macOS
        # Check if we're using clang or gcc
        # On macOS, we typically need libomp (LLVM OpenMP)
        # Try to match numpy/scipy's OpenMP backend
        try:
            import subprocess
            result = subprocess.run(['cc', '--version'], capture_output=True, text=True)
            if 'clang' in result.stdout.lower():
                return ['-Xpreprocessor', '-fopenmp'], ['-lomp']
            else:
                # Using GCC
                return ['-fopenmp'], ['-fopenmp']
        except Exception:
            # Default to libomp for macOS
            return ['-Xpreprocessor', '-fopenmp'], ['-lomp']

    elif system == 'Linux':
        return ['-fopenmp'], ['-fopenmp']

    elif system == 'Windows':
        return ['/openmp'], []

    else:
        # Unknown platform, try standard GCC flags
        return ['-fopenmp'], ['-fopenmp']


openmp_compile_args, openmp_link_args = get_openmp_flags()

base_compile_args = ['-O3', '-march=native', '-ffast-math']
base_link_args = ['-O3', '-march=native', '-ffast-math']

# Combine with OpenMP flags if applicable
compile_args = base_compile_args + openmp_compile_args
link_args = base_link_args + openmp_link_args

print(f"Compile args: {compile_args}")
print(f"Link args: {link_args}")

extensions = [
    Extension(
        "faster_multitensor",
        sources=["faster_multitensor.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    ),
    Extension(
        "vector_search",
        sources=["vector_search.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    ),
    Extension(
        "cython_matching",
        sources=["cython_matching.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args
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
        },
        compile_time_env={
            'USE_OPENMP': True  # Always enable OpenMP for all extensions
        }
    )
)