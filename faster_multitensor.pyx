# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport exp
from cython.parallel import prange

def multi_tensor(double[:, ::1] mevals,
                 double[:, :, ::1] evecs,
                 double[::1] fractions,
                 double[::1] bvals,
                 double[:, ::1] bvecs):
    cdef:
        int i, j, n = bvals.shape[0], n_tensors = fractions.shape[0]
        double[::1] S = np.zeros(n, dtype=np.float64)
        double[:, ::1] tmp_S_all = np.zeros((n_tensors, n), dtype=np.float64)
        double tmp

    with nogil:
        # Compute each tensor's contribution in parallel
        for i in prange(n_tensors, schedule='static'):
            single_tensor(mevals[i], evecs[i], bvals, bvecs, tmp_S_all[i])
            tmp = fractions[i] / 100.0
            for j in range(n):
                tmp_S_all[i, j] *= tmp

        # Sum all contributions sequentially
        for i in range(n_tensors):
            for j in range(n):
                S[j] += tmp_S_all[i, j]

    return np.asarray(S)

cdef inline void single_tensor(double[::1] evals,
                        double[:, ::1] evec,
                        double[::1] bvals,
                        double[:, ::1] bvecs,
                        double[::1] S) noexcept nogil:
    cdef:
        int i
        double D00, D01, D02, D11, D12, D22
        double bx, by, bz, val
        double S0 = 1.0

    # Compute diffusion tensor components
    D00 = evec[0,0]*evals[0]*evec[0,0] + evec[0,1]*evals[1]*evec[0,1] + evec[0,2]*evals[2]*evec[0,2]
    D01 = evec[0,0]*evals[0]*evec[1,0] + evec[0,1]*evals[1]*evec[1,1] + evec[0,2]*evals[2]*evec[1,2]
    D02 = evec[0,0]*evals[0]*evec[2,0] + evec[0,1]*evals[1]*evec[2,1] + evec[0,2]*evals[2]*evec[2,2]
    D11 = evec[1,0]*evals[0]*evec[1,0] + evec[1,1]*evals[1]*evec[1,1] + evec[1,2]*evals[2]*evec[1,2]
    D12 = evec[1,0]*evals[0]*evec[2,0] + evec[1,1]*evals[1]*evec[2,1] + evec[1,2]*evals[2]*evec[2,2]
    D22 = evec[2,0]*evals[0]*evec[2,0] + evec[2,1]*evals[1]*evec[2,1] + evec[2,2]*evals[2]*evec[2,2]

    # Vectorized computation over b-values
    for i in range(bvals.shape[0]):
        bx = bvecs[i, 0]
        by = bvecs[i, 1]
        bz = bvecs[i, 2]
        
        val = (D00 * bx * bx + 2 * D01 * bx * by + 2 * D02 * bx * bz +
               D11 * by * by + 2 * D12 * by * bz + D22 * bz * bz)
        
        S[i] = S0 * exp(-bvals[i] * val)