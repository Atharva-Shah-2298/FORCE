# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False

"""
High-performance k-NN search using:
- SciPy BLAS (sgemm) for large batch matrix multiplication
- FAISS SIMD (AVX2/FMA) for small batches
- Cython heap with prange for parallel top-k selection
"""

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Import SciPy BLAS
from scipy.linalg.cython_blas cimport sgemm

# Import FAISS SIMD functions
cdef extern from "src/distances.h" namespace "faiss" nogil:
    float fvec_inner_product(const float* x, const float* y, size_t d)
    float fvec_L2sqr(const float* x, const float* y, size_t d)
    float fvec_norm_L2sqr(const float* x, size_t d)

# Import heap functions
from heap cimport select_top_k_parallel


# Threshold for choosing between SIMD and BLAS
# Below this, SIMD is faster (less overhead)
# Above this, BLAS is faster (better optimization)
cdef size_t BLAS_THRESHOLD = 500


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_distances_simd(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    float* distances
) noexcept nogil:
    """
    Compute all inner products using FAISS SIMD.
    Used for small batches where BLAS overhead is too high.
    """
    cdef size_t n_queries = queries.shape[0]
    cdef size_t n_database = database.shape[0]
    cdef size_t d = queries.shape[1]
    cdef size_t i, j

    # Parallel loop over queries
    for i in prange(n_queries, schedule='static', nogil=True):
        for j in range(n_database):
            # Call FAISS SIMD-optimized inner product (AVX2/FMA)
            distances[i * n_database + j] = fvec_inner_product(
                &queries[i, 0],
                &database[j, 0],
                d
            )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_distances_blas(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    float* distances_out
) noexcept nogil:
    """
    Compute all inner products using SciPy BLAS sgemm.
    Used for large batches where BLAS is highly optimized.

    Computes: distances_out[i, j] = queries[i, :] . database[j, :]

    C-contiguous trick: A C-contiguous (m, n) matrix is equivalent to
    a Fortran-contiguous (n, m) transposed matrix in memory. So BLAS
    sees queries(nq, d) as queries_f^T where queries_f is F(d, nq),
    and similarly for database. We compute:
        result_f = database_f^T @ queries_f
    which gives a C-contiguous (nq, nd) result.
    """
    cdef int nd = database.shape[0]
    cdef int nq = queries.shape[0]
    cdef int d = queries.shape[1]

    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef char trans_a = b'T'  # Transpose database_f to get (nd, d)
    cdef char trans_b = b'N'  # queries_f as-is (d, nq)

    cdef int m = nd   # rows of op(A)
    cdef int n = nq   # cols of op(B)
    cdef int kk = d   # inner dimension
    cdef int lda = d   # leading dim of database_f (d, nd)
    cdef int ldb = d   # leading dim of queries_f (d, nq)
    cdef int ldc = nd  # leading dim of result_f (nd, nq)

    # sgemm expects non-const float* (Fortran convention), cast away const
    cdef float* db_ptr = <float*>&database[0, 0]
    cdef float* q_ptr = <float*>&queries[0, 0]

    sgemm(
        &trans_a,
        &trans_b,
        &m,
        &n,
        &kk,
        &alpha,
        db_ptr,
        &lda,
        q_ptr,
        &ldb,
        &beta,
        distances_out,
        &ldc
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void knn_inner_product(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    size_t k,
    float[:, ::1] distances_out,
    long[:, ::1] indices_out
) noexcept nogil:
    """
    Main k-NN search function.

    Chooses between SIMD and BLAS based on batch size,
    then uses parallel heap for top-k selection.
    """
    cdef size_t n_queries = queries.shape[0]
    cdef size_t n_database = database.shape[0]
    cdef size_t d = queries.shape[1]
    cdef size_t batch_size = n_queries * n_database

    cdef float* distances = NULL
    cdef bint use_simd = batch_size < BLAS_THRESHOLD

    try:
        # Allocate distance matrix
        distances = <float*>malloc(batch_size * sizeof(float))
        if distances == NULL:
            with gil:
                raise MemoryError("Cannot allocate distance matrix")

        if use_simd:
            # Small batch: Use FAISS SIMD (less overhead)
            compute_distances_simd(queries, database, distances)
        else:
            # Large batch: Use SciPy BLAS (highly optimized)
            compute_distances_blas(queries, database, distances)

        # Select top-k using parallel heap
        select_top_k_parallel(
            distances,
            n_queries,
            n_database,
            k,
            &distances_out[0, 0],
            &indices_out[0, 0]
        )

    finally:
        if distances != NULL:
            free(distances)


@cython.boundscheck(False)
@cython.wraparound(False)
def search_flat_ip(
    const float[:, ::1] queries not None,
    const float[:, ::1] database not None,
    int k
):
    """
    Search for k-nearest neighbors using inner product.

    Automatically chooses between:
    - FAISS SIMD (AVX2/FMA) for small batches
    - SciPy BLAS (sgemm) for large batches

    Then uses parallel Cython heap for top-k selection.

    Parameters
    ----------
    queries : float32 memoryview (n_queries, d)
        Query vectors (C-contiguous)
    database : float32 memoryview (n_database, d)
        Database vectors (C-contiguous)
    k : int
        Number of nearest neighbors

    Returns
    -------
    distances : float32 array (n_queries, k)
        Inner products (descending order)
    indices : int64 array (n_queries, k)
        Indices of nearest neighbors
    """
    cdef int n_queries = queries.shape[0]
    cdef int n_database = database.shape[0]
    cdef int d = queries.shape[1]

    if queries.shape[1] != database.shape[1]:
        raise ValueError(f"Dimension mismatch: queries {queries.shape[1]} != database {database.shape[1]}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if k > n_database:
        raise ValueError(f"k ({k}) cannot be larger than database size ({n_database})")

    # Allocate output arrays (C-contiguous)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] distances = \
        np.empty((n_queries, k), dtype=np.float32, order='C')
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] indices = \
        np.empty((n_queries, k), dtype=np.int64, order='C')

    # Get memoryviews
    cdef float[:, ::1] distances_view = distances
    cdef long[:, ::1] indices_view = indices

    # Call optimized search
    with nogil:
        knn_inner_product(
            queries,
            database,
            k,
            distances_view,
            indices_view
        )

    return distances, indices


@cython.boundscheck(False)
@cython.wraparound(False)
def inner_product_single(
    float[::1] x not None,
    float[::1] y not None
):
    """
    Compute inner product between two vectors using FAISS SIMD.

    Uses AVX2/FMA instructions for 3-4x speedup over naive code.

    Parameters
    ----------
    x, y : float32 memoryviews (d,)
        Input vectors (C-contiguous)

    Returns
    -------
    float
        Inner product
    """
    cdef int d = x.shape[0]

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Dimension mismatch: {x.shape[0]} != {y.shape[0]}")

    cdef float result
    with nogil:
        result = fvec_inner_product(&x[0], &y[0], d)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def l2sqr_single(
    float[::1] x not None,
    float[::1] y not None
):
    """
    Compute squared L2 distance using FAISS SIMD.

    Parameters
    ----------
    x, y : float32 memoryviews (d,)
        Input vectors

    Returns
    -------
    float
        Squared L2 distance
    """
    cdef int d = x.shape[0]

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Dimension mismatch: {x.shape[0]} != {y.shape[0]}")

    cdef float result
    with nogil:
        result = fvec_L2sqr(&x[0], &y[0], d)

    return result
