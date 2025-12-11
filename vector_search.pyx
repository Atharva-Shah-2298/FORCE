# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""Functional-style implementation of FAISS IndexFlatIP with OpenMP.

This module provides a high-performance inner product search index using
functional programming principles. It uses cross-platform BLAS via scipy,
OpenMP parallelization, and custom heap-based algorithms for top-k selection.

The implementation is fully cross-platform (macOS, Linux, Windows) and provides
both class-based and pure functional APIs.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, memset
cimport cython
from cython.parallel cimport prange, parallel

# Use scipy's cross-platform BLAS interface
from scipy.linalg.cython_blas cimport sgemm

cnp.import_array()

# OpenMP header (always enabled)
cdef extern from *:
    """
    #ifdef _OPENMP
      #include <omp.h>
    #else
      static inline int  omp_get_max_threads(void) { return 1; }
      static inline void omp_set_num_threads(int x) { (void)x; }
      static inline int  omp_get_thread_num(void)  { return 0; }
    #endif
    """
    int omp_get_max_threads() nogil
    void omp_set_num_threads(int) nogil
    int omp_get_thread_num() nogil


# Data structure to hold index state
cdef struct IndexState:
    int d           # dimension
    int ntotal      # number of vectors
    int capacity    # allocated capacity
    float* xb       # database vectors (ntotal x d), row-major


# Heap structure for top-k selection
cdef struct HeapItem:
    float value
    int index


# ============================================================================
# Internal cdef functions (not exposed to Python)
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void heap_swap(HeapItem* heap, int i, int j) noexcept nogil:
    """Swap two elements in a heap.

    Parameters
    ----------
    heap : HeapItem*
        Pointer to heap array.
    i : int
        Index of first element.
    j : int
        Index of second element.
    """
    cdef HeapItem tmp = heap[i]
    heap[i] = heap[j]
    heap[j] = tmp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void heap_down(HeapItem* heap, int k, int pos) noexcept nogil:
    """Sift element down in min-heap to maintain heap property.

    Parameters
    ----------
    heap : HeapItem*
        Pointer to heap array.
    k : int
        Size of the heap.
    pos : int
        Starting position to sift down from.
    """
    cdef int child
    cdef HeapItem item = heap[pos]

    while pos < k // 2:
        child = 2 * pos + 1

        if child + 1 < k and heap[child + 1].value < heap[child].value:
            child += 1

        if item.value <= heap[child].value:
            break

        heap[pos] = heap[child]
        pos = child

    heap[pos] = item


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void heap_up(HeapItem* heap, int pos) noexcept nogil:
    """Sift element up in min-heap to maintain heap property.

    Parameters
    ----------
    heap : HeapItem*
        Pointer to heap array.
    pos : int
        Starting position to sift up from.
    """
    cdef int parent
    cdef HeapItem item = heap[pos]

    while pos > 0:
        parent = (pos - 1) // 2

        if heap[parent].value <= item.value:
            break

        heap[pos] = heap[parent]
        pos = parent

    heap[pos] = item


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void select_top_k_heap(float* scores, int n, int k,
                            float* top_scores, int* top_indices) noexcept nogil:
    """Select top-k maximum values using a min-heap algorithm.

    This function uses a min-heap to maintain the k largest elements seen so far.
    The heap maintains the minimum of the top-k elements at the root, allowing
    efficient updates when processing the remaining elements. Time complexity is
    O(n log k).

    Parameters
    ----------
    scores : float*
        Pointer to array of n scores to search.
    n : int
        Number of scores in the array.
    k : int
        Number of top scores to select.
    top_scores : float*
        Output array of size k for top scores (sorted descending).
    top_indices : int*
        Output array of size k for indices of top scores.
    """
    cdef int i, j
    cdef HeapItem* heap = <HeapItem*>malloc(k * sizeof(HeapItem))

    for i in range(k):
        heap[i].value = scores[i]
        heap[i].index = i

    for i in range(k // 2 - 1, -1, -1):
        heap_down(heap, k, i)

    for i in range(k, n):
        if scores[i] > heap[0].value:
            heap[0].value = scores[i]
            heap[0].index = i
            heap_down(heap, k, 0)

    cdef int heap_size = k
    cdef HeapItem* sorted_heap = <HeapItem*>malloc(k * sizeof(HeapItem))

    for i in range(k):
        sorted_heap[i] = heap[i]

    for i in range(k - 1, -1, -1):
        top_scores[i] = sorted_heap[0].value
        top_indices[i] = sorted_heap[0].index

        sorted_heap[0] = sorted_heap[heap_size - 1]
        heap_size -= 1
        if heap_size > 0:
            heap_down(sorted_heap, heap_size, 0)

    free(heap)
    free(sorted_heap)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef IndexState* create_index(int d) noexcept nogil:
    """Create a new index state structure.

    Parameters
    ----------
    d : int
        Vector dimension.

    Returns
    -------
    IndexState*
        Pointer to newly allocated IndexState, or NULL on failure.
    """
    cdef IndexState* state = <IndexState*>malloc(sizeof(IndexState))
    if state == NULL:
        return NULL

    state.d = d
    state.ntotal = 0
    state.capacity = 0
    state.xb = NULL

    return state


cdef void free_index(IndexState* state) noexcept nogil:
    """Free index state and all associated memory.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state to free.
    """
    if state == NULL:
        return

    if state.xb != NULL:
        free(state.xb)

    free(state)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int add_vectors(IndexState* state, const float* vectors, int n) noexcept nogil:
    """Add vectors to the index database.

    Automatically expands capacity using 1.5x growth strategy when needed.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state.
    vectors : const float*
        Pointer to n vectors of dimension d stored in row-major order.
    n : int
        Number of vectors to add.

    Returns
    -------
    int
        0 on success, -1 on failure (memory allocation error).
    """
    if state == NULL or vectors == NULL or n <= 0:
        return -1

    cdef int new_total = state.ntotal + n
    cdef int new_capacity
    cdef float* new_xb

    if new_total > state.capacity:
        new_capacity = new_total
        if new_capacity < state.capacity * 3 // 2:
            new_capacity = state.capacity * 3 // 2
        if new_capacity < 1024:
            new_capacity = 1024

        new_xb = <float*>realloc(state.xb, new_capacity * state.d * sizeof(float))
        if new_xb == NULL:
            return -1

        state.xb = new_xb
        state.capacity = new_capacity

    memcpy(&state.xb[state.ntotal * state.d], vectors, n * state.d * sizeof(float))
    state.ntotal = new_total

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int search_vectors(IndexState* state,
                       const float* queries, int nq,
                       int k,
                       float* distances, int* indices,
                       int n_threads) noexcept nogil:
    """Search for k nearest neighbors using batched processing for memory efficiency.

    Processes queries in batches to avoid excessive memory allocation, making it
    suitable for large datasets similar to FAISS.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state.
    queries : const float*
        Query vectors (nq x d) stored in row-major order.
    nq : int
        Number of query vectors.
    k : int
        Number of neighbors to find per query.
    distances : float*
        Output array (nq x k) for inner product scores, row-major.
    indices : int*
        Output array (nq x k) for neighbor indices, row-major.
    n_threads : int
        Number of OpenMP threads. 0 uses default (all available cores).

    Returns
    -------
    int
        0 on success, -1 on failure (invalid input or memory allocation error).

    Notes
    -----
    Memory usage: batch_size × ntotal floats (default: 1000 × ntotal ≈ 2GB for 500k library)
    Previously allocated: nq × ntotal floats (could be 650k × 500k ≈ 1.3TB!)
    """
    if state == NULL or queries == NULL or distances == NULL or indices == NULL:
        return -1

    if state.ntotal == 0 or nq <= 0 or k <= 0:
        return -1

    if k > state.ntotal:
        k = state.ntotal

    # Set OpenMP threads
    if n_threads > 0:
        omp_set_num_threads(n_threads)

    # Adaptive batch size based on library size to keep memory reasonable
    # Target: ~2GB per batch (2GB / (ntotal × 4 bytes))
    cdef int batch_size
    if state.ntotal < 50000:
        batch_size = 10000  # Small library: larger batches OK
    elif state.ntotal < 200000:
        batch_size = 2000   # Medium library
    else:
        batch_size = 1000   # Large library: conservative batches

    # Declare all variables at the beginning
    cdef int batch_start, batch_end, batch_nq
    cdef int i, j
    cdef float* scores = NULL
    cdef int ret = 0
    cdef int blas_m, blas_n, blas_k
    cdef int lda, ldb, ldc
    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef char trans_t = ord('T')
    cdef char trans_n = ord('N')

    # Allocate scores buffer for one batch (reused across batches)
    scores = <float*>malloc(batch_size * state.ntotal * sizeof(float))
    if scores == NULL:
        return -1

    # Process batches (C-style loop for nogil compatibility)
    batch_start = 0
    while batch_start < nq:
        batch_end = batch_start + batch_size
        if batch_end > nq:
            batch_end = nq
        batch_nq = batch_end - batch_start

        # BLAS parameters for this batch
        blas_m = state.ntotal
        blas_n = batch_nq
        blas_k = state.d
        lda = state.d
        ldb = state.d
        ldc = state.ntotal

        # Compute scores for this batch using BLAS
        # scores_batch[batch_nq × ntotal] = queries_batch[batch_nq × d] @ xb[ntotal × d]^T
        sgemm(
            &trans_t,                           # transa: transpose xb
            &trans_n,                           # transb: no transpose queries
            &blas_m,                            # m: ntotal
            &blas_n,                            # n: batch_nq
            &blas_k,                            # k: d
            &alpha,                             # alpha: 1.0
            state.xb,                           # A: xb database
            &lda,                               # lda: d
            &queries[batch_start * state.d],   # B: queries for this batch
            &ldb,                               # ldb: d
            &beta,                              # beta: 0.0
            scores,                             # C: output scores
            &ldc                                # ldc: ntotal
        )

        # Extract top-k for each query in this batch
        if n_threads == 1:
            # Serial execution (single thread)
            for i in range(batch_nq):
                select_top_k_heap(
                    &scores[i * state.ntotal],
                    state.ntotal,
                    k,
                    &distances[(batch_start + i) * k],
                    &indices[(batch_start + i) * k]
                )
        else:
            # Parallel execution with OpenMP
            for i in prange(batch_nq, schedule='static'):
                select_top_k_heap(
                    &scores[i * state.ntotal],
                    state.ntotal,
                    k,
                    &distances[(batch_start + i) * k],
                    &indices[(batch_start + i) * k]
                )

        # Move to next batch
        batch_start = batch_start + batch_size

    free(scores)
    return 0


cdef void reset_index(IndexState* state) noexcept nogil:
    """Reset index by clearing all vectors while retaining allocated memory.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state.
    """
    if state == NULL:
        return

    state.ntotal = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int reconstruct_vector(IndexState* state, int idx, float* out) noexcept nogil:
    """Reconstruct a single vector from the index.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state.
    idx : int
        Index of vector to reconstruct.
    out : float*
        Output buffer of size d.

    Returns
    -------
    int
        0 on success, -1 on failure (invalid index or NULL pointers).
    """
    if state == NULL or out == NULL:
        return -1

    if idx < 0 or idx >= state.ntotal:
        return -1

    memcpy(out, &state.xb[idx * state.d], state.d * sizeof(float))
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int reconstruct_vectors(IndexState* state, int i0, int ni, float* out) noexcept nogil:
    """Reconstruct a contiguous range of vectors from the index.

    Parameters
    ----------
    state : IndexState*
        Pointer to index state.
    i0 : int
        Starting index of range.
    ni : int
        Number of vectors to reconstruct.
    out : float*
        Output buffer of size ni * d.

    Returns
    -------
    int
        0 on success, -1 on failure (invalid range or NULL pointers).
    """
    if state == NULL or out == NULL:
        return -1

    if i0 < 0 or i0 + ni > state.ntotal or ni <= 0:
        return -1

    memcpy(out, &state.xb[i0 * state.d], ni * state.d * sizeof(float))
    return 0


# ============================================================================
# Python interface (minimal def functions)
# ============================================================================

cdef class IndexFlatIP:
    """Functional-style flat index for inner product similarity search.

    This implementation uses a functional programming style with cdef functions,
    cross-platform BLAS via scipy, and OpenMP parallelization for maximum performance.

    Parameters
    ----------
    d : int
        Dimensionality of the vectors.

    Attributes
    ----------
    d : int
        Vector dimension.
    ntotal : int
        Number of vectors currently in the index.
    is_trained : bool
        Always True (no training required for flat index).

    Examples
    --------
    >>> import numpy as np
    >>> index = IndexFlatIP(d=128)
    >>> vectors = np.random.random((1000, 128)).astype('float32')
    >>> index.add(vectors)
    >>> queries = np.random.random((10, 128)).astype('float32')
    >>> distances, indices = index.search(queries, k=5, n_threads=4)
    """

    cdef IndexState* state

    def __init__(self, int d):
        """Initialize the index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors.

        Raises
        ------
        MemoryError
            If index state allocation fails.
        """
        self.state = create_index(d)
        if self.state == NULL:
            raise MemoryError("Failed to create index")

    def __dealloc__(self):
        """Free allocated memory when object is destroyed."""
        free_index(self.state)
        self.state = NULL

    def __reduce__(self):
        """Support for pickling (needed for Ray serialization).

        Returns a tuple (callable, args) to reconstruct the object.
        """
        if self.state == NULL:
            return (IndexFlatIP, (0,))

        # Extract all data from C structure
        cdef int d = self.state.d
        cdef int ntotal = self.state.ntotal

        # Copy vector data to numpy array
        cdef cnp.ndarray[cnp.float32_t, ndim=2] data = np.empty((ntotal, d), dtype=np.float32)
        if ntotal > 0 and self.state.xb != NULL:
            memcpy(&data[0, 0], self.state.xb, ntotal * d * sizeof(float))

        # Return constructor and state
        return (_rebuild_index, (d, data))

    @property
    def d(self):
        """int: Vector dimension."""
        return self.state.d if self.state != NULL else 0

    @property
    def ntotal(self):
        """int: Number of vectors in the index."""
        return self.state.ntotal if self.state != NULL else 0

    @property
    def is_trained(self):
        """bool: Always True (flat index requires no training)."""
        return True

    def add(self, cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] x):
        """Add vectors to the index database.

        Parameters
        ----------
        x : ndarray, shape (n, d), dtype=float32
            Array containing n vectors of dimension d. Must be C-contiguous.

        Raises
        ------
        RuntimeError
            If index is not initialized.
        ValueError
            If vector dimension does not match index dimension.
        MemoryError
            If memory allocation fails during vector addition.
        """
        if self.state == NULL:
            raise RuntimeError("Index not initialized")

        cdef int n = x.shape[0]
        cdef int d = x.shape[1]

        if d != self.state.d:
            raise ValueError(f"Vector dimension {d} does not match index dimension {self.state.d}")

        cdef int ret
        with nogil:
            ret = add_vectors(self.state, &x[0, 0], n)

        if ret != 0:
            raise MemoryError("Failed to add vectors to index")

    def search(self, cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] x, int k, int n_threads=0):
        """Search for k nearest neighbors by inner product.

        Uses BLAS for matrix multiplication and OpenMP for parallel top-k selection.

        Parameters
        ----------
        x : ndarray, shape (nq, d), dtype=float32
            Query vectors. Must be C-contiguous.
        k : int
            Number of nearest neighbors to return per query.
        n_threads : int, optional
            Number of OpenMP threads. 0 uses default (all available cores).

        Returns
        -------
        distances : ndarray, shape (nq, k), dtype=float32
            Inner product scores (higher is better).
        indices : ndarray, shape (nq, k), dtype=int32
            Indices of nearest neighbors in database.

        Raises
        ------
        RuntimeError
            If index is not initialized or is empty.
        ValueError
            If query dimension does not match index dimension.
        """
        if self.state == NULL:
            raise RuntimeError("Index not initialized")

        if self.state.ntotal == 0:
            raise RuntimeError("Cannot search in empty index")

        cdef int nq = x.shape[0]
        cdef int d = x.shape[1]

        if d != self.state.d:
            raise ValueError(f"Query dimension {d} does not match index dimension {self.state.d}")

        k = min(k, self.state.ntotal)

        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] distances = np.empty(
            (nq, k), dtype=np.float32, order='C'
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] indices = np.empty(
            (nq, k), dtype=np.int32, order='C'
        )

        cdef int ret
        with nogil:
            ret = search_vectors(
                self.state,
                &x[0, 0], nq,
                k,
                &distances[0, 0], <int*>&indices[0, 0],
                n_threads
            )

        if ret != 0:
            raise RuntimeError("Search failed")

        return distances, indices

    def reset(self):
        """Clear all vectors from the index while retaining allocated memory.

        Raises
        ------
        RuntimeError
            If index is not initialized.
        """
        if self.state == NULL:
            raise RuntimeError("Index not initialized")

        with nogil:
            reset_index(self.state)

    def reconstruct(self, int i):
        """Reconstruct a single vector from the index.

        Parameters
        ----------
        i : int
            Index of the vector to reconstruct.

        Returns
        -------
        ndarray, shape (d,), dtype=float32
            The reconstructed vector.

        Raises
        ------
        RuntimeError
            If index is not initialized or reconstruction fails.
        IndexError
            If index i is out of range.
        """
        if self.state == NULL:
            raise RuntimeError("Index not initialized")

        if i < 0 or i >= self.state.ntotal:
            raise IndexError(f"Index {i} out of range [0, {self.state.ntotal})")

        cdef cnp.ndarray[cnp.float32_t, ndim=1] result = np.empty(self.state.d, dtype=np.float32)

        cdef int ret
        with nogil:
            ret = reconstruct_vector(self.state, i, &result[0])

        if ret != 0:
            raise RuntimeError("Failed to reconstruct vector")

        return result

    def reconstruct_n(self, int i0, int ni):
        """Reconstruct a contiguous range of vectors from the index.

        Parameters
        ----------
        i0 : int
            Starting index.
        ni : int
            Number of vectors to reconstruct.

        Returns
        -------
        ndarray, shape (ni, d), dtype=float32
            The reconstructed vectors.

        Raises
        ------
        RuntimeError
            If index is not initialized or reconstruction fails.
        IndexError
            If range [i0, i0+ni) is out of bounds.
        """
        if self.state == NULL:
            raise RuntimeError("Index not initialized")

        if i0 < 0 or i0 + ni > self.state.ntotal:
            raise IndexError(f"Range [{i0}, {i0+ni}) out of bounds [0, {self.state.ntotal})")

        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] result = np.empty(
            (ni, self.state.d), dtype=np.float32, order='C'
        )

        cdef int ret
        with nogil:
            ret = reconstruct_vectors(self.state, i0, ni, &result[0, 0])

        if ret != 0:
            raise RuntimeError("Failed to reconstruct vectors")

        return result


# ============================================================================
# Pickle support function (module-level for unpickling)
# ============================================================================

def _rebuild_index(int d, data):
    """Rebuild IndexFlatIP from pickled data.

    This function is called during unpickling to reconstruct the index.

    Parameters
    ----------
    d : int
        Vector dimension.
    data : ndarray of shape (ntotal, d), dtype=float32
        Vector data to restore.

    Returns
    -------
    IndexFlatIP
        Reconstructed index with restored data.
    """
    cdef IndexFlatIP index = IndexFlatIP(d)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] data_c

    if data.shape[0] > 0:
        # Ensure data is C-contiguous and properly typed
        data_c = np.ascontiguousarray(data, dtype=np.float32)
        index.add(data_c)
    return index


# ============================================================================
# Pure functional API (no class, just functions)
# ============================================================================

def create_flat_ip_index(int d):
    """Create a new flat inner product index.

    Parameters
    ----------
    d : int
        Vector dimension.

    Returns
    -------
    IndexFlatIP
        New index instance.
    """
    return IndexFlatIP(d)


def add_to_index(IndexFlatIP index, cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] vectors):
    """Add vectors to an index.

    Parameters
    ----------
    index : IndexFlatIP
        Index instance to add vectors to.
    vectors : ndarray of shape (n, d), dtype=float32
        Vectors to add to the database.
    """
    index.add(vectors)


def search_index(IndexFlatIP index,
                cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] queries,
                int k,
                int n_threads=0):
    """Search an index for nearest neighbors.

    Parameters
    ----------
    index : IndexFlatIP
        Index instance to search.
    queries : ndarray of shape (nq, d), dtype=float32
        Query vectors.
    k : int
        Number of nearest neighbors to return.
    n_threads : int, optional
        Number of OpenMP threads to use (0 = use default).

    Returns
    -------
    distances : ndarray of shape (nq, k), dtype=float32
        Inner product scores for each query-neighbor pair.
    indices : ndarray of shape (nq, k), dtype=int64
        Database indices of nearest neighbors.
    """
    return index.search(queries, k, n_threads)


def reset_flat_ip_index(IndexFlatIP index):
    """Reset an index by clearing all vectors.

    Parameters
    ----------
    index : IndexFlatIP
        Index instance to reset.
    """
    index.reset()
