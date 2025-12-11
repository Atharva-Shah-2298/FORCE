# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3
"""
Parallel matching utilities for FORCE pipeline.

This module provides efficient parallel selection of best matches from top-k
candidates after applying penalties based on model complexity.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t INT_t


cdef void select_best_matches_parallel(
    float* top_scores,
    cnp.int32_t* top_indices,
    int nvoxels,
    int k,
    float* penalty_array,
    cnp.int32_t* output_indices,
    int n_threads
) noexcept nogil:
    """
    Apply penalty and select best match for each voxel in parallel.

    For each voxel (already has top-k from vector_search/FAISS):
    1. Apply penalty based on number of fibers
    2. Select best match after penalization

    Parameters
    ----------
    top_scores : float*
        Pre-computed similarity scores for top-k candidates (nvoxels × k)
    top_indices : int*
        Pre-computed indices for top-k candidates (nvoxels × k)
    nvoxels : int
        Number of voxels to process
    k : int
        Number of top candidates per voxel
    penalty_array : float*
        Penalty value per simulation (nsims,)
    output_indices : int*
        Output array for best indices (nvoxels,)
    n_threads : int
        Number of OpenMP threads (0 = auto-detect)
    """
    cdef int i, j, best_idx
    cdef float best_score, penalized_score

    # Use parallel or serial loop based on n_threads
    if n_threads == 1:
        # Serial execution (single thread)
        for i in range(nvoxels):
            best_idx = top_indices[i * k]
            best_score = top_scores[i * k] - penalty_array[best_idx]
            for j in range(1, k):
                penalized_score = top_scores[i * k + j] - penalty_array[top_indices[i * k + j]]
                if penalized_score > best_score:
                    best_score = penalized_score
                    best_idx = top_indices[i * k + j]
            output_indices[i] = best_idx
    else:
        # Parallel execution with OpenMP
        for i in prange(nvoxels, schedule='static', num_threads=n_threads):
            # Initialize with first candidate
            best_idx = top_indices[i * k]
            best_score = top_scores[i * k] - penalty_array[best_idx]

            # Check remaining k-1 candidates
            for j in range(1, k):
                penalized_score = top_scores[i * k + j] - penalty_array[top_indices[i * k + j]]
                if penalized_score > best_score:
                    best_score = penalized_score
                    best_idx = top_indices[i * k + j]

            # Store result
            output_indices[i] = best_idx


def select_best_from_topk(
    cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] top_scores,
    cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] top_indices,
    cnp.ndarray[cnp.float32_t, ndim=1, mode='c'] penalty_array,
    int n_threads=0
):
    """
    Select best match from top-k candidates after applying penalties.

    This function takes pre-computed top-k similarity scores and indices
    from vector_search.IndexFlatIP.search() or faiss.IndexFlatIP.search(),
    applies model complexity penalties, and selects the best match for each
    voxel in parallel using OpenMP.

    Parameters
    ----------
    top_scores : ndarray, shape (nvoxels, k), dtype=float32
        Pre-computed similarity scores for top-k candidates from vector_search
        or FAISS. Higher scores indicate better matches.
    top_indices : ndarray, shape (nvoxels, k), dtype=int32
        Pre-computed indices for top-k candidates. These are indices into the
        library array.
    penalty_array : ndarray, shape (nsims,), dtype=float32
        Penalty value per simulation, typically based on the number of fiber
        populations in the model. Higher penalties discourage complex models.
    n_threads : int, optional, default=0
        Number of OpenMP threads to use for parallel processing. If 0 or not
        specified, OpenMP will auto-detect the optimal number based on the
        system.

    Returns
    -------
    best_indices : ndarray, shape (nvoxels,), dtype=int32
        Best library index for each voxel after penalization. These can be
        used to index into library arrays to retrieve matched parameters.

    Notes
    -----
    This function expects that vector_search.IndexFlatIP.search() or
    faiss.IndexFlatIP.search() has already been called to obtain top_scores
    and top_indices.

    The penalty is subtracted from the similarity score:
        penalized_score = similarity_score - penalty_array[library_idx]

    The candidate with the highest penalized score is selected as the best
    match for each voxel.

    Examples
    --------
    >>> from vector_search import IndexFlatIP
    >>> from cython_matching import select_best_from_topk
    >>> # Build index and search
    >>> index = IndexFlatIP(d=150)
    >>> index.add(library_signals)
    >>> D, I = index.search(voxel_signals, k=50, n_threads=24)
    >>> # Apply penalties and select best
    >>> best_idx = select_best_from_topk(D, I, penalty_array, n_threads=24)
    >>> # Retrieve matched parameters
    >>> matched_params = library_params[best_idx]
    """
    cdef int nvoxels = top_scores.shape[0]
    cdef int k = top_scores.shape[1]

    # Validate inputs
    if top_indices.shape[0] != nvoxels or top_indices.shape[1] != k:
        raise ValueError(
            f"Shape mismatch: top_scores ({nvoxels}, {k}), "
            f"top_indices ({top_indices.shape[0]}, {top_indices.shape[1]})"
        )

    # Allocate output
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] best_indices = \
        np.empty(nvoxels, dtype=np.int32)

    # Call C-level function
    with nogil:
        select_best_matches_parallel(
            &top_scores[0, 0],
            &top_indices[0, 0],
            nvoxels,
            k,
            &penalty_array[0],
            &best_indices[0],
            n_threads
        )

    return best_indices
