cdef void select_top_k_parallel(
    const float* distances,
    size_t n_queries,
    size_t n_database,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil
