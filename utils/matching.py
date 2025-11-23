import faiss
import numpy as np
import ray


def create_faiss_index(norm_signal):
    dim = norm_signal.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(norm_signal)

    return index


def dict_search(
    index, chunk_indices, maskdata_chunk_normalized, labels, penalized_array, top_k=50
):
    D, I = index.search(maskdata_chunk_normalized, top_k)

    D = D - penalized_array[I]

    best_indices = np.argmax(D, axis=1)

    final_indices = I[np.arange(len(best_indices)), best_indices]

    closest_labels = labels[final_indices]

    return closest_labels, chunk_indices, final_indices
