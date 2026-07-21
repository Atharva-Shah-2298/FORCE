"""Peak-detection and fiber-count metrics.

Peaks are scored with an optimal one-to-one (Hungarian) assignment, so
duplicated or spurious peaks cannot inflate the detection rate. Angles are
antipodal-symmetric, i.e. in [0, 90] degrees.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_angles(a, b):
    """Antipodal-symmetric angles in degrees between two sets of unit vectors."""
    cos = np.clip(np.asarray(a) @ np.asarray(b).T, -1.0, 1.0)
    return np.rad2deg(np.arccos(np.abs(cos)))


def optimal_peak_matching(true_dirs, pred_dirs, threshold_deg):
    """One-to-one matching maximising pairs under `threshold_deg`.

    Ties are broken by minimising total angular error. Returns the number of
    matches and their angular errors in degrees.
    """
    m, n = len(true_dirs), len(pred_dirs)
    if m == 0 or n == 0:
        return 0, np.zeros(0, dtype=np.float32)

    unmatched_cost = float(threshold_deg) + 1e-3
    dist = pairwise_angles(true_dirs, pred_dirs)

    k = max(m, n)
    cost = np.full((k, k), unmatched_cost, dtype=np.float32)
    cost[:m, :n] = dist
    cost[:m, :n][cost[:m, :n] > threshold_deg] = 1e6

    rows, cols = linear_sum_assignment(cost)
    angles = [
        dist[i, j]
        for i, j in zip(rows, cols)
        if i < m and j < n and cost[i, j] < unmatched_cost
    ]
    return len(angles), np.asarray(angles, dtype=np.float32)


def valid_peak_dirs(peak_dirs):
    """Split a padded (Q, K, 3) peak array into per-voxel direction lists."""
    peak_dirs = np.asarray(peak_dirs, dtype=np.float32)
    dirs_list, counts = [], np.zeros(len(peak_dirs), dtype=np.int32)
    for i, dirs in enumerate(peak_dirs):
        dirs = np.atleast_2d(dirs)
        keep = dirs[np.linalg.norm(dirs, axis=-1) > 1e-6]
        if len(keep):
            keep = keep / np.linalg.norm(keep, axis=-1, keepdims=True)
        dirs_list.append(keep)
        counts[i] = len(keep)
    return dirs_list, counts


def evaluate(indices, pred_dirs_list, pred_counts, true_labels, true_num_fibers,
             target_dirs, threshold_deg=20.0):
    """Score one group of voxels (an angle bucket or a fiber-count class)."""
    total_true = total_pred = total_matched = 0
    errors = []

    for idx in indices:
        true_idx = np.flatnonzero(true_labels[idx] == 1)
        pred_dirs = pred_dirs_list[idx]
        total_true += len(true_idx)
        total_pred += len(pred_dirs)
        if len(true_idx) == 0:
            continue
        n_matched, angles = optimal_peak_matching(
            target_dirs[true_idx], pred_dirs, threshold_deg
        )
        total_matched += n_matched
        errors.extend(angles.tolist())

    recall = total_matched / total_true if total_true else 0.0
    precision = total_matched / total_pred if total_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "fiber_count_accuracy": float(np.mean(true_num_fibers[indices] == pred_counts[indices])),
        "peak_detection_rate": float(recall),
        "peak_precision": float(precision),
        "peak_f1": float(f1),
        "mean_angular_error": float(np.mean(errors)) if errors else 0.0,
        "num_voxels": int(len(indices)),
        "num_true_peaks": int(total_true),
        "num_pred_peaks": int(total_pred),
        "num_matched_pairs": int(total_matched),
    }
