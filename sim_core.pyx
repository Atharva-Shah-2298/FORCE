# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport exp
from utils.geometry import angle_between, is_angle_valid
import faster_multitensor

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t LABEL_t


cdef inline double get_dperp_extra(double d_par, double f_intra) noexcept:
    return d_par * (1.0 - f_intra) / (1.0 + f_intra)


cdef inline double fa_stick_zeppelin(double d_par, double d_perp, double f_intra) noexcept:
    """
    microFA for a mixture of sticks (Da, 0, 0) and zeppelins (Da, dperp_ex, dperp_ex)
    with same axial diffusivity for intra and extra.
    """
    cdef double e = 1.0 - f_intra
    cdef double num = d_par - e * d_perp
    cdef double den = (d_par * d_par + 2.0 * (e * e) * d_perp * d_perp) ** 0.5
    return num / den


cdef Py_ssize_t _closest_direction(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=1] vec
) noexcept:
    cdef Py_ssize_t i, n = target_sphere.shape[0]
    cdef double best = 1e300
    cdef double dist, dx, dy, dz
    cdef Py_ssize_t best_idx = 0
    cdef const double[:, :] ts_mv = target_sphere

    for i in range(n):
        dx = ts_mv[i, 0] - vec[0]
        dy = ts_mv[i, 1] - vec[1]
        dz = ts_mv[i, 2] - vec[2]
        dist = dx * dx + dy * dy + dz * dz
        if dist < best:
            best = dist
            best_idx = i

    return best_idx


########################
# Single fiber
########################

cpdef tuple generate_single_fiber(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    bint tortuisity
):
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double f_intra = float(np.random.uniform(0.6, 0.9))
        double f_extra = 1.0 - f_intra
        double d_par = float(np.random.uniform(2.0e-3, 3.0e-3))
        double d_perp_extra
        double S0 = 100.0
        int idx

    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, f_intra)
    else:
        d_perp_extra = float(np.random.uniform(0.3e-3, 1.5e-3))

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    idx = int(np.random.randint(0, n_dirs))
    true_stick = target_sphere[idx]

    factor = float(np.random.choice(odi_list))

    fodf_gt = bingham_sf[idx][factor]
    fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
    fodf_gt = fodf_gt / np.sum(fodf_gt)

    S_in = faster_multitensor.multi_tensor(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
    S_ex = faster_multitensor.multi_tensor(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

    S = f_intra * S_in + f_extra * S_ex

    nearest = _closest_direction(target_sphere, true_stick)
    labels[nearest] = 1

    return (
        S * S0,
        labels,
        1,
        factor,
        0.0,
        1.0 * f_intra,       # wm_nd
        fodf_gt,
        d_par,
        f_extra * d_perp_extra,
        [1.0, 0.0, 0.0],      # fiber fractions inside WM
        [f_intra],            # f_ins per fiber
    )


########################
# Two fibers
########################

cpdef tuple generate_two_fibers(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    bint tortuisity
):
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double S0 = 100.0
        double d_par, d_perp_extra
        double fiber_frac1
        double wm_nd, wm_d_perp
        int i
        int idx0, idx1

    f_in = np.random.uniform(0.6, 0.9, 2).astype(np.float64)
    fiber_frac1 = float(np.random.uniform(0.2, 0.8))
    fiber_fractions = [fiber_frac1, 1.0 - fiber_frac1]

    d_par = float(np.random.uniform(2.0e-3, 3.0e-3))
    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, float(f_in[0]))
    else:
        d_perp_extra = float(np.random.uniform(0.3e-3, 1.5e-3))

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    index = np.random.randint(0, n_dirs, 2)
    while not is_angle_valid(
        angle_between(target_sphere[index[0]], target_sphere[index[1]])
    ):
        index = np.random.randint(0, n_dirs, 2)

    idx0 = int(index[0])
    idx1 = int(index[1])
    true_stick1 = target_sphere[idx0]
    true_stick2 = target_sphere[idx1]

    factor = float(np.random.choice(odi_list))

    S = np.zeros(bvals.shape[0], dtype=np.float64)
    fodf = np.zeros(n_dirs, dtype=np.float64)

    for i in range(2):
        fodf_gt = bingham_sf[int(index[i])][factor]
        fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        fodf += fiber_fractions[i] * fodf_gt

        S_in = faster_multitensor.multi_tensor(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
        S_ex = faster_multitensor.multi_tensor(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

        f_intra = float(f_in[i])
        f_extra = 1.0 - f_intra

        S += fiber_fractions[i] * (f_intra * S_in + f_extra * S_ex)

    labels[_closest_direction(target_sphere, true_stick1)] = 1
    labels[_closest_direction(target_sphere, true_stick2)] = 1

    wm_nd = fiber_fractions[0] * float(f_in[0]) + fiber_fractions[1] * float(f_in[1])
    wm_d_perp = (
        fiber_fractions[0] * (1.0 - float(f_in[0])) * d_perp_extra
        + fiber_fractions[1] * (1.0 - float(f_in[1])) * d_perp_extra
    )

    return (
        S * S0,
        labels,
        2,
        factor,
        0.0,
        wm_nd,
        fodf,
        d_par,
        wm_d_perp,
        [fiber_frac1, 1.0 - fiber_frac1, 0.0],
        f_in.tolist(),  # f_ins
    )


########################
# Three fibers
########################

cpdef tuple generate_three_fibers(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    bint tortuisity
):
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double S0 = 100.0
        double d_par, d_perp_extra
        double wm_nd = 0.0
        double wm_d_perp = 0.0
        int k
        int idx0, idx1, idx2

    f_in = np.random.uniform(0.6, 0.9, 3).astype(np.float64)
    fiber_fracs = np.random.dirichlet([1.0, 1.0, 1.0]).astype(np.float64)
    while np.any(fiber_fracs < 0.2):
        fiber_fracs = np.random.dirichlet([1.0, 1.0, 1.0]).astype(np.float64)

    d_par = float(np.random.uniform(2.0e-3, 3.0e-3))
    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, float(f_in[0]))
    else:
        d_perp_extra = float(np.random.uniform(0.3e-3, 1.5e-3))

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    index = np.random.randint(0, n_dirs, 3)
    while (
        not is_angle_valid(angle_between(target_sphere[index[0]], target_sphere[index[1]]), threshold=60)
        or not is_angle_valid(angle_between(target_sphere[index[0]], target_sphere[index[2]]), threshold=60)
        or not is_angle_valid(angle_between(target_sphere[index[1]], target_sphere[index[2]]), threshold=60)
    ):
        index = np.random.randint(0, n_dirs, 3)

    idx0 = int(index[0])
    idx1 = int(index[1])
    idx2 = int(index[2])

    true_stick1 = target_sphere[idx0]
    true_stick2 = target_sphere[idx1]
    true_stick3 = target_sphere[idx2]

    factor = float(np.random.choice(odi_list))

    fodf = np.zeros(n_dirs, dtype=np.float64)
    S = np.zeros(bvals.shape[0], dtype=np.float64)

    for k in range(3):
        fodf_gt = bingham_sf[int(index[k])][factor]
        fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        fodf += fiber_fracs[k] * fodf_gt

        S_in = faster_multitensor.multi_tensor(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
        S_ex = faster_multitensor.multi_tensor(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

        S += fiber_fracs[k] * (float(f_in[k]) * S_in + (1.0 - float(f_in[k])) * S_ex)

    labels[_closest_direction(target_sphere, true_stick1)] = 1
    labels[_closest_direction(target_sphere, true_stick2)] = 1
    labels[_closest_direction(target_sphere, true_stick3)] = 1

    for k in range(3):
        wm_nd += fiber_fracs[k] * float(f_in[k])
        wm_d_perp += fiber_fracs[k] * (1.0 - float(f_in[k])) * d_perp_extra

    return (
        S * S0,
        labels,
        3,
        factor,
        0.0,
        wm_nd,
        fodf,
        d_par,
        wm_d_perp,
        fiber_fracs.tolist(),
        f_in.tolist(),
    )


########################
# WM / GM / CSF wrappers
########################

cpdef tuple create_wm_signal(
    int num_fib,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    bint tortuisity
):
    if num_fib == 1:
        return generate_single_fiber(target_sphere, evecs, bingham_sf, odi_list, bvals, bvecs, tortuisity)
    elif num_fib == 2:
        return generate_two_fibers(target_sphere, evecs, bingham_sf, odi_list, bvals, bvecs, tortuisity)
    elif num_fib == 3:
        return generate_three_fibers(target_sphere, evecs, bingham_sf, odi_list, bvals, bvecs, tortuisity)
    else:
        raise ValueError("num_fib must be 1, 2 or 3")


cpdef tuple create_gm_signal(
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere
):
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double d = float(np.random.uniform(0.7e-3, 1.2e-3))

    signal = np.exp(-bvals * d) * 100.0
    labels = np.zeros(n_dirs, dtype=np.uint8)
    gm_odf = np.ones(n_dirs, dtype=np.float64) / float(n_dirs)

    return (
        signal,
        labels,
        0,
        1.0,          # gm_disp
        0.0,          # placeholder
        0.0,          # gm_nd
        gm_odf,
        d,            # gm_d_par
        d,            # gm_d_perp
    )


cpdef tuple create_csf_signal(
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere
):
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double d = 3.0e-3

    signal = np.exp(-bvals * d) * 100.0
    labels = np.zeros(n_dirs, dtype=np.uint8)
    csf_odf = np.zeros(n_dirs, dtype=np.float64)

    return (
        signal,
        labels,
        0,
        1.0,          # placeholder
        1.0,          # csf_fw
        0.0,          # placeholder
        csf_odf,
        d,            # csf_d_par
        d,            # csf_d_perp
    )


########################
# Mixed tissue simulation
########################

cpdef tuple create_mixed_signal(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    double wm_threshold,
    bint tortuisity
):
    """
    Core mixed WM/GM/CSF simulation, matching your Python version.
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double wm_fraction
        double gm_fraction
        double csf_fraction
        int num_fiber
        double odi
        double nd
        double ufa_wm = 0.0
        double ufa_voxel
        int k

    fractions = np.random.dirichlet([2.0, 1.0, 1.0]).astype(np.float64)
    wm_fraction = float(fractions[0])
    gm_fraction = float(fractions[1])
    csf_fraction = float(fractions[2])

    num_fiber = int(np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7]))

    wm_signal, wm_label, wm_num_fib, wm_disp, _, wm_nd, wm_odf, wm_d_par, wm_d_perp, fracs, f_ins = \
        create_wm_signal(
            num_fiber,
            target_sphere,
            evecs,
            bingham_sf,
            odi_list,
            bvals,
            bvecs,
            tortuisity,
        )

    gm_signal, _, _, gm_disp, _, gm_nd, gm_odf, gm_d_par, gm_d_perp = \
        create_gm_signal(bvals, target_sphere)

    csf_signal, _, _, _, csf_fw, _, csf_odf, csf_d_par, csf_d_perp = \
        create_csf_signal(bvals, target_sphere)

    odi = wm_fraction * float(wm_disp) + gm_fraction * float(gm_disp) + csf_fraction * 1.0
    nd = wm_fraction * float(wm_nd) + gm_fraction * float(gm_nd)

    combined_signal = (
        wm_fraction * wm_signal
        + gm_fraction * gm_signal
        + csf_fraction * csf_signal
    )

    if wm_fraction > wm_threshold:
        combined_odf = 50.0 * wm_fraction * wm_odf
    else:
        wm_label = np.zeros(n_dirs, dtype=np.uint8)
        combined_odf = np.zeros(n_dirs, dtype=np.float16)

    # microFA in WM
    for k in range(wm_num_fib):
        ufa_wm += fa_stick_zeppelin(
            float(wm_d_par),
            float(wm_d_perp),
            float(f_ins[k]),
        ) * float(fracs[k])

    ufa_voxel = ufa_wm * wm_fraction

    # Ensure fractions array of length 3 like your original
    frac_arr = np.zeros(3, dtype=np.float32)
    for k in range(min(3, len(fracs))):
        frac_arr[k] = float(fracs[k])

    return (
        combined_signal,                  # 0
        wm_label,                         # 1
        wm_num_fib,                       # 2
        odi,                              # 3
        wm_fraction,                      # 4
        gm_fraction,                      # 5
        csf_fraction,                     # 6
        nd,                               # 7
        combined_odf.astype(np.float16),  # 8
        float(ufa_wm),                    # 9
        float(ufa_voxel),                 # 10
        frac_arr,                         # 11
    )
