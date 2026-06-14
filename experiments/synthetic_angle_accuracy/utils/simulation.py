import numpy as np
from faster_multitensor import multi_tensor

from utils.geometry import angle_between, is_angle_valid


def get_diffusion_perp_ex(d_par, f_intra):
    return d_par * (1 - f_intra) / (1 + f_intra)


def create_mixed_signal(
    sphere,
    gtab,
    evecs,
    bingham_sf,
    thres=0.5,
    tortuisity=False,
    index=None,
    num_fib=2,
):
    """Generate a mixed signal combining WM, GM, and CSF."""
    # Randomly sample fractions for WM, GM, and CSF
    evecs = np.ascontiguousarray(evecs).astype(np.float64)
    target_sphere = sphere.vertices
    bvals = np.ascontiguousarray(gtab.bvals).astype(np.float64)
    bvecs = np.ascontiguousarray(gtab.bvecs).astype(np.float64)

    def generate_single_fiber(index=None):
        # choose intra and extra axonal fractions with free water for compartments between 30 and 70%
        f_intra = np.random.uniform(0.6, 0.9)
        f_extra = 1 - f_intra

        fiber_fractions = 1

        labels = np.zeros((target_sphere.shape[0]))
        S0 = 100
        d_par = np.random.uniform(2.0e-3, 3.0e-3)
        if tortuisity:
            d_perp_extra = get_diffusion_perp_ex(d_par, f_intra)

        else:
            d_perp_extra = np.random.uniform(0.1e-3, 0.6e-3)

        mevals_ex = np.zeros(np.shape(target_sphere))
        mevals_ex[:, 0] = d_par
        mevals_ex[:, 1] = d_perp_extra
        mevals_ex[:, 2] = d_perp_extra

        mevals_in = np.zeros(np.shape(target_sphere))
        mevals_in[:, 0] = d_par
        mevals_in[:, 1] = 0.0e-3
        mevals_in[:, 2] = 0.0e-3

        # Select a random fiber orientation from the target sphere
        if index is not None:
            index = index
        else:
            index = np.random.randint(0, target_sphere.shape[0], 1)[0]
        true_stick = target_sphere[index]
        factor = np.random.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

        # fodf_gt = bingham_to_sf(1, k1, k2 , major_axis, minor_axis, target_sphere)
        fodf_gt = bingham_sf[index][factor]
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        S_in = multi_tensor(mevals_in, evecs, fodf_gt * 100, bvals, bvecs)
        S_ex = multi_tensor(mevals_ex, evecs, fodf_gt * 100, bvals, bvecs)

        # simulate the signal
        S = fiber_fractions * (f_intra * S_in + f_extra * S_ex)

        for j in range(1):
            labels[np.argmin(np.linalg.norm(target_sphere - true_stick, axis=1))] = 1
            labels[np.argmin(np.linalg.norm(target_sphere + true_stick, axis=1))] = 1

        return S * S0, labels, 1, factor, 0, fiber_fractions * f_intra, fodf_gt

    def generate_two_fibers(index=None):
        f_in = np.random.uniform(0.5, 0.9, 2)
        fiber_frac1 = np.random.uniform(0.2, 0.8)
        fiber_fractions = [fiber_frac1, 1 - fiber_frac1]

        d_par = np.random.uniform(2.0e-3, 3.0e-3)
        if tortuisity:
            d_perp_extra = get_diffusion_perp_ex(d_par, f_in[0])
        else:
            d_perp_extra = np.random.uniform(0.1e-3, 0.6e-3)

        labels = np.zeros((target_sphere.shape[0]))
        S0 = 100
        mevals_ex = np.zeros(np.shape(target_sphere))
        mevals_ex[:, 0] = d_par
        mevals_ex[:, 1] = d_perp_extra
        mevals_ex[:, 2] = d_perp_extra

        mevals_in = np.zeros(np.shape(target_sphere))
        mevals_in[:, 0] = d_par
        mevals_in[:, 1] = 0.0e-3
        mevals_in[:, 2] = 0.0e-3

        # Select a random fiber orientation from the target sphere
        if index is not None:
            index = index
        else:
            index = np.random.randint(0, target_sphere.shape[0], 2)

        # if indexes are closer than 30 degrees, choose another one
        while not is_angle_valid(
            angle_between(target_sphere[index[0]], target_sphere[index[1]])
        ):
            index = np.random.randint(0, target_sphere.shape[0], 2)

        true_stick1 = target_sphere[index[0]]
        true_stick2 = target_sphere[index[1]]
        factor = np.random.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

        S = np.zeros((len(bvals)))
        fodf = np.zeros((target_sphere.shape[0]))

        for i in range(2):
            fodf_gt = bingham_sf[index[i]][factor]
            fodf_gt = fodf_gt / np.sum(fodf_gt)

            fodf += fiber_fractions[i] * fodf_gt

            S_in = multi_tensor(mevals_in, evecs, fodf_gt * 100, bvals, bvecs)
            S_ex = multi_tensor(mevals_ex, evecs, fodf_gt * 100, bvals, bvecs)

            # choose intra and extra axonal fractions with free water for compartments between 30 and 70%
            f_intra = f_in[i]
            f_extra = 1 - f_intra

            S += fiber_fractions[i] * (f_intra * S_in + f_extra * S_ex)

            labels[np.argmin(np.linalg.norm(target_sphere - true_stick1, axis=1))] = 1
            labels[np.argmin(np.linalg.norm(target_sphere + true_stick1, axis=1))] = 1
            labels[np.argmin(np.linalg.norm(target_sphere - true_stick2, axis=1))] = 1
            labels[np.argmin(np.linalg.norm(target_sphere + true_stick2, axis=1))] = 1
        return (
            S * S0,
            labels,
            2,
            factor,
            0,
            fiber_fractions[0] * f_in[0] + fiber_fractions[1] * f_in[1],
            fodf,
        )

    def generate_three_fibers(index=None):
        f_in = np.random.uniform(0.4, 0.9, 3)
        fiber_fracs = np.random.dirichlet([1, 1, 1])

        d_par = np.random.uniform(2.0e-3, 3.0e-3)
        if tortuisity:
            d_perp_extra = get_diffusion_perp_ex(d_par, f_in[0])
        else:
            d_perp_extra = np.random.uniform(0.1e-3, 0.6e-3)

        labels = np.zeros((target_sphere.shape[0]))
        S0 = 100
        mevals_ex = np.zeros(np.shape(target_sphere))
        mevals_ex[:, 0] = d_par
        mevals_ex[:, 1] = d_perp_extra
        mevals_ex[:, 2] = mevals_ex[:, 1]

        mevals_in = np.zeros(np.shape(target_sphere))
        mevals_in[:, 0] = d_par
        mevals_in[:, 1] = 0.0e-3
        mevals_in[:, 2] = 0.0e-3

        # Select a random fiber orientation from the target sphere
        if index is not None:
            index = index
        else:
            index = np.random.randint(0, target_sphere.shape[0], 3)

        # if indexes are closer than 30 degrees, choose another one
        while (
            not is_angle_valid(
                angle_between(target_sphere[index[0]], target_sphere[index[1]])
            )
            or not is_angle_valid(
                angle_between(target_sphere[index[0]], target_sphere[index[2]])
            )
            or not is_angle_valid(
                angle_between(target_sphere[index[1]], target_sphere[index[2]])
            )
        ):
            index = np.random.randint(0, target_sphere.shape[0], 3)

        true_stick1 = target_sphere[index[0]]
        true_stick2 = target_sphere[index[1]]
        true_stick3 = target_sphere[index[2]]
        factor = np.random.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

        fodf = np.zeros((target_sphere.shape[0]))
        S = np.zeros((len(bvals)))
        for k in range(3):
            fodf_gt = bingham_sf[index[k]][factor]
            fodf_gt = fodf_gt / np.sum(fodf_gt)

            fodf += fiber_fracs[k] * fodf_gt

            S_in = multi_tensor(mevals_in, evecs, fodf_gt * 100, bvals, bvecs)
            S_ex = multi_tensor(mevals_ex, evecs, fodf_gt * 100, bvals, bvecs)

            S += fiber_fracs[k] * (f_in[k] * S_in + (1 - f_in[k]) * S_ex)

        labels[np.argmin(np.linalg.norm(target_sphere - true_stick1, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere + true_stick1, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere - true_stick2, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere + true_stick2, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere - true_stick3, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere + true_stick3, axis=1))] = 1
        return (
            S * S0,
            labels,
            3,
            factor,
            0,
            fiber_fracs[0] * f_in[0]
            + fiber_fracs[1] * f_in[1]
            + fiber_fracs[2] * f_in[2],
            fodf,
        )

    def create_wm_signal(num_fib):
        """Generate a white matter signal with random fiber configuration."""
        if num_fib == 1:
            return generate_single_fiber(index=index)
        elif num_fib == 2:
            return generate_two_fibers(index=index)
        elif num_fib == 3:
            return generate_three_fibers(index=index)

    def create_gm_signal():
        d = np.random.uniform(0.7e-3, 1.2e-3)
        signal = np.exp(-bvals * d) * 100
        return (
            signal,
            np.zeros(len(target_sphere)),
            0,
            1.0,
            0.0,
            0.0,
            np.ones(len(target_sphere)) / len(target_sphere),
        )

    def create_csf_signal():
        """Generate a CSF signal (isotropic)."""
        d = 3.0e-3
        signal = np.exp(-bvals * d) * 100
        return (
            signal,
            np.zeros(len(target_sphere)),
            0,
            1.0,
            1.0,
            0.0,
            np.zeros(len(target_sphere)),
        )

    fractions = np.random.dirichlet([2, 1, 1])  # WM, GM, CSF
    wm_fraction = fractions[0]
    gm_fraction = fractions[1]
    csf_fraction = fractions[2]
    if num_fib is None:
        num_fiber = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
    else:
        num_fiber = num_fib
    wm_signal, wm_label, wm_num_fib, wm_disp, _, wm_nd, wm_odf = create_wm_signal(
        num_fib=num_fiber
    )
    gm_signal, _, _, gm_disp, _, gm_nd, gm_odf = create_gm_signal()
    csf_signal, _, _, _, csf_fw, _, csf_odf = create_csf_signal()

    odi = wm_fraction * wm_disp + gm_fraction * gm_disp + csf_fraction * 1.0
    nd = wm_fraction * wm_nd + gm_fraction * gm_nd
    # Combine signals and ODFs
    combined_signal = (
        fractions[0] * wm_signal + fractions[1] * gm_signal + fractions[2] * csf_signal
    )
    combined_odf = (
        fractions[0] * wm_odf + fractions[1] * gm_odf + fractions[2] * csf_odf
    )

    if wm_fraction > thres:
        wm_label = wm_label
        wm_num_fib = wm_num_fib
    else:
        wm_label = np.zeros(len(target_sphere))
        wm_num_fib = 0
    return (
        combined_signal,
        wm_label,
        wm_num_fib,
        odi,
        wm_fraction,  # WM fraction
        gm_fraction,  # GM fraction
        csf_fraction,  # CSF fraction as free water
        nd,
        combined_odf,
    )
