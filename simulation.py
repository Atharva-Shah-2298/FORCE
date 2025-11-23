import numpy as np
import os
import time
import gc
import ray
from tqdm import tqdm
import psutil
# Dipy and related imports
from dipy.data import get_sphere, default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import all_tensor_evecs
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere
from dipy.io.peaks import save_peaks
import dipy.reconst.dti as dti
from dipy.reconst.dki import DiffusionKurtosisModel
import dipy.reconst.msdki as msdki

from faster_multitensor import multi_tensor
from utils.geometry import angle_between, is_angle_valid
from utils.distribution import bingham_dictionary
from utils.analytical import multi_tensor_dki

###################################### bval and bvec paths #########################################

bval_path = "/home/athshah/Phi/165840/bvals"
bvec_path = "/home/athshah/Phi/165840/bvecs"
output_dir = "/home/athshah/submission/simulated_data"
os.makedirs(output_dir, exist_ok=True)

###################################### sphere and gtab imports ######################################

sphere = get_sphere(name='repulsion724')
target_sphere = sphere.vertices
bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals, bvecs)
bvecs = np.ascontiguousarray(bvecs)

###################################### Simulation Functions ######################################

# list of odi values between 0.01 and 0.3 spaced out equally
odi_list = np.linspace(0.01, 0.3, 10)
evecs = np.array([all_tensor_evecs(tuple(point)) for point in target_sphere])
bingham_sf = bingham_dictionary(target_sphere, odi_list)

####################################### Global Variables #########################################
tortuisity = False
num_simulations = 500_000
wm_threshold = 0.5  # minimum white matter volume fraction for a voxel to have valid peaks
dtype_config = np.float32
label_dtype = np.uint8
num_cpus = 24

# Optional heavy model fits for a one on one comparison
run_dki = False
run_msdki = True

###################################### Helper ##########################################
def get_dperp_extra(d_par, f_intra):
    return d_par * (1 - f_intra) / (1 + f_intra)

################ MicroFA helpers #######################
def microfa_stick_zeppelin(Da, dperp_ex, W_in, W_ex, eps=1e-12):
    """
    microFA for a mixture of sticks (Da, 0, 0) and zeppelins (Da, dperp_ex, dperp_ex),
    with weights W_in and W_ex for intra and extra, respectively.
    """
    if (W_in + W_ex) <= eps:
        return 0.0
    if W_ex <= eps:
        return 1.0
    num = W_in * (Da ** 2) + W_ex * ((Da - dperp_ex) ** 2)
    den = W_in * (Da ** 2) + W_ex * (Da ** 2 + 2.0 * (dperp_ex ** 2))
    return float(np.sqrt(num / den)) if den > 0 else 0.0

def microfa_voxel_from_parts(Da, dperp_ex, W_in, W_ex, wm_frac, gm_frac, d_gm, csf_frac, d_csf, eps=1e-12):
    """
    microFA for full voxel: WM anisotropy in numerator, WM + GM + CSF in denominator.
    GM and CSF are isotropic so they add to the denominator only.
    """
    if wm_frac + gm_frac + csf_frac <= eps:
        return 0.0
    wm_num = W_in * (Da ** 2) + W_ex * ((Da - dperp_ex) ** 2)
    wm_den = W_in * (Da ** 2) + W_ex * (Da ** 2 + 2.0 * (dperp_ex ** 2))
    gm_den = 3.0 * (d_gm ** 2)
    csf_den = 3.0 * (d_csf ** 2)
    total_num = wm_frac * wm_num
    total_den = wm_frac * wm_den + gm_frac * gm_den + csf_frac * csf_den
    return float(np.sqrt(total_num / total_den)) if total_den > eps else 0.0

###################################### Single fiber simulation ##########################################
def generate_single_fiber():
    """Generate a single-fiber diffusion signal with random parameters."""
    f_intra = np.random.uniform(0.6, 0.9)
    f_extra = 1 - f_intra

    fiber_fractions = 1.0

    labels = np.zeros((target_sphere.shape[0]), dtype=label_dtype)
    S0 = 100.0
    d_par = np.random.uniform(1.5e-3, 3.0e-3)
    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, f_intra)
    else:
        d_perp_extra = np.random.uniform(0.3e-3, 1.5e-3)

    mevals_ex = np.zeros(np.shape(target_sphere))
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros(np.shape(target_sphere))
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    index = np.random.randint(0, target_sphere.shape[0], 1)[0]
    true_stick = target_sphere[index]
    factor = np.random.choice(odi_list)

    fodf_gt = bingham_sf[index][factor]
    fodf_gt = fodf_gt / np.sum(fodf_gt)

    S_in = multi_tensor(mevals_in, evecs, fodf_gt * 100, bvals, bvecs)
    S_ex = multi_tensor(mevals_ex, evecs, fodf_gt * 100, bvals, bvecs)

    S = fiber_fractions * (f_intra * S_in + f_extra * S_ex)

    labels[np.argmin(np.linalg.norm(target_sphere - true_stick, axis=1))] = 1

    # wm_d_perp returned here equals (1 - f_intra) * d_perp_extra
    return (
        S * S0,
        labels,
        1,
        factor,
        0.0,
        fiber_fractions * f_intra,  # wm_nd
        fodf_gt,
        d_par,
        f_extra * d_perp_extra,     # weighted
        [1.0, 0.0, 0.0],            # fiber fractions inside WM
        [f_intra],                  # f_ins per fiber
    )

###################################### Two fiber simulation ##########################################
def generate_two_fibers():
    """Generate a two-fiber diffusion signal with random parameters."""
    f_in = np.random.uniform(0.6, 0.9, 2)
    fiber_frac1 = np.random.uniform(0.2, 0.8)
    fiber_fractions = [fiber_frac1, 1 - fiber_frac1]

    d_par = np.random.uniform(1.5e-3, 3.0e-3)
    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, f_in[0])
    else:
        d_perp_extra = np.random.uniform(0.3e-3, 1.5e-3)

    labels = np.zeros((target_sphere.shape[0]), dtype=label_dtype)
    S0 = 100.0
    mevals_ex = np.zeros(np.shape(target_sphere))
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros(np.shape(target_sphere))
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    index = np.random.randint(0, target_sphere.shape[0], 2)

    while not is_angle_valid(angle_between(target_sphere[index[0]], target_sphere[index[1]])):
        index = np.random.randint(0, target_sphere.shape[0], 2)

    true_stick1 = target_sphere[index[0]]
    true_stick2 = target_sphere[index[1]]
    factor = np.random.choice(odi_list)

    S = np.zeros((len(bvals)))
    fodf = np.zeros((target_sphere.shape[0]))
    for i in range(2):
        fodf_gt = bingham_sf[index[i]][factor]
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        fodf += fiber_fractions[i] * fodf_gt

        S_in = multi_tensor(mevals_in, evecs, fodf_gt * 100, bvals, bvecs)
        S_ex = multi_tensor(mevals_ex, evecs, fodf_gt * 100, bvals, bvecs)

        f_intra = f_in[i]
        f_extra = 1 - f_intra

        S += fiber_fractions[i] * (f_intra * S_in + f_extra * S_ex)

        labels[np.argmin(np.linalg.norm(target_sphere - true_stick1, axis=1))] = 1
        labels[np.argmin(np.linalg.norm(target_sphere - true_stick2, axis=1))] = 1

    return (
        S * S0,
        labels,
        2,
        factor,
        0.0,
        fiber_fractions[0] * f_in[0] + fiber_fractions[1] * f_in[1],  # wm_nd
        fodf,
        d_par,
        fiber_fractions[0] * (1 - f_in[0]) * d_perp_extra + fiber_fractions[1] * (1 - f_in[1]) * d_perp_extra,
        [fiber_frac1, 1 - fiber_frac1, 0.0],
        f_in.tolist(),
    )

###################################### Three fiber simulation ##########################################
def generate_three_fibers():
    """Generate a three-fiber diffusion signal with random parameters."""
    f_in = np.random.uniform(0.6, 0.9, 3)
    fiber_fracs = np.random.dirichlet([1, 1, 1])
    while any(fiber_fracs < 0.2):
        fiber_fracs = np.random.dirichlet([1, 1, 1])

    d_par = np.random.uniform(1.5e-3, 3.0e-3)
    if tortuisity:
        d_perp_extra = get_dperp_extra(d_par, f_in[0])
    else:
        d_perp_extra = np.random.uniform(0.3e-3, 1.5e-3)

    labels = np.zeros((target_sphere.shape[0]), dtype=label_dtype)
    S0 = 100.0
    mevals_ex = np.zeros(np.shape(target_sphere))
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = mevals_ex[:, 1]

    mevals_in = np.zeros(np.shape(target_sphere))
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    index = np.random.randint(0, target_sphere.shape[0], 3)

    while (
        not is_angle_valid(angle_between(target_sphere[index[0]], target_sphere[index[1]]), threshold=60)
        or not is_angle_valid(angle_between(target_sphere[index[0]], target_sphere[index[2]]), threshold=60)
        or not is_angle_valid(angle_between(target_sphere[index[1]], target_sphere[index[2]]), threshold=60)
    ):
        index = np.random.randint(0, target_sphere.shape[0], 3)

    true_stick1 = target_sphere[index[0]]
    true_stick2 = target_sphere[index[1]]
    true_stick3 = target_sphere[index[2]]
    factor = np.random.choice(odi_list)

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
    labels[np.argmin(np.linalg.norm(target_sphere - true_stick2, axis=1))] = 1
    labels[np.argmin(np.linalg.norm(target_sphere - true_stick3, axis=1))] = 1

    return (
        S * S0,
        labels,
        3,
        factor,
        0.0,
        fiber_fracs[0] * f_in[0] + fiber_fracs[1] * f_in[1] + fiber_fracs[2] * f_in[2],  # wm_nd
        fodf,
        d_par,
        fiber_fracs[0] * (1 - f_in[0]) * d_perp_extra + fiber_fracs[1] * (1 - f_in[1]) * d_perp_extra + fiber_fracs[2] * (1 - f_in[2]) * d_perp_extra,
        fiber_fracs.tolist(),
        f_in.tolist(),
    )

def create_wm_signal(num_fib):
    """Generate a white matter signal with random fiber configuration."""
    if num_fib == 1:
        return generate_single_fiber()
    elif num_fib == 2:
        return generate_two_fibers()
    elif num_fib == 3:
        return generate_three_fibers()

def create_gm_signal():
    """Generate a gray matter signal (isotropic)."""
    d = np.random.uniform(0.7e-3, 1.2e-3)
    signal = np.exp(-bvals * d) * 100.0
    return signal, np.zeros(len(target_sphere), dtype=label_dtype), 0, 1.0, 0.0, 0.0, np.ones(len(target_sphere)) / len(target_sphere), d, d

def create_csf_signal():
    """Generate a CSF signal (isotropic)."""
    d = 3.0e-3
    signal = np.exp(-bvals * d) * 100.0
    return signal, np.zeros(len(target_sphere), dtype=label_dtype), 0, 1.0, 1.0, 0.0, np.zeros(len(target_sphere)), d, d

################ Mixed tissue simulation #######################

def create_mixed_signal():
    """Generate a mixed signal combining WM, GM, and CSF."""

    # Randomly sample fractions for WM, GM, and CSF
    fractions = np.random.dirichlet([2.0, 1.0, 1.0])  # WM, GM, CSF
    wm_fraction = float(fractions[0])
    gm_fraction = float(fractions[1])
    csf_fraction = float(fractions[2])

    # Choose a random number of fibers for WM
    num_fiber = int(np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7]))

    # create signals for each tissue type
    wm_signal, wm_label, wm_num_fib, wm_disp, _, wm_nd, wm_odf, wm_d_par, wm_d_perp, fracs, f_ins = create_wm_signal(num_fib=num_fiber)
    gm_signal, _, _, gm_disp, _, gm_nd, gm_odf, gm_d_par, gm_d_perp = create_gm_signal()
    csf_signal, _, _, _, csf_fw, _, csf_odf, csf_d_par, csf_d_perp = create_csf_signal()

    # Combined ODI and ND
    odi = float(wm_fraction * wm_disp + gm_fraction * gm_disp + csf_fraction * 1.0)
    nd  = float(wm_fraction * wm_nd + gm_fraction * gm_nd)

    # Combine signals
    combined_signal = wm_fraction * wm_signal + gm_fraction * gm_signal + csf_fraction * csf_signal

    # If WM fraction is below threshold, set WM labels and ODF to zero for export
    if wm_fraction > wm_threshold:
        combined_odf = 50.0 * wm_fraction * wm_odf  # scaled for visualization
    else:
        wm_label = np.zeros(len(target_sphere), dtype=label_dtype)
        combined_odf = np.zeros(len(target_sphere), dtype=np.float16)

    # ---------- microFA of WM only ----------
    fracs_arr = np.asarray(fracs[:num_fiber], dtype=float)
    f_ins_arr = np.asarray(f_ins[:num_fiber], dtype=float)

    W_in = float(np.sum(fracs_arr * f_ins_arr)) if fracs_arr.size else 0.0
    W_ex = float(np.sum(fracs_arr * (1.0 - f_ins_arr))) if fracs_arr.size else 0.0

    # Recover shared extra radial diffusivity
    d_perp_ex = float(wm_d_perp) / W_ex if W_ex > 0 else 0.0
    Da = float(wm_d_par)

    ufa_wm = microfa_stick_zeppelin(Da, d_perp_ex, W_in, W_ex)

    # ---------- microFA of full voxel ----------
    ufa_voxel = microfa_voxel_from_parts(
        Da=Da,
        dperp_ex=d_perp_ex,
        W_in=W_in,
        W_ex=W_ex,
        wm_frac=wm_fraction,
        gm_frac=gm_fraction,
        d_gm=float(gm_d_par),
        csf_frac=csf_fraction,
        d_csf=float(csf_d_par),
    )

    # return all the metrics and signals
    # Indices must match the consumer loop below
    return (
        combined_signal,                    # 0
        wm_label,                           # 1
        wm_num_fib,                         # 2
        odi,                                # 3
        wm_fraction,                        # 4
        gm_fraction,                        # 5
        csf_fraction,                       # 6
        nd,                                 # 7
        combined_odf.astype(np.float16),    # 8
        ufa_wm,                             # 9  microFA for WM only
        ufa_voxel,                          # 10 microFA for full voxel
        fracs                                # 11 fiber fractions inside WM (len 1..3)
    )

@ray.remote
def generate_mixed():
    return create_mixed_signal()

################################## Main Simulation Loop ##########################################
ray.init(num_cpus=num_cpus)

results = []
futures = [generate_mixed.remote() for _ in range(num_simulations)]
for future in tqdm(futures, desc="Generating mixed signals"):
    results.append(ray.get(future))

##################################### Preallocation ##########################################
signals = np.zeros((num_simulations, len(bvals)), dtype=dtype_config)
labels = np.zeros((num_simulations, len(target_sphere)), dtype=label_dtype)
num_fibers = np.zeros(num_simulations, dtype=dtype_config)
dispersion = np.zeros(num_simulations, dtype=dtype_config)
wm_fraction_arr = np.zeros(num_simulations, dtype=dtype_config)
gm_fraction_arr = np.zeros(num_simulations, dtype=dtype_config)
csf_fraction_arr = np.zeros(num_simulations, dtype=dtype_config)
nd_arr = np.zeros(num_simulations, dtype=dtype_config)
odfs = np.zeros((num_simulations, len(target_sphere)), dtype=np.float16)
fraction_array = np.zeros((num_simulations, 3), dtype=dtype_config)
ufa_wm_arr = np.zeros(num_simulations, dtype=dtype_config)
ufa_voxel_arr = np.zeros(num_simulations, dtype=dtype_config)

for i, res in enumerate(results):
    signals[i] = res[0]
    labels[i] = res[1]
    num_fibers[i] = res[2]
    dispersion[i] = res[3]
    wm_fraction_arr[i] = res[4]
    gm_fraction_arr[i] = res[5]
    csf_fraction_arr[i] = res[6]
    nd_arr[i] = res[7]
    odfs[i] = res[8]
    ufa_wm_arr[i] = res[9]
    ufa_voxel_arr[i] = res[10]
    fraction_array[i] = res[11]

################################### DTI computation ##########################################
# take the lowest shell only for DTI

def smallest_shell_bval(bvals, b0_threshold=50, shell_tolerance=50):
    """
    Return (min_shell_bval, mask) where:
      - min_shell_bval is the smallest non-zero shell (rounded by tolerance)
      - mask is a boolean array selecting volumes in that shell
    """
    bvals = np.asarray(bvals, dtype=float)
    non_b0 = bvals > b0_threshold
    if not np.any(non_b0):
        raise ValueError("No non-b0 volumes found.")
    rounded = np.round(bvals[non_b0] / shell_tolerance) * shell_tolerance
    min_shell = float(np.min(rounded))
    shell_mask = np.isclose(np.round(bvals / shell_tolerance) * shell_tolerance, min_shell)
    return min_shell, shell_mask

min_b, shell_mask = smallest_shell_bval(bvals, b0_threshold=50, shell_tolerance=50)
print("Smallest non-zero shell b-value:", min_b)

# Subset for DTI on lowest shell plus b0s
b0_mask = bvals <= 50
use_mask = shell_mask | b0_mask
bvals_small = bvals[use_mask]
bvecs_small = bvecs[use_mask]
gtab_small = gradient_table(bvals_small, bvecs_small)

dti_model = dti.TensorModel(gtab_small)
dti_fit = dti_model.fit(signals[:, use_mask])

fa_dti = dti_fit.fa.astype(dtype_config)
md_dti = dti_fit.md.astype(dtype_config)
rd_dti = dti_fit.rd.astype(dtype_config)

################################### Optional DKI computation ##########################################
if run_dki:
    # Only include b-values up to 2500 for DKI
    mask_dki = bvals <= 2500
    bvals_dki = bvals[mask_dki]
    bvecs_dki = bvecs[mask_dki]
    signals_dki = signals[:, mask_dki]
    gtab_dki = gradient_table(bvals_dki, bvecs_dki)

    dki_model = DiffusionKurtosisModel(gtab_dki)
    # Fit per voxel. For very large num_simulations this will be slow.
    dki_fit = dki_model.fit(signals_dki)
    ak_arr  = dki_fit.ak().astype(dtype_config)
    rk_arr  = dki_fit.rk().astype(dtype_config)
    mk_arr  = dki_fit.mk().astype(dtype_config)
    kfa_arr = dki_fit.kfa().astype(dtype_config)
else:
    ak_arr  = np.zeros(num_simulations, dtype=dtype_config)
    rk_arr  = np.zeros(num_simulations, dtype=dtype_config)
    mk_arr  = np.zeros(num_simulations, dtype=dtype_config)
    kfa_arr = np.zeros(num_simulations, dtype=dtype_config)

################################### Optional msdki microFA ##########################################
if run_msdki:
    mask_dki = bvals <= 2500
    bvals_dki = bvals[mask_dki]
    bvecs_dki = bvecs[mask_dki]
    signals_dki = signals[:, mask_dki]
    gtab_dki = gradient_table(bvals_dki, bvecs_dki)

    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab_dki)
    msdki_fit = msdki_model.fit(data=signals_dki, mask=None)
    ufa_smt2 = msdki_fit.smt2uFA.astype(dtype_config)
else:
    ufa_smt2 = np.zeros(num_simulations, dtype=dtype_config)

# Save the results to disk
np.savez_compressed(
    os.path.join(output_dir, 'simulated_data.npz'),
    signals=signals,
    labels=labels,
    num_fibers=num_fibers,
    dispersion=dispersion,
    wm_fraction=wm_fraction_arr,
    gm_fraction=gm_fraction_arr,
    csf_fraction=csf_fraction_arr,
    nd=nd_arr,
    odfs=odfs,
    # DTI metrics from the smallest shell
    fa=fa_dti,
    md=md_dti,
    rd=rd_dti,
    # MicroFA exports
    ufa_wm=ufa_wm_arr,
    ufa_voxel=ufa_voxel_arr,
    # Optional DKI exports
    ak=ak_arr,
    rk=rk_arr,
    mk=mk_arr,
    kfa=kfa_arr,
    # Optional msdki microFA
    ufa_smt2=ufa_smt2,
    fraction_array=fraction_array
)

print("Saved", os.path.join(output_dir, 'simulated_data.npz'))
