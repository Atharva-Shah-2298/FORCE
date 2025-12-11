import os
import platform
from tqdm import tqdm

import numpy as np
from cython_matching import select_best_from_topk


# Dipy and related imports
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.core.sphere import Sphere
from dipy.io.peaks import save_pam
from dipy.direction import peaks_from_model

###################################### global configs ######################################
dtype_config = np.float32
label_dtype = np.uint8
chunk_size = 10000   # voxels per chunk
num_cpus = 24

###################################### data paths #########################################

if "serge" in platform.node().lower():
    bval_path = "/home/serge/data/stanford_hardi/HARDI150.bval"
    bvec_path = "/home/serge/data/stanford_hardi/HARDI150.bvec"
    data_path = "/home/serge/data/stanford_hardi/HARDI150.nii.gz"
    mask_path = "/home/serge/data/stanford_hardi/brain_mask.nii.gz"
    sims_dir = "/Users/skoudoro/data/stanford_hardi/simulated_data"
    output_dir = "/Users/skoudoro/data/stanford_hardi/hcp_output"
elif "athshah" in platform.node().lower():
    bval_path = "/home/athshah/Phi/165840/bvals"
    bvec_path = "/home/athshah/Phi/165840/bvecs"
    data_path = "/home/athshah/Phi/165840/data.nii.gz"
    mask_path = "/home/athshah/Phi/165840/mask.nii.gz"
    sims_dir = "/home/athshah/Phi/165840/simulated_data"
    output_dir = "/home/athshah/Phi/165840/hcp_output"
else:
    bval_path = ""
    bvec_path = ""
    data_path = ""
    mask_path = ""
    sims_dir = "" # folder that holds simulated_data.npz
    output_dir = ""

os.makedirs(output_dir, exist_ok=True)

###################################### load data #########################################
data, affine = load_nifti(data_path)
mask, _ = load_nifti(mask_path)
bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(bvals=bvals, bvecs=bvecs)
sphere = get_sphere(name='repulsion724')
target_sphere = sphere.vertices

# Mask is slighttly going into meninges. So erode a bit for hcp subject 165840. (optional)
# Uncomment the following lines if you want to erode the mask for hcp subject 165840.
# from scipy.ndimage import binary_erosion
# mask = binary_erosion(mask, structure=np.ones((3, 3, 3)), iterations=2)
# No need to do it for every data, particular subject has a faulty mask.
#################################### load simulated library #####################################
sims_path = os.path.join(sims_dir, 'simulated_data.npz')
sims_data = np.load(sims_path, allow_pickle=False)
use_faiss = False  # Use improved vector_search with batched processing (memory efficient like FAISS)
if not use_faiss:
    import vector_search as faiss
else:
    import faiss
# Required fields from the coherent simulator
signals        = sims_data['signals']          # (Nsims, Ngrad)
labels         = sims_data['labels']           # (Nsims, 724), 0 or 1 at peak dirs
num_fibers     = sims_data['num_fibers']       # (Nsims,)
dispersion     = sims_data['dispersion']       # (Nsims,)
wm_fraction    = sims_data['wm_fraction']      # (Nsims,)
gm_fraction    = sims_data['gm_fraction']      # (Nsims,)
csf_fraction   = sims_data['csf_fraction']     # (Nsims,)
nd             = sims_data['nd']               # (Nsims,)
odfs           = sims_data['odfs']             # (Nsims, 724)
fa             = sims_data['fa']               # (Nsims,)
md             = sims_data['md']               # (Nsims,)
rd             = sims_data['rd']               # (Nsims,)
fractions      = sims_data['fraction_array']   # (Nsims, 3), WM fiber fractions padded

# DKI metrics if saved by the simulator
ak  = sims_data['ak']  if 'ak'  in sims_data.files else None
rk  = sims_data['rk']  if 'rk'  in sims_data.files else None
mk  = sims_data['mk']  if 'mk'  in sims_data.files else None
kfa = sims_data['kfa'] if 'kfa' in sims_data.files else None

# New microFA products
ufa_wm        = sims_data['ufa_wm']            # (Nsims,)
ufa_voxel     = sims_data['ufa_voxel']         # (Nsims,)

# Optional microFA from MSDKI if present
ufa_smt = sims_data['ufa_smt2'] if 'ufa_smt2' in sims_data.files else None

###################################### parameters ######################################
penalty = 1e-5

###################################### Helper Functions ###############################
def compute_uncertainty_and_ambiguity(profile):
    """
    Compute uncertainty (IQR) and ambiguity (FWHM fraction) for each row of 'profile'.
    'profile' shape (N, M).
    """
    p75 = np.percentile(profile, 75, axis=1)
    p25 = np.percentile(profile, 25, axis=1)
    uncertainties = p75 - p25
    max_values = np.max(profile, axis=1)
    half_max = max_values / 2.0
    widths = np.sum(profile > half_max[:, None], axis=1)
    ambiguities = widths / profile.shape[1]
    return uncertainties.astype(np.float32), ambiguities.astype(np.float32)

###################################### Prep input data ######################################
maskdata = data * mask[..., None]
x, y, z, n = maskdata.shape
maskdata_flattened = maskdata.reshape((-1, n))

# Preallocate outputs in flattened voxel space
Nvox = maskdata_flattened.shape[0]
voxel_flattened = np.zeros((Nvox, target_sphere.shape[0]), dtype=dtype_config)
dispersion_map = np.zeros((Nvox,), dtype=dtype_config)
fw_map = np.zeros((Nvox,), dtype=dtype_config)
wm_map = np.zeros((Nvox,), dtype=dtype_config)
gm_map = np.zeros((Nvox,), dtype=dtype_config)
nd_map = np.zeros((Nvox,), dtype=dtype_config)
odf_map = np.zeros((Nvox, target_sphere.shape[0]), dtype=dtype_config)
num_fibers_map = np.zeros((Nvox,), dtype=dtype_config)
cleaned_dwi = np.zeros((Nvox, bvals.shape[0]), dtype=dtype_config)
fa_map = np.zeros((Nvox,), dtype=dtype_config)
md_map = np.zeros((Nvox,), dtype=dtype_config)
rd_map = np.zeros((Nvox,), dtype=dtype_config)
fraction_output = np.zeros((Nvox, 3), dtype=dtype_config)
ak_map  = np.zeros((maskdata_flattened.shape[0]), dtype=dtype_config) if ak  is not None else None
rk_map  = np.zeros((maskdata_flattened.shape[0]), dtype=dtype_config) if rk  is not None else None
mk_map  = np.zeros((maskdata_flattened.shape[0]), dtype=dtype_config) if mk  is not None else None
kfa_map = np.zeros((maskdata_flattened.shape[0]), dtype=dtype_config) if kfa is not None else None

# New microFA maps
ufa_wm_map = np.zeros((Nvox,), dtype=dtype_config)
ufa_voxel_map = np.zeros((Nvox,), dtype=dtype_config)
ufa_smt_map = np.zeros((Nvox,), dtype=dtype_config) if ufa_smt is not None else None

###################################### Normalize and build FAISS ######################################
# Normalize voxel signals
mask_norm = np.linalg.norm(maskdata_flattened, axis=1, keepdims=True)
mask_norm[mask_norm == 0] = 1.0
maskdata_norm = (maskdata_flattened / mask_norm).astype(np.float32, copy=False)
maskdata_norm = np.ascontiguousarray(maskdata_norm)

# Normalize library signals
lib_norm = np.linalg.norm(signals, axis=1, keepdims=True)
lib_norm[lib_norm == 0] = 1.0
signals_norm = (signals / lib_norm).astype(np.float32, copy=False)
signals_norm = np.ascontiguousarray(signals_norm)


###################################### Library matching with Cython ######################################
print("Building index and searching for matches...", flush=True)

# Step 1: Build index and perform top-k search using vector_search or FAISS
# Note: 'faiss' is already aliased to vector_search or real FAISS at line 51-54
ndwi = signals_norm.shape[1]
index = faiss.IndexFlatIP(ndwi)
index.add(signals_norm)
print(f"Index built with {index.ntotal if hasattr(index, 'ntotal') else len(signals_norm)} vectors, dimension {ndwi}", flush=True)

# Search: vector_search supports n_threads parameter and batched processing, real FAISS doesn't need n_threads
if use_faiss:
    print(f"Searching {maskdata_norm.shape[0]} voxels (k=50) using FAISS...", flush=True)
    D, I = index.search(maskdata_norm, k=50)
else:
    # Improved vector_search with internal batched processing (memory efficient)
    print(f"Searching {maskdata_norm.shape[0]} voxels (k=50, n_threads={num_cpus}) using vector_search...", flush=True)
    D, I = index.search(maskdata_norm, k=50, n_threads=num_cpus)

print(f"Top-k search completed for {Nvox} voxels", flush=True)

# Step 2: Apply penalties and select best match using Cython
# Penalty by number of fibers in library
num_fibers = np.ascontiguousarray(num_fibers.astype(np.float32))
penalty_array = penalty * num_fibers

best_indices = select_best_from_topk(
    top_scores=D,                # (nvoxels, 50)
    top_indices=I,               # (nvoxels, 50)
    penalty_array=penalty_array, # (nsims,)
    n_threads=num_cpus           # 24
)

print("Best matches selected after penalization", flush=True)

# Step 3: Directly index all output arrays with best_indices
voxel_flattened = labels[best_indices]
dispersion_map = dispersion[best_indices]
fw_map = csf_fraction[best_indices]
wm_map = wm_fraction[best_indices]
gm_map = gm_fraction[best_indices]
nd_map = nd[best_indices]
odf_map = odfs[best_indices]
num_fibers_map = num_fibers[best_indices]
cleaned_dwi = signals[best_indices]
fa_map = fa[best_indices]
md_map = md[best_indices]
rd_map = rd[best_indices]
fraction_output = fractions[best_indices]
ufa_wm_map = ufa_wm[best_indices]
ufa_voxel_map = ufa_voxel[best_indices]
if ufa_smt_map is not None:
    ufa_smt_map = ufa_smt[best_indices]
if ak is not None:
    ak_map = ak[best_indices]
    rk_map = rk[best_indices]
    mk_map = mk[best_indices]
    kfa_map = kfa[best_indices]

print("All parameters matched and indexed", flush=True)


###################################### Peaks postprocessing ######################################
def postprocess_peaks(preds, target_sphere, fracs):
    """
    Convert binary peak masks to exactly 5 peaks per sample by padding zeros.
    Returns:
        peaks_output: (N, 5, 3)
        peak_indices: (N, 5)
        peak_values:  (N, 5)
    """
    peaks_output = []
    peak_indices = []
    peak_vals = []

    for i in tqdm(range(preds.shape[0]), desc="Postprocessing peaks"):
        coords = target_sphere[preds[i] == 1]
        indices = np.where(preds[i] == 1)[0]
        vals = np.zeros(5, dtype=np.float32)

        tmp_peaks = np.zeros((5, 3), dtype=np.float32)
        tmp_indices = np.ones(5, dtype=np.int32) * -1

        n = min(len(coords), 5)
        if n > 0:
            tmp_peaks[:n] = coords[:n]
            tmp_indices[:n] = indices[:n]
        vals[:min(5, fracs[i].shape[0])] = fracs[i][:min(5, fracs[i].shape[0])]
        peaks_output.append(tmp_peaks)
        peak_indices.append(tmp_indices)
        peak_vals.append(vals)

    return np.array(peaks_output), np.array(peak_indices), np.array(peak_vals)

peaks, peak_indices, peak_values = postprocess_peaks(voxel_flattened, target_sphere, fraction_output)

# Reshape back to 3D
peaks_output = peaks.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 5, 3)
peak_indices = peak_indices.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 5)
peak_values = peak_values.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 5)

# Apply brain mask
peaks_output = peaks_output * mask[..., None, None]
peak_indices = peak_indices * mask[..., None]
peak_values = peak_values * mask[..., None]

dispersion_map = dispersion_map.reshape(mask.shape) * mask
fw_map = fw_map.reshape(mask.shape) * mask
nd_map = nd_map.reshape(mask.shape) * mask
wm_map = wm_map.reshape(mask.shape) * mask
gm_map = gm_map.reshape(mask.shape) * mask
num_fibers_map = num_fibers_map.reshape(mask.shape) * mask
cleaned_dwi = cleaned_dwi.reshape(mask.shape[0], mask.shape[1], mask.shape[2], bvals.shape[0]) * mask[..., None]
fa_map = fa_map.reshape(mask.shape) * mask
md_map = md_map.reshape(mask.shape) * mask
rd_map = rd_map.reshape(mask.shape) * mask

# microFA maps back to 3D
ufa_wm_map = ufa_wm_map.reshape(mask.shape) * mask
ufa_voxel_map = ufa_voxel_map.reshape(mask.shape) * mask
if ufa_smt_map is not None:
    ufa_smt_map = ufa_smt_map.reshape(mask.shape) * mask

if ak is not None:
    ak_map  = ak_map.reshape(mask.shape)  * mask
    rk_map  = rk_map.reshape(mask.shape)  * mask
    mk_map  = mk_map.reshape(mask.shape)  * mask
    kfa_map = kfa_map.reshape(mask.shape) * mask


###################################### CSA peaks on raw data ######################################

print("Computing CSA peaks.", flush=True)
csa_model = CsaOdfModel(gtab, sh_order_max=8)
csa_peaks = peaks_from_model(
    csa_model, data, sphere,
    relative_peak_threshold=.5,
    min_separation_angle=45,
    parallel=False,
    mask=mask,
    normalize_peaks=True
)

###################################### Package and save ######################################
print("Saving outputs...", flush=True)

class PeakObject:
    def __init__(self, peak_dirs, peak_values, peak_indices, affine):
        self.peak_dirs = peak_dirs
        self.affine = affine
        self.peak_values = peak_values
        self.peak_indices = peak_indices
        self.shm_coeff = odf_map
        self.sphere = Sphere(xyz=target_sphere)
        # attributes expected by save_pam so populate from csa_peaks but not used anywhere in the tracking or any downstream tasks
        self.B = csa_peaks.B
        self.total_weight = csa_peaks.total_weight
        self.ang_thr = csa_peaks.ang_thr
        self.gfa = csa_peaks.gfa
        self.qa = csa_peaks.qa
        self.odf = odf_map

pam_obj = PeakObject(peak_dirs=peaks_output, peak_values=peak_values, peak_indices=peak_indices, affine=affine)
save_pam(os.path.join(output_dir, f'peaks_{penalty}.pam5'), pam_obj)
np.save(os.path.join(output_dir, f'odf_map_raw_{penalty}.npy'), odf_map)

save_nifti(os.path.join(output_dir, f'dispersion_map_{penalty}.nii.gz'), dispersion_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'fw_map_{penalty}.nii.gz'), fw_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'wm_map_{penalty}.nii.gz'), wm_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'gm_map_{penalty}.nii.gz'), gm_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'num_fibers_map_{penalty}.nii.gz'), num_fibers_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'cleaned_dwi_{penalty}.nii.gz'), cleaned_dwi.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'fa_map_{penalty}.nii.gz'), fa_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'md_map_{penalty}.nii.gz'), md_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'rd_map_{penalty}.nii.gz'), rd_map.astype(np.float32), affine)

# microFA maps
save_nifti(os.path.join(output_dir, f'ufa_wm_map_{penalty}.nii.gz'), ufa_wm_map.astype(np.float32), affine)
save_nifti(os.path.join(output_dir, f'ufa_voxel_map_{penalty}.nii.gz'), ufa_voxel_map.astype(np.float32), affine)
if ufa_smt_map is not None:
    save_nifti(os.path.join(output_dir, f'ufa_smt_map_{penalty}.nii.gz'), ufa_smt_map.astype(np.float32), affine)
if ak is not None:
    save_nifti(os.path.join(output_dir, f'ak_map_{penalty}.nii.gz'),  ak_map.astype(np.float32),  affine)
    save_nifti(os.path.join(output_dir, f'rk_map_{penalty}.nii.gz'),  rk_map.astype(np.float32),  affine)
    save_nifti(os.path.join(output_dir, f'mk_map_{penalty}.nii.gz'),  mk_map.astype(np.float32),  affine)
    save_nifti(os.path.join(output_dir, f'kfa_map_{penalty}.nii.gz'), kfa_map.astype(np.float32), affine)

print("Done.")
