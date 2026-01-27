import os
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import psutil  # for memory usage

from dipy.data import default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import all_tensor_evecs, multi_tensor_dki
from dipy.reconst import dti
from dipy.reconst.dki import (
    DiffusionKurtosisModel,
    axial_kurtosis,
    radial_kurtosis,
    mean_kurtosis,
    kurtosis_fractional_anisotropy,
)
import dipy.reconst.msdki as msdki

from utils.distribution import bingham_dictionary
from sim_core import create_mixed_signal, set_diffusivity_ranges

# Limit BLAS threading so our parallelism is controllable
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def print_mem_usage(label):
    """Print RSS memory usage of this Python process in GB."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM] {label}: {rss_gb:.2f} GB RSS")


def smallest_shell_bval(bvals, b0_threshold=50, shell_tolerance=50):
    """
    Return (min_shell_bval, mask) where:
      - min_shell_bval is the smallest non zero shell (rounded by tolerance)
      - mask is a boolean array selecting volumes in that shell
    """
    bvals = np.asarray(bvals, dtype=float)
    non_b0 = bvals > b0_threshold
    if not np.any(non_b0):
        raise ValueError("No non b0 volumes found.")
    rounded = np.round(bvals[non_b0] / shell_tolerance) * shell_tolerance
    min_shell = float(np.min(rounded))
    shell_mask = np.isclose(
        np.round(bvals / shell_tolerance) * shell_tolerance, min_shell
    )
    return min_shell, shell_mask


def compute_analytical_dki_batch(
    start_idx,
    batch_size,
    target_sphere,
    bingham_sf,
    odi_list,
    gtab_bvals,
    gtab_bvecs,
    input_memmap_info,
    output_memmap_info,
):
    """
    Worker function to compute analytical DKI for a batch of voxels.
    """
    from dipy.core.gradients import gradient_table
    from dipy.sims.voxel import multi_tensor_dki
    from dipy.reconst import dti
    from dipy.reconst.dki import (
        axial_kurtosis,
        radial_kurtosis,
        mean_kurtosis,
        kurtosis_fractional_anisotropy,
    )
    
    # Unpack input memmap info
    (
        num_fibers_path, num_fibers_shape, num_fibers_dtype,
        labels_path, labels_shape, labels_dtype,
        wm_disp_path, wm_disp_shape, wm_disp_dtype,
        wm_d_par_path, wm_d_par_shape, wm_d_par_dtype,
        wm_d_perp_path, wm_d_perp_shape, wm_d_perp_dtype,
        gm_d_par_path, gm_d_par_shape, gm_d_par_dtype,
        csf_d_par_path, csf_d_par_shape, csf_d_par_dtype,
        wm_fraction_path, wm_fraction_shape, wm_fraction_dtype,
        gm_fraction_path, gm_fraction_shape, gm_fraction_dtype,
        csf_fraction_path, csf_fraction_shape, csf_fraction_dtype,
        fraction_array_path, fraction_array_shape, fraction_array_dtype,
        f_ins_path, f_ins_shape, f_ins_dtype,
    ) = input_memmap_info
    
    # Unpack output memmap info
    (
        ak_path, ak_shape, ak_dtype,
        rk_path, rk_shape, rk_dtype,
        mk_path, mk_shape, mk_dtype,
        kfa_path, kfa_shape, kfa_dtype,
    ) = output_memmap_info
    
    # Open input memmaps (read-only)
    num_fibers_mm = np.memmap(num_fibers_path, mode="r", dtype=num_fibers_dtype, shape=num_fibers_shape)
    labels_mm = np.memmap(labels_path, mode="r", dtype=labels_dtype, shape=labels_shape)
    wm_disp_mm = np.memmap(wm_disp_path, mode="r", dtype=wm_disp_dtype, shape=wm_disp_shape)
    wm_d_par_mm = np.memmap(wm_d_par_path, mode="r", dtype=wm_d_par_dtype, shape=wm_d_par_shape)
    wm_d_perp_mm = np.memmap(wm_d_perp_path, mode="r", dtype=wm_d_perp_dtype, shape=wm_d_perp_shape)
    gm_d_par_mm = np.memmap(gm_d_par_path, mode="r", dtype=gm_d_par_dtype, shape=gm_d_par_shape)
    csf_d_par_mm = np.memmap(csf_d_par_path, mode="r", dtype=csf_d_par_dtype, shape=csf_d_par_shape)
    wm_fraction_mm = np.memmap(wm_fraction_path, mode="r", dtype=wm_fraction_dtype, shape=wm_fraction_shape)
    gm_fraction_mm = np.memmap(gm_fraction_path, mode="r", dtype=gm_fraction_dtype, shape=gm_fraction_shape)
    csf_fraction_mm = np.memmap(csf_fraction_path, mode="r", dtype=csf_fraction_dtype, shape=csf_fraction_shape)
    fraction_array_mm = np.memmap(fraction_array_path, mode="r", dtype=fraction_array_dtype, shape=fraction_array_shape)
    f_ins_mm = np.memmap(f_ins_path, mode="r", dtype=f_ins_dtype, shape=f_ins_shape)
    
    # Open output memmaps (read-write)
    ak_mm = np.memmap(ak_path, mode="r+", dtype=ak_dtype, shape=ak_shape)
    rk_mm = np.memmap(rk_path, mode="r+", dtype=rk_dtype, shape=rk_shape)
    mk_mm = np.memmap(mk_path, mode="r+", dtype=mk_dtype, shape=mk_shape)
    kfa_mm = np.memmap(kfa_path, mode="r+", dtype=kfa_dtype, shape=kfa_shape)
    
    # Recreate gradient table in worker
    gtab = gradient_table(gtab_bvals, bvecs=gtab_bvecs)
    
    L = len(target_sphere)
    
    for i in range(batch_size):
        idx = start_idx + i
        num_fiber = int(num_fibers_mm[idx])
        
        if num_fiber == 0:
            continue
        
        wm_label = labels_mm[idx]
        wm_disp = float(wm_disp_mm[idx])
        wm_d_par = float(wm_d_par_mm[idx])
        wm_d_perp = float(wm_d_perp_mm[idx])
        gm_d_par = float(gm_d_par_mm[idx])
        csf_d_par = float(csf_d_par_mm[idx])
        wm_fraction = float(wm_fraction_mm[idx])
        gm_fraction = float(gm_fraction_mm[idx])
        csf_fraction = float(csf_fraction_mm[idx])
        fracs = fraction_array_mm[idx]
        f_ins = f_ins_mm[idx]
        
        mevals_total = np.zeros(((2 * num_fiber) * L + 2, 3))
        angles_orient = np.vstack([target_sphere] * (2 * num_fiber)).astype(np.float64)
        angles_orient = np.vstack([angles_orient, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        fractions_total = np.zeros((2 * num_fiber * L + 2))
        
        index_label = np.where(wm_label == 1)[0]
        
        # Find nearest ODI value
        odi_idx = np.argmin(np.abs(odi_list - wm_disp))
        nearest_odi = odi_list[odi_idx]
        
        for k in range(num_fiber):
            if k < len(index_label):
                odf_w = bingham_sf[index_label[k]][nearest_odi].astype(np.float64)
            else:
                odf_w = np.ones(L, dtype=np.float64) / L
            odf_w = odf_w / odf_w.sum()
            
            start = k * 2 * L
            mid = start + L
            end = mid + L
            
            mevals_total[start:mid, 0] = wm_d_par
            mevals_total[start:mid, 1] = 0.0
            mevals_total[start:mid, 2] = 0.0
            fractions_total[start:mid] = wm_fraction * fracs[k] * f_ins[k] * odf_w
            
            mevals_total[mid:end, 0] = wm_d_par
            mevals_total[mid:end, 1] = wm_d_perp
            mevals_total[mid:end, 2] = wm_d_perp
            fractions_total[mid:end] = wm_fraction * fracs[k] * (1.0 - f_ins[k]) * odf_w
        
        mevals_total[-2, :] = gm_d_par
        fractions_total[-2] = gm_fraction
        mevals_total[-1, :] = csf_d_par
        fractions_total[-1] = csf_fraction
        
        try:
            _, dt, kt = multi_tensor_dki(
                gtab, mevals_total, angles=angles_orient,
                fractions=fractions_total * 100, S0=100
            )
            
            dt_evals, dt_evecs = dti.decompose_tensor(dti.from_lower_triangular(dt))
            dki_params = np.concatenate([dt_evals.ravel(), dt_evecs.ravel(), kt.ravel()])
            
            ak_mm[idx] = axial_kurtosis(dki_params)
            rk_mm[idx] = radial_kurtosis(dki_params)
            mk_mm[idx] = mean_kurtosis(dki_params)
            kfa_mm[idx] = kurtosis_fractional_anisotropy(dki_params)
        except Exception:
            pass
    
    ak_mm.flush()
    rk_mm.flush()
    mk_mm.flush()
    kfa_mm.flush()
    
    return batch_size


def generate_batch_to_memmap(
    start_idx,
    batch_size,
    sphere,
    evecs,
    bingham,
    odi,
    bval,
    bvec,
    wm_thresh,
    tort,
    memmap_info,
    diffusivity_cfg,
):
    """
    Worker function for multiprocessing parallelism.
    Opens memmaps by path in each process and writes directly.
    Returns batch_size for progress tracking.
    """
    # Unpack memmap info and open them in this process
    (
        signals_path, signals_shape, signals_dtype,
        labels_path, labels_shape, labels_dtype,
        num_fibers_path, num_fibers_shape, num_fibers_dtype,
        dispersion_path, dispersion_shape, dispersion_dtype,
        wm_fraction_path, wm_fraction_shape, wm_fraction_dtype,
        gm_fraction_path, gm_fraction_shape, gm_fraction_dtype,
        csf_fraction_path, csf_fraction_shape, csf_fraction_dtype,
        nd_path, nd_shape, nd_dtype,
        odfs_path, odfs_shape, odfs_dtype,
        ufa_wm_path, ufa_wm_shape, ufa_wm_dtype,
        ufa_voxel_path, ufa_voxel_shape, ufa_voxel_dtype,
        fraction_array_path, fraction_array_shape, fraction_array_dtype,
        # Additional DKI parameters
        wm_disp_path, wm_disp_shape, wm_disp_dtype,
        wm_d_par_path, wm_d_par_shape, wm_d_par_dtype,
        wm_d_perp_path, wm_d_perp_shape, wm_d_perp_dtype,
        gm_d_par_path, gm_d_par_shape, gm_d_par_dtype,
        csf_d_par_path, csf_d_par_shape, csf_d_par_dtype,
        f_ins_path, f_ins_shape, f_ins_dtype,
    ) = memmap_info

    # Apply diffusivity ranges inside the worker so it also works with spawn
    set_diffusivity_ranges(**diffusivity_cfg)

    # Open memmaps in read-write mode
    signals_mm = np.memmap(signals_path, mode="r+", dtype=signals_dtype, shape=signals_shape)
    labels_mm = np.memmap(labels_path, mode="r+", dtype=labels_dtype, shape=labels_shape)
    num_fibers_mm = np.memmap(num_fibers_path, mode="r+", dtype=num_fibers_dtype, shape=num_fibers_shape)
    dispersion_mm = np.memmap(dispersion_path, mode="r+", dtype=dispersion_dtype, shape=dispersion_shape)
    wm_fraction_mm = np.memmap(wm_fraction_path, mode="r+", dtype=wm_fraction_dtype, shape=wm_fraction_shape)
    gm_fraction_mm = np.memmap(gm_fraction_path, mode="r+", dtype=gm_fraction_dtype, shape=gm_fraction_shape)
    csf_fraction_mm = np.memmap(csf_fraction_path, mode="r+", dtype=csf_fraction_dtype, shape=csf_fraction_shape)
    nd_mm = np.memmap(nd_path, mode="r+", dtype=nd_dtype, shape=nd_shape)
    odfs_mm = np.memmap(odfs_path, mode="r+", dtype=odfs_dtype, shape=odfs_shape)
    ufa_wm_mm = np.memmap(ufa_wm_path, mode="r+", dtype=ufa_wm_dtype, shape=ufa_wm_shape)
    ufa_voxel_mm = np.memmap(ufa_voxel_path, mode="r+", dtype=ufa_voxel_dtype, shape=ufa_voxel_shape)
    fraction_array_mm = np.memmap(fraction_array_path, mode="r+", dtype=fraction_array_dtype, shape=fraction_array_shape)
    # Additional DKI memmaps
    wm_disp_mm = np.memmap(wm_disp_path, mode="r+", dtype=wm_disp_dtype, shape=wm_disp_shape)
    wm_d_par_mm = np.memmap(wm_d_par_path, mode="r+", dtype=wm_d_par_dtype, shape=wm_d_par_shape)
    wm_d_perp_mm = np.memmap(wm_d_perp_path, mode="r+", dtype=wm_d_perp_dtype, shape=wm_d_perp_shape)
    gm_d_par_mm = np.memmap(gm_d_par_path, mode="r+", dtype=gm_d_par_dtype, shape=gm_d_par_shape)
    csf_d_par_mm = np.memmap(csf_d_par_path, mode="r+", dtype=csf_d_par_dtype, shape=csf_d_par_shape)
    f_ins_mm = np.memmap(f_ins_path, mode="r+", dtype=f_ins_dtype, shape=f_ins_shape)

    for i in range(batch_size):
        idx = start_idx + i
        res = create_mixed_signal(
            sphere, evecs, bingham, odi, bval, bvec, wm_thresh, tort
        )
        (
            signals_mm[idx],
            labels_mm[idx],
            num_fibers_mm[idx],
            dispersion_mm[idx],
            wm_fraction_mm[idx],
            gm_fraction_mm[idx],
            csf_fraction_mm[idx],
            nd_mm[idx],
            odfs_mm[idx],
            ufa_wm_mm[idx],
            ufa_voxel_mm[idx],
            fraction_array_mm[idx],
            wm_disp_mm[idx],
            wm_d_par_mm[idx],
            wm_d_perp_mm[idx],
            gm_d_par_mm[idx],
            csf_d_par_mm[idx],
            f_ins_mm[idx],
        ) = res

    # Flush memmaps
    signals_mm.flush()
    labels_mm.flush()
    num_fibers_mm.flush()
    dispersion_mm.flush()
    wm_fraction_mm.flush()
    gm_fraction_mm.flush()
    csf_fraction_mm.flush()
    nd_mm.flush()
    odfs_mm.flush()
    ufa_wm_mm.flush()
    ufa_voxel_mm.flush()
    fraction_array_mm.flush()
    wm_disp_mm.flush()
    wm_d_par_mm.flush()
    wm_d_perp_mm.flush()
    gm_d_par_mm.flush()
    csf_d_par_mm.flush()
    f_ins_mm.flush()

    return batch_size


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)

    run_dki = False
    run_msdki = False
    run_analytical_dki = False  # Use analytical DKI (multi_tensor_dki) for comparison

    num_simulations = 500_000
    dtype_config = np.float32
    label_dtype = np.uint8
    num_cpus = 24

    BATCH_SIZE = 1000
    DTI_BATCH_SIZE = 2000  # batch size for DTI fitting

    # Diffusivity ranges (mm^2/s) — adjust as needed
    diffusivity_cfg = {
        "wm_d_par_range": (2.0e-3, 3.0e-3),
        "wm_d_perp_range": (0.3e-3, 1.5e-3),
        "gm_d_iso_range": 1.0e-3,
        "csf_d": 3.0e-3,
    }

    print_mem_usage("start of script")

    # -------------------------------------------------------------------------
    # Acquisition and model setup
    # -------------------------------------------------------------------------
    sphere = default_sphere
    target_sphere = np.ascontiguousarray(sphere.vertices, dtype=np.float64)

    # Load bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(
        "/home/athshah/Phi/165840/bvals", "/home/athshah/Phi/165840/bvecs"
    )
    bvals = np.ascontiguousarray(bvals.astype(np.float64))
    bvecs = np.ascontiguousarray(bvecs.astype(np.float64))

    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    # Pre compute eigenvectors and Bingham dictionary
    evecs = np.array(
        [all_tensor_evecs(tuple(point)) for point in target_sphere],
        dtype=np.float64,
    )
    odi_list = np.linspace(0.01, 0.3, 10).astype(np.float64)
    bingham_sf = bingham_dictionary(target_sphere, odi_list)

    wm_threshold = 0.5
    tortuisity = False

    n_dirs = target_sphere.shape[0]
    n_bvals = bvals.shape[0]

    # -------------------------------------------------------------------------
    # Create memmaps for large arrays
    # -------------------------------------------------------------------------
    memmap_paths = {}

    def create_memmap(name, dtype, shape):
        path = os.path.join(output_dir, f"{name}.mmap")
        memmap_paths[name] = path
        return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

    signals_mm = create_memmap("signals", dtype_config, (num_simulations, n_bvals))
    labels_mm = create_memmap("labels", label_dtype, (num_simulations, n_dirs))
    num_fibers_mm = create_memmap("num_fibers", dtype_config, (num_simulations,))
    dispersion_mm = create_memmap("dispersion", dtype_config, (num_simulations,))
    wm_fraction_mm = create_memmap("wm_fraction", dtype_config, (num_simulations,))
    gm_fraction_mm = create_memmap("gm_fraction", dtype_config, (num_simulations,))
    csf_fraction_mm = create_memmap("csf_fraction", dtype_config, (num_simulations,))
    nd_mm = create_memmap("nd", dtype_config, (num_simulations,))
    odfs_mm = create_memmap("odfs", np.float16, (num_simulations, n_dirs))
    ufa_wm_mm = create_memmap("ufa_wm", dtype_config, (num_simulations,))
    ufa_voxel_mm = create_memmap("ufa_voxel", dtype_config, (num_simulations,))
    fraction_array_mm = create_memmap("fraction_array", dtype_config, (num_simulations, 3))
    # Additional memmaps for analytical DKI
    wm_disp_mm = create_memmap("wm_disp", dtype_config, (num_simulations,))
    wm_d_par_mm = create_memmap("wm_d_par", dtype_config, (num_simulations,))
    wm_d_perp_mm = create_memmap("wm_d_perp", dtype_config, (num_simulations,))
    gm_d_par_mm = create_memmap("gm_d_par", dtype_config, (num_simulations,))
    csf_d_par_mm = create_memmap("csf_d_par", dtype_config, (num_simulations,))
    f_ins_mm = create_memmap("f_ins", dtype_config, (num_simulations, 3))

    memmaps = [
        signals_mm,
        labels_mm,
        num_fibers_mm,
        dispersion_mm,
        wm_fraction_mm,
        gm_fraction_mm,
        csf_fraction_mm,
        nd_mm,
        odfs_mm,
        ufa_wm_mm,
        ufa_voxel_mm,
        fraction_array_mm,
        wm_disp_mm,
        wm_d_par_mm,
        wm_d_perp_mm,
        gm_d_par_mm,
        csf_d_par_mm,
        f_ins_mm,
    ]

    print_mem_usage("after memmap allocation")

    # Apply diffusivity ranges for the parent process
    set_diffusivity_ranges(**diffusivity_cfg)

    # -------------------------------------------------------------------------
    # Run simulations in batches into memmaps using OpenMP-style threading
    # -------------------------------------------------------------------------
    num_batches_full = num_simulations // BATCH_SIZE
    remainder = num_simulations % BATCH_SIZE
    total_batches = num_batches_full + (1 if remainder > 0 else 0)

    # Build list of (start_idx, batch_size) for all batches
    batch_specs = []
    current_start = 0
    for batch_idx in range(total_batches):
        bs = BATCH_SIZE if batch_idx < num_batches_full else remainder
        batch_specs.append((current_start, bs))
        current_start += bs

    print(f"Running {total_batches} batches with {num_cpus} processes (multiprocessing)")

    # Pack memmap info for workers to reopen
    memmap_info = (
        memmap_paths["signals"], (num_simulations, n_bvals), dtype_config,
        memmap_paths["labels"], (num_simulations, n_dirs), label_dtype,
        memmap_paths["num_fibers"], (num_simulations,), dtype_config,
        memmap_paths["dispersion"], (num_simulations,), dtype_config,
        memmap_paths["wm_fraction"], (num_simulations,), dtype_config,
        memmap_paths["gm_fraction"], (num_simulations,), dtype_config,
        memmap_paths["csf_fraction"], (num_simulations,), dtype_config,
        memmap_paths["nd"], (num_simulations,), dtype_config,
        memmap_paths["odfs"], (num_simulations, n_dirs), np.float16,
        memmap_paths["ufa_wm"], (num_simulations,), dtype_config,
        memmap_paths["ufa_voxel"], (num_simulations,), dtype_config,
        memmap_paths["fraction_array"], (num_simulations, 3), dtype_config,
        # Additional DKI parameters
        memmap_paths["wm_disp"], (num_simulations,), dtype_config,
        memmap_paths["wm_d_par"], (num_simulations,), dtype_config,
        memmap_paths["wm_d_perp"], (num_simulations,), dtype_config,
        memmap_paths["gm_d_par"], (num_simulations,), dtype_config,
        memmap_paths["csf_d_par"], (num_simulations,), dtype_config,
        memmap_paths["f_ins"], (num_simulations, 3), dtype_config,
    )

    
    with tqdm(total=num_simulations, desc="Simulating (multiprocessing)") as pbar:
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit all batches
            futures = {
                executor.submit(
                    generate_batch_to_memmap,
                    start_idx,
                    bs,
                    target_sphere,
                    evecs,
                    bingham_sf,
                    odi_list,
                    bvals,
                    bvecs,
                    wm_threshold,
                    tortuisity,
                    memmap_info,
                    diffusivity_cfg,
                ): (start_idx, bs)
                for start_idx, bs in batch_specs
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                batch_size_done = future.result()
                pbar.update(batch_size_done)

    # Flush memmaps to disk
    for mm in memmaps:
        mm.flush()

    print_mem_usage("after simulations")

    # -------------------------------------------------------------------------
    # Compute DTI metrics from lowest non zero shell
    # -------------------------------------------------------------------------
    min_b, shell_mask = smallest_shell_bval(
        bvals, b0_threshold=50, shell_tolerance=50
    )
    print("Smallest non zero shell b value:", min_b)

    b0_mask = bvals <= 50
    use_mask = shell_mask | b0_mask

    bvals_small = bvals[use_mask]
    bvecs_small = bvecs[use_mask]
    gtab_small = gradient_table(bvals_small, bvecs_small)

    dti_model = dti.TensorModel(gtab_small)

    fa_dti = np.empty(num_simulations, dtype=dtype_config)
    md_dti = np.empty(num_simulations, dtype=dtype_config)
    rd_dti = np.empty(num_simulations, dtype=dtype_config)

    with tqdm(total=num_simulations, desc="DTI fitting") as pbar:
        for start in range(0, num_simulations, DTI_BATCH_SIZE):
            end = min(start + DTI_BATCH_SIZE, num_simulations)
            data_batch = signals_mm[start:end][:, use_mask]
            dti_fit_batch = dti_model.fit(data_batch)

            fa_dti[start:end] = dti_fit_batch.fa.astype(dtype_config)
            md_dti[start:end] = dti_fit_batch.md.astype(dtype_config)
            rd_dti[start:end] = dti_fit_batch.rd.astype(dtype_config)

            pbar.update(end - start)

    # -------------------------------------------------------------------------
    # Optional DKI and msdki
    # -------------------------------------------------------------------------
    if run_dki or run_msdki:
        mask_dki = bvals <= 2500
        bvals_dki = bvals[mask_dki]
        bvecs_dki = bvecs[mask_dki]
        gtab_dki = gradient_table(bvals_dki, bvecs_dki)
        signals_dki = signals_mm[:, mask_dki]

    if run_dki:
        print("Running DKI. This can be slow for many simulations.")
        dki_model = DiffusionKurtosisModel(gtab_dki)
        dki_fit = dki_model.multi_fit(signals_dki)[0]

        ak_arr = dki_fit.ak().astype(dtype_config)
        rk_arr = dki_fit.rk().astype(dtype_config)
        mk_arr = dki_fit.mk().astype(dtype_config)
        kfa_arr = dki_fit.kfa.astype(dtype_config)
    else:
        ak_arr = np.zeros(num_simulations, dtype=dtype_config)
        rk_arr = np.zeros(num_simulations, dtype=dtype_config)
        mk_arr = np.zeros(num_simulations, dtype=dtype_config)
        kfa_arr = np.zeros(num_simulations, dtype=dtype_config)

    if run_msdki:
        print("Running msdki uFA. This can be slow for many simulations.")
        msdki_model = msdki.MeanDiffusionKurtosisModel(gtab_dki)
        msdki_fit = msdki_model.fit(data=signals_dki, mask=None)
        ufa_smt2 = msdki_fit.smt2uFA.astype(dtype_config)
    else:
        ufa_smt2 = np.zeros(num_simulations, dtype=dtype_config)

    # -------------------------------------------------------------------------
    # Analytical DKI computation
    # -------------------------------------------------------------------------
    
    if run_analytical_dki:
        print(f"Running analytical DKI computation with {num_cpus} processes...")
        
        # Create output memmaps for analytical DKI
        ak_analytical_mm = create_memmap("ak_analytical", dtype_config, (num_simulations,))
        rk_analytical_mm = create_memmap("rk_analytical", dtype_config, (num_simulations,))
        mk_analytical_mm = create_memmap("mk_analytical", dtype_config, (num_simulations,))
        kfa_analytical_mm = create_memmap("kfa_analytical", dtype_config, (num_simulations,))
        
        # Build batch specs for DKI computation
        DKI_BATCH_SIZE = 500
        num_dki_batches_full = num_simulations // DKI_BATCH_SIZE
        dki_remainder = num_simulations % DKI_BATCH_SIZE
        total_dki_batches = num_dki_batches_full + (1 if dki_remainder > 0 else 0)
        
        dki_batch_specs = []
        current_start = 0
        for batch_idx in range(total_dki_batches):
            bs = DKI_BATCH_SIZE if batch_idx < num_dki_batches_full else dki_remainder
            dki_batch_specs.append((current_start, bs))
            current_start += bs
        
        # Pack input memmap info
        input_memmap_info = (
            memmap_paths["num_fibers"], (num_simulations,), dtype_config,
            memmap_paths["labels"], (num_simulations, n_dirs), label_dtype,
            memmap_paths["wm_disp"], (num_simulations,), dtype_config,
            memmap_paths["wm_d_par"], (num_simulations,), dtype_config,
            memmap_paths["wm_d_perp"], (num_simulations,), dtype_config,
            memmap_paths["gm_d_par"], (num_simulations,), dtype_config,
            memmap_paths["csf_d_par"], (num_simulations,), dtype_config,
            memmap_paths["wm_fraction"], (num_simulations,), dtype_config,
            memmap_paths["gm_fraction"], (num_simulations,), dtype_config,
            memmap_paths["csf_fraction"], (num_simulations,), dtype_config,
            memmap_paths["fraction_array"], (num_simulations, 3), dtype_config,
            memmap_paths["f_ins"], (num_simulations, 3), dtype_config,
        )
        
        # Pack output memmap info
        output_memmap_info = (
            memmap_paths["ak_analytical"], (num_simulations,), dtype_config,
            memmap_paths["rk_analytical"], (num_simulations,), dtype_config,
            memmap_paths["mk_analytical"], (num_simulations,), dtype_config,
            memmap_paths["kfa_analytical"], (num_simulations,), dtype_config,
        )
        
        with tqdm(total=num_simulations, desc="Analytical DKI (parallel)") as pbar:
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                futures = {
                    executor.submit(
                        compute_analytical_dki_batch,
                        start_idx,
                        bs,
                        target_sphere,
                        bingham_sf,
                        odi_list,
                        bvals,
                        bvecs,
                        input_memmap_info,
                        output_memmap_info,
                    ): (start_idx, bs)
                    for start_idx, bs in dki_batch_specs
                }
                
                for future in as_completed(futures):
                    batch_size_done = future.result()
                    pbar.update(batch_size_done)
        
        # Use analytical values
        ak_arr = np.array(ak_analytical_mm)
        rk_arr = np.array(rk_analytical_mm)
        mk_arr = np.array(mk_analytical_mm)
        kfa_arr = np.array(kfa_analytical_mm)
        
        # Add to memmaps list for cleanup
        memmaps.extend([ak_analytical_mm, rk_analytical_mm, mk_analytical_mm, kfa_analytical_mm])

    print_mem_usage("after metrics (DTI / DKI / msdki)")

    # -------------------------------------------------------------------------
    # Save final npz, then delete memmap temp files
    # -------------------------------------------------------------------------
    final_npz_path = os.path.join(output_dir, "simulated_data.npz")
    print("Saving npz to:", final_npz_path)

    np.savez_compressed(
        final_npz_path,
        signals=signals_mm,
        labels=labels_mm,
        num_fibers=num_fibers_mm,
        dispersion=dispersion_mm,
        wm_fraction=wm_fraction_mm,
        gm_fraction=gm_fraction_mm,
        csf_fraction=csf_fraction_mm,
        nd=nd_mm,
        odfs=odfs_mm,
        fa=fa_dti,
        md=md_dti,
        rd=rd_dti,
        ufa_wm=ufa_wm_mm,
        ufa_voxel=ufa_voxel_mm,
        ak=ak_arr,
        rk=rk_arr,
        mk=mk_arr,
        kfa=kfa_arr,
        ufa_smt2=ufa_smt2,
        fraction_array=fraction_array_mm,
    )

    print("Saved", final_npz_path)
    print_mem_usage("after saving npz before cleanup")

    # Clean up memmaps and backing files
    for mm in memmaps:
        del mm
    gc.collect()

    for name, path in memmap_paths.items():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"Warning: could not remove memmap file {path}: {e}")

    gc.collect()
    print_mem_usage("after cleanup and GC")


if __name__ == "__main__":
    main()

