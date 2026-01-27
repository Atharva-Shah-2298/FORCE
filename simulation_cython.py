import os
import gc
import numpy as np
import ray
from tqdm import tqdm
import psutil  # for memory usage

from dipy.data import default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import all_tensor_evecs
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
from sim_core import create_mixed_signal

# Limit BLAS threading so Ray parallelism is controllable
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


@ray.remote
def generate_batch_remote(
    batch_size, sphere, evecs, bingham, odi, bval, bvec, wm_thresh, tort
):
    """
    Ray worker function.
    Generates a batch of synthetic signals and packs them into arrays so the driver
    can write them directly into memmaps.
    """
    n_dirs = sphere.shape[0]
    n_bvals = bval.shape[0]

    signals_batch = np.empty((batch_size, n_bvals), dtype=np.float32)
    labels_batch = np.empty((batch_size, n_dirs), dtype=np.uint8)
    num_fibers_batch = np.empty(batch_size, dtype=np.float32)
    dispersion_batch = np.empty(batch_size, dtype=np.float32)
    wm_fraction_batch = np.empty(batch_size, dtype=np.float32)
    gm_fraction_batch = np.empty(batch_size, dtype=np.float32)
    csf_fraction_batch = np.empty(batch_size, dtype=np.float32)
    nd_batch = np.empty(batch_size, dtype=np.float32)
    odfs_batch = np.empty((batch_size, n_dirs), dtype=np.float16)
    ufa_wm_batch = np.empty(batch_size, dtype=np.float32)
    ufa_voxel_batch = np.empty(batch_size, dtype=np.float32)
    fraction_array_batch = np.empty((batch_size, 3), dtype=np.float32)

    for i in range(batch_size):
        res = create_mixed_signal(
            sphere, evecs, bingham, odi, bval, bvec, wm_thresh, tort
        )
        (
            signals_batch[i],
            labels_batch[i],
            num_fibers_batch[i],
            dispersion_batch[i],
            wm_fraction_batch[i],
            gm_fraction_batch[i],
            csf_fraction_batch[i],
            nd_batch[i],
            odfs_batch[i],
            ufa_wm_batch[i],
            ufa_voxel_batch[i],
            fraction_array_batch[i],
        ) = res

    return (
        signals_batch,
        labels_batch,
        num_fibers_batch,
        dispersion_batch,
        wm_fraction_batch,
        gm_fraction_batch,
        csf_fraction_batch,
        nd_batch,
        odfs_batch,
        ufa_wm_batch,
        ufa_voxel_batch,
        fraction_array_batch,
    )


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    output_dir = "/home/athshah/Phi/FORCE/simulated_data_cython"
    os.makedirs(output_dir, exist_ok=True)

    run_dki = False
    run_msdki = False

    num_simulations = 500_000
    dtype_config = np.float32
    label_dtype = np.uint8
    num_cpus = 24

    BATCH_SIZE = 1000
    DTI_BATCH_SIZE = 2000  # batch size for DTI fitting

    print_mem_usage("start of script")

    # -------------------------------------------------------------------------
    # Initialize Ray
    # -------------------------------------------------------------------------
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

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

    # Put constants into Ray object store
    sphere_ref = ray.put(target_sphere)
    evecs_ref = ray.put(evecs)
    bingham_sf_ref = ray.put(bingham_sf)
    odi_list_ref = ray.put(odi_list)
    bvals_ref = ray.put(bvals)
    bvecs_ref = ray.put(bvecs)
    wm_threshold_ref = ray.put(wm_threshold)
    tortuisity_ref = ray.put(tortuisity)

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
    ]

    print_mem_usage("after memmap allocation")

    # -------------------------------------------------------------------------
    # Run simulations in batches into memmaps
    # -------------------------------------------------------------------------
    num_batches_full = num_simulations // BATCH_SIZE
    remainder = num_simulations % BATCH_SIZE
    total_batches = num_batches_full + (1 if remainder > 0 else 0)

    max_in_flight = num_cpus * 2
    futures = []
    next_batch_index = 0
    next_start = 0

    def submit_batch(start_idx, batch_size):
        obj_ref = generate_batch_remote.remote(
            batch_size,
            sphere_ref,
            evecs_ref,
            bingham_sf_ref,
            odi_list_ref,
            bvals_ref,
            bvecs_ref,
            wm_threshold_ref,
            tortuisity_ref,
        )
        futures.append((start_idx, obj_ref))

    # Seed the first set of batches
    while next_batch_index < total_batches and len(futures) < max_in_flight:
        bs = BATCH_SIZE if next_batch_index < num_batches_full else remainder
        submit_batch(next_start, bs)
        next_start += bs
        next_batch_index += 1

    with tqdm(total=num_simulations, desc="Simulating (batched, memmap)") as pbar:
        while futures:
            obj_refs = [f[1] for f in futures]
            done, _ = ray.wait(obj_refs, num_returns=1)
            done_ref = done[0]

            # Find which batch finished
            idx = next(i for i, (_, ref) in enumerate(futures) if ref == done_ref)
            start_idx, _ = futures.pop(idx)

            (
                signals_b,
                labels_b,
                num_fibers_b,
                dispersion_b,
                wm_fraction_b,
                gm_fraction_b,
                csf_fraction_b,
                nd_b,
                odfs_b,
                ufa_wm_b,
                ufa_voxel_b,
                fraction_array_b,
            ) = ray.get(done_ref)

            batch_size = signals_b.shape[0]
            end_idx = start_idx + batch_size

            # Write to memmaps
            signals_mm[start_idx:end_idx] = signals_b
            labels_mm[start_idx:end_idx] = labels_b
            num_fibers_mm[start_idx:end_idx] = num_fibers_b
            dispersion_mm[start_idx:end_idx] = dispersion_b
            wm_fraction_mm[start_idx:end_idx] = wm_fraction_b
            gm_fraction_mm[start_idx:end_idx] = gm_fraction_b
            csf_fraction_mm[start_idx:end_idx] = csf_fraction_b
            nd_mm[start_idx:end_idx] = nd_b
            odfs_mm[start_idx:end_idx] = odfs_b
            ufa_wm_mm[start_idx:end_idx] = ufa_wm_b
            ufa_voxel_mm[start_idx:end_idx] = ufa_voxel_b
            fraction_array_mm[start_idx:end_idx] = fraction_array_b

            pbar.update(batch_size)

            # Submit next batch if any remain
            if next_batch_index < total_batches:
                bs = BATCH_SIZE if next_batch_index < num_batches_full else remainder
                submit_batch(next_start, bs)
                next_start += bs
                next_batch_index += 1

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

    ray.shutdown()


if __name__ == "__main__":
    main()
