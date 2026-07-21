"""Ground-truth synthetic test sets for the angular-resolution / NUFO experiment.

Built on the DIPY FORCE simulation stack, so the test signals use the same
forward model as the library they are matched against. Only the sampling
differs: the two-fiber generator takes sphere vertex indices, so crossings
can be placed at a prescribed angle.

Produces two test sets: `angular` (two-fiber crossings binned by angle) and
`nufo` (equal numbers of one-, two- and three-fiber voxels).
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.data import default_sphere
from dipy.sims._multi_tensor_omp import multi_tensor
from dipy.sims.force import dispersion_lut
from dipy.sims.voxel import all_tensor_evecs

BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]

S0 = 100.0
GM_D = 1.0e-3
CSF_D = 3.0e-3


def build_gtab(n_dirs=75, bval=2000.0, seed=0):
    """Single-shell gradient table: `n_dirs` dispersed directions, mirrored, plus a b0."""
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.random(n_dirs)
    phi = 2 * np.pi * rng.random(n_dirs)
    hsph, _ = disperse_charges(HemiSphere(theta=theta, phi=phi), 5000)
    sph = Sphere(xyz=np.vstack((hsph.vertices, -hsph.vertices)))

    bvecs = np.zeros((len(sph.vertices) + 1, 3))
    bvecs[1:] = sph.vertices
    bvals = np.zeros(len(sph.vertices) + 1)
    bvals[1:] = bval
    return gradient_table(bvals, bvecs=bvecs)


def crossing_angles(vertices):
    """Antipodal-symmetric pairwise angles between sphere vertices, in degrees."""
    cos = np.clip(vertices @ vertices.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(np.abs(cos)))


def sample_pairs_in_bucket(vertices, lower_deg, upper_deg, n_pairs, rng):
    """Draw `n_pairs` distinct vertex pairs whose crossing angle is in the bucket."""
    ang = crossing_angles(vertices)
    iu = np.triu_indices(len(vertices), k=1)
    ok = (ang[iu] >= lower_deg) & (ang[iu] < upper_deg)
    cand = np.stack([iu[0][ok], iu[1][ok]], axis=1)
    if len(cand) == 0:
        raise ValueError(f"no vertex pair falls in bucket {lower_deg}-{upper_deg} deg")
    replace = len(cand) < n_pairs
    sel = rng.choice(len(cand), size=n_pairs, replace=replace)
    return cand[sel], ang[iu][ok][sel]


class VoxelSimulator:
    """Multi-compartment voxel simulator on the FORCE target sphere.

    Each fiber is a Bingham-dispersed stick plus zeppelin; GM and CSF are
    isotropic. Same forward model DIPY uses to build the FORCE library.
    """

    def __init__(self, gtab, sphere=None, odi_range=(0.01, 0.3), num_odi_values=10):
        sphere = default_sphere if sphere is None else sphere
        self.sphere = sphere
        self.target_sphere = np.ascontiguousarray(sphere.vertices, dtype=np.float64)
        self.n_dirs = len(self.target_sphere)
        self.bvals = np.ascontiguousarray(gtab.bvals, dtype=np.float64)
        self.bvecs = np.ascontiguousarray(gtab.bvecs, dtype=np.float64)
        self.evecs = np.ascontiguousarray(
            [all_tensor_evecs(tuple(v)) for v in self.target_sphere], dtype=np.float64
        )
        self.odi_list = np.linspace(odi_range[0], odi_range[1], num_odi_values)
        self.bingham = dispersion_lut(self.target_sphere, self.odi_list)
        self.gm_signal = np.exp(-self.bvals * GM_D) * S0
        self.csf_signal = np.exp(-self.bvals * CSF_D) * S0
        self._angles = None

    @property
    def angles(self):
        """Cached matrix of antipodal-symmetric angles between sphere vertices."""
        if self._angles is None:
            self._angles = crossing_angles(self.target_sphere)
        return self._angles

    def wm_signal(self, indices, fiber_fracs, f_in, d_par, d_perp, odi):
        """Signal of `len(indices)` dispersed fibers."""
        mevals_ex = np.zeros((self.n_dirs, 3))
        mevals_ex[:, 0] = d_par
        mevals_ex[:, 1] = d_perp
        mevals_ex[:, 2] = d_perp
        mevals_in = np.zeros((self.n_dirs, 3))
        mevals_in[:, 0] = d_par

        signal = np.zeros(len(self.bvals))
        labels = np.zeros(self.n_dirs, dtype=np.uint8)
        for k, idx in enumerate(indices):
            fodf = self.bingham[int(idx)][odi]
            fodf = np.ascontiguousarray(fodf / fodf.sum() * 100.0)
            s_in = multi_tensor(mevals_in, self.evecs, fodf, self.bvals, self.bvecs)
            s_ex = multi_tensor(mevals_ex, self.evecs, fodf, self.bvals, self.bvecs)
            signal += fiber_fracs[k] * (f_in[k] * s_in + (1 - f_in[k]) * s_ex)
            labels[int(idx)] = 1
        return signal * S0, labels

    def add_tissue(self, wm_signal, rng, dirichlet=(2.0, 1.0, 1.0), wm_min=0.8):
        """Mix a WM signal with isotropic GM/CSF, resampling if WM falls below `wm_min`."""
        fractions = rng.dirichlet(dirichlet)
        while fractions[0] < wm_min:
            fractions = rng.dirichlet(dirichlet)
        signal = (
            fractions[0] * wm_signal
            + fractions[1] * self.gm_signal
            + fractions[2] * self.csf_signal
        )
        return signal, fractions

    def two_fiber_at(self, pair, rng, d_perp_range=(0.3e-3, 1.5e-3)):
        """Two crossing fibers at the given pair of sphere vertex indices."""
        f_in = rng.uniform(0.6, 0.9, 2)
        frac1 = rng.uniform(0.2, 0.8)
        fiber_fracs = [frac1, 1 - frac1]
        d_par = rng.uniform(2.0e-3, 3.0e-3)
        d_perp = rng.uniform(*d_perp_range)
        odi = rng.choice(self.odi_list)
        return self.wm_signal(pair, fiber_fracs, f_in, d_par, d_perp, odi)

    def n_fiber(self, num_fibers, rng, d_perp_range=(0.3e-3, 0.8e-3), min_sep_deg=70.0):
        """One, two or three fibers at random orientations.

        `min_sep_deg` applies between every pair when `num_fibers == 3`.
        """
        d_par = rng.uniform(2.0e-3, 3.0e-3)
        d_perp = rng.uniform(*d_perp_range)
        odi = rng.choice(self.odi_list)
        f_in = rng.uniform(0.6, 0.9, num_fibers)

        if num_fibers == 1:
            indices = rng.integers(0, self.n_dirs, 1)
            fiber_fracs = [1.0]
        elif num_fibers == 2:
            indices = rng.integers(0, self.n_dirs, 2)
            while indices[0] == indices[1]:
                indices = rng.integers(0, self.n_dirs, 2)
            frac1 = rng.uniform(0.2, 0.8)
            fiber_fracs = [frac1, 1 - frac1]
        elif num_fibers == 3:
            ang = self.angles
            while True:
                indices = rng.integers(0, self.n_dirs, 3)
                sep = [
                    ang[indices[0], indices[1]],
                    ang[indices[0], indices[2]],
                    ang[indices[1], indices[2]],
                ]
                if min(sep) >= min_sep_deg:
                    break
            fiber_fracs = rng.dirichlet([1, 1, 1])
            while np.any(fiber_fracs < 0.2):
                fiber_fracs = rng.dirichlet([1, 1, 1])
        else:
            raise ValueError(f"num_fibers must be 1, 2 or 3, got {num_fibers}")

        return self.wm_signal(indices, fiber_fracs, f_in, d_par, d_perp, odi)


def make_angular_testset(sim, n_per_bucket, seed=0, wm_min=0.8, verbose=True):
    """Two-fiber crossings binned by crossing angle."""
    rng = np.random.default_rng(seed)
    signals, labels, bucket_index, true_angles = [], [], [], []

    for b, (lo, hi) in enumerate(BUCKETS):
        pairs, angles = sample_pairs_in_bucket(sim.target_sphere, lo, hi, n_per_bucket, rng)
        if verbose:
            print(f"  bucket {lo}-{hi} deg: {len(pairs)} voxels", flush=True)
        for pair, angle in zip(pairs, angles):
            wm, lab = sim.two_fiber_at(pair, rng)
            signal, _ = sim.add_tissue(wm, rng, dirichlet=(2.0, 0.01, 1.0), wm_min=wm_min)
            signals.append(signal)
            labels.append(lab)
            bucket_index.append(b)
            true_angles.append(angle)

    return {
        "signals": np.asarray(signals, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "num_fibers": np.full(len(signals), 2, dtype=np.int32),
        "bucket_index": np.asarray(bucket_index, dtype=np.int32),
        "true_angle": np.asarray(true_angles, dtype=np.float32),
    }


def make_nufo_testset(sim, n_per_class, seed=1, wm_min=0.8, verbose=True):
    """Equal numbers of one-, two- and three-fiber voxels."""
    rng = np.random.default_rng(seed)
    signals, labels, num_fibers = [], [], []

    for n_fib in (1, 2, 3):
        if verbose:
            print(f"  {n_fib}-fiber voxels: {n_per_class}", flush=True)
        for _ in range(n_per_class):
            wm, lab = sim.n_fiber(n_fib, rng)
            signal, _ = sim.add_tissue(wm, rng, dirichlet=(2.0, 1.0, 1.0), wm_min=wm_min)
            signals.append(signal)
            labels.append(lab)
            num_fibers.append(n_fib)

    return {
        "signals": np.asarray(signals, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "num_fibers": np.asarray(num_fibers, dtype=np.int32),
    }
