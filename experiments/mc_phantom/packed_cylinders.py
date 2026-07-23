"""Generate a periodic packed-parallel-cylinder triangular mesh for disimpy.

The substrate is a square grid of impermeable cylinders parallel to +z,
fully inside a box [0,L] x [0,L] x [0,H], so the mesh is periodically
valid (no cylinder crosses a boundary). With disimpy `init_pos='uniform'`
and `periodic=True`, walkers placed inside cylinders undergo *restricted*
diffusion, those outside undergo *hindered* (tortuous) diffusion, and the
intra-axonal volume fraction equals the geometric area fraction exactly.

All lengths in metres (disimpy SI units). Typical axon radius ~1e-6 m.
"""
import numpy as np


def square_packing(icvf, radius, n_side, height_factor=2.0):
    """Square grid of n_side x n_side cylinders hitting a target ICVF.

    icvf      target intra-axonal volume (area) fraction in (0, 0.78)
    radius    cylinder radius [m]
    n_side    cylinders per axis (n_side**2 total)
    Returns (centers (M,2), L, H, realized_icvf).
    """
    if icvf >= np.pi / 4:
        raise ValueError(f"square packing ICVF must be < pi/4≈0.785, got {icvf}")
    spacing = radius * np.sqrt(np.pi / icvf)          # one cylinder per spacing^2 cell
    if radius >= 0.5 * spacing:
        raise ValueError("radius too large for spacing (cylinders would touch/cross)")
    L = spacing * n_side
    H = height_factor * spacing
    i = (np.arange(n_side) + 0.5) * spacing
    cx, cy = np.meshgrid(i, i)
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    realized = centers.shape[0] * np.pi * radius ** 2 / (L * L)
    return centers, L, H, realized


def cylinder_wall_mesh(centers, radius, H, n_theta=24):
    """Open-tube side-wall mesh for all cylinders (periodic in z).

    Returns (vertices (V,3), faces (F,3) int).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    verts = []
    faces = []
    voff = 0
    for (cx, cy) in centers:
        ring0 = np.column_stack([cx + radius * ct, cy + radius * st,
                                 np.zeros(n_theta)])
        ring1 = np.column_stack([cx + radius * ct, cy + radius * st,
                                 np.full(n_theta, H)])
        verts.append(ring0)
        verts.append(ring1)
        for k in range(n_theta):
            kn = (k + 1) % n_theta
            a = voff + k            # ring0[k]
            b = voff + kn           # ring0[k+1]
            c = voff + n_theta + kn  # ring1[k+1]
            d = voff + n_theta + k   # ring1[k]
            faces.append([a, b, c])
            faces.append([a, c, d])
        voff += 2 * n_theta
    vertices = np.concatenate(verts, axis=0).astype(np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


def build_substrate_mesh(icvf, radius, n_side=5, n_theta=24, height_factor=2.0):
    """Convenience: geometry + mesh. Returns dict with mesh + ground truth."""
    centers, L, H, realized = square_packing(icvf, radius, n_side, height_factor)
    vertices, faces = cylinder_wall_mesh(centers, radius, H, n_theta)
    # Pad in x,y so the disimpy voxel (= bbox + 2*padding) spans exactly [0,L].
    # bbox of triangles spans [min cx - r, max cx + r]; we want [0, L].
    bb_lo = vertices[:, :2].min(0)
    bb_hi = vertices[:, :2].max(0)
    pad_xy = bb_lo                      # lower margin (centers symmetric -> = upper)
    padding = np.array([pad_xy[0], pad_xy[1], 0.0])
    return {
        "vertices": vertices, "faces": faces, "padding": padding,
        "centers": centers, "radius": radius, "L": L, "H": H,
        "icvf": realized, "n_cyl": centers.shape[0],
    }
