"""Packed-sphere (soma) substrate for a grey-matter compartment.

A cubic grid of impermeable spheres inside a periodic box. With disimpy
`init_pos='uniform'`, walkers inside spheres undergo restricted diffusion
(soma) and those outside undergo hindered diffusion — an isotropic,
sub-D0 signal that is NOT free water and NOT a stick. Soma volume fraction
= geometry. All lengths in metres.
"""
import numpy as np


def uv_sphere(center, radius, n_lat=12, n_lon=16):
    """Closed UV-sphere triangular mesh. Returns (verts (V,3), faces (F,3))."""
    cx, cy, cz = center
    verts = [[cx, cy, cz + radius]]                 # top pole (index 0)
    for i in range(1, n_lat):                       # interior latitude rings
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            verts.append([cx + radius * np.sin(th) * np.cos(ph),
                          cy + radius * np.sin(th) * np.sin(ph),
                          cz + radius * np.cos(th)])
    verts.append([cx, cy, cz - radius])             # bottom pole
    V = len(verts)
    bot = V - 1
    faces = []

    def ring(i):                                    # vertex index of ring i, col j
        return 1 + (i - 1) * n_lon

    for j in range(n_lon):                          # top fan
        faces.append([0, ring(1) + j, ring(1) + (j + 1) % n_lon])
    for i in range(1, n_lat - 1):                   # quad strips
        for j in range(n_lon):
            a = ring(i) + j
            b = ring(i) + (j + 1) % n_lon
            c = ring(i + 1) + (j + 1) % n_lon
            d = ring(i + 1) + j
            faces.append([a, b, c]); faces.append([a, c, d])
    for j in range(n_lon):                          # bottom fan
        faces.append([bot, ring(n_lat - 1) + (j + 1) % n_lon, ring(n_lat - 1) + j])
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def build_gm_substrate(f_soma, radius, n_side=4, n_lat=12, n_lon=16):
    """Cubic grid of spheres hitting a target soma volume fraction.

    Returns dict with mesh + ground truth (realized soma fraction).
    """
    if f_soma >= np.pi / 6:
        raise ValueError(f"cubic sphere packing f_soma < pi/6≈0.524, got {f_soma}")
    spacing = radius * (4 * np.pi / (3 * f_soma)) ** (1.0 / 3.0)
    if radius >= 0.5 * spacing:
        raise ValueError("radius too large: spheres would touch/cross")
    L = spacing * n_side
    c = (np.arange(n_side) + 0.5) * spacing
    cx, cy, cz = np.meshgrid(c, c, c, indexing="ij")
    centers = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])
    verts_all, faces_all, voff = [], [], 0
    for ctr in centers:
        v, f = uv_sphere(ctr, radius, n_lat, n_lon)
        verts_all.append(v); faces_all.append(f + voff); voff += v.shape[0]
    vertices = np.concatenate(verts_all, 0)
    faces = np.concatenate(faces_all, 0)
    realized = centers.shape[0] * (4.0 / 3.0) * np.pi * radius ** 3 / L ** 3
    padding = vertices.min(0).copy()                # voxel -> [0,L]^3
    return {"vertices": vertices, "faces": faces, "padding": padding,
            "centers": centers, "radius": radius, "L": L,
            "f_soma": realized, "n_sphere": centers.shape[0]}
