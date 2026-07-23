"""True-MC substrates for fiber CROSSING and BENDING (no kernel composition).

CROSSING: two interpenetrating cylinder populations as interleaved orthogonal
layers (x-cylinders and y-cylinders stacked alternately along z). Periodic in
all axes, non-overlapping by construction. Walkers diffuse through the combined
geometry, so extra-axonal water genuinely sees both populations — the physics a
volume-weighted kernel sum cannot capture.

BENDING: undulating (sinusoidal) cylinders, y = y0 + A*sin(2*pi*x/wavelength).
The local fiber orientation rotates continuously along x by +-atan(A*k); the
voxel-average orientation spread comes from real curvature, not a distribution
of straight segments. Periodic in x (wavelength divides L).

All lengths in metres (disimpy SI).
"""
import numpy as np


def _straight_tube(center, axis, L, r, n_theta, voff):
    """Open tube spanning [0,L] along `axis` (0/1/2) through `center` in the
    perpendicular plane. Periodic in `axis`. Returns (verts, faces)."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    perp = [a for a in (0, 1, 2) if a != axis]
    rings = []
    for end in (0.0, L):
        ring = np.zeros((n_theta, 3))
        ring[:, axis] = end
        ring[:, perp[0]] = center[perp[0]] + r * np.cos(th)
        ring[:, perp[1]] = center[perp[1]] + r * np.sin(th)
        rings.append(ring)
    verts = np.concatenate(rings, 0)
    faces = []
    for k in range(n_theta):
        kn = (k + 1) % n_theta
        a = voff + k; b = voff + kn
        c = voff + n_theta + kn; d = voff + n_theta + k
        faces.append([a, b, c]); faces.append([a, c, d])
    return verts, np.asarray(faces, np.int64)


def build_crossing(radius, L, layer_t, spacing, n_theta=20):
    """Interleaved orthogonal cylinder layers in a periodic box [0,L]^3.

    Even z-layers: cylinders along x; odd z-layers: along y. Returns dict.
    """
    if radius >= 0.5 * min(layer_t, spacing):
        raise ValueError("radius too large for layer_t/spacing (overlap)")
    nz = int(round(L / layer_t))
    npl = int(round(L / spacing))                      # cylinders per layer
    V, F, voff = [], [], 0
    n_x = n_y = 0
    for k in range(nz):
        zc = (k + 0.5) * layer_t
        axis = 0 if k % 2 == 0 else 1                  # x or y cylinders
        for j in range(npl):
            pc = (j + 0.5) * spacing
            center = np.zeros(3); center[2] = zc
            center[1 if axis == 0 else 0] = pc          # perp in-plane coord
            v, f = _straight_tube(center, axis, L, radius, n_theta, voff)
            V.append(v); F.append(f); voff += v.shape[0]
            if axis == 0:
                n_x += 1
            else:
                n_y += 1
    vertices = np.concatenate(V, 0)
    faces = np.concatenate(F, 0)
    vol = (n_x + n_y) * np.pi * radius ** 2 * L
    icvf = vol / L ** 3
    padding = np.array([0.0, 0.0, 0.0])
    return {"vertices": vertices, "faces": faces, "padding": padding,
            "L": L, "icvf": icvf, "n_x": n_x, "n_y": n_y,
            "frac_x": n_x / (n_x + n_y)}


def _sinusoidal_tube(y0, z0, A, wavelength, L, r, n_theta, n_seg, voff):
    """Tube whose centerline is y = y0 + A sin(2 pi x / wavelength), z=z0,
    swept with circular cross-section of radius r perpendicular to the tangent."""
    k = 2 * np.pi / wavelength
    xs = np.linspace(0.0, L, n_seg + 1)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rings = []
    for x in xs:
        c = np.array([x, y0 + A * np.sin(k * x), z0])
        t = np.array([1.0, A * k * np.cos(k * x), 0.0]); t /= np.linalg.norm(t)
        # orthonormal frame perpendicular to tangent
        ref = np.array([0.0, 0.0, 1.0])
        n1 = np.cross(t, ref); n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1)
        ring = c[None, :] + r * (np.cos(th)[:, None] * n1[None, :]
                                 + np.sin(th)[:, None] * n2[None, :])
        rings.append(ring)
    verts = np.concatenate(rings, 0)
    faces = []
    for s in range(n_seg):
        base = voff + s * n_theta
        for kk in range(n_theta):
            kn = (kk + 1) % n_theta
            a = base + kk; b = base + kn
            c = base + n_theta + kn; d = base + n_theta + kk
            faces.append([a, b, c]); faces.append([a, c, d])
    return verts, np.asarray(faces, np.int64)


def build_undulating(radius, L, amplitude, wavelength, spacing_y, spacing_z,
                     n_theta=20, n_seg=40):
    """Packed in-phase undulating cylinders along x in a periodic box [0,L]^3.

    amplitude=0 -> straight bundle. Local orientation spread ~ atan(A*2pi/lambda).
    """
    ny = int(round(L / spacing_y)); nz = int(round(L / spacing_z))
    V, F, voff, ncyl = [], [], 0, 0
    for iy in range(ny):
        for iz in range(nz):
            y0 = (iy + 0.5) * spacing_y
            z0 = (iz + 0.5) * spacing_z
            v, f = _sinusoidal_tube(y0, z0, amplitude, wavelength, L, radius,
                                    n_theta, n_seg, voff)
            V.append(v); F.append(f); voff += v.shape[0]; ncyl += 1
    vertices = np.concatenate(V, 0)
    faces = np.concatenate(F, 0)
    # arc length factor for ICVF (mean |d centerline/dx|)
    k = 2 * np.pi / wavelength
    xs = np.linspace(0, L, 200)
    arc = np.mean(np.sqrt(1 + (amplitude * k * np.cos(k * xs)) ** 2))
    vol = ncyl * np.pi * radius ** 2 * L * arc
    icvf = vol / L ** 3
    max_angle = np.degrees(np.arctan(amplitude * k))
    return {"vertices": vertices, "faces": faces, "padding": np.zeros(3),
            "L": L, "icvf": icvf, "n_cyl": ncyl, "max_tilt_deg": max_angle}


def _oriented_tube(center, d, length, r, n_theta, n_seg, voff):
    """Straight tube of given length along unit direction d through center."""
    d = d / np.linalg.norm(d)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n1 = np.cross(d, ref); n1 /= np.linalg.norm(n1)
    n2 = np.cross(d, n1)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ts = np.linspace(-length / 2, length / 2, n_seg + 1)
    rings = []
    for t in ts:
        c = center + t * d
        rings.append(c[None, :] + r * (np.cos(th)[:, None] * n1[None, :]
                                       + np.sin(th)[:, None] * n2[None, :]))
    verts = np.concatenate(rings, 0)
    faces = []
    for s in range(n_seg):
        base = voff + s * n_theta
        for k in range(n_theta):
            kn = (k + 1) % n_theta
            faces.append([base + k, base + kn, base + n_theta + kn])
            faces.append([base + k, base + n_theta + kn, base + n_theta + k])
    return verts, np.asarray(faces, np.int64)


def _watson_x(kappa, n, rng):
    """n unit vectors ~ Watson(mu=+x, kappa)."""
    out = np.zeros((n, 3)); i = 0
    while i < n:
        t = rng.uniform(-1, 1, 4 * (n - i) + 8)
        u = rng.uniform(0, 1, t.size)
        keep = u < np.exp(kappa * (t * t - 1)); tk = t[keep][:n - i]
        ph = rng.uniform(0, 2 * np.pi, tk.size); s = np.sqrt(np.maximum(0, 1 - tk ** 2))
        out[i:i + tk.size, 0] = tk                      # mu = +x
        out[i:i + tk.size, 1] = s * np.cos(ph)
        out[i:i + tk.size, 2] = s * np.sin(ph)
        i += tk.size
    return out


def build_fanning(odi, radius, L, n_target, n_theta=16, n_seg=24, seed=0, margin=1.1):
    """Non-periodic box of Watson(mu=x, ODI)-oriented straight cylinders
    (a true unimodal FAN), placed non-overlapping by rejection. Returns mesh +
    realized ICVF and the actual orientation dispersion."""
    rng = np.random.default_rng(seed)
    kappa = 1.0 / np.tan(np.pi * odi / 2.0)
    dirs_pool = _watson_x(kappa, n_target * 6, rng)
    centers, dirs = [], []
    V, F, voff = [], [], 0
    length = 1.8 * L
    pidx = 0
    attempts = 0
    while len(centers) < n_target and attempts < n_target * 200:
        attempts += 1
        d = dirs_pool[pidx % len(dirs_pool)]; pidx += 1
        c = rng.uniform(0.15 * L, 0.85 * L, 3)
        ok = True
        for c2, d2 in zip(centers, dirs):
            cr = np.cross(d, d2); n = np.linalg.norm(cr)
            if n < 1e-6:
                dist = np.linalg.norm(np.cross(c - c2, d))
            else:
                dist = abs(np.dot(c - c2, cr)) / n
            if dist < 2 * radius * margin:
                ok = False; break
        if not ok:
            continue
        v, f = _oriented_tube(c, d, length, radius, n_theta, n_seg, voff)
        V.append(v); F.append(f); voff += v.shape[0]
        centers.append(c); dirs.append(d)
    vertices = np.concatenate(V, 0); faces = np.concatenate(F, 0)
    dirs = np.array(dirs)
    # realized ICVF (cylinder vol clipped to box ~ pi r^2 * L per cylinder)
    icvf = len(centers) * np.pi * radius ** 2 * L / L ** 3
    # actual orientation dispersion as ODI from the mean-direction concentration
    mean_dir = np.array([1.0, 0, 0])
    cos2 = (dirs @ mean_dir) ** 2
    return {"vertices": vertices, "faces": faces, "padding": np.zeros(3),
            "L": L, "icvf": icvf, "n_cyl": len(centers),
            "mean_cos2": float(cos2.mean()),
            "disp_deg": float(np.degrees(np.std(np.arccos(np.abs(dirs @ mean_dir)))))}
