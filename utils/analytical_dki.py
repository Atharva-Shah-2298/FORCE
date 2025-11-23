import numpy as np
I3 = np.eye(3)

def second_moment_from_odf(U, w):
    """
    U: (Nori, 3) unit vectors (your target_sphere)
    w: (Nori,) non-negative weights; if sum==0 returns zeros
    returns M = sum_u w(u) * u u^T   (3x3)
    """
    w = np.asarray(w, dtype=np.float64)
    if w.sum() <= 0:
        return np.zeros((3, 3), dtype=np.float64)
    w = w / w.sum()
    # M = U^T diag(w) U = U.T @ (w[:,None] * U)
    return U.T @ (w[:, None] * U)

def analytic_D_from_params(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                           f_gm=0.0, d_gm=0.0, f_csf=0.0, d_csf=0.0):
    """
    Eq. (24) specialized to axis-symmetric intra/extracellular + isotropic pools.
    Returns 3x3 diffusion tensor D (float64 for stability).
    """
    M = second_moment_from_odf(U, w_odf)  # orientation 2nd moment
    f_ex = max(0.0, 1.0 - f_in)
    D_wm = (f_wm * f_in) * (d_par * M) + (f_wm * f_ex) * (d_perp_ex * I3 + (d_par - d_perp_ex) * M)
    D_iso = (f_gm * d_gm + f_csf * d_csf) * I3
    return (D_wm + D_iso).astype(np.float64)

def kapp_analytic_for_dirs(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                           f_gm, d_gm, f_csf, d_csf, dirs):
    """
    Uses K(n) = 3 * Var_f[D_m(n)] / D(n)^2 derived from Eq. (25).
    U: (Nori,3); w_odf: (Nori,) (will be normalized inside)
    dirs: (Nd,3) measurement directions (assumed unit-norm)
    Returns Kapp for each dir: (Nd,)
    """
    w = np.asarray(w_odf, dtype=np.float64)
    if w.sum() > 0: w = w / w.sum()
    else:          w = w  # zero ODF -> no WM contribution

    dirs = np.asarray(dirs, dtype=np.float64)
    # cos^2 between every orientation and every measurement dir
    cos2 = (U @ dirs.T) ** 2  # (Nori, Nd)

    # Intra and extra directional diffusivities
    D_in = d_par * cos2                                   # (Nori, Nd)
    D_ex = d_perp_ex + (d_par - d_perp_ex) * cos2        # (Nori, Nd)

    f_ex = max(0.0, 1.0 - f_in)

    # Weighted means and second moments across *WM orientations*
    mu_wm  = f_wm * ( f_in * (w @ D_in)  + f_ex * (w @ D_ex) )         # (Nd,)
    mu2_wm = f_wm * ( f_in * (w @ (D_in**2)) + f_ex * (w @ (D_ex**2)) )# (Nd,)

    # Isotropic pools
    mu_iso  = f_gm * d_gm + f_csf * d_csf                              # scalar
    mu2_iso = f_gm * d_gm**2 + f_csf * d_csf**2                        # scalar

    # D(n) and E[D_m(n)^2]
    mu  = mu_wm  + mu_iso
    mu2 = mu2_wm + mu2_iso

    # K(n) = 3 * (mu2 - mu^2) / mu^2 ; guard very small mu
    eps = 1e-12
    K = 3.0 * np.maximum(mu2 - mu**2, 0.0) / np.maximum(mu**2, eps)
    return K

def mk_ak_rk_from_D_and_odf(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                            f_gm, d_gm, f_csf, d_csf, sphere_dirs):
    """
    Returns MK (spherical average of K), AK (along e1), RK (avg over plane ⟂ e1).
    """
    # D from Eq. 24 (for evecs)
    D = analytic_D_from_params(U, w_odf, f_wm, f_in, d_par, d_perp_ex, f_gm, d_gm, f_csf, d_csf)
    evals, evecs = np.linalg.eigh(D)
    idx = evals.argsort()[::-1]
    e1, e2, e3 = evecs[:, idx[0]], evecs[:, idx[1]], evecs[:, idx[2]]

    # MK: average over a (dense) sphere
    K_sphere = kapp_analytic_for_dirs(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                                      f_gm, d_gm, f_csf, d_csf, sphere_dirs)
    MK = float(np.mean(K_sphere))

    # AK: K along the principal diffusion direction
    AK = float(kapp_analytic_for_dirs(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                                      f_gm, d_gm, f_csf, d_csf, e1[None, :])[0])

    # RK: average K on the circle perpendicular to e1
    # sample 64 equally spaced directions in span{e2,e3}
    thetas = np.linspace(0, 2*np.pi, 64, endpoint=False)
    circle = np.array([np.cos(t)*e2 + np.sin(t)*e3 for t in thetas])
    circle /= np.linalg.norm(circle, axis=1, keepdims=True)
    RK = float(np.mean(kapp_analytic_for_dirs(U, w_odf, f_wm, f_in, d_par, d_perp_ex,
                                              f_gm, d_gm, f_csf, d_csf, circle)))
    return D, MK, AK, RK

def _principal_axes(D):
    evals, evecs = np.linalg.eigh(D)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]
