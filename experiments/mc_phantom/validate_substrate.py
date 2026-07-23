"""Validate the packed-cylinder substrate produces TRUE restricted/hindered
diffusion (not Gaussian): perpendicular ADC must drop with b and the signal
must sit above a free-Gaussian curve, while parallel ADC ~ D0.
"""
import argparse
import numpy as np
import np2_shim  # noqa: F401  (patches numpy>=2 aliases before disimpy import)
from disimpy import gradients, simulations, substrates
from packed_cylinders import build_substrate_mesh

D0 = 2.0e-9                                        # intrinsic diffusivity m^2/s


def run(icvf, radius_um, n_side, n_walkers, n_t, delta, DELTA):
    s = build_substrate_mesh(icvf, radius_um * 1e-6, n_side=n_side, n_theta=24)
    print(f"substrate: ICVF(geom)={s['icvf']:.3f}  r={radius_um}um  "
          f"n_cyl={s['n_cyl']}  L={s['L']*1e6:.1f}um  H={s['H']*1e6:.1f}um  "
          f"verts={s['vertices'].shape[0]} faces={s['faces'].shape[0]}")
    sub = substrates.mesh(s["vertices"], s["faces"], periodic=True,
                          padding=s["padding"], init_pos="uniform",
                          n_sv=np.array([20, 20, 20]), quiet=True)

    bvals_mm = np.array([0, 500, 1000, 2000, 3000, 5000, 8000], dtype=float)
    b_SI = bvals_mm * 1e6
    perp = np.tile([1.0, 0, 0], (len(b_SI), 1))    # x: perpendicular to cylinders
    par = np.tile([0, 0, 1.0], (len(b_SI), 1))     # z: parallel

    out = {}
    for name, bvecs in (("perp", perp), ("par", par)):
        grad, dt = gradients.pgse(delta, DELTA, n_t, b_SI, bvecs)
        sig = simulations.simulation(n_walkers, D0, grad, dt, sub,
                                     seed=42, quiet=True)
        out[name] = sig / sig[0]
    # free-diffusion reference on same scheme
    grad, dt = gradients.pgse(delta, DELTA, n_t, b_SI, perp)
    free = simulations.simulation(n_walkers, D0, grad, dt, substrates.free(),
                                  seed=42, quiet=True)
    out["free"] = free / free[0]

    print(f"\nDelta={DELTA*1e3:.0f}ms delta={delta*1e3:.0f}ms  "
          f"dt={dt*1e6:.2f}us  step_rms={np.sqrt(6*D0*dt)*1e6:.3f}um "
          f"(r={radius_um}um)")
    print(f"{'b(s/mm2)':>9}{'S_perp':>9}{'S_par':>9}{'S_free':>9}"
          f"{'ADCperp':>9}{'ADCpar':>9}")
    for i, b in enumerate(bvals_mm):
        adcp = -np.log(out['perp'][i]) / b_SI[i] * 1e9 if b > 0 else np.nan
        adcl = -np.log(out['par'][i]) / b_SI[i] * 1e9 if b > 0 else np.nan
        print(f"{b:>9.0f}{out['perp'][i]:>9.4f}{out['par'][i]:>9.4f}"
              f"{out['free'][i]:>9.4f}{adcp:>9.3f}{adcl:>9.3f}")
    # restriction signature: ADC_perp should fall with b and S_perp > S_free
    adc_perp = -np.log(out['perp'][1:]) / b_SI[1:] * 1e9
    print(f"\nADC_perp(b=500)={adc_perp[0]:.3f} -> ADC_perp(b=8000)="
          f"{adc_perp[-1]:.3f}  (restriction => decreasing)")
    print(f"S_perp > S_free at all b>0: {np.all(out['perp'][1:] > out['free'][1:])}")
    print(f"ADC_par(mean over b)={(-np.log(out['par'][1:])/b_SI[1:]*1e9).mean():.3f} "
          f"(expect ~{D0*1e9:.2f}=D0, hindered<D0 if tortuous)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--icvf", type=float, default=0.6)
    ap.add_argument("--radius_um", type=float, default=2.0)
    ap.add_argument("--n_side", type=int, default=6)
    ap.add_argument("--n_walkers", type=int, default=int(1e5))
    ap.add_argument("--n_t", type=int, default=4000)
    ap.add_argument("--delta", type=float, default=10e-3)
    ap.add_argument("--DELTA", type=float, default=30e-3)
    a = ap.parse_args()
    run(a.icvf, a.radius_um, a.n_side, a.n_walkers, a.n_t, a.delta, a.DELTA)
