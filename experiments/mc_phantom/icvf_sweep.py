import os, time, numpy as np
import np2_shim
from disimpy import gradients, simulations, substrates
from packed_cylinders import build_substrate_mesh
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel, softmax_stable
P="/home/athshah/Phi/165840"; D0=2.2e-9
bvals=np.loadtxt(P+"/bvals").ravel(); bvecs=np.loadtxt(P+"/bvecs")
if bvecs.shape[0]==3: bvecs=bvecs.T
sigs={}
for icvf in (0.60,0.70,0.75):
    s=build_substrate_mesh(icvf,1.0e-6,n_side=6,n_theta=24)
    sub=substrates.mesh(s["vertices"],s["faces"],periodic=True,padding=s["padding"],
                        init_pos="uniform",n_sv=np.array([20,20,20]),quiet=True)
    g,dt=gradients.pgse(10e-3,30e-3,8500,bvals*1e6,bvecs)
    t0=time.time(); sig=simulations.simulation(int(1e5),D0,g,dt,sub,seed=7,quiet=True)
    sigs[icvf]=(sig/sig[bvals<=50].mean()).astype(np.float64)
    print(f"ICVF(geom)={s['icvf']:.3f} {time.time()-t0:.0f}s")
gtab=gradient_table(bvals,bvecs=bvecs,b0_threshold=50)
m=FORCEModel(gtab,n_neighbors=50,use_posterior=True,posterior_beta=2000.,compute_odf=False,verbose=False)
m.generate(num_simulations=500000,use_cache=True,num_cpus=-1,compute_dti=True,compute_dki=False)
sims=m.simulations
Q=np.array([sigs[i] for i in (0.60,0.70,0.75)])
qn=np.linalg.norm(Q,axis=1,keepdims=True); D,nb=m._index.search(np.ascontiguousarray((Q/qn).astype(np.float32)),k=50)
W=softmax_stable(2000.*(D-m._penalty_array[nb]),axis=1)
out={f:np.einsum('nk,nk->n',W,sims[f][nb]) for f in ("wm_fraction","gm_fraction","csf_fraction","nd","dispersion")}
print(f"\n{'ICVF':>6}{'WM':>6}{'GM':>6}{'CSF':>6}{'iso':>6}{'ND':>6}  (real CC: WM0.89 GM0.06 CSF0.04 ND0.74)")
for i,icvf in enumerate((0.60,0.70,0.75)):
    print(f"{icvf:>6.2f}{out['wm_fraction'][i]:>6.2f}{out['gm_fraction'][i]:>6.2f}{out['csf_fraction'][i]:>6.2f}"
          f"{out['gm_fraction'][i]+out['csf_fraction'][i]:>6.2f}{out['nd'][i]:>6.2f}")
