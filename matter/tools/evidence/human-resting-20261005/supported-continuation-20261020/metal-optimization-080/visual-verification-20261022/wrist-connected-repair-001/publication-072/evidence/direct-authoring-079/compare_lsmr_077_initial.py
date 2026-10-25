from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
import sys
D=Path("/Users/n/Work/numi-human-connected-surface-069")
M=D/"canonical/matter/tools"
sys.path.insert(0,str(M))
import human_skin_fat_connected_repair as repair
I=D/"repair-run-embedded-077/iteration-001"
z=np.load(I/"input.npz")
X=z["points_world_f32"];R=z["reference_points_m"];T=z["tetrahedra"];F=z["free_nodes"]
with np.load(I/"contact-rows.npz") as q:C=sparse.csr_matrix((q["data"],q["indices"],q["indptr"]),shape=q["shape"])
# npz layout could use standard scipy keys instead.
target=np.load(I/"contact-targets.npy") if (I/"contact-targets.npy").exists() else None
if target is None:
 import json
 prov=json.loads((I/"contact-provenance.json").read_text())
 raise RuntimeError("contact target file unavailable; use input archive metadata")
J,G,quality,Q=repair.cell_values_gradients(X,R,T)
active=np.flatnonzero(np.isin(T,F).any(axis=1))
rhs0=1000*(target-C@X.astype(float).ravel());hot=rhs0>1e-8
dcols=(3*F[:,None]+np.arange(3)).ravel()
DG=repair.sparse_displacement_gradient(R,T[active],F)
ridge=.002**.5
offset=1000*(X[F].astype(float)-z["construction_origin_f32"][F])
A=sparse.vstack([C[hot][:,dcols],ridge*sparse.eye(3*len(F)),2.*DG],format="csr")
b=np.r_[rhs0[hot],-ridge*offset,np.zeros(DG.shape[0])]
scale=1/np.sqrt(np.maximum(np.asarray(A.power(2).sum(axis=0)).ravel(),1e-30))
B=A@sparse.diags(scale)
t=time.monotonic();sol=lsmr(B,b,atol=1e-6,btol=1e-6,maxiter=400);elapsed=time.monotonic()-t
r=np.asarray(B@sol[0]-b);nr=np.asarray(B.T@r)
out={"source_report_sha256":hashlib.sha256((I/"construction-report.json").read_bytes()).hexdigest(),
 "input_npz_sha256":hashlib.sha256((I/"input.npz").read_bytes()).hexdigest(),
 "contact_npz_sha256":hashlib.sha256((I/"contact-rows.npz").read_bytes()).hexdigest(),
 "target_sha256":hashlib.sha256((I/"contact-targets.npy").read_bytes()).hexdigest() if (I/"contact-targets.npy").exists() else None,
 "method":"scipy_lsmr", "stop":int(sol[1]),"iterations":int(sol[2]),"reported_residual_norm":float(sol[3]),
 "reported_normal_residual_norm":float(sol[4]),"reported_normal_residual_relative_to_rhs":float(sol[4]/max(np.linalg.norm(B.T@b),1e-300)),
 "recomputed_normal_residual_norm":float(np.linalg.norm(nr)),"recomputed_normal_residual_relative_to_rhs":float(np.linalg.norm(nr)/max(np.linalg.norm(B.T@b),1e-300)),
 "normal_backward_error":float(np.linalg.norm(nr)/max(np.linalg.norm(B.data)*np.linalg.norm(r),1e-300)),
 "shape":list(B.shape),"nnz":int(B.nnz),"elapsed_s":elapsed}
P=Path("/Users/n/Work/numi-human-connected-surface-069/direct-authoring-079/lsmr-comparison-001.json")
P.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
