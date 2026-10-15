from pathlib import Path
import numpy as np,json
from scipy import sparse
from scipy.sparse.linalg import eigsh
p=Path(__file__).parent;a=sparse.load_npz(p/'report.operator.npz');sym=(a+a.T)*.5;v=np.load(p/'offline-direct-correction.npz');w,q=eigsh(sym,k=12,sigma=0,which='LM',tol=1e-10,v0=np.ones(a.shape[0]));order=np.argsort(np.abs(w));w=w[order];q=q[:,order];b=v['rhs'];x=v['direct'];native=v['native'];norm=lambda a:float(np.linalg.norm(a));rows=[]
for i in range(len(w)):
 rows.append({'mode':i,'symmetric_part_eigenvalue':float(w[i]),'symmetric_eigen_residual':norm(sym@q[:,i]-w[i]*q[:,i]),'rhs_projection':float(q[:,i]@b),'direct_correction_projection':float(q[:,i]@x),'native_correction_projection':float(q[:,i]@native)})
np.savez_compressed(p/'symmetric-part-modes.npz',eigenvalues=w,eigenvectors=q)
r={'schema':'numi.cardboard.offline-modal-conditioning.v1','rows':rows,'first12_direct_direction_fraction':norm(q.T@x)/norm(x),'limits':'Symmetric-part near-zero diagnostic modes only. Full unsymmetrized physical operator retained, no native preconditioner implementation or physical qualification.'};(p/'modal-report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
