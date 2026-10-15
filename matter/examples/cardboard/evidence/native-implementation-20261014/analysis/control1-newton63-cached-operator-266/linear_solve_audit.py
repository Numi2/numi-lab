from pathlib import Path
import numpy as np,json,hashlib
from scipy import sparse
from scipy.sparse.linalg import spsolve
p=Path(__file__).parent;z=np.load(p/'report.vectors.npz');a=sparse.load_npz(p/'report.operator.npz');b=z['native_rhs_free_impulse_Ns'].reshape(-1);x=z['direction'].reshape(-1);exact=spsolve(a,b);r=b-a@x;re=b-a@exact
report={'schema':'numi.cardboard.captured-linear-solve-audit.v1','operator_sha256':hashlib.sha256((p/'report.operator.npz').read_bytes()).hexdigest(),'vectors_sha256':hashlib.sha256((p/'report.vectors.npz').read_bytes()).hexdigest(),'dofs':int(a.shape[0]),'rhs_l2':float(np.linalg.norm(b)),'native_correction_l2':float(np.linalg.norm(x)),'offline_direct_correction_l2':float(np.linalg.norm(exact)),'native_true_residual_l2':float(np.linalg.norm(r)),'native_true_residual_relative':float(np.linalg.norm(r)/np.linalg.norm(b)),'offline_direct_residual_relative':float(np.linalg.norm(re)/np.linalg.norm(b)),'direction_difference_l2_relative_to_direct':float(np.linalg.norm(exact-x)/np.linalg.norm(exact)),'direction_cosine':float(np.dot(x,exact)/(np.linalg.norm(x)*np.linalg.norm(exact))),'scope':'Offline diagnostic unsymmetrized sparse solve of captured native cached operator. Does not reproject material, change solver/physics/tolerance, or qualify a native solution.'}
np.savez_compressed(p/'offline-direct-correction.npz',native=x,direct=exact,rhs=b,native_residual=r,direct_residual=re)
(p/'linear-solve-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
