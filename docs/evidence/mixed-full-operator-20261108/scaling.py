"""Post-hoc diagonal scaling; no new acceptance threshold is applied."""
import json,pathlib,numpy as np
p=pathlib.Path(__file__).resolve().parent
A=np.array(json.loads((p/"analysis.json").read_text())["matrix"])
assert A.shape==(9,9) and np.isfinite(A).all() and np.all(np.diag(A)>0)
d=np.sqrt(np.diag(A));S=A/np.outer(d,d)
N=S[np.ix_([0,3,6],[0,3,6])]
def eig(x):return np.linalg.eigvalsh((x+x.T)*.5)
e=eig(S);n=eig(N)
assert e[0]>0 and n[0]>0
m={"scaled_full_eigenvalues":e.tolist(),"scaled_full_condition":float(e[-1]/e[0]),"scaled_normal_eigenvalues":n.tolist(),"scaled_normal_condition":float(n[-1]/n[0]),"scaled_rod_normal_cross":float(N[1,2]),"scope":"post-hoc scaling of one captured local linear operator, not proof of nonlinear solver mechanism"}
(p/"scaling.json").write_text(json.dumps(m,indent=2)+"\n")
print(json.dumps(m,indent=2))
