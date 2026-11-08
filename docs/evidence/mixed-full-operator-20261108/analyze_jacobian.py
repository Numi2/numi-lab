import json,pathlib
import numpy as np
p=pathlib.Path(__file__).resolve().parent
read=lambda n:json.loads((p/(n+".json")).read_text())
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ("row_trace","scratch")}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
base=read("baseline")
for name in ("jac0","jac1","rigid0","rigid1","rigid2"):
 a=read(name)
 assert clean(a)==clean(base) and a["repeat"] and a["capture_noninvasive"] and a["exact_trace"]
step=7
J=np.empty((9,6));reg=np.empty(9)
for contact in range(3):
 v=read("jac0")["steps"][step]["rows"][contact]["scratch"]+read("jac1")["steps"][step]["rows"][contact]["scratch"]
 J[3*contact:3*contact+3,:]=np.asarray(v[:18]).reshape(3,6)
 reg[3*contact:3*contact+3]=v[18:21]
assert np.isfinite(J).all() and np.isfinite(reg).all() and (reg>=0).all()
inv=np.diag([1.,1.,1.,10.,10.,10.]);reference=J@inv@J.T+np.diag(reg)
observed=np.zeros((9,9))
for axis in range(3):
 for target,row in enumerate(read("rigid"+str(axis))["steps"][step]["rows"]):
  s=row["scratch"];ids=[int(x) for x in s[9:12]]
  assert sorted(ids)==[0,1,2] and ids==s[9:12]
  for slot,src in enumerate(ids):observed[3*target+axis,3*src:3*src+3]=s[3*slot:3*slot+3]
assert np.isfinite(observed).all()
error=float(np.max(np.abs(reference-observed)));scale=float(np.max(np.abs(reference)));relative=error/scale
assert error<=1e-4 and relative<=1e-5,(error,relative)
negative=J@np.diag([.9,1,1,10,10,10])@J.T+np.diag(reg)
assert np.max(np.abs(negative-observed))>1e-4
result={"exposed_capture_equivalence":True,"jacobian":J.tolist(),"regularization":reg.tolist(),"analytic_mass_kg":1.,"analytic_inertia_kg_m2":.1,"reference_matrix":reference.tolist(),"captured_matrix":observed.tolist(),"max_absolute_difference":error,"relative_difference":relative,"gate_passed":True,"perturbed_mass_negative_rejected":True,"independent_geometric_jacobian_validated":False}
(p/"jac-analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if not k.endswith("matrix") and k!="jacobian"},indent=2))
