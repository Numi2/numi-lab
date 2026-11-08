import hashlib,json,pathlib
import numpy as np
p=pathlib.Path(__file__).resolve().parent
names=("baseline","axis0","axis1","axis2")
data={n:json.loads((p/(n+".json")).read_text()) for n in names}
old=json.loads((p.parent/"mixed-failing-row-witness-20261108/baseline.json").read_text())
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ("row_trace","scratch")}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
base=data["baseline"]
assert exposed(base)==exposed(old),"retained trajectory changed"
assert base["success"]==7 and base["failed"]==13 and base["steps"][7]["code"]==11
for n,v in data.items():
 assert v["repeat"] and v["capture_noninvasive"] and v["exact_trace"],n
 assert exposed(v)==exposed(base),n
s=7
W=np.empty((9,9),dtype=np.float64)
for axis in range(3):
 rows=data["axis"+str(axis)]["steps"][s]["rows"]
 assert len(rows)==3
 for target,row in enumerate(rows):
  scratch=row["scratch"]
  assert len(scratch)==12
  ids=[int(x) for x in scratch[9:12]]
  assert sorted(ids)==[0,1,2] and ids==scratch[9:12]
  for slot,source in enumerate(ids):
   W[3*target+axis,3*source:3*source+3]=scratch[3*slot:3*slot+3]
assert np.isfinite(W).all()
def stats(A):
 sym=(A+A.T)/2
 eig=np.linalg.eigvalsh(sym)
 sing=np.linalg.svd(A,compute_uv=False)
 scale=float(np.max(np.abs(eig)))
 asym=float(np.max(np.abs(A-A.T)))
 return {"relative_max_asymmetry":asym/scale,"symmetric_eigenvalues":eig.tolist(),"singular_values":sing.tolist(),"condition_2":float(sing[0]/sing[-1]),"symmetry_gate":asym/scale<=1e-4,"positive_gate":float(eig[0])>=-1e-5*scale}
normal=W[np.ix_([0,3,6],[0,3,6])]
ans={"source_revision":"740d6082f", "earlier_baseline_exposed_equal":True,
 "all_capture_exposed_equal_and_replayed":True,"requested_minimum_sweeps":2,
 "shader_maximum_sweeps":128,"step":7,"row_source_order":[0,1,2],
 "matrix_units":"velocity per impulse, including diagonal velocity-per-impulse regularization",
 "matrix":W.tolist(),"full":stats(W),"normal":dict(matrix=normal.tolist(),**stats(normal)),
 "normal_tangent_cross_max":float(np.max(np.abs(W[np.ix_([0,3,6],[1,2,4,5,7,8])]))),
 "captured_action_independently_validated":False,
 "baseline_sha256":hashlib.sha256((p/"baseline.json").read_bytes()).hexdigest()}
(p/"analysis.json").write_text(json.dumps(ans,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:ans[k] for k in ("full","normal","normal_tangent_cross_max")},indent=2))
