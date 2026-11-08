import json,pathlib,hashlib
import numpy as np
p=pathlib.Path(__file__).resolve().parent
read=lambda n:json.loads((p/(n+".json")).read_text())
base=read("baseline")
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ("row_trace","scratch")}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
for n in ("rigid0","rigid1","rigid2","factor0","factor1","factor2"):
 a=read(n)
 assert clean(a)==clean(base) and a["repeat"] and a["capture_noninvasive"] and a["exact_trace"],n
witness=json.loads((p.parent/"mixed-failing-row-witness-20261108/witness.json").read_text())
assert clean(witness)==clean(base)
step=7
factor=[]
for part in range(3):
 rows=read("factor"+str(part))["steps"][step]["rows"]
 assert len(rows)==3
 for row in rows:factor.extend(row["scratch"])
factor=np.asarray(factor[:85],dtype=np.float64)
assert np.isfinite(factor).all()
L=np.zeros((9,9))
for r in range(9):
 for c in range(r+1):L[r,c]=factor[9*r+r-c]
T=np.array([[factor[81],0.],[factor[84],factor[83]]])
assert np.all(np.diag(L)>0) and np.all(np.diag(T)>0)
A=np.zeros((11,11));A[:9,:9]=L@L.T;A[9:,9:]=T@T.T
J=np.zeros((9,11))
for ri in (1,2):
 row=witness["steps"][step]["rows"][ri]
 scratch=row["scratch"]
 a,b=int(scratch[0]),int(scratch[1]);weight=scratch[2];radius=scratch[3]
 edge=np.array(scratch[4:7]);radial=np.array(scratch[8:11]);normal=np.array(row["normal"]);tangent=np.array(row["tangent"])
 unit=edge/np.linalg.norm(edge);length=np.linalg.norm(edge)
 axes=[normal,tangent,np.cross(normal,tangent)]
 for col in range(11):
  va=np.zeros(3);vb=np.zeros(3)
  if col//3==a and col<9:va[col%3]=1
  if col//3==b and col<9:vb[col%3]=1
  twist=1.0 if col==9+a else 0.0
  diff=vb-va;tdot=(diff-unit*np.dot(unit,diff))/length
  surface=(1-weight)*va+weight*vb+np.cross(np.cross(unit,tdot)+twist*unit,radius*radial)
  for axis in range(3):J[3*ri+axis,col]=-np.dot(axes[axis],surface)
expected=J@np.linalg.solve(A,J.T)
def matrix(prefix):
 W=np.zeros((9,9))
 for axis in range(3):
  for target,row in enumerate(read(prefix+str(axis))["steps"][step]["rows"]):
   x=row["scratch"];ids=[int(v) for v in x[9:12]]
   assert sorted(ids)==[0,1,2] and ids==x[9:12]
   for slot,src in enumerate(ids):W[3*target+axis,3*src:3*src+3]=x[3*slot:3*slot+3]
 assert np.isfinite(W).all()
 return W
full=matrix("axis");rigid=matrix("rigid");measured=full-rigid
error=np.abs(expected-measured)
max_error=float(error.max());scale=float(np.abs(expected).max());relative=max_error/scale
assert max_error<=.01 and relative<=1e-5,(max_error,relative)
negative=measured.copy();negative[3,3]+=1
negative_rejected=float(np.max(np.abs(expected-negative)))>.01
assert negative_rejected
result={"exposed_capture_equivalence":True,"positive_factor_diagonals":True,
 "translation_diagonal":np.diag(L).tolist(),"twist_diagonal":np.diag(T).tolist(),
 "rod_reference_matrix":expected.tolist(),"captured_rod_matrix":measured.tolist(),
 "max_absolute_difference":max_error,"max_reference_entry":scale,"relative_difference":relative,
 "absolute_gate":.01,"relative_gate":1e-5,"gate_passed":True,
 "negative_control_rejected":negative_rejected,"rigid_contribution_independently_validated":False,
 "note":"FP64 reconstruction validates rod response of this captured factor and witness; articulated response remains from runtime helper."}
(p/"factor-analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if not k.endswith("matrix")},indent=2))
