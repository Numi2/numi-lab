import json,pathlib
import numpy as np
p=pathlib.Path(__file__).resolve().parent;out=p/'operator'
def read(n):return json.loads((out/(n+'.stdout')).read_text())
base=json.loads((p/'runs/rod9-long.stdout').read_text())
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('row_trace','scratch')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
names=['axis'+str(i) for i in range(3)]+['rigid'+str(i) for i in range(3)]+['factor'+str(i) for i in range(8)]+['witness']
equiv={n:clean(read(n))==clean(base) and all(read(n)[k] for k in ('repeat','capture_noninvasive','exact_trace')) for n in names}
assert all(equiv.values()),equiv
step=5
factor=[]
for part in range(8):
 rows=read('factor'+str(part))['steps'][step]['rows'];assert len(rows)==3
 for row in rows:factor.extend(row['scratch'])
factor=np.asarray(factor[:259],dtype=np.float64);assert np.isfinite(factor).all()
L=np.zeros((27,27))
for r in range(27):
 for c in range(max(0,r-8),r+1):L[r,c]=factor[9*r+r-c]
T=np.zeros((8,8))
for r in range(8):
 T[r,r]=factor[243+2*r]
 if r>0:T[r,r-1]=factor[243+2*r+1]
assert np.all(np.diag(L)>0) and np.all(np.diag(T)>0)
A=np.zeros((35,35));A[:27,:27]=L@L.T;A[27:,27:]=T@T.T;J=np.zeros((9,35))
witness=read('witness')['steps'][step]['rows']; identity=json.loads((p/'witness/rod9.stdout').read_text())['steps'][step]['rows']
rod_rows=[]
for ri,row in enumerate(witness):
 if not (row['flags']&8):continue
 rod_rows.append(ri);x=row['scratch'];a,b=int(x[0]),int(x[1]);edge_index=int(identity[ri]['scratch'][2]);assert [a,b]==[int(y) for y in identity[ri]['scratch'][:2]] and 0<=edge_index<8
 weight,radius=x[2:4];edge=np.array(x[4:7]);radial=np.array(x[8:11]);normal=np.array(row['normal']);tangent=np.array(row['tangent']);unit=edge/np.linalg.norm(edge);length=np.linalg.norm(edge);axes=[normal,tangent,np.cross(normal,tangent)]
 for col in range(35):
  va=np.zeros(3);vb=np.zeros(3)
  if col<27 and col//3==a:va[col%3]=1
  if col<27 and col//3==b:vb[col%3]=1
  twist=1. if col==27+edge_index else 0.;diff=vb-va;tdot=(diff-unit*np.dot(unit,diff))/length
  surface=(1-weight)*va+weight*vb+np.cross(np.cross(unit,tdot)+twist*unit,radius*radial)
  for axis in range(3):J[3*ri+axis,col]=-np.dot(axes[axis],surface)
assert len(rod_rows)==2,rod_rows
expected=J@np.linalg.solve(A,J.T)
def matrix(prefix):
 W=np.zeros((9,9))
 for axis in range(3):
  for target,row in enumerate(read(prefix+str(axis))['steps'][step]['rows']):
   x=row['scratch'];ids=[int(v) for v in x[9:12]];assert sorted(ids)==[0,1,2] and ids==x[9:12]
   for slot,src in enumerate(ids):W[3*target+axis,3*src:3*src+3]=x[3*slot:3*slot+3]
 assert np.isfinite(W).all();return W
full=matrix('axis');rigid=matrix('rigid');measured=full-rigid
max_error=float(np.abs(expected-measured).max());scale=float(np.abs(expected).max());relative=max_error/scale
negative=measured.copy();negative[3*rod_rows[0],3*rod_rows[0]]+=1.;negative_rejected=float(np.abs(expected-negative).max())>.01
passed=max_error<=.01 and relative<=1e-5 and negative_rejected
result={'step_index':step,'capture_equivalence':equiv,'rod_rows':rod_rows,'positive_factor_diagonals':True,'rod_reference_matrix':expected.tolist(),'captured_rod_matrix':measured.tolist(),'max_absolute_difference':max_error,'max_reference_entry':scale,'relative_difference':relative,'max_full_entry':float(np.abs(full).max()),'max_rigid_entry':float(np.abs(rigid).max()),'absolute_gate':.01,'relative_gate':1e-5,'gate_passed':passed,'negative_control_rejected':negative_rejected,'note':'FP64 reconstruction validates rod response of this captured factor and witness; articulated response remains from runtime helper. Full-minus-rigid subtraction may amplify FP32 cancellation.'}
(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if not k.endswith('matrix')},indent=2));raise SystemExit(0 if passed else 1)
