import sys,pathlib,time,json
from decimal import Decimal,localcontext
import numpy as np
Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'; REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools'; MPDIR=Q/'analysis/native531-all23-corrected-hill-highprecision-013-001/implementation'
sys.path[:0]=[str(PREP),str(REF),str(MPDIR)]
import strict_layers, cardboard_compensated_fp64_reference as strict
import audit222_corrected as mp
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
A=np.load(Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz')
h,n,t,_=strict_layers.geometry.capture_records(CAP)
t=sorted((x for x in t if x['environment']==0),key=lambda x:x['tet_id'])
mat=np.asarray(A['material']); paper=np.flatnonzero(mat!=2); rowmap={int(tid):i for i,tid in enumerate(paper)}
sel=sorted(set([1747,1754,1757,828,435,420]+list(range(18))))
spec={x['index']:x for x in h['materials']}; pars=np.asarray(h['parameters_by_environment'][0],np.float32)
dt=float(np.float32(h['dt_seconds']))
f32=strict_layers.f32
def decmat(x): return [[mp.dfloat(v) for v in row] for row in np.asarray(x)]
def decvec(x): return [mp.dfloat(v) for v in np.asarray(x).reshape(-1)]
start=time.monotonic(); counts=[]
with localcontext() as ctx:
 ctx.prec=90
 for tid in sel:
  ri=rowmap[tid]; mm=int(mat[tid]); off=spec[mm]['parameter_offset']; pp=[mp.dfloat(v) for v in pars[off:off+28]]
  f0=decmat(A['F_center'][tid]); old=decvec(A['accepted_paper_history'][ri]); H=decmat(f32(t[tid]['center']['F_direction_local_per_s']))
  for power in (26,28):
   ep=Decimal(1)/(Decimal(2)**power); delta=ep*mp.dfloat(dt)
   for sgn in (-1,1):
    F=[[f0[i][j]+sgn*delta*H[i][j] for j in range(3)] for i in range(3)]
    z,info=mp.project(pp,F,old,maxiter=80)
    counts.append((tid,power,sgn,info['iterations'],info['residual']))
mp_time=time.monotonic()-start
print(json.dumps({'phase':'mp','rows':len(sel),'endpoint_solves':len(counts),'seconds':mp_time,'max_iter':max(x[3] for x in counts),'counts':counts[:8]}))
# strict FP64 API pilot on same subset
ref=strict.Reference(strict_layers.ref)
P=[];F0=[];OLD=[];H=[]
for tid in sel:
 ri=rowmap[tid]; mm=int(mat[tid]); off=spec[mm]['parameter_offset']
 P.append(f32(pars[off:off+28])); F0.append(A['F_center'][tid]); OLD.append(A['accepted_paper_history'][ri]); H.append(f32(t[tid]['center']['F_direction_local_per_s']))
P=np.asarray(P);F0=np.asarray(F0);OLD=np.asarray(OLD);H=np.asarray(H)
start=time.monotonic(); fcounts=[]
for power in (26,28):
 ep=2.0**(-power)*dt
 for sgn in (-1,1):
  F=F0+sgn*ep*H
  z,it,rn,cn,ok,why,prov=strict.project_with_metrics(ref,P,F,OLD)
  fcounts.append({'power':power,'sign':sgn,'pass':int(ok.sum()),'fail':int((~ok).sum()),'maxres':float(np.nanmax(rn)),'maxcorr':float(np.nanmax(cn)),'itersmax':int(np.max(it))})
print(json.dumps({'phase':'fp64','seconds':time.monotonic()-start,'cases':fcounts},allow_nan=False))
