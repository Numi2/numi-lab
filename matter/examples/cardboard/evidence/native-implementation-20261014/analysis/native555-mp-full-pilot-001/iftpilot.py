import sys,pathlib,time
from decimal import Decimal,localcontext
import numpy as np
Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001';REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools';MPDIR=Q/'analysis/native531-all23-corrected-hill-highprecision-013-001/implementation'
sys.path[:0]=[str(PREP),str(REF),str(MPDIR)]
import strict_layers,audit222_corrected as mp
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'; A=np.load(Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz')
h,n,t,_=strict_layers.geometry.capture_records(CAP); t=sorted((x for x in t if x['environment']==0),key=lambda x:x['tet_id'])
mat=np.asarray(A['material']); paper=np.flatnonzero(mat!=2); rowmap={int(tid):i for i,tid in enumerate(paper)}; spec={x['index']:x for x in h['materials']}; pars=np.asarray(h['parameters_by_environment'][0],np.float32); f32=strict_layers.f32
sel=sorted(set([1747,1754,1757,828,435,420]+list(range(18))))
def dm(x):return [[mp.dfloat(v) for v in row] for row in np.asarray(x)]
def dv(x):return [mp.dfloat(v) for v in np.asarray(x).reshape(-1)]
start=time.monotonic();results=[]
with localcontext() as ctx:
 ctx.prec=90
 for tid in sel:
  ri=rowmap[tid];m=int(mat[tid]);off=spec[m]['parameter_offset'];p=[mp.dfloat(v) for v in pars[off:off+28]]
  F=dm(A['F_center'][tid]);old=dv(A['accepted_paper_history'][ri]); H=f32(t[tid]['center']['F_direction_local_per_s'])
  z,info=mp.project(p,F,old); J=mp.jacobian(p,F,old,z); hsmall=Decimal('1e-25'); Hd=dm(H)
  Fp=[[F[i][j]+hsmall*Hd[i][j] for j in range(3)] for i in range(3)]; Fm=[[F[i][j]-hsmall*Hd[i][j] for j in range(3)] for i in range(3)]
  rp,_=mp.residual(p,Fp,old,z);rm,_=mp.residual(p,Fm,old,z)
  dr=[(rp[i]-rm[i])/(2*hsmall) for i in range(15)]; dz=mp.solve(J,[-x for x in dr])
  zp=[z[i]+hsmall*dz[i] for i in range(15)];zm=[z[i]-hsmall*dz[i] for i in range(15)]
  sp=mp.stress(p,Fp,zp);sm=mp.stress(p,Fm,zm)
  T=[[ (sp[i][j]-sm[i][j])/(2*hsmall) for j in range(3)] for i in range(3)]
  results.append(tid)
print({'rows':len(results),'seconds':time.monotonic()-start})
