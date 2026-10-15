#!/usr/bin/env python3
"""FP32 RAP admission/solve sensitivity for optional coarse-mode controls."""
import importlib.util, json, math, os, hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy import linalg
ROOT=Path(__file__).resolve().parent; BASE=Path('/Users/home/cardboard-implementation-20261014')
spec=importlib.util.spec_from_file_location('recycle_study',ROOT/'study.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def f32(x): return np.float32(x)
def dot3(a,b):
 return f32(f32(f32(a[0]*b[0])+f32(a[1]*b[1]))+f32(a[2]*b[2]))
def simd_sum_nodes(P,AP):
 lanes=np.zeros(32,dtype=np.float32)
 for lane in range(32):
  acc=f32(0)
  for node in range(lane,P.shape[0],32): acc=f32(acc+dot3(P[node],AP[node]))
  lanes[lane]=acc
 width=16
 while width:
  for lane in range(width): lanes[lane]=f32(lanes[lane]+lanes[lane+width])
  width//=2
 return lanes[0]
def rap_build(P,AP):
 m=P.shape[2]; R=np.zeros((m,m),np.float32)
 for i in range(m):
  for j in range(m): R[i,j]=simd_sum_nodes(P[:,:,i],AP[:,:,j])
 return R
def factor_source(R):
 a=np.asarray(R,dtype=np.float32).copy();m=a.shape[0]; piv=np.arange(m,dtype=np.uint32)
 scale=f32(0)
 for v in a.reshape(-1):
  if not np.isfinite(v): return {'valid':False,'reason':'nonfinite input'}
  scale=max(scale,abs(v))
 floor=max(f32(scale*f32(1e-10)),f32(1e-30)); minp=float('inf');
 if not (scale>0): return {'valid':False,'reason':'zero scale'}
 for k in range(m):
  pr=k; pm=f32(0)
  for row in range(k,m):
   mag=abs(a[row,k])
   if mag>pm:pm=mag;pr=row
  if not (pm>floor) or not np.isfinite(pm): return {'valid':False,'reason':'pivot below floor','pivot_index':k,'pivot':float(pm),'floor':float(floor),'matrix':a,'pivots':piv}
  if pr!=k:
   a[[k,pr],:]=a[[pr,k],:];piv[[k,pr]]=piv[[pr,k]]
  diag=a[k,k]
  if not (abs(diag)>floor) or not np.isfinite(diag):return {'valid':False,'reason':'diag below floor','pivot_index':k,'pivot':float(diag),'floor':float(floor),'matrix':a,'pivots':piv}
  minp=min(minp,float(abs(diag)))
  for row in range(k+1,m):
   lower=f32(a[row,k]/diag)
   if not np.isfinite(lower):return {'valid':False,'reason':'nonfinite lower','pivot_index':k}
   a[row,k]=lower
   for col in range(k+1,m):
    a[row,col]=f32(a[row,col]-f32(lower*a[k,col]))
    if not np.isfinite(a[row,col]):return {'valid':False,'reason':'nonfinite update','pivot_index':k}
 return {'valid':True,'matrix':a,'pivots':piv,'scale':float(scale),'floor':float(floor),'min_abs_U_diagonal':minp,'min_relative_U_diagonal':minp/max(float(scale),1e-300)}
def solve_source(lu,piv,b):
 m=len(b); y=np.asarray(b,dtype=np.float32)[piv].copy()
 for i in range(m):
  v=f32(y[i])
  for j in range(i):v=f32(v-f32(lu[i,j]*y[j]))
  y[i]=v
 x=np.zeros(m,np.float32)
 for i in range(m-1,-1,-1):
  v=f32(y[i])
  for j in range(i+1,m):v=f32(v-f32(lu[i,j]*x[j]))
  x[i]=f32(v/lu[i,i])
 return x

def main():
 report,A,vectors,topo,pm,graph,state,free,b,native,Z,orth=S.load_inputs()
 Pfull=np.fromfile(S.P_DIR/'P.bin',dtype='<f4').reshape(26,882,4)
 pfree=np.transpose(Pfull[:,free,:3],(1,2,0)).reshape(2538,26).astype(np.float64)
 candidates={'baseline26':None,
  'lowest_sym_eigenmode_oracle':np.load(S.AUDIT/'symmetric-part-modes.npz')['eigenvectors'][:,0],
  'captured_current_native_FGMRES_correction_future_mode':native}
 out={'schema':'numi.cardboard.conditioning-recycle-fp32-rap-controls.v1','classification':'CPU source arithmetic emulation; not Metal/native execution',
  'selection':report.get('capture_selection'),'basis_source_P_sha256':S.sha(S.P_DIR/'P.bin'),'operator_npz_sha256':S.sha(S.AUDIT/'report.operator.npz'),
  'source_fgmres_sha256':S.sha(S.FGMRES_METAL),'controls':{}}
 state_full=np.zeros((882,3,26),np.float32);state_full[free]=pfree.reshape(846,3,26).astype(np.float32)
 for name,v in candidates.items():
  if v is None: Zfull=state_full
  else:
   Zplus, meta=S.candidate_basis(pfree,np.asarray(v,dtype=np.float64));
   z=Zplus[:,-1].astype(np.float32).reshape(846,3)
   Zfull=np.concatenate([state_full,np.zeros((882,3,1),np.float32)],axis=2);Zfull[free,:,-1]=z
  m=Zfull.shape[2]
  # Apply captured sparse operator to each source-mode basis; convert AP to the
  # float32 storage used by the source coarse builder.
  Zfree=np.transpose(Zfull[free],(0,1,2)).reshape(2538,m).astype(np.float32)
  APfree=np.asarray(A@Zfree.astype(np.float64),dtype=np.float32)
  APfull=np.zeros_like(Zfull);APfull[free]=APfree.reshape(846,3,m)
  R=rap_build(Zfull,APfull)
  f=factor_source(R)
  result={'mode_count':m,'candidate_orthogonalization':None if v is None else meta,
    'RAP_max_abs':float(np.max(np.abs(R))),'RAP_finite':bool(np.isfinite(R).all()),
    'RAP_condition_double':float(np.linalg.cond(R.astype(np.float64))),
    'RAP_f32_factor_valid_source_pivot_floor':bool(f.get('valid',False)),
    'RAP_factor':{k:v for k,v in f.items() if k not in ('matrix','pivots')},
    'RAP_eigenvalues_double':np.linalg.eigvals(R.astype(np.float64)).tolist(),
    'RAP_sha256_f32':hashlib.sha256(np.asarray(R,dtype='<f4').tobytes()).hexdigest()}
  if f.get('valid'):
   rhs=np.zeros(m,np.float32)
   for i in range(m): rhs[i]=simd_sum_nodes(Zfull[:,:,i],np.pad(b.reshape(846,3).astype(np.float32),((0,36),(0,0))))
   # fixed nodes have no P support; the padded residual is only a packing aid.
   coeff=solve_source(f['matrix'],f['pivots'],rhs)
   result['projected_rhs_norm']=float(np.linalg.norm(rhs.astype(np.float64)))
   result['coarse_coeff_l2']=float(np.linalg.norm(coeff.astype(np.float64)))
   result['coarse_coeff_finite']=bool(np.isfinite(coeff).all())
  out['controls'][name]=result
 outpath=ROOT/'run-004-fp32-rap.json';outpath.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(outpath)
if __name__=='__main__':main()
