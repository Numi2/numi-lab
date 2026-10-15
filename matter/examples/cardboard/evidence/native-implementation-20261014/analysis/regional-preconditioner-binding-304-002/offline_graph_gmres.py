#!/usr/bin/env python3
"""Same-captured-A offline approximate replay of source091 selector-5 additive RAP+graph preconditioner."""
from pathlib import Path
import json, subprocess, hashlib, time
import numpy as np
from scipy import sparse, linalg
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'solver/rationalized-paper-control2-newton63-capture-304'
RAP=RUN/'output/rejected-conditioning-000003'
AN=ROOT/'analysis/rationalized-paper-control2-newton63-operator-304'
LAY=ROOT/'analysis/regional-preconditioner-binding-304-002'
report=json.load(open(AN/'operator-report.json'))
A=sparse.load_npz(AN/'operator-report.operator.npz').tocsr()
with np.load(AN/'operator-report.vectors.npz',allow_pickle=False) as z: vec={k:np.array(z[k]) for k in z.files}
manifest=json.load(open(RAP/'manifest.json')); E,N,M=manifest['environment_count'],manifest['fem_node_count'],manifest['mode_count']
free=vec['free_node_ids'].astype(np.int64); ndof=A.shape[0]
P=np.fromfile(RAP/'P.bin',dtype='<f4').reshape(M,N,4)
eff=np.fromfile(RAP/'effective_P.bin',dtype='<f4').reshape(E,M,N,4)
AP=np.fromfile(RAP/'AP.bin',dtype='<f4').reshape(E,M,N,4)
lu=np.fromfile(RAP/'pivoted_RAP.bin',dtype='<f4').reshape(E,M,M)[0]
piv=np.fromfile(RAP/'pivots.bin',dtype='<u4').reshape(E,M)[0].astype(int)
# Reconstruct exact layout from the source091 production C++ builder and the captured compiled topology/free mask.
cmd=[str(LAY/'layout'),'882','1920','2',str(LAY/'topology.u32'),str(LAY/'free_mask.u8')]
text=subprocess.check_output(cmd,text=True)
lines=text.splitlines(); summary=lines[0]
patch_desc=[]
for line in lines[1:]:
 if line=='nodes': break
 patch_desc.append(tuple(map(int,line.split())))
node_line=lines[1+len(patch_desc)]
assert node_line=='nodes'
patch_nodes=list(map(int,lines[2+len(patch_desc)].split()))
coverage_line=3+len(patch_desc); assert lines[coverage_line]=='coverage'
coverage=np.array(list(map(int,lines[coverage_line+1].split())),dtype=int)
patches=[]
for no,nc,_,_ in patch_desc: patches.append(np.asarray(patch_nodes[no:no+nc],dtype=int))
assert len(coverage)==N and all(len(p)>0 for p in patches)
print('layout',summary,flush=True)
# Map free FEM node IDs to the captured operator's contiguous node-major xyz dof ordering.
free_index={int(n):i for i,n in enumerate(free)}
# Native RAP's final row-permutation LU solve; emulate native float operations for the small coarse block.
def coarse_solve(rhs):
 y=np.asarray(rhs,dtype=np.float32)[piv].copy()
 for i in range(M):
  if i: y[i]=np.float32(y[i]-np.dot(lu[i,:i],y[:i]))
 x=np.empty_like(y)
 for i in range(M-1,-1,-1):
  if i+1<M: y[i]=np.float32(y[i]-np.dot(lu[i,i+1:],x[i+1:]))
  x[i]=np.float32(y[i]/lu[i,i])
 return x
Pfree=eff[0][:,free,:3].astype(np.float32).transpose(1,2,0).reshape(ndof,M)
APfree=AP[0][:,free,:3].astype(np.float32).transpose(1,2,0).reshape(ndof,M)
# The local graph matrices are reconstructed from same-generation rich-capture A, then cast to native f32.
# Source patch matrices use the restricted free-free principal block; initially fixed nodes have identity rows and are skipped on application.
start=time.time(); factors=[]; patch_shapes=[]
for k,nodes in enumerate(patches):
 fnodes=[int(n) for n in nodes if int(n) in free_index]
 if not fnodes:
  factors.append(([],None,None)); patch_shapes.append(len(nodes)); continue
 dofs=np.concatenate([np.arange(3*free_index[n],3*free_index[n]+3) for n in fnodes])
 local=A[dofs][:,dofs].toarray().astype(np.float32)
 # Exact production local matrix has identity for initially fixed nodes, but no fixed nodes are in this local solve.
 with np.errstate(all='ignore'):
  fact,p=linalg.lu_factor(local,overwrite_a=True,check_finite=True)
 factors.append((dofs,fact,p)); patch_shapes.append(len(nodes))
print('built patch factors seconds',round(time.time()-start,3),'patch sizes',patch_shapes,flush=True)
def apply(rhs):
 r=np.asarray(rhs,dtype=np.float64)
 coarse_rhs=Pfree.astype(np.float64).T@r
 c=coarse_solve(coarse_rhs)
 out=Pfree.astype(np.float64)@c.astype(np.float64)
 fine=r-APfree.astype(np.float64)@c.astype(np.float64)
 accum=np.zeros(ndof,dtype=np.float64)
 for nodes,(dofs,fact,pivloc) in zip(patches,factors):
  if len(dofs)==0: continue
  with np.errstate(all='ignore'):
   local=linalg.lu_solve((fact,pivloc),fine[dofs].astype(np.float32),check_finite=False).astype(np.float64)
  # GPU apply divides each local-node xyz correction by global coverage count.
  for j,d in enumerate(dofs):
   nodepos=j//3
   node=int(nodes[nodepos])
   accum[d]+=local[j]/coverage[node]
 return out+accum
# Evaluate same operator and RHS with restarted right-preconditioned GMRES; report true residual at each update.
b=np.asarray(vec['native_rhs_free_impulse_Ns'],dtype=np.float64)
bnorm=np.linalg.norm(b); x=np.zeros_like(b); hist=[]
for cycle in range(2):
 r=b-A@x; beta=np.linalg.norm(r); V=np.zeros((ndof,17)); Z=np.zeros((ndof,16)); H=np.zeros((17,16)); V[:,0]=r/beta
 best=x.copy()
 for j in range(16):
  Z[:,j]=apply(V[:,j]); w=A@Z[:,j]
  # source FGMRES uses modified GS; repeat reorth pass for numerical stability here and report algorithm difference.
  for i in range(j+1):
   h=float(np.dot(V[:,i],w)); H[i,j]+=h; w-=h*V[:,i]
  H[j+1,j]=np.linalg.norm(w)
  if H[j+1,j]>0: V[:,j+1]=w/H[j+1,j]
  target=np.zeros(j+2); target[0]=beta
  y,*_=np.linalg.lstsq(H[:j+2,:j+1],target,rcond=None)
  best=x+Z[:,:j+1]@y
  rel=float(np.linalg.norm(b-A@best)/bnorm)
  hist.append({'iteration':cycle*16+j+1,'true_relative_residual':rel})
 x=best
 print('cycle',cycle,'true residual rel',float(np.linalg.norm(b-A@x)/bnorm),flush=True)
# Captured native correction vector and final residual from rich capture.
x_native=np.asarray(vec['direction'],dtype=np.float64)
r_native=b-A@x_native
result={'schema':'numi.cardboard.selector5-offline-replay-304.v1','bound_operator_report_sha256':hashlib.sha256((AN/'operator-report.json').read_bytes()).hexdigest(),'bound_rap_manifest_sha256':hashlib.sha256((RAP/'manifest.json').read_bytes()).hexdigest(),'layout_builder_source_sha256':hashlib.sha256((RUN/'inputs/source/matter/src/regional_graph_schwarz.hpp').read_bytes()).hexdigest(),'layout_summary':summary,'patch_sizes':patch_shapes,'patch_factorization':'FP64 sparse A assembled from captured source091 rich operator then local dense matrices cast to float32 and scipy LU; not a native factor dump','rhs_norm_Ns':float(bnorm),'offline_history':hist,'offline_final_residual_relative':float(np.linalg.norm(b-A@x)/bnorm),'native_captured_x_norm':float(np.linalg.norm(x_native)),'native_captured_x_true_residual_relative':float(np.linalg.norm(r_native)/bnorm),'native_true_residual_reported_cycle0':0.000265761395/0.000278883468,'native_true_residual_reported_cycle1':0.000265578186/0.000278883468,'limitations':['graph patch layout reconstructed from immutable source builder and exact captured topology/fixed mask; runtime did not export selected-generation graph patch arenas in the regional capture','local patch operator matrices reconstructed from same-generation rich operator and are not byte-identical native single-precision graph factor inputs','offline Arnoldi/orthogonalization and coarse solve are not bitwise identical to Metal production','diagnostic only, not a native preconditioner parity test or qualification']}
out=LAY/'offline-graph-gmres.json'; out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print('wrote',out,'sha256',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
