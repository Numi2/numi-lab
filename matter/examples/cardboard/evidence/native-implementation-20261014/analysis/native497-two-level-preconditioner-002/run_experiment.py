#!/usr/bin/env python3
"""CPU-only two-level preconditioner experiment on native497's captured cached A.
Reconstructs the exact FP32-cached FEM tangent in binary64 using the frozen source
helper. Does not re-evaluate constitutive physics or alter A.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, struct, sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import solve as dense_solve, lstsq as scipy_lstsq
from scipy.sparse.linalg import splu
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
CASE=Q/"solver/warm-cgls-control20-diagnostic-replay-497"
SRC=Q/"snapshots/source-admission-encoder-182-003"
OUT=Q/"analysis/native497-two-level-preconditioner-002"
STATE=CASE/"diagnostics/capture-base-cgls-capture/capture.state.json"
PAYLOAD=CASE/"diagnostics/capture-base-cgls-capture/capture.state.bin"
CGLS=CASE/"diagnostics/capture-base-cgls-capture/capture"
FD=CASE/"diagnostics/capture-operator-fd-capture/capture"
HELPER=SRC/"matter/tools/cardboard_cached_operator_audit.py"
EXPECTED={"cgls":"d2cebe5b9e0215f2a328a92efc74db707862034bc785eccb3f8a41a9992e5646","state":"b03e1bf2ee33162b2cca750ac14127d1e4a1b2e6df68c1fe71b081b0b251975c","payload":"10ef4ce317fc99f86f338e84ecc92b3e1a465afc55853b2e73720ef66614fbd1","fd":"0666362ab1b94a1307f4f07d66f51a43355d4b32bbfb5246e0993323ae9359d3","helper":"4a7e9c0cbf09eee0ad1f8f5b2b09314a2c0789a908aaab530b56cfff6a1c82ff"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def strict_obj(path):
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d: raise ValueError("duplicate JSON key "+k)
   d[k]=v
  return d
 def constant(x): raise ValueError("nonstandard JSON constant "+x)
 return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=constant)
def strict_jsonl(path):
 rows=[]
 with path.open() as f:
  for line in f:
   if line.strip(): rows.append(json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))))
 return rows
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if not spec or not spec.loader: raise ValueError("import failed "+str(path))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def parse_cgls(path):
 raw=path.read_bytes()
 if len(raw)<32: raise ValueError("short CGLS")
 magic,ver,E,N,U,cap,nseg,res=struct.unpack_from("<8I",raw)
 if (magic,ver,E,N,U,cap,nseg,res)!=(0x534c4743,3,2,882,3528,64,13,0): raise ValueError(f"unexpected CGLS header {(hex(magic),ver,E,N,U,cap,nseg,res)}")
 specs=[]
 for name in ("x","r","s","p","q","b","mask"): specs.append((name,"<f4",U*4,(U,4)))
 specs += [("row_environment","<u4",U,(U,)),("scalars","<f4",E*4,(E,4)),("control","<u4",E*4,(E,4)),("errors","<u4",E,(E,)),("selected","<u4",E,(E,)),("right_scale_rows","<f4",E*N*3*4,(E,N,3,4))]
 arr={};cur=32
 for name,dt,count,shape in specs:
  start=(cur+15)&~15
  if any(raw[cur:start]):raise ValueError("nonzero CGLS padding "+name)
  end=start+count*4
  if end>len(raw):raise ValueError("truncated "+name)
  arr[name]=np.frombuffer(raw,dtype=dt,count=count,offset=start).reshape(shape).copy();cur=end
 if cur!=len(raw):raise ValueError("extra CGLS bytes")
 for k,v in arr.items():
  if v.dtype.kind=="f" and not np.isfinite(v).all():raise ValueError("nonfinite "+k)
 return arr
def state_array(doc,payload,name,dtype,shape):
 m={x["name"]:x for x in doc["arrays"]};d=m[name];a=payload[d["offset"]:d["offset"]+d["byte_count"]]
 if len(a)!=d["byte_count"]:raise ValueError("truncated state array "+name)
 v=np.frombuffer(a,dtype=dtype).reshape(shape).copy()
 if v.dtype.kind=="f" and not np.isfinite(v).all():raise ValueError("nonfinite "+name)
 return v
def make_source_rigid_basis(nodes,tets,free_ids):
 N=len(nodes);labels=np.array([(int(t["tet_flags"])>>8)&7 for t in tets],dtype=np.int32);regions=sorted(set(map(int,labels)))
 if regions!=list(range(len(regions))) or not regions or len(regions)>5:raise ValueError("noncompact/out-of-bound region labels")
 x=np.array([n["accepted"]["x_m"] for n in nodes],dtype=np.float64)
 if x.shape!=(N,3) or not np.isfinite(x).all():raise ValueError("bad rest geometry")
 free=np.zeros(N,dtype=bool);free[free_ids]=True
 nv=np.zeros((len(regions),N),dtype=np.float64);vol=np.zeros(len(regions));mom=np.zeros((len(regions),3));conns=[]
 for ti,t in enumerate(tets):
  if not (int(t["tet_flags"])&1):raise ValueError("inactive tet")
  r=int(labels[ti]);v=float(t["rest_volume_m3"]);ids=np.asarray(t["nodes"],dtype=np.int64)
  if ids.shape!=(4,) or ids.min()<0 or ids.max()>=N or not (v>0 and math.isfinite(v)):raise ValueError("bad tet")
  c=x[ids].mean(axis=0);vol[r]+=v;mom[r]+=v*c
  for node in ids:nv[r,node]+=0.25*v
  conns.append(ids)
 if np.any(vol<=0):raise ValueError("empty region")
 centers=mom/vol[:,None];second=np.zeros(len(regions))
 for ti,t in enumerate(tets):
  r=int(labels[ti]);ids=conns[ti];v=float(t["rest_volume_m3"])
  for node in ids:
   d=x[node]-centers[r];second[r]+=0.25*v*float(d@d)
 total=nv.sum(axis=0);Qm=[];labels_out=[]
 for r in regions:
  radius=math.sqrt(second[r]/vol[r])
  for mode in range(6):
   q=np.zeros((N,3),dtype=np.float64)
   for node in range(N):
    if not free[node] or total[node]<=0:continue
    w=nv[r,node]/total[node]
    if not w>0:continue
    rx,ry,rz=x[node]-centers[r]
    if mode<3:q[node,mode]=w
    elif radius>1e-15:
     s=w/radius
     if mode==3:q[node]=[0,-s*rz,s*ry]
     elif mode==4:q[node]=[s*rz,0,-s*rx]
     else:q[node]=[-s*ry,s*rx,0]
   cand=q.reshape(-1);raw_sq=0.0
   for value in cand:raw_sq+=float(value)*float(value)
   raw_norm=math.sqrt(raw_sq)
   if raw_norm==0:continue
   for _ in range(2):
    for base in Qm:
     projection=0.0
     for i in range(cand.size):projection+=float(base[i])*float(cand[i])
     for i in range(cand.size):cand[i]-=projection*base[i]
   norm_sq=0.0
   for value in cand:norm_sq+=float(value)*float(value)
   norm=math.sqrt(norm_sq)
   if not (norm>1e-12*raw_norm and math.isfinite(norm)):continue
   cand/=norm
   if not np.isfinite(cand).all(): print("BADMODE",r,mode,raw_norm,norm,float(np.nanmax(np.abs(cand))))
   Qm.append(cand);labels_out.append((r,mode))
 if not Qm:raise ValueError("empty rigid coarse basis")
 return np.column_stack(Qm),labels_out,labels,x
def norm2(v):
 total=0.0
 for a in np.asarray(v).reshape(-1): total+=float(a)*float(a)
 return math.sqrt(total)
def dot2(a,b):
 total=0.0
 aa=np.asarray(a).reshape(-1); bb=np.asarray(b).reshape(-1)
 for i in range(aa.size): total+=float(aa[i])*float(bb[i])
 return total
def sparse_lstsq(A,b): return scipy_lstsq(A,b,lapack_driver='gelsd')[0]
def sparse_mv(A,v):
 v=np.asarray(v).reshape(-1); out=np.zeros(A.shape[0],dtype=np.float64)
 for i in range(A.shape[0]):
  total=0.0
  for k in range(A.indptr[i],A.indptr[i+1]): total+=float(A.data[k])*float(v[A.indices[k]])
  out[i]=total
 return out
def sparse_mm(A,V):
 V=np.asarray(V); out=np.zeros((A.shape[0],V.shape[1]),dtype=np.float64)
 for i in range(A.shape[0]):
  row=out[i]
  for k in range(A.indptr[i],A.indptr[i+1]): row+=float(A.data[k])*V[A.indices[k]]
 return out
def combine_columns(V,c):
 out=np.zeros(V.shape[0],dtype=np.float64)
 for j in range(V.shape[1]):
  if c[j]!=0.0: out+=float(c[j])*V[:,j]
 return out
def gmres_right(A,M,b,x0,restart=16,cycles=8):
 x=x0.copy(); r=b-sparse_mv(A,x); beta0=norm2(r)
 hist=[{"iteration":0,"residual_l2":beta0,"relative_to_initial":1.0}]
 n=b.size; anorm=norm2(A.data)
 for cyc in range(cycles):
  r=b-sparse_mv(A,x); beta=norm2(r)
  if beta==0.0: break
  V=np.zeros((n,restart+1)); Z=np.zeros((n,restart)); H=np.zeros((restart+1,restart))
  V[:,0]=r/beta; xbase=x.copy(); breakdown=False
  for j in range(restart):
   Z[:,j]=M(V[:,j]); w=sparse_mv(A,Z[:,j])
   for _ in range(2):
    for i in range(j+1):
     hij=dot2(V[:,i],w); H[i,j]+=hij; w-=hij*V[:,i]
   H[j+1,j]=norm2(w)
   if H[j+1,j]>np.finfo(float).eps*max(1.0,anorm): V[:,j+1]=w/H[j+1,j]
   rhs=np.zeros(j+2); rhs[0]=beta
   y=sparse_lstsq(H[:j+2,:j+1],rhs)
   x=xbase+combine_columns(Z[:,:j+1],y)
   rn=norm2(b-sparse_mv(A,x))
   hist.append({"iteration":cyc*restart+j+1,"residual_l2":rn,
                "relative_to_initial":rn/beta0 if beta0 else None,
                "cycle":cyc+1,"inner_iteration":j+1})
   if H[j+1,j]<=np.finfo(float).eps*max(1.0,anorm): breakdown=True; break
  if breakdown: break
 return x,hist
def main():
 sys.path.insert(0,str(HELPER.parent))
 OUT.mkdir(parents=True,exist_ok=True)
 ps={"cgls":CGLS,"state":STATE,"payload":PAYLOAD,"fd":FD,"helper":HELPER};binds={}
 for name,p in ps.items():
  actual=sha(p)
  if actual!=EXPECTED[name]:raise ValueError(f"{name} hash mismatch {actual}")
  binds[name]={"path":str(p),"sha256":actual,"bytes":p.stat().st_size}
 doc=strict_obj(STATE);payload=PAYLOAD.read_bytes();recs=strict_jsonl(FD)
 if recs[0].get("record")!="header":raise ValueError("FD header missing")
 header=recs[0];integrity=[r for r in recs if r.get("record")=="integrity"]
 nodes=[r for r in recs if r.get("record")=="node" and int(r["environment"])==0];tets=[r for r in recs if r.get("record")=="tetrahedron" and int(r["environment"])==0]
 if len(nodes)!=882 or len(tets)!=1920 or len(integrity)!=1:raise ValueError("capture record count mismatch")
 op=module("cached_operator_497",HELPER)
 mass,K,A,_,meta=op.validate_and_assemble(header,[r for r in recs if r.get("record")=="node"],[r for r in recs if r.get("record")=="tetrahedron"],integrity[0],0)
 if A.shape!=(2538,2538) or A.nnz!=71046:raise ValueError(f"A mismatch {A.shape}/{A.nnz}")
 c=parse_cgls(CGLS)
 if c["selected"].tolist()!=[1,0]:raise ValueError("selected env mask mismatch")
 E,N,U=2,882,3528;free_ids=np.asarray(meta["free_nodes"],dtype=np.int64)
 x0all=state_array(doc,payload,"original_newton_correction","<f4",(U,4));ball=state_array(doc,payload,"base_residual_rhs","<f4",(U,4))
 x0=x0all[free_ids,:3].astype(np.float64).reshape(-1);b=ball[free_ids,:3].astype(np.float64).reshape(-1)
 X=c["x"][free_ids,:3].astype(np.float64).reshape(-1)
 Hblocks=c["right_scale_rows"][0,free_ids,:,:3].astype(np.float64)
 if Hblocks.shape!=(846,3,3) or not np.isfinite(Hblocks).all():raise ValueError("captured block scale shape/finite failure")
 def h_apply(v):return np.einsum("nij,nj->ni",Hblocks,v.reshape(-1,3),optimize=False).reshape(-1)
 P,mode_labels,region_labels,coords=make_source_rigid_basis(nodes,tets,free_ids);P=P.reshape(N,3,-1)[free_ids].reshape(3*len(free_ids),-1).astype(np.float32).astype(np.float64)
 AP=sparse_mm(A,P);R=np.zeros((P.shape[1],P.shape[1]),dtype=np.float64)
 for i in range(P.shape[1]):
  for j in range(P.shape[1]): R[i,j]=dot2(P[:,i],AP[:,j])
 if np.linalg.matrix_rank(R)<R.shape[0]:raise ValueError("regional RAP rank deficient")
 ccond=float(np.linalg.cond(R))
 def coarse_coeff(v):return dense_solve(R,np.array([dot2(P[:,j],v) for j in range(P.shape[1])]),assume_a="gen",check_finite=True)
 def M_identity(v):return np.asarray(v,dtype=np.float64)
 def M_block(v):return h_apply(v)
 def M_regional(v):return combine_columns(P,coarse_coeff(v))
 def M_two(v):
  cc=coarse_coeff(v);pc=combine_columns(P,cc);apc=combine_columns(AP,cc)
  return pc+h_apply(v-apc)
 coarse_test=combine_columns(P,np.linspace(-0.75,1.25,P.shape[1]));two_err=norm2(M_two(sparse_mv(A,coarse_test))-coarse_test)/max(norm2(coarse_test),1e-300)
 block_err=norm2(M_block(sparse_mv(A,coarse_test))-coarse_test)/max(norm2(coarse_test),1e-300)
 lu=splu(A.tocsc(),permc_spec="COLAMD",diag_pivot_thresh=1.0);xstar=lu.solve(b);r0=b-sparse_mv(A,x0)
 methods={"identity":M_identity,"captured_block3":M_block,"regional_coarse_only":M_regional,"source_block_RAP_two_level":M_two};histories={};solutions={}
 for name,M in methods.items():solutions[name],histories[name]=gmres_right(A,M,b,x0,16,8)
 exact_res=norm2(b-sparse_mv(A,xstar))
 np.save(OUT/"exact_lu_solution_free_xyz.npy",np.asarray(xstar,dtype="<f8"),allow_pickle=False)
 np.save(OUT/"source_rigid_basis_free_xyz.npy",np.asarray(P,dtype="<f8"),allow_pickle=False)
 np.save(OUT/"captured_block3_right_scale_free_xyz.npy",np.asarray(Hblocks,dtype="<f8"),allow_pickle=False)
 report={
  "schema":"numi.cardboard.native497_two_level_regional_preconditioner_cpu_replay.v1",
  "verdict":"CPU-only fixed-cached-operator Krylov comparison; no native preconditioner execution, no physical qualification",
  "source_binding":{"source_snapshot":str(SRC),"source_snapshot_manifest_sha256":sha(SRC/"frozen-source.json"),"cached_operator_helper_sha256":sha(HELPER),"basis_algorithm":"runtime.mm buildRegionalGalerkinBasis rigid 6 modes per compact region; volume-overlap weights, two-pass modified Gram-Schmidt; captured accepted positionAndMass geometry and region tags are from exact capture; optional global bending enrichment omitted because executionBudgets.z is unavailable"},
  "input_bindings":binds,
  "capture":{"root":doc["root"],"newton_iteration":doc["newton_iteration"],"environment":0,"selected_words":[int(x) for x in c["selected"]],"nodes":N,"tets":len(tets),"free_nodes":int(len(free_ids)),"free_xyz_dimension":A.shape[0],"nnz":int(A.nnz),"region_counts":{str(int(r)):int(np.count_nonzero(region_labels==r)) for r in sorted(set(region_labels.tolist()))},"region_tet_counts":{str(r):int(sum(1 for t in tets if ((int(t["tet_flags"])>>8)&7)==r)) for r in sorted(set(region_labels.tolist()))},"dt_seconds":float(meta["dt_seconds"])},
  "operator":{"definition":"A=M+dt^2 K from exact captured FP32 algorithmic tangent; assembled once in binary64 for all methods","frobenius_norm":norm2(A.data),"nnz":int(A.nnz),"relative_skew_norm":float(sparse.linalg.norm(A-A.T)/sparse.linalg.norm(A)),"rhs":"captured same-generation base_residual_rhs on free xyz rows","initial_guess":"captured original Newton correction x0; solve remaining residual b-A*x0","initial_residual_l2":float(np.linalg.norm(r0)),"direct_LU_residual_l2":exact_res,"direct_LU_solution_l2":norm2(xstar),"captured_CGLS_total_X_residual_l2":norm2(b-sparse_mv(A,X)),"captured_CGLS_total_X_forward_error_to_LU":norm2(X-xstar)},
  "preconditioners":{"identity":"M=I","captured_block3":"M=H, exact same-generation captured per-node 3x3 right_scale_rows on free xyz","regional_coarse_only":"M=P(P^T A P)^-1 P^T","source_block_RAP_two_level":"M(r)=P c + H(r-AP c), c=(P^T A P)^-1 P^T r; exact algebraic form of nm_fgmres_regional_apply_block_rap with the captured block H and source rigid P; AP and RAP recomputed in binary64 from the same A"},
  "coarse_space":{"dimension":int(P.shape[1]),"mode_labels":[{"region":int(r),"mode":int(m)} for r,m in mode_labels],"R_condition_2":ccond,"R_solve":"general dense solve; no symmetry assumption","two_level_coarse_reproduction_relative_error":two_err,"block3_coarse_reproduction_relative_error":block_err,"basis_file":{"path":"source_rigid_basis_free_xyz.npy","sha256":sha(OUT/"source_rigid_basis_free_xyz.npy")},"source_geometry":"captured accepted positionAndMass.xyz; source preconditioner region bits from tet_flags; same operator free mask","max_abs_current_position_minus_rest_m":float(np.max(np.abs(coords-np.array([n["accepted"]["rest_m"] for n in nodes],dtype=np.float64)))),"basis_storage_precision":"double generation rounded to float32 as source host upload then widened"},
  "gmres_restart16":{"cycles":8,"max_inner_iterations":128,"method":"custom right-preconditioned restarted GMRES(16), twice-reorthogonalized modified Gram-Schmidt; true unpreconditioned residual b-A*x recomputed in binary64 each inner iteration","histories":histories,"solutions":{"exact_lu":{"path":"exact_lu_solution_free_xyz.npy","sha256":sha(OUT/"exact_lu_solution_free_xyz.npy")},"captured_block3_x_l2":norm2(solutions["captured_block3"]),"two_level_x_l2":norm2(solutions["source_block_RAP_two_level"]),"captured_block3_final_residual_l2":histories["captured_block3"][-1]["residual_l2"],"two_level_final_residual_l2":histories["source_block_RAP_two_level"][-1]["residual_l2"],"captured_block3_solution_error_to_lu":norm2(solutions["captured_block3"]-xstar),"two_level_solution_error_to_lu":norm2(solutions["source_block_RAP_two_level"]-xstar)}},
  "limits":["Only selected environment 0 free mechanical xyz block; field backing, env1, rigid/contact/cohesive rows excluded.","A is reconstructed from native FP32 tangent cache in binary64, not re-evaluated material law or native GPU action.","P uses captured accepted positionAndMass for source regional translation/rotation modes; optional global bending enrichment remains unknown because executionBudgets.z is not captured, and production P/AP/RAP are not captured.","H is exact same-generation captured right scale; GMRES with fixed H is not native CGLS/FGMRES replay.","All methods use the same A, b, x0 and binary64 residual history; no acceptance tolerance/gate is altered.","Exact LU is a numerical reference for this captured matrix only, not nonlinear convergence or physical assembly qualification."]
 }
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"out":str(OUT),"dimension":A.shape[0],"nnz":A.nnz,"coarse_modes":P.shape[1],"Rcond2":ccond,"r0":float(np.linalg.norm(r0)),"block16":histories["captured_block3"][15]["residual_l2"],"block128":histories["captured_block3"][-1]["residual_l2"],"two16":histories["source_block_RAP_two_level"][15]["residual_l2"],"two128":histories["source_block_RAP_two_level"][-1]["residual_l2"],"direct":exact_res},indent=2))
if __name__=="__main__":main()
