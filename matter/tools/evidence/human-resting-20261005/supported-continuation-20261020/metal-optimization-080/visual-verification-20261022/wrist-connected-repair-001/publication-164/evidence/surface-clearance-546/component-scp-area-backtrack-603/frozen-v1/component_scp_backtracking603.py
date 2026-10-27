"""Independent component SCP backtracking using frozen 408 gates and 549 stencils."""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
from scipy import sparse

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
P549=D/"D/component-area-backtracking-549/component_area_backtracking549.py"
H549="523b1aef89d92eef09bf63e9de2d4f3fa6bd938ef53a45915f9ea138f5b8f249"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def csr(x,name):
 a=sparse.csr_matrix(x,dtype=np.float64,copy=True);a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
 if not np.isfinite(a.data).all():raise ValueError(f"{name} has nonfinite values")
 return a

def row_owner_audit(mat,labels,name):
 a=csr(mat,name);labels=np.asarray(labels,dtype=np.int64)
 if a.shape[1]!=3*len(labels):raise ValueError(f"{name} columns mismatch support")
 own=np.full(a.shape[0],-1,dtype=np.int64);zero=0
 for r in range(a.shape[0]):
  lo,hi=a.indptr[r:r+2]
  if lo==hi:zero+=1;continue
  q=np.unique(labels[a.indices[lo:hi]//3]);q=q[q>=0]
  if len(q)>1:raise ValueError(f"{name} row {r} crosses components {q[:8].tolist()}")
  if len(q):own[r]=q[0]
 return {"name":name,"rows":int(a.shape[0]),"zero_rows":zero,
         "single_owner_rows":int(np.count_nonzero(own>=0)),
         "fixed_only_rows":int(np.count_nonzero((own<0)&(np.diff(a.indptr)>0)))}

def one_hot_map(mat,nrows,labels):
 m=csr(mat,"output_map")
 if m.shape[0]!=3*nrows or m.shape[1]%3:raise ValueError("output map not node-major xyz")
 q=m[0::3,0::3].tocsr()
 diag=(m[1::3,1::3],m[2::3,2::3])
 off=(m[0::3,1::3],m[0::3,2::3],m[1::3,0::3],m[1::3,2::3],m[2::3,0::3],m[2::3,1::3])
 if any(x.nnz for x in off) or any((x-q).nnz for x in diag):raise ValueError("output map mixes xyz axes")
 rc=np.diff(q.indptr);cc=np.bincount(q.indices,minlength=q.shape[1])
 if np.any((rc!=0)&(rc!=1)) or np.any(cc!=1) or q.nnz!=q.shape[1] or not np.all(q.data==1):
  raise ValueError("adapter requires one-hot independent full-coordinate controls")
 labels=np.asarray(labels,dtype=np.int64)
 expected_rows=np.flatnonzero(labels>=0)
 if q.shape!=(nrows,len(expected_rows)):raise ValueError("one-hot output scope/control count mismatch")
 expected=sparse.csr_matrix((np.ones(len(expected_rows)),(expected_rows,np.arange(len(expected_rows)))),shape=q.shape)
 if (q-expected).nnz:raise ValueError("Q is not the 564 independent-node identity map")
 return q

def componentwise_line_search(current,current_delta,target_delta,support,groups,try_component):
 current=np.asarray(current);cur=np.asarray(current_delta,dtype=np.float64).reshape(-1,3).copy()
 target=np.asarray(target_delta,dtype=np.float64).reshape(-1,3);support=np.asarray(support,dtype=np.int64)
 if current.dtype!=np.float32 or cur.shape!=target.shape or cur.shape!=(len(support),3):raise ValueError("line search domain mismatch")
 pos={int(n):i for i,n in enumerate(support)};events=[];accepted=0
 for ci,nodes in enumerate(groups):
  nodes=np.asarray(nodes,dtype=np.int64);ix=np.asarray([pos[int(n)] for n in nodes],dtype=np.int64)
  proposal=cur.copy();proposal[ix]=target[ix]
  if np.array_equal(proposal[ix],cur[ix]):
   events.append({"component":ci,"nodes":len(nodes),"accepted":False,"selected_alpha":None,"reason":"NO_COMPONENT_DELTA","trials":[]});continue
  cand,delta,rec=try_component(current,proposal.ravel())
  if cand is None or delta is None:
   events.append({"component":ci,"nodes":len(nodes),"accepted":False,"selected_alpha":None,"reason":rec.get("status","gate_failed"),"trials":rec.get("trials",[])});continue
  cand=np.asarray(cand);delta=np.asarray(delta,dtype=np.float64).reshape(-1,3)
  outside=np.ones(len(support),bool);outside[ix]=False
  if cand.dtype!=np.float32 or cand.shape!=current.shape or delta.shape!=cur.shape:raise ValueError("408 returned malformed trial")
  if not np.array_equal(cand[support[outside]].view(np.uint32),current[support[outside]].view(np.uint32)):raise ValueError("408 trial moved nodes outside its component")
  current=cand;cur=delta.copy();accepted+=1
  events.append({"component":ci,"nodes":len(nodes),"accepted":True,"selected_alpha":rec.get("accepted_alpha"),"reason":"accepted","trials":rec.get("trials",[])})
 return current,cur.ravel(),events,accepted

def _load549():
 if sha(P549)!=H549:raise ValueError("549 source pin mismatch")
 s=importlib.util.spec_from_file_location("component_area_backtracking549_for603",P549)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def install(owner,self_pairs_path):
 pair_path=Path(self_pairs_path).resolve()
 if not pair_path.is_file():raise FileNotFoundError(pair_path)
 pairs=np.load(pair_path,allow_pickle=False)
 if pairs.dtype.kind not in "iu" or pairs.ndim!=2 or pairs.shape[1]!=2:raise ValueError("bad self-pair schema")
 part=_load549();solve0=owner.solve_sparse_l1_clearance_in_basis;back0=owner.backtrack_area_feasible_scp_delta
 state={"last":None}
 def capture(*a,**kw):
  if len(a)<9:raise ValueError("basis solver omitted matrices")
  A,b,t,lo,hi,C,h,w,M=a[:9];r=solve0(*a,**kw)
  if r.get("success"):
   state["last"]={"A":csr(A,"A"),"b":np.asarray(b,float).copy(),"C":csr(C,"C"),
    "h":np.asarray(h,float).copy(),"w":np.asarray(w,float).copy(),"M":csr(M,"M"),"solver":r}
  return r
 owner.solve_sparse_l1_clearance_in_basis=capture
 def back(*a,**kw):
  if len(a)<18 or state["last"] is None:return back0(*a,**kw)
  base,current,xmm,refmm,faces,support,anchors,prefix,target,lower,upper,C,source,trust,cap,floor,EG,egb=a[:18]
  rec=state["last"];support=np.asarray(support,dtype=np.int64);base=np.asarray(base);current=np.asarray(current);target=np.asarray(target,float).reshape(-1)
  if target.shape!=(3*len(support),) or rec["A"].shape[1]!=target.size:return back0(*a,**kw)
  q=np.asarray(rec["solver"].get("basis_coordinates"),dtype=float).reshape(-1)
  if not np.allclose(rec["M"]@q,target,rtol=0,atol=2e-9):raise ValueError("Qq does not reconstruct LP delta")
  try:
   anchors=np.asarray(anchors,dtype=np.int64);free=np.setdiff1d(support,np.intersect1d(support,anchors))
   groups,proof=part.components(len(base),faces,free,pairs)
   labels=np.full(len(base),-1,dtype=np.int64)
   for ci,g in enumerate(groups):labels[np.asarray(g,dtype=np.int64)]=ci
   sl=labels[support]
   if np.any((sl<0)&~np.isin(support,anchors)):raise ValueError("support node lacks component")
   one_hot_map(rec["M"],len(support),sl)
   audits=[row_owner_audit(rec["A"],sl,"A_hard"),row_owner_audit(rec["C"],sl,"C_soft"),row_owner_audit(EG,sl,"edge_gradient")]
   sizes=rec["solver"].get("component_decomposition_475",{}).get("component_control_counts")
   if sizes is not None and [3*len(g) for g in groups]!=list(sizes):raise ValueError("549 stencils and 485 LP components differ")
   curdelta=((current[support].astype(float)-base[support].astype(float))*1000).ravel()
   now=np.asarray(rec["A"]@curdelta).ravel()-rec["b"];end=np.asarray(rec["A"]@target).ravel()-rec["b"]
   if min(np.min(now,initial=np.inf),np.min(end,initial=np.inf)) < -2e-7:raise ValueError("current or raw endpoint fails original hard rows")
  except Exception as e:
   p,d,r=back0(*a,**kw);r["component_scp_backtracking603"]={"status":"GLOBAL_408_FALLBACK_PREFLIGHT_REJECTED","reason":str(e),"native_admission":False};return p,d,r
  points,delta,events,naccepted=componentwise_line_search(current,curdelta,target,support,groups,
   lambda cur,goal:back0(base,cur,xmm,refmm,faces,support,anchors,prefix,goal,lower,upper,C,source,trust,cap,floor,EG,egb,**kw))
  if naccepted==0:
   p,d,r=back0(*a,**kw);r["component_scp_backtracking603"]={"status":"GLOBAL_408_FALLBACK_NO_COMPONENT_STEP","components":events,"native_admission":False};return p,d,r
  df=np.asarray(delta,float).ravel();hard=np.asarray(rec["A"]@df).ravel()-rec["b"]
  progress=np.asarray(rec["C"]@df).ravel();slack=np.maximum(0,(rec["h"]-progress)/rec["h"]);solver=rec["solver"]
  return points,delta,{"status":"ACCEPTED_COMPONENTWISE","accepted":True,"accepted_alpha":None,
   "component_scp_backtracking603":{"status":"ACCEPTED_COMPONENTWISE","components":events,"stencil_proof":proof,
    "component_node_counts":[int(len(g)) for g in groups],"matrix_row_owner_audits":audits,
    "minimum_hard_residual_before":float(np.min(now,initial=np.inf)),"minimum_hard_residual_raw_lp":float(np.min(end,initial=np.inf)),
    "minimum_emitted_f32_hard_residual_diagnostic":float(np.min(hard,initial=np.inf)),
    "post_projection_weighted_slack_diagnostic":float(np.dot(rec["w"],slack)),
    "solver_stage2_slack_budget":(solver.get("stage2") or {}).get("slack_budget"),
    "post_projection_lp_optimality_claim":False,"native_admission":False},
   "semantics":"per-component original 408 F32 gate; no single global alpha; final exact 545 remains mandatory"}
 owner.backtrack_area_feasible_scp_delta=back
 return {"status":"INSTALLED","component_549_path":str(P549),"component_549_sha256":H549,
  "self_pairs_path":str(pair_path),"self_pairs_sha256":sha(pair_path),
  "A_C_EG_rows_single_owner_required":True,"frozen_408_gate_per_component":True,"native_admission":False}

