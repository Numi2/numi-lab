from pathlib import Path
from collections import defaultdict
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-032-self16-joint-correction-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
CAND=D/"candidate-segment-partial-alpha-0p0625.npz"
BLAYER=W/"combined-d071-b109-b61-001/skin-fat-layers-541.npz"
TRANSFER=W/"b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
OWNER=W/"physical-layer-diagnostic-001/transfer_current.py"
FITTER=D/"solve_self16_joint.py"
BOUND_MM=.05
TRUST_MM=.05
TARGET_SKIN={(34588,0),(34708,1),(34741,1)}
TARGET_FAT={35427}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def flags_summary(new,old):
 out={}
 for k in ("authored_bad","nonpositive","conditioned"):
  a=np.asarray(old[k]);b=np.asarray(new[k]);added=b&~a;removed=a&~b
  out[k]={"base_count":int(a.sum()),"trial_count":int(b.sum()),"added":int(added.sum()),"removed":int(removed.sum())}
  if b.ndim==1:
   out[k]["added_face_rows"]=np.flatnonzero(added).astype(int).tolist();out[k]["removed_face_rows"]=np.flatnonzero(removed).astype(int).tolist()
  else:
   out[k]["added_face_tets"]=[[int(f),int(t)] for f,t in np.argwhere(added)]
   out[k]["removed_face_tets"]=[[int(f),int(t)] for f,t in np.argwhere(removed)]
 return out
def normals(O,LF):
 N=np.zeros_like(O)
 tri=O[LF];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 for j in range(3):np.add.at(N,LF[:,j],n)
 nn=np.linalg.norm(N,axis=1);valid=nn>1e-20;N[valid]/=nn[valid,None]
 return N
def volgrad(q):
 q0,q1,q2,q3=q
 g1=np.cross(q2-q0,q3-q0)/6.0
 g2=np.cross(q3-q0,q1-q0)/6.0
 g3=np.cross(q1-q0,q2-q0)/6.0
 return float(np.dot(np.cross(q1-q0,q2-q0),q3-q0)/6.0),np.asarray([-g1-g2-g3,g1,g2,g3])
def main():
 t0=time.monotonic()
 with np.load(BASE,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 with np.load(CAND,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 with np.load(BLAYER,allow_pickle=False) as z:l={k:z[k].copy() for k in z.files}
 X=np.asarray(c["points_world_m"],np.float32).astype(np.float64)*1000.0
 F=np.asarray(c["source_O_faces"],np.int64);LF=F[:,[0,2,1]]
 if not np.array_equal(F,np.asarray(b["source_O_faces"],np.int64)):raise ValueError("candidate changed outer topology")
 O=X
 baseO=np.asarray(l["outer_points_f64"],np.float64)*1000.0
 baseI=np.asarray(l["inner_points_f64"],np.float64)*1000.0
 baseD=np.asarray(l["deep_points_f64"],np.float64)*1000.0
 skin=np.asarray(l["skin_thickness_transferred_m"],np.float64)*1000.0
 fat=np.asarray(l["fat_thickness_transferred_m"],np.float64)*1000.0
 Icur=O+(baseI-baseO);Dcur=Icur+(baseD-baseI)
 transfer=mod("layer_transfer_032",TRANSFER);owner=mod("layer_owner_032",OWNER)
 baseS=transfer.face_flags(baseO,baseI,LF,l["skin_thickness_transferred_m"])
 baseF=transfer.face_flags(baseI,baseD,LF,l["fat_thickness_transferred_m"])
 curS=transfer.face_flags(O,Icur,LF,l["skin_thickness_transferred_m"])
 curF=transfer.face_flags(Icur,Dcur,LF,l["fat_thickness_transferred_m"])
 fitter=mod("layer_fitter_032",FITTER);seg=json.loads((D/"exact-segment-boundary-preflight.json").read_text())
 inside=[]
 with (W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv").open(newline="") as fh:
  for r in csv.DictReader(fh):
   if (r["bank_body"],r["bank_stable_id"],r["classification"])==("109","114","inside"):inside.append(int(r["query_row"]))
 sp=json.loads((D/"self16-pairs.json").read_text());selfpairs=np.asarray([r["skin_face_rows"] for r in sp["pair_ids"]],np.int64)
 patch,comps,boundary,support,edges=fitter.make_patch(F,seg,np.asarray(inside,np.int64),selfpairs)
 target_faces=[34588,34708,34741,35427]
 seeds=set(map(int,np.unique(LF[np.asarray(target_faces,np.int64)])))
 edge_adj=defaultdict(set)
 for tri in LF:
  for a,bn in zip(tri,np.roll(tri,-1)):edge_adj[int(a)].add(int(bn));edge_adj[int(bn)].add(int(a))
 ring=set(seeds)
 for n in list(seeds):ring.update(edge_adj[n])
 support=set(map(int,support));boundary=set(map(int,boundary))
 var=sorted((ring&support)-boundary);vmap={n:i for i,n in enumerate(var)}
 if any(not set(LF[f].tolist())<=support for f in target_faces):raise ValueError("target layer face outside exact patch support")
 nvar=len(var);nd=6*nvar
 if not nvar:raise ValueError("no movable layer node")
 O32=O.astype(np.float32).astype(np.float64);I32=Icur.astype(np.float32).astype(np.float64);D32=Dcur.astype(np.float32).astype(np.float64)
 # Every face incident to an adjustable I or D node is included in the cell guard.
 affected=np.flatnonzero(np.any(np.isin(LF,np.asarray(var,np.int64)),axis=1))
 N=normals(O,LF)
 constraints=[];lowers=[];labels=[];types=[]
 def addrow(items,lower,label,kind,scale=1.0):
  row=defaultdict(float)
  for col,val in items:
   if abs(float(val))>0:row[int(col)]+=float(val)
  row={k:v/scale for k,v in row.items() if abs(v)>1e-18}
  lower=float(lower)/scale
  constraints.append(row);lowers.append(lower);labels.append(label);types.append(kind)
 # Per-node thickness constraints keep the carried O/I/D spacing within 0.05 mm.
 for n in var:
  i=vmap[n];nn=N[n];sd=float(np.dot(O[n]-Icur[n],nn));fd=float(np.dot(Icur[n]-Dcur[n],nn))
  if sd<=0 or fd<=0:raise ValueError(f"carried layer has nonpositive projected thickness at node {n}: {sd}/{fd}")
  # -band <= n dot uI <= band
  addrow([(3*i+k,nn[k]) for k in range(3)],-BOUND_MM,f"skin-thickness-low:{n}","thickness")
  addrow([(3*i+k,-nn[k]) for k in range(3)],-BOUND_MM,f"skin-thickness-high:{n}","thickness")
  # -band <= n dot (uI-uD) <= band
  items=[(3*i+k,nn[k]) for k in range(3)]+[(3*nvar+3*i+k,-nn[k]) for k in range(3)]
  addrow(items,-BOUND_MM,f"fat-thickness-low:{n}","thickness")
  addrow([(col,-val) for col,val in items],-BOUND_MM,f"fat-thickness-high:{n}","thickness")
 # Current affected cell guards and one linearized repair for each exact flagged cell.
 def add_layer_cells(layer,A0,B0,curflags,baseflags,field_mm):
  for fi in affected:
   fi=int(fi);src=LF[fi];order=np.argsort(src);a,bn,cc=map(int,src[order])
   inv=int((src[0]>src[1])+(src[0]>src[2])+(src[1]>src[2]));exp=1.0 if inv%2 else -1.0
   sig=[exp,-exp,exp]
   node_tets=[[(a,"A"),(bn,"A"),(cc,"A"),(cc,"B")],
              [(a,"A"),(bn,"A"),(bn,"B"),(cc,"B")],
              [(a,"A"),(a,"B"),(bn,"B"),(cc,"B")]]
   qcur=[]
   for tet in node_tets:
    qcur.append(np.asarray([(A0[n] if kind=="A" else B0[n]) for n,kind in tet],np.float64))
   maxfield=float(np.max(field_mm[LF[fi]],initial=0.0))
   tol=max(1e-5,maxfield)**3*1e-12
   face_bad=bool(curflags["authored_bad"][fi])
   for ti,(tet,pts) in enumerate(zip(node_tets,qcur)):
    v,g=volgrad(pts);sv=float(sig[ti]*v)
    # Reproduce the helper's emitted-F32 longest-edge scale for this cell.
    q32=np.asarray([(A0[n] if kind=="A" else B0[n]) for n,kind in tet],np.float32).astype(np.float64)
    longest=max(float(np.linalg.norm(q32[i]-q32[j])) for i in range(4) for j in range(i+1,4))
    threshold=1.2e-5*longest**3/6.0
    targeted=(layer=="skin" and (fi,ti) in TARGET_SKIN) or (layer=="fat" and fi in TARGET_FAT)
    is_cond=bool(curflags["conditioned"][fi,ti])
    is_nonpos=bool(curflags["nonpositive"][fi,ti])
    if targeted:
     lower=max(tol,threshold)
    elif face_bad or is_cond or is_nonpos:
     lower=sv
    else:
     # New failures are forbidden for every currently passing affected cell.
     lower=max(tol,threshold)
    items=[]
    for (node,kind),grad in zip(tet,g):
     i=vmap.get(int(node))
     if i is None or np.array_equal(grad,np.zeros(3)):continue
     offset=0 if (layer=="skin" or kind=="A") else 3*nvar
     # Skin A is fixed O; skin B is I. Fat A is I, B is D.
     if layer=="skin" and kind=="A":continue
     for k in range(3):items.append((offset+3*i+k,sig[ti]*grad[k]))
    scale=max(float(np.linalg.norm([v for _,v in items])),1e-8)
    addrow(items,lower-sv,f"{layer}-volume:{fi}:{ti}","cell-volume",scale)
 # Skin uses O/I; fat uses I/D. The DOF map for A/B is dispatched above.
 add_layer_cells("skin",O,Icur,curS,baseS,skin)
 add_layer_cells("fat",Icur,Dcur,curF,baseF,fat)
 # Per-coordinate trust and smooth quadratic graph energy.
 rr=[];cc=[];vv=[]
 for r,row in enumerate(constraints):
  for col,val in row.items():rr.append(r);cc.append(col);vv.append(val)
 Acon=sparse.coo_matrix((vv,(rr,cc)),shape=(len(constraints),nd)).tocsc()
 lo=np.asarray(lowers,np.float64);hi=np.full(len(lo),np.inf)
 coord_lo=np.full(nd,-TRUST_MM);coord_hi=np.full(nd,TRUST_MM)
 # Restrict the local relaxation to a small share of original transferred depth.
 # The normal thickness rows above independently constrain I/D separation.
 L=sparse.lil_matrix((nvar,nvar),dtype=np.float64)
 for u,v in edges:
  u=int(u);v=int(v)
  if u in vmap and v in vmap:
   iu,iv=vmap[u],vmap[v]
   if iu==iv:continue
   L[iu,iu]+=1;L[iv,iv]+=1;L[iu,iv]-=1;L[iv,iu]-=1
 L=L.tocsr();L3=sparse.kron(L,sparse.eye(3),format="csc")
 P=(2*sparse.eye(nd,format="csc")+0.25*sparse.block_diag((L3,L3),format="csc")).tocsc()
 Aall=sparse.vstack([Acon,sparse.eye(nd,format="csc")],format="csc")
 lall=np.r_[lo,coord_lo];uall=np.r_[hi,coord_hi]
 sys.path.insert(0,str(TRANSFER.parent));import osqp
 opt=osqp.OSQP();opt.setup(P=P,q=np.zeros(nd),A=Aall,l=lall,u=uall,verbose=False,eps_abs=1e-9,eps_rel=1e-8,max_iter=200000,polishing=True,adaptive_rho=True,scaling=30,scaled_termination=True)
 sol=opt.solve();ok=int(sol.info.status_val)==1
 report={"schema":"numi.human.b109-114.offset-carry-targeted-layer-only-correction.v1","candidate_sha256":sha(CAND),"baseline_candidate_sha256":sha(BASE),"baseline_layer_sha256":sha(BLAYER),"candidate_outer_kept_fixed":True,"target_skin_cells":[list(x) for x in sorted(TARGET_SKIN)],"target_fat_face_rows":sorted(TARGET_FAT),"target_vertex_rows":sorted(seeds),"variable_node_rows":var,"variable_node_count":nvar,"fixed_boundary_count":len(boundary),"support_node_count":len(support),"affected_face_count":len(affected),"thickness_band_mm":BOUND_MM,"coordinate_trust_mm":TRUST_MM,"constraint_count":len(constraints),"constraint_kinds":{k:types.count(k) for k in sorted(set(types))},"solver":{"status":str(sol.info.status),"status_val":int(sol.info.status_val),"iterations":int(sol.info.iter),"primal_residual":float(sol.info.prim_res),"dual_residual":float(sol.info.dual_res),"success":bool(ok)},"geometry_admitted":False,"physical_steps":0}
 if ok:
  u=np.asarray(sol.x,np.float64).reshape(2,nvar,3)
  Itrial=Icur.copy();Dtrial=Dcur.copy()
  for n,i in vmap.items():Itrial[n]+=u[0,i];Dtrial[n]+=u[1,i]
  # Emit layer geometry at F64 and F32, preserving every typed/material field.
  newS=transfer.face_flags(O,Itrial,LF,l["skin_thickness_transferred_m"])
  newF=transfer.face_flags(Itrial,Dtrial,LF,l["fat_thickness_transferred_m"])
  oldFlags={"skin":baseS,"fat":baseF};carriedFlags={"skin":curS,"fat":curF}
  newFlags={"skin":newS,"fat":newF}
  summaries={layer:{"vs_baseline":flags_summary(newFlags[layer],oldFlags[layer]),"vs_offset_carry":flags_summary(newFlags[layer],carriedFlags[layer])} for layer in ("skin","fat")}
  changed_faces=np.flatnonzero(np.any(np.isin(LF,np.asarray(var,np.int64)),axis=1))
  maxd=float(np.max(np.linalg.norm(u,axis=2),initial=0))
  report.update(max_layer_node_displacement_mm:maxd,changed_face_count:int(len(changed_faces)),changed_face_rows:changed_faces.astype(int).tolist(),cell_flags:summaries,target_cells={})
  for layer,faces in (("skin",sorted(set(f for f,t in TARGET_SKIN))),("fat",sorted(TARGET_FAT))):
   for fi in faces:
    flags=newFlags[layer];report["target_cells"][f"{layer}:{fi}"]={"authored_bad":bool(flags["authored_bad"][fi]),"nonpositive":flags["nonpositive"][fi].tolist(),"conditioned":flags["conditioned"][fi].tolist(),"normalized_det":flags["normalized_det"][fi].tolist(),"volumes_mm3":flags["volumes"][fi].tolist()}
  no_added=all(not summaries[layer]["vs_baseline"][flag]["added"] for layer in ("skin","fat") for flag in ("authored_bad","nonpositive","conditioned"))
  no_worse_existing=True;worse=[]
  for layer in ("skin","fat"):
   prev=carriedFlags[layer]["normalized_det"];now=newFlags[layer]["normalized_det"]
   for fi in affected:
    for ti in range(3):
     if (carriedFlags[layer]["conditioned"][fi,ti] or carriedFlags[layer]["nonpositive"][fi,ti]) and float(now[fi,ti])+1e-12<float(prev[fi,ti]):
      no_worse_existing=False;worse.append([layer,int(fi),int(ti),float(prev[fi,ti]),float(now[fi,ti)])
  target_ok=not bool(newF["authored_bad"][35427]) and all(not bool(newS[f,t]) for f,t in TARGET_SKIN) and all(not bool(newS["nonpositive"][f,t]) and not bool(newS["conditioned"][f,t]) for f,t in TARGET_SKIN)
  report.update(no_new_invalid_vs_baseline=no_added,no_worsened_existing=no_worse_existing,worsened_existing_conditioned_or_nonpositive_cells=worse,target_cell_gate=target_ok,all_layer_cell_gates_pass=bool(no_added and no_worse_existing and target_ok))
  if report["all_layer_cell_gates_pass"]:
   output={k:v.copy() for k,v in l.items()}
   output["outer_points_f64"]=O/1000.0;output["inner_points_f64"]=Itrial/1000.0;output["deep_points_f64"]=Dtrial/1000.0
   output["outer_points_f32"]=(O/1000.).astype(np.float32);output["inner_points_f32"]=(Itrial/1000.).astype(np.float32);output["deep_points_f32"]=(Dtrial/1000.).astype(np.float32)
   nrm=normals(O,LF)
   output["skin_thickness_actual_norm_m"]=np.einsum("ij,ij->i",O-Itrial,nrm)/1000.0
   output["fat_thickness_actual_norm_m"]=np.einsum("ij,ij->i",Itrial-Dtrial,nrm)/1000.0
   layerout=D/"skin-fat-layers-offset-carry-corrected.npz"
   with layerout.open("xb") as f:np.savez_compressed(f,**output)
   report["layer_candidate_path"]=str(layerout);report["layer_candidate_sha256"]=sha(layerout)
 report["status"]="LAYER_CORRECTION_PASS" if report.get("all_layer_cell_gates_pass") else ("SOLVED_LAYER_GATE_FAIL" if ok else "QP_INFEASIBLE")
 report["elapsed_s"]=time.monotonic()-t0
 rp=D/"offset-carry-layer-correction-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"report":str(rp),"sha256":sha(rp),"solver":report["solver"],"variable_nodes":nvar,"affected_faces":len(affected),"constraints":len(constraints),"max_layer_motion_mm":report.get("max_layer_node_displacement_mm"),"new_invalid":report.get("no_new_invalid_vs_baseline"),"no_worse":report.get("no_worsened_existing"),"target_cells":report.get("target_cells"),"added_flags":report.get("cell_flags")},indent=2))
if __name__=="__main__":main()

