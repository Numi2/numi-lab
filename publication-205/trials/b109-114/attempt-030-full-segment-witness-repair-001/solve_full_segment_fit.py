#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict,deque
from fractions import Fraction
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-030-full-segment-witness-repair-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
LAYER=W/"combined-d071-b109-b61-001/skin-fat-layers-541.npz"
SEG=D/"exact-segment-boundary-preflight.json"
SELF_PRE=D/"alpha1-self-witness-barrier-preflight.json"
AUD130=W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"
AUD159=W/"combined-d071-b109-b61-001/audit-159/report.json"
AUDIT=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
LAYER_OWNER=S/"per-bank-workers-20261027/b65-99/root-layer-guard-002/fit.py"
SELF_OWNER=ROOT/"D/publication-198-stage-001/scripts/owners/self_separation_026.py"
QPATH=ROOT/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
SOURCE=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
OSQP_DEPS=Path("/Users/numi/numi-human-fds132-m6-1350/attempt-311-posed-area-sqp-003/osqp-deps")
BASE_SHA="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
LAYER_SHA="eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57"
BODY=109;BONE=114
INSIDE_MARGIN_MM=.10
SEGMENT_MARGIN_MM=.05
TRUST_MM=3.0
SOURCE_CAP_MM=12.0
EDGE_GRAD=.20
BACKTRACKS=[1.0,.5,.25,.125,.0625]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def face_metrics(points,faces):
 tri=np.asarray(points,dtype=np.float64)[np.asarray(faces,dtype=np.int64)]
 e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];e3=tri[:,2]-tri[:,1];n=np.cross(e1,e2);twice=np.linalg.norm(n,axis=1)
 lens=np.stack((np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1),np.linalg.norm(e3,axis=1)),axis=1)
 angles=[]
 for i in range(3):
  u=tri[:,(i+1)%3]-tri[:,i];v=tri[:,(i+2)%3]-tri[:,i]
  c=np.einsum("ij,ij->i",u,v)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-300)
  angles.append(np.degrees(np.arccos(np.clip(c,-1,1))))
 return {"area":.5*twice,"angle":np.min(np.stack(angles,axis=1),axis=1),"aspect":np.max(lens,axis=1)**2/(2*np.sqrt(3)*np.maximum(twice,1e-300)),"normal":n}
def shape_gate(base,trial,F,rows):
 a=face_metrics(base,F[rows]);b=face_metrics(trial,F[rows])
 ar=b["area"]/np.maximum(a["area"],1e-300);an=b["angle"]/np.maximum(a["angle"],1e-300);asp=b["aspect"]/np.maximum(a["aspect"],1e-300)
 dot=np.einsum("ij,ij->i",a["normal"],b["normal"])/np.maximum(np.linalg.norm(a["normal"],axis=1)*np.linalg.norm(b["normal"],axis=1),1e-300)
 bad=(ar<.95)|(an<.95)|(asp>1.05)|(dot<np.cos(np.deg2rad(18)))
 return {"pass":not bool(bad.any()),"affected_face_count":len(rows),"violating_face_count":int(bad.sum()),"violating_face_rows":rows[bad].astype(int).tolist(),"min_area_ratio":float(ar.min(initial=np.inf)),"min_min_angle_ratio":float(an.min(initial=np.inf)),"max_aspect_ratio":float(asp.max(initial=-np.inf)),"min_normal_dot":float(dot.min(initial=np.inf)),"max_normal_turn_deg":float(np.degrees(np.arccos(np.clip(dot,-1,1))).max(initial=0.0))}
def make_patch(F,seg,inside_nodes,self_pairs):
 target_faces={int(r["skin_face_row"]) for r in seg["target_pairs"]}
 seed=set(target_faces)
 inside_set=set(map(int,inside_nodes))
 for fi,t in enumerate(F):
  if any(int(n) in inside_set for n in t):seed.add(fi)
 for fa,fb in self_pairs:seed.update((int(fa),int(fb)))
 e2f=defaultdict(list)
 for fi,t in enumerate(F):
  for j in range(3):e2f[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append(fi)
 adj=[set() for _ in range(len(F))]
 for uses in e2f.values():
  for f in uses:adj[f].update(g for g in uses if g!=f)
 patch=set(seed)
 for _ in range(3):patch|=set().union(*(adj[f] for f in patch))
 rem=set(patch);comps=[]
 while rem:
  r=min(rem);rem.remove(r);q=deque([r]);c=[]
  while q:
   f=q.popleft();c.append(f)
   for g in adj[f]&rem:rem.remove(g);q.append(g)
  comps.append(sorted(c))
 comps=sorted(comps,key=lambda c:(-len(c),c[0]))
 edge_count=defaultdict(int)
 for fi in patch:
  t=F[fi]
  for j in range(3):edge_count[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))]+=1
 bedges=[e for e,k in edge_count.items() if k==1]
 bg=defaultdict(set)
 for u,v in bedges:bg[u].add(v);bg[v].add(u)
 bnodes=sorted(bg)
 if any(len(v)!=2 for v in bg.values()):raise ValueError("ring-3 patch boundary is not a simple degree-two loop set")
 return np.asarray(sorted(patch),np.int64),comps,np.asarray(bnodes,np.int64),np.asarray(sorted({n for fi in patch for n in F[fi]}),np.int64),np.asarray(sorted({tuple(sorted((int(t[j]),int(t[(j+1)%3])))) for fi in patch for t in [F[fi]] for j in range(3)}),np.int64).reshape(-1,2)
def bone_orientation(BT):
 def key(v):return tuple(np.asarray(v,dtype=np.float64).view(np.uint64).tolist())
 verts={};ti=np.empty((len(BT),3),np.int64)
 for i,t in enumerate(BT):
  for j,v in enumerate(t):ti[i,j]=verts.setdefault(key(v),len(verts))
 uses=defaultdict(list)
 for i,t in enumerate(ti):
  for j in range(3):
   a,b=int(t[j]),int(t[(j+1)%3]);k=tuple(sorted((a,b)));uses[k].append((i,1 if (a,b)==k else -1))
 if any(len(v)!=2 for v in uses.values()):raise ValueError("target bone geometry is not a closed two-manifold")
 adj=[[] for _ in BT]
 for items in uses.values():
  (i,si),(j,sj)=items;rel=-si*sj;adj[i].append((j,rel));adj[j].append((i,rel))
 orient=np.zeros(len(BT),np.int8);components=[]
 for root in range(len(BT)):
  if orient[root]:continue
  orient[root]=1;q=deque([root]);comp=[]
  while q:
   i=q.popleft();comp.append(i)
   for j,rel in adj[i]:
    want=orient[i]*rel
    if not orient[j]:orient[j]=want;q.append(j)
    elif orient[j]!=want:raise ValueError("bone orientation conflict")
  components.append(comp)
 if len(components)!=1:raise ValueError("target bone shell has multiple components")
 vol=0.
 c=BT.mean(axis=(0,1))
 for i in components[0]:
  t=BT[i]*int(orient[i])-c;vol+=float(np.dot(t[0],np.cross(t[1],t[2]))/6)
 if vol<0:orient*=-1
 return orient
def filter_pair_rows(rows,locidx,movable,zero_receipt,kind):
 kept=[]
 for row in rows:
  vec=np.zeros(len(movable),np.float64)
  for node,coef in zip(row["nodes"],row["coefficients"]):
   li=locidx.get(int(node))
   if li is None:raise ValueError(f"{kind} row references node outside support: {row.get('label')}")
   vec[3*li:3*li+3]+=np.asarray(coef,dtype=np.float64)
  norm=float(np.linalg.norm(vec*movable));slack=-float(row["lower"])
  if not np.isfinite(norm) or not np.isfinite(slack):raise ValueError(f"nonfinite {kind} row")
  if norm<=1e-14:
   rec={"kind":kind,"label":row.get("label"),"movable_norm":norm,"lower":float(row["lower"]),"zero_motion_slack":slack}
   zero_receipt["rows"].append(rec)
   if slack < -1e-12:
    zero_receipt["status"]="INFEASIBLE_ZERO_MOVABLE_SUPPORT";zero_receipt["unsatisfied"]=rec
    (D/"zero-movable-preflight.json").write_text(json.dumps(zero_receipt,indent=2,sort_keys=True)+"\n")
    raise ValueError(f"{kind} row has no movable support and is unsatisfied: {rec}")
  else:kept.append(row)
 return kept
def filter_matrix(G,rhs,movable,indices,zero_receipt,kind,face_rows=None):
 local=G[:,indices].tocsr();eff=local.multiply(movable.reshape(1,-1));norms=np.sqrt(np.asarray(eff.multiply(eff).sum(axis=1)).ravel());keep=[]
 for i,n in enumerate(norms):
  if not np.isfinite(n):raise ValueError(f"nonfinite {kind} row")
  if n<=1e-14:
   slack=-float(rhs[i]);rec={"kind":kind,"row_index":int(i),"face_row":None if face_rows is None else int(face_rows[i]),"movable_norm":float(n),"lower":float(rhs[i]),"zero_motion_slack":slack}
   zero_receipt["rows"].append(rec)
   if slack < -1e-12:
    zero_receipt["status"]="INFEASIBLE_ZERO_MOVABLE_SUPPORT";zero_receipt["unsatisfied"]=rec
    (D/"zero-movable-preflight.json").write_text(json.dumps(zero_receipt,indent=2,sort_keys=True)+"\n")
    raise ValueError(f"{kind} row has no movable support and is unsatisfied: {rec}")
  else:keep.append(i)
 return G[keep].tocsr(),np.asarray(rhs)[keep],len(keep)
def main():
 t0=time.monotonic()
 for p,h in ((BASE,BASE_SHA),(LAYER,LAYER_SHA)):
  if sha(p)!=h:raise ValueError("pinned candidate/layer mismatch "+str(p))
 seg=json.loads(SEG.read_text());sp=json.loads(SELF_PRE.read_text())
 if sha(SEG)!="238cee2dcd394379fb9e01bff244840da913cf85434ca7b15be9294f9bcc745d":raise ValueError("segment preflight pin mismatch")
 if sha(SELF_PRE)!="805626acb0d09769b951b54fc7d7ebc364cacb198c6949144951dcb4a7a60fed":raise ValueError("self barrier preflight pin mismatch")
 with np.load(BASE,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
 x=np.asarray(payload["points_world_m"],np.float32);F=np.asarray(payload["source_O_faces"],np.int64);bm=x.astype(np.float64)*1000.
 if len(x)!=21212 or len(F)!=42379:raise ValueError("pinned mesh shape changed")
 helper=mod("fit_audit_helper_030",HELPER);audit=mod("fit_audit_owner_030",AUDIT);exact=audit.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 mask=(ids[:,0]==BODY)&(ids[:,1]==BONE);BT=np.asarray(T[mask],np.float64)*1000.;stable=np.asarray(ids[mask,3],np.int64)
 if len(BT)!=392:raise ValueError("unexpected target bank face count")
 orient=bone_orientation(BT)
 local=((x.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY])*1000.
 selfpairs=[tuple(map(int,r["skin_face_rows"])) for r in sp["pair_ids"]]
 inside_rows=[]
 with AUD130.open(newline="") as f:
  for r in csv.DictReader(f):
   if (r["bank_body"],r["bank_stable_id"],r["classification"])==("109","114","inside"):inside_rows.append(r)
 inside_nodes=np.asarray([int(r["query_row"]) for r in inside_rows],np.int64)
 patch_faces,components,boundary_nodes,support,patch_edges=make_patch(F,seg,inside_nodes,selfpairs)
 if sorted(map(len,components))!=[47,211] or len(patch_faces)!=258 or len(support)!=156 or len(boundary_nodes)!=50:raise ValueError("ring-3 patch no longer matches certified topology preflight")
 # Validate the fixed ring-3 boundary against the saved exact target-bone location receipt.
 loc_checks=seg["boundary_location_checks"][0]
 if int(loc_checks["radius"])!=3 or not all(c["boundary_all_outside_target_bone"] for c in loc_checks["components"]):raise ValueError("ring-3 boundary is not certified outside the target bone")
 bcert=sorted(n for c in loc_checks["components"] for n in c["boundary_nodes"])
 if not np.array_equal(boundary_nodes,np.asarray(bcert,np.int64)):raise ValueError("boundary node IDs disagree with exact point-location receipt")
 # Exact segment endpoints provide barycentric rows on the skin child and outward normals from the closed bone.
 segment_rows=[];segment_meta=[]
 for rec in seg["target_pairs"]:
  fi=int(rec["skin_face_row"]);bi=int(rec["bone_triangle_row"]);nlocal=np.cross(BT[bi,1]-BT[bi,0],BT[bi,2]-BT[bi,0])*int(orient[bi]);nlocal/=np.linalg.norm(nlocal);nworld=M[BODY].T@nlocal;nworld/=np.linalg.norm(nworld)
  if len(rec["intersection_vertices"])!=2:raise ValueError(f"target pair {fi}/{bi} is not one exact segment")
  for ei,ep in enumerate(rec["intersection_vertices"]):
   bary=np.asarray(ep["skin_barycentric"],np.float64)
   if abs(float(bary.sum())-1)>1e-10 or bary.min()<-1e-8 or bary.max()>1+1e-8:raise ValueError("invalid endpoint barycentric")
   segment_rows.append({"nodes":F[fi].astype(int).tolist(),"coefficients":[(float(w)*nworld).tolist() for w in bary],"lower":SEGMENT_MARGIN_MM,"label":f"target-segment:{fi}:{bi}:endpoint:{ei}"})
   segment_meta.append({"skin_face_row":fi,"source_parent_face_row":int(rec["source_parent_face_row"]),"bone_triangle_row":bi,"endpoint_index":ei,"barycentric":bary.tolist(),"outward_normal_world":nworld.tolist()})
 # Point-depth rows use the 24 baseline nearest-boundary witnesses and target actual sampled depth.
 point_rows=[];point_meta=[]
 for r in inside_rows:
  node=int(r["query_row"]);q=np.asarray([float(r[f"query_local_{k}"]) for k in "xyz"],np.float64)*1000.;c=np.asarray([float(r[f"closest_{k}"]) for k in "xyz"],np.float64)*1000.;v=c-q;dist=float(np.linalg.norm(v));reported=float(r["nearest_distance_m"])*1000.
  if abs(dist-reported)>1e-5:raise ValueError(f"inside nearest-point distance mismatch node={node} exact={dist} reported={reported}")
  if dist<=0:raise ValueError("inside node has zero nearest distance")
  nlocal=v/dist;nworld=M[BODY].T@nlocal;nworld/=np.linalg.norm(nworld)
  bi=int(r["nearest_triangle"]);bt=BT[bi];vb=c-bt[0];e1=bt[1]-bt[0];e2=bt[2]-bt[0];d00=e1@e1;d01=e1@e2;d11=e2@e2;d20=vb@e1;d21=vb@e2;den=d00*d11-d01*d01;vv=(d11*d20-d01*d21)/den;ww=(d00*d21-d01*d20)/den;bb=np.asarray([1-vv-ww,vv,ww])
  if bb.min()<-2e-5 or bb.max()>1+2e-5:raise ValueError(f"nearest point outside declared target triangle node={node} bary={bb.tolist()}")
  point_rows.append({"nodes":[node],"coefficients":[nworld.tolist()],"lower":dist+INSIDE_MARGIN_MM,"label":f"inside-depth:{node}:triangle:{bi}"})
  point_meta.append({"node_row":node,"baseline_nearest_depth_mm":dist,"target_outward_progress_mm":dist+INSIDE_MARGIN_MM,"nearest_bone_triangle_row":bi,"nearest_triangle_stable_id":int(stable[bi]),"closest_point_barycentric":bb.tolist(),"outward_normal_local":nlocal.tolist(),"outward_normal_world":nworld.tolist()})
 # Preserve the four exact self-threat pairs already separated in the 806 baseline.
 sep=mod("self_separation_owner_030",SELF_OWNER)
 self_rows,self_records=sep.build_pair_rows(bm,F,np.asarray(selfpairs,np.int64),margin_fraction=.25)
 # The local movement support is exactly the two ring-3 disks; the 50 outside boundary nodes remain fixed.
 support=np.asarray(support,np.int64);locidx={int(n):i for i,n in enumerate(support)}
 movable=np.ones(3*len(support),bool)
 anchors=np.asarray(payload["typed_coarse_anchor_output_rows_018"],np.int64)
 q=mod("connected_fit_owner_030",QPATH);area=mod("area_fit_owner_030",q.DEFAULT_AREA_OWNER);q.verify_area_linearization_layout(area)
 with np.load(BASE,allow_pickle=False) as a:
  source_mm=q.compose_source_reference(a,SOURCE)[0]*1000.
 lo,hi,bound_receipt=q.source_safe_trust_bounds(bm,source_mm,support,anchors,TRUST_MM,SOURCE_CAP_MM)
 for node in boundary_nodes:
  li=locidx[int(node)];lo[3*li:3*li+3]=0.0;hi[3*li:3*li+3]=0.0
 movable=np.asarray(hi,dtype=np.float64)-np.asarray(lo,dtype=np.float64)>1e-15
 # Generate local area and normal-turn guards over precisely the two disks.
 affected=patch_faces
 normals,current_area,reference_area,required=q.fixed_reference_area_requirements(bm,np.asarray(payload["area_reference_037_points_world_m_f64"],np.float64)*1000.,F)
 G,grhs,_,_=area.area_linearization(bm,F,affected,normals[affected],current_area[affected],minimum_ratio=.95)
 normal_rows=[];angle=np.deg2rad(10.0)
 for fi in affected:
  a,b,c=map(int,F[int(fi)]);p0,p1,p2=bm[[a,b,c]];e1=p1-p0;e2=p2-p0;n=np.cross(e1,e2);area2=float(np.linalg.norm(n))
  if area2<=0:raise ValueError(f"degenerate baseline shape face {fi}")
  nh=n/area2;t1=e1/np.linalg.norm(e1);t2=np.cross(nh,t1);t2/=np.linalg.norm(t2);cap=np.tan(angle)*area2/np.sqrt(2)
  for t in (t1,t2):
   c0=-np.cross(e2,t)-np.cross(t,e1);c1=np.cross(e2,t);c2=np.cross(t,e1);nodes=[a,b,c];co=[c0,c1,c2]
   normal_rows.append({"nodes":nodes,"coefficients":[v.tolist() for v in co],"lower":-cap,"label":f"normal+:{int(fi)}"})
   normal_rows.append({"nodes":nodes,"coefficients":[(-v).tolist() for v in co],"lower":-cap,"label":f"normal-:{int(fi)}"})
 layer_owner=mod("layer_guard_owner_030",LAYER_OWNER)
 LG,lb,layer_receipt=layer_owner.make_layer_guard(LAYER,F,affected,support)
 zero_receipt={"schema":"numi.human.b109-114.full-segment-zero-support-preflight.v1","status":"PASS","tolerance":1e-14,"rows":[]}
 rows=filter_pair_rows(segment_rows+point_rows+self_rows+normal_rows,locidx,movable,zero_receipt,"witness/self/shape")
 G,grhs,area_count=filter_matrix(G,grhs,movable,q.interleaved_coordinate_columns(support),zero_receipt,"area",affected)
 LG,lb,layer_count=filter_matrix(LG,lb,movable,np.arange(3*len(support)),zero_receipt,"layer",affected[:LG.shape[0]] if LG.shape[0]==len(affected) else None)
 (D/"zero-movable-preflight.json").write_text(json.dumps(zero_receipt,indent=2,sort_keys=True)+"\n")
 A,rhs=q._owner_matrix(rows,G,grhs,support)
 EG,eb,edge_receipt=q.build_edge_gradient_constraints(bm,F,support,EDGE_GRAD)
 # Keep only edges in the two selected disks; no smoothing terms reach unrelated tissue.
 er=[];ec=[];ev=[];nr=0
 for u,v in patch_edges:
  iu=locidx.get(int(u));iv=locidx.get(int(v));weight=1/max(float(np.linalg.norm(bm[u]-bm[v])),.05)
  for k in range(3):
   if iu is not None:er.append(nr);ec.append(3*iu+k);ev.append(weight)
   if iv is not None:er.append(nr);ec.append(3*iv+k);ev.append(-weight)
   nr+=1
 E0=sparse.coo_matrix((ev,(er,ec)),shape=(nr,3*len(support))).tocsr()
 n=3*len(support);H=5.0*(E0.T@E0)+.02*sparse.eye(n,format="csr")
 I=sparse.eye(n,format="csr");K=sparse.vstack((A,LG,EG,I),format="csc")
 lower=np.r_[rhs,lb,eb,lo];upper=np.r_[np.full(A.shape[0],np.inf),np.full(LG.shape[0],np.inf),np.full(EG.shape[0],np.inf),hi]
 sys.path.insert(0,str(OSQP_DEPS));import osqp
 opt=osqp.OSQP();opt.setup(P=H.tocsc(),q=np.zeros(n),A=K,l=lower,u=upper,verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=200000,polishing=True,adaptive_rho=True,scaling=30,scaled_termination=True)
 sol=opt.solve();ok=int(sol.info.status_val)==1
 report={"schema":"numi.human.b109-114.full-exact-segment-depth-self-fit.v1","status":"QP_FAILED","baseline_sha256":BASE_SHA,"baseline_audit130_sha256":sha(AUD130),"baseline_audit159_sha256":sha(AUD159),"layer_sha256":sha(LAYER),"segment_preflight_sha256":sha(SEG),"self_preflight_sha256":sha(SELF_PRE),"target_bank":[BODY,BONE],"target_raw_pair_count":len(seg["target_pairs"]),"target_skin_face_count":len({r["skin_face_row"] for r in seg["target_pairs"]}),"target_segment_endpoint_rows":len(segment_rows),"target_segment_constraint_margin_mm":SEGMENT_MARGIN_MM,"inside_node_witnesses":point_meta,"inside_depth_margin_mm":INSIDE_MARGIN_MM,"self_witness_pairs":[list(x) for x in selfpairs],"self_barrier_records":self_records,"ring3_patch_face_count":len(affected),"ring3_patch_component_face_counts":[len(c) for c in components],"ring3_patch_node_count":len(support),"fixed_boundary_nodes":boundary_nodes.astype(int).tolist(),"fixed_boundary_count":len(boundary_nodes),"moving_support_node_count":int(np.count_nonzero(np.any(movable.reshape(-1,3),axis=1))),"source_bound_receipt":bound_receipt,"trust_mm":TRUST_MM,"source_cap_mm":SOURCE_CAP_MM,"area_guard_rows":area_count,"normal_guard_rows":len(normal_rows),"normal_turn_linear_bound_degrees":10.0,"layer_guard_rows":layer_count,"layer_guard_receipt":layer_receipt,"edge_gradient_cap":EDGE_GRAD,"edge_gradient_receipt":edge_receipt,"self_barrier_row_count":len(self_rows),"zero_movable_preflight":zero_receipt,"solver":{"status":str(sol.info.status),"status_val":int(sol.info.status_val),"iterations":int(sol.info.iter),"primal_residual":float(sol.info.prim_res),"dual_residual":float(sol.info.dual_res),"success":bool(ok)},"backtrack_trials":[],"geometry_admitted":False,"physical_steps":0,"elapsed_s":time.monotonic()-t0}
 if ok:
  d=np.asarray(sol.x,np.float64).reshape(-1,3)
  for alpha in BACKTRACKS:
   y=x.copy();y[support]=(x[support].astype(np.float64)+alpha*d/1000.).astype(np.float32)
   changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1));disp=np.linalg.norm((y.astype(np.float64)-x.astype(np.float64))*1000,axis=1)
   sh=shape_gate(bm,y.astype(np.float64)*1000,F,affected)
   delta=(y.astype(np.float64)-x.astype(np.float64))*1000.
   target_progress=[]
   for wr in segment_rows:
    val=0.0
    for node,coef in zip(wr["nodes"],wr["coefficients"]):val+=float(np.dot(np.asarray(coef),delta[int(node)]))
    target_progress.append(val)
   inside_progress=[float(np.dot(np.asarray(r["outward_normal_world"]),delta[int(r["node_row"])])) for r in point_meta]
   rec={"alpha":alpha,"shape":sh,"changed_node_count":len(changed),"changed_face_count":int(np.count_nonzero(np.any(np.isin(F,changed),axis=1))),"max_node_step_mm":float(disp.max(initial=0)),"min_target_segment_outward_progress_mm":float(min(target_progress,default=0)),"target_segment_progress_min_median_max_mm":[float(min(target_progress,default=0)),float(np.median(target_progress)) if target_progress else 0.0,float(max(target_progress,default=0))],"inside_outward_progress_min_median_max_mm":[float(min(inside_progress,default=0)),float(np.median(inside_progress)) if inside_progress else 0.0,float(max(inside_progress,default=0))],"inside_depths_baseline_mm":[float(r["baseline_nearest_depth_mm"]) for r in point_meta],"inside_depth_linear_residuals_mm":[float(r["baseline_nearest_depth_mm"]-v) for r,v in zip(point_meta,inside_progress)],"candidate_path":None,"candidate_sha256":None}
   if sh["pass"] and disp.max(initial=0)<=TRUST_MM+1e-5 and min(inside_progress,default=0)>0:
    cp=D/f"candidate-segment-alpha-{str(alpha).replace('.','p')}.npz"
    out_payload={k:v.copy() for k,v in payload.items()};out_payload["points_world_m"]=y
    n0=len(x);m0=len(F);out_payload["remesh806_old_to_new_node_rows"]=np.arange(n0,dtype=np.int64);out_payload["remesh806_new_to_old_node_rows"]=np.arange(n0,dtype=np.int64)
    out_payload["remesh806_baseline_sha256"]=np.asarray([BASE_SHA],dtype="<U64");out_payload["remesh806_face_parent_indptr"]=np.arange(m0+1,dtype=np.int64);out_payload["remesh806_face_parent_face_rows"]=np.arange(m0,dtype=np.int64)
    with cp.open("xb") as f:np.savez_compressed(f,**out_payload)
    rec["candidate_path"]=str(cp);rec["candidate_sha256"]=sha(cp)
   report["backtrack_trials"].append(rec)
  passing=[r for r in report["backtrack_trials"] if r["candidate_path"]]
  report["status"]="SHAPE_AND_DEPTH_PROGRESS_CANDIDATES_EMITTED" if passing else "NO_BACKTRACK_PASSED_SHAPE_DEPTH_OR_TRUST"
  report["candidate_candidates"]=passing
 out=D/"fit-report.json";out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":report["status"],"report":str(out),"report_sha256":sha(out),"target_segment_rows":len(segment_rows),"inside_nodes":len(point_meta),"self_rows":len(self_rows),"support_nodes":len(support),"fixed_boundary":len(boundary_nodes),"solver":report["solver"],"trials":[{"alpha":r["alpha"],"shape_pass":r["shape"]["pass"],"changed_nodes":r["changed_node_count"],"max_node_step_mm":r["max_node_step_mm"],"min_target_progress_mm":r["min_target_segment_outward_progress_mm"],"inside_progress_min_median_max_mm":r["inside_outward_progress_min_median_max_mm"],"inside_depth_linear_residual_min_max_mm":[min(r["inside_depth_linear_residuals_mm"],default=0),max(r["inside_depth_linear_residuals_mm"],default=0)],"candidate_path":r["candidate_path"],"candidate_sha256":r["candidate_sha256"],"shape_metrics":{k:r["shape"][k] for k in ("min_area_ratio","min_min_angle_ratio","max_aspect_ratio","max_normal_turn_deg","violating_face_count")}} for r in report["backtrack_trials"]]},indent=2,allow_nan=False))
if __name__=="__main__":main()
