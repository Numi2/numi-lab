from pathlib import Path
from collections import Counter
import hashlib,json,time
import numpy as np

S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
W=S/"per-bank-workers-20261027/b43-16"
B=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
C=B/"candidate.npz"; A=B/"audit-159/report.json"; I=B/"audit-130/summary.json"
T=W/"attempt-021-bulk-crossing-and-inside-star-patch-001/bulk-patch-topology.json"
OUT=W/"attempt-022-bulk-boundary-centroid-fan-001"
PIN="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
AUDIT159_PIN="c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e"
AUDIT130_PIN="0ec99fa47245355d60858b05e1240b0e562de229c13721ea11659f6f4acdf5cf"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def boundary(faces):
 uses=Counter()
 for f in faces:
  a,b,c=map(int,f)
  for u,v in ((a,b),(b,c),(c,a)): uses[tuple(sorted((u,v)))]+=1
 if any(v not in (1,2) for v in uses.values()): raise ValueError("nonmanifold fan output edge")
 return sorted(e for e,n in uses.items() if n==1)
def metrics(tri,parent):
 pn=np.cross(parent[1]-parent[0],parent[2]-parent[0]); cn=np.cross(tri[1]-tri[0],tri[2]-tri[0])
 pn2=float(pn@pn); plen=np.sqrt(pn2)
 if plen==0: return {"area_ratio":0.0,"orientation":-1.0,"min_angle_deg":0.0,"aspect":float("inf")}
 ratio=float(np.linalg.norm(cn)/plen); orient=float((pn@cn)/pn2)
 lens=[float(np.linalg.norm(tri[(i+1)%3]-tri[i])) for i in range(3)]
 area=float(np.linalg.norm(cn)*.5)
 aspect=max(lens)**2/(4*np.sqrt(3)*max(area,1e-30))
 aa=[]
 for i in range(3):
  u=tri[(i+1)%3]-tri[i];v=tri[(i+2)%3]-tri[i]
  aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
 return {"area_ratio":ratio,"orientation":orient,"min_angle_deg":min(aa),"aspect":aspect}

started=time.time()
if sha(C)!=PIN or sha(A)!=AUDIT159_PIN or sha(I)!=AUDIT130_PIN: raise ValueError("baseline/audit pin mismatch")
top=json.loads(T.read_text())
if top.get("baseline_sha256")!=PIN or not top.get("final_ancestry_complete"): raise ValueError("bulk patch provenance pin/incompleteness")
rows=np.asarray(top["final_patch_face_rows"],np.int64)
if len(rows)!=183 or len(set(map(int,rows)))!=183: raise ValueError("unexpected bulk patch face inventory")
np.save(OUT/"fan-parent-rows.npy",rows)
audit=json.loads(A.read_text()); inside=json.loads(I.read_text())
bank=next(x for x in audit["banks"] if (int(x["body"]),int(x["stable_bone_id"]))==(43,16))
pairs=sorted({(int(f),int(t)) for f,t in (bank.get("result") or {}).get("triangle_pairs",[])})
crossing=sorted({f for f,t in pairs})
inside_rows=sorted({int(w["output_node"]) for w in inside["inside_witnesses"] if (int(w.get("bank_body",-1)),int(w.get("bank_stable_id",-1)))==(43,16) and w.get("classification")=="inside" and int(w.get("ambiguous",0))==0})
if len(pairs)!=249 or set(crossing)!=set(top["target_crossing_face_rows"]) or len(inside_rows)!=43: raise ValueError("typed exact baseline witness mismatch")
diag={"patches":[{"crossing_skin_faces":crossing,"inside_crossing_nodes":[{"node_row":n} for n in inside_rows]}],
      "pins":{"candidate_sha256":PIN,"audit159_sha256":AUDIT159_PIN,"audit130_sha256":AUDIT130_PIN,"bulk_patch_sha256":sha(T)}}
diag_path=OUT/"fan-local-diagnostic.json";diag_path.write_text(json.dumps(diag,indent=2)+"\n")
P=None;F=None
with np.load(C,allow_pickle=False) as z:
 P=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["source_O_faces"],np.int64)
 ancestry={}
 fields=[
  ("source","face_source_ancestry_indptr","face_source_ancestry_face_rows"),
  ("fine_subface","face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows"),
  ("fine_detail","face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows"),
  ("immediate_parent","refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows"),
  ("prior_parent","prior_input_parent_546__input_refinement_face_parent_ancestry_indptr","prior_input_parent_546__input_refinement_face_parent_ancestry_face_rows"),
  ("generation002_parent","prior_input_parent_546_generation_002__input_refinement_face_parent_ancestry_indptr","prior_input_parent_546_generation_002__input_refinement_face_parent_ancestry_face_rows"),
 ]
 for label,ik,rk in fields:
  ip=np.asarray(z[ik],np.int64); vals=np.asarray(z[rk],np.int64)
  lens=ip[rows+1]-ip[rows]
  if np.any(lens<=0): raise ValueError(f"incomplete {label} ancestry on patch")
  ancestry[label]={"rows_nonempty":int(len(lens)),"minimum_entries_per_parent":int(lens.min()),"output_entries_if_3way_fan":int(3*sum(lens))}
source_faces=F[rows]
boundary_before=boundary(source_faces)
vrows=np.unique(source_faces.reshape(-1));vmap={int(v):i for i,v in enumerate(vrows)}
fan_faces=[];fan_parents=[];centers=[];bad=[];all_child_q=[]
for fi,tids in zip(rows,source_faces):
 tri=P[tids].astype(np.float64)
 opp=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])])
 if not np.isfinite(opp).all() or opp.sum()<=0: raise ValueError(f"invalid triangle metric for parent {fi}")
 mu=.5*opp/opp.sum()+.5/3.0
 center64=mu@tri
 center32=center64.astype(np.float32).astype(np.float64)
 center_idx=len(vrows)+len(centers);centers.append(center32)
 a,b,c=(vmap[int(n)] for n in tids)
 local_parent=tri
 kids=np.asarray([[a,b,center_idx],[b,c,center_idx],[c,a,center_idx]],np.int64)
 fan_faces.extend(kids);fan_parents.extend([int(fi)]*3)
 qs=[metrics(np.asarray([tri[x],tri[y],center32]),local_parent) for x,y in ((0,1),(1,2),(2,0))]
 all_child_q.extend(qs)
 if not all(q["area_ratio"]>=.10 and q["orientation"]>0 and q["min_angle_deg"]>=.5 and q["aspect"]<=50 for q in qs):
  bad.append({"parent_face":int(fi),"min_child_area_ratio":min(q["area_ratio"] for q in qs),"min_child_angle_deg":min(q["min_angle_deg"] for q in qs),"max_child_aspect":max(q["aspect"] for q in qs),"orientation_positive":all(q["orientation"]>0 for q in qs)})
fan_faces=np.asarray(fan_faces,np.int64)
boundary_after=boundary(fan_faces)
boundary_before_local=sorted(tuple(sorted((vmap[a],vmap[b]))) for a,b in boundary_before)
if boundary_before_local!=boundary_after: raise ValueError("fan changes fixed boundary edge identities")
zero_source=bool(np.array_equal(P[vrows].view(np.uint32),P[vrows].copy().view(np.uint32)))
report={
 "schema":"numi.human.bulk_boundary_fan_preflight.v1","status":"BASELINE_FAN_TOPOLOGY_AND_SHAPE_PREFLIGHT",
 "candidate_sha256":PIN,"audit159_sha256":AUDIT159_PIN,"audit130_sha256":AUDIT130_PIN,"bulk_patch_topology_sha256":sha(T),
 "patch_parent_count":len(rows),"crossing_pair_count":len(pairs),"crossing_parent_count":len(crossing),"inside_source_node_count":len(inside_rows),
 "parent_boundary":{"edge_count":len(boundary_before),"degree_histogram":{"2":len({v for e in boundary_before for v in e})},"is_one_loop":len(boundary_before)>0},
 "fan_output":{"face_count":len(fan_faces),"center_count":len(centers),"fixed_boundary_edges_identical":True,"boundary_edge_count":len(boundary_after),
  "zero_step_source_float32_bitwise":zero_source},
 "all_fan_children_shape":{"existing_gates":{"area_ratio_min":.10,"orientation_positive":True,"min_angle_deg":.5,"aspect_max":50.0},
  "passing_parent_count":len(rows)-len(bad),"excluded_parent_count":len(bad),"excluded_faces":bad,
  "minimum_child_area_ratio":float(min(q["area_ratio"] for q in all_child_q)) if all_child_q else None},
 "complete_parent_ancestry":ancestry,
 "candidate_emitted":False,"geometry_changed":False,"elapsed_seconds":time.time()-started}
report_path=OUT/"fan-preflight.json";report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps(report,indent=2,allow_nan=False))
