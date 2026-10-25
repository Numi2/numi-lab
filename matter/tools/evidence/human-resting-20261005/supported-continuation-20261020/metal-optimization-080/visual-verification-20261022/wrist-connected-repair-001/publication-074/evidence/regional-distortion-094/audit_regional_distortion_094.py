#!/usr/bin/env python3
"""Read-only regional distortion audit for connected-clearance-091 iteration 004.

Maps the candidate's 084-node displacement through the frozen 075->084 W map,
then compares detailed float32 triangles/edges with their fixed-O detail origin.
This is an offline construction diagnostic, not a native or physical gate.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from scipy import sparse

ROOT=Path("/Users/n/Work/numi-human-connected-surface-069")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
RUN=ROOT/"whole-reference-tetgen-084-bounded/connected-clearance-091/run-001"
CAND=RUN/"iteration-004/candidate.npz"
INPUT=ROOT/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
MAP=ROOT/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002"
EXPORT=E/"accepted-transfer-row-export-075/run-001/source-preflight"
ADAPTER=E/"E/accepted-transfer-row-adapter-075"
DETAIL_PAYLOAD=E/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
IDENTITY=E/"torso-distal-arm-decoupling-022/body-identity-inputs.json"
HAND_REPORT=E/"bilateral-hand-functional-rebind-020/run-001/report.json"
OUT=ROOT/"regional-distortion-094"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ADAPTER))
from load_transfer_075 import load_transfer075

EXPECTED={
 "candidate": "0675413e628d157cf48fc8a00508ed25b718126c40881fc3e5c03cdb069bb903",
 "input": "5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db",
 "W": "55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519",
 "transfer_rows": "71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14",
 "coarse_reference": "9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2",
 "coarse_faces": "7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2",
 "detail_payload": "7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",
}
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def require_sha(key,path):
 got=sha(path)
 assert got==EXPECTED[key], f"{key} SHA mismatch: {got}"
 return got
for k,p in [("candidate",CAND),("input",INPUT),("W",MAP/"Wnew-detail-to-084-nodes.npz"),
            ("transfer_rows",EXPORT/"accepted-transfer-rows.csv"),
            ("coarse_reference",EXPORT/"accepted-coarse-reference.csv"),
            ("coarse_faces",EXPORT/"accepted-coarse-faces.csv"),("detail_payload",DETAIL_PAYLOAD)]:
 require_sha(k,p)

z=np.load(INPUT)
candidate=np.load(CAND)
coarse0=z["current037_points_m_f32"]
newcoarse=candidate["points_world_f32"]
assert coarse0.dtype==np.float32 and newcoarse.dtype==np.float32
assert coarse0.shape==(7360,3) and newcoarse.shape==coarse0.shape
W=sparse.load_npz(MAP/"Wnew-detail-to-084-nodes.npz").tocsr()
W.sum_duplicates();W.sort_indices()
W075,detail0,coarse075,faces=load_transfer075(
 EXPORT/"accepted-transfer-rows.csv", EXPORT/"accepted-coarse-reference.csv",
 EXPORT/"accepted-coarse-faces.csv", DETAIL_PAYLOAD,
 expected_sha256={"transfer_rows":EXPECTED["transfer_rows"],
                  "coarse_reference":EXPECTED["coarse_reference"],
                  "coarse_faces":EXPECTED["coarse_faces"],
                  "detail_payload":EXPECTED["detail_payload"]})
assert W.shape==(61950,7360) and W075.shape==(61950,2411)
assert detail0.dtype==np.float32 and faces.shape==(108987,3)
assert np.array_equal(coarse075,coarse0[:2411]),"075 coarse origin is not the exact 084 source prefix"
# Exactly the 091 mapped-surface convention: fixed detail origin + Wnew displacement
# from the pinned 084 coarse origin. Emission is rounded to float32 before metrics.
delta=newcoarse.astype(np.float64)-coarse0.astype(np.float64)
detail1=(detail0.astype(np.float64)+W@delta).astype(np.float32)
assert np.isfinite(detail1).all() and np.array_equal((detail0.astype(np.float64)+W@np.zeros_like(delta)).astype(np.float32),detail0)

# 075 row donor-body IDs are direct per-detail-row values; do not infer names from geometry.
donor=np.full(len(detail0),-1,dtype=np.int32)
active=np.zeros(len(detail0),dtype=bool)
POSES=EXPORT/"donor-rest-poses.csv"
pose_rows=list(csv.DictReader(POSES.open()))
pose_ids={int(r["body_id"]) for r in pose_rows}
assert len(pose_rows)==157
with open(EXPORT/"accepted-transfer-rows.csv",newline="") as f:
 for row in csv.DictReader(f):
  i=int(row["detail_row"])
  assert 0<=i<len(donor) and donor[i]==-1
  donor[i]=int(row["detail_donor_body"])
  active[i]=bool(int(row["active"]))
  if active[i]:
   assert int(row["transfer_donor_body"])==donor[i]
assert np.all(donor>=0) and set(np.unique(donor)).issubset(pose_ids)
F=faces.astype(np.int64,copy=False)
labels=donor[F]
# Majority donor per face; all-distinct triples are kept as mixed instead of guessed.
face_label=np.full(len(F),-1,dtype=np.int32)
face_label[(labels[:,0]==labels[:,1])|(labels[:,0]==labels[:,2])]=labels[(labels[:,0]==labels[:,1])|(labels[:,0]==labels[:,2]),0]
m=(face_label==-1)&(labels[:,1]==labels[:,2])
face_label[m]=labels[m,1]
mixed=(face_label==-1)

def tri_metrics(P):
 tri=P[F].astype(np.float64)
 e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
 cr=np.cross(e1,e2)
 norm=np.linalg.norm(cr,axis=1)
 return .5*norm, cr/np.maximum(norm[:,None],1e-300)
area0,norm0=tri_metrics(detail0); area1,norm1=tri_metrics(detail1)
assert np.all(area0>0) and np.all(area1>0)
area_ratio=area1/area0
normal_cos=np.clip(np.einsum("ij,ij->i",norm0,norm1),-1.,1.)
normal_deg=np.degrees(np.arccos(normal_cos))
vertex_move=np.linalg.norm(detail1.astype(np.float64)-detail0.astype(np.float64),axis=1)
face_move=np.max(vertex_move[F],axis=1)
face_mean_move=np.mean(vertex_move[F],axis=1)

# Build unique edges once, retaining a deterministic first incident face orientation.
directed=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]],axis=0)
edge_face=np.tile(np.arange(len(F),dtype=np.int64),3)
undirected=np.sort(directed,axis=1)
unique_edges,inv,edge_counts=np.unique(undirected,axis=0,return_inverse=True,return_counts=True)
order=np.argsort(inv,kind="stable")
starts=np.r_[0,np.cumsum(edge_counts[:-1])]
manifold_ids=np.flatnonzero(edge_counts==2)
i0=order[starts[manifold_ids]]
i1=order[starts[manifold_ids]+1]
ef0=edge_face[i0];ef1=edge_face[i1]
edge_rows=unique_edges[manifold_ids]
edge_dir=directed[i0]
edgevec=detail0[edge_dir[:,1]].astype(float)-detail0[edge_dir[:,0]].astype(float)
edgevec1=detail1[edge_dir[:,1]].astype(float)-detail1[edge_dir[:,0]].astype(float)
elen0=np.linalg.norm(edgevec,axis=1);elen1=np.linalg.norm(edgevec1,axis=1)
assert np.all(elen0>0) and np.all(elen1>0)
u0=edgevec/elen0[:,None];u1=edgevec1/elen1[:,None]
def signed_dihedral(U,N0,N1):
 return np.arctan2(np.einsum("ij,ij->i",U,np.cross(N0,N1)),
                   np.clip(np.einsum("ij,ij->i",N0,N1),-1.,1.))
dih0=signed_dihedral(u0,norm0[ef0],norm0[ef1])
dih1=signed_dihedral(u1,norm1[ef0],norm1[ef1])
dih_delta=(dih1-dih0+np.pi)%(2*np.pi)-np.pi
dih_abs=np.abs(dih_delta)
same_direction_edges=int(np.count_nonzero((directed[i1,0]==directed[i0,0])&(directed[i1,1]==directed[i0,1])))
opposite_direction_edges=int(np.count_nonzero((directed[i1,0]==directed[i0,1])&(directed[i1,1]==directed[i0,0])))
edge_label0=face_label[ef0];edge_label1=face_label[ef1]
edge_pair=np.sort(np.c_[edge_label0,edge_label1],axis=1)
edge_pair_text=np.array([f"{int(a)}|{int(b)}" for a,b in edge_pair],dtype=object)

def qs(x):
 x=np.asarray(x,dtype=float)
 if not len(x): return {"count":0}
 return {"count":int(len(x)),"min":float(np.min(x)),"p50":float(np.quantile(x,.50)),
         "p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),
         "max":float(np.max(x))}
def top_face(metric,idx,n=8,largest=True):
 if not len(idx): return []
 order_local=np.argsort(metric[idx],kind="stable")
 chosen=idx[order_local[-n:][::-1]] if largest else idx[order_local[:n]]
 out=[]
 for k in chosen:
  out.append({"face_row":int(k),"vertices":F[k].tolist(),"donor_vertices":labels[k].tolist(),
              "majority_donor_body_id":int(face_label[k]),"area_ratio":float(area_ratio[k]),
              "normal_change_deg":float(normal_deg[k]),"max_vertex_displacement_mm":float(face_move[k]*1000),
              "centroid_baseline_m":np.mean(detail0[F[k]].astype(float),axis=0).tolist(),
              "centroid_candidate_m":np.mean(detail1[F[k]].astype(float),axis=0).tolist()})
 return out
def top_edge(n=10):
 ix=np.argsort(dih_abs,kind="stable")[-n:][::-1]
 return [{"edge_vertex_ids":edge_rows[k].tolist(),"incident_face_rows":[int(ef0[k]),int(ef1[k])],
          "incident_face_donor_bodies":[int(edge_label0[k]),int(edge_label1[k])],
          "baseline_signed_dihedral_deg":float(np.degrees(dih0[k])),
          "candidate_signed_dihedral_deg":float(np.degrees(dih1[k])),
          "signed_change_deg":float(np.degrees(dih_delta[k])),
          "absolute_change_deg":float(np.degrees(dih_abs[k])),
          "baseline_edge_length_mm":float(elen0[k]*1000),
          "candidate_edge_length_mm":float(elen1[k]*1000)}
         for k in ix]

# Only identity labels backed by the frozen typed identity records are named.
identity_data=json.loads(IDENTITY.read_text())
core_map=identity_data["core_body_to_source"]
named={}
for bid,v in core_map.items():
 name=v.get("name")
 if name: named[int(bid)]={"name":name,"source_body_id":v.get("source_body_id"),
                           "identity_source":"torso-distal-arm-decoupling-022/body-identity-inputs.json"}
# The hand rebind report identifies the selected row sets by donor IDs, but not wrist-vs-finger.
hand=json.loads(HAND_REPORT.read_text())
handid={}
for side,info in hand.get("coverage",{}).items():
 before=info.get("before_active_body_ids",[])
 if len(before)==1:
  handid[int(before[0])]={"name":f"{side}_selected_hand_rebind_donor",
                           "identity_source":"bilateral-hand-functional-rebind-020/run-001/report.json",
                           "limitation":"selected-hand donor identity only; no wrist/finger subregion separation"}
for bid,x in handid.items():
 named.setdefault(bid,x)

def face_group_stats(ids):
 ids=np.asarray(ids,dtype=np.int64)
 out={"face_count":int(len(ids)),"area_ratio":qs(area_ratio[ids]),
      "normal_cosine":qs(normal_cos[ids]),"normal_change_deg":qs(normal_deg[ids]),"max_vertex_displacement_mm":qs(face_move[ids]*1000),
      "top_area_ratio_low":top_face(area_ratio,ids,3,False),
      "top_area_ratio_high":top_face(area_ratio,ids,3,True),
      "top_normal_change":top_face(normal_deg,ids,5,True)}
 return out
def edge_group_stats(mask):
 ids=np.flatnonzero(mask)
 return {"edge_count":int(len(ids)),"absolute_dihedral_change_deg":qs(np.degrees(dih_abs[ids])),
         "signed_dihedral_change_deg":qs(np.degrees(dih_delta[ids])),
         "top_edges":[top_edge_record(int(i)) for i in ids[np.argsort(dih_abs[ids],kind="stable")[-5:][::-1]]]}
def top_edge_record(k):
 return {"edge_vertex_ids":edge_rows[k].tolist(),"incident_face_rows":[int(ef0[k]),int(ef1[k])],
         "baseline_midpoint_m":np.mean(detail0[edge_rows[k]].astype(float),axis=0).tolist(),
         "candidate_midpoint_m":np.mean(detail1[edge_rows[k]].astype(float),axis=0).tolist(),
         "incident_face_donor_bodies":[int(edge_label0[k]),int(edge_label1[k])],
         "baseline_signed_dihedral_deg":float(np.degrees(dih0[k])),
         "candidate_signed_dihedral_deg":float(np.degrees(dih1[k])),
         "signed_change_deg":float(np.degrees(dih_delta[k])),
         "absolute_change_deg":float(np.degrees(dih_abs[k]))}

donor_groups={}
for bid in sorted(map(int,np.unique(donor))):
 ids=np.flatnonzero(face_label==bid)
 touch=np.flatnonzero(np.any(labels==bid,axis=1))
 touchmask=np.zeros(len(F),dtype=bool);touchmask[touch]=True
 emask=touchmask[ef0]|touchmask[ef1]
 donor_groups[str(bid)]={"typed_identity":named.get(bid),
     "detail_vertex_rows":int(np.count_nonzero(donor==bid)),
     "active_transfer_rows":int(np.count_nonzero((donor==bid)&active)),
     "majority_faces":face_group_stats(ids),
     "touching_faces":face_group_stats(touch),
     "incident_dihedral_edges":edge_group_stats(emask)}
donor_groups["mixed_face_donor"]={"face_count":int(np.count_nonzero(mixed)),
    "normal_change_deg":qs(normal_deg[mixed]),"area_ratio":qs(area_ratio[mixed]),
    "max_vertex_displacement_mm":qs(face_move[mixed]*1000)}

# Region aggregates are unions of explicit donor IDs. No spatial/coordinate guessing.
regions={
 "torso":[20],
 "right_arm":[41,42,43],
 "left_arm":[91,92,93],
}
for bid,obj in handid.items():
 regions[f"{obj['name']} donor id {bid}"]=[bid]
region_groups={}
for region,ids0 in regions.items():
 ids0=np.asarray(ids0,dtype=np.int32)
 fmask=np.any(np.isin(labels,ids0),axis=1)
 fids=np.flatnonzero(fmask)
 emask=fmask[ef0]|fmask[ef1]
 region_groups[region]={"donor_body_ids":ids0.astype(int).tolist(),"face_statistics":face_group_stats(fids),
                        "edge_statistics":edge_group_stats(emask)}
all_metrics={
 "mapped_detail_vertex_displacement_mm":qs(vertex_move*1000),
 "area_ratio":qs(area_ratio),
 "normal_cosine":qs(normal_cos),
 "normal_change_deg":qs(normal_deg),
 "signed_dihedral_change_deg":qs(np.degrees(dih_delta)),
 "absolute_dihedral_change_deg":qs(np.degrees(dih_abs)),
}
report={
 "schema":"regional_distortion_094_v1",
 "status":"READ_ONLY_CONSTRUCTION_DISTORTION_AUDIT",
 "scope":"Detailed 075 surface mapped from fixed detail origin through the frozen 075-to-084 displacement map using connected-clearance-091 iteration-004 coarse points. No geometry is changed and no physical/native admission is implied.",
 "comparison":"baseline=detail_origin from frozen 075 owner; candidate=Float32(detail_origin + Wnew @ (091_candidate_points - compacted_084_current037_origin)).",
 "inputs":{
  "091_candidate":{"path":str(CAND),"sha256":sha(CAND)},
  "084_input":{"path":str(INPUT),"sha256":sha(INPUT)},
  "Wnew":{"path":str(MAP/'Wnew-detail-to-084-nodes.npz'),"sha256":sha(MAP/'Wnew-detail-to-084-nodes.npz')},
  "Wnew_report":{"path":str(MAP/'transfer-report.json'),"sha256":sha(MAP/'transfer-report.json')},
  "075_transfer_rows":{"path":str(EXPORT/'accepted-transfer-rows.csv'),"sha256":sha(EXPORT/'accepted-transfer-rows.csv')},
  "075_coarse_reference":{"path":str(EXPORT/'accepted-coarse-reference.csv'),"sha256":sha(EXPORT/'accepted-coarse-reference.csv')},
  "075_coarse_faces":{"path":str(EXPORT/'accepted-coarse-faces.csv'),"sha256":sha(EXPORT/'accepted-coarse-faces.csv')},
  "075_donor_rest_poses":{"path":str(POSES),"sha256":sha(POSES)},
  "detail_payload":{"path":str(DETAIL_PAYLOAD),"sha256":sha(DETAIL_PAYLOAD)},
  "regional_identity_table":{"path":str(IDENTITY),"sha256":sha(IDENTITY)},
  "hand_rebind_report":{"path":str(HAND_REPORT),"sha256":sha(HAND_REPORT)},
  "mapping_owner":"publication-074/canonical/matter/tools/human_skin_fat_embedded_surface.py::MappedSkinSurface.positions"
 },
 "counts":{"detail_vertices":int(len(detail0)),"detail_faces":int(len(F)),"candidate_coarse_nodes":int(len(newcoarse)),
           "active_detail_transfer_rows":int(active.sum()),"inactive_fixed_detail_rows":int((~active).sum()),
           "unique_detail_donor_body_ids":int(len(np.unique(donor))),"faces_all_vertices_same_donor":int(np.count_nonzero((labels[:,0]==labels[:,1])&(labels[:,1]==labels[:,2]))),
           "faces_two_or_more_donors":int(np.count_nonzero(np.any(labels!=labels[:,0,None],axis=1))),
           "faces_all_three_donors_distinct":int(np.count_nonzero(mixed)),"unique_edges":int(len(unique_edges)),
           "boundary_edges":int(np.count_nonzero(edge_counts==1)),"manifold_two-face_edges":int(np.count_nonzero(edge_counts==2)),
           "nonmanifold_edges":int(np.count_nonzero(edge_counts>2)),
           "two-face_edges_with_opposite_directed_incidence":opposite_direction_edges,
           "two-face_edges_with_same_directed_incidence":same_direction_edges},
 "global":all_metrics,
 "global_worst_faces":{"largest_normal_change":top_face(normal_deg,np.arange(len(F)),10,True),
                       "smallest_area_ratio":top_face(area_ratio,np.arange(len(F)),10,False),
                       "largest_area_ratio":top_face(area_ratio,np.arange(len(F)),10,True)},
 "global_worst_dihedral_edges":top_edge(15),
 "typed_region_groups":region_groups,
 "donor_body_groups":donor_groups,
 "identity_limits":{
   "typed_source":str(IDENTITY),
   "explicit_numeric_donor_id_labels":{str(k):v for k,v in sorted(named.items())},
   "head_identity_in_available_frozen_map":False,
   "per_finger_identity_in_075_row_metadata":False,
   "hand_ids_49_99":"020 report identifies right/left selected-hand rebind donor sets only; metrics retain numeric IDs and do not split wrist versus fingers.",
   "unmapped_donor_ids":[int(x) for x in np.unique(donor) if int(x) not in named],
   "no_regions_inferred_from_coordinates":True},
 "limitations":[
   "Mapped detailed coordinates are a construction prolongation through Wnew, not a native pose emission or physics result.",
   "Face groups use the majority of three 075 detail_donor_body row IDs; all-distinct donor triples are kept in an explicit mixed group.",
   "Edge donor labels are the two incident faces' majority donor IDs; numeric donor groups with any mixed face may not capture the full blend.",
   "Reported dihedral change is signed around the first incident face's directed edge; edge orientation is deterministic from source face order.",
   "No self-intersection, bone clearance, material thickness, constitutive, or native-renderer gate was run here.",
   "Arm/torso labels use the frozen explicit body-identity table where present. Wrist/finger/head subregions are not separately named by the available 075 row export; IDs 49/99 are only selected-hand donor sets."
 ]
}
path=OUT/"report.json"
path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(path),"detail_displacement_mm":all_metrics["mapped_detail_vertex_displacement_mm"],
 "area_ratio":all_metrics["area_ratio"],"normal_change_deg":all_metrics["normal_change_deg"],
 "abs_dihedral_change_deg":all_metrics["absolute_dihedral_change_deg"],
 "face_count":len(F),"body_groups":len(donor_groups)-1,
 "top_dihedral":report["global_worst_dihedral_edges"][:3],
 "report_sha256":sha(path)},indent=2))

