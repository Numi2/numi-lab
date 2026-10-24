#!/usr/bin/env python3
"""Bounded metadata/ancestry audit of the separate actual007 O descriptor."""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

BASE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
DESCRIPTOR = BASE / "bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz"
SOURCE = BASE / "left-ankle-rest-preserving-weight-repair-128/candidate.nhskin"
REGISTERED = BASE / "native-app-main-header-integration-151/source-preflight-007/registered-skin.csv"
REPORT = BASE / "bilateral-source-cut-plane-271/report.json"
OUT = BASE / "duplicate-source-batch-compose-256/o271-lineage-audit.json"
MODULE = BASE / "duplicate-source-batch-compose-256/tools/rebind_human_skin_reference.py"

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def rows_csv(path):
 with open(path,newline="") as f:return list(csv.DictReader(f))

def main():
 pins={p.name:sha(p) for p in (DESCRIPTOR,SOURCE,REGISTERED,REPORT,MODULE)}
 spec=importlib.util.spec_from_file_location("frozen_skin_256",MODULE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 source=m.decode(SOURCE.read_bytes())
 with np.load(DESCRIPTOR,allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
 reg=rows_csv(REGISTERED)
 if len(reg)!=source["nv"] or not np.array_equal(np.asarray([int(r["vertex"]) for r in reg]),np.arange(source["nv"])):
  raise ValueError("registered source vertex rows do not exactly cover the pinned NHSKIN domain")
 Xsrc=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in reg],np.float64)
 X=np.asarray(d["outer_world_actual007_m"],np.float64)
 F=np.asarray(d["surface_faces"],np.int64)
 kind=np.asarray(d["surface_face_kind"])
 detail=np.asarray(d["surface_face_detailed_source_ids"],np.int64)
 node_kind=np.asarray(d["node_kind"]).astype(str)
 sid=np.asarray(d["source_vertex_ids"],np.int64)
 edge=np.asarray(d["source_cut_edge_vertices"],np.int64)
 ew=np.asarray(d["source_cut_edge_weights"],np.float64)
 rows=np.flatnonzero(kind==2)
 parent=detail[rows]
 if len(rows)!=len(d["source_face_parent_rows"]) or not np.array_equal(parent,d["source_face_parent_rows"]):
  raise ValueError("detailed-face identity sequence differs from source_face_parent_rows")
 if len(parent)==0 or np.any(parent<0) or np.any(parent>=len(source["inds"])):
  raise ValueError("detailed parent face IDs are not source NHSKIN face rows")
 used=np.unique(F[rows].ravel())
 mesh_types={k:int(np.count_nonzero(node_kind[used]==k)) for k in np.unique(node_kind[used])}
 if set(mesh_types)-{"source_vertex","source_cut"}:
  raise ValueError("a detailed source face depends on non-source O vertices")
 # Replay world point ancestry and full binding rows from source payload IDs.
 Wrec=np.empty((len(used),source["nb"]),np.float64)
 Xrec=np.empty((len(used),3),np.float64)
 for j,vi in enumerate(used):
  if node_kind[vi]=="source_vertex":
   s=int(sid[vi])
   if not 0<=s<source["nv"]:raise ValueError("invalid direct source vertex ID")
   Xrec[j]=Xsrc[s];Wrec[j]=source["weights"][s]
  elif node_kind[vi]=="source_cut":
   e=edge[vi];w=ew[vi]
   if np.any(e<0) or np.any(e>=source["nv"]) or not np.isfinite(w).all() or np.min(w)<0 or abs(w.sum()-1)>1e-12:
    raise ValueError("invalid clipped source-edge ancestry")
   Xrec[j]=w@Xsrc[e];Wrec[j]=w@source["weights"][e]
  else:raise ValueError("unsupported detailed-row ancestry")
 if np.max(np.abs(Wrec.sum(axis=1)-1))>5e-7 or np.min(Wrec)<-1e-12:
  raise ValueError("interpolated dense source bindings do not remain a partition")
 pos_err=np.linalg.norm(X[used]-Xrec,axis=1)
 # Every detailed O triangle must lie in its declared original source face.
 T=Xsrc[source["inds"][parent]]
 Q=X[F[rows]]
 v0=T[:,1]-T[:,0];v1=T[:,2]-T[:,0]
 d00=np.einsum("ij,ij->i",v0,v0);d01=np.einsum("ij,ij->i",v0,v1);d11=np.einsum("ij,ij->i",v1,v1)
 denom=d00*d11-d01*d01
 if np.any(denom<=0):raise ValueError("degenerate declared source parent triangle")
 v2=Q-T[:,None,0,:]
 d20=np.einsum("fki,fi->fk",v2,v0);d21=np.einsum("fki,fi->fk",v2,v1)
 bary1=(d11[:,None]*d20-d01[:,None]*d21)/denom[:,None]
 bary2=(d00[:,None]*d21-d01[:,None]*d20)/denom[:,None]
 bary=np.stack((1-bary1-bary2,bary1,bary2),axis=2)
 reconstructed=np.einsum("fkj,fji->fki",bary,T)
 face_pos_err=np.linalg.norm(Q-reconstructed,axis=2)
 bary_min=float(bary.min());bary_max=float(bary.max())
 signed2=(bary[:,1,1]-bary[:,0,1])*(bary[:,2,2]-bary[:,0,2])-(bary[:,1,2]-bary[:,0,2])*(bary[:,2,1]-bary[:,0,1])
 # The builder deliberately reverses source-component winding to match the physical O descriptor.
 orient_counts={"same":int(np.count_nonzero(signed2>1e-12)),"reversed":int(np.count_nonzero(signed2<-1e-12)),"nearzero":int(np.count_nonzero(np.abs(signed2)<=1e-12))}
 summary=json.loads(REPORT.read_text())
 scope=summary.get("plane",{}).get("source_basis","")
 # Anatomical cut is defined from tibia minus calcaneus anchors; this is an ankle/lower-leg descriptor.
 if "tibia" not in scope.lower() or "calcaneus" not in scope.lower():
  raise ValueError("unexpected O271 anatomical source scope; inspect report before interpreting")
 result={"schema":"numi.human.outer271-source-lineage-binding-feasibility.v1",
  "audit_script_sha256":sha(Path(__file__)),
  "python_version":sys.version,
  "input_pins":pins,
  "descriptor_semantics":"separate physical O surface descriptor; not an NHSKIN candidate",
  "anatomical_scope":"bilateral lower-leg/ankle collar, selected from tibia-minus-calcaneus landmarks; not shoulder/arm/hand",
  "source_correspondence":{
   "kind2_face_count":int(len(rows)),"unique_source_payload_face_rows":int(len(np.unique(parent))),
   "repeated_parent_faces":int(len(parent)-len(np.unique(parent))),"parent_ids_in_original_nhskin_face_row_domain":True,
   "face_parent_sequence_exact":True,"source_nodes_used":int(len(used)),"node_ancestry_kinds":mesh_types,
   "world_point_replay_max_error_m":float(pos_err.max(initial=0)),"world_point_replay_rms_error_m":float(np.sqrt(np.mean(pos_err**2))),
   "parent_triangle_barycentric_min":bary_min,"parent_triangle_barycentric_max":bary_max,
   "parent_triangle_replay_max_error_m":float(face_pos_err.max(initial=0)),"orientation_vs_source_payload":orient_counts,
   "full_dense_binding_rows_replayable":True,"max_binding_partition_error":float(np.max(np.abs(Wrec.sum(axis=1)-1))),
   "max_binding_negative_weight":float(max(0.0,-Wrec.min(initial=0)))},
  "feasibility":{
   "can_reconstruct_fine_surface_on_detailed_kind2_region_from_existing_O_ancestry":bool(bary_min>=-1e-7 and bary_max<=1+1e-7 and face_pos_err.max(initial=0)<1e-7 and pos_err.max(initial=0)<1e-7),
   "qualification":"ancestry/binding feasibility only; no pose, collision, tissue, mechanical, or mass claim",
   "limitation":"this O271 descriptor covers ankle/lower-leg patches, so it does not establish arm/hand reconstruction feasibility"}}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2))

def load_report(path):
 return json.loads(path.read_text())

if __name__=="__main__":main()
