from __future__ import annotations
import hashlib, importlib, json, sys, time, types
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/Work/numi-human-connected-surface-069")
OUT=ROOT/"joint-interface-facet-contraction-165/output-015"
OUT.mkdir(parents=True,exist_ok=False)
CHECKPOINT=ROOT/"connected-volume-extraction-157/true-o-foundation-159/run-029-closed-volume-field-repair/run-003-nonzero-regularization/output-002/regularized-foundation-checkpoint.npz"
CAND=ROOT/"joint-interface-facet-contraction-165/output-012/common-field-gauge-candidate.npz"
CUTTER=ROOT/"joint-interface-facet-extraction-164/canonical/matter/tools/human_skin_fat_layer_reconstruction.py"
PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def qstats(points,facets):
 tri=np.asarray(points,dtype=np.float32)[np.asarray(facets,dtype=np.int64)].astype(np.float64)
 a=tri[:,1]-tri[:,0]; b=tri[:,2]-tri[:,0]; c=tri[:,2]-tri[:,1]
 den=np.maximum.reduce([np.einsum("ij,ij->i",a,a),np.einsum("ij,ij->i",b,b),np.einsum("ij,ij->i",c,c)])
 return np.linalg.norm(np.cross(a,b),axis=1)/np.maximum(den,np.finfo(float).tiny)
def audit(P,F):
 package="numilab_human"; pkg=types.ModuleType(package)
 pkg.__path__=[str(PRED.parent),str(PRED.parents[3]/"incremental-exact-audit-owner-253/src/numilab_human")]
 sys.modules[package]=pkg
 mod=importlib.import_module("numilab_human.cardiac_cavity_intersections")
 records,_=mod.float32_surface_records(np.asarray(P,dtype=np.float32),np.asarray(F,dtype=np.int64),source_seam_groups=())
 return mod._audit_self_pairs_touching_faces(records,set(range(len(F))))
def main():
 t0=time.monotonic()
 for p in (CHECKPOINT,CAND,CUTTER,PRED):
  if not p.is_file(): raise FileNotFoundError(p)
 cp=np.load(CHECKPOINT,allow_pickle=False); c=np.load(CAND,allow_pickle=False)
 P=cp["points_world_m"]; T=cp["tetrahedra_zero_based"].astype(np.int64)
 I=c["skin_levelset_m"].astype(np.float64); D=c["total_levelset_m"].astype(np.float64)
 if P.dtype!=np.float32 or I.shape!=(len(P),) or D.shape!=(len(P),): raise ValueError("full-domain source shape/dtype mismatch")
 if np.any(D-I<=0): raise ValueError("candidate fat depth sign failed")
 spec=importlib.util.spec_from_file_location("joint_facet_cutter_164",CUTTER)
 cutter=importlib.util.module_from_spec(spec); sys.modules[spec.name]=cutter; spec.loader.exec_module(cutter)
 allids=np.arange(len(T),dtype=np.int64)
 result=cutter.extract_joint_zero_level_facets(P,T,I,D,global_vertex_ids=np.arange(len(P),dtype=np.int64),parent_tet_ids=allids)
 Psurf=np.asarray(result.points_float32,dtype=np.float32); F=np.asarray(result.facets,dtype=np.int64)
 interfaces=np.asarray(result.interface_ids,dtype=np.int8)
 q=qstats(Psurf,F); self_audit=audit(Psurf,F)
 quality={}
 for iid,label in ((1,"skin"),(2,"fat")):
  m=interfaces==iid; qi=q[m]
  quality[label]={"facet_count":int(m.sum()),"q_min":float(qi.min()) if len(qi) else None,
   "q_below_1e-5":int(np.count_nonzero(qi<1e-5)),"q_nonpositive_or_nonfinite":int(np.count_nonzero(~np.isfinite(qi)|(qi<=0)))}
 np.savez_compressed(OUT/"full-after-facets.npz",points=Psurf,facets=F,interface_ids=interfaces,
  facet_parent_indptr=result.facet_parent_indptr,facet_parent_tet_rows=result.facet_parent_tet_rows)
 report={"status":"FULL_AFTER_SURFACE_AUDIT_COMPLETE","construction_only":True,"physical_or_native_admission":False,
  "inputs":{"checkpoint_sha256":sha(CHECKPOINT),"candidate_fields_path":str(CAND),"candidate_fields_sha256":sha(CAND),
   "cutter_sha256":sha(CUTTER),"exact_self_predicate_sha256":sha(PRED),"source_tet_count":int(len(T))},
  "output":{"path":str(OUT/"full-after-facets.npz"),"sha256":sha(OUT/"full-after-facets.npz"),
   "facet_count":int(len(F)),"interface_counts":{"skin":int(np.count_nonzero(interfaces==1)),"fat":int(np.count_nonzero(interfaces==2))}},
  "quality_by_interface":quality,
  "exact_global_self":{"count":int(self_audit["count"]),"examples":self_audit.get("triangle_pairs",[])[:100],
   "complete_surface_pair_audit":True},
  "passed_surface_gates":bool(self_audit["count"]==0 and all(x["q_below_1e-5"]==0 and x["q_nonpositive_or_nonfinite"]==0 for x in quality.values())),
  "runtime_seconds":time.monotonic()-t0}
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__": main()

