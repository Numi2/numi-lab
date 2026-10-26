#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
D = Path("/Users/n/Work/numi-human-connected-surface-069")
C394 = D/"barycentric-connected-clearance-170/combined-clearance-394/run-001/candidate.npz"
R159 = D/"barycentric-connected-clearance-170/combined-clearance-394/run-001/audit-159/report.json"
C362 = D/"barycentric-connected-clearance-170/coherent-support-continuation-362/uniform-scale-recovery/backtrack/pass-001/candidate.npz"
OLD_REPORT = D/"barycentric-connected-clearance-170/coherent-support-continuation-391/run-001/round-001/attempt-001/fixed-boundary/report.json"
OLD_NPZ = OLD_REPORT.parent/"introduced-skin-bone-barriers.npz"
P364 = D/"D/connected-remesh-reuse-364/remap_legacy_face_constraints_364.py"
P314 = D/"publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
P313 = D/"publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
P396 = D/"D/precision-phase-one-396/propose_joint_clearance_witnesses_396.py"
P026 = D/"barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
OUT = D/"D/fixed-bone-guards-394/run-003"
EXPECTED = {
 "candidate394":"251b7e0a8c39d2e1e3ef57b2654bf4912cfb3e62bc9cc38fb9ee4d7a9a52e8c3",
 "candidate362":"814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3",
 "old_report391":"6f9e72fa56e631e7b10ba170ec7373c43d98e22aaea5bd445d55bc4adb2a39e3",
 "old_npz391":"4d45ff97bba9d20db1d9e17ff53d627f6a7901362c2971e1e91d8aeca1072610",
 "audit159_394":"a16543bc13f8fb35eec62e0f8cad6dc3b3fe587cb92366c93f33874dda8b6774",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def arrsha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise ImportError(path)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def readnpz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def json_default(value):
 if isinstance(value,np.generic): return value.item()
 if isinstance(value,Path): return value.as_posix()
 raise TypeError(f"not JSON serializable: {type(value).__name__}")
def main():
 for k,p in (("candidate394",C394),("candidate362",C362),("old_report391",OLD_REPORT),("old_npz391",OLD_NPZ),("audit159_394",R159)):
  if sha(p)!=EXPECTED[k]:raise ValueError(f"{k} SHA mismatch: {sha(p)}")
 OUT.mkdir(parents=True,exist_ok=False)
 remap=loadmod("remap_364",P364); ex=loadmod("emit_314",P314); fx=loadmod("fixed_owner_313",P313); fit=loadmod("fit_owner_026",P026); ph=loadmod("phase_one_396",P396)
 old=readnpz(C362);cur=readnpz(C394)
 points=np.asarray(cur["points_world_m"]);faces=np.asarray(cur["source_O_faces"]);of=np.asarray(old["source_O_faces"])
 if points.dtype!=np.float32 or faces.shape!=of.shape or not np.array_equal(of,old["area_reference_037_source_O_faces"]):raise ValueError("invalid 362 source topology/reference")
 if not np.array_equal(faces,cur["area_reference_037_source_O_faces"]):raise ValueError("394 current/matched-037 connectivity mismatch")
 oldreport=json.loads(OLD_REPORT.read_text());olda=readnpz(OLD_NPZ)
 if oldreport.get("output_npz_sha256")!=sha(OLD_NPZ):raise ValueError("391 report does not pin old NPZ")
 oldpoints=np.asarray(old["points_world_m"])
 ph.validate_fixed_boundary_payload(olda,oldreport,candidate_sha256=sha(C362),candidate_points=oldpoints,candidate_faces=of,npz_path=OLD_NPZ)
 histn=len(olda["skin_face_rows"])
 hist=list(zip(olda["skin_face_rows"].astype(int),olda["bank_body"].astype(int),olda["stable_bone_id"].astype(int),olda["bank_triangle_stable_id"].astype(int)))
 if len(set(hist))!=histn:raise ValueError("duplicate historical typed rows")
 changed=np.flatnonzero(np.any(of!=faces,axis=1)).astype(np.int64)
 meta=remap._metadata_changed(old,cur,len(faces))
 identity=remap.filter_constraint_rows(of,of,faces,changed,np.empty((0,2),np.int64),olda["skin_face_rows"],meta)
 keep=np.asarray(identity["retained_bone_mask"],bool)
 audit=json.loads(R159.read_text())
 bh=loadmod("bank_helper_314",ex.BANK_HELPER)
 T,ids,_g,_b,M,off,pos,keys,pins,_unused=bh.read_bank_inputs(ex.BANK_DIR,ex.BANK_EVIDENCE)
 bankkeys=tuple((int(x[0]),int(x[1])) for x in np.asarray(keys))
 current=ex.validate_audit_report(audit,candidate_sha256=sha(C394),point_count=len(points),face_count=len(faces),
  expected_bank_keys=bankkeys,expected_script_sha256=sha(ex.AUDIT_159),expected_predicate_sha256=sha(ex.EXACT_159))
 if len(bankkeys)!=185:raise ValueError("expected 185 unique typed banks")
 if len(current)!=sum(int(r["result"]["count"]) for r in audit["banks"] if r.get("result") is not None):raise ValueError("159 report pair counts mismatch")
 overlap=[]; retained=[]
 for i,(f,b,bo,t) in enumerate(hist):
  if not keep[i]:continue
  k=(b,bo,f,t)
  (overlap if k in current else retained).append(k)
 if len(set(retained))!=len(retained):raise ValueError("duplicate retained fixed guard")
 lookup=ex.typed_triangle_lookup(T,ids)
 transforms={int(b):(M[int(b)],off[int(b)],pos[int(b)]) for b in M}
 arrays=ex.make_barrier_arrays(retained,lookup,transforms)
 # 313 owner sorts by face/body/bone/triangle; persist arrays in that same record order.
 order=sorted(range(len(arrays["skin_face_rows"])),key=lambda i:(int(arrays["skin_face_rows"][i]),int(arrays["bank_body"][i]),int(arrays["stable_bone_id"][i]),int(arrays["bank_triangle_stable_id"][i])))
 arrays={k:np.asarray(v)[order].copy() for k,v in arrays.items()}
 recs=[]
 for i in range(len(order)):
  recs.append({"skin_face_row":int(arrays["skin_face_rows"][i]),"bank_body":int(arrays["bank_body"][i]),
    "stable_bone_id":int(arrays["stable_bone_id"][i]),"bank_triangle_stable_id":int(arrays["bank_triangle_stable_id"][i]),
    "fixed_triangle_world_mm":arrays["fixed_triangles_world_mm"][i]})
 pointsmm=points.astype(np.float64)*1000.
 rows,typed=ph.build_fixed_boundary_barriers(pointsmm,faces,recs,fit,fx,margin_fraction=.25)
 if len(typed)!=len(retained):raise ValueError("313 pair count changed")
 npz=OUT/"introduced-skin-bone-barriers.npz";np.savez_compressed(npz,**arrays)
 typed_report=[{"skin_face_row":f,"bank_body":b,"stable_bone_id":bo,"bank_triangle_stable_id":t} for b,bo,f,t in sorted(retained)]
 report={
  "status":"COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT","schema":"numi.fixed_skin_bone_barriers.v1",
  "filtered_builder_path":str(Path(__file__)),"filtered_builder_sha256":sha(__file__),
  "base_candidate_path":str(C394),"base_candidate_sha256":sha(C394),
  "candidate_points_world_m_array_sha256":arrsha(points),"skin_faces_array_sha256":arrsha(faces),
  "point_count":len(points),"face_count":len(faces),"candidate_points_world_m_shape":list(points.shape),"skin_faces_shape":list(faces.shape),
  "bank_count":185,"expected_bank_count":185,"audit_script_path":str(ex.AUDIT_159),"audit_script_sha256":sha(ex.AUDIT_159),
  "exact_predicate_path":str(ex.EXACT_159),"exact_predicate_sha256":sha(ex.EXACT_159),
  "bank_reader_path":str(ex.BANK_HELPER),"bank_reader_sha256":sha(ex.BANK_HELPER),
  "bank_source_pins":pins,"bank_source_paths":{n:(ex.BANK_DIR/n).as_posix() for n in pins},
  "introduced_pair_count":len(retained),"typed_pairs":typed_report,"fixed_triangles_units":"mm","fixed_triangles_frame":"world",
  "fixed_triangle_transform":"Current pinned raw bank triangles transformed to world-frame millimetres in Float64 using 314; construction-only fixed geometry.",
  "inputs":[{"role":"base","report_path":str(R159),"report_sha256":sha(R159),"candidate_path":str(C394),"candidate_sha256":sha(C394),
   "point_count":len(points),"face_count":len(faces),"candidate_points_world_m_array_sha256":arrsha(points),"source_O_faces_array_sha256":arrsha(faces),
   "introduced_or_base_pair_count":len(current)}],
  "output_npz":npz.name,"output_npz_sha256":sha(npz),"geometry_or_native_admission":False,
  "construction_scope":{
   "selection_scope":"FILTERED_HISTORICAL_391_FIXED_GUARDS_MINUS_CURRENT_394_159_PAIRS",
   "historical_barrier_report_path":str(OLD_REPORT),"historical_barrier_report_sha256":sha(OLD_REPORT),
   "historical_barrier_npz_path":str(OLD_NPZ),"historical_barrier_npz_sha256":sha(OLD_NPZ),
   "historical_barrier_base_candidate_sha256":oldreport["base_candidate_sha256"],
   "identity_source_candidate_path":str(C362),"identity_source_candidate_sha256":sha(C362),
   "current_candidate_path":str(C394),"current_candidate_sha256":sha(C394),
   "changed_connectivity_face_rows":len(changed),"metadata_or_ancestry_changed_face_rows":len(meta),
   "identity_unsafe_face_rows":len(identity["unsafe_face_rows"]),"historical_input_pair_count":histn,
   "identity_retained_before_current_pair_subtraction":int(keep.sum()),"identity_stale_rows_removed":int(histn-keep.sum()),
   "retained_pairs_removed_because_current_159_pair":len(overlap),"filtered_guard_pair_count":len(retained),
   "current_394_159_pair_count":len(current),"current_159_audit_path":str(R159),"current_159_audit_sha256":sha(R159),
   "is_complete_current_159_pair_set":False,
   "interpretation":"Construction guard subset only. Historical identities are retained solely when exact face connectivity and row-aligned source ancestry are unchanged from 362 to 394; exact pair identities in the complete current 394 159 report are removed. This subset is not a complete current 159 audit, and absence is not evidence of clearance.",
   "filter_owner_path":str(P364),"filter_owner_sha256":sha(P364),
   "filter_method":"Exact face-row connectivity diff plus P364 row-aligned face metadata and CSR source/fine ancestry equality; no nearest-row or geometric identity mapping.",
   "rebuild_owner_314_path":str(P314),"rebuild_owner_314_sha256":sha(P314),"fixed_row_owner_313_path":str(P313),"fixed_row_owner_313_sha256":sha(P313),
   "pair_owner_026_path":str(P026),"pair_owner_026_sha256":sha(P026),"rows_emitted_by_313":len(rows),"barrier_margin_fraction":.25,
   "source_old_payload_validated_against_362":True,"current_394_159_complete_typed_banks_validated":True},
  "limitations":["Filtered historical construction guard subset only, not complete current 159 surface audit.","Missing pairs do not establish clearance; complete current 159 report remains authoritative.","Construction-only; no native, physical, or anatomy qualification."]
 }
 rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,default=json_default)+"\n")
 loaded=readnpz(npz)
 records,receipt=ph.validate_fixed_boundary_payload(loaded,report,candidate_sha256=sha(C394),candidate_points=points,candidate_faces=faces,npz_path=npz)
 rebuilt,typed2=ph.build_fixed_boundary_barriers(pointsmm,faces,records,fit,fx,margin_fraction=.25)
 if len(rebuilt)!=len(rows) or len(typed2)!=len(typed):raise ValueError("396 row roundtrip changed")
 validation={"status":"PASS_396_FIXED_BOUNDARY_PAYLOAD_AND_313_ROW_VALIDATION","candidate_sha256":sha(C394),
  "npz_sha256":sha(npz),"report_sha256":sha(rp),"filtered_guard_pairs":len(records),"313_rows":len(rebuilt),
  "all_rows_zero_feasible":all(float(r["lower"])<=2e-8 for r in rebuilt),"receipt":receipt}
 vp=OUT/"validation.json";vp.write_text(json.dumps(validation,indent=2,sort_keys=True,default=json_default)+"\n")
 print(json.dumps({"candidate_sha256":sha(C394),"npz":str(npz),"npz_sha256":sha(npz),"report":str(rp),
  "report_sha256":sha(rp),"pairs":len(records),"rows":len(rebuilt),"changed_faces":len(changed),
  "metadata_changed_faces":len(meta),"stale":histn-int(keep.sum()),"current_overlap_removed":len(overlap)}))
if __name__=="__main__":main()

