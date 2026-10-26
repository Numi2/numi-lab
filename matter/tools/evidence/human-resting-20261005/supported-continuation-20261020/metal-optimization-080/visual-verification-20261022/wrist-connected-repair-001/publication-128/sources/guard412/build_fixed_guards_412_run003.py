#!/usr/bin/env python3
"""Refresh construction-only 313 fixed-bone guards for selected 410 coordinates.

The guard identity is the union of the previously validated 942-pair guard subset
and every exact typed 159 crossing newly appearing from 408 to 410, less all
crossings in the complete current 410 report. Absence from this subset is not
evidence of clearance.
"""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
C408=D/"D/postfit-backtrack-408/run-001/backtrack-373/pass-003/candidate.npz"
A408=D/"D/postfit-backtrack-408/run-001/backtrack-373/pass-003/audit-159/report.json"
G410R=D/"D/fixed-bone-guards-410/run-002/report.json"
G410N=G410R.parent/"introduced-skin-bone-barriers.npz"
C410=D/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz"
A410=D/"D/postfit-backtrack-410/run-001/backtrack-373/pass-002/audit-159/report.json"
P314=D/"publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
P313=D/"publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
P401=D/"D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
P026=D/"publication-101/evidence/204/source/fit/self_separation_026.py"
OUT=D/"D/fixed-bone-guards-412/run-003"
PINS={
 "c408":"a661a0ac9ff4690a5e67b1dc06753289b7cf267c188d8b812b929297ae7fecbe",
 "a408":"b498988d21224bdcfd23e47b9330656a0789d14c165d71fc66dd20766f24ccba",
 "g410r":"edafeea29a52d76f374ff5f99886c03e31ed99f454a25781c7e634ec5f9e7350",
 "g410n":"711957268a457a17071fdffbc5bb05344edea6d18b28cba9783c6774628fe9e8",
 "c410":"141ca8ff15a4377092e198c9d7d52d08034541cbf295979e7b539c34d508a090",
 "a410":"f8aed46baed83a7260ce12e874baedeb143e3bdf25be19798b2a94cfc83ab0e8",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def mod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def audit_set(report):
 j=json.loads(Path(report).read_text()); s=set()
 for b in j["banks"]:
  for f,t in ((b.get("result") or {}).get("triangle_pairs") or []):
   s.add((int(b["body"]),int(b["stable_bone_id"]),int(f),int(t)))
 return j,s
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 for k,p in (("c408",C408),("a408",A408),("g410r",G410R),("g410n",G410N),("c410",C410),("a410",A410)):
  got=sha(p)
  if got!=PINS[k]: raise ValueError(f"{k} SHA mismatch: expected {PINS[k]}, got {got}")
 old,cur=load(C408),load(C410)
 points=cur["points_world_m"]; faces=cur["source_O_faces"]
 if points.dtype!=np.float32 or points.shape!=(20889,3) or faces.shape!=(41733,3):
  raise ValueError("410 candidate point/face schema mismatch")
 if old["points_world_m"].dtype!=np.float32 or old["points_world_m"].shape!=points.shape:
  raise ValueError("408 baseline point schema mismatch")
 if old["source_O_faces"].dtype!=faces.dtype or old["source_O_faces"].tobytes()!=faces.tobytes():
  raise ValueError("410 source face identity differs from 408; cannot reuse face-row guards")
 if cur["area_reference_037_source_O_faces"].dtype!=faces.dtype or cur["area_reference_037_source_O_faces"].tobytes()!=faces.tobytes():
  raise ValueError("410 current faces differ from embedded matched-037 reference faces")
 phase=mod("phase401_guard412",P401); fit=mod("fit026_guard412",P026)
 ex=mod("em314_guard412",P314); fx=mod("fx313_guard412",P313)
 oldr=json.loads(G410R.read_text()); oldn=load(G410N)
 old_arrays,old_receipt=phase.validate_fixed_boundary_payload(oldn,oldr,candidate_sha256=PINS["c408"],candidate_points=old["points_world_m"],candidate_faces=old["source_O_faces"],npz_path=G410N)
 oldset=set()
 for f,b,bo,t in zip(oldn["skin_face_rows"],oldn["bank_body"],oldn["stable_bone_id"],oldn["bank_triangle_stable_id"]):
  key=(int(b),int(bo),int(f),int(t))
  if key in oldset: raise ValueError("duplicate historical 410 guard identity")
  oldset.add(key)
 if len(oldset)!=942: raise ValueError(f"expected 942 validated historical pairs, got {len(oldset)}")
 j408,s408=audit_set(A408); j410,s410=audit_set(A410)
 for label,j,expected in (("408",j408,PINS["c408"]),("410",j410,PINS["c410"])):
  if (j.get("complete") is not True or j.get("bank_count")!=185 or j.get("expected_bank_count")!=185
      or j.get("candidate_sha256")!=expected or j.get("surface_point_count")!=20889 or j.get("surface_face_count")!=41733):
   raise ValueError(f"{label} complete 185-bank audit incomplete or mismatched")
 if len(s408)!=3302 or len(s410)!=3268:
  raise ValueError(f"unexpected 408/410 exact typed crossing totals {len(s408)}/{len(s410)}")
 new=s410-s408; resolved=s408-s410
 if (new|resolved)!=(s408^s410) or new&s408 or resolved&s410:
  raise ValueError("typed pair set difference is inconsistent")
 union=oldset|resolved
 removed=union&s410
 kept=union-s410
 if removed|kept!=union or removed&kept: raise ValueError("current crossing filter partition failed")
 if not resolved.issubset(kept): raise ValueError("resolved 408-to-410 typed pairs were not retained as guards")
 # Load the pinned exact bank owner and verify every one of the 185 typed bank keys.
 bank=mod("bankreader_guard412",ex.BANK_HELPER)
 T,ids,_g,_b,M,off,pos,bkeys,bpins,_=bank.read_bank_inputs(ex.BANK_DIR,ex.BANK_EVIDENCE)
 bank_keys=tuple((int(x[0]),int(x[1])) for x in np.asarray(bkeys))
 if len(bank_keys)!=185 or len(set(bank_keys))!=185: raise ValueError("source bank inventory is not 185 unique typed banks")
 validated=ex.validate_audit_report(j410,candidate_sha256=PINS["c410"],point_count=len(points),face_count=len(faces),expected_bank_keys=bank_keys,expected_script_sha256=sha(ex.AUDIT_159),expected_predicate_sha256=sha(ex.EXACT_159))
 if validated!=s410: raise ValueError("validated 410 exact pair set differs from extracted 159 identities")
 transforms={int(b):(M[int(b)],off[int(b)],pos[int(b)]) for b in M}
 arrays=ex.make_barrier_arrays(sorted(kept),ex.typed_triangle_lookup(T,ids),transforms)
 order=sorted(range(len(arrays["skin_face_rows"])),key=lambda i:(int(arrays["skin_face_rows"][i]),int(arrays["bank_body"][i]),int(arrays["stable_bone_id"][i]),int(arrays["bank_triangle_stable_id"][i])))
 arrays={k:np.asarray(v)[order].copy() for k,v in arrays.items()}
 records=[{"skin_face_row":int(arrays["skin_face_rows"][i]),"bank_body":int(arrays["bank_body"][i]),"stable_bone_id":int(arrays["stable_bone_id"][i]),"bank_triangle_stable_id":int(arrays["bank_triangle_stable_id"][i]),"fixed_triangle_world_mm":arrays["fixed_triangles_world_mm"][i]} for i in range(len(order))]
 rows,typed=phase.build_fixed_boundary_barriers(points.astype(np.float64)*1000.,faces,records,fit,fx,margin_fraction=.25)
 if len(rows)!=6*len(kept) or len(typed)!=len(kept): raise ValueError("313 row count mismatch")
 OUT.mkdir(parents=True,exist_ok=False)
 npz=OUT/"introduced-skin-bone-barriers.npz";np.savez_compressed(npz,**arrays)
 typedlist=[{"skin_face_row":f,"bank_body":b,"stable_bone_id":bo,"bank_triangle_stable_id":t} for b,bo,f,t in sorted(kept)]
 removedlist=[{"skin_face_row":f,"bank_body":b,"stable_bone_id":bo,"bank_triangle_stable_id":t,
               "source":"historical_guard_408" if (b,bo,f,t) in oldset else "resolved_408_vs_410",
               "reason":"PRESENT_IN_COMPLETE_CURRENT_410_159_AUDIT"} for b,bo,f,t in sorted(removed)]
 report={
  "status":"COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT","schema":"numi.fixed_skin_bone_barriers.v1",
  "filtered_builder_path":str(Path(__file__)),"filtered_builder_sha256":sha(__file__),
  "base_candidate_path":str(C410),"base_candidate_sha256":PINS["c410"],
  "candidate_points_world_m_array_sha256":arrsha(points),"skin_faces_array_sha256":arrsha(faces),
  "candidate_points_world_m_shape":list(points.shape),"skin_faces_shape":list(faces.shape),"point_count":len(points),"face_count":len(faces),
  "bank_count":185,"expected_bank_count":185,"bank_source_pins":bpins,"bank_source_paths":{n:(ex.BANK_DIR/n).as_posix() for n in bpins},
  "exact_predicate_path":str(ex.EXACT_159),"exact_predicate_sha256":sha(ex.EXACT_159),"audit_script_path":str(ex.AUDIT_159),"audit_script_sha256":sha(ex.AUDIT_159),
  "bank_reader_path":str(ex.BANK_HELPER),"bank_reader_sha256":sha(ex.BANK_HELPER),"fixed_triangles_units":"mm","fixed_triangles_frame":"world",
  "output_npz":npz.name,"output_npz_sha256":sha(npz),"introduced_pair_count":len(kept),"typed_pairs":typedlist,"geometry_or_native_admission":False,
  "inputs":[
   {"role":"base","candidate_path":str(C410),"candidate_sha256":PINS["c410"],"report_path":str(A410),"report_sha256":PINS["a410"],"pair_count":len(s410)},
   {"role":"historical_guard_subset_410_at_408","candidate_path":str(C408),"candidate_sha256":PINS["c408"],"report_path":str(G410R),"report_sha256":PINS["g410r"],"npz_path":str(G410N),"npz_sha256":PINS["g410n"],"pair_count":len(oldset)},
   {"role":"baseline_complete_surface_408","candidate_path":str(C408),"candidate_sha256":PINS["c408"],"report_path":str(A408),"report_sha256":PINS["a408"],"pair_count":len(s408)}
  ],
  "construction_scope":{
   "selection_scope":"HISTORICAL_942_GUARDS_AT_408_UNION_EXACT_408_MINUS_410_RESOLVED_TYPED_PAIRS_MINUS_CURRENT_COMPLETE_410_159",
   "historical_guard_pairs":len(oldset),"new_typed_pairs_410_vs_408":len(new),"resolved_typed_pairs_408_vs_410":len(resolved),"resolved_pairs_added_as_zero_feasible_guards":len(resolved&kept),
   "new_typed_pairs_in_current_410":len(new&s410),"old_guard_pairs_filtered_as_current":len(oldset&s410),
   "union_pairs_before_current_filter":len(union),"current_410_pairs_removed":len(removed),"filtered_guard_pairs":len(kept),
   "rows_emitted_by_313":len(rows),"current_410_audit_complete_typed_banks_validated":True,
   "current_410_audit_path":str(A410),"current_410_audit_sha256":PINS["a410"],
   "baseline_408_pair_count":len(s408),"current_410_pair_count":len(s410),
   "matched_037_faces_equal":True,"face_rows_identical_408_to_410":True,
   "interpretation":"Incremental construction guard subset only. Complete current 410 159 report remains authoritative; current typed crossings are excluded from zero-feasible guard constraints, and absence from this subset does not establish clearance."
  },
  "new_pairs_410_vs_408":[{"bank_body":b,"stable_bone_id":bo,"skin_face_row":f,"bank_triangle_stable_id":t,"current_crossing":True,"constraint_emitted":False} for b,bo,f,t in sorted(new)],
  "resolved_pairs_408_vs_410":[{"bank_body":b,"stable_bone_id":bo,"skin_face_row":f,"bank_triangle_stable_id":t,"current_crossing":False,"constraint_emitted":((b,bo,f,t) in kept)} for b,bo,f,t in sorted(resolved)],
  "removed_current_410_pairs":removedlist,
  "limitations":["Incremental construction guard subset; complete current 410 159 report remains authoritative.","Absence from this subset does not establish clearance.","No native, physical, or anatomy qualification."]
 }
 rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n")
 payload=load(npz)
 rec,receipt=phase.validate_fixed_boundary_payload(payload,report,candidate_sha256=PINS["c410"],candidate_points=points,candidate_faces=faces,npz_path=npz)
 rebuilt,typed2=phase.build_fixed_boundary_barriers(points.astype(np.float64)*1000.,faces,rec,fit,fx,margin_fraction=.25)
 zero=all(float(x["lower"])<=2e-8 for x in rebuilt)
 if not zero or len(rebuilt)!=len(rows) or len(typed2)!=len(typed):raise ValueError("401/313 zero-baseline validation failed")
 vp=OUT/"validation.json"
 vp.write_text(json.dumps({"status":"PASS_401_FIXED_BOUNDARY_PAYLOAD_AND_313_ROW_VALIDATION","candidate_sha256":PINS["c410"],"npz_sha256":sha(npz),"report_sha256":sha(rp),"filtered_guard_pairs":len(rec),"313_rows":len(rebuilt),"all_rows_zero_feasible":zero,"receipt":receipt},indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":"PASS","candidate":str(C410),"npz":str(npz),"npz_sha256":sha(npz),"report":str(rp),"report_sha256":sha(rp),"validation":str(vp),"validation_sha256":sha(vp),"previous_guard_pairs":len(oldset),"new_410_vs_408":len(new),"removed_408_vs_410":len(resolved),"new_pairs_filtered_current":len(new&s410),"old_guard_pairs_current":len(oldset&s410),"current_pairs":len(s410),"final_pairs":len(rec),"rows":len(rebuilt)},indent=2))
if __name__=="__main__":main()
