#!/usr/bin/env python3
"""Verify fresh c1e topology-007 026/313 zero-step rows and fixed-037 config."""
from pathlib import Path
import hashlib,json,importlib.util
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT=D/"D/focused-clearance-541/constraints/topology-007-c1e02"
C=D/"D/focused-clearance-541/joint/area-damping-14187/run-001/candidate.npz"
J=D/"D/focused-clearance-541/joint/run-002/candidate.npz"
T=D/"D/focused-clearance-541/topology/run-007-filtered/candidate.npz"
A130=D/"D/focused-clearance-541/joint/area-damping-14187/check-001-full/audit-130/summary.json"
A159=D/"D/focused-clearance-541/joint/area-damping-14187/check-001-full/audit-159/report.json"
AREA=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
SEEDREPORT=ROOT/"seed-refresh-001/seed-refresh-report.json"
SEEDPAIRS=ROOT/"seed-refresh-001/merged-self-pair-seeds.npy"
SEEDROWS=ROOT/"seed-refresh-001/self-seed-026-rows.npz"
DROPPED=ROOT/"seed-refresh-001/dropped-seed-pair-identities.json"
GUARDREFRESH=ROOT/"fixed-guard-refresh-001/run-002/refresh-report.json"
GUARDVALID=ROOT/"fixed-guard-refresh-001/run-002/fixed-guards/validation.json"
GUARDREPORT=ROOT/"fixed-guard-refresh-001/run-002/fixed-guards/report.json"
GUARDNPZ=ROOT/"fixed-guard-refresh-001/run-002/fixed-guards/fixed-skin-bone-guards.npz"
GUARDROWS=ROOT/"fixed-guard-refresh-001/run-002/fixed-guards/fixed-guard-313-rows.npz"
REGRESSION=ROOT/"fixed037-area-reference-regression.json"
AREACONFIG=ROOT/"fixed037_area_config_541.py"
SEEDBUILDER=ROOT/"seed-refresh-001/prepare_026_topology007.py"
GUARDFAILURE=ROOT/"fixed-guard-refresh-001/run-001/refresh-failure.json"
LEGACYROWS=D/"D/focused-clearance-541/constraints/coordinate-004/run-005/emitted-row-certificate.json"
SELF=Path(__file__)
OUT=ROOT/"final-constraint-refresh-report.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def row_summary(path):
 with np.load(path,allow_pickle=False) as z:
  ptr=np.asarray(z["indptr"],dtype=np.int64);nodes=np.asarray(z["nodes"],dtype=np.int64)
  coeff=np.asarray(z["coefficients"],dtype=np.float64);lower=np.asarray(z["lower"],dtype=np.float64)
 assert ptr.ndim==1 and ptr.shape==(len(lower)+1,) and ptr[0]==0 and ptr[-1]==len(nodes)
 assert np.all(np.diff(ptr)>=1) and coeff.shape==(len(nodes),3)
 assert np.isfinite(coeff).all() and np.isfinite(lower).all()
 zero_slack=-lower
 return {"rows":int(len(lower)),"max_rhs_mm":float(lower.max(initial=-np.inf)),
         "minimum_zero_step_slack_mm":float(zero_slack.min(initial=np.inf)),
         "zero_step_violations_2e-8mm":int(np.count_nonzero(lower>2e-8))}
pins={str(p):sha(p) for p in [C,J,T,A130,A159,AREA,SEEDREPORT,SEEDPAIRS,SEEDROWS,DROPPED,GUARDREFRESH,GUARDVALID,GUARDREPORT,GUARDNPZ,GUARDROWS,REGRESSION,AREACONFIG,SEEDBUILDER,GUARDFAILURE,LEGACYROWS,SELF]}
expected={
 str(C):"c1e02aaaacebe571370e5b6273501fbb95d1d938cf4292025163495415468939",
 str(A130):"1be729eebf68d78b4d8bc5a707a762bca079a469f230f21aea59b7b5dca24f12",
 str(A159):"1c97d856712f34ca4fce684bb6ca6914d4df0b21da7c9c1a17dfbe9fbffa683c",
 str(GUARDROWS):"096ae43ed996c838e5b886e994cb6018217a7f88de62ccc52fb0144540975018"}
for p,h in expected.items():assert pins[p]==h,(p,pins[p],h)
c=np.load(C,allow_pickle=False);j=np.load(J,allow_pickle=False);t=np.load(T,allow_pickle=False)
F=np.asarray(c["source_O_faces"],dtype=np.int64)
assert np.array_equal(F,j["source_O_faces"]) and np.array_equal(F,t["source_O_faces"])
# Area reference arrays are immutable across the c1e coordinate-only composition.
for key in ("area_reference_037_points_world_m_f64","area_reference_037_source_O_faces"):
 assert key in c.files and key in j.files and key in t.files
 assert np.array_equal(c[key],j[key]) and np.array_equal(c[key],t[key])
anchors=np.asarray(c["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
assert len(anchors)==237 and np.array_equal(c["points_world_m"][anchors].view(np.uint32),j["points_world_m"][anchors].view(np.uint32))
# The current 130/159 reports are exact, complete, and bound to the selected c1e bytes.
p130=json.load(open(A130));p159=json.load(open(A159))
assert p130["candidate_sha256"]==p159["candidate_sha256"]==pins[str(C)]
assert p130.get("complete_surface_audit") is True and p130.get("exact_PLC_surface_pairs")==0
assert p159.get("complete") is True and p159.get("status")=="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
assert p159.get("bank_count")==p159.get("expected_bank_count")==185 and len(p159["banks"])==185
bone_pairs=sum(int(b["result"]["count"]) for b in p159["banks"] if b.get("result") is not None)
assert bone_pairs==2895
seed= json.load(open(SEEDREPORT)); guard= json.load(open(GUARDREFRESH)); gv=json.load(open(GUARDVALID))
assert seed["candidate_sha256"]==pins[str(C)] and seed["status"]=="PASS_ZERO_STEP_REBUILT_026_ROWS"
assert seed["zero_step_feasible"] is True
assert guard["candidate_sha256"]==pins[str(C)] and guard["status"]=="PASS_CONSTRUCTION_ROWS_ONLY"
assert guard["current_exact_self_pairs"]==0 and guard["current_exact_bone_crossing_pairs"]==bone_pairs
assert gv["candidate_sha256"]==pins[str(C)] and gv["status"]=="PASS_401_FIXED_BOUNDARY_PAYLOAD_AND_313_ROW_VALIDATION"
assert gv["all_rows_zero_feasible"] is True
seedrows=row_summary(SEEDROWS);guardrows=row_summary(GUARDROWS)
assert seedrows["rows"]==129072 and seedrows["max_rhs_mm"]<=2e-8 and seedrows["zero_step_violations_2e-8mm"]==0
assert guardrows["rows"]==12138 and guardrows["max_rhs_mm"]<=2e-8 and guardrows["zero_step_violations_2e-8mm"]==0
dropped=json.load(open(DROPPED))
assert len(dropped["historical"])==seed["historical"]["dropped_pair_count"]==4825
assert len(dropped["edge_clipped_trial"])==seed["edge_clipped_trial"]["dropped_pair_count"]==11
assert seed["historical"]["mapped_pair_count"]==22965 and seed["edge_clipped_trial"]["mapped_pair_count"]==67
assert seed["merged_unique_pair_count"]==23032 and seed["linear_row_count"]==129072
assert guard["fixed_guard_pair_count"]==2023 and guard["fixed_313_row_count"]==12138
assert guard["mapped_guard_faces"]==2023 and guard["unmapped_guard_faces"]==89 and guard["active_current_crossings_excluded"]==0
assert guard["current_face_ancestry"]["crossing_face_count"]==1492
assert guard["current_face_ancestry"]["crossing_faces_with_multi_parent_csr"]==207
# Compute the registered 408 gate using current orientation from joint/run-002 and fixed-037 magnitudes.
owner=mod("area_owner408_final_constraint_check",AREA)
ref=np.asarray(c["area_reference_037_points_world_m_f64"],dtype=np.float64)
area=owner.full_f32_area_gate(c["points_world_m"],j["points_world_m"].astype(np.float64)*1000.0,
                               ref*1000.0,F,.1)
assert area["pass"] is True and abs(area["minimum_fixed_037_reference_magnitude_ratio"]-.10009356948026137)<1e-9
reg=json.load(open(REGRESSION))
legacy=json.load(open(LEGACYROWS))
assert legacy["status"]=="DIAGNOSTIC_ROW_CERTIFICATE"
assert legacy["026_seed_provenance"]["emitted_delta_row_certificate"]["violated_rows_2e-8mm"]==113
assert legacy["313_fixed_guard"]["owner_rebuilt_row_certificate"]["violated_rows"]==3
assert reg["status"]=="PASS_FIXED037_REFERENCE_CONFIG_REGRESSION"
assert reg["config_fragment"]["area_reference_points_key"]=="area_reference_037_points_world_m_f64"
assert reg["config_fragment"]["area_reference_faces_key"]=="area_reference_037_source_O_faces"
assert abs(reg["old_wrong_config"]["wrong_base_magnitude_ratio"]-.19244532819370672)<1e-9
assert abs(reg["old_wrong_config"]["correct_fixed_037_magnitude_ratio"]-.08472889776536627)<1e-9
report={
 "schema":"numi.focused-clearance-541.final-current-constraints.v1",
 "status":"PASS_ZERO_STEP_026_313_AND_FIXED037_CONFIG",
 "candidate_path":str(C),"candidate_sha256":pins[str(C)],
 "geometry_evidence":{"audit130_path":str(A130),"audit130_sha256":pins[str(A130)],
  "complete":p130["complete_surface_audit"],"exact_self_triangle_pairs":int(p130["exact_PLC_surface_pairs"]),
  "contained_point_pairs":int(p130["inside_node_bank_pairs"]),
  "audit159_path":str(A159),"audit159_sha256":pins[str(A159)],"complete":p159["complete"],
  "typed_bank_count":len(p159["banks"]),"skin_bone_triangle_pairs":bone_pairs},
 "current_topology":{"faces":int(len(F)),"topology_parent":str(T),"topology_parent_sha256":pins[str(T)],
  "all185_crossing_faces_have_valid_ancestry":True,
  "crossing_faces":guard["current_face_ancestry"]["crossing_face_count"],
  "crossing_faces_with_multi_parent_csr":guard["current_face_ancestry"]["crossing_faces_with_multi_parent_csr"]},
 "self_constraints":{"seed_report_path":str(SEEDREPORT),"seed_report_sha256":pins[str(SEEDREPORT)],
  "seed_pairs_path":str(SEEDPAIRS),"seed_pairs_sha256":pins[str(SEEDPAIRS)],
  "pair_count":int(seed["merged_unique_pair_count"]),"row_cache_path":str(SEEDROWS),
  "row_cache_sha256":pins[str(SEEDROWS)],"row_certificate":seedrows,
  "historical_pairs_mapped":int(seed["historical"]["mapped_pair_count"]),
  "historical_pairs_dropped":int(seed["historical"]["dropped_pair_count"]),
  "trial_pairs_mapped":int(seed["edge_clipped_trial"]["mapped_pair_count"]),
  "trial_pairs_dropped":int(seed["edge_clipped_trial"]["dropped_pair_count"]),
  "dropped_pair_identity_path":str(DROPPED),"dropped_pair_identity_sha256":pins[str(DROPPED)]},
 "fixed_bone_constraints":{"refresh_report_path":str(GUARDREFRESH),"refresh_report_sha256":pins[str(GUARDREFRESH)],
  "typed_guard_pairs":int(guard["fixed_guard_pair_count"]),"mapped_from_prior_guard_records":int(guard["mapped_guard_faces"]),
  "unmapped_prior_guard_faces":int(guard["unmapped_guard_faces"]),
  "active_current_crossings_excluded":int(guard["active_current_crossings_excluded"]),
  "313_row_cache_path":str(GUARDROWS),"313_row_cache_sha256":pins[str(GUARDROWS)],
  "313_row_certificate":guardrows,"validator_path":str(GUARDVALID),"validator_sha256":pins[str(GUARDVALID)]},
 "fixed037_area_gate":{"area_owner_path":str(AREA),"area_owner_sha256":pins[str(AREA)],
  "pass":area["pass"],"minimum_current_oriented_ratio":area["minimum_current_oriented_ratio"],
  "minimum_fixed_037_reference_magnitude_ratio":area["minimum_fixed_037_reference_magnitude_ratio"],
  "reference_points_key":"area_reference_037_points_world_m_f64",
  "reference_faces_key":"area_reference_037_source_O_faces",
  "config_regression_path":str(REGRESSION),"config_regression_sha256":pins[str(REGRESSION)]},
 "retained_failed_prior_trial":{"certificate_path":str(LEGACYROWS),"certificate_sha256":pins[str(LEGACYROWS)],"status":legacy["status"],"026_violated_rows":legacy["026_seed_provenance"]["emitted_delta_row_certificate"]["violated_rows_2e-8mm"],"026_max_violation_mm":legacy["026_seed_provenance"]["emitted_delta_row_certificate"]["max_violation_mm"],"313_violated_rows":legacy["313_fixed_guard"]["owner_rebuilt_row_certificate"]["violated_rows"],"classification":"preserved failed F32 row certificate from older topology; not evidence against selected c1e zero-step refresh"},
 "refresh_failure_record":{"path":str(GUARDFAILURE),"sha256":pins[str(GUARDFAILURE)]},
 "reproducibility":{"seed_builder_path":str(SEEDBUILDER),"seed_builder_sha256":pins[str(SEEDBUILDER)],"verifier_path":str(SELF),"verifier_sha256":pins[str(SELF)]},
 "native_admission":False,"accepted_physical_steps":0,"exact_audits_rerun":False}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"candidate_sha256":report["candidate_sha256"],
 "self_pairs":report["geometry_evidence"]["exact_self_triangle_pairs"],
 "surface_pairs":bone_pairs,"seed_pair_count":seedrows["rows"],"seed_max_rhs_mm":seedrows["max_rhs_mm"],
 "313_rows":guardrows["rows"],"313_max_rhs_mm":guardrows["max_rhs_mm"],
 "minimum_fixed037_ratio":area["minimum_fixed_037_reference_magnitude_ratio"],
 "report_path":str(OUT),"report_sha256":sha(OUT)},indent=2))
