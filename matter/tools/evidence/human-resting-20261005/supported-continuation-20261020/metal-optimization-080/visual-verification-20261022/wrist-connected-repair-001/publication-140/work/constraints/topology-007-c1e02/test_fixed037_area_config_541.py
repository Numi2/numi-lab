#!/usr/bin/env python3
"""Regression: 524 must use fixed-037 magnitudes and the immediate base orientation."""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT=D/"D/focused-clearance-541/constraints/topology-007-c1e02"
HELPER=ROOT/"fixed037_area_config_541.py"
CAND=D/"D/focused-clearance-541/joint/area-damping-14187/run-001/candidate.npz"
JOINT=D/"D/focused-clearance-541/joint/run-002/candidate.npz"
OLD_BASE=D/"D/focused-clearance-541/saved-reuse/run-004-focused/candidate.npz"
OLD_TRIAL=D/"D/focused-clearance-541/constraints/coordinate-004/run-005/backtrack-373/pass-003/candidate.npz"
RUNNER=D/"publication-137/sources/typed-point-regression-524/run_local_witness_backtracking_524.py"
OWNER=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OUT=ROOT/"fixed037-area-reference-regression.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helper=mod("fixed037_area_config_541",HELPER)
tree=ast.parse(RUNNER.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="validate_area_reference")
ns={"np":np}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(RUNNER),"exec"),ns)
validate_area_reference=ns["validate_area_reference"]
owner=mod("area_owner408_fixed037_regression",OWNER)
cfg=helper.apply_fixed037_reference({"maximum_exact_audits":8},CAND)
assert cfg["area_reference_points_key"]=="area_reference_037_points_world_m_f64"
assert cfg["area_reference_faces_key"]=="area_reference_037_source_O_faces"
z=np.load(CAND,allow_pickle=False);cur=np.load(JOINT,allow_pickle=False)
F=np.asarray(z["source_O_faces"],dtype=np.int64)
ref=np.asarray(z[cfg["area_reference_points_key"]],dtype=np.float64)
rf=np.asarray(z[cfg["area_reference_faces_key"]],dtype=np.int64)
assert np.array_equal(F,rf)
c0,den,arearef=validate_area_reference(cur["points_world_m"].astype(float),F,ref,rf)
trial=np.asarray(z["points_world_m"],dtype=np.float32)
cross=np.cross(trial[F[:,1]].astype(float)-trial[F[:,0]],trial[F[:,2]].astype(float)-trial[F[:,0]])
oriented=np.einsum("ij,ij->i",cross,c0)/den
magnitude=np.linalg.norm(cross,axis=1)/arearef
gate=owner.full_f32_area_gate(trial,cur["points_world_m"].astype(float)*1000,ref*1000,F,.1)
assert gate["pass"] and abs(float(oriented.min())-gate["minimum_current_oriented_ratio"])<1e-12
assert abs(float(magnitude.min())-gate["minimum_fixed_037_reference_magnitude_ratio"])<1e-12
assert abs(float(magnitude.min())-.10009356948026137)<1e-9

# Reproduce the previous wrong configuration to ensure the regression catches it.
ob=np.load(OLD_BASE,allow_pickle=False);ot=np.load(OLD_TRIAL,allow_pickle=False)
of=np.asarray(ob["source_O_faces"],dtype=np.int64); ox=np.asarray(ob["points_world_m"],dtype=np.float32)
oref=np.asarray(ob["area_reference_037_points_world_m_f64"],dtype=np.float64)
_,old_den,old_refmag=validate_area_reference(ox.astype(float),of,ox.astype(float),of)
ocross=np.cross(np.asarray(ot["points_world_m"])[of[:,1]].astype(float)-np.asarray(ot["points_world_m"])[of[:,0]],
                np.asarray(ot["points_world_m"])[of[:,2]].astype(float)-np.asarray(ot["points_world_m"])[of[:,0]])
wrong_magnitude=np.linalg.norm(ocross,axis=1)/old_refmag
true_gate=owner.full_f32_area_gate(ot["points_world_m"],ox.astype(float)*1000,oref*1000,of,.1)
assert float(wrong_magnitude.min())>.1 and not true_gate["pass"]
assert abs(float(wrong_magnitude.min())-.19244532819370672)<1e-9
assert abs(float(true_gate["minimum_fixed_037_reference_magnitude_ratio"])-.08472889776536627)<1e-9
rec={"schema":"numi.focused-clearance-541.fixed037-area-reference-regression.v1",
 "status":"PASS_FIXED037_REFERENCE_CONFIG_REGRESSION",
 "runner_sha256":sha(RUNNER),"area_owner_sha256":sha(OWNER),
 "config_fragment":cfg,"candidate_sha256":sha(CAND),
 "selected_candidate_gate":{"pass":gate["pass"],
  "minimum_current_oriented_ratio":gate["minimum_current_oriented_ratio"],
  "minimum_fixed_037_reference_magnitude_ratio":gate["minimum_fixed_037_reference_magnitude_ratio"]},
 "old_wrong_config":{"candidate_sha256":sha(OLD_TRIAL),"wrong_base_magnitude_ratio":float(wrong_magnitude.min()),
  "correct_fixed_037_magnitude_ratio":true_gate["minimum_fixed_037_reference_magnitude_ratio"],
  "correct_gate_pass":true_gate["pass"],"failing_fixed037_face_rows":true_gate["bad_fixed_037_reference_face_rows_first_64"]},
 "native_admission":False,"accepted_physical_steps":0}
OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":rec["status"],"config_fragment":cfg,
 "selected_min_fixed037_ratio":gate["minimum_fixed_037_reference_magnitude_ratio"],
 "old_wrong_ratio":float(wrong_magnitude.min()),
 "old_correct_fixed037_ratio":true_gate["minimum_fixed_037_reference_magnitude_ratio"],
 "failing_face":true_gate["bad_fixed_037_reference_face_rows_first_64"][0]},indent=2))
