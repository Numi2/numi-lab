"""Compile the joint boundary/volume quotient using existing 806/808/677/818/849/834/1022/938 owners."""
from pathlib import Path
import json,sys,subprocess,hashlib,importlib.util,time
import numpy as np
W=Path("/Users/numi/Work/human-integrated-gaps-20261104")
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=W/"joint-boundary-rebuild-64c"
C=D/"compiler";C.mkdir(exist_ok=False)
def pin(p):
 p=Path(p).resolve();return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+"\n")
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run(name,args):
 cmd=[sys.executable,*map(str,args)];start=time.time()
 print("START",name,flush=True)
 with (C/(name+".stdout.log")).open("w") as o,(C/(name+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=o,stderr=e)
 save(C/(name+".command.json"),{"command":cmd,"returncode":r.returncode,"wall_s":time.time()-start})
 print("END",name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError((name,(C/(name+".stderr.log")).read_text()[-4500:]))
report=load(D/"role-volumes-v2/report.json")
for role in ("Skin","Fat"):
 assert report["roles"][role]["native_quality_failures"]==report["roles"][role]["nonpositive_cells"]==0
 assert report["roles"][role]["joint_source_coordinate_difference_max_m"]==0
ind=load(D/"independent-checks.json");assert ind["status"].startswith("PASS_")
assert load(D/"full-context/report.json")["status"].startswith("PASS_")
# Reuse the existing, pinned construction auditor required by the downstream state contract.
audit=H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_806_extended_250um_construction.py"
ap=C/"proposal-exact-806";logs=C/"proposal-audit-logs";logs.mkdir()
cmd=[sys.executable,str(audit),"--config",str(D/"context-config.json"),"--out",str(ap)]
with (logs/"audit806.stdout.log").open("w") as o,(logs/"audit806.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=o,stderr=e)
save(logs/"audit806.command.json",{"command":cmd,"returncode":r.returncode})
assert r.returncode==0
assert load(ap/"report.json")["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
# Exact per-material boundary/typed target proof, including original collar walls.
helper=module(H/"bounded625-local-cavity-repair/finalize_677_target_metadata_625_v2.py","target677")
wall_report=load(H/"current-native-1024-root/volume-prefix-818c/volume-prefix-818b-report.json")
wallcfg=Path(wall_report["wall_config"]["path"]);cfg=load(wallcfg)
collar=helper.load_npz(Path(cfg["collar"]["path"]));wall=Path(cfg["wall_csv"]["path"])
surf={r:helper.load_npz(D/f"candidate-{r}-typed-current.npz") for r in "OID"}
proof={}
for role in ("Skin","Fat"):
 v=helper.load_npz(Path(report["roles"][role]["path"]))
 target,*rest=helper.build_target(role,v["points_world_m_f32"],surf,wall,collar)
 assert len(target)==len(v["boundary_faces"])
 assert helper.directed_chain(target)==helper.directed_chain(v["boundary_faces"])
 proof[role]={"target_faces":len(target),"closed_oriented_target_matches_complete_volume_boundary":True}
save(C/"boundary-proof.json",{"status":"PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY","roles":proof,"producer":pin(__file__),"physical_admission":False,"native_steps":0})
base=H/"current-native-1024-root/volume-prefix-818c/verified-state.json"
state={"status":"PASS_SOURCE_ONLY_CONNECTED_BOUNDARY_VOLUME_QUALITY",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","source_state":pin(base),
 "source_surfaces":{r:pin(D/f"candidate-{r}-typed-current.npz") for r in "OID"},
 "source_audit_status":"PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",
 "source_crossing_audit_status":"PASS_CURRENT_806_EXACT_SOURCE_CONTEXT",
 "surface_crossings":0,"exact_context":pin(ap/"report.json"),
 "independent_boundary_proof":pin(C/"boundary-proof.json"),"independent_replay":pin(D/"independent-checks.json"),
 "joint_volume_bridge":pin(D/"role-volumes-v2/report.json")}
for role in ("Skin","Fat"):state[role+"_current"]=dict(report["roles"][role])
sp=C/"verified-state.json";save(sp,state)
ver=module(H/"verify_current_source_evidence_808_v26a.py","source808")
save(C/"source-proof-808.json",ver.verify_current_source_evidence(sp))
run("prefix818",[H/"bounded625-local-cavity-repair/finalize_818c_volume_only_current_source_prefix_808.py","--state",sp,"--wall-config",wallcfg,"--output",C/"volume-prefix"])
current=C/"volume-prefix/verified-state.json"
run("manifest849",[H/"canonical-volume-parent-unions-849-nativepreflight/build_current_manifest_849d_explicit_reference_base.py","--base-state",base,"--current-state",current,"--manifest-out",C/"manifest.json","--report-out",C/"manifest849-report.json","--validate-only"])
run("canonical834",[H/"canonical-volume-parent-unions-834-nativepreflight/canonical_volume_parent_unions_834.py","--manifest",C/"manifest.json","--output-dir",C/"canonical","--report",C/"canonical834-report.json"])
old=load(H/"current-shell-1022-native/actual-001/receipt.json")["inputs"]
run("shell1022",[H/"current-shell-1022-native/build_current_shell_1022.py","--state",current,"--rim-receipt",old["rim_receipt"]["path"],"--parent-view-manifest",old["parent_view_manifest"]["path"],"--output",C/"shells"])
run("handoff938",[H/"current-role-p1-support-938-1025-active-source-map/current_fine_volume_to_217_938_818c_849d_active_rows.py",
 "--skin-volume-npz",C/"canonical/skin-canonical-parent-union.npz","--fat-volume-npz",C/"canonical/fat-canonical-parent-union.npz",
 "--skin-shell-npz",C/"shells/skin-shell.npz","--fat-shell-npz",C/"shells/fat-shell.npz",
 "--inner-surface-npz",D/"candidate-I-typed-current.npz","--current-state",current,
 "--canonical-834-report",C/"canonical834-report.json","--canonical-849-report",C/"manifest849-report.json",
 "--volume-prefix-818b-report",C/"volume-prefix/volume-prefix-818b-report.json","--shell-assembly-receipt",C/"shells/receipt.json","--output",C/"handoff938"])
print("EXISTING_COMPILER_HANDOFF_COMPLETE_NOT_NATIVE_ADMISSION",flush=True)
