#!/usr/bin/env python3
import argparse, datetime, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

REPO=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=REPO/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/hand-actuation-qualification-001"
BIN=REPO/"build-integrated-validation-20261019/profile-build-039/bin/numi-human-native"
RIGID=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
MUSCLES=RIGID.with_name("myosim-fullbody-muscle-reference.nhmyo")
EQ=RIGID.with_name("myosim-fullbody-joint-equalities.nheq")
BONES=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.nhbones")
SUPPORT=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/skin729-flat-bed-native003-witness-registration-871/myosim-fullbody-resting-bed-support.nhcnt")
TENDON=E/"results/left-apl-envelope-under-2p2/numi-human-tendon-attachments.nhtendon"
BINARY_SHA="b9606aac3fd7f2705da1c4da1a9e9f54be693e95269406cc825b89a4bae4f7ad"
TENDON_SHA="fb9b6a92081c218b4306a7dd08f39303d424cd92695ecfd5666b7a8a829b48c6"
SKIN=E.parent/"finger-profile-060/replacement-bilateral-hands038.nhskin"
SKIN_SHA="e30bd35e4c7cee48433c50cafb0987f23987a95d377034aecc017aea28d28ae1"
SOURCE_MANIFEST=Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
SOURCE_MANIFEST_SHA="c794687f4b60dab86b510658365fba74b29bcf749b1665f0ec232d4b6e451054"
SOURCE_ARCHIVE_SHA="280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
PINNED={
 str(RIGID):"2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4",
 str(MUSCLES):"e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b",
 str(EQ):"0e3c593792a8c4ccd619f1999f6685b68e6ff0fd5ae5f6f0bca66b594a0e9b88",
 str(BONES):"43a31ef4771e2a383ae2c7caab1b994f0c28076c2a43501860cd1ba3fe7708c0",
 str(SUPPORT):"4c31ce250a63bd730c066f6f92571295c761ddb51215a3eeb09275bc42d6992d",
}
COHORTS={
 "thumb":[256,257,258,259,319,320,321,322],
 "bilateral_finger_flexion":list(range(242,250))+list(range(305,313)),
 "bilateral_finger_extension":list(range(250,256))+list(range(313,319)),
 "bilateral_wrist_flexion":[237,238,239,300,301,302],
 "bilateral_wrist_extension":[234,235,236,297,298,299],
 "bilateral_intrinsic_radial":[261,264,267,270,324,327,330,333],
 "bilateral_intrinsic_ulnar":[263,266,269,272,326,329,332,335],
 "bilateral_opponens":[260,323],
}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--cohort",choices=COHORTS,default="thumb")
 ap.add_argument("--output",type=Path)
 ap.add_argument("--execute",action="store_true",help="execute one bounded native transaction; default is hash-qualified plan only")
 ap.add_argument("--binary",type=Path,default=BIN,help="native executable; custom binaries require their exact build manifest")
 ap.add_argument("--source-manifest",type=Path,help="exact isolated native build manifest for --binary")
 ap.add_argument("--binary-sha256",default=BINARY_SHA)
 ap.add_argument("--tendon-payload",type=Path,default=TENDON)
 ap.add_argument("--tendon-sha256",default=TENDON_SHA)
 ap.add_argument("--step-count",type=int,default=1)
 ap.add_argument("--timestep-seconds",type=float,default=0.000125)
 ap.add_argument("--activation",type=float,default=0.5)
 ap.add_argument("--timeout-seconds",type=float,default=60.0,help="native subprocess wall limit; at most 300 seconds")
 ap.add_argument("--skin-payload",type=Path,help="optional pinned NHSKIN1 pose-snapshot input")
 ap.add_argument("--skin-sha256",default=SKIN_SHA)
 ap.add_argument("--initial-reference-posture",action="store_true",help="use the native fixed bilateral hand reference seed q36=q74=0.1 rad; initial configuration only, no pose lock")
 a=ap.parse_args()
 if not (1<=a.step_count<=800):raise SystemExit("--step-count must be 1..800")
 if not (1e-6<=a.timestep_seconds<=1e-3):raise SystemExit("--timestep-seconds must be 1e-6..1e-3")
 if not (0<a.activation<=1):raise SystemExit("--activation must be >0 and <=1")
 if not (0<a.timeout_seconds<=300):raise SystemExit("--timeout-seconds must be >0 and <=300")
 binary=a.binary.resolve()
 if binary!=BIN.resolve() and a.source_manifest is None:raise SystemExit("custom --binary requires --source-manifest")
 tendon_payload=a.tendon_payload
 files=[binary,RIGID,MUSCLES,EQ,BONES,SUPPORT,tendon_payload]
 if a.skin_payload is not None:files.append(a.skin_payload)
 hashes={str(p):sha(p) for p in files}
 for path,expected in PINNED.items():
  if hashes[path]!=expected:raise SystemExit("pinned input SHA mismatch: "+path+" got="+hashes[path]+" expected="+expected)
 if hashes[str(binary)]!=a.binary_sha256:raise SystemExit("binary SHA mismatch: "+hashes[str(binary)]+" expected="+a.binary_sha256)
 if hashes[str(tendon_payload)]!=a.tendon_sha256:raise SystemExit("tendon SHA mismatch: "+hashes[str(tendon_payload)]+" expected="+a.tendon_sha256)
 if a.skin_payload is not None and hashes[str(a.skin_payload)]!=a.skin_sha256:raise SystemExit("skin SHA mismatch: "+hashes[str(a.skin_payload)]+" expected="+a.skin_sha256)
 build_manifest={}
 if a.source_manifest is not None:
  source_manifest=a.source_manifest.resolve()
  manifest_sha=sha(source_manifest);hashes[str(source_manifest)]=manifest_sha
  build_manifest=json.loads(source_manifest.read_text())
  if Path(build_manifest.get("binary_path", "")).resolve()!=binary:raise SystemExit("source manifest binary path mismatch")
  if build_manifest.get("binary_sha256")!=hashes[str(binary)]:raise SystemExit("source manifest binary SHA mismatch")
 if sha(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:raise SystemExit("source manifest SHA mismatch")
 hashes[str(SOURCE_MANIFEST)]=SOURCE_MANIFEST_SHA
 src=json.loads(SOURCE_MANIFEST.read_text())
 if src.get("source",{}).get("archive_sha256")!=SOURCE_ARCHIVE_SHA:raise SystemExit("source manifest archive SHA mismatch")
 myo=MUSCLES.read_bytes()
 if myo[:8]!=b"NHMYO2\0\0" or myo[44:76].hex()!=SOURCE_ARCHIVE_SHA:raise SystemExit("NHMYO2 embedded source archive identity mismatch")
 ids=COHORTS[a.cohort]
 output_dir=a.output or (E/"results"/(a.cohort+f"-dt{a.timestep_seconds:g}-steps{a.step_count}"))
 argv=[str(binary),str(RIGID),str(MUSCLES),str(BONES),str(output_dir),
  "--muscle-step-seconds",format(a.timestep_seconds,".9g"),"--muscle-step-count",str(a.step_count),"--muscle-activation",format(a.activation,".9g"),
  "--support-contact-payload",str(SUPPORT),"--joint-equality-payload",str(EQ),
  "--tendon-payload",str(tendon_payload),"--selected-tendon-control"]
 if a.initial_reference_posture:
  if a.cohort=="thumb":raise SystemExit("the bilateral thumb certificate currently excludes initial-reference posture; use a non-certificate cohort")
  argv += ["--initial-reference-posture"]
 for i in ids:argv += ["--activated-source-muscle-index",str(i)]
 if a.skin_payload is not None:argv += ["--skin-payload",str(a.skin_payload)]
 if a.cohort=="thumb":argv += ["--bilateral-thumb-tendon-certificate"]
 if any(x in argv for x in ["--resting-rigid-hands","--pose-q","--root-pose","--persistent-metal-stand","--stand-root-assistance"]):raise SystemExit("forbidden pose/lock/assistance flag")
 receipt={"schema":"numi.human.hand-actuation-invocation.v2","argv":argv,"cwd":str(REPO),
  "binary_path":str(binary),"binary_sha256":hashes[str(binary)],"input_hashes":hashes,"cohort":a.cohort,
  "step_count":a.step_count,"timestep_seconds":a.timestep_seconds,"activation":a.activation,"timeout_seconds":a.timeout_seconds,
  "initial_configuration":"bilateral_hand_q36_0p1_q74_0p1" if a.initial_reference_posture else "source_default_q",
  "initial_configuration_only":bool(a.initial_reference_posture),"ongoing_q_lock":False,
  "initial_configuration_flags":["--initial-reference-posture"] if a.initial_reference_posture else [],
  "pose_or_lock_flags":[],
  "tendon_payload":str(tendon_payload),"skin_payload":str(a.skin_payload) if a.skin_payload else None,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
 # Bind native runs to exact working-source identity.
 if a.source_manifest is not None:
  receipt["build_source_manifest_path"]=str(source_manifest)
  receipt["build_source_manifest_sha256"]=manifest_sha
  shadow_header=build_manifest.get("shadow_header",{})
  receipt["native_source_sha256"]={build_manifest.get("source_path"):build_manifest.get("source_sha256"),shadow_header.get("path"):shadow_header.get("sha256")}
  receipt["build_dependency_count"]=build_manifest.get("dependency_closure",{}).get("dependency_path_count")
  receipt["build_dependency_file_sha256"]=build_manifest.get("dependency_closure",{}).get("depfile_sha256")
  receipt["linked_library_identity"]=build_manifest.get("linked_library")
  receipt["build_head"]=build_manifest.get("head")
 else:
  cpp=REPO/"apps/numilab_human_myosim_visual_probe.mm"
  header=REPO/"include/metalrobo/mujoco_muscle_gpu.h"
  receipt["git_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
  tracked_diff=subprocess.check_output(["git","diff","--binary","HEAD","--",str(cpp.relative_to(REPO)),str(header.relative_to(REPO))],cwd=REPO)
  receipt["tracked_source_diff_sha256"]=hashlib.sha256(tracked_diff).hexdigest()
  receipt["native_source_sha256"]={str(x):sha(x) for x in (cpp,header)}
 receipt["qualification_helper_sha256"]=sha(Path(__file__))
 print(json.dumps({"preflight":receipt,"argv":argv},indent=2))
 if not a.execute:
  print("plan_only=true");return 0
 output_dir.mkdir(parents=True,exist_ok=False)
 (output_dir/"invocation.json").write_text(json.dumps(receipt,indent=2)+"\n")
 env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_HUMAN_RESTING_","NUMI_HUMAN_SUPPORTED_"))}
 for key in ("NUMI_HUMAN_WRIST_CONTOUR_ENABLE","NUMI_HUMAN_WRIST_CONTOUR_SCALE"):env.pop(key,None)
 env["NUMI_HUMAN_SUPPORTED_INITIALIZATION"]="0"
 native_started=time.perf_counter()
 try:
  r=subprocess.run(argv,cwd=REPO,env=env,text=True,capture_output=True,timeout=a.timeout_seconds,check=False)
  stdout_text,stderr_text,rc,execution=r.stdout,r.stderr,r.returncode,"completed"
 except subprocess.TimeoutExpired as e:
  stdout_text=e.stdout.decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or "")
  stderr_text=e.stderr.decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")
  rc,execution=124,"timeout"
 native_wall_elapsed_seconds=round(time.perf_counter()-native_started,6)
 (output_dir/"stdout.log").write_text(stdout_text);(output_dir/"stderr.log").write_text(stderr_text)
 line=next((x for x in stdout_text.splitlines() if x.startswith("selected_hand_wrist_actuation=")),None)
 reasons=[]
 if rc:reasons.append("native_exit_"+str(rc))
 try: t=json.loads(line.split("=",1)[1]) if line else {}
 except Exception:t={};reasons.append("telemetry_malformed")
 rows=t.get("q_rows",[]); muscles=t.get("selected_muscles",[])
 nonfinite=[]
 def scrub(value,path="telemetry"):
  if isinstance(value,float) and not math.isfinite(value):nonfinite.append(path);return None
  if isinstance(value,dict):return {k:scrub(v,path+"."+str(k)) for k,v in value.items()}
  if isinstance(value,list):return [scrub(v,path+"["+str(i)+"]") for i,v in enumerate(value)]
  return value
 t=scrub(t)
 if nonfinite:reasons.append("nonfinite_telemetry:"+",".join(nonfinite[:8]))
 rows=t.get("q_rows",[]); muscles=t.get("selected_muscles",[])
 def finite_num(value):
  try:
   x=float(value)
   return x if math.isfinite(x) else 0.0
  except (ValueError,TypeError):return 0.0
 if not line:reasons.append("telemetry_missing")
 initial_state={}
 if a.initial_reference_posture:
  terminal_line=next((x for x in stdout_text.splitlines() if x.startswith("stand_terminal_state=")),None)
  try:
   terminal=json.loads(terminal_line.split("=",1)[1]) if terminal_line else {}
   initial_q=terminal.get("initial_q",[])
   if len(initial_q)<=74:reasons.append("initial_reference_state_q_missing")
   else:
    initial_state={"q36_rad":float(initial_q[36]),"q74_rad":float(initial_q[74]),"source":"stand_terminal_state.initial_q"}
    if not all(math.isfinite(initial_state[k]) for k in ("q36_rad","q74_rad")):reasons.append("initial_reference_state_nonfinite")
    if abs(initial_state["q36_rad"]-0.1)>1e-6 or abs(initial_state["q74_rad"]-0.1)>1e-6:reasons.append("initial_reference_posture_not_honored")
  except Exception:reasons.append("initial_reference_state_malformed")
 if t.get("schema")!="numi.human.selected-hand-wrist-actuation.v1":reasons.append("wrong_schema")
 if t.get("matched_baseline")!="same_input_zero_activation_increment":reasons.append("no_matched_baseline")
 if t.get("replay")!="bitwise":reasons.append("replay_not_bitwise")
 if t.get("accepted_steps")!=a.step_count:reasons.append("no_expected_steps_accepted")
 if t.get("target_coordinate_count")!=44 or len(rows)!=44:reasons.append("not_44_coordinates")
 if [m.get("source_actuator_index") for m in muscles]!=ids:reasons.append("actuator_ids_mismatch")
 for m in muscles:
  if finite_num(m.get("commanded_activation_increment",0))<=0:reasons.append("activation_missing_"+str(m.get("source_actuator_index")))
  if abs(finite_num(m.get("source_force_increment_n",0)))<=0:reasons.append("source_force_missing_"+str(m.get("source_actuator_index")))
  signed=finite_num(m.get("source_force_n",0)); increment=finite_num(m.get("source_force_increment_n",0))
  m["baseline_source_force_n_derived"]=signed-increment
  m["source_pull_magnitude_n"]=max(-signed,0.0)
  if "tendon_tension_normalized" in m:
   normalized=finite_num(m.get("tendon_tension_normalized"))
   tendon_n=finite_num(m.get("tendon_tension_n"))
   if normalized<=0:reasons.append("normalized_tendon_tension_missing_"+str(m.get("source_actuator_index")))
   if tendon_n<=0:reasons.append("tendon_force_n_missing_"+str(m.get("source_actuator_index")))
   m["source_pull_vs_tendon_force_abs_difference_n"]=abs(m["source_pull_magnitude_n"]-tendon_n)
   m["normalized_tendon_tension"]=normalized
   m["tendon_tension_n"]=tendon_n
   m["normalized_equilibrium_residual"]=finite_num(m.get("normalized_equilibrium_residual"))
   m["baseline_normalized_equilibrium_residual"]=finite_num(m.get("baseline_normalized_equilibrium_residual"))
  else:
   # Preserve the older bad-unit output separately; never call it newtons.
   legacy_normalized=finite_num(m.get("tendon_tension_n",0))
   if legacy_normalized<=0:reasons.append("normalized_tendon_tension_missing_"+str(m.get("source_actuator_index")))
   m["normalized_tendon_tension_from_legacy_field"]=legacy_normalized
   m.pop("tendon_tension_n",None)
 if a.cohort=="thumb": target={43,44,45,46,81,82,83,84}
 elif "finger" in a.cohort: target=set(range(47,63))|set(range(85,101))
 elif "intrinsic" in a.cohort: target={48,52,56,60,86,90,94,98}
 elif "opponens" in a.cohort: target={43,44,81,82}
 else:target={41,42,79,80}
 tr=[r for r in rows if r.get("q_index") in target]
 tq=sum(abs(finite_num(r.get("selected_muscle_generalized_torque_increment_l1_nm",0))) for r in tr)
 q=sum(abs(finite_num(r.get("q_increment_rad",0))) for r in tr);v=sum(abs(finite_num(r.get("v_increment_rad_s",0))) for r in tr)
 endpoint_fields=("accepted_q_rad","baseline_q_rad","accepted_v_rad_s","baseline_v_rad_s","source_dof_flags","source_lower_limit_rad","source_upper_limit_rad")
 endpoint_rows=[]
 for row in tr:
  if any(k not in row for k in endpoint_fields):
   reasons.append("endpoint_qv_or_limit_telemetry_missing_q"+str(row.get("q_index")));continue
  flags=int(row["source_dof_flags"])
  accepted_q=finite_num(row["accepted_q_rad"]); baseline_q=finite_num(row["baseline_q_rad"])
  accepted_v=finite_num(row["accepted_v_rad_s"]); baseline_v=finite_num(row["baseline_v_rad_s"])
  lower=finite_num(row["source_lower_limit_rad"]); upper=finite_num(row["source_upper_limit_rad"])
  limited=bool(flags & (1 << 2))
  inside=(lower<=accepted_q<=upper) if limited else None
  baseline_inside=(lower<=baseline_q<=upper) if limited else None
  accepted_violation=max(0.0,lower-accepted_q,accepted_q-upper) if limited else 0.0
  baseline_violation=max(0.0,lower-baseline_q,baseline_q-upper) if limited else 0.0
  accepted_bound=lower if accepted_q<lower else upper
  baseline_bound=lower if baseline_q<lower else upper
  accepted_slop=16.0*1.1920928955078125e-7*max(1.0,abs(accepted_q),abs(accepted_bound)) if limited else 0.0
  baseline_slop=16.0*1.1920928955078125e-7*max(1.0,abs(baseline_q),abs(baseline_bound)) if limited else 0.0
  accepted_owner_ok=(accepted_violation<=accepted_slop) if limited else None
  baseline_owner_ok=(baseline_violation<=baseline_slop) if limited else None
  if limited and (not accepted_owner_ok or not baseline_owner_ok):reasons.append("joint_position_limit_outside_owner_slop_q"+str(row.get("q_index")))
  endpoint_rows.append({"q_index":row["q_index"],"accepted_q_rad":accepted_q,"baseline_q_rad":baseline_q,
   "accepted_v_rad_s":accepted_v,"baseline_v_rad_s":baseline_v,"source_dof_flags":flags,
   "position_limit_enabled":limited,"source_lower_limit_rad":lower,"source_upper_limit_rad":upper,
   "accepted_q_within_source_limits":inside,"baseline_q_within_source_limits":baseline_inside,
   "accepted_limit_violation_rad":accepted_violation,"baseline_limit_violation_rad":baseline_violation,
   "accepted_owner_slop_rad":accepted_slop,"baseline_owner_slop_rad":baseline_slop,
   "accepted_q_within_owner_slop":accepted_owner_ok,"baseline_q_within_owner_slop":baseline_owner_ok})
 if not tr:reasons.append("target_rows_missing")
 if tq<=0:reasons.append("no_target_torque")
 if q<=0 and v<=0:reasons.append("no_target_motion")
 cert_line=next((x for x in stdout_text.splitlines() if x.startswith("numi_human_bilateral_thumb_tendon_force_transfer=")),None)
 cert={}
 if a.cohort=="thumb":
  if cert_line is None or "numi_human_bilateral_thumb_tendon_force_transfer=ok" not in cert_line:reasons.append("thumb_certificate_missing");
  else:
   import re
   cert={k:v.strip("\"") for k,v in re.findall(r'([A-Za-z0-9_]+)=("(?:[^"]*)"|\S+)',cert_line)}
  if cert.get("replay")!="bitwise":reasons.append("thumb_replay_not_bitwise")
  if cert.get("rollback")!="consumer_rejection_preserved_result":reasons.append("thumb_rollback_missing")
  if cert.get("borrowed_consumer")!="same_command_buffer_exact_snapshot":reasons.append("thumb_consumer_missing")
  try:
   if int(cert.get("accepted_steps","0"))!=a.step_count:reasons.append("thumb_steps_not_accepted")
   if not math.isfinite(float(cert.get("bilateral_force_relative_difference","nan"))) or float(cert.get("bilateral_force_relative_difference","nan"))>0.05:reasons.append("thumb_force_parity_or_nonfinite")
   for k in ("right_represented_endpoint_force_increment_l1_n","left_represented_endpoint_force_increment_l1_n","right_q_increment_l1_rad","left_q_increment_l1_rad","right_v_increment_l1_rad_s","left_v_increment_l1_rad_s"):
    value=float(cert.get(k,"nan"))
    if not math.isfinite(value) or value<=0:reasons.append("thumb_no_or_nonfinite_"+k)
  except (ValueError,TypeError):reasons.append("thumb_numeric_certificate_malformed")
 res={"schema":"numi.human.hand-actuation-result.v2","status":"PASS" if not reasons else "FAIL","reasons":reasons,
  "native_wall_elapsed_seconds":native_wall_elapsed_seconds,
  "thumb_certificate_fields":cert,"execution":execution,"exit_code":rc,"cohort":a.cohort,"source_actuator_indices":ids,
  "selected_hand_wrist_actuation":t,"initial_state":initial_state,"target_q_indices":sorted(target),"target_q_rows":tr,
  "target_generalized_torque_l1_nm":tq,"target_q_increment_l1_rad":q,"target_v_increment_l1_rad_s":v,
  "target_endpoint_rows":endpoint_rows,
  "target_endpoint_q_bounds_rad":{"min":min((x["accepted_q_rad"] for x in endpoint_rows),default=None),"max":max((x["accepted_q_rad"] for x in endpoint_rows),default=None)},
  "target_endpoint_v_bounds_rad_s":{"min":min((x["accepted_v_rad_s"] for x in endpoint_rows),default=None),"max":max((x["accepted_v_rad_s"] for x in endpoint_rows),default=None)},
  "target_joint_limit_audit":{"limited_coordinate_count":sum(x["position_limit_enabled"] for x in endpoint_rows),"within_limits":all(x["accepted_q_within_source_limits"] is not False for x in endpoint_rows),
   "within_existing_owner_slop":all(x["accepted_q_within_owner_slop"] is not False and x["baseline_q_within_owner_slop"] is not False for x in endpoint_rows),
   "owner_slop_rule":"16 * FLT_EPSILON * max(1, abs(position), abs(bound)) from mrNumiHumanPositionLimitSlop",
   "minimum_signed_limit_margin_rad":min((min(x["accepted_q_rad"]-x["source_lower_limit_rad"],x["source_upper_limit_rad"]-x["accepted_q_rad"]) for x in endpoint_rows if x["position_limit_enabled"]),default=None),
   "worst_margin_q_index":min((x for x in endpoint_rows if x["position_limit_enabled"]),key=lambda x:min(x["accepted_q_rad"]-x["source_lower_limit_rad"],x["source_upper_limit_rad"]-x["accepted_q_rad"]),default={}).get("q_index"),
   "rows":endpoint_rows},
  "skin_pose_snapshot_files":sorted(str(x) for x in output_dir.glob("*.skin-poses.json")),
  "finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "boundary":f"matched finite-horizon selected-tendon force and free hand/wrist response ({a.step_count} steps at {a.timestep_seconds:g} s) with zero-increment baseline; skin pose output is geometry readback only; not whole-body FEM, sustained grasp, Brain control, or clinical validation"}
 if a.skin_payload is not None and not res["skin_pose_snapshot_files"]:res["reasons"].append("skin_pose_snapshot_missing");res["status"]="FAIL"
 (output_dir/"result.json").write_text(json.dumps(res,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"output":str(output_dir),"result":{k:v for k,v in res.items() if k not in ("selected_hand_wrist_actuation","target_q_rows")}},indent=2))
 return 0 if res["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
