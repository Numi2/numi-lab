#!/usr/bin/env python3
"""Prepare/execute/assess matched fresh-process full-shared-FEM joint cohorts.

CPU-only commands: list, prepare, self-test, assess. The execute command
requires --coordinator-authorized. It modifies only output paths, 4/8-step
horizon at 125 us, activation, selected source-muscle IDs, and the diagnostic
common-failure receipt path. It does not change model assets or physics flags.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

REPO=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
HERE=Path(__file__).resolve().parent
DEFAULT_BASE=REPO/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/selector6-scalar-inverse-fix-001/results/root0-activation0-fixed-scalar-8steps-125us/invocation.json"
SOURCE=Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
RUNS=HERE/"results"
SELECTED="--activated-source-muscle-index"
OLD_RIGID="2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
OLD_EQ="0e3c593792a8c4ccd619f1999f6685b68e6ff0fd5ae5f6f0bca66b594a0e9b88"
PROJECTED_RIGID="c10924a45efc1cfb3916bd22baabeeca491da5e412bbaad37b1328fe364db53d"
REQ=("--persistent-metal-stand","--selected-tendon-control","--mechanics-only","--resting-scene","--skin-payload","--muscle-step-seconds","--muscle-step-count","--muscle-activation")
FORBID=("--stand-root-assistance","--stand-remove-assistance","--stand-deterministic-replay","--resting-rigid-hands","--initial-reference-joint-q")
# q/v are source NHRIGID2 indices. Sign is source-oracle direction at the
# pinned source pose, diagnostic only and never a physical pass threshold.
P={
"elbow-flexors":([232,233,295,296],["BRA","BRD","BRA_l","BRD_l"],[("right elbow",39,38,"+",[0,2.269]),("left elbow",77,76,"+",[0,2.269])],["Retained as a runner fixture; not ROM qualification."]),
"shoulder-supraspinatus":([213,276],["SUPSP","SUPSP_l"],[("right shoulder elevation",36,35,"+",[0,3.142]),("left shoulder elevation",74,73,"+",[0,3.142])],["Primary elevation only; q37/q38 and q75/q76 remain untested."]),
"shoulder-deltoid2":([211,274],["DELT2","DELT2_l"],[("right shoulder elevation",36,35,"-",[0,3.142]),("left shoulder elevation",74,73,"-",[0,3.142])],["Source direction depends on posture; primary elevation only."]),
"knee-flexors":([342,343,367,368,382,383,407,408],["bflh_r","bfsh_r","semimem_r","semiten_r","bflh_l","bfsh_l","semimem_l","semiten_l"],[("right knee",106,105,"+",[0,2.0944]),("left knee",120,119,"+",[0,2.0944])],["Requires reviewed projected dependent-knee envelope; old 2c78/0e3c pair conflicts at q114/q128."]),
"knee-extensors":([365,373,374,375,405,413,414,415],["recfem_r","vasint_r","vaslat_r","vasmed_r","recfem_l","vasint_l","vaslat_l","vasmed_l"],[("right knee",106,105,"-",[0,2.0944]),("left knee",120,119,"-",[0,2.0944])],["Requires projected envelope and authorized one-time flexed seed; retained q106/q120=0 at lower limit. No ongoing lock."]),
"ankle-dorsiflexors":([371,411],["tibant_r","tibant_l"],[("right ankle",109,108,"+",[-.698132,.523599]),("left ankle",123,122,"+",[-.698132,.523599])],["Tibialis anterior only; no EDL/EHL co-selection."]),
"ankle-plantarflexors":([348,349,369,388,389,409],["gaslat_r","gasmed_r","soleus_r","gaslat_l","gasmed_l","soleus_l"],[("right ankle",109,108,"-",[-.698132,.523599]),("left ankle",123,122,"-",[-.698132,.523599])],["Only ankle coordinates are response targets."]),
"mtp-flexors-nonsaturated":([346,347,387],["fdl_r","fhl_r","fhl_l"],[("right aggregate MTP/toes",111,110,"+",[-.523599,.523599]),("left aggregate MTP/toes",125,124,"+",[-.523599,.523599])],["q111/q125 are aggregate toes-body coordinates; no individual toe/IP DOFs. Left FDL 386 omitted because baseline activation was 1.0 and prior cohort lacked an effective increment."]),
"mtp-extensors":([344,345,384,385],["edl_r","ehl_r","edl_l","ehl_l"],[("right aggregate MTP/toes",111,110,"-",[-.523599,.523599]),("left aggregate MTP/toes",125,124,"-",[-.523599,.523599])],["q111/q125 are aggregate toes-body coordinates; no individual toe/IP DOFs. EDL/EHL also cross the ankle."]),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def J(p): return json.loads(Path(p).read_text())
def at(argv,flag):
 ix=[i for i,x in enumerate(argv) if x==flag]
 if len(ix)!=1: raise RuntimeError(f"expected one {flag}, got {ix}")
 return ix[0]
def val(argv,flag): return argv[at(argv,flag)+1]
def source_sha():
 m=J(SOURCE); muscles=m.get("muscles",[])
 if len(muscles)!=416: raise RuntimeError(f"expected 416 muscles, found {len(muscles)}")
 for c,(ids,names,_,_) in P.items():
  for i,n in zip(ids,names):
   if muscles[i].get("source_actuator_index")!=i or muscles[i].get("name")!=n: raise RuntimeError(f"source mapping mismatch {c}:{i}/{n}")
 return sha(SOURCE)
def artifact(x,k):
 if not isinstance(x,dict) or not Path(x.get("path","")).is_file() or sha(Path(x["path"]))!=x.get("sha256"): raise RuntimeError(f"{k} hash/path mismatch")
def load_base(path):
 path=Path(path).resolve(); b=J(path); a=list(b.get("argv",[]))
 if b.get("activation")!=0.0 or float(val(a,"--muscle-activation"))!=0: raise RuntimeError("base must be zero-activation full-scene invocation")
 for f in REQ:
  if f not in a: raise RuntimeError(f"base missing {f}")
 for f in FORBID:
  if f in a: raise RuntimeError(f"base contains disallowed {f}")
 if float(val(a,"--muscle-step-seconds"))!=.000125 or int(val(a,"--muscle-step-count")) not in (4,8): raise RuntimeError("base must use 4/8 steps at 125us")
 e=b.get("simulation_environment_override",{})
 for k in ("NUMI_HUMAN_RESTING_SKIN_FAT_FEM","NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM","NUMI_HUMAN_RESTING_PASSIVE_FASCIA"):
  if e.get(k)!="1": raise RuntimeError(f"base missing shared-FEM owner {k}")
 if "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT" not in e: raise RuntimeError("base missing diagnostic receipt")
 for k in ("binary","private_runtime_library","metallib"): artifact(b.get(k),k)
 ins=b.get("inputs_verified",{})
 if not ins: raise RuntimeError("base missing input hashes")
 for p,h in ins.items():
  if not Path(p).is_file() or sha(Path(p))!=h: raise RuntimeError(f"input hash mismatch: {p}")
 return path,b
def strip_ids(a):
 z=[];i=0
 while i<len(a):
  if a[i]==SELECTED:
   if i+1>=len(a): raise RuntimeError("truncated source-ID flag")
   i+=2
  else: z.append(a[i]);i+=1
 return z
def make_argv(base,c,steps,activation,out):
 a=list(base["argv"])
 original_output=Path(a[4]).resolve()
 if original_output!=Path(base.get("output_directory",""),"output").resolve(): raise RuntimeError("base argv[4] does not match recorded base output directory")
 fresh_output=(out/"output").resolve()
 if fresh_output==original_output: raise RuntimeError("refusing to reuse retained base argv[4]")
 if fresh_output.exists(): raise RuntimeError(f"refusing to overwrite existing native output {fresh_output}")
 a[4]=str(fresh_output)
 if Path(a[4]).resolve()==original_output: raise RuntimeError("native argv[4] still points at retained output")
 a[at(a,"--muscle-step-count")+1]=str(steps)
 a[at(a,"--muscle-activation")+1]=format(activation,".9g")
 a=strip_ids(a)
 ids=P[c][0]; a.extend(y for i in ids for y in (SELECTED,str(i)))
 if [int(a[i+1]) for i,x in enumerate(a[:-1]) if x==SELECTED]!=ids: raise RuntimeError("selected ID transform failed")
 if float(val(a,"--muscle-step-seconds"))!=.000125: raise RuntimeError("dt changed")
 return a
def knee_blockers(base,c):
 if not c.startswith("knee-"): return []
 ins=base["inputs_verified"]
 rigid=next((h for p,h in ins.items() if p.endswith(".nhrigid")),None)
 eq=next((h for p,h in ins.items() if p.endswith(".nheq")),None)
 if rigid==OLD_RIGID and eq==OLD_EQ: out=["known q114/q128 dependent-knee limit conflict in retained source pair"]
 elif rigid!=PROJECTED_RIGID: out=["projected dependent-knee envelope identity not verified"]
 else: out=[]
 if c=="knee-extensors": out.append("q106/q120 start at lower limit; no authorized one-time flexed seed in retained base invocation")
 return out
def prepare(a):
 bp,b=load_base(a.base_invocation); sh=source_sha()
 if a.steps not in (4,8) or not (math.isfinite(a.activation) and 0<a.activation<=1): raise RuntimeError("invalid horizon or activation")
 rd=Path(a.output_dir).resolve() if a.output_dir else RUNS/f"shared-fem-016-{a.cohort}-{a.steps}steps-125us"
 if rd.exists(): raise RuntimeError(f"refusing overwrite {rd}")
 rd.mkdir(parents=True)
 ids,names,targets,limits=P[a.cohort]
 identity={k:b[k] for k in ("binary","private_runtime_library","metallib")}
 pair={"schema":"numi.human.full-shared-fem-joint-cohort-pair.v1","cohort":a.cohort,"source_actuator_ids":ids,"source_actuator_names":names,
       "source_manifest_path":str(SOURCE),"source_manifest_sha256":sh,"base_invocation_path":str(bp),"base_invocation_sha256":sha(bp),
       "binary_identity":identity,"verified_inputs":b["inputs_verified"],"steps":a.steps,"dt_s":.000125,
       "simulated_seconds":a.steps*.000125,"preflight_blockers":knee_blockers(b,a.cohort),
       "targets":[{"joint":n,"q_index":q,"v_index":v,"source_oracle_direction":d,"source_range_rad":r,"direction_is_acceptance_gate":False} for n,q,v,d,r in targets],
       "limitations":limits,"pairing":"fresh-process zero-activation full-scene control and same selected IDs at loaded activation",
       "scope":"short full-shared-FEM muscle response measurement; not ROM, physiology, grasp, or production qualification",
       "physics_tolerances_iterations_materials_contact_settings_changed":False}
 (rd/"pair.json").write_text(json.dumps(pair,indent=2,sort_keys=True)+"\n")
 for cond,act in (("baseline",0.0),("loaded",a.activation)):
  td=rd/cond;td.mkdir(); argv=make_argv(b,a.cohort,a.steps,act,td)
  env=dict(b["simulation_environment_override"]);key="NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT";old=env[key];env[key]=str(td/"common-field-failure.json")
  inv={"schema":"numi.human.full-shared-fem-joint-cohort-invocation.v1","condition":cond,"cohort":a.cohort,"activation":act,
       "steps":a.steps,"step_seconds":.000125,"argv":argv,"cwd":str(REPO),"binary":identity["binary"],
       "private_runtime_library":identity["private_runtime_library"],"metallib":identity["metallib"],
       "inputs_verified":b["inputs_verified"],"simulation_environment_override":env,
       "environment_changed":{key:{"from":old,"to":env[key]}},
       "source_actuator_ids":ids,"source_actuator_names":names,"source_manifest_sha256":sh,"base_invocation_sha256":pair["base_invocation_sha256"],
       "physics_tolerances_iterations_materials_contact_settings_changed":False,"timeout_seconds":a.timeout,"scope":pair["scope"]}
  (td/"invocation.json").write_text(json.dumps(inv,indent=2,sort_keys=True)+"\n");(td/"argv.json").write_text(json.dumps(argv,indent=2)+"\n")
 pair["baseline_invocation_sha256"]=sha(rd/"baseline/invocation.json");pair["loaded_invocation_sha256"]=sha(rd/"loaded/invocation.json")
 (rd/"pair.json").write_text(json.dumps(pair,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"prepared_only":True,"run_dir":str(rd),"cohort":a.cohort,"ids":ids,"steps":a.steps,"blockers":pair["preflight_blockers"],
                   "baseline_invocation_sha256":pair["baseline_invocation_sha256"],"loaded_invocation_sha256":pair["loaded_invocation_sha256"]},indent=2))
 return 0
def validate(td,pair):
 ip=td/"invocation.json"; inv=J(ip)
 if sha(ip)!=pair[f"{inv['condition']}_invocation_sha256"]: raise RuntimeError("invocation hash changed")
 if inv["base_invocation_sha256"]!=pair["base_invocation_sha256"] or inv["source_manifest_sha256"]!=pair["source_manifest_sha256"] or source_sha()!=pair["source_manifest_sha256"]: raise RuntimeError("base/source identity changed")
 for k in ("binary","private_runtime_library","metallib"): artifact(inv[k],k)
 for p,h in inv["inputs_verified"].items():
  if not Path(p).is_file() or sha(Path(p))!=h: raise RuntimeError(f"input changed: {p}")
 return inv
def execute(a):
 rd=Path(a.run_dir).resolve();pair=J(rd/"pair.json")
 if pair["preflight_blockers"]: raise RuntimeError("blocked: "+"; ".join(pair["preflight_blockers"]))
 if not a.coordinator_authorized: raise RuntimeError("requires --coordinator-authorized after GPU slot is assigned")
 for cond in ("baseline","loaded"):
  td=rd/cond;inv=validate(td,pair)
  actual=Path(inv["argv"][4]).resolve()
  expected=(td/"output").resolve()
  if actual!=expected: raise RuntimeError(f"argv[4] mismatch: {actual} != fresh {expected}")
  if actual==Path(J(pair["base_invocation_path"])["argv"][4]).resolve(): raise RuntimeError("refusing retained base argv[4]")
  if actual.exists(): raise RuntimeError(f"refusing to overwrite existing native output {actual}")
  if any(x.name not in ("invocation.json","argv.json") for x in td.iterdir()): raise RuntimeError(f"refusing overwrite {td}")
  env=os.environ.copy();env.update(inv["simulation_environment_override"]);start=time.monotonic();timed=False
  try:
   with (td/"stdout.log").open("wb") as o,(td/"stderr.log").open("wb") as e:
    p=subprocess.run(inv["argv"],cwd=inv["cwd"],env=env,stdout=o,stderr=e,timeout=a.timeout,check=False)
   rc=p.returncode
  except subprocess.TimeoutExpired: rc=None;timed=True
  r={"schema":"numi.human.full-shared-fem-joint-cohort-result.v1","condition":cond,"native_exit_code":rc,"timed_out":timed,
     "wall_seconds":time.monotonic()-start,"invocation_sha256":sha(td/"invocation.json"),
     "stdout_path":str(td/"stdout.log"),"stderr_path":str(td/"stderr.log")}
  (td/"result.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps(r),flush=True)
  if rc!=0 or timed: return 1
 return 0
def rec(lines,prefix):
 x=[l[len(prefix)+1:] for l in lines if l.startswith(prefix+"=")]
 if len(x)!=1: raise RuntimeError(f"missing/duplicate {prefix}")
 return json.loads(x[0])
def kv(line):
 z={}
 for p in line.split(","):
  if "=" in p:
   k,v=p.split("=",1);z[k]=v
 return z
def parse_run(td,pair):
 inv=validate(td,pair);result=J(td/"result.json");lines=(td/"stdout.log").read_text(errors="replace").splitlines()
 terminal=rec(lines,"stand_terminal_state");selected=rec(lines,"selected_hand_wrist_actuation")
 audits=[l for l in lines if l.startswith("resting_FEM_terminal_audit=")]
 if len(audits)!=1: raise RuntimeError("missing shared FEM terminal audit")
 audit=audits[0];status=audit.split("=",1)[1].split(" ",1)[0]
 nodes={};nt=0;minj=math.inf;maxj=-math.inf
 for line in lines:
  if line.startswith("resting_FEM_node,"):
   r=kv(line[len("resting_FEM_node,"):]);key=(r["object"],int(r["local"]))
   if key in nodes: raise RuntimeError(f"duplicate FEM node {key}")
   v=[float(r[f"accepted_{x}"]) for x in "xyz"]
   if not all(math.isfinite(x) for x in v): raise RuntimeError("nonfinite FEM node")
   nodes[key]=v
  elif line.startswith("resting_FEM_tet,"):
   j=float(kv(line[len("resting_FEM_tet,"):])["J"])
   if not math.isfinite(j): raise RuntimeError("nonfinite FEM J")
   nt+=1;minj=min(minj,j);maxj=max(maxj,j)
 for key,n in (("q",129),("initial_q",129),("v",128),("initial_v",128)):
  if len(terminal.get(key,[]))!=n or not all(math.isfinite(float(x)) for x in terminal[key]): raise RuntimeError(f"missing/nonfinite full {key}")
 if terminal.get("step_count")!=pair["steps"] or terminal.get("timestep_seconds")!=pair["dt_s"]: raise RuntimeError("terminal horizon mismatch")
 if selected.get("accepted_steps")!=pair["steps"] or selected.get("matched_baseline")!="same_input_zero_activation_increment": raise RuntimeError("selected owner accepted-state telemetry missing")
 muscles=selected.get("selected_muscles",[]);ids=[int(m["source_actuator_index"]) for m in muscles]
 if ids!=pair["source_actuator_ids"]: raise RuntimeError("selected muscle output IDs mismatch")
 route=[]
 for m in muscles:
  for k in ("accepted_activation","commanded_activation","commanded_activation_increment","baseline_accepted_activation","source_force_n","tendon_tension_n","tendon_tension_normalized","accepted_path_length_m","accepted_fiber_length_m","normalized_equilibrium_residual"):
   if k not in m or not math.isfinite(float(m[k])): raise RuntimeError(f"missing/nonfinite selected muscle {k}")
  inc=float(m["commanded_activation_increment"]); obs=float(m["accepted_activation"])-float(m["baseline_accepted_activation"])
  expected=0 if inv["condition"]=="baseline" else float(inv["activation"])
  route.append({"id":int(m["source_actuator_index"]),"accepted_activation":m["accepted_activation"],
                "baseline_accepted_activation":m["baseline_accepted_activation"],"activation_increment":inc,
                "accepted_minus_internal_baseline":obs,"increment_identity":math.isclose(obs,inc,rel_tol=0,abs_tol=1e-7),
                "matches_requested_increment":math.isclose(inc,expected,rel_tol=0,abs_tol=1e-7),
                "source_force_n":m["source_force_n"],"tendon_tension_n":m["tendon_tension_n"],
                "normalized_tension":m["tendon_tension_normalized"],"path_length_m":m["accepted_path_length_m"],
                "fiber_length_m":m["accepted_fiber_length_m"],"normalized_equilibrium_residual":m["normalized_equilibrium_residual"]})
 replay="not_requested" if "--stand-deterministic-replay" not in inv["argv"] else selected.get("replay")
 return dict(inv=inv,result=result,terminal=terminal,selected=selected,audit=audit,status=status,nodes=nodes,tets=nt,minj=minj,maxj=maxj,route=route,replay=replay)
def assess(a):
 rd=Path(a.run_dir).resolve();pair=J(rd/"pair.json")
 b=parse_run(rd/"baseline",pair);l=parse_run(rd/"loaded",pair)
 bt,lt=b["terminal"],l["terminal"];sameq=bt["initial_q"]==lt["initial_q"];samev=bt["initial_v"]==lt["initial_v"]
 common=set(b["nodes"])&set(l["nodes"]);samekeys=set(b["nodes"])==set(l["nodes"])
 diffs=[math.sqrt(sum((b["nodes"][k][i]-l["nodes"][k][i])**2 for i in range(3))) for k in common]
 targets=[]
 for name,q,v,d,rng in P[pair["cohort"]][2]:
  dq=float(lt["q"][q])-float(bt["q"][q]);dv=float(lt["v"][v])-float(bt["v"][v])
  targets.append({"joint":name,"q_index":q,"v_index":v,"source_oracle_direction":d,"source_range_rad":rng,
                  "baseline_q_rad":bt["q"][q],"loaded_q_rad":lt["q"][q],"delta_q_rad":dq,
                  "baseline_v_rad_s":bt["v"][v],"loaded_v_rad_s":lt["v"][v],"delta_v_rad_s":dv,
                  "direction_match_diagnostic":dq>0 if d=="+" else dq<0,"direction_is_acceptance_gate":False})
 sameassets=all(b["inv"][k]==l["inv"][k] for k in ("binary","private_runtime_library","metallib"))
 sameinputs=b["inv"]["inputs_verified"]==l["inv"]["inputs_verified"]
 ok=(b["result"].get("native_exit_code",b["result"].get("exit_code"))==0 and l["result"].get("native_exit_code",l["result"].get("exit_code"))==0 and
     not b["result"].get("timed_out",False) and not l["result"].get("timed_out",False) and b["status"]==l["status"]=="passed" and
     sameq and samev and sameassets and sameinputs and samekeys and bool(common) and b["tets"]>0 and l["tets"]>0)
 out={"schema":"numi.human.full-shared-fem-joint-cohort-assessment.v1","cohort":pair["cohort"],
      "status":"paired_shared_fem_endpoints_observed" if ok else "pair_failed_or_inconclusive",
      "scope":pair["scope"],"limitations":pair["limitations"],"preflight_blockers":pair["preflight_blockers"],
      "base_invocation_sha256":pair["base_invocation_sha256"],"source_manifest_sha256":pair["source_manifest_sha256"],
      "binary_identity":pair["binary_identity"],"steps":pair["steps"],"dt_s":pair["dt_s"],"simulated_seconds":pair["simulated_seconds"],
      "matched_pair":{"baseline_activation":0.0,"loaded_activation":l["inv"]["activation"],"same_initial_q_bitwise":sameq,
                      "same_initial_v_bitwise":samev,"same_input_hashes":sameinputs,"same_binary_runtime_metallib":sameassets,
                      "baseline_replay":b["replay"],"loaded_replay":l["replay"],
                      "replay_note":"raw selected-control replay='failed' means not_requested when deterministic replay is absent from argv."},
      "source_muscles":{"ids":pair["source_actuator_ids"],"names":pair["source_actuator_names"],
                        "baseline_route":b["route"],"loaded_route":l["route"],
                        "force_tension_are_measurements_not_minimum_force_gates":True},
      "joint_targets":targets,
      "shared_fem":{"baseline_nodes":len(b["nodes"]),"loaded_nodes":len(l["nodes"]),"matched_nodes":len(common),
                    "baseline_tets":b["tets"],"loaded_tets":l["tets"],
                    "max_endpoint_node_difference_m":max(diffs) if diffs else None,
                    "rms_endpoint_node_difference_m":math.sqrt(sum(x*x for x in diffs)/len(diffs)) if diffs else None,
                    "baseline_min_J":b["minj"],"loaded_min_J":l["minj"],"baseline_max_J":b["maxj"],"loaded_max_J":l["maxj"],
                    "baseline_terminal_audit":b["audit"],"loaded_terminal_audit":l["audit"]},
      "interpretation":"Short full-shared-FEM accepted endpoint measurement; source direction and magnitudes are diagnostics, not ROM, physiology, grasp, contact-completeness, or anatomy-candidate qualification."}
 dest=rd/"assessment.json"
 if dest.exists(): raise RuntimeError(f"refusing overwrite {dest}")
 dest.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps(out,indent=2,sort_keys=True));return 0 if ok else 1
def selftest():
 basepath,b=load_base(DEFAULT_BASE); sm=source_sha()
 for c in P:
  td=Path("/tmp/shared-fem-harness-selftest")/c/"loaded"
  a=make_argv(b,c,8,.1,td)
  assert Path(a[4]).resolve()==(td/"output").resolve()
  assert Path(a[4]).resolve()!=Path(b["argv"][4]).resolve()
  assert [int(a[i+1]) for i,x in enumerate(a[:-1]) if x==SELECTED]==P[c][0]
  assert int(val(a,"--muscle-step-count"))==8 and float(val(a,"--muscle-step-seconds"))==.000125 and float(val(a,"--muscle-activation"))==.1
 def f(d):
  lines=(d/"stdout.log").read_text(errors="replace").splitlines()
  t=rec(lines,"stand_terminal_state");s=rec(lines,"selected_hand_wrist_actuation")
  audits=[x for x in lines if x.startswith("resting_FEM_terminal_audit=")]
  return t,s,audits,sum(x.startswith("resting_FEM_node,") for x in lines),sum(x.startswith("resting_FEM_tet,") for x in lines)
 base_dir=DEFAULT_BASE.parent
 active_dir=DEFAULT_BASE.parent.parent/"external-shared-activation-0.10-8steps-125us-001"
 bdata=f(base_dir);ldata=f(active_dir)
 assert len(bdata[0]["q"])==len(ldata[0]["q"])==129 and len(bdata[0]["v"])==len(ldata[0]["v"])==128
 assert bdata[0]["initial_q"]==ldata[0]["initial_q"] and bdata[0]["initial_v"]==ldata[0]["initial_v"]
 assert bdata[2][0].startswith("resting_FEM_terminal_audit=passed ") and ldata[2][0].startswith("resting_FEM_terminal_audit=passed ")
 assert bdata[3:]==ldata[3:] and bdata[3]>0 and bdata[4]>0
 ids=[232,233,295,296]
 assert [int(x["source_actuator_index"]) for x in bdata[1]["selected_muscles"]]==ids
 assert [int(x["source_actuator_index"]) for x in ldata[1]["selected_muscles"]]==ids
 print(json.dumps({"self_test_passed":True,"native_launched":False,"source_manifest_sha256":sm,
                   "cohorts":list(P),"accepted_q_length":129,"accepted_v_length":128,
                   "shared_fem_nodes":bdata[3],"shared_fem_tets":bdata[4],
                   "retained_fixture":str(active_dir),"fixture_scope":"parser and argv-transform check only"},indent=2))
def list_targets():
 sm=source_sha()
 data={"schema":"numi.human.full-shared-fem-joint-cohort-target-map.v1","source_manifest_path":str(SOURCE),"source_manifest_sha256":sm,
       "q_v_provenance":"NHRIGID2 source q/v layout from base invocation; shoulder elevation q36/74-v35/73; elbow q39/77-v38/76; knee q106/120-v105/119; ankle q109/123-v108/122; aggregate MTP q111/125-v110/124.",
       "limitations":["MTP coordinates are aggregate toes-body coordinates; no toe/IP DOFs.","Shoulder coverage here is primary elevation only.","The existing full-shared scene uses retained older skin payloads; newer hand candidates remain unqualified.",
       "Four/eight-step outcomes are finite-horizon engineering observations only."],"cohorts":{}}
 for c,(ids,names,t,lim) in P.items():
  data["cohorts"][c]={"ids":ids,"names":names,"targets":[{"joint":n,"q":q,"v":v,"source_direction":d,"range_rad":r} for n,q,v,d,r in t],"limitations":lim}
 print(json.dumps(data,indent=2,sort_keys=True))
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest="cmd",required=True)
 p=sp.add_parser("list");p.set_defaults(fn=lambda a:list_targets() or 0)
 p=sp.add_parser("self-test");p.set_defaults(fn=lambda a:selftest() or 0)
 p=sp.add_parser("prepare");p.add_argument("cohort",choices=sorted(P));p.add_argument("--base-invocation",default=str(DEFAULT_BASE));p.add_argument("--steps",type=int,choices=(4,8),default=8);p.add_argument("--activation",type=float,default=.1);p.add_argument("--timeout",type=float,default=300);p.add_argument("--output-dir");p.set_defaults(fn=prepare)
 p=sp.add_parser("execute");p.add_argument("run_dir");p.add_argument("--timeout",type=int,default=300);p.add_argument("--coordinator-authorized",action="store_true");p.set_defaults(fn=execute)
 p=sp.add_parser("assess");p.add_argument("run_dir");p.set_defaults(fn=assess)
 a=ap.parse_args()
 try:return int(a.fn(a) or 0)
 except Exception as e: print(f"ERROR: {e}",file=sys.stderr);return 2
if __name__=="__main__": raise SystemExit(main())
