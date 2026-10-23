#!/usr/bin/env python3
"""Assess matched 011 selected-tendon core joint-actuation pairs from raw logs."""
import argparse, json, math, re, sys
from pathlib import Path

EXPECTED = {
    "knee-flexors": {"ids": [342,343,367,368,382,383,407,408], "targets": [("right knee",106,105,1,0.0,2.0944),("left knee",120,119,1,0.0,2.0944)]},
    "knee-extensors": {"ids": [365,373,374,375,405,413,414,415], "targets": [("right knee",106,105,-1,0.0,2.0944),("left knee",120,119,-1,0.0,2.0944)]},
    "achilles-plantarflexors": {"ids": [348,349,369,388,389,409], "targets": [("right ankle",109,108,-1,-0.6981,0.5236),("left ankle",123,122,-1,-0.6981,0.5236)]},
}
MIN_DELTA_Q = 1e-9

def one_record(text, prefix):
    rows=[json.loads(line.split("=",1)[1]) for line in text.splitlines() if line.startswith(prefix+"=")]
    if len(rows)!=1: raise RuntimeError(f"expected exactly one {prefix}, found {len(rows)}")
    return rows[0]

def scalar(text, key):
    rows=re.findall(r"(?:^|\s)"+re.escape(key)+r"=([^\s]+)",text)
    if len(rows)!=1: raise RuntimeError(f"expected exactly one {key}, found {len(rows)}")
    return float(rows[0])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir",type=Path); ns=ap.parse_args()
    rd=ns.run_dir.resolve(); pair=json.loads((rd/"pair.json").read_text()); cohort=pair["cohort"]
    if cohort not in EXPECTED: raise SystemExit("unsupported cohort")
    expected=EXPECTED[cohort]; states={}; ctrls={}; raw={}; errors=[]
    if not pair.get("complete"): errors.append("pair_not_complete")
    for condition in ("baseline","loaded"):
        td=rd/condition
        try:
            result=json.loads((td/"execution.json").read_text())
            if result.get("native_exit_code") != 0: errors.append(condition+"_native_exit_nonzero")
            text=(td/"stdout.log").read_text(errors="replace"); raw[condition]=text
            q=one_record(text,"stand_terminal_state"); c=one_record(text,"selected_hand_wrist_actuation")
            if q.get("step_count")!=pair["steps"] or q.get("timestep_seconds")!=pair["dt_s"]: errors.append(condition+"_horizon_mismatch")
            if q.get("root_assistance") is not False: errors.append(condition+"_root_assistance")
            if len(q.get("q",[]))!=129 or len(q.get("v",[]))!=128 or len(q.get("initial_q",[]))!=129 or len(q.get("initial_v",[]))!=128: errors.append(condition+"_qv_dimensions")
            numeric=[*q.get("q",[]),*q.get("v",[]),*q.get("initial_q",[]),*q.get("initial_v",[])]
            if not all(math.isfinite(float(x)) for x in numeric): errors.append(condition+"_nonfinite_qv")
            if c.get("accepted_steps")!=pair["steps"] or c.get("replay")!="bitwise": errors.append(condition+"_accepted_or_replay")
            muscles={int(x["source_actuator_index"]):x for x in c.get("selected_muscles",[])}
            if sorted(muscles)!=expected["ids"]: errors.append(condition+"_actuator_identity")
            for idx,m in muscles.items():
                for key in ("accepted_activation","commanded_activation","commanded_activation_increment","source_force_n","source_force_increment_n","tendon_tension_normalized","tendon_tension_n","normalized_equilibrium_residual"):
                    if not math.isfinite(float(m[key])): errors.append(f"{condition}_nonfinite_{idx}_{key}")
                if float(m["accepted_activation"])!=float(m["commanded_activation"]): errors.append(f"{condition}_activation_not_accepted_{idx}")
                inc=float(m["commanded_activation_increment"])
                if condition=="baseline" and inc!=0.0: errors.append(f"baseline_nonzero_increment_{idx}")
                if condition=="loaded" and inc<=0.0: errors.append(f"loaded_increment_missing_{idx}")
            limits=[x for x in text.splitlines() if x.startswith("pose_joint_ranges=")]
            if len(limits)!=1 or "pose_joint_ranges=passed" not in limits[0] or "failed_range_count=0" not in limits[0]: errors.append(condition+"_native_limits")
            equality=[x for x in text.splitlines() if x.startswith("joint_equality_payload=")]
            if equality!=["joint_equality_payload=NHEQ1 joint_equalities=51"]: errors.append(condition+"_equality_owner")
            states[condition]=q; ctrls[condition]=c
        except Exception as e:
            errors.append(condition+"_parse_error:"+repr(e))
    targets=[]
    if set(states)=={"baseline","loaded"}:
        if states["baseline"]["initial_q"]!=states["loaded"]["initial_q"] or states["baseline"]["initial_v"]!=states["loaded"]["initial_v"]: errors.append("initial_qv_not_bitwise_matched")
        for name,qi,vi,direction,lower,upper in expected["targets"]:
            b=states["baseline"]; a=states["loaded"]
            dq=float(a["q"][qi])-float(b["q"][qi]); dv=float(a["v"][vi])-float(b["v"][vi])
            values=(float(b["q"][qi]),float(a["q"][qi]),float(b["v"][vi]),float(a["v"][vi]),dq,dv)
            if not all(math.isfinite(x) for x in values): errors.append(name+"_nonfinite_endpoint")
            in_range=(lower<=values[0]<=upper and lower<=values[1]<=upper)
            if not in_range: errors.append(name+"_endpoint_outside_source_range")
            targets.append({"joint":name,"q_index":qi,"v_index":vi,"source_direction":direction,"source_range_rad":[lower,upper],"baseline_q_rad":values[0],"loaded_q_rad":values[1],"delta_q_rad":dq,"baseline_v_rad_s":values[2],"loaded_v_rad_s":values[3],"delta_v_rad_s":dv,"directional_response_pass":direction*dq>MIN_DELTA_Q,"endpoint_source_range_pass":in_range})
            if direction*dq<=MIN_DELTA_Q: errors.append(name+"_directional_response_below_gate")
    closure={}
    for condition,text in raw.items():
        try:
            closure[condition]={"tendon_step_max_force_residual_n":scalar(text,"tendon_step_max_force_residual_n"),"tendon_step_max_moment_residual_nm":scalar(text,"tendon_step_max_moment_residual_nm")}
        except Exception as e: errors.append(condition+"_transfer_closure_parse_error:"+repr(e))
    max_eq={condition:max(abs(float(m["normalized_equilibrium_residual"])) for m in c["selected_muscles"]) for condition,c in ctrls.items()} if ctrls else {}
    report={"schema":"numi.human.native-joint-pair-assessment-011.v1","cohort":cohort,"status":"core_directional_response_pass" if not errors else "fail","errors":errors,"binary":pair["binary"],"build_identity":pair["build_identity"],"candidate_rigid":pair["candidate_rigid"],"source_assets":pair["source_assets"],"steps":pair["steps"],"dt_s":pair["dt_s"],"initial_posture":pair["initial_posture"],"root_assistance":False,"accepted_steps":{k:v.get("accepted_steps") for k,v in ctrls.items()},"replay":{k:v.get("replay") for k,v in ctrls.items()},"initial_qv_bitwise_matched":not any(x=="initial_qv_not_bitwise_matched" for x in errors),"targets":targets,"selected_source_muscle_ids":expected["ids"],"max_abs_normalized_equilibrium_residual":max_eq,"nhtendon3_transfer_closure":closure,"native_joint_limits":"pose_joint_ranges passed; 118 enabled source ranges; 0 failed in both runs" if not errors or all("_native_limits" not in x for x in errors) else "failed","equality":"NHEQ1 with 51 records in both runs" if not any("_equality_owner" in x for x in errors) else "failed","scope":"selected-tendon articulated native core only; not whole-body FEM/contact, ROM, grasp, gait, or production qualification","predeclared_directional_threshold_rad":MIN_DELTA_Q}
    out=rd/"assessment-native-joint-pair-011-v4.json"
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    if errors: raise SystemExit(1)

if __name__=="__main__": main()
