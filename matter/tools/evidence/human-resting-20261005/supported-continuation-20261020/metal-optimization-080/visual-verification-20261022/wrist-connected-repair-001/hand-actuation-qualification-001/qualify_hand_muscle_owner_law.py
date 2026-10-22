#!/usr/bin/env python3
"""Derived core-only qualification of selected hand/wrist muscle actuation."""
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

K_MIN = 1.0e-7
RESIDUAL_LIMIT = 1.0e-4
TENSION_ULP_LIMIT = 4
FORCE_ULP_LIMIT = 1

def f32(x):
    try: return struct.unpack("<f", struct.pack("<f", float(x)))[0]
    except (OverflowError, ValueError, TypeError, struct.error): return float("nan")
def bits(x): return struct.unpack("<I", struct.pack("<f", f32(x)))[0]
def frombits(x): return struct.unpack("<f", struct.pack("<I", x & 0xffffffff))[0]
def ordered(x):
    b = bits(x)
    return (~b & 0xffffffff) if b & 0x80000000 else b | 0x80000000
def fromordered(x):
    b = (x & 0x7fffffff) if x & 0x80000000 else (~x & 0xffffffff)
    return frombits(b)
def ulps(a, b):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)): return None
    return abs(ordered(a) - ordered(b))
def neighbors(x, radius=3):
    x = f32(x)
    if not math.isfinite(x): return []
    center = ordered(x)
    return [fromordered(center+d) for d in range(-radius, radius+1) if 0 <= center+d <= 0xffffffff]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def loadj(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _:float("nan"))
def finite(x):
    try: v=float(x)
    except (TypeError,ValueError,OverflowError): return None
    return v if math.isfinite(v) else None
def fail(out, code, detail=None, idx=None):
    item={"code":code}
    if detail is not None: item["detail"]=str(detail)
    if idx is not None: item["source_actuator_index"]=idx
    out.append(item)

def owner_tension(strain, arch):
    """Float32 transcription of normalizedTendonForceFromStrain in Metal."""
    strain=f32(strain)
    if strain<=0: return 0.0
    toe=f32(arch[4]); stiff=f32(arch[3])
    toe_strain=f32(f32(arch[2])-f32(f32(1.0-toe)/max(f32(K_MIN),stiff)))
    if strain>=toe_strain: return f32(toe+f32(stiff*f32(strain-toe_strain)))
    t=f32(min(1.0,max(0.0,f32(strain/max(f32(K_MIN),toe_strain)))))
    t2=f32(t*t); t3=f32(t2*t)
    left=f32(f32(f32(f32(-2.0)*t)*t)*t)
    left=f32(left+f32(f32(f32(3.0)*t)*t)); left=f32(left*toe)
    right=f32(f32(f32(t3-t2)*toe_strain)*stiff)
    return f32(left+right)

def max_force(values):
    direct=f32(values[7])  # gainParameters[2]
    if direct<0: return f32(f32(values[8])/max(f32(K_MIN),f32(values[2])))
    return direct

def reconstruct(row, endpoint, dt, arch):
    """Invert the published Float32 fiber state; retain all round-trip candidates."""
    p="accepted_" if endpoint=="accepted" else "baseline_"
    path=f32(row[p+"path_length_m"]); pv=f32(row[p+"path_velocity_m_s"])
    fiber=f32(row[p+"fiber_length_m"]); fv=f32(row[p+"fiber_velocity_m_s"])
    if not all(math.isfinite(x) for x in (path,pv,fiber,fv,dt)) or dt<=0:
        raise ValueError("nonfinite path/fiber value or nonpositive timestep")
    predicted=f32(max(f32(K_MIN),f32(path+f32(dt*pv))))
    inc_guess=f32(fv*dt); slack=f32(arch[1]); cands={}
    for inc in neighbors(inc_guess,3):
        if f32(inc/dt)!=fv: continue
        for accepted in neighbors(f32(fiber-inc),3):
            if f32(accepted+inc)!=fiber: continue
            ext=f32(f32(predicted-accepted)-slack)
            strain=f32(f32(ext-inc)/slack)
            t=owner_tension(strain,arch)
            cands[bits(t)]=(t,strain)
    if not cands: raise ValueError("fiber length/velocity have no Float32 round-trip owner state")
    return predicted,list(cands.values())

def parse_model(path):
    raw=Path(path).read_bytes()
    if len(raw)<76: raise ValueError("truncated NHMYO2 header")
    magic,abi,bodies,count,sites,wraps,routes,tendons,r0,r1,src=struct.unpack_from("<8s9I32s",raw)
    if magic!=b"NHMYO2\0\0" or abi!=2 or r0!=count or r1!=32: raise ValueError("unexpected NHMYO2 header/ABI")
    mo=76+sites*16+wraps*64+routes*16; ao=mo+count*164
    if ao+count*32>len(raw): raise ValueError("truncated NHMYO2 records")
    return raw,count,mo,ao,src.hex()

def qualify(result_path, output_path, evidence_root=None):
    rp=Path(result_path).resolve(); op=Path(output_path).resolve()
    ip=rp.parent/"invocation.json"
    root=Path(evidence_root).resolve() if evidence_root else rp.parent.parent.parent.resolve()
    d=loadj(rp); inv=loadj(ip); cmap=loadj(root/"hand-actuator-cohort-map.json"); errors=[]
    argv=inv.get("argv",[])
    dtmeta=finite(inv.get("timestep_seconds"))
    positions=[i for i,x in enumerate(argv) if x=="--muscle-step-seconds"]
    dtargv=finite(argv[positions[0]+1]) if len(positions)==1 and positions[0]+1<len(argv) else None
    if dtmeta is None or dtargv is None or dtmeta!=dtargv: fail(errors,"invocation_timestep_not_bound_to_exact_argv")
    dt=f32(dtmeta) if dtmeta is not None else float("nan")
    if inv.get("initial_configuration_only") is not True or inv.get("ongoing_q_lock") is not False:
        fail(errors,"initial_posture_not_one_time_and_unlocked")
    if "--initial-reference-posture" not in argv or inv.get("pose_or_lock_flags"):
        fail(errors,"initial_reference_posture_missing_or_conflicting")
    if any(x in argv for x in ("--pose-q","--root-pose","--stand-root-assistance","--stand-remove-assistance")):
        fail(errors,"scripted_pose_or_assistance_present")
    if not inv.get("skin_payload"): fail(errors,"skin_payload_not_bound")

    cohort=d.get("cohort"); ce=cmap.get("cohorts",{}).get(cohort)
    if not ce: fail(errors,"cohort_not_in_pinned_map",cohort); ce={"muscles":[]}
    expected=[int(x["source_actuator_index"]) for x in ce["muscles"]]
    t=d.get("selected_hand_wrist_actuation",{}); rows=t.get("selected_muscles",[])
    ids=[int(x.get("source_actuator_index",-1)) for x in rows]
    if ids!=expected or d.get("source_actuator_indices")!=expected: fail(errors,"actuator_indices_not_equal_pinned_cohort")
    if t.get("schema")!="numi.human.selected-hand-wrist-actuation.v1": fail(errors,"selected_actuation_schema_mismatch")
    if t.get("matched_baseline")!="same_input_zero_activation_increment": fail(errors,"matched_baseline_missing")
    if t.get("replay")!="bitwise": fail(errors,"bitwise_replay_missing")
    if t.get("accepted_steps")!=inv.get("step_count"): fail(errors,"accepted_steps_do_not_match_invocation")
    if t.get("target_coordinate_count")!=44 or len(t.get("q_rows",[]))!=44: fail(errors,"44_coordinate_qv_readback_missing")
    if d.get("execution") != "completed" or d.get("exit_code")!=0: fail(errors,"native_execution_did_not_complete")
    if d.get("target_joint_limit_audit",{}).get("within_existing_owner_slop") is not True: fail(errors,"authored_joint_limit_slop_failed")
    for key in ("target_generalized_torque_l1_nm","target_q_increment_l1_rad","target_v_increment_l1_rad_s"):
        x=finite(d.get(key))
        if x is None or x<=0: fail(errors,"cohort_torque_q_v_response_missing_or_zero",key)
    qrows=d.get("target_q_rows",[])
    if not qrows: fail(errors,"target_q_rows_missing")
    for key in ("selected_muscle_generalized_torque_increment_l1_nm","q_increment_rad","v_increment_rad_s"):
        vals=[finite(x.get(key)) for x in qrows]
        if any(x is None for x in vals) or sum(abs(x or 0) for x in vals)<=0: fail(errors,"target_coordinate_response_missing_or_zero",key)

    ih=inv.get("input_hashes",{}); mp=[x for x in ih if str(x).endswith(".nhmyo")]
    model=Path(mp[0]) if len(mp)==1 else None
    if model is None: fail(errors,"pinned_nhmyo_not_unique"); raw,count,mo,ao,embedded=b"",0,0,0,""
    else:
        if not model.is_file() or sha(model)!=ih[mp[0]] or ih[mp[0]]!=cmap.get("nhmyo_sha256"): fail(errors,"pinned_nhmyo_hash_mismatch")
        try: raw,count,mo,ao,embedded=parse_model(model)
        except Exception as ex: fail(errors,"nhmyo_decode_failed",ex); raw,count,mo,ao,embedded=b"",0,0,0,""
        if embedded!=cmap.get("source_archive_sha256"): fail(errors,"embedded_source_archive_hash_mismatch")
    binary=inv.get("binary_path"); manifest=inv.get("build_source_manifest_path"); mh=inv.get("build_source_manifest_sha256")
    if not binary or ih.get(binary)!=inv.get("binary_sha256"): fail(errors,"binary_hash_not_bound")
    elif Path(binary).is_file() and sha(binary)!=inv.get("binary_sha256"): fail(errors,"binary_hash_changed")
    if not manifest or not Path(manifest).is_file() or sha(manifest)!=mh: fail(errors,"build_source_manifest_hash_mismatch")

    names={int(x["source_actuator_index"]):x["name"] for x in ce["muscles"]}
    reports=[]; slack_ids=[]; taut_ids=[]
    max_at=max_bt=max_af=max_an=max_bf=0; maxres=0.0; force_delta_l1=0.0
    for row in rows:
        idx=int(row.get("source_actuator_index",-1)); local=[]
        if idx not in names or idx>=count:
            fail(local,"actuator_not_in_pinned_model",idx=idx); errors.extend(local)
            reports.append({"source_actuator_index":idx,"checks":local}); continue
        rec=struct.unpack_from("<4I37f",raw,mo+idx*164); arch=struct.unpack_from("<8f",raw,ao+idx*32)
        mf=max_force(rec[4:])
        av=[finite(row.get(k)) for k in ("accepted_activation","baseline_accepted_activation","commanded_activation","commanded_activation_increment","accepted_excitation")]
        if any(x is None for x in av): fail(local,"activation_missing_or_nonfinite",idx=idx)
        else:
            acc,base,cmd,cmdinc,exc=av
            if not all(0<=x<=1 for x in (acc,base,cmd,exc)) or cmdinc<=0 or acc<=base: fail(local,"activation_increment_not_realized",idx=idx)
            if ulps(acc,cmd)>1 or ulps(acc,exc)>1: fail(local,"activation_state_mismatch",idx=idx)
            delta=f32(f32(acc)-f32(base))
            if ulps(delta,cmdinc)>1: fail(local,"activation_increment_mismatch",idx=idx)

        eps={}
        for endpoint in ("accepted","baseline"):
            try: predicted,cands=reconstruct(row,endpoint,dt,arch)
            except Exception as ex: fail(local,endpoint+"_owner_state_reconstruction_failed",ex,idx); continue
            ts=[x[0] for x in cands]
            residual_key="normalized_equilibrium_residual" if endpoint=="accepted" else "baseline_normalized_equilibrium_residual"
            residual=finite(row.get(residual_key))
            if residual is None: fail(local,endpoint+"_residual_missing_or_nonfinite",idx=idx)
            else:
                maxres=max(maxres,abs(residual))
                if abs(residual)>RESIDUAL_LIMIT: fail(local,endpoint+"_residual_exceeds_1e-4_source_tolerance",idx=idx)
            if endpoint=="accepted":
                obs_t=finite(row.get("tendon_tension_normalized")); obs_f=finite(row.get("source_force_n")); obs_n=finite(row.get("tendon_tension_n"))
            else:
                obs_f=finite(row.get("baseline_source_force_n_derived")); obs_n=abs(obs_f) if obs_f is not None else None
                obs_t=f32(abs(obs_f)/mf) if obs_f is not None and mf>0 else None
            if any(x is None for x in (obs_t,obs_f,obs_n)):
                fail(local,endpoint+"_force_or_tension_missing_or_nonfinite",idx=idx); continue
            tu=min(ulps(obs_t,x) for x in ts)
            fc=[f32(-f32(mf*x)) for x in ts]
            fu=min(ulps(obs_f,x) for x in fc)
            if endpoint=="accepted":
                expectedf=f32(-f32(mf*f32(obs_t))); expectedn=f32(mf*f32(obs_t))
                fu_obs=ulps(obs_f,expectedf); nu=ulps(obs_n,expectedn)
                max_at=max(max_at,tu); max_af=max(max_af,fu_obs); max_an=max(max_an,nu)
                if fu_obs>FORCE_ULP_LIMIT: fail(local,"source_force_not_binary32_owner_product",f"{fu_obs} ULP",idx)
                if nu>FORCE_ULP_LIMIT: fail(local,"tendon_force_not_binary32_owner_product",f"{nu} ULP",idx)
                inc=finite(row.get("source_force_increment_n"))
                if inc is None: fail(local,"source_force_increment_missing_or_nonfinite",idx=idx)
                else: force_delta_l1+=abs(inc)
            else:
                max_bt=max(max_bt,tu); max_bf=max(max_bf,fu)
            if tu>TENSION_ULP_LIMIT: fail(local,endpoint+"_tension_owner_law_mismatch",f"{tu} ULP > {TENSION_ULP_LIMIT}",idx)
            if fu>TENSION_ULP_LIMIT: fail(local,endpoint+"_source_force_owner_law_mismatch",f"{fu} ULP > {TENSION_ULP_LIMIT}",idx)

            all_slack=all(x==0 for x in ts); all_taut=all(x>0 for x in ts)
            if endpoint=="accepted":
                if all_slack:
                    slack_ids.append(idx)
                    if obs_t!=0 or obs_f!=0 or obs_n!=0: fail(local,"slack_branch_has_nonzero_force_or_tension",idx=idx)
                elif all_taut:
                    taut_ids.append(idx)
                    if obs_t<=0 or obs_f>=0 or obs_n<=0: fail(local,"positive_tension_has_zero_or_wrong_sign_force",idx=idx)
                else: fail(local,"accepted_owner_branch_ambiguous",idx=idx)
            elif not (all_slack or all_taut): fail(local,"baseline_owner_branch_ambiguous",idx=idx)
            eps[endpoint]={
                "predicted_path_length_m":predicted,
                "fiber_length_m":f32(row[("accepted_" if endpoint=="accepted" else "baseline_")+"fiber_length_m"]),
                "fiber_velocity_m_s":f32(row[("accepted_" if endpoint=="accepted" else "baseline_")+"fiber_velocity_m_s"]),
                "roundtrip_owner_state_count":len(cands),"owner_tension_candidates_normalized":ts,
                "observed_tension_normalized":obs_t,"observed_source_force_n":obs_f,"observed_tendon_force_n":obs_n,
                "tension_error_float32_ulps":tu,"source_force_error_float32_ulps":fu,
                "normalized_equilibrium_residual":residual,
                "classification":"slack-owner-branch" if all_slack else ("positive-owner-tension" if all_taut else "ambiguous")}
        reports.append({"source_actuator_index":idx,"source_name":names[idx],"maximum_isometric_force_n":mf,"checks":local,**eps})
        errors.extend(local)

    allowed=set()
    for idx in slack_ids:
        allowed.update((f"normalized_tendon_tension_missing_{idx}",f"tendon_force_n_missing_{idx}"))
        row=next((x for x in rows if int(x.get("source_actuator_index",-1))==idx),{})
        if finite(row.get("source_force_increment_n"))==0: allowed.add(f"source_force_missing_{idx}")
    reasons=d.get("reasons",[])
    if d.get("status")=="FAIL":
        if not reasons or not set(reasons).issubset(allowed): fail(errors,"raw_failure_not_only_zero_force_expectations_on_slack_channels")
    elif d.get("status")!="PASS" or reasons: fail(errors,"raw_runner_status_or_reasons_invalid")
    if not taut_ids: fail(errors,"cohort_has_no_positive_tension_muscle")
    if force_delta_l1<=0: fail(errors,"cohort_source_force_increment_is_zero")

    report={
        "schema":"numi.human.hand-actuation-owner-law-qualification.v1",
        "derived_status":"PASS_CORE_ONLY_OWNER_LAW" if not errors else "FAIL",
        "qualification_scope":"selected articulated hand/wrist muscle-tendon owner only; excludes whole-body FEM/contact, grasp, Brain control, biological, clinical and production qualification",
        "raw_runner_status_preserved":d.get("status"),"raw_runner_reasons_preserved":reasons,
        "raw_result_sha256":sha(rp),"invocation_sha256":sha(ip),"invocation_timestep_seconds":dtmeta,
        "binary_sha256":inv.get("binary_sha256"),"build_source_manifest_sha256":mh,
        "nhmyo2_sha256":sha(model) if model and model.is_file() else None,"embedded_source_archive_sha256":embedded,
        "cohort":cohort,"accepted_steps":t.get("accepted_steps"),"selected_muscle_count":len(rows),
        "accepted_slack_actuator_indices":sorted(set(slack_ids)),
        "accepted_positive_tension_actuator_indices":sorted(set(taut_ids)),
        "source_force_increment_l1_n":force_delta_l1,
        "response":{k:d.get(k) for k in ("target_generalized_torque_l1_nm","target_q_increment_l1_rad","target_v_increment_l1_rad_s")},
        "numerical_checks":{
            "source_force":"Metal force=-maximumForce*tendonTension; binary32 multiplication, <=1 ULP.",
            "tendon_force":"maximumForce*tendonTension; binary32 multiplication, <=1 ULP.",
            "tendon_law":"Invert acceptedFiber/increment from exact published Float32 fiberLength=f32(acceptedFiber+increment), fiberVelocity=f32(increment/dt), evaluate the same piecewise Metal tendon law, compare <=4 ULP.",
            "roundtrip_search":"All pairs in +/-3 Float32 ULP neighborhoods that reproduce both published fiber state values; not selected based on result.",
            "max_accepted_tension_error_ulp":max_at,"max_baseline_tension_error_ulp":max_bt,
            "max_accepted_force_product_error_ulp":max_af,"max_accepted_tendon_force_product_error_ulp":max_an,
            "max_baseline_force_law_error_ulp":max_bf,"max_abs_normalized_equilibrium_residual":maxres,
            "native_residual_gate":"dynamic Metal path requires finite residual; stationary path explicitly accepts abs(residual)<=1e-4. This derived check also requires <=1e-4 and does not alter solver behavior.",
            "residual_qualification_limit":RESIDUAL_LIMIT,"minimum_force_threshold":"none",
            "zero_force_policy":"Accept only when every reconstructed Float32 owner state is on the slack branch and measured force/tension are exactly zero."},
        "failures":errors,"channels":reports}
    op.parent.mkdir(parents=True,exist_ok=True)
    op.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--evidence-root",type=Path)
    a=p.parse_args()
    try:r=qualify(a.result,a.output,a.evidence_root)
    except Exception as ex:
        print(f"qualification input error: {ex}",file=sys.stderr);return 2
    print(json.dumps({"output":str(a.output),"derived_status":r["derived_status"],
        "raw_runner_status_preserved":r["raw_runner_status_preserved"],
        "failures":r["failures"],"numerical_checks":r["numerical_checks"]},indent=2))
    return 0 if r["derived_status"]=="PASS_CORE_ONLY_OWNER_LAW" else 1
if __name__=="__main__": raise SystemExit(main())
