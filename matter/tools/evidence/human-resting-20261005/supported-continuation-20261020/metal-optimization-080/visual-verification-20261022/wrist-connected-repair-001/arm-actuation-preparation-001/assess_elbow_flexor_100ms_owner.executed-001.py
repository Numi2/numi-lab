#!/usr/bin/env python3
"""CPU-only fail-preserving assessment of the completed 100 ms elbow-flexor run."""
from __future__ import annotations
import hashlib, importlib.util, json, math, struct
from pathlib import Path
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-actuation-preparation-001")
RUN = E / "results/bilateral-elbow-flexors006-100ms"
OUT = RUN / "owner-assessment-001.json"
HELPER = E.parent / "hand-actuation-qualification-001/qualify_hand_muscle_owner_law.py"
SOURCE_MANIFEST = Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
SOURCE_NAMES = {230:"BIClong",231:"BICshort",232:"BRA",233:"BRD",293:"BIClong_l",294:"BICshort_l",295:"BRA_l",296:"BRD_l"}
ELBOWS = {"right":(39,38),"left":(77,76)}
DT = 0.000125
STEPS = 800
TENSION_ULP_LIMIT = 4
PRODUCT_ULP_LIMIT = 1
RESIDUAL_LIMIT = 1e-4

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _:float("nan"))
def line_json(path,prefix):
    found=[x.split("=",1)[1] for x in Path(path).read_text().splitlines() if x.startswith(prefix+"=")]
    if len(found)!=1: raise ValueError(f"expected one {prefix} record; found {len(found)}")
    return json.loads(found[0],parse_constant=lambda _:float("nan"))
def finite(x):
    try: y=float(x)
    except (TypeError,ValueError,OverflowError): return None
    return y if math.isfinite(y) else None
def main():
    out={"schema":"numi.human.elbow-flexor-100ms-owner-assessment.v1",
         "scope":"Core selected-tendon elbow actuation and muscle owner-law audit. Not whole-body FEM/contact, grasp, useful-range motion, Brain control, or biological/clinical/production qualification.",
         "strict_exact_limit_audit":{"status":"NOT_RUN","violations":[]},
         "active_elbow_response":{"status":"NOT_RUN","elbows":{}},
         "muscle_owner_law":{"status":"NOT_RUN","channels":[]},
         "input_integrity":{"status":"NOT_RUN","checks":{}},
         "errors":[]}
    try:
        invp=RUN/"invocation.json"; resultp=RUN/"result.json"; stdoutp=RUN/"stdout.log"
        inv=load(invp); result=load(resultp)
        if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
        out["invocation"]={"path":str(invp),"sha256":sha(invp),"binary_path":inv.get("binary_path"),"binary_sha256":inv.get("binary_sha256"),"build_source_manifest_path":inv.get("build_source_manifest_path"),"build_source_manifest_sha256":inv.get("build_source_manifest_sha256"),"build_head":inv.get("build_head"),"steps":inv.get("step_count"),"timestep_seconds":inv.get("timestep_seconds"),"simulated_seconds":inv.get("simulated_seconds"),"cohort":inv.get("cohort"),"argv":inv.get("argv")}
        out["native_result"]={"path":str(resultp),"sha256":sha(resultp),"native_exit_code":result.get("native_exit_code"),"wall_elapsed_seconds":result.get("wall_elapsed_seconds"),"reported_simulated_seconds":result.get("simulated_seconds"),"result_status":result.get("status")}
        errors=out["errors"]
        integrity={}
        for name,path,expected in [("binary",Path(inv["binary_path"]),inv["binary_sha256"]),("source_manifest",Path(inv["build_source_manifest_path"]),inv["build_source_manifest_sha256"])]:
            actual=sha(path) if path.is_file() else None
            integrity[name]={"path":str(path),"expected_sha256":expected,"actual_sha256":actual,"matches":actual==expected}
            if actual!=expected: errors.append(f"{name}_hash_mismatch")
        for raw,expected in inv.get("input_hashes",{}).items():
            p=Path(raw); p=p if p.is_absolute() else Path("/Users/n/numi-human-aortic-contact-opt-20261019")/p
            actual=sha(p) if p.is_file() else None
            integrity[str(p)]={"expected_sha256":expected,"actual_sha256":actual,"matches":actual==expected}
            if actual!=expected: errors.append("input_hash_mismatch:"+str(p))
        out["input_integrity"]={"status":"PASS" if integrity and all(x["matches"] for x in integrity.values()) else "FAIL","checks":integrity}
        if result.get("invocation_sha256")!=sha(invp): errors.append("native_result_invocation_hash_mismatch")
        if result.get("native_exit_code")!=0: errors.append("native_exit_not_zero")
        if inv.get("step_count")!=STEPS or inv.get("timestep_seconds")!=DT or inv.get("simulated_seconds")!=0.1: errors.append("invocation_horizon_mismatch")
        control=line_json(stdoutp,"selected_hand_wrist_actuation")
        terminal=line_json(stdoutp,"stand_terminal_state")
        if terminal.get("step_count")!=STEPS or terminal.get("timestep_seconds")!=DT: errors.append("terminal_horizon_mismatch")
        if terminal.get("root_assistance") is not False: errors.append("root_assistance_not_false")
        if control.get("accepted_steps")!=STEPS: errors.append("accepted_step_count_mismatch")
        if control.get("matched_baseline")!="same_input_zero_activation_increment": errors.append("matched_zero_activation_baseline_missing")
        if control.get("replay")!="bitwise": errors.append("replay_not_bitwise")
        if len(control.get("q_rows",[]))!=44 or len(control.get("arm_q_rows",[]))!=10: errors.append("expected_endpoint_rows_missing")
        rows=control.get("q_rows",[])+control.get("arm_q_rows",[])
        violations=[]
        for r in rows:
            for which in ("accepted","baseline"):
                q=finite(r.get(which+"_q_rad")); lo=finite(r.get("source_lower_limit_rad")); hi=finite(r.get("source_upper_limit_rad"))
                if q is None or lo is None or hi is None:
                    errors.append(f"nonfinite_limit_telemetry:{r.get('q_index')}:{which}")
                elif q<lo or q>hi:
                    violations.append({"side":r.get("side"),"q_index":r.get("q_index"),"endpoint":which,"q_rad":q,"lower_rad":lo,"upper_rad":hi,"excursion_beyond_bound_rad":(lo-q if q<lo else q-hi)})
        out["strict_exact_limit_audit"]={"status":"FAIL" if violations else "PASS","comparison":"literal source_lower_limit_rad <= q <= source_upper_limit_rad; no slop/tolerance applied","violation_count":len(violations),"violations":violations}
        arm={int(r["q_index"]):r for r in control.get("arm_q_rows",[])}
        elbow_errors=[]
        for side,(qi,vi) in ELBOWS.items():
            r=arm.get(qi)
            if r is None or int(r.get("dof_index",-1))!=vi:
                elbow_errors.append("missing_or_wrong_coordinate:"+side); continue
            vals={k:finite(r.get(k)) for k in ("accepted_q_rad","baseline_q_rad","accepted_v_rad_s","baseline_v_rad_s","source_lower_limit_rad","source_upper_limit_rad","selected_muscle_generalized_torque_increment_nm","selected_muscle_generalized_torque_increment_l1_nm")}
            if any(v is None for v in vals.values()): elbow_errors.append("nonfinite_elbow_telemetry:"+side); continue
            active_inside=vals["source_lower_limit_rad"]<=vals["accepted_q_rad"]<=vals["source_upper_limit_rad"]
            baseline_inside=vals["source_lower_limit_rad"]<=vals["baseline_q_rad"]<=vals["source_upper_limit_rad"]
            qdelta=vals["accepted_q_rad"]-vals["baseline_q_rad"]; vdelta=vals["accepted_v_rad_s"]-vals["baseline_v_rad_s"]
            positive=qdelta>0 and vdelta>0 and vals["selected_muscle_generalized_torque_increment_nm"]>0 and vals["selected_muscle_generalized_torque_increment_l1_nm"]>0
            if not active_inside: elbow_errors.append("active_elbow_outside_authored_limits:"+side)
            if not positive: elbow_errors.append("positive_flexor_directional_response_missing:"+side)
            out["active_elbow_response"]["elbows"][side]={"q_index":qi,"v_index":vi,**vals,"q_delta_vs_matched_baseline_rad":qdelta,"v_delta_vs_matched_baseline_rad_s":vdelta,"accepted_endpoint_within_authored_limits":active_inside,"baseline_endpoint_within_authored_limits":baseline_inside,"positive_q_v_and_generalized_torque_response":positive}
        out["active_elbow_response"]["status"]="PASS_DIRECTIONAL_ACTIVE_ENDPOINTS_IN_BOUNDS" if not elbow_errors else "FAIL"
        out["active_elbow_response"]["errors"]=elbow_errors
        spec=importlib.util.spec_from_file_location("owner_law",HELPER)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        nhmyo=Path(next(p for p in inv["input_hashes"] if str(p).endswith(".nhmyo")))
        raw,count,mo,ao,embedded=mod.parse_model(nhmyo)
        source_manifest=load(SOURCE_MANIFEST)
        names={int(x["source_actuator_index"]):x["name"] for x in source_manifest.get("muscles",[])}
        law_errors=[]; channel_reports=[]
        if count!=416: law_errors.append("unexpected_muscle_count")
        if embedded!="280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975": law_errors.append("embedded_source_archive_hash_mismatch")
        observed={int(x.get("source_actuator_index",-1)):x for x in control.get("selected_muscles",[])}
        if sorted(observed)!=sorted(SOURCE_NAMES): law_errors.append("selected_actuator_ids_mismatch")
        for idx,expected_name in SOURCE_NAMES.items():
            r=observed.get(idx)
            if r is None: continue
            local=[]
            if names.get(idx)!=expected_name: local.append("source_manifest_actuator_name_mismatch")
            rec=struct.unpack_from("<4I37f",raw,mo+idx*164)
            arch=struct.unpack_from("<8f",raw,ao+idx*32)
            mf=mod.max_force(rec[4:])
            try: _,accepted=mod.reconstruct(r,"accepted",DT,arch)
            except Exception as ex: accepted=[]; local.append("accepted_owner_reconstruction_failed:"+str(ex))
            try: _,baseline=mod.reconstruct(r,"baseline",DT,arch)
            except Exception as ex: baseline=[]; local.append("baseline_owner_reconstruction_failed:"+str(ex))
            obs_t=finite(r.get("tendon_tension_normalized")); obs_f=finite(r.get("source_force_n")); tendon_n=finite(r.get("tendon_tension_n"))
            if any(x is None for x in (obs_t,obs_f,tendon_n,mf)) or not accepted or not baseline:
                local.append("missing_or_nonfinite_owner_fields")
                accepted_t_ulps=source_product_ulps=tendon_product_ulps=baseline_force_ulps=None
            else:
                accepted_ts=[x[0] for x in accepted]; baseline_ts=[x[0] for x in baseline]
                accepted_t_ulps=min(mod.ulps(obs_t,x) for x in accepted_ts)
                expected_force=mod.f32(-mod.f32(mf*mod.f32(obs_t)))
                expected_tendon=mod.f32(mf*mod.f32(obs_t))
                source_product_ulps=mod.ulps(obs_f,expected_force)
                tendon_product_ulps=mod.ulps(tendon_n,expected_tendon)
                if accepted_t_ulps>TENSION_ULP_LIMIT: local.append("accepted_tension_owner_law_mismatch")
                if source_product_ulps>PRODUCT_ULP_LIMIT: local.append("accepted_source_force_product_mismatch")
                if tendon_product_ulps>PRODUCT_ULP_LIMIT: local.append("accepted_tendon_force_product_mismatch")
                inferred_baseline=mod.f32(mod.f32(obs_f)-mod.f32(finite(r.get("source_force_increment_n"))))
                baseline_force_candidates=[mod.f32(-mod.f32(mf*mod.f32(t))) for t in [x[0] for x in baseline]]
                nearest=min(baseline_force_candidates,key=lambda x:abs(float(x)-float(inferred_baseline)))
                baseline_force_ulps=mod.ulps(inferred_baseline,nearest)
                if baseline_force_ulps>TENSION_ULP_LIMIT: local.append("inferred_baseline_force_owner_law_mismatch")
                if not all(x==0 for x in accepted_ts) and not all(x>0 for x in accepted_ts): local.append("accepted_owner_branch_ambiguous")
                if not all(x==0 for x in [x[0] for x in baseline]) and not all(x>0 for x in [x[0] for x in baseline]): local.append("baseline_owner_branch_ambiguous")
            residual=finite(r.get("normalized_equilibrium_residual")); base_res=finite(r.get("baseline_normalized_equilibrium_residual"))
            if residual is None or abs(residual)>RESIDUAL_LIMIT: local.append("accepted_equilibrium_residual_missing_or_over_owner_limit")
            if base_res is None or abs(base_res)>RESIDUAL_LIMIT: local.append("baseline_equilibrium_residual_missing_or_over_owner_limit")
            activation={k:finite(r.get(k)) for k in ("commanded_activation","commanded_activation_increment","accepted_activation","accepted_excitation","baseline_accepted_activation")}
            if any(v is None for v in activation.values()) or not (activation["commanded_activation"]==activation["accepted_activation"]==activation["accepted_excitation"]==0.5 and activation["commanded_activation_increment"]==0.5 and activation["baseline_accepted_activation"]==0.0):
                local.append("activation_not_realized_as_requested")
            report={"source_actuator_index":idx,"source_name":expected_name,"max_isometric_force_n":mf,"accepted_tendon_tension_candidates":[x[0] for x in accepted],"accepted_tension_error_float32_ulps":accepted_t_ulps,"accepted_tendon_tension_normalized":obs_t,"accepted_source_force_n":obs_f,"expected_source_force_from_observed_tension_n":(mod.f32(-mod.f32(mf*mod.f32(obs_t))) if obs_t is not None else None),"accepted_source_force_product_error_float32_ulps":source_product_ulps,"accepted_tendon_force_n":tendon_n,"expected_tendon_force_from_observed_tension_n":(mod.f32(mf*mod.f32(obs_t)) if obs_t is not None else None),"accepted_tendon_force_product_error_float32_ulps":tendon_product_ulps,"baseline_tension_candidates_diagnostic_only":[x[0] for x in baseline],"baseline_source_force_directly_emitted":False,"baseline_source_force_inferred_from_increment_n":(mod.f32(mod.f32(obs_f)-mod.f32(r["source_force_increment_n"])) if obs_f is not None else None),"baseline_inferred_force_owner_error_float32_ulps_diagnostic_only":baseline_force_ulps,"accepted_residual":residual,"baseline_residual":base_res,"accepted_path_length_m":r.get("accepted_path_length_m"),"accepted_path_velocity_m_s":r.get("accepted_path_velocity_m_s"),"accepted_fiber_length_m":r.get("accepted_fiber_length_m"),"accepted_fiber_velocity_m_s":r.get("accepted_fiber_velocity_m_s"),"baseline_path_length_m":r.get("baseline_path_length_m"),"baseline_path_velocity_m_s":r.get("baseline_path_velocity_m_s"),"baseline_fiber_length_m":r.get("baseline_fiber_length_m"),"baseline_fiber_velocity_m_s":r.get("baseline_fiber_velocity_m_s"),"activation":activation,"errors":local}
            channel_reports.append(report)
            law_errors.extend(f"{idx}:{x}" for x in local)
        out["muscle_owner_law"]={"status":"PASS" if not law_errors else "FAIL","limits":{"tendon_law_max_float32_ulp":TENSION_ULP_LIMIT,"force_products_max_float32_ulp":PRODUCT_ULP_LIMIT,"normalized_equilibrium_residual_abs_max":RESIDUAL_LIMIT,"residual_limit_owner":"existing muscle-equilibrium owner acceptance; no solver changes","baseline_force":"inferred only from accepted force minus source_force_increment; not native direct telemetry"},"nhmyo2_sha256":sha(nhmyo),"embedded_source_archive_sha256":embedded,"source_manifest_sha256":sha(SOURCE_MANIFEST),"channels":channel_reports,"errors":law_errors}
        out["dynamic_limit_acceptance_tolerance"]={"status":"NOT_ESTABLISHED_FROM_THIS_RUN","detail":"Selected-tendon 100 ms stdout records source bounds and terminal q, but no dynamic joint-limit residual/slop acceptance field. No initialization or unrelated articulated-world tolerance is substituted."}
        out["overall_status"]="PARTIAL_CORE_DIRECTIONAL_OWNER_LAW_PASS_BASELINE_STRICT_LIMIT_FAIL" if out["active_elbow_response"]["status"].startswith("PASS") and out["muscle_owner_law"]["status"]=="PASS" and out["strict_exact_limit_audit"]["status"]=="FAIL" else ("PASS_CORE_ONLY" if not errors and out["strict_exact_limit_audit"]["status"]=="PASS" else "FAIL")
    except Exception as ex:
        out["errors"].append("assessment_exception:"+repr(ex))
        out["overall_status"]="FAIL_ASSESSMENT_ERROR"
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(OUT),"overall_status":out.get("overall_status"),"strict_exact_limit_audit":out["strict_exact_limit_audit"],"active_elbow_response":out["active_elbow_response"],"muscle_owner_law_status":out["muscle_owner_law"]["status"],"errors":out["errors"]},indent=2))
if __name__=="__main__": main()

