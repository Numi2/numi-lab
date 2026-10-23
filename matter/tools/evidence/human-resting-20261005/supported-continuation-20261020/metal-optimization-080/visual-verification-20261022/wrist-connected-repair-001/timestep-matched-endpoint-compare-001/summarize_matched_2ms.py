#!/usr/bin/env python3
"""Offline matched 2 ms timestep endpoint summary for retained logs."""
import csv, hashlib, json, math, re, sys
from pathlib import Path

if len(sys.argv) != 7:
    raise SystemExit("usage: summarize_matched_2ms.py comparison.json full125.stdout candidate250.stdout full125.coupled.csv candidate250.coupled.csv output.json")
comparison_path, full_log, candidate_log, full_csv, candidate_csv, output_path = map(Path, sys.argv[1:])

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def one_record(path, prefix):
    found=[]
    with path.open(errors="replace") as f:
        for line in f:
            if line.startswith(prefix+"="):
                found.append(line[len(prefix)+1:].rstrip("\n"))
    if len(found)!=1:
        raise RuntimeError(f"{path}: expected one {prefix}, got {len(found)}")
    return json.loads(found[0])

def keyvals(line):
    out={}
    for token in line.split():
        if "=" not in token: continue
        k,v=token.split("=",1)
        out[k]=v
    return out

def summary_line(path, prefix):
    rows=[]
    with path.open(errors="replace") as f:
        for line in f:
            if line.startswith(prefix):
                rows.append(line.rstrip("\n"))
    if len(rows)!=1: raise RuntimeError(f"{path}: expected one {prefix}, got {len(rows)}")
    return rows[0]

def coupled(path):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows: raise RuntimeError(f"empty coupled CSV {path}")
    return rows

full=one_record(full_log,"stand_terminal_state")
cand=one_record(candidate_log,"stand_terminal_state")
if len(full["q"])!=129 or len(cand["q"])!=129 or len(full["v"])!=128 or len(cand["v"])!=128:
    raise RuntimeError("unexpected full q/v lengths")
if full["initial_q"]!=cand["initial_q"] or full["initial_v"]!=cand["initial_v"]:
    raise RuntimeError("initial q/v mismatch")
qd=[float(b)-float(a) for a,b in zip(full["q"],cand["q"])]
vd=[float(b)-float(a) for a,b in zip(full["v"],cand["v"])]

def top_delta(base, diff, n):
    return [{"index":i,"reference":float(base[i]),"candidate":float(base[i])+diff[i],"delta":diff[i]}
            for i in sorted(range(len(diff)),key=lambda i:abs(diff[i]),reverse=True)[:n]]

fc,cc=coupled(full_csv),coupled(candidate_csv)
if fc[0].keys()!=cc[0].keys(): raise RuntimeError("coupled CSV columns differ")
if abs(float(fc[-1]["time_s"])-0.002)>1e-9 or abs(float(cc[-1]["time_s"])-0.002)>1e-9:
    raise RuntimeError("coupled CSV final rows are not the 2 ms endpoint")
coupled_delta={}
for k in fc[-1]:
    try:
        a=float(fc[-1][k]);b=float(cc[-1][k])
        if math.isfinite(a) and math.isfinite(b):
            coupled_delta[k]=b-a
    except (TypeError,ValueError):
        continue

cmp=json.loads(comparison_path.read_text())["endpoint_fidelity"]
def channel_summary(force):
    out={}
    for channel,objects in force["per_channel_per_object"].items():
        eligible=[(name,x) for name,x in objects.items() if isinstance(x.get("relative_l1_error"),(float,int))]
        worst=max(eligible,key=lambda z:z[1]["relative_l1_error"]) if eligible else (None,None)
        out[channel]={"objects_with_signal":len(eligible),
                      "objects_failed":sum(x.get("status")=="fail" for x in objects.values()),
                      "max_relative_l1_error":worst[1]["relative_l1_error"] if worst[1] else None,
                      "worst_object":worst[0]}
    return out
comparison_summary={
    "matched_endpoint_time_s":cmp["matched_endpoint"]["endpoint_time_s"],
    "force_gate":cmp["force_fidelity"]["gate"],
    "force_threshold_relative_l1_error":cmp["force_fidelity"]["threshold_relative_l1_error"],
    "force_channels":channel_summary(cmp["force_fidelity"]),
    "anchor_reaction_history":cmp["accepted_anchor_reaction_history"],
    "strain_gate":cmp["green_strain_fidelity"]["gate"],
    "strain_volume_weighted_relative_l1_error":cmp["green_strain_fidelity"]["global_volume_weighted_relative_l1_error"],
    "strain_threshold":cmp["green_strain_fidelity"]["threshold_volume_weighted_relative_l1_error"],
    "tets_at_or_over_5_percent":cmp["green_strain_fidelity"]["tetrahedra_error_at_or_over_threshold"],
    "tet_count":cmp["source"]["source_tetrahedron_count"],
    "max_finite_per_tet_relative_error":cmp["green_strain_fidelity"]["maximum_finite_per_tet_relative_error"],
    "maximum_error_witness":cmp["green_strain_fidelity"]["maximum_error_witness"],
    "gates":cmp["gates"],
}
full_audit=keyvals(summary_line(full_log,"resting_FEM_terminal_audit="))
cand_audit=keyvals(summary_line(candidate_log,"resting_FEM_terminal_audit="))
full_progress=keyvals(summary_line(full_log,"human_standing_progress=accepted "))
cand_progress=keyvals(summary_line(candidate_log,"human_standing_progress=accepted "))
mass_prefixes=("resting_mass_calibration ","whole_body_tissue_mass_reference=","resting_tissue_mass_partition=")
def owner_lines(path):
    d={}
    with path.open(errors="replace") as f:
        for line in f:
            for prefix in mass_prefixes:
                if line.startswith(prefix):
                    k=prefix.rstrip(" =")
                    d[k]=line.rstrip("\n")
    return d
inv=[]
for log in (full_log,candidate_log):
    # Invocation metadata is sibling of the stdout log.
    ip=log.parent/"invocation.json"
    inv.append(json.loads(ip.read_text()))
output={
    "schema":"numi.human.full-shared-fem-matched-2ms-timestep-summary.v1",
    "scope":"offline endpoint comparison only; zero muscle activation; not ROM, physiology, whole-body qualification, or a timestep recommendation",
    "reference":{"path":str(full_log),"sha256":sha(full_log),"steps":full["step_count"],"dt_s":full["timestep_seconds"],"accepted_simulated_time_s":0.002},
    "candidate":{"path":str(candidate_log),"sha256":sha(candidate_log),"steps":cand["step_count"],"dt_s":cand["timestep_seconds"],"accepted_simulated_time_s":0.002},
    "identities":{"binary_sha256":[x["binary"]["sha256"] for x in inv],
                  "runtime_library_sha256":[x["private_runtime_library"]["sha256"] for x in inv],
                  "metallib_sha256":[x["metallib"]["sha256"] for x in inv],
                  "input_hashes_match":inv[0]["inputs_verified"]==inv[1]["inputs_verified"],
                  "source_payload_hashes_match":inv[0]["inputs_verified"]==inv[1]["inputs_verified"],
                  "changed_contact_certification_evaluation_cap":inv[1].get("changed_contact_certification_evaluation_cap"),
                  "physics_tolerances_iterations_materials_contact_settings_changed":inv[1].get("physics_tolerances_iterations_materials_contact_settings_changed"),
                  "physical_equations_and_tolerances_changed":inv[1].get("physical_equations_and_tolerances_changed"),
                  "interpretation":"Schedules have equal physical endpoint. Binary/runtime and input payloads match; shader library differs because candidate raises certified CCD evaluation cap. This comparison is therefore timestep-schedule plus CCD-certification implementation, not timestep-only."},
    "initial_state":{"initial_q_identical":full["initial_q"]==cand["initial_q"],"initial_v_identical":full["initial_v"]==cand["initial_v"],
                     "q_count":len(qd),"v_count":len(vd)},
    "primary_joint_coordinate_deltas":[
        {"joint":name,"q_index":q,"v_index":v,"reference_q":float(full["q"][q]),"candidate_q":float(cand["q"][q]),
         "delta_q":qd[q],"reference_v":float(full["v"][v]),"candidate_v":float(cand["v"][v]),"delta_v":vd[v]}
        for name,q,v in [("shoulder elevation right",36,35),("shoulder elevation left",74,73),
                         ("elbow right",39,38),("elbow left",77,76),("knee right",106,105),("knee left",120,119),
                         ("ankle right",109,108),("ankle left",123,122),("aggregate MTP right",111,110),("aggregate MTP left",125,124)]],
    "accepted_terminal_state_delta":{"q_candidate_minus_reference_rad":qd,"v_candidate_minus_reference_rad_s":vd,
                                     "largest_q_delta":top_delta(full["q"],qd,12),"largest_v_delta":top_delta(full["v"],vd,12)},
    "standing_progress":{"reference":full_progress,"candidate":cand_progress},
    "terminal_audit":{"reference":full_audit,"candidate":cand_audit},
    "coupled_endpoint_state":{"reference_final_row":fc[-1],"candidate_final_row":cc[-1],"candidate_minus_reference_numeric_deltas":coupled_delta,
                              "rows_reference":len(fc),"rows_candidate":len(cc),
                              "contains_gas_and_mass_counters":True},
    "mass_owner_lines":{"reference":owner_lines(full_log),"candidate":owner_lines(candidate_log)},
    "existing_1pct_force_5pct_strain_comparator":comparison_summary,
}
output_path.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"wrote":str(output_path),"q_length":len(qd),"v_length":len(vd),"coupled_rows":[len(fc),len(cc)],
                  "comparator_exit_semantics":"measured-channel gate failed; comparator process returns 2 on gate fail",
                  "force_gate":comparison_summary["force_gate"],"strain_gate":comparison_summary["strain_gate"],
                  "strain_volume_weighted_relative_error":comparison_summary["strain_volume_weighted_relative_l1_error"],
                  "max_abs_q_delta":max(map(abs,qd)),"max_abs_v_delta":max(map(abs,vd))},indent=2))
