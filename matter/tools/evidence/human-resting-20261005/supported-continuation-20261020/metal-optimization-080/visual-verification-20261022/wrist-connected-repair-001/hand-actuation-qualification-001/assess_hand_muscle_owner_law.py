#!/usr/bin/env python3
"""Post-run NHMYO2 compliant tendon endpoint assessment; never overrides runner status."""
import argparse, hashlib, json, math, struct
from pathlib import Path

def owner_tendon_force(normalized_length, a):
    strain = normalized_length - 1.0
    if strain <= 0.0:
        return 0.0
    toe, stiffness = a[4], a[3]
    toe_strain = a[2] - (1.0 - toe) / stiffness
    if strain >= toe_strain:
        return toe + stiffness * (strain - toe_strain)
    t = min(1.0, max(0.0, strain / max(1.0e-12, toe_strain)))
    return ((-2.0*t*t*t + 3.0*t*t)*toe +
            (t*t*t - t*t)*toe_strain*stiffness)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--comparison-result", type=Path)
    ap.add_argument("--timestep-seconds", type=float, required=True)
    a = ap.parse_args()
    result = json.loads(a.result.read_text())
    evidence_dir = a.result.parent.parent.parent
    invocation = json.loads((a.result.parent/"invocation.json").read_text())
    cmap = json.loads((evidence_dir/"hand-actuator-cohort-map.json").read_text())
    src = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-muscle-reference.nhmyo")
    payload = src.read_bytes()
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha != cmap["nhmyo_sha256"]:
        raise SystemExit("NHMYO2 SHA does not match cohort map")
    magic, abi, bodies, count, sites, wraps, routes, tendons, r0, r1, source_sha = struct.unpack_from("<8s9I32s", payload)
    if magic != b"NHMYO2\0\0" or abi != 2 or r0 != count or r1 != 32:
        raise SystemExit("unexpected NHMYO2 header/ABI")
    if source_sha.hex() != cmap["source_archive_sha256"]:
        raise SystemExit("embedded source SHA does not match cohort map")
    moff = 76 + sites*16 + wraps*64 + routes*16
    aoff = moff + count*164
    name_by_idx = {int(m["source_actuator_index"]):m["name"]
                   for m in cmap["cohorts"][result["cohort"]]["muscles"]}
    selected = result["selected_hand_wrist_actuation"]
    channels = []
    for row in selected["selected_muscles"]:
        idx = int(row["source_actuator_index"])
        if idx not in name_by_idx:
            raise SystemExit(f"unmapped actuator {idx}")
        muscle = struct.unpack_from("<4I37f", payload, moff+idx*164)
        arch = struct.unpack_from("<8f", payload, aoff+idx*32)
        v = muscle[4:]
        max_force = v[7] if v[7] >= 0.0 else v[8]/max(1.0e-12,v[2])
        endpoints = {}
        for key in ("accepted","baseline"):
            vals = [float(row[f"{key}_{name}"]) for name in
                    ("path_length_m","path_velocity_m_s","fiber_length_m","fiber_velocity_m_s")]
            if not all(math.isfinite(x) for x in vals):
                raise SystemExit(f"nonfinite {key} path/fiber telemetry for {idx}")
            path, path_v, fiber, fiber_v = vals
            predicted = path+a.timestep_seconds*path_v
            norm_length = (predicted-fiber)/arch[1]
            tension = owner_tendon_force(norm_length,arch)
            force = -max_force*tension
            obs_force = float(row["source_force_n"] if key=="accepted"
                              else row["baseline_source_force_n_derived"])
            obs_tension = float(row["tendon_tension_normalized"] if key=="accepted"
                                else -obs_force/max_force)
            residual = float(row["normalized_equilibrium_residual"] if key=="accepted"
                             else row["baseline_normalized_equilibrium_residual"])
            if not all(math.isfinite(x) for x in (predicted,norm_length,tension,force,obs_force,obs_tension,residual)):
                raise SystemExit(f"nonfinite force/residual telemetry for {idx} {key}")
            slack = norm_length <= 1.0
            if slack and tension==0.0 and obs_tension==0.0 and obs_force==0.0:
                cls = "zero-tension-consistent-with-owner-slack-branch"
            elif tension > 0.0:
                cls = "positive-tension-owner-branch"
            else:
                cls = "owner-law-data-disagreement"
            endpoints[key] = {
                "path_length_m":path,"path_velocity_m_s":path_v,
                "predicted_path_length_m":predicted,"fiber_length_m":fiber,
                "fiber_velocity_m_s":fiber_v,"normalized_fiber_length":fiber/arch[0],
                "normalized_tendon_length":norm_length,
                "owner_computed_tendon_tension_normalized":tension,
                "owner_computed_source_force_n":force,"observed_tendon_tension_normalized":obs_tension,
                "observed_source_force_n":obs_force,"source_force_difference_n":obs_force-force,
                "tension_difference":obs_tension-tension,
                "normalized_equilibrium_residual":residual,"equilibrium_residual_finite":True,
                "slack_branch":slack,"classification":cls
            }
        channels.append({
            "source_actuator_index":idx,"source_name":name_by_idx[idx],
            "source_tendon_index":muscle[0],"maximum_isometric_force_n":max_force,
            "optimal_fiber_length_m":arch[0],"tendon_slack_length_m":arch[1],
            "tendon_strain_at_one_normalized_force":arch[2],
            "tendon_stiffness_at_one_normalized_force":arch[3],
            "tendon_force_at_toe_end_normalized":arch[4],
            "accepted_activation":row["accepted_activation"],
            "baseline_activation":row["baseline_accepted_activation"],
            **endpoints
        })
    comparison = None
    if a.comparison_result:
        old=json.loads(a.comparison_result.read_text())["selected_hand_wrist_actuation"]["q_rows"]
        new=selected["q_rows"]
        keys=("accepted_q_rad","baseline_q_rad","accepted_v_rad_s","baseline_v_rad_s")
        comparison={"path":str(a.comparison_result),"old_count":len(old),"new_count":len(new),
                    "q_v_exactly_equal":len(old)==len(new) and all(
                        all(x[k]==y[k] for k in keys) for x,y in zip(old,new)),
                    "fields":list(keys)}
    residuals=[abs(x[p]["normalized_equilibrium_residual"]) for x in channels for p in ("accepted","baseline")]
    out={
        "schema":"numi.human.hand-muscle-owner-law-assessment.v1",
        "runner_status_preserved":result["status"],"runner_reasons_preserved":result["reasons"],
        "native_execution":result["execution"],"native_exit_code":result["exit_code"],
        "accepted_steps":selected["accepted_steps"],"timestep_seconds":a.timestep_seconds,
        "nhmyo2_sha256":payload_sha,"embedded_source_sha256":source_sha.hex(),
        "native_binary_sha256":invocation["binary_sha256"],
        "native_build_manifest_path":invocation["build_source_manifest_path"],
        "comparison":comparison,
        "summary":{
            "selected_muscle_count":len(channels),
            "accepted_slack_branch_count":sum(x["accepted"]["slack_branch"] for x in channels),
            "accepted_zero_tension_slack_consistent_count":sum(
                x["accepted"]["classification"]=="zero-tension-consistent-with-owner-slack-branch" for x in channels),
            "all_equilibrium_residuals_finite":all(x[p]["equilibrium_residual_finite"]
                for x in channels for p in ("accepted","baseline")),
            "maximum_absolute_normalized_equilibrium_residual":max(residuals,default=0.0),
            "minimum_force_threshold":"none","runner_status_overridden":False},
        "owner_law":{
            "source":"src/core/MujocoMuscleReference.cpp:719-721,853-855",
            "normalized_tendon_length":"(path_length + timestep * path_velocity - candidate_fiber_length) / tendon_slack_length",
            "slack_branch":"normalized tendon length <= 1 returns zero tension",
            "source_force":"negative maximum isometric force times normalized tendon tension",
            "boundary":"endpoint muscle-law assessment only; not grasp, whole-body FEM, biological or clinical qualification"},
        "channels":channels}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(a.output),"strict_runner_status":result["status"],
        "zero_tension_slack_consistent":out["summary"]["accepted_zero_tension_slack_consistent_count"],
        "residuals_finite":out["summary"]["all_equilibrium_residuals_finite"],
        "q_v_exact_parity":comparison and comparison["q_v_exactly_equal"]},indent=2))
if __name__=="__main__":
    main()
