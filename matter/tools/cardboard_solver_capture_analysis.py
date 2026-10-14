#!/usr/bin/env python3
"""Replay a sealed native operator capture through the independent FP64 law.

The capture, replay executable, and this analyzer are hashed. Every local
failure stays in the output. A branch-crossing finite difference and an
unresolved FP32 difference can never produce a smooth-derivative verdict.
Requires NumPy; it does not create or advance a native simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value, size=None):
    a = np.asarray(value, dtype=np.float64)
    if size is not None and a.size != size:
        raise ValueError(f"expected {size} values, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError("nonfinite captured input")
    return a


def metric(actual, expected, noise=0.0):
    a, b = array(actual), array(expected)
    if a.size != b.size:
        raise ValueError("comparison shape differs")
    a,b = a.ravel(),b.ravel()
    absolute = float(np.max(np.abs(a-b)))
    scale = float(np.max(np.abs(b)))
    # An empirical noise floor must be supplied by repeated/swept evidence.
    # Machine epsilon alone is not evidence of the projection noise floor.
    return {"absolute_error": absolute, "reference_scale": scale,
            "relative_error": absolute/scale if scale > noise and scale > 0 else None,
            "noise_floor": noise, "resolved": scale > noise and scale > 0}


def rotation(q):
    x, y, z, w = array(q, 4)
    n = np.linalg.norm([x,y,z,w])
    if n == 0:
        raise ValueError("zero material frame quaternion")
    x,y,z,w = np.asarray([x,y,z,w])/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def force(p, inverse, volume):
    h = -volume*np.asarray(p).reshape(3,3)@inverse.T
    return np.vstack((-h.sum(axis=1), h.T))


def starch_oracle(state, direction, parameters):
    """Independent derivative of the locked finite-starch Neo-Hookean law."""
    f = array(state["F_local"],9).reshape(3,3)
    h = array(direction,9).reshape(3,3)
    determinant = float(np.linalg.det(f))
    if determinant <= 0:
        return {"valid":False,"reason":"nonpositive starch determinant"}
    mu, lam = parameters["mu"], parameters["lambda"]
    inverse = np.linalg.inv(f)
    a = lam*np.log(determinant)-mu
    p = mu*f+a*inverse.T
    tangent = mu*h+lam*np.trace(inverse@h)*inverse.T-a*inverse.T@h.T@inverse.T
    return {"valid":True,"oracle_P_Pa":p.ravel().tolist(),
            "oracle_tangent_Pa":tangent.ravel().tolist(),"oracle_state":[],
            "independent_residual_at_metal_state":0,"lambda_increment":0,
            "elastic_trial_q_dimensionless":None}


def normalize_native(records):
    """Normalize the explicit native JSONL names without regenerating inputs."""
    if not records or records[0].get("schema") != "numi.cardboard.fem-newton-capture.v1":
        return records
    # The native exporter writes GPU floats with nine round-trip digits. Widen
    # the reconstructed FP32 values, not the decimal text treated as exact
    # real numbers; the latter gives a different input to the FP64 oracle.
    def widen(value):
        if isinstance(value, float):
            result = float(np.float32(value))
            if not np.isfinite(result):
                raise ValueError("native capture value outside finite FP32 range")
            return result
        if isinstance(value, list):
            return [widen(item) for item in value]
        if isinstance(value, dict):
            return {key:widen(item) for key,item in value.items()}
        return value
    records = [widen(record) for record in records]
    for r in records:
        r["record_type"] = r["record"]
        if r["record"] == "header":
            r["native_schema"] = r["schema"]
            r["schema"] = "numi.cardboard.fem_operator_fd_capture.v1"
            r["numeric_input_contract"] = "native_fp32_roundtrip_then_exact_fp64_widening"
            r["dt_s"] = r["dt_seconds"]
        elif r["record"] == "integrity":
            r["inputs_unchanged"] = r["production_inputs_bitwise_unchanged"]
            # Count validation below, plus a terminal record, defines complete.
            r["rich_capture_complete"] = True
        elif r["record"] == "tetrahedron":
            r["node_ids"] = [int(n) for n in r["nodes"]]
            r["inverse_rest_rows"] = r["inverse_rest_rows_per_m"]
            r["frame_xyzw"] = r["material_frame_quaternion_xyzw"]
            r["reference_volume_m3"] = r["rest_volume_m3"]
            r["accepted_state"] = r["center"]["accepted_state"]
            c = r["center"]
            c["local_F_direction_per_unit_velocity"] = c["F_direction_local_per_s"]
            c["dP_local_Pa_per_mps"] = c["tangent_local_Pa_per_s"]
            c["dP_world_Pa_per_mps"] = c["tangent_world_Pa_per_s"]
            c["operator_element_force_direction_nodes_N_per_s"] = c["operator_force_node_world_N_per_s"]
            r["perturbations"] = r["sides"]
            for i,s in enumerate(r["perturbations"]):
                s["side"] = "plus" if i%2 == 0 else "minus"
    header = records[0]
    for kind,count in (("node",header["node_count_per_environment"]),
                       ("tetrahedron",header["tetrahedron_count_per_environment"])):
        actual = [r for r in records if r["record_type"] == kind]
        expected = count*header["environment_count"]
        if len(actual) != expected:
            raise ValueError(f"incomplete {kind} capture: {len(actual)} != {expected}")
        identity = [(r["environment"],r["node_id" if kind == "node" else "tet_id"]) for r in actual]
        if len(set(identity)) != expected:
            raise ValueError(f"duplicate {kind} capture identity")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--material-map", default="0:liner,1:medium,2:starch",
                    help="explicit native material indices for this locked recipe")
    ap.add_argument("--stress-noise-floor-pa", type=float, default=0,
                    help="measured absolute floor from repeat/epsilon sweeps; zero means unmeasured")
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("output exists; preserve earlier analyses")
    mapping = {int(k):v for k,v in (part.split(":") for part in args.material_map.split(","))}
    if set(mapping.values()) - {"liner", "medium", "starch"}:
        raise ValueError("FP64 replay supports only the two authored paper grades")
    records = normalize_native([json.loads(line) for line in args.capture.read_text().splitlines() if line.strip()])
    headers = [r for r in records if r.get("record_type") == "header"]
    integrity = [r for r in records if r.get("record_type") == "integrity"]
    if len(headers) != 1 or len(integrity) != 1:
        raise ValueError("exactly one header and terminal integrity record required")
    header = headers[0]
    if header.get("schema") != "numi.cardboard.fem_operator_fd_capture.v1":
        raise ValueError("unsupported capture schema")
    if not integrity[0].get("inputs_unchanged") or not integrity[0].get("rich_capture_complete"):
        raise ValueError("capture integrity failed")
    dt = float(header["dt_s"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid native timestep")
    tets = [r for r in records if r.get("record_type") == "tetrahedron"]
    if not tets:
        raise ValueError("empty capture is inconclusive")
    rows, keys, sources = [], [], {}
    for tet in tets:
        if mapping.get(tet["material_index"]) not in {"liner", "medium"}:
            continue
        for si, state in enumerate([tet["center"]]+tet["perturbations"]):
            key = f"e{tet['environment']}_t{tet['tet_id']}_s{si}"
            f = array(state["F_local"],9).ravel()
            h = array(tet["center"]["local_F_direction_per_unit_velocity"],9).ravel()
            old = array(tet["accepted_state"][:13],13)
            metal = array(state["projected_state"][:13],13)
            rows.append(" ".join([key,mapping[tet["material_index"]]]+
                                 [format(float(v),".17g") for v in np.concatenate((f,h,old,metal))]))
            keys.append(key)
            sources[key] = (tet, si, state)
    if not rows:
        raise ValueError("no supported paper material records")
    all_parameters = header["parameters_by_environment"]
    if any(values != all_parameters[0] for values in all_parameters[1:]):
        raise ValueError("replay requires separately bound parameter files for differing environments")
    parameter_lines, starch_parameters = [], None
    expected_names = ["density","c11","c22","c33","c12","c13","c23","g12","g13","g23","sigma_ref",
                      "hill_F","hill_G","hill_H","hill_L","hill_M","hill_N","q_floor","fb_epsilon","residual_scale"]
    for material in header["materials"]:
        kind = mapping.get(material["index"])
        offset = material["parameter_offset"]
        names = material["parameter_names"]
        values = array(all_parameters[0][offset:offset+len(names)])
        if kind in {"liner","medium"}:
            if names != expected_names or material["name"] != f"hajali2009_{kind}_hill_ideal":
                raise ValueError("captured paper program differs from the independent oracle contract")
            parameter_lines.append(" ".join([kind]+[format(float(v),".17g") for v in values]))
        elif kind == "starch":
            if names != ["density","mu","lambda"] or material["name"] != "starch2007_finite_glue_elastic":
                raise ValueError("captured starch program differs from the independent oracle contract")
            starch_parameters = dict(zip(names,values.tolist()))
    if len(parameter_lines) != 2:
        raise ValueError("both captured paper parameter sets required")
    args.output.mkdir(parents=True)
    parameter_path = args.output/"captured-material-parameters.txt"
    parameter_path.write_text("\n".join(parameter_lines)+"\n")
    result = subprocess.run([str(args.replay.resolve()),"--parameters",str(parameter_path.resolve())], input="\n".join(rows)+"\n",
                            text=True, capture_output=True, check=False)
    oracle = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(oracle) != len(rows) or [r["key"] for r in oracle] != keys:
        raise ValueError("incomplete or reordered independent replay")
    (args.output/"replay-input.txt").write_text("\n".join(rows)+"\n")
    (args.output/"replay-output.jsonl").write_text(result.stdout)
    (args.output/"replay-stderr.txt").write_text(result.stderr)
    by_key = {r["key"]:r for r in oracle}
    for tet in tets:
        if mapping.get(tet["material_index"]) == "starch":
            for si, state in enumerate([tet["center"]]+tet["perturbations"]):
                if starch_parameters is None:
                    raise ValueError("captured starch parameters absent")
                by_key[f"e{tet['environment']}_t{tet['tet_id']}_s{si}"] = starch_oracle(state,tet["center"]["local_F_direction_per_unit_velocity"],starch_parameters)
    audits = []
    for tet in tets:
        prefix = f"e{tet['environment']}_t{tet['tet_id']}_s"
        row = {"environment":tet["environment"],"tet_id":tet["tet_id"],
               "material_index":tet["material_index"]}
        if tet["material_index"] not in mapping:
            row["verdict"] = "material_outside_independent_paper_oracle"
            audits.append(row)
            continue
        center = tet["center"]
        o = by_key[prefix+"0"]
        if not o["valid"]:
            row.update(verdict="independent_projection_failed",oracle=o)
            audits.append(row)
            continue
        q = rotation(tet["frame_xyzw"])
        inv = array(tet["inverse_rest_rows"],9).reshape(3,3)
        volume = float(tet["reference_volume_m3"])
        row["projection_stress"] = metric(center["P_local_Pa"], o["oracle_P_Pa"],args.stress_noise_floor_pa)
        row["projection_state"] = metric(center["projected_state"][:13],o["oracle_state"]) if o["oracle_state"] else {"absolute_error":0,"reference_scale":0,"relative_error":None,"resolved":False,"noise_floor":0}
        row["independent_residual_at_metal_state"] = o["independent_residual_at_metal_state"]
        row["production_residual_scale"] = 10
        row["analytic_tangent"] = metric(center["dP_local_Pa_per_mps"],o["oracle_tangent_Pa"])
        row["frame_input"] = metric(array(center["F_local"],9).reshape(3,3),array(center["F_world"],9).reshape(3,3)@q)
        row["frame_stress"] = metric(array(center["P_world_Pa"],9).reshape(3,3),array(center["P_local_Pa"],9).reshape(3,3)@q.T)
        row["frame_tangent"] = metric(array(center["dP_world_Pa_per_mps"],9).reshape(3,3),array(center["dP_local_Pa_per_mps"],9).reshape(3,3)@q.T)
        row["element_force"] = metric(center["element_force_node_world_N"],force(center["P_world_Pa"],inv,volume))
        row["oracle_element_tangent_N"] = force(array(o["oracle_tangent_Pa"],9).reshape(3,3)@q.T,inv,volume).tolist()
        side_audits, grouped = [], {}
        for si, side in enumerate(tet["perturbations"],1):
            oside = by_key[prefix+str(si)]
            item = {"epsilon":side["epsilon"],"side":side["side"],"oracle_valid":oside["valid"]}
            if oside["valid"]:
                item["projection_stress"] = metric(side["P_local_Pa"],oside["oracle_P_Pa"],args.stress_noise_floor_pa)
                item["independent_residual_at_metal_state"] = oside["independent_residual_at_metal_state"]
                item["lambda_increment"] = oside["lambda_increment"]
                grouped.setdefault(float(side["epsilon"]),{})[side["side"]] = (side,oside)
            side_audits.append(item)
        row["sides"] = side_audits
        row["finite_differences"] = []
        for epsilon, sides in sorted(grouped.items()):
            if set(sides) != {"plus","minus"}:
                continue
            (plus,op),(minus,om) = sides["plus"],sides["minus"]
            fd = (array(plus["P_local_Pa"],9)-array(minus["P_local_Pa"],9))/(2*epsilon*dt)
            realized = (array(plus["F_local"],9)-array(minus["F_local"],9))/(2*epsilon*dt)
            nominal = array(center["local_F_direction_per_unit_velocity"],9)
            def branch(value):
                trial_q = value["elastic_trial_q_dimensionless"]
                if trial_q is None:
                    return "elastic_glue"
                if abs(trial_q-1) < 1e-6:
                    return "transition_unresolved"
                return "plastic" if trial_q > 1 else "elastic"
            branches = [branch(om),branch(o),branch(op)]
            item = {"epsilon":epsilon,"branches":branches,"branch_crossing":len(set(branches))>1,
                    "realized_F_direction":metric(realized,nominal),
                    "metal_centered_tangent":metric(fd,center["dP_local_Pa_per_mps"]),
                    "oracle_centered_tangent":metric((array(op["oracle_P_Pa"],9)-array(om["oracle_P_Pa"],9))/(2*epsilon*dt),o["oracle_tangent_Pa"]),
                    "one_sided_plus":metric((array(plus["P_local_Pa"],9)-array(center["P_local_Pa"],9))/(epsilon*dt),center["dP_local_Pa_per_mps"]),
                    "one_sided_minus":metric((array(center["P_local_Pa"],9)-array(minus["P_local_Pa"],9))/(epsilon*dt),center["dP_local_Pa_per_mps"])}
            item["smooth_derivative_admissible"] = not item["branch_crossing"] and "transition_unresolved" not in branches and args.stress_noise_floor_pa > 0 and float(np.max(np.abs(array(plus["P_local_Pa"],9)-array(minus["P_local_Pa"],9)))) > 2*args.stress_noise_floor_pa
            row["finite_differences"].append(item)
        row["verdict"] = "captured_layers_compared"
        audits.append(row)
    (args.output/"tetrahedron-audit.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in audits))
    assembly = []
    nodes = [r for r in records if r.get("record_type") == "node"]
    for env in sorted({r["environment"] for r in nodes}):
        env_nodes = sorted([r for r in nodes if r["environment"] == env],key=lambda r:r["node_id"])
        count = len(env_nodes)
        if [r["node_id"] for r in env_nodes] != list(range(count)):
            raise ValueError("node IDs are not contiguous")
        mass = array([r["accepted"]["mass_kg"] for r in env_nodes])
        free = np.asarray([r["center"]["constraint_tag"] not in (1,3) and mass[i]>0 for i,r in enumerate(env_nodes)])
        dv = array([r["correction_delta_v_m_s"] for r in env_nodes])
        mass_action = mass[:,None]*dv
        actual_force = np.zeros((count,3))
        tangent_force = np.zeros((count,3))
        oracle_tangent_force = np.zeros((count,3))
        side_force = [np.zeros((count,3)) for _ in range(4)]
        epsilon = [None,None]
        oracle_complete = True
        for tet in (r for r in tets if r["environment"] == env):
            ids = tet["node_ids"]
            inv = array(tet["inverse_rest_rows"],9).reshape(3,3)
            volume = tet["reference_volume_m3"]
            c = tet["center"]
            np.add.at(actual_force,ids,array(c["element_force_node_world_N"]))
            np.add.at(tangent_force,ids,array(c["operator_element_force_direction_nodes_N_per_s"]))
            o = by_key.get(f"e{env}_t{tet['tet_id']}_s0")
            if o and o["valid"]:
                q = rotation(tet["frame_xyzw"])
                np.add.at(oracle_tangent_force,ids,force(array(o["oracle_tangent_Pa"],9).reshape(3,3)@q.T,inv,volume))
            else:
                oracle_complete = False
            for si,s in enumerate(tet["perturbations"]):
                np.add.at(side_force[si],ids,force(s["P_world_Pa"],inv,volume))
                epsilon[si//2] = float(s["epsilon"])
        action = array([r["production_A_delta_v_impulse_Ns"][:3] for r in env_nodes])
        reconstructed = mass_action-dt*dt*tangent_force
        captured_r = array([r["center_residual_impulse_Ns"][:3] for r in env_nodes])
        center_v = array([r["center"]["v_m_s"] for r in env_nodes])
        accepted_v = array([r["accepted"]["v_m_s"] for r in env_nodes])
        # External impulse is frozen across perturbations. Retain its inferred
        # value instead of assuming that all source forces are zero.
        external = captured_r-dt*actual_force+mass[:,None]*(center_v-accepted_v)
        reconstructed[~free] = 0
        env_report = {"environment":env,"free_nodes":int(free.sum()),
                      "production_gather":metric(action[free],reconstructed[free]),
                      "mass_action_l2_Ns":float(np.linalg.norm(mass_action[free])),
                      "stiffness_action_l2_Ns":float(np.linalg.norm(dt*dt*tangent_force[free])),
                      "external_impulse_l2_Ns":float(np.linalg.norm(external[free])),
                      "oracle_complete":oracle_complete,
                      "oracle_assembled_action":metric(action[free],(mass_action-dt*dt*oracle_tangent_force)[free]) if oracle_complete else None,
                      "perturbations":[]}
        for pair,eps in enumerate(epsilon):
            side_r = []
            for side in range(2):
                si = 2*pair+side
                key = f"epsilon_pair_{pair}_{'plus' if side == 0 else 'minus'}"
                velocities = array([r[key]["v_m_s"] for r in env_nodes])
                expected_r = dt*side_force[si]-mass[:,None]*(velocities-accepted_v)+external
                native_r = array([r["epsilon_pair_residual_impulses_Ns"][si][:3] for r in env_nodes])
                env_report["perturbations"].append({"epsilon":eps,"side":side,
                                                    "residual_assembly":metric(native_r[free],expected_r[free])})
                side_r.append(native_r)
            fd = -(side_r[0]-side_r[1])/(2*eps)
            env_report.setdefault("centered_actions",[]).append({"epsilon":eps,"comparison":metric(action[free],fd[free]),
                                                                  "fd_action_l2_Ns":float(np.linalg.norm(fd[free]))})
        env_report["nonvacuous"] = float(np.linalg.norm(action[free])) > 0
        assembly.append(env_report)
    (args.output/"assembled-audit.json").write_text(json.dumps(assembly,indent=2,allow_nan=False)+"\n")
    supported = [r for r in audits if r.get("verdict") == "captured_layers_compared"]
    maxima = {}
    for field in ("projection_stress","projection_state","analytic_tangent","frame_input","frame_stress","frame_tangent","element_force"):
        values = [(r[field]["relative_error"],r["tet_id"],r["environment"]) for r in supported if r[field]["relative_error"] is not None]
        maxima[field] = {"relative_error":max(values)[0],"tet_id":max(values)[1],"environment":max(values)[2]} if values else None
    summary = {"schema":"numi.cardboard.solver-capture-analysis.v1",
               "binding":{"capture_sha256":digest(args.capture),"replay_sha256":digest(args.replay),"analyzer_sha256":digest(__file__)},
               "native_header":header,"integrity":integrity[0],"replay_returncode":result.returncode,
               "tetrahedra":len(tets),"material_tetrahedra_compared":len(supported),"layer_maxima":maxima,
               "stress_noise_floor_measured":args.stress_noise_floor_pa>0,
               "assembled_comparison":assembly,
               "solver_correctness":"unqualified", "physical_validation":False}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,allow_nan=False))


if __name__ == "__main__":
    main()
