#!/usr/bin/env python3
"""Audit captured bulk FEM transformations and gather using a source-bound FP64 replay.

This is a diagnostic of the selected operator capture, not the complete contact
operator or a numerical-convergence certificate. No solver step is executed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import cardboard_solver_pair_analyzer as pair
from cardboard_solver_capture_analysis import normalize_native, rotation, force

SCHEMA = 'numi.cardboard.solver-assembly-audit.v1'


def metric(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        raise ValueError('comparison shape differs')
    return pair.metric(a.ravel().tolist(), b.ravel().tolist())


def matrix(values):
    a = np.asarray(values, dtype=float)
    if a.size != 9 or not np.isfinite(a).all():
        raise ValueError('invalid matrix input')
    return a.reshape(3, 3)


def starch_reference(tet, header):
    descriptor = next((m for m in header['materials']
                       if int(m['index']) == int(tet['material_index'])), None)
    if descriptor is None or descriptor['name'] != 'starch2007_finite_glue_elastic':
        raise ValueError('unsupported non-paper material in assembled capture')
    if descriptor['parameter_names'] != ['density', 'mu', 'lambda']:
        raise ValueError('starch parameter names/order differ')
    offset = int(descriptor['parameter_offset'])
    values = header['parameters_by_environment'][tet['environment']][offset:offset+3]
    if len(values) != 3 or min(values) <= 0:
        raise ValueError('invalid captured starch parameters')
    _, mu, lam = values
    f = matrix(tet['center']['F_local'])
    h = matrix(tet['center']['local_F_direction_per_unit_velocity'])
    determinant = np.linalg.det(f)
    if determinant <= 0:
        raise ValueError('nonpositive captured starch determinant')
    inv = np.linalg.inv(f)
    a = lam*np.log(determinant)-mu
    stress = mu*f + a*inv.T
    tangent = mu*h + lam*np.trace(inv@h)*inv.T-a*inv.T@h.T@inv.T
    return stress, tangent


def audit_environment(header, nodes, tets, replay, environment):
    nodes = sorted((r for r in nodes if r['environment'] == environment), key=lambda r:r['node_id'])
    count = int(header['node_count_per_environment'])
    if [r['node_id'] for r in nodes] != list(range(count)):
        raise ValueError('missing, duplicated, or noncontiguous node identity')
    dt = float(header['dt_seconds_by_environment'][environment])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('invalid object timestep')
    masses = np.array([r['accepted']['mass_kg'] for r in nodes])
    if any(r['center']['mass_kg'] != r['accepted']['mass_kg'] for r in nodes):
        raise ValueError('candidate masses differ from accepted masses')
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError('invalid exact nodal masses')
    free = np.array([r['center']['constraint_tag'] not in (1,3) and masses[i] > 0
                     for i,r in enumerate(nodes)])
    dv = np.array([r['correction_delta_v_m_s'] for r in nodes])
    mass_action = masses[:,None]*dv
    force_sum = np.zeros((count,3))
    tangent_sum = np.zeros((count,3))
    oracle_candidate = np.zeros((count,3))
    oracle_reprojected = np.zeros((count,3))
    side_force = [np.zeros((count,3)) for _ in range(4)]
    audits = []
    paper_branch_classes = [set() for _ in range(2)]
    branch_crossing_tets = [0,0]
    near_yield_tets = [0,0]
    env_tets = sorted((t for t in tets if t['environment'] == environment), key=lambda t:t['tet_id'])
    if [t['tet_id'] for t in env_tets] != list(range(int(header['tetrahedron_count_per_environment']))):
        raise ValueError('missing, duplicated, or noncontiguous tetrahedron identity')
    for tet in env_tets:
        ids = tet['node_ids']
        if len(ids) != 4 or len(set(ids)) != 4 or min(ids)<0 or max(ids)>=count:
            raise ValueError('invalid tetrahedron node identities')
        c = tet['center']
        tet_dt = c.get('dt_seconds_bulk_modulus_kind_frame_active', [dt])[0]
        if tet_dt != dt:
            raise ValueError('mixed object timesteps require a separate assembly audit')
        q = rotation(tet['frame_xyzw'])
        inv = matrix(tet['inverse_rest_rows'])
        volume = float(tet['reference_volume_m3'])
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError('invalid reference volume')
        p = matrix(c['P_local_Pa'])
        dp = matrix(c['dP_local_Pa_per_mps'])
        pworld = matrix(c['P_world_Pa'])
        dpworld = matrix(c['dP_world_Pa_per_mps'])
        elem = np.array(c['element_force_node_world_N'])
        elem_dp = np.array(c['operator_element_force_direction_nodes_N_per_s'])
        if elem.shape != (4,3) or elem_dp.shape != (4,3):
            raise ValueError('invalid captured four-node force shape')
        row = dict(environment=environment,tet_id=tet['tet_id'],material_index=tet['material_index'],
                   stress_frame=metric(pworld,p@q.T), tangent_frame=metric(dpworld,dp@q.T),
                   element_force=metric(elem,force(pworld,inv,volume)),
                   element_tangent=metric(elem_dp,force(dpworld,inv,volume)))
        key = f"e{environment}_t{tet['tet_id']}_s0"
        if key in replay:
            reference = replay[key]
            if reference.get('valid') is not True:
                raise ValueError('invalid independent paper reference row')
            oc = matrix(reference['oracle_tangent_at_captured_candidate_Pa_per_s'])
            op = matrix(reference['oracle_tangent_at_reprojected_state_Pa_per_s'])
            for ep in range(2):
                keys = [f"e{environment}_t{tet['tet_id']}_s{s}" for s in (2*ep+2,0,2*ep+1)]
                if any(k not in replay or replay[k].get('valid') is not True for k in keys):
                    raise ValueError('missing independent side reference')
                classes = [pair.classify_trial_q(replay[k]['elastic_trial_q'],1e-6) for k in keys]
                paper_branch_classes[ep].update(classes)
                near_yield_tets[ep] += 'near_yield_unresolved' in classes
                branch_crossing_tets[ep] += len(set(classes))>1 and 'near_yield_unresolved' not in classes
        else:
            _, oc = starch_reference(tet,header)
            op = oc
        np.add.at(force_sum,ids,elem)
        np.add.at(tangent_sum,ids,elem_dp)
        np.add.at(oracle_candidate,ids,force(oc@q.T,inv,volume))
        np.add.at(oracle_reprojected,ids,force(op@q.T,inv,volume))
        for si,s in enumerate(tet['perturbations']):
            np.add.at(side_force[si],ids,force(matrix(s['P_world_Pa']),inv,volume))
        audits.append(row)
    native_action = np.array([r['production_A_delta_v_impulse_Ns'][:3] for r in nodes])
    assembled = mass_action-dt*dt*tangent_sum
    assembled[~free] = 0
    native_residual = np.array([r['center_residual_impulse_Ns'][:3] for r in nodes])
    velocity = np.array([r['center']['v_m_s'] for r in nodes])
    old_velocity = np.array([r['accepted']['v_m_s'] for r in nodes])
    external = native_residual - dt*force_sum + masses[:,None]*(velocity-old_velocity)
    # This external term is inferred, not an independently measured source force.
    report = dict(environment=environment,free_nodes=int(free.sum()),
        production_gather=metric(native_action[free],assembled[free]),
        captured_candidate_FP64_assembled_action=metric(native_action[free],(mass_action-dt*dt*oracle_candidate)[free]),
        reprojected_FP64_assembled_action=metric(native_action[free],(mass_action-dt*dt*oracle_reprojected)[free]),
        mass_action_l2_Ns=float(np.linalg.norm(mass_action[free])),
        captured_stiffness_action_l2_Ns=float(np.linalg.norm(dt*dt*tangent_sum[free])),
        production_action_l2_Ns=float(np.linalg.norm(native_action[free])),
        inferred_frozen_external_impulse_l2_Ns=float(np.linalg.norm(external[free])),
        external_source_independently_validated=False,
        strict_gate='inconclusive_no_measured_noise_floor',perturbations=[],centered_actions=[])
    for ep,epsilon in enumerate(header['epsilon_values']):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('invalid representable perturbation')
        sides = []
        for side,sign in enumerate((1,-1)):
            si=2*ep+side
            key=f"epsilon_pair_{ep}_{'plus' if side==0 else 'minus'}"
            v=np.array([r[key]['v_m_s'] for r in nodes])
            native=np.array([r['epsilon_pair_residual_impulses_Ns'][si][:3] for r in nodes])
            expected=dt*side_force[si]-masses[:,None]*(v-old_velocity)+external
            actual_du=np.array([r['epsilon_pair_realized_u_delta_m'][si] for r in nodes])
            actual_dv=np.array([r['epsilon_pair_realized_velocity_delta_m_s'][si] for r in nodes])
            report['perturbations'].append(dict(epsilon=epsilon,side='plus' if sign==1 else 'minus',
                residual_gather=metric(native[free],expected[free]),
                realized_delta_v=metric(actual_dv[free],(sign*epsilon*dv)[free]),
                realized_u_velocity_relation=metric(actual_du[free],(dt*actual_dv)[free])))
            sides.append(native)
        fd=-(sides[0]-sides[1])/(2*epsilon)
        report['centered_actions'].append(dict(epsilon=epsilon,comparison=metric(native_action[free],fd[free]),
            branch_crossing_tetrahedra=branch_crossing_tets[ep],near_yield_tetrahedra=near_yield_tets[ep],
            smooth_branch_eligible=bool(np.linalg.norm(native_action[free]) > 0 and
                not branch_crossing_tets[ep] and not near_yield_tets[ep]),
            strict_gate='inconclusive_no_measured_noise_floor'))
    return report,audits


def run(capture,pair_analysis,output):
    if output.exists(): raise ValueError('output exists; preserve earlier analyses')
    summary=json.loads((pair_analysis/'summary.json').read_text())
    replay_path=pair_analysis/'replay-output.jsonl'
    replay_status = summary.get('replay',{})
    if (summary.get('schema')!=pair.SCHEMA or replay_status.get('invalid_rows')!=0 or
            replay_status.get('returncode')!=0 or replay_status.get('output_errors')):
        raise ValueError('complete valid pair-aware source replay is required')
    binding=summary.get('binding',{})
    if binding.get('capture_sha256')!=pair.sha256(capture) or binding.get('replay_output_sha256')!=pair.sha256(replay_path):
        raise ValueError('capture/replay hash differs from admitted pair analysis')
    records=normalize_native([json.loads(line) for line in capture.read_text().splitlines() if line.strip()])
    header=records[0]
    if len([r for r in records if r.get('record_type')=='integrity' and r.get('inputs_unchanged')])!=1:
        raise ValueError('capture terminal integrity is absent')
    replay={r['key']:r for r in (json.loads(line) for line in replay_path.read_text().splitlines() if line.strip())}
    nodes=[r for r in records if r['record_type']=='node']
    tets=[r for r in records if r['record_type']=='tetrahedron']
    environments,element_audits=[],[]
    for env in range(header['environment_count']):
        report,audits=audit_environment(header,nodes,tets,replay,env)
        environments.append(report); element_audits.extend(audits)
    layer_maxima={}
    for key in ('stress_frame','tangent_frame','element_force','element_tangent'):
        values=[(a[key]['relative_error'],a['environment'],a['tet_id']) for a in element_audits if a[key]['relative_error'] is not None]
        layer_maxima[key]={'max_relative_error':max(values)[0],'environment':max(values)[1],'tet_id':max(values)[2]} if values else None
    report=dict(schema=SCHEMA,binding=dict(capture_sha256=pair.sha256(capture),pair_summary_sha256=pair.sha256(pair_analysis/'summary.json'),
        replay_output_sha256=pair.sha256(replay_path),analyzer_sha256=pair.sha256(Path(__file__)),
        normalization_source_sha256=pair.sha256(Path(__file__).with_name('cardboard_solver_capture_analysis.py')),
        pair_analyzer_source_sha256=pair.sha256(Path(pair.__file__))),
        scope='captured bulk-plus-mass operator and element-frame transformations; full nonlinear contact operator not qualified',
        run_completion=summary['run_completion'],layer_maxima=layer_maxima,environments=environments,
        strict_gate='inconclusive_no_measured_noise_floor',physical_validation=False,terminal_failure_explained=False)
    output.mkdir(parents=True)
    (output/'element-audit.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in element_audits))
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('pair_analysis',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    try:
        result=run(args.capture,args.pair_analysis,args.output)
        print(json.dumps(dict(schema=SCHEMA,environments=result['environments'],layer_maxima=result['layer_maxima'],strict_gate=result['strict_gate']),allow_nan=False))
        return 3
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(json.dumps(dict(error=str(error),admission=False)))
        return 2


if __name__=='__main__': raise SystemExit(main())
