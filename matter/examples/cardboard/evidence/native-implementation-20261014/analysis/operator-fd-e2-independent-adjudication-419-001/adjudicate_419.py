#!/usr/bin/env python3
"""Strict, hash-bound adjudication of the 419/422/424 source125 diagnostics."""
from __future__ import annotations
import hashlib, json, math, pathlib, subprocess
import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
OUT = BASE / 'analysis/operator-fd-e2-independent-adjudication-419-001'
SMALL = BASE / 'analysis/operator-fd-e2-adapter-001/independent-smallstep-419-003'
COUPON = BASE / 'analysis/operator-fd-e2-adapter-001/coupon419-003'
TERM = BASE / 'analysis/native-operator-term-replay-result-422'
SWEEP = BASE / 'analysis/native424-residual-sweep-audit-001'
RUN = BASE / 'solver/two-stage-first-failure-rich-fd-419'
DECODER = BASE / 'analysis/native415-status5-parity-001/decode_status5'


def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def read_json(p):
    with open(p, 'r', encoding='utf8') as f: return json.load(f, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonstandard JSON constant {x}')))

def finite_tree(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, list): return all(finite_tree(v) for v in value)
    return True

def norm(a): return math.sqrt(math.fsum(float(x)*float(x) for x in np.asarray(a).reshape(-1)))

def main():
    small = read_json(SMALL/'report.json')
    coupon = read_json(COUPON/'report.json')
    term = read_json(TERM/'result/replay-report.json')
    sweep = read_json(SWEEP/'report.json')
    receipt = read_json(RUN/'receipt.json')
    status5 = read_json(RUN/'first-status5.json')
    assert small['status'] == 'complete_diagnostic' and finite_tree(small)
    assert coupon['status'] == 'complete_diagnostic' and finite_tree(coupon)
    assert receipt['execution_started'] and receipt['source_inputs_unchanged'] and receipt['exit_code'] == 2 and not receipt['timed_out']
    assert term['physical_transactions'] == 0
    assert term['source125_world_fingerprint_hex'] == status5['world_fingerprint']
    assert sweep['all_retained_endpoints_bitwise_equal'] is True
    assert all(x['node_word_mismatches'] == x['residual_word_mismatches'] == 0 for x in sweep['parity'])
    assert status5['records_present'] == 1
    # Decode the 419 payload itself using the already-built source125 ABI reader.
    assert sha(RUN/'first-status5.bin') == status5['payload']['sha256']
    decoded_bytes=subprocess.run([str(DECODER),str(RUN/'first-status5.bin')],check=True,capture_output=True).stdout
    decoded=json.loads(decoded_bytes,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonstandard JSON constant {x}')))
    assert decoded['records_present'] == 1
    recs=[r for r in decoded['records'] if r['capture'][0] != 0]
    assert len(recs)==1
    r=recs[0]
    assert r['capture']==[1,31,1,2] and r['context']==[1,0,1,26]
    assert r['identity'][1:3]==[1464,1464] and r['status'][0]==5
    assert r['status'][9] > r['material_validity'][2]

    # Bind endpoint results by exact epsilon. Unavailable values must be null;
    # no NaN/Infinity may enter serialized report.
    epsrows={float(x['epsilon']):x for x in small['independent_fp64_epsilon_sweep']}
    e22=epsrows[2.0**-22]
    e18=epsrows[2.0**-18]
    e24=epsrows[2.0**-24]
    assert e22['all_center_and_endpoint_projection_rows_passed'] is True
    assert e18['all_center_and_endpoint_projection_rows_passed'] is True
    # A null secant is unavailable, never a zero or pass.
    assert e18['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'] is not None
    assert e22['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'] is not None
    assert e24['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'] is not None
    for row in small['independent_fp64_epsilon_sweep']:
        for key in ('assembled_fp64_A_endpoint_action_vs_center_tangent_action','assembled_fp64_A_endpoint_action_vs_419_independent_center_action'):
            obj=row.get(key)
            if obj and obj.get('status','').startswith('unavailable'):
                assert obj.get('relative_to_reference') is None

    env0=term['environment_0']
    operator=small['operator_action_crosscheck']
    c=small['center_projection']
    ncap=small['native_endpoint_projection_validation']
    sample64=next(x for x in sweep['samples'].values() if x['epsilon']==2.0**-6)
    nativefd64=next(x for x in sample64['environments'] if x['environment']==0)
    samplebest=next(x for x in sweep['samples'].values() if x['epsilon']==2.0**-14)
    nativebest=next(x for x in samplebest['environments'] if x['environment']==0)
    admitted_native=[(s,env) for s in sweep['samples'].values() for env in s['environments']
                     if env['environment']==0 and env['status']==[0,0]
                     and env['centered']['relative_error'] is not None]
    best_admitted=min(admitted_native,key=lambda pair:pair[1]['centered']['relative_error'])
    # Exact force-action decomposition from the captured per-tet tangent actions.
    npz=np.load(COUPON/'environment-0.npz', allow_pickle=False)
    required={'mass_kg','correction_delta_v_m_s','native_production_A_delta_v_impulse_Ns','independent_A_delta_v_impulse_Ns','free_node_mask','material_index'}
    assert required.issubset(npz.files)
    dt=float(term['dt_seconds'])
    dv=npz['correction_delta_v_m_s']
    mass=npz['mass_kg']
    nativeA=npz['native_production_A_delta_v_impulse_Ns']
    independentA=npz['independent_A_delta_v_impulse_Ns']
    # The mass-only replay is the actual native term. Independent mass is analytic M*dv;
    # stiffness is formed by subtraction from each total action, with cancellation retained.
    mpath=TERM/'result/mass-only-action.float4.bin'
    native_mass=np.fromfile(mpath,dtype='<f4').reshape(-1,4)[:882,:3].astype(np.float64)
    independent_mass=mass[:,None]*dv
    native_stiff=nativeA-native_mass
    independent_stiff=independentA-independent_mass
    mass_diff=norm(native_mass-independent_mass)
    stiff_diff=norm(native_stiff-independent_stiff)
    native_norm=norm(nativeA)
    independent_norm=norm(independentA)
    # Native status in 424 is stage-global for this pair, not a per-tet status.
    out={
      'schema':'numi.cardboard.independent_operator_fd_adjudication.v1',
      'status':'complete_diagnostic',
      'evidence_class':'source-bound CPU replay/adjudication; not a converged solver, calibration, or physical qualification',
      'inputs':{},
      'native_capture':{
        'run_exit_code':receipt['exit_code'],'timed_out':receipt['timed_out'],'source_inputs_unchanged':receipt['source_inputs_unchanged'],
        'physical_transactions':term['physical_transactions'],'selection':term['capture_selection'],
        'node_count':882,'tetrahedron_count':1920,'environment_count':2,'dt_seconds':dt,
        'status5_first_record':{
          'record_present':True,'domain':r['domain_name'],'producer_stage':r['stage_name'],
          'control_step':r['context'][0],'microtick':r['context'][1],'microtick_count':r['context'][2],
          'newton_iteration':r['context'][3],'tetrahedron_id':r['identity'][1],
          'constitutive_kind':r['material_words'][0],'parameter_count':r['material_words'][2],
          'state_count':r['material_words'][3],'determinant':r['status'][8],
          'reported_max_abs_stress_pa':r['status'][9],
          'validity_max_stress_pa':r['material_validity'][2],
          'interpretation':'one captured failing tet in a private Armijo evaluation; not the per-tet meaning of later FD-side stage status arrays'
        }
      },
      'center':{
        'all_tet_projection_rows_passed':sum(x['projection_success'] for x in c.values()),
        'tet_rows':sum(x['rows'] for x in c.values()),
        'maximum_reference_scaled_residual':max(x['max_reference_scaled_residual'] for x in c.values()),
        'maximum_dynamic_correction':max(x['max_dynamic_correction'] for x in c.values()),
        'native_fresh_operator_vs_capture_l2_ns':env0['fresh_vs_captured_action_difference_l2_Ns'],
        'native_cache_vs_fresh_l2_ns':env0['cached_vs_fresh_action_difference_l2_Ns'],
        'independent_fp64_A_action_l2_ns':operator['independent_fp64_center_action_vs_native422_fresh_action']['reference_l2'],
        'native_fresh_A_action_l2_ns':env0['fresh_action_l2_Ns'],
        'independent_vs_native_fresh_A_action_abs_l2_ns':operator['independent_fp64_center_action_vs_native422_fresh_action']['difference_l2'],
        'independent_vs_native_fresh_A_action_relative':operator['independent_fp64_center_action_vs_native422_fresh_action']['relative_to_reference'],
        'native422_fresh_rhs_minus_Ax_l2_ns':env0['fresh_rhs_minus_Ax_l2_Ns'],
        'mass_term_native_vs_independent_l2_ns':mass_diff,
        'stiffness_term_native_vs_independent_l2_ns':stiff_diff,
        'native_total_A_action_l2_ns_recomputed':native_norm,
        'independent_total_A_action_l2_ns_recomputed':independent_norm,
        'mass_plus_stiffness_cancellation':True
      },
      'independent_fp64_small_step':{
        'eps_2^-18':{
          'all_center_endpoint_projection_rows_passed':e18['all_center_and_endpoint_projection_rows_passed'],
          'assembled_secant_vs_center_tangent_relative':e18['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'],
          'liner_local_tangent_relative':e18['material']['0']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'medium_local_tangent_relative':e18['material']['1']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'lambda_activity_crossings_liner':e18['material']['0']['plus']['lambda_activity_crossing_rows_vs_center']+e18['material']['0']['minus']['lambda_activity_crossing_rows_vs_center'],
          'lambda_activity_crossings_medium':e18['material']['1']['plus']['lambda_activity_crossing_rows_vs_center']+e18['material']['1']['minus']['lambda_activity_crossing_rows_vs_center']
        },
        'eps_2^-22':{
          'all_center_endpoint_projection_rows_passed':e22['all_center_and_endpoint_projection_rows_passed'],
          'assembled_secant_vs_center_tangent_relative':e22['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'],
          'liner_local_tangent_relative':e22['material']['0']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'medium_local_tangent_relative':e22['material']['1']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'lambda_activity_crossings_liner':e22['material']['0']['plus']['lambda_activity_crossing_rows_vs_center']+e22['material']['0']['minus']['lambda_activity_crossing_rows_vs_center'],
          'lambda_activity_crossings_medium':e22['material']['1']['plus']['lambda_activity_crossing_rows_vs_center']+e22['material']['1']['minus']['lambda_activity_crossing_rows_vs_center']
        },
        'eps_2^-24':{
          'all_center_endpoint_projection_rows_passed':e24['all_center_and_endpoint_projection_rows_passed'],
          'assembled_secant_vs_center_tangent_relative':e24['assembled_fp64_A_endpoint_action_vs_center_tangent_action']['relative_to_reference'],
          'liner_local_tangent_relative':e24['material']['0']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'medium_local_tangent_relative':e24['material']['1']['fd_tangent_vs_center_tangent']['relative_to_reference'],
          'interpretation':'the smaller step worsens the assembled secant comparison; measured subtraction floor, not a universal noise floor'
        },
        'strict_json_unavailable_metrics':'Nonfinite endpoint metrics are encoded as null with explicit unavailable status in the preserved 003 report; this adjudication also serializes with allow_nan=False.'
      },
      'native_fd':{
        'source424_exact_endpoint_parity':sweep['all_retained_endpoints_bitwise_equal'],
        'all_native_endpoint_node_residual_word_mismatches':sum(x['node_word_mismatches']+x['residual_word_mismatches'] for x in sweep['parity']),
        'best_admitted_native_relative_error':best_admitted[1]['centered']['relative_error'],
        'best_admitted_native_epsilon':best_admitted[0]['epsilon'],
        'epsilon_1_over_32_status':next(env for env in next(x for x in sweep['samples'].values() if x['epsilon']==2.0**-5)['environments'] if env['environment']==0)['status'],
        'epsilon_1_over_64_status':nativefd64['status'],
        'epsilon_1_over_64_relative_error':nativefd64['centered']['relative_error'],
        'best_observed_native_epsilon':samplebest['epsilon'],
        'best_observed_native_relative_error':nativebest['centered']['relative_error'],
        'strict_1e_3_native_tangent_gate_passed':False,
        'interpretation':'endpoint replay is exact, but residual secants do not resolve the captured A direction; environment status is stage-propagated, not a rowwise tet failure'
      },
      'adjudication':{
        'center_projection_or_tangent_is_first_failure':False,
        'cache_vs_fresh_operator_is_first_failure':False,
        'first_unresolved_layer':'native finite-difference residual/action response versus captured assembled A at selected Newton48; no native epsilon meets the requested derivative gate',
        'likely_numerical_context':'large-scale cancellation between mass and stiffness action leaves a small total A*dv; smaller FP64 eps also shows subtraction degradation, while the 1/64 path has substantial finite state changes and limited lambda branch crossings',
        'no_claim':'This does not diagnose the eventual Newton terminal failure beyond Newton48, and the native partial endpoint run is not convergence or physical qualification.'
      }
    }
    out['inputs']={
      '419_run_receipt_sha256':sha(RUN/'receipt.json'),
      '419_prediction_sha256':sha(RUN/'prediction.json'),
      '419_rich_capture_sha256':sha(RUN/'operator-capture'),
      '419_status5_json_sha256':sha(RUN/'first-status5.json'),
      '419_status5_payload_sha256':sha(RUN/'first-status5.bin'),
      'status5_decoder_sha256':sha(DECODER),
      'status5_decoder_source_sha256':sha(DECODER.parent/'decode_status5.cpp'),
      'status5_decoded_419_sha256':hashlib.sha256(decoded_bytes).hexdigest(),
      'smallstep_003_report_sha256':sha(SMALL/'report.json'),
      'coupon_003_report_sha256':sha(COUPON/'report.json'),
      'coupon_003_environment0_npz_sha256':sha(COUPON/'environment-0.npz'),
      'native422_replay_report_sha256':sha(TERM/'result/replay-report.json'),
      'native422_receipt_sha256':sha(TERM/'receipt.json'),
      'native422_fresh_action_sha256':sha(TERM/'result/fresh-action.float4.bin'),
      'native422_mass_action_sha256':sha(TERM/'result/mass-only-action.float4.bin'),
      'native424_report_sha256':sha(SWEEP/'report.json'),
      'native424_manifest_sha256':sha(SWEEP/'manifest.json'),
      'source_snapshot_frozen_source_sha256':sha(BASE/'snapshots/source-status5-vtccd-diagnostics-125/frozen-source.json')
    }
    # Verify no non-finite floats and strict JSON encoding.
    assert finite_tree(out)
    payload=json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n'
    (OUT/'report.json').write_text(payload,encoding='utf8')
    return out

if __name__ == '__main__':
    main()
