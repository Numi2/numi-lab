#!/usr/bin/env python3
"""Refresh a source-bound summary and capability frontier from retained cases."""
from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bulk"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def file_record(path: Path, base: Path) -> dict[str, Any]:
    return {'path':path.relative_to(base).as_posix(),'sha256':sha(path),'size_bytes':path.stat().st_size}

def check_inventory(case: Path, hashes: dict[str,str]) -> dict[str,Any]:
    rows=[]
    for rel,expected in sorted(hashes.items()):
        p=case/'inputs'/rel
        actual=sha(p) if p.is_file() else None
        rows.append((rel,expected,actual,actual==expected))
    return {
      'expected_count':len(rows),'verified_count':sum(r[3] for r in rows),
      'all_verified':bool(rows) and all(r[3] for r in rows),
      'hash_map_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'missing_or_mismatched':[{'path':p,'expected':e,'actual':a} for p,e,a,ok in rows if not ok]
    }

def case_binding(rel: str, label: str) -> dict[str,Any]:
    case=ROOT/rel
    pred_path=case/'prediction.json'; receipt_path=case/'receipt.json'
    pred=read_json(pred_path) if pred_path.is_file() else {}
    receipt=read_json(receipt_path) if receipt_path.is_file() else None
    inv=check_inventory(case,pred.get('inputs_sha256',{})) if isinstance(pred.get('inputs_sha256'),dict) else {'expected_count':0,'verified_count':0,'all_verified':False,'hash_map_sha256':None,'missing_or_mismatched':[]}
    outputs=[]
    candidates=[case/'result.json',case/'postrun-audit.json',case/'source-binding-verification.json',case/'binding-verification.json',case/'states.csv',case/'stdout.log',case/'stderr.log',case/'output/result.json',case/'output/observations.csv',case/'output/tool-observations.csv',case/'output/box-native-steps.csv',case/'output/box-actions.csv']
    for p in candidates:
        if p.is_file(): outputs.append(file_record(p,ROOT))
    if not outputs:
        for p in (case/'output').glob('*') if (case/'output').is_dir() else []:
            if p.is_file() and p.suffix in {'.json','.csv','.log','.txt'}: outputs.append(file_record(p,ROOT))
    status=None
    result_paths=[case/'result.json',case/'postrun-audit.json',case/'output/result.json']
    for rp in result_paths:
        if rp.is_file():
            rd=read_json(rp)
            status=rd.get('status')
            if status: break
    return {
      'case':label,'path':rel,'status':status,'source_snapshot':pred.get('source_snapshot'),
      'prediction_sha256':sha(pred_path) if pred_path.is_file() else None,
      'receipt_sha256':sha(receipt_path) if receipt_path.is_file() else None,
      'receipt':receipt,'sealed_inputs':inv,'reviewable_outputs':outputs
    }

def native_score_comparison() -> dict[str,Any]:
    case_ids=('scoring/armijo-002','scoring/armijo-003')
    predictions={rel:read_json(ROOT/rel/'prediction.json') for rel in case_ids}
    results={rel:read_json(ROOT/rel/'output/result.json') for rel in case_ids}
    receipts={rel:read_json(ROOT/rel/'receipt.json') for rel in case_ids}
    rows={}
    for rel in case_ids:
        with (ROOT/rel/'output/observations.csv').open(newline='') as stream:
            rows[rel]=[r for r in csv.DictReader(stream)
                       if r.get('environment')=='0' and r.get('arm')=='indented']
    by_step={rel:{int(r['step']):r for r in data} for rel,data in rows.items()}
    prefix_steps=range(60)
    compared=0; changed=[]
    for step in prefix_steps:
        a=by_step[case_ids[0]][step]; b=by_step[case_ids[1]][step]
        if a.get('step_accepted')!='1' or b.get('step_accepted')!='1':
            changed.append({'step':step,'kind':'prefix_not_accepted_in_both'})
        for key in a:
            compared+=1
            if a[key] != b[key]:
                changed.append({'step':step,'field':key,'left':a[key],'right':b[key]})
    argv={rel:predictions[rel].get('argv',[]) for rel in case_ids}
    def option(args: list[str], name: str) -> str | None:
        try: return args[args.index(name)+1]
        except (ValueError,IndexError): return None
    plans={rel:sum(int(option(args,name) or 0) for name in
                   ('--steps','--hold-steps','--unload-steps','--relax-steps'))
           for rel,args in argv.items()}
    tool_rows=[]
    tool_path=ROOT/case_ids[1]/'output/tool-observations.csv'
    if tool_path.is_file():
        with tool_path.open(newline='') as stream:
            tool_rows=[r for r in csv.DictReader(stream) if r.get('environment')=='0']
    def float_values(name: str) -> list[float]:
        return [float(r[name]) for r in tool_rows if r.get(name) not in (None,'')]
    travel=float_values('commanded_end_travel_m')
    force=float_values('punch_force_z_N')
    end_gap=float_values('min_end_punch_node_gap_m')
    anvil_gap=float_values('min_anvil_node_gap_m')
    contacts=[int(r['punch_contact_count']) for r in tool_rows if r.get('punch_contact_count')]
    by_phase={}
    for r in rows[case_ids[1]]:
        by_phase[r['phase']]=by_phase.get(r['phase'],0)+int(r['step_accepted']=='1')
    accepted_long=[r for r in rows[case_ids[1]] if r.get('step_accepted')=='1']
    certificate_maxima={}
    for field in ('certificate_residual','certificate_correction','certificate_volume','certificate_pressure'):
        vals=[(float(r[field]),int(r['step'])) for r in accepted_long if r.get(field) not in (None,'')]
        certificate_maxima[field]={'maximum':max(vals)[0] if vals else None,
                                   'step':max(vals)[1] if vals else None}
    residual_row=next((r for r in rows[case_ids[0]] if r.get('step_accepted')=='0'),None)
    return {
      'classification':'NATIVE_PRESCRIBED_PATH_SENSITIVITY_ONLY',
      'input_hash_maps_identical':predictions[case_ids[0]].get('inputs_sha256')==predictions[case_ids[1]].get('inputs_sha256'),
      'compiled_world_fingerprints_identical':results[case_ids[0]].get('compiled_world_fingerprint')==results[case_ids[1]].get('compiled_world_fingerprint'),
      'shared_prefix':{
        'step_indices':'0..59, the common accepted loading/hold prefix',
        'all_60_accepted_in_both':all(by_step[case_ids[0]][i].get('step_accepted')=='1' and by_step[case_ids[1]][i].get('step_accepted')=='1' for i in prefix_steps),
        'compared_csv_cells':compared,'changed_cells':changed
      },
      'protocol':{
        'same_frozen_runtime':predictions[case_ids[0]].get('runtime_dir')==predictions[case_ids[1]].get('runtime_dir'),
        'same_native_fingerprint':results[case_ids[0]].get('compiled_world_fingerprint')==results[case_ids[1]].get('compiled_world_fingerprint'),
        'only_unload_steps_changed':option(argv[case_ids[0]],'--unload-steps')=='24' and option(argv[case_ids[1]],'--unload-steps')=='96',
        'fixed_dt_seconds':option(argv[case_ids[1]],'--dt'),
        'requested_steps':plans,
        'planned_duration_seconds':{rel:plans[rel]*float(option(args,'--dt') or 0) for rel,args in argv.items()},
        'warning':'Longer withdrawal at fixed dt changes the prescribed path duration and per-step schedule; this is not a temporal convergence study.'
      },
      'short_withdrawal':{
        'status':results[case_ids[0]].get('status'),'accepted_steps':results[case_ids[0]].get('accepted_steps'),
        'exit_code':receipts[case_ids[0]].get('exit_code'),'wall_seconds':receipts[case_ids[0]].get('wall_seconds'),
        'first_rejected_row':({k:residual_row.get(k) for k in ('step','phase','status_code','fgmres_iterations','status_diagnostic_x','status_diagnostic_y','status_diagnostic_z','status_diagnostic_w')} if residual_row else None)
      },
      'long_withdrawal':{
        'status':results[case_ids[1]].get('status'),'accepted_steps':results[case_ids[1]].get('accepted_steps'),
        'exit_code':receipts[case_ids[1]].get('exit_code'),'wall_seconds':receipts[case_ids[1]].get('wall_seconds'),
        'accepted_steps_by_phase':by_phase,'accepted_step_certificate_maxima':certificate_maxima,
        'physical_validation':results[case_ids[1]].get('physical_validation'),
        'diagnostic_note':'The accepted step-60 status_diagnostic_x is a J-related diagnostic (~0.9964), not a residual. Accepted-step residuals are read from certificate_residual.'
      },
      'long_withdrawal_tool_observations':{
        'record_count':len(tool_rows),'steps_with_punch_contacts':sum(v>0 for v in contacts),
        'maximum_punch_contact_count':max(contacts) if contacts else None,
        'peak_reported_punch_force_z_N':max(force) if force else None,
        'commanded_travel_min_m':min(travel) if travel else None,
        'commanded_travel_max_m':max(travel) if travel else None,
        'minimum_end_punch_node_gap_m':min(end_gap) if end_gap else None,
        'minimum_anvil_node_gap_m':min(anvil_gap) if anvil_gap else None,
        'last_step':({k:tool_rows[-1].get(k) for k in ('step','commanded_end_travel_m','punch_force_z_N','punch_contact_count','min_end_punch_node_gap_m','min_anvil_node_gap_m','step_accepted')} if tool_rows else None)
      },
      'interpretation':'The longer native withdrawal schedule completes from an exactly matching 60-row accepted prefix where the shorter schedule rejects its first unload step. This supports only a prescribed solver-path sensitivity statement; it is not calibrated damage, a physical crease, or time-step convergence.'
    }

def main() -> None:
    point=read_json(OUT/'source-point-006/result.json')
    seed=read_json(OUT/'seeded-sensitivity.json')
    fem=read_json(OUT/'fem-coupling-001/result.json')
    cohesive=read_json(ROOT/'cohesive/native-003/result.json')
    score=read_json(ROOT/'scoring/armijo-001/postrun-audit.json')
    score_record=case_binding('scoring/armijo-001','Frozen-027 punch diagnostic; stationary endpoint scope not qualified')
    score_record['raw_status']=score_record['status']
    score_record['status']='SCOPE_NOT_QUALIFIED_FOR_STATIONARY_ENDPOINT_PROOF'
    score_record['scope_reason']='BOX dispatch selectors in frozen build 027 bypass the stationary-endpoint gate. Accepted indentation force/contact and rejected-withdrawal solve failure remain raw diagnostics.'
    score002_record=case_binding('scoring/armijo-002','Frozen-028 short-withdrawal native path; 60/96 accepted then failed')
    score003_record=case_binding('scoring/armijo-003','Frozen-028 long-withdrawal native path; 168/168 accepted')
    operator_fd=read_json(ROOT/'operator-fd/outcome-comparison.json')
    operator_fd_baseline=case_binding('solver/operator-fd-baseline-001','Frozen-031 no-contact nx16 operator-FD baseline')
    operator_fd_diagnostic=case_binding('solver/operator-fd-001','Frozen-031 no-contact nx16 root19/Newton3 operator-FD diagnostic')
    operator_fd_baseline['status']='NATIVE_PATH_REJECTION_WITH_DIAGNOSTIC_DISABLED'
    operator_fd_diagnostic['status']='NATIVE_PATH_REJECTION_WITH_OPERATOR_DIAGNOSTIC'
    operator_fd_baseline['interpretation']='The unchanged no-contact path naturally rejects at the predicted root19/Newton3 region; no mesh convergence is admitted.'
    operator_fd_diagnostic['interpretation']='Same native path and accepted records with operator-FD capture enabled; the diagnostic does not alter CSV, meshes, result JSON, or compiled pack.'
    scale_comparison_path=ROOT/'operator-fd/scaled-epsilon-comparison.json'
    scale_comparison=read_json(scale_comparison_path) if scale_comparison_path.is_file() else None
    yield_audit_path=ROOT/'operator-fd/accepted-history-yield-audit.json'
    yield_audit=read_json(yield_audit_path) if yield_audit_path.is_file() else None
    later_operator_records=[]
    for rel,label in (
      ('solver/operator-fd-002','Frozen-033 operator-FD capture at epsilon scale 8'),
      ('solver/operator-fd-003','Frozen-033 operator-FD capture at epsilon scale 16'),
      ('solver/operator-fd-scope-034','Frozen-034 finite-positive-mass scope admission check'),
      ('native-tests-012/cohesive-reset','Native-012 cohesive reset regression'),
      ('native-tests-012/capsule-box','Native-012 capsule/box regression'),
      ('native-tests-013/cohesive-reset','Native-013 cohesive reset regression'),
      ('native-tests-013/capsule-box','Native-013 capsule/box regression'),
      ('solver/operator-fd-scope-036','Frozen-036 finite-positive-mass scope admission check'),
      ('solver/local-projection-accuracy-035','Frozen-035 alternate local projection threshold experiment')):
        if (ROOT/rel/'receipt.json').is_file():
            later_operator_records.append(case_binding(rel,label))
    box=read_json(ROOT/'box/native-003/output/result.json')
    box005=read_json(ROOT/'box/native-005/output/result.json')
    invalid=read_json(ROOT/'native-tests-008/invocation-audit.json')
    corrected_tests=read_json(ROOT/'native-tests-009/invocation-audit.json')
    reset010_record=case_binding('native-tests-010/cohesive-reset','Native-010 cohesive-reset partial run with fixture admission failure')
    reset010_record['status']='INCOMPLETE_FIXTURE_ADMISSION_FAILURE'
    reset010_record['interpretation']='Earlier damage/reset/isolation assertions ran; final rejected-reset target did not enter because the fixture target count omitted environments. No cohesive mechanical failure is inferred.'
    reset011_record=case_binding('native-tests-011/cohesive-reset','Native-011 corrected cohesive reset and coupon checks')
    reset011_record['status']='PASS_NATIVE_SOFTWARE_CHECKS' if reset011_record['receipt'] and reset011_record['receipt'].get('exit_code')==0 and not reset011_record['receipt'].get('timed_out') else 'NATIVE_CHECK_NOT_PASSED'
    capsule011_record=case_binding('native-tests-011/capsule-box','Native-011 capsule/box Armijo and endpoint checks')
    capsule011_record['status']='PASS_NATIVE_SOFTWARE_CHECKS' if capsule011_record['receipt'] and capsule011_record['receipt'].get('exit_code')==0 and not capsule011_record['receipt'].get('timed_out') else 'NATIVE_CHECK_NOT_PASSED'
    box004=case_binding('box/native-004','Frozen-027 instrumented endpoint-gap capture (scope not qualified)')
    box004['raw_status']=box004['status']
    box004['status']='SCOPE_NOT_QUALIFIED'
    box004['scope_reason']='All-selector audit found BOX dispatch paths that bypass the stationary-endpoint gate; accepted step records cannot qualify endpoint-gap admission or box mechanics.'
    box005_record=case_binding('box/native-005','Frozen-028 full-blank native contact attempt; failed at step 32')
    local_projection_result_path=ROOT/'solver/local-projection-accuracy-035/output/result.json'
    local_projection_result=read_json(local_projection_result_path) if local_projection_result_path.is_file() else None
    local_projection_receipt_path=ROOT/'solver/local-projection-accuracy-035/receipt.json'
    local_projection_receipt=read_json(local_projection_receipt_path) if local_projection_receipt_path.is_file() else None
    local_projection_failure_row=None
    local_projection_csv=ROOT/'solver/local-projection-accuracy-035/output/observations.csv'
    if local_projection_csv.is_file():
        with local_projection_csv.open(newline='') as stream:
            local_projection_failure_row=next((
              {k:r.get(k) for k in ('step','environment','arm','status_code','object_index','failing_index','completed_microsteps','fgmres_iterations','step_accepted')}
              for r in csv.DictReader(stream) if r.get('step_accepted')=='0'),None)
    mesh004_records=[]
    for nx in (8,16,32):
        record=case_binding(f'solver/mesh-armijo-004/nx{nx}',f'Frozen-028 prescribed FEM path nx={nx}')
        receipt=record.get('receipt') or {}
        result_path=ROOT/f'solver/mesh-armijo-004/nx{nx}/output/result.json'
        result=read_json(result_path) if result_path.is_file() else None
        if receipt.get('timed_out'):
            record['status']='TIMEOUT_PARTIAL_PATH'
        elif result:
            record['status']=result.get('status')
            record['accepted_steps']=result.get('accepted_steps')
        mesh004_records.append(record)
    named=[
      dict(case_binding('bulk/source-point-004','Preserved invalid rest-tangent instrument attempt'),
           status='INVALID_INSTRUMENTATION',
           interpretation='The fixture supplied F13 while asserting a tensile ZD F33 tangent. It exited before a damage-path experiment and is retained only as a test-instrument defect; no constitutive failure is inferred.'),
      case_binding('bulk/source-point-006','Robertsson full high-shear point path'),
      *(case_binding(f'bulk/source-point-seeded-{n}',f'Seeded continuation, {n} subdivisions') for n in (1,2,4)),
      case_binding('bulk/fem-coupling-001','Fully prescribed one-tet FEM/point parity'),
      case_binding('cohesive/native-003','Separate bleached-paperboard positive Mode-I cohesive coupon'),
      score_record,
      score002_record,
      score003_record,
      *[r for r in later_operator_records if r.get('path')=='solver/local-projection-accuracy-035'],
      operator_fd_baseline,
      operator_fd_diagnostic,
      *later_operator_records,
      reset010_record,
      reset011_record,
      capsule011_record,
      case_binding('box/native-003','FEFCO 0201 full-blank early score-die-contact rejection'),
      box004,
      box005_record,
      *mesh004_records,
      *(dict(case_binding(f'native-tests-009/{name}',f'Corrected native-test 009: {name}'),status=case['status']) for name,case in corrected_tests['cases'].items())]
    material_paths={
      'Robertsson material builder':'bulk/source-point-006/inputs/matter/tools/robertsson2023_material.hpp',
      'Robertsson damage architecture':'bulk/source-point-006/inputs/matter/materials/robertsson2023_bulk_damage_architecture.md',
      'Tryding source material':'cohesive/native-003/inputs/matter/materials/tryding2023_bleached_paperboard_modeI_coupon.nmatter',
      'Haj-Ali liner material':'box/native-003/inputs/matter/materials/hajali2009_liner_hill_ideal.nmatter',
      'Haj-Ali medium material':'box/native-003/inputs/matter/materials/hajali2009_medium_hill_ideal.nmatter',
      'Corrugated elastic glue material':'box/native-003/inputs/matter/materials/starch2007_finite_glue_elastic.nmatter'
    }
    source_files=[]
    for label,rel in material_paths.items():
        p=ROOT/rel
        if not p.is_file():
            # The cohesive paper material can be retained under the supplemental archive.
            candidates=list((ROOT/'cohesive/native-003').glob('supplemental-inputs/**/'+Path(rel).name))
            p=candidates[0] if candidates else p
        if p.is_file(): source_files.append({'label':label,**file_record(p,ROOT)})
    summary={
      'schema':'numi.cardboard.evidence-summary.v1',
      'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'source_root':str(ROOT),
      'scope':'Source-bound software/material-point/FEM coupling/contact evidence. The listed studies do not establish physical cardboard qualification.',
      'source_identities':[
        {'name':'Robertsson et al. 2023 single-ply softwood damage model','doi':'10.1002/pts.2774','grade_limit':'single-ply softwood; not the corrugated kraft/NSSC stack'},
        {'name':'Tryding/Biel bleached clay-coated paperboard Mode-I law','doi':'10.1016/j.ijsolstr.2023.112365','grade_limit':'separate bleached paperboard grade; not corrugated starch'}
      ],
      'source_files_sha256':source_files,
      'material_authoring_limits':{
        'damage_area_reference_mm2':0.0011,
        'authored_default_area_MD_mm2':0.0011,
        'authored_default_area_CD_mm2':0.0011,
        'effective_default_area_ratios':{'MD':1.0,'CD':1.0},
        'per_element_projected_area_scaling_in_current_fem_call_path':False,
        'interpretation':'The source law uses projected-area scaling, but current calls use default unity ratios rather than computed per-tetrahedron material-direction projected areas; native parity is not a mesh-regularization reproduction.',
        'zd_compression_energy_normalization':'The authored branch subtracts an additive constant to impose Psi(I)=0. Fixed-damage stress is unchanged, but the absolute energy level and any energy-derived damage-conjugate term shift; exact source energy-level reproduction is unresolved.'
      },
      'selected_case_records':named,
      'material_point_high_shear':{
        'source_point_006_status':point['status'],'failure':point['failure'],
        'local_projection_iteration_limit':point['failure']['projection_iterations'],
        'convergence_rule':point['failure']['projection_convergence_rule'],
        'residual_inf':point['failure']['projection_residual_inf'],
        'state_inf':point['failure']['projection_state_inf'],
        'fp32_threshold':point['failure']['residual_threshold_fp32'],
        'residual_excess':point['failure']['residual_excess'],
        'seeded_sensitivity':seed
      },
      'fully_prescribed_fem_point_parity':{
        'status':'PASS_MANUFACTURED_COUPLING_PARITY',
        'physical_validation':False,
        'free_mesh_degrees_of_freedom':0,
        'comparison_metrics':fem['comparison']['metrics'],
        'input_scope':fem.get('input_binding',{})
      },
      'separate_mode_i_coupon':cohesive,
      'scored_tool_trial':{
        **score,
        'stationary_endpoint_scope':'SCOPE_NOT_QUALIFIED due to the frozen-027 BOX selector bypass',
        'rejected_row_gap_pair_semantics':'Rollback snapshot of the last accepted state, not the proposed withdrawal endpoint; do not interpret as realized withdrawal contact or separation.'
      },
      'native_score_withdrawal_schedule_comparison':native_score_comparison(),
      'production_operator_fd_diagnostic':{
        'classification':'MATRIX_FREE_OPERATOR_MISMATCH_DIAGNOSTIC_NOT_NATIVE_PATH_REPAIR',
        'postrun_comparison':operator_fd,
        'interpretation':'At the root19/Newton3 state, both nonzero correction perturbations are valid and preserve positive J, but the production matrix-free action disagrees strongly with centered residual differences. The second environment is a zero-direction control and its zero error is vacuous. The diagnostic and baseline native runs have byte-identical CSV, results, compiled pack and all exported mesh hashes, and both naturally reject; the diagnostic does not repair or change the path. The comparison does not identify the cause; projection, active-set behavior, and FP32 effects remain possible contributors.'
      },
      'accepted_history_yield_audit':yield_audit,
      'scaled_operator_fd_sensitivity':scale_comparison if scale_comparison else {
        'status':'PREPARED_NOT_RUN',
        'planned_cases':['solver/operator-fd-002','solver/operator-fd-003'],
        'note':'Postrun comparison waits for both native receipts; it will explicitly compare source/runtime identities and common epsilon rather than claiming input maps match.'
      },
      'alternate_local_projection_threshold_experiment':{
        'case':case_binding('solver/local-projection-accuracy-035','Frozen-035 alternate local projection threshold experiment'),
        'production_default_unchanged':'2e-5',
        'isolated_experiment_threshold':'2e-6',
        'accepted_steps':local_projection_result.get('accepted_steps') if local_projection_result else None,
        'result_status':local_projection_result.get('status') if local_projection_result else None,
        'receipt':local_projection_receipt,
        'first_rejected_row':local_projection_failure_row,
        'failure_subtype':'not_instrumented_or_inferred',
        'interpretation':'The isolated alternate build rejects before a microstep is accepted. This result does not identify cause and does not adopt the alternate threshold.'
      },
      'full_blank_attempt':{
        'status':box['status'],'accepted_steps':box['accepted_steps'],'nodes':box['nodes'],'tetrahedra':box['tetrahedra'],
        'failure':box['failure'],'physical_validation':box['physical_validation'],
        'failing_step_record':next((r for r in csv.DictReader((ROOT/'box/native-003/output/box-native-steps.csv').open()) if r['accepted']=='0'),None)
      },
      'full_blank_frozen028_attempt':{
        'status':box005.get('status'),'accepted_steps':box005.get('accepted_steps'),
        'nodes':box005.get('nodes'),'tetrahedra':box005.get('tetrahedra'),
        'failure':box005.get('failure'),'physical_validation':box005.get('physical_validation'),
        'terminal_record':next((r for r in csv.DictReader((ROOT/'box/native-005/output/box-native-steps.csv').open()) if r['accepted']=='0'),None),
        'scope':'Repaired frozen-028 selector path; early native contact failure before fold/closure actions.'
      },
      'mesh_armijo_004_path_coverage':mesh004_records,
      'frozen027_instrumented_box_capture':{
        'classification':'SCOPE_NOT_QUALIFIED', 'exit_code':2, 'wall_seconds':502.5261783330352,
        'accepted_steps':32,
        'failure':{'native_step':32,'stage':'score_die_contact_approach','status':10,'fgmres':512,'residual':1.63902e-5},
        'endpoint_gate_reason':'BOX dispatch selectors bypass the stationary-endpoint gate; retained as diagnostic evidence, not endpoint admission or box qualification.',
        'physical_validation':False
      },
      'native_tests_008_invocation_audit':invalid,
      'native_tests_009_invocation_audit':corrected_tests,
      'native_tests_010_013':{
        '010_cohesive_reset':reset010_record,
        '011_cohesive_reset_stdout':(ROOT/'native-tests-011/cohesive-reset/stdout.log').read_text().strip(),
        '011_capsule_box_stdout':(ROOT/'native-tests-011/capsule-box/stdout.log').read_text().strip(),
        '012_cohesive_reset':case_binding('native-tests-012/cohesive-reset','Native-012 cohesive reset and paperboard coupon checks'),
        '012_capsule_box':case_binding('native-tests-012/capsule-box','Native-012 capsule/box endpoint, rollback and Armijo checks'),
        'scope_034':case_binding('solver/operator-fd-scope-034','Frozen-034 finite-positive-mass scope admission check'),
        '013_cohesive_reset':case_binding('native-tests-013/cohesive-reset','Native-013 cohesive reset and paperboard coupon checks'),
        '013_capsule_box':case_binding('native-tests-013/capsule-box','Native-013 capsule/box endpoint, rollback and Armijo checks'),
        'scope_036':case_binding('solver/operator-fd-scope-036','Frozen-036 finite-positive-mass scope admission check'),
        'interpretation':'Corrected 011/012 and scope-034 demonstrate native software checks and one eligible finite-mass step. They do not calibrate corrugated cardboard or qualify physical box assembly.'
      },
      'preservation_rule':'Rejected, incomplete, superseded, and invalid-invocation records remain included; none is silently treated as a successful mechanics test.'
    }
    def passed_native_receipt(rel: str) -> bool:
        receipt_path=ROOT/rel/'receipt.json'
        if not receipt_path.is_file(): return False
        receipt=read_json(receipt_path)
        return receipt.get('exit_code')==0 and not receipt.get('timed_out')
    native013_status=('013_PASS' if all(passed_native_receipt(rel) for rel in
                       ('native-tests-013/cohesive-reset','native-tests-013/capsule-box'))
                      else '013_PENDING_OR_NOT_PASSED')
    frontier={
      'schema':'numi.cardboard.evidence-frontier.v1',
      'created_at_utc':summary['created_at_utc'],
      'evidence_summary_sha256':None,
      'capabilities':[
        {'capability':'Published single-ply softwood damage constitutive slice','status':'IMPLEMENTED_BOUNDED_NATIVE_POINT_EVIDENCE','evidence':['bulk/source-point-005','bulk/source-point-006','bulk/source-point-seeded-1','bulk/source-point-seeded-2','bulk/source-point-seeded-4'],'boundary':'The exact 0.005 path rejects at .595; seeded 2/4 subdivisions accept .59→.595 and return only to .59, with small nonzero partition differences. No full high-shear unload-to-zero or physical corrugated validation.'},
        {'capability':'Robertsson FEM state/force coupling','status':'PASS_MANUFACTURED_FULLY_PRESCRIBED_PARITY_ONLY','evidence':['bulk/fem-coupling-001'],'boundary':'One tetrahedron, every node prescribed, zero free mesh degrees of freedom; no independent material oracle or mesh objective proof.'},
        {'capability':'Positive pure-Mode-I cohesive face law','status':'PASS_SEPARATE_BLEACHED_GRADE_COUPON','evidence':['cohesive/native-003'],'boundary':'Single-ply bleached clay-coated paperboard; no mixed mode, contact closure, crack propagation, or corrugated starch calibration.'},
        {'capability':'Corrugated strip scoring contact and withdrawal','status':'DIAGNOSTIC_LOADING_CONTACT_THEN_WITHDRAWAL_REJECTED; STATIONARY_ENDPOINT_UNQUALIFIED','evidence':['scoring/armijo-001'],'boundary':'Accepted indentation force/contact and status-10 first-withdrawal failure are retained as diagnostics. Frozen-027 BOX selectors bypass the stationary-endpoint gate; zero contacts and the gap on the rejected withdrawal row are rollback-snapshot fields, not end-pose evidence. Self-contact was disabled and irreversible crease damage is not qualified.'},
        {'capability':'Frozen-028 prescribed punch withdrawal path','status':'ONE_SHORT_WITHDRAWAL_FAILURE; LONGER_SCHEDULE_COMPLETED_NATIVE_ONLY','evidence':['scoring/armijo-002','scoring/armijo-003'],'boundary':'All 60 accepted rows before withdrawal are cell-identical. The 24-step withdrawal rejects its first unload step; a 96-step withdrawal schedule completes 168/168 at the same fixed dt and same runtime. This changes path duration and is not temporal convergence, calibrated damage, or physical crease validation; self-contact is disabled.'},
        {'capability':'FEFCO 0201 full-blank score/fold/closure','status':'EARLY_NATIVE_CONTACT_FAILURE; FROZEN027 CAPTURE UNQUALIFIED','evidence':['box/native-003','box/native-004','box/native-005'],'boundary':'Native-003 accepted 32 steps then rejected step 32 during score-die contact approach. Frozen-027 native-004 also failed step 32 (residual 1.63902e-5) but its endpoint capture is scope-not-qualified due selector bypass. Repaired frozen-028 native-005 with default self-contact and FGMRES1024 accepted 32 steps then rejected step 32 (status 10, residual 1.25349e-6). No fold/closure actions were completed.'},
        {'capability':'Frozen-028 native FEM path coverage by mesh resolution','status':'ALL_THREE_PATHS_INCOMPLETE','evidence':['solver/mesh-armijo-004/nx8','solver/mesh-armijo-004/nx16','solver/mesh-armijo-004/nx32'],'boundary':'nx8 failed at step43 after43/96 accepted; nx16 failed step19 after19/96; nx32 timed out at1200.027s after17/96 rows without a terminal native failure. These are incomplete path-coverage diagnostics, not mesh convergence.'},
        {'capability':'Frozen-031 production matrix-free operator versus residual finite differences','status':'DIAGNOSTIC_IDENTIFIES_OPERATOR_MISMATCH; NATIVE_PATH_UNCHANGED','evidence':['solver/operator-fd-baseline-001','solver/operator-fd-001'],'boundary':'At root19/Newton3, nonzero epsilon 1/32 and1/64 perturbations have positive J, but operator-versus-centered-FD relative errors are about1.0224 and1.0248 with cosines about−0.0671 and0.00175. The second environment has a zero correction direction and is only a vacuous control. Baseline and diagnostic outputs are byte-identical and both naturally reject; this is not a solver repair, convergence result, or physical claim.'},
        {'capability':'Host-scaled operator-FD perturbation sensitivity','status':('DIAGNOSTIC_SENSITIVITY_MEASURED_NOT_CAUSE_IDENTIFIED' if scale_comparison else 'PREPARED_NOT_RUN'),'evidence':['solver/operator-fd-baseline-001','solver/operator-fd-002','solver/operator-fd-003'],'boundary':'Frozen-033 changes the host-only diagnostic epsilon scale; a common epsilon 0.25 comparison checks cross-scale repeatability while production Metal remains hash-bound. Even exact repeatability would characterize only the selected rejected state and would not identify projection, active-set, FP32, or other causes or certify the Jacobian.'},
        {'capability':'Alternate local projection threshold diagnostic','status':('REJECTED_AFTER_2_ACCEPTED_STEPS_NOT_ADOPTED' if local_projection_receipt else 'PREPARED_NOT_RUN'),'evidence':['solver/local-projection-accuracy-035'],'boundary':'An isolated Metal build changes the local projection threshold from 2e-5 to 2e-6; it rejects the next step with status 5 at element 1430 before microstep completion or FGMRES. The terminal subtype/cause is not inferred; production remains at 2e-5.'},
        {'capability':'Native endpoint and cohesive reset regressions 008–013','status':f'008_SUPERSEDED; 009_PASS; 010_PARTIAL_FIXTURE_FAILURE; 011/012_PASS; {native013_status}','evidence':['native-tests-008','native-tests-009','native-tests-010/cohesive-reset','native-tests-011/cohesive-reset','native-tests-011/capsule-box','native-tests-012/cohesive-reset','native-tests-012/capsule-box','native-tests-013/cohesive-reset','native-tests-013/capsule-box'],'boundary':'Frozen-027 008 endpoint observation is not moved-body proof; four 008 calls are UNEXECUTED_INVALID_INVOCATION. Corrected 009 passed five calls. Native010 ran earlier damage/reset/isolation assertions but its final rejected-reset target was not admitted due an environment×node count bug in the fixture. Corrected 011/012 pass cohesive reset and capsule/box endpoint software checks. Native013 outcomes are classified from receipts and remain software checks, not corrugated calibration or physical box qualification.'}
      ],
      'open_scientific_gaps':[
        'No experimentally calibrated bulk crease-damage parameters for the specific corrugated kraft liner and NSSC medium stack; Robertsson softwood values cannot be transferred by assumption.',
        'No local starch cohesive traction, fracture energy, mixed-mode interaction or unloading/closure parameters identified for the current corrugated glue region; glue remains elastic/tied.',
        'No validated tetrahedral damage regularization/mesh-objectivity study against matching board experiments.',
        'No experimentally validated irreversible crease cycle or physical board validation; one prescribed native punch/full-withdrawal schedule completes, but self-contact is disabled and no matching-grade crease law is calibrated. Full-box fold-and-close remains unqualified.'
      ],
      'pending_case_roots_at_generation':[
        rel for rel in ['scoring/armijo-002','scoring/armijo-003','box/native-005',
                        'native-tests-010/cohesive-reset','native-tests-011/cohesive-reset',
                        'native-tests-011/capsule-box','solver/operator-fd-baseline-001',
                        'solver/operator-fd-001','solver/operator-fd-002',
                        'solver/operator-fd-003','solver/operator-fd-scope-034',
                        'native-tests-012/cohesive-reset','native-tests-012/capsule-box',
                        'native-tests-013/cohesive-reset','native-tests-013/capsule-box',
                        'solver/operator-fd-scope-036',
                        'solver/local-projection-accuracy-035']
        if not (ROOT/rel/'receipt.json').is_file()
      ] + (['solver/mesh-armijo-004'] if not all((ROOT/f'solver/mesh-armijo-004/nx{nx}/receipt.json').is_file() for nx in (8,16,32)) else []),
      'claim_limit':'Software checks, source-grade coupons and manufactured FEM parity are not physical cardboard qualification.'
    }
    summary_path=OUT/'evidence-summary.json'; frontier_path=OUT/'evidence-frontier.json'
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    frontier['evidence_summary_sha256']=sha(summary_path)
    frontier_path.write_text(json.dumps(frontier,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'summary':str(summary_path),'frontier':str(frontier_path),'case_count':len(named),'source_files':len(source_files),'frontier_statuses':[(x['capability'],x['status']) for x in frontier['capabilities']]},indent=2))

if __name__=='__main__': main()
