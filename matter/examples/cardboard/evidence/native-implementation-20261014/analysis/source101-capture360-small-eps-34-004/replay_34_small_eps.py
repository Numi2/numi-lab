#!/usr/bin/env python3
"""One bounded CPU FP64 smaller-step replay of the 34 rows failing 1e-6 in directional003.

This is a diagnostic subset of a rejected Newton capture, not a native or physical result.
It keeps accepted history fixed at both endpoints and records every failure/branch change.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, pathlib, signal, struct, sys, time
import numpy as np

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
DIRECTIONAL=BASE/'analysis/source101-capture360-directional-fd-003'
DIRECTIONAL_REPORT=DIRECTIONAL/'output/report.json'
DIRECTIONAL_ROWS=DIRECTIONAL/'output/all_1728_row_metrics.csv'
DIRECTIONAL_MANIFEST=DIRECTIONAL/'manifest.json'
FP_HELPER=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
FP_REFERENCE=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
MP90=BASE/'analysis/source101-capture360-selected-mp90-002'
MP90_REPORT=MP90/'output/report.json'
MP90_SELECTED=MP90/'selected-summary.json'
OUT=pathlib.Path(__file__).resolve().parent/'output-002'
TARGETS=(1.0e-7,3.0e-8,1.0e-8)
LOCAL_RESIDUAL_TOL=2.0e-10
TIME_BUDGET_SECONDS=450.0

class BoundedTimeout(Exception): pass

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader: raise RuntimeError(f'cannot load {path}')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def parse_int(token): return -0.0 if token=='-0' else int(token)
def f32words(xs):
    return [f'0x{struct.unpack("<I",struct.pack("<f",float(x)))[0]:08x}'
            for x in np.asarray(xs).reshape(-1)]

def branch_signature(ref,p,old,z):
    z=np.asarray(z,dtype=np.float64).reshape(15)
    old=np.asarray(old,dtype=np.float64).reshape(15)
    pp=np.asarray(p,dtype=np.float64).reshape(28)
    dl=float(z[12]-old[12])
    q=float(ref.hill(pp,z[6:12])[0])
    gap=float(1.0+pp[17]*pp[19]*pp[21]*z[13]/pp[10]+
              pp[18]*pp[20]*pp[22]*z[14]/pp[10]-q)
    fb='elastic_smoothed' if gap>dl else 'plastic_smoothed'
    return {
        'delta_multiplier':dl,
        'multiplier_sign':'positive' if dl>0.0 else ('negative' if dl<0.0 else 'zero'),
        'yield_gap':gap,
        'yield_sign':'nonnegative' if gap>=0.0 else 'negative',
        'fb_selection':fb,
        'hill_q':q,
        'hardening_factor':float(1.0+gap+q),
    }

def branch_key(b): return (b['multiplier_sign'],b['yield_sign'],b['fb_selection'])

def projection_record(ref,p,F,old,z,iterations,residual):
    scales=np.array([1e4]*6+[1.0]*6+[1e4]*3,dtype=np.float64)
    r=np.asarray(ref.residual(p[None,:],F[None,:,:],old[None,:],z[None,:]),dtype=np.float64).reshape(15)
    scaled_res=float(np.max(np.abs(r)*scales))
    jac=np.asarray(ref.jacobian(p[None,:],F[None,:,:],old[None,:],z[None,:]),dtype=np.float64).reshape(15,15)
    correction=np.linalg.solve(jac,-r)
    correction_inf=float(np.max(np.abs(correction)))
    relative_scaled_correction=float(max(abs(correction[i])/max(abs(z[i]),abs(old[i]),1e-8) for i in range(15)))
    finite=all(math.isfinite(x) for x in (scaled_res,correction_inf,relative_scaled_correction)) and np.all(np.isfinite(z))
    return {
        'projector_returned':True,
        'projector_iterations':int(iterations),
        'projector_reported_scaled_residual':float(residual),
        'recomputed_scaled_residual':scaled_res,
        'absolute_newton_correction_inf':correction_inf,
        'scaled_newton_correction_inf':relative_scaled_correction,
        'residual_converged_at_reference_tolerance':bool(finite and scaled_res<=LOCAL_RESIDUAL_TOL),
        'scaled_correction_converged_at_reference_tolerance':bool(finite and relative_scaled_correction<=LOCAL_RESIDUAL_TOL),
        'both_residual_and_scaled_correction_converged':bool(finite and scaled_res<=LOCAL_RESIDUAL_TOL and relative_scaled_correction<=LOCAL_RESIDUAL_TOL),
        'branch':branch_signature(ref,p,old,z),
    }

def project_one(fd,ref,p,F,old):
    z,it,res,ok,why=fd.project_resilient(ref,p[None,:],F[None,:,:],old[None,:],maxiter=80)
    if not bool(ok[0]): return None, {'projector_returned':False,'failure':why[0]}
    zz=np.asarray(z[0],dtype=np.float64)
    return zz,projection_record(ref,p,F,old,zz,int(it[0]),float(res[0]))

def rel_metrics(a,b):
    aa=np.asarray(a,dtype=np.float64).reshape(-1)
    bb=np.asarray(b,dtype=np.float64).reshape(-1)
    diff=aa-bb
    den=float(np.linalg.norm(bb))
    return {
        'absolute_l2':float(np.linalg.norm(diff)),
        'max_component_absolute':float(np.max(np.abs(diff))),
        'reference_l2':den,
        'relative_l2':float(np.linalg.norm(diff)/den) if den>0.0 else None,
    }

class RowTimeout(Exception): pass

def main():
    if OUT.exists(): raise RuntimeError(f'refusing to overwrite output directory {OUT}')
    OUT.mkdir(parents=True,exist_ok=False)
    need_hashes={
        'capture':(CAP,CAP_SHA),
        'directional_rows':(DIRECTIONAL_ROWS,'226a12e13a5dae5cf17f4189a65d71ba6a5e26c94d14d11407e5712a76979180'),
        'directional_report':(DIRECTIONAL_REPORT,'0b97764b85914b19931984b392a3ec4d7d8e0a29313ff82924f0321248e2791d'),
        'directional_manifest':(DIRECTIONAL_MANIFEST,None),
        'fp64_reference':(FP_REFERENCE,'c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3'),
        'directional_helper':(FP_HELPER,'742b0920c4f543e284e41ed18f8dbc78253915a31f39b3ef8d32ff7dec01b5e8'),
        'mp90_report':(MP90_REPORT,'f029e40fbd720c6a2589451b9f927dab3285ea921ddb45eefca366c77ebe4b37'),
        'mp90_selected_summary':(MP90_SELECTED,'1146ab9891d10074f526d8e0828b4b22d20d50a95ce6183fe1dc339d9ef12941'),
        'mp90_runner':(MP90/'replay_selected.py','aa73eda519a30807f00b741d3b6cc9b73fb8434e9082fbf39955739572f5c089'),
    }
    for name,(path,expected) in need_hashes.items():
        actual=sha(path)
        if expected is not None and actual!=expected: raise RuntimeError(f'{name} SHA mismatch: {actual}')
    directional_manifest=json.loads(DIRECTIONAL_MANIFEST.read_text())
    if directional_manifest['capture_sha256']!=CAP_SHA or directional_manifest['reference_sha256']!=sha(FP_REFERENCE):
        raise RuntimeError('directional003 source bindings do not match replay inputs')
    mp90_report=json.loads(MP90_REPORT.read_text())
    mp90_points={int(point['tet_id']):point for point in mp90_report['points']}
    rows=list(csv.DictReader(DIRECTIONAL_ROWS.open(newline='')))
    selected=[r for r in rows if float(r['central_fd_relative_error'])>1.0e-6]
    selected.sort(key=lambda r:int(r['tet_id']))
    if len(rows)!=1728 or len(selected)!=34: raise RuntimeError(f'expected 1728 source rows and exactly 34 >1e-6, got {len(rows)} / {len(selected)}')
    selected_ids=[int(r['tet_id']) for r in selected]
    if len(set(selected_ids))!=34: raise RuntimeError('directional subset contains duplicate IDs')
    fd=load_module(FP_HELPER,'directional_fd_003_replay')
    ref=load_module(FP_REFERENCE,'bilinear15_fp64_reference_34')
    records=[]
    for line_number,line in enumerate(CAP.open(),1):
        if line.strip():
            records.append(json.loads(line,parse_int=parse_int))
    header=records[0]
    if header.get('record')!='header' or (header.get('root'),header.get('newton_iteration'))!=(2,23):
        raise RuntimeError('capture selection is not root2/Newton23')
    if records[-1].get('record')!='integrity' or records[-1].get('production_inputs_bitwise_unchanged') is not True:
        raise RuntimeError('captured source inputs lack unchanged integrity record')
    by_id={(int(r['environment']),int(r['tet_id'])):r for r in records if r.get('record')=='tetrahedron'}
    ptab=fd.f32(header['parameters_by_environment'][0])
    started=time.monotonic()
    timeout_hit=False
    completed=[]
    completed_records=[]
    row_path=OUT/'rows.jsonl'
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(BoundedTimeout()))
    signal.setitimer(signal.ITIMER_REAL,TIME_BUDGET_SECONDS)
    try:
        with row_path.open('x') as output:
            for selection in selected:
                tet_id=int(selection['tet_id'])
                native=by_id.get((0,tet_id))
                if native is None: raise RuntimeError(f'missing selected capture row {tet_id}')
                mat=int(native['material_index'])
                expected_mat=0 if selection['material']=='liner' else 1
                if mat!=expected_mat or int(native.get('material_state_count',-1))!=15:
                    raise RuntimeError(f'capture/directional row identity mismatch at tet {tet_id}')
                p=np.asarray(fd.f32(ptab[mat*28:(mat+1)*28]),dtype=np.float64)
                center=native['center']
                old=np.asarray(fd.f32(center['accepted_state']),dtype=np.float64)
                Fhi=fd.f32(center['F_local']); Flo=fd.f32(center.get('F_local_low',np.zeros_like(center['F_local'])))
                F=(Fhi+Flo).reshape(3,3).astype(np.float64)
                h=np.asarray(fd.f32(center['F_direction_local_per_s']),dtype=np.float64).reshape(3,3)
                dt=float(fd.f32([center['dt_seconds_bulk_modulus_kind_frame_active'][0]])[0])
                dFbase=dt*h
                direction_norm=float(np.linalg.norm(dFbase))
                if not math.isfinite(direction_norm) or direction_norm==0.0: raise RuntimeError(f'bad direction at tet {tet_id}')
                z0,center_diag=project_one(fd,ref,p,F,old)
                result={
                    'environment':0,'root':2,'newton_iteration':23,'tet_id':tet_id,'material_index':mat,
                    'source_material':selection['material'],
                    'original_directional003_relative_error':float(selection['central_fd_relative_error']),
                    'accepted_history_seed':'same captured float32 accepted_state for center and every plus/minus endpoint; perturbed projected states are never reused as history',
                    'center_F_input':'captured float32 F_local high+low words widened exactly to float64',
                    'center_F_high_f32_bits':f32words(Fhi.reshape(-1)),
                    'center_F_low_f32_bits':f32words(Flo.reshape(-1)),
                    'accepted_history_f32_bits':f32words(old),
                    'parameters_f32_bits':f32words(p),
                    'direction_f32_bits':f32words(h.reshape(-1)),
                    'dt_f32_bits':f32words([dt]),
                    'captured_rate_direction_norm':direction_norm,
                    'center_projection':center_diag,
                    'epsilon_cases':[],
                }
                if z0 is not None:
                    P0=np.asarray(ref.stress(p[None,:],F[None,:,:],z0[None,:]),dtype=np.float64).reshape(3,3)
                    tangent=np.asarray(ref.tangent(p[None,:],F[None,:,:],old[None,:],z0[None,:],h[None,:,:]),dtype=np.float64).reshape(3,3)
                    result['center_reference_stress_Pa']=P0.tolist()
                    result['center_rate_tangent_action_Pa_per_s']=tangent.tolist()
                    result['center_action_denominator_l2_Pa_per_s']=float(np.linalg.norm(tangent))
                    # MP90-v2 contains two overlapping rows in this 34-row subset.
                    if tet_id in mp90_points:
                        mp=center_mp=mp90_points[tet_id]['center']
                        mbranch=mp['mp90_projection']['branch']
                        result['mp90_center_crosscheck']={
                            'source_mp90_material_label':mp90_points[tet_id]['label'],
                            'mp90_converged':mp['mp90_projection']['converged'],
                            'mp90_branch':{k:mbranch.get(k) for k in ('yield_sign','fb_selection')},
                            'fp64_vs_mp90_state':mp['fp64_state_vs_mp90_root'],
                            'fp64_vs_mp90_stress_Pa':mp['fp64_stress_vs_mp90_Pa'],
                            'fp64_vs_mp90_tangent_action':mp['fp64_directional_tangent_vs_mp90_Pa_per_s'],
                            'branch_signature_matches':(
                                result['center_projection']['branch']['yield_sign']==mbranch.get('yield_sign') and
                                result['center_projection']['branch']['fb_selection']==mbranch.get('fb_selection') and
                                result['center_projection']['branch']['multiplier_sign']==('positive' if float(mbranch.get('delta_multiplier',0.0))>0 else 'zero')
                            ),
                        }
                    for target in TARGETS:
                        eps=target/direction_norm
                        Fplus=F+eps*dFbase
                        Fminus=F-eps*dFbase
                        actual_plus=Fplus-F
                        actual_minus=F-Fminus
                        zplus,plus_diag=project_one(fd,ref,p,Fplus,old)
                        zminus,minus_diag=project_one(fd,ref,p,Fminus,old)
                        case={
                            'target_deltaF_frobenius':target,
                            'epsilon_scalar':eps,
                            'actual_deltaF_plus_frobenius':float(np.linalg.norm(actual_plus)),
                            'actual_deltaF_minus_frobenius':float(np.linalg.norm(actual_minus)),
                            'plus_minus_realized_direction_match':rel_metrics(actual_plus/(eps*dt),h),
                            'plus_projection':plus_diag,
                            'minus_projection':minus_diag,
                        }
                        if zplus is not None and zminus is not None:
                            pplus=np.asarray(ref.stress(p[None,:],Fplus[None,:,:],zplus[None,:]),dtype=np.float64).reshape(3,3)
                            pminus=np.asarray(ref.stress(p[None,:],Fminus[None,:,:],zminus[None,:]),dtype=np.float64).reshape(3,3)
                            fdcentral=(pplus-pminus)/(2.0*eps*dt)
                            fdplus=(pplus-P0)/(eps*dt)
                            fdminus=(P0-pminus)/(eps*dt)
                            signature_stable=(branch_key(plus_diag['branch'])==branch_key(center_diag['branch'])==branch_key(minus_diag['branch']))
                            both_endpoint_converged=(plus_diag['both_residual_and_scaled_correction_converged'] and minus_diag['both_residual_and_scaled_correction_converged'])
                            central_error=rel_metrics(fdcentral,tangent)
                            strict_status=(
                                'inconclusive_endpoint_convergence' if not both_endpoint_converged else
                                ('inconclusive_branch_change' if not signature_stable else
                                 ('within_threshold_for_this_direction_and_step' if central_error['relative_l2'] is not None and central_error['relative_l2']<=1e-6 else 'exceeds_threshold_for_this_direction_and_step')))
                            case.update({
                                'plus_stress_Pa':pplus.tolist(),'minus_stress_Pa':pminus.tolist(),
                                'central_fd_action_Pa_per_s':fdcentral.tolist(),
                                'central_fd_vs_fp64_tangent':central_error,
                                'plus_one_sided_vs_fp64_tangent':rel_metrics(fdplus,tangent),
                                'minus_one_sided_vs_fp64_tangent':rel_metrics(fdminus,tangent),
                                'one_sided_gap':rel_metrics(fdplus,fdminus),
                                'branch_signature_stable':bool(signature_stable),
                                'both_endpoints_fully_converged':bool(both_endpoint_converged),
                                'strict_1e-6_diagnostic':strict_status,
                            })
                        else:
                            case.update({'branch_signature_stable':False,'both_endpoints_fully_converged':False,
                                         'strict_1e-6_diagnostic':'inconclusive_projection_failure'})
                        result['epsilon_cases'].append(case)
                else:
                    result['center_reference_stress_Pa']=None
                    result['center_rate_tangent_action_Pa_per_s']=None
                    result['center_action_denominator_l2_Pa_per_s']=None
                    result['row_status']='center_projection_failed'
                output.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
                output.flush(); os.fsync(output.fileno())
                completed.append(tet_id)
                completed_records.append(result)
                print(f'completed row {len(completed)}/34 tet={tet_id}',flush=True)
    except BoundedTimeout:
        timeout_hit=True
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.0)
    elapsed=time.monotonic()-started
    endpoint_summary=[]
    for target in TARGETS:
        cases=[case for row in completed_records for case in row.get('epsilon_cases',[])
               if case['target_deltaF_frobenius']==target]
        endpoint_summary.append({
            'target_deltaF_frobenius':target,
            'endpoint_pair_count':len(cases),
            'both_endpoints_residual_and_scaled_correction_converged_count':sum(c['both_endpoints_fully_converged'] for c in cases),
            'both_endpoints_same_branch_signature_count':sum(c['branch_signature_stable'] for c in cases),
            'both_converged_and_same_branch_count':sum(c['both_endpoints_fully_converged'] and c['branch_signature_stable'] for c in cases),
            'diagnostic_within_1e-6_count':sum(c['strict_1e-6_diagnostic']=='within_threshold_for_this_direction_and_step' for c in cases),
            'inconclusive_count':sum(c['strict_1e-6_diagnostic'].startswith('inconclusive') for c in cases),
        })
    center_residual_failures=[r['tet_id'] for r in completed_records
                              if not r.get('center_projection',{}).get('residual_converged_at_reference_tolerance',False)]
    center_scaled_correction_failures=[r['tet_id'] for r in completed_records
                                       if not r.get('center_projection',{}).get('scaled_correction_converged_at_reference_tolerance',False)]
    report={
        'schema':'numi.cardboard.source101-capture360-small-eps-34.v1',
        'classification':'One bounded CPU-only independent FP64 sensitivity replay of exactly the 34 rows above 1e-6 in directional003. Rejected Newton 23 only; not a native noise-floor study, full local derivative qualification, coupled-system result, or physical validation.',
        'status':'timed_out_partial' if timeout_hit else 'completed_subset',
        'binding':{
            'capture_path':str(CAP),'capture_sha256':sha(CAP),
            'directional003_manifest_path':str(DIRECTIONAL_MANIFEST),'directional003_manifest_sha256':sha(DIRECTIONAL_MANIFEST),
            'directional003_rows_path':str(DIRECTIONAL_ROWS),'directional003_rows_sha256':sha(DIRECTIONAL_ROWS),
            'directional003_report_path':str(DIRECTIONAL_REPORT),'directional003_report_sha256':sha(DIRECTIONAL_REPORT),
            'fp64_reference_path':str(FP_REFERENCE),'fp64_reference_sha256':sha(FP_REFERENCE),
            'directional_helper_path':str(FP_HELPER),'directional_helper_sha256':sha(FP_HELPER),
            'mp90_v2_report_path':str(MP90_REPORT),'mp90_v2_report_sha256':sha(MP90_REPORT),
            'mp90_v2_selected_summary_path':str(MP90_SELECTED),'mp90_v2_selected_summary_sha256':sha(MP90_SELECTED),
            'mp90_v2_runner_path':str(MP90/'replay_selected.py'),'mp90_v2_runner_sha256':sha(MP90/'replay_selected.py'),
            'failed_preliminary_attempt':{
                'output_path':str(pathlib.Path(__file__).resolve().parent/'output'),
                'status':'failed_before_first_row',
                'cause':'diagnostic recorder referenced the newly-added central_fd_vs_fp64_tangent key while constructing the same dict update; no projections were counted',
            },
            'source_snapshot':'source-boundary-reset-order-101',
        },
        'selection':{
            'rule':'exactly directional003 all_1728_row_metrics.csv rows whose central_fd_relative_error > 1e-6',
            'selected_count':len(selected),'completed_count':len(completed),'completed_tet_ids':completed,
            'selected_tet_ids':selected_ids,
            'selected_tet_ids_u32le_sha256':hashlib.sha256(np.asarray(selected_ids,dtype='<u4').tobytes()).hexdigest(),
            'by_material':{'liner':sum(r['material']=='liner' for r in selected),'medium':sum(r['material']=='medium' for r in selected)},
            'center_projection_residual_failure_tet_ids':center_residual_failures,
            'center_projection_scaled_correction_failure_tet_ids':center_scaled_correction_failures,
        },
        'endpoint_summary':endpoint_summary,
        'numerics':{
            'target_deltaF_frobenius_norms':list(TARGETS),
            'perturbation':'F± = captured F_high32 + F_low32 + epsilon * dt32 * captured Fdot32, evaluated in independent FP64; epsilon = target Frobenius norm / ||dt*Fdot||. Actual realized FP64 endpoint directions and norms are recorded.',
            'accepted_history':'same captured FP32 accepted old-state widened exactly to FP64 for center and every endpoint; endpoint projected state is never reused as history; reference projector initializes each endpoint from that fixed accepted state and computes elastic trial state.',
            'reference_projector':'bilinear15_reference.py stock FP64 safeguarded projection, max 80, residual stop 2e-10 scaled.',
            'additional_endpoint_check':'At every returned endpoint, independently recompute scaled residual, absolute Newton correction, and state-scaled Newton correction max(abs(delta_i)/max(abs(z_i),abs(old_i),1e-8)); residual and scaled-correction criteria are reported separately. Strict row comparisons are diagnostic only when both endpoints satisfy both checks and reported branch signature is unchanged.',
            'branch_signature':'Exact multiplier increment sign, yield-gap sign, and elastic_smoothed/plastic_smoothed complementarity selector. Bilinear hardening contributes the smooth factor 1+yield_gap+Hill_q; the independent evaluator exposes no separate hardening max/sign branch, so no such branch stability is claimed.',
            'mp90_use':'MP90-v2 is bound as an independent evaluator cross-check for overlapping selected rows (tet 457 and 1635); its own original three-point/ten-target outputs are not rerun or overwritten. It does not validate all 34 centers.',
            'noise_floor':'No native FP32 noise-floor estimate is made; decreasing-epsilon stability is not called a noise floor.',
            'time_budget_seconds':TIME_BUDGET_SECONDS,
            'elapsed_seconds':elapsed,
        },
        'limits':['Only the listed subset of one rejected Newton 23 capture is replayed.','No physical transaction, native Metal, or accepted whole-system step is implied.','The smaller-step derivative comparisons do not change the 1e-6 threshold and are not all-nine-direction or full-material qualification.','All rows, branch changes, projection failures, and nonconverged endpoints are retained.'],
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'report':str(OUT/'report.json'),'status':report['status'],'completed':len(completed),'elapsed_seconds':elapsed,'ids':completed},indent=2),flush=True)
    if timeout_hit: raise SystemExit(124)

if __name__=='__main__':
    main()
