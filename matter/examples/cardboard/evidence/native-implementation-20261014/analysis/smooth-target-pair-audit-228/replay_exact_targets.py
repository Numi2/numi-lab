#!/usr/bin/env python3
"""Source-bound exact FP32-word replay for six selected paper tets in 216/218/220."""
import hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
OUT = BASE/'analysis/smooth-target-pair-audit-228'
CAPS = {
    '216': BASE/'solver/graph-schwarz-first-control-final-newton-capture-216/operator-capture',
    '218': BASE/'solver/graph-schwarz-final-newton-fd-scale-1over16-218/operator-capture',
    '220': BASE/'solver/graph-schwarz-final-newton-fd-scale-1over256-220/operator-capture',
}
TARGETS = {1388:'smooth_liner', 633:'smooth_medium', 441:'smooth_liner',
           518:'yield_control_liner', 702:'yield_control_medium', 776:'yield_control_medium'}
REPLAY = BASE/'analysis/smooth-target-pair-audit-225/build/cardboard_solver_pair_replay_realized'
SOURCE_ROOT = BASE/'snapshots/source-refined-graph-angular-residual-instrument-071'
MATS = [SOURCE_ROOT/'matter/materials/hajali2009_liner_hill_ideal.nmatter',
        SOURCE_ROOT/'matter/materials/hajali2009_medium_hill_ideal.nmatter']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def f32(x):
    a=np.asarray(x,dtype=np.float32)
    if not np.isfinite(a).all(): raise ValueError('nonfinite FP32 capture word')
    return a

def vec(x,n=None):
    a=f32(x).reshape(-1)
    if n is not None and len(a)!=n: raise ValueError(f'expected {n} words, got {len(a)}')
    return a

def words(a):
    return ' '.join(format(float(x),'.17g') for x in np.asarray(a).reshape(-1))

def flatf(a,n):
    return vec(a,n).astype(np.float64)

def maxabs(a):
    a=np.asarray(a,dtype=np.float64)
    return float(np.max(np.abs(a))) if a.size else 0.0

def metric(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1); b=np.asarray(b,dtype=np.float64).reshape(-1)
    d=maxabs(a-b); s=maxabs(b)
    return {'absolute_error':d,'reference_max_abs':s,'relative_error':d/s if s>0 else None}

def branch(q):
    if abs(q-1.0) <= 1e-6: return 'transition_unresolved'
    return 'plastic' if q>1 else 'elastic'

def rotation(q):
    x,y,z,w=map(float,vec(q,4)); n=math.sqrt(x*x+y*y+z*z+w*w)
    if not n: raise ValueError('zero quaternion')
    x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)

def element_force(p, inv, vol):
    h=-vol*np.asarray(p,dtype=np.float64).reshape(3,3)@np.asarray(inv,dtype=np.float64).reshape(3,3).T
    return np.vstack((-h.sum(axis=1),h.T))

def native_P(s):
    return flatf(s['P_local_Pa'],9)

def world_force_from_native(s,tet):
    return element_force(f32(s['P_world_Pa']),f32(tet['inverse_rest_rows_per_m']),float(f32(tet['rest_volume_m3'])))

def addrow(rows, keys, src, key, kind, tet, state, direction, params):
    fhi=vec(state['F_local'],9); flo=vec(state.get('F_local_low',np.zeros(9)),9)
    chi=vec(state['projected_state'],13); clo=vec(state.get('projected_state_low',np.zeros(13)),13)
    old=vec(tet['center']['accepted_state'],13)
    dhi=f32(np.asarray(direction,dtype=np.float64)); dlo=f32(np.asarray(direction,dtype=np.float64)-dhi.astype(np.float64))
    row=' '.join([key,kind,words(fhi),words(flo),words(dhi),words(dlo),words(old),words(chi),words(clo),words(vec(params,20))])
    rows.append(row); keys.append(key); src[key]=(tet,state)

all_results={}
for label,cap in CAPS.items():
    recs=[]
    with open(cap) as f:
        for line in f:
            if line.strip(): recs.append(json.loads(line))
    if len(recs)<3 or recs[0].get('record')!='header' or recs[-1].get('record')!='integrity':
        raise ValueError(f'{label}: missing header/integrity')
    header=recs[0]; integrity=recs[-1]
    if header.get('schema')!='numi.cardboard.fem-newton-capture.v1' or not integrity.get('production_inputs_bitwise_unchanged'):
        raise ValueError(f'{label}: capture schema or input integrity invalid')
    if (header.get('root'),header.get('newton_iteration'),header.get('environment_count'))!=(0,23,2):
        raise ValueError(f'{label}: unexpected capture selection')
    dt=float(f32(header['dt_seconds']))
    params_all=vec(header['parameters_by_environment'][0])
    if len(params_all)<43: raise ValueError('parameter array is incomplete')
    materials={m['index']:m for m in header['materials']}
    param_by_kind={}
    for mi,k in ((0,'liner'),(1,'medium')):
        m=materials[mi]; off=int(m['parameter_offset'])
        vals=params_all[off:off+20]
        if m['name']!=f'hajali2009_{k}_hill_ideal': raise ValueError('captured material identity mismatch')
        param_by_kind[k]=vals
    found={}; rows=[]; keys=[]; src={}
    for tet in recs[1:-1]:
        if tet.get('record')!='tetrahedron' or int(tet.get('environment',-1))!=0: continue
        tid=int(tet['tet_id'])
        if tid not in TARGETS: continue
        kind='liner' if int(tet['material_index'])==0 else 'medium' if int(tet['material_index'])==1 else None
        if kind is None or kind!=TARGETS[tid].split('_')[-1]: raise ValueError(f'tet {tid} material mismatch')
        center=tet['center']; direction=vec(center['F_direction_local_per_s'],9)
        prefix=f'e0_t{tid}'
        addrow(rows,keys,src,prefix+'_c',kind,tet,center,direction,param_by_kind[kind])
        by_pair={}
        fc=vec(center['F_local'],9).astype(np.float64)+vec(center.get('F_local_low',np.zeros(9)),9).astype(np.float64)
        for si,s in enumerate(tet['sides']):
            eps=float(f32(s['epsilon']))
            fs=vec(s['F_local'],9).astype(np.float64)+vec(s.get('F_local_low',np.zeros(9)),9).astype(np.float64)
            signed=float(np.dot(fs-fc,direction.astype(np.float64)))
            side='plus' if signed>0 else 'minus' if signed<0 else 'zero'
            pair=int(np.argmin(np.abs(np.asarray(header['epsilon_values'],dtype=np.float64)-eps)))
            by_pair.setdefault(pair,{})[side]=s
            addrow(rows,keys,src,f'{prefix}_s{si}',kind,tet,s,direction,param_by_kind[kind])
        # Calculate exact realized direction from the FP32 high+low endpoint pairs,
        # and replay its center tangent separately (rounded to the replay's FP32 H input).
        for pair, sides in by_pair.items():
            if 'plus' not in sides or 'minus' not in sides: raise ValueError(f'{prefix} pair {pair}: missing signed side')
            eps=float(f32(sides['plus']['epsilon']))
            fp=vec(sides['plus']['F_local'],9).astype(np.float64)+vec(sides['plus'].get('F_local_low',np.zeros(9)),9).astype(np.float64)
            fm=vec(sides['minus']['F_local'],9).astype(np.float64)+vec(sides['minus'].get('F_local_low',np.zeros(9)),9).astype(np.float64)
            heff=(fp-fm)/(2*eps*dt)
            addrow(rows,keys,src,f'{prefix}_h{pair}',kind,tet,center,heff,param_by_kind[kind])
        found[tid]={'tet':tet,'pair_sides':by_pair}
    if set(found)!=set(TARGETS): raise ValueError(f'{label}: selected IDs missing: {set(TARGETS)-set(found)}')
    input_text='\n'.join(rows)+'\n'
    run=OUT/f'run{label}'; run.mkdir(exist_ok=False)
    (run/'replay-input.txt').write_text(input_text)
    cp=subprocess.run([str(REPLAY),str(MATS[0]),str(MATS[1]),'--direction-high-low'],input=input_text,text=True,capture_output=True)
    (run/'replay-stdout.jsonl').write_text(cp.stdout); (run/'replay-stderr.txt').write_text(cp.stderr)
    if cp.returncode not in (0,2): raise RuntimeError(f'{label}: replay failed {cp.returncode}: {cp.stderr[:1000]}')
    oracle=[json.loads(x) for x in cp.stdout.splitlines() if x.strip()]
    if len(oracle)!=len(keys) or [o['key'] for o in oracle]!=keys: raise RuntimeError(f'{label}: replay row mismatch')
    bykey={o['key']:o for o in oracle}
    per_tet=[]
    for tid,obj in found.items():
        tet=obj['tet']; center=tet['center']; k=f'e0_t{tid}'; kind='liner' if tet['material_index']==0 else 'medium'
        oc=bykey[k+'_c']; native_center=native_P(center)
        native_cand=vec(center['projected_state'],13).astype(np.float64)+vec(center.get('projected_state_low',np.zeros(13)),13).astype(np.float64)
        qcenter=float(oc['elastic_trial_q']); branch_center=branch(qcenter)
        tangent_native=flatf(center['tangent_local_Pa_per_s'],9)
        t_candidate=np.asarray(oc['oracle_tangent_at_captured_candidate_Pa_per_s'],dtype=np.float64)
        t_reproject=np.asarray(oc['oracle_tangent_at_reprojected_state_Pa_per_s'],dtype=np.float64)
        state_proj=np.asarray(oc['oracle_reprojected_state'],dtype=np.float64)
        center_report={'tet_id':tid,'grade':kind,'material_index':tet['material_index'],
            'center':{'native_trial_q_from_capture':center['hill_q_lambda_a_b'][0],
                      'FP64_trial_q_at_exact_native_F':qcenter,'branch':branch_center,
                      'native_lambda_increment':float(native_cand[12]-vec(center['accepted_state'],13)[12]),
                      'FP64_lambda_increment':float(oc['lambda_increment_reprojected']),
                      'native_projected_state_vs_FP64':metric(native_cand,state_proj),
                      'native_stress_vs_FP64_candidate_state':metric(native_center,oc['oracle_candidate_P_Pa']),
                      'native_stress_vs_FP64_reprojected_state':metric(native_center,oc['oracle_reprojected_P_Pa']),
                      'native_algorithmic_tangent_vs_FP64_candidate':metric(tangent_native,t_candidate),
                      'native_algorithmic_tangent_vs_FP64_reprojected':metric(tangent_native,t_reproject),
                      'native_algorithmic_tangent_abs_max_Pa_per_s':maxabs(tangent_native),
                      'observed_native_minus_FP64_stress_max_abs_Pa':maxabs(native_center-np.asarray(oc['oracle_candidate_P_Pa'],dtype=np.float64)),
                      'native_local_residual_at_captured_candidate_scaled_inf':float(oc['source_vm_compensated_scaled_residual_inf'])}}
        # Element internal force from same captured material stress; compare native and oracle at exact state.
        inv=f32(tet['inverse_rest_rows_per_m']); vol=float(f32(tet['rest_volume_m3'])); q=rotation(tet['material_frame_quaternion_xyzw'])
        o_p=np.asarray(oc['oracle_candidate_P_Pa'],dtype=np.float64).reshape(3,3)
        o_pw=o_p@q.T
        native_force=np.asarray(f32(center['element_force_node_world_N']),dtype=np.float64).reshape(4,3)
        oracle_force=element_force(o_pw,inv,vol)
        center_report['center']['native_internal_force_vs_FP64_Pa_stress_force']=metric(native_force,oracle_force)
        center_report['center']['native_internal_force_observed_abs_discrepancy_max_N']=maxabs(native_force-oracle_force)
        fd_rows=[]
        for pair,sides in sorted(obj['pair_sides'].items()):
            eps=float(f32(next(iter(sides.values()))['epsilon']))
            sp,sm=sides['plus'],sides['minus']
            opi=bykey[f'{k}_s{tet["sides"].index(sp)}']; omi=bykey[f'{k}_s{tet["sides"].index(sm)}']
            fplus=vec(sp['F_local'],9).astype(np.float64)+vec(sp.get('F_local_low',np.zeros(9)),9).astype(np.float64)
            fminus=vec(sm['F_local'],9).astype(np.float64)+vec(sm.get('F_local_low',np.zeros(9)),9).astype(np.float64)
            realized=(fplus-fminus)/(2*eps*dt); nominal=vec(center['F_direction_local_per_s'],9).astype(np.float64)
            pn=native_P(sp); mn=native_P(sm)
            po=np.asarray(opi['oracle_candidate_P_Pa'],dtype=np.float64); mo=np.asarray(omi['oracle_candidate_P_Pa'],dtype=np.float64)
            pcenter_n=native_center; pcenter_o=np.asarray(oc['oracle_candidate_P_Pa'],dtype=np.float64)
            native_fd=(pn-mn)/(2*eps*dt); fp64_fd=(po-mo)/(2*eps*dt)
            oh=bykey[f'{k}_h{pair}']
            t_eff=np.asarray(oh['oracle_tangent_at_captured_candidate_Pa_per_s'],dtype=np.float64)
            t_nom_cand=t_candidate
            qplus=float(opi['elastic_trial_q']); qminus=float(omi['elastic_trial_q'])
            branches=[branch(qminus),branch(qcenter),branch(qplus)]
            crossing=len(set(branches))>1 or 'transition_unresolved' in branches
            # Force response differences use fixed reference geometry/material orientation.
            native_force_p=world_force_from_native(sp,tet); native_force_m=world_force_from_native(sm,tet)
            oracle_force_p=element_force(np.asarray(opi['oracle_candidate_P_Pa']).reshape(3,3)@q.T,inv,vol)
            oracle_force_m=element_force(np.asarray(omi['oracle_candidate_P_Pa']).reshape(3,3)@q.T,inv,vol)
            forcefd_native=(native_force_p-native_force_m)/(2*eps*dt)
            forcefd_oracle=(oracle_force_p-oracle_force_m)/(2*eps*dt)
            fd_rows.append({'pair':pair,'epsilon':eps,'dt_s':dt,'branches_minus_center_plus':branches,
                'branch_crossing_or_unresolved':crossing,
                'FP64_trial_q_minus_center_plus':[qminus,qcenter,qplus],
                'FP64_lambda_increment_minus_center_plus':[float(omi['lambda_increment_reprojected']),float(oc['lambda_increment_reprojected']),float(opi['lambda_increment_reprojected'])],
                'native_lambda_increment_minus_center_plus':[float(vec(sm['projected_state'],13)[12]+vec(sm.get('projected_state_low',np.zeros(13)),13)[12]-vec(tet['center']['accepted_state'],13)[12]),float(native_cand[12]-vec(tet['center']['accepted_state'],13)[12]),float(vec(sp['projected_state'],13)[12]+vec(sp.get('projected_state_low',np.zeros(13)),13)[12]-vec(tet['center']['accepted_state'],13)[12])],
                'native_centered_stress_derivative_Pa_per_s':metric(native_fd,tangent_native),
                'FP64_exact_endpoint_stress_derivative_vs_FP64_nominal_direction_tangent':metric(fp64_fd,t_nom_cand),
                'FP64_exact_endpoint_stress_derivative_vs_FP64_realized_direction_tangent':metric(fp64_fd,t_eff),
                'native_minus_FP64_exact_endpoint_stress_difference_abs_max_Pa':maxabs((pn-mn)-(po-mo)),
                'native_vs_FP64_plus_stress_abs_max_Pa':maxabs(pn-po),
                'native_vs_FP64_minus_stress_abs_max_Pa':maxabs(mn-mo),
                'exact_realized_F_direction_vs_nominal':metric(realized,nominal),
                'realized_F_direction_max_abs_delta_per_s':maxabs(realized-nominal),
                'nominal_direction_inf_per_s':maxabs(nominal),
                'realized_direction_inf_per_s':maxabs(realized),
                'nominal_F_endpoint_delta_inf':2*eps*dt*maxabs(nominal),
                'actual_native_F_endpoint_delta_inf':maxabs(fp-fm),
                'native_F_endpoint_delta_inf_error':maxabs((fp-fm)-2*eps*dt*nominal),
                'native_centered_force_derivative_N_per_s':metric(forcefd_native,element_force(tangent_native.reshape(3,3)@q.T,inv,vol)),
                'native_minus_FP64_exact_endpoint_force_difference_abs_max_N':maxabs((native_force_p-native_force_m)-(oracle_force_p-oracle_force_m)),
                'native_plus_force_vs_FP64_abs_max_N':maxabs(native_force_p-oracle_force_p),
                'native_minus_force_vs_FP64_abs_max_N':maxabs(native_force_m-oracle_force_m),
                'native_plus_minus_stress_abs_delta_Pa':maxabs(pn-mn),
                'fp64_plus_minus_stress_abs_delta_Pa':maxabs(po-mo),
                'smooth_tangent_verdict':('not_admissible_branch_crossing_or_near_yield' if crossing else
                    ('center_tangent_compared; input pair response separately reported'))})
        center_report['pairs']=fd_rows
        # Endogenous reference stress-force error envelope across center and all perturbations.
        per_tet.append(center_report)
    receipt=json.loads((cap.parent/'receipt.json').read_text())
    prediction=json.loads((cap.parent/'prediction.json').read_text())
    source_build=json.loads((cap.parent/'inputs/external/source-build.json').read_text())
    build_artifacts=source_build.get('artifacts_sha256',{})
    all_results[label]={'capture':str(cap),'receipt':str(cap.parent/'receipt.json'),
       'prediction':str(cap.parent/'prediction.json'),'source_build':str(cap.parent/'inputs/external/source-build.json'),
       'capture_sha256':sha(cap),'receipt_sha256':sha(cap.parent/'receipt.json'),
       'prediction_sha256':sha(cap.parent/'prediction.json'),'source_build_sha256':sha(cap.parent/'inputs/external/source-build.json'),
       'receipt_summary':{k:receipt.get(k) for k in ('exit_code','timed_out','source_inputs_unchanged','wall_seconds')},
       'prediction_summary':{k:prediction.get(k) for k in ('source_snapshot','source_binding_limitation','executed_build_verified','build_provenance','timeout_seconds')},
       'native_probe_binary_sha256':next((v for k,v in build_artifacts.items() if k.endswith('/numi-matter-cardboard-probe')),None),
       'native_metallib_sha256':next((v for k,v in build_artifacts.items() if k.endswith('/NumiMatter.metallib')),None),
       'native_source_material_sha256':{
          'liner':sha(cap.parent/'inputs/source/matter/materials/hajali2009_liner_hill_ideal.nmatter'),
          'medium':sha(cap.parent/'inputs/source/matter/materials/hajali2009_medium_hill_ideal.nmatter')},
       'native_source_materials_match_FP64_replay':{
          'liner':sha(cap.parent/'inputs/source/matter/materials/hajali2009_liner_hill_ideal.nmatter')==sha(MATS[0]),
          'medium':sha(cap.parent/'inputs/source/matter/materials/hajali2009_medium_hill_ideal.nmatter')==sha(MATS[1])},
       'source_build_materials_match':{
          'liner':source_build.get('source_sha256',{}).get('matter/materials/hajali2009_liner_hill_ideal.nmatter')==sha(MATS[0]),
          'medium':source_build.get('source_sha256',{}).get('matter/materials/hajali2009_medium_hill_ideal.nmatter')==sha(MATS[1])},
       'header':header,'integrity':integrity,'selected_tets':per_tet,
       'replay_returncode':cp.returncode,'replay_stdout_sha256':sha(run/'replay-stdout.jsonl'),
       'replay_input_sha256':sha(run/'replay-input.txt')}

# Deterministic implementation-discrepancy envelopes across all captured target samples.
for label,run in all_results.items():
    stress=[]; force=[]; tangent=[]; dirs=[]; deltaP=[]; deltaF=[]
    for tet in run['selected_tets']:
        c=tet['center']
        stress.append(c['native_stress_vs_FP64_candidate_state']['absolute_error'])
        force.append(c['native_internal_force_observed_abs_discrepancy_max_N'])
        tangent.append(c['native_algorithmic_tangent_vs_FP64_candidate']['relative_error'])
        for pair in tet['pairs']:
            stress.extend((pair['native_vs_FP64_plus_stress_abs_max_Pa'],pair['native_vs_FP64_minus_stress_abs_max_Pa']))
            force.extend((pair['native_plus_force_vs_FP64_abs_max_N'],pair['native_minus_force_vs_FP64_abs_max_N']))
            deltaP.append(pair['native_minus_FP64_exact_endpoint_stress_difference_abs_max_Pa'])
            deltaF.append(pair['native_minus_FP64_exact_endpoint_force_difference_abs_max_N'])
            dirs.append(pair['exact_realized_F_direction_vs_nominal']['relative_error'])
    run['deterministic_discrepancy_envelopes']={'max_native_vs_FP64_candidate_stress_abs_Pa':max(stress),
        'max_native_vs_FP64_element_force_abs_N':max(force),
        'max_native_vs_FP64_candidate_tangent_relative':max(tangent),
        'max_native_vs_FP64_centered_pair_stress_response_abs_Pa':max(deltaP),
        'max_native_vs_FP64_centered_pair_force_response_abs_N':max(deltaF),
        'max_realized_direction_relative_deviation':max(dirs),
        'interpretation':'captured deterministic evaluator/rounding discrepancies, not measured stochastic noise floor'}
manifest={'schema':'numi.cardboard.exact-native-pair-replay-audit.v1','evidence_class':'CPU independent FP64 constitutive replay of captured native states; not a new native solver run or physical validation',
 'replay_source_snapshot':'source-refined-graph-angular-residual-instrument-071 plus isolated two-word-direction reader extension',
 'exact_direction_replay_source_sha256':sha(BASE/'analysis/smooth-target-pair-audit-225/cardboard_solver_pair_replay_realized.cpp'),'captures':all_results,
 'replay_executable':str(REPLAY),'replay_executable_sha256':sha(REPLAY),
 'replay_source_sha256':sha(SOURCE_ROOT/'matter/tools/cardboard_solver_pair_replay.cpp'),
 'oracle_header_sha256':sha(SOURCE_ROOT/'matter/tools/cardboard_solver_reference.hpp'),
 'projection_scaling_header_sha256':sha(SOURCE_ROOT/'matter/tools/cardboard_paper_projection_scaling.hpp'),
 'liner_material_sha256':sha(MATS[0]),'medium_material_sha256':sha(MATS[1]),
 'capture_source_snapshot':'source-graph-tail-guard-angular-instrument-069',
 'capture_run_invariant_source_files_sha256':{
    'matter/tools/cardboard_solver_pair_replay.cpp':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/tools/cardboard_solver_pair_replay.cpp'),
    'matter/tools/cardboard_solver_reference.hpp':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/tools/cardboard_solver_reference.hpp'),
    'matter/tools/cardboard_paper_projection_scaling.hpp':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/tools/cardboard_paper_projection_scaling.hpp'),
    'matter/include/numi/matter/paper_compensated_bytecode.h':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/include/numi/matter/paper_compensated_bytecode.h'),
    'matter/materials/hajali2009_liner_hill_ideal.nmatter':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/materials/hajali2009_liner_hill_ideal.nmatter'),
    'matter/materials/hajali2009_medium_hill_ideal.nmatter':sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/materials/hajali2009_medium_hill_ideal.nmatter')},
 'capture_source_and_replay_source_relevant_files_match':all(sha(BASE/'snapshots/source-graph-tail-guard-angular-instrument-069'/rel)==sha(SOURCE_ROOT/rel) for rel in ['matter/tools/cardboard_solver_pair_replay.cpp','matter/tools/cardboard_solver_reference.hpp','matter/tools/cardboard_paper_projection_scaling.hpp','matter/include/numi/matter/paper_compensated_bytecode.h','matter/materials/hajali2009_liner_hill_ideal.nmatter','matter/materials/hajali2009_medium_hill_ideal.nmatter']),
 'numeric_contract':'All native float-valued capture words, including F high/low, projected-state high/low, accepted state, direction, and parameters, are rounded to float32 and widened exactly. The independent replay sums each high+low pair in FP64, including a two-word decomposition of the exact realized endpoint direction; exact-endpoint stress differences remain unrounded FP64.',
 'limits':['Only env0 tets 1388, 633, 441, 518, 702, 776 are replayed.','Capture JSON exposes directional tangent only, not the full 9x9 native tangent; exact realized-pair derivative is compared with independent tangent, not native tangent transformed to a new direction.','Absolute deterministic discrepancies are evaluator error envelopes, not stochastic/repeatability noise floors.','Near-yield or branch-crossing endpoint pairs are not admissible smooth-derivative tests.','No physical grade calibration, box qualification, or convergence claim.']}
(OUT/'exact-pair-replay-audit.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(json.dumps({'out':str(OUT/'exact-pair-replay-audit.json'),'captures':list(all_results),'selected':{k:len(v['selected_tets']) for k,v in all_results.items()}},indent=2))
