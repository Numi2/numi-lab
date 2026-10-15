#!/usr/bin/env python3
"""Exact-word FP64 replay of every paper tetrahedron in native capture 216.

CPU diagnostic only. It preserves the failed Newton capture and never runs a
native step. Inputs/outputs are hash-bound in the generated manifest.
"""
from __future__ import annotations
import hashlib, json, math, subprocess
from collections import Counter
from pathlib import Path
import numpy as np

BASE=Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/graph-schwarz-first-control-final-newton-capture-216/operator-capture'
RUN=CAP.parent
OUT=BASE/'analysis/all-paper-tet-replay-231/output'
REPLAY=BASE/'analysis/smooth-target-pair-audit-225/build/cardboard_solver_pair_replay_realized'
LINER=RUN/'inputs/source/matter/materials/hajali2009_liner_hill_ideal.nmatter'
MEDIUM=RUN/'inputs/source/matter/materials/hajali2009_medium_hill_ideal.nmatter'
if OUT.exists(): raise SystemExit(f'preserving existing output: {OUT}')
OUT.mkdir(parents=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def f32(a):
    x=np.asarray(a,dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('nonfinite FP32 input')
    return x
def f32flat(a,n):
    x=f32(a).reshape(-1)
    if x.size != n: raise ValueError(f'expected {n} words, got {x.size}')
    return x
def widen_pair(hi,lo):
    return f32flat(hi,len(np.asarray(hi).reshape(-1))).astype(np.float64)+f32flat(lo,len(np.asarray(lo).reshape(-1))).astype(np.float64)
def fmt(a): return ' '.join(format(float(v),'.17g') for v in np.asarray(a).reshape(-1))
def maxabs(x):
    a=np.asarray(x,dtype=np.float64)
    return float(np.max(np.abs(a))) if a.size else 0.0
def metric(actual,ref):
    a=np.asarray(actual,dtype=np.float64).reshape(-1); b=np.asarray(ref,dtype=np.float64).reshape(-1)
    d=maxabs(a-b); scale=maxabs(b)
    return {'absolute_error':d,'reference_max_abs':scale,'relative_error':d/scale if scale>0 else None}
def qbranch(q):
    return 'transition_unresolved' if abs(q-1.0)<=1e-6 else ('plastic' if q>1 else 'elastic')
def rot(q):
    x,y,z,w=map(float,f32flat(q,4)); n=math.sqrt(x*x+y*y+z*z+w*w)
    if n==0: raise ValueError('zero material quaternion')
    x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)
def element_force(p,inv,vol):
    h=-float(vol)*np.asarray(p,dtype=np.float64).reshape(3,3)@np.asarray(inv,dtype=np.float64).reshape(3,3).T
    return np.vstack((-h.sum(axis=1),h.T))

records=[]
with open(CAP) as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))
if records[0].get('record')!='header' or records[-1].get('record')!='integrity': raise ValueError('incomplete capture framing')
header, integrity=records[0],records[-1]
if header.get('schema')!='numi.cardboard.fem-newton-capture.v1' or (header['root'],header['newton_iteration'],header['environment_count'])!=(0,23,2): raise ValueError('unexpected capture selection')
if not integrity.get('production_inputs_bitwise_unchanged'): raise ValueError('native capture integrity failed')
receipt=json.loads((RUN/'receipt.json').read_text())
prediction=json.loads((RUN/'prediction.json').read_text())
if receipt.get('source_inputs_unchanged') is not True: raise ValueError('source receipt says inputs changed')
if receipt.get('exit_code')==0 or receipt.get('timed_out'): raise ValueError('capture is expected to be a complete failed transaction, not accepted success/timeout')
if not LINER.exists() or not MEDIUM.exists(): raise ValueError('source-bound copied paper programs missing')
materials={int(m['index']):m for m in header['materials']}
if materials[0]['name']!='hajali2009_liner_hill_ideal' or materials[1]['name']!='hajali2009_medium_hill_ideal': raise ValueError('paper material identities changed')
params_by_env=[]
for env,raw in enumerate(header['parameters_by_environment']):
    vals=f32(raw)
    if vals.size < 40: raise ValueError('captured paper parameter blocks missing')
    params_by_env.append(vals)
if not np.array_equal(params_by_env[0],params_by_env[1]):
    # Row-local parameters make this executable, but source data must still bind
    # both environment blocks. Preserve the difference in the result.
    params_equal=False
else: params_equal=True
paper=[]; nodes=[]; counts=Counter()
for r in records[1:-1]:
    if r.get('record')=='node': nodes.append(r)
    elif r.get('record')=='tetrahedron':
        counts[(int(r['environment']),int(r['material_index']))]+=1
        if int(r['material_index']) in (0,1): paper.append(r)
if len(nodes)!=header['environment_count']*header['node_count_per_environment']: raise ValueError('incomplete nodes')
if len(paper)!=2*(1152+576): raise ValueError(f'paper tet count mismatch: {len(paper)}')
for env in range(2):
    if counts[(env,0)]!=1152 or counts[(env,1)]!=576 or counts[(env,2)]!=192: raise ValueError(f'per-env material population mismatch: {env}: {counts}')
if len({(int(t['environment']),int(t['tet_id'])) for t in paper})!=len(paper): raise ValueError('duplicate paper tet ids')

def row_for(key,kind,tet,state,direction,exact_direction=False):
    env=int(tet['environment']); block=params_by_env[env]; off=int(materials[0 if kind=='liner' else 1]['parameter_offset'])
    pars=block[off:off+20]
    fhi=f32flat(state['F_local'],9); flo=f32flat(state.get('F_local_low',[0]*9),9)
    dhi=f32flat(direction,9)
    # Keep residual as a second word for the derived exact direction.
    dlo=f32(np.asarray(direction,dtype=np.float64)-dhi.astype(np.float64)).reshape(-1) if exact_direction else np.zeros(9,dtype=np.float32)
    acc=f32flat(tet['center']['accepted_state'],13)
    chi=f32flat(state['projected_state'],13); clo=f32flat(state.get('projected_state_low',[0]*13),13)
    fields=[key,kind,fmt(fhi),fmt(flo),fmt(dhi),fmt(dlo),fmt(acc),fmt(chi),fmt(clo),fmt(pars)]
    return ' '.join(fields)

rows=[]; source_by_key={}; per_tet={}
for tet in sorted(paper,key=lambda t:(int(t['environment']),int(t['tet_id']))):
    env=int(tet['environment']); tid=int(tet['tet_id']); kind='liner' if int(tet['material_index'])==0 else 'medium'
    center=tet['center']; keybase=f'e{env}_t{tid}'
    nominal=f32(center['F_direction_local_per_s']).reshape(9)
    def add(suffix,state,direction=nominal,exact=False):
        key=keybase+suffix
        rows.append(row_for(key,kind,tet,state,direction,exact))
        source_by_key[key]=(tet,state,kind)
    add('_c',center)
    pairs={}
    for si,s in enumerate(tet['sides']):
        add(f'_s{si}',s)
        pair=int(si//2); side='plus' if si%2==0 else 'minus'
        pairs.setdefault(pair,{})[side]=s
    fcenter=widen_pair(center['F_local'],center.get('F_local_low',[0]*9))
    for pair,sides in pairs.items():
        if set(sides)!={'plus','minus'}: raise ValueError(f'{keybase} pair {pair} incomplete')
        eps=float(f32(sides['plus']['epsilon'])); dt=float(f32(sides['plus']['dt_seconds']))
        fp=widen_pair(sides['plus']['F_local'],sides['plus'].get('F_local_low',[0]*9))
        fm=widen_pair(sides['minus']['F_local'],sides['minus'].get('F_local_low',[0]*9))
        heff=(fp-fm)/(2*eps*dt)
        add(f'_h{pair}',center,heff,True)
    per_tet[(env,tid)]={'tet':tet,'kind':kind,'pairs':pairs}

input_path=OUT/'replay-input.jsonl'
with open(input_path,'w') as f:
    for row in rows: f.write(row+'\n')
rawout=OUT/'replay-output.jsonl'; errpath=OUT/'replay-stderr.txt'
with open(input_path,'r') as inp, open(rawout,'w') as out, open(errpath,'w') as err:
    cp=subprocess.run([str(REPLAY),str(LINER),str(MEDIUM),'--direction-high-low'],stdin=inp,stdout=out,stderr=err,text=True,check=False)
if cp.returncode not in (0,2): raise RuntimeError(f'FP64 replay exited {cp.returncode}: {errpath.read_text()[:2000]}')

result={}; expected_keys=iter(source_by_key.keys()); obs={}; output_count=0
with open(rawout) as f:
    for line in f:
        if not line.strip(): continue
        o=json.loads(line); output_count+=1; key=o.get('key')
        if key not in source_by_key: raise ValueError(f'unexpected replay key {key}')
        obs[key]=o
if output_count!=len(rows): raise ValueError(f'incomplete replay rows {output_count} != {len(rows)}')

all_center_stress=[]; all_center_tangent=[]; all_candidate_state=[]; all_force=[]; all_fd=[]
branch_counts=Counter(); smooth_stats={'pair0':Counter(),'pair1':Counter()}; 
audit_path=OUT/'paper-tetrahedron-audit.jsonl'
element_tangent_oracle_by_env={0:{},1:{}}
for ident,item in sorted(per_tet.items()):
    env,tid=ident; tet=item['tet']; kind=item['kind']; c=tet['center']; base=f'e{env}_t{tid}'
    oc=obs[base+'_c']; nativeP=f32flat(c['P_local_Pa'],9).astype(np.float64)
    nativeT=f32flat(c['tangent_local_Pa_per_s'],9).astype(np.float64)
    nativeState=widen_pair(c['projected_state'],c.get('projected_state_low',[0]*13))
    if not oc.get('valid'): raise ValueError(f'center replay invalid {base}: {oc}')
    oracleP=np.asarray(oc['oracle_candidate_P_Pa'],dtype=np.float64)
    oracleT=np.asarray(oc['oracle_tangent_at_captured_candidate_Pa_per_s'],dtype=np.float64)
    oracleState=np.asarray(oc['candidate_state_FP64'],dtype=np.float64)
    qnative=float(f32(c['hill_q_lambda_a_b'][0])); qfp=float(oc['elastic_trial_q'])
    branch=qbranch(qfp); branch_counts[(env,kind,branch)]+=1
    inv=f32(tet['inverse_rest_rows_per_m']).reshape(3,3).astype(np.float64)
    volume=float(f32(tet['rest_volume_m3'])); Q=rot(tet['material_frame_quaternion_xyzw'])
    oracleForce=element_force(oracleT.reshape(3,3)@Q.T,inv,volume)
    nativeForce=f32(tet['center']['operator_force_node_world_N_per_s']).reshape(4,3).astype(np.float64)
    element_tangent_oracle_by_env[env][tid]=oracleForce
    cr={'environment':env,'tet_id':tid,'material_index':int(tet['material_index']),'grade':kind,
        'center_trial_q_native':qnative,'center_trial_q_fp64':qfp,'center_branch':branch,
        'projection_scaled_residual_inf':float(oc['oracle_residual_scaled_inf']),
        'source_vm_scaled_residual_inf':float(oc['source_vm_compensated_scaled_residual_inf']),
        'native_stress_vs_fp64_candidate':metric(nativeP,oracleP),
        'native_tangent_vs_fp64_candidate':metric(nativeT,oracleT),
        'native_tangent_absolute_error_max_Pa_per_s':maxabs(nativeT-oracleT),
        'native_candidate_state_vs_fp64_reprojected':metric(nativeState,oc['oracle_reprojected_state']),
        'candidate_state_component_abs_error_max':maxabs(nativeState-np.asarray(oc['oracle_reprojected_state'])),
        'element_tangent_nodal_force_native_vs_fp64':metric(nativeForce,oracleForce),
        'element_tangent_force_abs_error_max_N_per_s':maxabs(nativeForce-oracleForce),
        'pairs':[]}
    all_center_stress.append(cr['native_stress_vs_fp64_candidate']['relative_error'])
    all_center_tangent.append(cr['native_tangent_vs_fp64_candidate']['relative_error'])
    all_candidate_state.append(cr['candidate_state_component_abs_error_max'])
    all_force.append(cr['element_tangent_nodal_force_native_vs_fp64']['relative_error'])
    for pair,sides in sorted(item['pairs'].items()):
        plus=sides['plus']; minus=sides['minus']; eps=float(f32(plus['epsilon'])); dt=float(f32(plus['dt_seconds']))
        ip=tet['sides'].index(plus); im=tet['sides'].index(minus)
        op=obs[f'{base}_s{ip}']; om=obs[f'{base}_s{im}']; oh=obs[f'{base}_h{pair}']
        qp=float(op['elastic_trial_q']); qm=float(om['elastic_trial_q'])
        branches=[qbranch(qm),branch,qbranch(qp)]
        crossing=(len(set(branches))>1 or 'transition_unresolved' in branches)
        fp=widen_pair(plus['F_local'],plus.get('F_local_low',[0]*9)); fm=widen_pair(minus['F_local'],minus.get('F_local_low',[0]*9))
        realized=(fp-fm)/(2*eps*dt)
        nominal=f32(c['F_direction_local_per_s']).reshape(9).astype(np.float64)
        opP=np.asarray(op['oracle_candidate_P_Pa']); omP=np.asarray(om['oracle_candidate_P_Pa'])
        fd=(opP-omP)/(2*eps*dt)
        h=np.asarray(oh['oracle_tangent_at_captured_candidate_Pa_per_s'])
        native_plus=f32flat(plus['P_local_Pa'],9).astype(np.float64); native_minus=f32flat(minus['P_local_Pa'],9).astype(np.float64)
        native_fd=(native_plus-native_minus)/(2*eps*dt)
        hmetric=metric(fd,h)
        resolved=(not crossing and maxabs(h)>0 and maxabs(fd)>0)
        pairrow={'pair':pair,'epsilon':eps,'dt_s':dt,'branches_minus_center_plus':branches,
           'branch_crossing_or_unresolved':crossing,'resolved_smooth_branch':resolved,
           'trial_q_minus_center_plus':[qm,qfp,qp],
           'fp64_endpoint_fd_vs_realized_direction_analytic':hmetric,
           'fp64_endpoint_fd_abs_difference_max_Pa_per_s':maxabs(fd-h),
           'native_endpoint_fd_vs_native_center_tangent':metric(native_fd,nativeT),
           'realized_direction_vs_nominal':metric(realized,nominal),
           'realized_direction_relative_error':metric(realized,nominal)['relative_error'],
           'actual_endpoint_delta_inf':maxabs(fp-fm),
           'nominal_endpoint_delta_inf':2*eps*dt*maxabs(nominal),
           'native_plus_stress_vs_fp64_abs_Pa':maxabs(f32flat(plus['P_local_Pa'],9)-np.asarray(op['oracle_candidate_P_Pa'])),
           'native_minus_stress_vs_fp64_abs_Pa':maxabs(f32flat(minus['P_local_Pa'],9)-np.asarray(om['oracle_candidate_P_Pa'])),
           'own_fp64_fd_gate_1e-6':'PASS' if resolved and hmetric['relative_error'] is not None and hmetric['relative_error']<=1e-6 else ('FAIL' if resolved and hmetric['relative_error'] is not None else 'INCONCLUSIVE_BRANCH_OR_ZERO_SCALE')}
        cr['pairs'].append(pairrow)
        all_fd.append(pairrow['fp64_endpoint_fd_vs_realized_direction_analytic']['relative_error'])
        smooth_stats[f'pair{pair}'][pairrow['own_fp64_fd_gate_1e-6']]+=1
    result[ident]=cr

with open(audit_path,'w') as f:
    for key,cr in sorted(result.items()): f.write(json.dumps(cr,allow_nan=False)+'\n')

# Glue tangent is reconstructed independently from the exact native FP32 F/H
# and captured finite Neo-Hookean parameters; paper tangents come from the full
# FP64 history-bearing replay above. Then assemble the selected native A action
# over the captured free displacement rows, no solver state is modified.
from importlib.util import spec_from_file_location, module_from_spec
oldtool=BASE/'snapshots/source-refined-graph-angular-residual-instrument-071/matter/tools/cardboard_solver_capture_analysis.py'
spec=spec_from_file_location('capture_analysis',oldtool); ca=module_from_spec(spec); spec.loader.exec_module(ca)
node_records=[r for r in records if r.get('record')=='node']
tet_records=[r for r in records if r.get('record')=='tetrahedron']
assembly=[]
for env in range(2):
    envnodes=sorted((n for n in node_records if int(n['environment'])==env),key=lambda n:int(n['node_id']))
    if [int(n['node_id']) for n in envnodes]!=list(range(header['node_count_per_environment'])): raise ValueError('node identity sequence mismatch')
    n=len(envnodes); mass=np.asarray([float(f32(x['accepted']['mass_kg'])) for x in envnodes])
    free=np.asarray([int(x['center']['constraint_tag']) not in (1,3) and mass[i]>0 for i,x in enumerate(envnodes)])
    dv=np.asarray([f32flat(x['correction_delta_v_m_s'],3) for x in envnodes],dtype=np.float64)
    mass_action=mass[:,None]*dv
    native_tangent_force=np.zeros((n,3),dtype=np.float64); oracle_tangent_force=np.zeros((n,3),dtype=np.float64)
    missing=[]
    for t in (t for t in tet_records if int(t['environment'])==env):
        tid=int(t['tet_id']); ids=np.asarray(t['nodes'],dtype=int)
        np.add.at(native_tangent_force,ids,f32(t['center']['operator_force_node_world_N_per_s']).astype(np.float64))
        if int(t['material_index']) in (0,1):
            local=element_tangent_oracle_by_env[env].get(tid)
            if local is None: missing.append(tid); continue
            np.add.at(oracle_tangent_force,ids,local)
        elif int(t['material_index'])==2:
            center=t['center']; f=widen_pair(center['F_local'],center.get('F_local_low',[0]*9)).reshape(3,3)
            h=f32(center['F_direction_local_per_s']).reshape(3,3).astype(np.float64)
            params=params_by_env[env]; off=int(materials[2]['parameter_offset'])
            mu=float(params[off+1]); lam=float(params[off+2]); det=float(np.linalg.det(f))
            if det<=0: raise ValueError(f'nonpositive glue determinant env{env} tet{tid}')
            fi=np.linalg.inv(f); a=lam*math.log(det)-mu
            p=mu*h+lam*np.trace(fi@h)*fi.T-a*fi.T@h.T@fi.T
            local=element_force(p@rot(t['material_frame_quaternion_xyzw']).T,
                                f32(t['inverse_rest_rows_per_m']).reshape(3,3),float(f32(t['rest_volume_m3'])))
            np.add.at(oracle_tangent_force,ids,local)
        else: raise ValueError('unknown material index in capture')
    if missing: raise ValueError(f'missing independent paper tangent tet records: {missing[:10]}')
    dt=float(f32(header['dt_seconds']))
    action=np.asarray([f32flat(x['production_A_delta_v_impulse_Ns'],4)[:3] for x in envnodes])
    expected_native=mass_action-dt*dt*native_tangent_force
    expected_oracle=mass_action-dt*dt*oracle_tangent_force
    action_report={'environment':env,'free_node_count':int(free.sum()),'captured_tet_count':sum(int(t['environment'])==env for t in tet_records),
       'paper_tet_count':sum(int(t['environment'])==env and int(t['material_index']) in (0,1) for t in tet_records),
       'glue_tet_count':sum(int(t['environment'])==env and int(t['material_index'])==2 for t in tet_records),
       'native_production_A_action_vs_mass_minus_native_element_tangent':metric(action[free],expected_native[free]),
       'native_production_A_action_vs_mass_minus_independent_FP64_element_tangent':metric(action[free],expected_oracle[free]),
       'max_abs_native_operator_action_Ns':maxabs(action[free]),
       'max_abs_independent_element_action_Ns':maxabs(expected_oracle[free]),
       'relative_action_difference_native_tangent_vs_oracle':metric(expected_native[free],expected_oracle[free]),
       'mass_action_l2_Ns':float(np.linalg.norm(mass_action[free])),
       'dt_squared_stiffness_action_l2_Ns':float(np.linalg.norm((dt*dt*native_tangent_force)[free])),
       'native_correction_delta_v_max_m_s':maxabs(dv[free]),
       'displacement_dof_reconstruction':'mass*delta_v - dt^2*assembled element (dP/dF):(dF/dv) direction; first 3 node action components only',
       'limitations':'captured rejected Newton transaction; independent paper 13-state FP64 tangent plus analytic finite-glue tangent; no global physical reaction or accepted solver claim'}
    assembly.append(action_report)

# Compact, source-bound summary. A per-tet JSONL preserves every raw outcome.
def summary_stats(values):
    x=np.asarray([v for v in values if v is not None and math.isfinite(v)],dtype=np.float64)
    return {'n':int(x.size),'max':float(np.max(x)) if x.size else None,'p99':float(np.quantile(x,.99)) if x.size else None,
            'median':float(np.median(x)) if x.size else None}
summary={'schema':'numi.cardboard.all_paper_tet_fp64_audit.v1','capture':str(CAP),'capture_sha256':sha(CAP),
 'prediction_sha256':sha(RUN/'prediction.json'),'receipt_sha256':sha(RUN/'receipt.json'),'native_receipt':{k:receipt.get(k) for k in ('exit_code','timed_out','source_inputs_unchanged','wall_seconds')},
 'selection':{'root':header['root'],'newton_iteration':header['newton_iteration'],'environment_count':header['environment_count'],'node_count_per_environment':header['node_count_per_environment'],'tet_count_per_environment':header['tetrahedron_count_per_environment'],'dt_seconds':float(f32(header['dt_seconds'])),
              'paper_tets':len(paper),'paper_tets_by_environment_material':{f'{e}:{m}':counts[(e,m)] for e in range(2) for m in (0,1)},'starch_tets_per_environment':192},
 'source':{'source_snapshot':prediction.get('source_snapshot'),'source_tree_hash':prediction.get('source_tree_hash'),'liner_nmatter_sha256':sha(LINER),'medium_nmatter_sha256':sha(MEDIUM),'replay_cpp_sha256':sha(OUT/'unused') if False else sha(BASE/'analysis/smooth-target-pair-audit-228/cardboard_solver_pair_replay_realized.cpp'),'replay_binary_sha256':sha(REPLAY),'parameter_blocks_equal_across_environments':params_equal,'inputs_bitwise_unchanged':integrity.get('production_inputs_bitwise_unchanged')},
 'counts':{'replayed_rows_expected':len(rows),'replayed_rows_observed':output_count,'paper_tetrahedron_audit_rows':len(result),'branch_counts':{f'env{e}:{k}:{b}':v for (e,k,b),v in branch_counts.items()},
           'own_fp64_fd_gate_by_pair':{k:dict(v) for k,v in smooth_stats.items()}},
 'all_tet_distributions':{'native_vs_FP64_center_stress_relative':summary_stats([x['native_stress_vs_fp64_candidate']['relative_error'] for x in result.values()]),
  'native_vs_FP64_center_tangent_relative':summary_stats([x['native_tangent_vs_fp64_candidate']['relative_error'] for x in result.values()]),
  'native_vs_FP64_center_tangent_absolute_Pa_per_s':summary_stats([x['native_tangent_absolute_error_max_Pa_per_s'] for x in result.values()]),
  'native_vs_FP64_projected_state_abs_component':summary_stats([x['candidate_state_component_abs_error_max'] for x in result.values()]),
  'native_vs_FP64_element_tangent_force_relative':summary_stats([x['element_tangent_nodal_force_native_vs_fp64']['relative_error'] for x in result.values()]),
  'native_vs_FP64_element_tangent_force_abs_N_per_s':summary_stats([x['element_tangent_force_abs_error_max_N_per_s'] for x in result.values()])},
 'assembled_free_A_action':assembly,
 'gates':{'native_directional_tangent_vs_independent_FP64_relative_le_1e-3':'counted in rows; no gate accepted when reference scale is zero','own_FP64_endpoint_FD_vs_exact_realized_direction_tangent_relative_le_1e-6':'only eligible same-branch, non-near-transition rows are counted; observed endpoint pairs use captured high+low F words','assembled_free_displacement_A_action_relative_le_1e-2':'reported per environment; capture uses rejected transaction and is not physical/solver qualification'},
 'limits':['This native transaction was rejected with zero accepted steps; capture is diagnostic only.',
  'Native capture carries a 9-component directional tangent, not a complete exported 9x9 per-element constitutive Jacobian; the full 9x9 matrix gate is unavailable.',
  'Independent FP64 own-FD sweeps at infinitesimal smooth steps are a separate operation from native endpoint pairs; native realized F perturbation can differ from requested direction.',
  'Element force and assembled A checks are not physical calibration, fold validation, or an accepted global response.'],
 'artifact_sha256':{'replay_input':sha(input_path),'replay_output':sha(rawout),'audit_jsonl':sha(audit_path)}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(OUT/'assembled-action.json').write_text(json.dumps(assembly,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(OUT),'rows':len(rows),'tet_rows':len(result),'stress':summary['all_tet_distributions']['native_vs_FP64_center_stress_relative'],
                  'tangent':summary['all_tet_distributions']['native_vs_FP64_center_tangent_relative'],
                  'assembly':assembly},indent=2))
