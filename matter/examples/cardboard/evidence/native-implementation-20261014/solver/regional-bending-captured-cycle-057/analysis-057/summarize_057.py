#!/usr/bin/env python3
import json, pathlib, collections, math, hashlib
import numpy as np
from analyze_capture_pair import array, metric, rotation, force

base = pathlib.Path('/Users/home/cardboard-implementation-20261014/solver/regional-bending-captured-cycle-057')
cap_path = base/'capture.jsonl'
out = base/'analysis-057/replay-results-candidate-tangent'
records = [json.loads(line) for line in cap_path.read_text().splitlines() if line.strip()]
header = records[0]
tets = [r for r in records if r.get('record')=='tetrahedron']
nodes = [r for r in records if r.get('record')=='node']
audits = [json.loads(x) for x in (out/'tetrahedron-audit.jsonl').read_text().splitlines()]
replay = [json.loads(x) for x in (out/'replay-output.jsonl').read_text().splitlines()]
audit_by = {(r['environment'],r['tet_id']):r for r in audits}
replay_by = {r['key']:r for r in replay}

def rel(a,b):
 a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel()
 scale=float(np.max(np.abs(b))) if b.size else 0.
 return None if scale==0 else float(np.max(np.abs(a-b))/scale)
def max_metric(rows, getter):
 vals=[]
 for r in rows:
  v=getter(r)
  if v is not None and math.isfinite(v): vals.append((v,r))
 vals.sort(key=lambda x:x[0])
 if not vals:return None
 v,r=vals[-1]
 return {'relative_error':v,'environment':r['environment'],'tet_id':r['tet_id']}
def stats(vals):
 vals=sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
 if not vals:return {'count':0}
 return {'count':len(vals),'p50':vals[int((len(vals)-1)*.50)],'p90':vals[int((len(vals)-1)*.90)],'p99':vals[int((len(vals)-1)*.99)],'max':vals[-1]}

def candidate_word(state):
 hi=array(state['projected_state'],13); lo=array(state.get('projected_state_low',[0]*13),13)
 return hi+lo

def f_words(state):
 hi=array(state['F_local'],9); lo=array(state.get('F_local_low',[0]*9),9)
 return hi+lo

by_mat=collections.defaultdict(list)
branch_triples=collections.Counter()
center_branches=collections.Counter()
low_counts=collections.Counter()
low_max={'F':0.0,'state':0.0}
residuals=collections.defaultdict(list)
state_err=collections.defaultdict(list)
stress_err_projected=collections.defaultdict(list)
stress_err_candidate=collections.defaultdict(list)
tangent_err_projected=collections.defaultdict(list)
tangent_err_candidate=collections.defaultdict(list)
near_counts=collections.Counter()
side_cross_counts=collections.Counter()
side_unresolved_counts=collections.Counter()
case_details=[]

for tet in tets:
 env,tid,mat=tet['environment'],tet['tet_id'],tet['material_index']
 audit=audit_by[(env,tid)]
 if mat not in [0,1]: continue
 kind='liner' if mat==0 else 'medium'
 c=tet['center']; key=f'e{env}_t{tid}_s0'; o=replay_by[key]
 f=f_words(c); h=array(c['F_direction_local_per_s'],9)
 old=array(c['accepted_state'],13); cand=candidate_word(c)
 native_p=array(c['P_local_Pa'],9); native_t=array(c['tangent_local_Pa_per_s'],9)
 exactp=np.asarray(o['oracle_P_Pa']); candp=np.asarray(o['oracle_candidate_P_Pa'])
 exactt=np.asarray(o['oracle_tangent_Pa']); candt=np.asarray(o['oracle_tangent_at_candidate_Pa'])
 qtrial=o['elastic_trial_q_dimensionless']
 near=abs(qtrial-1)<1e-6
 center_branches[(mat,'transition_unresolved' if near else ('plastic' if qtrial>1 else 'elastic'))]+=1
 if near: near_counts[mat]+=1
 residuals[mat].append(o['candidate_residual'])
 state_err[mat].append(rel(cand,o['oracle_state']))
 stress_err_projected[mat].append(rel(native_p,exactp))
 stress_err_candidate[mat].append(rel(native_p,candp))
 tangent_err_projected[mat].append(rel(native_t,exactt))
 tangent_err_candidate[mat].append(rel(native_t,candt))
 by_mat[mat].append({'environment':env,'tet_id':tid,'elastic_trial_q':qtrial,'near_transition':near,
                      'candidate_residual':o['candidate_residual'],
                      'candidate_state_error':rel(cand,o['oracle_state']),
                      'stress_vs_exact_projection':rel(native_p,exactp),
                      'stress_vs_captured_candidate':rel(native_p,candp),
                      'tangent_vs_exact_projection':rel(native_t,exactt),
                      'tangent_vs_captured_candidate':rel(native_t,candt)})
 # Count nonzero low words for F and projected state; each center/side row is independently captured.
 for si,s in enumerate([c]+tet['sides']):
  f_low=np.asarray(s.get('F_local_low',[]),dtype=np.float64).ravel()
  p_low=np.asarray(s.get('projected_state_low',[]),dtype=np.float64).ravel()
  if f_low.size and np.any(f_low!=0):low_counts[(mat,'F',si)]+=1
  if p_low.size and np.any(p_low!=0):low_counts[(mat,'state',si)]+=1
  low_max['F']=max(low_max['F'],float(np.max(np.abs(f_low))) if f_low.size else 0.0)
  low_max['state']=max(low_max['state'],float(np.max(np.abs(p_low))) if p_low.size else 0.0)
 # Check two centered perturbation pairs for oracle branch changes.
 branches=[]
 for si in range(5):
  oside=replay_by[f'e{env}_t{tid}_s{si}']
  q=oside['elastic_trial_q_dimensionless']
  branches.append('transition_unresolved' if abs(q-1)<1e-6 else ('plastic' if q>1 else 'elastic'))
 for epsidx,eps in enumerate(header['epsilon_values']):
  plus=branches[1+2*epsidx]; minus=branches[2+2*epsidx]
  crossing=plus!=minus
  unresolved='transition_unresolved' in (branches[0],plus,minus)
  branch_triples[(mat,float(eps),branches[0],plus,minus)]+=1
  if crossing:side_cross_counts[(mat,float(eps))]+=1
  if unresolved:side_unresolved_counts[(mat,float(eps))]+=1

# Rebuild FEM tangent action using the exact FP64 law evaluated at the captured high+low candidate state.
# The contact term is zero for this run; this only checks assembled element tangent + mass action.
assembly=[]
for env in sorted({n['environment'] for n in nodes}):
 env_nodes=sorted([n for n in nodes if n['environment']==env],key=lambda n:n['node_id'])
 count=len(env_nodes)
 mass=np.asarray([n['accepted']['mass_kg'] for n in env_nodes],dtype=np.float64)
 free=np.asarray([n['center']['constraint_tag'] not in (1,3) and mass[i]>0 for i,n in enumerate(env_nodes)])
 dv=np.asarray([n['correction_delta_v_m_s'] for n in env_nodes],dtype=np.float64)
 native_A=np.asarray([n['production_A_delta_v_impulse_Ns'][:3] for n in env_nodes],dtype=np.float64)
 tangent_force=np.zeros((count,3),dtype=np.float64)
 for tet in (t for t in tets if t['environment']==env):
  mat=tet['material_index']; tid=tet['tet_id']; c=tet['center']
  inv=array(tet['inverse_rest_rows_per_m'],9).reshape(3,3); volume=tet['rest_volume_m3']; q=rotation(tet['material_frame_quaternion_xyzw'])
  if mat in (0,1):
   k=f'e{env}_t{tid}_s0'; tlocal=np.asarray(replay_by[k]['oracle_tangent_at_candidate_Pa']).reshape(3,3)
  else:
   # Use the same exact captured-F/FP64 derivative for the published finite-glue Neo-Hookean control.
   F=f_words(c).reshape(3,3); H=array(c['F_direction_local_per_s'],9).reshape(3,3)
   params=header['parameters_by_environment'][env][40:43]
   mu=float(np.float32(params[1])); lam=float(np.float32(params[2]))
   det=float(np.linalg.det(F)); invF=np.linalg.inv(F); a=lam*math.log(det)-mu
   tlocal=(mu*H+lam*np.trace(invF@H)*invF.T-a*invF.T@H.T@invF.T)
  tworld=tlocal@q.T
  nodal=force(tworld,inv,volume) # N/s, four rows x xyz
  np.add.at(tangent_force,np.asarray(tet['nodes'],dtype=int),nodal)
 predicted=mass[:,None]*dv-((header['dt_seconds'])**2)*tangent_force
 predicted[~free]=0
 native=native_A.copy(); native[~free]=0
 assembly.append({'environment':env,'free_node_count':int(free.sum()),
                  'native_production_A_vs_candidate_state_oracle':metric(native[free],predicted[free]),
                  'native_action_l2_Ns':float(np.linalg.norm(native[free])),
                  'candidate_state_oracle_action_l2_Ns':float(np.linalg.norm(predicted[free]))})

summary={
 'schema':'numi.cardboard.capture057.fp64-layer-audit.v1',
 'scope':{'capture_root':1,'newton_iteration':0,'center_residual_source':header['center_residual_source'],
          'environment_count':header['environment_count'],'tetrahedra':len(tets),'nodes':len(nodes),
          'paper_tetrahedra':sum(r['material_index'] in (0,1) for r in tets),
          'capture_complete':True,'terminal_newton_failure_explained':False,'physical_validation':False},
 'input_word_handling':{'F_contract':header['F_local_word_encoding'],
                        'candidate_state_high_low_widened_separately_before_f64_sum':True,
                        'nonzero_low_word_max_abs':low_max,
                        'paper_record_low_word_counts':{f'material{m}_{kind}_sample{i}':n for (m,kind,i),n in sorted(low_counts.items())}},
 'per_material':{},
 'branching':{'center_classification_threshold_abs_trial_q_minus_1_lt_1e-6':dict((f'material{m}_{b}',n) for (m,b),n in center_branches.items()),
              'near_transition_center_count':dict((str(k),v) for k,v in near_counts.items()),
              'plus_minus_actual_branch_crossings':{f'material{m}_epsilon_{eps}':n for (m,eps),n in sorted(side_cross_counts.items())},
              'any_center_or_side_transition_unresolved':{f'material{m}_epsilon_{eps}':n for (m,eps),n in sorted(side_unresolved_counts.items())}},
 'layers':{'local_F_frame_maxima':{'input':max((audit_by[(t['environment'],t['tet_id'])]['frame_input']['relative_error'] or 0 for t in tets),default=0),
                                  'P_world_frame':max((audit_by[(t['environment'],t['tet_id'])]['frame_stress']['relative_error'] or 0 for t in tets),default=0),
                                  'tangent_world_frame':max((audit_by[(t['environment'],t['tet_id'])]['frame_tangent']['relative_error'] or 0 for t in tets),default=0)},
            'element_force_reconstruction_max':max((audit_by[(t['environment'],t['tet_id'])]['element_force']['relative_error'] or 0 for t in tets),default=0),
            'production_gather_from_native_element_tangent':next(x['production_gather'] for x in json.loads((out/'assembled-audit.json').read_text()) if x['environment']==0),
            'oracle_gather_from_reprojected_exact_tangent':next(x['oracle_assembled_action'] for x in json.loads((out/'assembled-audit.json').read_text()) if x['environment']==0),
            'production_gather_from_oracle_tangent_at_captured_candidate':assembly},
 'capture055_comparison_contract':'same physical path as 055 per sealed prediction; only FGMRES iteration ceiling changed 256 to 1024; 057 is still a failed prefix',
 'terminal_failure_boundary':{'failed_newton_iteration':19,'nonlinear_residual':2.9222736e-5,'residual_tolerance':1e-6,'terminal_status_code':10,
                              'capture_selected_iteration':0,'therefore_layer_audit_is_not_a_terminal_iteration_diagnosis':True},
 'per_tet_records':by_mat
}
for mat,kind in [(0,'liner'),(1,'medium')]:
 rs=by_mat[mat]
 summary['per_material'][kind]={
   'count':len(rs),
   'candidate_residual':stats(r['candidate_residual'] for r in rs),
   'candidate_state_error_vs_converged_fp64':stats(r['candidate_state_error'] for r in rs),
   'native_P_vs_converged_fp64_P':stats(r['stress_vs_exact_projection'] for r in rs),
   'native_P_vs_same_captured_candidate_fp64_P':stats(r['stress_vs_captured_candidate'] for r in rs),
   'native_tangent_vs_converged_fp64_state_tangent':stats(r['tangent_vs_exact_projection'] for r in rs),
   'native_tangent_vs_same_captured_candidate_state_tangent':stats(r['tangent_vs_captured_candidate'] for r in rs),
   'native_tangent_vs_converged_fp64_state_tangent_outside_transition_band':stats(r['tangent_vs_exact_projection'] for r in rs if not r['near_transition']),
   'native_tangent_vs_same_candidate_tangent_near_transition_band':stats(r['tangent_vs_captured_candidate'] for r in rs if r['near_transition']),
   'near_transition_tet_ids':[{'environment':r['environment'],'tet_id':r['tet_id'],'trial_q':r['elastic_trial_q'],
                               'candidate_residual':r['candidate_residual'],
                               'tangent_error_exact_projection':r['tangent_vs_exact_projection'],
                               'tangent_error_same_candidate':r['tangent_vs_captured_candidate']}
                              for r in rs if r['near_transition']]
 }
path=out/'capture057-layer-analysis.json'
path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='per_tet_records'},indent=2,allow_nan=False))
