#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
p=Path(__file__).parent
baseline=Path(sys.argv[1]); candidate=json.loads((p/'on.json').read_text())['runs']; off=json.loads((p/'off.json').read_text())['runs']; original=json.loads((baseline/'on.json').read_text())['runs']; geometry=json.loads((p/'geometry-v3.json').read_text())
def clean(x):return {k:v for k,v in x.items() if k!='elapsed_with_initialization'}
assert len(candidate)==len(off)==len(original)==12
assert all(clean(x)==clean(y) for x,y in zip(candidate,off))
assert all(x['repeat_identical'] and x['rejected_state_restored'] for x in candidate)
assert geometry['all_accepted_segments_qualified'] and len(geometry['accepted_runs'])==sum(r['accepted'] for r in candidate)
events={};key=None
for line in (p/'on.stderr').read_text().splitlines():
 if line.startswith('ccd_case='):
  c=json.loads(line.split('=',1)[1]);key=(c['scale'],c['fixture']);events[key]=[]
 elif line.startswith('ccd_event_trace='):events[key].append(json.loads(line.split('=',1)[1]))
rows=[]
for run,control in zip(candidate,original):
 assert all(run[k]==control[k] for k in ('scale','limit','mode','fixture'))
 k=(run['scale'],run['fixture']);e=events[k];assert len(e)==8 and e[:4]==e[4:]
 maxima=[]
 for frame in e[:4]:
  if frame['status']!=0 or not frame['contacts']:continue
  residuals=[]
  for c in frame['contacts']:
   assert (c['body_a'],c['body_b'])==(0,1)
   j=c['jacobians'];v=frame['v'];n=c['normal']
   jv=sum(n[axis]*sum((j[18+axis*6+k]-j[axis*6+k])*v[k] for k in range(6)) for axis in range(3))
   eq=jv-c['evaluated_target']+c['regularization']*c['J']
   residuals.append(abs(eq) if c['J']>1e-6 else max(-eq,0.))
  calc=max(residuals);assert abs(calc-frame['residuals'][1])<2e-5
  maxima.append({'pass':frame['pass'],'reported_residual_m_s':frame['residuals'][1],'reconstructed_residual_m_s':calc})
 rows.append({'scale':run['scale'],'fixture':run['fixture'],'main_accepted':control['accepted'],'candidate_accepted':run['accepted'],'candidate_status':run['status'],'candidate_normal_impulse_Ns':run['normal_impulse_Ns'],'successful_event_residuals':maxima})
misses=all(r['candidate_accepted'] and r['candidate_normal_impulse_Ns']==0 for r in rows if r['fixture'] in (13,15))
impacts=all(r['candidate_accepted'] for r in rows if r['fixture'] in (12,14))
residuals=all(e['reconstructed_residual_m_s']<=.002 for r in rows for e in r['successful_event_residuals'])
assert misses and not impacts and not residuals
print(json.dumps({'schema':'numi.ccd.cold-hybrid.v1','trace_on_off_results_equal':True,'repeat_and_rollback_gates_pass':True,'accepted_trajectory_geometry_gate_pass':True,'near_miss_gate_pass':misses,'thin_impact_completion_gate_pass':impacts,'normal_residual_gate_pass':residuals,'candidate_promotable':False,'rows':rows,'limitations':['The candidate has an independently retained mixed rod-rigid residual failure.','This is a scoped correctness experiment, not a force calibration or performance measurement.','The observer bounds only its stated floating-box interpolation domain and sampled acceleration envelope.']},indent=2))
