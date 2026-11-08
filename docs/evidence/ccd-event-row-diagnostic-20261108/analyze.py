#!/usr/bin/env python3
"""Independently verify the opt-in event-row snapshot against the rigid fixture."""
import argparse,json,math,tarfile
from pathlib import Path
p=Path(__file__).parent
before=json.loads((p/'off.json').read_text())['runs']
after=json.loads((p/'on.json').read_text())['runs']
parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);archive=parser.parse_args().archive
with tarfile.open(archive) as t:
 old=json.load(t.extractfile('ccd-useful-motion005/world-steps.json'))['runs']
old={(r['scale'],r['limit'],r['mode'],r['fixture']):r for r in old}
def without_timing(x):return {k:v for k,v in x.items() if k!='elapsed_with_initialization'}
assert len(before)==len(after)==12
for x,y in zip(before,after):
 assert without_timing(x)==without_timing(y)
 assert {k:v for k,v in without_timing(y).items() if k not in ('final_q','final_v')}==without_timing(old[tuple(y[k] for k in ('scale','limit','mode','fixture'))])
 assert len(y['final_q'])==7 and len(y['final_v'])==6
 assert y['repeat_identical'] and y['rejected_state_restored']
cases={};key=None
for line in (p/'on.stderr').read_text().splitlines():
 if line.startswith('ccd_case='):
  c=json.loads(line.split('=',1)[1]);key=(c['scale'],c['fixture']);cases[key]=[]
 elif line.startswith('ccd_event_trace='):
  assert key is not None;cases[key].append(json.loads(line.split('=',1)[1]))
assert len(cases)==12
out=[]
for case in after:
 key=(case['scale'],case['fixture']); events=cases[key]
 assert len(events)==8 and events[:4]==events[4:]
 failure=next((i for i,e in enumerate(events[:4]) if e['status']!=0),None)
 selected=events[(failure-1) if failure is not None else 0]
 contacts=[]
 for c in selected['contacts']:
  assert (c['body_a'],c['body_b'])==(0,1)
  j=c['jacobians'];v=selected['v'];n=c['normal']; assert len(j)==36
  jv=[sum((j[18+axis*6+k]-j[axis*6+k])*v[k] for k in range(6)) for axis in range(3)]
  vn=sum(n[i]*jv[i] for i in range(3))
  q=selected['q'];point=c['point']; w=v[3:6]
  geometric=v[1]+w[2]*(point[0]-q[0])-w[0]*(point[2]-q[2])
  equation=vn-c['evaluated_target']+c['regularization']*c['J']
  residual=abs(equation) if c['J']>1e-6 else max(-equation,0.)
  assert abs(c['row_gap']-c['gap'])<1e-7 and abs(c['row_target']+max(c['gap'],0.)/.01)<1e-6
  assert abs(vn-geometric)<2e-5
  contacts.append({'jacobian_normal_velocity_m_s':vn,'geometric_normal_velocity_m_s':geometric,'evaluated_target_m_s':c['evaluated_target'],'impulse_Ns':c['J'],'normal_equation_residual_m_s':residual})
 if contacts:
  assert abs(max(c['normal_equation_residual_m_s'] for c in contacts)-selected['residuals'][1])<2e-5
 if case['fixture'] in (12,14):
  assert failure is not None and contacts and selected['status']==0
  assert max(c['normal_equation_residual_m_s'] for c in contacts)>.4
 if case['fixture'] in (13,15):assert case['accepted'] and case['normal_impulse_Ns']==0
 out.append({'fixture':case['fixture'],'scale':case['scale'],'accepted':case['accepted'],'first_failure_pass':failure,'selected_pass':selected['pass'],'solver_status':selected['status'],'reported_normal_residual_m_s':selected['residuals'][1],'contacts':contacts})
print(json.dumps({'schema':'numi.ccd.event-row-diagnostic.v1','trace_on_off_physical_results_identical':True,'matches_retained_physical_results':True,'repeated_trace_identical':True,'cases':out,'limitations':['The trace is a read-only snapshot of the selected event boundary; events in ultimately rejected steps are not published trajectories.','Matching the normal residual does not by itself explain the cause of incomplete solver convergence.','This covers the single-floating-box hybrid CCD fixture only; no performance or general contact qualification follows.']},indent=2))
