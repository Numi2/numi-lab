from pathlib import Path
import collections,gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-hip-push-geometry-20261113';E=R/'artifacts/crow-v15-inrange-knee-push-20261113';O=R/'artifacts/crow-v15-leg-work-observer-20261113';S=R/'artifacts/crow-v15-slow-leg-push-20261113';arms={'inrange':A/'push-nowings-20261127-rebound-hip','baseline':E/'push-nowings-20261127-rebound-inrange','hold':O/'nopush-nowings-20261127-rebound-work','prior_slow':S/'push-nowings-20261127-rebound-slow160'}
def load(p):
 d={'state':{},'contact':{},'work':{},'toe':{},'errors':[]};d['run_result']=json.loads((p/'run-result.json').read_text());d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];
    if 1<=t<=400:assert (e,t) not in d['state'];d['state'][(e,t)]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in d['contact'];d['contact'][(e,t,k)]=list(map(float,z[6:11]));
    if int(z[4])>int(z[5]):d['errors'].append(['overflow',e,t,k])
   elif line.startswith('CROW_LEG_EFFORT '):
    z=line.split();t,k,e=map(int,z[1:4]);assert tuple(map(int,z[4:7]))==(1,0,0);v=list(map(float,z[7:]));assert len(v)==30;d['work'][(e,t,k)]=[v[i:i+5] for i in range(0,30,5)]
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);d['toe'][(e,t,k)]=max(abs(float(x)) for x in z[4:])
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 if d['run_result']['exit_code']!=0:d['errors'].append('nonzero exit')
 for e in range(8):
  if d['meta']['steps_by_environment'][e]<400:d['errors'].append(['early first-episode end',e])
  if sum((e,t) in d['state'] for t in range(1,401))!=400:d['errors'].append(['missing state',e])
  if sum((e,t,k) in d['contact'] for t in range(335,381) for k in range(16))!=46*16:d['errors'].append(['missing contact',e])
  if sum((e,t,k) in d['work'] for t in range(335,381) for k in range(16))!=46*16:d['errors'].append(['missing effort',e])
  if sum((e,t,k) in d['toe'] for t in range(335,381) for k in range(16))!=46*16:d['errors'].append(['missing toe',e])
 return d
D={n:load(p) for n,p in arms.items()}
def same(a,b):return all(a[k]==b[k] for k in ('q','v','policy_action','contacts'))
parity=[]
for e in range(8):
 missing=[t for t in range(1,346) if not same(D['inrange']['state'][(e,t)],D['hold']['state'][(e,t)])];contact=[(t,k) for t in range(335,346) for k in range(16) if D['inrange']['contact'][(e,t,k)]!=D['hold']['contact'][(e,t,k)]];parity.append({'lane':e,'endpoint_mismatches':len(missing),'contact_mismatches':len(contact),'first_endpoint':missing[:1],'first_contact':contact[:1]})
for e in range(8):
 missing=[t for t in range(1,346) if not same(D['inrange']['state'][(e,t)],D['baseline']['state'][(e,t)])];contact=[(t,k) for t in range(335,346) for k in range(16) if D['inrange']['contact'][(e,t,k)]!=D['baseline']['contact'][(e,t,k)]];parity.append({'lane':e,'reference':'baseline','endpoint_mismatches':len(missing),'contact_mismatches':len(contact),'first_endpoint':missing[:1],'first_contact':contact[:1]})
if any(d['errors'] for d in D.values()) or any(x['endpoint_mismatches'] or x['contact_mismatches'] for x in parity):
 (A/'integrity-failed.json').write_text(json.dumps({'parity':parity,'errors':{n:x['errors'] for n,x in D.items()}},indent=2)+'\n');raise SystemExit('Trial invalid: see integrity-failed.json')
rows=[]
for e in range(8):
 row={'lane':e}
 for name,d in D.items():
  imp=sum(sum(d['contact'][(e,t,k)][:2]) for t in range(348,376) for k in range(16));normal=[any(sum(d['contact'][(e,t,k)][2:4])>1e-7 for k in range(16)) for t in range(351,381)];release=next((351+i for i in range(len(normal)-2) if not any(normal[i:i+3])),None);rise=max(d['state'][(e,t)]['q'][2]-d['state'][(e,350)]['q'][2] for t in range(351,381));work=sum(j[0]*(j[2]-j[1]) for t in range(348,376) for k in range(16) for j in d['work'][(e,t,k)]);nonfoot=sum(d['contact'][(e,t,k)][4] for t in range(348,381) for k in range(16));caps=sum(d['toe'][(e,t,k)]>=11.99 for t in range(348,381) for k in range(16))
  row[name]={'upward_foot_impulse_ns':imp,'root_vz375_m_s':d['state'][(e,375)]['v'][2],'maximum_root_rise_m':rise,'first_three_unloaded_intervals':release,'net_leg_work_proxy_j':work,'other_ground_impulse_ns':nonfoot,'toe_speed_cap_substeps':caps}
 row['impulse_difference_vs_baseline_ns']=row['inrange']['upward_foot_impulse_ns']-row['baseline']['upward_foot_impulse_ns'];row['impulse_difference_vs_hold_ns']=row['inrange']['upward_foot_impulse_ns']-row['hold']['upward_foot_impulse_ns'];row['screen_impulse_pass']=row['impulse_difference_vs_hold_ns']>=.05;row['screen_lift_pass']=row['inrange']['first_three_unloaded_intervals'] is not None and row['inrange']['maximum_root_rise_m']>=.05;rows.append(row)
out={'parity':parity,'errors':{n:d['errors'] for n,d in D.items()},'rows':rows,'scope':'One development seed, correlated lanes; modeled generalized leg effort, passive aero and active crouch controls. Screen only; no physical or mission qualification.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print('lane',r['lane'],'delta impulse',round(r['impulse_difference_vs_hold_ns'],5),'rise',round(r['inrange']['maximum_root_rise_m'],5),'release',r['inrange']['first_three_unloaded_intervals'],'v375',round(r['inrange']['root_vz375_m_s'],5),'work',round(r['inrange']['net_leg_work_proxy_j'],5),'other',round(r['inrange']['other_ground_impulse_ns'],6),'toe caps',r['inrange']['toe_speed_cap_substeps'],'screens',r['screen_impulse_pass'],r['screen_lift_pass'])
