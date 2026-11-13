from pathlib import Path
import gzip,json,collections,math,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-front-toe-friction-ablation-20261113';paths={'baseline':R/'crow-v15-supported-wing-transition-20261113/push-wing-20261126-rebound-supportedwing','ablation':A/'push-wing-20261126-rebound-frontablation'};D={}
for name,p in paths.items():
 d={'state':{},'contact':{},'toe':{},'ground':collections.defaultdict(list),'errors':[],'meta':json.loads((p/'run.stdout').read_text())['first_episode_navigation'],'exit':json.loads((p/'run-result.json').read_text())['exit_code']};D[name]=d
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];assert key not in d['state'];d['state'][key]=x
   else:
    z=line.split()
    if not z:continue
    typ=z[0]
    if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_GROUND_CONTACT'):
     t,k,e=map(int,z[1:4]);key=e,t,k
     if typ=='CROW_TWO_TOE_SUBSTEP':assert key not in d['contact'];d['contact'][key]=list(map(float,z[6:11]));
     elif typ=='CROW_TOE_V_SUBSTEP':assert key not in d['toe'];d['toe'][key]=list(map(float,z[4:]))
     elif 348<=t<=375:
      a,b=map(int,z[4:6]);body=a if b==35 else b;vals=list(map(float,z[8:]));d['ground'][key].append({'body':body,'friction':vals[10:12],'impulses':vals[12:15]})
    elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(line.strip())
 for e in range(8):
  if d['meta']['steps_by_environment'][e]<400:d['errors'].append(['early end',e])
  if sum((e,t) in d['state'] for t in range(1,401))!=400:d['errors'].append(['state coverage',e])
  if sum((e,t,k) in d['contact'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['contact coverage',e])
  if sum((e,t,k) in d['toe'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['toe coverage',e])
 if d['exit']!=0:d['errors'].append(['exit',d['exit']])
front=set(range(19,25))|set(range(27,33));parity=[];rows=[]
for e in range(8):
 statebad=[t for t in range(1,351) if any(D['baseline']['state'][e,t][k]!=D['ablation']['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];cbad=[(t,k) for t in range(335,351) for k in range(16) if D['baseline']['contact'][e,t,k]!=D['ablation']['contact'][e,t,k]];parity.append({'lane':e,'state_mismatches_through_350':len(statebad),'contact_mismatches_through_350':len(cbad),'first_state_mismatch':min(statebad,default=None)})
for e in range(2,8):
 row={'lane':e}
 for name,d in D.items():
  support=lambda t:any(sum(d['contact'][e,t,k][2:4])>1e-7 for k in range(16))
  release=next((t for t in range(351,399) if not any(support(u) for u in range(t,t+3))),None)
  gp=[c for (ee,t,k),v in d['ground'].items() if ee==e and 351<=t<=375 for c in v];fp=[c for c in gp if c['body'] in front];other=[c for c in gp if c['body'] not in front];
  row[name]={'first_three_unloaded':release,'pre_release_gaps':[] if release is None else [t for t in range(348,release) if not support(t)],'recontacts':[] if release is None else [t for t in range(release+3,401) if support(t)],'toe_speed_threshold_substeps':sum(max(map(abs,d['toe'][e,t,k]))>=11.99 for t in range(348,401) for k in range(16)),'front_contact_points':len(fp),'front_nonzero_friction_points':sum(any(abs(y)>1e-7 for y in x['friction']) for x in fp),'front_nonzero_tangent_impulse_points':sum(any(abs(y)>1e-7 for y in x['impulses'][1:]) for x in fp),'other_nonzero_friction_points':sum(any(abs(y)>1e-7 for y in x['friction']) for x in other),'front_normal_impulse_ns':sum(x['impulses'][0] for x in fp),'front_tangent_sum_ns':sum(math.hypot(*x['impulses'][1:]) for x in fp),'peak_root_rise_m':max(d['state'][e,t]['q'][2]-d['state'][e,350]['q'][2] for t in range(351,401))}
 rows.append(row)
errors={k:v['errors'] for k,v in D.items()};valid=not any(errors.values()) and not any(x['state_mismatches_through_350'] or x['contact_mismatches_through_350'] for x in parity) and all(r['ablation']['front_contact_points'] and not r['ablation']['front_nonzero_friction_points'] and not r['ablation']['front_nonzero_tangent_impulse_points'] and r['ablation']['other_nonzero_friction_points'] for r in rows)
(A/'analysis.json').write_text(json.dumps({'valid':valid,'errors':errors,'parity':parity,'rows':rows,'scope':'Research-only transient friction ablation on one used seed, correlated lanes. No production or biological qualification.'},indent=2)+'\n')
print('valid',valid,'errors',errors,'parity',parity)
for x in rows:print(x)
if not valid:raise SystemExit('Ablation invalid: inspect analysis.json')
