from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-drive-step-ablation-20261113';P={'baseline':R/'crow-v15-wing-effort-observer-20261113/push-wing-20261126-rebound-effort','ablated':A/'push-wing-20261126-rebound-driveoff'}
def read(p):
 d={'state':{},'contact':{},'wing':{},'aero':{},'errors':[]};d['result']=json.loads((p/'run-result.json').read_text());d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation']
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_SUPPORT_SAMPLE':
    x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step'];assert (e,t) not in d['state'];d['state'][e,t]=x
   elif z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);assert (e,t,k) not in d['contact'];d['contact'][e,t,k]=list(map(float,z[6:11]));
    if int(z[4])>int(z[5]):d['errors'].append('overflow')
   elif z[0]=='CROW_WING_EFFORT':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==14 and all(map(math.isfinite,v)) and (e,t,k) not in d['wing'];d['wing'][e,t,k]=[v[:7],v[7:]]
   elif z[0]=='CROW_AERO_FORCE':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[5:]));assert len(v)==12 and v[-1]==4 and (e,t,k) not in d['aero'];d['aero'][e,t,k]=v
   elif z[0] in ('CROW_AERO_OBSERVER_ERROR','CROW_SUBSTEP_ERROR'):d['errors'].append(l.strip())
 assert d['result']['exit_code']==0 and not d['errors'] and all(x>=400 for x in d['meta']['steps_by_environment'])
 assert len(d['wing'])==len(d['aero'])==8*28*16
 return d
d={n:read(p) for n,p in P.items()};b=d['baseline'];a=d['ablated'];mismatch=[]
for e in range(8):
 for t in range(1,369):
  if any(a['state'][e,t][k]!=b['state'][e,t][k] for k in ('q','v','policy_action','contacts')):mismatch.append(['state',e,t])
 for t in range(335,369):
  for k in range(16):
   if a['contact'][e,t,k]!=b['contact'][e,t,k]:mismatch.append(['contact',e,t,k])
 if a['aero'][e,369,0]!=b['aero'][e,369,0]:mismatch.append(['aero_start',e])
 for j in (0,1):
  if a['wing'][e,369,0][j][2]!=b['wing'][e,369,0][j][2] or a['wing'][e,369,0][j][4]!=b['wing'][e,369,0][j][4]:mismatch.append(['start',e,j])
 for k in range(16):
  if any(a['wing'][e,369,k][j][0]!=0 for j in (0,1)):mismatch.append(['nonzero_wing_effort',e,k])
if mismatch:
 (A/'integrity-failed.json').write_text(json.dumps(mismatch,indent=2)+'\n');raise SystemExit('Invalid ablation: parity or suppression failure')
rows=[]
for e in range(2,8):
 for j in range(2):
  x=b['wing'][e,369,0][j];y=a['wing'][e,369,0][j];rows.append({'lane':e,'wing':j,'baseline_working_effort':x[0],'ablated_working_effort':y[0],'matched_q':x[2],'matched_v':x[4],'baseline_presolve_v':x[5],'ablated_presolve_v':y[5],'drive_removal_presolve_delta_v':y[5]-x[5],'baseline_total_vertical_aero_force':b['aero'][e,369,0][2]})
out={'parity_through_step':368,'errors':mismatch,'scope':'one-step flap working-effort-buffer ablation; first-substep immediate effect at matched initial state; all later differences evolve from changed state','rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['lane'],x['wing'],'delta_v',round(x['drive_removal_presolve_delta_v'],5),'effort',x['baseline_working_effort'],x['ablated_working_effort'],'baseline aero Z',round(x['baseline_total_vertical_aero_force'],4))
