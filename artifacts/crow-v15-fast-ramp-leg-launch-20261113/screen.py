from pathlib import Path
import collections,gzip,json,hashlib
A=Path('artifacts/crow-v15-fast-ramp-leg-launch-20261113');a=json.loads((A/'analysis.json').read_text());c=json.loads((A/'com-analysis.json').read_text());cm={(r['arm'],r['lane']):r for r in c['rows']};D={}
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-fastramped/run.stderr.gz');d={'kin':{},'aero':{},'contact':{},'toe':{},'target':{},'state':{}};D[arm]=d
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d['state'][x['lane'],x['step']]=x;continue
   z=line.split()
   if not z or z[0] not in ('CROW_KIN_STATE','CROW_AERO_FORCE','CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_DRIVE_TARGET'):continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);
    if e>=2 and 348<=t<=400:d['target'][e,t]=list(map(float,z[3:]))
    continue
   t,k,e=map(int,z[1:4]);
   if e<2 or not 348<=t<=400:continue
   key=e,t,k
   if z[0]=='CROW_KIN_STATE':d['kin'][key]=list(map(float,z[47:]))
   elif z[0]=='CROW_AERO_FORCE':d['aero'][key]=float(z[7])
   elif z[0]=='CROW_TWO_TOE_SUBSTEP':d['contact'][key]=list(map(float,z[6:11]))
   else:d['toe'][key]=max(abs(float(x)) for x in z[4:])
for arm,d in D.items():
 for e in range(2,8):
  for t in range(348,401):
   assert (e,t) in d['target']
   for k in range(16):assert all((e,t,k) in d[x] for x in ('kin','aero','contact','toe'))
rows=[];parity=[]
for e in range(2,8):
 first=next((t for t in range(348,401) if D['push']['target'][e,t]!=D['hold']['target'][e,t]),None);assert first is not None; sm=[t for t in range(1,first) if any(D['push']['state'][e,t][k]!=D['hold']['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];cc=[(t,k) for t in range(348,first) for k in range(16) if D['push']['contact'][e,t,k]!=D['hold']['contact'][e,t,k]];parity.append({'lane':e,'first_target_difference':first,'prior_state_mismatches':sm,'prior_contact_mismatches':cc});assert not sm and not cc,(e,first,sm,cc)
 for arm in ('push','hold'):
  d=D[arm];ar=a['rows'][e][arm];cr=cm[arm,e];release=ar['first_three_unloaded'];end=release if release is not None else 400;keys=[(e,t,k) for t in range(348,end+1) for k in range(16)]; toe=sum(d['toe'][q]>=11.99 for q in keys);nonfoot=sum(d['contact'][q][4] for q in keys);flap=max(abs(d['kin'][q][j]) for q in keys for j in (8,9));aero=sum(d['aero'][q] for q in keys)*.00125
  footdelta=sum(sum(D['push']['contact'][e,t,k][:2])-sum(D['hold']['contact'][e,t,k][:2]) for t in range(371,end+1) for k in range(16)) if end>=371 else None
  row={'arm':arm,'lane':e,'release_tick':release,'com_vz_at_release_m_s':cr['com_vz_at_release_m_s'],'peak_com_rise_m':cr['com_peak_rise_from_350_m'],'push_minus_hold_foot_normal_impulse_371_to_release_ns':footdelta,'toe_threshold_substeps_through_release':toe,'nonfoot_normal_impulse_through_release_ns':nonfoot,'peak_flap_speed_through_release_rad_s':flap,'vertical_aero_impulse_through_release_ns':aero,'leg_target_samples':{str(t):d['target'][e,t][13:19] for t in (370,373,375,380,385,390,395,400)}}
  row['screen_pass']=arm=='push' and release is not None and 371<=release<398 and cr['com_vz_at_release_m_s']>0 and footdelta is not None and footdelta>0 and nonfoot<=1e-7 and toe==0 and flap<=1. and abs(aero)<=.02
  rows.append(row)
out={'rows':rows,'target_divergence':parity,'pair_pass':{str(e)+'-'+str(e+1):all(next(x for x in rows if x['arm']=='push' and x['lane']==k)['screen_pass'] for k in (e,e+1)) for e in (2,4,6)},'source_logs_sha256':{arm:hashlib.sha256((A/(arm+'-wing-20261126-rebound-fastramped/run.stderr.gz')).read_bytes()).hexdigest() for arm in D},'scope':'Development seed and correlated settings; support impulse is not isolated muscle work; active wing holding and research aerodynamics persist. No mission or biological qualification.'};(A/'screen.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['arm'],r['lane'],'release',r['release_tick'],'COMvz',r['com_vz_at_release_m_s'],'delta impulse',r['push_minus_hold_foot_normal_impulse_371_to_release_ns'],'toe',r['toe_threshold_substeps_through_release'],'nonfoot',r['nonfoot_normal_impulse_through_release_ns'],'flap',r['peak_flap_speed_through_release_rad_s'],'aero',r['vertical_aero_impulse_through_release_ns'],'pass',r['screen_pass'])
