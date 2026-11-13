from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-consistent-wingheld-leg-control-20261113');states={};forces={};work={}
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-wingheld/run.stderr.gz')
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if not z or z[0] not in ('CROW_KIN_STATE','CROW_AERO_FORCE','CROW_LEG_EFFORT'):continue
   t,k,e=map(int,z[1:4]);
   if e<2 or not (348<=t<=365):continue
   key=arm,e,t,k
   if z[0]=='CROW_KIN_STATE':assert key not in states;states[key]=list(map(float,z[47:]));assert len(states[key])==40
   elif z[0]=='CROW_AERO_FORCE':assert key not in forces;forces[key]=float(z[7])
   else:
    assert key not in work;vals=list(map(float,z[7:]));assert len(vals)==30;work[key]=sum(vals[j]*(vals[j+2]-vals[j+1]) for j in range(0,30,5))
a=json.loads((A/'analysis.json').read_text());c=json.loads((A/'com-analysis.json').read_text());cs={(x['arm'],x['lane']):x for x in c['rows']};rows=[]
for e in range(2,8):
 r={'lane':e};x=a['rows'][e];r['early_delta_348_355_ns']=x['impulse_348_355_delta_ns'];r['early_delta_348_357_ns']=x['early_impulse_delta_ns'];r['delta_com_vz_365_m_s']=cs['push',e]['com_vz_after_m_s']-cs['hold',e]['com_vz_after_m_s']
 for arm in ('push','hold'):
  keys=[(arm,e,t,k) for t in range(348,366) for k in range(16)];assert all(q in states and q in forces and q in work for q in keys)
  r[arm]={'max_flap_speed_rad_s':max(abs(states[q][j]) for q in keys for j in (8,9)),'max_sweep_speed_rad_s':max(abs(states[q][j]) for q in keys for j in (6,7)),'max_pronation_speed_rad_s':max(abs(states[q][j]) for q in keys for j in (10,11)),'vertical_aero_impulse_ns':sum(forces[q] for q in keys)*.00125,'leg_effort_displacement_proxy_j':sum(work[q] for q in keys)}
 r['isolation_pass']=r['early_delta_348_355_ns']>.05 and r['early_delta_348_357_ns']>.05 and r['delta_com_vz_365_m_s']>.05 and all(r[arm]['max_flap_speed_rad_s']<=1 and abs(r[arm]['vertical_aero_impulse_ns'])<=.02 for arm in ('push','hold'))
 rows.append(r)
res={'rows':rows,'scope':'Exploratory wing-held counterfactual in one training seed. Aero impulse includes wing and body; foot impulse difference does not isolate muscle force. No biological qualification.'};(A/'isolation-analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
