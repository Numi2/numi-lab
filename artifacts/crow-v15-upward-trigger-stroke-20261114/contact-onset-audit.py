from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-upward-trigger-stroke-20261114');d={}
for arm in ('push','hold'):
 with gzip.open(A/(arm+'-window.json.gz'),'rt') as f:x=json.load(f)
 d[arm]={'state':{(v['lane'],v['step']):v for v in x['state']},'contact':{tuple(v[:3]):v[3:] for v in x['contact']},'toe':{tuple(v[:3]):v[3:] for v in x['toe']},'aero':{tuple(v[:3]):v[3:] for v in x['aero']},'target':{(v[0],v[1]):v[2:] for v in x['target']},'errors':x['errors']}
 out=json.loads((A/(arm+'-20261126-rebound-phase/run.stdout')).read_text());d[arm]['meta']=out['first_episode_navigation'];d[arm]['teacher_off']=out['birdflow_journey_teacher'] is False;d[arm]['inputs']=json.loads((A/(arm+'-20261126-rebound-phase/inputs.json')).read_text())
parity=[]
for e in range(8):
 end=368 if e in (6,7) else 500;bad=[]
 for t in range(1,end+1):
  a=d['push']['state'].get((e,t));b=d['hold']['state'].get((e,t))
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 parity.append({'lane':e,'through':end,'mismatch_count':len(bad),'first_mismatches':bad[:8]})
rows=[]
for arm in ('push','hold'):
 for e in (6,7):
  a=d[arm];c=a['contact'];toe=a['toe'];aero=a['aero'];s=a['state'];tar=a['target'];end=min(a['meta']['steps_by_environment'][e],500)
  missing=[(t,k) for t in range(335,end+1) for k in range(16) if any((e,t,k) not in a[n] for n in ('contact','toe','aero'))]
  def loaded(t):return any(sum(c[e,t,k][2:4])>1e-7 for k in range(16) if (e,t,k) in c)
  first=next((t for t in range(371,end+1) if all((e,t,k) in c for k in range(16)) and not loaded(t)),None)
  later=next((t for t in range((first or end)+1,end+1) if loaded(t)),None)
  targetend=first if first is not None else end;expected=[0.,0.,.35,.35,0.,0.];targetdev=[abs(tar[e,t][j]-expected[j-6]) for t in range(335,targetend+1) if (e,t) in tar for j in range(6,12)]
  nonfoot=sum(abs(c[e,t,k][4]) for t in range(335,end+1) for k in range(16) if (e,t,k) in c);caps=sum(max(map(abs,toe[e,t,k]))>=11.99 for t in range(335,end+1) for k in range(16) if (e,t,k) in toe)
  firststate=s.get((e,370));flap=max((abs(s[e,t]['q'][j]-firststate['q'][j]) for t in range(371,(first or end)+1) if (e,t) in s for j in (9,10)),default=None) if firststate else None
  before=range(371,first) if first else range(371,end+1);av=[aero[e,t,k][3]*.02/16 for t in before for k in range(16) if (e,t,k) in aero]
  rows.append({'arm':arm,'lane':e,'end':end,'first_unloaded':first,'first_recontact':later,'loaded_at_370':all((e,370,k) in c for k in range(16)) and loaded(370),'continuous':first is not None and later is None and end==500,'target_samples':len(targetdev),'target_expected':6*(targetend-334),'pre_unload_max_trim_deviation_rad':max(targetdev,default=None),'pre_unload_targets_held':len(targetdev)==6*(targetend-334) and max(targetdev,default=1)<1e-5,'max_observer_flap_displacement_pre_release_rad':flap,'modeled_pre_release_vertical_aero_impulse_Ns':sum(av),'missing_substeps':len(missing),'nonfoot_abs_impulse_Ns':nonfoot,'toe_threshold_substeps':caps,'observer_errors':a['errors'],'teacher_off':a['teacher_off'],'physical_failure':a['meta']['physical_failure_by_environment'][e]})
shared=('host_sha256','landing_policy_sha256','route_policy_sha256','frozen_flight_sha256');identities={'expected_shader':{arm:d[arm]['inputs']['metallib_sha256']==json.loads((A/(arm+'-hashes.json')).read_text())['metallib'] for arm in ('push','hold')},'matched_nonshader_inputs':all(d['push']['inputs'][k]==d['hold']['inputs'][k] for k in shared)}
out={'parity':parity,'identities':identities,'rows':rows,'scope':'Used-seed contact-gated development test; target and state epochs for these lanes are inferred; modeled aerodynamics are not biological measurements.'};(A/'contact-onset-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
