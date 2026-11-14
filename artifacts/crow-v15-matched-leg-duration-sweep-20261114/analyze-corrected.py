from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-matched-leg-duration-sweep-20261114');data={};meta={};inputs={}
for arm in ('push','hold'):
 d=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt')); data[arm]={k:{(tuple(v[:3]) if k in ('contact','aero','leg','toe') else tuple(v[:2])):v[(3 if k in ('contact','aero','leg','toe') else 2):] for v in d[k]} for k in ('target','contact','aero','leg','toe')};data[arm]['state']={(v['lane'],v['step']):v for v in d['state']};data[arm]['errors']=d['errors'];p=A/(arm+'-20261126-rebound-nonleg');meta[arm]=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];inputs[arm]=json.loads((p/'inputs.json').read_text())
parity=[];rows=[]
for e in range(8):
 bad=[];missing=[];wing=[]
 for t in range(1,369):
  x=data['push']['state'].get((e,t));y=data['hold']['state'].get((e,t))
  if not x or not y:missing.append(t)
  elif any(x[k]!=y[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 for t in range(335,501):
  x=data['push']['target'].get((e,t));y=data['hold']['target'].get((e,t));
  if x is None or y is None or len(x)<14 or len(y)<14 or x[6:12]!=y[6:12]:wing.append(t)
 parity.append({'lane':e,'state_prefix_through':368,'state_missing':missing,'state_mismatch':bad,'wing_command_interval':[335,500],'wing_mismatch_or_missing':wing})
 for arm in ('push','hold'):
  d=data[arm];c=d['contact'];a=d['aero'];state=d['state'];end=min(meta[arm]['steps_by_environment'][e],500)
  def unloaded(t):return all((e,t,k) in c and sum(c[e,t,k][2:4])<=1e-7 for k in range(16))
  release=next((t for t in range(371,end-1) if all(unloaded(u) for u in range(t,t+3))),None)
  recontact=next((t for t in range((release or end)+3,end+1) if not unloaded(t)),None)
  sample_end=release if release is not None else end+1;keys=[(e,t,k) for t in range(335,sample_end) for k in range(16)];full=all(k in a and k in c for k in keys)
  impulse=sum(a[k][3]*.02/16 for k in keys) if full else None;ground=sum(sum(c[k][2:4]) for k in keys) if full else None
  extras=[]
  for typ in ('contact','aero','leg','toe'):
   for t in range(335,end+1):
    for k in range(16):
     v=d[typ].get((e,t,k))
     if v is None or not all(math.isfinite(x) for x in v):extras.append([typ,t,k])
  speed=[abs(v) for (lane,t,k),vs in d['toe'].items() if lane==e and 335<=t<=end for v in vs]
  rows.append({'arm':arm,'lane':e,'episode_end':end,'physical_failure':meta[arm]['physical_failure_by_environment'][e],'first_three_interval_unload':release,'first_later_contact_or_missing':recontact,'integral_window':[335,sample_end-1],'integral_complete':full,'modeled_vertical_aero_impulse_Ns':impulse,'foot_upward_impulse_Ns':ground,'state_at_first_unload':{'position_m':state[e,release]['q'][:3],'linear_velocity_m_s':state[e,release]['v'][:3]} if release is not None and (e,release) in state else None,'max_toe_speed_rad_s':max(speed,default=None),'missing_or_nonfinite_substep_count':len(extras),'first_missing_or_nonfinite':extras[:8]})
out={'parity':parity,'rows':rows,'shared_nonshader_inputs':all(inputs['push'][k]==inputs['hold'][k] for k in ('host_sha256','frozen_flight_sha256','landing_policy_sha256','route_policy_sha256')),'observer_errors':{k:v['errors'] for k,v in data.items()},'scope':'matched time-scheduled wing targets, used-seed physical model; force is modeled and observer epochs are not independently aligned'};(A/'analysis-corrected.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
