from pathlib import Path
import gzip, json, math
A=Path('artifacts/crow-v15-gust-wrench-observer-20261114'); D=A/'push-20261126-rebound-gust'
B=A/'push-20261126-rebound-control'
def read(path):
 d={k:{} for k in ('state','target','foot','toe','force')};d['errors']=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);table='state';key=(x['lane'],x['step']);value=x
   else:
    z=line.split()
    if not z: continue
    kind=z[0]
    if 'OBSERVER_ERROR' in kind or kind=='CROW_SUBSTEP_ERROR': d['errors'].append(line.strip());continue
    if kind=='CROW_DRIVE_TARGET':
     t,e=map(int,z[1:3]);table='target';key=e,t;value=list(map(float,z[3:]))
    elif kind in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_PRE_AERO_WRENCH'):
     t,sub,e=map(int,z[1:4]);key=e,t,sub
     if not 0<=sub<16:d['errors'].append('invalid substep '+str(key))
     if kind=='CROW_TWO_TOE_SUBSTEP':
      if int(z[4])>int(z[5]):d['errors'].append('contact overflow '+str(key))
      table='foot';value=list(map(float,z[6:]))
     elif kind=='CROW_TOE_V_SUBSTEP':table='toe';value=list(map(float,z[4:]))
     else:table='force';value=list(map(float,z[4:]))
    else:continue
   if key in d[table]:d['errors'].append('duplicate '+table+' '+str(key))
   d[table][key]=value
 return d
n=read(D/'run.stderr.gz');b=read(B/'run.stderr.gz'); meta=json.loads((D/'run.stdout').read_text());basein=json.loads((B/'inputs.json').read_text());actual=json.loads((D/'inputs.json').read_text());build=json.loads(Path('artifacts/crow-v15-launch-gust-20261114/push-hashes.json').read_text());
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def delta(a,b):return math.atan2(math.sin(a-b),math.cos(a-b))
def tilt(q):return math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))
def all_zero_wrench(obs,lanes):
 return all((e,t,k) in obs['force'] and len(obs['force'][e,t,k])==6 and all(math.isfinite(v) and abs(v)<1e-9 for v in obs['force'][e,t,k]) for e in lanes for t in range(895,1021) for k in range(16))
control_zero=all_zero_wrench(b,range(8));unaffected_zero=all_zero_wrench(n,range(6))
rows=[]
for e in range(8):
 end=meta['first_episode_navigation']['steps_by_environment'][e];s=n['state'];ref=b['state'];stop=900 if e>=6 else 2400
 mismatch={'state':sum(any(s.get((e,t),{}).get(k)!=ref.get((e,t),{}).get(k) for k in ('q','v','policy_action','contacts')) or (e,t) not in s or (e,t) not in ref for t in range(1,stop+1)), 'target':sum(n['target'].get((e,t))!=b['target'].get((e,t)) or (e,t) not in n['target'] or (e,t) not in b['target'] for t in range(335,stop+1)), 'contact':sum(n['foot'].get((e,t,k))!=b['foot'].get((e,t,k)) or (e,t,k) not in n['foot'] or (e,t,k) not in b['foot'] for t in range(335,stop+1) for k in range(16))}
 def unloaded(t):return all((e,t,k) in n['foot'] and sum(n['foot'][e,t,k][2:4])<=1e-7 for k in range(16))
 release=next((t for t in range(371,min(end,2400)-1) if all(unloaded(j) for j in range(t,t+3))),None)
 missing=sum((e,t) not in s for t in range(1,2401))+sum((e,t) not in n['target'] for t in range(335,2401))+sum((e,t,k) not in n[typ] for typ in ('foot','toe') for t in range(335,2401) for k in range(16))
 recontact=next((t for t in range((release or 2400)+3,min(end,2400)+1) if any((e,t,k) in n['foot'] and sum(n['foot'][e,t,k][2:4])>1e-7 for k in range(16))),None)
 nonfoot=next(((t,k) for t in range(release or 371,min(end,2400)+1) for k in range(16) if (e,t,k) in n['foot'] and abs(n['foot'][e,t,k][4])>1e-7),None)
 cap=next(((t,k) for t in range(release or 371,min(end,2400)+1) for k in range(16) if (e,t,k) in n['toe'] and max(map(abs,n['toe'][e,t,k]))>=11.99),None)
 samples=[s[e,t] for t in range(450,2401) if (e,t) in s];heights=[x['q'][2] for x in samples];tilts=[tilt(x['q']) for x in samples];lateral=[abs(x['q'][1]-s[e,450]['q'][1]) for x in samples] if (e,450) in s else []
 level=[abs(s[e,t]['q'][2]-.55) for t in range(950,1151) if (e,t) in s];climb=[abs(s[e,t]['q'][2]-.80) for t in range(1500,1751) if (e,t) in s];descent=[abs(s[e,t]['q'][2]-.55) for t in range(2100,2351) if (e,t) in s]
 finite=all(all(map(math.isfinite,x['q']+x['v']+x['policy_action'])) for x in samples) and all(all(map(math.isfinite,v)) for kind in ('target','foot','toe') for key,v in n[kind].items() if key[0]==e)
 checks={'control_wrench_zero':control_zero,'unaffected_wrench_zero':unaffected_zero,'shared_inputs_except_metallib':all(actual[k]==basein[k] for k in basein if k!='metallib_sha256'),'expected_metallib':actual['metallib_sha256']==build['metallib'],'teacher_off_no_scheduled_resets':not meta['birdflow_journey_teacher'] and not meta['scheduled_resets'],'complete_first_episode':end==2400 and not meta['first_episode_navigation']['physical_failure_by_environment'][e],'matched_prefix':not any(mismatch.values()),'complete_observations':not missing and not n['errors'] and not b['errors'],'finite':finite,'three_interval_release':release is not None,'no_foot_recontact':release is not None and recontact is None and missing==0,'no_nonfoot_impulse':nonfoot is None,'no_toe_speed_cap':cap is None,'height_tilt_lateral':len(heights)==1951 and min(heights)>=.25 and max(heights)<=1.3 and max(tilts)<=.35 and max(lateral)<=2.,'level_error':len(level)==201 and sum(level)/len(level)<=.05,'climb_error':len(climb)==251 and sum(climb)/len(climb)<=.05,'descent_error':len(descent)==251 and sum(descent)/len(descent)<=.05}
 row={'lane':e,'checks':checks,'prefix_mismatches':mismatch,'first_release':release,'first_recontact':recontact,'first_nonfoot_impulse':nonfoot,'first_toe_speed_cap':cap,'missing_count':missing,'height_range':[min(heights),max(heights)] if heights else None,'max_tilt':max(tilts,default=None),'max_lateral':max(lateral,default=None),'level_mae':sum(level)/len(level) if level else None,'climb_mae':sum(climb)/len(climb) if climb else None,'descent_mae':sum(descent)/len(descent) if descent else None}
 if e>=6:
  samplesw=[(t,k,n['force'].get((e,t,k)),b['force'].get((e,t,k))) for t in range(895,1021) for k in range(16)]
  valid=[(t,k,[x-y for x,y in zip(a,c)]) for t,k,a,c in samplesw if a is not None and c is not None and len(a)==len(c)==6 and all(map(math.isfinite,a+c))]
  active=[(t,k,v) for t,k,v in valid if max(map(abs,v))>1e-8];steps=sorted({t for t,_,_ in active});outside=[(t,k) for t,k,v in active if not 899<=t<=1005];maxforce=max((max(map(abs,v[:3])) for _,_,v in valid),default=None);norms=[math.sqrt(sum(x*x for x in v[3:])) for _,_,v in active];mean_y=sum(v[4] for _,_,v in active)/len(active) if active else 0
  checks['measured_external_disturbance']=len(valid)==126*16 and len(active)==1600 and len(steps)==100 and steps==list(range(steps[0],steps[0]+100)) and not outside and maxforce is not None and maxforce<1e-8 and all(abs(v-.04)<1e-6 for v in norms) and (mean_y>1e-5 if e==6 else mean_y< -1e-5)
  wave=[(t,delta(pitch(s[e,t]['q']),pitch(ref[e,t]['q'])),s[e,t]['v'][4]-ref[e,t]['v'][4]) for t in range(901,1051) if (e,t) in s and (e,t) in ref];dp=delta(pitch(s[e,1200]['q']),pitch(ref[e,1200]['q'])) if (e,1200) in s and (e,1200) in ref else None;dv=s[e,1200]['v'][4]-ref[e,1200]['v'][4] if dp is not None else None
  peakp=max((abs(x[1]) for x in wave),default=None);peakv=max((abs(x[2]) for x in wave),default=None)
  checks['measurable_response']=len(wave)==150 and (peakp>=.04 or peakv>=.10);checks['recovery_at_1200']=dp is not None and abs(dp)<.06 and abs(dv)<.15
  row.update({'wrench_active_substeps':len(active),'wrench_active_steps':steps,'wrench_max_force_N':maxforce,'wrench_torque_range_Nm':[min(norms),max(norms)] if norms else None,'wrench_mean_world_y_Nm':mean_y,'wrench_outside_window':outside,'peak_pitch_delta_rad':peakp,'peak_world_y_angular_rate_delta_rad_s':peakv,'pitch_delta_at_1200_rad':dp,'world_y_angular_rate_delta_at_1200_rad_s':dv})
 row['pass']=all(checks.values());rows.append(row)
out={'rows':rows,'errors':n['errors']+b['errors'],'scope':'Used-seed in-flight disturbance screen only; no perturbed-start, generalized command, landing or mission qualification. Contact/state epoch synchronization not independently established.'}
(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
