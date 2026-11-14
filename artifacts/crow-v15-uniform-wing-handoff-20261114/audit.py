from pathlib import Path
import gzip, json, math
A=Path('artifacts/crow-v15-uniform-wing-handoff-20261114'); D=A/'push-20261126-rebound-uniform'
B=Path('artifacts/crow-v15-descent-observer-extension-20261114/push-20261126-rebound-observed')
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
def states(path):
 d={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d[x['lane'],x['step']]=x
 return d
n=read(D/'run.stderr.gz');control=read(B/'run.stderr.gz');base=control['state'];straight=states(Path('artifacts/crow-v15-early-long-flight-20261114/push-20261126-rebound-long/run.stderr.gz'));meta=json.loads((D/'run.stdout').read_text());inputs=json.loads((D/'inputs.json').read_text());basein=json.loads((B/'inputs.json').read_text());build=json.loads((A/'push-hashes.json').read_text());
def roll(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def tilt(q):return math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))
rows=[]
for e in range(8):
 s=n['state'];end=meta['first_episode_navigation']['steps_by_environment'][e]
 def unloaded(t):return all((e,t,k) in n['foot'] and sum(n['foot'][e,t,k][2:4])<=1e-7 for k in range(16))
 release=next((t for t in range(371,min(end,2400)-1) if all(unloaded(j) for j in range(t,t+3))),None)
 missing=sum((e,t) not in s for t in range(1,2401))+sum((e,t) not in n['target'] for t in range(335,2401))+sum((e,t,k) not in n[typ] for typ in ('foot','toe') for t in range(335,2401) for k in range(16))
 recontact=next((t for t in range((release or 2400)+3,min(end,2400)+1) if any((e,t,k) in n['foot'] and sum(n['foot'][e,t,k][2:4])>1e-7 for k in range(16))),None)
 nonfoot=next(((t,k) for t in range(release or 371,min(end,2400)+1) for k in range(16) if (e,t,k) in n['foot'] and abs(n['foot'][e,t,k][4])>1e-7),None)
 cap=next(((t,k) for t in range(release or 371,min(end,2400)+1) for k in range(16) if (e,t,k) in n['toe'] and max(map(abs,n['toe'][e,t,k]))>=11.99),None)
 samples=[s[e,t] for t in range(450,2401) if (e,t) in s];heights=[x['q'][2] for x in samples];tilts=[tilt(x['q']) for x in samples];lateral=[abs(x['q'][1]-s[e,450]['q'][1]) for x in samples] if (e,450) in s else []
 level=[abs(s[e,t]['q'][2]-.55) for t in range(950,1151) if (e,t) in s];climb=[abs(s[e,t]['q'][2]-.80) for t in range(1500,1751) if (e,t) in s];descent=[abs(s[e,t]['q'][2]-.55) for t in range(2100,2351) if (e,t) in s]
 finite=all(all(map(math.isfinite,x['q']+x['v']+x['policy_action'])) for x in samples) and all(all(map(math.isfinite,v)) for kind in ('target','foot','toe') for key,v in n[kind].items() if key[0]==e)
 first=next((t for t in range(335,2401) if n['target'].get((e,t))!=control['target'].get((e,t))),None)
 stop=2400 if e in (1,5) else (first-1 if first is not None else 2400)
 mismatch={'state':sum((e,t) not in s or (e,t) not in base or any(s[e,t].get(k)!=base[e,t].get(k) for k in ('q','v','policy_action','contacts')) for t in range(1,stop+1)),'target':sum((e,t) not in n['target'] or (e,t) not in control['target'] or n['target'][e,t]!=control['target'][e,t] for t in range(335,stop+1)),'contact':sum((e,t,k) not in n['foot'] or (e,t,k) not in control['foot'] or n['foot'][e,t,k]!=control['foot'][e,t,k] for t in range(335,stop+1) for k in range(16))}
 pre_state_bad=sum((e,t) not in s or (e,t) not in base or s[e,t]['q']!=base[e,t]['q'] or s[e,t]['v']!=base[e,t]['v'] for t in range(1,(release or 371)+1))
 def foot_imp(obs):return sum(sum(obs['foot'][e,t,k][2:4]) for t in range(335,(release or 371)+1) for k in range(16) if (e,t,k) in obs['foot'])
 impulse_difference=foot_imp(n)-foot_imp(control)
 turns=[(t,wrap(wrap(yaw(s[e,t]['q'])-yaw(s[e,600]['q']))-wrap(yaw(straight[e,t]['q'])-yaw(straight[e,600]['q'])))) for t in range(600,1201) if (e,t) in s and (e,600) in s and (e,t) in straight and (e,600) in straight];extreme=(min(turns,key=lambda x:x[1]) if e%2==0 else max(turns,key=lambda x:x[1])) if turns else None
 checks={'expected_metallib':inputs['metallib_sha256']==build['metallib'],'unchanged_other_inputs':all(inputs[k]==basein[k] for k in basein if k!='metallib_sha256'),'matched_intervention_prefix':not any(mismatch.values()) and ((first is None) if e in (1,5) else (first is not None)),'teacher_off_no_scheduled_resets':not meta['birdflow_journey_teacher'] and not meta['scheduled_resets'],'complete_first_episode':end==2400 and not meta['first_episode_navigation']['physical_failure_by_environment'][e],'complete_observations':not missing and not n['errors'],'finite':finite,'three_interval_release':release is not None,'no_foot_recontact':release is not None and recontact is None and missing==0,'no_nonfoot_impulse':nonfoot is None,'no_toe_speed_cap':cap is None,'height_tilt_lateral':len(heights)==1951 and min(heights)>=.25 and max(heights)<=1.3 and max(tilts)<=.35 and max(lateral)<=2.,'signed_turn_vs_straight':len(turns)==601 and extreme is not None and (extreme[1]<=-.20 if e%2==0 else extreme[1]>=.20),'level_error':len(level)==201 and sum(level)/len(level)<=.05,'climb_error':len(climb)==251 and sum(climb)/len(climb)<=.05,'descent_error':len(descent)==251 and sum(descent)/len(descent)<=.05}
 rows.append({'lane':e,'checks':checks,'pass':all(checks.values()),'first_changed_target':first,'prefix_mismatches':mismatch,'preunload_state_mismatch_count':pre_state_bad,'observed_335_through_release_foot_impulse_difference_Ns':impulse_difference,'first_release':release,'first_recontact':recontact,'first_nonfoot_impulse':nonfoot,'first_toe_speed_cap':cap,'missing_count':missing,'height_range':[min(heights),max(heights)] if heights else None,'max_tilt':max(tilts,default=None),'max_lateral':max(lateral,default=None),'signed_yaw_vs_straight':extreme,'level_mae':sum(level)/len(level) if level else None,'climb_mae':sum(climb)/len(climb) if climb else None,'descent_mae':sum(descent)/len(descent) if descent else None})
out={'rows':rows,'errors':n['errors'],'scope':'Uniform launch-amplitude screen on used seed/layout; not general flight, landing or mission qualification. Contact/state timing not independently verified.'};(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
