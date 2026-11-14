from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-release-turn-altitude-20261114');D=A/'push-20261126-rebound-integrated';R=Path('artifacts/crow-v15-early-long-flight-20261114/push-20261126-rebound-long'); meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation']
def read(path,ref=False):
 d={'state':{},'target':{},'foot':{},'toe':{},'pron':{},'errors':[]}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=(x['lane'],x['step']);
    if k in d['state']:d['errors'].append('duplicate state '+str(k))
    d['state'][k]=x;continue
   z=line.split()
   if not z:continue
   kind=z[0]
   if 'OBSERVER_ERROR' in kind or kind=='CROW_SUBSTEP_ERROR':d['errors'].append(line.strip());continue
   if kind=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);k=(e,t)
    if not ref or t<=600:
     if k in d['target']:d['errors'].append('duplicate target '+str(k))
     d['target'][k]=list(map(float,z[3:]))
    continue
   if kind not in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_PRONATION_EFFORT'):continue
   t,sub,e=map(int,z[1:4]);k=(e,t,sub)
   if ref and (kind!='CROW_TWO_TOE_SUBSTEP' or t>600):continue
   tab={'CROW_TWO_TOE_SUBSTEP':'foot','CROW_TOE_V_SUBSTEP':'toe','CROW_PRONATION_EFFORT':'pron'}[kind]
   if k in d[tab] or not 0<=sub<16:d['errors'].append('duplicate or invalid '+str((kind,k)))
   if kind=='CROW_TWO_TOE_SUBSTEP':
    if int(z[4])>int(z[5]):d['errors'].append('contact overflow '+str(k))
    d[tab][k]=list(map(float,z[6:]))
   else:d[tab][k]=list(map(float,z[4:]))
 return d
n=read(D/'run.stderr.gz');r=read(R/'run.stderr.gz',True)
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def tilt(q):return math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))
rows=[]
for e in range(8):
 end=min(meta['steps_by_environment'][e],1800);s=n['state'];ref=r['state'];errors=n['errors']+r['errors'];mismatch={'state':sum(s.get((e,t))!=ref.get((e,t)) for t in range(1,601)), 'target':sum(n['target'].get((e,t))!=r['target'].get((e,t)) for t in range(335,601)), 'contact':sum(n['foot'].get((e,t,k))!=r['foot'].get((e,t,k)) for t in range(335,601) for k in range(16))}
 def unloaded(t):return all((e,t,k) in n['foot'] and sum(n['foot'][e,t,k][2:4])<=1e-7 for k in range(16))
 release=next((t for t in range(371,end-1) if all(unloaded(j) for j in range(t,t+3))),None);recontact=next((t for t in range((release or end)+3,end+1) if not unloaded(t)),None)
 nonfoot=next(((t,k) for t in range(release or 371,end+1) for k in range(16) if (e,t,k) in n['foot'] and abs(n['foot'][e,t,k][4])>1e-7),None);cap=next(((t,k) for t in range(release or 371,end+1) for k in range(16) if (e,t,k) in n['toe'] and max(map(abs,n['toe'][e,t,k]))>=11.99),None)
 missing=sum((e,t) not in s for t in range(1,end+1))+sum((e,t) not in n['target'] for t in range(335,end+1))+sum((e,t,k) not in n[typ] for typ in ('foot','toe') for t in range(335,end+1) for k in range(16));finite=all(all(map(math.isfinite,x['q']+x['v']+x['policy_action'])) for (ee,t),x in s.items() if ee==e and t<=end) and all(all(map(math.isfinite,v)) for typ in ('target','foot','toe') for key,v in n[typ].items() if key[0]==e and key[1]<=end)
 samples=[s[e,t] for t in range(450,end+1) if (e,t) in s];h=[x['q'][2] for x in samples];tilts=[tilt(x['q']) for x in samples];lat=[abs(x['q'][1]-s[e,450]['q'][1]) for x in samples] if (e,450) in s else []
 yawdiff=[(t,wrap(yaw(s[e,t]['q'])-yaw(ref[e,t]['q']))) for t in range(600,min(end,1200)+1) if (e,t) in s and (e,t) in ref];turn=(min(yawdiff,key=lambda v:v[1]) if e%2==0 else max(yawdiff,key=lambda v:v[1])) if yawdiff else None
 level=[abs(s[e,t]['q'][2]-.55) for t in range(950,1151) if (e,t) in s];climb=[abs(s[e,t]['q'][2]-.80) for t in range(1500,1751) if (e,t) in s];pron=[n['pron'][e,t,k] for t in range(600,min(end,1200)+1) for k in range(16) if (e,t,k) in n['pron']];eff=[abs(x[j]) for x in pron if len(x)==14 for j in (0,7)];track=[abs(x[j]-x[j+1]) for x in pron if len(x)==14 for j in (1,8)]
 checks={'complete_first_episode':end==1800 and not meta['physical_failure_by_environment'][e],'matched_prefix':not any(mismatch.values()),'complete_observations':not missing and not errors,'finite':finite,'three_interval_release':release is not None,'no_foot_recontact':release is not None and recontact is None,'no_nonfoot_impulse':nonfoot is None,'no_toe_speed_cap':cap is None,'height_tilt_lateral':len(h)==1351 and min(h)>=.25 and max(h)<=1.30 and max(tilts)<=.35 and max(lat)<=2.,'signed_turn_vs_straight':len(yawdiff)==601 and turn is not None and (turn[1]<=-.20 if e%2==0 else turn[1]>=.20),'level_error':len(level)==201 and sum(level)/len(level)<=.05,'climb_error':len(climb)==251 and sum(climb)/len(climb)<=.05,'pronation_observed':len(pron)==601*16 and all(len(x)==14 and all(map(math.isfinite,x)) for x in pron)}
 rows.append({'lane':e,'checks':checks,'pass':all(checks.values()),'prefix_mismatches':mismatch,'first_release':release,'first_recontact':recontact,'first_nonfoot_impulse':nonfoot,'first_toe_speed_cap':cap,'missing_count':missing,'height_range':[min(h),max(h)] if h else None,'max_tilt':max(tilts,default=None),'max_lateral':max(lat,default=None),'signed_yaw_vs_straight':turn,'level_mae':sum(level)/len(level) if level else None,'climb_mae':sum(climb)/len(climb) if climb else None,'pronation_saturated_fraction':sum(x>=.1499 for x in eff)/len(eff) if eff else None,'pronation_max_tracking_error':max(track,default=None)})
out={'rows':rows,'errors':n['errors']+r['errors'],'scope':'Fixed-profile integrated development screen on an already-used layout and seed; contact/state epochs not independently aligned; not flight, biological, landing or mission qualification.'};(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
