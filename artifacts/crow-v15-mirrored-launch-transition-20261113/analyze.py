from pathlib import Path
import collections,gzip,json,math,os,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-mirrored-launch-transition-20261113';L=R/'build/crow-v15-mirrored-leg-isolation-20261113';D={};inputs=[];keys={};idx=0
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-transition');d={'state':{},'contact':{},'toe':{},'target':{},'effort':{},'aero':{},'frames':{},'masks':[],'errors':[]};d['meta']=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];d['inputs']=json.loads((p/'inputs.json').read_text());d['exit']=json.loads((p/'run-result.json').read_text())['exit_code']
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);
    if key in d['state']:d['errors'].append(['duplicate_state',key])
    d['state'][key]=x;continue
   z=line.split()
   if not z:continue
   k=z[0]
   if k=='CROW_WING_FRAME':d['frames'][int(z[1])]=z[2:]
   elif k=='CROW_SCENE_MASK':d['masks'].append(z[1:])
   elif k=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);key=(e,t);v=list(map(float,z[3:]));
    if key in d['target'] or len(v)!=40 or not all(map(math.isfinite,v)):d['errors'].append(['target',key])
    d['target'][key]=v
   elif k in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_WING_EFFORT','CROW_AERO_FORCE'):
    t,sub,e=map(int,z[1:4]);key=(e,t,sub)
    if k=='CROW_AERO_FORCE':v=[float(z[7])];table='aero'
    elif k=='CROW_TWO_TOE_SUBSTEP':v=list(map(float,z[6:11]));table='contact';
    elif k=='CROW_TOE_V_SUBSTEP':v=list(map(float,z[4:]));table='toe'
    else:v=list(map(float,z[4:]));table='effort'
    if key in d[table] or not all(map(math.isfinite,v)):d['errors'].append([table,key])
    if k=='CROW_TWO_TOE_SUBSTEP' and (len(v)!=5 or int(z[4])>int(z[5])):d['errors'].append(['contact_overflow',key])
    if k=='CROW_TOE_V_SUBSTEP' and len(v)!=16:d['errors'].append(['toe_shape',key])
    if k=='CROW_WING_EFFORT' and len(v)!=14:d['errors'].append(['effort_shape',key])
    d[table][key]=v
   elif 'OBSERVER_ERROR' in k or k=='CROW_SUBSTEP_ERROR':d['errors'].append([k,line.strip()])
 D[arm]=d
 for (e,t),x in sorted(d['state'].items()):
  if 345<=t<=500:
   keys[idx]=(arm,e,t);inputs.append(str(idx)+' 0 '+' '.join(map(str,x['q']+x['v'])));idx+=1
q=D['push']['state'][2,347]['q'];zero=[0.]*40;move=zero.copy();move[:3]=[.3,-.2,.1]
for j,v in ((idx,zero),(idx+1,move)):inputs.append(str(j)+' 0 '+' '.join(map(str,q+v)))
r=subprocess.run([str(L/'kinematics')],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(L)});fk={}
for line in r.stdout.splitlines():
 z=line.split();assert z[0]=='C';i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));fk[i]=v
assert len(fk)==idx+2;cal=max(abs((fk[idx+1][4+j]-fk[idx][4+j])-move[j]) for j in range(3));assert cal<1e-8;com={keys[i]:v for i,v in fk.items() if i<idx};p,h=D['push'],D['hold'];axes=[]
for j in (4,5):v=p['frames'].get(j,[]);axes.append(list(map(float,v[4:7])) if len(v)>=7 else None)
shared=all(d['exit']==0 and not d['errors'] for d in D.values()) and p['masks']==h['masks'] and p['frames']==h['frames'] and axes==[[0.,1.,0.],[0.,1.,0.]] and all(p['inputs'][k]==h['inputs'][k] for k in ('host_sha256','landing_policy_sha256','route_policy_sha256'));rows=[]
for e in range(2,8):
 end=min(500,p['meta']['steps_by_environment'][e]);hend=min(500,h['meta']['steps_by_environment'][e]);first=next((t for t in range(335,min(end,hend)+1) if p['target'].get((e,t))!=h['target'].get((e,t))),None);prefix=[t for t in range(1,first or 501) if (e,t) not in p['state'] or (e,t) not in h['state'] or any(p['state'][e,t][k]!=h['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];wingdiff=[t for t in range(335,min(end,hend)+1) if (e,t) not in p['target'] or (e,t) not in h['target'] or p['target'][e,t][6:12]!=h['target'][e,t][6:12]]
 coverage={};
 for arm,d,stop in (('push',p,end),('hold',h,hend)):
  coverage[arm]={'state':sum((e,t) not in d['state'] for t in range(1,stop+1)),'target':sum((e,t) not in d['target'] for t in range(335,stop+1)),'contact':sum((e,t,k) not in d['contact'] for t in range(335,stop+1) for k in range(16)),'toe':sum((e,t,k) not in d['toe'] for t in range(335,stop+1) for k in range(16)),'effort':sum((e,t,k) not in d['effort'] for t in range(348,stop+1) for k in range(16)),'aero':sum((e,t,k) not in d['aero'] for t in range(335,min(stop,400)+1) for k in range(16))}
 def support(d,t):return any(sum(d['contact'].get((e,t,k),[0]*5)[2:4])>1e-7 for k in range(16))
 release=next((t for t in range(351,min(end-1,398)+1) if all(not support(p,u) for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,end+1) if support(p,t)),None) if release else None;hold_supported=support(h,release) if release and release<=hend else None;first_wing=next((t for t in range(371,min(end,hend)+1) if p['target'].get((e,t),[])[6:12]!=p['target'].get((e,370),[])[6:12]),None)
 start=first or 348;last=release or min(end,400);footdelta=sum(sum(p['contact'].get((e,t,k),[0]*5)[:2])-sum(h['contact'].get((e,t,k),[0]*5)[:2]) for t in range(start,last+1) for k in range(16)) if last<=min(end,hend) else None;aero=sum((p['aero'].get((e,t,k),[0])[0]-h['aero'].get((e,t,k),[0])[0])*.00125 for t in range(start,min(last,400)+1) for k in range(16));cv=com.get(('push',e,release));hv=com.get(('hold',e,release));before=[(t,k) for t in range(start,last+1) for k in range(16)];after=[(t,k) for t in range(release or end,end+1) for k in range(16)];nonfoot=lambda ks:sum(p['contact'].get((e,t,k),[0]*5)[4] for t,k in ks);toecount=lambda ks:sum(max(map(abs,p['toe'].get((e,t,k),[0])))>=11.99 for t,k in ks);effmax=max((max(abs(p['effort'][e,t,k][0]),abs(p['effort'][e,t,k][7])) for t in range(348,end+1) for k in range(16) if (e,t,k) in p['effort']),default=None);samples=[p['state'][e,t] for t in range(450,end+1) if (e,t) in p['state']];vx=[x['v'][0] for x in samples];q400=p['state'].get((e,400));q500=p['state'].get((e,500));dx=q500['q'][0]-q400['q'][0] if q400 and q500 else None;sweep=max((max(abs(x['q'][7]),abs(x['q'][8])) for x in samples),default=None)
 checks={'shared':shared,'prefix':first is not None and not prefix,'wing_targets':not wingdiff,'complete':end==500 and hend==500 and all(not any(x.values()) for x in coverage.values()),'release':release is not None and cv is not None and cv[6]>0 and hv is not None and cv[6]-hv[6]>.05 and hold_supported and footdelta is not None and footdelta>.10,'pre_release_contacts':nonfoot(before)<=1e-7 and toecount(before)==0,'sustained_contacts':release is not None and recontact is None and nonfoot(after)<=1e-7 and toecount(after)<=1,'actuator':effmax is not None and effmax<=3.00001 and sweep is not None and sweep<.2,'forward':dx is not None and dx>.15 and .15<=dx/2<=1. and len(vx)==51 and min(vx)>0 and max(vx)<=1.}
 rows.append({'lane':e,'end':end,'hold_end':hend,'first_leg_target_difference':first,'first_wing_target_departure':first_wing,'prefix_differences':prefix,'wing_target_differences':wingdiff,'coverage':coverage,'release_tick':release,'hold_supported_at_push_release':hold_supported,'recontact':recontact,'push_com_vz':cv[6] if cv else None,'hold_com_vz_at_push_release':hv[6] if hv else None,'push_minus_hold_foot_impulse_ns':footdelta,'push_minus_hold_vertical_aero_impulse_ns':aero,'pre_release_nonfoot':nonfoot(before),'pre_release_toe_events':toecount(before),'post_release_nonfoot':nonfoot(after),'post_release_toe_events':toecount(after),'flap_effort_max':effmax,'sweep_max_abs':sweep,'forward_displacement_400_500':dx,'vx_range_450_500':[min(vx),max(vx)] if vx else None,'checks':checks,'passed':all(checks.values())})
out={'shared':shared,'corrected_axes':axes,'com_translation_calibration_error':cal,'rows':rows,'pairs':{str(e)+'-'+str(e+1):rows[e-2]['passed'] and rows[e-1]['passed'] for e in (2,4,6)},'scope':'Used-seed timed onset developmental transition, not generalized contact-triggered takeoff or flight qualification.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
