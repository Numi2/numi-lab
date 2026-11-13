from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-bank-damping-episode-trigger-20261113');P=A/sys.argv[1];seed=next(x for x in ('20261126','20261127') if x in P.name)
R=Path('artifacts/crow-v15-mirrored-amplitude-screen-20261113/pronation-small-20261126-rebound-small')
def read(p):
 d={'state':{},'contact':collections.defaultdict(lambda:[0.,0.,0]),'toe_count':collections.Counter(),'toe_events':collections.Counter(),'effort_count':collections.Counter(),'effort_max':collections.defaultdict(float),'body_count':collections.Counter(),'body_max':collections.defaultdict(float),'body_hits':collections.Counter(),'target':{},'ground':{},'mask':[],'model':{},'model_errors':[],'errors':[],'duplicates':[]};seen=set();body_indices=list(range(6,22))+[38,39]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step']
    if key in d['state']:d['duplicates'].append(['state',key])
    d['state'][key]=x;continue
   z=line.split()
   if not z:continue
   kind=z[0]
   if kind=='CROW_SCENE_MASK':d['mask'].append((z[1],int(z[2]),int(z[3])))
   if 'OBSERVER_ERROR' in kind or kind=='CROW_SUBSTEP_ERROR':d['errors'].append(line.strip())
   if kind=='CROW_MODEL_DOF_LIMIT':
    index,v,flags=int(z[1]),int(z[2]),int(z[3]);lim=float(z[4]);value=(index,flags,lim,z[5])
    if v in d['model'] and d['model'][v]!=value:d['model_errors'].append(['conflicting_model_metadata',v])
    d['model'][v]=value
   if kind in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_WING_EFFORT','CROW_BODY_JOINT_V_SUBSTEP'):
    t,k,e=map(int,z[1:4]);key=(kind,e,t,k)
    if key in seen:d['duplicates'].append(['substep',key])
    seen.add(key)
    if kind=='CROW_TWO_TOE_SUBSTEP':d['contact'][e,t][0]+=sum(map(float,z[8:10]));d['contact'][e,t][1]+=float(z[10]);d['contact'][e,t][2]+=1
    elif kind=='CROW_TOE_V_SUBSTEP':
     vals=list(map(float,z[4:]));d['toe_count'][e,t]+=1
     if len(vals)!=16 or not all(math.isfinite(x) for x in vals):d['errors'].append('invalid toe velocity sample')
     else:d['toe_events'][e,t]+=int(max(map(abs,vals))>=11.99)
    elif kind=='CROW_WING_EFFORT':
     vals=list(map(float,z[4:]));d['effort_count'][e,t]+=1
     if len(vals)!=14 or not all(math.isfinite(x) for x in vals):d['errors'].append('invalid wing effort sample')
     else:d['effort_max'][e]=max(d['effort_max'][e],abs(vals[0]),abs(vals[7]))
    else:
     vals=list(map(float,z[4:]));d['body_count'][e,t]+=1
     if len(vals)!=len(body_indices) or not all(math.isfinite(x) for x in vals):d['errors'].append('invalid body velocity sample')
     elif t>=500:
      for j,val in zip(body_indices,vals):
       d['body_max'][e,j]=max(d['body_max'][e,j],abs(val));m=d['model'].get(j)
       if m and m[1]&8 and m[2]>0 and abs(val)>=m[2]-1e-4:d['body_hits'][e,j]+=1
   elif kind=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);key=(e,t)
    if key in d['target']:d['duplicates'].append(['target',key])
    d['target'][key]=list(map(float,z[3:]))
   elif kind=='CROW_GROUND_CONTACT':
    t,k,e=map(int,z[1:4]);d['ground'].setdefault((e,t),line.strip())
 return d
old=read(R);new=read(P);om=json.loads((R/'run.stdout').read_text())['first_episode_navigation'];nm=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ni=json.loads((P/'inputs.json').read_text());oi=json.loads((R/'inputs.json').read_text());same=ni['host_sha256']=='5e4e8f807d87f430356956813e94d61144dddd8d53fba7966e98d3b16dc65a03' and oi['host_sha256']=='a9eb6747e68b1ba48d9aeb60f46467f11f4b84740331d36f7597d3e82a165998' and all(ni[k]==oi[k] for k in ('landing_policy_sha256','route_policy_sha256'));masks=new['mask']==old['mask'];rows=[]
def yaw(x):
 a,b,c,w=x['q'][3:7];return math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))
def roll(x):
 a,b,c,w=x["q"][3:7];return math.atan2(2*(w*a+b*c),1-2*(a*a+b*b))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def smooth(a,b,x):
 u=max(0.,min(1.,(x-a)/(b-a)));return u*u*(3.-2.*u)
for e in range(8):
 end=min(nm['steps_by_environment'][e],1800);stop=om['steps_by_environment'][e] if e<6 else 600;missing=[t for t in range(1,stop+1) if (e,t) not in old['state'] or (e,t) not in new['state']];diff=[t for t in range(1,stop+1) if (e,t) in old['state'] and (e,t) in new['state'] and any(old['state'][e,t][k]!=new['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];release=next((t for t in range(351,max(351,end-1)) if all(new['contact'][e,u][2]==16 and new['contact'][e,u][0]<=1e-7 for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,end+1) if new['contact'][e,t][0]>1e-7),None) if release else None;ground=next(((t,new['ground'][e,t]) for t in range((release or end)+1,end+1) if (e,t) in new['ground']),None);first=ground[0] if ground else recontact
 samples=[new['state'][e,t] for t in range(500,end+1) if (e,t) in new['state']];late=[x for x in samples if x['step']>=550];hs=[x['q'][2] for x in samples];vx=[x['v'][0] for x in late];vy=[abs(x['v'][1]) for x in late];qs=[x['q'][j] for x in samples for j in (11,12)];dyaw=[];dy=[];droll=[]
 for t in range(650,min(end,1150)+1):
  a=old['state'].get((e,t));b=new['state'].get((e,t))
  if a and b:dyaw.append(wrap(yaw(b)-yaw(a)));dy.append(b['q'][1]-a['q'][1]);droll.append(wrap(roll(b)-roll(a)))
 errs=[];amp=.06 if e<4 else .10 if e<6 else .08
 for t in range(435,end+1):
  target=new['target'].get((e,t));seconds=(t-2)*.02
  if target is not None:
   
   if e<6: errs.append(0.)
   elif seconds>=9.5: errs.append(max(0.,abs((target[8]-target[9])/2)-.025*abs(math.sin(2*math.pi*6.4*(seconds-7.44)))))
 cnt=lambda d,lo:sum(d[e,t]!=16 for t in range(lo,end+1));row={'lane':e,'end':end,'physical_failure':nm['physical_failure_by_environment'][e],'prefix_missing':len(missing),'prefix_different':len(diff),'release':release,'recontact':recontact,'first_ground':ground,'accepted_before_first_contact':new['state'].get((e,first-1)) if first else None,'targets_before_first_contact':new['target'].get((e,first-1)) if first else None,'height_range':[min(hs),max(hs)] if hs else None,'vx_range':[min(vx),max(vx)] if vx else None,'max_abs_vy':max(vy) if vy else None,'pronation_range':[min(qs),max(qs)] if qs else None,'roll_difference_range':[min(droll),max(droll)] if droll else None,'yaw_difference_range':[min(dyaw),max(dyaw)] if dyaw else None,'y_difference_range':[min(dy),max(dy)] if dy else None,'positive_signed_event':any(y>=.10 and d>=.20 for y,d in zip(dyaw,dy)),'negative_signed_event':any(y<=-.10 and d<=-.20 for y,d in zip(dyaw,dy)),'toe_events':sum(new['toe_events'][e,t] for t in range(348,end+1)),'flap_effort_max':new['effort_max'][e],'nonfoot_impulse':sum(new['contact'][e,t][1] for t in range(release,end+1)) if release else None,'missing_states':sum((e,t) not in new['state'] for t in range(1,end+1)),'missing_contacts':sum(new['contact'][e,t][2]!=16 for t in range(335,end+1)),'missing_toe':cnt(new['toe_count'],335),'missing_effort':cnt(new['effort_count'],348),'missing_body':cnt(new['body_count'],335),'missing_target':sum((e,t) not in new['target'] for t in range(335,end+1)),'target_samples':len(errs),'target_max_error':max(errs,default=None)};rows.append(row)
shared=same and masks and not new['errors'] and not new['duplicates'] and all(x['prefix_missing']==0 and x['prefix_different']==0 for x in rows[:6]);out={'rows':rows,'shared_validity':shared,'expected_observer_host_and_matching_policies':same,'scene_masks_match':masks,'observer_errors':new['errors'],'duplicate_count':len(new['duplicates']),'scope':'Opposite bounded bank commands and physical flap targets on one used seed; not qualified turns or navigation.'};(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print('lane',x['lane'],'end',x['end'],'prefix',x['prefix_missing'],x['prefix_different'],'height',x['height_range'],'vx',x['vx_range'],'dyaw',x['yaw_difference_range'],'dy',x['y_difference_range'],'target err',x['target_max_error'])
print('shared valid',shared)
