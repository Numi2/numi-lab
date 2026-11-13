from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-position-flight-screen-20261113');P=A/sys.argv[1];seed=next(x for x in ('20261126','20261127') if x in P.name)
R=Path('artifacts/crow-v15-lateral-uniform-confirmation-20261113')/('lateral-uniform-'+seed+'-rebound-uniform')
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
old=read(R);new=read(P);meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];oldmeta=json.loads((R/'run.stdout').read_text())['first_episode_navigation'];test_inputs=json.loads((P/'inputs.json').read_text());base_inputs=json.loads((R/'inputs.json').read_text());policy_ok=all(test_inputs[k]==base_inputs[k] for k in ('landing_policy_sha256','route_policy_sha256'));host_ok=test_inputs['host_sha256']==json.loads((A/'position-flight-20261126-rebound-position'/'inputs.json').read_text())['host_sha256'];policy_ok=policy_ok and host_ok;mask_ok=new['mask']==old['mask']
indices=list(range(6,22))+[38,39];metadata_ok=not new['model_errors'] and all(j in new['model'] and new['model'][j][0]==j and new['model'][j][1]&8 and new['model'][j][2]>0 for j in indices+list(range(22,38)))
def smooth(a,b,x):
 u=max(0.,min(1.,(x-a)/(b-a)));return u*u*(3.-2.*u)
def feedback(x,t,target,e):
 xx,yy,zz,w=x['q'][3:7];n=math.sqrt(xx*xx+yy*yy+zz*zz+w*w);xx/=n;yy/=n;zz/=n;w/=n
 roll=math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy));wx,wy,wz=x['v'][3:6];rate=(1-2*(yy*yy+zz*zz))*wx+2*(xx*yy+w*zz)*wy+2*(xx*zz-w*yy)*wz
 yawx=1-2*(yy*yy+zz*zz);yawy=2*(w*zz+xx*yy);norm=math.hypot(yawx,yawy);vyaw=(-yawy*x['v'][0]+yawx*x['v'][1])/norm;gain=0. if e<4 else .10 if e<6 else .20;correction=max(-.20,min(.20,gain*x['q'][1]));desired=max(-.18,min(.18,1.20*(vyaw+correction)));seconds=(t-2)*.02;on=smooth(9.,9.5,seconds);delta=on*max(-.025,min(.025,.08*(desired-roll)-.025*rate))
 sine=(target[8]+target[9]-.70)/1.40
 if abs(sine)>=.85:return None
 return max(abs(target[8]-(.35+(.70+delta)*sine)),abs(target[9]-(.35+(.70-delta)*sine))),abs(sine-math.sin(2*math.pi*6.4*(seconds-7.44)))
rows=[]
for e in range(2,8):
 end=min(meta['steps_by_environment'][e],1800);prefix=1200 if e<4 else 450;missing=[t for t in range(1,prefix+1) if (e,t) not in old['state'] or (e,t) not in new['state']];diff=[t for t in range(1,prefix+1) if (e,t) in old['state'] and (e,t) in new['state'] and any(old['state'][e,t][k]!=new['state'][e,t][k] for k in ('q','v','policy_action','contacts'))]
 miss_state=[t for t in range(1,end+1) if (e,t) not in new['state']];miss_contact=[t for t in range(335,end+1) if new['contact'][e,t][2]!=16];miss_toe=[t for t in range(335,end+1) if new['toe_count'][e,t]!=16];miss_body=[t for t in range(335,end+1) if new['body_count'][e,t]!=16];miss_effort=[t for t in range(348,end+1) if new['effort_count'][e,t]!=16];miss_target=[t for t in range(335,end+1) if (e,t) not in new['target']]
 release=next((t for t in range(351,max(351,end-1)) if all(new['contact'][e,u][2]==16 and new['contact'][e,u][0]<=1e-7 for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,end+1) if new['contact'][e,t][0]>1e-7),None) if release else None;ground=next(((t,new['ground'][e,t]) for t in range((release or end)+1,min(1800,end+1)+1) if (e,t) in new['ground']),None);nonfoot=sum(new['contact'][e,t][1] for t in range(release,end+1)) if release else None
 samples=[new['state'][e,t] for t in range(500,end+1) if (e,t) in new['state']];late=[new['state'][e,t] for t in range(550,end+1) if (e,t) in new['state']];start=new['state'].get((e,500));last=new['state'].get((e,1800));height=[min((x['q'][2] for x in samples),default=None),max((x['q'][2] for x in samples),default=None)];vx=[min((x['v'][0] for x in late),default=None),max((x['v'][0] for x in late),default=None)];vy=max((abs(x['v'][1]) for x in late),default=None);dy=max((abs(x['q'][1]-start['q'][1]) for x in samples),default=None) if start else None;dx=last['q'][0]-start['q'][0] if last and start else None
 rec=[feedback(new['state'][e,t-1],t,new['target'][e,t],e) for t in range(455,end+1) if (e,t-1) in new['state'] and (e,t) in new['target']];rec=[x for x in rec if x is not None];fb=max((x[0] for x in rec),default=None);wave=max((x[1] for x in rec),default=None);sweep=all((e,t) in new['target'] and all(-.090001<=x<=.025001 for x in new['target'][e,t][6:8]) for t in range(435,end+1))
 body=[{'v_index':j,'name':new['model'].get(j,(None,None,None,None))[3],'max_abs_rad_s':new['body_max'][e,j],'limit_rad_s':new['model'].get(j,(None,None,None))[2],'at_limit_substeps':new['body_hits'][e,j]} for j in indices];toe_events=sum(new['toe_events'][e,t] for t in range(348,end+1));effort=new['effort_max'][e]
 passed=bool(end==1800 and not meta['physical_failure_by_environment'][e] and not missing and not diff and mask_ok and policy_ok and metadata_ok and not new['errors'] and not new['duplicates'] and not miss_state and not miss_contact and not miss_toe and not miss_body and not miss_effort and not miss_target and release is not None and recontact is None and ground is None and nonfoot is not None and nonfoot<=1e-7 and toe_events==0 and effort<=3.00001 and len(rec)>=30 and fb is not None and fb<=1e-4 and wave is not None and wave<=2e-4 and sweep and height[0] is not None and .25<=height[0] and height[1]<=1.5 and vx[0] is not None and vx[0]>0 and vx[1]<=1 and vy is not None and vy<=.4 and dx is not None and dx>1 and dy is not None and dy<=1)
 rows.append({'lane':e,'end':end,'termination_reason':meta['termination_reason_by_environment'][e],'physical_failure':meta['physical_failure_by_environment'][e],'missing_prefix_ticks':missing,'different_prefix_ticks':diff,'exact_required_prefix':not missing and not diff,'missing_state_ticks':miss_state,'missing_contact_ticks':miss_contact,'missing_toe_ticks':miss_toe,'missing_body_velocity_ticks':miss_body,'missing_effort_ticks':miss_effort,'missing_target_ticks':miss_target,'release':release,'recontact':recontact,'first_ground_record':ground,'accepted_state_before_ground':new['state'].get((e,ground[0]-1)) if ground else None,'wing_targets_before_ground':new['target'].get((e,ground[0]-1),[])[6:12] if ground else None,'nonfoot_impulse_ns':nonfoot,'toe_threshold_events':toe_events,'flap_effort_max':effort,'body_joint_speeds':body,'height_range_m':height,'vx_range_m_s':vx,'max_abs_vy_m_s':vy,'max_abs_y_displacement_m':dy,'forward_displacement_m':dx,'feedback_samples':len(rec),'feedback_max_error_rad':fb,'waveform_max_error':wave,'sweep_targets_valid':sweep,'preliminary_engineering_pass':passed})
controls=[]
for e in (0,1):
 stop=oldmeta['steps_by_environment'][e];missing=[t for t in range(1,stop+1) if (e,t) not in old['state'] or (e,t) not in new['state']];diff=[t for t in range(1,stop+1) if (e,t) in old['state'] and (e,t) in new['state'] and any(old['state'][e,t][k]!=new['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];controls.append({'lane':e,'old_end':stop,'new_end':meta['steps_by_environment'][e],'missing_ticks':missing,'different_ticks':diff,'exact':stop==meta['steps_by_environment'][e] and not missing and not diff})
out={'seed':seed,'rows':rows,'controls':controls,'model_metadata':new['model'],'model_metadata_valid':metadata_ok,'model_metadata_errors':new['model_errors'],'scene_mask_matches_baseline':mask_ok,'policy_bytes_and_seed26_host_match':policy_ok,'observer_errors':new['errors'],'duplicates':new['duplicates'],'reference_controls_exact':all(c['exact'] for c in controls) and all(r['exact_required_prefix'] for r in rows[:2]),'gain_pair_pass':{str(g):all(r['preliminary_engineering_pass'] for r in rows[i:i+2]) and all(c['exact'] for c in controls) and all(r['exact_required_prefix'] for r in rows[:2]) for g,i in ((0.10,2),(0.20,4))},'scope':'Used-seed obstacle-free 36-second position-hold development screen; not flight qualification or freeze. Speed caps are current model engineering limits, not biological data.'}
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print('lane',r['lane'],'end',r['end'],'prefix',r['exact_required_prefix'],'dy',r['max_abs_y_displacement_m'],'vy',r['max_abs_vy_m_s'],'dx',r['forward_displacement_m'],'first_ground',r['first_ground_record'],'body_missing',len(r['missing_body_velocity_ticks']),'feedback',r['feedback_max_error_rad'],'waveform',r['waveform_max_error'],'pass',r['preliminary_engineering_pass'])
print('controls_exact',all(c['exact'] for c in controls),'metadata_valid',metadata_ok,'reference_controls_exact',out['reference_controls_exact'],'gain_pair_pass',out['gain_pair_pass'])
