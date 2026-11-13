from pathlib import Path
import collections,gzip,json,math,statistics,sys
A=Path('artifacts/crow-v15-bank-damping-sensitivity-20261113');P=A/sys.argv[1];R=Path('artifacts/crow-v15-mirrored-amplitude-screen-20261113/pronation-small-20261126-rebound-small');lanes=(2,3,4,5,6,7);substeps=range(16);ticks=range(650,1151);kinds={'CROW_AERO_INPUT','CROW_KIN_STATE','CROW_AERO_WRENCH','CROW_AERO_FORCE','CROW_PRONATION_EFFORT'}
def read(p, detailed):
 state={};tables={k:{} for k in kinds};model={};mask=[];errors=[];dup=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step']
    if key in state:dup.append(['state',key])
    state[key]=x;continue
   z=line.split()
   if not z:continue
   typ=z[0]
   if typ=='CROW_SCENE_MASK':mask.append((z[1],int(z[2]),int(z[3])))
   if 'OBSERVER_ERROR' in typ or typ=='CROW_SUBSTEP_ERROR':errors.append(line.strip())
   if typ=='CROW_PRONATION_MODEL':
    i=int(z[1]);item=[int(z[2])]+list(map(float,z[3:]));
    if i in model and model[i]!=item:errors.append('conflicting model '+str(i))
    model[i]=item
   if detailed and typ in kinds:
    t,k,e=map(int,z[1:4]);
    if e not in lanes or t not in ticks:continue
    key=(e,t,k)
    if typ=='CROW_AERO_WRENCH':key+=(int(z[4]),);vals=list(map(float,z[5:]))
    elif typ=='CROW_AERO_FORCE':vals=list(map(float,z[5:]))
    elif typ=='CROW_KIN_STATE':
     if z[4:6]!=['41','40']:errors.append('kin stride mismatch')
     vals=list(map(float,z[6:]))
    else:vals=list(map(float,z[4:]))
    if key in tables[typ]:dup.append([typ,key])
    if not all(math.isfinite(v) for v in vals):errors.append('nonfinite '+typ)
    tables[typ][key]=vals
 return dict(state=state,tables=tables,model=model,mask=mask,errors=errors,dup=dup)
old=read(R,False);new=read(P,True);im=json.loads((P/'inputs.json').read_text());ir=json.loads((R/'inputs.json').read_text());inputs=all(im[k]==ir[k] for k in ('landing_policy_sha256','route_policy_sha256'));mask=new['mask']==old['mask'];meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ometa=json.loads((R/'run.stdout').read_text())['first_episode_navigation'];model=new['model'];expected=[45.,.45,.00002,0.,-.35,.35,25.,.15];model_ok=all(i in model and model[i][0]&8 and model[i][0]&16 and len(model[i])==9 and all(abs(x-y)<1e-5 for x,y in zip(model[i][1:],expected)) for i in (10,11));parity=[]
for e in range(8):
 stop=ometa['steps_by_environment'][e] if e<6 else 600;missing=[];different=[]
 for t in range(1,stop+1):
  a=old['state'].get((e,t));b=new['state'].get((e,t))
  if a is None or b is None:missing.append(t)
  elif any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):different.append(t)
 parity.append(dict(lane=e,stop=stop,end=meta['steps_by_environment'][e],missing=missing,different=different,exact=(e>=6 or meta['steps_by_environment'][e]==stop) and not missing and not different))
T=new['tables'];rows=[]
for e in lanes:
 keys=[(e,t,k) for t in ticks for k in substeps];counts={typ:0 for typ in kinds};timing=0.;force_residual=0.;invalid=0;eff=[[],[]];tracking=[[],[]];pos=[[],[]];speed=[[],[]];force={5:[[],[],[]],6:[[],[],[]]};wing_yaw={5:[],6:[]};root_yaw=[]
 for key in keys:
  for typ in kinds:
   if typ=='CROW_AERO_WRENCH':counts[typ]+=sum(key+(b,) in T[typ] for b in (0,5,6,7))
   else:counts[typ]+=key in T[typ]
  a=T['CROW_AERO_INPUT'].get(key);k=T['CROW_KIN_STATE'].get(key);f=T['CROW_AERO_FORCE'].get(key);p=T['CROW_PRONATION_EFFORT'].get(key);w=[T['CROW_AERO_WRENCH'].get(key+(b,)) for b in (0,5,6,7)]
  if not all(x is not None for x in (a,k,f,p,*w)):continue
  if len(a)!=81 or len(k)!=81 or len(f)!=12 or len(p)!=14 or any(len(x)!=6 for x in w) or f[-1]!=4:invalid+=1;continue
  timing=max(timing,max(abs(x-y) for x,y in zip(a,k)));force_residual=max(force_residual,max(abs(sum(x[j] for x in w)-f[j]) for j in range(3)))
  for j in range(2):
   v=p[j*7:(j+1)*7];eff[j].append(v[0]);tracking[j].append(v[3]-v[1]);pos[j].append(v[3]);speed[j].append(v[6])
  for b,wi in ((5,w[1]),(6,w[2])):
   for j in range(3):force[b][j].append(wi[j])
   wing_yaw[b].append(wi[5])
  root_yaw.append(w[0][5])
 req=len(keys);complete=all(counts[t]==(req*4 if t=='CROW_AERO_WRENCH' else req) for t in kinds);calc=lambda v:dict(min=min(v),max=max(v),mean=statistics.mean(v)) if v else None
 joint=[]
 for j in range(2):
  v=eff[j];lim=model.get(10+j,[0,0,0,0,0,0,0,0,0])[-1];joint.append(dict(index=10+j,samples=len(v),effort=calc(v),at_effort_limit_count=sum(abs(x)>=lim-1e-5 for x in v) if lim else None,over_effort_limit_count=sum(abs(x)>lim+1e-5 for x in v) if lim else None,position=calc(pos[j]),velocity=calc(speed[j]),mean_abs_end_position_minus_target=statistics.mean(abs(x) for x in tracking[j]) if tracking[j] else None,max_abs_end_position_minus_target=max(map(abs,tracking[j])) if tracking[j] else None))
 rows.append(dict(lane=e,counts=counts,complete=complete,invalid_records=invalid,kernel_input_vs_substep_start_max_abs=timing,max_unaccounted_world_force_n=force_residual,pronation=joint,wing_force_n={str(b):[calc(v) for v in force[b]] for b in (5,6)},wing_com_world_z_torque_nm={str(b):calc(wing_yaw[b]) for b in (5,6)},root_world_z_torque_nm=calc(root_yaw)))
valid=inputs and mask and model_ok and not new['errors'] and not new['dup'] and all(x['exact'] for x in parity) and all(r['complete'] and r['invalid_records']==0 and r['kernel_input_vs_substep_start_max_abs']<1e-7 and r['max_unaccounted_world_force_n']<1e-7 for r in rows)
out=dict(valid=valid,policy_bytes_match=inputs,scene_mask_matches=mask,model_metadata=model,model_matches_expected=model_ok,errors=new['errors'],duplicate_count=len(new['dup']),parity=parity,rows=rows,scope='Modeled world-frame aerodynamic increments at each body COM; wing-COM torque is not net bird yaw torque. Control effort is modeled actuator output, not biological muscle work. Correlated lanes and a used seed.')
(P/'force-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('valid',valid,'model',model_ok,'parity',all(x['exact'] for x in parity),'errors',len(new['errors']),'duplicates',len(new['dup']))
for r in rows:print('lane',r['lane'],'coverage',r['complete'],'timing',r['kernel_input_vs_substep_start_max_abs'],'force residual',r['max_unaccounted_world_force_n'],'effort',[(j['effort'],j['at_effort_limit_count'],j['over_effort_limit_count']) for j in r['pronation']],'wing COM yaw',r['wing_com_world_z_torque_nm'])
