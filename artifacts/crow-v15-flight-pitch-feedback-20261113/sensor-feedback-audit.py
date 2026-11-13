from pathlib import Path
import gzip,json,math,collections
A=Path('artifacts/crow-v15-flight-pitch-feedback-20261113');O=Path('artifacts/crow-v15-overlap-obstacle-free-20261113');P=A/'push-wing-20261126-rebound-pitch';Q=O/'push-wing-20261126-rebound-clear';meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];oldmeta=json.loads((Q/'run.stdout').read_text())['first_episode_navigation'];data=[]
for path in (Q/'run.stderr.gz',P/'run.stderr.gz'):
 d={'state':{},'sensor':{},'target':{},'mask':[]};data.append(d)
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d['state'][x['lane'],x['step']]=x;continue
   z=line.split()
   if not z:continue
   if z[0]=='CROW_FEEDBACK_SENSOR':t,e=map(int,z[1:3]);d['sensor'][e,t]=list(map(float,z[3:]));continue
   if z[0]=='CROW_DRIVE_TARGET':t,e=map(int,z[1:3]);d['target'][e,t]=list(map(float,z[3:]));continue
   if z[0]=='CROW_SCENE_MASK':d['mask'].append({'name':z[1],'old':int(z[2]),'new':int(z[3])})
old,new=data;expected={'crow_course_gate_left/collision','crow_course_gate_right/collision','crow_course_slalom_a/collision','crow_course_slalom_b/collision','crow_course_perch/collision'};course=[x for x in new['mask'] if x['name'].startswith('crow_course_')];ground=[x for x in new['mask'] if x['name']=='locomotion_ground/collision'];maskvalid=set(x['name'] for x in course)==expected and len(course)==5 and all(x['old']!=0 and x['new']==0 for x in course) and len(ground)==1 and ground[0]['old']==ground[0]['new'] and ground[0]['new']!=0 and all(x['old']==x['new'] for x in new['mask'] if x not in course)
a=json.loads((P/'inputs.json').read_text());b=json.loads((Q/'inputs.json').read_text());policyvalid=all(a[k]==b[k] for k in ('landing_policy_sha256','route_policy_sha256'));rows=[]
for e in range(2,8):
 end=min(900,meta['steps_by_environment'][e]);common_end=min(end,oldmeta['steps_by_environment'][e]);first=next((t for t in range(335,common_end+1) if (e,t) in old['target'] and (e,t) in new['target'] and old['target'][e,t][6:12]!=new['target'][e,t][6:12]),None);missing=[];diff={k:[] for k in ('q','v','policy_action','contacts')}
 if first is not None:
  for t in range(1,first):
   if (e,t) not in old['state'] or (e,t) not in new['state']:missing.append(t);continue
   for k in diff:
    if old['state'][e,t][k]!=new['state'][e,t][k]:diff[k].append(t)
 else:missing.append('no feedback target divergence')
 errors=[];sensor_missing=[]
 for t in range(335,end+1):
  if (e,t) not in new['state'] or (e,t) not in new['sensor']:sensor_missing.append(t);continue
  x=new['state'][e,t];q=x['q'];v=x['v'];xx,yy,zz,w=q[3:7];yawx=1.-2.*(yy*yy+zz*zz);yawy=2.*(w*zz+xx*yy);norm=math.hypot(yawx,yawy);speed=(yawx*v[0]+yawy*v[1])/norm;expectedv=(speed,q[2],v[2]);sensor=new['sensor'][e,t];errors.append([abs(sensor[i]-expectedv[i]) for i in range(3)])
 maxima=[max(x[i] for x in errors) for i in range(3)] if errors else None; sensorsvalid=not sensor_missing and maxima is not None and all(v<=1e-4 for v in maxima);parity=first is not None and not missing and not any(diff.values());rows.append({'lane':e,'first_wing_target_difference':first,'reference_parity_before_first_change':parity,'missing_reference_ticks':missing,'reference_mismatches':diff,'sensor_missing_ticks':sensor_missing,'sensor_max_abs_error_yaw_speed_height_vz':maxima,'sensor_values_valid':sensorsvalid})
main=json.loads((A/'analysis.json').read_text());pairs={str(e)+'-'+str(e+1):bool(maskvalid and policyvalid and main['paired_long_pass'][str(e)+'-'+str(e+1)] and all(next(x for x in rows if x['lane']==k)['sensor_values_valid'] and next(x for x in rows if x['lane']==k)['reference_parity_before_first_change'] for k in (e,e+1))) for e in (2,4,6)};out={'mask_valid':maskvalid,'policy_pack_bytes_identical':policyvalid,'rows':rows,'paired_sensor_validated_screen_pass':pairs,'scope':'Task-state samples are compared to accepted root state at the same logged step. The subsequent action kernel consumes the persisted state; target and state parity enforce intervention timing.'};(A/'sensor-feedback-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
