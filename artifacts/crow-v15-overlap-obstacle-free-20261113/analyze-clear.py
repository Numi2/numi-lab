from pathlib import Path
import gzip,json,math,collections,hashlib
A=Path('artifacts/crow-v15-overlap-obstacle-free-20261113');O=Path('artifacts/crow-v15-overlap-leg-wing-20261113');P=A/'push-wing-20261126-rebound-clear';meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation']; assert json.loads((P/'run-result.json').read_text())['exit_code']==0
state={};contacts=collections.defaultdict(lambda:[0.,0.,0]);toes=collections.Counter();pair={};ground={};duplicates=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];
   if key in state:duplicates.append(['state',key])
   state[key]=x;continue
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);key=e,t
   if z[0]=='CROW_TWO_TOE_SUBSTEP':contacts[key][0]+=sum(map(float,z[8:10]));contacts[key][1]+=float(z[10]);contacts[key][2]+=1
   elif max(abs(float(v)) for v in z[4:])>=11.99:toes[key]+=1
  elif z[0]=='CROW_WING_TARGET_PAIR':
   t,e=map(int,z[1:3]);key=e,t
   if key in pair:duplicates.append(['target_pair',key])
   pair[key]=list(map(float,z[3:]))
  elif z[0]=='CROW_GROUND_CONTACT':
   t,k,e=map(int,z[1:4]);ground.setdefault((e,t),{'tick':t,'substep':k,'body_a':int(z[4]),'body_b':int(z[5]),'pair_key':z[6],'feature_key':z[7],'record':z[8:]})
old={}
with gzip.open(O/'push-wing-20261126-rebound-overlap/run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if x['lane']>=2 and x['step']<=500:old[x['lane'],x['step']]=x
rows=[]
for e in range(2,8):
 end=min(900,meta['steps_by_environment'][e]); first_release=next((t for t in range(351,max(351,end-1)) if all(contacts[e,u][2]==16 and contacts[e,u][0]<=1e-7 for u in range(t,t+3))),None); first_recontact=next((t for t in range(first_release+3,end+1) if contacts[e,t][0]>1e-7),None) if first_release else None
 missing_state=[t for t in range(1,end+1) if (e,t) not in state];missing_contact=[t for t in range(335,end+1) if contacts[e,t][2]!=16];missing_pair=[t for t in range(335,end+1) if (e,t) not in pair];target_bad=[t for t in range(336,end+1) if (e,t) in pair and (e,t-1) in pair and (pair[e,t][1]!=pair[e,t-1][0] or pair[e,t][3]!=pair[e,t-1][2])]
 # Toe coverage is checked independently by recounting the exact records below.
 missing_reference=[t for t in range(1,501) if (e,t) not in old or (e,t) not in state];different_reference=[t for t in range(1,501) if (e,t) in old and (e,t) in state and any(old[e,t][k]!=state[e,t][k] for k in ('q','v','policy_action','contacts'))]
 toe=sum(toes[e,t] for t in range(348,end+1));nonfoot=sum(contacts[e,t][1] for t in range(first_release,end+1)) if first_release else None
 samples=[state[e,t] for t in range(500,end+1) if (e,t) in state];late=[state[e,t]['v'][0] for t in range(550,end+1) if (e,t) in state];sweep=[state[e,t]['q'][j+1] for t in range(348,end+1) if (e,t) in state for j in (6,7)];dx=state[e,900]['q'][0]-state[e,500]['q'][0] if end==900 and (e,900) in state and (e,500) in state else None
 def attitude(x):
  q=x['q'];xx,yy,zz,w=q[3:7];return (math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy)),math.asin(max(-1.,min(1.,2*(w*yy-zz*xx)))))
 rp=[attitude(x) for x in samples];angular=[math.sqrt(sum(v*v for v in x['v'][3:6])) for x in samples];points=[t for t in range((first_release or 900)+1,end+1) if (e,t) in ground];first_ground=ground[e,points[0]] if points else None
 before=state.get((e,points[0]-1)) if points else None
 row={'lane':e,'first_episode_steps':end,'termination_reason':meta['termination_reason_by_environment'][e],'physical_failure':meta['physical_failure_by_environment'][e],'release':first_release,'recontact':first_recontact,'first_recorded_ground_pair_after_release':first_ground,'accepted_state_before_first_recorded_ground_pair':before,'missing_state_ticks':missing_state,'missing_contact_ticks':missing_contact,'missing_wing_target_ticks':missing_pair,'wing_target_history_mismatches':target_bad,'reference_missing_ticks':missing_reference,'reference_different_ticks':different_reference,'toe_threshold_substeps':toe,'nonfoot_normal_impulse_after_release_ns':nonfoot,'dx_500_to_900_m':dx,'height_500_to_end_range_m':[min(x['q'][2] for x in samples),max(x['q'][2] for x in samples)] if samples else None,'vx_550_to_end_range_m_s':[min(late),max(late)] if late else None,'roll_range_rad':[min(x[0] for x in rp),max(x[0] for x in rp)] if rp else None,'pitch_range_rad':[min(x[1] for x in rp),max(x[1] for x in rp)] if rp else None,'max_angular_speed_rad_s':max(angular) if angular else None,'sweep_hit_modeled_stop':bool(sweep and max(map(abs,sweep))>=.19999)}
 rows.append(row)
# Count toe records separately, including below-threshold samples; absence must not look like zero toe speed.
toecover=collections.Counter()
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_TOE_V_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);toecover[e,t]+=1
for x in rows:
 e=x['lane'];end=x['first_episode_steps'];missing=[t for t in range(335,end+1) if toecover[e,t]!=16];x['missing_toe_ticks']=missing
 x['exploratory_long_screen_pass']=bool(not duplicates and end==900 and not x['missing_state_ticks'] and not x['missing_contact_ticks'] and not missing and not x['missing_wing_target_ticks'] and not x['wing_target_history_mismatches'] and not x['reference_missing_ticks'] and not x['reference_different_ticks'] and x['release'] is not None and x['recontact'] is None and x['nonfoot_normal_impulse_after_release_ns'] is not None and x['nonfoot_normal_impulse_after_release_ns']<=1e-7 and x['toe_threshold_substeps']<=1 and not x['sweep_hit_modeled_stop'] and x['dx_500_to_900_m'] is not None and x['dx_500_to_900_m']>.5 and x['height_500_to_end_range_m'] is not None and x['height_500_to_end_range_m'][0]>=.25 and x['height_500_to_end_range_m'][1]<=1.5 and x['vx_550_to_end_range_m_s'] is not None and x['vx_550_to_end_range_m_s'][0]>=0 and x['vx_550_to_end_range_m_s'][1]<=1)
out={'duplicates':duplicates,'rows':rows,'paired_long_pass':{str(e)+'-'+str(e+1):all(next(x for x in rows if x['lane']==k)['exploratory_long_screen_pass'] for k in (e,e+1)) for e in (2,4,6)},'source_log_sha256':hashlib.sha256((P/'run.stderr.gz').read_bytes()).hexdigest(),'scope':'Obstacle-free mask diagnostic on a used development layout/seed, one unqualified open-loop continuation. Aero and individual wing effort after tick 500 were not observed; do not infer them.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print('lane',r['lane'],'end',r['first_episode_steps'],'reason',r['termination_reason'],'release',r['release'],'recontact',r['recontact'],'dx',r['dx_500_to_900_m'],'height',r['height_500_to_end_range_m'],'vx',r['vx_550_to_end_range_m_s'],'toe',r['toe_threshold_substeps'],'reference_diff',len(r['reference_different_ticks']),'pass',r['exploratory_long_screen_pass'])
