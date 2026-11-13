from pathlib import Path
import collections,gzip,hashlib,json,math
A=Path('artifacts/crow-v15-wingheld-retraction-screen-20261113')
a=json.loads((A/'analysis.json').read_text()); c=json.loads((A/'com-analysis.json').read_text()); cm={(x['arm'],x['lane']):x for x in c['rows']}; allrows=[]
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-retraction/run.stderr.gz'); states={};forces={};contacts=collections.defaultdict(lambda:[0.,0.,0]);targets={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if not z or z[0] not in ('CROW_KIN_STATE','CROW_AERO_FORCE','CROW_GROUND_CONTACT','CROW_DRIVE_TARGET'):continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);
    if e>=2 and 348<=t<=380:targets[e,t]=list(map(float,z[3:]))
    continue
   t,k,e=map(int,z[1:4]);
   if e<2 or not 348<=t<=380:continue
   if z[0]=='CROW_KIN_STATE':states[e,t,k]=list(map(float,z[47:]))
   elif z[0]=='CROW_AERO_FORCE':forces[e,t,k]=float(z[7])
   else:
    b1,b2=map(int,z[4:6]);body=b1 if b2==35 else b2
    if body in list(range(19,35)):
     vals=list(map(float,z[8:]));jn,jt,jv=vals[12:15]
     contacts[e][0]+=jn;contacts[e][1]+=math.hypot(jt,jv);contacts[e][2]+=1
 for e in range(2,8):
  keys=[(e,t,k) for t in range(348,381) for k in range(16)];assert all(q in states and q in forces for q in keys)
  assert all((e,t) in targets for t in range(348,381))
  ar=a['rows'][e][arm];cr=cm[arm,e];release=ar['first_three_unloaded']; flap=max(abs(states[q][j]) for q in keys for j in (8,9));aero=sum(forces[q] for q in keys)*.00125
  row={'arm':arm,'lane':e,'first_unload_tick':release,'com_vz_at_unload_m_s':cr['com_vz_at_release_m_s'],'com_peak_rise_m':cr['com_peak_rise_from_350_m'],'root_rise_m':ar['root_rise_max_m'],'nonfoot_normal_impulse_ns_348_400':ar['nonfoot_impulse_348_400_ns'],'toe_speed_threshold_substeps_348_400':ar['toe_speed_threshold_substeps'],'peak_flap_speed_rad_s_348_380':flap,'vertical_aero_impulse_ns_348_380':aero,'foot_ground_normal_logged_ns_348_380':contacts[e][0],'foot_ground_tangent_magnitudes_sum_ns_348_380':contacts[e][1],'ground_contact_points_348_380':contacts[e][2],'leg_target_samples':{str(t):targets[e,t][13:19] for t in (350,353,355,357,360,365,370,380)}}
  row['screen_pass']=release is not None and release<380 and cr['com_vz_at_release_m_s']>0 and ar['nonfoot_impulse_348_400_ns']<=1e-7 and ar['toe_speed_threshold_substeps']==0 and flap<=1.0 and abs(aero)<=.02
  allrows.append(row)
result={'source_logs_sha256':{arm:hashlib.sha256((A/(arm+'-wing-20261126-rebound-retraction/run.stderr.gz')).read_bytes()).hexdigest() for arm in ('push','hold')},'rows':allrows,'pair_pass':{f'{i}-{i+1}':all(next(x for x in allrows if x['arm']=='push' and x['lane']==e)['screen_pass'] for e in (i,i+1)) for i in (2,4,6)},'scope':'Development seed, correlated lanes; tangential sum is magnitudes of logged impulses, not net force or measured slip. Wing holding is active; no biological or mission qualification.'}
(A/'screen.json').write_text(json.dumps(result,indent=2)+'\n')
for row in allrows: print(row['arm'],row['lane'],'release',row['first_unload_tick'],'COMvz',row['com_vz_at_unload_m_s'],'rise',round(row['com_peak_rise_m'],4),'toe',row['toe_speed_threshold_substeps_348_400'],'nonfoot',row['nonfoot_normal_impulse_ns_348_400'],'flap',round(row['peak_flap_speed_rad_s_348_380'],3),'aero',round(row['vertical_aero_impulse_ns_348_380'],5),'pass',row['screen_pass'])
