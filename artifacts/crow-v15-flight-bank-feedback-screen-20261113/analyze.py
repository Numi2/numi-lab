from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-flight-bank-feedback-screen-20261113')
R=Path('artifacts/crow-v15-uniform-flight-24s-screen-20261113/push-wing-012-20261126-rebound-uniform')
P=A/sys.argv[1]
def read(p):
 d={'state':{},'contact':collections.defaultdict(lambda:[0.,0.,0]),'toe':collections.Counter(),'toe_count':collections.Counter(),'effort':{},'target':{},'ground':{},'masks':[],'errors':[],'duplicates':[]}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]); key=x['lane'],x['step']
    if key in d['state']:d['duplicates'].append(['state',key])
    d['state'][key]=x;continue
   z=line.split()
   if not z:continue
   if z[0]=='CROW_SCENE_MASK': d['masks'].append((z[1],int(z[2]),int(z[3])))
   if 'OBSERVER_ERROR' in z[0]:d['errors'].append(line.strip())
   if z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);d['contact'][e,t][0]+=sum(map(float,z[8:10]));d['contact'][e,t][1]+=float(z[10]);d['contact'][e,t][2]+=1
   elif z[0]=='CROW_TOE_V_SUBSTEP':
    t,k,e=map(int,z[1:4]);d['toe_count'][e,t]+=1;d['toe'][e,t]+=int(max(abs(float(v)) for v in z[4:])>=11.99)
   elif z[0]=='CROW_WING_EFFORT':
    t,k,e=map(int,z[1:4]);d['effort'][e,t,k]=list(map(float,z[4:]))
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);d['target'][e,t]=list(map(float,z[3:]))
   elif z[0]=='CROW_GROUND_CONTACT':
    t,k,e=map(int,z[1:4]);d['ground'].setdefault((e,t),line.strip())
 return d
old=read(R);new=read(P);meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation']
def angles(x):
 xx,yy,zz,w=x['q'][3:7]
 return (math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy)),math.asin(max(-1,min(1,2*(w*yy-zz*xx)))),math.atan2(2*(w*zz+xx*yy),1-2*(yy*yy+zz*zz)))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def mean(xs):return sum(xs)/len(xs) if xs else None
rows=[]
for e in range(2,8):
 end=min(meta['steps_by_environment'][e],1200); required=1200 if e<4 else 600;missing=[];diff=[]
 for t in range(1,required+1):
  if (e,t) not in old['state'] or (e,t) not in new['state']:missing.append(t)
  elif any(old['state'][e,t][k]!=new['state'][e,t][k] for k in ('q','v','policy_action','contacts')):diff.append(t)
 release=next((t for t in range(351,max(351,end-1)) if all(new['contact'][e,u][2]==16 and new['contact'][e,u][0]<=1e-7 for u in range(t,t+3))),None)
 recontact=next((t for t in range(release+3,end+1) if new['contact'][e,t][0]>1e-7),None) if release else None
 windows={}
 for name,lo,hi in [('pulse',800,850),('recovery',1100,1150)]:
  ticks=[t for t in range(lo,hi+1) if t<=end and (e,t) in new['state'] and (e,t) in old['state']]
  delta=[tuple(wrap(a-b) for a,b in zip(angles(new['state'][e,t]),angles(old['state'][e,t]))) for t in ticks]
  windows[name]={'samples':len(ticks),'mean_delta_roll_pitch_yaw_rad':[mean([v[i] for v in delta]) for i in range(3)],'mean_delta_height_m':mean([new['state'][e,t]['q'][2]-old['state'][e,t]['q'][2] for t in ticks]),'mean_delta_velocity_m_s':[mean([new['state'][e,t]['v'][i]-old['state'][e,t]['v'][i] for t in ticks]) for i in range(3)]}
 samples=[new['state'][e,t] for t in range(348,end+1) if (e,t) in new['state']];eff=[new['effort'][e,t,k] for t in range(348,end+1) for k in range(16) if (e,t,k) in new['effort']];missing_eff=(end-347)*16-len(eff)
 badcontact=[t for t in range(335,end+1) if new['contact'][e,t][2]!=16];badtoe=[t for t in range(335,end+1) if new['toe_count'][e,t]!=16]
 ground=[(t,new['ground'][e,t]) for t in range((release or end)+1,min(1200,end+1)+1) if (e,t) in new['ground']]
 target_samples=[new['target'][e,t][6:8] for t in range(601,min(900,end)+1) if (e,t) in new['target']]
 target_ticks=[t for t in range(601,min(900,end)+1) if (e,t) in new['target'] and (e,t) in old['target']]
 flap_delta=[(new['target'][e,t][8]-old['target'][e,t][8],new['target'][e,t][9]-old['target'][e,t][9]) for t in target_ticks]
 rows.append({'lane':e,'end':end,'termination_reason':meta['termination_reason_by_environment'][e],'physical_failure':meta['physical_failure_by_environment'][e],'comparison_ticks':required,'missing_comparison_ticks':missing,'different_comparison_ticks':diff,'exact_required_parity':not missing and not diff,'release':release,'recontact':recontact,'nonfoot_normal_impulse_after_release_ns':sum(new['contact'][e,t][1] for t in range(release,end+1)) if release else None,'first_ground_record_after_release_or_termination_boundary':ground[0] if ground else None,'accepted_state_immediately_before_first_ground':new['state'].get((e,ground[0][0]-1)) if ground else None,'physical_wing_targets_before_first_ground':new['target'].get((e,ground[0][0]-1),[])[6:12] if ground else None,'missing_contact_ticks':badcontact,'missing_toe_ticks':badtoe,'toe_speed_threshold_events':sum(new['toe'][e,t] for t in range(348,end+1)),'missing_effort_substeps':missing_eff,'flap_effort_max_abs':max((max(abs(x[0]),abs(x[7])) for x in eff if len(x)==14),default=None),'sampled_sweep_q_range':[min((x['q'][j] for x in samples for j in (7,8)),default=None),max((x['q'][j] for x in samples for j in (7,8)),default=None)],'pulse_sweep_target_range':[min((v for x in target_samples for v in x),default=None),max((v for x in target_samples for v in x),default=None)],'pulse_flap_target_comparison_samples':len(target_ticks),'pulse_flap_target_delta_extrema':[[min((x[j] for x in flap_delta),default=None),max((x[j] for x in flap_delta),default=None)] for j in (0,1)],'windows':windows})
out={'rows':rows,'scene_mask_matches_baseline':new['masks']==old['masks'],'observer_errors':new['errors'],'duplicate_samples':new['duplicates'],'input_hashes':json.loads((P/'inputs.json').read_text()),'baseline_input_hashes':json.loads((R/'inputs.json').read_text()),'scope':'Used-seed bounded bank feedback identification on uncalibrated obstacle-free research body; neither command-tracking qualification nor flight freeze.'}
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print('lane',r['lane'],'end',r['end'],'parity',r['exact_required_parity'],'release/recontact',r['release'],r['recontact'],'pulse delta RPY',r['windows']['pulse']['mean_delta_roll_pitch_yaw_rad'],'recovery',r['windows']['recovery']['mean_delta_roll_pitch_yaw_rad'],'missing contacts',len(r['missing_contact_ticks']))
