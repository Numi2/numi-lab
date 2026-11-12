from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-sweep-pitch-feedback-screen-20261112')
B=Path('artifacts/crow-v15-differential-wingbeat-screen-20261112')
def load(root,name):
 result=json.loads((root/(name+'.stdout')).read_text())['first_episode_navigation'];ends=result['steps_by_environment'];samples={};contacts=collections.Counter();errors=[];nonfoot=collections.Counter()
 with gzip.open(root/(name+'.stderr.gz'),'rt') as f:
  for line in f:
   if 'CROW_SUBSTEP_ERROR' in line:errors.append(line.strip())
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if t<=ends[e]:samples[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=ends[e]:
     contacts[e,t,k]+=1
     if int(z[4])>int(z[5]):errors.append('overflow')
     if t>=450 and abs(float(z[10]))>1e-7:nonfoot[e]+=1
 return result,samples,contacts,errors,nonfoot
base=load(B,'baseline');cand=load(A,'feedback')
bad=[];count=0
for e in range(6):
 for t in sorted(t for i,t in base[1] if i==e and t<450):
  count+=1;b=base[1][e,t];c=cand[1].get((e,t))
  if c is None or any(b[k]!=c[k] for k in ('q','v','policy_action')):bad.append([e,t])
def attitude(q):
 x,y,z,w=q[3:7]
 return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
rows=[]
for e in range(6):
 end=cand[0]['steps_by_environment'][e]
 missing=sum(cand[2][e,t,k]!=1 for t in range(1,end+1) for k in range(16))
 times=sorted(t for i,t in cand[1] if i==e and t>=500 and (e,t) in base[1])
 obs=[]
 for t in times:
  c=cand[1][e,t];b=base[1][e,t];ca=attitude(c['q']);ba=attitude(b['q'])
  obs.append({'step':t,'candidate_height_m':c['q'][2],'control_height_m':b['q'][2],'candidate_roll_pitch_yaw_rad':ca,'control_roll_pitch_yaw_rad':ba,'candidate_vertical_velocity_m_s':c['v'][2],'control_vertical_velocity_m_s':b['v'][2],'candidate_speed_m_s':math.sqrt(sum(z*z for z in c['v'][:3])),'control_speed_m_s':math.sqrt(sum(z*z for z in b['v'][:3])),'candidate_heading_error_rad':abs(math.atan2(math.sin((math.pi/2 if e%2==0 else -math.pi/2)-ca[2]),math.cos((math.pi/2 if e%2==0 else -math.pi/2)-ca[2]))),'candidate_q7_to_q13':c['q'][7:14],'control_q7_to_q13':b['q'][7:14]})
 rows.append({'lane':e,'pitch_gain':[.20,.20,.40,.40,.80,.80][e],'residual_limit':.12,'first_episode_end':end,'physical_failure':cand[0]['physical_failure_by_environment'][e],'termination_reason':cand[0]['termination_reason_by_environment'][e],'missing_or_duplicate_contact_substeps':missing,'nonfoot_ground_substeps_from_intervention':cand[4][e],'paired_samples':obs})
for row in rows:
 obs=row['paired_samples']
 row['sampled_stability_pass']=(row['first_episode_end']>=1600 and not row['physical_failure'] and row['nonfoot_ground_substeps_from_intervention']==0 and len(obs)==12 and all(.65<=x['candidate_height_m']<=1.15 and abs(x['candidate_roll_pitch_yaw_rad'][1])<.35 for x in obs))
pairs=[{'pitch_gain':g,'lanes':[i,i+1],'pass':rows[i]['sampled_stability_pass'] and rows[i+1]['sampled_stability_pass']} for i,g in ((0,.2),(2,.4),(4,.8))]
out={'prefix':{'expected_samples':48,'compared':count,'mismatches':bad,'pass':count==48 and not bad},'rows':rows,'gain_pairs':pairs,'observer_errors':cand[3], 'scope':'One development seed, two different turn requests, sparse endpoint states; no direct body yaw/pitch moment after 9 s. Applied force and biological validity not measured.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('prefix',out['prefix'],'errors',len(cand[3]))
for x in rows:
 print(x['lane'],x['pitch_gain'],x['residual_limit'],'end',x['first_episode_end'],'fail',x['physical_failure'],'missing',x['missing_or_duplicate_contact_substeps'],'nonfoot contacts',x['nonfoot_ground_substeps_from_intervention'],'first',x['paired_samples'][0] if x['paired_samples'] else None)
if not out['prefix']['pass'] or any(x['missing_or_duplicate_contact_substeps'] for x in rows) or cand[3]:raise SystemExit('replay validity failed')
