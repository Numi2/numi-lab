from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-differential-wingbeat-screen-20261112');names=('baseline','feedback');states={};results={}
for arm in names:
 D=A if arm=='baseline' else Path('artifacts/crow-v15-stronger-bank-feedback-screen-20261112'); result=json.loads((D/(arm+'.stdout')).read_text())['first_episode_navigation'];results[arm]=result
 with gzip.open(D/(arm+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step']
    if t<=result['steps_by_environment'][e]:states[arm,e,t]=x
compare=[];rows=[]
for arm in names[1:]:
 pre=[];matched=0;expected=0
 for e in range(6):
  bt={t for a,i,t in states if a=='baseline' and i==e and t<450};ct={t for a,i,t in states if a==arm and i==e and t<450};expected+=len(bt);matched+=len(bt&ct)
  for t in sorted(bt&ct):
   b,c=states['baseline',e,t],states[arm,e,t]
   if any(b[k]!=c[k] for k in ('q','v','policy_action')):pre.append([e,t])
  times=sorted({t for a,i,t in states if a=='baseline' and i==e and t>=500}&{t for a,i,t in states if a==arm and i==e and t>=500})
  values=[];target=math.pi/2 if e%2==0 else -math.pi/2
  for t in times:
   q=states[arm,e,t]['q'];b=states['baseline',e,t]['q'];x,y,z,w=q[3:7];bx,by,bz,bw=b[3:7];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));byaw=math.atan2(2*(bw*bz+bx*by),1-2*(by*by+bz*bz));roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));heading_err=abs(math.atan2(math.sin(target-yaw),math.cos(target-yaw)));base_err=abs(math.atan2(math.sin(target-byaw),math.cos(target-byaw)))
   values.append({'step':t,'yaw_rad':yaw,'roll_rad':roll,'height_m':q[2],'heading_error_rad':heading_err,'baseline_error_rad':base_err,'flap_q_delta_rad':[q[9]-b[9],q[10]-b[10]]})
  rows.append({'arm':arm,'lane':e,'bank_limit':[.12,.18,.24][e//2],'target_rad':target,'end':results[arm]['steps_by_environment'][e],'physical_failure':results[arm]['physical_failure_by_environment'][e],'termination_reason':results[arm]['termination_reason_by_environment'][e],'paired_samples':values})
 compare.append({'arm':arm,'expected_samples':expected,'matched_samples':matched,'mismatches':pre,'pass':expected==48 and matched==expected and not pre})
out={'prefix':compare,'rows':rows,'limits':'Sparse endpoint samples from one used seed per mode. No turn-direction inference from an absolute error alone; post-failure states excluded. Prior direct body moments are suppressed only after 9 s.'};(Path('artifacts/crow-v15-stronger-bank-feedback-screen-20261112')/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print('prefix',json.dumps(compare))
for r in rows:
 s=r['paired_samples'];last=s[-1] if s else None
 print(r['arm'],r['lane'],'gain',r['bank_limit'],'end',r['end'],'fail',r['physical_failure'],'last',json.dumps(last))
