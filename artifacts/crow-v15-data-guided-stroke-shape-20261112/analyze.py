from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-data-guided-stroke-shape-20261112')
def load(path,name):
 result=json.loads((path/(name+'.stdout')).read_text())['first_episode_navigation'];ends=result['steps_by_environment'];samples={};contacts=collections.Counter();errors=[];nonfoot=collections.Counter();first={}
 with gzip.open(path/(name+'.stderr.gz'),'rt') as f:
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
     if t>=450 and abs(float(z[10]))>1e-7:nonfoot[e]+=1;first.setdefault(e,[t,k])
 return result,samples,contacts,errors,nonfoot,first
def measure(x,target):
 q=x['q'];v=x['v'];a,b,c,d=q[3:7]
 pitch=math.asin(max(-1,min(1,2*(d*b-c*a))))
 yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
 forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]
 return {'height_m':q[2],'pitch_rad':pitch,'yaw_rad':yaw,'forward_m_s':forward,'lateral_m_s':lateral,'sideslip_rad':math.atan2(lateral,forward),'heading_error_rad':abs(math.atan2(math.sin(target-yaw),math.cos(target-yaw)))}
base=load(A,'sham');cand=load(A,'warped');prefix=[];count=0
for e in range(6):
 for t in sorted(t for i,t in base[1] if i==e and t<450):
  count+=1;b=base[1][e,t];c=cand[1].get((e,t))
  if c is None or any(b[k]!=c[k] for k in ('q','v','policy_action')):prefix.append([e,t])
rows=[]
for e in range(6):
 end=cand[0]['steps_by_environment'][e];missing=sum(cand[2][e,t,k]!=1 for t in range(1,end+1) for k in range(16));target=math.pi/2 if e%2==0 else -math.pi/2
 samples=[]
 for t in sorted(t for i,t in cand[1] if i==e and 500<=t<=1600):
  s={'step':t,'candidate':measure(cand[1][e,t],target)};b=base[1].get((e,t));s['reference']=measure(b,target) if b else None;samples.append(s)
 valid=len(samples)==12
 passed=(end>=1600 and not cand[0]['physical_failure_by_environment'][e] and not cand[4][e] and valid and all(.65<=s['candidate']['height_m']<=1.15 and abs(s['candidate']['pitch_rad'])<.35 for s in samples) and all(s['candidate']['forward_m_s']>=.5 and abs(s['candidate']['sideslip_rad'])<=math.pi/4 for s in samples if s['step']>=800) and samples[-1]['candidate']['heading_error_rad']<=math.pi/12)
 rows.append({'lane':e,'bank_limit_rad':[.12,.18,.24][e//2],'pitch_gain':[.2,.4,.8][e//2],'forward_speed_target_m_s':.75,'end_step':end,'physical_failure':cand[0]['physical_failure_by_environment'][e],'reference_end_step':base[0]['steps_by_environment'][e],'missing_substeps':missing,'post_intervention_nonfoot_substeps':cand[4][e],'first_nonfoot':cand[5].get(e),'sampled_pass':passed,'samples':samples})
out={'prefix':{'count':count,'expected':48,'mismatches':prefix,'pass':count==48 and not prefix},'errors':cand[3],'rows':rows,'paired_passes':[{'lanes':[i,i+1],'pass':rows[i]['sampled_pass'] and rows[i+1]['sampled_pass']} for i in (0,2,4)],'scope':'Single used development seed; sparse 100-interval samples, not biological or controller qualification; direct body moments remain before 9 seconds.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2));print([(x['lane'],x['end_step'],x['physical_failure'],x['first_nonfoot'],x['sampled_pass']) for x in rows])
if not out['prefix']['pass'] or cand[3] or base[3] or any(x['missing_substeps'] for x in rows):raise SystemExit('trace or prefix validity failed')
