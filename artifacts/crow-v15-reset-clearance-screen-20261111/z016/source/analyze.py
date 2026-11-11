from pathlib import Path
import collections,gzip,json,subprocess
A=Path('artifacts/crow-v15-reset-clearance-screen-20261111/z016/source');B=Path('artifacts/crow-v15-reset-clearance-screen-20261111/z016');states={};initial={};obs={};reference={}
def read(p,ret,collect):
 with gzip.open(p/'stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if x['step']<=4:ret['state',x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP ') or l.startswith('CROW_TOE_V_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);
    if t<=4:ret[a[0],e,t,k]=a[4:]
   elif collect and l.startswith('CROW_QV_SOURCE '):
    a=l.split();t,k,e=map(int,a[1:4]);initial[e,t,k]=list(map(float,a[4:]))
read(A,obs,True);read(B,reference,False);differences=[k for k in set(obs)|set(reference) if obs.get(k)!=reference.get(k)];continuity=[]
for e in range(8):
 for t in (3,4):
  v=initial.get((e,t,0));previous=obs.get(('state',e,t-1));
  if v is not None and previous is not None:continuity.extend(abs(a-b) for a,b in zip(v,previous['q']+previous['v']))
path=Path('/tmp/crow-reset-source-states.txt')
with path.open('w') as f:
 for e in range(8):
  v=initial[e,1,0];f.write(f'{e} 1 '+' '.join(map(str,v))+'\n')
p=subprocess.run(['build/crow-v15-two-segment-gait-geometry-20261111/probe',str(path)],capture_output=True,text=True,check=True);rows=collections.defaultdict(list)
for line in p.stdout.splitlines():
 a=line.split();e,t=map(int,a[:2]);name=a[2];pose=list(map(float,a[3:]));
 if 'toe' not in name and not name.endswith('_foot'):continue
 half=[.01,.01725,.0092] if name.endswith('_foot') else [.009,.0025,.0092];x,y,z,w=pose[3:7];row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));gap=pose[2]-sum(abs(a)*b for a,b in zip(row,half));rows[e].append((gap,name))
r={'compared_record_counts':[len(obs),len(reference)],'differing_records':len(differences),'first_difference':str(differences[:1]),'source_samples':len(initial),'expected_source_samples':8*4*16,'previous_endpoint_compared_components':len(continuity),'previous_endpoint_max_abs_error':max(continuity) if continuity else None,'lanes':[{'lane':e,'root_position_m':initial[e,1,0][:3],'minimum_foot_toe_gap_m':min(rows[e])[0],'lowest_shape':min(rows[e])[1],'toe_v_max':max(abs(x) for x in initial[e,1,0][39+22:39+38])} for e in range(8)],'scope':'initial preflight step is duplicated; continuity compares actual successive steps 3 and 4; independent FK of unmodified reset source; negative gap denotes overlap with plane, not automatically the cause of all cap hits'}
(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
