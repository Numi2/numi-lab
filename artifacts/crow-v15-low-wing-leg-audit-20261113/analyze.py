from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts')
A=R/'crow-v15-low-wing-leg-audit-20261113';A.mkdir(exist_ok=True)
S=R/'crow-v15-low-wing-effort-screen-20261113'
allrows=[]
for arm in ('push','hold'):
 path=S/(arm+'-wing-20261126-rebound-low')/'run.stderr.gz'
 D={}; targets={}; errors=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_LEG_EFFORT':
    t,k,e=map(int,z[1:4]);mode=tuple(map(int,z[4:7]));v=list(map(float,z[7:]));assert len(v)==30 and mode==(1,0,0) and (e,t,k) not in D
    D[e,t,k]=[v[i:i+5] for i in range(0,30,5)]
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);v=list(map(float,z[3:]));targets[e,t]=v
   elif z[0]=='CROW_SUBSTEP_ERROR':errors.append(line.strip())
 assert not errors and len(D)==8*66*16
 for e in range(2,8):
  for start,end in ((348,355),(348,357),(348,365)):
   rows=[]
   for j in range(6):
    samples=[D[e,t,k][j] for t in range(start,end+1) for k in range(16)]
    works=[tau*(q1-q0) for tau,q0,q1,v0,v1 in samples]
    rows.append({'joint_v_index':13+j,'q_start':samples[0][1],'q_end':samples[-1][2],'target_start':targets[e,start][13+j],'target_end':targets[e,end][13+j],'net_work_proxy':sum(works),'positive_work_proxy':sum(max(x,0.) for x in works),'max_abs_effort':max(abs(x[0]) for x in samples)})
   allrows.append({'arm':arm,'lane':e,'ticks':[start,end],'joints':rows,'net_work_proxy':sum(x['net_work_proxy'] for x in rows),'positive_work_proxy':sum(x['positive_work_proxy'] for x in rows)})
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc per-joint generalized working effort times accepted coordinate displacement; not isolated muscle work. Matched low-effort run, used development seed, one correlated cohort. No qualification.','rows':allrows},indent=2)+'\n')
for r in allrows:
 if r['ticks']==[348,357]:
  print(r['arm'],r['lane'],'net',round(r['net_work_proxy'],4),'qstart',[round(j['q_start'],3) for j in r['joints'][:3]],'qend',[round(j['q_end'],3) for j in r['joints'][:3]],'targets end',[round(j['target_end'],3) for j in r['joints'][:3]])
