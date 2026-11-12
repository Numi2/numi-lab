from pathlib import Path
import collections,gzip,json,math,subprocess,os
P=Path('artifacts/crow-v15-pitch-response-separated-20261111/zero');A=Path('artifacts/crow-v15-perch-negative-start-support-audit-20261112')
qv={};contacts=collections.defaultdict(lambda:[0.]*37);counts=collections.Counter()
with gzip.open(P/'trace.log.gz','rt') as f:
 for l in f:
  z=l.split()
  if z and z[0]=='step':
   e,t=int(z[1]),int(z[2])+1
   if e<2 and t<=150:
    i=z.index('q');j=z.index('v');qv[e,t]=[list(map(float,z[i+1:j])),list(map(float,z[j+1:]))]
with gzip.open(P/'substeps.log.gz','rt') as f:
 for l in f:
  z=l.split()
  if z and z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4])
   if e<2 and t<=150:
    counts[e,t,k]+=1
    for i,x in enumerate(map(float,z[6:])):contacts[e,t][i]+=x
assert len(qv)==300 and all(counts[e,t,k]==1 for e in range(2) for t in range(1,151) for k in range(32))
inp=''.join(f'{e} {t} '+' '.join(map(str,q))+'\n' for (e,t),(q,v) in sorted(qv.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path.cwd()/'build/crow-v15-hip-roll-journey-integration-20261111')}
r=subprocess.run(['build/crow-v15-perch-com-moment-audit-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
com={}
for l in r.stdout.splitlines():
 z=l.split()
 if z[0]=='C':com[int(z[1]),int(z[2])]=list(map(float,z[4:7]))
assert len(com)==len(qv)
rows=[];mass=.45006;threshold=.5*mass*9.81*.02
for e in (0,1):
 vals=[]
 for t in range(1,151):
  q,v=qv[e,t];s=contacts[e,t];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))));left_front=sum(s[21:27]);left_rear=sum(s[27:29]);right_front=sum(s[29:35]);right_rear=sum(s[35:37]);balanced=min(s[0],s[1])>.0002 and s[0]+s[1]>=threshold and left_front>=.0002 and left_rear<=-.0002 and right_front>=.0002 and right_rear<=-.0002 and abs(s[4])<=1e-7
  vals.append({'step':t,'pitch':pitch,'root_height':q[2],'com_x':com[e,t][0],'support_impulse_left':s[0],'support_impulse_right':s[1],'left_front_normal_x':left_front,'left_rear_normal_x':left_rear,'right_front_normal_x':right_front,'right_rear_normal_x':right_rear,'nonfoot':s[4],'supported_opposing':balanced})
 rows.append({'lane':e,'initial_pitch':.04 if e==0 else -.04,'samples':vals})
(A/'analysis.json').write_text(json.dumps({'rows':rows,'support_threshold_Ns':threshold,'limits':'Development bench, body and contact model estimates; selected endpoints and integrated substeps are not simultaneous. This audit describes association and does not establish biological contact forces or causation.'},indent=2)+'\n')
for row in rows:
 print('lane',row['lane'])
 for x in row['samples']:
  if x['step'] in (10,20,40,60,80,100,110,120,130,136,140,150):print(x['step'],'pitch',round(x['pitch'],3),'comx',round(x['com_x'],4),'up',round(x['support_impulse_left'],4),round(x['support_impulse_right'],4),'front/rear',*[round(x[k],5) for k in ('left_front_normal_x','left_rear_normal_x','right_front_normal_x','right_rear_normal_x')],'qualified',x['supported_opposing'])
