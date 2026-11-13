from pathlib import Path
import gzip,json,collections,subprocess,os
R=Path.cwd();A=R/'artifacts/crow-v15-leg-coordination-screen-20261113';A.mkdir(exist_ok=True);D={};inputs=[];keys=[]
for arm in ('push','hold'):
 p=R/'artifacts/crow-v15-leg-coordination-screen-20261113'/(arm+'-wing-20261126-rebound-coordination/run.stderr.gz');d={'state':{},'contact':collections.defaultdict(lambda:[0.]*5),'joint':{}};D[arm]=d
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step']
    if e in (2,4,6) and 368<=t<=395:d['state'][e,t]=x
    continue
   z=l.split()
   if not z or z[0] not in ('CROW_TWO_TOE_SUBSTEP','CROW_LEG_EFFORT'):continue
   t,k,e=map(int,z[1:4]);
   if e not in (2,4,6) or not 370<=t<=395:continue
   if z[0]=='CROW_TWO_TOE_SUBSTEP':
    v=list(map(float,z[6:11]));d['contact'][e,t]=[a+b for a,b in zip(d['contact'][e,t],v)]
   elif k==0:
    v=list(map(float,z[7:]));d['joint'][e,t]=[v[j:j+5] for j in range(0,30,5)]
 for e in (2,4,6):
  for t in range(368,396):
   x=d['state'][e,t];keys.append((arm,e,t));inputs.append(str(len(keys)-1)+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});com={}
for line in r.stdout.splitlines():
 z=line.split();assert z[0]=='C';com[keys[int(z[1])]]=list(map(float,z[3:]))
rows=[]
for e in (2,4,6):
 for t in range(370,396):
  row={'lane':e,'tick':t}
  for arm in ('push','hold'):
   d=D[arm];x=d['state'][e,t];row[arm]={'com_z':com[arm,e,t][3],'com_vz':com[arm,e,t][6],'foot_vertical_ns':sum(d['contact'][e,t][:2]),'q_hip_knee_ankle':x['q'][14:17],'v_hip_knee_ankle':x['v'][13:16],'leg_effort_start': [x[0] for x in d['joint'][e,t][:3]],'toe_contacts':d['contact'][e,t][2:4]}
  rows.append(row)
(A/'timeline.json').write_text(json.dumps({'rows':rows,'scope':'Posthoc used-seed wing-held leg push diagnostic; contact forces not muscle forces and COM accounting excludes unlogged terms.'},indent=2)+'\n')
for r in rows:
 if r['lane'] in (2,4,6):
  p,h=r['push'],r['hold'];print(r['tick'],'foot',round(p['foot_vertical_ns'],3),round(h['foot_vertical_ns'],3),'COMvz',round(p['com_vz'],3),round(h['com_vz'],3),'q',*[round(x,2) for x in p['q_hip_knee_ankle']],'eff',*[round(x,2) for x in p['leg_effort_start']])
