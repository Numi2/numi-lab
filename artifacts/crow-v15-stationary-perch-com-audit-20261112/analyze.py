from pathlib import Path
import gzip,json,math,os,subprocess
P=Path('artifacts/crow-v15-pitch-response-separated-20261111/zero');A=Path('artifacts/crow-v15-stationary-perch-com-audit-20261112')
times={1,10,20,40,60,80,100,120,130,136,150,200,400,550};data={}
with gzip.open(P/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0] not in ('step','event'):continue
  e,t=int(z[1]),int(z[2])+1
  if e>1 or t not in times:continue
  a=data.setdefault((e,t),{})
  if z[0]=='step':
   qi,vi=z.index('q'),z.index('v');q=list(map(float,z[qi+1:vi]));v=list(map(float,z[vi+1:]));a['q']=q;a['v']=v;a['pitch']=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))))
  else:a['event']=z[3:]
assert len(data)==2*len(times) and all('q' in x and 'event' in x for x in data.values())
inp=''.join(f'{e} {t} '+' '.join(map(str,x['q']))+'\n' for (e,t),x in sorted(data.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path.cwd()/'build/crow-v15-hip-roll-journey-integration-20261111')}
p=subprocess.run(['build/crow-v15-perch-com-moment-audit-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
for line in p.stdout.splitlines():
 z=line.split()
 if z[0]=='C':data[int(z[1]),int(z[2])]['com']=list(map(float,z[4:7]))
rows=[]
for e in (0,1):
 vals=[]
 for t in sorted(times):
  x=data[e,t];vals.append({'step':t,'root_position':x['q'][:3],'pitch':x['pitch'],'pitch_rate_world_y':x['v'][4],'root_vx':x['v'][0],'com':x['com'],'event_fields':x['event']})
 rows.append({'lane':e,'initial_pitch_rad':.04 if e==0 else -.04,'samples':vals})
out={'rows':rows,'limits':'Constructed stationary-perch bench and estimated v15 body, not flight-to-perch landing or a biological observation. Perch spans x=-0.025 to +0.025 m; COM projected outside that span alone does not determine stability because contact moments, toe forces, inertia and motion matter. Event fields retain their bench-defined meanings.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 print(r['lane'],[(v['step'],round(v['com'][0],4),round(v['pitch'],3)) for v in r['samples']])
