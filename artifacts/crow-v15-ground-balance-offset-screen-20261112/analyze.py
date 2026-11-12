from pathlib import Path
import json,gzip,math,statistics
p=Path('artifacts/crow-v15-ground-balance-offset-screen-20261112'); trial=p/'candidate-20261127'; prior=Path('artifacts/crow-v15-shank-proxy-trim-screen-20261112/candidate-20261127')
def read(d):
 result=json.loads((d/'run.stdout').read_text());end=result['first_episode_navigation'];states={};sub={};errors=[]
 with gzip.open(d/'run.stderr.gz','rt') as f:
  for line in f:
   if 'CROW_SUBSTEP_ERROR' in line:errors.append(line.strip())
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]); states[x['lane'],x['step']]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);sub[e,t,k]=z[4:]
    if int(z[4])>int(z[5]):errors.append('overflow')
 return end,states,sub,errors
end,states,sub,errors=read(trial);old,ost,osub,oerr=read(prior)
assert all((e,t) in states for e in range(8) for t in range(1,301))
parity={'endpoint_mismatches':sum(states[3,t]!=ost[3,t] for t in range(1,301)),'substep_mismatches':sum(sub[3,t,k]!=osub[3,t,k] for t in range(1,301) for k in range(16))}
rows=[]
for e,off in enumerate([-.3,-.2,-.1,0,.05,.1,.2,.3]):
 finish=end['steps_by_environment'][e]; samples={}
 for t in [25,50,75,100,125,150,175,200]:
  if t>finish:continue
  x=states[e,t];a,b,c,w=x['q'][3:7];pitch=math.asin(max(-1,min(1,2*(w*b-c*a))));samples[str(t)]={'pitch_rad':pitch,'position_m':x['q'][:3]}
 rows.append({'lane':e,'hip_offset_action':off,'first_episode_end':finish,'failure':end['physical_failure_by_environment'][e],'reason':end['termination_reason_by_environment'][e],'samples':samples})
out={'zero_offset_parity':parity,'rows':rows,'errors':errors,'scope':'Used-seed diagnostic screen; first-episode values only. Different offsets occupy different lanes, not matched randomized replicates. Offset is an action sensitivity, not a biological claim or feedback controller.'}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('parity',parity,'errors',errors)
for x in rows:print('lane',x['lane'],'offset',x['hip_offset_action'],'end',x['first_episode_end'],'failure',x['failure'],'pitch @50,100,150',[(k,round(x['samples'][k]['pitch_rad'],3)) for k in ['50','100','150'] if k in x['samples']])
