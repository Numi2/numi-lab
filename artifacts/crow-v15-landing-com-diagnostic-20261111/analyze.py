from pathlib import Path
import gzip,json,math,subprocess,os,hashlib
D=Path('artifacts/crow-v15-landing-com-diagnostic-20261111');A=Path('artifacts/crow-v15-perch-strong-balance-screen-20261111');analysis=json.loads((A/'analysis.json').read_text());ep=analysis['lanes'];samples={}
for p in sorted((A/'trial').glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step'];
    if t%100==0 and 3300<=t<=min(4100,ep[e]['first_episode_end_step']):samples[e,t]=x
with gzip.open(A/'trial/replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==1
frames={f['step']:f for f in replay['frames']}
ref=[(t,x) for (e,t),x in samples.items() if e==1 and t in frames and t-1 in frames]
errors={offset:max(max(abs(v-w) for v,w in zip(x['q'],frames[t+offset]['q'])) for t,x in ref) for offset in [0,-1]}
offset=min(errors,key=errors.get);assert errors[offset]<1e-6,errors
input_text=''.join(f'{e} {t} '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(samples.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-landing-approach-dev-20261111/trace').resolve())}
p=subprocess.run(['build/crow-v15-landing-com-diagnostic-20261111/kinematics'],input=input_text,text=True,capture_output=True,check=True,env=env)
bodies={};com={}
for line in p.stdout.splitlines():
 x=line.split();e=int(x[1]);t=int(x[2]);
 if x[0]=='body':bodies[e,t,x[3]]=list(map(float,x[4:]))
 else:com[e,t]=list(map(float,x[3:]))
maxpos=maxrot=0.;perbody={}
for t,x in ref:
 f=frames[t+offset]
 for i,name in enumerate(replay['body_names'][:33]):
  y=bodies[1,t,name];z=f['body_states'][i*13:i*13+13]
  err=math.sqrt(sum((a-b)**2 for a,b in zip(y[:3],z[:3])));maxpos=max(maxpos,err);perbody[name]=max(perbody.get(name,0),err)
  maxrot=max(maxrot,min(math.sqrt(sum((a-b)**2 for a,b in zip(y[3:7],z[3:7]))),math.sqrt(sum((a+b)**2 for a,b in zip(y[3:7],z[3:7])))))
print('per-body maximum position mismatch:',sorted(perbody.items(),key=lambda x:-x[1])[:12]);assert maxpos<1e-5 and maxrot<1e-5,(maxpos,maxrot)
rows=[]
for (e,t),z in sorted(com.items()):
 l=bodies[e,t,'crow_left_foot'];r=bodies[e,t,'crow_right_foot'];mid=[(l[i]+r[i])/2 for i in range(3)]
 rows.append({'lane':e,'step':t,'mass_kg':z[0],'com_m':z[1:4],'foot_midpoint_m':mid,'com_minus_foot_m':[z[i+1]-mid[i] for i in range(3)],'first_perch_contact_step':ep[e]['first_scene_contact']['step'],'episode_end':ep[e]['first_episode_end_step']})
result={'source':str(A),'kinematics_replay_validation':{'lane':1,'offset':offset,'q_errors_by_offset':errors,'max_position_error_m':maxpos,'max_quaternion_error':maxrot,'samples':len(ref)},'rows':rows,'limitations':'COM uses reconstructed v15 model body masses and endpoint poses; foot midpoint is not the actual center of pressure or perch center. The comparison estimates posture only and does not measure stability margin, torque, continuous support, or biological realism.'}
(D/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('validated',result['kinematics_replay_validation'])
for e in range(8):
 rr=[x for x in rows if x['lane']==e and x['step']>=x['first_perch_contact_step']]
 print(e,[(x['step'],round(x['com_minus_foot_m'][0],3),round(x['com_minus_foot_m'][1],3)) for x in rr])
