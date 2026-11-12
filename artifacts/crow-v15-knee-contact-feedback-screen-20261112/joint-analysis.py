from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess,sys
A=Path('artifacts/crow-v15-knee-contact-feedback-screen-20261112')
seed=int(sys.argv[1]); arm=sys.argv[2]; assert seed in (20261126,20261127) and arm in ('baseline','phase')
D=A/f'candidate-{seed}-rebound-{arm}'
q={};v={};contact={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);key=e,t,k;d=q if z[0]=='CROW_Q_SUBSTEP' else v;assert key not in d;d[key]=list(map(float,z[4:]))
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4])
   if 201<=t<=300:
    key=e,t,k;assert key not in contact;contact[key]=list(map(float,z[6:]));assert len(contact[key])==37
assert len(q)==len(v)==len(contact)==12800 and q.keys()==v.keys()==contact.keys()
inp=''.join(str(e)+' '+str((t-201)*16+k+1)+' '+' '.join(map(str,q[e,t,k]+v[e,t,k]))+'\n' for e,t,k in sorted(q))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
needed={f'crow_{side}_{name}' for side in ('left','right') for name in ('thigh','shank','foot','front_toe_2_proximal')}|{'crow_head'}
b={}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body' and z[3] in needed:b[int(z[1]),int(z[2])-1,z[3]]=list(map(float,z[4:]))
assert len(b)==8*1600*9
with gzip.open(D/'replay.json.gz','rt') as f:r=json.load(f)['payload']
names=r['body_names'];err=0.
for fr in r['frames']:
 t=fr['step']+1
 if not 201<=t<=300:continue
 i=(t-201)*16+15
 for name in needed:
  idx=names.index(name)*13;raw=fr['body_states'][idx:idx+13];calc=b[5,i,name]
  err=max(err,math.dist(raw[:3],calc[:3]))
assert err<1e-5
def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(t*t for t in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def point(body,local):
 z=rot(body[3:7],local);return [z[i]+body[i] for i in range(3)]
def angle(h,k,a):
 x=[h[i]-k[i] for i in range(3)];y=[a[i]-k[i] for i in range(3)];d=sum(x[i]*y[i] for i in range(3))/(math.sqrt(sum(i*i for i in x))*math.sqrt(sum(i*i for i in y)));return math.degrees(math.acos(max(-1,min(1,d))))
records=[];reject=collections.Counter();maxerr=0.;body_motion=[]
for e in range(8):
 root=[q[e,t,k] for t in range(201,301) for k in range(16)]
 rolls=[math.atan2(2*(x[6]*x[3]+x[4]*x[5]),1-2*(x[3]*x[3]+x[4]*x[4])) for x in root]
 pitches=[math.asin(max(-1,min(1,2*(x[6]*x[4]-x[5]*x[3])))) for x in root]
 head_local=[]
 for i,x in enumerate(root):
  h=b[e,i,'crow_head'];head_local.append(rot([-x[3],-x[4],-x[5],x[6]],[h[j]-x[j] for j in range(3)]))
 body_motion.append({'lane':e,'root_z_span_m':max(x[2] for x in root)-min(x[2] for x in root),'roll_span_rad':max(rolls)-min(rolls),'pitch_span_rad':max(pitches)-min(pitches),'head_torso_local_span_m':[max(x[j] for x in head_local)-min(x[j] for x in head_local) for j in range(3)]})
 for side,sd in enumerate(('left','right')):
  samples=[]
  for t in range(201,301):
   for k in range(16):
    z=contact[e,t,k];toes=z[5+side*8:13+side*8]
    samples.append((z[2+side]>1e-7,sum(toes[:6])>1e-7,sum(toes[6:])>1e-7))
  active=[i for i,val in enumerate(samples) if val[0]];groups=[]
  for i in active:
   if not groups or i-groups[-1][-1]>3:groups.append([i])
   else:groups[-1].append(i)
  events=[]
  for group in groups:
   start,end=group[0],group[-1]
   if start==0 or end==len(samples)-1:reject['partial_contact']+=1;events.append(None);continue
   front=[i for i in range(start,end+1) if samples[i][1]];rear=[i for i in range(start,end+1) if samples[i][2]]
   if not front or not rear:reject['missing_toe_group']+=1;events.append(None);continue
   if (end-start+1)*.00125<.04:reject['contact_under_40ms']+=1;events.append(None);continue
   events.append((rear[0],front[-1]))
  for prev,nxt in zip(events,events[1:]):
   if not prev or not nxt:reject['incomplete_pair']+=1;continue
   start,end=prev[1],nxt[0]
   if end-start<32:reject['recovery_under_40ms']+=1;continue
   knee=[];ankle=[]
   for i in range(start+1,end):
    thigh=b[e,i,'crow_'+sd+'_thigh'];sh=b[e,i,'crow_'+sd+'_shank'];foot=b[e,i,'crow_'+sd+'_foot'];toe=b[e,i,'crow_'+sd+'_front_toe_2_proximal']
    hip=point(thigh,[0,0,.031625]);k1=point(thigh,[0,0,-.031625]);k2=point(sh,[0,0,.031625]);an=point(sh,[0,0,-.031625]);an2=point(foot,[-.02875,0,.0092]);toeprox=point(toe,[-.009,0,0]);maxerr=max(maxerr,math.dist(k1,k2),math.dist(an,an2))
    knee.append(angle(hip,k1,an));ankle.append(angle(k1,an,toeprox))
   def smooth(x):
    return [statistics.mean(x[max(0,i-8):min(len(x),i+9)]) for i in range(len(x))]
   result={}
   for label,k,a in [('raw',knee,ankle),('21ms_boxcar',smooth(knee),smooth(ankle))]:
    ki=min(range(len(k)),key=k.__getitem__);ai=min(range(len(a)),key=a.__getitem__)
    result[label]={'knee_min_deg':k[ki],'ankle_proxy_min_deg':a[ai],'ankle_minus_knee_minimum_seconds':(ai-ki)*.00125,'knee_minimum_recovery_percent':100*(ki+1)/(end-start),'ankle_proxy_minimum_recovery_percent':100*(ai+1)/(end-start),'order':'knee_first' if ai>ki else 'ankle_first' if ai<ki else 'same_substep'}
   records.append({'lane':e,'side':sd,'recovery_seconds':(end-start)*.00125,**result})
out={'magpie_walking_reference':{'doi':'10.1242/jeb.203.14.2159','table':1,'knee_minimum_percent_mean_sd_n':[27.3,7.9,16],'ankle_minimum_percent_mean_sd_n':[52.3,9.3,16],'speed_m_s':[0.5,1.2]},'substep_seconds':.00125,'seed':seed,'arm':arm,'window_seconds':[4,6],'recovery_count':len(records),'order_counts':{label:dict(collections.Counter(x[label]['order'] for x in records)) for label in ('raw','21ms_boxcar')},'rejections':dict(reject),'max_joint_anchor_error_m':maxerr,'max_replay_body_position_error_m':err,'body_motion':body_motion,'records':records,'scope':'Development baseline, matched accepted contact substeps and 800 Hz q/v; 3D geometric ankle proxy uses second modeled front-toe base, whose anatomical identity is unverified. Group <=2 silent substeps, reject incomplete contacts, missing toe groups, contact or recovery under 40 ms; smoothing is a centered 21.25 ms boxcar without a biological sampling calibration. Minimum-angle ordering is a proxy for extension onset, not direct onset measurement. Different species, speeds, surfaces and anatomy prevent biological validation.'}
(A/f'joint-analysis-{seed}-{arm}.json').write_text(json.dumps(out,indent=2)+'\n')
print('recoveries',len(records),'orders',out['order_counts'],'rejections',dict(reject),'joint error',maxerr)
print('body motion',body_motion)
