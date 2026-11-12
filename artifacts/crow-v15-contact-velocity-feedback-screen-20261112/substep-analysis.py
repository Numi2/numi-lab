from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess
A=Path('artifacts/crow-v15-contact-velocity-feedback-screen-20261112');D=A/'candidate-20261127-rebound-phase'
q={};v={};points=collections.defaultdict(list);endpoints={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);key=(e,t,k);d=q if z[0]=='CROW_Q_SUBSTEP' else v
   assert key not in d;d[key]=list(map(float,z[4:]))
  elif z[0]=='CROW_LOADED_POINT':
   t,k,e,body=map(int,z[1:5]);p=list(map(float,z[5:]));assert len(p)==7 and p[-1]>0 and all(math.isfinite(x) for x in p);points[e,t,k].append((body,p))
  elif z[0]=='CROW_SUPPORT_SAMPLE ':
   pass
  elif line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1])
   if 201<=x['step']<=300:endpoints[x['lane'],x['step']]=x
assert len(q)==len(v)==12800 and q.keys()==v.keys() and len(endpoints)==800,(len(q),len(v),len(endpoints))
assert all(len(x)==41 for x in q.values()) and all(len(x)==40 for x in v.values())
assert all(key in q for key in points)
# Cross-check the final substep's loaded contact geometry against the
# independently published endpoint contact record in every lane.
mismatches=[]
for (e,t),sample in endpoints.items():
 native=collections.Counter()
 for c in sample['contacts']:
  bird=c['a'] if c['b_name']=='locomotion_ground' else c['b'] if c['a_name']=='locomotion_ground' else -1
  if bird in (11,15) or 19<=bird<=34:
   if c['impulses'][0]>1e-7:native[(bird,*c['point'],*c['normal'],c['impulses'][0])]+=1
 observed=collections.Counter((body,*a) for body,a in points.get((e,t,15),[]))
 if native!=observed:mismatches.append((e,t,len(native),len(observed)))
assert not mismatches, mismatches[:10]

endpoint_q=max(abs(a-b) for (e,t),x in endpoints.items() for a,b in zip(q[e,t,15],x['q']))
endpoint_v=max(abs(a-b) for (e,t),x in endpoints.items() for a,b in zip(v[e,t,15],x['v']))
assert endpoint_q<1e-6 and endpoint_v<1e-6,(endpoint_q,endpoint_v)
with gzip.open(D/'replay.json.gz','rt') as f:r=json.load(f)['payload']
assert r['environment']==5 and r['body_state_stride']==13
names=r['body_names']; ground=names.index('locomotion_ground');ground_speed=max(abs(z) for fr in r['frames'] if 200<=fr['step']<=299 for z in fr['body_states'][ground*13+7:ground*13+13]);assert ground_speed<1e-7
keys=sorted(q);inp=''.join(str(e)+' '+str((t-201)*16+k+1)+' '+' '.join(map(str,q[e,t,k]+v[e,t,k]))+'\n' for e,t,k in keys)
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
bodies={}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body':
  e,i=int(z[1]),int(z[2]);name=z[3]
  if name.startswith(('crow_left_','crow_right_')) and ('foot' in name or 'toe' in name):
   vals=list(map(float,z[4:]));assert len(vals)==13 and all(math.isfinite(x) for x in vals)
   bodies[e,i,names.index(name)]=vals
max_pos=max_vel=max_ang=0.
for fr in r['frames']:
 t=fr['step']+1
 if not 201<=t<=300:continue
 i=(t-201)*16+16
 for name in names:
  if not name.startswith(('crow_left_','crow_right_')) or not ('foot' in name or 'toe' in name):continue
  b=names.index(name);raw=fr['body_states'][b*13:b*13+13];calc=bodies[5,i,b]
  max_pos=max(max_pos,math.dist(raw[:3],calc[:3]));max_vel=max(max_vel,math.dist(raw[7:10],calc[7:10]));max_ang=max(max_ang,math.dist(raw[10:13],calc[10:13]))
assert max_pos<1e-5 and max_vel<1e-4 and max_ang<1e-4,(max_pos,max_vel,max_ang)
speeds=collections.defaultdict(list);timeseries=collections.defaultdict(dict);body_metric=collections.defaultdict(list);loaded=collections.Counter();excluded=0
for (e,t,k),ps in points.items():
 groups=collections.defaultdict(list);i=(t-201)*16+k+1
 for body,a in ps:
  normal=a[3:6];n=math.sqrt(sum(x*x for x in normal))
  assert abs(n-1)<1e-4,(e,t,k,normal)
  if abs(normal[2])<.99:excluded+=1;continue
  name=names[body];side='left' if name.startswith('crow_left_') else 'right' if name.startswith('crow_right_') else None
  assert side is not None,(body,name)
  b=bodies[e,i,body];rx,ry,rz=[a[j]-b[j] for j in range(3)];wx,wy,wz=b[10:13]
  vv=[b[7]+wy*rz-wz*ry,b[8]+wz*rx-wx*rz,b[9]+wx*ry-wy*rx];dot=sum(vv[j]*normal[j] for j in range(3));tang=math.sqrt(max(0,sum(x*x for x in vv)-dot*dot))
  groups[side].append((tang,a[-1]));body_metric[body].append((tang,a[-1]))
 for side,items in groups.items():
  value=sum(speed*imp for speed,imp in items)/sum(imp for _,imp in items)
  speeds[e,side].append(value);timeseries[e,side][(t-201)*16+k]=value;loaded[e,side]+=1
rows=[]
for e in range(8):
 for side in ('left','right'):
  values=sorted(speeds[e,side]);assert values
  # Sensitivity analysis: drop the first and last 8 samples of each
  # contiguous loaded interval, removing 10 ms near each contact transition.
  series=timeseries[e,side]; runs=[];run=[]
  for t in sorted(series):
   if run and t!=run[-1]+1:runs.append(run);run=[]
   run.append(t)
  if run:runs.append(run)
  interior=[series[t] for run in runs if len(run)>16 for t in run[8:-8]]
  interior.sort()
  boundary=sorted(value for t,value in series.items() if t%16==15)
  rows.append({'lane':e,'side':side,'loaded_substeps':loaded[e,side],'mean_m_s':statistics.mean(values),'p95_m_s':values[int(.95*(len(values)-1))],'integral_of_mean_contact_speed_m':sum(values)*.02/16,'max_m_s':values[-1], 'frame_boundary_weighted_p95_m_s':boundary[int(.95*(len(boundary)-1))] if boundary else None, 'interior_after_10ms_edge_trim':{'samples':len(interior),'mean_m_s':statistics.mean(interior) if interior else None,'p95_m_s':interior[int(.95*(len(interior)-1))] if interior else None}})
out={'window_seconds':[4,6],'substep_duration_seconds':.02/16,'states':len(q),'loaded_contact_groups':sum(loaded.values()),'excluded_nonvertical_contact_points':excluded,'validation':{'q_endpoint_max_abs':endpoint_q,'v_endpoint_max_abs':endpoint_v,'foot_position_max_abs_m':max_pos,'foot_velocity_max_abs_m_s':max_vel,'foot_angular_velocity_max_abs_rad_s':max_ang,'ground_max_speed':ground_speed,'endpoint_contact_record_mismatches':len(mismatches)},'rows':rows,'per_body':[{'body':names[b],'contact_points':len(pts),'impulse_weighted_mean_m_s':sum(speed*imp for speed,imp in pts)/sum(imp for _,imp in pts),'normal_impulse_ns':sum(imp for _,imp in pts)} for b,pts in sorted(body_metric.items())],'limitations':'Impulse-weighted speed integrated only while loaded; multiple changing contact points, and contact points are from collision while q/v are end-of-substep states. This is a speed integral proxy, not tracked material-point slip distance.'}
(A/'substep-contact-speed.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x)
