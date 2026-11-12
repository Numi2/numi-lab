from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-flight-height-feedback-screen-20261112');seed=int(sys.argv[1]);assert seed in (20261126,20261127);P=A/f'candidate-{seed}-rebound-flexed';mp=Path('artifacts/crow-v15-drive-work-observer-20261112/dof-map.txt');mapping={}
for line in mp.read_text().splitlines():
 z=line.split();mapping[int(z[0])]={'name':z[1],'flags':int(z[3]),'limit_rad_s':float(z[-1])}
meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ends=meta['steps_by_environment'];counts=collections.Counter();hits=collections.Counter();mx=collections.defaultdict(float);unloaded=collections.Counter();nonfoot=collections.Counter();errors=[];over=[];states=collections.defaultdict(list);scene=[]
def period(t):return 'walk_4_6' if 201<=t<=300 else 'takeoff_6_8' if 301<=t<=400 else 'flight_8plus' if t>=401 else 'early'
def speeds(e,t,kind,indices,values):
 assert len(indices)==len(values)
 for j,value in zip(indices,values):
  assert math.isfinite(value);m=mapping[j];lim=m['limit_rad_s'];assert lim>0 and m['flags']&8
  mx[e,j,period(t)]=max(mx[e,j,period(t)],abs(value));hits[e,j,period(t)]+=int(abs(value)>=lim-1e-4)
 counts[e,kind]+=1
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];
   if t<=ends[e]:states[e].append(x)
  elif line.startswith('CROW_BODY_JOINT_V_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e]:speeds(e,t,'body',list(range(6,22))+[38,39],list(map(float,z[4:])))
  elif line.startswith('CROW_TOE_V_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e]:speeds(e,t,'toe',list(range(22,38)),list(map(float,z[4:])))
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e]:
    counts[e,'contact']+=1;unloaded[e,period(t)]+=int(float(z[8])+float(z[9])<=1e-7);nonfoot[e,period(t)]+=int(float(z[10])>1e-7)
    if int(z[4])>int(z[5]):over.append([e,t,k])
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e]:scene.append({'step':t,'substep':k,'lane':e,'bodyA':int(z[4]),'bodyB':int(z[5]),'normal_impulse':float(z[6])})
  elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
rows=[]
for e in range(8):
 ss=states[e];assert len(ss)==ends[e],(e,len(ss),ends[e]); expected=ends[e]*16
 def attitude(q):
  x,y,z,w=q[3:7];return {'yaw':math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),'pitch':math.asin(max(-1,min(1,2*(w*y-z*x)))),'roll':math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))}
 h=[{'joint':mapping[j]['name'],'period':p,'limit_rad_s':mapping[j]['limit_rad_s'],'max_abs_rad_s':mx[e,j,p],'hits':n} for (lane,j,p),n in sorted(hits.items()) if lane==e and n]
 rows.append({'lane':e,'first_episode_steps':ends[e],'reported_failure':meta['physical_failure_by_environment'][e],'termination_reason':meta['termination_reason_by_environment'][e],'waypoints':meta['maximum_waypoints_by_environment'][e],'first_episode_coverage':{k:counts[e,k] for k in ('body','toe','contact')},'expected_substeps':expected,'unloaded_by_period':{p:unloaded[e,p] for p in ('early','walk_4_6','takeoff_6_8','flight_8plus')},'nonfoot_ground_by_period':{p:nonfoot[e,p] for p in ('early','walk_4_6','takeoff_6_8','flight_8plus')},'speed_cap_hits':h,'final_position_m':ss[-1]['q'][:3],'final_velocity_m_s':ss[-1]['v'][:3],'final_attitude_rad':attitude(ss[-1]['q']),'first_scene_contact':next((x for x in scene if x['lane']==e),None)})
out={'seed':seed,'rows':rows,'contact_overflows':over,'substep_errors':errors,'scope':'Used development starts, first episodes only. Full-substep observed velocities compared with engineering velocity limits from the matching research model; final substep velocity at a limit indicates clamping but not its cause. Endpoint q/attitude and recorded collision impulses do not establish continuous clearance. No biological, perturbed-flight, mission or landing qualification.'}
(A/f'analysis-{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['lane'],'end',r['first_episode_steps'],'reason',r['termination_reason'],'wp',r['waypoints'],'coverage',r['first_episode_coverage'],'caps',[(v['joint'],v['period'],v['hits']) for v in r['speed_cap_hits']],'nonfoot',r['nonfoot_ground_by_period'],'scene',r['first_scene_contact'])
