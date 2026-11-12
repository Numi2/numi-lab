from pathlib import Path
import collections,gzip,json,math,subprocess
A=Path('artifacts/crow-v15-platform-pitch-balance-screen-20261112');D=A/'autonomous-20261126-posture';meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];ends=meta['steps_by_environment'];v=collections.defaultdict(lambda:[0.]*5);state={};coverage=collections.Counter(); errors=[]
p=subprocess.Popen(['cat',*[str(x) for x in sorted(D.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE)
with gzip.open(p.stdout,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];
   if t<=ends[e] and t>=3100:state[e,t]=x
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if 3100<=t<=ends[e]:coverage[e,t]+=1
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if t<3100 or t>ends[e] or 40 not in (ba,bb):continue
   b=bb if ba==40 else ba;sign=1. if ba==40 else -1.;normal,t1,t2=map(float,z[6:9]);nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]);otherz=nx*ty-ny*tx
   side=0 if b==11 or 19<=b<=26 else 1 if b==15 or 27<=b<=34 else -1
   if side<0:v[e,t][4]+=normal;continue
   v[e,t][side]+=sign*(nz*normal+tz*t1+otherz*t2);v[e,t][2+side]+=normal
  elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
assert p.wait()==0
threshold=.5*.45006*9.81*.02
rows=[]
for e,end in enumerate(ends):
 samples=[]
 for t in range(3100,end+1):
  if (e,t) not in state: continue
  x=state[e,t];q=x['q'];j=v[e,t]
  samples.append({'t':t,'left_mean_proximal_rad':sum(q[i] for i in (23,25,27,29))/4,'right_mean_proximal_rad':sum(q[i] for i in (31,33,35,37))/4,'left_up_Ns':j[0],'right_up_Ns':j[1],'left_normal_Ns':j[2],'right_normal_Ns':j[3],'nonfoot_normal_Ns':j[4], 'root_z_m':q[2],'root_vz_m_s':x['v'][2], 'root_speed_m_s':math.sqrt(sum(c*c for c in x['v'][:3])), 'root_tilt_rad':math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))})
 tail=[s for s in samples if s['t']>=3800];support=[s for s in tail if s['left_up_Ns']+s['right_up_Ns']>=threshold and s['nonfoot_normal_Ns']<=1e-7]
 unilateral=[s for s in support if s['right_up_Ns']>=threshold and s['left_normal_Ns']<=1e-7]
 def runs(xs):
  groups=[]
  for t in xs:
   if not groups or t!=groups[-1][-1]+1:groups.append([t])
   else:groups[-1].append(t)
  return [{'first_step':g[0],'last_step':g[-1],'intervals':len(g)} for g in groups]
 open_release=[s['t'] for s in unilateral if s['left_mean_proximal_rad']<.20]
 closed_bilateral=[s['t'] for s in support if s['left_mean_proximal_rad']>.60 and s['right_mean_proximal_rad']>.60 and s['left_normal_Ns']>1e-7 and s['right_normal_Ns']>1e-7]
 events={'open_unilateral_release_runs':runs(open_release),'closed_bilateral_support_runs':runs(closed_bilateral),'last_25_supported_intervals':sum(s['left_up_Ns']+s['right_up_Ns']>=threshold and s['nonfoot_normal_Ns']<=1e-7 for s in samples[-25:]),'last_25_closed_bilateral_intervals':sum(s['t'] in closed_bilateral for s in samples[-25:]), 'last_25_max_root_speed_m_s':max((s['root_speed_m_s'] for s in samples[-25:]),default=None), 'last_25_max_root_tilt_rad':max((s['root_tilt_rad'] for s in samples[-25:]),default=None)}
 rows.append({'events':events,'lane':e,'end':end,'failure':meta['physical_failure_by_environment'][e],'waypoints':meta['maximum_waypoints_by_environment'][e], 'covered_expected_after_3100':all(coverage[e,t]==16 for t in range(3100,end+1)), 'post_3800_supported_intervals':len(support),'post_3800_right_only_supported_intervals':len(unilateral),'post_3800_left_proximal_range_rad':[min((s['left_mean_proximal_rad'] for s in tail),default=None),max((s['left_mean_proximal_rad'] for s in tail),default=None)],'post_3800_right_proximal_range_rad':[min((s['right_mean_proximal_rad'] for s in tail),default=None),max((s['right_mean_proximal_rad'] for s in tail),default=None)],'last_25_samples':samples[-25:]})
out={'rows':rows,'substep_errors':errors,'scope':'Previous development seed and layout only. Impulse support threshold is half body weight per 20ms; absence of contact is meaningful only with complete substep coverage. Flexion averages do not establish voluntary control. This is not the newer gait or qualified flight controller.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print({k:v for k,v in r.items() if k!='last_25_samples'})
