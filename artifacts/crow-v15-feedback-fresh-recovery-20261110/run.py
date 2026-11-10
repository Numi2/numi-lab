from pathlib import Path
import difflib,gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-feedback-fresh-recovery-20261110';B=R/'build/crow-v15-feedback-fresh-recovery-20261110';SRC=R/'build/crow-v15-feedback-flight-envelope-20261110/cruise';HOST=R/'build/crow-v15-corrected-wing-action-observer-20261110';POLICY=R/'build/crow-v15-wing-inertia-prototype-20261110/zero.policypack'
source=A/'flight.stderr.gz'
with gzip.open(source,'rt') as f:lines=f.readlines()
samples=[json.loads(l.split(' ',1)[1]) for l in lines if l.startswith('CROW_FLIGHT_SAMPLE ')];actions=[json.loads(l.split(' ',1)[1]) for l in lines if l.startswith('CROW_ACTION_SAMPLE ')]
templates=[]
for lane in range(0,40,5):
 x=next(x for x in samples if x['step']==700 and x['lane']==lane);a=next(x for x in actions if x['episode_step']==700 and x['lane']==lane);x['applied_filtered_action']=a['filtered'];assert len(x['q'])==31 and len(x['v'])==30 and len(a['filtered'])==18;templates.append(x)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'airborne-templates.json').write_text(json.dumps({'source_sha256':h(source),'source_description':'first episode of corrected-body fixed-head development flight at step 700; filtered actions from matched observer','templates':templates},indent=2)+'\n')
def fmt(v):
 s=format(v,'.10g');return s+('.0' if '.' not in s and 'e' not in s else '')+'f'
def array(name,key,width):return 'constant float '+name+'[8]['+str(width)+'] = {\n'+''.join('  {'+','.join(fmt(v) for v in t[key])+'},\n' for t in templates)+'};\n'
base=(SRC/'tracking.metal').read_text();s=base
def sub(old,new):
 global s
 assert s.count(old)==1,(old[:80],s.count(old));s=s.replace(old,new)
sub('namespace {\n','namespace {\n'+array('kAirQ','q',31)+array('kAirV','v',30)+array('kAirActions','applied_filtered_action',18))
# Reuse the previously archived perturbation and accepted-state-feedback setup,
# changing only the q/v dimensions for the current articulated toe body.
old=(R/'artifacts/crow-airborne-speed-recovery-20261109/run.py').read_text();start=old.index("sub('''        const bool projectileEpisode = randomUnit(");end=old.index("sub('''        const float rate =",start);fragment=old[start:end].replace('dispatch.counts.z,23u','dispatch.counts.z,31u').replace('dispatch.counts.w,22u','dispatch.counts.w,30u');exec(fragment)
sub('        const float rate = mode == 1u ? 0.03f : mode == 2u ? -0.03f : 0.0f;','        const float rate = 0.0f;')
sub('        const float targetHeight = seconds >= 6.0f ? (mode == 3u ? 1.05f : mode == 4u ? 0.65f : 0.85f) : 0.85f;','        const float targetHeight = 0.85f;')
(B/'airborne.metal').write_text(s);(A/'recovery.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/flight-tracking.metal',tofile='b/research-airborne-recovery.metal')))
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(SRC),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'airborne.metal'),'-o',str(B/'airborne.air')],check=True)
airs=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in airs],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'airborne.air'),'-o',str(B/'airborne.metallib')],check=True)
cmd=json.loads((A/'flight-analysis.json').read_text())['argv']
for key,val in [('--metallib',str(B/'airborne.metallib')),('--policy-pack',str(POLICY)),('--state-trace',str(B/'airborne.tsv')),('--state-trace-environment','1'),('--envs','80'),('--steps','800'),('--seed','18722117')]:cmd[cmd.index(key)+1]=val
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_ACTION_OBSERVER':'1','DYLD_LIBRARY_PATH':str(HOST),'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/'airborne.stdout').write_text(p.stdout)
with gzip.GzipFile(filename=str(A/'airborne.stderr.gz'),mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('return='+str(p.returncode)+'\n'+p.stderr[-5000:])
data=json.loads(p.stdout);first=data['first_episode_navigation'];samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')];rows=[]
for lane in range(80):
 ss=sorted((x for x in samples if x['lane']==lane and 300<=x['step']<=700 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));fw=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lat=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((abs(yaw),abs(lat),abs(fw-.7),abs(q[2]-.85)))
 maxima=[max((x[i] for x in vals),default=None) for i in range(4)];passed=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]>=800 and [x['step'] for x in ss]==list(range(300,701,100)) and all(maxima[i]<=lim for i,lim in enumerate([.15,.20,.20,.15])));rows.append({'lane':lane,'template':lane//10,'perturbation':lane%10,'physical_failure':first['physical_failure_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'screen_pass':passed})
groups=[{'perturbation':i,'passes':sum(x['screen_pass'] for x in rows if x['perturbation']==i),'physical_failures':sum(x['physical_failure'] for x in rows if x['perturbation']==i)} for i in range(10)];report={'argv':cmd,'source_sha256':h(B/'airborne.metal'),'shader_sha256':h(B/'airborne.metallib'),'library_sha256':h(HOST/'libmetalrobo.dylib'),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'groups':groups,'scope':'airborne reset from development trajectories, not continuous autonomous execution or biological validation'};(A/'airborne-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2),flush=True)
