from pathlib import Path
import difflib, gzip, hashlib, json, math, os, statistics, subprocess
R=Path.cwd()
A=R/'artifacts/crow-wing-phase-control-20261109'
B=R/'build/crow-wing-phase-20261109'
B.mkdir(exist_ok=True)
LIB=R/'build/crow-phase-lib-20261109/build/crow-envelope-restore-lib/build/crow-flight-source-diagnostics-01/versioned-flight-bench-lib/libmetalrobo.dylib'
POLICY=R/'build/crow-phase-policy-20261109/build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'
base=(R/'src/metal/LocomotionTask.metal').read_text()
src=B/'phase.metal'
subprocess.run(['patch','-s','-o',str(src),str(R/'src/metal/LocomotionTask.metal'),str(R/'artifacts/crow-flight-envelope-20261109/yaw-negative.patch')],check=True)
s=src.read_text()
old='''            const float speedBias = 0.55f;
            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? speedBias : -speedBias), -1.0f, 1.0f);'''
new='''            const uint group = environment % 9u;
            const float amplitude = group == 1u || group == 5u ? 0.30f :
                                    group == 2u || group == 6u ? -0.30f :
                                    group == 3u || group == 7u ? 0.60f :
                                    group == 4u || group == 8u ? -0.60f : 0.0f;
            const float phaseOffset = amplitude * (group <= 4u ? sin(state.commandAndPhase.w) : cos(state.commandAndPhase.w));
            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? 0.55f : -0.55f) + phaseOffset, -1.0f, 1.0f);'''
assert s.count(old)==1
s=s.replace(old,new)
src.write_text(s)
(A/'phase.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/src/metal/LocomotionTask.metal')))
air=B/'phase.air'; shader=B/'phase.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
cmd=json.loads((R/'artifacts/crow-flight-envelope-20261109/yaw-negative-analysis.json').read_text())['argv']
for key,val in [('--metallib',str(shader)),('--policy-pack',str(POLICY)),('--state-trace',str(B/'phase-lane0.tsv')),('--envs','72')]:
 cmd[cmd.index(key)+1]=val
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(LIB.parent),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480)
(A/'phase.stdout').write_text(p.stdout)
with (A/'phase.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('return='+str(p.returncode)+'\n'+p.stderr[-5000:])
data=json.loads(p.stdout); first=data['first_episode_navigation']
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
rows=[]
for lane in range(72):
 ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step'])
 vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,d=q[3:7]
  yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
  forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]
  lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]
  vals.append((yaw,abs(lateral),abs(forward-.7),abs(q[2]-.85)))
 rates=[math.atan2(math.sin(y[0]-x[0]),math.cos(y[0]-x[0]))/2 for x,y in zip(vals,vals[1:])]
 rows.append({'lane':lane,'group':lane%9,'physical_failure':first['physical_failure_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'mean_yaw_rad_s':statistics.mean(rates) if len(rates)==5 else None,'max_abs_lateral_m_s':max((x[1] for x in vals),default=None),'max_abs_speed_error_m_s':max((x[2] for x in vals),default=None),'max_abs_height_error_m':max((x[3] for x in vals),default=None)})
groups=[]
for g in range(9):
 rr=[x for x in rows if x['group']==g]
 rates=[x['mean_yaw_rad_s'] for x in rr if x['mean_yaw_rad_s'] is not None]
 med=statistics.median(rates) if len(rates)==8 else None
 passed=(data['failed_environment_steps']==0 and len(rates)==8 and med<=-.03 and all(not x['physical_failure'] and x['steps']==1600 and x['sample_steps']==list(range(1000,1501,100)) and x['max_abs_lateral_m_s']<=.2 and x['max_abs_speed_error_m_s']<=.25 and x['max_abs_height_error_m']<=.2 for x in rr))
 groups.append({'group':g,'median_mean_yaw_rad_s':med,'physical_failures':sum(x['physical_failure'] for x in rr),'sample_complete':len(rates),'maximum_lateral_m_s':max((x['max_abs_lateral_m_s'] for x in rr if x['max_abs_lateral_m_s'] is not None),default=None),'maximum_speed_error_m_s':max((x['max_abs_speed_error_m_s'] for x in rr if x['max_abs_speed_error_m_s'] is not None),default=None),'maximum_height_error_m':max((x['max_abs_height_error_m'] for x in rr if x['max_abs_height_error_m'] is not None),default=None),'screen_pass':passed})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'argv':cmd,'source_sha256':h(src),'shader_sha256':h(shader),'library_sha256':h(LIB),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'groups':groups,'sample_spacing_s':2,'scope':'exploratory nominal v15 flight bench, not qualification'}
(A/'phase-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(groups,indent=2),flush=True)
