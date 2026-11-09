from pathlib import Path
import difflib,gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd(); seed=os.environ['CROW_START_SEED']; A=R/'artifacts/crow-start-recovery-20261109'/seed; A.mkdir(exist_ok=True); B=R/'build/crow-start-recovery-20261109'/seed; B.mkdir(parents=True,exist_ok=True)
LIB=R/'build/crow-phase-lib-20261109/build/crow-envelope-restore-lib/build/crow-flight-source-diagnostics-01/versioned-flight-bench-lib/libmetalrobo.dylib'
POLICY=R/'build/crow-phase-policy-20261109/build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'
base=(R/'src/metal/LocomotionTask.metal').read_text(); src=B/'tracking.metal'
subprocess.run(['patch','-s','-o',str(src),str(R/'src/metal/LocomotionTask.metal'),str(R/'artifacts/crow-flight-envelope-20261109/yaw-negative.patch')],check=True)
s=src.read_text()
def sub(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new)
sub('''            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? speedBias : -speedBias), -1.0f, 1.0f);''','''            const float yawDemand = clamp(16.0f * state.commandExtension.x - 0.50f * state.commandExtension.y - 0.35f * state.threatTeacher.x, -1.0f, 1.0f);
            const float phaseOffset = 0.60f * clamp(-yawDemand, 0.0f, 1.0f) * sin(state.commandAndPhase.w);
            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? speedBias : -speedBias) + phaseOffset, -1.0f, 1.0f);''')
sub('''        if (avianNavigationTeacher && float(state.episode.x) * dispatch.timing.x >= 6.0f && action == 13u) navigationTeacherRequested = -1.00f;
''','')
sub('''        const uint mode = environment % 5u;
        const float seconds = float(state.episode.x + 1u) * dispatch.timing.x;
        const float rate = 0.0f;''','''        const uint mode = environment % 5u;
        const float seconds = float(state.episode.x + 1u) * dispatch.timing.x;
        const float rate = mode == 1u ? 0.03f : mode == 2u ? -0.03f : 0.0f;
        const float targetHeight = seconds >= 6.0f ? (mode == 3u ? 1.05f : mode == 4u ? 0.65f : 0.85f) : 0.85f;''')
sub('''        state.threatGeometry.x = 0.85f - rootWorldPosition(program, q).z;''','''        state.threatGeometry.x = targetHeight - rootWorldPosition(program, q).z;''')
sub("? (state.navigation.z >= 5.0f ? 0.0f : 1.0f)", "? 1.0f")
sub("""        const bool projectileEpisode = randomUnit(""","""        // Diagnostic reset perturbations; body, solver and joint limits unchanged.
        if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u) {
            const uint perturb = environment % 8u;
            const float roll = perturb == 1u ? 0.12f : perturb == 2u ? -0.12f : perturb == 7u ? 0.08f : 0.0f;
            const float pitch = perturb == 3u ? 0.08f : perturb == 4u ? -0.08f : perturb == 7u ? -0.06f : 0.0f;
            const float yaw = perturb == 5u ? 0.12f : perturb == 6u ? -0.12f : perturb == 7u ? 0.10f : 0.0f;
            const float4 qx = float4(sin(0.5f*roll), 0.0f, 0.0f, cos(0.5f*roll));
            const float4 qy = float4(0.0f, sin(0.5f*pitch), 0.0f, cos(0.5f*pitch));
            const float4 qz = float4(0.0f, 0.0f, sin(0.5f*yaw), cos(0.5f*yaw));
            const float4 existing = float4(resetQ[qBase+program.root.z+3u],resetQ[qBase+program.root.z+4u],resetQ[qBase+program.root.z+5u],resetQ[qBase+program.root.z+6u]);
            const float4 disturbed = quaternionProduct(quaternionProduct(qz,quaternionProduct(qy,qx)),existing);
            for (uint j=0u; j<4u; ++j) resetQ[qBase+program.root.z+3u+j] = disturbed[j];
            if (perturb != 0u) resetQ[qBase+program.root.z+2u] += 0.04f;
            if (perturb == 7u) {
                resetV[vBase+program.root.w+1u] += 0.08f;
                resetV[vBase+program.root.w+3u] += 0.15f;
            }
        }
        const bool projectileEpisode = randomUnit(""")
src.write_text(s); (A/'tracking.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/src/metal/LocomotionTask.metal')))
air=B/'tracking.air'; shader=B/'tracking.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
cmd=json.loads((R/'artifacts/crow-flight-envelope-20261109/yaw-negative-analysis.json').read_text())['argv']
for key,val in [('--metallib',str(shader)),('--policy-pack',str(POLICY)),('--state-trace',str(B/'tracking-lane0.tsv')),('--envs','80'),('--seed',seed),('--state-trace-environment','1')]:cmd[cmd.index(key)+1]=val
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(LIB.parent),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480); (A/'tracking.stdout').write_text(p.stdout)
with (A/'tracking.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('return='+str(p.returncode)+'\n'+p.stderr[-5000:])
data=json.loads(p.stdout); first=data['first_episode_navigation']; samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
rows=[]
for lane in range(80):
 mode=lane%5; rate=[0,.03,-.03,0,0][mode]; target=[.85,.85,.85,1.05,.65][mode]
 ss=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']); vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,d=q[3:7]; yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c)); desired=rate*(x['step']*.02-6)
  heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)))
  forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]; lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]
  vals.append((heading,abs(lateral),abs(forward-.7),abs(q[2]-target),q[2]))
 maxima=[max((x[i] for x in vals),default=None) for i in range(4)]
 passed=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and [x['step'] for x in ss]==list(range(1000,1501,100)) and all(maxima[i]<=lim for i,lim in enumerate([.15,.20,.20,.15])))
 rows.append({'lane':lane,'perturbation':lane%8,'mode':mode,'physical_failure':first['physical_failure_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'mean_height_m':statistics.mean(x[4] for x in vals) if len(vals)==6 else None,'screen_pass':passed})
groups=[]
for mode in range(5):
 rr=[x for x in rows if x['mode']==mode]; heights=[x['mean_height_m'] for x in rr if x['mean_height_m'] is not None]
 groups.append({'mode':mode,'physical_failures':sum(x['physical_failure'] for x in rr),'passing':sum(x['screen_pass'] for x in rr),'median_mean_height_m':statistics.median(heights) if len(heights)==16 else None,'mode_pass':all(x['screen_pass'] for x in rr)})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'argv':cmd,'source_sha256':h(src),'shader_sha256':h(shader),'library_sha256':h(LIB),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'groups':groups,'climb_minus_descent_m':groups[3]['median_mean_height_m']-groups[4]['median_mean_height_m'] if all(groups[x]['median_mean_height_m'] is not None for x in (3,4)) else None,'scope':'exploratory command tracking, not qualification'}
(A/'tracking-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2),flush=True);print('climb minus descent',report['climb_minus_descent_m'],flush=True)
