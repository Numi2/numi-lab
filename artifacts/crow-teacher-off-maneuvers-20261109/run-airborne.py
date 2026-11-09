from pathlib import Path
import difflib,gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd();arm=os.environ['CROW_SPEED_ARM'];A=R/'artifacts/crow-teacher-off-maneuvers-20261109'/('airborne-'+arm);A.mkdir(exist_ok=True);B=R/'build/crow-teacher-off-maneuvers-20261109'/('airborne-'+arm);B.mkdir(parents=True,exist_ok=True)
LIB=R/'build/crow-phase-lib-20261109/build/crow-envelope-restore-lib/build/crow-flight-source-diagnostics-01/versioned-flight-bench-lib/libmetalrobo.dylib'
POLICY=R/'build/crow-phase-policy-20261109/build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'
source=R/'artifacts/crow-command-tracking-handoff-20261109/tracking.stderr.gz'
with gzip.open(source,'rt') as f:
 samples=[json.loads(l.split(' ',1)[1]) for l in f if l.startswith('CROW_FLIGHT_SAMPLE ')]
templates=[next(x for x in samples if x['step']==700 and x['lane']==lane) for lane in range(0,40,5)]
(A/'airborne-templates.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_description':'accepted first episode of matched controller, t=14 s, straight lanes','templates':templates},indent=2)+'\n')
def fmt(v):
 s=format(v,'.10g')
 return s+('.0' if '.' not in s and 'e' not in s else '')+'f'
def array(name,key,width):
 return 'constant float '+name+'[8]['+str(width)+'] = {\n'+''.join('  {'+','.join(fmt(v) for v in t[key])+'},\n' for t in templates)+'};\n'
base=(R/'src/metal/LocomotionTask.metal').read_text(); src=B/'airborne.metal'
subprocess.run(['patch','-s','-o',str(src),str(R/'src/metal/LocomotionTask.metal'),str(R/'artifacts/crow-command-tracking-handoff-20261109/tracking.patch')],check=True)
s=src.read_text()
def sub(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new)
sub('namespace {\n','namespace {\n'+array('kAirQ','q',23)+array('kAirV','v',22)+array('kAirActions','applied_filtered_action',18))
sub('''        const bool projectileEpisode = randomUnit(''','''        // Diagnostic initialization from physically reached airborne states.
        if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u) {
            const uint source = environment / 10u;
            const uint perturb = environment % 10u;
            for (uint i=0u; i<min(dispatch.counts.z,23u); ++i) resetQ[qBase+i]=kAirQ[source][i];
            for (uint i=0u; i<min(dispatch.counts.w,22u); ++i) resetV[vBase+i]=kAirV[source][i];
            const float roll = perturb==1u ? 0.15f : perturb==2u ? -0.15f : perturb==9u ? 0.10f : 0.0f;
            const float pitch = perturb==3u ? 0.10f : perturb==4u ? -0.10f : perturb==9u ? -0.08f : 0.0f;
            const float yaw = perturb==5u ? 0.15f : perturb==6u ? -0.15f : perturb==9u ? 0.12f : 0.0f;
            const float4 qx=float4(sin(0.5f*roll),0.0f,0.0f,cos(0.5f*roll));
            const float4 qy=float4(0.0f,sin(0.5f*pitch),0.0f,cos(0.5f*pitch));
            const float4 qz=float4(0.0f,0.0f,sin(0.5f*yaw),cos(0.5f*yaw));
            const float4 existing=float4(resetQ[qBase+program.root.z+3u],resetQ[qBase+program.root.z+4u],resetQ[qBase+program.root.z+5u],resetQ[qBase+program.root.z+6u]);
            const float4 disturbed=quaternionProduct(quaternionProduct(qz,quaternionProduct(qy,qx)),existing);
            for (uint j=0u;j<4u;++j) resetQ[qBase+program.root.z+3u+j]=disturbed[j];
            resetV[vBase+program.root.w+1u] += perturb==7u ? 0.15f : perturb==8u ? -0.15f : perturb==9u ? 0.12f : 0.0f;
            if (perturb==9u) resetV[vBase+program.root.w+3u] += 0.20f;
            for (uint a=0u;a<min(program.counts0.x,18u);++a) {
                for (uint d=0u;d<program.layout.w;++d) actionHistory[delayBase+d*program.counts0.x+a]=kAirActions[source][a];
                rawPolicyActions[environment*program.counts0.x+a]=kAirActions[source][a];
                const uint vi=actions[a].indices.w;
                const uint qi=actions[a].indices.z;
                previousJointVelocity[previousVelocityBase+a]=vi==MR_INVALID_INDEX ? 0.0f : resetV[vBase+vi];
                previousJointVelocity[previousVelocityBase+program.counts0.x+a]=qi==MR_INVALID_INDEX ? 0.0f : resetQ[qBase+qi];
            }
            state.episode.x=700u;
            state.commandAndPhase=float4(0.70f,0.0f,0.0f,fmod(700.0f*kTwoPi*dispatch.timing.x/program.locomotion.y,kTwoPi));
            const float3 lin=rootWorldLinearVelocity(program,resetQ+qBase,resetV+vBase);
            const float3 ang=rotateInverse(disturbed,rootWorldAngularVelocity(program,resetV+vBase));
            const float angle=atan2(2.0f*(disturbed.w*disturbed.z+disturbed.x*disturbed.y),1.0f-2.0f*(disturbed.y*disturbed.y+disturbed.z*disturbed.z));
            const float fw=cos(angle)*lin.x+sin(angle)*lin.y;
            const float lat=-sin(angle)*lin.x+cos(angle)*lin.y;
            const float pitchAngle=asin(clamp(2.0f*(disturbed.w*disturbed.y-disturbed.z*disturbed.x),-1.0f,1.0f));
            const float rollAngle=atan2(2.0f*(disturbed.w*disturbed.x+disturbed.y*disturbed.z),1.0f-2.0f*(disturbed.x*disturbed.x+disturbed.y*disturbed.y));
            state.commandExtension.xyz=float3(-angle,ang.z,fw);
            state.threatGeometry=float4(0.85f-resetQ[qBase+program.root.z+2u],lin.z,pitchAngle,ang.y);
            state.threatTeacher=float4(lat,rollAngle,ang.x,0.0f);
        }
        const bool projectileEpisode = randomUnit(''')
sub('''        const float rate = mode == 1u ? 0.03f : mode == 2u ? -0.03f : 0.0f;''','''        const float rate = 0.0f;''')
sub('''        const float targetHeight = seconds >= 6.0f ? (mode == 3u ? 1.05f : mode == 4u ? 0.65f : 0.85f) : 0.85f;''','''        const float targetHeight = 0.85f;''')
if arm=='trim07':
 sub('const float speedBias = 0.55f;','const float speedBias = clamp(0.55f + 0.70f * (state.commandExtension.z - 0.70f), 0.25f, 0.95f);')
elif arm=='trim14':
 sub('const float speedBias = 0.55f;','const float speedBias = clamp(0.55f + 1.40f * (state.commandExtension.z - 0.70f), 0.25f, 0.95f);')
elif arm=='pitch18':
 sub('0.12f * verticalRate - 1.20f * forwardSpeedError,','0.12f * verticalRate - 1.80f * forwardSpeedError,')
else: raise ValueError(arm)
sub('''        const bool avianJourneyTeacher =
            dispatch.sampling.w != 0u &&''','''        const bool fixedFlightController =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u;
        const bool avianJourneyTeacher =
            dispatch.sampling.w != 0u &&''')
sub('avianJourneyTeacher || approachSupervisorBlend > 0.0f','avianJourneyTeacher || fixedFlightController || approachSupervisorBlend > 0.0f')
sub('''        const bool avianNavigationTeacher =
            avianJourneyTeacher &&''','''        const bool avianNavigationTeacher =
            (avianJourneyTeacher || fixedFlightController) &&''')
sub('''        const float requestedBeforeAvailability = avianJourneyTeacher
            ? mix(''','''        const float requestedBeforeAvailability = fixedFlightController && action < 15u
            ? navigationTeacherRequested
            : avianJourneyTeacher
            ? mix(''')
src.write_text(s);(A/'airborne.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/src/metal/LocomotionTask.metal')))
air=B/'airborne.air';shader=B/'airborne.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air');subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
cmd=json.loads((R/'artifacts/crow-flight-envelope-20261109/yaw-negative-analysis.json').read_text())['argv']
for key,val in [('--metallib',str(shader)),('--policy-pack',str(POLICY)),('--state-trace',str(B/'airborne-lane1.tsv')),('--state-trace-environment','1'),('--envs','80'),('--steps','800'),('--seed','20261116')]:cmd[cmd.index(key)+1]=val
cmd.remove('--birdflow-journey-teacher')
authority=cmd.index('--birdflow-journey-student-authority'); del cmd[authority:authority+2]
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(LIB.parent),'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480)
(A/'airborne.stdout').write_text(p.stdout)
with (A/'airborne.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('return='+str(p.returncode)+'\n'+p.stderr[-5000:])
data=json.loads(p.stdout);first=data['first_episode_navigation'];samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')];rows=[]
for lane in range(80):
 ss=sorted((x for x in samples if x['lane']==lane and 300<=x['step']<=700 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step']);vals=[]
 for x in ss:
  q=x['q'];v=x['v'];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));fw=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lat=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]
  vals.append((abs(yaw),abs(lat),abs(fw-.7),abs(q[2]-.85)))
 maxima=[max((x[i] for x in vals),default=None) for i in range(4)]
 passed=(data['failed_environment_steps']==0 and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]>=800 and [x['step'] for x in ss]==list(range(300,701,100)) and all(maxima[i]<=lim for i,lim in enumerate([.15,.20,.20,.15])))
 rows.append({'lane':lane,'template':lane//10,'perturbation':lane%10,'physical_failure':first['physical_failure_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_steps':[x['step'] for x in ss],'max_heading_error_rad':maxima[0],'max_lateral_m_s':maxima[1],'max_speed_error_m_s':maxima[2],'max_height_error_m':maxima[3],'screen_pass':passed})
groups=[{'perturbation':i,'passes':sum(x['screen_pass'] for x in rows if x['perturbation']==i),'physical_failures':sum(x['physical_failure'] for x in rows if x['perturbation']==i)} for i in range(10)]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report={'argv':cmd,'source_sha256':h(src),'shader_sha256':h(shader),'library_sha256':h(LIB),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'groups':groups,'scope':'airborne state reset from measured development trajectories, not qualification'}
(A/'airborne-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2),flush=True)
