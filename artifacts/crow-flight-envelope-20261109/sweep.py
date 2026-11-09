from pathlib import Path
import difflib,gzip,hashlib,json,math,os,re,subprocess
R=Path.cwd()
A=R/'artifacts/crow-flight-envelope-20261109'
B=R/'build/crow-flight-envelope-20261109'
B.mkdir(exist_ok=True)
LIB=R/'build/crow-envelope-restore-lib/build/crow-flight-source-diagnostics-01/versioned-flight-bench-lib/libmetalrobo.dylib'
POLICY=R/'build/crow-envelope-restore-policy/build/crow-flight-source-diagnostics-01/versioned-flight-bench.policypack'
base=(R/'src/metal/LocomotionTask.metal').read_text()
def replace_one(text,old,new):
 if text.count(old)!=1: raise RuntimeError('Replacement mismatch: '+old[:100])
 return text.replace(old,new)
for name,gain,bias in [('sweep-positive',16.0,0.30),('sweep-negative',16.0,-0.30)]:
 startbank=base.index('    const float navigationBankCorrection = navigation')
 endbank=base.index('    const float navigationPitchTarget = navigation',startbank)
 s=base[:startbank]+f'''    const float targetRoll = clamp(0.25f * lateralSpeedError, -0.12f, 0.12f);
    const float navigationBankCorrection = navigation
        ? clamp(1.50f * (targetRoll - state.threatTeacher.y) - 0.25f * state.threatTeacher.z, -0.45f, 0.45f)
        : 0.0f;
'''+base[endbank:]
 s=replace_one(s,'0.12f * verticalRate,\n              -0.25f,','0.12f * verticalRate - 1.20f * forwardSpeedError,\n              -0.25f,')
 s=replace_one(s,'3.00f * headingError - 0.50f * yawRate -',f'{gain:.2f}f * headingError - 0.50f * yawRate -')
 start=s.index('        const float navigationTeacherRequested = !avianNavigationTeacher')
 end=s.index('        // Keep the training-only teacher active through the final target.',start)
 s=s[:start]+'''        float navigationTeacherRequested = avianNavigationTeacher && action >= 15u
            ? studentRequested : teacherRequested;
'''+s[end:]
 s=replace_one(s,'            ? studentRequested : teacherRequested;\n',
   '            ? studentRequested : teacherRequested;\n' +
   '        if (avianNavigationTeacher && float(state.episode.x) * dispatch.timing.x >= 6.0f && (action == 4u || action == 5u)) {\n' +
   f'            const float speedBias = {"clamp(0.30f + 0.8f * (state.commandExtension.z - 0.70f), -0.40f, 0.75f)" if name == "speed-feedback" else "0.55f"};\n' +
   '            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? speedBias : -speedBias), -1.0f, 1.0f);\n' +
   '        }\n' +
   f'        if (avianNavigationTeacher && float(state.episode.x) * dispatch.timing.x >= 6.0f && (action == 2u || action == 3u)) navigationTeacherRequested = clamp(navigationTeacherRequested + {bias:.2f}f, -1.0f, 1.0f);\n')
 needle='''        if (binding.actuator.x != MR_TASK_ACTUATOR_JOINT_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_GRIPPER_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_FLAPPING_POSITION) {'''
 s=replace_one(s,needle,'''        if (avianJourneyTeacher) { teacherActions[actionBase + action] = filtered; }
'''+needle)
 marker='''    if (avianJourneyTakeoffCruise) {
        // Feed the next teacher action from accepted state without a host'''
 injection='''    // Exploratory fixed-command flight bench. Not a deployment controller.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u) {
        const uint mode = environment % 5u;
        const float seconds = float(state.episode.x + 1u) * dispatch.timing.x;
        const float rate = mode == 1u ? 0.03f : mode == 2u ? -0.03f :
                           mode == 3u ? 0.06f : mode == 4u ? -0.06f : 0.0f;
        const float desired = rate * max(seconds - 6.0f, 0.0f);
        const float heading = atan2(yawBasis.y, yawBasis.x);
        const float error = atan2(sin(desired-heading), cos(desired-heading));
        state.commandAndPhase.xyz = float3(0.70f, 0.0f, rate);
        state.commandExtension.xyz = float3(error, baseAngular.z, yawFrameLinear.x);
        state.threatGeometry.x = 0.85f - rootWorldPosition(program, q).z;
        state.threatTeacher.x = yawFrameLinear.y;
        state.threatTeacher.y = atan2(2.0f * (orientation.w * orientation.x + orientation.y * orientation.z),
                                     1.0f - 2.0f * (orientation.x * orientation.x + orientation.y * orientation.y));
        state.threatTeacher.z = baseAngular.x;
    }
'''
 s=replace_one(s,marker,injection+marker)
 (A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/src/metal/LocomotionTask.metal')))
 src=B/(name+'.metal'); src.write_text(s)
 air=B/(name+'.air'); shader=B/(name+'.metallib')
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
 cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15','--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--metallib',str(shader),'--birdflow-navigation-course','training','--envs','32','--steps','1600','--repeats','1','--chunk','1','--physics-substeps','4','--seed','20261111','--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10','--policy-pack',str(POLICY),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json','--state-trace',str(B/(name+'-lane0.tsv')),'--state-trace-environment','0']
 env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(LIB.parent),'PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
 (A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 data=json.loads(p.stdout); first=data['first_episode_navigation']
 samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
 rows=[]
 for lane in range(32):
  rate=[0,.03,-.03,.06,-.06][lane%5]
  ss=[x for x in samples if x['lane']==lane and 600<=x['step']<=first['steps_by_environment'][lane]]
  errs=[]
  for x in ss:
   q=x['q'];v=x['v'];a,b,c,d=q[3:7]
   yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
   target=rate*max(x['step']*.02-6,0)
   err=math.atan2(math.sin(target-yaw),math.cos(target-yaw))
   forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]
   errs.append((abs(err),abs(forward-.7),abs(q[2]-.85)))
  m=[max((e[i] for e in errs),default=None) for i in range(3)]
  passed=(not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==1600 and len(ss)==11 and all(m[i]<=[.15,.20,.20][i] for i in range(3)))
  rows.append({'lane':lane,'rate_rad_s':rate,'physical_failure':first['physical_failure_by_environment'][lane],'steps':first['steps_by_environment'][lane],'sample_count':len(ss),'max_abs_heading_error_rad':m[0],'max_abs_forward_speed_error_m_s':m[1],'max_abs_height_error_m':m[2],'screen_pass':passed})
 h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 report={'arm':name,'argv':cmd,'source_sha256':h(src),'shader_sha256':h(shader),'library_sha256':h(LIB),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'passing_lanes':sum(x['screen_pass'] for x in rows),'sample_spacing_s':2,'scope':'exploratory nominal v15 flight bench, not qualification'}
 (A/(name+'-analysis.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(name,'physical_failures',sum(x['physical_failure'] for x in rows),'passing_lanes',report['passing_lanes'],'failed_sim_steps',report['failed_environment_steps'],flush=True)
