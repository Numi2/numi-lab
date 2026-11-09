from pathlib import Path
import difflib,gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd(); seed='20261124'; A=R/'artifacts/crow-landing-vertical-damping-20261109'; B=R/'build/crow-landing-vertical-damping-20261109'; B.mkdir(parents=True,exist_ok=True)
LIB=R/'build/crow-support-observer-20261109/libmetalrobo.dylib'
POLICY=R/'build/crow-extended-flight-20261109/collision-on/extended.policypack'
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
sub('const float speedBias = 0.55f;','const float speedBias = clamp(0.55f + 1.40f * (state.commandExtension.z - max(state.commandAndPhase.x, 0.15f)), 0.25f, 0.95f);')
sub('state.threatGeometry.x = targetHeight - rootWorldPosition(program, q).z;','''const float benchPitch = asin(clamp(2.0f * (orientation.w * orientation.y - orientation.z * orientation.x), -1.0f, 1.0f));
        state.threatGeometry = float4(targetHeight - rootWorldPosition(program, q).z, rootLinearVelocity.z, benchPitch, baseAngular.y);''')
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
sub('inline uint taskActionHoldPeriod', '#include "CrowFlightControllerV1.metal"\n\ninline uint taskActionHoldPeriod')
sub('''            ? birdFlowJourneyTeacherAction(
                  action,
                  state,
                  dispatch.timing.x,
                  (program.schedule.w &
                   MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u
              )
            : 0.0f;''','''            ? (fixedFlightController
               ? crowFlightControllerV1Action(action, state, dispatch.timing.x)
               : birdFlowJourneyTeacherAction(
                   action, state, dispatch.timing.x,
                   (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u))
            : 0.0f;''')
sub('''        if (avianNavigationTeacher && float(state.episode.x) * dispatch.timing.x >= 6.0f && (action == 4u || action == 5u)) {
            const float speedBias = clamp(0.55f + 1.40f * (state.commandExtension.z - max(state.commandAndPhase.x, 0.15f)), 0.25f, 0.95f);
            const float yawDemand = clamp(16.0f * state.commandExtension.x - 0.50f * state.commandExtension.y - 0.35f * state.threatTeacher.x, -1.0f, 1.0f);
            const float phaseOffset = 0.60f * clamp(-yawDemand, 0.0f, 1.0f) * sin(state.commandAndPhase.w);
            navigationTeacherRequested = clamp(navigationTeacherRequested + (action == 4u ? speedBias : -speedBias) + phaseOffset, -1.0f, 1.0f);
        }
''','')
# Discard the perturbed-start diagnostic injection for full-start route trials.
start=s.index('        // Diagnostic reset perturbations;')
end=s.index('        const bool projectileEpisode = randomUnit(',start)
s=s[:start]+s[end:]
start=s.index('    // Exploratory fixed-command flight bench.')
end=s.index('    if (avianJourneyTakeoffCruise)',start)
s=s[:start]+'''    // Privileged development route teacher; not autonomous policy.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u) {
        uint course = MR_INVALID_INDEX;
        for (uint i=0u; i<program.counts1.w; ++i) {
            const MRTaskRewardOperatorGPU operation = rewards[i];
            if (operation.source.x == MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS ||
                operation.source.x == MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) {
                course = operation.source.z;
                break;
            }
        }
        const uint wp = min(uint(max(state.navigation.z,0.0f)), kCrowNavigationWaypointCount-1u);
        if (course != MR_INVALID_INDEX) {
            const float3 target = crowNavigationWaypointTarget(sceneState + sceneBase, course, wp);
            const float3 pos = rootWorldPosition(program,q);
            float3 lateralTarget = target;
            if (wp >= 1u && wp <= 3u) {
                const float3 previous = crowNavigationWaypointTarget(sceneState + sceneBase, course, wp-1u);
                const float transition = smoothstep(previous.x+(wp==2u ? -0.16f : wp==3u ? 0.20f : 0.35f), previous.x+(wp==2u ? -0.01f : wp==3u ? 0.35f : 0.50f), pos.x);
                lateralTarget.y = mix(previous.y, target.y, transition);
            }
            const float3 delta = lateralTarget-pos;
            const float localLateral = -yawBasis.y*delta.x + yawBasis.x*delta.y;
            const float lateralCommand = clamp(0.80f*localLateral,-0.30f,0.30f);
            const float heightTarget = clamp(target.z+0.30f,1.15f,1.30f);
            const float heading = atan2(yawBasis.y,yawBasis.x);
            const float pitch = asin(clamp(2.0f*(orientation.w*orientation.y-orientation.z*orientation.x),-1.0f,1.0f));
            const float roll = atan2(2.0f*(orientation.w*orientation.x+orientation.y*orientation.z),1.0f-2.0f*(orientation.x*orientation.x+orientation.y*orientation.y));
            const bool landing = state.navigation.z >= float(kCrowNavigationWaypointCount);
            float landingPhase = state.threatTeacher.w;
            float desiredLateral = lateralCommand;
            float desiredHeight = heightTarget;
            float errorOffset = 0.0f;
            if (landing) {
                const float3 perch = sceneState[sceneBase+course+4u].position.xyz;
                const float3 landingTarget = perch + float3(0.0f,0.55f,0.0f);
                const float xError = landingTarget.x-pos.x;
                const float yError = landingTarget.y-pos.y;
                const float commandedForward = clamp(0.45f*xError,-0.20f,0.20f);
                desiredLateral = clamp(0.80f*yError,-0.30f,0.30f);
                errorOffset = 0.25f-commandedForward;
                if (length(float2(xError,yError)) < 0.08f &&
                    length(yawFrameLinear.xy) < 0.08f) landingPhase=1.0f;
                uint loadedFeet=0u;
                for (uint gi=0u; gi<program.counts0.w; ++gi) {
                    const MRTaskContactGroupGPU group=contactGroups[gi];
                    if ((group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u &&
                        compactContact[compactBase+group.members.w] > 0.5f) ++loadedFeet;
                }
                if (landingPhase > 0.5f && loadedFeet >= 2u &&
                    abs(xError) < 0.06f && abs(yError) < 0.16f &&
                    length(yawFrameLinear.xy) < 0.08f &&
                    abs(rootLinearVelocity.z) < 0.12f) landingPhase=2.0f;
                desiredHeight = landingPhase > 0.5f ? clamp(perch.z+0.34f-0.30f*rootLinearVelocity.z,perch.z+0.30f,perch.z+0.38f) : 1.30f;
            }
            state.commandAndPhase.xyz=float3(0.15f,desiredLateral,0.0f);
            state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x+errorOffset);
            state.threatGeometry=float4(desiredHeight-pos.z,rootLinearVelocity.z,pitch,baseAngular.y);
            state.threatTeacher=float4(yawFrameLinear.y-desiredLateral,roll,baseAngular.x,landingPhase);
        }
    }
'''+s[end:]
sub('''        const float teacherRequested =
            avianJourneyTeacher || fixedFlightController || approachSupervisorBlend > 0.0f''','''        MRTaskStateGPU carrierState = state;
        carrierState.episode.x = min(carrierState.episode.x, 1500u);
        const float teacherRequested =
            avianJourneyTeacher || fixedFlightController || approachSupervisorBlend > 0.0f''')
sub('crowFlightControllerV1Action(action, state, dispatch.timing.x)','crowFlightControllerV1Action(action, carrierState, dispatch.timing.x)')
sub('''        const float navigationTeacherBlend = avianNavigationTeacher''','''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
        const float navigationTeacherBlend = avianNavigationTeacher''')
sub('''        const float avianWingAmplitude =
            avianGroundCurriculum &&''','''        const float avianWingAmplitude =
            fixedFlightController && state.threatTeacher.w >= 1.5f &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f
            : avianGroundCurriculum &&''')
src.write_text(s); (A/'tracking.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/src/metal/LocomotionTask.metal')))
air=B/'tracking.air'; shader=B/'tracking.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
cmd=json.loads((R/'artifacts/crow-flight-envelope-20261109/yaw-negative-analysis.json').read_text())['argv']
for key,val in [('--metallib',str(shader)),('--policy-pack',str(POLICY)),('--state-trace',str(B/'tracking-lane0.tsv')),('--envs','64'),('--steps','3000'),('--seed',seed),('--state-trace-environment','1')]:cmd[cmd.index(key)+1]=val
cmd.remove('--birdflow-journey-teacher')
authority=cmd.index('--birdflow-journey-student-authority'); del cmd[authority:authority+2]
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','DYLD_LIBRARY_PATH':str(LIB.parent),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480); (A/'tracking.stdout').write_text(p.stdout)
with (A/'tracking.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('return='+str(p.returncode)+'\n'+p.stderr[-5000:])

data=json.loads(p.stdout); first=data['first_episode_navigation']
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
contacts=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
rows=[]
for lane in range(64):
 fail=first['physical_failure_by_environment'][lane]; end=first['steps_by_environment'][lane]
 ss=[x for x in samples if x['lane']==lane and x['step']<=end]
 last=ss[-1] if ss else None
 cc=[x for x in contacts if x['lane']==lane and x['step']<=end+1]
 rows.append({'lane':lane,'steps':end,'maximum_waypoints':first['maximum_waypoints_by_environment'][lane],'completion_seen':first['completion_seen_by_environment'][lane],'physical_failure':fail,'first_completion_step':first['first_completion_step_by_environment'][lane],'last_sample':last,'first_contact_record':cc[0] if cc else None})
dist={str(i):sum(x['maximum_waypoints']==i for x in rows) for i in range(6)}
passed=(sum(x['maximum_waypoints']>=5 for x in rows)==64 and not any(x['physical_failure'] for x in rows) and data['failed_environment_steps']==0)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'argv':cmd,'source_sha256':h(src),'frozen_controller_sha256':h(R/'src/metal/CrowFlightControllerV1.metal'),'shader_sha256':h(shader),'library_sha256':h(LIB),'policy_sha256':h(POLICY),'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'waypoint_distribution':dist,'lanes':rows,'screen_pass':passed,'scope':'privileged route teacher, obstacle-collision development control'}
(A/'route-results.json').write_text(json.dumps(report,indent=2)+'\n'); print('waypoint distribution',dist,'physical failures',sum(x['physical_failure'] for x in rows),'failed sim',data['failed_environment_steps'],'screen',passed,flush=True)

support=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
per=[]
for lane in range(64):
 end=first['steps_by_environment'][lane]; samples=[x for x in support if x['lane']==lane and x['step']<=end]
 qualifying=[]
 for z in samples:
  feet=[c for c in z['contacts'] if ('foot' in c['a_name'] or 'foot' in c['b_name'])]
  nonfeet=[c for c in z['contacts'] if not ('foot' in c['a_name'] or 'foot' in c['b_name'])]
  upward=sum(((-1 if 'foot' in c['a_name'] else 1)*c['normal'][2]*c['impulses'][0]) for c in feet)
  speed=math.hypot(z['v'][0],z['v'][1])
  if feet and not nonfeet and upward>0 and speed<=.12 and abs(z['v'][2])<=.12: qualifying.append(z['step'])
 longest=0;current=0;last=None
 for t in qualifying:
  current=current+1 if last is not None and t==last+1 else 1
  longest=max(longest,current);last=t
 per.append({'lane':lane,'contact_samples':len(samples),'foot_samples':sum(any('foot' in c['a_name'] or 'foot' in c['b_name'] for c in z['contacts']) for z in samples),'longest_qualifying_contact_run':longest,'candidate_supported_landing':longest>=25 and not first['physical_failure_by_environment'][lane]})
out={'lanes':per,'candidate_supported_landings':sum(x['candidate_supported_landing'] for x in per),'support_samples':len(support),'scope':'last-substep diagnostic samples, not continuous-contact proof'}
(A/'support-results.json').write_text(json.dumps(out,indent=2)+'\n')
print('support samples',len(support),'candidate landings',out['candidate_supported_landings'],'longest',max(x['longest_qualifying_contact_run'] for x in per),flush=True)
