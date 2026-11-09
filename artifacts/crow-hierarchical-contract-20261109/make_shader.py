from pathlib import Path
import difflib,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-hierarchical-contract-20261109'; B=R/'build/crow-hierarchical-contract-20261109';mode=os.environ['CROW_HIERARCHY_MODE']
assert mode in ('oracle','actor','dagger')
base=(R/'build/crow-hierarchical-oracle-20261109/tracking.metal').read_text()
s=base.replace('if (length(float2(xError,yError)) < 0.08f &&','if (landingPhase < 0.5f && length(float2(xError,yError)) < 0.08f &&')
def sub(old,new):
 global s
 assert s.count(old)==1,(s.count(old),old[:160])
 s=s.replace(old,new)
sub('''        const float4 oracleCommand = state.nutrition;''','''        const float4 actorLatent = clamp(float4(
            actionStream[actionBase+0u],actionStream[actionBase+1u],
            actionStream[actionBase+2u],actionStream[actionBase+3u]),-1.0f,1.0f);
        const float4 actorCommand = float4(
            0.10f+0.30f*actorLatent.x,0.30f*actorLatent.y,
            0.15f*actorLatent.z,0.975f+0.325f*actorLatent.w);
        const uint recoveryTakeover = 700u + 150u*(environment % 4u);
        const bool privilegedCommand = HIERARCHY_ORACLE ||
            (dispatch.sampling.w != 0u &&
             (dispatch.assistance.x < 0.5f || (HIERARCHY_RECOVER && state.episode.x >= recoveryTakeover)));
        const float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;''')
sub('''        if (avianJourneyTeacher) { teacherActions[actionBase + action] = filtered; }''','''        if (avianJourneyTeacher) {
            const float4 label = float4(
                clamp((state.nutrition.x-0.10f)/0.30f,-1.0f,1.0f),
                clamp(state.nutrition.y/0.30f,-1.0f,1.0f),
                clamp(state.nutrition.z/0.15f,-1.0f,1.0f),
                clamp((state.nutrition.w-0.975f)/0.325f,-1.0f,1.0f));
            teacherActions[actionBase + action] = action < 4u ? label[action] : 0.0f;
        }''')
sub('''            state.commandAndPhase.xyz=float3(0.15f,desiredLateral,0.0f);''','''            // Policy observations do not expose privileged teacher commands.
            state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);''')
start=s.index('    // Privileged development route teacher; not autonomous policy.')
end=s.index('    if (avianJourneyTakeoffCruise)',start)
block=s[start:end]
old='''    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u) {'''
assert block.count(old)==1
block=block.replace(old,'''    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u &&
        (dispatch.sampling.w != 0u || HIERARCHY_ORACLE)) {''')
block+='''    else if ((program.schedule.w & (1u << 28u)) != 0u) {
        // Accepted proprioceptive feedback and explicit final waypoint goal.
        // The actor, not the route teacher, owns navigation commands.
        const float3 pos = rootWorldPosition(program,q);
        const float heading = atan2(yawBasis.y,yawBasis.x);
        const float pitch = asin(clamp(2.0f*(orientation.w*orientation.y-orientation.z*orientation.x),-1.0f,1.0f));
        const float roll = atan2(2.0f*(orientation.w*orientation.x+orientation.y*orientation.z),1.0f-2.0f*(orientation.x*orientation.x+orientation.y*orientation.y));
        float phase=state.threatTeacher.w;
        if (state.navigation.z >= float(kCrowNavigationWaypointCount)) {
            phase=max(phase,1.0f);
            uint course=MR_INVALID_INDEX;
            for (uint i=0u;i<program.counts1.w;++i) {
                const MRTaskRewardOperatorGPU operation=rewards[i];
                if (operation.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS ||
                    operation.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) {course=operation.source.z;break;}
            }
            if (course!=MR_INVALID_INDEX) {
                // The fifth waypoint is an explicit goal observation; no
                // obstacle collision geometry or teacher route is consulted.
                const float3 finalGoal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                uint loadedFeet=0u;
                for (uint gi=0u;gi<program.counts0.w;++gi) {
                    const MRTaskContactGroupGPU group=contactGroups[gi];
                    if ((group.members.z & MR_TASK_CONTACT_SUPPORT)!=0u &&
                        compactContact[compactBase+group.members.w]>0.05f) ++loadedFeet;
                }
                if (loadedFeet>=2u && abs(finalGoal.x-pos.x)<0.06f &&
                    abs(finalGoal.y-pos.y)<0.16f && length(yawFrameLinear.xy)<0.08f &&
                    abs(rootLinearVelocity.z)<0.12f) phase=2.0f;
            }
        }
        state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);
        state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x);
        state.threatGeometry=float4(-pos.z,rootLinearVelocity.z,pitch,baseAngular.y);
        state.threatTeacher=float4(yawFrameLinear.y,roll,baseAngular.x,phase);
    }
'''
s=s[:start]+block+s[end:]
sub('''        } else if (routeComponent < 6u &&
            waypoint < kCrowNavigationWaypointCount) {
            const uint requested = routeComponent < 3u
                ? waypoint
                : min(waypoint + 1u,
                      kCrowNavigationWaypointCount - 1u);''','''        } else if (routeComponent < 6u &&
            (waypoint < kCrowNavigationWaypointCount ||
             ((program.schedule.w & (1u << 28u)) != 0u &&
              waypoint == kCrowNavigationWaypointCount))) {
            const uint visibleWaypoint = min(waypoint,kCrowNavigationWaypointCount-1u);
            const uint requested = routeComponent < 3u
                ? visibleWaypoint
                : min(visibleWaypoint + 1u,
                      kCrowNavigationWaypointCount - 1u);''')
s=('#define HIERARCHY_ORACLE '+('true' if mode=='oracle' else 'false')+'\n#define HIERARCHY_RECOVER '+('true' if mode=='dagger' else 'false')+'\n')+s
src=B/(mode+'.metal');src.write_text(s)
(A/(mode+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/oracle-tracking.metal',tofile='b/hierarchical-'+mode+'.metal')))
air=B/(mode+'.air');shader=B/(mode+'.metallib')
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/(mode+'-shader.json')).write_text(json.dumps({'mode':mode,'source_sha256':h(src),'shader_sha256':h(shader),'frozen_controller_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
