from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/"artifacts/crow-v15-student-landing-controller-20261110";B=R/"build/crow-v15-student-landing-controller-20261110";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110";base=(R/"build/crow-v15-bounded-rebound-20261110/teacher.metal").read_text();s=base
changes=[("const bool closing=state.navigation.z>=5.0f;","const bool closing=state.navigation.z>=5.0f && state.threatTeacher.w<2.5f;"),("state.navigation.z>=5.0f && state.threatTeacher.w<1.5f","state.navigation.z>=5.0f && state.threatTeacher.w<2.5f"),("if (state.navigation.z >= float(kCrowNavigationWaypointCount) && loadedFeet>=2u && abs(state.nutrition.z)<0.045f &&\n                    abs(finalGoal.y-pos.y)<0.16f && length(yawFrameLinear.xy)<0.08f &&\n                    abs(rootLinearVelocity.z)<0.12f) phase=2.0f;","""if (phase >= 1.5f && state.nutrition.x+state.nutrition.y < 0.25f) phase=1.0f;
                if (state.navigation.z >= float(kCrowNavigationWaypointCount) && phase<2.5f && loadedFeet>=2u && abs(state.nutrition.z)<0.045f &&
                    abs(finalGoal.y-pos.y)<0.16f && length(yawFrameLinear.xy)<0.08f &&
                    abs(rootLinearVelocity.z)<0.12f) phase=2.0f;
                if (state.navigation.z >= float(kCrowNavigationWaypointCount)) {
                    if (phase < 1.5f) state.threatMetadata.x=0u;
                    else if (phase < 2.5f) {
                        if (state.threatMetadata.x==0u || state.threatMetadata.x==MR_INVALID_INDEX) state.threatMetadata.x=state.episode.x;
                        if (state.episode.x >= state.threatMetadata.x+200u && state.nutrition.x>0.10f && state.nutrition.y>0.10f && state.nutrition.x+state.nutrition.y>2.2f) phase=3.0f;
                    }
                }""")]
changes.extend([
("float2 accumulated = pass.physicsSubstep == 0u ? float2(0.0f) : as_type<float2>(state.threatMetadata.zw);", """float2 accumulated = pass.physicsSubstep == 0u ? float2(0.0f) : as_type<float2>(state.threatMetadata.zw);
        if (state.threatMetadata.z == MR_INVALID_INDEX) accumulated.x=0.0f;
        if (state.threatMetadata.w == MR_INVALID_INDEX) accumulated.y=0.0f;"""),
("state.nutrition.xy=state.navigation.z>=5.0f ? as_type<float2>(state.threatMetadata.zw)/dispatch.timing.x : float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);", """state.nutrition.xy=state.navigation.z>=5.0f ? as_type<float2>(state.threatMetadata.zw)/dispatch.timing.x : float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);
        if (state.navigation.z>=5.0f && state.threatMetadata.z == MR_INVALID_INDEX) state.nutrition.x=0.0f;
        if (state.navigation.z>=5.0f && state.threatMetadata.w == MR_INVALID_INDEX) state.nutrition.y=0.0f;""")])
changes.extend([
("""        if (fixedFlightController && state.episode.z == 10u && float(state.episode.x)*dispatch.timing.x < 16.0f) { actorCommand.x = 0.70f; actorCommand.y = 0.0f; actorCommand.z = 0.0f; actorCommand.w = 0.30f; }""", """        if (fixedFlightController && dispatch.sampling.w==0u && state.navigation.z>=5.0f) {
            // Sensor-driven terminal control, bounded to the qualified flight envelope.
            actorCommand.x=clamp(0.04f+0.40f*state.nutrition.z-0.40f*state.commandExtension.z,-0.08f,0.10f);
            actorCommand.y=state.commandAndPhase.y;
            actorCommand.z=state.commandExtension.x;
            actorCommand.w=state.nutrition.w;
        }
        if (fixedFlightController && state.episode.z == 10u && float(state.episode.x)*dispatch.timing.x < 16.0f) { actorCommand.x = 0.70f; actorCommand.y = 0.0f; actorCommand.z = 0.0f; actorCommand.w = 0.30f; }"""),
("        float phase=state.threatTeacher.w;\n        if (state.navigation.z >= float(kCrowNavigationWaypointCount-1u))", "        float phase=state.threatTeacher.w;\n        float landingLateral=0.0f;\n        if (state.navigation.z >= float(kCrowNavigationWaypointCount-1u))"),
("state.nutrition.z=clamp(finalGoal.x+0.015f-feetX,-0.50f,0.50f);", "state.nutrition.z=clamp(finalGoal.x+(state.navigation.z>=5.0f ? 0.020f : 0.015f)-feetX,-0.50f,0.50f);"),
("""                if (state.navigation.z >= float(kCrowNavigationWaypointCount) && phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;""", """                if (state.navigation.z >= float(kCrowNavigationWaypointCount)) {
                    const float xError=state.nutrition.z;
                    const float yError=finalGoal.y-pos.y;
                    landingLateral=clamp(0.80f*yError,-0.30f,0.30f);
                    if (phase < 0.5f && length(float2(xError,yError))<0.08f && length(yawFrameLinear.xy)<0.08f) phase=1.0f;
                    if (phase >= 1.2f && phase < 1.5f && (length(yawFrameLinear.xy)>0.16f || (pos.z<finalGoal.z+0.05f && rootLinearVelocity.z < -0.25f) || (pos.z>finalGoal.z+0.30f && rootLinearVelocity.z>0.10f))) phase=1.0f;
                    if (phase>0.5f && phase<1.2f && state.nutrition.x>0.05f && state.nutrition.y>0.05f && abs(xError)<0.04f && abs(yError)<0.10f && length(yawFrameLinear.xy)<0.15f && abs(rootLinearVelocity.z)<0.90f) phase=1.25f;
                    const bool canLower=abs(xError)<0.040f && abs(yError)<0.12f && length(yawFrameLinear.xy)<0.11f && abs(rootLinearVelocity.z)<0.18f;
                    const bool footContact=state.nutrition.x+state.nutrition.y>0.20f;
                    const float priorHeight=state.nutrition.w>0.5f ? state.nutrition.w : 1.30f;
                    state.nutrition.w=phase>0.5f ? (footContact ? max(priorHeight,pos.z) : (canLower ? max(finalGoal.z+0.105f,priorHeight-0.12f*dispatch.timing.x) : priorHeight)) : 1.30f;
                }"""),
("        state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);\n        state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x);", "        state.commandAndPhase.xyz=float3(0.15f,state.navigation.z>=5.0f ? landingLateral : 0.0f,0.0f);\n        state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x);")])
for old,new in changes:
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
(B/"teacher.metal").write_text(s);(A/"student-landing.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/student.metal",tofile="b/student.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True);other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air","MetalRoboCollision.air"));subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air"),str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
