from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/"artifacts/crow-v15-student-release-candidate-20261110";B=R/"build/crow-v15-student-release-candidate-20261110";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110";base=(R/"build/crow-v15-bounded-rebound-20261110/teacher.metal").read_text();s=base
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
for old,new in changes:
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
(B/"teacher.metal").write_text(s);(A/"student-release.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/student.metal",tofile="b/student.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True);other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air","MetalRoboCollision.air"));subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air"),str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
