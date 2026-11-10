from pathlib import Path
import difflib,subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-load-responsive-wing-20261110"; B=R/"build/crow-v15-load-responsive-wing-20261110"; F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-toe-height-bracket-20261110/teacher.metal").read_text(); s=old
changes=[("const uint supportDuration = (state.status.w >> 25u) & 127u;", "if (landingPhase >= 1.0f && landingPhase < 1.5f) {\n                    const float previousSupport=clamp((landingPhase-1.0f)*2.5f,0.0f,1.0f);\n                    const float measuredSupport=clamp((state.nutrition.x+state.nutrition.y)/4.41f,0.0f,1.0f);\n                    landingPhase=1.0f+0.4f*mix(previousSupport,measuredSupport,0.10f);\n                }\n                const uint supportDuration = (state.status.w >> 25u) & 127u;"), ("binding.drive.z + binding.drive.w * avianWingPolicyCommand,\n                0.0f,\n                1.0f\n            );", "binding.drive.z + binding.drive.w * avianWingPolicyCommand,\n                0.0f,\n                1.0f\n            ) * (fixedFlightController && state.navigation.z>=5.0f && state.threatTeacher.w>=1.0f && state.threatTeacher.w<1.5f && binding.actuator.x==MR_TASK_ACTUATOR_FLAPPING_POSITION ? 1.0f-0.65f*clamp((state.threatTeacher.w-1.0f)*2.5f,0.0f,1.0f) : 1.0f);")]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(B/"teacher.metal").write_text(s);(A/"teacher.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher-stop.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
