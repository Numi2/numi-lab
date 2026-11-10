from pathlib import Path
import difflib,subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-support-debounced-landing-20261110"; B=R/"build/crow-v15-support-debounced-landing-20261110"; F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-ramped-perch-approach-20261110/teacher.metal").read_text(); s=old
changes=[("if (landingPhase > 0.5f && loadedFeet >= 2u &&", "if (landingPhase >= 1.5f && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;\n                const uint supportDuration = (state.status.w >> 25u) & 127u;\n                if (landingPhase > 0.5f && landingPhase < 1.5f && loadedFeet >= 2u && supportDuration >= 25u &&"), ("// Accepted contact resultant and articulated COM supply a bounded", "if (state.navigation.z>=5.0f && state.threatTeacher.w<1.5f) {\n                const uint prior=(state.status.w>>25u)&127u;\n                const bool supported=state.nutrition.x>0.10f && state.nutrition.y>0.10f && state.nutrition.x+state.nutrition.y>2.2f;\n                const uint duration=supported ? min(prior+1u,100u) : 0u;\n                packed |= duration<<25u;\n            }\n            // Accepted contact resultant and articulated COM supply a bounded")]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(B/"teacher.metal").write_text(s);(A/"teacher.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher-stop.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
