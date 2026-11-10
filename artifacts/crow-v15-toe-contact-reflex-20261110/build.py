from pathlib import Path
import difflib,subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-toe-contact-reflex-20261110"; B=R/"build/crow-v15-toe-contact-reflex-20261110"; F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-perch-trim-20261110/teacher.metal").read_text(); s=old
changes=[("max(perch.z+0.23f,state.nutrition.w-0.025f*dispatch.timing.x)", "max(perch.z+0.28f,state.nutrition.w-0.025f*dispatch.timing.x)"), ("min(rearPrevious+1u,43u) : (rearPrevious>0u ? rearPrevious-1u : 0u)", "min(rearPrevious+6u,43u) : (rearPrevious>3u ? rearPrevious-3u : 0u)"), ("(footLoad>0.05f && rearPrevious>=12u)", "(footLoad>0.05f && rearPrevious>=1u)"), ("min(previous+1u,48u) : (previous>0u ? previous-1u : 0u)", "min(previous+6u,48u) : (previous>3u ? previous-3u : 0u)")]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(B/"teacher.metal").write_text(s);(A/"teacher.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher-stop.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
