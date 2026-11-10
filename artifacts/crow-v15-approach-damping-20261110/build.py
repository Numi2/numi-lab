from pathlib import Path
import difflib,subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-approach-damping-20261110"; B=R/"build/crow-v15-approach-damping-20261110"; F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-teacher-stop-approach-20261110/teacher.metal").read_text(); s=old
changes=[("const float commandedForward = clamp(0.45f*xError,-0.20f,0.20f);", "const float commandedForward = clamp(0.25f*xError-0.80f*yawFrameLinear.x,-0.10f,0.20f);")]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(B/"teacher.metal").write_text(s);(A/"teacher.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher-stop.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
