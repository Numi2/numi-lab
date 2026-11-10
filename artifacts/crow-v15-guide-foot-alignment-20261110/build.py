from pathlib import Path
import difflib, subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-guide-foot-alignment-20261110";B=R/"build/crow-v15-guide-foot-alignment-20261110";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-toe-height-bracket-20261110/teacher.metal").read_text()
a="const float xError = landingTarget.x-pos.x-0.04f;"
b="""float feetX = pos.x;
                if (program.counts0.w >= 2u) {
                    const uint lf=contactGroups[0].reference.x, rf=contactGroups[1].reference.x;
                    if (lf>=robotFirst && lf<robotEnd && rf>=robotFirst && rf<robotEnd) {
                        const MRBodyStateGPU left=bodyStates[bodyBase+lf], right=bodyStates[bodyBase+rf];
                        const float3 lj=left.position.xyz+rotate(left.orientation,contactGroups[0].kinematicReference.xyz);
                        const float3 rj=right.position.xyz+rotate(right.orientation,contactGroups[1].kinematicReference.xyz);
                        feetX=0.5f*(lj.x+rj.x);
                    }
                }
                const float xError = landingTarget.x+0.045f-feetX;"""
assert old.count(a)==1
s=old.replace(a,b);(B/"teacher.metal").write_text(s);(A/"teacher.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
