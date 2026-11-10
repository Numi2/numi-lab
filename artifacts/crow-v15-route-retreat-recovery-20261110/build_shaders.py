from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-route-retreat-recovery-20261110";B=R/"build/crow-v15-route-retreat-recovery-20261110";base=(R/"build/crow-v15-route-teacher-recovery-20261110/recovery.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="state.nutrition=float4(0.25f-errorOffset,desiredLateral,-heading,desiredHeight);"
new="""float desiredForward=0.25f-errorOffset;
            if (!landing && wp<=2u && pos.x>target.x+0.10f) {
                // Retreat on the current side before attempting the missed gate.
                desiredForward=-0.05f;
                desiredLateral=0.0f;
            }
            state.nutrition=float4(desiredForward,desiredLateral,-heading,desiredHeight);"""
assert s0.count(old)==1
s0=s0.replace(old,new)
s=s0;name="retreat";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
