from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-route-retreat-clearance-20261110";B=R/"build/crow-v15-route-retreat-clearance-20261110";base=(R/"build/crow-v15-route-retreat-recovery-20261110/retreat.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="            state.nutrition=float4(desiredForward,desiredLateral,-heading,desiredHeight);"
new="""            if (!landing && wp==2u && pos.x>target.x-1.0f && abs(pos.y-target.y)>0.30f)
                desiredForward=0.10f; // effective hover while clearing the second slalom.
            state.nutrition=float4(desiredForward,desiredLateral,-heading,desiredHeight);"""
assert s0.count(old)==1
s0=s0.replace(old,new)
s=s0;name="clearance";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
