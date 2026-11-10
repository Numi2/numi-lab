from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-route-teacher-earlier-turn-20261110";B=R/"build/crow-v15-route-teacher-earlier-turn-20261110";base=(R/"build/crow-v15-route-teacher-signal-fix-20261110/teacherfix.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="wp==3u ? 0.20f : 0.20f"
new="wp==3u ? -0.10f : 0.20f"
assert s0.count(old)==1
s0=s0.replace(old,new)
old="wp==3u ? 0.35f : 0.35f"
new="wp==3u ? 0.05f : 0.35f"
assert s0.count(old)==1
s0=s0.replace(old,new)
s=s0;name="earlier";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
