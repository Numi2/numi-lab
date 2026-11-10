from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-perch-toe-debounce-20261110";B=R/"build/crow-v15-perch-toe-debounce-20261110";base=(R/"build/crow-v15-perch-toe-damped-20261110/velocity08c.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="const uint rearNext=(footLoad>0.05f || rearPrevious>0u) ? min(rearPrevious+1u,43u) : 0u;"
new="const bool bilateral=state.nutrition.x>0.05f && state.nutrition.y>0.05f;\n                    const uint rearNext=(rearPrevious>=5u || bilateral) ? min(rearPrevious+1u,48u) : 0u;"
assert s0.count(old)==1
s0=s0.replace(old,new)
old2="const uint next=loaded ? min(previous+1u,48u) : (previous>0u ? previous-1u : 0u);"
new2="const uint next=rearNext>=5u ? min(previous+1u,48u) : 0u;"
assert s0.count(old2)==1
s0=s0.replace(old2,new2)
old3="min(0.020f*float(rearRamp),0.85f)"
new3="min(0.020f*float(rearRamp>5u ? rearRamp-5u : 0u),0.85f)"
assert s0.count(old3)==1
s0=s0.replace(old3,new3)
old="            // Experimental target marker never present"
new=old
s=s0;name="debounce";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
