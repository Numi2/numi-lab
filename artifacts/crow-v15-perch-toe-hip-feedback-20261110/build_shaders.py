from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-perch-toe-hip-feedback-20261110";B=R/"build/crow-v15-perch-toe-hip-feedback-20261110";base=(R/"build/crow-v15-perch-toe-velocity-20261110/velocity08.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
for name,kp,kd in (("hip3","3.0","0.25"),("hip6","6.0","0.35")):
 extra="""        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f &&
            state.threatTeacher.w>=1.5f && (state.status.w&63u)>=28u && ((state.status.w>>6u)&63u)>=28u && (action==7u || action==9u || action==10u || action==12u) &&
            ((action==7u || action==9u) ? state.nutrition.x : state.nutrition.y)>0.05f) {
            const uint encoded=(state.status.w>>25u)&127u;
            if(encoded!=0u) {
                const float supportError=(float(encoded)-64.0f)*0.002f;
                const float correction=clamp("""+kp+"""f*supportError-"""+kd+"""f*state.threatGeometry.w,-0.30f,0.30f);
                navigationTeacherRequested=(action==9u || action==12u) ? -1.077f*correction : correction;
            }
        }
"""
 s=s0.replace(needle,needle+extra);src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/toe-ramp.metal",tofile="b/"+name+".metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
