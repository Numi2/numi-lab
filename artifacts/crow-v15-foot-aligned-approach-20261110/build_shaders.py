from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-foot-aligned-approach-20261110";B=R/"build/crow-v15-foot-aligned-approach-20261110";base=(R/"build/crow-v15-perch-toe-damped-20261110/velocity08c.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="""        const float4 actorCommand = float4(
            0.10f+0.30f*actorLatent.x,0.30f*actorLatent.y,
            0.15f*actorLatent.z,0.975f+0.325f*actorLatent.w);"""
new="""        float4 actorCommand = float4(
            0.10f+0.30f*actorLatent.x,0.30f*actorLatent.y,
            0.15f*actorLatent.z,0.975f+0.325f*actorLatent.w);
        if (fixedFlightController && dispatch.sampling.w==0u &&
            state.navigation.z>=5.0f && state.threatTeacher.w<1.5f) {
            // Bounded approach speed from accepted foot-to-goal error.
            actorCommand.x=clamp(0.10f+1.0f*state.nutrition.z,-0.05f,0.30f);
        }"""
assert s0.count(old)==1
s0=s0.replace(old,new)
old="""                const float3 finalGoal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                if (phase < 0.5f"""
new="""                const float3 finalGoal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                if (dispatch.sampling.w==0u && program.counts0.w>=2u) {
                    const uint lf=contactGroups[0].reference.x,rf=contactGroups[1].reference.x;
                    if(lf>=robotFirst && lf<robotEnd && rf>=robotFirst && rf<robotEnd) {
                        const float feetX=0.5f*(bodyStates[bodyBase+lf].position.x+bodyStates[bodyBase+rf].position.x);
                        state.nutrition.z=clamp(finalGoal.x+0.015f-feetX,-0.25f,0.25f);
                    }
                }
                if (phase < 0.5f"""
assert s0.count(old)==1
s0=s0.replace(old,new)
s=s0;name="footalign";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
