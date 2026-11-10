from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-route-cross-behind-20261110";B=R/"build/crow-v15-route-cross-behind-20261110";base=(R/"build/crow-v15-route-side-clearance-20261110/sideclear.metal").read_text()
s0=base
needle="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
"""
assert s0.count(needle)==1
old="""            if (!landing && wp<=2u && pos.x>target.x+0.10f) {
                // Retreat on the current side before attempting the missed gate.
                const float3 missedObstacle=sceneState[sceneBase+course+1u+wp].position.xyz;
                const float side=pos.y>=missedObstacle.y ? 1.0f : -1.0f;
                const float lateralClearance=side*(pos.y-missedObstacle.y);
                const float safeY=missedObstacle.y+side*0.95f;
                desiredForward=lateralClearance<0.85f ? 0.10f : -0.05f;
                desiredLateral=clamp(0.8f*yawBasis.x*(safeY-pos.y),-0.30f,0.30f);
            }"""
new="""            if (!landing && wp>=1u && wp<=2u) {
                const float3 missedObstacle=sceneState[sceneBase+course+1u+wp].position.xyz;
                const float side=pos.y>=missedObstacle.y ? 1.0f : -1.0f;
                const bool wrongSide=(pos.y-missedObstacle.y)*(target.y-missedObstacle.y)<0.0f;
                const bool retreat=pos.x>(wrongSide ? target.x-0.75f : target.x+0.10f);
                if (retreat) {
                    const float lateralClearance=side*(pos.y-missedObstacle.y);
                    const float safeY=missedObstacle.y+side*0.95f;
                    desiredForward=lateralClearance<0.85f ? 0.10f : -0.05f;
                    desiredLateral=clamp(0.8f*yawBasis.x*(safeY-pos.y),-0.30f,0.30f);
                } else if (wrongSide) {
                    // Cross the missed obstacle's line only while safely upstream.
                    desiredForward=0.10f;
                    desiredLateral=clamp(0.8f*yawBasis.x*(target.y-pos.y),-0.30f,0.30f);
                }
            }"""
assert s0.count(old)==1
s0=s0.replace(old,new)
s=s0;name="crossbehind";src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/velocity08c.metal",tofile="b/supportgate.metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
