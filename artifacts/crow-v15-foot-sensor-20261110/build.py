from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-foot-sensor-20261110';B=R/'build/crow-v15-foot-sensor-20261110';base=(R/'build/crow-v15-causal-route-20261110/route.metal').read_text()
needle='            state.status.w=packed;\n'
assert base.count(needle)==1
sensor="""            // Diagnostic only: millimetre-quantized foot midpoint to explicit goal.
            if (state.navigation.z>=5.0f) {
                uint course=MR_INVALID_INDEX;
                for(uint i=0u;i<program.counts1.w;++i) {
                    const MRTaskRewardOperatorGPU op=rewards[i];
                    if(op.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS || op.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) { course=op.source.z; break; }
                }
                if(course!=MR_INVALID_INDEX) {
                    const float3 goal=crowNavigationWaypointTarget(sceneState+sceneBase,course,4u);
                    const float footX=0.5f*(bodyStates[bodyBase+contactGroups[0].reference.x].position.x+bodyStates[bodyBase+contactGroups[1].reference.x].position.x);
                    const int mm=int(round(clamp(goal.x-footX,-0.5f,0.5f)*1000.0f));
                    packed |= uint(mm+512) << 13u;
                }
            }
            state.status.w=packed;
"""
s=base.replace(needle,sensor);(B/'route.metal').write_text(s);(A/'sensor.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/causal-route.metal',tofile='b/sensor.metal')))
air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'route.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
