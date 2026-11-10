from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-staged-descent-20261110';B=R/'build/crow-v15-staged-descent-20261110';base=(R/'build/crow-v15-grip-latch-20261110/latch.metal').read_text();s=base
old='''            actorCommand.x=clamp(0.00f+0.80f*state.nutrition.z-0.80f*state.commandExtension.z,-0.15f,0.20f);
        }
        const bool privilegedCommand'''
new='''            actorCommand.x=clamp(0.00f+0.80f*state.nutrition.z-0.80f*state.commandExtension.z,-0.15f,0.20f);
        }
        if (fixedFlightController && dispatch.sampling.w==0u && state.navigation.z>=5.0f && state.threatTeacher.w<1.5f)
            actorCommand.w=clamp(state.nutrition.w,0.65f,1.50f);
        const bool privilegedCommand'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                if (state.navigation.z >= float(kCrowNavigationWaypointCount) && phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;
                uint loadedFeet=0u;'''
new='''                if (state.navigation.z >= float(kCrowNavigationWaypointCount)) {
                    // Hold above the perch until feet, lateral position, and velocity settle;
                    // then lower the bounded flight height target gradually.
                    if (phase < 0.5f) {
                        state.nutrition.w=finalGoal.z+0.25f;
                        if (abs(state.nutrition.z)<0.018f && abs(finalGoal.y-pos.y)<0.08f &&
                            length(yawFrameLinear.xy)<0.025f && pos.z>finalGoal.z+0.18f)
                            phase=1.0f;
                    } else if (phase < 1.5f) {
                        state.nutrition.w=max(state.nutrition.w-0.001f,finalGoal.z+0.15f);
                    }
                }
                uint loadedFeet=0u;'''
assert s.count(old)==1;s=s.replace(old,new);name='staged';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/latch.metal',tofile='b/staged.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
