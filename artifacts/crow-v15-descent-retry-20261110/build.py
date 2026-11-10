from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-descent-retry-20261110';B=R/'build/crow-v15-descent-retry-20261110';base=(R/'build/crow-v15-staged-descent-height-gate-20261110/heightgate.metal').read_text()
old='''                    } else if (phase < 1.5f) {
                        state.nutrition.w=max(state.nutrition.w-0.001f,finalGoal.z+0.15f);
                    }'''
new='''                    } else if (phase < 1.5f) {
                        if (abs(state.nutrition.z)>0.04f || length(yawFrameLinear.xy)>0.055f) {
                            phase=0.0f;
                            state.nutrition.w=finalGoal.z+0.25f;
                        } else {
                            state.nutrition.w=max(state.nutrition.w-0.0004f,finalGoal.z+0.15f);
                        }
                    }'''
assert base.count(old)==1;s=base.replace(old,new);name='retry';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/heightgate.metal',tofile='b/retry.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
