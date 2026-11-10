from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-grip-latch-20261110';B=R/'build/crow-v15-grip-latch-20261110';base=(R/'build/crow-v15-approach-speed-bias-20261110/speedbias.metal').read_text()
old='const uint next=loaded ? min(previous+1u,48u) : (previous>0u ? previous-1u : 0u);'
new='const uint next=loaded || (state.threatTeacher.w>=1.5f && previous>=12u) ? min(previous+1u,48u) : (previous>0u ? previous-1u : 0u);'
assert base.count(old)==1;s=base.replace(old,new)
for name,candidate in [('latch',s),('slow_latch',s.replace('abs(finalGoal.y-pos.y)<0.16f && length(yawFrameLinear.xy)<0.08f &&\n                    abs(rootLinearVelocity.z)<0.12f','abs(finalGoal.y-pos.y)<0.16f && length(yawFrameLinear.xy)<0.025f &&\n                    abs(rootLinearVelocity.z)<0.08f'))]:
 if name=='slow_latch':assert candidate!=s
 src=B/(name+'.metal');src.write_text(candidate);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/speedbias.metal',tofile='b/'+name+'.metal')))
 air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
