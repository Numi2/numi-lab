from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-approach-speed-bias-20261110';B=R/'build/crow-v15-approach-speed-bias-20261110';base=(R/'build/crow-v15-qualified-reverse-approach-20261110/reverseapproach.metal').read_text()
old='actorCommand.x=clamp(0.10f+0.80f*state.nutrition.z-0.80f*state.commandExtension.z,-0.15f,0.20f);'
new='actorCommand.x=clamp(0.00f+0.80f*state.nutrition.z-0.80f*state.commandExtension.z,-0.15f,0.20f);'
assert base.count(old)==1;s=base.replace(old,new)
name='speedbias';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/referencefix.metal',tofile='b/reverseapproach.metal')))
air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
