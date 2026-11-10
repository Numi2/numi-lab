from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-launch-pronation-limit-20261110';B=R/'build/crow-v15-launch-pronation-limit-20261110';base=(R/'build/crow-v15-launch-pronation-20261110/negative.metal').read_text()
for name,offset,height in [('m16h50',-1.6,.5),('m20h50',-2.0,.5),('m16h30',-1.6,.3),('m20h30',-2.0,.3)]:
 old='(action==4u ? -0.40f : 0.40f)*fade';new=f'(action==4u ? {offset:.2f}f : {-offset:.2f}f)*fade';assert base.count(old)==1;s=base.replace(old,new)
 old='actorCommand.w = 0.50f;';new=f'actorCommand.w = {height:.2f}f;';assert s.count(old)==1;s=s.replace(old,new)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/negative.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
