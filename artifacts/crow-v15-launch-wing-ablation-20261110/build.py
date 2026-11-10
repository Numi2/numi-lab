from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-launch-wing-ablation-20261110';B=R/'build/crow-v15-launch-wing-ablation-20261110'
for name,srcname in [('push','build/crow-v15-rear-toe-bounce-release-20261110/bouncerelease.metal'),('nopush','build/crow-v15-integrated-push-causality-20261110/nopush.metal')]:
 base=(R/srcname).read_text();old='(state.status.w & 4096u)==0u';assert base.count(old)==2;s=base.replace(old,'true')
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/'+name+'.metal',tofile='b/'+name+'-no-wing.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
