from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-leg-push-timing-20261110';B=R/'build/crow-v15-leg-push-timing-20261110';base=(R/'build/crow-v15-launch-speed-height-20261110/low.metal').read_text()
for name,end in [('ramp08',13.04),('ramp12',13.08),('ramp20',13.16)]:
 old='smoothstep(12.96f,13.00f,seconds)';new=f'smoothstep(12.96f,{end:.2f}f,seconds)';assert base.count(old)==1;s=base.replace(old,new)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/low.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
