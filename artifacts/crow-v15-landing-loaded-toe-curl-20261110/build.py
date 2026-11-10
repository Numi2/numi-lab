from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-landing-loaded-toe-curl-20261110';B=R/'build/crow-v15-landing-loaded-toe-curl-20261110';base=(R/'build/crow-v15-landing-support-alignment-20261110/ahead.metal').read_text()
for name in ('front','both'):
 old='clamp(1.12f-0.8f*state.threatGeometry.z-0.80f*state.threatGeometry.w-2.0f*state.commandExtension.z,0.85f,1.20f)';new='clamp(1.12f-0.8f*state.threatGeometry.z-0.80f*state.threatGeometry.w-2.0f*state.commandExtension.z,1.10f,1.20f)';assert base.count(old)==1;s=base.replace(old,new)
 if name=='both':
  old='min(0.020f*float(rearRamp),0.85f)';new='min(0.026f*float(rearRamp),1.10f)';assert s.count(old)==1;s=s.replace(old,new)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/ahead.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
