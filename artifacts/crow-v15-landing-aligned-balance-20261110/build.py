from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-landing-aligned-balance-20261110';B=R/'build/crow-v15-landing-aligned-balance-20261110';base=(R/'build/crow-v15-landing-support-alignment-20261110/ahead.metal').read_text()
for name in ('strong','early'):
 if name=='strong':old='clamp(-0.60f*state.threatGeometry.z-0.12f*state.threatGeometry.w,-0.08f,0.08f)';new='clamp(-1.50f*state.threatGeometry.z-0.30f*state.threatGeometry.w,-0.20f,0.20f)'
 else:old='(state.status.w&63u)>=40u && ((state.status.w>>6u)&63u)>=40u';new='(state.status.w&63u)>=20u && ((state.status.w>>6u)&63u)>=20u'
 assert base.count(old)==1;s=base.replace(old,new)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/ahead.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
