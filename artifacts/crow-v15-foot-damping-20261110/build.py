from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-foot-damping-20261110';B=R/'build/crow-v15-foot-damping-20261110';base=(R/'build/crow-v15-foot-guidance-20261110/route.metal').read_text()
old='oracleCommand.x=clamp(1.2f*footError,-0.18f,0.18f);';assert base.count(old)==1
s=base.replace(old,'oracleCommand.x=clamp(0.45f*footError-0.7f*state.commandExtension.z,-0.12f,0.12f);');(B/'route.metal').write_text(s);(A/'damping.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/guidance.metal',tofile='b/damping.metal')))
air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'route.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
