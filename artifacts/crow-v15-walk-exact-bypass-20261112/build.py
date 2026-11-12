from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-walk-exact-bypass-20261112';B=R/'build/crow-v15-walk-exact-bypass-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';I=B;replaced={p.name:p for p in (W/'shaders').glob('*.air')};out={}
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'World.metal'),'-o',str(B/'World.air')],check=True)
for n in (0,5):
 air=B/f'task{n}.air';lib=B/f'lead{n}.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/f'task{n}.metal'),'-o',str(air)],check=True)
 airs=[str(I/'World.air' if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
 subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(lib)],check=True);out[str(n)]=hashlib.sha256(lib.read_bytes()).hexdigest()
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
