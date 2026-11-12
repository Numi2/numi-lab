from pathlib import Path
import subprocess,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-aero-closure-decomposition-20261112';B=R/'build/crow-v15-aero-closure-decomposition-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'closure.metal'),'-o',str(B/'closure.air')],check=True)
replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str((R/'build/crow-v15-walk-exact-bypass-20261112/World.air') if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(R/'build/crow-v15-common-command-pronation-screen-20261112/flexed.air'),str(B/'closure.air'),'-o',str(B/'flexed.metallib')],check=True)
(A/'metal-hashes.json').write_text(json.dumps({'source':hashlib.sha256((A/'closure.metal').read_bytes()).hexdigest(),'metallib':hashlib.sha256((B/'flexed.metallib').read_bytes()).hexdigest()},indent=2)+'\n')
