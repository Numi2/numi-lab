from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-substep-target-interpolation-screen-20261112';B=R/'build/crow-v15-substep-target-interpolation-screen-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110'
base=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal')]
for src,dst in [('task.metal','task.air'),('BoundedDrive.metal','candidate-drive.air')]:subprocess.run(base+['-c',str(A/src),'-o',str(B/dst)],check=True)
replaced={p.name:p for p in (W/'shaders').glob('*.air')}
for name in ('sham','candidate'):
 airs=[str((B/'candidate-drive.air') if name=='candidate' and p.name=='MetalRoboBoundedDrive.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
 subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'task.air'),'-o',str(B/(name+'.metallib'))],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'metal-hashes.json').write_text(json.dumps({name:h(B/(name+'.metallib')) for name in ('sham','candidate')},indent=2)+'\n')
