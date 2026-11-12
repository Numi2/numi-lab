from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-implicit-target-interpolation-screen-20261112';B=R/'build/crow-v15-implicit-target-interpolation-screen-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';P=R/'build/crow-v15-substep-target-interpolation-screen-20261112'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'World.metal'),'-o',str(B/'World.air')],check=True)
replaced={p.name:p for p in (W/'shaders').glob('*.air')}
airs=[str(B/'World.air' if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(P/'task.air'),'-o',str(B/'candidate.metallib')],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'hashes.json').write_text(json.dumps({'world_source':h(A/'World.metal'),'candidate_metallib':h(B/'candidate.metallib'),'shared_host':h(P/'libmetalrobo.dylib'),'shared_task_air':h(P/'task.air'),'sham_metallib':h(P/'sham.metallib')},indent=2)+'\n')
