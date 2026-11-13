from pathlib import Path
import hashlib,json,subprocess,sys
name=sys.argv[1];assert name in ("push-wing","hold-wing")
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-front-toe-preload-reach-20261113';B=R/'build/crow-v15-front-toe-preload-reach-20261113';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/(name+'.metal')),'-o',str(B/(name+'.air'))],check=True)
replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str((B/'World.air') if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/(name+'.air')),'-o',str(B/(name+'.metallib'))],check=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/(name+'-hashes.json')).write_text(json.dumps({'host':h(R/'build/crow-v15-wing-effort-observer-20261113/libmetalrobo.dylib'),'metallib':h(B/(name+'.metallib')),'task_source':h(A/(name+'.metal'))},indent=2)+'\n');print('Built candidate metallib:',h(B/(name+'.metallib')))
