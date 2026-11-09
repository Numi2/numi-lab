from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-gate-recovery-dagger-20261109';B=R/'build/crow-gate-recovery-dagger-20261109'
base=(R/'build/crow-hierarchical-goal-memory-20261109/dagger.metal').read_text()
old='''        const uint recoveryTakeover = 700u + 150u*(environment % 4u);'''
new='''        const uint recoveryTakeover = 850u + 100u*(environment % 4u);'''
assert base.count(old)==1
s=base.replace(old,new);src=B/'full-student.metal';src.write_text(s)
(A/'locomotion.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/goal-memory-dagger.metal',tofile='b/full-student-dagger.metal')))
air=B/'full-student.air';shader=B/'full-student.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboPolicyInference.air'))
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),str(R/'build/crow-stage-selected-policy-20261109/PolicyInference.air'),'-o',str(shader)],check=True)
(A/'shader.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'frozen_flight_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest()},indent=2)+'\n')
