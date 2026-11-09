from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-teacher-route-recovery-20261109';B=R/'build/crow-teacher-route-recovery-20261109'
base=(R/'src/metal/LocomotionTask.metal').read_text()
old='''            float errorOffset = 0.0f;
            if (landing) {'''
new='''            // Development-only teacher: avoid flying past an unreached
            // target; reduce forward speed to allow cross-track correction.
            // The frozen flight law and student observations are unchanged.
            float forwardCommand = clamp(0.55f * (target.x - pos.x), -0.20f, 0.25f);
            if (abs(delta.y) > 0.28f) forwardCommand = min(forwardCommand, 0.08f);
            float errorOffset = 0.25f - forwardCommand;
            if (landing) {'''
assert base.count(old)==1
s=base.replace(old,new);src=B/'teacher.metal';src.write_text(s)
(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/development/teacher-route-recovery.metal')))
air=B/'teacher.air';shader=B/'teacher.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'shader.json').write_text(json.dumps({'source_sha256':sha(src),'shader_sha256':sha(shader),'frozen_flight_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'scope':'development-only privileged teacher route correction, teacher off leaves student actuation unchanged'},indent=2)+'\n')
