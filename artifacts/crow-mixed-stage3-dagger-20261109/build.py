from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-mixed-stage3-dagger-20261109';B=R/'build/crow-mixed-stage3-dagger-20261109'
base=(R/'build/crow-stage2-head-experiment-20261109/stage2.metal').read_text()
old='const bool privilegedCommand = (dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f);'
new='const bool privilegedCommand = dispatch.sampling.w != 0u && (dispatch.assistance.x < 0.5f || state.episode.x >= (1300u + 100u*(environment % 4u)));'
assert base.count(old)==1
s=base.replace(old,new);src=B/'recovery.metal';src.write_text(s)
(A/'recovery.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/development-stage2-head.metal',tofile='b/development-recovery.metal')))
air=B/'recovery.air';shader=B/'recovery.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
(A/'shader.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'frozen_flight_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest(),'note':'production source with only a diagnostic teacher takeover schedule; teacher disabled in controls'},indent=2)+'\n')
