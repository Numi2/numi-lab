from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-stage2-head-experiment-20261109';B=R/'build/crow-stage2-head-experiment-20261109'
base=(R/'src/metal/LocomotionTask.metal').read_text()
old='''        const float4 actorLatent = fixedFlightController
            ? clamp(float4(actionStream[actionBase+0u],actionStream[actionBase+1u],
                           actionStream[actionBase+2u],actionStream[actionBase+3u]),-1.0f,1.0f)
            : float4(0.0f);'''
new='''        const uint commandOffset = state.navigation.z >= 2.0f && state.navigation.z < 3.0f ? 4u : 0u;
        const float4 actorLatent = fixedFlightController
            ? clamp(float4(actionStream[actionBase+commandOffset+0u],actionStream[actionBase+commandOffset+1u],
                           actionStream[actionBase+commandOffset+2u],actionStream[actionBase+commandOffset+3u]),-1.0f,1.0f)
            : float4(0.0f);'''
assert base.count(old)==1
s=base.replace(old,new);src=B/'stage2.metal';src.write_text(s)
(A/'shader.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/development-stage2-head.metal')))
air=B/'stage2.air';shader=B/'stage2.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
(A/'shader.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'frozen_flight_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest(),'scope':'development-only alternative action mapping; same physical model'},indent=2)+'\n')
