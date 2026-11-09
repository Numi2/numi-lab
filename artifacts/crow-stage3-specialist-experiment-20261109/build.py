from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-stage3-specialist-experiment-20261109';B=R/'build/crow-stage3-specialist-experiment-20261109'
base=(R/'src/metal/LocomotionTask.metal').read_text()
old='''            state.navigation.z >= 2.0f && state.navigation.z < 3.0f ? 4u : 0u;'''
new='''            state.navigation.z >= 2.0f && state.navigation.z < 3.0f ? 4u :
            ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION) != 0u &&
             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u : 0u);'''
assert base.count(old)==1
s=base.replace(old,new);src=B/'specialist.metal';src.write_text(s)
(A/'shader.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/development/specialist.metal')))
air=B/'specialist.air';shader=B/'specialist.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'shader.json').write_text(json.dumps({'source_sha256':sha(src),'shader_sha256':sha(shader),'frozen_flight_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'scope':'development-only stage3 expert routing; not production task contract'},indent=2)+'\n')
