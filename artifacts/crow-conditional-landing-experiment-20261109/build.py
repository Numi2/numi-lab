from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-conditional-landing-experiment-20261109';B=R/'build/crow-conditional-landing-experiment-20261109'
base=(R/'src/metal/LocomotionTask.metal').read_text()
old='''        const uint commandOffset =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION)'''
new='''        // Development-only: a learned late landing recovery head, gated by
        // elapsed time and proprioceptive support, never teacher commands.
        const uint commandOffset =
            fixedFlightController && state.navigation.z >= 5.0f &&
            state.episode.x >= 2200u && state.threatTeacher.w < 1.5f ? 4u :
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION)'''
assert base.count(old)==1;s=base.replace(old,new);src=B/'conditional.metal';src.write_text(s)
(A/'shader.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/development/conditional-landing.metal')))
air=B/'conditional.air';shader=B/'conditional.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'shader.json').write_text(json.dumps({'source_sha256':sha(src),'shader_sha256':sha(shader),'frozen_flight_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'scope':'development-only conditional learned recovery after 2200 steps while unsupported; distinct fingerprinted task required before promotion'},indent=2)+'\n')
