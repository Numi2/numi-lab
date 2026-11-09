from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-four-expert-experiment-20261109';B=R/'build/crow-four-expert-experiment-20261109'
base=(R/'src/metal/LocomotionTask.metal').read_text()
old='''             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u : 0u);'''
new='''             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u :
             ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION) != 0u &&
              state.navigation.z >= 4.0f && state.navigation.z < 5.0f ? 12u : 0u));'''
assert base.count(old)==1;s=base.replace(old,new)
old='''        const float studentRequested = actionStream[actionBase + action];'''
new='''        // In this diagnostic contract slot 15 carries a stage-four command;
        // the installed route actor's physical head action 15 is exactly zero.
        const float studentRequested = fixedFlightController && state.navigation.z < 5.0f && action == 15u
            ? 0.0f : actionStream[actionBase + action];'''
assert s.count(old)==1;s=s.replace(old,new);src=B/'stage4.metal';src.write_text(s)
(A/'shader.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/development/four-expert.metal')))
air=B/'stage4.air';shader=B/'stage4.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'shader.json').write_text(json.dumps({'shader_sha256':sha(shader),'frozen_flight_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'scope':'development-only independent stage4 expert with explicit zero physical head action; distinct contract required before promotion'},indent=2)+'\n')
