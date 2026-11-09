from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-stage-selected-policy-20261109';B=R/'build/crow-stage-selected-policy-20261109'
base=(R/'src/metal/PolicyInference.metal').read_text()
old='''    const bool liftOffBand = state.episode.z == 2u;'''
new='''    // The early-route actor is frozen. At physical completion of waypoint
    // five select the independently trained landing actor; no teacher is used.
    const bool liftOffBand = state.navigation.z >= 5.0f;'''
assert base.count(old)==1
s=base.replace(old,new)
old2='''            actions[actionBase + action] = clamp(
                baseAction + residualAction,
                -program.limits.y,
                program.limits.y
            );'''
new2='''            actions[actionBase + action] = residualAction;'''
assert s.count(old2)==1;s=s.replace(old2,new2)
src=B/'PolicyInference.metal';src.write_text(s)
(A/'policy-inference.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/metal/PolicyInference.metal',tofile='b/stage-selected-policy-inference.metal')))
air=B/'PolicyInference.air';shader=B/'stage-selected.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboPolicyInference.air'))
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(R/'build/crow-hierarchical-goal-memory-20261109/actor.air'),str(air),'-o',str(shader)],check=True)
(A/'shader.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'flight_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest(),'scope':'deterministic evaluation-only stage selector; not a PPO likelihood contract'},indent=2)+'\n')
