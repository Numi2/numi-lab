from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-goal-memory-terminal-dagger-20261109';B=R/'build/crow-goal-memory-terminal-dagger-20261109'
base=(R/'build/crow-hierarchical-goal-memory-20261109/dagger.metal').read_text()
old='''        const uint recoveryTakeover = 700u + 150u*(environment % 4u);
        const bool privilegedCommand = HIERARCHY_ORACLE ||
            (dispatch.sampling.w != 0u &&
             (dispatch.assistance.x < 0.5f || (HIERARCHY_RECOVER && state.episode.x >= recoveryTakeover)));'''
new='''        // Teacher reaches the final goal physically, then the student owns
        // landing approach until the explicit recovery intervention.
        const uint recoveryTakeover = 2100u + 100u*(environment % 4u);
        const bool privilegedCommand = HIERARCHY_ORACLE ||
            (dispatch.sampling.w != 0u &&
             (dispatch.assistance.x < 0.5f || state.navigation.z < 5.0f ||
              (HIERARCHY_RECOVER && state.episode.x >= recoveryTakeover)));'''
assert base.count(old)==1
s=base.replace(old,new)
src=B/'terminal.metal';src.write_text(s)
(A/'terminal.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/goal-memory-dagger.metal',tofile='b/terminal-dagger.metal')))
air=B/'terminal.air';shader=B/'terminal.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
(A/'shader.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'frozen_controller_sha256':hashlib.sha256((R/'src/metal/CrowFlightControllerV1.metal').read_bytes()).hexdigest()},indent=2)+'\n')
