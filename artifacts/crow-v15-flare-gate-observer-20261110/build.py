from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-flare-gate-observer-20261110';B=R/'build/crow-v15-flare-gate-observer-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'build/crow-v15-touchdown-hold-20261110/teacher.metal').read_text();s=base
needle='''            state.nutrition=float4(0.0f,0.0f,encoded,desiredHeight);'''
replacement='''            if (landing && landingPhase >= 0.5f && landingPhase < 1.5f) {
                const float3 perch = sceneState[sceneBase+course+4u].position.xyz;
                const uint lf=contactGroups[0].reference.x, rf=contactGroups[1].reference.x;
                const MRBodyStateGPU left=bodyStates[bodyBase+lf], right=bodyStates[bodyBase+rf];
                const float3 lj=left.position.xyz+rotate(left.orientation,contactGroups[0].kinematicReference.xyz);
                const float3 rj=right.position.xyz+rotate(right.orientation,contactGroups[1].kinematicReference.xyz);
                const float xErr=perch.x+0.045f-0.5f*(lj.x+rj.x);
                const float yErr=perch.y+0.55f-pos.y;
                const uint bits=(state.nutrition.x>0.05f && state.nutrition.y>0.05f ? 1u : 0u) | (abs(xErr)<0.04f ? 2u : 0u) | (abs(yErr)<0.10f ? 4u : 0u) | (length(yawFrameLinear.xy)<0.15f ? 8u : 0u) | (abs(rootLinearVelocity.z)<0.40f ? 16u : 0u);
                landingPhase=1.0f+float(bits)*0.001f;
            }
            state.nutrition=float4(0.0f,0.0f,encoded,desiredHeight);'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
(B/'teacher.metal').write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')));subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air','MetalRoboCollision.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air'),str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True)
