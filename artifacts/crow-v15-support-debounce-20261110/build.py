from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-support-debounce-20261110';B=R/'build/crow-v15-support-debounce-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'build/crow-v15-pose-trim-strong-20261110/teacher.metal').read_text();s=base
needle='if (landingPhase >= 1.5f && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;'
replacement='if (landingPhase >= 1.5f && ((state.status.w >> 25u) & 127u) >= 5u && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;'
assert s.count(needle)==1;s=s.replace(needle,replacement)
start=s.index('            // Accepted contact resultant and articulated COM supply a bounded');end=s.index('            state.status.w=packed;',start)
replacement='''            // Count consecutive accepted low-load samples only while in supported phase.
            if (state.navigation.z>=5.0f && state.threatTeacher.w>=1.5f) {
                const uint prior=(state.status.w>>25u)&127u;
                const bool noLoad=state.nutrition.x+state.nutrition.y<0.25f;
                const uint duration=noLoad ? min(prior+1u,100u) : 0u;
                packed |= duration<<25u;
            }
'''
s=s[:start]+replacement+s[end:]
(B/'teacher.metal').write_text(s);(A/'debounce.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')));subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air','MetalRoboCollision.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air'),str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True)
