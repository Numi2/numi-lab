from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-contact-flare-20261110';B=R/'build/crow-v15-contact-flare-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'build/crow-v15-touchdown-hold-20261110/teacher.metal').read_text();s=base
needle='''                if (landingPhase >= 1.5f && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;'''
replacement='''                const uint toeTimers = (state.status.w & 63u) + ((state.status.w >> 6u) & 63u);
                if (landingPhase >= 1.2f && landingPhase < 1.5f && toeTimers == 0u && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;
                if (landingPhase > 0.5f && landingPhase < 1.2f && state.nutrition.x>0.20f && state.nutrition.y>0.20f && abs(xError)<0.04f && abs(yError)<0.10f && length(yawFrameLinear.xy)<0.15f && abs(rootLinearVelocity.z)<0.40f) landingPhase=1.25f;
                if (landingPhase >= 1.5f && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='''? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.5f &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION'''
replacement='''? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.2f &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
(B/'teacher.metal').write_text(s);(A/'contact-flare.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')));subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air','MetalRoboCollision.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air'),str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True)
