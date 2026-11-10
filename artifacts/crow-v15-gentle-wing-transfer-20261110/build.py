from pathlib import Path
import difflib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-gentle-wing-transfer-20261110';B=R/'build/crow-v15-gentle-wing-transfer-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';old=(R/'build/crow-v15-averaged-contact-control-20261110/teacher.metal').read_text()
for label,k in [('gentle15','0.15f'),('gentle30','0.30f')]:
 s=old;a='''                const uint supportDuration = (state.status.w >> 25u) & 127u;''';b='''                if (landingPhase >= 1.0f && landingPhase < 1.5f) {
                    const float previousSupport=clamp((landingPhase-1.0f)*2.5f,0.0f,1.0f);
                    const float measuredSupport=clamp((state.nutrition.x+state.nutrition.y)/4.41f,0.0f,1.0f);
                    landingPhase=1.0f+0.4f*mix(previousSupport,measuredSupport,0.05f);
                }
                const uint supportDuration = (state.status.w >> 25u) & 127u;''';assert s.count(a)==1;s=s.replace(a,b)
 a='''                binding.drive.z + binding.drive.w * avianWingPolicyCommand,
                0.0f,
                1.0f
            );''';b='''                binding.drive.z + binding.drive.w * avianWingPolicyCommand,
                0.0f,
                1.0f
            ) * (fixedFlightController && state.navigation.z>=5.0f && state.threatTeacher.w>=1.0f && state.threatTeacher.w<1.5f && binding.actuator.x==MR_TASK_ACTUATOR_FLAPPING_POSITION ? 1.0f-K*clamp((state.threatTeacher.w-1.0f)*2.5f,0.0f,1.0f) : 1.0f);'''.replace('1.0f-K*','1.0f-'+k+'*');assert s.count(a)==1;s=s.replace(a,b);(B/(label+'.metal')).write_text(s);(A/(label+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/'+label+'.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/(label+'.metal')),'-o',str(B/(label+'.air'))],check=True)
 other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/(label+'.air')),'-o',str(B/(label+'.metallib'))],check=True)
