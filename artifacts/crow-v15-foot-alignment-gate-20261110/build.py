from pathlib import Path
import difflib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-foot-alignment-gate-20261110';B=R/'build/crow-v15-foot-alignment-gate-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';old=(R/'build/crow-v15-contact-onset-reflex-20261110/teacher.metal').read_text()
needle='desiredHeight = landingPhase > 0.5f ? max(perch.z+0.245f,state.nutrition.w-0.025f*dispatch.timing.x) : 1.30f;'
replacement='''const bool alignedForDescent = abs(xError)<0.025f && abs(yError)<0.12f && length(yawFrameLinear.xy)<0.065f;
                desiredHeight = landingPhase > 0.5f ? (alignedForDescent ? max(perch.z+0.245f,state.nutrition.w-0.025f*dispatch.timing.x) : max(perch.z+0.31f,state.nutrition.w)) : 1.30f;'''
toe='const float target=!closing ? 0.0f : toe==3u ? min(0.020f*float(rearRamp),0.85f) : min(0.025f*float(ramp),clamp(1.12f-0.8f*state.threatGeometry.z-0.80f*state.threatGeometry.w-2.0f*state.commandExtension.z,0.85f,1.20f));'
forward='const float commandedForward = clamp(0.04f+0.20f*xError-0.20f*yawFrameLinear.x,-0.08f,0.10f);'
for label in ['gate','zero_bias']:
 s=old
 for a,b in [(needle,replacement),(toe,'const float target=0.0f;')]+([(forward,'const float commandedForward = clamp(0.60f*xError-0.50f*yawFrameLinear.x,-0.08f,0.10f);')] if label=='zero_bias' else []):
  assert s.count(a)==1,(label,a,s.count(a));s=s.replace(a,b)
 f=B/(label+'.metal');f.write_text(s);(A/(label+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(f),'-o',str(B/(label+'.air'))],check=True)
 other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/(label+'.air')),'-o',str(B/(label+'.metallib'))],check=True)
