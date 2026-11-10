from pathlib import Path
import difflib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-landing-rate-feedback-20261110';B=R/'build/crow-v15-landing-rate-feedback-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';old=(R/'build/crow-v15-contact-onset-reflex-20261110/teacher.metal').read_text()
needle='lateralMilli * 0.001f - 0.30f, state.commandExtension.x, state.nutrition.w);'
for label,gain in [('gain03','0.30f'),('gain06','0.60f')]:
 change='lateralMilli * 0.001f - 0.30f, state.commandExtension.x, state.nutrition.w + ((state.navigation.z >= 5.0f && state.threatTeacher.w > 0.5f && state.threatTeacher.w < 1.5f) ? clamp(-'+gain+'*(state.threatGeometry.y+0.025f),-0.08f,0.08f) : 0.0f));'
 assert old.count(needle)==1;s=old.replace(needle,change);f=B/(label+'.metal');f.write_text(s);(A/(label+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(f),'-o',str(B/(label+'.air'))],check=True)
 other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/(label+'.air')),'-o',str(B/(label+'.metallib'))],check=True)
