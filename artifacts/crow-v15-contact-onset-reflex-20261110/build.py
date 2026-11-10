from pathlib import Path
import difflib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-onset-reflex-20261110';B=R/'build/crow-v15-contact-onset-reflex-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';old=(R/'build/crow-v15-averaged-contact-control-20261110/teacher.metal').read_text();s=old
pairs=[('footLoad>0.05f || (state.threatTeacher.w>=1.5f && rearPrevious>=12u)) ? min(rearPrevious+6u,43u)','footLoad>0.20f || (state.threatTeacher.w>=1.5f && rearPrevious>=12u)) ? min(rearPrevious+16u,43u)'),('length(force)>0.05f || (footLoad>0.05f && rearPrevious>=1u)','length(force)>0.05f || footLoad>0.20f'),('min(previous+6u,48u)','min(previous+16u,48u)')]
for a,b in pairs:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
(B/'teacher.metal').write_text(s);(A/'control.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')))
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True)
