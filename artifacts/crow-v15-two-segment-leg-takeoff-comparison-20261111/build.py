from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-leg-takeoff-comparison-20261111';B=R/'build/crow-v15-two-segment-leg-takeoff-comparison-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110'
base=(R/'artifacts/crow-v15-two-segment-gait-uniform-adaptive-20261111/candidate.metal').read_text()
a='        // Experimental gait ends before the launch crouch takes control.'
b='        // Keep the training-only teacher active through the final target.'
c='        const float avianWingAmplitude =\n'
assert base.count(a)==base.count(b)==base.count(c)==1
quiet='        if (fixedFlightController && state.episode.z == 10u && action <= 6u && float(state.episode.x)*dispatch.timing.x >= 6.0f && float(state.episode.x)*dispatch.timing.x < 9.0f) navigationTeacherRequested=0.0f;\n'
neutral='        if (fixedFlightController && state.episode.z == 10u && action >= 7u && action <= 12u && float(state.episode.x)*dispatch.timing.x >= 6.3f && float(state.episode.x)*dispatch.timing.x < 9.0f) navigationTeacherRequested=0.0f;\n'
wing='            fixedFlightController && float(state.episode.x)*dispatch.timing.x >= 6.0f && float(state.episode.x)*dispatch.timing.x < 9.0f && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION ? 0.0f :\n'
replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];hashes={}
for arm in ('push','control'):
 s=base.replace(a,quiet+a).replace(c,c+wing)
 if arm=='control':s=s.replace(b,neutral+b)
 src=A/(arm+'.metal');src.write_text(s);air=B/(arm+'.air');lib=B/(arm+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(lib)],check=True);hashes[arm]={'source':hashlib.sha256(src.read_bytes()).hexdigest(),'metallib':hashlib.sha256(lib.read_bytes()).hexdigest()}
(A/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
