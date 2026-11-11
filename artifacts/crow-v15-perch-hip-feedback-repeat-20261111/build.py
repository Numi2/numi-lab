from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perch-hip-feedback-repeat-20261111';B=R/'build/crow-v15-perch-hip-feedback-repeat-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110'
base=(R/'artifacts/crow-v15-walk-strength085-confirm-20261111/candidate.metal').read_text()
old="""        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;"""
new=old+"""
        // Development-only perching feedback derived from the privileged
        // teacher's unsaturated landing command. Not an autonomous sensor.
        if (privilegedCommand && fixedFlightController && state.navigation.z>=5.0f &&
            state.threatTeacher.w>=1.5f && (state.status.w & 4095u)!=0u &&
            (action==7u || action==10u) &&
            state.nutrition.x>-.099f && state.nutrition.x<.349f) {
            const float vx=state.commandExtension.z;
            const float err=(state.nutrition.x-.10f+.60f*vx)/.45f;
            const float delta=clamp(.5f*(err-.032f-.20f*vx),-.16f,.16f);
            navigationTeacherRequested=clamp(delta/max(abs(actions[action].parameters.x),.001f),-1.0f,1.0f);
        }"""
assert base.count(old)==1
s=base.replace(old,new)
oldRamp="const uint ramp=loaded[side] ? min(oldRamp+1u,34u) : (oldRamp>0u ? oldRamp-1u : 0u);"
newRamp="const uint ramp=loaded[side] ? min(oldRamp+1u,34u) : (oldRamp>0u ? oldRamp-1u : 0u);"
assert s.count(oldRamp)==1
s=s.replace(oldRamp,newRamp)
(A/'candidate.metal').write_text(s)
(A/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/revised_gait.metal',tofile='b/root_balance_latch.metal')))
air=B/'tracking.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'candidate.metal'),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/'tracking.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);(A/'hashes.json').write_text(json.dumps({'shader_sha256':hashlib.sha256(ml.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((A/'candidate.metal').read_bytes()).hexdigest(),'frozen_flight_source_sha256':hashlib.sha256((F/'CrowFlightControllerResearchV2.metal').read_bytes()).hexdigest()},indent=2)+'\n')
