from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-contact-flight-handoff-20261111';B=R/'build/crow-v15-two-segment-contact-flight-handoff-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'artifacts/crow-v15-two-segment-gait-uniform-adaptive-20261111/candidate.metal').read_text()
a='        // Experimental gait ends before the launch crouch takes control.';b='        const float avianWingAmplitude =\n';c='        state.nutrition.xy=float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);\n'
assert base.count(a)==base.count(b)==base.count(c)==1
quiet='        if (fixedFlightController && state.episode.z == 10u && action <= 6u && float(state.episode.x)*dispatch.timing.x >= 6.0f && (state.status.w & 8192u)==0u) navigationTeacherRequested=0.0f;\n'
wing='            fixedFlightController && float(state.episode.x)*dispatch.timing.x >= 6.0f && (state.status.w & 8192u)==0u && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION ? 0.0f :\n'
latch='''        if (program.counts0.w>=5u && dispatch.counts.w==38u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            const uint previous=state.status.w;
            const bool low=seconds>=6.96f && state.nutrition.x<=.05f && state.nutrition.y<=.05f;
            const uint low1=low ? 16384u : 0u;
            const uint low2=low && (previous&16384u)!=0u ? 32768u : 0u;
            const bool released=(previous&8192u)!=0u || (low && (previous&32768u)!=0u && state.airReturnTracking.y>.195f && state.threatGeometry.y>.05f);
            const uint retract=released ? 4096u : 0u;
            state.status.w=(previous & ~(4096u|8192u|16384u|32768u)) | low1 | low2 | retract | (released?8192u:0u);
        }
'''
s=base.replace(a,quiet+a).replace(b,b+wing).replace(c,c+latch);(A/'candidate.metal').write_text(s);(A/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/uniform-walk.metal',tofile='b/contact-handoff.metal')));air=B/'tracking.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'candidate.metal'),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/'tracking.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);(A/'hashes.json').write_text(json.dumps({'shader_sha256':hashlib.sha256(ml.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((A/'candidate.metal').read_bytes()).hexdigest(),'frozen_flight_source_sha256':hashlib.sha256((F/'CrowFlightControllerResearchV2.metal').read_bytes()).hexdigest()},indent=2)+'\n')
