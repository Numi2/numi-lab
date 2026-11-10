from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-contact-wing-fade-20261110';B=R/'build/crow-v15-contact-wing-fade-20261110';base=(R/'build/crow-v15-grip-latch-20261110/latch.metal').read_text();s=base
old='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;'''
new='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u) {
            const float loadedRamp=float(min(state.status.w&63u,(state.status.w>>6u)&63u));
            const float remaining=1.0f-smoothstep(0.0f,48.0f,loadedRamp);
            navigationTeacherRequested=(action<7u || action==13u || action==14u)
                ? remaining*navigationTeacherRequested : 0.0f;
        }'''
assert s.count(old)==1;s=s.replace(old,new)
old='''            ? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.5f &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f
            : avianGroundCurriculum'''
new='''            ? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.5f &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? clamp(binding.drive.z + binding.drive.w * avianWingPolicyCommand,0.0f,1.0f) *
                (1.0f-smoothstep(0.0f,48.0f,float(min(state.status.w&63u,(state.status.w>>6u)&63u))))
            : avianGroundCurriculum'''
assert s.count(old)==1;s=s.replace(old,new);name='wingfade';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/latch.metal',tofile='b/wingfade.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
