from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-v1-load-relieved-landing-20261111';B=R/'build/crow-v15-v1-load-relieved-landing-20261111';base=(R/'build/crow-v15-v1-deeper-touchdown-20261111/teacher.metal').read_text();old='''        // Keep the training-only teacher active through the final target.''';new='''        // Experiment: measured-foot-load wing relief during perch approach only.
        if (fixedFlightController && state.navigation.z>=5.0f && state.threatTeacher.w>0.5f &&
            state.threatTeacher.w<1.5f && action<=5u) {
            const float footLoad=max(state.nutrition.x,0.0f)+max(state.nutrition.y,0.0f);
            navigationTeacherRequested *= 1.0f-0.70f*smoothstep(0.10f,2.0f,footLoad);
        }
        // Keep the training-only teacher active through the final target.''';assert base.count(old)==1;s=base.replace(old,new);(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/deeper.metal',tofile='b/wing_relief.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
