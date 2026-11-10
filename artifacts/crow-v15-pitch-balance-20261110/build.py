from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-pitch-balance-20261110';B=R/'build/crow-v15-pitch-balance-20261110';base=(R/'build/crow-v15-grip-latch-20261110/latch.metal').read_text()
old='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
'''
add='''        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f &&
            state.threatTeacher.w>=1.5f && (state.status.w&63u)>=40u && ((state.status.w>>6u)&63u)>=40u &&
            (action==7u || action==9u || action==10u || action==12u)) {
            const float correction=clamp(-0.20f*state.threatGeometry.z-0.04f*state.threatGeometry.w,-0.04f,0.04f);
            navigationTeacherRequested=(action==9u || action==12u) ? -correction : correction;
        }
'''
assert base.count(old)==1;s=base.replace(old,old+add);name='pitchbalance';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/latch.metal',tofile='b/pitchbalance.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
