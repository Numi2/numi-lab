from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-loaded-balance-20261110';B=R/'build/crow-v15-loaded-balance-20261110';base=(R/'build/crow-v15-grip-latch-20261110/latch.metal').read_text()
old='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
'''
new=old+'''        // Apply bounded leg adjustments only after sustained bilateral toe load.
        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f &&
            state.threatTeacher.w>=1.5f && (state.status.w&63u)>=40u && ((state.status.w>>6u)&63u)>=40u &&
            (action==7u || action==9u || action==10u || action==12u) &&
            ((action==7u || action==9u) ? state.nutrition.x : state.nutrition.y)>0.05f) {
            const uint encoded=(state.status.w>>25u)&127u;
            if(encoded!=0u) {
                const float supportError=(float(encoded)-64.0f)*0.002f;
                const float correction=clamp(3.0f*supportError-0.25f*state.threatGeometry.w,-0.20f,0.20f);
                navigationTeacherRequested=(action==9u || action==12u) ? -1.077f*correction : correction;
            }
        }
'''
assert base.count(old)==1;s=base.replace(old,new);name='balance';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/latch.metal',tofile='b/balance.metal')))
air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
