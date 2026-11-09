from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-release-balance-20261109';B=R/'build/crow-v15-release-balance-20261109';base=(R/'build/crow-v15-release-timing-20261109/positive.metal').read_text();needle='''        const float navigationTeacherBlend = fixedFlightController'''
assert base.count(needle)==1
for name,sign in [('positive','4.0f'),('negative','10.0f')]:
 patch='''        // Experimental ankle balance; enabled only on bilateral measured support.
        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f &&
            (action==9u || action==12u) && state.nutrition.x>0.05f && state.nutrition.y>0.05f && float(state.episode.x)*dispatch.timing.x>=45.0f) {
            navigationTeacherRequested=clamp(SIGN*(0.20f*state.threatGeometry.z+0.10f*state.threatGeometry.w),-0.60f,0.60f);
        }
'''.replace('SIGN',sign);s=base.replace(needle,patch+needle);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/toe-active.metal',tofile='b/hip-balance-'+name+'.metal')))
 air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
