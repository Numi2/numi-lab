from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-landing-posture-transition-20261110';B=R/'build/crow-v15-landing-posture-transition-20261110';base=(R/'build/crow-v15-rear-toe-bounce-release-20261110/bouncerelease.metal').read_text()
old='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;'''
assert base.count(old)==1
for name,expr in [('soft','1.0f-smoothstep(0.0f,43.0f,float(min((state.status.w>>13u)&63u,(state.status.w>>19u)&63u)))'),('hold','1.0f')]:
 new='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u) {
            const float remaining='''+expr+''';
            navigationTeacherRequested=(action>=7u && action<=12u) ? remaining*teacherRequested : 0.0f;
        }'''
 s=base.replace(old,new);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/bouncerelease.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
