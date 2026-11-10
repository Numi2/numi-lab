from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-landing-joint-identification-20261110';B=R/'build/crow-v15-landing-joint-identification-20261110';base=(R/'build/crow-v15-grip-latch-20261110/latch.metal').read_text()
old='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;
'''
for name,condition,val in [('hip_plus','action==7u || action==10u','0.03f'),('hip_minus','action==7u || action==10u','-0.03f'),('ankle_plus','action==9u || action==12u','0.03f'),('ankle_minus','action==9u || action==12u','-0.03f')]:
 add='''        if (fixedFlightController && state.episode.z==10u && state.threatTeacher.w>=1.5f &&
            state.episode.x>=2500u && state.episode.x<2520u && ('''+condition+'''))
            navigationTeacherRequested = '''+val+''';
'''
 assert base.count(old)==1;s=base.replace(old,old+add);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/latch.metal',tofile='b/'+name+'.metal')))
 air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
