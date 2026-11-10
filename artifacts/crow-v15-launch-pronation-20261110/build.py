from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-launch-pronation-20261110';B=R/'build/crow-v15-launch-pronation-20261110';base=(R/'build/crow-v15-launch-speed-height-20261110/low.metal').read_text()
old='''        // Experimental gait ends before the launch crouch takes control.
'''
for name,offset in [('positive',.4),('negative',-.4)]:
 add=f'''        if (fixedFlightController && state.episode.z==10u && float(state.episode.x)*dispatch.timing.x>=12.96f && float(state.episode.x)*dispatch.timing.x<16.0f && (action==4u || action==5u)) {{
            const float fade=1.0f-smoothstep(15.0f,16.0f,float(state.episode.x)*dispatch.timing.x);
            navigationTeacherRequested=clamp(navigationTeacherRequested+(action==4u ? {offset:.2f}f : {-offset:.2f}f)*fade,-1.0f,1.0f);
        }}
'''
 assert base.count(old)==1;s=base.replace(old,add+old)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/low.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
