from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-ramped-pitch-balance-20261110';B=R/'build/crow-v15-ramped-pitch-balance-20261110';base=(R/'build/crow-v15-strong-pitch-balance-20261110/strong.metal').read_text();s=base
old='''            const float correction=clamp(-1.50f*state.threatGeometry.z-0.30f*state.threatGeometry.w,-0.20f,0.20f);
            navigationTeacherRequested=(action==9u || action==12u) ? -correction : correction;'''
new='''            const float correction=clamp(-1.50f*state.threatGeometry.z-0.30f*state.threatGeometry.w,-0.30f,0.30f) * clamp(state.nutrition.w,0.0f,1.0f);
            navigationTeacherRequested=(action==9u || action==12u) ? -correction : correction;'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);
        state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x);'''
new='''        // Landing-only scalar grows gradually; the controller remains physically actuated.
        if (state.navigation.z>=5.0f)
            state.nutrition.w=phase>=1.5f ? min(state.nutrition.w+0.01f,1.0f) : 0.0f;
        state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);
        state.commandExtension.xyz=float3(-heading,baseAngular.z,yawFrameLinear.x);'''
assert s.count(old)==1;s=s.replace(old,new);name='ramped';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/strong.metal',tofile='b/ramped.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
