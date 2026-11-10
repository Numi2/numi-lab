from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-foot-handoff-20261110';B=R/'build/crow-v15-foot-handoff-20261110';base=(R/'build/crow-v15-foot-aligned-approach-20261110/footalign.metal').read_text()
s=base.replace('actorCommand.x=clamp(0.10f+1.0f*state.nutrition.z,-0.05f,0.30f);','actorCommand.x=clamp(0.10f+0.65f*state.nutrition.z-0.60f*state.commandExtension.z,0.00f,0.20f);')
old='''if (loadedFeet>=2u && abs(finalGoal.x-pos.x)<0.06f &&
                    abs(finalGoal.y-pos.y)<0.16f'''
new='''if (loadedFeet>=2u && abs(state.nutrition.z)<0.045f &&
                    abs(finalGoal.y-pos.y)<0.16f'''
assert s.count(old)==1;s=s.replace(old,new)
name='handoff';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/footalign.metal',tofile='b/handoff.metal')))
air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
