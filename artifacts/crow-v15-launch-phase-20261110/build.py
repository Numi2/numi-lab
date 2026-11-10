from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-launch-phase-20261110';B=R/'build/crow-v15-launch-phase-20261110';base=(R/'build/crow-v15-launch-transition-20261110/forward.metal').read_text()
for name,phase in [('quarter',1.570796327),('half',3.141592654),('threequarter',4.71238898)]:
 fade=f'({phase:.9f}f * (1.0f-smoothstep(15.0f,16.0f,float(state.episode.x)*dispatch.timing.x)))'
 old='carrierState.episode.x = min(carrierState.episode.x, 1500u);';new=old+'\n        if (state.episode.z == 10u) carrierState.commandAndPhase.w += '+fade+';';assert base.count(old)==1;s=base.replace(old,new)
 old2='avianWingAmplitude *\n                    sin(state.commandAndPhase.w)';new2='avianWingAmplitude *\n                    sin(state.commandAndPhase.w + (fixedFlightController && state.episode.z == 10u ? '+fade+' : 0.0f))';assert s.count(old2)==1;s=s.replace(old2,new2)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/forward.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
