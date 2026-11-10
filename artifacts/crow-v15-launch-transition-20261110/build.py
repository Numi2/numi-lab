from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-launch-transition-20261110';B=R/'build/crow-v15-launch-transition-20261110';base=(R/'build/crow-v15-rear-toe-bounce-release-20261110/bouncerelease.metal').read_text()
old=': bilateral ? mix(crouched,extended,smoothstep(12.96f,13.00f,seconds)) : crouched;'
new=': (bilateral || (state.status.w & 4096u)!=0u) ? mix(crouched,extended,smoothstep(12.96f,13.00f,seconds)) : crouched;'
assert base.count(old)==1
for name in ('extension','forward'):
 s=base.replace(old,new)
 if name=='forward':
  old2='const bool privilegedCommand = (dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f);'
  new2='if (fixedFlightController && state.episode.z == 10u && float(state.episode.x)*dispatch.timing.x < 16.0f) { actorCommand.x = 0.40f; actorCommand.y = 0.0f; actorCommand.z = 0.0f; actorCommand.w = 0.90f; }\n        '+old2
  assert s.count(old2)==1;s=s.replace(old2,new2)
 src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/bouncerelease.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
