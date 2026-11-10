from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-causal-launch-20261110';B=R/'build/crow-v15-causal-launch-20261110';base=(R/'build/crow-v15-toe-launch-control-20261110/push.metal').read_text();needle='''            uint packed=0u;''';replacement='''            // Preserve a launch latch in a spare high bit of this experimental task state.
            uint packed=state.status.w & 4096u;
            if (seconds>=4.96f && seconds<5.4f && state.nutrition.x<0.05f && state.nutrition.y<0.05f && rootLinearVelocity.z>0.03f) packed |= 4096u;
            if (seconds>=5.4f) packed &= ~4096u;''';assert base.count(needle)==1;s=base.replace(needle,replacement)
flap='''float(state.episode.x)*dispatch.timing.x < 5.4f && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION''';assert s.count(flap)==1
for name in ('push-trigger','push-timer','control-trigger'):
 candidate=s
 if name!='push-timer':candidate=candidate.replace(flap,'''float(state.episode.x)*dispatch.timing.x < 5.4f && (state.status.w & 4096u)==0u && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION''')
 if name=='control-trigger':
  n='''                    : bilateral ? mix(crouched,extended,smoothstep(4.96f,5.00f,seconds)) : crouched;''';assert candidate.count(n)==1;candidate=candidate.replace(n,'''                    : crouched;''')
 src=B/(name+'.metal');src.write_text(candidate);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/toe-launch-push.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
