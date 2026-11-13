from pathlib import Path
import gzip,subprocess
R=Path.cwd();old='crow-v15-wing-target-velocity-screen-20261113';new='crow-v15-passive-aero-flap-amplitude-20261113';A=R/'artifacts'/new;B=R/'build'/new;O=R/'artifacts'/old
for name in ('push-wing','hold-wing'):
 s=(O/(name+'.metal')).read_text();needle='const float triggered=defaultQ[binding.indices.z]+0.65f*envelope*phase;';replacement='const float amplitude=environment%8u<4u ? 0.25f : environment%8u<6u ? 0.45f : 0.65f;\n                const float triggered=defaultQ[binding.indices.z]+amplitude*envelope*phase;';assert s.count(needle)==1;s=s.replace(needle,replacement);(A/(name+'.metal')).write_text(s)
s=(O/'World.metal').read_text();needle='const float kp=environment%8u<4u ? 4.0f : environment%8u<6u ? 10.0f : 20.0f;';assert s.count(needle)==2;s=s.replace(needle,'const float kp=10.0f;');needle='const float kd=environment%8u<4u ? 0.05f : environment%8u<6u ? 0.10f : 0.20f;';assert s.count(needle)==2;s=s.replace(needle,'const float kd=0.10f;');(A/'World.metal').write_text(s)
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'build/crow-v15-wide-contact-isolation-20261111/include'),'-I',str(R/'src/metal'),'-c',str(A/'World.metal'),'-o',str(B/'World.air')],check=True)
s=(O/'build.py').read_text().replace(old,new).replace("build/crow-v15-wing-aero-alignment-20261110/aero.air","build/crow-v15-energy-consistent-aero-20261113/aero.air");(A/'build.py').write_text(s)
s=gzip.decompress((R/'artifacts/crow-v15-aero-power-audit-20261113/observer.mm.gz').read_bytes()).decode();src=gzip.decompress((O/'observer.mm.gz').read_bytes()).decode();line=next(l for l in src.splitlines(True) if 'CROW_WING_TARGET_PAIR %llu' in l);needle=next(l for l in s.splitlines(True) if 'CROW_DRIVE_TARGET %llu' in l);assert s.count(needle)==1;s=s.replace(needle,needle+line);(B/'MetalWorld.mm').write_text(s);(A/'observer.mm.gz').write_bytes(gzip.compress(s.encode(),mtime=0))
s=(O/'build-observer.py').read_text().replace(old,new);(A/'build-observer.py').write_text(s);subprocess.run(['python3',str(A/'build-observer.py')],check=True)
for name in ('push-wing','hold-wing'):subprocess.run(['python3',str(A/'build.py'),name],check=True)
(A/'run.py').write_text((O/'run.py').read_text().replace(old,new))
for f in ('analyze.py','com-analyze.py','strict-gates.py','target-pair-check.py','wing-tracking.py'):
 s=(O/f).read_text().replace(old,new).replace('rebound-targetvelocity','rebound-amplitude');(A/f).write_text(s)
