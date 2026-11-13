from pathlib import Path
import gzip,hashlib,json,shutil,subprocess
R=Path.cwd();new='crow-v15-consistent-long-flight-20261113';previous='crow-v15-consistent-wing-drive-20261113';amp='crow-v15-passive-aero-flap-amplitude-20261113';A=R/'artifacts'/new;B=R/'build'/new;O=R/'artifacts'/amp
for name in ('push-wing','hold-wing'):
 s=(O/(name+'.metal')).read_text();needle='if (seconds>=6.80f && seconds<8.02f)';assert s.count(needle)==1;s=s.replace(needle,'if (seconds>=6.80f && seconds<10.02f)');(A/(name+'.metal')).write_text(s)
shutil.copyfile(R/'build'/amp/'World.air',B/'World.air')
s=(O/'build.py').read_text().replace(amp,new).replace("R/'build/crow-v15-consistent-long-flight-20261113/libmetalrobo.dylib'", "R/'build/crow-v15-consistent-wing-drive-20261113/libmetalrobo.dylib'");(A/'build.py').write_text(s)
for name in ('push-wing','hold-wing'):subprocess.run(['python3',str(A/'build.py'),name],check=True)
s=gzip.decompress((R/'artifacts'/previous/'observer.mm.gz').read_bytes()).decode();assert s.count('traceStep<=400')==1 and s.count('crowTraceStep <= 400')==1 and s.count('crowTraceStep<=400')==5;s=s.replace('traceStep<=400','traceStep<=500').replace('crowTraceStep <= 400','crowTraceStep <= 500').replace('crowTraceStep<=400','crowTraceStep<=500');(B/'MetalWorld.mm').write_text(s);(A/'observer.mm.gz').write_bytes(gzip.compress(s.encode(),mtime=0))
s=(R/'artifacts'/previous/'build-observer.py').read_text().replace(previous,new).replace("B/'c_api.cpp.o'", "R/'build/crow-v15-consistent-wing-drive-20261113/c_api.cpp.o'");(A/'build-observer.py').write_text(s);subprocess.run(['python3',str(A/'build-observer.py')],check=True)
s=(R/'artifacts'/previous/'run.py').read_text().replace(previous,new).replace("M=R/'build/crow-v15-consistent-long-flight-20261113'", "M=R/'build/crow-v15-consistent-wing-drive-20261113'").replace("('--steps','400')", "('--steps','500')");(A/'run.py').write_text(s)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({str(p.relative_to(R)):h(p) for p in (B/'MetalWorld.mm',B/'libmetalrobo.dylib',B/'push-wing.metallib',B/'hold-wing.metallib',B/'World.air')},indent=2)+'\n');print((A/'build-hashes.json').read_text())
