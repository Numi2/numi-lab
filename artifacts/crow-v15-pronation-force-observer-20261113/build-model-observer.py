from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path.cwd();B=R/'build/crow-v15-pronation-force-observer-20261113';W=R/'build/crow-v15-wide-contact-isolation-20261111';A=R/'artifacts/crow-v15-pronation-force-observer-20261113'
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/'c_api.cpp');args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'src/metal'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109'),'-I'+str(R/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111')];subprocess.run(args,cwd=R/'build',check=True)
(A/'model-observer-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'c_api.cpp',obj)},indent=2)+'\n')
