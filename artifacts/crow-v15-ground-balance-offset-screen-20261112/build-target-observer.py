from pathlib import Path
import shlex,subprocess
R=Path.cwd();B=R/'build/crow-v15-ground-balance-offset-screen-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/'MetalWorld.mm');args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'src/metal'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')];subprocess.run(args,cwd=R/'build',check=True)
s=(R/'artifacts/crow-v15-shank-proxy-trim-screen-20261112/build.py').read_text().replace('artifacts/crow-v15-shank-proxy-trim-screen-20261112','artifacts/crow-v15-ground-balance-offset-screen-20261112').replace('build/crow-v15-shank-proxy-trim-screen-20261112','build/crow-v15-ground-balance-offset-screen-20261112').replace('H/"MetalWorld.mm.o"','B/"MetalWorld.mm.o"').replace('(A/"hashes.json")','(A/"target-host-hashes.json")');exec(compile(s,'observer-build','exec'))
