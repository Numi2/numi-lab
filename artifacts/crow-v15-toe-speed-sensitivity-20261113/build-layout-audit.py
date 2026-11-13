from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-toe-speed-sensitivity-20261113';B=R/'build/crow-v15-toe-speed-sensitivity-20261113';W=R/'build/crow-v15-wide-contact-isolation-20261111';T=R/'build/crow-v15-corvid-toe-contact-admission-20261112/TaskProgram-layout-audit.cpp.o';P=R/'build/crow-v15-drive-target-audit-20261113';O=R/'build/crow-v15-gait-slip-observer-20261112/c_api.cpp.o';lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);report={}
for label,obj in [('baseline',O),('modified',B/'c_api.cpp.o')]:
 d=B/('layout-'+label);d.mkdir(exist_ok=False);args=shlex.split(line[len(': && '):-len(' && :')]);out=d/'libmetalrobo.dylib';args[args.index('-o')+1]=str(out);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(T)
 files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
 for i,n in enumerate(files):
  key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(P/'MetalWorld.mm.o' if i==4 else W/(str(i)+'.o'))
 subprocess.run(args,cwd=R/'build',check=True);report[label]=hashlib.sha256(out.read_bytes()).hexdigest()
report['audit_task_object_sha256']=hashlib.sha256(T.read_bytes()).hexdigest();(A/'layout-host-hashes.json').write_text(json.dumps(report,indent=2)+'\n')
