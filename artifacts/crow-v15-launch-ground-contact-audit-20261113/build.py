from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-launch-ground-contact-audit-20261113';B=R/'build/crow-v15-launch-ground-contact-audit-20261113';M=R/'build/crow-v15-matched-binary-gait-screen-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';P=R/'build/crow-v15-gait-slip-observer-20261112'
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/'MetalWorld.mm');args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'src/metal'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')];subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(P/'c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(M/'TaskProgram.cpp.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(obj if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
def h(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()
(A/'observer-build.json').write_text(json.dumps({'host_sha256':h(B/'libmetalrobo.dylib'),'source_sha256':h(B/'MetalWorld.mm'),'object_sha256':h(obj)},indent=2)+'\n');print('Observer built:',h(B/'libmetalrobo.dylib'))
