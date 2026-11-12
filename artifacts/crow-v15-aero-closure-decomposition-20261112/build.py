from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-aero-closure-decomposition-20261112'; B=R/'build/crow-v15-aero-closure-decomposition-20261112'; W=R/'build/crow-v15-wide-contact-isolation-20261111'; M=R/'build/crow-v15-matched-binary-gait-screen-20261112'; P=R/'build/crow-v15-gait-slip-observer-20261112'; original=R/'build/crow-v15-aerodynamic-wrench-audit-20261112/MetalWorld.mm'; source=B/'MetalWorld.mm'
(A/'observer.patch').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),source.read_text().splitlines(True),fromfile='a/observer.mm',tofile='b/slip_observer.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);args.insert(1,'-I'+str(W/'include'));args.insert(1,'-I'+str(R/'src/metal'));obj=B/'MetalWorld.mm.o'
for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(source);subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(R/'build/crow-v15-smooth-launch-open-flight-20261112/c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(M/'TaskProgram.cpp.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(obj if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/'hashes.json').write_text(json.dumps({'host_sha256':h(B/'libmetalrobo.dylib'),'source_sha256':h(source),'reference_host_sha256':h(R/'build/crow-v15-stance-contact-velocity-audit-20261112/libmetalrobo.dylib')},indent=2)+'\n')
