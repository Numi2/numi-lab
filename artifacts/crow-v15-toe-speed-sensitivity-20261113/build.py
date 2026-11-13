from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-toe-speed-sensitivity-20261113';B=R/'build/crow-v15-toe-speed-sensitivity-20261113';W=R/'build/crow-v15-wide-contact-isolation-20261111';M=R/'build/crow-v15-matched-binary-gait-screen-20261112';P=R/'build/crow-v15-drive-target-audit-20261113'
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')];obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/'c_api.cpp');subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(M/'TaskProgram.cpp.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(P/'MetalWorld.mm.o' if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({'host_sha256':h(B/'libmetalrobo.dylib'),'c_api_source_sha256':h(B/'c_api.cpp'),'c_api_object_sha256':h(obj),'world_object_sha256':h(P/'MetalWorld.mm.o')},indent=2)+'\n');print('built',h(B/'libmetalrobo.dylib'))
