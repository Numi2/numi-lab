from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-toe-damping-slip-screen-20261112';B=R/'build/crow-v15-toe-damping-slip-screen-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';M=R/'build/crow-v15-matched-binary-gait-screen-20261112';P=R/'build/crow-v15-gait-slip-observer-20261112';O=R/'build/crow-v15-substep-slip-observer-20261112'
base=(P/'c_api.cpp').read_text();old='d.drive={.1f,.003f,.00005f,0};';assert base.count(old)==1
source=base.replace(old,'d.drive={.1f,.030f,.00005f,0};')
marker='m.dofs.push_back(d);m.dofNames.push_back(name+"_flex");'
assert source.count(marker)==1
source=source.replace(marker,'if(std::getenv("CROW_VERIFY_TOE_DAMPING")) std::fprintf(stderr,"CROW_TOE_DAMPING %s %.9g\\n",name.c_str(),d.drive.y);'+marker)
(B/'c_api.cpp').write_text(source)
(A/'physical-model.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/observer_c_api.cpp',tofile='b/toe_damping_c_api.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(line);args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')]
obj=B/'c_api.cpp.o'
for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(B/'c_api.cpp');subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(M/'TaskProgram.cpp.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(O/'MetalWorld.mm.o' if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/'hashes.json').write_text(json.dumps({'reference_host_sha256':h(O/'libmetalrobo.dylib'),'candidate_host_sha256':h(B/'libmetalrobo.dylib'),'candidate_source_sha256':h(B/'c_api.cpp'),'toe_drive_damping_candidate':.03,'reference':.003},indent=2)+'\n')
