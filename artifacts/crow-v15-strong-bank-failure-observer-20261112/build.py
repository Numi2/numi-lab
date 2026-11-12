from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-strong-bank-failure-observer-20261112';B=R/'build/crow-v15-strong-bank-failure-observer-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';O=R/'build/crow-v15-hip-roll-obstacle-free-control-20261111'
base=(R/'build/crow-v15-gait-target-observer-20261112/MetalWorld.mm').read_text();s=base
s=s.replace('sub==0 && crowTraceStep>=201 && crowTraceStep<=300','sub==0 && lane==4 && crowTraceStep>=1280 && crowTraceStep<=1305').replace('crowTraceStep >= 201 && crowTraceStep <= 300','lane==4 && crowTraceStep >= 1280 && crowTraceStep <= 1305')
needle='for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];'
addition='''for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];
                                if(lane==4 && crowTraceStep>=1280 && crowTraceStep<=1305 && c.impulses.x>0.f)
                                    std::fprintf(stderr,"CROW_FAILURE_PAIR %llu %lu %lu %u %u %llu %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,(unsigned long long)c.pairKey,c.impulses.x,c.impulses.y,c.impulses.z,c.pointAndSeparation.x,c.pointAndSeparation.y,c.pointAndSeparation.z,c.pointAndSeparation.w,c.normal.x,c.normal.y,c.normal.z,c.targetVelocityAndPreSolveNormal.w);'''
assert s.count(needle)==1;s=s.replace(needle,addition);world=B/'MetalWorld.mm';world.write_text(s);(A/'world-observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.mm',tofile='b/observer.mm')))
base=(O/'c_api.cpp').read_text();s=base;needle='(handle->submittedControlSteps + 1u) % 100u == 0u ||';replacement='(handle->submittedControlSteps + 1u) % 100u == 0u || (handle->submittedControlSteps >= 1278u && handle->submittedControlSteps < 1305u) ||';assert s.count(needle)==1;s=s.replace(needle,replacement);api=B/'c_api.cpp';api.write_text(s);(A/'api-observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference-api.cpp',tofile='b/observer-api.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
def compile(src, original, obj, includes):
 line=next(x for x in lines if ' -c '+str(R/original) in x);args=shlex.split(line)
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src);args[1:1]=includes;subprocess.run(args,cwd=R/'build',check=True)
compile(world,'src/metal/MetalWorld.mm',B/'world.o',['-I'+str(W/'include'),'-I'+str(R/'src/metal'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')])
compile(api,'src/c_api.cpp',B/'api.o',['-I'+str(W/'include'),'-I'+str(R/'artifacts/crow-v15-two-segment-frozen-flight-setup-20261111'),'-I'+str(R/'artifacts/crow-clearance-observer-20261109')])
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(B/'api.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 key='CMakeFiles/metalrobo.dir/'+n+'.o';assert key in args;args[args.index(key)]=str(B/'world.o' if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'hashes.json').write_text(json.dumps({'observer_host':h(B/'libmetalrobo.dylib'),'reference_host':h(O/'libmetalrobo.dylib'),'world_source':h(world),'api_source':h(api)},indent=2)+'\n')
