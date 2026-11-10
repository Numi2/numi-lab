from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-contact-group-diagnostic-20261110';B=R/'build/crow-v15-contact-group-diagnostic-20261110';base=(R/'build/crow-v15-integrated-route-baseline-20261110/c_api.cpp').read_text()
needle='''        if (feeding && !cueConfig) {
            handle->resourceRuntime='''
insert='''        if (std::getenv("CROW_GROUP_DIAGNOSTIC")) {
            const auto& model=handle->run.model();
            const auto& groups=handle->taskProgram.contactGroups();
            for (std::size_t i=0;i<groups.size();++i) {
                const auto& g=groups[i];auto b=g.reference.x;
                std::fprintf(stderr,"CROW_GROUP %zu body %u name %s local %.9g %.9g %.9g kinematic %.9g %.9g %.9g flags %u\\n",i,b,b<model.bodyNames.size()?model.bodyNames[b].c_str():"invalid",g.localReference.x,g.localReference.y,g.localReference.z,g.kinematicReference.x,g.kinematicReference.y,g.kinematicReference.z,g.members.z);
            }
        }
'''
assert base.count(needle)==1;s=base.replace(needle,insert+needle);src=B/'c_api.cpp';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/base.cpp',tofile='b/groups.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-causal-launch-20261110/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest())
