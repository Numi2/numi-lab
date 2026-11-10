from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-toe-contact-isolation-20261110';B=R/'build/crow-v15-toe-contact-isolation-20261110';base=(R/'build/crow-v15-toe-free-flight-20261110/c_api.cpp').read_text()
needle='''        if (std::getenv("CROW_CLEARANCE_TRACE") != nullptr &&
'''
insert='''        if (std::getenv("CROW_TOE_PAIR_TRACE") && control_step_count==1u &&
            ((handle->submittedControlSteps+1u)%100u)==0u) {
          const auto stride=published.layout.contactDispatch.constraintStride;
          for (size_t lane=0; lane<environmentCount && lane<published.contactStatuses.size(); ++lane) {
            const size_t count=published.contactStatuses[lane].requiredConstraints;
            std::fprintf(stderr,"TOE_PAIRS {\\"step\\":%llu,\\"lane\\":%zu,\\"count\\":%zu,\\"pairs\\":[",
              static_cast<unsigned long long>(handle->submittedControlSteps+1u),lane,count);
            bool first=true;
            for(size_t i=0;i<count && lane*stride+i<published.contactEvidence.contacts.size();++i) {
              const auto& c=published.contactEvidence.contacts[lane*stride+i];
              if (c.bodyA>=handle->model.bodyNames.size() || c.bodyB>=handle->model.bodyNames.size()) continue;
              std::fprintf(stderr,"%s{\\"a\\":\\"%s\\",\\"b\\":\\"%s\\",\\"impulse\\":%.9g,\\"separation\\":%.9g}",
                first?"":",",handle->model.bodyNames[c.bodyA].c_str(),handle->model.bodyNames[c.bodyB].c_str(),c.impulses.x,c.pointAndSeparation.w);
              first=false;
            }
            std::fprintf(stderr,"]}\\n");
          }
        }
'''
assert base.count(needle)==1
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();compile=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
for name in ('on','off'):
 C=B/name;C.mkdir(exist_ok=True);srcText=base.replace(needle,insert+needle)
 if name=='off':
  old='MRShapeGPU shape=m.shapes.front(); shape.bodyIndex=bi;'; assert srcText.count(old)==1;srcText=srcText.replace(old,old+' shape.collisionMask=0u;')
 src=C/'c_api.cpp';src.write_text(srcText);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),srcText.splitlines(True),fromfile='a/base.cpp',tofile='b/'+name+'.cpp')));args=shlex.split(compile);obj=C/'c_api.cpp.o'
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
 args=shlex.split(link[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
