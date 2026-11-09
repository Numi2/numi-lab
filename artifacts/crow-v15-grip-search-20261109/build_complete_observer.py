from pathlib import Path
import subprocess,shlex,difflib,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer';C.mkdir(exist_ok=True)
for base,patch,out in [(R/'src/c_api.cpp',R/'artifacts/crow-v15-ground-contact-20261109/late-observer.patch',C/'c_api.cpp'),(R/'src/metal/MetalWorld.mm',R/'artifacts/crow-v15-contact-slip-20261109/materialized-velocity-host.patch',C/'MetalWorld.mm')]:
 subprocess.run(['patch','--silent','-o',str(out),str(base),str(patch)],check=True)
src=C/'c_api.cpp';s=src.read_text();old='''                    if (opened) std::fprintf(stderr,"]}\\n");
''';new='''                    if (opened) std::fprintf(stderr,"]}\\n");
                    else {
                        std::fprintf(stderr, "CROW_SUPPORT_SAMPLE {\\\"step\\\":%llu,\\\"lane\\\":%zu",
                            static_cast<unsigned long long>(handle->submittedControlSteps+1u), lane);
                        array("q",published.finalQ,lane*handle->world.nq(),handle->world.nq());
                        array("v",published.finalV,lane*handle->world.nv(),handle->world.nv());
                        array("policy_action",published.policyActions,lane*handle->taskProgram.layout().actionCount,handle->taskProgram.layout().actionCount);
                        std::fprintf(stderr, ",\\\"contacts\\\":[]}\\n");
                    }
'''
assert s.count(old)==1;s=s.replace(old,new);src.write_text(s)
(A/'complete-observer.patch').write_text(''.join(difflib.unified_diff((R/'src/c_api.cpp').read_text().splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/complete-contact-observer.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
for source,extra in [('src/c_api.cpp',R/'artifacts/crow-clearance-observer-20261109'),('src/metal/MetalWorld.mm',R/'src/metal')]:
 original=R/source;line=next(x for x in lines if ' -c '+str(original) in x);args=shlex.split(line);local=C/Path(source).name;obj=C/(Path(source).name+'.o')
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(local);args.insert(1,'-I'+str(extra));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(C/'c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(C/'MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'complete-observer-build.json').write_text(json.dumps({'library_sha256':h(lib),'observer_patch_sha256':h(A/'complete-observer.patch'),'materialization_patch_sha256':h(R/'artifacts/crow-v15-contact-slip-20261109/materialized-velocity-host.patch'),'plan_sha256':h(A/'observer-plan.md')},indent=2)+'\n')
print('built',lib,'sha256',h(lib))
