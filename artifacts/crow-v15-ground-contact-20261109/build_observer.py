from pathlib import Path
import difflib, hashlib, json, shlex, subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-ground-contact-20261109'; B=R/'build/crow-v15-ground-contact-20261109'
original=(R/'src/c_api.cpp').read_text()
base=(R/'build/crow-conditional-native-observer-20261109/c_api.cpp').read_text()
old='''if (a.find("crow_course_perch")==std::string::npos &&
                            b.find("crow_course_perch")==std::string::npos) continue;'''
new='''if (a.find("crow_course_perch")==std::string::npos &&
                            b.find("crow_course_perch")==std::string::npos &&
                            a.find("locomotion_ground")==std::string::npos &&
                            b.find("locomotion_ground")==std::string::npos) continue;'''
assert base.count(old)==2,base.count(old)
s=base.replace(old,new)
old='if (consecutive[lane]>50u && step<2976u && step%100u!=0u) continue;'
assert s.count(old)==1
s=s.replace(old,'if (step>350u) continue;')
old='''array("v",published.finalV,lane*handle->world.nv(),handle->world.nv());
                            std::fprintf(stderr,",\\\"contacts\\\":[");'''
new='''array("v",published.finalV,lane*handle->world.nv(),handle->world.nv());
                            array("policy_action",published.policyActions,lane*handle->taskProgram.layout().actionCount,handle->taskProgram.layout().actionCount);
                            std::fprintf(stderr,",\\\"contacts\\\":[");'''
assert s.count(old)==1,s.count(old)
s=s.replace(old,new)
src=B/'c_api.cpp';src.write_text(s)
(A/'observer.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/ground-contact-observer.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj)
subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'build.json').write_text(json.dumps({'observer_source_sha256':h(src),'observer_library_sha256':h(lib),'original_api_sha256':h(R/'src/c_api.cpp'),'scope':'read-only accepted-contact instrumentation; no physical control changes'},indent=2)+'\n')
