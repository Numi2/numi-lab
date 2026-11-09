from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';C=B/'late-observer';C.mkdir(exist_ok=True)
base=(B/'c_api.cpp').read_text();old='if (step>350u) continue;';new='if (step>350u && consecutive[lane]>50u && step<2976u && step%100u!=0u) continue;';assert base.count(old)==1;s=base.replace(old,new);src=C/'c_api.cpp';src.write_text(s);original=(R/'src/c_api.cpp').read_text()
(A/'late-observer.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/late-contact-observer.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=C/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'late-observer-build.json').write_text(json.dumps({'source_sha256':h(src),'library_sha256':h(lib),'patch_sha256':h(A/'late-observer.patch'),'scope':'read-only accepted-contact observer; launch and terminal perch'},indent=2)+'\n')
