from pathlib import Path
import difflib,hashlib,json,os,shlex,subprocess
R=Path.cwd();B=R/'build/crow-route-collision-control-20261109';B.mkdir(exist_ok=True);A=R/'artifacts/crow-route-collision-control-20261109'
full=(R/'artifacts/crow-flight-source-diagnostics-20261108/flight-bench-runtime-diagnostic.patch').read_text()
patch=full.split('diff --git a/src/core/RunProgram.cpp')[0]
patchfile=B/'observer.patch';patchfile.write_text(patch)
source=B/'c_api.cpp'
subprocess.run(['patch','-s','-o',str(source),str(R/'src/c_api.cpp'),str(patchfile)],check=True)
(A/'observer-c-api.patch').write_text(''.join(difflib.unified_diff((R/'src/c_api.cpp').read_text().splitlines(True),source.read_text().splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/src/c_api.cpp')))
out=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True)
lines=out.splitlines()
compile_line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(compile_line)
obj=B/'c_api.o'
for key,val in [('-o',str(obj)),('-MF',str(B/'c_api.d')),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(source)
subprocess.run(args,cwd=R/'build',check=True)
link_line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
assert link_line.endswith(' && :')
args=shlex.split(link_line[len(': && '):-len(' && :')])
lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
i=args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o');args[i]=str(obj)
subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'diagnostic_source_sha256':h(source),'diagnostic_object_sha256':h(obj),'diagnostic_library_sha256':h(lib),'base_c_api_sha256':h(R/'src/c_api.cpp'),'original_collision_scene_object_sha256':h(R/'build/CMakeFiles/metalrobo.dir/src/core/RunProgram.cpp.o'),'compile_argv':args if False else shlex.split(compile_line),'note':'observer c_api only; original collision-enabled RunProgram object linked'}
(A/'observer-build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='compile_argv'},indent=2),flush=True)
