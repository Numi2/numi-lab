from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-extended-flight-20261109';B=R/'build/crow-extended-flight-20261109';B.mkdir(exist_ok=True)
base=(R/'src/c_api.cpp').read_text(); observed=(R/'build/crow-route-collision-control-20261109/c_api.cpp').read_text()
needle='''        applyRunProfile(manifest, *config);
        manifest.profile.capacities = manifest.task.capacities;
        std::optional<metalrobo::FiniteResourceConfig> resources;'''
repl='''        applyRunProfile(manifest, *config);
        if (activeNavigation) manifest.task.maximumEpisodeSteps = 3000u;
        manifest.profile.capacities = manifest.task.capacities;
        std::optional<metalrobo::FiniteResourceConfig> resources;'''
assert observed.count(needle)==1
extended=observed.replace(needle,repl);csrc=B/'c_api.cpp';csrc.write_text(extended)
(A/'runtime.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),extended.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/src/c_api.cpp')))
full=(R/'artifacts/crow-flight-source-diagnostics-20261108/flight-bench-runtime-diagnostic.patch').read_text(); segment='diff --git a/src/core/RunProgram.cpp'+full.split('diff --git a/src/core/RunProgram.cpp',1)[1]
pf=B/'collision-disable.patch';pf.write_text(segment);rsrc=B/'RunProgram.cpp'
subprocess.run(['patch','-s','-o',str(rsrc),str(R/'src/core/RunProgram.cpp'),str(pf)],check=True)
(A/'collision-disable.patch').write_text(''.join(difflib.unified_diff((R/'src/core/RunProgram.cpp').read_text().splitlines(True),rsrc.read_text().splitlines(True),fromfile='a/src/core/RunProgram.cpp',tofile='b/src/core/RunProgram.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
def compile_one(original,source,name):
 line=next(x for x in lines if ' -c '+str(original) in x);args=shlex.split(line);obj=B/(name+'.o')
 for k,v in [('-o',str(obj)),('-MF',str(B/(name+'.d'))),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(source);subprocess.run(args,cwd=R/'build',check=True);return obj,args
cobj,cargs=compile_one(R/'src/c_api.cpp',csrc,'c_api');robj,rargs=compile_one(R/'src/core/RunProgram.cpp',rsrc,'RunProgram')
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);assert line.endswith(' && :')
baseargs=shlex.split(line[len(': && '):-len(' && :')]);libs={}
for kind in ('collision-on','collision-off'):
 d=B/kind;d.mkdir(exist_ok=True);lib=d/'libmetalrobo.dylib';args=baseargs[:]
 args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(cobj)
 if kind=='collision-off':args[args.index('CMakeFiles/metalrobo.dir/src/core/RunProgram.cpp.o')]=str(robj)
 subprocess.run(args,cwd=R/'build',check=True);libs[kind]=hashlib.sha256(lib.read_bytes()).hexdigest()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'runtime-build.json').write_text(json.dumps({'runtime_source_sha256':h(csrc),'collision_off_scene_source_sha256':h(rsrc),'libraries_sha256':libs,'compile_commands':[cargs,rargs],'task_horizon':3000},indent=2)+'\n')
print(json.dumps(libs,indent=2),flush=True)
