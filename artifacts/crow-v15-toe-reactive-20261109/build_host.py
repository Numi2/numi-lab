from pathlib import Path
import difflib,shlex,subprocess,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-toe-reactive-20261109';B=R/'build/crow-v15-toe-reactive-20261109';old=R/'build/crow-v15-grip-search-20261109/toe-envelope/c_api.cpp';base=old.read_text()
needle='''        if (retinal) {
            auto p=visualSensorProgram(*visual_sensor);p.retinal=retina;'''
insert='''        if (activeNavigation) {
            manifest.task.contactGroups.push_back({.id="left_rear_toe_wrench",.bodies={"crow_left_rear_toe"},.referenceBody="crow_left_rear_toe"});
            manifest.task.contactGroups.push_back({.id="right_rear_toe_wrench",.bodies={"crow_right_rear_toe"},.referenceBody="crow_right_rear_toe"});
        }
        if (retinal) {
            auto p=visualSensorProgram(*visual_sensor);p.retinal=retina;''';assert base.count(needle)==1;s=base.replace(needle,insert);src=B/'c_api.cpp';src.write_text(s);(A/'host.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/toe-envelope.cpp',tofile='b/toe-reactive.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True)
(A/'host-build.json').write_text(json.dumps({'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'host_patch_sha256':hashlib.sha256((A/'host.patch').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((A/'plan.md').read_bytes()).hexdigest()},indent=2)+'\n');print('built')
