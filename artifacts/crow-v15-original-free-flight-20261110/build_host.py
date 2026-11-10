from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-original-free-flight-20261110';B=R/'build/crow-v15-original-free-flight-20261110';base=(R/'build/crow-v15-toe-free-flight-20261110/c_api.cpp').read_text();old='        manifest.robot = activeNavigation ? crowToeModel() : std::move(*robot);';assert base.count(old)==1;s=base.replace(old,'        manifest.robot = std::move(*robot);')
old='''        if (activeNavigation) {
            for (auto& group : manifest.task.contactGroups) {
                const std::string side=group.id=="left_foot_contact" ? "left" : group.id=="right_foot_contact" ? "right" : "";
                if (side.empty()) continue;
                for (unsigned digit=1u;digit<=3u;++digit) group.bodies.push_back("crow_"+side+"_front_toe_"+std::to_string(digit));
                group.bodies.push_back("crow_"+side+"_rear_toe");
            }
            manifest.task.contactGroups.push_back({.id="left_rear_toe_wrench",.bodies={"crow_left_rear_toe"},.referenceBody="crow_left_rear_toe"});
            manifest.task.contactGroups.push_back({.id="right_rear_toe_wrench",.bodies={"crow_right_rear_toe"},.referenceBody="crow_right_rear_toe"});
        }
''';assert s.count(old)==1;s=s.replace(old,'');src=B/'c_api.cpp';src.write_text(s);(A/'original-body.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/toe-body.cpp',tofile='b/original-body.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
