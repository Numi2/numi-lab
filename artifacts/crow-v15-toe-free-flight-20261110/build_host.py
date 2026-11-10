from pathlib import Path
import difflib,shlex,subprocess,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-toe-free-flight-20261110';B=R/'build/crow-v15-toe-free-flight-20261110';base=(R/'build/crow-v15-visual-binding-fix-20261110/c_api.cpp').read_text();needle='''        manifest.sensors.id = cueConfig ? "camera_admission_proprioception" : journey''';assert base.count(needle)==1
insert='''        // Experimental obstacle-free control: retain bodies and ground, disable only course collisions.
        if (activeNavigation) {
            unsigned removed=0;
            for (auto& object:manifest.scene.objects) {
                auto& mechanics=object.mechanics;
                for (std::size_t i=0;i<mechanics.shapeNames.size();++i) {
                    if (mechanics.shapeNames[i].starts_with("crow_course_")) {
                        mechanics.shapes.at(i).collisionMask=0u;
                        ++removed;
                    }
                }
            }
            if (removed!=5u) throw std::logic_error("expected five collision-disabled course shapes");
        }
'''+needle
s=base.replace(needle,insert);src=B/'c_api.cpp';src.write_text(s);(A/'obstacle-off.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/obstacles-on.cpp',tofile='b/obstacles-off.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
