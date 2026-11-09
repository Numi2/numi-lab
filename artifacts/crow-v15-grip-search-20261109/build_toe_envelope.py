from pathlib import Path
import difflib,shlex,subprocess,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'toe-envelope';C.mkdir(exist_ok=True)
base=(B/'toe-integration/c_api.cpp').read_text();needle='''      shape.contactRestAndBoundingRadius.z=std::sqrt(.010f*.010f+shape.dimensions.y*shape.dimensions.y+shape.dimensions.z*shape.dimensions.z);''';assert base.count(needle)==1;s=base.replace(needle,'      // Keep the parent sole bounding envelope conservative for legacy observations.')
src=C/'c_api.cpp';src.write_text(s);(A/'toe-envelope-host.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/toe-integration.cpp',tofile='b/toe-envelope.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=C/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(B/'complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'toe-envelope-build.json').write_text(json.dumps({'library_sha256':h(lib),'patch_sha256':h(A/'toe-envelope-host.patch'),'plan_sha256':h(A/'toe-envelope-plan.md')},indent=2)+'\n')
