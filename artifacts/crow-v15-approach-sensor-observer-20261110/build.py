from pathlib import Path
import subprocess,shlex,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-approach-sensor-observer-20261110';B=R/'build/crow-v15-approach-sensor-observer-20261110';base=(R/'build/crow-v15-causal-launch-20261110/MetalWorld.mm').read_text()
old='''std::fprintf(stderr, "CROW_TASK_STATE %u %u %u %.9g %.9g %.9g\\n", lane, x.episode.x, x.status.w, x.navigation.z, x.nutrition.x, x.nutrition.y);'''
new='''std::fprintf(stderr, "CROW_TASK_STATE %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g\\n", lane, x.episode.x, x.status.w, x.navigation.z, x.nutrition.x, x.nutrition.y, x.nutrition.z, x.threatTeacher.w, x.commandExtension.z);'''
assert base.count(old)==1;s=base.replace(old,new);src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/base.mm',tofile='b/observer.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(R/'build/crow-v15-integrated-route-baseline-20261110/c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest())
