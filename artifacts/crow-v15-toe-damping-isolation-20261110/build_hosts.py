from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-toe-damping-isolation-20261110';B=R/'build/crow-v15-toe-damping-isolation-20261110';base=(R/'build/crow-v15-toe-free-flight-20261110/c_api.cpp').read_text();old='b.dampingAndSpeedLimits={.015f,.015f,30,35};';assert base.count(old)==1
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();compile=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
for name,value in [('zero','unused')]:
 C=B/name;C.mkdir(exist_ok=True);s=base.replace(old,'b.dampingAndSpeedLimits={0.f,0.f,30,35};');src=C/'c_api.cpp';src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/base.cpp',tofile='b/'+name+'.cpp')));args=shlex.split(compile);obj=C/'c_api.cpp.o'
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
 args=shlex.split(link[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
