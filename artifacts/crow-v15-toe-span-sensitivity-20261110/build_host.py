from pathlib import Path
import difflib,shlex,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-toe-span-sensitivity-20261110';B=R/'build/crow-v15-toe-span-sensitivity-20261110';base=(R/'build/crow-v15-long-horizon-20261110/c_api.cpp').read_text()
old='float dx[4]={.020666667f,.020666667f,.020666667f,-.062f},dy[4]={-.012f,0,.012f,0};'
new='float dx[4]={.015f,.015f,.015f,-.045f},dy[4]={-.012f,0,.012f,0};'
assert base.count(old)==1;s=base.replace(old,new);src=B/'c_api.cpp';src.write_text(s);(A/'toe-span.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/long-c_api.cpp',tofile='b/toe-span-c_api.cpp')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();compile=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(compile);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-causal-launch-20261110/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
