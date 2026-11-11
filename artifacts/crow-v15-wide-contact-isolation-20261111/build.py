from pathlib import Path
import hashlib,shlex,subprocess,shutil
R=Path.cwd();A=R/'artifacts/crow-v15-wide-contact-isolation-20261111';B=R/'build/crow-v15-wide-contact-isolation-20261111';P=R/'build/crow-v15-aba-40-body-isolation-20261111';(B/'include/metalrobo').mkdir(parents=True,exist_ok=True);shutil.copy2(P/'include/metalrobo/engine_types.h',B/'include/metalrobo/engine_types.h');shutil.copy2(R/'build/crow-v15-aba-floating-gravity-screen-20261111/aba.metal',B/'source/ArticulatedABA.metal');shutil.copy2(R/'src/metal/ArticulatedParallelABA.metal',B/'source/ArticulatedParallelABA.metal')
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();sourceFiles=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm'];objects={}
for i,name in enumerate(sourceFiles):
 line=next(x for x in lines if ' -c '+str(R/name) in x);args=shlex.split(line);args.insert(1,'-I'+str(B/'include'));obj=B/(str(i)+'.o');original=args[args.index('-o')+1];objects[original]=str(obj)
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 subprocess.run(args,cwd=R/'build',check=True);print('compiled host',name,flush=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
for old,new in objects.items():assert old in args,old;args[args.index(old)]=new
subprocess.run(args,cwd=R/'build',check=True);print('linked host',hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
master=next(x for x in lines if 'ArticulatedInverseMass.metal' in x and 'metallib' in x);replaced={};names={'ArticulatedABA.metal','ArticulatedParallelABA.metal','ArticulatedInverseMass.metal','FunctionBasedDenseDynamics.metal'}
for part in master.split(' && '):
 if not part.startswith('xcrun -sdk macosx metal '):continue
 args=shlex.split(part);source=Path(args[args.index('-c')+1]);
 if source.name not in names:continue
 out=Path(args[args.index('-o')+1]);dest=B/'shaders'/out.name;args[args.index('-o')+1]=str(dest);args[4:4]=['-I',str(B/'include'),'-I',str(R/'src/metal')]
 if source.name in ('ArticulatedABA.metal','ArticulatedParallelABA.metal'):args[args.index('-c')+1]=str(B/'source'/source.name)
 subprocess.run(args,cwd=R/'build',check=True);replaced[out.name]=dest;print('compiled shader',out.name,flush=True)
assert len(replaced)>=12,len(replaced)
airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air'))];ml=B/'MetalRobo.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,'-o',str(ml)],check=True);print('linked metallib',hashlib.sha256(ml.read_bytes()).hexdigest(),flush=True)
