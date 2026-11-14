from pathlib import Path
import json,subprocess,shlex,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107');a=R/'artifacts/crow-v15-full-flight-pronation-observer-20261114';b=R/'build/crow-v15-full-flight-pronation-observer-20261114';b.mkdir(exist_ok=True)
src=a/'observer.mm'
entries=json.loads((R/'build/compile_commands.json').read_text());c=shlex.split(next(x['command'] for x in entries if x['file'].endswith('/MetalWorld.mm')));c.insert(1,'-I'+str(R/'build/crow-v15-wide-contact-isolation-20261111/include'));c.extend(['-I',str(R/'src/metal')]);c[c.index('-o')+1]=str(b/'MetalWorld.mm.o');c[c.index('-c')+1]=str(src);subprocess.run(c,cwd=R/'build',check=True)
out=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands','lib/libmetalrobo.dylib'],text=True).splitlines()[-1];assert out.startswith(': && ') and out.endswith(' && :');args=shlex.split(out[5:-5]);args[args.index('-o')+1]=str(b/'libmetalrobo.dylib');oldobj='CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o';assert oldobj in args;args[args.index(oldobj)]=str(b/'MetalWorld.mm.o');wide=R/'build/crow-v15-wide-contact-isolation-20261111';names={0:'src/core/CompiledMetalMultiArticulatedProgram.cpp',1:'src/core/CompiledMetalMultiArticulatedContactProgram.cpp',2:'src/metal/MetalArticulatedInverseMass.mm',3:'src/metal/MetalArticulatedABA.mm',5:'src/metal/MetalMultiArticulatedContact.mm',6:'src/metal/MetalMultiArticulatedABA.mm',7:'src/metal/MetalMultiArticulatedInverseMass.mm'}
for i,n in names.items():
 k='CMakeFiles/metalrobo.dir/'+n+'.o';assert k in args,k;args[args.index(k)]=str(wide/(str(i)+'.o'))
taskobj='CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o';assert taskobj in args;args[args.index(taskobj)]=str(R/'build/crow-v15-matched-binary-gait-screen-20261112/TaskProgram.cpp.o')
capi='CMakeFiles/metalrobo.dir/src/c_api.cpp.o';assert capi in args;args[args.index(capi)]=str(R/'build/crow-v15-mirrored-pronation-20261113/c_api.cpp.o');subprocess.run(args,cwd=R/'build',check=True)
(a/'build-commands.json').write_text(json.dumps({'compile':c,'link':args},indent=2)+'\n');print('built observational host')
