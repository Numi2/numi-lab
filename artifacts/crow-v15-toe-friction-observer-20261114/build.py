from pathlib import Path
import difflib,hashlib,json,shlex,subprocess,gzip
R=Path.cwd();A=R/'artifacts/crow-v15-toe-friction-observer-20261114';B=R/'build/crow-v15-toe-friction-observer-20261114';O=R/'build/crow-v15-toe-effort-observer-20261114';P=R/'build/crow-v15-wide-contact-isolation-20261111';B.mkdir(parents=True,exist_ok=True)
base=(O/'MetalWorld.mm').read_text();s=base
def once(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
needle='if(side<0){other+=c.impulses.x;continue;}const float vz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;'
once(needle,'if(side<0){other+=c.impulses.x;continue;}std::fprintf(stderr,"CROW_TOE_FRICTION %llu %lu %lu %u %.9g %.9g %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,body,c.impulses.x,c.impulses.y,c.impulses.z,c.friction.x,c.friction.y);const float vz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;')
(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/toe-effort-observer.mm',tofile='b/toe-friction-observer.mm')));src=B/'MetalWorld.mm';src.write_text(s)
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);args.insert(1,'-I'+str(P/'include'));args.insert(1,'-I'+str(R/'src/metal'));obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');names=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,name in enumerate(names):
 old='CMakeFiles/metalrobo.dir/'+name+'.o';assert old in args;args[args.index(old)]=str(obj if i==4 else P/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lib=B/'libmetalrobo.dylib';out={'prior_observer_source':h(O/'MetalWorld.mm'),'prior_library':h(O/'libmetalrobo.dylib'),'new_source':h(src),'new_library':h(lib),'patch':h(A/'observer.patch')};(A/'build-hashes.json').write_text(json.dumps(out,indent=2)+'\n')
with (A/'libmetalrobo.dylib.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(lib.read_bytes())
