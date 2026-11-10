from pathlib import Path
import subprocess,shlex,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-grip-transition-substep-audit-20261110';B=R/'build/crow-v15-grip-transition-substep-audit-20261110';old=(R/'build/crow-v15-substep-contact-observer-20261110/MetalWorld.mm').read_text();s=old
s=s.replace('crowTraceStep >= 4375 && crowTraceStep <= 4410','crowTraceStep >= 1 && crowTraceStep <= 120')
a='''                                const bool toeA=c.bodyA>=17u && c.bodyA<=24u, toeB=c.bodyB>=17u && c.bodyB<=24u;
                                if ((toeA && c.bodyB==30u)||(toeB && c.bodyA==30u))
                                    std::fprintf(stderr,"CROW_SUBSTEP_TOE %llu %lu %lu %u %u %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,c.impulses.x,c.pointAndSeparation.w,c.normal.z);'''
b='''                                std::fprintf(stderr,"CROW_SUBSTEP_ALL %llu %lu %lu %u %u %.9g %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,c.impulses.x,c.pointAndSeparation.w,c.normal.x,c.normal.z);'''
assert s.count(a)==1;s=s.replace(a,b);src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/MetalWorld.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
print('diagnostic_library_sha256',hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest())
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I','include','-I','artifacts/crow-v15-wing-inertia-prototype-20261110',str(A/'perch_probe.cpp'),'-L',str(B),'-lmetalrobo','-Wl,-rpath,'+str(B),'-o',str(B/'probe')],cwd=R,check=True)
