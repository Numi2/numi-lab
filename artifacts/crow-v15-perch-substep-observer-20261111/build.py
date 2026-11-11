from pathlib import Path
import difflib,hashlib,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perch-substep-observer-20261111';B=R/'build/crow-v15-perch-substep-observer-20261111';base=(R/'build/crow-v15-bounded-launch-commands-20261111/MetalWorld.mm').read_text();s=base
old='crowTraceStep >= 1 && crowTraceStep <= 505';new='crowTraceStep >= 1 && crowTraceStep <= 3005';assert s.count(old)==1;s=s.replace(old,new)
old='float up[2]={},normal[2]={},toe[2]={},other=0;';new=old+' float perchUp[2]={},perchNormal[2]={},perchToe[8]={},perchX[8]={},perchOther=0;';assert s.count(old)==1;s=s.replace(old,new)
old='''if (!((c.bodyA<25u&&c.bodyB==25u)||(c.bodyB<25u&&c.bodyA==25u))||c.impulses.x<=0)continue;'''
new='''if (((c.bodyA<25u&&c.bodyB==30u)||(c.bodyB<25u&&c.bodyA==30u))&&c.impulses.x>0) {
                                    const float psign=c.bodyB==30u?-1.f:1.f; const unsigned pb=c.bodyB==30u?c.bodyA:c.bodyB;
                                    const int pside=(pb==10u||(pb>=17u&&pb<=20u))?0:((pb==13u||(pb>=21u&&pb<=24u))?1:-1);
                                    if(pside<0)perchOther+=c.impulses.x; else { const float pvz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;
                                        perchUp[pside]+=psign*(c.normal.z*c.impulses.x+c.tangent.z*c.impulses.y+pvz*c.impulses.z);perchNormal[pside]+=c.impulses.x;
                                        if(pb>=17u&&pb<=24u){perchToe[pb-17u]+=c.impulses.x;perchX[pb-17u]+=psign*c.normal.x*c.impulses.x;}}
                                }
                                '''+old
assert s.count(old)==1;s=s.replace(old,new)
old='''std::fprintf(stderr,"CROW_WALK_SUBSTEP %llu %lu %lu %u %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,statuses[lane].requiredConstraints,statuses[lane].physicsSubstep,up[0],up[1],normal[0],normal[1],toe[0],toe[1],other);'''
new=old+'''
                            std::fprintf(stderr,"CROW_PERCH_SUBSTEP %llu %lu %lu %u %lu %.9g %.9g %.9g %.9g %.9g",crowTraceStep,(unsigned long)sub,(unsigned long)lane,statuses[lane].requiredConstraints,(unsigned long)crowTraceCapacity,perchUp[0],perchUp[1],perchNormal[0],perchNormal[1],perchOther);
                            for(int k=0;k<8;k++)std::fprintf(stderr," %.9g",perchToe[k]);
                            for(int k=0;k<8;k++)std::fprintf(stderr," %.9g",perchX[k]);
                            std::fprintf(stderr,"\\n");'''
assert s.count(old)==1;s=s.replace(old,new);src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/full_perch_observer.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src)
for d in ('src/metal','artifacts/crow-clearance-observer-20261109'):args.insert(1,'-I'+str(R/d))
subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(R/'build/crow-v15-perturbed-leg-handoff-20261111/c_api.o');subprocess.run(args,cwd=R/'build',check=True);print('library_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
