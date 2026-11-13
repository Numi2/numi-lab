from pathlib import Path
import difflib,subprocess,shlex
R=Path.cwd();A=R/'artifacts/crow-v15-hiproll-toe-observer-20261113';B=R/'build/crow-v15-hiproll-toe-observer-20261113';P=R/'build/crow-v15-wide-contact-isolation-20261111';base=(R/'src/metal/MetalWorld.mm').read_text();old=(R/'build/crow-v15-perch-substep-observer-20261111/MetalWorld.mm').read_text();diff=list(difflib.unified_diff(base.splitlines(True),old.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/MetalWorld.mm'));headers=diff[:2];hunks=[];h=None
for line in diff[2:]:
 if line.startswith('@@ '):
  if h:hunks.append(h)
  h=[line]
 elif h is not None:h.append(line)
if h:hunks.append(h)
selected=[h for h in hunks if any(h[0].startswith('@@ -'+n+',') for n in ('21422','22032','22348'))];assert len(selected)==3,[h[0] for h in hunks];src=B/'MetalWorld.mm';src.write_text(base);subprocess.run(['patch',str(src)],input=''.join(headers+sum(selected,[])),text=True,check=True)
s=src.read_text();start=s.index('                            float up[2]={},normal[2]={},toe[2]={},other=0;');end=s.index('                        }\n                    }\n                }];',start)
body='''                            float up[2]={}, normal[2]={}, toe[16]={}, signedX[16]={}, other=0;
                            for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];
                                if (!((c.bodyA<35u&&c.bodyB==35u)||(c.bodyB<35u&&c.bodyA==35u)) || c.impulses.x<=0)continue;
                                const float sign=c.bodyB==35u?-1.f:1.f;const unsigned body=c.bodyB==35u?c.bodyA:c.bodyB;
                                const int side=(body==11u||(body>=19u&&body<=26u))?0:((body==15u||(body>=27u&&body<=34u))?1:-1);
                                if(side<0){other+=c.impulses.x;continue;}const float vz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;
                                up[side]+=sign*(c.normal.z*c.impulses.x+c.tangent.z*c.impulses.y+vz*c.impulses.z);normal[side]+=c.impulses.x;
                                if(body>=19u&&body<=34u){toe[body-19u]+=c.impulses.x;signedX[body-19u]+=sign*c.normal.x*c.impulses.x;}
                            }
                            std::fprintf(stderr,"CROW_TWO_TOE_SUBSTEP %llu %lu %lu %u %lu %.9g %.9g %.9g %.9g %.9g",crowTraceStep,(unsigned long)sub,(unsigned long)lane,statuses[lane].requiredConstraints,(unsigned long)crowTraceCapacity,up[0],up[1],normal[0],normal[1],other);
                            for(int k=0;k<16;k++)std::fprintf(stderr," %.9g",toe[k]);for(int k=0;k<16;k++)std::fprintf(stderr," %.9g",signedX[k]);std::fprintf(stderr,"\\n");
''';s=s[:start]+body+s[end:];src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/two_toe_observer.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);args.insert(1,'-I'+str(P/'include'));args.insert(1,'-I'+str(R/'src/metal'));obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');sourceFiles=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,name in enumerate(sourceFiles):
 oldobj='CMakeFiles/metalrobo.dir/'+name+'.o';assert oldobj in args,oldobj;args[args.index(oldobj)]=str(obj if i==4 else P/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True)
