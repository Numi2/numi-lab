from pathlib import Path
import difflib,hashlib,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-walk-substep-damping-comparison-20261111';B=R/'build/crow-v15-walk-substep-damping-comparison-20261111';old=(R/'build/crow-v15-grip-transition-substep-audit-20261110/MetalWorld.mm').read_text();s=old.replace('crowTraceStep <= 120','crowTraceStep <= 205')
a=s.index('                    for (NSUInteger sub=0; sub<crowTraceBuffers.count/2; ++sub) {');b=s.index('\n                }];',a)
replacement='''                    for (NSUInteger sub=0; sub<crowTraceBuffers.count/2; ++sub) {
                        id<MTLBuffer> cb=crowTraceBuffers[2*sub], sb=crowTraceBuffers[2*sub+1];
                        const auto* contacts=static_cast<const MRContactConstraintGPU*>(cb.contents);
                        const auto* statuses=static_cast<const MRMetalWorldContactStatusGPU*>(sb.contents);
                        for (NSUInteger lane=0;lane<crowTraceLanes;++lane) {
                            const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);
                            float up[2]={},normal[2]={},toe[2]={},other=0;
                            for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];
                                if (!((c.bodyA<25u&&c.bodyB==25u)||(c.bodyB<25u&&c.bodyA==25u))||c.impulses.x<=0)continue;
                                const float sign=c.bodyB==25u?-1.f:1.f;const unsigned body=c.bodyB==25u?c.bodyA:c.bodyB;
                                const int side=(body==10u||(body>=17u&&body<=20u))?0:((body==13u||(body>=21u&&body<=24u))?1:-1);
                                if(side<0){other+=c.impulses.x;continue;}const float vz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;
                                up[side]+=sign*(c.normal.z*c.impulses.x+c.tangent.z*c.impulses.y+vz*c.impulses.z);normal[side]+=c.impulses.x;
                                if(body>=17u)toe[side]+=c.impulses.x;
                            }
                            std::fprintf(stderr,"CROW_WALK_SUBSTEP %llu %lu %lu %u %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,statuses[lane].requiredConstraints,statuses[lane].physicsSubstep,up[0],up[1],normal[0],normal[1],toe[0],toe[1],other);
                        }
                    }'''
s=s[:a]+replacement+s[b:];src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/MetalWorld.mm')))
base=(R/'build/crow-v15-toe-support-20261109/c_api.cpp').read_text();needle='b.dampingAndSpeedLimits={.015f,.015f,30,35};';assert base.count(needle)==1
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
def compile(src,original,obj,extra):
 line=next(x for x in lines if ' -c '+str(R/original) in x);args=shlex.split(line)
 for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
 args[-1]=str(src)
 for d in extra:args.insert(1,'-I'+str(R/d))
 subprocess.run(args,cwd=R/'build',check=True)
compile(src,'src/metal/MetalWorld.mm',B/'MetalWorld.mm.o',['src/metal','artifacts/crow-clearance-observer-20261109'])
for name,code in [('old',base),('zero',base.replace(needle,'b.dampingAndSpeedLimits={0.f,0.f,30,35};'))]:
 src=B/(name+'.cpp');src.write_text(code);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),code.splitlines(True),fromfile='a/old.cpp',tofile='b/'+name+'.cpp')))
 obj=B/(name+'.o');compile(src,'src/c_api.cpp',obj,['artifacts/crow-clearance-observer-20261109'])
 line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/name/'libmetalrobo.dylib';lib.parent.mkdir(exist_ok=True);args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(B/'MetalWorld.mm.o');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
