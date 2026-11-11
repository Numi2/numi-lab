from pathlib import Path
import subprocess,shlex,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-regrip-rear-force-feedback-20261111';B=R/'build/crow-v15-regrip-rear-force-feedback-20261111';old=(R/'build/crow-v15-grip-transition-substep-audit-20261110/MetalWorld.mm').read_text();s=old
needle='#include <mutex>\n';extra='''#include <mutex>
static std::mutex gCrowToeMutex;
static float gCrowToeInterval[432]={};
static unsigned long long gCrowToeSequence=0;
extern "C" unsigned long long crowCopyToeInterval(float* out){std::lock_guard<std::mutex> lock(gCrowToeMutex);for(unsigned i=0;i<432;i++)out[i]=gCrowToeInterval[i];return gCrowToeSequence;}
''';assert s.count(needle)==1;s=s.replace(needle,extra);s=s.replace('crowTraceStep <= 120','crowTraceStep <= 1500')
a=s.index('                    for (NSUInteger sub=0; sub<crowTraceBuffers.count/2; ++sub) {');b=s.index('\n                }];',a)
replacement='''                    float interval[432]={};
                    for (NSUInteger sub=0; sub<crowTraceBuffers.count/2; ++sub) {
                        id<MTLBuffer> cb=crowTraceBuffers[2*sub], sb=crowTraceBuffers[2*sub+1];
                        const auto* contacts=static_cast<const MRContactConstraintGPU*>(cb.contents);
                        const auto* statuses=static_cast<const MRMetalWorldContactStatusGPU*>(sb.contents);
                        if(crowTraceLanes!=16u)continue;
                        for(unsigned lane=0;lane<16;lane++){const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);
                        for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];
                            if ((c.bodyA<25u&&c.bodyB==25u)||(c.bodyB<25u&&c.bodyA==25u)) {
                                const float sign=c.bodyB==25u?-1.f:1.f;const unsigned body=c.bodyB==25u?c.bodyA:c.bodyB;
                                const int side=(body==10u||(body>=17u&&body<=20u))?0:((body==13u||(body>=21u&&body<=24u))?1:-1);
                                if(side<0&&c.impulses.x>0)interval[lane*27+26]+=c.impulses.x;
                                if(side>=0){const float vz=c.normal.x*c.tangent.y-c.normal.y*c.tangent.x;const float jz=sign*(c.normal.z*c.impulses.x+c.tangent.z*c.impulses.y+vz*c.impulses.z);interval[lane*27+8+side]+=std::max(0.f,jz);}
                            }
                            const bool f=c.bodyA>=17u&&c.bodyA<=24u&&c.bodyB==25u;
                            const bool r=c.bodyB>=17u&&c.bodyB<=24u&&c.bodyA==25u;
                            if(!(f||r)||c.impulses.x<=0)continue;
                            const unsigned k=(f?c.bodyA:c.bodyB)-17u;interval[lane*27+10+k]+=c.impulses.x;const float sign=f?-1.f:1.f;const float vx=c.normal.y*c.tangent.z-c.normal.z*c.tangent.y;interval[lane*27+18+k]+=sign*(c.normal.x*c.impulses.x+c.tangent.x*c.impulses.y+vx*c.impulses.z);const float nx=(f?1.f:-1.f)*c.normal.x;
                            if((k%4u==3u&&nx>.5f)||(k%4u!=3u&&nx<-.5f))interval[lane*27+k]+=c.impulses.x;
                        }}
                    }
                    {std::lock_guard<std::mutex> lock(gCrowToeMutex);for(unsigned k=0;k<432;k++)gCrowToeInterval[k]=interval[k];gCrowToeSequence=crowTraceStep;}'''
s=s[:a]+replacement+s[b:];src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/MetalWorld.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
print('diagnostic_library_sha256',hashlib.sha256((B/'libmetalrobo.dylib').read_bytes()).hexdigest())
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I','include','-I','artifacts/crow-v15-wing-inertia-prototype-20261110',str(A/'perch_probe.cpp'),'-L',str(B),'-lmetalrobo','-Wl,-rpath,'+str(B),'-o',str(B/'probe')],cwd=R,check=True)
