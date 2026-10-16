#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#ifndef NUMI_FEM_CGLS_ARENA_METALLIB
#error "Build must define NUMI_FEM_CGLS_ARENA_METALLIB"
#endif
namespace {
using U=std::uint32_t;
struct alignas(16) F4{float x,y,z,w;};
unsigned checks=0;
void need(bool ok,const char* text){++checks;if(!ok)throw std::runtime_error(text);}
id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,const char*n){
 NSString*name=[NSString stringWithFormat:@"numi_matter_metal::%s",n];
 id<MTLFunction>f=[l newFunctionWithName:name];need(f!=nil,"missing kernel");
 NSError*err=nil;id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&err];
 if(p==nil){const char*detail=err.localizedDescription.UTF8String;throw std::runtime_error(detail?detail:"kernel pipeline failed");}return p;
}
using Bind=std::function<void(id<MTLComputeCommandEncoder>)>;
void launch(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,U n,const Bind&bind){
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];need(c&&e,"encoder");
 [e setComputePipelineState:p];bind(e);
 NSUInteger w=std::max<NSUInteger>(1,std::min<NSUInteger>(256,p.maxTotalThreadsPerThreadgroup));
 [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(w,1,1)];
 [e endEncoding];[c commit];[c waitUntilCompleted];need(c.status==MTLCommandBufferStatusCompleted,"dispatch failed");
}
template<class T>void arg(id<MTLComputeCommandEncoder>e,const T&v,NSUInteger i){[e setBytes:&v length:sizeof(T) atIndex:i];}
template<class T>id<MTLBuffer> buf(id<MTLDevice>d,const T*p,size_t n){
 return [d newBufferWithBytes:p length:n*sizeof(T) options:MTLResourceStorageModeShared];
}
template<class T>id<MTLBuffer> zero(id<MTLDevice>d,size_t n){
 id<MTLBuffer>b=[d newBufferWithLength:n*sizeof(T) options:MTLResourceStorageModeShared];
 need(b!=nil,"buffer allocation");std::memset(b.contents,0,n*sizeof(T));return b;
}
struct Input{
 const char*name;
 float cglsResidual;
 float cglsAlpha;
 bool accepted,healthy,eligible,exhausted;
 bool environmentAlphaMismatch,newtonAlphaMismatch;
 float outcome;
 bool keepCgls;
};
void run(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLLibrary>lib,const Input&t){
 constexpr U E=2,N=2,UCount=8;
 NMMatterDispatchGPU ds{};ds.environmentCount=E;ds.objectCount=1;ds.femNodeCount=N;
 NMMixedSolverGPU sv{};sv.residualTolerances.x=.1f;
 NMFGMRESLayoutGPU ly{};ly.unknownCount=UCount;
 const float newtonAlpha=t.newtonAlphaMismatch?.5f:.5f;
 F4 met[E]={{1.f,.8f,newtonAlpha,t.eligible?1.f:0.f},{.25f,.2f,.125f,0.f}};
 F4 line[E]={{newtonAlpha,.8f,0.f,0.f},{.25f,.6f,0.f,0.f}};
 float ea[E]={t.newtonAlphaMismatch?.25f:newtonAlpha,.25f};
 NMFGMRESStateGPU st[E]{};st[0].diagnostics.z=1.f;st[1].diagnostics.z=.25f;
 NMMatterStatusGPU status[E]{},trial[E]{};
 F4 old[UCount],sol[UCount],oldSol[UCount],newSol[UCount]{};
 for(U i=0;i<UCount;++i)old[i]=sol[i]=oldSol[i]={float(i+1),float(i+10),float(i+20),float(i+30)};
 if(t.exhausted)met[0].w=3.f;
 const F4 oldMet[E]={met[0],met[1]},oldLine[E]={line[0],line[1]};
 const float oldEA[E]={ea[0],ea[1]};
 const NMFGMRESStateGPU oldSt[E]={st[0],st[1]};
 const NMMatterStatusGPU oldStatus[E]={status[0],status[1]},oldTrial[E]={trial[0],trial[1]};
 auto bds=buf(d,&ds,1),bsv=buf(d,&sv,1),bly=buf(d,&ly,1);
 auto bm=buf(d,met,E),bl=buf(d,line,E),bea=buf(d,ea,E),bst=buf(d,st,E);
 auto bs=buf(d,status,E),bt=buf(d,trial,E),bm0=zero<F4>(d,E),bl0=zero<F4>(d,E);
 auto be0=zero<float>(d,E),bc0=zero<float>(d,E),bs0=zero<NMMatterStatusGPU>(d,E);
 auto bt0=zero<NMMatterStatusGPU>(d,E),bf=zero<U>(d,E),bsol=buf(d,sol,UCount),bnew=buf(d,newSol,UCount);
 auto ps=pipeline(d,lib,"nm_fgmres_base_cgls_best_of_save");
 auto pd=pipeline(d,lib,"nm_fgmres_base_cgls_best_of_save_direction");
 auto pr=pipeline(d,lib,"nm_fgmres_base_cgls_best_of_resolve");
 auto px=pipeline(d,lib,"nm_fgmres_base_cgls_best_of_restore_direction");
 launch(q,ps,E,[&](id<MTLComputeCommandEncoder>e){
 [e setBuffer:bds offset:0 atIndex:0];[e setBuffer:bsv offset:0 atIndex:1];[e setBuffer:bm offset:0 atIndex:2];[e setBuffer:bl offset:0 atIndex:3];[e setBuffer:bea offset:0 atIndex:4];[e setBuffer:bst offset:0 atIndex:5];[e setBuffer:bs offset:0 atIndex:6];[e setBuffer:bt offset:0 atIndex:7];[e setBuffer:bm0 offset:0 atIndex:8];[e setBuffer:bl0 offset:0 atIndex:9];[e setBuffer:be0 offset:0 atIndex:10];[e setBuffer:bc0 offset:0 atIndex:11];[e setBuffer:bs0 offset:0 atIndex:12];[e setBuffer:bt0 offset:0 atIndex:13];[e setBuffer:bf offset:0 atIndex:14];});
 const U flag=t.exhausted?2u:(t.eligible&&!t.newtonAlphaMismatch?1u:0u);
 need(static_cast<U*>(bf.contents)[0]==flag,"eligibility flag");need(static_cast<U*>(bf.contents)[1]==0,"unselected flag");
 launch(q,pd,UCount,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:bds offset:0 atIndex:0];[e setBuffer:bly offset:0 atIndex:30];[e setBuffer:bsol offset:0 atIndex:1];[e setBuffer:bnew offset:0 atIndex:2];[e setBuffer:bf offset:0 atIndex:3];});
 auto*m=static_cast<F4*>(bm.contents);auto*l=static_cast<F4*>(bl.contents);auto*a=static_cast<float*>(bea.contents);auto*s=static_cast<NMFGMRESStateGPU*>(bst.contents);auto*stp=static_cast<NMMatterStatusGPU*>(bs.contents);auto*tp=static_cast<NMMatterStatusGPU*>(bt.contents);auto*x=static_cast<F4*>(bsol.contents);
 m[0]={1.f,t.cglsResidual,t.cglsAlpha,t.accepted?1.f:0.f};
 l[0]={t.cglsAlpha,.2f,0.f,0.f};
 a[0]=t.environmentAlphaMismatch?std::nextafter(t.cglsAlpha,0.f):t.cglsAlpha;
 if(t.exhausted)m[0].w=1.f;
 s[0].diagnostics.z=0.f;
 stp[0].code=t.healthy?NM_STATUS_SUCCESS:1u;tp[0].code=t.accepted?NM_STATUS_SUCCESS:77u;
 for(U i=0;i<UCount;++i)x[i]={-float(i+1),-float(i+10),-float(i+20),-float(i+30)};
 // Adversarially mutate every unknown row, including both environment blocks.
 // The losing/unselected restoration must recover the full saved vector.
 m[1]={91.f,92.f,93.f,94.f};l[1]={81.f,82.f,83.f,84.f};a[1]=71.f;
 s[1].diagnostics.z=61.f;stp[1].code=51u;tp[1].code=41u;
 F4 cgls[UCount];std::memcpy(cgls,x,sizeof(cgls));
 launch(q,pr,E,[&](id<MTLComputeCommandEncoder>e){
 [e setBuffer:bds offset:0 atIndex:0];[e setBuffer:bm offset:0 atIndex:1];[e setBuffer:bl offset:0 atIndex:2];[e setBuffer:bea offset:0 atIndex:3];[e setBuffer:bst offset:0 atIndex:4];[e setBuffer:bs offset:0 atIndex:5];[e setBuffer:bt offset:0 atIndex:6];[e setBuffer:bm0 offset:0 atIndex:7];[e setBuffer:bl0 offset:0 atIndex:8];[e setBuffer:be0 offset:0 atIndex:9];[e setBuffer:bc0 offset:0 atIndex:10];[e setBuffer:bs0 offset:0 atIndex:11];[e setBuffer:bt0 offset:0 atIndex:12];[e setBuffer:bf offset:0 atIndex:13];});
 launch(q,px,UCount,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:bds offset:0 atIndex:0];[e setBuffer:bly offset:0 atIndex:30];[e setBuffer:bsol offset:0 atIndex:1];[e setBuffer:bnew offset:0 atIndex:2];[e setBuffer:bm0 offset:0 atIndex:3];[e setBuffer:bf offset:0 atIndex:4];});
 const auto*out=static_cast<const F4*>(bm0.contents);
 const auto*final=static_cast<const F4*>(bsol.contents);
 const auto*fm=static_cast<const F4*>(bm.contents);
 const auto*fl=static_cast<const F4*>(bl.contents);
 const auto*fa=static_cast<const float*>(bea.contents);
 const auto*fs=static_cast<const NMFGMRESStateGPU*>(bst.contents);
 const auto*fst=static_cast<const NMMatterStatusGPU*>(bs.contents);
 const auto*ft=static_cast<const NMMatterStatusGPU*>(bt.contents);
 need(out[0].w==t.outcome,t.name);
 for(U i=0;i<UCount;++i){U rows=E*N,local=i<rows?i:i-rows,env=local/N;bool restore=env==1u||!t.keepCgls;
 const F4&want=restore?oldSol[i]:cgls[i];need(std::memcmp(final+i,&want,sizeof(F4))==0,t.name);}
 // Environment 1 is outside this CGLS transaction. Every saved metadata and
 // status field must remain byte-identical in every outcome.
 need(std::memcmp(fm+1,oldMet+1,sizeof(F4))==0,"unselected metrics changed");
 need(std::memcmp(fl+1,oldLine+1,sizeof(F4))==0,"unselected line search changed");
 need(std::memcmp(fa+1,oldEA+1,sizeof(float))==0,"unselected environment alpha changed");
 need(std::memcmp(fs+1,oldSt+1,sizeof(NMFGMRESStateGPU))==0,"unselected certificate/state changed");
 need(std::memcmp(fst+1,oldStatus+1,sizeof(NMMatterStatusGPU))==0,"unselected status changed");
 need(std::memcmp(ft+1,oldTrial+1,sizeof(NMMatterStatusGPU))==0,"unselected trial status changed");
 if(t.keepCgls&&!t.exhausted){need(fm[0].y==t.cglsResidual,"winner metric");need(fs[0].diagnostics.z==0.f,"CGLS inherited Newton certificate");
   if(t.accepted&&t.healthy&&!t.environmentAlphaMismatch){need(fm[0].z==fl[0].x&&fm[0].z==fa[0],"winner applied alpha mismatch");need(out[0].z==fm[0].z,"winner outcome alpha mismatch");}}
 if(!t.keepCgls&&!t.exhausted){need(std::memcmp(fm,oldMet,sizeof(oldMet))==0,"metrics restore");need(std::memcmp(fl,oldLine,sizeof(oldLine))==0,"line restore");need(std::memcmp(fa,oldEA,sizeof(oldEA))==0,"environment alpha restore");need(std::memcmp(fs,oldSt,sizeof(oldSt))==0,"certificate restore");need(std::memcmp(fst,oldStatus,sizeof(oldStatus))==0,"status restore");need(std::memcmp(ft,oldTrial,sizeof(oldTrial))==0,"trial status restore");}
}
}
int main(){@autoreleasepool{try{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();need(d!=nil,"no Metal device");
 id<MTLCommandQueue>q=[d newCommandQueue];need(q!=nil,"no queue");
 NSError*err=nil;id<MTLLibrary>lib=[d newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_FEM_CGLS_ARENA_METALLIB]] error:&err];
 if(lib==nil){const char*detail=err.localizedDescription.UTF8String;throw std::runtime_error(detail?detail:"metallib load failed");}
 run(d,q,lib,{"win",.7f,.125f,true,true,true,false,false,false,2.f,true});
 run(d,q,lib,{"tie",.8f,.125f,true,true,true,false,false,false,1.f,false});
 run(d,q,lib,{"worse",.9f,.125f,true,true,true,false,false,false,1.f,false});
 run(d,q,lib,{"nonfinite",NAN,.125f,true,true,true,false,false,false,3.f,false});
 run(d,q,lib,{"bad-status",.7f,.125f,true,false,true,false,false,false,3.f,false});
 run(d,q,lib,{"rejected-armijo",.7f,.125f,false,true,true,false,false,false,3.f,false});
 run(d,q,lib,{"alpha-mismatch",.7f,.125f,true,true,true,false,true,false,3.f,false});
 run(d,q,lib,{"no-newton",.7f,.125f,true,true,false,false,false,false,0.f,false});
 run(d,q,lib,{"newton-alpha-mismatch",.7f,.125f,true,true,true,false,false,true,0.f,false});
 run(d,q,lib,{"legacy-fallback",.7f,.125f,true,true,false,true,false,false,4.f,true});
 std::cout<<"pure_fem_cgls_best_of_production_kernel=PASS checks="<<checks<<" physical_transactions=0\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"pure_fem_cgls_best_of_production_kernel=FAIL "<<e.what()<<'\n';return 1;}}}
