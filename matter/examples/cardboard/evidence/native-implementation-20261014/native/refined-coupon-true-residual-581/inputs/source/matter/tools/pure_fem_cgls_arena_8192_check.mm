#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include "numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#ifndef NUMI_FEM_CGLS_ARENA_METALLIB
#error "Build must define NUMI_FEM_CGLS_ARENA_METALLIB"
#endif
namespace {
using U=std::uint32_t;
struct alignas(16) F4{float x,y,z,w;};
struct alignas(16) U4{U x,y,z,w;};
unsigned checks=0;
void need(bool ok,const std::string& s){++checks;if(!ok)throw std::runtime_error(s);}
id<MTLComputePipelineState> pipe(id<MTLDevice>d,id<MTLLibrary>l,const char*n){
 NSString*name=[NSString stringWithFormat:@"numi_matter_metal::%s",n];id<MTLFunction>f=[l newFunctionWithName:name];
 need(f!=nil,std::string("missing production kernel ")+n);NSError*e=nil;
 id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];need(p!=nil,std::string("pipeline ")+n);return p;
}
template<class T>void bytes(id<MTLComputeCommandEncoder>e,const T&v,NSUInteger slot){[e setBytes:&v length:sizeof(T) atIndex:slot];}
id<MTLBuffer> hostbuf(id<MTLDevice>d,const void*p,std::size_t n,const char*label){
 need(p&&n,std::string("bad input buffer ")+label);id<MTLBuffer>b=[d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
 need(b!=nil,std::string("allocation failed ")+label);return b;
}
template<class T> id<MTLBuffer> hostbuf(id<MTLDevice>d,const T&v,const char*label){return hostbuf(d,&v,sizeof(v),label);}
using Bind=std::function<void(id<MTLComputeCommandEncoder>)>;
void threads(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,U n,const Bind&f){
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];
 need(c&&e,"no compute encoder");[e setComputePipelineState:p];f(e);
 NSUInteger w=std::max<NSUInteger>(1,std::min<NSUInteger>(256,p.maxTotalThreadsPerThreadgroup));
 [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(w,1,1)];
 [e endEncoding];[c commit];[c waitUntilCompleted];need(c.status==MTLCommandBufferStatusCompleted,"Metal command failed");
}
void groups(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,U n,const Bind&f){
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];
 need(c&&e,"no group encoder");[e setComputePipelineState:p];f(e);
 [e dispatchThreadgroups:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 [e endEncoding];[c commit];[c waitUntilCompleted];need(c.status==MTLCommandBufferStatusCompleted,"Metal group command failed");
}
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
 using namespace numi::matter::detail;
 need(argc==1||(argc==2&&(std::string(argv[1])=="--refined-coupon"||std::string(argv[1])=="--maximum-coupon-arena")),"usage: checker [--refined-coupon|--maximum-coupon-arena]");
 const bool refined=argc==2&&std::string(argv[1])=="--refined-coupon";
 const bool maximum=argc==2&&std::string(argv[1])=="--maximum-coupon-arena";
 const U E=2,N=refined?20682u:(maximum?32768u:8192u),T=refined?89088u:(maximum?131072u:32768u),UCount=2*E*N;
 FEMPureArmijoArenaShape shape{};shape.environmentCount=E;shape.femNodeCount=N;shape.tetrahedronCount=T;
 shape.unknownCount=UCount;shape.maxBufferLength=512ull*1024ull*1024ull;shape.oneStaticExplicitFEM=true;
 shape.includeCauchy=true;shape.includeCGLS=true;shape.includeRightScale=true;
 auto plan=planFEMPureArmijoCGLSArena(shape);
 need(plan.admitted()&&plan.bufferCount==25,"N8192 two-env right-scaled CGLS plan rejected");
 need(plan.scratchBytes<kFEMPureArmijoMaximumScratchBytes,"N8192 arena cap");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();need(d!=nil,"no Metal device");id<MTLCommandQueue>q=[d newCommandQueue];need(q!=nil,"no queue");
 NSError*err=nil;id<MTLLibrary>lib=[d newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_FEM_CGLS_ARENA_METALLIB]] error:&err];
 need(lib!=nil,std::string("load metallib: ")+(err.localizedDescription.UTF8String?err.localizedDescription.UTF8String:"unknown"));
 std::unordered_map<std::string,id<MTLBuffer>> a;
 for(std::size_t i=0;i<plan.bufferCount;++i){const auto&b=plan.buffers[i];id<MTLBuffer>m=[d newBufferWithLength:static_cast<NSUInteger>(b.allocatedBytes) options:MTLResourceStorageModeShared];need(m!=nil,std::string("planned allocation ")+b.name);std::memset(m.contents,0,b.allocatedBytes);a.emplace(b.name,m);}
 auto get=[&](const char*n){auto i=a.find(n);need(i!=a.end(),std::string("missing arena buffer ")+n);return i->second;};
 NMMatterDispatchGPU dispatch{};dispatch.environmentCount=E;dispatch.objectCount=1;dispatch.femNodeCount=N;
 NMFGMRESLayoutGPU layout{};layout.unknownCount=UCount;
 auto dispatchB=hostbuf(d,dispatch,"dispatch");auto layoutB=hostbuf(d,layout,"layout");(void)layoutB;
 auto nodes=get("armijo_candidate_nodes"),mask=get("cgls_mask"),rowEnv=get("cgls_environment_map");
 auto selected=get("cgls_selected"),rhs=get("cauchy_rhs"),b=get("cgls_b"),x=get("cgls_x"),r=get("cgls_r");
 auto s=get("cgls_s"),p=get("cgls_p"),qv=get("cgls_q"),scalars=get("cgls_scalars"),control=get("cgls_control");
 auto errors=get("cgls_errors"),scaleRows=get("cgls_right_scale_rows"),physical=get("cgls_right_scale_physical");
 auto*sel=static_cast<U*>(selected.contents);sel[0]=1;sel[1]=0;
 auto*masks=static_cast<F4*>(mask.contents);auto*env=static_cast<U*>(rowEnv.contents);
 auto*node=static_cast<NMFEMNodeStateGPU*>(nodes.contents);auto*rhsData=static_cast<F4*>(rhs.contents);
 auto*bData=static_cast<F4*>(b.contents);auto*atb=static_cast<F4*>(s.contents);auto*sr=static_cast<F4*>(scaleRows.contents);
 for(U e=0;e<E;++e)for(U n=0;n<N;++n){
  U mi=e*N+n,fi=E*N+e*N+n;env[mi]=e;env[fi]=e;
  if(e==0){masks[mi]={1,1,1,0};node[mi].positionAndMass={0,0,0,1};node[mi].restAndFixed={0,0,0,0};
   U k=mi*3;sr[k]={1,0,0,1};sr[k+1]={0,1,0,0};sr[k+2]={0,0,1,0};
   rhsData[mi]={float(n%13+1)*.01f,float(n%17+1)*-.02f,float(n%19+1)*.015f,0};bData[mi]=rhsData[mi];atb[mi]=rhsData[mi];
  }
 }
 auto state=pipe(d,lib,"nm_fgmres_base_cgls_initialize_state"),validate=pipe(d,lib,"nm_fgmres_base_cgls_validate_right_scale");
 auto init=pipe(d,lib,"nm_fgmres_base_cgls_initialize_vectors"),measure=pipe(d,lib,"nm_fgmres_base_cgls_measure_initial");
 auto applyS=pipe(d,lib,"nm_fgmres_base_cgls_apply_right_scale"),alpha=pipe(d,lib,"nm_fgmres_base_cgls_form_alpha");
 auto updateXR=pipe(d,lib,"nm_fgmres_base_cgls_update_xr"),complete=pipe(d,lib,"nm_fgmres_base_cgls_complete_iteration");
 auto beta=pipe(d,lib,"nm_fgmres_base_cgls_form_beta"),finalize=pipe(d,lib,"nm_fgmres_base_cgls_finalize");
 auto identity=pipe(d,lib,"nm_cgls_arena_8192_identity_action");
 const U cap=64;
 threads(q,state,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,E,0);bytes(e,cap,1);[e setBuffer:selected offset:0 atIndex:2];[e setBuffer:scalars offset:0 atIndex:3];[e setBuffer:control offset:0 atIndex:4];[e setBuffer:errors offset:0 atIndex:5];});
 threads(q,validate,E*N,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:dispatchB offset:0 atIndex:0];[e setBuffer:nodes offset:0 atIndex:1];[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:selected offset:0 atIndex:3];[e setBuffer:scaleRows offset:0 atIndex:4];[e setBuffer:errors offset:0 atIndex:5];});
 const U useScale=1;
 threads(q,init,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:rowEnv offset:0 atIndex:3];[e setBuffer:selected offset:0 atIndex:4];[e setBuffer:rhs offset:0 atIndex:5];[e setBuffer:s offset:0 atIndex:6];[e setBuffer:x offset:0 atIndex:7];[e setBuffer:r offset:0 atIndex:8];[e setBuffer:s offset:0 atIndex:9];[e setBuffer:p offset:0 atIndex:10];[e setBuffer:qv offset:0 atIndex:11];[e setBuffer:errors offset:0 atIndex:12];bytes(e,useScale,13);[e setBuffer:scaleRows offset:0 atIndex:14];});
 groups(q,measure,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:rowEnv offset:0 atIndex:3];[e setBuffer:rhs offset:0 atIndex:4];[e setBuffer:r offset:0 atIndex:5];[e setBuffer:s offset:0 atIndex:6];[e setBuffer:scalars offset:0 atIndex:7];[e setBuffer:control offset:0 atIndex:8];[e setBuffer:errors offset:0 atIndex:9];});
 threads(q,identity,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);[e setBuffer:p offset:0 atIndex:1];[e setBuffer:qv offset:0 atIndex:2];});
 const auto*pv=static_cast<const F4*>(p.contents);const auto*qdata=static_cast<const F4*>(qv.contents);
 for(U i=0;i<UCount;++i)need(std::memcmp(pv+i,qdata+i,sizeof(F4))==0,"production N8192 CGLS A*p action mismatch");
 const float damping=0;
 groups(q,alpha,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:rowEnv offset:0 atIndex:3];[e setBuffer:qv offset:0 atIndex:4];[e setBuffer:scalars offset:0 atIndex:5];[e setBuffer:control offset:0 atIndex:6];[e setBuffer:errors offset:0 atIndex:7];[e setBuffer:p offset:0 atIndex:8];bytes(e,damping,9);});
 need(std::abs(static_cast<const F4*>(scalars.contents)[0].y-1.0f)<2e-6f,"identity CGLS alpha should be one");
 threads(q,updateXR,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:rowEnv offset:0 atIndex:3];[e setBuffer:scalars offset:0 atIndex:4];[e setBuffer:control offset:0 atIndex:5];[e setBuffer:p offset:0 atIndex:6];[e setBuffer:qv offset:0 atIndex:7];[e setBuffer:x offset:0 atIndex:8];[e setBuffer:r offset:0 atIndex:9];[e setBuffer:errors offset:0 atIndex:10];});
 threads(q,complete,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,E,0);[e setBuffer:control offset:0 atIndex:1];[e setBuffer:errors offset:0 atIndex:2];});
 threads(q,identity,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);[e setBuffer:r offset:0 atIndex:1];[e setBuffer:s offset:0 atIndex:2];});
 const U trans=1;
 threads(q,applyS,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bytes(e,trans,2);[e setBuffer:mask offset:0 atIndex:3];[e setBuffer:rowEnv offset:0 atIndex:4];[e setBuffer:selected offset:0 atIndex:5];[e setBuffer:scaleRows offset:0 atIndex:6];[e setBuffer:s offset:0 atIndex:7];[e setBuffer:s offset:0 atIndex:8];[e setBuffer:errors offset:0 atIndex:9];});
 groups(q,beta,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);[e setBuffer:mask offset:0 atIndex:2];[e setBuffer:rowEnv offset:0 atIndex:3];[e setBuffer:s offset:0 atIndex:4];[e setBuffer:scalars offset:0 atIndex:5];[e setBuffer:control offset:0 atIndex:6];[e setBuffer:errors offset:0 atIndex:7];[e setBuffer:x offset:0 atIndex:8];bytes(e,damping,9);});
 threads(q,finalize,E,[&](id<MTLComputeCommandEncoder>e){bytes(e,E,0);[e setBuffer:control offset:0 atIndex:1];[e setBuffer:errors offset:0 atIndex:2];});
 const U trans0=0;
 threads(q,applyS,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bytes(e,trans0,2);[e setBuffer:mask offset:0 atIndex:3];[e setBuffer:rowEnv offset:0 atIndex:4];[e setBuffer:selected offset:0 atIndex:5];[e setBuffer:scaleRows offset:0 atIndex:6];[e setBuffer:x offset:0 atIndex:7];[e setBuffer:physical offset:0 atIndex:8];[e setBuffer:errors offset:0 atIndex:9];});
 auto*ctrl=static_cast<const U4*>(control.contents);auto*errBits=static_cast<const U*>(errors.contents);
 need(errBits[0]==0&&errBits[1]==0,"CGLS reported an error");
 need(ctrl[0].x==3&&ctrl[0].y==1,"active N8192 CGLS identity action did not solve in one iteration");
 need(ctrl[1].x==0&&ctrl[1].y==0,"quiet environment changed its solver state");
 auto*sol=static_cast<const F4*>(physical.contents);auto*rhsPtr=static_cast<const F4*>(rhs.contents);
 for(U i=0;i<UCount;++i){if(i<E*N)need(std::memcmp(sol+i,rhsPtr+i,sizeof(F4))==0,"identity correction differs from b");
  else need(sol[i].x==0&&sol[i].y==0&&sol[i].z==0,"field/quiet output is nonzero");}
 std::cout<<"pure_fem_cgls_arena_"<<N<<"_native_check=PASS checks="<<checks<<" planned_bytes="<<plan.scratchBytes<<" buffers="<<plan.bufferCount<<" active_environments=1 quiet_environments=1 physical_transactions=0\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"pure_fem_cgls_arena_8192_native_check=FAIL reason="<<e.what()<<'\n';return 1;}}}
