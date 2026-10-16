#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_WARM_START_METALLIB
#error "Build must define NUMI_WARM_START_METALLIB"
#endif

namespace {
using U=std::uint32_t;
struct alignas(16) F4 { float x,y,z,w; };
using U4=std::array<U,4>;
float component(const F4& v, U i) { return i==0u?v.x:(i==1u?v.y:(i==2u?v.z:v.w)); }
float& component(F4& v, U i) { return i==0u?v.x:(i==1u?v.y:(i==2u?v.z:v.w)); }
constexpr U E=1u,N=3u,UCount=2u*E*N;
constexpr U kRunning=1u,kExhausted=2u;
unsigned checks=0;
void require(bool x,const std::string&m){++checks;if(!x)throw std::runtime_error(m);}
id<MTLBuffer> makeBuffer(id<MTLDevice>d,const void*p,std::size_t n,const char*name){require(p&&n>0,std::string("bad buffer ")+name);id<MTLBuffer>b=[d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];require(b!=nil,std::string("alloc ")+name);return b;}
template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice>d,const std::vector<T>&v,const char*n){return makeBuffer(d,v.data(),v.size()*sizeof(T),n);}
id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,const char*s){NSString*n=[NSString stringWithUTF8String:(std::string("numi_matter_metal::")+s).c_str()];id<MTLFunction>f=[l newFunctionWithName:n];require(f!=nil,std::string("missing kernel ")+s);NSError*e=nil;id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];require(p!=nil,std::string("pipeline ")+s+":"+((e.localizedDescription.UTF8String ? e.localizedDescription.UTF8String : "unknown")));return p;}
void bind(id<MTLComputeCommandEncoder>e,id<MTLBuffer>b,NSUInteger i){[e setBuffer:b offset:0 atIndex:i];}
template<class T> void bytes(id<MTLComputeCommandEncoder>e,const T&v,NSUInteger i){[e setBytes:&v length:sizeof(v) atIndex:i];}
void run(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,U n,const std::function<void(id<MTLComputeCommandEncoder>)>&f){id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"command encoder");[e setComputePipelineState:p];f(e);[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];require(c.status==MTLCommandBufferStatusCompleted,std::string("command failed: ")+((c.error.localizedDescription.UTF8String ? c.error.localizedDescription.UTF8String : "unknown")));(void)d;}
void solve6(const std::vector<float>&a,const std::vector<F4>&b,std::vector<float>&x){double m[6][7]{};for(U r=0;r<6;++r){for(U c=0;c<6;++c)m[r][c]=a[r*6+c];const U node=(r/3==0?0u:2u),axis=r%3;x[r]=0.0f;m[r][6]=component(b[node],axis);}for(U k=0;k<6;++k){U pivot=k;for(U r=k+1;r<6;++r)if(std::abs(m[r][k])>std::abs(m[pivot][k]))pivot=r;require(std::abs(m[pivot][k])>1e-12,"singular checker A");for(U c=k;c<7;++c)std::swap(m[k][c],m[pivot][c]);for(U r=k+1;r<6;++r){double f=m[r][k]/m[k][k];for(U c=k;c<7;++c)m[r][c]-=f*m[k][c];}}for(int r=5;r>=0;--r){double v=m[r][6];for(U c=U(r)+1;c<6;++c)v-=m[r][c]*x[c];x[r]=float(v/m[r][r]);}}
void install(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,id<MTLBuffer>dispatch,id<MTLBuffer>layout,id<MTLBuffer>metrics,id<MTLBuffer>nodes,id<MTLBuffer>rhs,id<MTLBuffer>candidate,id<MTLBuffer>ax,id<MTLBuffer>control,id<MTLBuffer>errors,id<MTLBuffer>solution,id<MTLBuffer>line,id<MTLBuffer>envLine,id<MTLBuffer>trace,id<MTLBuffer>states,id<MTLBuffer>status,id<MTLBuffer>trial,U iter,id<MTLBuffer>baseline,U requireImprovement){run(d,q,p,E,[&](id<MTLComputeCommandEncoder>e){bind(e,dispatch,0);bind(e,metrics,1);bind(e,nodes,2);bind(e,rhs,3);bind(e,candidate,4);bind(e,ax,5);bind(e,control,6);bind(e,errors,7);bind(e,solution,8);bind(e,line,9);bind(e,envLine,10);bind(e,trace,11);bind(e,states,12);bind(e,status,13);bind(e,trial,14);bytes(e,iter,15);bind(e,baseline,16);bytes(e,requireImprovement,17);bind(e,layout,30);});}
}
int main(){@autoreleasepool{try{
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();require(d!=nil,"no Metal device");id<MTLCommandQueue>q=[d newCommandQueue];require(q!=nil,"no Metal queue");NSError*err=nil;NSString*path=[NSString stringWithUTF8String:NUMI_WARM_START_METALLIB];id<MTLLibrary>lib=[d newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];require(lib!=nil,std::string("load library: ")+((err.localizedDescription.UTF8String ? err.localizedDescription.UTF8String : "unknown")));
    auto pseed=pipeline(d,lib,"nm_fgmres_base_cgls_seed_residual"),padd=pipeline(d,lib,"nm_fgmres_base_cgls_add_initial_solution"),pinstall=pipeline(d,lib,"nm_fgmres_install_base_cgls_direction"),pclear=pipeline(d,lib,"nm_warm_check_clear"),papply=pipeline(d,lib,"nm_warm_check_apply_A");
    const std::vector<float>A{4,1,0,.2f,0,0,.1f,3,1,0,0,0,0,.2f,2.5f,1,0,0,0,0,.1f,4,1,0,1,0,0,.1f,3,1,0,0,1,0,.2f,2};
    const std::vector<U>freeNodes{0,2};std::vector<U>cols;std::vector<float>vals;std::vector<U>rowOff{0};for(U r=0;r<6;++r){for(U c=0;c<6;++c)if(A[r*6+c]!=0){cols.push_back(c);vals.push_back(A[r*6+c]);}rowOff.push_back(U(cols.size()));}
    std::vector<F4>rhs(UCount,F4{0,0,0,0}),mask(UCount,F4{0,0,0,0}),x0(UCount,F4{0,0,0,0});rhs[0]=F4{1,-2,.5f,0};rhs[2]=F4{3,-1,2.5f,0};mask[0]=mask[2]=F4{1,1,1,0};x0[0]=F4{.2f,-.1f,.05f,0};x0[2]=F4{-.1f,.15f,.04f,0};
    std::vector<U>rowEnv(UCount,0),selected{1};std::vector<NMFEMNodeStateGPU>nodes(E*N);for(U n=0;n<N;++n){nodes[n].positionAndMass={0,0,0,(n==0||n==2)?1.0f:0.0f};nodes[n].restAndFixed={0,0,0,n==1?1.0f:0.0f};}
    auto dispatchRecord=NMMatterDispatchGPU{};dispatchRecord.environmentCount=E;dispatchRecord.objectCount=1;dispatchRecord.femNodeCount=N;auto layoutRecord=NMFGMRESLayoutGPU{};layoutRecord.unknownCount=UCount;
    auto dispatchB=makeBuffer(d,&dispatchRecord,sizeof(dispatchRecord),"dispatch"),layoutB=makeBuffer(d,&layoutRecord,sizeof(layoutRecord),"layout"),nodesB=makeBuffer(d,nodes,"nodes"),rhsB=makeBuffer(d,rhs,"rhs"),maskB=makeBuffer(d,mask,"mask"),envB=makeBuffer(d,rowEnv,"row env"),selectedB=makeBuffer(d,selected,"selected"),offB=makeBuffer(d,rowOff,"CSR offsets"),colB=makeBuffer(d,cols,"CSR columns"),valB=makeBuffer(d,vals,"CSR values"),freeB=makeBuffer(d,freeNodes,"free nodes");
    std::vector<U4>runControl{{kRunning,0,0,0}};auto controlB=makeBuffer(d,runControl,"operator control");auto x0B=makeBuffer(d,x0,"x0");auto ax0B=makeBuffer(d,std::vector<F4>(UCount),"Ax0");
    run(d,q,pclear,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bind(e,ax0B,1);});
    run(d,q,papply,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,N,0);bytes(e,U(2),1);bytes(e,U(6),2);bind(e,offB,3);bind(e,colB,4);bind(e,valB,5);bind(e,freeB,6);bind(e,x0B,7);bind(e,ax0B,8);bind(e,controlB,9);});
    auto residualB=makeBuffer(d,std::vector<F4>(UCount),"seeded residual");std::vector<U>errors0{0};auto errorsB=makeBuffer(d,errors0,"errors");
    run(d,q,pseed,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,rhsB,5);bind(e,x0B,6);bind(e,ax0B,7);bind(e,residualB,8);bind(e,errorsB,9);});
    const auto*seed=static_cast<const F4*>(residualB.contents);require(static_cast<const U*>(errorsB.contents)[0]==0,"seed kernel raised error");for(U node:{0u,2u})for(U k=0;k<3;++k)require(std::abs(component(seed[node],k)-(component(rhs[node],k)-component(static_cast<const F4*>(ax0B.contents)[node],k)))<2e-6f,"seed residual differs from b-A*x0");for(U i=1;i<UCount;++i)if(i!=2)require(seed[i].x==0&&seed[i].y==0&&seed[i].z==0&&seed[i].w==0,"constrained/field seed row is nonzero");
    std::vector<float>xTrue(6);solve6(A,rhs,xTrue);std::vector<F4>correction(UCount,F4{0,0,0,0});for(U k=0;k<3;++k){component(correction[0],k)=xTrue[k]-component(x0[0],k);component(correction[2],k)=xTrue[k+3]-component(x0[2],k);}auto candidateB=makeBuffer(d,correction,"correction candidate");
    run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,x0B,5);bind(e,candidateB,6);bind(e,errorsB,7);});require(static_cast<const U*>(errorsB.contents)[0]==0,"add x0 kernel raised error");const auto*total=static_cast<const F4*>(candidateB.contents);for(U k=0;k<3;++k){require(std::abs(component(total[0],k)-xTrue[k])<2e-6f,"x0+correction node0");require(std::abs(component(total[2],k)-xTrue[k+3])<2e-6f,"x0+correction node2");}require(total[1].x==0&&total[3].x==0,"fixed/field candidate rows not zero");
    auto axB=makeBuffer(d,std::vector<F4>(UCount),"candidate A action");run(d,q,pclear,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bind(e,axB,1);});run(d,q,papply,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,N,0);bytes(e,U(2),1);bytes(e,U(6),2);bind(e,offB,3);bind(e,colB,4);bind(e,valB,5);bind(e,freeB,6);bind(e,candidateB,7);bind(e,axB,8);bind(e,controlB,9);});
    std::vector<F4>metrics0{{0,0,0,3}},lines0{{.25f,2,3,4}},traces0(1,F4{});std::vector<float>envLines0{.5f};std::vector<NMFGMRESStateGPU>states0(1);states0[0].diagnostics={0,0,1,0};std::vector<NMMatterStatusGPU>statuses0(1),trial0(1);std::vector<U4>cgls0{{kExhausted,64,64,1}};auto metricsB=makeBuffer(d,metrics0,"metrics"),lineB=makeBuffer(d,lines0,"line search"),envLineB=makeBuffer(d,envLines0,"environment line"),traceB=makeBuffer(d,traces0,"trace"),statesB=makeBuffer(d,states0,"FGMRES states"),statusB=makeBuffer(d,statuses0,"status"),trialB=makeBuffer(d,trial0,"trial status"),cglsB=makeBuffer(d,cgls0,"CGLS control"),solutionB=makeBuffer(d,x0,"production solution");
    U iter=48,requireImprovement=1;install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,rhsB,candidateB,axB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,residualB,requireImprovement);const auto*installed=static_cast<const F4*>(solutionB.contents);require(static_cast<const NMMatterStatusGPU*>(statusB.contents)[0].code==0,"strict improved candidate status");require(static_cast<const NMFGMRESStateGPU*>(statesB.contents)[0].diagnostics.z==0,"strict improved candidate did not clear old certificate");require(std::abs(installed[0].x-total[0].x)<1e-6f&&std::abs(installed[2].z-total[2].z)<1e-6f,"strict lower-residual candidate not installed");require(static_cast<const F4*>(traceB.contents)[0].w==2.0f,"warm trace does not mark installed candidate");
    // A refinement may strictly lower the current residual while b dot A*x is
    // negative. The warm gate must use the requested residual comparison alone.
    std::vector<F4> negativeRHS(UCount,F4{0,0,0,0}),desiredX0Action(UCount,F4{0,0,0,0});
    negativeRHS[0].x=1.0f; desiredX0Action[0].x=100.0f;
    std::vector<float> x0NegativeDense(6); solve6(A,desiredX0Action,x0NegativeDense);
    std::vector<F4> x0Negative(UCount,F4{0,0,0,0});
    for(U k=0;k<3;++k){component(x0Negative[0],k)=x0NegativeDense[k];component(x0Negative[2],k)=x0NegativeDense[k+3];}
    auto negativeRHSB=makeBuffer(d,negativeRHS,"negative-descent rhs");
    auto x0NegativeB=makeBuffer(d,x0Negative,"negative-descent x0");
    auto ax0NegativeB=makeBuffer(d,std::vector<F4>(UCount),"negative-descent Ax0");
    run(d,q,pclear,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bind(e,ax0NegativeB,1);});
    run(d,q,papply,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,N,0);bytes(e,U(2),1);bytes(e,U(6),2);bind(e,offB,3);bind(e,colB,4);bind(e,valB,5);bind(e,freeB,6);bind(e,x0NegativeB,7);bind(e,ax0NegativeB,8);bind(e,controlB,9);});
    auto negativeBaselineB=makeBuffer(d,std::vector<F4>(UCount),"negative-descent baseline");
    std::memcpy(errorsB.contents,errors0.data(),sizeof(U));
    run(d,q,pseed,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,negativeRHSB,5);bind(e,x0NegativeB,6);bind(e,ax0NegativeB,7);bind(e,negativeBaselineB,8);bind(e,errorsB,9);});
    std::vector<F4> desiredCandidateAction(UCount,F4{0,0,0,0}); desiredCandidateAction[0].x=-90.0f;
    std::vector<float> candidateNegativeDense(6); solve6(A,desiredCandidateAction,candidateNegativeDense);
    std::vector<F4> candidateNegative(UCount,F4{0,0,0,0});
    for(U k=0;k<3;++k){component(candidateNegative[0],k)=candidateNegativeDense[k];component(candidateNegative[2],k)=candidateNegativeDense[k+3];}
    auto candidateNegativeB=makeBuffer(d,candidateNegative,"negative-descent candidate");
    auto axNegativeB=makeBuffer(d,std::vector<F4>(UCount),"negative-descent Ax");
    run(d,q,pclear,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bind(e,axNegativeB,1);});
    run(d,q,papply,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,N,0);bytes(e,U(2),1);bytes(e,U(6),2);bind(e,offB,3);bind(e,colB,4);bind(e,valB,5);bind(e,freeB,6);bind(e,candidateNegativeB,7);bind(e,axNegativeB,8);bind(e,controlB,9);});
    double baseR2=0.0,candidateR2=0.0,negativeDescent=0.0;
    for(U node:{0u,2u})for(U k=0;k<3;++k){double br=component(negativeRHS[node],k)-component(static_cast<const F4*>(ax0NegativeB.contents)[node],k);double cr=component(negativeRHS[node],k)-component(static_cast<const F4*>(axNegativeB.contents)[node],k);baseR2+=br*br;candidateR2+=cr*cr;negativeDescent+=component(negativeRHS[node],k)*component(static_cast<const F4*>(axNegativeB.contents)[node],k);}
    require(candidateR2<baseR2&&negativeDescent<0.0,"negative-descent fixture does not have strict true-residual reduction");
    std::memcpy(solutionB.contents,x0Negative.data(),UCount*sizeof(F4));std::memcpy(statesB.contents,states0.data(),sizeof(NMFGMRESStateGPU));std::memcpy(statusB.contents,statuses0.data(),sizeof(NMMatterStatusGPU));std::memcpy(metricsB.contents,metrics0.data(),sizeof(F4));std::memcpy(lineB.contents,lines0.data(),sizeof(F4));std::memcpy(envLineB.contents,envLines0.data(),sizeof(float));std::memcpy(trialB.contents,trial0.data(),sizeof(NMMatterStatusGPU));std::memcpy(traceB.contents,traces0.data(),sizeof(F4));std::memcpy(errorsB.contents,errors0.data(),sizeof(U));
    install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,negativeRHSB,candidateNegativeB,axNegativeB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,negativeBaselineB,requireImprovement);
    require(std::memcmp(solutionB.contents,candidateNegative.data(),UCount*sizeof(F4))==0,"strict improving negative-descent candidate was declined");
    require(static_cast<const F4*>(traceB.contents)[0].w==2.0f,"negative-descent improvement not marked installed");
    // A strict fresh-residual reduction can legitimately install the zero
    // vector when the retained x0 is poor. Keep this separate from the legacy
    // alternative-direction validity rule, which still requires xn > 0.
    std::vector<F4> zeroCandidate(UCount,F4{0,0,0,0});
    auto zeroCandidateB=makeBuffer(d,zeroCandidate,"zero total candidate");
    auto axZeroB=makeBuffer(d,std::vector<F4>(UCount),"zero candidate Ax");
    run(d,q,pclear,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bind(e,axZeroB,1);});
    run(d,q,papply,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,N,0);bytes(e,U(2),1);bytes(e,U(6),2);bind(e,offB,3);bind(e,colB,4);bind(e,valB,5);bind(e,freeB,6);bind(e,zeroCandidateB,7);bind(e,axZeroB,8);bind(e,controlB,9);});
    double poorInitialResidual2=0.0,zeroCandidateResidual2=0.0;
    for(U node:{0u,2u})for(U k=0;k<3;++k){const double b=component(negativeRHS[node],k);const double initial=component(static_cast<const F4*>(ax0NegativeB.contents)[node],k);const double candidate=component(static_cast<const F4*>(axZeroB.contents)[node],k);poorInitialResidual2+=(b-initial)*(b-initial);zeroCandidateResidual2+=(b-candidate)*(b-candidate);}
    require(zeroCandidateResidual2<poorInitialResidual2,"zero-total fixture lacks strict fresh-residual reduction");
    std::memcpy(solutionB.contents,x0Negative.data(),UCount*sizeof(F4));std::memcpy(statesB.contents,states0.data(),sizeof(NMFGMRESStateGPU));std::memcpy(statusB.contents,statuses0.data(),sizeof(NMMatterStatusGPU));std::memcpy(metricsB.contents,metrics0.data(),sizeof(F4));std::memcpy(lineB.contents,lines0.data(),sizeof(F4));std::memcpy(envLineB.contents,envLines0.data(),sizeof(float));std::memcpy(trialB.contents,trial0.data(),sizeof(NMMatterStatusGPU));std::memcpy(traceB.contents,traces0.data(),sizeof(F4));std::memcpy(errorsB.contents,errors0.data(),sizeof(U));requireImprovement=1;
    install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,negativeRHSB,zeroCandidateB,axZeroB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,negativeBaselineB,requireImprovement);
    require(static_cast<const U*>(errorsB.contents)[0]==0,"zero-total strict improvement raised CGLS error");require(std::memcmp(solutionB.contents,zeroCandidate.data(),UCount*sizeof(F4))==0,"zero-total strict improvement was not installed");require(static_cast<const F4*>(traceB.contents)[0].w==2.0f,"zero-total strict improvement not marked installed");require(static_cast<const NMFGMRESStateGPU*>(statesB.contents)[0].diagnostics.z==0.0f,"installed zero-total candidate retained original certificate");
    std::memcpy(solutionB.contents,x0Negative.data(),UCount*sizeof(F4));std::memcpy(statesB.contents,states0.data(),sizeof(NMFGMRESStateGPU));std::memcpy(statusB.contents,statuses0.data(),sizeof(NMMatterStatusGPU));std::memcpy(metricsB.contents,metrics0.data(),sizeof(F4));std::memcpy(lineB.contents,lines0.data(),sizeof(F4));std::memcpy(envLineB.contents,envLines0.data(),sizeof(float));std::memcpy(trialB.contents,trial0.data(),sizeof(NMMatterStatusGPU));std::memcpy(traceB.contents,traces0.data(),sizeof(F4));std::memcpy(errorsB.contents,errors0.data(),sizeof(U));requireImprovement=0;
    install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,negativeRHSB,zeroCandidateB,axZeroB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,negativeBaselineB,requireImprovement);
    require(static_cast<const NMMatterStatusGPU*>(statusB.contents)[0].code!=0,"legacy zero-norm direction was not rejected");require(std::memcmp(solutionB.contents,x0Negative.data(),UCount*sizeof(F4))==0,"legacy zero-norm rejection changed x0");require(static_cast<const NMFGMRESStateGPU*>(statesB.contents)[0].diagnostics.z==1.0f,"legacy zero-norm rejection cleared the original certificate");
    // Equal residual must decline the refinement without touching x0/cert/Armijo state.
    std::memcpy(solutionB.contents,x0.data(),UCount*sizeof(F4));std::memcpy(candidateB.contents,x0.data(),UCount*sizeof(F4));std::memcpy(axB.contents,ax0B.contents,UCount*sizeof(F4));std::memcpy(statesB.contents,states0.data(),sizeof(NMFGMRESStateGPU));std::memcpy(statusB.contents,statuses0.data(),sizeof(NMMatterStatusGPU));std::memcpy(metricsB.contents,metrics0.data(),sizeof(F4));std::memcpy(lineB.contents,lines0.data(),sizeof(F4));std::memcpy(envLineB.contents,envLines0.data(),sizeof(float));std::memcpy(trialB.contents,trial0.data(),sizeof(NMMatterStatusGPU));std::memcpy(traceB.contents,traces0.data(),sizeof(F4));std::memcpy(errorsB.contents,errors0.data(),sizeof(U));install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,rhsB,candidateB,axB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,residualB,requireImprovement);require(std::memcmp(solutionB.contents,x0.data(),UCount*sizeof(F4))==0,"equal-residual refinement changed x0");require(static_cast<const NMFGMRESStateGPU*>(statesB.contents)[0].diagnostics.z==1.0f,"declined refinement cleared original certificate");require(static_cast<const F4*>(metricsB.contents)[0].w==0.0f,"declined refinement left CGLS policy marker armed");require(std::memcmp(lineB.contents,lines0.data(),sizeof(F4))==0&&std::memcmp(envLineB.contents,envLines0.data(),sizeof(float))==0,"declined refinement changed Armijo state");require(static_cast<const F4*>(traceB.contents)[0].w==1.0f,"warm trace does not mark decline");
    // Legacy mode remains an alternative direction and does not acquire the new residual gate.
    std::vector<F4>twice(UCount,F4{0,0,0,0}),twiceAx(UCount,F4{0,0,0,0});for(U n:{0u,2u})for(U k=0;k<3;++k){component(twice[n],k)=2.0f*xTrue[(n==0?0u:3u)+k];component(twiceAx[n],k)=2.0f*component(rhs[n],k);}std::memcpy(solutionB.contents,x0.data(),UCount*sizeof(F4));std::memcpy(candidateB.contents,twice.data(),UCount*sizeof(F4));std::memcpy(axB.contents,twiceAx.data(),UCount*sizeof(F4));std::memcpy(statesB.contents,states0.data(),sizeof(NMFGMRESStateGPU));std::memcpy(statusB.contents,statuses0.data(),sizeof(NMMatterStatusGPU));std::memcpy(metricsB.contents,metrics0.data(),sizeof(F4));std::memcpy(errorsB.contents,errors0.data(),sizeof(U));requireImprovement=0;install(d,q,pinstall,dispatchB,layoutB,metricsB,nodesB,rhsB,candidateB,axB,cglsB,errorsB,solutionB,lineB,envLineB,traceB,statesB,statusB,trialB,iter,residualB,requireImprovement);require(std::memcmp(solutionB.contents,twice.data(),UCount*sizeof(F4))==0,"default legacy installer behavior changed");
    std::cout<<"warm-start Metal kernel checker source prepared; checks="<<checks<<"; physical_transactions=0; GPU_NOT_RUN\n";return 0;
}catch(const std::exception&e){std::cerr<<"warm-start checker error: "<<e.what()<<"\n";return 1;}}}
