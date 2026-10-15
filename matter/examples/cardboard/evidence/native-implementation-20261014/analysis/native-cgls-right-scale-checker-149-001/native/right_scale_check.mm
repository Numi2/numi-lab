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

#ifndef NUMI_CGLS_RIGHT_SCALE_METALLIB
#error "Build must define NUMI_CGLS_RIGHT_SCALE_METALLIB"
#endif

namespace {
struct alignas(16) F4 { float x, y, z, w; };
using U = std::uint32_t;
using U4 = std::array<U,4>;
constexpr U kRunning=1u, kExhausted=2u, kZeroNormal=3u;
constexpr U kInvalidRightScale=64u;
constexpr U E=2u, N=3u, UCount=2u*E*N, MechanicalRows=E*N;
constexpr U FreeCount=2u, DofCount=3u*FreeCount, Cap=32u;
unsigned checks=0;
void require(bool ok,const std::string& what){++checks;if(!ok)throw std::runtime_error(what);}

id<MTLBuffer> buffer(id<MTLDevice> d,const void* p,std::size_t n,const char* label){
    require(p!=nullptr&&n>0,std::string("invalid empty buffer: ")+label);
    id<MTLBuffer> b=[d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
    require(b!=nil,std::string("Metal allocation failed: ")+label);return b;
}
template<class T> id<MTLBuffer> buffer(id<MTLDevice>d,const std::vector<T>&v,const char*l){return buffer(d,v.data(),v.size()*sizeof(T),l);}
id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>lib,const char* suffix){
    NSString* name=[NSString stringWithUTF8String:(std::string("numi_matter_metal::")+suffix).c_str()];
    id<MTLFunction> f=[lib newFunctionWithName:name];require(f!=nil,std::string("missing kernel: ")+suffix);
    NSError* err=nil;id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&err];
    const char* msg=(err&&err.localizedDescription)?err.localizedDescription.UTF8String:"no diagnostic";
    require(p!=nil,std::string("pipeline error ")+suffix+": "+msg);return p;
}
void run(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,NSUInteger groups,NSUInteger threads,
         const std::function<void(id<MTLComputeCommandEncoder>)>& bind){
    id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> e=cmd?[cmd computeCommandEncoder]:nil;
    require(cmd!=nil&&e!=nil,"cannot create command encoder");[e setComputePipelineState:p];bind(e);
    [e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[e endEncoding];[cmd commit];[cmd waitUntilCompleted];
    const char* msg=(cmd.error&&cmd.error.localizedDescription)?cmd.error.localizedDescription.UTF8String:"no diagnostic";
    require(cmd.status==MTLCommandBufferStatusCompleted,std::string("Metal execution failed: ")+msg);
    (void)d;
}
void bind(id<MTLComputeCommandEncoder>e,id<MTLBuffer>b,NSUInteger i){[e setBuffer:b offset:0 atIndex:i];}
void bytes(id<MTLComputeCommandEncoder>e,const void*p,NSUInteger n,NSUInteger i){[e setBytes:p length:n atIndex:i];}
void uintArg(id<MTLComputeCommandEncoder>e,U v,NSUInteger i){bytes(e,&v,sizeof(v),i);}

struct CSR { U n=0;std::vector<U>offsets,columns;std::vector<float>values; };
CSR makeCSR(const std::vector<float>&a,U n,bool transpose){CSR c;c.n=n;c.offsets.push_back(0);for(U r=0;r<n;++r){for(U j=0;j<n;++j){float v=transpose?a[std::size_t(j)*n+r]:a[std::size_t(r)*n+j];if(v!=0){c.columns.push_back(j);c.values.push_back(v);}}c.offsets.push_back(static_cast<U>(c.columns.size()));}return c;}
float dotMask(const std::vector<F4>&a,const std::vector<F4>&b,const std::vector<F4>&mask,U count){float s=0;for(U i=0;i<count;++i)for(U k=0;k<3;++k)s=std::fma((&a[i].x)[k]*(&mask[i].x)[k],(&b[i].x)[k]*(&mask[i].x)[k],s);return s;}

std::array<float,9> block(const std::vector<F4>&rows,U node){U base=node*3;return {rows[base].x,rows[base].y,rows[base].z,rows[base+1].x,rows[base+1].y,rows[base+1].z,rows[base+2].x,rows[base+2].y,rows[base+2].z};}
std::vector<F4> scaleCPU(const std::vector<F4>&in,const std::vector<F4>&rows,const std::vector<F4>&mask,const std::vector<U>&selected,bool transpose){
    std::vector<F4>out(UCount,F4{0,0,0,0});for(U e=0;e<E;++e)if(selected[e])for(U node=0;node<N;++node){U idx=e*N+node;if(mask[idx].x==0&&mask[idx].y==0&&mask[idx].z==0)continue;auto m=block(rows,e*N+node);float v[3]={in[idx].x,in[idx].y,in[idx].z};float r[3]={};for(U i=0;i<3;++i)for(U j=0;j<3;++j)r[i]+= (transpose?m[j*3+i]:m[i*3+j])*v[j];out[idx]=F4{r[0],r[1],r[2],0};}return out;
}
void setBlock(std::vector<F4>&rows,U node,const std::array<float,9>&m){U base=node*3;rows[base]=F4{m[0],m[1],m[2],1};rows[base+1]=F4{m[3],m[4],m[5],0};rows[base+2]=F4{m[6],m[7],m[8],0};}

void validateScale(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,id<MTLBuffer> dispatch,id<MTLBuffer> candidate,id<MTLBuffer> mask,id<MTLBuffer> selected,id<MTLBuffer> rows,id<MTLBuffer> errors){
    run(d,q,p,1,64,[&](id<MTLComputeCommandEncoder>e){bind(e,dispatch,0);bind(e,candidate,1);bind(e,mask,2);bind(e,selected,3);bind(e,rows,4);bind(e,errors,5);});
}
void applyScale(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,U transpose,id<MTLBuffer>mask,id<MTLBuffer>rowEnv,id<MTLBuffer>selected,id<MTLBuffer>rows,id<MTLBuffer>input,id<MTLBuffer>output,id<MTLBuffer>errors){
    run(d,q,p,(UCount+63)/64,64,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);uintArg(e,transpose,2);bind(e,mask,3);bind(e,rowEnv,4);bind(e,selected,5);bind(e,rows,6);bind(e,input,7);bind(e,output,8);bind(e,errors,9);});
}
void clear(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,id<MTLBuffer>out){run(d,q,p,(UCount+63)/64,64,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);bind(e,out,1);});}
void action(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>p,const CSR&A,id<MTLBuffer>offsets,id<MTLBuffer>cols,id<MTLBuffer>values,id<MTLBuffer>freeNodes,id<MTLBuffer>input,id<MTLBuffer>output,id<MTLBuffer>control){
    (void)A;
    run(d,q,p,1,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,N,0);uintArg(e,FreeCount,1);uintArg(e,DofCount,2);bind(e,offsets,3);bind(e,cols,4);bind(e,values,5);bind(e,freeNodes,6);bind(e,input,7);bind(e,output,8);bind(e,control,9);});
}

std::vector<F4> cpuScaledCGLS(const CSR&A,const std::vector<float>&denseS,const std::vector<U>&freeNodes,const std::vector<F4>&rhs,U cap){
    const U n=DofCount;std::vector<float>B(std::size_t(n)*n,0),BT(std::size_t(n)*n,0);
    // Reconstruct dense A from CSR, then B=A*S and B^T=S^T*A^T.
    std::vector<float> denseA(std::size_t(n)*n,0);for(U r=0;r<n;++r)for(U k=A.offsets[r];k<A.offsets[r+1];++k)denseA[std::size_t(r)*n+A.columns[k]]=A.values[k];
    for(U i=0;i<n;++i)for(U j=0;j<n;++j){double sum=0;for(U k=0;k<n;++k)sum+=double(denseA[std::size_t(i)*n+k])*denseS[std::size_t(k)*n+j];B[std::size_t(i)*n+j]=float(sum);BT[std::size_t(j)*n+i]=B[std::size_t(i)*n+j];}
    std::vector<double> b(n),r(n),s(n),p(n),q(n),y(n,0);for(U fi=0;fi<freeNodes.size();++fi)for(U a=0;a<3;++a)b[3*fi+a]=(&rhs[freeNodes[fi]].x)[a];r=b;
    auto multiply=[&](const std::vector<float>&m,const std::vector<double>&x){std::vector<double>o(n,0);for(U i=0;i<n;++i)for(U j=0;j<n;++j)o[i]+=double(m[std::size_t(i)*n+j])*x[j];return o;};
    s=multiply(BT,r);p=s;auto dot=[](const std::vector<double>&x,const std::vector<double>&z){double v=0;for(std::size_t i=0;i<x.size();++i)v+=x[i]*z[i];return v;};double gamma=dot(s,s);
    for(U it=0;it<cap&&gamma>1e-28;++it){q=multiply(B,p);double den=dot(q,q);if(!(den>0)||!std::isfinite(den))break;double alpha=gamma/den;for(U i=0;i<n;++i){y[i]+=alpha*p[i];r[i]-=alpha*q[i];}s=multiply(BT,r);double next=dot(s,s);if(!(next>1e-28)||!std::isfinite(next))break;double beta=next/gamma;for(U i=0;i<n;++i)p[i]=s[i]+beta*p[i];gamma=next;}
    std::vector<double>x(n,0);for(U i=0;i<n;++i)for(U j=0;j<n;++j)x[i]+=double(denseS[std::size_t(i)*n+j])*y[j];std::vector<F4>out(N,F4{0,0,0,0});for(U fi=0;fi<freeNodes.size();++fi)for(U a=0;a<3;++a)(&out[freeNodes[fi]].x)[a]=float(x[3*fi+a]);return out;
}
}

int main(){@autoreleasepool{try{
    const std::vector<U> freeNodes{0,2};
    const std::vector<float> denseA{
        4.0f,1.0f,0.0f,0.2f,0.0f,0.0f,
        0.1f,3.0f,1.0f,0.0f,0.0f,0.0f,
        0.0f,0.2f,2.5f,1.0f,0.0f,0.0f,
        0.0f,0.0f,0.1f,4.0f,1.0f,0.0f,
        1.0f,0.0f,0.0f,0.1f,3.0f,1.0f,
        0.0f,0.0f,1.0f,0.0f,0.2f,2.0f};
    const CSR A=makeCSR(denseA,DofCount,false), AT=makeCSR(denseA,DofCount,true);
    const std::vector<F4> bFree{{1.0f,-2.0f,0.5f,0.0f},{0,0,0,0},{3.0f,-1.0f,2.5f,0.0f}};
    std::vector<F4> rhs(UCount,F4{0,0,0,0}),mask(UCount,F4{0,0,0,0});
    for(U i=0;i<N;++i)rhs[i]=bFree[i];mask[0]=F4{1,1,1,0};mask[2]=F4{1,1,1,0};
    std::vector<U> rowEnvironment(UCount),selected{1,0};
    for(U i=0;i<UCount;++i){U local=i<MechanicalRows?i:i-MechanicalRows;rowEnvironment[i]=local/N;}
    // Deliberately nonsymmetric, full off-diagonal blocks; constrained and quiet blocks stay zero.
    std::vector<F4> scaleRows(E*N*3,F4{0,0,0,0});
    const std::array<float,9>S0{1.0f,0.25f,-0.125f,0.1f,0.8f,0.15f,0.0f,-0.2f,1.1f};
    const std::array<float,9>S2{0.7f,0.05f,0.2f,-0.1f,1.2f,0.1f,0.08f,0.0f,0.9f};
    setBlock(scaleRows,0,S0);setBlock(scaleRows,2,S2);
    std::vector<float> denseS(DofCount*DofCount,0.0f);for(U bidx=0;bidx<2;++bidx){const auto&blockValues=bidx==0?S0:S2;for(U i=0;i<3;++i)for(U j=0;j<3;++j)denseS[std::size_t(3*bidx+i)*DofCount+3*bidx+j]=blockValues[3*i+j];}

    id<MTLDevice>d=MTLCreateSystemDefaultDevice();require(d!=nil,"Metal device unavailable");id<MTLCommandQueue>q=[d newCommandQueue];require(q!=nil,"Metal queue unavailable");
    NSError*err=nil;NSURL*url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_CGLS_RIGHT_SCALE_METALLIB]];id<MTLLibrary>lib=[d newLibraryWithURL:url error:&err];const char*msg=(err&&err.localizedDescription)?err.localizedDescription.UTF8String:"no diagnostic";require(lib!=nil,std::string("load metallib: ")+msg);
    auto make=[&](const char*suffix){return pipeline(d,lib,suffix);};
    auto validate=make("nm_fgmres_base_cgls_validate_right_scale");auto applyScaleKernel=make("nm_fgmres_base_cgls_apply_right_scale");
    auto initState=make("nm_fgmres_base_cgls_initialize_state");auto initVectors=make("nm_fgmres_base_cgls_initialize_vectors");auto measure=make("nm_fgmres_base_cgls_measure_initial");auto formAlpha=make("nm_fgmres_base_cgls_form_alpha");auto updateXR=make("nm_fgmres_base_cgls_update_xr");auto complete=make("nm_fgmres_base_cgls_complete_iteration");auto formBeta=make("nm_fgmres_base_cgls_form_beta");auto updateP=make("nm_fgmres_base_cgls_update_p");auto finalize=make("nm_fgmres_base_cgls_finalize");
    auto clearKernel=make("nm_base_cgls_check_clear");auto applyA=make("nm_base_cgls_check_apply_A");auto applyAT=make("nm_base_cgls_check_apply_AT");
    auto maskB=buffer(d,mask,"mask");auto envB=buffer(d,rowEnvironment,"row environments");auto selectedB=buffer(d,selected,"selected envs");auto scaleB=buffer(d,scaleRows,"inverse block rows");auto rhsB=buffer(d,rhs,"rhs");auto freeB=buffer(d,freeNodes,"free nodes");
    std::vector<std::uint32_t> errors0(E,0);auto errorsB=buffer(d,errors0,"errors");
    std::vector<U4> controls(E,U4{0,0,0,0});auto controlB=buffer(d,controls,"control");
    auto makeDispatch=[](){NMMatterDispatchGPU x{};x.environmentCount=E;x.femNodeCount=N;return x;};auto dispatchRecord=makeDispatch();auto dispatchB=buffer(d,&dispatchRecord,sizeof(dispatchRecord),"dispatch");
    std::vector<NMFEMNodeStateGPU> candidates(E*N);for(U e=0;e<E;++e)for(U n=0;n<N;++n){auto&v=candidates[e*N+n];v.positionAndMass={0,0,0,((e==0&&(n==0||n==2))?1.0f:0.0f)};v.restAndFixed={0,0,0,(e==0&&n==1)?1.0f:0.0f};}
    // Candidate tag 2 in an inactive environment is deliberately unsupported; inactive data must be ignored.
    candidates[N].restAndFixed.w=2.0f;auto candidateB=buffer(d,candidates,"candidate nodes");
    validateScale(d,q,validate,dispatchB,candidateB,maskB,selectedB,scaleB,errorsB);
    require(static_cast<const U*>(errorsB.contents)[0]==0&&static_cast<const U*>(errorsB.contents)[1]==0,"valid selected blocks or inactive invalid environment rejected");

    std::vector<F4> probe(UCount,F4{0,0,0,0});probe[0]=F4{0.3f,-1.2f,2.4f,0};probe[2]=F4{-0.7f,0.9f,1.6f,0};auto probeB=buffer(d,probe,"scale probe");auto outB=buffer(d,std::vector<F4>(UCount),"scale output");
    applyScale(d,q,applyScaleKernel,0,maskB,envB,selectedB,scaleB,probeB,outB,errorsB);
    std::vector<F4> nativeS(static_cast<const F4*>(outB.contents),static_cast<const F4*>(outB.contents)+UCount);auto expectedS=scaleCPU(probe,scaleRows,mask,selected,false);
    applyScale(d,q,applyScaleKernel,1,maskB,envB,selectedB,scaleB,probeB,outB,errorsB);
    std::vector<F4> nativeST(static_cast<const F4*>(outB.contents),static_cast<const F4*>(outB.contents)+UCount);auto expectedST=scaleCPU(probe,scaleRows,mask,selected,true);
    float maxS=0,maxST=0,difference=0;for(U i=0;i<UCount;++i)for(U k=0;k<4;++k){maxS=std::max(maxS,std::abs((&nativeS[i].x)[k]-(&expectedS[i].x)[k]));maxST=std::max(maxST,std::abs((&nativeST[i].x)[k]-(&expectedST[i].x)[k]));difference=std::max(difference,std::abs((&nativeS[i].x)[k]-(&nativeST[i].x)[k]));}
    require(maxS<2e-6f&&maxST<2e-6f,"native S or S^T differs from independent block multiplication");require(difference>0.1f,"nonsymmetric block fixture did not distinguish S from S^T");
    // The runtime relies on exact in-place S^T and final S; test the alias contract.
    auto aliasB=buffer(d,probe,"alias input");applyScale(d,q,applyScaleKernel,1,maskB,envB,selectedB,scaleB,aliasB,aliasB,errorsB);auto alias=static_cast<const F4*>(aliasB.contents);for(U i=0;i<UCount;++i)for(U k=0;k<4;++k)require(std::abs((&alias[i].x)[k]-(&expectedST[i].x)[k])<2e-6f,"in-place transpose scale mismatch");

    // Inactive environment may contain invalid block bytes, but it remains exactly quiet and unobserved.
    scaleRows[(N+0)*3].x=INFINITY;std::memcpy(scaleB.contents,scaleRows.data(),scaleRows.size()*sizeof(F4));std::fill(static_cast<U*>(errorsB.contents),static_cast<U*>(errorsB.contents)+E,0u);
    validateScale(d,q,validate,dispatchB,candidateB,maskB,selectedB,scaleB,errorsB);require(static_cast<const U*>(errorsB.contents)[0]==0&&static_cast<const U*>(errorsB.contents)[1]==0,"inactive nonfinite scale block was not ignored");
    applyScale(d,q,applyScaleKernel,0,maskB,envB,selectedB,scaleB,probeB,outB,errorsB);require(static_cast<const U*>(errorsB.contents)[1]==0,"quiet inactive rows raised scale error");
    scaleRows[(N+0)*3].x=0.0f;scaleRows[0].x=INFINITY;std::memcpy(scaleB.contents,scaleRows.data(),scaleRows.size()*sizeof(F4));std::fill(static_cast<U*>(errorsB.contents),static_cast<U*>(errorsB.contents)+E,0u);
    validateScale(d,q,validate,dispatchB,candidateB,maskB,selectedB,scaleB,errorsB);require((static_cast<const U*>(errorsB.contents)[0]&kInvalidRightScale)!=0,"selected nonfinite inverse block was not rejected");
    scaleRows[0].x=S0[0];std::memcpy(scaleB.contents,scaleRows.data(),scaleRows.size()*sizeof(F4));std::fill(static_cast<U*>(errorsB.contents),static_cast<U*>(errorsB.contents)+E,0u);

    auto offsetsAB=buffer(d,A.offsets,"A offsets");auto colsAB=buffer(d,A.columns,"A columns");auto valuesAB=buffer(d,A.values,"A values");auto offsetsATB=buffer(d,AT.offsets,"AT offsets");auto colsATB=buffer(d,AT.columns,"AT columns");auto valuesATB=buffer(d,AT.values,"AT values");
    auto xB=buffer(d,std::vector<F4>(UCount),"CGLS y/x");auto rB=buffer(d,std::vector<F4>(UCount),"CGLS residual");auto sB=buffer(d,std::vector<F4>(UCount),"CGLS normal residual");auto pB=buffer(d,std::vector<F4>(UCount),"CGLS search");auto qB=buffer(d,std::vector<F4>(UCount),"CGLS operator");auto scalePhysicalB=buffer(d,std::vector<F4>(UCount),"physical S p");auto scalarsB=buffer(d,std::vector<F4>(E),"scalars");
    std::fill(static_cast<U*>(errorsB.contents),static_cast<U*>(errorsB.contents)+E,0u);
    run(d,q,initState,1,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,E,0);uintArg(e,Cap,1);bind(e,selectedB,2);bind(e,scalarsB,3);bind(e,controlB,4);bind(e,errorsB,5);});
    clear(d,q,clearKernel,sB);action(d,q,applyAT,AT,offsetsATB,colsATB,valuesATB,freeB,rhsB,sB,controlB);
    run(d,q,initVectors,(UCount+63)/64,64,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,rhsB,5);bind(e,sB,6);bind(e,xB,7);bind(e,rB,8);bind(e,sB,9);bind(e,pB,10);bind(e,qB,11);bind(e,errorsB,12);uintArg(e,1,13);bind(e,scaleB,14);});
    run(d,q,measure,E,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,rhsB,4);bind(e,rB,5);bind(e,sB,6);bind(e,scalarsB,7);bind(e,controlB,8);bind(e,errorsB,9);});
    for(U it=0;it<Cap;++it){
        applyScale(d,q,applyScaleKernel,0,maskB,envB,selectedB,scaleB,pB,scalePhysicalB,errorsB);clear(d,q,clearKernel,qB);action(d,q,applyA,A,offsetsAB,colsAB,valuesAB,freeB,scalePhysicalB,qB,controlB);
        run(d,q,formAlpha,E,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,qB,4);bind(e,scalarsB,5);bind(e,controlB,6);bind(e,errorsB,7);});
        run(d,q,updateXR,(UCount+63)/64,64,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,scalarsB,4);bind(e,controlB,5);bind(e,pB,6);bind(e,qB,7);bind(e,xB,8);bind(e,rB,9);bind(e,errorsB,10);});
        run(d,q,complete,E,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,E,0);bind(e,controlB,1);bind(e,errorsB,2);});
        clear(d,q,clearKernel,sB);action(d,q,applyAT,AT,offsetsATB,colsATB,valuesATB,freeB,rB,sB,controlB);applyScale(d,q,applyScaleKernel,1,maskB,envB,selectedB,scaleB,sB,sB,errorsB);
        run(d,q,formBeta,E,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,sB,4);bind(e,scalarsB,5);bind(e,controlB,6);bind(e,errorsB,7);});
        run(d,q,updateP,(UCount+63)/64,64,[&](id<MTLComputeCommandEncoder>e){uintArg(e,UCount,0);uintArg(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,scalarsB,4);bind(e,controlB,5);bind(e,sB,6);bind(e,pB,7);bind(e,errorsB,8);});
    }
    run(d,q,finalize,1,32,[&](id<MTLComputeCommandEncoder>e){uintArg(e,E,0);bind(e,controlB,1);bind(e,errorsB,2);});
    applyScale(d,q,applyScaleKernel,0,maskB,envB,selectedB,scaleB,xB,xB,errorsB);
    const U4* nativeControl=static_cast<const U4*>(controlB.contents);const U* nativeErrors=static_cast<const U*>(errorsB.contents);const F4* nativeX=static_cast<const F4*>(xB.contents);
    require((nativeControl[0][0]==kExhausted||nativeControl[0][0]==kZeroNormal)&&nativeControl[0][1]>0,"scaled CGLS did not reach a valid terminal state");require(nativeControl[1][0]==0&&nativeControl[1][1]==0,"quiet environment was modified by scaled recurrence");require(nativeErrors[0]==0&&nativeErrors[1]==0,"scaled CGLS raised validation or recurrence errors");
    std::vector<F4> physicalX(nativeX,nativeX+N);const auto cpuX=cpuScaledCGLS(A,denseS,freeNodes,rhs,Cap);float maxAbs=0,maxScale=0;for(U node:freeNodes)for(U k=0;k<3;++k){float v=(&physicalX[node].x)[k],ref=(&cpuX[node].x)[k];require(std::isfinite(v),"nonfinite physical solution");maxAbs=std::max(maxAbs,std::abs(v-ref));maxScale=std::max(maxScale,std::abs(ref));}require(maxAbs<1.5e-2f*std::max(maxScale,1.0e-5f),"native physical CGLS solution differs from independent B=A*S recurrence");
    // Evaluate the actual unscaled residual with a control record that permits the independent A action after CGLS has terminated.
    std::vector<U4> opControl(E,U4{0,0,0,0});opControl[0][0]=kRunning;auto opControlB=buffer(d,opControl,"post-solve operator control");clear(d,q,clearKernel,qB);action(d,q,applyA,A,offsetsAB,colsAB,valuesAB,freeB,xB,qB,opControlB);std::vector<F4> ax(static_cast<const F4*>(qB.contents),static_cast<const F4*>(qB.contents)+N);std::vector<F4> bOnly(rhs.begin(),rhs.begin()+N);float start=std::sqrt(dotMask(bOnly,bOnly,mask,N));for(U i=0;i<N;++i)for(U k=0;k<3;++k)(&bOnly[i].x)[k]-=(&ax[i].x)[k];float end=std::sqrt(dotMask(bOnly,bOnly,mask,N));require(std::isfinite(end)&&end<start,"physical x=S*y did not reduce true unscaled b-Ax residual");
    std::cout<<"source149 right-scale native checker checks="<<checks<<" terminal="<<nativeControl[0][0]<<" iterations="<<nativeControl[0][1]<<" scale_S_abs="<<maxS<<" scale_ST_abs="<<maxST<<" S_vs_ST="<<difference<<" direction_rel_abs="<<maxAbs<<" direction_scale="<<maxScale<<" true_residual_before="<<start<<" true_residual_after="<<end<<" physical_transactions=0 gpu_checker_not_run_here=1\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"CGLS right-scale checker failure: "<<e.what()<<"\n";return 1;}}}
