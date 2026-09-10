#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_equality_gpu.h"
#include "metalrobo/NumiHumanCompliantEquilibrium.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<char> read(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot open fixture");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
std::uint32_t u32(const std::vector<char>& bytes, std::size_t offset) {
    require(offset + 4u <= bytes.size(), "truncated fixture");
    std::uint32_t value; std::memcpy(&value, bytes.data()+offset, 4u); return value;
}
struct Fixture {
    std::uint32_t nq=12u, nv=11u;
    std::vector<NMHumanJointEqualityGPU> rows;
    std::vector<float> q=std::vector<float>(12u, 0.0f);
};
Fixture synthetic() {
    Fixture fixture;
    for (std::uint32_t i=0u; i<4u; ++i) {
        NMHumanJointEqualityGPU row{};
        row.indices={7u+i,6u+i,11u,10u};
        row.referencesAndCoefficients0={0.0f,0.0f,0.01f*float(i),0.2f};
        row.coefficients1={-0.3f,0.1f,-0.01f,0.0f};
        row.solref={0.02f,1.0f,0.0f,0.0f};
        row.solimp0={0.9999f,0.9999f,0.001f,0.5f};
        row.solimp1={2.0f,0.0f,0.0f,0.0f};
        row.sourceInverseWeights={3.25f,4.5f,0.0f,0.0f};
        if (i==0u) { row.indices.z=row.indices.w=NM_INVALID_INDEX; row.sourceInverseWeights.y=0.0f; }
        if (i==1u) row.coefficients1={0.0f,0.0f,0.0f,0.0f};
        if (i==3u) { row.solref={-2500.0f,-100.0f,0.0f,0.0f}; row.solimp0={0.1f,0.9f,0.5f,0.3f}; }
        fixture.rows.push_back(row);
    }
    return fixture;
}
Fixture authored(const char* rigidPath, const char* equalityPath) {
    const auto rigid=read(rigidPath), equality=read(equalityPath);
    require(rigid.size()>=80u && std::memcmp(rigid.data(),"NHRIGID2",8u)==0,"invalid NHRIGID2 fixture");
    require(equality.size()>=80u && std::memcmp(equality.data(),"NHEQ2\0\0\0",8u)==0,"invalid NHEQ2 fixture");
    require(u32(equality,8u)==2u && u32(equality,24u)==112u && u32(equality,32u)==1u &&
            u32(equality,36u)==1u && u32(equality,40u)==0u && u32(equality,44u)==0u,"invalid NHEQ2 header");
    require(std::memcmp(rigid.data()+48u,equality.data()+48u,32u)==0,"source identity mismatch");
    Fixture result;result.nq=u32(rigid,28u);result.nv=u32(rigid,32u);
    const std::uint32_t count=u32(equality,20u);
    require(u32(equality,12u)==result.nq && u32(equality,16u)==result.nv &&
            u32(equality,28u)==count && equality.size()==80u+112u*count,"NHEQ2 dimensions mismatch");
    const std::size_t qOffset=80u+96u+48u+160u*u32(rigid,20u)+144u*u32(rigid,24u)+64u*result.nv;
    require(qOffset+4u*result.nq<=rigid.size(),"NHRIGID2 q is truncated");
    result.q.resize(result.nq);std::memcpy(result.q.data(),rigid.data()+qOffset,4u*result.nq);
    result.rows.resize(count);std::memcpy(result.rows.data(),equality.data()+80u,112u*count);
    for (const auto& row:result.rows)
        require(row.indices.x<result.nq && row.indices.y<result.nv &&
                ((row.indices.z==NM_INVALID_INDEX && row.indices.w==NM_INVALID_INDEX) ||
                 (row.indices.z<result.nq && row.indices.w<result.nv)),"invalid equality indices");
    return result;
}
id<MTLBuffer> buffer(id<MTLDevice> device, std::size_t bytes) {
    id<MTLBuffer> result=[device newBufferWithLength:std::max<std::size_t>(bytes,16u)
                                           options:MTLResourceStorageModeShared];
    require(result!=nil,"Metal allocation failed");std::memset(result.contents,0,result.length);return result;
}
template<typename T>T* data(id<MTLBuffer> b) {return static_cast<T*>(b.contents);}
struct Reference {double derivative,bDelta,inverseR,phi;};
Reference reference(const NMHumanJointEqualityGPU& row,const float* q,const float* v,const float* free,
                    double h,bool refsafe) {
    const bool fixed=row.indices.z==NM_INVALID_INDEX;
    const double x=fixed?0.0:double(q[row.indices.z])-row.referencesAndCoefficients0.y;
    const std::array<double,5> a={row.referencesAndCoefficients0.z,row.referencesAndCoefficients0.w,
        row.coefficients1.x,row.coefficients1.y,row.coefficients1.z};
    double polynomial=0.0,derivative=0.0;
    for (unsigned k=0u;k<5u;++k) polynomial+=a[k]*std::pow(x,double(k));
    if (!fixed) for (unsigned k=1u;k<5u;++k) derivative+=double(k)*a[k]*std::pow(x,double(k-1u));
    const double phi=double(q[row.indices.x])-row.referencesAndCoefficients0.x-polynomial;
    const double d0=std::clamp(double(row.solimp0.x),0.0001,0.9999);
    const double dw=std::clamp(double(row.solimp0.y),0.0001,0.9999);
    const double width=std::max(double(row.solimp0.z),0.0);
    const double midpoint=std::clamp(double(row.solimp0.w),0.0001,0.9999);
    const double power=std::max(double(row.solimp1.x),1.0);
    double impedance=0.5*(d0+dw);
    if (d0!=dw && width>1e-15) {
        const double u=std::clamp(std::abs(phi)/width,0.0,1.0);
        const double y=power==1.0?u:(u<=midpoint?
            std::pow(u,power)/std::pow(midpoint,power-1.0):
            1.0-std::pow(1.0-u,power)/std::pow(1.0-midpoint,power-1.0));
        impedance=d0+(dw-d0)*y;
    }
    const double t=refsafe?std::max(double(row.solref.x),2.0*h):row.solref.x;
    const double stiffness=row.solref.x>0.0f?1.0/std::max(1e-15,dw*dw*t*t*row.solref.y*row.solref.y):
        -row.solref.x/std::max(1e-15,dw*dw);
    const double damping=row.solref.x>0.0f?2.0/std::max(1e-15,dw*t):-row.solref.y/std::max(1e-15,dw);
    const double sourceV=double(v[row.indices.y])-(fixed?0.0:derivative*v[row.indices.w]);
    const double increment=double(free[row.indices.y])-v[row.indices.y]-(fixed?0.0:
        derivative*(double(free[row.indices.w])-v[row.indices.w]));
    const double regularizer=std::max(1e-15,(1.0-impedance)/impedance*
        (double(row.sourceInverseWeights.x)+row.sourceInverseWeights.y));
    return {derivative,h*(-damping*sourceV-stiffness*impedance*phi)-increment,1.0/regularizer,phi};
}
void run(const Fixture& fixture) {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
    require(library!=nil,"Matter metallib unavailable");
    auto pipeline=[&](NSString* name) {
        id<MTLFunction> function=[library newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];require(function!=nil,"missing equality kernel");
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:function error:&error];
        require(p!=nil,"equality pipeline failed");return p;
    };
    const auto prepare=pipeline(@"nm_human_equality_prepare"), residual=pipeline(@"nm_human_equality_residual"),
               action=pipeline(@"nm_human_equality_operator"),
               factor=pipeline(@"nm_human_equality_factor"), precondition=pipeline(@"nm_human_equality_precondition");
    const std::uint32_t envs=2u, count=std::uint32_t(fixture.rows.size());
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=envs;dispatch.rigidGeneralizedCapacity=fixture.nv;
    dispatch.femNodeCount=3u;dispatch.mpmActiveNodeCapacity=5u;
    const std::size_t base=envs*(2u*dispatch.femNodeCount+dispatch.mpmActiveNodeCapacity);
    const std::size_t vectors=base+envs*fixture.nv;
    NMHumanEqualityDispatchGPU eq{};eq.count=count;eq.qCount=fixture.nq;eq.dofCount=fixture.nv;
    eq.policy=NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC;
    const auto rows=buffer(device,count*sizeof(NMHumanJointEqualityGPU));
    std::memcpy(rows.contents,fixture.rows.data(),count*sizeof(NMHumanJointEqualityGPU));
    const auto q=buffer(device,envs*fixture.nq*4u), v=buffer(device,envs*fixture.nv*4u),free=buffer(device,envs*fixture.nv*4u),
               lin=buffer(device,envs*count*16u),statuses=buffer(device,envs*sizeof(NMMatterStatusGPU)),
               delta=buffer(device,envs*fixture.nv*4u),direction=buffer(device,vectors*16u),
               r=buffer(device,vectors*16u),op=buffer(device,vectors*16u),states=buffer(device,envs*sizeof(NMFGMRESStateGPU));
    const auto sourceL=buffer(device,envs*fixture.nv*fixture.nv*4u), factorB=buffer(device,sourceL.length),
               preconditioned=buffer(device,vectors*16u);
    for(unsigned e=0u;e<envs;++e)for(unsigned i=0u;i<fixture.nv;++i)for(unsigned j=0u;j<=i;++j)
        data<float>(sourceL)[e*fixture.nv*fixture.nv+i*fixture.nv+j]=i==j?
            0.8f+0.004f*float(i+e):0.006f*std::sin(float(i+3u*j+e));
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"Metal queue unavailable");
    double linError=0.0,residualError=0.0,actionError=0.0,fdError=0.0,initialDefect=0.0;
    double factorError=0.0,inverseError=0.0,inverseResidual=0.0,staticForceError=0.0;
    for (unsigned trial=0u;trial<4u;++trial) {
        eq.flags=trial==2u?0u:NM_HUMAN_EQUALITY_REFSAFE;eq.time.x=trial==0u?1e-4f:0.02f;
        for (unsigned e=0u;e<envs;++e) {
            for (unsigned j=0u;j<fixture.nq;++j) data<float>(q)[e*fixture.nq+j]=fixture.q[j]+(e==0u?0.0f:0.1f*std::sin(float(j)));
            for (unsigned j=0u;j<fixture.nv;++j) {
                const unsigned k=e*fixture.nv+j;
                data<float>(v)[k]=(e==0u || trial==3u)?0.0f:0.2f*std::sin(float(j+2u));
                data<float>(free)[k]=data<float>(v)[k]+(trial==3u?0.0f:0.015f*std::cos(float(j)));
                data<float>(delta)[k]=trial==3u?0.0f:0.03f*std::cos(float(j+3u));
                data<nm_float4>(direction)[base+k].x=0.2f*std::sin(float(j+1u));
            }
        }
        auto execute=[&](bool expectSuccess=true) {
            std::vector<char> sourceBytes(sourceL.length);std::memcpy(sourceBytes.data(),sourceL.contents,sourceL.length);
            for(std::size_t k=0u;k<vectors;++k)data<nm_float4>(preconditioned)[k]={0.0f,7.0f,8.0f,9.0f};
            std::memset(lin.contents,0,lin.length);std::memset(statuses.contents,0,statuses.length);
            for (std::size_t k=0u;k<vectors;++k) {data<nm_float4>(r)[k]={0.0f,7.0f,8.0f,9.0f};data<nm_float4>(op)[k]={0.0f,7.0f,8.0f,9.0f};}
            id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            auto bind=[&](id<MTLComputePipelineState> p) {[enc setComputePipelineState:p];[enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];[enc setBytes:&eq length:sizeof(eq) atIndex:1u];[enc setBuffer:rows offset:0u atIndex:2u];};
            auto submit=[&](std::size_t n) {[enc dispatchThreads:MTLSizeMake(n,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];};
            bind(prepare);[enc setBuffer:q offset:0u atIndex:3u];[enc setBuffer:v offset:0u atIndex:4u];[enc setBuffer:free offset:0u atIndex:5u];[enc setBuffer:lin offset:0u atIndex:6u];[enc setBuffer:statuses offset:0u atIndex:7u];submit(envs*count);
            bind(residual);[enc setBuffer:lin offset:0u atIndex:3u];[enc setBuffer:delta offset:0u atIndex:4u];[enc setBuffer:r offset:0u atIndex:5u];submit(envs*fixture.nv);
            bind(action);[enc setBuffer:lin offset:0u atIndex:3u];[enc setBuffer:direction offset:0u atIndex:4u];[enc setBuffer:op offset:0u atIndex:5u];[enc setBuffer:states offset:0u atIndex:6u];submit(envs*fixture.nv);
            bind(factor);[enc setBuffer:lin offset:0u atIndex:3u];[enc setBuffer:sourceL offset:0u atIndex:4u];[enc setBuffer:factorB offset:0u atIndex:5u];[enc setBuffer:statuses offset:0u atIndex:6u];[enc dispatchThreadgroups:MTLSizeMake(envs,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
            bind(precondition);[enc setBuffer:factorB offset:0u atIndex:2u];[enc setBuffer:direction offset:0u atIndex:3u];[enc setBuffer:preconditioned offset:0u atIndex:4u];[enc setBuffer:states offset:0u atIndex:5u];[enc setBuffer:statuses offset:0u atIndex:6u];[enc dispatchThreadgroups:MTLSizeMake(envs,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
            [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"equality command failed");
            require(std::memcmp(sourceBytes.data(),sourceL.contents,sourceL.length)==0,"equality preconditioner mutated source L0");
            if(expectSuccess)for (unsigned e=0u;e<envs;++e) require(data<NMMatterStatusGPU>(statuses)[e].code==0u,"equality prepare/factor rejected finite fixture");
            for(std::size_t k=0u;k<vectors;++k){const auto pc=data<nm_float4>(preconditioned)[k];require(pc.y==7.0f&&pc.z==8.0f&&pc.w==9.0f,"preconditioner overwrote unrelated vector channels");if(k<base)require(pc.x==0.0f,"preconditioner overwrote continuum prefix");}
            for (std::size_t k=0u;k<vectors;++k) {const auto rr=data<nm_float4>(r)[k],aa=data<nm_float4>(op)[k];require(rr.y==7.0f&&rr.z==8.0f&&rr.w==9.0f&&aa.y==7.0f&&aa.z==8.0f&&aa.w==9.0f,"equality kernel overwrote unrelated vector channels");if(k<base)require(rr.x==0.0f&&aa.x==0.0f,"equality kernel overwrote continuum prefix");}
        };
        execute();
        std::vector<double> expectedR(envs*fixture.nv,0.0), expectedA(expectedR.size(),0.0);
        for (unsigned e=0u;e<envs;++e) for (unsigned i=0u;i<count;++i) {
            const auto& row=fixture.rows[i];const auto ref=reference(row,data<float>(q)+e*fixture.nq,data<float>(v)+e*fixture.nv,data<float>(free)+e*fixture.nv,eq.time.x,(eq.flags&1u)!=0u);
            const auto actual=data<nm_float4>(lin)[e*count+i];const std::array<double,4> cpu={ref.derivative,ref.bDelta,ref.inverseR,ref.phi};const std::array<float,4> gpu={actual.x,actual.y,actual.z,actual.w};
            for(unsigned c=0u;c<4u;++c)linError=std::max(linError,std::abs(double(gpu[c])-cpu[c])/std::max(std::abs(cpu[c]),1.0));
            if(e==0u&&trial==0u)initialDefect=std::max(initialDefect,std::abs(ref.phi));
            const unsigned dep=e*fixture.nv+row.indices.y;const bool fixed=row.indices.w==NM_INVALID_INDEX;const unsigned master=fixed?dep:e*fixture.nv+row.indices.w;
            const double jdv=double(data<float>(delta)[dep])-(fixed?0.0:ref.derivative*data<float>(delta)[master]);
            const double jdir=double(data<nm_float4>(direction)[base+dep].x)-(fixed?0.0:ref.derivative*data<nm_float4>(direction)[base+master].x);
            if (trial==3u) {
                metalrobo::NumiHumanSourceScalarLaw law;
                std::memcpy(&law.solref,&row.solref,16);std::memcpy(&law.solimp0,&row.solimp0,16);std::memcpy(&law.solimp1,&row.solimp1,16);
                law.inverseWeight=double(row.sourceInverseWeights.x)+row.sourceInverseWeights.y;law.referenceSafe=(eq.flags&1u)!=0;
                double force=0;
                require(metalrobo::evaluateNumiHumanSourceStaticForce(law,ref.phi,eq.time.x,force),"static source force rejected");
                const double gpuForce=double(actual.y)*actual.z/eq.time.x;
                staticForceError=std::max(staticForceError,std::abs(force-gpuForce)/std::max(1.0,std::abs(force)));
            }
            const double lambda=(jdv-ref.bDelta)*ref.inverseR, product=jdir*ref.inverseR;
            expectedR[dep]-=lambda;expectedA[dep]+=product;
            if(!fixed){expectedR[master]+=ref.derivative*lambda;expectedA[master]-=ref.derivative*product;}
        }
        double rscale=1.0,ascale=1.0,re=0.0,ae=0.0;
        for(unsigned k=0u;k<envs*fixture.nv;++k){rscale=std::max(rscale,std::abs(expectedR[k]));ascale=std::max(ascale,std::abs(expectedA[k]));re=std::max(re,std::abs(data<nm_float4>(r)[base+k].x-expectedR[k]));ae=std::max(ae,std::abs(data<nm_float4>(op)[base+k].x-expectedA[k]));}
        residualError=std::max(residualError,re/rscale);actionError=std::max(actionError,ae/ascale);
        // Independent FP64 dense assembly/Cholesky, separate from the device rank-update algorithm.
        for(unsigned e=0u;e<envs;++e) {
            const unsigned n=fixture.nv, mb=e*n*n;
            std::vector<double> B(n*n,0.0), L(n*n,0.0), answer(n,0.0);
            for(unsigned i=0u;i<n;++i)for(unsigned j=0u;j<n;++j)
                for(unsigned k=0u;k<=std::min(i,j);++k)B[i*n+j]+=double(data<float>(sourceL)[mb+i*n+k])*data<float>(sourceL)[mb+j*n+k];
            for(unsigned i=0u;i<count;++i) {
                const auto& row=fixture.rows[i];const auto ref=reference(row,data<float>(q)+e*fixture.nq,data<float>(v)+e*n,data<float>(free)+e*n,eq.time.x,(eq.flags&1u)!=0u);
                const unsigned d=row.indices.y,m=row.indices.w;B[d*n+d]+=ref.inverseR;
                if(m!=NM_INVALID_INDEX){B[d*n+m]-=ref.derivative*ref.inverseR;B[m*n+d]-=ref.derivative*ref.inverseR;B[m*n+m]+=ref.derivative*ref.derivative*ref.inverseR;}
            }
            double fs=1.0,fe=0.0;
            for(unsigned i=0u;i<n;++i)for(unsigned j=0u;j<n;++j){double value=0.0;for(unsigned k=0u;k<=std::min(i,j);++k)value+=double(data<float>(factorB)[mb+i*n+k])*data<float>(factorB)[mb+j*n+k];fe=std::max(fe,std::abs(value-B[i*n+j]));fs=std::max(fs,std::abs(B[i*n+j]));}
            factorError=std::max(factorError,fe/fs);
            for(unsigned i=0u;i<n;++i)for(unsigned j=0u;j<=i;++j){double value=B[i*n+j];for(unsigned k=0u;k<j;++k)value-=L[i*n+k]*L[j*n+k];if(i==j){require(value>0.0,"FP64 SPD fixture lost positive pivot");L[i*n+j]=std::sqrt(value);}else L[i*n+j]=value/L[j*n+j];}
            for(unsigned i=0u;i<n;++i){double value=data<nm_float4>(direction)[base+e*n+i].x;for(unsigned j=0u;j<i;++j)value-=L[i*n+j]*answer[j];answer[i]=value/L[i*n+i];}
            for(unsigned rev=0u;rev<n;++rev){const unsigned i=n-1u-rev;double value=answer[i];for(unsigned j=i+1u;j<n;++j)value-=L[j*n+i]*answer[j];answer[i]=value/L[i*n+i];}
            double es=1e-6,ee=0.0,rs=1e-6,reconstructionError=0.0;
            for(unsigned i=0u;i<n;++i){const double value=data<nm_float4>(preconditioned)[base+e*n+i].x;es=std::max(es,std::abs(answer[i]));ee=std::max(ee,std::abs(value-answer[i]));double reconstructed=0.0;for(unsigned j=0u;j<n;++j)reconstructed+=B[i*n+j]*data<nm_float4>(preconditioned)[base+e*n+j].x;const double rhs=data<nm_float4>(direction)[base+e*n+i].x;rs=std::max(rs,std::abs(rhs));reconstructionError=std::max(reconstructionError,std::abs(reconstructed-rhs));}
            inverseError=std::max(inverseError,ee/es);inverseResidual=std::max(inverseResidual,reconstructionError/rs);
        }
        std::vector<char> factorBytes(factorB.length),preconditionedBytes(preconditioned.length);
        std::memcpy(factorBytes.data(),factorB.contents,factorB.length);std::memcpy(preconditionedBytes.data(),preconditioned.contents,preconditioned.length);
        std::vector<char> linBytes(lin.length),rBytes(r.length),opBytes(op.length);
        std::memcpy(linBytes.data(),lin.contents,lin.length);std::memcpy(rBytes.data(),r.contents,r.length);std::memcpy(opBytes.data(),op.contents,op.length);
        execute();require(std::memcmp(factorBytes.data(),factorB.contents,factorB.length)==0&&std::memcmp(preconditionedBytes.data(),preconditioned.contents,preconditioned.length)==0,"equality preconditioner failed byte replay");require(std::memcmp(linBytes.data(),lin.contents,lin.length)==0&&std::memcmp(rBytes.data(),r.contents,r.length)==0&&std::memcmp(opBytes.data(),op.contents,op.length)==0,"equality kernels failed byte replay");
        const std::vector<float> original(data<float>(delta),data<float>(delta)+envs*fixture.nv);const float epsilon=1e-3f;
        for(unsigned k=0u;k<envs*fixture.nv;++k)data<float>(delta)[k]=original[k]+epsilon*data<nm_float4>(direction)[base+k].x;
        execute();const std::vector<nm_float4> plus(data<nm_float4>(r),data<nm_float4>(r)+vectors);
        for(unsigned k=0u;k<envs*fixture.nv;++k)data<float>(delta)[k]=original[k]-epsilon*data<nm_float4>(direction)[base+k].x;
        execute();double difference=0.0,scale=1.0;
        for(unsigned k=0u;k<envs*fixture.nv;++k){const double fd=-(double(plus[base+k].x)-data<nm_float4>(r)[base+k].x)/(2.0*epsilon);const double gpu=data<nm_float4>(op)[base+k].x;difference=std::max(difference,std::abs(fd-gpu));scale=std::max(scale,std::abs(gpu));}
        fdError=std::max(fdError,difference/scale);
        if(trial==2u) {
            const float saved=data<float>(sourceL)[0];data<float>(sourceL)[0]=-1.0f;execute(false);
            require(data<NMMatterStatusGPU>(statuses)[0].code==NM_STATUS_LINEAR_SOLVER_FAILURE,"factor failed to reject nonpositive source pivot");
            data<float>(sourceL)[0]=std::numeric_limits<float>::quiet_NaN();execute(false);
            require(data<NMMatterStatusGPU>(statuses)[0].code==NM_STATUS_NONFINITE_INPUT,"factor failed to reject nonfinite source pivot");
            data<float>(sourceL)[0]=saved;
        }
    }
    std::cout<<"rows="<<count<<" environments=2 trials=4 prepare_scaled_error="<<linError<<" residual_relative_error="<<residualError<<" action_relative_error="<<actionError<<" finite_difference_relative_error="<<fdError<<" initial_position_defect="<<initialDefect<<" factor_relative_error="<<factorError<<" inverse_relative_error="<<inverseError<<" inverse_residual="<<inverseResidual<<" static_force_scaled_error="<<staticForceError<<" source_factor=immutable invalid_pivots=rejected replay=byte_identical\n";
    require(staticForceError<3e-5,"offline static force differs from Metal source law");
    require(linError<3e-5,"source equality preparation differs from FP64");require(residualError<3e-5&&actionError<3e-5,"source equality Schur action differs from FP64");require(fdError<3e-3,"source equality operator fails residual finite difference");
    require(factorError<3e-5,"equality factor differs from independent FP64 SPD assembly");
    require(inverseError<5e-4&&inverseResidual<5e-4,"equality preconditioner differs from independent FP64 inverse");
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {require(argc==1||argc==3,"usage: probe [NHRIGID2 NHEQ2]");run(argc==3?authored(argv[1],argv[2]):synthetic());return 0;}
        catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
}
