#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_BASE_CGLS_CHECK_METALLIB
#error "Build must define NUMI_BASE_CGLS_CHECK_METALLIB"
#endif

namespace {
struct alignas(16) F4 { float x, y, z, w; };
using U4 = std::array<std::uint32_t, 4>;
using U = std::uint32_t;
constexpr U kExhausted=2u, kZeroNormalResidual=3u, kInvalidInput=5u;

unsigned checks = 0;
void require(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}

id<MTLBuffer> buffer(id<MTLDevice> device, const void* bytes, std::size_t size,
                     const char* label) {
    require(bytes != nullptr && size != 0u, std::string("empty buffer: ") + label);
    id<MTLBuffer> result = [device newBufferWithBytes:bytes length:size
        options:MTLResourceStorageModeShared];
    require(result != nil, std::string("Metal buffer allocation failed: ") + label);
    return result;
}

template <typename T>
id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values,
                     const char* label) {
    return buffer(device, values.data(), values.size() * sizeof(T), label);
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSString* fullName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:fullName];
    require(function != nil, std::string("missing CGLS kernel: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    const char* message = (error != nil && error.localizedDescription != nil)
        ? error.localizedDescription.UTF8String : "no compiler diagnostic";
    require(result != nil, std::string("CGLS pipeline failed: ") + name + ": " + message);
    return result;
}

void run(id<MTLDevice> device, id<MTLCommandQueue> queue,
         id<MTLComputePipelineState> state, NSUInteger groups,
         NSUInteger threads,
         const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command != nil
        ? [command computeCommandEncoder] : nil;
    require(command != nil && encoder != nil, "cannot create Metal encoder");
    [encoder setComputePipelineState:state];
    bind(encoder);
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    const char* message = (command.error != nil &&
        command.error.localizedDescription != nil)
        ? command.error.localizedDescription.UTF8String : "no Metal diagnostic";
    require(command.status == MTLCommandBufferStatusCompleted,
        std::string("Metal command failed: ") + message);
    (void)device;
}

void b(id<MTLComputeCommandEncoder> e, id<MTLBuffer> value, NSUInteger slot) {
    [e setBuffer:value offset:0u atIndex:slot];
}
void c(id<MTLComputeCommandEncoder> e, const void* value, NSUInteger size,
       NSUInteger slot) {
    [e setBytes:value length:size atIndex:slot];
}
void count(id<MTLComputeCommandEncoder> e, U value, NSUInteger slot) {
    c(e, &value, sizeof(value), slot);
}

struct CSR {
    U n = 0;
    std::vector<U> offsets;
    std::vector<U> columns;
    std::vector<float> values;
};

CSR makeCSR(const std::vector<float>& dense, U n, bool transpose) {
    CSR out; out.n = n; out.offsets.push_back(0u);
    for (U row = 0u; row < n; ++row) {
        for (U column = 0u; column < n; ++column) {
            const float value = transpose
                ? dense[std::size_t(column) * n + row]
                : dense[std::size_t(row) * n + column];
            if (value != 0.0f) {
                out.columns.push_back(column);
                out.values.push_back(value);
            }
        }
        out.offsets.push_back(static_cast<U>(out.columns.size()));
    }
    return out;
}

std::vector<F4> applyCPU(const CSR& matrix, const std::vector<U>& freeNodes,
                         const std::vector<F4>& input, U nodeCount) {
    std::vector<F4> out(nodeCount, F4{0, 0, 0, 0});
    for (U freeIndex = 0; freeIndex < freeNodes.size(); ++freeIndex) {
        F4 result{0, 0, 0, 0};
        const U node = freeNodes[freeIndex];
        for (U axis = 0; axis < 3; ++axis) {
            const U row = 3u * freeIndex + axis;
            float sum = 0.0f;
            for (U entry = matrix.offsets[row]; entry < matrix.offsets[row + 1u]; ++entry) {
                const U column = matrix.columns[entry];
                const U columnNode = freeNodes[column / 3u];
                const float component = (&input[columnNode].x)[column % 3u];
                sum = std::fma(matrix.values[entry], component, sum);
            }
            (&result.x)[axis] = sum;
        }
        out[node] = result;
    }
    return out;
}

float dotMechanical(const std::vector<F4>& a, const std::vector<F4>& bvec,
                    const std::vector<F4>& mask, U nodeCount) {
    float sum = 0.0f;
    for (U node = 0; node < nodeCount; ++node) {
        for (U axis = 0; axis < 3; ++axis) {
            const float m = (&mask[node].x)[axis];
            sum = std::fma((&a[node].x)[axis] * m,
                           (&bvec[node].x)[axis] * m, sum);
        }
    }
    return sum;
}

void clearOutput(id<MTLDevice> device, id<MTLCommandQueue> queue,
                 id<MTLComputePipelineState> clear, id<MTLBuffer> output,
                 U rows) {
    run(device, queue, clear, (rows + 63u) / 64u, 64u,
        [&](id<MTLComputeCommandEncoder> e) {
            count(e, rows, 0u); b(e, output, 1u);
        });
}

void applyAction(id<MTLDevice> device, id<MTLCommandQueue> queue,
                 id<MTLComputePipelineState> action, id<MTLBuffer> offsets,
                 id<MTLBuffer> columns, id<MTLBuffer> values,
                 id<MTLBuffer> freeNodes, id<MTLBuffer> input,
                 id<MTLBuffer> output, id<MTLBuffer> control,
                 U nodeCount, U freeCount, U dofCount) {
    run(device, queue, action, 1u, 32u,
        [&](id<MTLComputeCommandEncoder> e) {
            count(e, nodeCount, 0u); count(e, freeCount, 1u); count(e, dofCount, 2u);
            b(e, offsets, 3u); b(e, columns, 4u); b(e, values, 5u);
            b(e, freeNodes, 6u); b(e, input, 7u); b(e, output, 8u); b(e, control, 9u);
        });
}

std::vector<F4> runCPU(const CSR& A, const CSR& AT, const std::vector<U>& freeNodes,
                      const std::vector<F4>& rhs, const std::vector<F4>& nodeMask,
                      U nodeCount, U cap) {
    std::vector<F4> x(nodeCount, F4{0,0,0,0});
    std::vector<F4> r = rhs;
    std::vector<F4> s = applyCPU(AT, freeNodes, r, nodeCount);
    std::vector<F4> p = s;
    float gamma = dotMechanical(s,s,nodeMask,nodeCount);
    for (U iteration=0; iteration<cap && gamma>0.0f; ++iteration) {
        const auto q = applyCPU(A,freeNodes,p,nodeCount);
        const float qnorm = dotMechanical(q,q,nodeMask,nodeCount);
        if (!(qnorm>0.0f) || !std::isfinite(qnorm)) break;
        const float alpha = gamma/qnorm;
        for (U i=0;i<nodeCount;++i) for (U k=0;k<3;++k) {
            (&x[i].x)[k] = ((&x[i].x)[k] + alpha*(&p[i].x)[k]) * (&nodeMask[i].x)[k];
            (&r[i].x)[k] = ((&r[i].x)[k] - alpha*(&q[i].x)[k]) * (&nodeMask[i].x)[k];
        }
        s = applyCPU(AT,freeNodes,r,nodeCount);
        const float next = dotMechanical(s,s,nodeMask,nodeCount);
        if (!(next>0.0f) || !std::isfinite(next)) break;
        const float beta = next/gamma;
        for (U i=0;i<nodeCount;++i) for (U k=0;k<3;++k)
            (&p[i].x)[k] = ((&s[i].x)[k] + beta*(&p[i].x)[k]) * (&nodeMask[i].x)[k];
        gamma=next;
    }
    return x;
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr U E=2u, N=3u, UCount=2u*E*N, FreeCount=2u, DofCount=3u*FreeCount;
            constexpr U Cap=32u;
            const std::vector<U> freeNodes{0u,2u};
            const std::vector<float> dense{
                4.0f,1.0f,0.0f,0.2f,0.0f,0.0f,
                0.1f,3.0f,1.0f,0.0f,0.0f,0.0f,
                0.0f,0.2f,2.5f,1.0f,0.0f,0.0f,
                0.0f,0.0f,0.1f,4.0f,1.0f,0.0f,
                1.0f,0.0f,0.0f,0.1f,3.0f,1.0f,
                0.0f,0.0f,1.0f,0.0f,0.2f,2.0f};
            const CSR A=makeCSR(dense,DofCount,false), AT=makeCSR(dense,DofCount,true);
            const std::vector<F4> bFree{{1.0f,-2.0f,0.5f,0.0f},
                {0.0f,0.0f,0.0f,0.0f},{3.0f,-1.0f,2.5f,0.0f}};
            std::vector<F4> rhs(UCount,F4{0,0,0,0});
            std::vector<F4> rowMask(UCount,F4{0,0,0,0});
            std::vector<U> rowEnv(UCount), selected{1u,0u};
            for (U i=0;i<E*N;++i) rowEnv[i]=i/N;
            for (U i=E*N;i<UCount;++i) rowEnv[i]=(i-E*N)/N;
            for (U i=0;i<N;++i) {
                rhs[i]=bFree[i];
                if (i==0u || i==2u) rowMask[i]=F4{1,1,1,0};
            }
            std::vector<F4> zeros(UCount,F4{0,0,0,0});
            const std::vector<U> errorsZero(E,0u);

            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device unavailable");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            require(queue!=nil,"Metal command queue unavailable");
            NSError* error=nil;
            NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_BASE_CGLS_CHECK_METALLIB]];
            id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
            const char* libMessage=(error!=nil && error.localizedDescription!=nil)
                ? error.localizedDescription.UTF8String : "no Metal diagnostic";
            require(library!=nil,std::string("cannot load CGLS check library: ")+libMessage);

            auto make=[&](const char* suffix) {
                return pipeline(device,library,(std::string("numi_matter_metal::")+suffix).c_str());
            };
            auto initState=make("nm_fgmres_base_cgls_initialize_state");
            auto initVectors=make("nm_fgmres_base_cgls_initialize_vectors");
            auto measure=make("nm_fgmres_base_cgls_measure_initial");
            auto formAlpha=make("nm_fgmres_base_cgls_form_alpha");
            auto updateXR=make("nm_fgmres_base_cgls_update_xr");
            auto complete=make("nm_fgmres_base_cgls_complete_iteration");
            auto formBeta=make("nm_fgmres_base_cgls_form_beta");
            auto updateP=make("nm_fgmres_base_cgls_update_p");
            auto finalize=make("nm_fgmres_base_cgls_finalize");
            auto clear=make("nm_base_cgls_check_clear");
            auto applyA=make("nm_base_cgls_check_apply_A");
            auto applyAT=make("nm_base_cgls_check_apply_AT");

            auto maskB=buffer(device,rowMask,"row mask");
            auto envB=buffer(device,rowEnv,"row environment");
            auto selectedB=buffer(device,selected,"selected environment");
            auto rhsB=buffer(device,rhs,"RHS");
            auto xB=buffer(device,zeros,"x"); auto rB=buffer(device,zeros,"r");
            auto sB=buffer(device,zeros,"s/ATb"); auto pB=buffer(device,zeros,"p");
            auto qB=buffer(device,zeros,"q"); auto scalarsB=buffer(device,std::vector<F4>(E),"scalars");
            auto controlB=buffer(device,std::vector<U4>(E),"control");
            auto errorsB=buffer(device,errorsZero,"errors");
            auto offsetAB=buffer(device,A.offsets,"A offsets");
            auto columnAB=buffer(device,A.columns,"A columns");
            auto valuesAB=buffer(device,A.values,"A values");
            auto offsetATB=buffer(device,AT.offsets,"AT offsets");
            auto columnATB=buffer(device,AT.columns,"AT columns");
            auto valuesATB=buffer(device,AT.values,"AT values");
            auto freeB=buffer(device,freeNodes,"free nodes");

            run(device,queue,initState,1u,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,E,0u);count(e,Cap,1u);b(e,selectedB,2u);b(e,scalarsB,3u);b(e,controlB,4u);b(e,errorsB,5u);
            });
            clearOutput(device,queue,clear,sB,UCount);
            applyAction(device,queue,applyAT,offsetATB,columnATB,valuesATB,freeB,rhsB,sB,controlB,N,FreeCount,DofCount);
            run(device,queue,initVectors,(UCount+63u)/64u,64u,[&](id<MTLComputeCommandEncoder> e){
                count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,selectedB,4u);
                b(e,rhsB,5u);b(e,sB,6u);b(e,xB,7u);b(e,rB,8u);b(e,sB,9u);b(e,pB,10u);b(e,qB,11u);b(e,errorsB,12u);count(e,0u,13u);b(e,errorsB,14u);
            });
            run(device,queue,measure,E,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,rhsB,4u);
                b(e,rB,5u);b(e,sB,6u);b(e,scalarsB,7u);b(e,controlB,8u);b(e,errorsB,9u);
            });
            for (U it=0u;it<Cap;++it) {
                clearOutput(device,queue,clear,qB,UCount);
                applyAction(device,queue,applyA,offsetAB,columnAB,valuesAB,freeB,pB,qB,controlB,N,FreeCount,DofCount);
                run(device,queue,formAlpha,E,32u,[&](id<MTLComputeCommandEncoder> e){
                    count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,qB,4u);
                    b(e,scalarsB,5u);b(e,controlB,6u);b(e,errorsB,7u);
                });
                run(device,queue,updateXR,(UCount+63u)/64u,64u,[&](id<MTLComputeCommandEncoder> e){
                    count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,scalarsB,4u);
                    b(e,controlB,5u);b(e,pB,6u);b(e,qB,7u);b(e,xB,8u);b(e,rB,9u);b(e,errorsB,10u);
                });
                run(device,queue,complete,1u,32u,[&](id<MTLComputeCommandEncoder> e){
                    count(e,E,0u);b(e,controlB,1u);b(e,errorsB,2u);
                });
                clearOutput(device,queue,clear,sB,UCount);
                applyAction(device,queue,applyAT,offsetATB,columnATB,valuesATB,freeB,rB,sB,controlB,N,FreeCount,DofCount);
                run(device,queue,formBeta,E,32u,[&](id<MTLComputeCommandEncoder> e){
                    count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,sB,4u);
                    b(e,scalarsB,5u);b(e,controlB,6u);b(e,errorsB,7u);
                });
                run(device,queue,updateP,(UCount+63u)/64u,64u,[&](id<MTLComputeCommandEncoder> e){
                    count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,scalarsB,4u);
                    b(e,controlB,5u);b(e,sB,6u);b(e,pB,7u);b(e,errorsB,8u);
                });
            }
            run(device,queue,finalize,1u,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,E,0u);b(e,controlB,1u);b(e,errorsB,2u);
            });
            const auto* nativeX=static_cast<const F4*>(xB.contents);
            const auto* nativeControl=static_cast<const U4*>(controlB.contents);
            const auto* nativeErrors=static_cast<const U*>(errorsB.contents);
            require(nativeControl[0][0]==kExhausted || nativeControl[0][0]==kZeroNormalResidual,
                    "selected environment did not finish with a valid CGLS status");
            require(nativeControl[1][0]==0u,"quiet environment state was not untouched");
            require(nativeErrors[0]==0u && nativeErrors[1]==0u,"unexpected CGLS error bits");
            const U selectedStatus=nativeControl[0][0];
            const U selectedIterations=nativeControl[0][1];
            const std::vector<F4> mechanicalMask(rowMask.begin(),rowMask.begin()+N);
            const auto cpuX=runCPU(A,AT,freeNodes,rhs,mechanicalMask,N,Cap);
            float maxAbs=0.0f,maxScale=0.0f;
            for (U node:freeNodes) for(U axis=0;axis<3u;++axis) {
                const float v=(&nativeX[node].x)[axis];
                const float ref=(&cpuX[node].x)[axis];
                require(std::isfinite(v),"native CGLS direction is nonfinite");
                maxAbs=std::max(maxAbs,std::abs(v-ref));
                maxScale=std::max(maxScale,std::abs(ref));
            }
            require(maxAbs<=5.0e-3f*std::max(maxScale,1.0e-6f),
                    "native CGLS direction differs from independent FP32 recurrence");
            const auto nativeVec=std::vector<F4>(nativeX,nativeX+N);
            const auto applied=applyCPU(A,freeNodes,nativeVec,N);
            const float startNorm=std::sqrt(dotMechanical(rhs,rhs,rowMask,N));
            std::vector<F4> residual=rhs;
            for(U i=0;i<N;++i) for(U k=0;k<3;++k)
                (&residual[i].x)[k]-=(&applied[i].x)[k];
            const float endNorm=std::sqrt(dotMechanical(residual,residual,rowMask,N));
            require(std::isfinite(endNorm) && endNorm<startNorm,
                    "CGLS candidate did not reduce the synthetic linear residual");

            // A deliberately invalid quiet-environment mask must fail closed.
            rowMask[3].x=1.0f;
            std::memcpy(maskB.contents,rowMask.data(),UCount*sizeof(F4));
            std::fill(static_cast<U*>(errorsB.contents),static_cast<U*>(errorsB.contents)+E,0u);
            run(device,queue,initState,1u,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,E,0u);count(e,Cap,1u);b(e,selectedB,2u);b(e,scalarsB,3u);b(e,controlB,4u);b(e,errorsB,5u);
            });
            clearOutput(device,queue,clear,sB,UCount);
            applyAction(device,queue,applyAT,offsetATB,columnATB,valuesATB,freeB,rhsB,sB,controlB,N,FreeCount,DofCount);
            run(device,queue,initVectors,(UCount+63u)/64u,64u,[&](id<MTLComputeCommandEncoder> e){
                count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,selectedB,4u);
                b(e,rhsB,5u);b(e,sB,6u);b(e,xB,7u);b(e,rB,8u);b(e,sB,9u);b(e,pB,10u);b(e,qB,11u);b(e,errorsB,12u);count(e,0u,13u);b(e,errorsB,14u);
            });
            run(device,queue,measure,E,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,UCount,0u);count(e,E,1u);b(e,maskB,2u);b(e,envB,3u);b(e,rhsB,4u);
                b(e,rB,5u);b(e,sB,6u);b(e,scalarsB,7u);b(e,controlB,8u);b(e,errorsB,9u);
            });
            run(device,queue,finalize,1u,32u,[&](id<MTLComputeCommandEncoder> e){
                count(e,E,0u);b(e,controlB,1u);b(e,errorsB,2u);
            });
            require(static_cast<const U4*>(controlB.contents)[1][0]==kInvalidInput,
                    "invalid quiet-environment data was not rejected");
            require(static_cast<const U*>(errorsB.contents)[1]!=0u,
                    "invalid quiet-environment data produced no error bits");
            std::cout<<"base CGLS direct-kernel checks="<<checks
                     <<" status="<<selectedStatus
                     <<" completed="<<selectedIterations
                     <<" max_direction_abs_error="<<maxAbs
                     <<" max_direction_scale="<<maxScale
                     <<" residual_before="<<startNorm
                     <<" residual_after="<<endNorm<<"\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr<<"base CGLS checker failure: "<<error.what()<<"\n";
            return 1;
        }
    }
}
