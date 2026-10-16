#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
constexpr std::uint32_t kEnvironmentCount = 2u;
constexpr std::uint32_t kNodeCount = 4u;
constexpr std::uint32_t kModeCount = 3u;

void need(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    need(!values.empty(), "test buffer must not be empty");
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
        length:values.size() * sizeof(T) options:MTLResourceStorageModeShared];
    need(result != nil, "Metal test-buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         const char* shortName) {
    NSString* qualified = [NSString stringWithFormat:
        @"numi_matter_metal::%s", shortName];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    need(function != nil, std::string("missing production kernel: ") + shortName);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    need(result != nil, std::string("pipeline compile failed: ") + shortName +
        (error ? std::string(": ") + error.localizedDescription.UTF8String : ""));
    return result;
}

void runGroups(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
               std::uint32_t groups,
               void (^bind)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    need(queue != nil, "Metal queue creation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    need(encoder != nil, "Metal encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    need(command.status == MTLCommandBufferStatusCompleted,
         command.error ? command.error.localizedDescription.UTF8String :
                         "Metal command did not complete");
}

void runThreads(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                std::uint32_t threads,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    need(queue != nil, "Metal queue creation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    need(encoder != nil, "Metal encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    [encoder dispatchThreads:MTLSizeMake(threads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    need(command.status == MTLCommandBufferStatusCompleted,
         command.error ? command.error.localizedDescription.UTF8String :
                         "Metal command did not complete");
}

void setBuffer(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, NSUInteger i) {
    [e setBuffer:b offset:0u atIndex:i];
}
void setU32(id<MTLComputeCommandEncoder> e, std::uint32_t x, NSUInteger i) {
    [e setBytes:&x length:sizeof(x) atIndex:i];
}
void exact(float actual, float expected, const char* label) {
    need(std::memcmp(&actual, &expected, sizeof(float)) == 0,
         std::string(label) + " changed");
}
std::size_t regionalVectorIndex(std::uint32_t env, std::uint32_t mode, std::uint32_t node) {
    return (env * kModeCount + mode) * kNodeCount + node;
}
std::size_t matrixIndex(std::uint32_t env, std::uint32_t row,
                        std::uint32_t column) {
    return (env * kModeCount + row) * kModeCount + column;
}
std::size_t extraIndex(std::uint32_t env, std::uint32_t node) {
    return (env * (kModeCount + 1u) + kModeCount) * kNodeCount + node;
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const char* libraryPath = argc > 1 ? argv[1] : NUMI_MATTER_METALLIB;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            need(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath]]
                error:&error];
            need(library != nil, error ? error.localizedDescription.UTF8String :
                 "source-bound metallib load failed");

            const auto build = makePipeline(device, library,
                "nm_fgmres_regional_restart_build_rap");
            const auto factor = makePipeline(device, library,
                "nm_fgmres_regional_restart_factor_rap");
            const auto fallback = makePipeline(device, library,
                "nm_fgmres_regional_restart_fallback_if_factor_invalid");
            const auto identity = makePipeline(device, library,
                "nm_fgmres_regional_restart_identity_if_invalid");
            const auto clear = makePipeline(device, library,
                "nm_fgmres_regional_restart_clear_if_invalid");
            const auto reset = makePipeline(device, library,
                "nm_fgmres_regional_restart_reset_metadata");

            NMMatterDispatchGPU dispatch{};
            dispatch.environmentCount = kEnvironmentCount;
            dispatch.femNodeCount = kNodeCount;
            const auto dispatchBuffer = makeBuffer(device,
                std::vector<NMMatterDispatchGPU>{dispatch});

            // Active env 0 forms a nonsymmetric, invertible 3x3 RAP matrix.
            // Preserved env 1 starts with per-entry sentinels.
            const float p[kModeCount][kNodeCount] = {
                {1, 0, 1, 0}, {0, 1, 1, 0}, {0, 0, 0, 1}
            };
            std::vector<nm_float4> basis(kEnvironmentCount*kModeCount*kNodeCount);
            std::vector<nm_float4> ap(basis.size());
            for (unsigned mode=0; mode<kModeCount; ++mode) {
                for (unsigned node=0; node<kNodeCount; ++node) {
                    basis[regionalVectorIndex(0,mode,node)] = {p[mode][node],0,0,0};
                    ap[regionalVectorIndex(0,mode,node)] = {
                        float(mode+2u)*p[mode][node],0,0,0};
                    basis[regionalVectorIndex(1,mode,node)] = {
                        float(10u+mode+node),float(20u+mode),float(node),0};
                    ap[regionalVectorIndex(1,mode,node)] = {
                        float(30u+mode),float(40u+node),float(mode+node),0};
                }
            }
            std::vector<float> rap(kEnvironmentCount*kModeCount*kModeCount);
            for (unsigned i=0;i<kModeCount*kModeCount;++i) rap[i]=91.0f;
            for (unsigned i=kModeCount*kModeCount;i<rap.size();++i)
                rap[i]=500.0f+float(i);
            const auto rapBefore = rap;
            std::vector<std::uint32_t> tail{0u,1u};
            const auto basisBuffer=makeBuffer(device,basis);
            const auto apBuffer=makeBuffer(device,ap);
            const auto rapBuffer=makeBuffer(device,rap);
            const auto tailBuffer=makeBuffer(device,tail);

            runGroups(device,build,kEnvironmentCount*kModeCount*kModeCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,basisBuffer,1);
                    setBuffer(e,apBuffer,2); setBuffer(e,rapBuffer,3);
                    setU32(e,kModeCount,4); setU32(e,kModeCount,5);
                    setBuffer(e,tailBuffer,6);
                });
            const float expected[9]={4,3,0,2,6,0,0,0,4};
            const auto* actualRap=static_cast<const float*>(rapBuffer.contents);
            for(unsigned i=0;i<9;++i) exact(actualRap[i],expected[i],"active RAP");
            for(unsigned i=9;i<rap.size();++i)
                exact(actualRap[i],rapBefore[i],"preserved RAP sentinel");

            // A valid active factor must be produced. A preserved invalid
            // factor (including NaN matrix words) must stay byte-identical.
            std::vector<float> factorMatrix(actualRap,actualRap+rap.size());
            const std::uint32_t qnan=0x7fc12345u;
            for(unsigned i=9;i<18;++i)
                std::memcpy(&factorMatrix[i],&qnan,sizeof(qnan));
            const auto factorMatrixBefore=factorMatrix;
            auto factorMatrixBuffer=makeBuffer(device,factorMatrix);
            std::vector<std::uint32_t> pivots(6u,0xdeadbeefu);
            const auto pivotsBefore=pivots;
            auto pivotsBuffer=makeBuffer(device,pivots);
            std::vector<std::uint32_t> valid{0u,0u};
            auto validBuffer=makeBuffer(device,valid);
            runGroups(device,factor,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,factorMatrixBuffer,1);
                    setBuffer(e,pivotsBuffer,2); setBuffer(e,validBuffer,3);
                    setU32(e,kModeCount,4); setBuffer(e,tailBuffer,5);
                });
            const auto* validOut=static_cast<const std::uint32_t*>(validBuffer.contents);
            need(validOut[0]==1u,"active RAP factor did not become valid");
            need(validOut[1]==0u,"preserved invalid RAP flag changed");
            need(std::memcmp(static_cast<const float*>(factorMatrixBuffer.contents)+9,
                    factorMatrixBefore.data()+9,9u*sizeof(float))==0,
                 "preserved invalid RAP matrix changed");
            need(std::memcmp(static_cast<const std::uint32_t*>(pivotsBuffer.contents)+3,
                    pivotsBefore.data()+3,3u*sizeof(std::uint32_t))==0,
                 "preserved invalid RAP pivots changed");

            // Also preserve a valid factor. The active singular matrix must
            // remain invalid, demonstrating the ordinary failure path.
            std::vector<float> singular(actualRap,actualRap+rap.size());
            std::fill(singular.begin(),singular.begin()+9u,0.0f);
            for(unsigned i=9;i<18;++i) singular[i]=float(i+1u);
            const auto singularBefore=singular;
            auto singularBuffer=makeBuffer(device,singular);
            std::vector<std::uint32_t> pivots2(6u,0xcafef00du);
            const auto pivots2Before=pivots2;
            auto pivots2Buffer=makeBuffer(device,pivots2);
            auto valid2Buffer=makeBuffer(device,std::vector<std::uint32_t>{0u,1u});
            runGroups(device,factor,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,singularBuffer,1);
                    setBuffer(e,pivots2Buffer,2); setBuffer(e,valid2Buffer,3);
                    setU32(e,kModeCount,4); setBuffer(e,tailBuffer,5);
                });
            const auto* valid2=static_cast<const std::uint32_t*>(valid2Buffer.contents);
            need(valid2[0]==0u && valid2[1]==1u,
                 "active-invalid/preserved-valid factor result mismatch");
            need(std::memcmp(static_cast<const float*>(singularBuffer.contents)+9,
                    singularBefore.data()+9,9u*sizeof(float))==0,
                 "preserved valid RAP matrix changed");
            need(std::memcmp(static_cast<const std::uint32_t*>(pivots2Buffer.contents)+3,
                    pivots2Before.data()+3,3u*sizeof(std::uint32_t))==0,
                 "preserved valid RAP pivots changed");

            // Invalid fallback only changes active env 0; preserved env 1
            // retains an already-valid or invalid mode-valid byte.
            auto coarseInvalid=makeBuffer(device,std::vector<std::uint32_t>{0u,0u});
            auto modeValid=makeBuffer(device,std::vector<std::uint32_t>{1u,1u});
            runGroups(device,fallback,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,coarseInvalid,1);
                    setBuffer(e,modeValid,2); setBuffer(e,tailBuffer,3);
                });
            const auto* modesAfter=static_cast<const std::uint32_t*>(modeValid.contents);
            need(modesAfter[0]==0u && modesAfter[1]==1u,
                 "active/preserved factor fallback mismatch");
            auto invalidMode=makeBuffer(device,std::vector<std::uint32_t>{1u,0u});
            runGroups(device,fallback,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,coarseInvalid,1);
                    setBuffer(e,invalidMode,2); setBuffer(e,tailBuffer,3);
                });
            const auto* invalidAfter=static_cast<const std::uint32_t*>(invalidMode.contents);
            need(invalidAfter[0]==0u && invalidAfter[1]==0u,
                 "preserved invalid mode-valid byte changed");

            // Active invalid mode gets the exact fallback identity row/column;
            // preserved invalid mode's full matrix remains sentinel bytes.
            std::vector<float> identityMatrix(18u);
            for(unsigned i=0;i<identityMatrix.size();++i) identityMatrix[i]=70.0f+float(i);
            const auto identityBefore=identityMatrix;
            auto identityBuffer=makeBuffer(device,identityMatrix);
            auto targetScale=makeBuffer(device,std::vector<float>{5.0f,7.0f});
            auto invalidModes=makeBuffer(device,std::vector<std::uint32_t>{0u,0u});
            runGroups(device,identity,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,identityBuffer,1);
                    setBuffer(e,invalidModes,2); setBuffer(e,targetScale,3);
                    setU32(e,kModeCount,4); setBuffer(e,tailBuffer,5);
                });
            const auto* identityOut=static_cast<const float*>(identityBuffer.contents);
            for(unsigned row=0;row<kModeCount;++row) {
                exact(identityOut[matrixIndex(0,row,2)],
                      row==2u?5.0f:0.0f,"active fallback identity column");
                exact(identityOut[matrixIndex(0,2,row)],
                      row==2u?5.0f:0.0f,"active fallback identity row");
            }
            need(std::memcmp(identityOut+9,identityBefore.data()+9,9u*sizeof(float))==0,
                 "preserved invalid-mode coarse matrix changed");

            // Clear only the active invalid extra mode; preserve every word
            // of the tail environment's basis and AP arrays.
            std::vector<nm_float4> q(kEnvironmentCount*(kModeCount+1u)*kNodeCount);
            std::vector<nm_float4> aq(q.size());
            for(unsigned i=0;i<q.size();++i) {
                q[i]={float(i+1u),-2.0f,3.0f,4.0f};
                aq[i]={5.0f,float(i+2u),-1.0f,6.0f};
            }
            const auto qBefore=q, aqBefore=aq;
            auto qBuffer=makeBuffer(device,q), aqBuffer=makeBuffer(device,aq);
            auto invalids=makeBuffer(device,std::vector<std::uint32_t>{0u,0u});
            runThreads(device,clear,kEnvironmentCount*kNodeCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,qBuffer,1);
                    setBuffer(e,aqBuffer,2); setBuffer(e,invalids,3);
                    setU32(e,kModeCount,4); setBuffer(e,tailBuffer,5);
                });
            const auto* qOut=static_cast<const nm_float4*>(qBuffer.contents);
            const auto* aqOut=static_cast<const nm_float4*>(aqBuffer.contents);
            const nm_float4 zero{};
            for(unsigned node=0;node<kNodeCount;++node) {
                const auto active=extraIndex(0,node), preserved=extraIndex(1,node);
                need(std::memcmp(&qOut[active],&zero,sizeof(zero))==0 &&
                     std::memcmp(&aqOut[active],&zero,sizeof(zero))==0,
                     "active invalid extra mode/AP was not cleared");
                need(std::memcmp(&qOut[preserved],&qBefore[preserved],sizeof(zero))==0 &&
                     std::memcmp(&aqOut[preserved],&aqBefore[preserved],sizeof(zero))==0,
                     "preserved invalid extra mode/AP sentinel changed");
            }

            // The existing per-Newton reset invalidates both environments'
            // stale preservation bits and factor validity.
            std::vector<std::uint32_t> generation(8u,0xdeadbeefu);
            auto generationBuffer=makeBuffer(device,generation);
            auto raw=makeBuffer(device,std::vector<float>{2.0f,3.0f});
            auto scaleOut=makeBuffer(device,std::vector<float>{4.0f,5.0f});
            auto modeScaleValid=makeBuffer(device,std::vector<std::uint32_t>{1u,1u});
            auto resetTarget=makeBuffer(device,std::vector<float>{6.0f,7.0f});
            auto factorValid=makeBuffer(device,std::vector<std::uint32_t>{1u,1u});
            auto staleTail=makeBuffer(device,std::vector<std::uint32_t>{1u,1u});
            runGroups(device,reset,kEnvironmentCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,generationBuffer,1);
                    setBuffer(e,raw,2); setBuffer(e,scaleOut,3);
                    setBuffer(e,modeScaleValid,4); setBuffer(e,resetTarget,5);
                    setBuffer(e,factorValid,6); setBuffer(e,staleTail,7);
                });
            const auto* tailOut=static_cast<const std::uint32_t*>(staleTail.contents);
            const auto* factorOut=static_cast<const std::uint32_t*>(factorValid.contents);
            need(tailOut[0]==0u && tailOut[1]==0u &&
                 factorOut[0]==0u && factorOut[1]==0u,
                 "Newton reset failed to clear tail/factor markers");

            // After the reset, both E2 environments must be eligible to rebuild
            // their next-Newton RAP. This catches a stale preservation marker
            // and proves the host/kernel E*m^2 launch extent covers env 1.
            std::vector<float> nextRAP(rap.size(), -123.0f);
            auto nextRAPBuffer = makeBuffer(device, nextRAP);
            runGroups(device,build,kEnvironmentCount*kModeCount*kModeCount,
                ^(id<MTLComputeCommandEncoder> e) {
                    setBuffer(e,dispatchBuffer,0); setBuffer(e,basisBuffer,1);
                    setBuffer(e,apBuffer,2); setBuffer(e,nextRAPBuffer,3);
                    setU32(e,kModeCount,4); setU32(e,kModeCount,5);
                    setBuffer(e,staleTail,6);
                });
            const auto* nextRAPOut =
                static_cast<const float*>(nextRAPBuffer.contents);
            for (unsigned env=0; env<kEnvironmentCount; ++env) {
                for (unsigned row=0; row<kModeCount; ++row) {
                    for (unsigned column=0; column<kModeCount; ++column) {
                        float expectedValue=0.0f;
                        for (unsigned node=0; node<kNodeCount; ++node) {
                            const auto& pv=basis[regionalVectorIndex(env,row,node)];
                            const auto& av=ap[regionalVectorIndex(env,column,node)];
                            expectedValue += pv.x*av.x + pv.y*av.y + pv.z*av.z;
                        }
                        exact(nextRAPOut[matrixIndex(env,row,column)],
                              expectedValue,"next-Newton E2 RAP rebuild");
                    }
                }
            }

            std::cout << "regional_rap_tail_preserve=pass cases=E2_mixed,"
                "active_RAP_and_LU,preserve_valid_and_invalid_LU,"
                "invalid_fallback_and_identity,invalid_extra_clear,newton_reset"
                " physical_transactions=0\n";
        } catch (const std::exception& error) {
            std::cerr << "regional RAP tail preserve checker failed: "
                      << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
