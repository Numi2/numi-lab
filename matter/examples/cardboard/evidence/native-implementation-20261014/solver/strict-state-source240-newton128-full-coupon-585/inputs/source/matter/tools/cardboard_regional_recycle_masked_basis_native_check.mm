#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "Compile with NUMI_MATTER_METALLIB set to the source-bound Metal library"
#endif

namespace {
constexpr std::uint32_t kEnvironmentCount = 2u;
constexpr std::uint32_t kNodeCount = 5u;
constexpr std::uint32_t kBaseModeCount = 2u;
constexpr std::uint32_t kModeCount = kBaseModeCount + 1u;
constexpr std::size_t kBasisWords =
    static_cast<std::size_t>(kEnvironmentCount) * kModeCount * kNodeCount;
constexpr float kVectorTolerance = 3.0e-5f;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

id<MTLBuffer> sharedBuffer(id<MTLDevice> device, const void* source,
                           std::size_t size, const char* label) {
    require(size != 0u, std::string(label) + " buffer must not be empty");
    id<MTLBuffer> buffer = [device newBufferWithBytes:source
                                               length:size
                                              options:MTLResourceStorageModeShared];
    require(buffer != nil, std::string("could not allocate ") + label);
    return buffer;
}

template<class T>
id<MTLBuffer> sharedVector(id<MTLDevice> device,
                           const std::vector<T>& values,
                           const char* label) {
    return sharedBuffer(device, values.data(), values.size() * sizeof(T), label);
}

template<class T>
id<MTLBuffer> zeroVector(id<MTLDevice> device, std::size_t count,
                         const char* label) {
    std::vector<T> zeros(count);
    return sharedVector(device, zeros, label);
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                    id<MTLLibrary> library,
                                    const char* shortName) {
    NSString* qualified = [NSString stringWithFormat:
        @"numi_matter_metal::%s", shortName];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    require(function != nil,
        std::string("source-bound kernel is missing: ") + shortName);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, std::string("cannot compile kernel ") + shortName +
        (error ? std::string(": ") + error.localizedDescription.UTF8String : ""));
    return result;
}

void setBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer,
               NSUInteger index) {
    [encoder setBuffer:buffer offset:0u atIndex:index];
}

void setU32(id<MTLComputeCommandEncoder> encoder, std::uint32_t value,
            NSUInteger index) {
    [encoder setBytes:&value length:sizeof(value) atIndex:index];
}

std::size_t basisIndex(std::uint32_t environment, std::uint32_t mode,
                       std::uint32_t node) {
    return (static_cast<std::size_t>(environment) * kModeCount + mode) *
        kNodeCount + node;
}

float component(const nm_float4& value, unsigned axis) {
    switch (axis) {
        case 0u: return value.x;
        case 1u: return value.y;
        default: return value.z;
    }
}

void checkNear(float actual, float expected, const std::string& what) {
    require(std::isfinite(actual) && std::isfinite(expected),
            what + " is nonfinite");
    require(std::abs(actual - expected) <= kVectorTolerance,
            what + " differs: actual=" + std::to_string(actual) +
                " expected=" + std::to_string(expected));
}

// A deterministic candidate-bound synthetic SPD action used only to verify
// that the selector does not alter the already projected source modes. It is
// not the Matter FEM operator and cannot qualify production A parity.
std::vector<float> syntheticAction(const std::vector<nm_float4>& basis,
                                   const std::vector<NMFEMNodeStateGPU>& nodes,
                                   std::uint32_t environment,
                                   std::uint32_t mode) {
    constexpr std::uint32_t dofCount = 3u * kNodeCount;
    std::array<float, dofCount * dofCount> matrix{};
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const auto& candidate = nodes[environment * kNodeCount + node];
        const float diagonal = 2.0f + 0.125f * static_cast<float>(node) +
            0.25f * static_cast<float>(environment) +
            0.0625f * candidate.restAndFixed.w;
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const std::uint32_t row = 3u * node + axis;
            matrix[row * dofCount + row] = diagonal + 0.1f * axis;
        }
    }
    for (std::uint32_t node = 0u; node + 1u < kNodeCount; ++node) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const std::uint32_t lhs = 3u * node + axis;
            const std::uint32_t rhs = 3u * (node + 1u) + axis;
            matrix[lhs * dofCount + rhs] = 0.0625f;
            matrix[rhs * dofCount + lhs] = 0.0625f;
        }
    }
    std::array<float, dofCount> x{};
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const nm_float4 value = basis[basisIndex(environment, mode, node)];
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            x[3u * node + axis] = component(value, axis);
    }
    std::vector<float> result(dofCount, 0.0f);
    for (std::uint32_t row = 0u; row < dofCount; ++row) {
        for (std::uint32_t column = 0u; column < dofCount; ++column)
            result[row] += matrix[row * dofCount + column] * x[column];
    }
    return result;
}

std::vector<nm_float4> authoredModes() {
    std::vector<nm_float4> result(kModeCount * kNodeCount);
    // Source rows 0 and 1 are orthogonal before projection. Removing node 1
    // makes them nonorthogonal but independent; removing nodes 1 and 2 makes
    // them exactly dependent. Row 2 is a deterministic static fallback.
    result[0u * kNodeCount + 0u] = {1.0f, 0.0f, 0.0f, 0.0f};
    result[0u * kNodeCount + 1u] = {1.0f, 0.0f, 0.0f, 0.0f};
    result[1u * kNodeCount + 0u] = {1.0f, 0.0f, 0.0f, 0.0f};
    result[1u * kNodeCount + 1u] = {-1.0f, 0.0f, 0.0f, 0.0f};
    result[1u * kNodeCount + 2u] = {1.0f, 0.0f, 0.0f, 0.0f};
    result[2u * kNodeCount + 3u] = {0.0f, 2.0f, 0.0f, 0.0f};
    result[2u * kNodeCount + 4u] = {0.0f, 1.0f, 0.0f, 0.0f};
    return result;
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const char* libraryPath = argc > 1 ? argv[1] : NUMI_MATTER_METALLIB;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSURL* url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:libraryPath]];
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            require(library != nil, std::string("cannot load source-bound metallib: ") +
                (error ? error.localizedDescription.UTF8String : "unknown error"));
            auto project = pipeline(device, library,
                "nm_fgmres_regional_project_basis");
            auto select = pipeline(device, library,
                "nm_fgmres_regional_select_recycle_mode");

            NMMatterDispatchGPU dispatch{};
            dispatch.environmentCount = kEnvironmentCount;
            dispatch.femNodeCount = kNodeCount;
            std::vector<NMFEMNodeStateGPU> candidate(
                kEnvironmentCount * kNodeCount);
            candidate[1u].restAndFixed.w = 1.0f;
            candidate[kNodeCount + 1u].restAndFixed.w = 3.0f;
            candidate[kNodeCount + 2u].restAndFixed.w = 1.0f;
            std::vector<NMMatterStatusGPU> statuses(kEnvironmentCount);
            const std::vector<nm_float4> authored = authoredModes();
            std::vector<nm_float4> correction(kEnvironmentCount * kNodeCount);
            correction[4u] = {1.0f, 0.0f, 0.0f, 0.0f};
            // Environment 1 proposes a correction in the dependent source span,
            // so selection must keep the static fallback.
            correction[kNodeCount + 0u] = {2.0f, 0.0f, 0.0f, 0.0f};
            std::vector<nm_float4> effective(kBasisWords);
            std::vector<nm_float4> effectiveBefore(kBasisWords);
            std::vector<nm_float4> orthogonal(
                kEnvironmentCount * kBaseModeCount * kNodeCount);
            std::vector<std::uint32_t> selected(kEnvironmentCount, 99u);

            id<MTLBuffer> dispatchBuffer = sharedBuffer(device, &dispatch,
                sizeof(dispatch), "dispatch");
            id<MTLBuffer> candidateBuffer = sharedVector(device, candidate,
                "candidate nodes");
            id<MTLBuffer> authoredBuffer = sharedVector(device, authored,
                "authored modes");
            id<MTLBuffer> correctionBuffer = sharedVector(device, correction,
                "accepted correction");
            id<MTLBuffer> effectiveBuffer = sharedVector(device, effective,
                "effective modes");
            id<MTLBuffer> orthogonalBuffer = sharedVector(device, orthogonal,
                "orthogonal workspace");
            id<MTLBuffer> selectedBuffer = sharedVector(device, selected,
                "selected mode flags");
            id<MTLBuffer> statusBuffer = sharedVector(device, statuses,
                "status");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            id<MTLCommandBuffer> projectionCommand = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [projectionCommand computeCommandEncoder];
            [encoder setComputePipelineState:project];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, authoredBuffer, 1u);
            setBuffer(encoder, candidateBuffer, 2u);
            setBuffer(encoder, effectiveBuffer, 3u);
            setU32(encoder, kModeCount, 4u);
            setU32(encoder, kModeCount, 5u); // no padding in this direct fixture.
            [encoder dispatchThreads:MTLSizeMake(kEnvironmentCount *
                kModeCount * kNodeCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            [encoder endEncoding];
            [projectionCommand commit];
            [projectionCommand waitUntilCompleted];
            require(projectionCommand.status == MTLCommandBufferStatusCompleted,
                "candidate mask projection command did not complete");
            std::memcpy(effectiveBefore.data(), effectiveBuffer.contents,
                        effectiveBefore.size() * sizeof(nm_float4));
            for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
                for (std::uint32_t mode = 0u; mode < kModeCount; ++mode) {
                    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                        const bool constrained =
                            candidate[env * kNodeCount + node].restAndFixed.w == 1.0f ||
                            candidate[env * kNodeCount + node].restAndFixed.w == 3.0f;
                        const nm_float4 expected = constrained
                            ? nm_float4{} : authored[mode * kNodeCount + node];
                        const auto index = basisIndex(env, mode, node);
                        require(std::memcmp(&effectiveBefore[index], &expected,
                            sizeof(nm_float4)) == 0,
                            "project kernel output did not match candidate mask and source P");
                    }
                }
            }

            id<MTLCommandBuffer> selectionCommand = [queue commandBuffer];
            encoder = [selectionCommand computeCommandEncoder];
            [encoder setComputePipelineState:select];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, authoredBuffer, 1u);
            setBuffer(encoder, candidateBuffer, 2u);
            setBuffer(encoder, correctionBuffer, 3u);
            setBuffer(encoder, effectiveBuffer, 4u);
            setBuffer(encoder, selectedBuffer, 5u);
            setBuffer(encoder, statusBuffer, 6u);
            setU32(encoder, kBaseModeCount, 7u);
            setU32(encoder, kModeCount, 8u);
            setBuffer(encoder, orthogonalBuffer, 9u);
            [encoder dispatchThreadgroups:MTLSizeMake(kEnvironmentCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            [encoder endEncoding];
            [selectionCommand commit];
            [selectionCommand waitUntilCompleted];
            require(selectionCommand.status == MTLCommandBufferStatusCompleted,
                "selector command buffer did not complete");

            std::memcpy(effective.data(), effectiveBuffer.contents,
                        effective.size() * sizeof(nm_float4));
            std::memcpy(selected.data(), selectedBuffer.contents,
                        selected.size() * sizeof(std::uint32_t));
            std::memcpy(orthogonal.data(), orthogonalBuffer.contents,
                        orthogonal.size() * sizeof(nm_float4));
            require(std::memcmp(authoredBuffer.contents, authored.data(),
                authored.size() * sizeof(nm_float4)) == 0,
                "selector modified source-authored P");
            require(std::memcmp(candidateBuffer.contents, candidate.data(),
                candidate.size() * sizeof(NMFEMNodeStateGPU)) == 0,
                "selector modified candidate nodes");
            require(std::memcmp(statusBuffer.contents, statuses.data(),
                statuses.size() * sizeof(NMMatterStatusGPU)) == 0,
                "selector modified solver status");
            require(selected[0u] == 1u,
                "environment 0 did not accept independent recycle mode");
            require(selected[1u] == 0u,
                "environment 1 did not fall back for dependent source span");

            // The two projected source rows are nonorthogonal in environment 0
            // and exactly dependent in environment 1.
            const auto inner = [&](std::uint32_t env, std::uint32_t lhs,
                                   std::uint32_t rhs) {
                float value = 0.0f;
                for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                    const nm_float4 a = effectiveBefore[basisIndex(env,lhs,node)];
                    const nm_float4 b = effectiveBefore[basisIndex(env,rhs,node)];
                    value += a.x*b.x + a.y*b.y + a.z*b.z;
                }
                return value;
            };
            const float gramOffDiagonal = inner(0u, 0u, 1u);
            require(std::abs(gramOffDiagonal) > 0.5f,
                "environment 0 did not create the intended nonorthogonal mask");
            const auto workspaceIndex = [](std::uint32_t env,
                                            std::uint32_t mode,
                                            std::uint32_t node) {
                return (static_cast<std::size_t>(env) * kBaseModeCount + mode) *
                    kNodeCount + node;
            };
            for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                const nm_float4 env0Mode0 = orthogonal[workspaceIndex(0u,0u,node)];
                const nm_float4 env0Mode1 = orthogonal[workspaceIndex(0u,1u,node)];
                checkNear(env0Mode0.x, node == 0u ? 1.0f : 0.0f,
                    "MGS environment 0 mode 0");
                checkNear(env0Mode1.x, node == 2u ? 1.0f : 0.0f,
                    "MGS environment 0 mode 1");
                const nm_float4 env1Mode1 = orthogonal[workspaceIndex(1u,1u,node)];
                checkNear(env1Mode1.x, 0.0f, "dependent MGS row");
                checkNear(env1Mode1.y, 0.0f, "dependent MGS row");
                checkNear(env1Mode1.z, 0.0f, "dependent MGS row");
            }

            // Base P rows remain bitwise identical; the new mode is orthogonal
            // to the projected base span and normalized.
            for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
                for (std::uint32_t mode = 0u; mode < kBaseModeCount; ++mode) {
                    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                        const auto index = basisIndex(env, mode, node);
                        require(std::memcmp(&effective[index],
                            &effectiveBefore[index], sizeof(nm_float4)) == 0,
                            "selector modified a physical source P row");
                    }
                }
            }
            float selectedNorm2 = 0.0f;
            for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                const nm_float4 value = effective[basisIndex(0u, 2u, node)];
                selectedNorm2 += value.x*value.x + value.y*value.y + value.z*value.z;
                checkNear(value.x, node == 4u ? 1.0f : 0.0f,
                    "accepted recycle raw x component");
                checkNear(value.y, 0.0f, "accepted recycle raw y component");
                checkNear(value.z, 0.0f, "accepted recycle raw z component");
            }
            checkNear(selectedNorm2, 1.0f, "accepted recycle mode norm squared");
            for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
                const nm_float4 actual = effective[basisIndex(1u, 2u, node)];
                const nm_float4 expected = effectiveBefore[basisIndex(1u, 2u, node)];
                require(std::memcmp(&actual, &expected, sizeof(nm_float4)) == 0,
                    "dependent environment did not restore its static fallback");
            }

            // Verify that this selection stage leaves the candidate-bound
            // synthetic A action on the source span unchanged. The production
            // FEM operator is outside this small kernel-level fixture.
            std::uint64_t checkedActions = 0u;
            for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
                for (std::uint32_t mode = 0u; mode < kBaseModeCount; ++mode) {
                    const auto before = syntheticAction(effectiveBefore,
                        candidate, env, mode);
                    const auto after = syntheticAction(effective,
                        candidate, env, mode);
                    require(before == after,
                        "candidate-bound synthetic A action changed under recycle selection");
                    ++checkedActions;
                }
            }
            std::cout << "{\"schema\":\"numi.matter.regional_recycle_masked_basis_native.v1\","
                      << "\"environments\":2,\"nodes_per_environment\":5,"
                      << "\"base_modes\":2,\"mode_count\":3,"
                      << "\"selected\":[1,0],\"env0_projected_gram_offdiagonal\":"
                      << gramOffDiagonal << ",\"candidate_bound_synthetic_A_actions_checked\":"
                      << checkedActions << ",\"production_FEM_A_qualified\":false,"
                      << "\"physical_steps\":0,\"native_kernel_execution\":true,\"physical_validation\":false}\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "regional recycle masked basis check failed: "
                      << exception.what() << "\n";
            return 1;
        }
    }
}
