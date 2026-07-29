#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalDiscreteElasticRod.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ranges>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

using Vec3 = std::array<double, 3>;
constexpr NSUInteger kThreadgroupSize = MR_ROD_GPU_MAX_NODES;
const char kImageAnchor = 0;

MetalDiscreteElasticRodDiagnostics reject(
    MetalDiscreteElasticRodDiagnostics diagnostics,
    const MetalDiscreteElasticRodHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3& value) {
    return finite(value[0]) &&
        finite(value[1]) &&
        finite(value[2]);
}

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

Vec3 multiply(const Vec3& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double dot(const Vec3& left, const Vec3& right) {
    return
        left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

Vec3 cross(const Vec3& left, const Vec3& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

bool normalize(const Vec3& value, Vec3& output) {
    const double magnitude = std::sqrt(dot(value, value));
    if (!(magnitude > 1.0e-14) || !finite(magnitude)) {
        return false;
    }
    output = multiply(value, 1.0 / magnitude);
    return finite(output);
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Vec3 rotate(
    const Vec3& value,
    const Vec3& axis,
    const double angle
) {
    return add(
        add(
            multiply(value, std::cos(angle)),
            multiply(cross(axis, value), std::sin(angle))
        ),
        multiply(
            axis,
            dot(axis, value) * (1.0 - std::cos(angle))
        )
    );
}

Vec3 leastAligned(const Vec3& tangent) {
    const std::array<Vec3, 3> axes{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    std::size_t selected = 0u;
    for (std::size_t index = 1u; index < axes.size(); ++index) {
        if (std::abs(dot(tangent, axes[index])) <
            std::abs(dot(tangent, axes[selected]))) {
            selected = index;
        }
    }
    Vec3 output = subtract(
        axes[selected],
        multiply(
            tangent,
            dot(axes[selected], tangent)
        )
    );
    (void)normalize(output, output);
    return output;
}

bool transport(
    const Vec3& director,
    const Vec3& from,
    const Vec3& to,
    Vec3& output
) {
    const Vec3 axis = cross(from, to);
    const double sine = std::sqrt(dot(axis, axis));
    const double cosine =
        std::clamp(dot(from, to), -1.0, 1.0);
    if (sine <= 1.0e-14) {
        if (cosine < 0.0) {
            return false;
        }
        output = director;
        return true;
    }
    output = rotate(
        director,
        multiply(axis, 1.0 / sine),
        std::atan2(sine, cosine)
    );
    output = subtract(
        output,
        multiply(to, dot(output, to))
    );
    return normalize(output, output);
}

bool curvature(
    const std::array<Vec3, 3>& positions,
    const std::array<double, 2>& twists,
    std::array<double, 2>& output
) {
    Vec3 left;
    Vec3 right;
    if (!normalize(
            subtract(positions[1], positions[0]),
            left
        ) ||
        !normalize(
            subtract(positions[2], positions[1]),
            right
        )) {
        return false;
    }
    const Vec3 referenceLeft = leastAligned(left);
    Vec3 referenceRight;
    if (!transport(
            referenceLeft,
            left,
            right,
            referenceRight
        )) {
        return false;
    }
    const Vec3 directorLeft =
        rotate(referenceLeft, left, twists[0]);
    const Vec3 directorRight =
        rotate(referenceRight, right, twists[1]);
    const Vec3 secondLeft = cross(left, directorLeft);
    const Vec3 secondRight = cross(right, directorRight);
    const double denominator = 1.0 + dot(left, right);
    if (!(denominator > 1.0e-8) || !finite(denominator)) {
        return false;
    }
    const Vec3 binormal = multiply(
        cross(left, right),
        2.0 / denominator
    );
    output = {
        0.5 * dot(binormal, add(secondLeft, secondRight)),
        -0.5 * dot(
            binormal,
            add(directorLeft, directorRight)
        ),
    };
    return finite(output[0]) && finite(output[1]);
}

bool validState(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state
) {
    const std::size_t nodes = model.restPositions.size();
    const std::size_t edges = nodes - 1u;
    return
        state.positions.size() == nodes &&
        state.velocities.size() == nodes &&
        state.twists.size() == edges &&
        state.twistRates.size() == edges &&
        std::ranges::all_of(
            state.positions,
            [](const Vec3& value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.velocities,
            [](const Vec3& value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.twists,
            [](const double value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.twistRates,
            [](const double value) { return finite(value); }
        );
}

bool validConfig(const DiscreteElasticRodStepConfig& config) {
    return
        config.timestep > 0.0 &&
        finite(config.timestep) &&
        std::ranges::all_of(
            config.gravity,
            [](const double value) { return finite(value); }
        ) &&
        config.solverIterations > 0u &&
        config.constraintTolerance > 0.0 &&
        finite(config.constraintTolerance) &&
        config.linearDamping >= 0.0 &&
        finite(config.linearDamping) &&
        config.twistDamping >= 0.0 &&
        finite(config.twistDamping) &&
        config.derivativeStep > 0.0 &&
        finite(config.derivativeStep);
}

std::string nsString(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string{value.UTF8String}
        : std::string{};
}

std::string errorString(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    std::string result = nsString(error.localizedDescription);
    return result.empty() ? nsString(error.description) : result;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            directory / "metalrobo/MetalRobo.metallib",
            directory.parent_path() / "shaders/MetalRobo.metallib",
        };
        for (const auto& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }
    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

template <typename T>
id<MTLBuffer> inputBuffer(
    id<MTLDevice> device,
    const std::vector<T>& values
) {
    if (values.empty()) {
        return [device
            newBufferWithLength:sizeof(T)
                        options:MTLResourceStorageModeShared];
    }
    return [device
        newBufferWithBytes:values.data()
                   length:values.size() * sizeof(T)
                  options:MTLResourceStorageModeShared];
}

template <typename T>
id<MTLBuffer> outputBuffer(
    id<MTLDevice> device,
    const std::size_t count
) {
    return [device
        newBufferWithLength:
            std::max<std::size_t>(count, 1u) * sizeof(T)
                    options:MTLResourceStorageModeShared];
}

template <typename T>
bool appendBytes(
    const std::size_t count,
    std::size_t& total
) {
    const std::size_t physical = std::max<std::size_t>(
        count,
        1u
    );
    if (physical >
        std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    const std::size_t bytes = physical * sizeof(T);
    if (bytes >
        std::numeric_limits<std::size_t>::max() - total) {
        return false;
    }
    total += bytes;
    return bytes <= std::numeric_limits<NSUInteger>::max();
}

} // namespace

MetalDiscreteElasticRodDiagnostics runMetalDiscreteElasticRod(
    const DiscreteElasticRodModel& model,
    const MetalDiscreteElasticRodInput& input,
    MetalDiscreteElasticRodResult& output,
    const MetalDiscreteElasticRodConfig& config
) {
    @autoreleasepool {
        MetalDiscreteElasticRodDiagnostics diagnostics;
        std::string modelReason;
        if (!model.valid(&modelReason)) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::invalidModel,
                std::move(modelReason)
            );
        }
        if (!validConfig(config.step)) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    invalidConfiguration,
                "Metal rod step configuration is invalid"
            );
        }
        const std::size_t environmentCount = input.states.size();
        const std::size_t nodeCount = model.restPositions.size();
        const std::size_t edgeCount = nodeCount - 1u;
        if (environmentCount == 0u ||
            environmentCount >
                std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::invalidState,
                "Metal rod requires at least one environment"
            );
        }
        if (nodeCount > MR_ROD_GPU_MAX_NODES ||
            input.attachmentCount >
                MR_ROD_GPU_MAX_ATTACHMENTS) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::capacityOverflow,
                "rod exceeds the SIMD threadgroup capacity"
            );
        }
        if (input.attachments.size() !=
            environmentCount * input.attachmentCount) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::invalidAttachment,
                "attachments are not packed environment-major"
            );
        }
        for (const DiscreteElasticRodState& state :
             input.states) {
            if (!validState(model, state)) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::invalidState,
                    "rod state is invalid"
                );
            }
        }
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::set<std::uint32_t> attachedNodes;
            for (std::size_t localAttachment = 0u;
                 localAttachment < input.attachmentCount;
                 ++localAttachment) {
                const DiscreteRodAttachment& attachment =
                    input.attachments[
                        environment * input.attachmentCount +
                        localAttachment
                    ];
                if (attachment.nodeIndex >= nodeCount ||
                    !finite(attachment.targetPosition) ||
                    !finite(attachment.targetVelocity) ||
                    !finite(attachment.compliance) ||
                    attachment.compliance < 0.0 ||
                    !attachedNodes.insert(
                        attachment.nodeIndex
                    ).second) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            invalidAttachment,
                        "attachment is invalid or duplicates a node"
                    );
                }
            }
        }

        MRRodGPUDispatch dispatch{};
        dispatch.abiVersion = MR_ROD_GPU_ABI_VERSION;
        dispatch.environmentCount =
            static_cast<std::uint32_t>(environmentCount);
        dispatch.nodeCount =
            static_cast<std::uint32_t>(nodeCount);
        dispatch.edgeCount =
            static_cast<std::uint32_t>(edgeCount);
        dispatch.attachmentCount =
            static_cast<std::uint32_t>(input.attachmentCount);
        dispatch.solverIterations =
            config.step.solverIterations;
        dispatch.stateNodeStride = dispatch.nodeCount;
        dispatch.stateEdgeStride = dispatch.edgeCount;
        dispatch.gravityAndTimestep = {
            static_cast<float>(config.step.gravity[0]),
            static_cast<float>(config.step.gravity[1]),
            static_cast<float>(config.step.gravity[2]),
            static_cast<float>(config.step.timestep),
        };
        dispatch.dampingDerivativeTolerance = {
            static_cast<float>(config.step.linearDamping),
            static_cast<float>(config.step.twistDamping),
            // The FP64 oracle's perturbation is below useful FP32
            // resolution at suture scale. Central differences need an
            // epsilon-aware floor or curvature gradients become quantized.
            static_cast<float>(std::max(
                config.step.derivativeStep,
                3.5e-4
            )),
            static_cast<float>(config.step.constraintTolerance),
        };

        std::vector<float> restLengths;
        std::vector<float> restTwists;
        std::vector<mr_float4> restCurvatures;
        std::vector<float> inverseMasses;
        std::vector<float> inverseRotationalInertias;
        std::vector<float> stretchStiffness;
        std::vector<float> bendStiffness;
        std::vector<float> twistStiffness;
        restLengths.reserve(edgeCount);
        restTwists.reserve(edgeCount);
        inverseMasses.reserve(nodeCount);
        inverseRotationalInertias.reserve(edgeCount);
        stretchStiffness.reserve(edgeCount);
        bendStiffness.reserve(edgeCount - 1u);
        twistStiffness.reserve(edgeCount - 1u);
        for (const double value : model.restLengths) {
            restLengths.push_back(static_cast<float>(value));
        }
        for (const double value : model.restTwists) {
            restTwists.push_back(static_cast<float>(value));
        }
        for (const double value : model.nodeMasses) {
            inverseMasses.push_back(
                static_cast<float>(1.0 / value)
            );
        }
        for (const double value :
             model.edgeRotationalInertias) {
            inverseRotationalInertias.push_back(
                static_cast<float>(1.0 / value)
            );
        }
        for (const double value : model.stretchStiffness) {
            stretchStiffness.push_back(
                static_cast<float>(value)
            );
        }
        for (const double value : model.bendStiffness) {
            bendStiffness.push_back(static_cast<float>(value));
        }
        for (const double value : model.twistStiffness) {
            twistStiffness.push_back(static_cast<float>(value));
        }
        for (std::size_t constraintIndex = 0u;
             constraintIndex + 1u < edgeCount;
             ++constraintIndex) {
            const std::array<Vec3, 3> localPositions{{
                model.restPositions[constraintIndex],
                model.restPositions[constraintIndex + 1u],
                model.restPositions[constraintIndex + 2u],
            }};
            const std::array<double, 2> localTwists{{
                model.restTwists[constraintIndex],
                model.restTwists[constraintIndex + 1u],
            }};
            std::array<double, 2> value{};
            if (!curvature(
                    localPositions,
                    localTwists,
                    value
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::invalidModel,
                    "rod rest curvature is degenerate"
                );
            }
            restCurvatures.push_back({
                static_cast<float>(value[0]),
                static_cast<float>(value[1]),
                0.0f,
                0.0f,
            });
        }

        const std::size_t nodeElements =
            environmentCount * nodeCount;
        const std::size_t edgeElements =
            environmentCount * edgeCount;
        std::vector<mr_float4> positions(nodeElements);
        std::vector<mr_float4> velocities(nodeElements);
        std::vector<float> twists(edgeElements);
        std::vector<float> twistRates(edgeElements);
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            for (std::size_t node = 0u;
                 node < nodeCount;
                 ++node) {
                const Vec3& position =
                    input.states[environment].positions[node];
                const Vec3& velocity =
                    input.states[environment].velocities[node];
                positions[environment * nodeCount + node] = {
                    static_cast<float>(position[0]),
                    static_cast<float>(position[1]),
                    static_cast<float>(position[2]),
                    1.0f,
                };
                velocities[environment * nodeCount + node] = {
                    static_cast<float>(velocity[0]),
                    static_cast<float>(velocity[1]),
                    static_cast<float>(velocity[2]),
                    0.0f,
                };
            }
            for (std::size_t edge = 0u;
                 edge < edgeCount;
                 ++edge) {
                twists[environment * edgeCount + edge] =
                    static_cast<float>(
                        input.states[environment].twists[edge]
                    );
                twistRates[environment * edgeCount + edge] =
                    static_cast<float>(
                        input.states[environment].
                            twistRates[edge]
                    );
            }
        }
        std::vector<MRRodGPUAttachment> attachments;
        attachments.reserve(input.attachments.size());
        for (const DiscreteRodAttachment& attachment :
             input.attachments) {
            MRRodGPUAttachment record{};
            record.targetAndCompliance = {
                static_cast<float>(attachment.targetPosition[0]),
                static_cast<float>(attachment.targetPosition[1]),
                static_cast<float>(attachment.targetPosition[2]),
                static_cast<float>(attachment.compliance),
            };
            record.velocity = {
                static_cast<float>(attachment.targetVelocity[0]),
                static_cast<float>(attachment.targetVelocity[1]),
                static_cast<float>(attachment.targetVelocity[2]),
                0.0f,
            };
            record.nodeIndex = attachment.nodeIndex;
            attachments.push_back(record);
        }

        std::size_t allocatedBytes = sizeof(dispatch);
        if (!appendBytes<float>(restLengths.size(), allocatedBytes) ||
            !appendBytes<float>(restTwists.size(), allocatedBytes) ||
            !appendBytes<mr_float4>(
                restCurvatures.size(),
                allocatedBytes
            ) ||
            !appendBytes<float>(inverseMasses.size(), allocatedBytes) ||
            !appendBytes<float>(
                inverseRotationalInertias.size(),
                allocatedBytes
            ) ||
            !appendBytes<float>(
                stretchStiffness.size(),
                allocatedBytes
            ) ||
            !appendBytes<float>(
                bendStiffness.size(),
                allocatedBytes
            ) ||
            !appendBytes<float>(
                twistStiffness.size(),
                allocatedBytes
            ) ||
            !appendBytes<mr_float4>(
                4u * nodeElements,
                allocatedBytes
            ) ||
            !appendBytes<float>(
                4u * edgeElements,
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUAttachment>(
                attachments.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUStatus>(
                environmentCount,
                allocatedBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::arithmeticOverflow,
                "rod Metal buffer size overflow"
            );
        }
        diagnostics.allocatedBytes = allocatedBytes;

        const std::string metallibPath =
            config.metallibPath.empty()
            ? defaultMetallibPath()
            : config.metallibPath;
        if (metallibPath.empty()) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalDeviceUnavailable,
                "no Metal device is available"
            );
        }
        diagnostics.deviceName = nsString(device.name);
        NSString* path = [NSString
            stringWithUTF8String:metallibPath.c_str()];
        NSError* error = nil;
        id<MTLLibrary> library = path == nil
            ? nil
            : [device
                newLibraryWithURL:[NSURL fileURLWithPath:path]
                            error:&error];
        if (library == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalLibraryFailure,
                "failed to load metallib: " + errorString(error)
            );
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_discrete_elastic_rod_step"];
        if (function == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalLibraryFailure,
                "metallib lacks discrete elastic rod kernel"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
                                           error:&error];
        if (pipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalPipelineFailure,
                "failed to create rod pipeline: " +
                    errorString(error)
            );
        }
        if (pipeline.threadExecutionWidth != 32u ||
            pipeline.maxTotalThreadsPerThreadgroup <
                kThreadgroupSize ||
            pipeline.staticThreadgroupMemoryLength >
                device.maxThreadgroupMemoryLength) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalDeviceUnsupported,
                "device cannot execute the SIMD32 rod cohort"
            );
        }

        id<MTLBuffer> buffers[19] = {};
        const std::vector<MRRodGPUDispatch> dispatchVector{dispatch};
        buffers[0] = inputBuffer(device, dispatchVector);
        buffers[1] = inputBuffer(device, restLengths);
        buffers[2] = inputBuffer(device, restTwists);
        buffers[3] = inputBuffer(device, restCurvatures);
        buffers[4] = inputBuffer(device, inverseMasses);
        buffers[5] = inputBuffer(
            device,
            inverseRotationalInertias
        );
        buffers[6] = inputBuffer(device, stretchStiffness);
        buffers[7] = inputBuffer(device, bendStiffness);
        buffers[8] = inputBuffer(device, twistStiffness);
        buffers[9] = inputBuffer(device, positions);
        buffers[10] = inputBuffer(device, velocities);
        buffers[11] = inputBuffer(device, twists);
        buffers[12] = inputBuffer(device, twistRates);
        buffers[13] = inputBuffer(device, attachments);
        buffers[14] = outputBuffer<mr_float4>(
            device,
            nodeElements
        );
        buffers[15] = outputBuffer<mr_float4>(
            device,
            nodeElements
        );
        buffers[16] = outputBuffer<float>(
            device,
            edgeElements
        );
        buffers[17] = outputBuffer<float>(
            device,
            edgeElements
        );
        buffers[18] = outputBuffer<MRRodGPUStatus>(
            device,
            environmentCount
        );
        for (id<MTLBuffer> buffer : buffers) {
            if (buffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        metalBufferFailure,
                    "failed to allocate rod Metal buffer"
                );
            }
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer =
            [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (queue == nil || commandBuffer == nil ||
            encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalCommandFailure,
                "failed to create rod command graph"
            );
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < 19u; ++index) {
            [encoder setBuffer:buffers[index]
                        offset:0u
                       atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                environmentCount,
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kThreadgroupSize,
                1u,
                1u
            )];
        [encoder endEncoding];
        const auto start = std::chrono::steady_clock::now();
        diagnostics.dispatched = true;
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalCommandFailure,
                "rod command failed: " +
                    errorString(commandBuffer.error)
            );
        }

        std::vector<MRRodGPUStatus> statuses(environmentCount);
        std::memcpy(
            statuses.data(),
            buffers[18].contents,
            statuses.size() * sizeof(MRRodGPUStatus)
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            if (statuses[environment].code !=
                    MR_ROD_GPU_SUCCESS ||
                statuses[environment].environment != environment) {
                diagnostics.firstFailingEnvironment =
                    static_cast<std::uint32_t>(environment);
                diagnostics.firstGPUStatusCode =
                    statuses[environment].code;
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        gpuEnvironmentFailure,
                    "a GPU rod environment failed at iteration " +
                        std::to_string(
                            statuses[environment].iterations
                        ) +
                        " with correction " +
                        std::to_string(
                            statuses[environment].diagnostics.y
                        )
                );
            }
        }

        const auto* outputPosition =
            static_cast<const mr_float4*>(buffers[14].contents);
        const auto* outputVelocity =
            static_cast<const mr_float4*>(buffers[15].contents);
        const auto* outputTwist =
            static_cast<const float*>(buffers[16].contents);
        const auto* outputTwistRate =
            static_cast<const float*>(buffers[17].contents);
        MetalDiscreteElasticRodResult staged;
        staged.states.resize(environmentCount);
        staged.statuses = std::move(statuses);
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            DiscreteElasticRodState& state =
                staged.states[environment];
            state.positions.resize(nodeCount);
            state.velocities.resize(nodeCount);
            state.twists.resize(edgeCount);
            state.twistRates.resize(edgeCount);
            for (std::size_t node = 0u;
                 node < nodeCount;
                 ++node) {
                const mr_float4 position =
                    outputPosition[
                        environment * nodeCount + node
                    ];
                const mr_float4 velocity =
                    outputVelocity[
                        environment * nodeCount + node
                    ];
                state.positions[node] = {
                    position.x,
                    position.y,
                    position.z,
                };
                state.velocities[node] = {
                    velocity.x,
                    velocity.y,
                    velocity.z,
                };
            }
            for (std::size_t edge = 0u;
                 edge < edgeCount;
                 ++edge) {
                state.twists[edge] =
                    outputTwist[
                        environment * edgeCount + edge
                    ];
                state.twistRates[edge] =
                    outputTwistRate[
                        environment * edgeCount + edge
                    ];
            }
            if (!validState(model, state)) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        internalFailure,
                    "GPU rod output is non-finite"
                );
            }
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

const char* metalDiscreteElasticRodHostStatusName(
    const MetalDiscreteElasticRodHostStatus status
) noexcept {
    switch (status) {
    case MetalDiscreteElasticRodHostStatus::success:
        return "success";
    case MetalDiscreteElasticRodHostStatus::invalidModel:
        return "invalid_model";
    case MetalDiscreteElasticRodHostStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalDiscreteElasticRodHostStatus::invalidState:
        return "invalid_state";
    case MetalDiscreteElasticRodHostStatus::invalidAttachment:
        return "invalid_attachment";
    case MetalDiscreteElasticRodHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalDiscreteElasticRodHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalDiscreteElasticRodHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalDiscreteElasticRodHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalDiscreteElasticRodHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalDiscreteElasticRodHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalDiscreteElasticRodHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalDiscreteElasticRodHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalDiscreteElasticRodHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalDiscreteElasticRodHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalDiscreteElasticRodHostStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
