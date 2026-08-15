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

bool finite(const mr_float4 value) {
    return
        finite(value.x) &&
        finite(value.y) &&
        finite(value.z) &&
        finite(value.w);
}

bool validRigidBodyState(const MRBodyStateGPU& state) {
    const double quaternionNormSquared =
        static_cast<double>(state.orientation.x) *
            state.orientation.x +
        static_cast<double>(state.orientation.y) *
            state.orientation.y +
        static_cast<double>(state.orientation.z) *
            state.orientation.z +
        static_cast<double>(state.orientation.w) *
            state.orientation.w;
    return
        finite(state.position) &&
        finite(state.orientation) &&
        finite(state.linearVelocityAndInverseMass) &&
        finite(state.angularVelocity) &&
        finite(state.inverseInertiaWorldRow0) &&
        finite(state.inverseInertiaWorldRow1) &&
        finite(state.inverseInertiaWorldRow2) &&
        std::abs(quaternionNormSquared - 1.0) <= 2.0e-4 &&
        state.linearVelocityAndInverseMass.w >= 0.0f;
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

bool referenceFrames(
    const std::vector<Vec3>& restPositions,
    std::vector<Vec3>& tangents,
    std::vector<Vec3>& directors
) {
    if (restPositions.size() < 2u) {
        return false;
    }
    const std::size_t edgeCount = restPositions.size() - 1u;
    tangents.resize(edgeCount);
    directors.resize(edgeCount);
    for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
        if (!normalize(
                subtract(
                    restPositions[edge + 1u],
                    restPositions[edge]
                ),
                tangents[edge]
            )) {
            return false;
        }
    }
    directors[0] = leastAligned(tangents[0]);
    for (std::size_t edge = 1u; edge < edgeCount; ++edge) {
        if (!transport(
                directors[edge - 1u],
                tangents[edge - 1u],
                tangents[edge],
                directors[edge]
            )) {
            return false;
        }
    }
    return true;
}

bool curvature(
    const std::array<Vec3, 3>& positions,
    const std::array<double, 2>& twists,
    const std::array<Vec3, 2>& referenceTangents,
    const std::array<Vec3, 2>& referenceDirectors,
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
    Vec3 referenceLeft;
    Vec3 referenceRight;
    if (!transport(
            referenceDirectors[0],
            referenceTangents[0],
            left,
            referenceLeft
        ) ||
        !transport(
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
        finite(config.derivativeStep) &&
        config.selfCollisionMargin >= 0.0 &&
        finite(config.selfCollisionMargin) &&
        config.selfCollisionCompliance >= 0.0 &&
        finite(config.selfCollisionCompliance);
}

bool validToolConfig(
    const MetalDiscreteElasticRodToolConfig& config
) {
    return
        config.outerIterations > 0u &&
        config.outerIterations <= 32u &&
        std::isfinite(config.contactOffset) &&
        config.contactOffset >= 0.0f &&
        std::isfinite(config.restOffset) &&
        config.restOffset >= 0.0f &&
        config.restOffset <= config.contactOffset &&
        std::isfinite(config.compliance) &&
        config.compliance >= 0.0f &&
        std::isfinite(config.damping) &&
        config.damping >= 0.0f &&
        std::isfinite(config.restitution) &&
        config.restitution >= 0.0f &&
        config.restitution <= 1.0f &&
        std::isfinite(config.restitutionThreshold) &&
        config.restitutionThreshold >= 0.0f &&
        std::isfinite(config.frictionScale) &&
        config.frictionScale > 0.0f &&
        std::isfinite(config.maximumDepenetrationVelocity) &&
        config.maximumDepenetrationVelocity > 0.0f &&
        config.collisionGroup != 0u &&
        config.collisionMask != 0u;
}

std::uint32_t rodToolPairClass(
    const std::uint32_t toolType
) {
    switch (toolType) {
    case MR_SHAPE_PLANE:
        return MR_COLLISION_PAIR_CAPSULE_PLANE;
    case MR_SHAPE_SPHERE:
        return MR_COLLISION_PAIR_SPHERE_CAPSULE;
    case MR_SHAPE_CAPSULE:
        return MR_COLLISION_PAIR_CAPSULE_CAPSULE;
    case MR_SHAPE_BOX:
        return MR_COLLISION_PAIR_CAPSULE_BOX;
    case MR_SHAPE_CYLINDER:
    case MR_SHAPE_CONVEX:
        return MR_COLLISION_PAIR_CONVEX;
    case MR_SHAPE_TRIANGLE_MESH:
        return MR_COLLISION_PAIR_MESH;
    default:
        return MR_COLLISION_PAIR_UNSUPPORTED;
    }
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
        if (config.tool.enabled &&
            !validToolConfig(config.tool)) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    invalidConfiguration,
                "Metal rod tool-contact configuration is invalid"
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
        if (input.rigidBodyCount == 0u) {
            if (!input.rigidBodies.empty() ||
                !input.rigidBindings.empty()) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        invalidAttachment,
                    "rigid rod bindings require rigid body state"
                );
            }
        } else if (
            input.rigidBodies.size() !=
                environmentCount * input.rigidBodyCount ||
            input.rigidBindings.size() != input.attachmentCount
        ) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    invalidAttachment,
                "rigid bodies or bindings are not packed as declared"
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
        if (input.rigidBodyCount > 0u) {
            if (input.rigidBodyCount >
                std::numeric_limits<mr_u32>::max()) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        capacityOverflow,
                    "rigid body stride exceeds the Metal ABI"
                );
            }
            for (const MRBodyStateGPU& body : input.rigidBodies) {
                if (!validRigidBodyState(body)) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            invalidState,
                        "rigid coupling state is invalid"
                    );
                }
            }
            std::set<std::uint32_t> coupledBodies;
            for (const DiscreteRodRigidAttachmentBinding& binding :
                 input.rigidBindings) {
                if (!finite(binding.localAnchor)) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            invalidAttachment,
                        "rigid attachment anchor is non-finite"
                    );
                }
                if (binding.bodyIndex ==
                    kDiscreteRodNoRigidBody) {
                    continue;
                }
                if (binding.bodyIndex >= input.rigidBodyCount ||
                    !coupledBodies.insert(
                        binding.bodyIndex
                    ).second) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            invalidAttachment,
                        "rigid attachment bodies must be valid and disjoint"
                    );
                }
                for (std::size_t environment = 0u;
                     environment < environmentCount;
                     ++environment) {
                    const MRBodyStateGPU& body =
                        input.rigidBodies[
                            environment *
                                input.rigidBodyCount +
                            binding.bodyIndex
                        ];
                    const std::uint32_t motion =
                        body.flagsAndIndices[0];
                    const bool dynamicTarget =
                        motion == MR_MOTION_DYNAMIC &&
                        body.linearVelocityAndInverseMass.w > 0.0f;
                    const bool kinematicTarget =
                        motion == MR_MOTION_KINEMATIC &&
                        body.linearVelocityAndInverseMass.w == 0.0f;
                    if (!dynamicTarget && !kinematicTarget) {
                        return reject(
                            std::move(diagnostics),
                            MetalDiscreteElasticRodHostStatus::
                                invalidAttachment,
                            "rod attachment target must be dynamic or "
                            "kinematic with consistent inverse mass"
                        );
                    }
                }
            }
        }

        std::vector<MRRodColliderGPU> rodColliders;
        std::vector<MRShapeGPU> rodShapeSources;
        std::vector<MRRodToolPairGPU> toolPairs;
        const EngineModel* toolModel =
            config.tool.enabled ? input.toolModel : nullptr;
        if (toolModel != nullptr) {
            std::string toolReason;
            if (!toolModel->valid(&toolReason) ||
                input.rigidBodyCount !=
                    toolModel->bodies.size() ||
                toolModel->shapes.empty() ||
                config.tool.rodMaterialIndex >=
                    toolModel->materials.size()) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        invalidConfiguration,
                    "thread/tool model or body packing is invalid: " +
                        toolReason
                );
            }
            if (edgeCount >
                    std::numeric_limits<mr_u32>::max() ||
                toolModel->shapes.size() >
                    std::numeric_limits<mr_u32>::max()) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        capacityOverflow,
                    "thread/tool topology exceeds 32-bit GPU indexing"
                );
            }
            rodColliders.reserve(edgeCount);
            rodShapeSources.reserve(edgeCount);
            for (std::uint32_t edge = 0u;
                 edge < edgeCount;
                 ++edge) {
                MRRodColliderGPU collider{};
                collider.rodIndex = 0u;
                collider.edgeIndex = edge;
                collider.nodeA = edge;
                collider.nodeB = edge + 1u;
                collider.materialIndex =
                    config.tool.rodMaterialIndex;
                collider.collisionGroup =
                    config.tool.collisionGroup;
                collider.collisionMask =
                    config.tool.collisionMask;
                collider.topologyGeneration = 1u;
                collider.radiusAndOffsets = {
                    static_cast<float>(model.radius),
                    config.tool.contactOffset,
                    config.tool.restOffset,
                    static_cast<float>(
                        model.radius +
                        0.5 * model.restLengths[edge]
                    ),
                };
                collider.flagsAndExclusions = {
                    static_cast<mr_u32>(
                        MR_ROD_GPU_FLAG_TOOL_COLLISION |
                        (
                            config.tool.enableCCD
                            ? MR_ROD_GPU_FLAG_ENABLE_CCD
                            : 0u
                        )
                    ),
                    0u,
                    0u,
                    0u,
                };
                rodColliders.push_back(collider);

                MRShapeGPU source{};
                source.bodyIndex = 0u;
                source.shapeType = MR_SHAPE_CAPSULE;
                source.materialIndex =
                    config.tool.rodMaterialIndex;
                source.collisionGroup =
                    config.tool.collisionGroup;
                source.collisionMask =
                    config.tool.collisionMask;
                source.slotGeneration = 1u;
                source.localRotation = {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                };
                source.dimensions = {
                    static_cast<float>(model.radius),
                    static_cast<float>(
                        0.5 * model.restLengths[edge]
                    ),
                    0.0f,
                    0.0f,
                };
                source.contactRestAndBoundingRadius = {
                    config.tool.contactOffset,
                    config.tool.restOffset,
                    collider.radiusAndOffsets.w,
                    0.0f,
                };
                rodShapeSources.push_back(source);
            }

            if (!input.toolPairs.empty()) {
                toolPairs.assign(
                    input.toolPairs.begin(),
                    input.toolPairs.end()
                );
                for (std::size_t pairIndex = 0u;
                     pairIndex < toolPairs.size();
                     ++pairIndex) {
                    const MRRodToolPairGPU& pair =
                        toolPairs[pairIndex];
                    if (pair.rodCollider >= edgeCount ||
                        pair.rigidCollider >=
                            toolModel->shapes.size() ||
                        pair.pairClass != rodToolPairClass(
                            toolModel
                                ->shapes[pair.rigidCollider]
                                .shapeType
                        ) ||
                        (pair.flags &
                         ~static_cast<mr_u32>(
                             MR_ROD_TOOL_PAIR_VALID |
                             MR_ROD_TOOL_PAIR_ENABLE_CCD
                         )) != 0u ||
                        (pair.flags &
                         MR_ROD_TOOL_PAIR_VALID) == 0u) {
                        return reject(
                            std::move(diagnostics),
                            MetalDiscreteElasticRodHostStatus::
                                invalidConfiguration,
                            "explicit thread/tool pair is invalid at " +
                                std::to_string(pairIndex)
                        );
                    }
                }
            } else {
                std::set<
                    std::pair<std::uint32_t, std::uint32_t>
                > attachmentExclusions;
                for (std::size_t attachment = 0u;
                     attachment < input.attachmentCount;
                     ++attachment) {
                    const auto& binding =
                        input.rigidBindings[attachment];
                    if (binding.bodyIndex ==
                        kDiscreteRodNoRigidBody) {
                        continue;
                    }
                    const std::uint32_t node =
                        input.attachments[attachment].nodeIndex;
                    for (std::size_t environment = 1u;
                         environment < environmentCount;
                         ++environment) {
                        if (input.attachments[
                                environment *
                                    input.attachmentCount +
                                attachment
                            ].nodeIndex != node) {
                            return reject(
                                std::move(diagnostics),
                                MetalDiscreteElasticRodHostStatus::
                                    invalidAttachment,
                                "tool collision requires homogeneous "
                                "attachment topology"
                            );
                        }
                    }
                    if (node > 0u) {
                        attachmentExclusions.emplace(
                            static_cast<std::uint32_t>(node - 1u),
                            binding.bodyIndex
                        );
                    }
                    if (node < edgeCount) {
                        attachmentExclusions.emplace(
                            node,
                            binding.bodyIndex
                        );
                    }
                }
                for (std::uint32_t edge = 0u;
                     edge < edgeCount;
                     ++edge) {
                    for (std::uint32_t collider = 0u;
                         collider < toolModel->shapes.size();
                         ++collider) {
                        const MRShapeGPU& shape =
                            toolModel->shapes[collider];
                        if ((shape.flags &
                             MR_SHAPE_FLAG_SIMULATION_DISABLED) !=
                                0u ||
                            (config.tool.collisionGroup &
                             shape.collisionMask) == 0u ||
                            (shape.collisionGroup &
                             config.tool.collisionMask) == 0u ||
                            attachmentExclusions.contains({
                                edge,
                                shape.bodyIndex,
                            })) {
                            continue;
                        }
                        const std::uint32_t pairClass =
                            rodToolPairClass(shape.shapeType);
                        if (pairClass ==
                            MR_COLLISION_PAIR_UNSUPPORTED) {
                            return reject(
                                std::move(diagnostics),
                                MetalDiscreteElasticRodHostStatus::
                                    invalidConfiguration,
                                "tool collider has no rod narrowphase"
                            );
                        }
                        toolPairs.push_back({
                            .rodCollider = edge,
                            .rigidCollider = collider,
                            .pairClass = pairClass,
                            .flags =
                                MR_ROD_TOOL_PAIR_VALID |
                                (
                                    config.tool.enableCCD &&
                                        (shape.flags &
                                         MR_SHAPE_FLAG_ENABLE_CCD) !=
                                            0u
                                    ? MR_ROD_TOOL_PAIR_ENABLE_CCD
                                    : 0u
                                ),
                        });
                    }
                }
            }
            const std::uint64_t witnessCount =
                static_cast<std::uint64_t>(
                    environmentCount
                ) *
                toolPairs.size() *
                MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
            if (toolPairs.size() >
                    std::numeric_limits<mr_u32>::max() /
                        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR ||
                witnessCount >
                    std::numeric_limits<std::size_t>::max()) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        capacityOverflow,
                    "thread/tool pair cache exceeds addressable capacity"
                );
            }
            if (!input.previousToolContacts.empty() &&
                input.previousToolContacts.size() !=
                    static_cast<std::size_t>(witnessCount)) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        invalidState,
                    "thread/tool cache is not pair-slot packed"
                );
            }
        } else if (
            config.tool.enabled ||
            !input.toolPairs.empty() ||
            !input.previousToolContacts.empty()
        ) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    invalidConfiguration,
                "tool collision requires an EngineModel"
            );
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
        dispatch.rigidBodyCount =
            static_cast<std::uint32_t>(input.rigidBodyCount);
        dispatch.stateBodyStride = dispatch.rigidBodyCount;
        dispatch.flags = config.step.enableSelfCollision
            ? MR_ROD_GPU_FLAG_SELF_COLLISION
            : 0u;
        dispatch.toolShapeCount = toolModel != nullptr
            ? static_cast<std::uint32_t>(
                  toolModel->shapes.size()
              )
            : 0u;
        dispatch.toolPairCount =
            static_cast<std::uint32_t>(toolPairs.size());
        dispatch.toolContactStride =
            dispatch.toolPairCount *
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
        dispatch.toolContactIterations =
            toolPairs.empty()
            ? 0u
            : config.tool.outerIterations;
        dispatch.rodMaterialIndex =
            config.tool.rodMaterialIndex;
        dispatch.rodNodeBase = 0u;
        dispatch.rodEdgeBase = 0u;
        dispatch.toolPairBase = 0u;
        dispatch.toolPairWorldStride =
            dispatch.toolPairCount;
        if (!toolPairs.empty()) {
            dispatch.flags |=
                MR_ROD_GPU_FLAG_TOOL_COLLISION;
            if (config.tool.enableCCD) {
                dispatch.flags |=
                    MR_ROD_GPU_FLAG_ENABLE_CCD;
            }
            if (config.tool.warmStart &&
                !input.previousToolContacts.empty()) {
                dispatch.flags |=
                    MR_ROD_GPU_FLAG_TOOL_WARM_START;
            }
        }
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
        dispatch.selfCollision = {
            static_cast<float>(model.radius),
            static_cast<float>(
                config.step.selfCollisionMargin
            ),
            static_cast<float>(
                config.step.selfCollisionCompliance
            ),
            0.0f,
        };
        dispatch.toolContact = {
            config.tool.contactOffset,
            config.tool.restOffset,
            config.tool.compliance,
            config.tool.damping,
        };
        dispatch.toolResponse = {
            config.tool.restitution,
            config.tool.restitutionThreshold,
            config.tool.frictionScale,
            config.tool.maximumDepenetrationVelocity,
        };

        std::vector<float> restLengths;
        std::vector<float> restTwists;
        std::vector<mr_float4> restCurvatures;
        std::vector<Vec3> referenceTangentValues;
        std::vector<Vec3> referenceDirectorValues;
        std::vector<mr_float4> referenceTangents;
        std::vector<mr_float4> referenceDirectors;
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
        if (!referenceFrames(
                model.restPositions,
                referenceTangentValues,
                referenceDirectorValues
            )) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::invalidModel,
                "rod rest reference frame is degenerate"
            );
        }
        referenceTangents.reserve(edgeCount);
        referenceDirectors.reserve(edgeCount);
        for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
            referenceTangents.push_back({
                static_cast<float>(referenceTangentValues[edge][0]),
                static_cast<float>(referenceTangentValues[edge][1]),
                static_cast<float>(referenceTangentValues[edge][2]),
                0.0f,
            });
            referenceDirectors.push_back({
                static_cast<float>(referenceDirectorValues[edge][0]),
                static_cast<float>(referenceDirectorValues[edge][1]),
                static_cast<float>(referenceDirectorValues[edge][2]),
                0.0f,
            });
        }
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
            const std::array<Vec3, 2> localReferenceTangents{{
                referenceTangentValues[constraintIndex],
                referenceTangentValues[constraintIndex + 1u],
            }};
            const std::array<Vec3, 2> localReferenceDirectors{{
                referenceDirectorValues[constraintIndex],
                referenceDirectorValues[constraintIndex + 1u],
            }};
            std::array<double, 2> value{};
            if (!curvature(
                    localPositions,
                    localTwists,
                    localReferenceTangents,
                    localReferenceDirectors,
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
        const std::size_t toolPairElements =
            environmentCount * toolPairs.size();
        const std::size_t toolWitnessElements =
            toolPairElements *
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
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
        std::vector<MRRodGPURigidBinding> rigidBindings(
            input.attachmentCount
        );
        for (std::size_t attachmentIndex = 0u;
             attachmentIndex < input.attachmentCount;
             ++attachmentIndex) {
            MRRodGPURigidBinding& record =
                rigidBindings[attachmentIndex];
            record.bodyIndex = MR_ROD_GPU_INVALID_BODY;
            if (!input.rigidBindings.empty()) {
                const DiscreteRodRigidAttachmentBinding& binding =
                    input.rigidBindings[attachmentIndex];
                record.localAnchor = {
                    static_cast<float>(binding.localAnchor[0]),
                    static_cast<float>(binding.localAnchor[1]),
                    static_cast<float>(binding.localAnchor[2]),
                    0.0f,
                };
                record.bodyIndex = binding.bodyIndex;
            }
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
                (toolPairs.empty() ? 4u : 6u) *
                    nodeElements,
                allocatedBytes
            ) ||
            !appendBytes<float>(
                (toolPairs.empty() ? 4u : 6u) *
                    edgeElements,
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUAttachment>(
                attachments.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUAttachment>(
                attachments.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPURigidBinding>(
                rigidBindings.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRBodyStateGPU>(
                (toolPairs.empty() ? 1u : 2u) *
                    input.rigidBodies.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRBodyStateGPU>(
                input.rigidBodies.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUAttachmentReaction>(
                attachments.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodGPUStatus>(
                environmentCount,
                allocatedBytes
            ) ||
            !appendBytes<MRRodColliderGPU>(
                rodColliders.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRShapeGPU>(
                rodShapeSources.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRRodToolPairGPU>(
                toolPairs.size(),
                allocatedBytes
            ) ||
            !appendBytes<MRShapeGPU>(
                toolModel != nullptr
                    ? toolModel->shapes.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<MRMaterialGPU>(
                toolModel != nullptr
                    ? toolModel->materials.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<MRGeometryHeaderGPU>(
                toolModel != nullptr
                    ? toolModel->geometryHeaders.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<mr_float4>(
                toolModel != nullptr
                    ? toolModel->geometryVertices.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<MRMeshBVHNodeGPU>(
                toolModel != nullptr
                    ? toolModel->meshBvhNodes.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<MRMeshTriangleGPU>(
                toolModel != nullptr
                    ? toolModel->meshTriangles.size()
                    : 0u,
                allocatedBytes
            ) ||
            !appendBytes<std::uint32_t>(
                toolPairElements,
                allocatedBytes
            ) ||
            !appendBytes<MRRodToolWitnessGPU>(
                3u * toolWitnessElements,
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
        id<MTLFunction> resolveFunction = [library
            newFunctionWithName:
                @"mr_resolve_rod_rigid_attachments"];
        id<MTLFunction> applyFunction = [library
            newFunctionWithName:
                @"mr_apply_rod_rigid_reactions"];
        id<MTLFunction> toolNarrowphaseFunction =
            toolPairs.empty()
            ? nil
            : [library
                newFunctionWithName:
                    @"mr_rod_tool_narrowphase"];
        id<MTLFunction> toolSolveFunction =
            toolPairs.empty()
            ? nil
            : [library
                newFunctionWithName:
                    @"mr_solve_rod_tool_contacts"];
        if (function == nil ||
            resolveFunction == nil ||
            applyFunction == nil ||
            (!toolPairs.empty() &&
             (
                 toolNarrowphaseFunction == nil ||
                 toolSolveFunction == nil
             ))) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalLibraryFailure,
                "metallib lacks the coupled discrete rod graph"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
                                           error:&error];
        id<MTLComputePipelineState> resolvePipeline = [device
            newComputePipelineStateWithFunction:resolveFunction
                                           error:&error];
        id<MTLComputePipelineState> applyPipeline = [device
            newComputePipelineStateWithFunction:applyFunction
                                           error:&error];
        id<MTLComputePipelineState> toolNarrowphasePipeline =
            toolPairs.empty()
            ? nil
            : [device
                newComputePipelineStateWithFunction:
                    toolNarrowphaseFunction
                error:&error];
        id<MTLComputePipelineState> toolSolvePipeline =
            toolPairs.empty()
            ? nil
            : [device
                newComputePipelineStateWithFunction:
                    toolSolveFunction
                error:&error];
        if (pipeline == nil ||
            resolvePipeline == nil ||
            applyPipeline == nil ||
            (!toolPairs.empty() &&
             (
                 toolNarrowphasePipeline == nil ||
                 toolSolvePipeline == nil
             ))) {
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
        if (!toolPairs.empty() &&
            (
                toolNarrowphasePipeline.threadExecutionWidth !=
                    32u ||
                toolSolvePipeline.threadExecutionWidth != 32u ||
                toolSolvePipeline.maxTotalThreadsPerThreadgroup <
                    kThreadgroupSize
            )) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalDeviceUnsupported,
                "device cannot execute the SIMD32 rod/tool graph"
            );
        }

        id<MTLBuffer> buffers[20] = {};
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
        // The canonical attachment stream is resolved by the preceding
        // projection kernel. Explicit attachments are copied unchanged.
        buffers[13] = outputBuffer<MRRodGPUAttachment>(
            device,
            attachments.size()
        );
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
        buffers[19] =
            outputBuffer<MRRodGPUAttachmentReaction>(
                device,
                attachments.size()
            );
        id<MTLBuffer> inputAttachmentBuffer =
            inputBuffer(device, attachments);
        id<MTLBuffer> rigidBindingBuffer =
            inputBuffer(device, rigidBindings);
        id<MTLBuffer> referenceTangentBuffer =
            inputBuffer(device, referenceTangents);
        id<MTLBuffer> referenceDirectorBuffer =
            inputBuffer(device, referenceDirectors);
        const std::vector<MRBodyStateGPU> rigidBodies{
            input.rigidBodies.begin(),
            input.rigidBodies.end(),
        };
        id<MTLBuffer> inputRigidBodyBuffer =
            inputBuffer(device, rigidBodies);
        id<MTLBuffer> outputRigidBodyBuffer =
            outputBuffer<MRBodyStateGPU>(
                device,
                rigidBodies.size()
            );
        std::vector<MRRodToolWitnessGPU> previousToolContacts(
            toolWitnessElements
        );
        if (!input.previousToolContacts.empty()) {
            std::copy(
                input.previousToolContacts.begin(),
                input.previousToolContacts.end(),
                previousToolContacts.begin()
            );
        }
        // The persistent world compacts active rod/tool pairs on device. The
        // standalone API already receives an explicit small pair list, so
        // author the same ABI as a deterministic dense list per environment.
        constexpr std::uint32_t standaloneRodCount = 1u;
        const std::size_t activeWitnessDomain =
            environmentCount * dispatch.toolContactStride;
        const std::size_t activePairDomain =
            environmentCount * dispatch.toolPairWorldStride;
        const std::size_t activeMetadataStride =
            MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS +
            MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS *
                standaloneRodCount;
        std::vector<std::uint32_t> activeToolPairs(
            activeWitnessDomain + activePairDomain +
                environmentCount * activeMetadataStride,
            0u
        );
        for (std::uint32_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const std::size_t activeBase =
                activeWitnessDomain +
                environment * dispatch.toolPairWorldStride;
            for (std::uint32_t pair = 0u;
                 pair < dispatch.toolPairCount;
                 ++pair) {
                activeToolPairs[activeBase + pair] = pair;
            }
            const std::size_t metadataBase =
                activeWitnessDomain + activePairDomain +
                environment * activeMetadataStride;
            activeToolPairs[metadataBase + 0u] =
                dispatch.toolPairCount;
            activeToolPairs[metadataBase + 1u] =
                (4u * dispatch.toolPairCount + 31u) / 32u;
            activeToolPairs[metadataBase + 2u] = 1u;
            activeToolPairs[metadataBase + 3u] = 1u;
            const std::size_t rodMetadataBase =
                metadataBase + MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS;
            activeToolPairs[rodMetadataBase + 0u] = 0u;
            activeToolPairs[rodMetadataBase + 1u] =
                dispatch.toolPairCount;
            activeToolPairs[rodMetadataBase + 2u] =
                (dispatch.toolPairCount + 31u) / 32u;
            activeToolPairs[rodMetadataBase + 3u] = 1u;
            activeToolPairs[rodMetadataBase + 4u] = 1u;
        }
        const std::vector<MRShapeGPU> emptyShapes;
        const std::vector<MRMaterialGPU> emptyMaterials;
        const std::vector<MRGeometryHeaderGPU> emptyGeometryHeaders;
        const std::vector<mr_float4> emptyGeometryVertices;
        const std::vector<MRMeshBVHNodeGPU> emptyMeshNodes;
        const std::vector<MRMeshTriangleGPU> emptyMeshTriangles;
        id<MTLBuffer> rodColliderBuffer =
            inputBuffer(device, rodColliders);
        id<MTLBuffer> rodShapeSourceBuffer =
            inputBuffer(device, rodShapeSources);
        id<MTLBuffer> toolPairBuffer =
            inputBuffer(device, toolPairs);
        id<MTLBuffer> toolShapeBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->shapes
                : emptyShapes
        );
        id<MTLBuffer> toolMaterialBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->materials
                : emptyMaterials
        );
        id<MTLBuffer> toolGeometryHeaderBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->geometryHeaders
                : emptyGeometryHeaders
        );
        id<MTLBuffer> toolGeometryVertexBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->geometryVertices
                : emptyGeometryVertices
        );
        id<MTLBuffer> toolMeshNodeBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->meshBvhNodes
                : emptyMeshNodes
        );
        id<MTLBuffer> toolMeshTriangleBuffer = inputBuffer(
            device,
            toolModel != nullptr
                ? toolModel->meshTriangles
                : emptyMeshTriangles
        );
        id<MTLBuffer> previousToolContactBuffer =
            inputBuffer(device, previousToolContacts);
        id<MTLBuffer> toolPairCountBuffer =
            outputBuffer<std::uint32_t>(
                device,
                toolPairElements
            );
        id<MTLBuffer> toolWitnessBuffer =
            outputBuffer<MRRodToolWitnessGPU>(
                device,
                toolWitnessElements
            );
        id<MTLBuffer> activeToolPairBuffer =
            inputBuffer(device, activeToolPairs);
        id<MTLBuffer> finalToolWitnessBuffer =
            outputBuffer<MRRodToolWitnessGPU>(
                device,
                toolWitnessElements
            );
        id<MTLBuffer> finalToolPositionBuffer =
            outputBuffer<mr_float4>(device, nodeElements);
        id<MTLBuffer> finalToolVelocityBuffer =
            outputBuffer<mr_float4>(device, nodeElements);
        id<MTLBuffer> finalToolTwistBuffer =
            outputBuffer<float>(device, edgeElements);
        id<MTLBuffer> finalToolTwistRateBuffer =
            outputBuffer<float>(device, edgeElements);
        id<MTLBuffer> finalToolBodyBuffer =
            outputBuffer<MRBodyStateGPU>(
                device,
                rigidBodies.size()
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
        if (inputAttachmentBuffer == nil ||
            rigidBindingBuffer == nil ||
            referenceTangentBuffer == nil ||
            referenceDirectorBuffer == nil ||
            inputRigidBodyBuffer == nil ||
            outputRigidBodyBuffer == nil ||
            rodColliderBuffer == nil ||
            rodShapeSourceBuffer == nil ||
            toolPairBuffer == nil ||
            toolShapeBuffer == nil ||
            toolMaterialBuffer == nil ||
            toolGeometryHeaderBuffer == nil ||
            toolGeometryVertexBuffer == nil ||
            toolMeshNodeBuffer == nil ||
            toolMeshTriangleBuffer == nil ||
            previousToolContactBuffer == nil ||
            toolPairCountBuffer == nil ||
            toolWitnessBuffer == nil ||
            activeToolPairBuffer == nil ||
            finalToolWitnessBuffer == nil ||
            finalToolPositionBuffer == nil ||
            finalToolVelocityBuffer == nil ||
            finalToolTwistBuffer == nil ||
            finalToolTwistRateBuffer == nil ||
            finalToolBodyBuffer == nil) {
            return reject(
                std::move(diagnostics),
                MetalDiscreteElasticRodHostStatus::
                    metalBufferFailure,
                "failed to allocate rigid rod coupling buffer"
            );
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
        if (!attachments.empty()) {
            [encoder setComputePipelineState:resolvePipeline];
            [encoder setBuffer:buffers[0]
                        offset:0u
                       atIndex:0u];
            [encoder setBuffer:inputAttachmentBuffer
                        offset:0u
                       atIndex:1u];
            [encoder setBuffer:rigidBindingBuffer
                        offset:0u
                       atIndex:2u];
            [encoder setBuffer:inputRigidBodyBuffer
                        offset:0u
                       atIndex:3u];
            [encoder setBuffer:buffers[13]
                        offset:0u
                       atIndex:4u];
            const NSUInteger resolveWidth = std::min<NSUInteger>(
                256u,
                resolvePipeline.maxTotalThreadsPerThreadgroup
            );
            [encoder
                dispatchThreads:MTLSizeMake(
                    attachments.size(),
                    1u,
                    1u
                )
                threadsPerThreadgroup:MTLSizeMake(
                    resolveWidth,
                    1u,
                    1u
                )];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < 20u; ++index) {
            [encoder setBuffer:buffers[index]
                        offset:0u
                       atIndex:index];
        }
        // The standalone rod solver runs one complete fixed microstep. Buffer
        // zero is a valid inert binding for the event cursor because the
        // FULL_MICROSTEP mode guarantees that the shader never dereferences it.
        const mr_u32 eventSegmentMode =
            MR_CCD_SEGMENT_FULL_MICROSTEP;
        [encoder setBuffer:buffers[0]
                    offset:0u
                   atIndex:20u];
        [encoder setBytes:&eventSegmentMode
                   length:sizeof(eventSegmentMode)
                  atIndex:21u];
        [encoder setBuffer:referenceTangentBuffer
                    offset:0u
                   atIndex:22u];
        [encoder setBuffer:referenceDirectorBuffer
                    offset:0u
                   atIndex:23u];
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
        if (input.rigidBodyCount > 0u) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:applyPipeline];
            [encoder setBuffer:buffers[0]
                        offset:0u
                       atIndex:0u];
            [encoder setBuffer:rigidBindingBuffer
                        offset:0u
                       atIndex:1u];
            [encoder setBuffer:inputRigidBodyBuffer
                        offset:0u
                       atIndex:2u];
            [encoder setBuffer:buffers[19]
                        offset:0u
                       atIndex:3u];
            [encoder setBuffer:buffers[18]
                        offset:0u
                       atIndex:4u];
            [encoder setBuffer:outputRigidBodyBuffer
                        offset:0u
                       atIndex:5u];
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
        }
        if (!toolPairs.empty()) {
            [encoder
                memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder
                setComputePipelineState:
                    toolNarrowphasePipeline];
            [encoder setBuffer:buffers[0]
                        offset:0u
                       atIndex:0u];
            [encoder setBuffer:rodColliderBuffer
                        offset:0u
                       atIndex:1u];
            [encoder setBuffer:rodShapeSourceBuffer
                        offset:0u
                       atIndex:2u];
            [encoder setBuffer:toolPairBuffer
                        offset:0u
                       atIndex:3u];
            [encoder setBuffer:toolShapeBuffer
                        offset:0u
                       atIndex:4u];
            [encoder setBuffer:outputRigidBodyBuffer
                        offset:0u
                       atIndex:5u];
            [encoder setBuffer:toolGeometryHeaderBuffer
                        offset:0u
                       atIndex:6u];
            [encoder setBuffer:toolGeometryVertexBuffer
                        offset:0u
                       atIndex:7u];
            [encoder setBuffer:toolMeshNodeBuffer
                        offset:0u
                       atIndex:8u];
            [encoder setBuffer:toolMeshTriangleBuffer
                        offset:0u
                       atIndex:9u];
            [encoder setBuffer:buffers[14]
                        offset:0u
                       atIndex:10u];
            [encoder setBuffer:buffers[15]
                        offset:0u
                       atIndex:11u];
            [encoder setBuffer:buffers[17]
                        offset:0u
                       atIndex:12u];
            [encoder setBuffer:previousToolContactBuffer
                        offset:0u
                       atIndex:13u];
            [encoder setBuffer:toolPairCountBuffer
                        offset:0u
                       atIndex:14u];
            [encoder setBuffer:toolWitnessBuffer
                        offset:0u
                       atIndex:15u];
            [encoder setBuffer:buffers[18]
                        offset:0u
                       atIndex:16u];
            [encoder setBuffer:activeToolPairBuffer
                        offset:0u
                       atIndex:17u];
            const NSUInteger narrowphaseWidth =
                std::min<NSUInteger>(
                    256u,
                    toolNarrowphasePipeline
                        .maxTotalThreadsPerThreadgroup
                );
            const std::uint32_t standaloneRodIndex = 0u;
            [encoder setBytes:&standaloneRodIndex
                       length:sizeof(standaloneRodIndex)
                      atIndex:18u];
            [encoder setBytes:&standaloneRodCount
                       length:sizeof(standaloneRodCount)
                      atIndex:19u];
            for (std::uint32_t environment = 0u;
                 environment < environmentCount;
                 ++environment) {
                [encoder setBytes:&environment
                           length:sizeof(environment)
                          atIndex:20u];
                [encoder
                    dispatchThreads:MTLSizeMake(
                        dispatch.toolPairCount,
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        narrowphaseWidth,
                        1u,
                        1u
                    )];
            }
            [encoder
                memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:toolSolvePipeline];
            [encoder setBuffer:buffers[0]
                        offset:0u
                       atIndex:0u];
            [encoder setBuffer:buffers[4]
                        offset:0u
                       atIndex:1u];
            [encoder setBuffer:buffers[5]
                        offset:0u
                       atIndex:2u];
            [encoder setBuffer:rodColliderBuffer
                        offset:0u
                       atIndex:3u];
            [encoder setBuffer:toolPairBuffer
                        offset:0u
                       atIndex:4u];
            [encoder setBuffer:toolShapeBuffer
                        offset:0u
                       atIndex:5u];
            [encoder setBuffer:toolMaterialBuffer
                        offset:0u
                       atIndex:6u];
            [encoder setBuffer:toolPairCountBuffer
                        offset:0u
                       atIndex:7u];
            [encoder setBuffer:toolWitnessBuffer
                        offset:0u
                       atIndex:8u];
            [encoder setBuffer:buffers[14]
                        offset:0u
                       atIndex:9u];
            [encoder setBuffer:buffers[15]
                        offset:0u
                       atIndex:10u];
            [encoder setBuffer:buffers[16]
                        offset:0u
                       atIndex:11u];
            [encoder setBuffer:buffers[17]
                        offset:0u
                       atIndex:12u];
            [encoder setBuffer:outputRigidBodyBuffer
                        offset:0u
                       atIndex:13u];
            [encoder setBuffer:buffers[18]
                        offset:0u
                       atIndex:14u];
            [encoder setBuffer:finalToolWitnessBuffer
                        offset:0u
                       atIndex:15u];
            [encoder setBuffer:finalToolPositionBuffer
                        offset:0u
                       atIndex:16u];
            [encoder setBuffer:finalToolVelocityBuffer
                        offset:0u
                       atIndex:17u];
            [encoder setBuffer:finalToolTwistBuffer
                        offset:0u
                       atIndex:18u];
            [encoder setBuffer:finalToolTwistRateBuffer
                        offset:0u
                       atIndex:19u];
            [encoder setBuffer:finalToolBodyBuffer
                        offset:0u
                       atIndex:20u];
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
        }
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
            static_cast<const mr_float4*>(
                (
                    toolPairs.empty()
                    ? buffers[14]
                    : finalToolPositionBuffer
                ).contents
            );
        const auto* outputVelocity =
            static_cast<const mr_float4*>(
                (
                    toolPairs.empty()
                    ? buffers[15]
                    : finalToolVelocityBuffer
                ).contents
            );
        const auto* outputTwist =
            static_cast<const float*>(
                (
                    toolPairs.empty()
                    ? buffers[16]
                    : finalToolTwistBuffer
                ).contents
            );
        const auto* outputTwistRate =
            static_cast<const float*>(
                (
                    toolPairs.empty()
                    ? buffers[17]
                    : finalToolTwistRateBuffer
                ).contents
            );
        const auto* outputReaction =
            static_cast<const MRRodGPUAttachmentReaction*>(
                buffers[19].contents
            );
        const auto* outputRigidBody =
            static_cast<const MRBodyStateGPU*>(
                (
                    toolPairs.empty()
                    ? outputRigidBodyBuffer
                    : finalToolBodyBuffer
                ).contents
            );
        MetalDiscreteElasticRodResult staged;
        staged.states.resize(environmentCount);
        staged.statuses = std::move(statuses);
        staged.reactions.resize(attachments.size());
        staged.rigidBodies.resize(rigidBodies.size());
        staged.toolContactCounts.resize(toolPairElements);
        staged.toolContacts.resize(toolWitnessElements);
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
        for (std::size_t reactionIndex = 0u;
             reactionIndex < attachments.size();
             ++reactionIndex) {
            const MRRodGPUAttachmentReaction value =
                outputReaction[reactionIndex];
            DiscreteRodAttachmentReaction& reaction =
                staged.reactions[reactionIndex];
            reaction.nodeIndex = value.nodeIndex;
            reaction.bodyIndex = value.bodyIndex;
            reaction.impulseOnTarget = {
                value.impulseAndError.x,
                value.impulseAndError.y,
                value.impulseAndError.z,
            };
            reaction.averageForceOnTarget = {
                value.averageForce.x,
                value.averageForce.y,
                value.averageForce.z,
            };
            reaction.finalPositionError =
                value.impulseAndError.w;
            if (!finite(reaction.impulseOnTarget) ||
                !finite(reaction.averageForceOnTarget) ||
                !finite(reaction.finalPositionError)) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        internalFailure,
                    "GPU rod reaction evidence is non-finite"
                );
            }
        }
        if (!rigidBodies.empty()) {
            std::copy_n(
                outputRigidBody,
                rigidBodies.size(),
                staged.rigidBodies.begin()
            );
            if (!std::ranges::all_of(
                    staged.rigidBodies,
                    validRigidBodyState
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalDiscreteElasticRodHostStatus::
                        internalFailure,
                    "GPU rigid reaction output is non-finite"
                );
            }
        }
        if (!toolPairs.empty()) {
            std::copy_n(
                static_cast<const std::uint32_t*>(
                    toolPairCountBuffer.contents
                ),
                toolPairElements,
                staged.toolContactCounts.begin()
            );
            std::copy_n(
                static_cast<const MRRodToolWitnessGPU*>(
                    finalToolWitnessBuffer.contents
                ),
                toolWitnessElements,
                staged.toolContacts.begin()
            );
            diagnostics.toolPairCount =
                static_cast<std::uint32_t>(toolPairs.size());
            std::uint64_t totalToolContacts = 0u;
            for (const std::uint32_t count :
                 staged.toolContactCounts) {
                if (count >
                    MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            internalFailure,
                        "GPU rod/tool count exceeded its pair slot"
                    );
                }
                totalToolContacts += count;
            }
            diagnostics.toolContactCount =
                static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(
                        totalToolContacts,
                        std::numeric_limits<std::uint32_t>::max()
                    )
                );
            for (const MRRodToolWitnessGPU& witness :
                 staged.toolContacts) {
                if ((witness.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_VALID) == 0u) {
                    continue;
                }
                if (!finite(witness.rodPointAndWeight) ||
                    !finite(
                        witness.toolPointAndSeparation
                    ) ||
                    !finite(
                        witness.normalAndPreSolveVelocity
                    ) ||
                    !finite(
                        witness
                            .tangentUAndTwistJacobian
                    ) ||
                    !finite(
                        witness
                            .radialAndTwistJacobianV
                    ) ||
                    !finite(witness.impulses)) {
                    return reject(
                        std::move(diagnostics),
                        MetalDiscreteElasticRodHostStatus::
                            internalFailure,
                        "GPU rod/tool evidence is non-finite"
                    );
                }
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
