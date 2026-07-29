#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalMultiArticulatedContact.hpp"
#include "metalrobo/ParallelABASchedule.hpp"

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
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kWaveWidth = 32u;
const char kMultiContactImageAnchor = 0;

struct PointOccurrence {
    std::size_t slot = 0u;
    bool second = false;
    bool equality = false;
    bool angular = false;
    std::uint32_t basis = 0u;
};

struct PointArenaSlice {
    std::size_t bodyPoseOffset = 0u;
    std::size_t pointWorldOffset = 0u;
    std::size_t generalizedOffset = 0u;
    std::size_t statusOffset = 0u;
};

struct Prepared {
    MetalMultiArticulatedContactLayout layout;
    const ParallelABASchedule* schedule = nullptr;
    std::vector<MRMultiContactGPU> contacts;
    std::vector<MRMultiContactEndpointsGPU> endpoints;
    std::vector<MRMultiContactGPU> pointEqualities;
    std::vector<MRMultiContactEndpointsGPU>
        pointEqualityEndpoints;
    std::vector<std::uint32_t> equalityKinds;
    std::vector<ConstraintIRRow> equalityRows;
    std::vector<std::uint32_t> sceneVelocityOffsets;
    std::vector<MRArticulatedPointImpulseGPU> pointQueries;
    std::vector<PointArenaSlice> pointArenas;
    std::size_t bodyPoseElements = 0u;
    std::size_t pointWorldElements = 0u;
    std::size_t pointGeneralizedElements = 0u;
    std::span<const float> equalityJacobian{};
};

MetalMultiArticulatedContactDiagnostics reject(
    MetalMultiArticulatedContactDiagnostics diagnostics,
    const MetalMultiArticulatedContactStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const float value) {
    return std::isfinite(value);
}

template <std::size_t Size>
bool finite(const std::array<double, Size>& values) {
    return std::ranges::all_of(
        values,
        [](const double value) {
            return finite(value);
        }
    );
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

double dot(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

std::array<double, 3> cross(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

std::array<double, 3> rotate(
    const mr_float4 quaternion,
    const std::array<double, 3>& value
) {
    const std::array<double, 3> vectorPart{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    const std::array<double, 3> first =
        cross(vectorPart, value);
    const std::array<double, 3> doubled{
        2.0 * first[0],
        2.0 * first[1],
        2.0 * first[2],
    };
    const std::array<double, 3> second =
        cross(vectorPart, doubled);
    return {
        value[0] + quaternion.w * doubled[0] + second[0],
        value[1] + quaternion.w * doubled[1] + second[1],
        value[2] + quaternion.w * doubled[2] + second[2],
    };
}

double norm(const std::array<double, 3>& value) {
    return std::sqrt(dot(value, value));
}

bool validFrame(
    const MultiArticulatedIslandContact& contact
) {
    if (!finite(contact.normal) ||
        !finite(contact.tangentU) ||
        !finite(contact.tangentV)) {
        return false;
    }
    return
        std::abs(norm(contact.normal) - 1.0) <= 2.0e-4 &&
        std::abs(norm(contact.tangentU) - 1.0) <= 2.0e-4 &&
        std::abs(norm(contact.tangentV) - 1.0) <= 2.0e-4 &&
        std::abs(dot(contact.normal, contact.tangentU)) <=
            4.0e-4 &&
        std::abs(dot(contact.normal, contact.tangentV)) <=
            4.0e-4 &&
        std::abs(dot(contact.tangentU, contact.tangentV)) <=
            4.0e-4 &&
        std::abs(
            dot(
                cross(contact.normal, contact.tangentU),
                contact.tangentV
            ) - 1.0
        ) <= 6.0e-4;
}

bool sameTopology(
    const MultiContactEndpoint& left,
    const MultiContactEndpoint& right
) {
    return left.kind == right.kind &&
        left.body == right.body;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& output
) {
    if (left != 0u &&
        right >
            std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& output
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

template <typename T>
bool addBytes(
    const std::size_t elements,
    std::size_t& bytes
) {
    std::size_t product = 0u;
    return checkedMultiply(
            std::max<std::size_t>(elements, 1u),
            sizeof(T),
            product
        ) &&
        checkedAdd(bytes, product, bytes) &&
        product <= std::numeric_limits<NSUInteger>::max();
}

std::string string(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string{value.UTF8String}
        : std::string{};
}

std::string errorString(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string result =
        string(error.localizedDescription);
    return result.empty()
        ? string(error.description)
        : result;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMultiContactImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            directory / "metalrobo/MetalRobo.metallib",
            directory.parent_path() /
                "shaders/MetalRobo.metallib",
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
    const T* data,
    const std::size_t elements
) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(elements, 1u) * sizeof(T)
    );
    if (elements == 0u) {
        return [device
            newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
    }
    return [device
        newBufferWithBytes:data
                   length:bytes
                  options:MTLResourceStorageModeShared];
}

template <typename T>
id<MTLBuffer> outputBuffer(
    id<MTLDevice> device,
    const std::size_t elements
) {
    return [device
        newBufferWithLength:static_cast<NSUInteger>(
            std::max<std::size_t>(elements, 1u) * sizeof(T)
        )
                    options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name,
    NSError** error
) {
    id<MTLFunction> function =
        [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    return [device
        newComputePipelineStateWithFunction:function
                                       error:error];
}

mr_float4 f4(
    const std::array<double, 3>& value,
    const float w = 0.0f
) {
    return {
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2]),
        w,
    };
}

bool validConfiguration(
    const MetalMultiArticulatedContactConfig& config
) {
    const auto& quality = config.quality;
    return
        quality.maximumNewtonIterations > 0u &&
        quality.maximumCGIterations > 0u &&
        quality.maximumLineSearchIterations > 0u &&
        finite(quality.convergenceTolerance) &&
        quality.convergenceTolerance > 0.0f &&
        finite(quality.armijoCoefficient) &&
        quality.armijoCoefficient > 0.0f &&
        quality.armijoCoefficient < 0.5f &&
        finite(quality.normalEquationRegularization) &&
        quality.normalEquationRegularization > 0.0f &&
        finite(quality.minimumCGDenominator) &&
        quality.minimumCGDenominator > 0.0f &&
        finite(config.delassusSymmetryTolerance) &&
        config.delassusSymmetryTolerance > 0.0f &&
        finite(config.delassusDiagonalTolerance) &&
        config.delassusDiagonalTolerance > 0.0f &&
        finite(config.equalityEvaluation.timestep) &&
        config.equalityEvaluation.timestep > 0.0 &&
        finite(
            config.equalityEvaluation
                .maximumDepenetrationVelocity
        ) &&
        config.equalityEvaluation
                .maximumDepenetrationVelocity >= 0.0 &&
        finite(
            config.equalityEvaluation.minimumTimeConstantRatio
        ) &&
        config.equalityEvaluation
                .minimumTimeConstantRatio >= 0.0 &&
        finite(config.equalityEvaluation.minimumRegularization) &&
        config.equalityEvaluation.minimumRegularization >= 0.0 &&
        finite(config.equalityPivotTolerance) &&
        config.equalityPivotTolerance > 0.0f &&
        finite(config.equalityResidualTolerance) &&
        config.equalityResidualTolerance > 0.0f;
}

bool validSceneBody(
    const MRBodyStateGPU& state,
    const std::uint32_t expectedMotion
) {
    const std::uint32_t motion = state.flagsAndIndices[0];
    if (motion != expectedMotion ||
        (motion != MR_MOTION_DYNAMIC &&
         motion != MR_MOTION_KINEMATIC &&
         motion != MR_MOTION_STATIC) ||
        state.flagsAndIndices[1] != MR_INVALID_INDEX ||
        !finite(state.position) ||
        !finite(state.orientation) ||
        !finite(state.linearVelocityAndInverseMass) ||
        !finite(state.angularVelocity) ||
        !finite(state.inverseInertiaWorldRow0) ||
        !finite(state.inverseInertiaWorldRow1) ||
        !finite(state.inverseInertiaWorldRow2)) {
        return false;
    }
    const double qn = std::sqrt(
        double(state.orientation.x) * state.orientation.x +
        double(state.orientation.y) * state.orientation.y +
        double(state.orientation.z) * state.orientation.z +
        double(state.orientation.w) * state.orientation.w
    );
    if (!finite(qn) || std::abs(qn - 1.0) > 2.0e-4) {
        return false;
    }
    if (motion != MR_MOTION_DYNAMIC) {
        return true;
    }
    if (!(state.linearVelocityAndInverseMass.w > 0.0f)) {
        return false;
    }
    const double a00 = state.inverseInertiaWorldRow0.x;
    const double a01 = 0.5 * (
        state.inverseInertiaWorldRow0.y +
        state.inverseInertiaWorldRow1.x
    );
    const double a02 = 0.5 * (
        state.inverseInertiaWorldRow0.z +
        state.inverseInertiaWorldRow2.x
    );
    const double a11 = state.inverseInertiaWorldRow1.y;
    const double a12 = 0.5 * (
        state.inverseInertiaWorldRow1.z +
        state.inverseInertiaWorldRow2.y
    );
    const double a22 = state.inverseInertiaWorldRow2.z;
    const double determinant =
        a00 * (a11 * a22 - a12 * a12) -
        a01 * (a01 * a22 - a12 * a02) +
        a02 * (a01 * a12 - a11 * a02);
    return a00 > 0.0 &&
        a00 * a11 - a01 * a01 > 0.0 &&
        determinant > 0.0 && finite(determinant);
}

MetalMultiArticulatedContactDiagnostics prepare(
    const EngineModel& model,
    const ParallelABASchedule& schedule,
    const std::span<const float> equalityJacobian,
    const MetalMultiArticulatedContactInput& input,
    const MetalMultiArticulatedContactConfig& config,
    Prepared& output
) {
    MetalMultiArticulatedContactDiagnostics diagnostics;
    std::string reason;
    if (!model.valid(&reason)) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::invalidModel,
            "invalid EngineModel: " + reason
        );
    }
    if (!validConfiguration(config)) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                invalidConfiguration,
            "multi-articulation contact configuration is invalid"
        );
    }
    if (input.environmentCount == 0u ||
        input.contactCount == 0u ||
        input.contactCount >
            MR_MULTI_CONTACT_MAX_CONTACTS ||
        input.pointEqualityCount >
            MR_MULTI_CONTACT_MAX_EQUALITY_ROWS / 3u ||
        input.angularEqualityCount >
            MR_MULTI_CONTACT_MAX_EQUALITY_ROWS / 3u ||
        input.pointEqualityCount >
            std::numeric_limits<std::size_t>::max() -
                input.angularEqualityCount ||
        input.pointEqualityCount +
                input.angularEqualityCount >
            MR_MULTI_CONTACT_MAX_EQUALITY_ROWS / 3u ||
        input.environmentCount >
            std::numeric_limits<std::uint32_t>::max() ||
        input.sceneBodyCount >
            std::numeric_limits<std::uint32_t>::max() ||
        model.articulations.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            std::move(diagnostics),
            input.contactCount >
                MR_MULTI_CONTACT_MAX_CONTACTS
                ? MetalMultiArticulatedContactStatus::
                      capacityOverflow
                : MetalMultiArticulatedContactStatus::
                      invalidDimensions,
            "batch dimensions exceed the contact ABI"
        );
    }

    std::size_t qElements = 0u;
    std::size_t articulatedVelocityElements = 0u;
    std::size_t sceneElements = 0u;
    std::size_t contactElements = 0u;
    std::size_t pointEqualityElements = 0u;
    std::size_t angularEqualityElements = 0u;
    if (!checkedMultiply(
            input.environmentCount,
            model.world.nq,
            qElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            model.world.nv,
            articulatedVelocityElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            input.sceneBodyCount,
            sceneElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            input.contactCount,
            contactElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            input.pointEqualityCount,
            pointEqualityElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            input.angularEqualityCount,
            angularEqualityElements
        ) ||
        input.q.size() != qElements ||
        input.freeArticulationVelocity.size() !=
            articulatedVelocityElements ||
        input.sceneBodies.size() != sceneElements ||
        input.contacts.size() != contactElements ||
        input.pointEqualities.size() !=
            pointEqualityElements ||
        input.angularEqualities.size() !=
            angularEqualityElements) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                invalidDimensions,
            "input spans do not match environment-major dimensions"
        );
    }
    if (!std::ranges::all_of(
            input.q,
            [](const float value) {
                return finite(value);
            }
        ) ||
        !std::ranges::all_of(
            input.freeArticulationVelocity,
            [](const float value) {
                return finite(value);
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::nonfiniteInput,
            "q or articulated velocity contains non-finite values"
        );
    }

    Prepared staged;
    staged.schedule = &schedule;
    staged.equalityJacobian = equalityJacobian;
    const std::size_t staticEqualityRows =
        model.constraintProgram.rows.size();
    const std::size_t equalityRows =
        staticEqualityRows +
        3u * (
            input.pointEqualityCount +
            input.angularEqualityCount
        );
    if (equalityRows > MR_MULTI_CONTACT_MAX_EQUALITY_ROWS ||
        equalityJacobian.size() !=
            staticEqualityRows * model.world.nv) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                unsupportedTopology,
            "compiled generalized equality layout is invalid"
        );
    }
    staged.sceneVelocityOffsets.assign(
        input.sceneBodyCount,
        MR_INVALID_INDEX
    );
    std::size_t totalNv = model.world.nv;
    for (std::size_t body = 0u;
         body < input.sceneBodyCount;
         ++body) {
        const std::uint32_t motion =
            input.sceneBodies[body].flagsAndIndices[0];
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            if (!validSceneBody(
                    input.sceneBodies[
                        environment * input.sceneBodyCount +
                        body
                    ],
                    motion
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        nonfiniteInput,
                    "scene-body state or motion class is invalid"
                );
            }
        }
        if (motion == MR_MOTION_DYNAMIC) {
            if (totalNv >
                std::numeric_limits<std::uint32_t>::max() -
                    6u) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        arithmeticOverflow,
                    "packed scene velocity exceeds the GPU ABI"
                );
            }
            staged.sceneVelocityOffsets[body] =
                static_cast<std::uint32_t>(totalNv);
            totalNv += 6u;
        }
    }

    std::vector<std::vector<PointOccurrence>> occurrences(
        model.articulations.size()
    );
    staged.endpoints.resize(input.contactCount);
    const auto compileEndpoint = [&](
        const MultiContactEndpoint& endpoint,
        const std::size_t slot,
        const bool second,
        const bool equality,
        const bool angular,
        MRMultiContactEndpointsGPU& gpu
    ) -> bool {
        const std::uint32_t kind =
            static_cast<std::uint32_t>(endpoint.kind);
        std::uint32_t body = endpoint.body;
        std::uint32_t slice = MR_INVALID_INDEX;
        std::uint32_t query = MR_INVALID_INDEX;
        if (endpoint.kind ==
            MultiContactEndpointKind::articulatedBody) {
            if (body >= model.bodies.size()) {
                return false;
            }
            const std::uint32_t articulation =
                model.bodies[body].articulationIndex;
            if (articulation == MR_INVALID_INDEX ||
                articulation >= model.articulations.size()) {
                return false;
            }
            slice = articulation;
            query = static_cast<std::uint32_t>(
                occurrences[articulation].size()
            );
            const std::uint32_t queryCount =
                angular ? 4u : 1u;
            for (std::uint32_t basis = 0u;
                 basis < queryCount;
                 ++basis) {
                occurrences[articulation].push_back({
                    slot,
                    second,
                    equality,
                    angular,
                    basis,
                });
            }
        } else if (endpoint.kind ==
            MultiContactEndpointKind::sceneBody) {
            if (body >= input.sceneBodyCount) {
                return false;
            }
        } else if (endpoint.kind ==
            MultiContactEndpointKind::staticWorld) {
            body = MR_INVALID_INDEX;
        } else {
            return false;
        }
        if (second) {
            gpu.kindB = kind;
            gpu.bodyB = body;
            gpu.sliceB = slice;
            gpu.queryB = query;
        } else {
            gpu.kindA = kind;
            gpu.bodyA = body;
            gpu.sliceA = slice;
            gpu.queryA = query;
        }
        return true;
    };
    const auto endpointResponds = [&](
        const MultiContactEndpoint& endpoint
    ) {
        if (endpoint.kind ==
            MultiContactEndpointKind::articulatedBody) {
            return true;
        }
        return endpoint.kind ==
                MultiContactEndpointKind::sceneBody &&
            endpoint.body < staged.sceneVelocityOffsets.size() &&
            staged.sceneVelocityOffsets[endpoint.body] !=
                MR_INVALID_INDEX;
    };
    for (std::size_t contact = 0u;
         contact < input.contactCount;
         ++contact) {
        const auto& topology = input.contacts[contact];
        if (sameTopology(
                topology.endpointA,
                topology.endpointB
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "a contact cannot use the same body at both endpoints"
            );
        }
        if (!endpointResponds(topology.endpointA) &&
            !endpointResponds(topology.endpointB)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "a contact requires at least one dynamic endpoint"
            );
        }
        const MultiArticulatedIslandContact& source =
            input.contacts[contact];
        if (!compileEndpoint(
                source.endpointA,
                contact,
                false,
                false,
                false,
                staged.endpoints[contact]
            ) ||
            !compileEndpoint(
                source.endpointB,
                contact,
                true,
                false,
                false,
                staged.endpoints[contact]
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    invalidDimensions,
                "contact endpoint topology is invalid"
            );
        }
    }
    const std::size_t authoredEqualityCount =
        input.pointEqualityCount +
        input.angularEqualityCount;
    staged.pointEqualityEndpoints.resize(
        authoredEqualityCount
    );
    for (std::size_t equality = 0u;
         equality < input.pointEqualityCount;
         ++equality) {
        const MultiArticulatedPointEquality& source =
            input.pointEqualities[equality];
        if ((
                !endpointResponds(source.endpointA) &&
                !endpointResponds(source.endpointB)
            ) ||
            (
                source.endpointA.kind ==
                    MultiContactEndpointKind::staticWorld &&
                source.endpointB.kind ==
                    MultiContactEndpointKind::staticWorld
            ) ||
            sameTopology(
                source.endpointA,
                source.endpointB
            ) ||
            !compileEndpoint(
                source.endpointA,
                equality,
                false,
                true,
                false,
                staged.pointEqualityEndpoints[equality]
            ) ||
            !compileEndpoint(
                source.endpointB,
                equality,
                true,
                true,
                false,
                staged.pointEqualityEndpoints[equality]
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "point equality endpoint topology is invalid"
            );
        }
    }
    for (std::size_t angular = 0u;
         angular < input.angularEqualityCount;
         ++angular) {
        const std::size_t slot =
            input.pointEqualityCount + angular;
        const MultiArticulatedAngularEquality& source =
            input.angularEqualities[angular];
        if ((
                !endpointResponds(source.endpointA) &&
                !endpointResponds(source.endpointB)
            ) ||
            (
                source.endpointA.kind ==
                    MultiContactEndpointKind::staticWorld &&
                source.endpointB.kind ==
                    MultiContactEndpointKind::staticWorld
            ) ||
            sameTopology(
                source.endpointA,
                source.endpointB
            ) ||
            !compileEndpoint(
                source.endpointA,
                angular,
                false,
                true,
                true,
                staged.pointEqualityEndpoints[slot]
            ) ||
            !compileEndpoint(
                source.endpointB,
                angular,
                true,
                true,
                true,
                staged.pointEqualityEndpoints[slot]
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "angular equality endpoint topology is invalid"
            );
        }
    }

    staged.contacts.resize(contactElements);
    for (std::size_t environment = 0u;
         environment < input.environmentCount;
         ++environment) {
        for (std::size_t contact = 0u;
             contact < input.contactCount;
             ++contact) {
            const MultiArticulatedIslandContact& source =
                input.contacts[
                    environment * input.contactCount + contact
                ];
            const MultiArticulatedIslandContact& topology =
                input.contacts[contact];
            const bool valid =
                sameTopology(
                    source.endpointA,
                    topology.endpointA
                ) &&
                sameTopology(
                    source.endpointB,
                    topology.endpointB
                ) &&
                finite(source.endpointA.localPoint) &&
                finite(source.endpointB.localPoint) &&
                finite(source.targetVelocity) &&
                finite(source.regularization) &&
                finite(source.warmImpulse) &&
                finite(source.friction) &&
                source.friction > 0.0 &&
                std::ranges::all_of(
                    source.regularization,
                    [](const double value) {
                        return finite(value) && value > 0.0;
                    }
                ) &&
                validFrame(source);
            if (!valid) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        nonfiniteInput,
                    "contact geometry or exact-cone semantics are invalid"
                );
            }
            MRMultiContactGPU& gpu =
                staged.contacts[
                    environment * input.contactCount + contact
                ];
            gpu.localPointA =
                f4(source.endpointA.localPoint);
            gpu.localPointB =
                f4(source.endpointB.localPoint);
            gpu.normal = f4(source.normal);
            gpu.tangentU = f4(source.tangentU);
            gpu.tangentV = f4(source.tangentV);
            gpu.targetVelocity =
                f4(source.targetVelocity);
            gpu.regularization =
                f4(source.regularization);
            gpu.warmImpulse =
                config.quality.enableWarmStart
                ? f4(source.warmImpulse)
                : mr_float4{};
            gpu.friction = {
                static_cast<float>(source.friction),
                0.0f,
                0.0f,
                0.0f,
            };
        }
    }
    staged.pointEqualities.resize(
        pointEqualityElements + angularEqualityElements
    );
    staged.equalityKinds.assign(
        authoredEqualityCount,
        MR_MULTI_EQUALITY_POINT
    );
    std::fill(
        staged.equalityKinds.begin() +
            static_cast<std::ptrdiff_t>(
                input.pointEqualityCount
            ),
        staged.equalityKinds.end(),
        MR_MULTI_EQUALITY_ANGULAR
    );
    staged.equalityRows.resize(
        input.environmentCount * equalityRows
    );
    for (std::size_t environment = 0u;
         environment < input.environmentCount;
         ++environment) {
        std::ranges::copy(
            model.constraintProgram.rows,
            staged.equalityRows.begin() +
                static_cast<std::ptrdiff_t>(
                    environment * equalityRows
                )
        );
        for (std::size_t equality = 0u;
             equality < input.pointEqualityCount;
             ++equality) {
            const MultiArticulatedPointEquality& source =
                input.pointEqualities[
                    environment *
                        input.pointEqualityCount +
                    equality
                ];
            const MultiArticulatedPointEquality& topology =
                input.pointEqualities[equality];
            MultiArticulatedIslandContact frame;
            frame.normal = source.axisX;
            frame.tangentU = source.axisY;
            frame.tangentV = source.axisZ;
            const bool valid =
                sameTopology(
                    source.endpointA,
                    topology.endpointA
                ) &&
                sameTopology(
                    source.endpointB,
                    topology.endpointB
                ) &&
                constraintIRKeyEqual(
                    source.key,
                    topology.key
                ) &&
                finite(source.endpointA.localPoint) &&
                finite(source.endpointB.localPoint) &&
                finite(source.positionError) &&
                finite(source.targetVelocity) &&
                finite(source.compliance) &&
                finite(source.dissipation) &&
                finite(source.warmImpulse) &&
                finite(source.timeConstant) &&
                finite(source.dampingRatio) &&
                source.timeConstant >= 0.0 &&
                source.dampingRatio >= 0.0 &&
                std::ranges::all_of(
                    source.compliance,
                    [](const double value) {
                        return value >= 0.0;
                    }
                ) &&
                std::ranges::all_of(
                    source.dissipation,
                    [](const double value) {
                        return value >= 0.0;
                    }
                ) &&
                validFrame(frame);
            if (!valid) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        nonfiniteInput,
                    "point equality semantics are invalid"
                );
            }
            MRMultiContactGPU& gpu =
                staged.pointEqualities[
                    environment *
                        authoredEqualityCount +
                    equality
                ];
            gpu.localPointA = f4(source.endpointA.localPoint);
            gpu.localPointB = f4(source.endpointB.localPoint);
            gpu.normal = f4(source.axisX);
            gpu.tangentU = f4(source.axisY);
            gpu.tangentV = f4(source.axisZ);
            for (std::uint32_t row = 0u;
                 row < 3u;
                 ++row) {
                ConstraintIRRow& semantics =
                    staged.equalityRows[
                        environment * equalityRows +
                        staticEqualityRows +
                        3u * equality + row
                    ];
                const std::array<double, 3>& axis =
                    row == 0u
                    ? source.axisX
                    : (row == 1u
                        ? source.axisY
                        : source.axisZ);
                double prescribedVelocity = 0.0;
                const auto addPrescribed = [&](
                    const MultiContactEndpoint& endpoint,
                    const double sign
                ) {
                    if (endpoint.kind !=
                            MultiContactEndpointKind::
                                sceneBody ||
                        staged.sceneVelocityOffsets[
                            endpoint.body
                        ] != MR_INVALID_INDEX) {
                        return;
                    }
                    const MRBodyStateGPU& body =
                        input.sceneBodies[
                            environment *
                                input.sceneBodyCount +
                            endpoint.body
                        ];
                    const std::array<double, 3> offset =
                        rotate(
                            body.orientation,
                            endpoint.localPoint
                        );
                    const std::array<double, 3> angular{
                        body.angularVelocity.x,
                        body.angularVelocity.y,
                        body.angularVelocity.z,
                    };
                    const std::array<double, 3> rotational =
                        cross(angular, offset);
                    const std::array<double, 3> velocity{
                        body.linearVelocityAndInverseMass.x +
                            rotational[0],
                        body.linearVelocityAndInverseMass.y +
                            rotational[1],
                        body.linearVelocityAndInverseMass.z +
                            rotational[2],
                    };
                    prescribedVelocity +=
                        sign * dot(axis, velocity);
                };
                addPrescribed(source.endpointA, -1.0);
                addPrescribed(source.endpointB, 1.0);
                semantics.direction = f4(axis);
                semantics.positionError =
                    static_cast<float>(
                        source.positionError[row]
                    );
                semantics.targetVelocity =
                    static_cast<float>(
                        source.targetVelocity[row] -
                        prescribedVelocity
                    );
                semantics.compliance =
                    static_cast<float>(
                        source.compliance[row]
                    );
                semantics.dissipation =
                    static_cast<float>(
                        source.dissipation[row]
                    );
                semantics.timeConstant =
                    static_cast<float>(
                        source.timeConstant
                    );
                semantics.dampingRatio =
                    static_cast<float>(
                        source.dampingRatio
                    );
                semantics.impulseLower =
                    -kConstraintIRUnbounded;
                semantics.impulseUpper =
                    kConstraintIRUnbounded;
                semantics.flags =
                    source.positionStabilized
                    ? constraintIRRowPositionStabilized
                    : 0u;
            }
        }
        for (std::size_t angular = 0u;
             angular < input.angularEqualityCount;
             ++angular) {
            const std::size_t slot =
                input.pointEqualityCount + angular;
            const MultiArticulatedAngularEquality& source =
                input.angularEqualities[
                    environment *
                        input.angularEqualityCount +
                    angular
                ];
            const MultiArticulatedAngularEquality& topology =
                input.angularEqualities[angular];
            MultiArticulatedIslandContact frame;
            frame.normal = source.axisX;
            frame.tangentU = source.axisY;
            frame.tangentV = source.axisZ;
            const bool valid =
                sameTopology(
                    source.endpointA,
                    topology.endpointA
                ) &&
                sameTopology(
                    source.endpointB,
                    topology.endpointB
                ) &&
                constraintIRKeyEqual(
                    source.key,
                    topology.key
                ) &&
                finite(source.endpointA.localPoint) &&
                finite(source.endpointB.localPoint) &&
                finite(source.orientationError) &&
                finite(source.targetVelocity) &&
                finite(source.compliance) &&
                finite(source.dissipation) &&
                finite(source.warmImpulse) &&
                finite(source.timeConstant) &&
                finite(source.dampingRatio) &&
                source.timeConstant >= 0.0 &&
                source.dampingRatio >= 0.0 &&
                std::ranges::all_of(
                    source.compliance,
                    [](const double value) {
                        return value >= 0.0;
                    }
                ) &&
                std::ranges::all_of(
                    source.dissipation,
                    [](const double value) {
                        return value >= 0.0;
                    }
                ) &&
                validFrame(frame);
            if (!valid) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        nonfiniteInput,
                    "angular equality semantics are invalid"
                );
            }
            MRMultiContactGPU& gpu =
                staged.pointEqualities[
                    environment * authoredEqualityCount +
                    slot
                ];
            gpu.localPointA = f4(source.endpointA.localPoint);
            gpu.localPointB = f4(source.endpointB.localPoint);
            gpu.normal = f4(source.axisX);
            gpu.tangentU = f4(source.axisY);
            gpu.tangentV = f4(source.axisZ);
            for (std::uint32_t row = 0u;
                 row < 3u;
                 ++row) {
                ConstraintIRRow& semantics =
                    staged.equalityRows[
                        environment * equalityRows +
                        staticEqualityRows +
                        3u * slot + row
                    ];
                const std::array<double, 3>& axis =
                    row == 0u
                    ? source.axisX
                    : (row == 1u
                        ? source.axisY
                        : source.axisZ);
                double prescribedVelocity = 0.0;
                const auto addPrescribed = [&](
                    const MultiContactEndpoint& endpoint,
                    const double sign
                ) {
                    if (endpoint.kind !=
                            MultiContactEndpointKind::
                                sceneBody ||
                        staged.sceneVelocityOffsets[
                            endpoint.body
                        ] != MR_INVALID_INDEX) {
                        return;
                    }
                    const MRBodyStateGPU& body =
                        input.sceneBodies[
                            environment *
                                input.sceneBodyCount +
                            endpoint.body
                        ];
                    prescribedVelocity += sign * (
                        axis[0] * body.angularVelocity.x +
                        axis[1] * body.angularVelocity.y +
                        axis[2] * body.angularVelocity.z
                    );
                };
                addPrescribed(source.endpointA, -1.0);
                addPrescribed(source.endpointB, 1.0);
                semantics.direction = f4(axis);
                semantics.positionError =
                    static_cast<float>(
                        source.orientationError[row]
                    );
                semantics.targetVelocity =
                    static_cast<float>(
                        source.targetVelocity[row] -
                        prescribedVelocity
                    );
                semantics.compliance =
                    static_cast<float>(
                        source.compliance[row]
                    );
                semantics.dissipation =
                    static_cast<float>(
                        source.dissipation[row]
                    );
                semantics.timeConstant =
                    static_cast<float>(
                        source.timeConstant
                    );
                semantics.dampingRatio =
                    static_cast<float>(
                        source.dampingRatio
                    );
                semantics.impulseLower =
                    -kConstraintIRUnbounded;
                semantics.impulseUpper =
                    kConstraintIRUnbounded;
                semantics.flags =
                    source.positionStabilized
                    ? constraintIRRowPositionStabilized
                    : 0u;
            }
        }
    }

    staged.layout.pointDispatches.resize(
        model.articulations.size()
    );
    staged.layout.pointSlices.resize(
        model.articulations.size()
    );
    staged.pointArenas.resize(model.articulations.size());
    for (std::size_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        if (articulation.bodyCount >
                MR_ARTICULATED_OPERATOR_MAX_BODIES ||
            articulation.nv >
                MR_ARTICULATED_OPERATOR_MAX_DOFS ||
            articulation.nv >
                MR_ARTICULATED_ABA_MAX_DOFS ||
            articulation.bodyCount >
                MR_ARTICULATED_ABA_MAX_BODIES ||
            occurrences[articulationIndex].size() >
                MR_ARTICULATED_OPERATOR_MAX_POINTS) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    unsupportedTopology,
                "articulation exceeds point or inverse-ABA capacity"
            );
        }
        const std::size_t queryCount =
            occurrences[articulationIndex].size();
        if (staged.layout.pointQueryElements >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.layout.pointJacobianElements >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.pointWorldElements >
                std::numeric_limits<std::uint32_t>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    arithmeticOverflow,
                "point arena offsets exceed the GPU ABI"
            );
        }
        MRArticulatedOperatorDispatchGPU& pointDispatch =
            staged.layout.pointDispatches[articulationIndex];
        pointDispatch.articulationIndex =
            static_cast<std::uint32_t>(articulationIndex);
        pointDispatch.environmentCount =
            static_cast<std::uint32_t>(
                input.environmentCount
            );
        pointDispatch.pointCount =
            static_cast<std::uint32_t>(queryCount);
        pointDispatch.flags =
            MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
        pointDispatch.qStride = model.world.nq;
        pointDispatch.pointStride =
            static_cast<std::uint32_t>(queryCount);
        pointDispatch.bodyPoseStride =
            articulation.bodyCount;
        pointDispatch.pointWorldStride =
            static_cast<std::uint32_t>(queryCount);
        pointDispatch.pointJacobianStride =
            static_cast<std::uint32_t>(
                queryCount * 3u * articulation.nv
            );
        pointDispatch.generalizedStride = articulation.nv;

        MRMultiContactJacobianSliceGPU& slice =
            staged.layout.pointSlices[articulationIndex];
        slice.articulationIndex =
            static_cast<std::uint32_t>(articulationIndex);
        slice.queryOffset =
            static_cast<std::uint32_t>(
                staged.layout.pointQueryElements
            );
        slice.queryCount =
            static_cast<std::uint32_t>(queryCount);
        slice.jacobianOffset =
            static_cast<std::uint32_t>(
                staged.layout.pointJacobianElements
            );
        slice.jacobianEnvironmentStride =
            pointDispatch.pointJacobianStride;
        slice.vOffset = articulation.vOffset;
        slice.nv = articulation.nv;

        PointArenaSlice& arena =
            staged.pointArenas[articulationIndex];
        arena.bodyPoseOffset = staged.bodyPoseElements;
        arena.pointWorldOffset = staged.pointWorldElements;
        arena.generalizedOffset =
            staged.pointGeneralizedElements;
        arena.statusOffset =
            articulationIndex * input.environmentCount;
        slice.pointWorldOffset =
            static_cast<std::uint32_t>(
                arena.pointWorldOffset
            );

        std::size_t count = 0u;
        if (!checkedMultiply(
                input.environmentCount,
                queryCount,
                count
            ) ||
            !checkedAdd(
                staged.layout.pointQueryElements,
                count,
                staged.layout.pointQueryElements
            ) ||
            !checkedMultiply(
                count,
                3u * articulation.nv,
                count
            ) ||
            !checkedAdd(
                staged.layout.pointJacobianElements,
                count,
                staged.layout.pointJacobianElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    arithmeticOverflow,
                "point-Jacobian arena size overflow"
            );
        }
        std::size_t poseCount = 0u;
        std::size_t generalizedCount = 0u;
        if (!checkedMultiply(
                input.environmentCount,
                articulation.bodyCount,
                poseCount
            ) ||
            !checkedAdd(
                staged.bodyPoseElements,
                poseCount,
                staged.bodyPoseElements
            ) ||
            !checkedAdd(
                staged.pointWorldElements,
                input.environmentCount * queryCount,
                staged.pointWorldElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                articulation.nv,
                generalizedCount
            ) ||
            !checkedAdd(
                staged.pointGeneralizedElements,
                generalizedCount,
                staged.pointGeneralizedElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    arithmeticOverflow,
                "point output arena size overflow"
            );
        }
    }

    staged.pointQueries.resize(
        staged.layout.pointQueryElements
    );
    for (std::size_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const auto& occurrence =
            occurrences[articulationIndex];
        const auto& slice =
            staged.layout.pointSlices[articulationIndex];
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            for (std::size_t query = 0u;
                 query < occurrence.size();
                 ++query) {
                const PointOccurrence& item =
                    occurrence[query];
                MultiContactEndpoint endpoint;
                if (item.angular) {
                    const MultiArticulatedAngularEquality&
                        equality =
                            input.angularEqualities[
                                environment *
                                    input.angularEqualityCount +
                                item.slot
                            ];
                    endpoint = item.second
                        ? equality.endpointB
                        : equality.endpointA;
                    if (item.basis != 0u) {
                        endpoint.localPoint[
                            item.basis - 1u
                        ] += 1.0;
                    }
                } else if (item.equality) {
                    const MultiArticulatedPointEquality&
                        equality =
                            input.pointEqualities[
                                environment *
                                    input.pointEqualityCount +
                                item.slot
                            ];
                    endpoint = item.second
                        ? equality.endpointB
                        : equality.endpointA;
                } else {
                    const MultiArticulatedIslandContact&
                        contact =
                            input.contacts[
                                environment *
                                    input.contactCount +
                                item.slot
                            ];
                    endpoint = item.second
                        ? contact.endpointB
                        : contact.endpointA;
                }
                MRArticulatedPointImpulseGPU point{};
                point.bodyIndex = endpoint.body;
                point.localPoint = f4(endpoint.localPoint);
                staged.pointQueries[
                    slice.queryOffset +
                    environment * occurrence.size() +
                    query
                ] = point;
            }
        }
    }

    const std::size_t rowCount = 3u * input.contactCount;
    const std::size_t responseRowCount =
        rowCount + equalityRows;
    std::size_t environmentRows = 0u;
    std::size_t environmentResponseRows = 0u;
    std::size_t rowPairs = 0u;
    if (!checkedMultiply(
            input.environmentCount,
            totalNv,
            staged.layout.packedVelocityElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            rowCount,
            environmentRows
        ) ||
        !checkedMultiply(
            input.environmentCount,
            responseRowCount,
            environmentResponseRows
        ) ||
        !checkedMultiply(
            environmentResponseRows,
            totalNv,
            staged.layout.jacobianElements
        ) ||
        !checkedMultiply(
            environmentRows,
            rowCount,
            rowPairs
        )) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                arithmeticOverflow,
            "contact operator size overflow"
        );
    }
    staged.layout.delassusElements = rowPairs;
    staged.layout.rowElements = environmentRows;
    staged.layout.responseRowElements =
        environmentResponseRows;
    if (!checkedMultiply(
            input.environmentCount,
            equalityRows * equalityRows,
            staged.layout.equalityOperatorElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            equalityRows * rowCount,
            staged.layout.equalityCouplingElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            equalityRows,
            staged.layout.equalityImpulseElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                arithmeticOverflow,
            "equality projection arena size overflow"
        );
    }
    std::size_t inverseEnvironmentStride = 0u;
    if (!checkedMultiply(
            responseRowCount,
            totalNv,
            inverseEnvironmentStride
        ) ||
        inverseEnvironmentStride >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                arithmeticOverflow,
            "inverse-response stride exceeds the GPU ABI"
        );
    }

    for (std::size_t rowBegin = 0u;
         rowBegin < responseRowCount;
         rowBegin += MR_ARTICULATED_INVERSE_MASS_MAX_RHS) {
        const std::size_t chunkRows = std::min<std::size_t>(
            MR_ARTICULATED_INVERSE_MASS_MAX_RHS,
            responseRowCount - rowBegin
        );
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            MRMultiInverseMassDispatchGPU work{};
            work.dispatch.articulationIndex =
                static_cast<std::uint32_t>(
                    articulationIndex
                );
            work.dispatch.environmentCount =
                static_cast<std::uint32_t>(
                    input.environmentCount
                );
            work.dispatch.rhsCount =
                static_cast<std::uint32_t>(chunkRows);
            work.dispatch.qStride = model.world.nq;
            work.dispatch.rhsEnvironmentStride =
                static_cast<std::uint32_t>(
                    inverseEnvironmentStride
                );
            work.dispatch.rhsVectorStride =
                static_cast<std::uint32_t>(totalNv);
            work.dispatch.outputEnvironmentStride =
                work.dispatch.rhsEnvironmentStride;
            work.dispatch.outputVectorStride =
                work.dispatch.rhsVectorStride;
            work.qBase = articulation.qOffset;
            work.rhsBase = static_cast<std::uint32_t>(
                rowBegin * totalNv + articulation.vOffset
            );
            work.outputBase = work.rhsBase;
            work.statusBase = static_cast<std::uint32_t>(
                staged.layout.inverseMassDispatches.size() *
                    input.environmentCount
            );
            staged.layout.inverseMassDispatches.push_back(work);
        }
    }

    MRMultiContactDispatchGPU& dispatch =
        staged.layout.dispatch;
    dispatch.abiVersion = MR_MULTI_CONTACT_ABI_VERSION;
    dispatch.environmentCount =
        static_cast<std::uint32_t>(input.environmentCount);
    dispatch.articulationCount =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    dispatch.sceneBodyCount =
        static_cast<std::uint32_t>(input.sceneBodyCount);
    dispatch.articulatedNv = model.world.nv;
    dispatch.totalNv = static_cast<std::uint32_t>(totalNv);
    dispatch.contactCount =
        static_cast<std::uint32_t>(input.contactCount);
    dispatch.rowCount =
        static_cast<std::uint32_t>(rowCount);
    dispatch.inverseWorkCount =
        static_cast<std::uint32_t>(
            staged.layout.inverseMassDispatches.size()
        );
    dispatch.equalityRowCount =
        static_cast<std::uint32_t>(equalityRows);
    dispatch.responseRowCount =
        static_cast<std::uint32_t>(responseRowCount);
    dispatch.staticEqualityRowCount =
        static_cast<std::uint32_t>(staticEqualityRows);
    dispatch.tolerances = {
        config.delassusSymmetryTolerance,
        config.delassusDiagonalTolerance,
        0.0f,
        0.0f,
    };
    dispatch.equalityEvaluation0 = {
        static_cast<float>(
            config.equalityEvaluation.timestep
        ),
        static_cast<float>(
            config.equalityEvaluation
                .maximumDepenetrationVelocity
        ),
        static_cast<float>(
            config.equalityEvaluation
                .minimumTimeConstantRatio
        ),
        static_cast<float>(
            config.equalityEvaluation.minimumRegularization
        ),
    };
    dispatch.equalityEvaluation1 = {
        config.equalityPivotTolerance,
        config.equalityResidualTolerance,
        0.0f,
        0.0f,
    };

    std::size_t bytes = 0u;
#define MR_ADD_BUFFER(Type, Count) \
    addBytes<Type>((Count), bytes)
    const std::size_t pointStatusElements =
        input.environmentCount * model.articulations.size();
    const std::size_t inverseStatusElements =
        input.environmentCount *
        staged.layout.inverseMassDispatches.size();
    const bool byteCountsValid =
        MR_ADD_BUFFER(MRWorldGPU, 1u) &&
        MR_ADD_BUFFER(
            MRArticulationGPU,
            model.articulations.size()
        ) &&
        MR_ADD_BUFFER(MRJointDescriptorGPU, model.joints.size()) &&
        MR_ADD_BUFFER(MRDofPropertiesGPU, model.dofs.size()) &&
        MR_ADD_BUFFER(MRBodyPropertiesGPU, model.bodies.size()) &&
        MR_ADD_BUFFER(
            MRArticulatedOperatorDispatchGPU,
            staged.layout.pointDispatches.size()
        ) &&
        MR_ADD_BUFFER(float, qElements) &&
        MR_ADD_BUFFER(
            MRArticulatedPointImpulseGPU,
            staged.layout.pointQueryElements
        ) &&
        MR_ADD_BUFFER(
            MRArticulatedBodyPoseGPU,
            staged.bodyPoseElements
        ) &&
        MR_ADD_BUFFER(
            MRArticulatedPointWorldGPU,
            staged.pointWorldElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.pointJacobianElements
        ) &&
        MR_ADD_BUFFER(float, staged.pointGeneralizedElements) &&
        MR_ADD_BUFFER(float, staged.pointGeneralizedElements) &&
        MR_ADD_BUFFER(
            MRArticulatedOperatorStatusGPU,
            pointStatusElements
        ) &&
        MR_ADD_BUFFER(MRMultiContactDispatchGPU, 1u) &&
        MR_ADD_BUFFER(MRMultiContactGPU, staged.contacts.size()) &&
        MR_ADD_BUFFER(
            MRMultiContactEndpointsGPU,
            staged.endpoints.size()
        ) &&
        MR_ADD_BUFFER(
            MRMultiContactGPU,
            staged.pointEqualities.size()
        ) &&
        MR_ADD_BUFFER(
            MRMultiContactEndpointsGPU,
            staged.pointEqualityEndpoints.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.equalityKinds.size()
        ) &&
        MR_ADD_BUFFER(
            MRMultiContactJacobianSliceGPU,
            staged.layout.pointSlices.size()
        ) &&
        MR_ADD_BUFFER(MRBodyStateGPU, sceneElements) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.sceneVelocityOffsets.size()
        ) &&
        MR_ADD_BUFFER(
            float,
            articulatedVelocityElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.packedVelocityElements
        ) &&
        MR_ADD_BUFFER(float, staged.layout.jacobianElements) &&
        MR_ADD_BUFFER(float, staged.layout.jacobianElements) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.rowElements * totalNv
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.packedVelocityElements
        ) &&
        MR_ADD_BUFFER(float, equalityJacobian.size()) &&
        MR_ADD_BUFFER(
            ConstraintIRRow,
            staged.equalityRows.size()
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityOperatorElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityCouplingElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityImpulseElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityImpulseElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityImpulseElements
        ) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.equalityImpulseElements
        ) &&
        MR_ADD_BUFFER(
            MRMultiContactEqualityStatusGPU,
            input.environmentCount
        ) &&
        MR_ADD_BUFFER(
            MRMultiInverseMassDispatchGPU,
            staged.layout.inverseMassDispatches.size()
        ) &&
        MR_ADD_BUFFER(
            MRInverseMassStatusGPU,
            inverseStatusElements
        ) &&
        MR_ADD_BUFFER(
            MRParallelABAArticulationGPU,
            staged.schedule->articulations.size()
        ) &&
        MR_ADD_BUFFER(
            MRParallelABALevelGPU,
            staged.schedule->levels.size()
        ) &&
        MR_ADD_BUFFER(
            MRParallelABAParentReductionGPU,
            staged.schedule->parentReductions.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.schedule->levelBodies.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.schedule->parentLocal.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.schedule->inboundJoint.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.schedule->childOffsets.size()
        ) &&
        MR_ADD_BUFFER(
            std::uint32_t,
            staged.schedule->childIndices.size()
        ) &&
        MR_ADD_BUFFER(float, staged.layout.delassusElements) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(float, staged.layout.delassusElements) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(
            MRMetalQualityStatusGPU,
            input.environmentCount
        ) &&
        MR_ADD_BUFFER(float, staged.layout.rowElements) &&
        MR_ADD_BUFFER(
            float,
            staged.layout.packedVelocityElements
        ) &&
        MR_ADD_BUFFER(
            MRMultiContactStatusGPU,
            input.environmentCount
        );
#undef MR_ADD_BUFFER
    if (!byteCountsValid) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                arithmeticOverflow,
            "aggregate contact buffer byte count overflow"
        );
    }
    staged.layout.totalAllocatedBytes = bytes;
    diagnostics.layout = staged.layout;
    output = std::move(staged);
    return diagnostics;
}

template <typename T>
void copyBuffer(
    std::vector<T>& output,
    id<MTLBuffer> buffer
) {
    if (output.empty()) {
        return;
    }
    std::memcpy(
        output.data(),
        buffer.contents,
        output.size() * sizeof(T)
    );
}

} // namespace

static MetalMultiArticulatedContactDiagnostics
solveMetalMultiArticulatedContactsImpl(
    const EngineModel& model,
    const ParallelABASchedule& schedule,
    const std::span<const float> equalityJacobian,
    const MetalMultiArticulatedContactInput& input,
    MetalMultiArticulatedContactResult& output,
    const MetalMultiArticulatedContactConfig& config
) {
    Prepared prepared;
    MetalMultiArticulatedContactDiagnostics diagnostics;
    try {
        diagnostics = prepare(
            model,
            schedule,
            equalityJacobian,
            input,
            config,
            prepared
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::
                metalBufferFailure,
            "host allocation failed while preparing contact graph"
        );
    }

    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalDeviceUnavailable,
                "no Metal device is available"
            );
        }
        diagnostics.deviceName = string(device.name);
        if (!device.hasUnifiedMemory) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalDeviceUnsupported,
                "multi-articulation contact requires unified memory"
            );
        }
        if (diagnostics.layout.totalAllocatedBytes >
            device.recommendedMaxWorkingSetSize &&
            device.recommendedMaxWorkingSetSize != 0u) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalBufferFailure,
                "contact graph exceeds the recommended working set"
            );
        }

        std::string metallibPath = config.metallibPath;
        if (metallibPath.empty()) {
            metallibPath = config.quality.metallibPath;
        }
        if (metallibPath.empty()) {
            metallibPath = defaultMetallibPath();
        }
        if (metallibPath.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:
                [NSURL fileURLWithPath:
                    [NSString
                        stringWithUTF8String:
                            metallibPath.c_str()]]
                        error:&error];
        if (library == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalLibraryFailure,
                "failed to load metallib: " +
                    errorString(error)
            );
        }

        const std::array<NSString*, 13> names{
            @"mr_articulated_operator",
            @"mr_multi_contact_assemble_jacobian",
            @"mr_multi_contact_pack_free_velocity",
            @"mr_multi_contact_scatter_equality_jacobian",
            @"mr_multi_contact_assemble_point_equalities",
            @"mr_parallel_multi_articulated_inverse_mass",
            @"mr_multi_contact_apply_scene_response",
            @"mr_multi_contact_project_equalities",
            @"mr_multi_contact_delassus",
            @"mr_multi_contact_free_contact_velocity",
            @"mr_multi_contact_prepare_quality_matrix",
            @"mr_multi_contact_prepare_quality_vector",
            @"mr_quality_contact_solve",
        };
        std::array<id<MTLComputePipelineState>, 13> pipelines{};
        for (std::size_t index = 0u;
             index < names.size();
             ++index) {
            error = nil;
            pipelines[index] = pipeline(
                device,
                library,
                names[index],
                &error
            );
            if (pipelines[index] == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        metalPipelineFailure,
                    "failed to create pipeline " +
                        string(names[index]) + ": " +
                        errorString(error)
                );
            }
        }
        error = nil;
        id<MTLComputePipelineState> finalizePipeline =
            pipeline(
                device,
                library,
                @"mr_multi_contact_finalize",
                &error
            );
        if (finalizePipeline == nil ||
            pipelines[0].threadExecutionWidth != kWaveWidth ||
            pipelines[5].threadExecutionWidth != kWaveWidth ||
            pipelines[7].threadExecutionWidth != kWaveWidth ||
            pipelines[12].threadExecutionWidth != kWaveWidth ||
            finalizePipeline.threadExecutionWidth != kWaveWidth) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalDeviceUnsupported,
                "required SIMD32 contact pipelines are unavailable"
            );
        }

        const auto& layout = prepared.layout;
        const std::size_t authoredEqualityCount =
            input.pointEqualityCount +
            input.angularEqualityCount;
        const std::size_t pointStatusElements =
            input.environmentCount * model.articulations.size();
        const std::size_t inverseStatusElements =
            input.environmentCount *
            layout.inverseMassDispatches.size();
        id<MTLBuffer> worldBuffer = inputBuffer(
            device, &model.world, 1u
        );
        id<MTLBuffer> articulationBuffer = inputBuffer(
            device,
            model.articulations.data(),
            model.articulations.size()
        );
        id<MTLBuffer> jointBuffer = inputBuffer(
            device, model.joints.data(), model.joints.size()
        );
        id<MTLBuffer> dofBuffer = inputBuffer(
            device, model.dofs.data(), model.dofs.size()
        );
        id<MTLBuffer> bodyBuffer = inputBuffer(
            device, model.bodies.data(), model.bodies.size()
        );
        id<MTLBuffer> pointDispatchBuffer = inputBuffer(
            device,
            layout.pointDispatches.data(),
            layout.pointDispatches.size()
        );
        id<MTLBuffer> qBuffer = inputBuffer(
            device, input.q.data(), input.q.size()
        );
        id<MTLBuffer> pointQueryBuffer = inputBuffer(
            device,
            prepared.pointQueries.data(),
            prepared.pointQueries.size()
        );
        id<MTLBuffer> bodyPoseBuffer =
            outputBuffer<MRArticulatedBodyPoseGPU>(
                device, prepared.bodyPoseElements
            );
        id<MTLBuffer> pointWorldBuffer =
            outputBuffer<MRArticulatedPointWorldGPU>(
                device, prepared.pointWorldElements
            );
        id<MTLBuffer> emptyMassBuffer =
            outputBuffer<float>(device, 0u);
        id<MTLBuffer> pointJacobianBuffer =
            outputBuffer<float>(
                device, layout.pointJacobianElements
            );
        id<MTLBuffer> pointGeneralizedImpulseBuffer =
            outputBuffer<float>(
                device, prepared.pointGeneralizedElements
            );
        id<MTLBuffer> pointDeltaVelocityBuffer =
            outputBuffer<float>(
                device, prepared.pointGeneralizedElements
            );
        id<MTLBuffer> pointStatusBuffer =
            outputBuffer<MRArticulatedOperatorStatusGPU>(
                device, pointStatusElements
            );
        id<MTLBuffer> dispatchBuffer = inputBuffer(
            device, &layout.dispatch, 1u
        );
        id<MTLBuffer> contactBuffer = inputBuffer(
            device,
            prepared.contacts.data(),
            prepared.contacts.size()
        );
        id<MTLBuffer> endpointBuffer = inputBuffer(
            device,
            prepared.endpoints.data(),
            prepared.endpoints.size()
        );
        id<MTLBuffer> sliceBuffer = inputBuffer(
            device,
            layout.pointSlices.data(),
            layout.pointSlices.size()
        );
        id<MTLBuffer> sceneBodyBuffer = inputBuffer(
            device,
            input.sceneBodies.data(),
            input.sceneBodies.size()
        );
        id<MTLBuffer> sceneOffsetBuffer = inputBuffer(
            device,
            prepared.sceneVelocityOffsets.data(),
            prepared.sceneVelocityOffsets.size()
        );
        id<MTLBuffer> freeArticulationBuffer = inputBuffer(
            device,
            input.freeArticulationVelocity.data(),
            input.freeArticulationVelocity.size()
        );
        id<MTLBuffer> packedFreeBuffer =
            outputBuffer<float>(
                device, layout.packedVelocityElements
            );
        id<MTLBuffer> jacobianBuffer =
            outputBuffer<float>(
                device, layout.jacobianElements
            );
        id<MTLBuffer> responseBuffer =
            outputBuffer<float>(
                device, layout.jacobianElements
            );
        id<MTLBuffer> equalityJacobianBuffer = inputBuffer(
            device,
            prepared.equalityJacobian.data(),
            prepared.equalityJacobian.size()
        );
        id<MTLBuffer> pointEqualityBuffer = inputBuffer(
            device,
            prepared.pointEqualities.data(),
            prepared.pointEqualities.size()
        );
        id<MTLBuffer> pointEqualityEndpointBuffer =
            inputBuffer(
                device,
                prepared.pointEqualityEndpoints.data(),
                prepared.pointEqualityEndpoints.size()
            );
        id<MTLBuffer> equalityKindBuffer = inputBuffer(
            device,
            prepared.equalityKinds.data(),
            prepared.equalityKinds.size()
        );
        id<MTLBuffer> equalityRowBuffer = inputBuffer(
            device,
            prepared.equalityRows.data(),
            prepared.equalityRows.size()
        );
        id<MTLBuffer> projectedResponseBuffer =
            outputBuffer<float>(
                device,
                layout.rowElements *
                    layout.dispatch.totalNv
            );
        id<MTLBuffer> projectedFreeBuffer =
            outputBuffer<float>(
                device,
                layout.packedVelocityElements
            );
        id<MTLBuffer> equalityOperatorBuffer =
            outputBuffer<float>(
                device,
                layout.equalityOperatorElements
            );
        id<MTLBuffer> equalityCouplingBuffer =
            outputBuffer<float>(
                device,
                layout.equalityCouplingElements
            );
        id<MTLBuffer> equalityFreeImpulseBuffer =
            outputBuffer<float>(
                device,
                layout.equalityImpulseElements
            );
        id<MTLBuffer> equalityTargetBuffer =
            outputBuffer<float>(
                device,
                layout.equalityImpulseElements
            );
        id<MTLBuffer> equalityRegularizationBuffer =
            outputBuffer<float>(
                device,
                layout.equalityImpulseElements
            );
        id<MTLBuffer> equalityImpulseBuffer =
            outputBuffer<float>(
                device,
                layout.equalityImpulseElements
            );
        id<MTLBuffer> equalityStatusBuffer =
            outputBuffer<MRMultiContactEqualityStatusGPU>(
                device,
                input.environmentCount
            );
        id<MTLBuffer> inverseDispatchBuffer = inputBuffer(
            device,
            layout.inverseMassDispatches.data(),
            layout.inverseMassDispatches.size()
        );
        id<MTLBuffer> inverseStatusBuffer =
            outputBuffer<MRInverseMassStatusGPU>(
                device, inverseStatusElements
            );
        const ParallelABASchedule& schedule =
            *prepared.schedule;
        id<MTLBuffer> scheduleArticulationBuffer = inputBuffer(
            device,
            schedule.articulations.data(),
            schedule.articulations.size()
        );
        id<MTLBuffer> scheduleLevelBuffer = inputBuffer(
            device,
            schedule.levels.data(),
            schedule.levels.size()
        );
        id<MTLBuffer> scheduleReductionBuffer = inputBuffer(
            device,
            schedule.parentReductions.data(),
            schedule.parentReductions.size()
        );
        id<MTLBuffer> scheduleLevelBodyBuffer = inputBuffer(
            device,
            schedule.levelBodies.data(),
            schedule.levelBodies.size()
        );
        id<MTLBuffer> scheduleParentBuffer = inputBuffer(
            device,
            schedule.parentLocal.data(),
            schedule.parentLocal.size()
        );
        id<MTLBuffer> scheduleInboundBuffer = inputBuffer(
            device,
            schedule.inboundJoint.data(),
            schedule.inboundJoint.size()
        );
        id<MTLBuffer> scheduleChildOffsetBuffer = inputBuffer(
            device,
            schedule.childOffsets.data(),
            schedule.childOffsets.size()
        );
        id<MTLBuffer> scheduleChildIndexBuffer = inputBuffer(
            device,
            schedule.childIndices.data(),
            schedule.childIndices.size()
        );
        id<MTLBuffer> delassusBuffer =
            outputBuffer<float>(
                device, layout.delassusElements
            );
        id<MTLBuffer> freeContactBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> qualityMatrixBuffer =
            outputBuffer<float>(
                device, layout.delassusElements
            );
        id<MTLBuffer> linearBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> warmBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> scaleBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> compactImpulseBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> qualityStatusBuffer =
            outputBuffer<MRMetalQualityStatusGPU>(
                device, input.environmentCount
            );
        id<MTLBuffer> physicalImpulseBuffer =
            outputBuffer<float>(device, layout.rowElements);
        id<MTLBuffer> nextVelocityBuffer =
            outputBuffer<float>(
                device, layout.packedVelocityElements
            );
        id<MTLBuffer> statusBuffer =
            outputBuffer<MRMultiContactStatusGPU>(
                device, input.environmentCount
            );
        const std::array allBuffers{
            worldBuffer, articulationBuffer, jointBuffer,
            dofBuffer, bodyBuffer, pointDispatchBuffer,
            qBuffer, pointQueryBuffer, bodyPoseBuffer,
            pointWorldBuffer, emptyMassBuffer,
            pointJacobianBuffer,
            pointGeneralizedImpulseBuffer,
            pointDeltaVelocityBuffer,
            pointStatusBuffer, dispatchBuffer, contactBuffer,
            endpointBuffer, sliceBuffer, sceneBodyBuffer,
            sceneOffsetBuffer, freeArticulationBuffer,
            packedFreeBuffer, jacobianBuffer, responseBuffer,
            equalityJacobianBuffer, equalityRowBuffer,
            pointEqualityBuffer,
            pointEqualityEndpointBuffer,
            equalityKindBuffer,
            projectedResponseBuffer, projectedFreeBuffer,
            equalityOperatorBuffer, equalityCouplingBuffer,
            equalityFreeImpulseBuffer, equalityTargetBuffer,
            equalityRegularizationBuffer,
            equalityImpulseBuffer, equalityStatusBuffer,
            inverseDispatchBuffer, inverseStatusBuffer,
            scheduleArticulationBuffer, scheduleLevelBuffer,
            scheduleReductionBuffer, scheduleLevelBodyBuffer,
            scheduleParentBuffer, scheduleInboundBuffer,
            scheduleChildOffsetBuffer,
            scheduleChildIndexBuffer,
            delassusBuffer, freeContactBuffer,
            qualityMatrixBuffer, linearBuffer, warmBuffer,
            scaleBuffer, compactImpulseBuffer,
            qualityStatusBuffer, physicalImpulseBuffer,
            nextVelocityBuffer, statusBuffer,
        };
        if (std::ranges::any_of(
                allBuffers,
                [](id<MTLBuffer> buffer) {
                    return buffer == nil;
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalBufferFailure,
                "failed to allocate the contact graph buffers"
            );
        }

        MRMetalQualityDispatchGPU qualityDispatch{};
        qualityDispatch.abiVersion =
            MR_METAL_QUALITY_SOLVER_ABI_VERSION;
        qualityDispatch.problemCount =
            static_cast<std::uint32_t>(
                input.environmentCount
            );
        qualityDispatch.contactCount =
            layout.dispatch.contactCount;
        qualityDispatch.dimension =
            layout.dispatch.rowCount;
        qualityDispatch.matrixStride =
            layout.dispatch.rowCount *
            layout.dispatch.rowCount;
        qualityDispatch.vectorStride =
            layout.dispatch.rowCount;
        qualityDispatch.maximumNewtonIterations =
            config.quality.maximumNewtonIterations;
        qualityDispatch.maximumCGIterations =
            config.quality.maximumCGIterations;
        qualityDispatch.maximumLineSearchIterations =
            config.quality.maximumLineSearchIterations;
        qualityDispatch.tolerances = {
            config.quality.convergenceTolerance,
            config.quality.armijoCoefficient,
            config.quality.normalEquationRegularization,
            config.quality.minimumCGDenominator,
        };
        id<MTLBuffer> qualityDispatchBuffer = inputBuffer(
            device, &qualityDispatch, 1u
        );
        if (qualityDispatchBuffer == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalBufferFailure,
                "failed to allocate the quality dispatch"
            );
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (queue == nil || command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalCommandFailure,
                "failed to create Metal command objects"
            );
        }
        const std::array zeroedBuffers{
            pointJacobianBuffer,
            jacobianBuffer,
            responseBuffer,
            delassusBuffer,
            projectedResponseBuffer,
            projectedFreeBuffer,
            equalityOperatorBuffer,
            equalityCouplingBuffer,
            equalityFreeImpulseBuffer,
            equalityTargetBuffer,
            equalityRegularizationBuffer,
            equalityImpulseBuffer,
            equalityStatusBuffer,
        };
        for (id<MTLBuffer> buffer : zeroedBuffers) {
            [blit fillBuffer:buffer
                       range:NSMakeRange(0u, buffer.length)
                       value:0u];
        }
        [blit endEncoding];

        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalCommandFailure,
                "failed to create the contact compute encoder"
            );
        }
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const auto& slice =
                layout.pointSlices[articulationIndex];
            const auto& arena =
                prepared.pointArenas[articulationIndex];
            [encoder setComputePipelineState:pipelines[0]];
            [encoder setBuffer:worldBuffer offset:0u atIndex:0u];
            [encoder setBuffer:articulationBuffer
                         offset:0u atIndex:1u];
            [encoder setBuffer:jointBuffer offset:0u atIndex:2u];
            [encoder setBuffer:dofBuffer offset:0u atIndex:3u];
            [encoder setBuffer:bodyBuffer offset:0u atIndex:4u];
            [encoder setBuffer:pointDispatchBuffer
                         offset:articulationIndex *
                             sizeof(
                                 MRArticulatedOperatorDispatchGPU
                             )
                        atIndex:5u];
            [encoder setBuffer:qBuffer
                         offset:model.articulations[
                             articulationIndex
                         ].qOffset * sizeof(float)
                        atIndex:6u];
            [encoder setBuffer:pointQueryBuffer
                         offset:slice.queryOffset *
                             sizeof(
                                 MRArticulatedPointImpulseGPU
                             )
                        atIndex:7u];
            [encoder setBuffer:bodyPoseBuffer
                         offset:arena.bodyPoseOffset *
                             sizeof(MRArticulatedBodyPoseGPU)
                        atIndex:8u];
            [encoder setBuffer:pointWorldBuffer
                         offset:arena.pointWorldOffset *
                             sizeof(MRArticulatedPointWorldGPU)
                        atIndex:9u];
            [encoder setBuffer:emptyMassBuffer
                         offset:0u atIndex:10u];
            [encoder setBuffer:pointJacobianBuffer
                         offset:slice.jacobianOffset *
                             sizeof(float)
                        atIndex:11u];
            [encoder setBuffer:pointGeneralizedImpulseBuffer
                         offset:arena.generalizedOffset *
                             sizeof(float)
                        atIndex:12u];
            [encoder setBuffer:pointDeltaVelocityBuffer
                         offset:arena.generalizedOffset *
                             sizeof(float)
                        atIndex:13u];
            [encoder setBuffer:pointStatusBuffer
                         offset:arena.statusOffset *
                             sizeof(
                                 MRArticulatedOperatorStatusGPU
                             )
                        atIndex:14u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    input.environmentCount,
                    1u,
                    1u
                )
                threadsPerThreadgroup:MTLSizeMake(
                    kWaveWidth,
                    1u,
                    1u
                )];
        }
        const std::array pointProducts{
            pointWorldBuffer,
            pointJacobianBuffer,
            pointStatusBuffer,
        };
        [encoder
            memoryBarrierWithResources:pointProducts.data()
                                 count:pointProducts.size()];

        [encoder setComputePipelineState:pipelines[1]];
        const std::array assembleBuffers{
            dispatchBuffer, contactBuffer, endpointBuffer,
            sliceBuffer, pointJacobianBuffer,
            sceneBodyBuffer, sceneOffsetBuffer, jacobianBuffer,
        };
        for (NSUInteger index = 0u;
             index < assembleBuffers.size();
             ++index) {
            [encoder setBuffer:assembleBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.totalNv,
                layout.dispatch.rowCount,
                input.environmentCount
            )
            threadsPerThreadgroup:MTLSizeMake(8u, 4u, 1u)];

        [encoder setComputePipelineState:pipelines[2]];
        const std::array packBuffers{
            dispatchBuffer, freeArticulationBuffer,
            sceneBodyBuffer, sceneOffsetBuffer,
            packedFreeBuffer,
        };
        for (NSUInteger index = 0u;
             index < packBuffers.size();
             ++index) {
            [encoder setBuffer:packBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.totalNv,
                input.environmentCount,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        if (layout.dispatch.staticEqualityRowCount != 0u) {
            [encoder setComputePipelineState:pipelines[3]];
            [encoder setBuffer:dispatchBuffer
                         offset:0u atIndex:0u];
            [encoder setBuffer:equalityJacobianBuffer
                         offset:0u atIndex:1u];
            [encoder setBuffer:jacobianBuffer
                         offset:0u atIndex:2u];
            [encoder
                dispatchThreads:MTLSizeMake(
                    layout.dispatch.totalNv,
                    layout.dispatch.staticEqualityRowCount,
                    input.environmentCount
                )
                threadsPerThreadgroup:MTLSizeMake(
                    32u, 1u, 1u
                )];
        }
        if (authoredEqualityCount != 0u) {
            [encoder setComputePipelineState:pipelines[4]];
            const std::array pointEqualityBuffers{
                dispatchBuffer, pointEqualityBuffer,
                pointEqualityEndpointBuffer, sliceBuffer,
                pointJacobianBuffer, pointWorldBuffer,
                equalityKindBuffer, sceneBodyBuffer,
                sceneOffsetBuffer, jacobianBuffer,
            };
            for (NSUInteger index = 0u;
                 index < pointEqualityBuffers.size();
                 ++index) {
                [encoder
                    setBuffer:pointEqualityBuffers[index]
                       offset:0u
                      atIndex:index];
            }
            [encoder
                dispatchThreads:MTLSizeMake(
                    layout.dispatch.totalNv,
                    3u * authoredEqualityCount,
                    input.environmentCount
                )
                threadsPerThreadgroup:MTLSizeMake(
                    32u, 1u, 1u
                )];
        }
        const std::array rowProducts{
            jacobianBuffer,
            packedFreeBuffer,
        };
        [encoder
            memoryBarrierWithResources:rowProducts.data()
                                 count:rowProducts.size()];

        [encoder setComputePipelineState:pipelines[5]];
        const std::array inverseBuffers{
            worldBuffer, articulationBuffer, jointBuffer,
            dofBuffer, bodyBuffer, inverseDispatchBuffer,
            qBuffer, jacobianBuffer, responseBuffer,
            inverseStatusBuffer,
            scheduleArticulationBuffer,
            scheduleLevelBuffer,
            scheduleReductionBuffer,
            scheduleLevelBodyBuffer,
            scheduleParentBuffer,
            scheduleInboundBuffer,
            scheduleChildOffsetBuffer,
            scheduleChildIndexBuffer,
        };
        for (NSUInteger index = 0u;
             index < inverseBuffers.size();
             ++index) {
            [encoder setBuffer:inverseBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                input.environmentCount,
                layout.inverseMassDispatches.size(),
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth, 1u, 1u
            )];

        [encoder setComputePipelineState:pipelines[6]];
        const std::array sceneResponseBuffers{
            dispatchBuffer, jacobianBuffer, sceneBodyBuffer,
            sceneOffsetBuffer, responseBuffer,
        };
        for (NSUInteger index = 0u;
             index < sceneResponseBuffers.size();
             ++index) {
            [encoder setBuffer:sceneResponseBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.totalNv,
                layout.dispatch.responseRowCount,
                input.environmentCount
            )
            threadsPerThreadgroup:MTLSizeMake(8u, 4u, 1u)];
        const std::array responseProducts{
            responseBuffer,
            inverseStatusBuffer,
        };
        [encoder
            memoryBarrierWithResources:responseProducts.data()
                                 count:responseProducts.size()];

        [encoder setComputePipelineState:pipelines[7]];
        const std::array equalityBuffers{
            dispatchBuffer, equalityRowBuffer,
            jacobianBuffer, responseBuffer,
            inverseStatusBuffer, packedFreeBuffer,
            projectedResponseBuffer, projectedFreeBuffer,
            equalityOperatorBuffer, equalityCouplingBuffer,
            equalityFreeImpulseBuffer, equalityTargetBuffer,
            equalityRegularizationBuffer,
            equalityStatusBuffer,
        };
        for (NSUInteger index = 0u;
             index < equalityBuffers.size();
             ++index) {
            [encoder setBuffer:equalityBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                input.environmentCount,
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth, 1u, 1u
            )];
        const std::array equalityProducts{
            projectedResponseBuffer,
            projectedFreeBuffer,
            equalityOperatorBuffer,
            equalityCouplingBuffer,
            equalityFreeImpulseBuffer,
            equalityTargetBuffer,
            equalityRegularizationBuffer,
            equalityStatusBuffer,
        };
        [encoder
            memoryBarrierWithResources:equalityProducts.data()
                                 count:equalityProducts.size()];

        [encoder setComputePipelineState:pipelines[8]];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:jacobianBuffer offset:0u atIndex:1u];
        [encoder setBuffer:projectedResponseBuffer
                     offset:0u atIndex:2u];
        [encoder setBuffer:delassusBuffer offset:0u atIndex:3u];
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.rowCount,
                layout.dispatch.rowCount,
                input.environmentCount
            )
            threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];

        [encoder setComputePipelineState:pipelines[9]];
        const std::array velocityBuffers{
            dispatchBuffer, contactBuffer, endpointBuffer,
            jacobianBuffer, projectedFreeBuffer, sceneBodyBuffer,
            sceneOffsetBuffer, freeContactBuffer,
        };
        for (NSUInteger index = 0u;
             index < velocityBuffers.size();
             ++index) {
            [encoder setBuffer:velocityBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.rowCount,
                input.environmentCount,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        const std::array contactProducts{
            delassusBuffer,
            freeContactBuffer,
        };
        [encoder
            memoryBarrierWithResources:contactProducts.data()
                                 count:contactProducts.size()];

        [encoder setComputePipelineState:pipelines[10]];
        const std::array matrixBuffers{
            dispatchBuffer, contactBuffer, delassusBuffer,
            qualityMatrixBuffer,
        };
        for (NSUInteger index = 0u;
             index < matrixBuffers.size();
             ++index) {
            [encoder setBuffer:matrixBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.rowCount,
                layout.dispatch.rowCount,
                input.environmentCount
            )
            threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];

        [encoder setComputePipelineState:pipelines[11]];
        const std::array vectorBuffers{
            dispatchBuffer, contactBuffer, freeContactBuffer,
            linearBuffer, warmBuffer, scaleBuffer,
        };
        for (NSUInteger index = 0u;
             index < vectorBuffers.size();
             ++index) {
            [encoder setBuffer:vectorBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreads:MTLSizeMake(
                layout.dispatch.rowCount,
                input.environmentCount,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        const std::array qualityInputs{
            qualityMatrixBuffer,
            linearBuffer,
            warmBuffer,
            scaleBuffer,
        };
        [encoder
            memoryBarrierWithResources:qualityInputs.data()
                                 count:qualityInputs.size()];

        [encoder setComputePipelineState:pipelines[12]];
        const std::array qualityBuffers{
            qualityDispatchBuffer, qualityMatrixBuffer,
            linearBuffer, warmBuffer, compactImpulseBuffer,
            qualityStatusBuffer,
        };
        for (NSUInteger index = 0u;
             index < qualityBuffers.size();
             ++index) {
            [encoder setBuffer:qualityBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                input.environmentCount,
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth, 1u, 1u
            )];
        const std::array qualityProducts{
            compactImpulseBuffer,
            qualityStatusBuffer,
        };
        [encoder
            memoryBarrierWithResources:qualityProducts.data()
                                 count:qualityProducts.size()];

        [encoder setComputePipelineState:finalizePipeline];
        const std::array finalizeBuffers{
            dispatchBuffer, pointStatusBuffer,
            inverseStatusBuffer, qualityStatusBuffer,
            scaleBuffer, compactImpulseBuffer,
            packedFreeBuffer, projectedResponseBuffer,
            delassusBuffer,
            physicalImpulseBuffer, nextVelocityBuffer,
            statusBuffer,
            projectedFreeBuffer, jacobianBuffer,
            equalityTargetBuffer,
            equalityRegularizationBuffer,
            equalityFreeImpulseBuffer,
            equalityCouplingBuffer, equalityImpulseBuffer,
            equalityStatusBuffer,
        };
        for (NSUInteger index = 0u;
             index < finalizeBuffers.size();
             ++index) {
            [encoder setBuffer:finalizeBuffers[index]
                        offset:0u atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                input.environmentCount,
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth, 1u, 1u
            )];
        [encoder endEncoding];

        diagnostics.dispatched = true;
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    metalCommandFailure,
                "Metal contact graph failed: " +
                    errorString(command.error)
            );
        }

        MetalMultiArticulatedContactResult staged;
        staged.layout = layout;
        staged.nextVelocity.resize(
            layout.packedVelocityElements
        );
        staged.impulses.resize(layout.rowElements);
        staged.delassus.resize(layout.delassusElements);
        staged.freeContactVelocity.resize(layout.rowElements);
        staged.equalityImpulses.resize(
            layout.equalityImpulseElements
        );
        staged.statuses.resize(input.environmentCount);
        staged.equalityStatuses.resize(
            input.environmentCount
        );
        staged.pointStatuses.resize(pointStatusElements);
        staged.inverseMassStatuses.resize(
            inverseStatusElements
        );
        staged.qualityStatuses.resize(
            input.environmentCount
        );
        copyBuffer(staged.nextVelocity, nextVelocityBuffer);
        copyBuffer(staged.impulses, physicalImpulseBuffer);
        copyBuffer(staged.delassus, delassusBuffer);
        copyBuffer(
            staged.freeContactVelocity,
            freeContactBuffer
        );
        copyBuffer(
            staged.equalityImpulses,
            equalityImpulseBuffer
        );
        copyBuffer(staged.statuses, statusBuffer);
        copyBuffer(
            staged.equalityStatuses,
            equalityStatusBuffer
        );
        copyBuffer(staged.pointStatuses, pointStatusBuffer);
        copyBuffer(
            staged.inverseMassStatuses,
            inverseStatusBuffer
        );
        copyBuffer(
            staged.qualityStatuses,
            qualityStatusBuffer
        );

        bool failed = false;
        for (std::size_t environment = 0u;
             environment < staged.statuses.size();
             ++environment) {
            const MRMultiContactStatusGPU& status =
                staged.statuses[environment];
            if (status.environment != environment ||
                status.code >
                    MR_MULTI_CONTACT_EQUALITY_FAILED ||
                !finite(status.diagnostics)) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        internalFailure,
                    "GPU returned malformed contact status"
                );
            }
            const MRMultiContactEqualityStatusGPU&
                equalityStatus =
                    staged.equalityStatuses[environment];
            if (equalityStatus.environment != environment ||
                equalityStatus.rowCount !=
                    layout.dispatch.equalityRowCount ||
                equalityStatus.code >
                    MR_MULTI_CONTACT_EQUALITY_RESIDUAL_FAILED ||
                !finite(equalityStatus.diagnostics)) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedContactStatus::
                        internalFailure,
                    "GPU returned malformed equality status"
                );
            }
            if (status.code != MR_MULTI_CONTACT_SUCCESS &&
                !failed) {
                failed = true;
                diagnostics.firstFailingEnvironment =
                    static_cast<std::uint32_t>(environment);
                diagnostics.firstGPUStatusCode = status.code;
            }
        }
        if (!failed &&
            (!std::ranges::all_of(
                 staged.nextVelocity,
                 [](const float value) {
                     return finite(value);
                 }
             ) ||
             !std::ranges::all_of(
                 staged.impulses,
                 [](const float value) {
                     return finite(value);
                 }
             ) ||
             !std::ranges::all_of(
                 staged.delassus,
                 [](const float value) {
                     return finite(value);
                 }
             ) ||
             !std::ranges::all_of(
                 staged.freeContactVelocity,
                 [](const float value) {
                     return finite(value);
                 }
             ) ||
             !std::ranges::all_of(
                 staged.equalityImpulses,
                 [](const float value) {
                     return finite(value);
                 }
             ))) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    nonfiniteResult,
                "successful GPU contact payload is non-finite"
            );
        }
        output = std::move(staged);
        diagnostics.published = true;
        if (failed) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedContactStatus::
                    gpuEnvironmentFailure,
                "one or more contact environments rolled back"
            );
        }
        return diagnostics;
    }
}

MetalMultiArticulatedContactDiagnostics
solveMetalMultiArticulatedContacts(
    const EngineModel& model,
    const MetalMultiArticulatedContactInput& input,
    MetalMultiArticulatedContactResult& output,
    const MetalMultiArticulatedContactConfig& config
) {
    CompiledMetalMultiArticulatedContactProgram program;
    auto diagnostics =
        compileMetalMultiArticulatedContactProgram(
            model,
            program
        );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return solveMetalMultiArticulatedContactsImpl(
        program.model(),
        program.abaSchedule(),
        program.generalizedEqualityJacobian(),
        input,
        output,
        config
    );
}

MetalMultiArticulatedContactDiagnostics
solveMetalMultiArticulatedContacts(
    const CompiledMetalMultiArticulatedContactProgram& program,
    const MetalMultiArticulatedContactInput& input,
    MetalMultiArticulatedContactResult& output,
    const MetalMultiArticulatedContactConfig& config
) {
    MetalMultiArticulatedContactDiagnostics diagnostics;
    if (!program.valid()) {
        return reject(
            std::move(diagnostics),
            MetalMultiArticulatedContactStatus::invalidModel,
            "compiled Metal contact program is invalid"
        );
    }
    return solveMetalMultiArticulatedContactsImpl(
        program.model(),
        program.abaSchedule(),
        program.generalizedEqualityJacobian(),
        input,
        output,
        config
    );
}

const char* metalMultiArticulatedContactStatusName(
    const MetalMultiArticulatedContactStatus status
) noexcept {
    switch (status) {
    case MetalMultiArticulatedContactStatus::success:
        return "success";
    case MetalMultiArticulatedContactStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalMultiArticulatedContactStatus::invalidModel:
        return "invalid_model";
    case MetalMultiArticulatedContactStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalMultiArticulatedContactStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalMultiArticulatedContactStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalMultiArticulatedContactStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalMultiArticulatedContactStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalMultiArticulatedContactStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalMultiArticulatedContactStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalMultiArticulatedContactStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalMultiArticulatedContactStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalMultiArticulatedContactStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalMultiArticulatedContactStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalMultiArticulatedContactStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalMultiArticulatedContactStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalMultiArticulatedContactStatus::nonfiniteResult:
        return "nonfinite_result";
    case MetalMultiArticulatedContactStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
