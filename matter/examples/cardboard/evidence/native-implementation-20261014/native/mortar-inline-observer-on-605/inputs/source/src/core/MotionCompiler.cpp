#include "metalrobo/MotionCompiler.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

MotionCompileResult reject(
    const MotionCompileStatus status,
    std::string message
) {
    return {status, std::move(message)};
}

std::uint32_t uniqueIndex(
    const std::vector<std::string>& names,
    const std::string_view target
) {
    std::uint32_t result = MR_INVALID_INDEX;
    for (std::size_t index = 0u; index < names.size(); ++index) {
        if (names[index] != target) {
            continue;
        }
        if (result != MR_INVALID_INDEX) {
            return MR_INVALID_INDEX;
        }
        result = static_cast<std::uint32_t>(index);
    }
    return result;
}

std::array<double, 4> normalized(
    const std::array<double, 4> value
) {
    const double norm = std::sqrt(
        value[0] * value[0] + value[1] * value[1] +
        value[2] * value[2] + value[3] * value[3]
    );
    if (!(norm > 1.0e-12) || !std::isfinite(norm)) {
        return {0.0, 0.0, 0.0, 1.0};
    }
    return {
        value[0] / norm,
        value[1] / norm,
        value[2] / norm,
        value[3] / norm,
    };
}

std::array<double, 4> product(
    const std::array<double, 4> left,
    const std::array<double, 4> right
) {
    return {
        left[3] * right[0] + left[0] * right[3] +
            left[1] * right[2] - left[2] * right[1],
        left[3] * right[1] - left[0] * right[2] +
            left[1] * right[3] + left[2] * right[0],
        left[3] * right[2] + left[0] * right[1] -
            left[1] * right[0] + left[2] * right[3],
        left[3] * right[3] - left[0] * right[0] -
            left[1] * right[1] - left[2] * right[2],
    };
}

std::array<double, 3> rotate(
    const std::array<double, 4> quaternion,
    const std::array<double, 3> value
) {
    const std::array<double, 4> pure{
        value[0], value[1], value[2], 0.0,
    };
    const std::array<double, 4> inverse{
        -quaternion[0], -quaternion[1], -quaternion[2], quaternion[3],
    };
    const std::array<double, 4> result = product(
        product(quaternion, pure), inverse
    );
    return {result[0], result[1], result[2]};
}

std::array<double, 3> rotateInverse(
    const std::array<double, 4> quaternion,
    const std::array<double, 3> value
) {
    return rotate(
        {
            -quaternion[0], -quaternion[1],
            -quaternion[2], quaternion[3],
        },
        value
    );
}

void appendBodyFeature(
    const ArticulatedBodyKinematics& anchor,
    const ArticulatedBodyKinematics& body,
    std::vector<float>& features
) {
    const std::array<double, 3> delta{
        body.centerOfMassPosition[0] - anchor.centerOfMassPosition[0],
        body.centerOfMassPosition[1] - anchor.centerOfMassPosition[1],
        body.centerOfMassPosition[2] - anchor.centerOfMassPosition[2],
    };
    const std::array<double, 3> position = rotateInverse(
        anchor.orientation,
        delta
    );
    const std::array<double, 4> relative = normalized(product(
        {
            -anchor.orientation[0], -anchor.orientation[1],
            -anchor.orientation[2], anchor.orientation[3],
        },
        body.orientation
    ));
    const double x = relative[0];
    const double y = relative[1];
    const double z = relative[2];
    const double w = relative[3];
    const std::array<double, 9> values{
        position[0], position[1], position[2],
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
    };
    for (const double value : values) {
        features.push_back(static_cast<float>(value));
    }
}

} // namespace

MotionCompileResult compileInteractionMotionPack(
    const InteractionPack& interactions,
    const EngineModel& model,
    const InteractionMotionCompileConfig& config,
    MotionPack& output
) {
    if (!validInteractionPack(interactions) || config.id.empty() ||
        config.anchorBody.empty() || config.trackedBodies.empty() ||
        config.trackedBodies.size() >
            std::numeric_limits<std::uint32_t>::max() / 9u) {
        return reject(
            MotionCompileStatus::invalidInput,
            "interaction intent and motion feature contract must be valid"
        );
    }
    std::string modelReason;
    if (!model.valid(&modelReason) ||
        config.articulationIndex >= model.articulations.size()) {
        return reject(
            MotionCompileStatus::invalidModel,
            modelReason.empty() ? "articulation does not exist" : modelReason
        );
    }
    if (std::ranges::any_of(
            config.trackedBodies,
            [&](const std::string& name) {
                return name.empty() ||
                    std::ranges::count(config.trackedBodies, name) != 1;
            }
        )) {
        return reject(
            MotionCompileStatus::invalidInput,
            "tracked body identities must be nonempty and unique"
        );
    }

    const InteractionClip* selected = nullptr;
    for (const InteractionClip& clip : interactions.clips) {
        if (!config.clipId.empty() && clip.id != config.clipId) {
            continue;
        }
        if (selected != nullptr) {
            return reject(
                MotionCompileStatus::unresolvedClip,
                "interaction clip identity is ambiguous"
            );
        }
        selected = &clip;
    }
    if (selected == nullptr) {
        return reject(
            MotionCompileStatus::unresolvedClip,
            "interaction clip does not exist"
        );
    }

    const MRArticulationGPU& articulation =
        model.articulations[config.articulationIndex];
    if (articulation.rootType != MR_ROOT_FIXED &&
        articulation.rootType != MR_ROOT_FLOATING) {
        return reject(
            MotionCompileStatus::unsupportedTopology,
            "motion compilation requires a fixed or floating articulation"
        );
    }
    const auto inArticulation = [&](const std::uint32_t body) {
        return body >= articulation.firstBody &&
            body < articulation.firstBody + articulation.bodyCount;
    };
    const std::uint32_t anchorBody = uniqueIndex(
        model.bodyNames,
        config.anchorBody
    );
    if (anchorBody == MR_INVALID_INDEX || !inArticulation(anchorBody)) {
        return reject(
            MotionCompileStatus::unresolvedBody,
            "motion anchor is unresolved in the selected articulation"
        );
    }
    std::vector<std::uint32_t> trackedBodies;
    trackedBodies.reserve(config.trackedBodies.size());
    for (const std::string& name : config.trackedBodies) {
        const std::uint32_t body = uniqueIndex(model.bodyNames, name);
        if (body == MR_INVALID_INDEX || !inArticulation(body)) {
            return reject(
                MotionCompileStatus::unresolvedBody,
                "tracked body is unresolved in the selected articulation: " +
                    name
            );
        }
        trackedBodies.push_back(body);
    }

    std::vector<std::uint32_t> jointCoordinates;
    jointCoordinates.reserve(interactions.jointNames.size());
    for (const std::string& name : interactions.jointNames) {
        const std::uint32_t jointIndex = uniqueIndex(
            model.jointNames,
            name
        );
        if (jointIndex == MR_INVALID_INDEX ||
            jointIndex < articulation.firstJoint ||
            jointIndex >= articulation.firstJoint + articulation.jointCount) {
            return reject(
                MotionCompileStatus::unresolvedJoint,
                "interaction joint is unresolved: " + name
            );
        }
        const MRJointDescriptorGPU& joint = model.joints[jointIndex];
        if (joint.nq != 1u || joint.qOffset < articulation.qOffset ||
            joint.qOffset >= articulation.qOffset + articulation.nq) {
            return reject(
                MotionCompileStatus::unsupportedTopology,
                "interaction intent requires scalar joint coordinates: " +
                    name
            );
        }
        jointCoordinates.push_back(joint.qOffset - articulation.qOffset);
    }

    if (articulation.qOffset > model.defaultQ.size() ||
        articulation.nq > model.defaultQ.size() - articulation.qOffset ||
        articulation.vOffset > model.defaultV.size() ||
        articulation.nv > model.defaultV.size() - articulation.vOffset) {
        return reject(
            MotionCompileStatus::invalidModel,
            "default articulation state is incomplete"
        );
    }
    std::vector<double> q(
        model.defaultQ.begin() + articulation.qOffset,
        model.defaultQ.begin() + articulation.qOffset + articulation.nq
    );
    const std::vector<double> v(
        model.defaultV.begin() + articulation.vOffset,
        model.defaultV.begin() + articulation.vOffset + articulation.nv
    );
    std::vector<ArticulatedBodyKinematics> kinematics(
        articulation.bodyCount
    );

    MotionPack staged{
        .id = config.id,
        .sourceRepository = interactions.sourceRepository,
        .sourceRevision = interactions.sourceRevision,
        .license = interactions.license,
        .anchorBody = config.anchorBody,
        .trackedBodies = config.trackedBodies,
        .featureCount = static_cast<std::uint32_t>(
            9u * config.trackedBodies.size()
        ),
        .clips = {{
            .id = selected->id,
            .framesPerSecond = selected->framesPerSecond,
        }},
    };
    MotionClip& target = staged.clips.front();
    target.features.reserve(
        static_cast<std::size_t>(selected->frameCount) *
        staged.featureCount
    );

    const MRBodyPropertiesGPU& rootBody =
        model.bodies[articulation.rootBody];
    for (std::uint32_t frame = 0u; frame < selected->frameCount; ++frame) {
        if (articulation.rootType == MR_ROOT_FLOATING) {
            const std::size_t rootBase =
                static_cast<std::size_t>(frame) *
                kInteractionRootTargetCount;
            const std::array<double, 4> orientation = normalized({
                selected->rootTargets[rootBase + 3u],
                selected->rootTargets[rootBase + 4u],
                selected->rootTargets[rootBase + 5u],
                selected->rootTargets[rootBase + 6u],
            });
            const std::array<double, 3> centerOffset = rotate(
                orientation,
                {
                    rootBody.centerOfMass.x,
                    rootBody.centerOfMass.y,
                    rootBody.centerOfMass.z,
                }
            );
            q[0] = selected->rootTargets[rootBase + 0u] + centerOffset[0];
            q[1] = selected->rootTargets[rootBase + 1u] + centerOffset[1];
            q[2] = selected->rootTargets[rootBase + 2u] + centerOffset[2];
            q[3] = orientation[0];
            q[4] = orientation[1];
            q[5] = orientation[2];
            q[6] = orientation[3];
        }
        const std::size_t jointBase =
            static_cast<std::size_t>(frame) * interactions.jointNames.size();
        for (std::size_t joint = 0u;
             joint < jointCoordinates.size();
             ++joint) {
            q[jointCoordinates[joint]] =
                selected->jointTargets[jointBase + joint];
        }
        const ArticulatedDynamicsDiagnostics diagnostics =
            computeArticulatedBodyKinematics(
                model,
                config.articulationIndex,
                q,
                v,
                kinematics
            );
        if (!diagnostics.succeeded()) {
            return reject(
                MotionCompileStatus::kinematicsFailure,
                "native articulated kinematics failed at frame " +
                    std::to_string(frame) + " with status " +
                    std::to_string(
                        static_cast<std::uint32_t>(diagnostics.status)
                    )
            );
        }
        const ArticulatedBodyKinematics& anchor =
            kinematics[anchorBody - articulation.firstBody];
        for (const std::uint32_t body : trackedBodies) {
            appendBodyFeature(
                anchor,
                kinematics[body - articulation.firstBody],
                target.features
            );
        }
    }
    output = std::move(staged);
    return {};
}

const char* motionCompileStatusName(
    const MotionCompileStatus status
) noexcept {
    switch (status) {
    case MotionCompileStatus::success:
        return "success";
    case MotionCompileStatus::invalidInput:
        return "invalid_input";
    case MotionCompileStatus::invalidModel:
        return "invalid_model";
    case MotionCompileStatus::unresolvedClip:
        return "unresolved_clip";
    case MotionCompileStatus::unresolvedJoint:
        return "unresolved_joint";
    case MotionCompileStatus::unresolvedBody:
        return "unresolved_body";
    case MotionCompileStatus::unsupportedTopology:
        return "unsupported_topology";
    case MotionCompileStatus::kinematicsFailure:
        return "kinematics_failure";
    }
    return "unknown";
}

} // namespace metalrobo
