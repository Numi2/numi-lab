#include "metalrobo/ArticulatedCollision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr double kTiny = 1.0e-14;
constexpr std::uint32_t kKnownConstraintFlags =
    MR_CONSTRAINT_FLAG_NEW_IMPACT |
    MR_CONSTRAINT_FLAG_WARM_STARTED |
    MR_CONSTRAINT_FLAG_DISABLED;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct EndpointBinding {
    bool articulated = false;
    std::uint32_t modelBody = MR_INVALID_INDEX;
};

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {
        left.x - right.x,
        left.y - right.y,
        left.z - right.z,
    };
}

Vec3 operator-(const Vec3 value) {
    return {-value.x, -value.y, -value.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale,
    };
}

Vec3 operator/(const Vec3 value, const double scale) {
    return value * (1.0 / scale);
}

double dot(const Vec3 left, const Vec3 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

std::array<double, 3> array(const Vec3 value) {
    return {value.x, value.y, value.z};
}

std::optional<Quaternion> checkedQuaternion(const mr_float4 value) {
    if (!finite(value)) {
        return std::nullopt;
    }
    Quaternion result{value.x, value.y, value.z, value.w};
    const double squared =
        result.x * result.x +
        result.y * result.y +
        result.z * result.z +
        result.w * result.w;
    if (!(squared > kTiny) ||
        std::abs(squared - 1.0) > 2.0e-4) {
        return std::nullopt;
    }
    const double inverseNorm = 1.0 / std::sqrt(squared);
    result.x *= inverseNorm;
    result.y *= inverseNorm;
    result.z *= inverseNorm;
    result.w *= inverseNorm;
    return result;
}

Quaternion conjugate(const Quaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

Vec3 rotate(const Quaternion quaternion, const Vec3 value) {
    const Vec3 vector{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    const Vec3 twiceCross = cross(vector, value) * 2.0;
    return
        value +
        twiceCross * quaternion.w +
        cross(vector, twiceCross);
}

Vec3 inverseRotate(
    const Quaternion quaternion,
    const Vec3 value
) {
    return rotate(conjugate(quaternion), value);
}

std::pair<Vec3, Vec3> contactBasis(const Vec3 unitNormal) {
    const Vec3 absolute{
        std::abs(unitNormal.x),
        std::abs(unitNormal.y),
        std::abs(unitNormal.z),
    };
    Vec3 reference{};
    if (absolute.x <= absolute.y && absolute.x <= absolute.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absolute.y <= absolute.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    const Vec3 tangent = cross(reference, unitNormal);
    const double tangentLength = norm(tangent);
    if (!(tangentLength > kTiny) || !finite(tangentLength)) {
        return {};
    }
    const Vec3 tangentU = tangent / tangentLength;
    return {tangentU, cross(unitNormal, tangentU)};
}

bool validState(const MRBodyStateGPU& state) {
    return
        state.flagsAndIndices[0] <= MR_MOTION_DYNAMIC &&
        finite(state.position) &&
        checkedQuaternion(state.orientation).has_value() &&
        finite(state.linearVelocityAndInverseMass) &&
        finite(state.angularVelocity) &&
        state.linearVelocityAndInverseMass.w >= 0.0f &&
        (
            state.flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
            state.linearVelocityAndInverseMass.w > 0.0f
        );
}

bool validConstraint(const MRContactConstraintGPU& contact) {
    const Vec3 normal = xyz(contact.normal);
    const Vec3 tangent = xyz(contact.tangent);
    const double normalSquared = dot(normal, normal);
    const double tangentSquared = dot(tangent, tangent);
    return
        (contact.flags & ~kKnownConstraintFlags) == 0u &&
        finite(contact.pointAndSeparation) &&
        finite(contact.normal) &&
        finite(contact.tangent) &&
        finite(contact.friction) &&
        finite(contact.response) &&
        finite(contact.targetVelocityAndPreSolveNormal) &&
        finite(contact.impulses) &&
        std::abs(normalSquared - 1.0) <= 2.0e-4 &&
        std::abs(tangentSquared - 1.0) <= 2.0e-4 &&
        std::abs(dot(normal, tangent)) <= 2.0e-4 &&
        contact.friction.x >= 0.0f &&
        contact.friction.y >= 0.0f &&
        contact.friction.x >= contact.friction.y &&
        contact.friction.z >= 0.0f &&
        contact.friction.w >= 0.0f &&
        contact.response.x >= 0.0f &&
        contact.response.x <= 1.0f &&
        contact.response.y >= 0.0f &&
        contact.response.z >= 0.0f &&
        contact.response.w >= 0.0f;
}

Vec3 pointVelocity(
    const MRBodyStateGPU& state,
    const Vec3 point
) {
    if (state.flagsAndIndices[0] == MR_MOTION_STATIC) {
        return {};
    }
    return
        xyz(state.linearVelocityAndInverseMass) +
        cross(
            xyz(state.angularVelocity),
            point - xyz(state.position)
        );
}

ArticulatedCollisionResult failure(
    ArticulatedCollisionDiagnostics diagnostics,
    const MRStepStatusCode code,
    const ArticulatedCollisionFailure reason,
    const std::uint32_t constraintIndex = MR_INVALID_INDEX
) {
    diagnostics.code = code;
    diagnostics.failure = reason;
    diagnostics.failedConstraintIndex = constraintIndex;
    return {diagnostics, {}, {}, {}};
}

std::optional<ArticulatedCollisionFailure> bindEndpoint(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const std::uint32_t articulationIndex,
    const MRBodyStateGPU& state,
    EndpointBinding& binding
) {
    if (!validState(state)) {
        return ArticulatedCollisionFailure::invalidBodyBinding;
    }
    if (state.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return std::nullopt;
    }
    const std::uint32_t boundArticulation =
        state.flagsAndIndices[1];
    const std::uint32_t boundBody = state.flagsAndIndices[2];
    if (boundArticulation == MR_INVALID_INDEX ||
        boundBody == MR_INVALID_INDEX) {
        return ArticulatedCollisionFailure::unboundDynamicBody;
    }
    if (boundArticulation != articulationIndex) {
        return ArticulatedCollisionFailure::crossArticulationContact;
    }
    if (boundBody < articulation.firstBody ||
        boundBody >= articulation.firstBody + articulation.bodyCount ||
        boundBody >= model.bodies.size() ||
        model.bodies[boundBody].articulationIndex !=
            articulationIndex) {
        return ArticulatedCollisionFailure::invalidBodyBinding;
    }
    binding.articulated = true;
    binding.modelBody = boundBody;
    return std::nullopt;
}

MRStepStatusCode codeForBindingFailure(
    const ArticulatedCollisionFailure reason
) {
    return
        reason == ArticulatedCollisionFailure::invalidBodyBinding
        ? MR_STEP_NONFINITE_INPUT
        : MR_STEP_UNSUPPORTED;
}

double normalTargetVelocity(
    const MRContactConstraintGPU& contact,
    const Vec3 normal,
    const ContactSolverConfig& config
) {
    const double penetration = std::min(
        static_cast<double>(contact.pointAndSeparation.w) +
            config.penetrationSlop,
        0.0
    );
    const double positional = std::min(
        config.maxDepenetrationVelocity,
        -config.errorReduction * penetration / config.timestep
    );
    double restitution = 0.0;
    if ((contact.flags & MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u &&
        contact.targetVelocityAndPreSolveNormal.w <
            -contact.response.y) {
        restitution =
            -static_cast<double>(contact.response.x) *
            contact.targetVelocityAndPreSolveNormal.w;
    }
    return
        dot(
            xyz(contact.targetVelocityAndPreSolveNormal),
            normal
        ) +
        std::max(positional, restitution);
}

} // namespace

ArticulatedCollisionResult adaptArticulatedContactConstraints(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const MRContactConstraintGPU> constraints,
    const std::span<const MRBodyStateGPU> bodyStates,
    const ArticulatedCollisionAdapterConfig& config
) {
    ArticulatedCollisionDiagnostics diagnostics;
    diagnostics.articulationIndex = articulationIndex;
    diagnostics.inputConstraintCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                constraints.size(),
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    const double timestepSquared =
        config.contact.timestep * config.contact.timestep;
    if (articulationIndex >= model.articulations.size() ||
        constraints.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        !(config.contact.timestep > 0.0) ||
        !finite(config.contact.timestep) ||
        !(timestepSquared > 0.0) ||
        !finite(timestepSquared) ||
        config.contact.errorReduction < 0.0 ||
        !finite(config.contact.errorReduction) ||
        config.contact.penetrationSlop < 0.0 ||
        !finite(config.contact.penetrationSlop) ||
        config.contact.maxDepenetrationVelocity < 0.0 ||
        !finite(config.contact.maxDepenetrationVelocity) ||
        !(config.qualityTangentialRegularization > 0.0) ||
        !finite(config.qualityTangentialRegularization)) {
        return failure(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedCollisionFailure::invalidConfiguration
        );
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if (articulation.firstBody > model.bodies.size() ||
        articulation.bodyCount >
            model.bodies.size() - articulation.firstBody ||
        articulation.bodyCount == 0u) {
        return failure(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedCollisionFailure::invalidConfiguration
        );
    }

    std::uint64_t activeCount = 0u;
    for (std::size_t constraintIndex = 0u;
         constraintIndex < constraints.size();
         ++constraintIndex) {
        const MRContactConstraintGPU& contact =
            constraints[constraintIndex];
        if ((contact.flags & ~kKnownConstraintFlags) != 0u) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidConstraint,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) == 0u) {
            ++activeCount;
        }
    }
    diagnostics.requiredContactCount =
        static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                activeCount,
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    if (activeCount > config.contactCapacity) {
        return failure(
            diagnostics,
            MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW,
            ArticulatedCollisionFailure::capacityOverflow
        );
    }

    std::vector<ArticulatedContact> adapted;
    std::vector<std::uint32_t> sourceIndices;
    adapted.reserve(static_cast<std::size_t>(activeCount));
    sourceIndices.reserve(static_cast<std::size_t>(activeCount));

    for (std::size_t constraintIndex = 0u;
         constraintIndex < constraints.size();
         ++constraintIndex) {
        const MRContactConstraintGPU& contact =
            constraints[constraintIndex];
        if ((contact.flags & MR_CONSTRAINT_FLAG_DISABLED) != 0u) {
            continue;
        }
        if (contact.bodyA >= bodyStates.size() ||
            contact.bodyB >= bodyStates.size() ||
            !validConstraint(contact)) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidConstraint,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if (contact.bodyA == contact.bodyB) {
            return failure(
                diagnostics,
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::invalidConstraint,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if (contact.friction.x != contact.friction.y ||
            contact.friction.z != 0.0f ||
            contact.friction.w != 0.0f ||
            contact.response.w != 0.0f ||
            contact.impulses.w != 0.0f) {
            return failure(
                diagnostics,
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::
                    unsupportedContactSemantics,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        const MRBodyStateGPU& stateA = bodyStates[contact.bodyA];
        const MRBodyStateGPU& stateB = bodyStates[contact.bodyB];
        EndpointBinding bindingA;
        EndpointBinding bindingB;
        if (const auto bindingFailure = bindEndpoint(
                model,
                articulation,
                articulationIndex,
                stateA,
                bindingA
            );
            bindingFailure.has_value()) {
            return failure(
                diagnostics,
                codeForBindingFailure(*bindingFailure),
                *bindingFailure,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if (const auto bindingFailure = bindEndpoint(
                model,
                articulation,
                articulationIndex,
                stateB,
                bindingB
            );
            bindingFailure.has_value()) {
            return failure(
                diagnostics,
                codeForBindingFailure(*bindingFailure),
                *bindingFailure,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if (!bindingA.articulated && !bindingB.articulated) {
            return failure(
                diagnostics,
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::invalidBodyBinding,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        if (bindingA.articulated && bindingB.articulated &&
            bindingA.modelBody == bindingB.modelBody) {
            return failure(
                diagnostics,
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::invalidConstraint,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        const Vec3 point = xyz(contact.pointAndSeparation);
        const Vec3 inputNormal = xyz(contact.normal);
        const double normalLength = norm(inputNormal);
        if (!(normalLength > kTiny) || !finite(normalLength)) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidConstraint,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }
        const Vec3 originalNormal = inputNormal / normalLength;
        const auto [originalTangentU, originalTangentV] =
            contactBasis(originalNormal);
        if (!finite(originalTangentU) ||
            !finite(originalTangentV)) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedCollisionFailure::nonfiniteResult,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        const bool swapped = !bindingA.articulated;
        const Vec3 adaptedNormal =
            swapped ? -originalNormal : originalNormal;
        const auto [adaptedTangentU, adaptedTangentV] =
            contactBasis(adaptedNormal);
        if (!finite(adaptedTangentU) ||
            !finite(adaptedTangentV)) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedCollisionFailure::nonfiniteResult,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        const Vec3 surfaceTarget =
            xyz(contact.targetVelocityAndPreSolveNormal);
        const double targetNormal = normalTargetVelocity(
            contact,
            originalNormal,
            config.contact
        );
        const Vec3 physicalTarget =
            surfaceTarget +
            originalNormal * (
                targetNormal -
                dot(surfaceTarget, originalNormal)
            );
        Vec3 nonDynamicRelative{};
        if (!bindingA.articulated) {
            nonDynamicRelative =
                nonDynamicRelative -
                pointVelocity(stateA, point);
        }
        if (!bindingB.articulated) {
            nonDynamicRelative =
                nonDynamicRelative +
                pointVelocity(stateB, point);
        }
        diagnostics.maximumKinematicTargetCompensation = std::max(
            diagnostics.maximumKinematicTargetCompensation,
            norm(nonDynamicRelative)
        );
        const Vec3 dynamicTarget =
            physicalTarget - nonDynamicRelative;
        const Vec3 adaptedTarget =
            swapped ? -dynamicTarget : dynamicTarget;

        const Vec3 originalImpulse =
            originalNormal * contact.impulses.x +
            originalTangentU * contact.impulses.y +
            originalTangentV * contact.impulses.z;
        const Vec3 adaptedImpulse =
            swapped ? -originalImpulse : originalImpulse;

        const MRBodyStateGPU& articulatedStateA =
            swapped ? stateB : stateA;
        const EndpointBinding& articulatedBindingA =
            swapped ? bindingB : bindingA;
        const auto articulatedRotationA =
            checkedQuaternion(articulatedStateA.orientation);
        if (!articulatedRotationA.has_value()) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidBodyBinding,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        ArticulatedContact result;
        result.bodyA = articulatedBindingA.modelBody;
        result.localPointA = array(inverseRotate(
            *articulatedRotationA,
            point - xyz(articulatedStateA.position)
        ));
        result.normal = array(adaptedNormal);
        result.tangentU = array(adaptedTangentU);
        result.tangentV = array(adaptedTangentV);
        result.targetVelocity = {
            dot(adaptedTarget, adaptedNormal),
            dot(adaptedTarget, adaptedTangentU),
            dot(adaptedTarget, adaptedTangentV),
        };
        const double normalRegularization =
            static_cast<double>(contact.response.z) /
                timestepSquared +
            config.qualityTangentialRegularization;
        result.regularization = {
            normalRegularization,
            config.qualityTangentialRegularization,
            config.qualityTangentialRegularization,
        };
        result.warmImpulse = {
            dot(adaptedImpulse, adaptedNormal),
            dot(adaptedImpulse, adaptedTangentU),
            dot(adaptedImpulse, adaptedTangentV),
        };
        result.friction = contact.friction.x;

        if (!swapped && bindingB.articulated) {
            const auto articulatedRotationB =
                checkedQuaternion(stateB.orientation);
            if (!articulatedRotationB.has_value()) {
                return failure(
                    diagnostics,
                    MR_STEP_NONFINITE_INPUT,
                    ArticulatedCollisionFailure::invalidBodyBinding,
                    static_cast<std::uint32_t>(constraintIndex)
                );
            }
            result.bodyB = bindingB.modelBody;
            result.localPointB = array(inverseRotate(
                *articulatedRotationB,
                point - xyz(stateB.position)
            ));
        } else {
            result.bodyB = kArticulatedStaticWorld;
            result.localPointB = array(point);
        }

        if (!std::ranges::all_of(
                result.targetVelocity,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !std::ranges::all_of(
                result.warmImpulse,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !std::ranges::all_of(
                result.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            ) ||
            !std::ranges::all_of(
                result.localPointA,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !std::ranges::all_of(
                result.localPointB,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !std::ranges::all_of(
                result.normal,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !std::ranges::all_of(
                result.tangentU,
                [](const double value) {
                    return finite(value);
                }
            ) ||
            !finite(result.friction) ||
            result.friction < 0.0
        ) {
            return failure(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedCollisionFailure::nonfiniteResult,
                static_cast<std::uint32_t>(constraintIndex)
            );
        }

        diagnostics.maximumNormalTargetVelocity = std::max(
            diagnostics.maximumNormalTargetVelocity,
            std::abs(result.targetVelocity[0])
        );
        if (swapped) {
            ++diagnostics.swappedEndpointCount;
        }
        adapted.push_back(result);
        sourceIndices.push_back(
            static_cast<std::uint32_t>(constraintIndex)
        );
    }

    diagnostics.adaptedContactCount =
        static_cast<std::uint32_t>(adapted.size());
    ArticulatedCollisionResult result;
    result.diagnostics = diagnostics;
    result.contacts = std::move(adapted);
    result.sourceConstraintIndices = std::move(sourceIndices);
    return result;
}

ArticulatedCollisionResult adaptEvaluatedArticulatedContacts(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const ConstraintIREvaluationView& semantics,
    const std::span<const MRBodyStateGPU> bodyStates,
    const std::uint32_t contactCapacity
) {
    ArticulatedCollisionDiagnostics diagnostics;
    diagnostics.articulationIndex = articulationIndex;
    diagnostics.inputConstraintCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            semantics.blocks.size(),
            std::numeric_limits<std::uint32_t>::max()
        ));
    diagnostics.semanticFingerprint =
        semantics.semanticFingerprint;

    const auto failBlock = [&diagnostics](
        const MRStepStatusCode code,
        const ArticulatedCollisionFailure reason,
        const std::uint32_t blockIndex = MR_INVALID_INDEX
    ) {
        ArticulatedCollisionResult result;
        diagnostics.code = code;
        diagnostics.failure = reason;
        diagnostics.failedBlockIndex = blockIndex;
        result.diagnostics = diagnostics;
        return result;
    };

    const ConstraintIRDiagnostics semanticDiagnostics =
        validateConstraintIREvaluationView(semantics);
    if (!semanticDiagnostics.succeeded()) {
        return failBlock(
            MR_STEP_NONFINITE_INPUT,
            ArticulatedCollisionFailure::invalidConstraint,
            semanticDiagnostics.blockIndex
        );
    }
    if (articulationIndex >= model.articulations.size()) {
        return failBlock(
            MR_STEP_NONFINITE_INPUT,
            ArticulatedCollisionFailure::invalidConfiguration
        );
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if (articulation.firstBody > model.bodies.size() ||
        articulation.bodyCount >
            model.bodies.size() - articulation.firstBody ||
        articulation.bodyCount == 0u) {
        return failBlock(
            MR_STEP_NONFINITE_INPUT,
            ArticulatedCollisionFailure::invalidConfiguration
        );
    }

    std::uint64_t activeCount = 0u;
    for (std::uint32_t blockIndex = 0u;
         blockIndex < semantics.blocks.size();
         ++blockIndex) {
        const ConstraintIRBlock& block =
            semantics.blocks[blockIndex];
        if ((block.flags & constraintIRBlockDisabled) != 0u) {
            continue;
        }
        if (block.type != MR_CONSTRAINT_CONTACT) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::
                    unsupportedContactSemantics,
                blockIndex
            );
        }
        ++activeCount;
    }
    diagnostics.requiredContactCount =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            activeCount,
            std::numeric_limits<std::uint32_t>::max()
        ));
    if (activeCount > contactCapacity) {
        return failBlock(
            MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW,
            ArticulatedCollisionFailure::capacityOverflow
        );
    }

    std::vector<ArticulatedContact> adapted;
    std::vector<std::uint32_t> sourceBlocks;
    adapted.reserve(static_cast<std::size_t>(activeCount));
    sourceBlocks.reserve(static_cast<std::size_t>(activeCount));

    for (std::uint32_t blockIndex = 0u;
         blockIndex < semantics.blocks.size();
         ++blockIndex) {
        const ConstraintIRBlock& block =
            semantics.blocks[blockIndex];
        if ((block.flags & constraintIRBlockDisabled) != 0u) {
            continue;
        }
        const EvaluatedConstraintIRCone& cone =
            semantics.cones[block.coneIndex];
        if (block.dimension != 3u ||
            cone.effectiveFrictionU !=
                cone.effectiveFrictionV ||
            cone.rollingLength != 0.0F ||
            cone.torsionalLength != 0.0F ||
            cone.adhesionImpulse != 0.0F ||
            cone.maximumNormalImpulse != 0.0F) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::
                    unsupportedContactSemantics,
                blockIndex
            );
        }
        const ConstraintIREndpoint& endpointA =
            semantics.endpoints[block.endpointOffset];
        const ConstraintIREndpoint& endpointB =
            semantics.endpoints[block.endpointOffset + 1u];
        if (endpointA.jacobianKind !=
                constraintIRJacobianWorldPoint ||
            endpointB.jacobianKind !=
                constraintIRJacobianWorldPoint ||
            endpointA.objectIndex >= bodyStates.size() ||
            endpointB.objectIndex >= bodyStates.size()) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::
                    unsupportedContactSemantics,
                blockIndex
            );
        }

        const MRBodyStateGPU& stateA =
            bodyStates[endpointA.objectIndex];
        const MRBodyStateGPU& stateB =
            bodyStates[endpointB.objectIndex];
        EndpointBinding bindingA;
        EndpointBinding bindingB;
        if (const auto bindingFailure = bindEndpoint(
                model,
                articulation,
                articulationIndex,
                stateA,
                bindingA
            );
            bindingFailure.has_value()) {
            return failBlock(
                codeForBindingFailure(*bindingFailure),
                *bindingFailure,
                blockIndex
            );
        }
        if (const auto bindingFailure = bindEndpoint(
                model,
                articulation,
                articulationIndex,
                stateB,
                bindingB
            );
            bindingFailure.has_value()) {
            return failBlock(
                codeForBindingFailure(*bindingFailure),
                *bindingFailure,
                blockIndex
            );
        }
        if (!bindingA.articulated && !bindingB.articulated) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::invalidBodyBinding,
                blockIndex
            );
        }
        if (bindingA.articulated && bindingB.articulated &&
            bindingA.modelBody == bindingB.modelBody) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::invalidConstraint,
                blockIndex
            );
        }

        const EvaluatedConstraintIRRow& normalRow =
            semantics.rows[block.rowOffset];
        const EvaluatedConstraintIRRow& tangentURow =
            semantics.rows[block.rowOffset + 1u];
        const EvaluatedConstraintIRRow& tangentVRow =
            semantics.rows[block.rowOffset + 2u];
        if (!(normalRow.regularization > 0.0F) ||
            !(tangentURow.regularization > 0.0F) ||
            !(tangentVRow.regularization > 0.0F)) {
            return failBlock(
                MR_STEP_UNSUPPORTED,
                ArticulatedCollisionFailure::
                    unsupportedContactSemantics,
                blockIndex
            );
        }

        const Vec3 originalNormal = xyz(normalRow.direction);
        const Vec3 originalTangentU = xyz(tangentURow.direction);
        const Vec3 originalTangentV = xyz(tangentVRow.direction);
        const double handedness = dot(
            cross(originalNormal, originalTangentU),
            originalTangentV
        );
        if (!finite(handedness) ||
            std::abs(handedness - 1.0) > 6.0e-4) {
            return failBlock(
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidConstraint,
                blockIndex
            );
        }

        const Vec3 pointA = xyz(endpointA.anchor);
        const Vec3 pointB = xyz(endpointB.anchor);
        Vec3 nonDynamicRelative{};
        if (!bindingA.articulated) {
            nonDynamicRelative =
                nonDynamicRelative -
                pointVelocity(stateA, pointA);
        }
        if (!bindingB.articulated) {
            nonDynamicRelative =
                nonDynamicRelative +
                pointVelocity(stateB, pointB);
        }
        diagnostics.maximumKinematicTargetCompensation = std::max(
            diagnostics.maximumKinematicTargetCompensation,
            norm(nonDynamicRelative)
        );

        std::array<double, 3> dynamicTarget{
            static_cast<double>(normalRow.targetVelocity) -
                dot(nonDynamicRelative, originalNormal),
            static_cast<double>(tangentURow.targetVelocity) -
                dot(nonDynamicRelative, originalTangentU),
            static_cast<double>(tangentVRow.targetVelocity) -
                dot(nonDynamicRelative, originalTangentV),
        };
        std::array<double, 3> warmImpulse{
            semantics.warmImpulses[block.impulseOffset],
            semantics.warmImpulses[block.impulseOffset + 1u],
            semantics.warmImpulses[block.impulseOffset + 2u],
        };

        const bool swapped = !bindingA.articulated;
        if (swapped) {
            // n'=-n, u'=u, v'=n'x u'=-v and the desired relative
            // velocity/impulse vector also changes sign.
            dynamicTarget[1] = -dynamicTarget[1];
            warmImpulse[1] = -warmImpulse[1];
        }

        const MRBodyStateGPU& articulatedStateA =
            swapped ? stateB : stateA;
        const EndpointBinding& articulatedBindingA =
            swapped ? bindingB : bindingA;
        const Vec3 articulatedPointA =
            swapped ? pointB : pointA;
        const auto rotationA =
            checkedQuaternion(articulatedStateA.orientation);
        if (!rotationA.has_value()) {
            return failBlock(
                MR_STEP_NONFINITE_INPUT,
                ArticulatedCollisionFailure::invalidBodyBinding,
                blockIndex
            );
        }

        ArticulatedContact result;
        result.bodyA = articulatedBindingA.modelBody;
        result.localPointA = array(inverseRotate(
            *rotationA,
            articulatedPointA - xyz(articulatedStateA.position)
        ));
        result.normal = array(
            swapped ? -originalNormal : originalNormal
        );
        result.tangentU = array(originalTangentU);
        result.tangentV = array(
            swapped ? -originalTangentV : originalTangentV
        );
        result.targetVelocity = dynamicTarget;
        result.regularization = {
            normalRow.regularization,
            tangentURow.regularization,
            tangentVRow.regularization,
        };
        result.warmImpulse = warmImpulse;
        result.friction = cone.effectiveFrictionU;

        if (!swapped && bindingB.articulated) {
            const auto rotationB =
                checkedQuaternion(stateB.orientation);
            if (!rotationB.has_value()) {
                return failBlock(
                    MR_STEP_NONFINITE_INPUT,
                    ArticulatedCollisionFailure::invalidBodyBinding,
                    blockIndex
                );
            }
            result.bodyB = bindingB.modelBody;
            result.localPointB = array(inverseRotate(
                *rotationB,
                pointB - xyz(stateB.position)
            ));
        } else {
            result.bodyB = kArticulatedStaticWorld;
            result.localPointB = array(
                swapped ? pointA : pointB
            );
        }

        const auto finiteArray = [](const auto& values) {
            return std::ranges::all_of(
                values,
                [](const double value) {
                    return finite(value);
                }
            );
        };
        if (!finiteArray(result.localPointA) ||
            !finiteArray(result.localPointB) ||
            !finiteArray(result.normal) ||
            !finiteArray(result.tangentU) ||
            !finiteArray(result.targetVelocity) ||
            !finiteArray(result.regularization) ||
            !finiteArray(result.warmImpulse) ||
            !finite(result.friction) ||
            result.friction < 0.0) {
            return failBlock(
                MR_STEP_NONFINITE_RESULT,
                ArticulatedCollisionFailure::nonfiniteResult,
                blockIndex
            );
        }

        diagnostics.maximumNormalTargetVelocity = std::max(
            diagnostics.maximumNormalTargetVelocity,
            std::abs(result.targetVelocity[0])
        );
        if (swapped) {
            ++diagnostics.swappedEndpointCount;
        }
        adapted.push_back(result);
        sourceBlocks.push_back(blockIndex);
    }

    diagnostics.adaptedContactCount =
        static_cast<std::uint32_t>(adapted.size());
    ArticulatedCollisionResult result;
    result.diagnostics = diagnostics;
    result.contacts = std::move(adapted);
    result.sourceBlockIndices = std::move(sourceBlocks);
    return result;
}

} // namespace metalrobo
