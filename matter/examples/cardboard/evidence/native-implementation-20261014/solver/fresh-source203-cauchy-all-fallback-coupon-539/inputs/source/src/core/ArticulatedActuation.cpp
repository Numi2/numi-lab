#include "metalrobo/ArticulatedActuation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <ranges>
#include <utility>

namespace metalrobo {
namespace {

constexpr mr_u32 kKnownDofFlags =
    MR_DOF_FLAG_ROOT |
    MR_DOF_FLAG_ACTUATED |
    MR_DOF_FLAG_POSITION_LIMIT |
    MR_DOF_FLAG_VELOCITY_LIMIT |
    MR_DOF_FLAG_EFFORT_LIMIT |
    MR_DOF_FLAG_DRIVE;

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool zero(const mr_float4 value) {
    return value.x == 0.0f &&
        value.y == 0.0f &&
        value.z == 0.0f &&
        value.w == 0.0f;
}

bool within(
    const mr_u32 offset,
    const mr_u32 count,
    const std::size_t size
) {
    return static_cast<std::size_t>(offset) <= size &&
        static_cast<std::size_t>(count) <=
            size - static_cast<std::size_t>(offset);
}

mr_u32 expectedQIndex(
    const MRJointDescriptorGPU& joint,
    const mr_u32 localDof
) {
    switch (joint.jointType) {
    case MR_JOINT_REVOLUTE:
    case MR_JOINT_PRISMATIC:
    case MR_JOINT_CONTINUOUS:
    case MR_JOINT_PLANAR:
    case MR_JOINT_FUNCTION_BASED:
        return joint.qOffset + localDof;
    case MR_JOINT_FREE:
        return localDof < 3u
            ? joint.qOffset + localDof
            : MR_INVALID_INDEX;
    case MR_JOINT_SPHERICAL:
    case MR_JOINT_FIXED:
    default:
        return MR_INVALID_INDEX;
    }
}

bool validNonRootDofMetadata(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const std::uint32_t articulationIndex,
    const mr_u32 globalV,
    const MRDofPropertiesGPU& dof
) {
    if (dof.articulationIndex != articulationIndex ||
        dof.vIndex != globalV ||
        dof.reserved0 != 0u ||
        dof.reserved1 != 0u ||
        (dof.flags & ~kKnownDofFlags) != 0u ||
        (dof.flags & MR_DOF_FLAG_ROOT) != 0u ||
        !finite(dof.limits) ||
        !finite(dof.drive) ||
        dof.drive.x < 0.0f ||
        dof.drive.y < 0.0f ||
        dof.drive.z < 0.0f ||
        dof.drive.w < 0.0f ||
        dof.jointIndex < articulation.firstJoint ||
        dof.jointIndex - articulation.firstJoint >=
            articulation.jointCount ||
        dof.jointIndex >= model.joints.size()) {
        return false;
    }

    const MRJointDescriptorGPU& joint =
        model.joints[dof.jointIndex];
    if (dof.localDof >= joint.nv ||
        joint.vOffset > globalV ||
        dof.localDof != globalV - joint.vOffset ||
        dof.qIndex != expectedQIndex(joint, dof.localDof) ||
        joint.qOffset < articulation.qOffset ||
        joint.qOffset - articulation.qOffset > articulation.nq ||
        joint.nq >
            articulation.nq -
                (joint.qOffset - articulation.qOffset) ||
        joint.vOffset < articulation.vOffset ||
        joint.vOffset - articulation.vOffset > articulation.nv ||
        joint.nv >
            articulation.nv -
                (joint.vOffset - articulation.vOffset)) {
        return false;
    }

    const bool actuated =
        (dof.flags & MR_DOF_FLAG_ACTUATED) != 0u;
    const bool positionLimited =
        (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
    const bool velocityLimited =
        (dof.flags & MR_DOF_FLAG_VELOCITY_LIMIT) != 0u;
    const bool effortLimited =
        (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u;
    const bool driven =
        (dof.flags & MR_DOF_FLAG_DRIVE) != 0u;

    return
        (actuated || (!effortLimited && !driven)) &&
        (!driven || actuated) &&
        (driven || dof.drive.x == 0.0f) &&
        (positionLimited
             ? (dof.qIndex != MR_INVALID_INDEX &&
                joint.jointType != MR_JOINT_CONTINUOUS &&
                dof.limits.x <= dof.limits.y)
             : (dof.limits.x == 0.0f &&
                dof.limits.y == 0.0f)) &&
        (velocityLimited
             ? dof.limits.z > 0.0f
             : dof.limits.z == 0.0f) &&
        (effortLimited
             ? dof.limits.w > 0.0f
             : dof.limits.w == 0.0f);
}

bool validRootDofMetadata(
    const MRDofPropertiesGPU& dof,
    const std::uint32_t articulationIndex,
    const MRArticulationGPU& articulation,
    const mr_u32 localV
) {
    const mr_u32 expectedQ =
        localV < 3u
            ? articulation.qOffset + localV
            : MR_INVALID_INDEX;
    return dof.articulationIndex == articulationIndex &&
        dof.jointIndex == MR_INVALID_INDEX &&
        dof.qIndex == expectedQ &&
        dof.vIndex == articulation.vOffset + localV &&
        dof.localDof == localV &&
        dof.flags == MR_DOF_FLAG_ROOT &&
        dof.reserved0 == 0u &&
        dof.reserved1 == 0u &&
        zero(dof.limits) &&
        zero(dof.drive);
}

bool finiteCommand(const ArticulatedDofCommand& command) {
    return finite(command.feedForward) &&
        finite(command.desiredPosition) &&
        finite(command.desiredVelocity) &&
        finite(command.stiffness) &&
        finite(command.damping);
}

bool zeroCommandScalars(const ArticulatedDofCommand& command) {
    return command.feedForward == 0.0 &&
        command.desiredPosition == 0.0 &&
        command.desiredVelocity == 0.0 &&
        command.stiffness == 0.0 &&
        command.damping == 0.0;
}

void updateMaximum(double& maximum, const double value) {
    maximum = std::max(maximum, std::abs(value));
}

ArticulatedActuationDiagnostics fail(
    ArticulatedActuationDiagnostics diagnostics,
    const ArticulatedActuationStatus status,
    const mr_u32 localDof = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.rejectedLocalDof = localDof;
    return diagnostics;
}

} // namespace

bool cookActuatorProfile(
    const ActuatorProfile& source,
    const std::uint32_t globalVIndex,
    MRActuatorProfileGPU& output,
    std::string* reason
) {
    const double stallTorque =
        source.jointTorqueConstant * source.currentLimit;
    const bool valid =
        globalVIndex != MR_INVALID_INDEX &&
        finite(source.jointTorqueConstant) &&
        finite(source.currentLimit) &&
        finite(source.noLoadSpeed) &&
        finite(source.efficiency) &&
        finite(source.backlash) &&
        finite(source.commandDelaySeconds) &&
        source.jointTorqueConstant > 0.0 &&
        source.currentLimit > 0.0 &&
        source.noLoadSpeed > 0.0 &&
        source.efficiency > 0.0 &&
        source.efficiency <= 1.0 &&
        source.backlash >= 0.0 &&
        source.commandDelaySeconds >= 0.0 &&
        finite(stallTorque) &&
        stallTorque > 0.0 &&
        stallTorque <=
            std::numeric_limits<float>::max() &&
        source.jointTorqueConstant <=
            std::numeric_limits<float>::max() &&
        source.currentLimit <=
            std::numeric_limits<float>::max() &&
        source.noLoadSpeed <=
            std::numeric_limits<float>::max() &&
        source.backlash <=
            std::numeric_limits<float>::max() &&
        source.commandDelaySeconds <=
            std::numeric_limits<float>::max();
    if (!valid) {
        if (reason != nullptr) {
            *reason =
                "actuator profile must contain finite positive motor "
                "parameters, efficiency in (0,1], and nonnegative "
                "backlash/delay";
        }
        return false;
    }
    MRActuatorProfileGPU cooked{};
    cooked.motorAndSpeed = {
        static_cast<float>(source.jointTorqueConstant),
        static_cast<float>(source.currentLimit),
        static_cast<float>(source.noLoadSpeed),
        static_cast<float>(source.efficiency),
    };
    cooked.transmissionAndEnvelope = {
        static_cast<float>(source.backlash),
        static_cast<float>(source.commandDelaySeconds),
        static_cast<float>(stallTorque),
        0.0f,
    };
    cooked.identity = {
        globalVIndex,
        MR_ACTUATOR_PROFILE_ACTIVE |
            (
                source.calibrated
                ? MR_ACTUATOR_PROFILE_CALIBRATED
                : 0u
            ),
        0u,
        0u,
    };
    output = cooked;
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

double actuatorTorqueEnvelope(
    const MRActuatorProfileGPU& profile,
    const double jointVelocity,
    const double authoredEffortLimit
) noexcept {
    const double speedFraction = std::clamp(
        std::abs(jointVelocity) /
            std::max(
                static_cast<double>(
                    profile.motorAndSpeed.z
                ),
                std::numeric_limits<double>::min()
            ),
        0.0,
        1.0
    );
    const double motorLimit =
        static_cast<double>(
            profile.transmissionAndEnvelope.z
        ) *
        profile.motorAndSpeed.w *
        (1.0 - speedFraction);
    return std::min(
        std::max(0.0, authoredEffortLimit),
        std::max(0.0, motorLimit)
    );
}

double updateActuatorBacklashTarget(
    const double previousEffectiveTarget,
    const double commandedTarget,
    const double backlashPlay
) noexcept {
    const double halfPlay =
        0.5 * std::max(0.0, backlashPlay);
    return std::clamp(
        previousEffectiveTarget,
        commandedTarget - halfPlay,
        commandedTarget + halfPlay
    );
}

ArticulatedActuationDiagnostics evaluateArticulatedActuation(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const ArticulatedDofCommand> commands,
    ArticulatedActuationResult& result,
    const ArticulatedActuationConfig& config
) {
    ArticulatedActuationDiagnostics diagnostics{};
    diagnostics.articulationIndex = articulationIndex;

    if (!finite(config.stictionVelocityThreshold) ||
        config.stictionVelocityThreshold < 0.0) {
        return fail(
            diagnostics,
            ArticulatedActuationStatus::invalidConfiguration
        );
    }
    if (articulationIndex >= model.articulations.size()) {
        return fail(
            diagnostics,
            ArticulatedActuationStatus::invalidArticulation
        );
    }

    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    diagnostics.dofCount = articulation.nv;
    const mr_u32 rootDofCount =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        rootDofCount > articulation.nv ||
        !within(
            articulation.qOffset,
            articulation.nq,
            model.defaultQ.size()
        ) ||
        !within(
            articulation.vOffset,
            articulation.nv,
            model.defaultV.size()
        ) ||
        !within(
            articulation.vOffset,
            articulation.nv,
            model.dofs.size()
        ) ||
        !within(
            articulation.firstJoint,
            articulation.jointCount,
            model.joints.size()
        ) ||
        articulation.qOffset > model.world.nq ||
        articulation.nq >
            model.world.nq - articulation.qOffset ||
        articulation.vOffset > model.world.nv ||
        articulation.nv >
            model.world.nv - articulation.vOffset) {
        return fail(
            diagnostics,
            ArticulatedActuationStatus::invalidArticulation
        );
    }
    if (q.size() != articulation.nq ||
        v.size() != articulation.nv ||
        commands.size() != articulation.nv) {
        return fail(
            diagnostics,
            ArticulatedActuationStatus::invalidDimensions
        );
    }
    if (!std::ranges::all_of(q, [](const double value) {
            return finite(value);
        }) ||
        !std::ranges::all_of(v, [](const double value) {
            return finite(value);
        })) {
        return fail(
            diagnostics,
            ArticulatedActuationStatus::nonfiniteInput
        );
    }

    ArticulatedActuationResult staged{};
    staged.actuatorEffort.assign(articulation.nv, 0.0);
    staged.passiveFrictionEffort.assign(articulation.nv, 0.0);
    staged.generalizedEffort.assign(articulation.nv, 0.0);

    for (mr_u32 localV = 0u;
         localV < articulation.nv;
         ++localV) {
        const mr_u32 globalV = articulation.vOffset + localV;
        const MRDofPropertiesGPU& dof = model.dofs[globalV];
        const ArticulatedDofCommand& command = commands[localV];
        const bool root = localV < rootDofCount;

        const bool validMetadata =
            root
                ? validRootDofMetadata(
                      dof,
                      articulationIndex,
                      articulation,
                      localV
                  )
                : validNonRootDofMetadata(
                      model,
                      articulation,
                      articulationIndex,
                      globalV,
                      dof
                  );
        if (!validMetadata) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::invalidDofMetadata,
                localV
            );
        }
        if (!finiteCommand(command)) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::nonfiniteInput,
                localV
            );
        }

        const std::uint32_t mode =
            static_cast<std::uint32_t>(command.mode);
        if (mode >
            static_cast<std::uint32_t>(
                ArticulatedActuationMode::effort
            )) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::invalidCommandMode,
                localV
            );
        }
        if (root &&
            command.mode != ArticulatedActuationMode::disabled) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::rootActuationForbidden,
                localV
            );
        }

        double stiffness = 0.0;
        double damping = 0.0;
        const bool active =
            command.mode != ArticulatedActuationMode::disabled;
        if (!active) {
            if (!zeroCommandScalars(command)) {
                return fail(
                    diagnostics,
                    ArticulatedActuationStatus::invalidCommandSemantics,
                    localV
                );
            }
        } else {
            if ((dof.flags & MR_DOF_FLAG_ACTUATED) == 0u) {
                return fail(
                    diagnostics,
                    ArticulatedActuationStatus::unactuatedDof,
                    localV
                );
            }
            if ((dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) == 0u ||
                !(dof.limits.w > 0.0f)) {
                return fail(
                    diagnostics,
                    ArticulatedActuationStatus::missingEffortLimit,
                    localV
                );
            }

            if (command.mode == ArticulatedActuationMode::modelPD) {
                if (command.stiffness != 0.0 ||
                    command.damping != 0.0) {
                    return fail(
                        diagnostics,
                        ArticulatedActuationStatus::
                            invalidCommandSemantics,
                        localV
                    );
                }
                if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
                    return fail(
                        diagnostics,
                        ArticulatedActuationStatus::missingModelDrive,
                        localV
                    );
                }
                stiffness = static_cast<double>(dof.drive.x);
                damping = static_cast<double>(dof.drive.y);
            } else if (
                command.mode == ArticulatedActuationMode::customPD
            ) {
                if (command.stiffness < 0.0 ||
                    command.damping < 0.0) {
                    return fail(
                        diagnostics,
                        ArticulatedActuationStatus::
                            invalidCommandSemantics,
                        localV
                    );
                }
                stiffness = command.stiffness;
                damping = command.damping;
            } else {
                if (command.desiredPosition != 0.0 ||
                    command.desiredVelocity != 0.0 ||
                    command.stiffness != 0.0 ||
                    command.damping != 0.0) {
                    return fail(
                        diagnostics,
                        ArticulatedActuationStatus::
                            invalidCommandSemantics,
                        localV
                    );
                }
            }
        }

        double actuator = 0.0;
        if (command.mode == ArticulatedActuationMode::modelPD ||
            command.mode == ArticulatedActuationMode::customPD) {
            if (dof.qIndex == MR_INVALID_INDEX ||
                dof.qIndex < articulation.qOffset ||
                dof.qIndex - articulation.qOffset >=
                    articulation.nq) {
                return fail(
                    diagnostics,
                    ArticulatedActuationStatus::
                        positionCoordinateUnavailable,
                    localV
                );
            }
            const mr_u32 localQ =
                dof.qIndex - articulation.qOffset;
            double positionError =
                command.desiredPosition - q[localQ];
            if (!root &&
                model.joints[dof.jointIndex].jointType ==
                    MR_JOINT_CONTINUOUS) {
                positionError = std::remainder(
                    positionError,
                    2.0 * std::numbers::pi_v<double>
                );
            }
            actuator =
                command.feedForward +
                stiffness * positionError +
                damping *
                    (command.desiredVelocity - v[localV]);
        } else if (
            command.mode == ArticulatedActuationMode::effort
        ) {
            actuator = command.feedForward;
        }
        if (!finite(actuator)) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::nonfiniteResult,
                localV
            );
        }

        updateMaximum(
            diagnostics.maximumUnclampedActuatorEffort,
            actuator
        );
        if (active) {
            double effortLimit =
                static_cast<double>(dof.limits.w);
            if (!model.actuatorProfiles.empty() &&
                (
                    model.actuatorProfiles[globalV].identity.y &
                    MR_ACTUATOR_PROFILE_ACTIVE
                ) != 0u) {
                effortLimit = actuatorTorqueEnvelope(
                    model.actuatorProfiles[globalV],
                    v[localV],
                    effortLimit
                );
            }
            const double clamped =
                std::clamp(actuator, -effortLimit, effortLimit);
            if (clamped != actuator) {
                ++diagnostics.saturatedDofCount;
                diagnostics.maximumSaturationExcess = std::max(
                    diagnostics.maximumSaturationExcess,
                    std::abs(actuator) - effortLimit
                );
            }
            actuator = clamped;
        }

        double friction = 0.0;
        const double dryFriction =
            static_cast<double>(dof.drive.w);
        if (dryFriction > 0.0) {
            if (std::abs(v[localV]) >
                config.stictionVelocityThreshold) {
                friction = std::copysign(
                    dryFriction,
                    -v[localV]
                );
                const double dissipatedPower =
                    -friction * v[localV];
                if (!finite(dissipatedPower) ||
                    !finite(
                        diagnostics.movingFrictionDissipation +
                        dissipatedPower
                    )) {
                    return fail(
                        diagnostics,
                        ArticulatedActuationStatus::nonfiniteResult,
                        localV
                    );
                }
                ++diagnostics.movingFrictionDofCount;
                diagnostics.movingFrictionDissipation +=
                    dissipatedPower;
            } else {
                friction = -std::clamp(
                    actuator,
                    -dryFriction,
                    dryFriction
                );
                ++diagnostics.stictionDofCount;
            }
            if (friction != 0.0) {
                ++diagnostics.frictionActiveDofCount;
            }
        }

        const double generalized = actuator + friction;
        if (!finite(friction) || !finite(generalized)) {
            return fail(
                diagnostics,
                ArticulatedActuationStatus::nonfiniteResult,
                localV
            );
        }

        staged.actuatorEffort[localV] = actuator;
        staged.passiveFrictionEffort[localV] = friction;
        staged.generalizedEffort[localV] = generalized;
        updateMaximum(
            diagnostics.maximumActuatorEffort,
            actuator
        );
        updateMaximum(
            diagnostics.maximumPassiveFrictionEffort,
            friction
        );
        updateMaximum(
            diagnostics.maximumGeneralizedEffort,
            generalized
        );
    }

    result = std::move(staged);
    return diagnostics;
}

const char* articulatedActuationStatusName(
    const ArticulatedActuationStatus status
) noexcept {
    switch (status) {
    case ArticulatedActuationStatus::success:
        return "success";
    case ArticulatedActuationStatus::invalidConfiguration:
        return "invalid_configuration";
    case ArticulatedActuationStatus::invalidArticulation:
        return "invalid_articulation";
    case ArticulatedActuationStatus::invalidDimensions:
        return "invalid_dimensions";
    case ArticulatedActuationStatus::nonfiniteInput:
        return "nonfinite_input";
    case ArticulatedActuationStatus::invalidDofMetadata:
        return "invalid_dof_metadata";
    case ArticulatedActuationStatus::invalidCommandMode:
        return "invalid_command_mode";
    case ArticulatedActuationStatus::invalidCommandSemantics:
        return "invalid_command_semantics";
    case ArticulatedActuationStatus::rootActuationForbidden:
        return "root_actuation_forbidden";
    case ArticulatedActuationStatus::unactuatedDof:
        return "unactuated_dof";
    case ArticulatedActuationStatus::missingEffortLimit:
        return "missing_effort_limit";
    case ArticulatedActuationStatus::missingModelDrive:
        return "missing_model_drive";
    case ArticulatedActuationStatus::positionCoordinateUnavailable:
        return "position_coordinate_unavailable";
    case ArticulatedActuationStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
