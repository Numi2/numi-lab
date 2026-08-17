#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include "metalrobo/SurgicalPSM.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ranges>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

constexpr std::array<std::uint32_t, 2> kJawAShapes{
    24u,
    26u,
};
constexpr std::array<std::uint32_t, 2> kJawBShapes{
    25u,
    27u,
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
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
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

double length(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3& value) {
    const double magnitude = length(value);
    require(
        magnitude > 1.0e-12 && std::isfinite(magnitude),
        "thread grasp direction is degenerate"
    );
    return multiply(value, 1.0 / magnitude);
}

Vec3 rotate(
    const std::array<double, 4>& quaternion,
    const Vec3& value
) {
    const Vec3 imaginary{
        quaternion[0],
        quaternion[1],
        quaternion[2],
    };
    const Vec3 first{
        imaginary[1] * value[2] - imaginary[2] * value[1],
        imaginary[2] * value[0] - imaginary[0] * value[2],
        imaginary[0] * value[1] - imaginary[1] * value[0],
    };
    const Vec3 inside = add(
        first,
        multiply(value, quaternion[3])
    );
    const Vec3 second{
        imaginary[1] * inside[2] - imaginary[2] * inside[1],
        imaginary[2] * inside[0] - imaginary[0] * inside[2],
        imaginary[0] * inside[1] - imaginary[1] * inside[0],
    };
    return add(value, multiply(second, 2.0));
}

std::vector<metalrobo::ArticulatedBodyKinematics> bodyKinematics(
    const metalrobo::EngineModel& model,
    const std::span<const double> q
) {
    std::vector<metalrobo::ArticulatedBodyKinematics> result(
        model.articulations.at(0u).bodyCount
    );
    const std::vector<double> velocity(model.world.nv, 0.0);
    const auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            velocity,
            result
        );
    require(
        diagnostics.succeeded(),
        "Large Needle Driver kinematics failed"
    );
    return result;
}

Vec3 shapeCenter(
    const metalrobo::EngineModel& model,
    const std::span<const metalrobo::ArticulatedBodyKinematics> bodies,
    const std::uint32_t shapeIndex
) {
    const MRShapeGPU& shape = model.shapes.at(shapeIndex);
    const auto found = std::ranges::find_if(
        bodies,
        [&](const metalrobo::ArticulatedBodyKinematics& body) {
            return body.bodyIndex == shape.bodyIndex;
        }
    );
    require(found != bodies.end(), "jaw insert body is missing");
    const Vec3 local{
        shape.localPosition.x,
        shape.localPosition.y,
        shape.localPosition.z,
    };
    return add(
        found->centerOfMassPosition,
        rotate(found->orientation, local)
    );
}

struct JawGeometry {
    Vec3 jawA{};
    Vec3 jawB{};
    Vec3 midpoint{};
    Vec3 rail{};
    double separation = 0.0;
};

JawGeometry jawGeometry(
    const metalrobo::EngineModel& model,
    const std::span<const double> q
) {
    const auto bodies = bodyKinematics(model, q);
    JawGeometry result;
    Vec3 negativeRailCenter{};
    Vec3 positiveRailCenter{};
    constexpr double jawWeight = 1.0 / kJawAShapes.size();
    constexpr double railWeight = 1.0 / kJawAShapes.size();
    for (std::size_t index = 0u; index < kJawAShapes.size(); ++index) {
        const Vec3 centerA = shapeCenter(
            model,
            bodies,
            kJawAShapes[index]
        );
        const Vec3 centerB = shapeCenter(
            model,
            bodies,
            kJawBShapes[index]
        );
        result.jawA = add(result.jawA, multiply(centerA, jawWeight));
        result.jawB = add(result.jawB, multiply(centerB, jawWeight));
        Vec3& railCenter = index < kJawAShapes.size() / 2u
            ? negativeRailCenter : positiveRailCenter;
        railCenter = add(
            railCenter,
            multiply(add(centerA, centerB), railWeight)
        );
    }
    result.midpoint = multiply(add(result.jawA, result.jawB), 0.5);
    result.separation = length(subtract(result.jawA, result.jawB));
    result.rail = normalized(subtract(
        positiveRailCenter,
        negativeRailCenter
    ));
    return result;
}

std::vector<double> jawConfiguration(
    const metalrobo::EngineModel& model,
    const double coordinate
) {
    std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
    q[6] = -coordinate;
    q[7] = coordinate;
    return q;
}

double calibratedThreadJawCoordinate(
    const metalrobo::EngineModel& model,
    const double threadRadius,
    const double radialPreload
) {
    const double railRadius =
        model.shapes.at(kJawAShapes[0]).dimensions.x;
    const double rowHalfSpacing = 0.5 * std::abs(
        static_cast<double>(
            model.shapes.at(kJawAShapes[1]).localPosition.z
        ) - model.shapes.at(kJawAShapes[0]).localPosition.z
    );
    const double contactRadius =
        threadRadius + railRadius - radialPreload;
    require(
        contactRadius > rowHalfSpacing,
        "authored insert groove cannot seat the PDO strand"
    );
    const double desiredSeparation = 2.0 * std::sqrt(
        contactRadius * contactRadius -
        rowHalfSpacing * rowHalfSpacing
    );
    double bestCoordinate = 0.0;
    double bestError = std::numeric_limits<double>::infinity();
    for (std::uint32_t sample = 0u; sample <= 2400u; ++sample) {
        const double coordinate =
            0.12 * static_cast<double>(sample) / 2400.0;
        const double error = std::abs(
            jawGeometry(
                model,
                jawConfiguration(model, coordinate)
            ).separation - desiredSeparation
        );
        if (error < bestError) {
            bestError = error;
            bestCoordinate = coordinate;
        }
    }
    require(
        bestCoordinate > 0.0 && bestCoordinate < 0.12 &&
            bestError <= 2.0e-5,
        "Large Needle Driver could not close on 3-0 PDO: coordinate=" +
            std::to_string(bestCoordinate) +
            " error=" + std::to_string(bestError) +
            " desired_separation=" +
            std::to_string(desiredSeparation)
    );
    return bestCoordinate;
}

std::vector<MRBodyStateGPU> kinematicBodyStates(
    const metalrobo::EngineModel& model,
    const std::span<const double> q
) {
    const auto bodies = bodyKinematics(model, q);
    std::vector<MRBodyStateGPU> result(model.bodies.size());
    for (const auto& body : bodies) {
        MRBodyStateGPU& state = result.at(body.bodyIndex);
        state.position = {
            static_cast<float>(body.centerOfMassPosition[0]),
            static_cast<float>(body.centerOfMassPosition[1]),
            static_cast<float>(body.centerOfMassPosition[2]),
            1.0f,
        };
        state.orientation = {
            static_cast<float>(body.orientation[0]),
            static_cast<float>(body.orientation[1]),
            static_cast<float>(body.orientation[2]),
            static_cast<float>(body.orientation[3]),
        };
        state.linearVelocityAndInverseMass = {};
        state.angularVelocity = {};
        state.inverseInertiaWorldRow0 = {};
        state.inverseInertiaWorldRow1 = {};
        state.inverseInertiaWorldRow2 = {};
        state.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = body.bodyIndex;
    }
    return result;
}

void placeRod(
    metalrobo::DiscreteElasticRodModel& model,
    const Vec3& midpoint,
    const Vec3& direction
) {
    const double halfLength = 0.5 *
        std::ranges::fold_left(
            model.restLengths,
            0.0,
            std::plus<>{}
        );
    for (std::size_t node = 0u;
         node < model.restPositions.size();
         ++node) {
        const double along =
            model.restPositions[node][0] - halfLength;
        model.restPositions[node] = add(
            midpoint,
            multiply(direction, along)
        );
    }
    std::string reason;
    require(model.valid(&reason), "placed PDO rod is invalid: " + reason);
}

std::vector<MRRodToolPairGPU> insertPairs(
    const metalrobo::DiscreteElasticRodModel& rod
) {
    std::vector<MRRodToolPairGPU> result;
    result.reserve(
        (rod.restPositions.size() - 1u) *
        (kJawAShapes.size() + kJawBShapes.size())
    );
    for (std::uint32_t edge = 0u;
         edge + 1u < rod.restPositions.size();
         ++edge) {
        for (const std::uint32_t shape : kJawAShapes) {
            result.push_back({
                .rodCollider = edge,
                .rigidCollider = shape,
                .pairClass = MR_COLLISION_PAIR_CAPSULE_BOX,
                .flags = MR_ROD_TOOL_PAIR_VALID,
            });
        }
        for (const std::uint32_t shape : kJawBShapes) {
            result.push_back({
                .rodCollider = edge,
                .rigidCollider = shape,
                .pairClass = MR_COLLISION_PAIR_CAPSULE_BOX,
                .flags = MR_ROD_TOOL_PAIR_VALID,
            });
        }
    }
    return result;
}

struct ClampOutcome {
    metalrobo::DiscreteElasticRodState state;
    double maximumNormalImpulse = 0.0;
    double maximumTangentialImpulse = 0.0;
    double maximumAttachmentLoadN = 0.0;
    double meanPullResistanceN = 0.0;
    double graspNodeDisplacementM = 0.0;
    std::uint32_t jawAContactSteps = 0u;
    std::uint32_t jawBContactSteps = 0u;
    std::uint32_t bilateralPullContactSteps = 0u;
    std::uint32_t firstMissingBilateralPullStep = MR_INVALID_INDEX;
    std::uint32_t lastMissingBilateralPullStep = MR_INVALID_INDEX;
    bool terminalBilateralContact = false;
    std::uint32_t failedSteps = 0u;
    std::string deviceName;
};

ClampOutcome runClamp(
    const metalrobo::EngineModel& toolModel,
    const metalrobo::DiscreteElasticRodModel& rod,
    const metalrobo::DiscreteElasticRodState& initialState,
    const std::span<const MRBodyStateGPU> toolBodies,
    const std::span<const MRRodToolPairGPU> pairs,
    const Vec3& pullDirection
) {
    constexpr double timestep = 1.0e-3;
    constexpr double pullSpeedMps = 2.0e-3;
    constexpr std::uint32_t settleSteps = 12u;
    constexpr std::uint32_t pullSteps = 40u;
    const std::uint32_t graspNode =
        static_cast<std::uint32_t>(initialState.positions.size() / 2u);
    const Vec3 initialGraspPosition = initialState.positions[graspNode];

    ClampOutcome outcome;
    outcome.state = initialState;
    metalrobo::DiscreteRodAttachment attachment{
        .nodeIndex = 0u,
        .targetPosition = initialState.positions.front(),
        .targetVelocity = {},
        .compliance = 0.0,
    };
    std::vector<MRRodToolWitnessGPU> previousContacts;
    metalrobo::MetalDiscreteElasticRodConfig config;
    config.step.timestep = timestep;
    config.step.gravity = {0.0, 0.0, 0.0};
    config.step.solverIterations = 1024u;
    config.step.constraintTolerance = 5.0e-6;
    config.step.linearDamping = 1.0;
    config.step.twistDamping = 1.0;
    config.step.derivativeStep = 3.5e-4;
    config.tool.enabled = true;
    config.tool.warmStart = true;
    config.tool.outerIterations = 8u;
    config.tool.rodMaterialIndex =
        static_cast<std::uint32_t>(toolModel.materials.size() - 1u);
    config.tool.contactOffset = 2.0e-5f;
    config.tool.compliance =
        metalrobo::surgicalPSMMetadata()
            .insertSystemNormalComplianceMPerN;
    config.tool.damping = 0.0f;
    config.tool.restitution = 0.0f;
    config.tool.frictionScale = 1.0f;

    for (std::uint32_t step = 0u;
         step < settleSteps + pullSteps;
         ++step) {
        if (step >= settleSteps) {
            attachment.targetPosition = add(
                attachment.targetPosition,
                multiply(pullDirection, pullSpeedMps * timestep)
            );
            attachment.targetVelocity = multiply(
                pullDirection,
                pullSpeedMps
            );
        }
        const std::array<metalrobo::DiscreteElasticRodState, 1> states{{
            outcome.state,
        }};
        const std::array<metalrobo::DiscreteRodAttachment, 1>
            attachments{{attachment}};
        const std::array<
            metalrobo::DiscreteRodRigidAttachmentBinding,
            1
        > rigidBindings{};
        metalrobo::MetalDiscreteElasticRodResult result;
        const auto diagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                rod,
                {
                    .states = states,
                    .attachmentCount = 1u,
                    .attachments = attachments,
                    .rigidBodyCount = toolBodies.size(),
                    .rigidBodies = toolBodies,
                    .rigidBindings = rigidBindings,
                    .toolModel = &toolModel,
                    .toolPairs = pairs,
                    .previousToolContacts = previousContacts,
                },
                result,
                config
            );
        require(
            diagnostics.succeeded() && diagnostics.published &&
                result.states.size() == 1u &&
                result.reactions.size() == 1u,
            "Apple Metal PDO clamp step failed at " +
                std::to_string(step) + ": " + diagnostics.message
        );
        outcome.deviceName = diagnostics.deviceName;
        outcome.state = std::move(result.states[0]);
        previousContacts = std::move(result.toolContacts);
        outcome.maximumAttachmentLoadN = std::max(
            outcome.maximumAttachmentLoadN,
            length(result.reactions[0].averageForceOnTarget)
        );
        bool jawAContact = false;
        bool jawBContact = false;
        for (const MRRodToolWitnessGPU& witness : previousContacts) {
            if ((witness.featuresAndFlags.w &
                 MR_ROD_TOOL_WITNESS_VALID) == 0u ||
                !(witness.impulses.x > 0.0f)) {
                continue;
            }
            outcome.maximumNormalImpulse = std::max(
                outcome.maximumNormalImpulse,
                static_cast<double>(witness.impulses.x)
            );
            outcome.maximumTangentialImpulse = std::max(
                outcome.maximumTangentialImpulse,
                std::hypot(
                    static_cast<double>(witness.impulses.y),
                    static_cast<double>(witness.impulses.z)
                )
            );
            jawAContact = jawAContact ||
                witness.featuresAndFlags.z == 7u;
            jawBContact = jawBContact ||
                witness.featuresAndFlags.z == 8u;
        }
        outcome.jawAContactSteps += jawAContact ? 1u : 0u;
        outcome.jawBContactSteps += jawBContact ? 1u : 0u;
        if (step >= settleSteps) {
            outcome.meanPullResistanceN += std::abs(dot(
                result.reactions[0].averageForceOnTarget,
                pullDirection
            ));
            outcome.bilateralPullContactSteps +=
                jawAContact && jawBContact ? 1u : 0u;
            if (!(jawAContact && jawBContact)) {
                outcome.firstMissingBilateralPullStep = std::min(
                    outcome.firstMissingBilateralPullStep,
                    step - settleSteps
                );
                outcome.lastMissingBilateralPullStep =
                    step - settleSteps;
            }
            outcome.terminalBilateralContact = jawAContact && jawBContact;
        }
    }
    outcome.meanPullResistanceN /= static_cast<double>(pullSteps);
    outcome.graspNodeDisplacementM = dot(
        subtract(outcome.state.positions[graspNode], initialGraspPosition),
        pullDirection
    );
    return outcome;
}

} // namespace

int main() {
    try {
        constexpr double radialPreloadM = 14.0e-6;
        const auto sutureSpec =
            metalrobo::makeBowelAnastomosisSutureSpec();
        const auto sutureWorld =
            metalrobo::makeBowelAnastomosisNeedleThreadWorldConfig(
                sutureSpec
            );
        metalrobo::EngineModel frictionalTool =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        frictionalTool.materials.push_back(
            sutureWorld.threadContactMaterial
        );
        frictionalTool.world.materialCount =
            static_cast<std::uint32_t>(frictionalTool.materials.size());
        std::string modelReason;
        require(
            frictionalTool.valid(&modelReason),
            "thread-grasp PSM is invalid: " + modelReason
        );
        metalrobo::EngineModel frictionlessTool = frictionalTool;
        for (const std::uint32_t shapeIndex : kJawAShapes) {
            const std::uint32_t materialIndex =
                frictionlessTool.shapes.at(shapeIndex).materialIndex;
            frictionlessTool.materials.at(materialIndex).friction = {};
        }
        for (const std::uint32_t shapeIndex : kJawBShapes) {
            const std::uint32_t materialIndex =
                frictionlessTool.shapes.at(shapeIndex).materialIndex;
            frictionlessTool.materials.at(materialIndex).friction = {};
        }
        require(
            frictionlessTool.valid(&modelReason),
            "frictionless thread-grasp control is invalid: " + modelReason
        );

        const double jawCoordinate = calibratedThreadJawCoordinate(
            frictionalTool,
            sutureSpec.threadRadiusM.value,
            radialPreloadM
        );
        const std::vector<double> q = jawConfiguration(
            frictionalTool,
            jawCoordinate
        );
        const JawGeometry jaws = jawGeometry(frictionalTool, q);
        const auto toolBodies = kinematicBodyStates(frictionalTool, q);
        metalrobo::DiscreteElasticRodModel rod =
            metalrobo::makeStraightSutureRod(
                65u,
                0.02,
                sutureWorld.threadMaterial
            );
        placeRod(rod, jaws.midpoint, jaws.rail);
        const metalrobo::DiscreteElasticRodState initialState =
            metalrobo::makeDiscreteElasticRodDefaultState(rod);
        const auto pairs = insertPairs(rod);
        const Vec3 pullDirection = multiply(jaws.rail, -1.0);

        const ClampOutcome frictionless = runClamp(
            frictionlessTool,
            rod,
            initialState,
            toolBodies,
            pairs,
            pullDirection
        );
        const ClampOutcome frictional = runClamp(
            frictionalTool,
            rod,
            initialState,
            toolBodies,
            pairs,
            pullDirection
        );
        const ClampOutcome replay = runClamp(
            frictionalTool,
            rod,
            initialState,
            toolBodies,
            pairs,
            pullDirection
        );
        const bool deterministicReplay =
            replay.state.positions == frictional.state.positions &&
            replay.state.velocities == frictional.state.velocities &&
            replay.state.twists == frictional.state.twists &&
            replay.state.twistRates == frictional.state.twistRates &&
            replay.maximumNormalImpulse ==
                frictional.maximumNormalImpulse &&
            replay.maximumTangentialImpulse ==
                frictional.maximumTangentialImpulse &&
            replay.meanPullResistanceN ==
                frictional.meanPullResistanceN &&
            replay.bilateralPullContactSteps ==
                frictional.bilateralPullContactSteps;
        const bool retained =
            frictional.jawAContactSteps > 0u &&
                frictional.jawBContactSteps > 0u &&
                frictional.maximumNormalImpulse > 0.0 &&
                frictional.maximumTangentialImpulse > 0.0 &&
                frictional.maximumAttachmentLoadN > 0.0 &&
                frictionless.maximumTangentialImpulse <= 1.0e-12 &&
                frictional.meanPullResistanceN >
                    1.2 * frictionless.meanPullResistanceN &&
                frictional.bilateralPullContactSteps >= 36u &&
                frictional.terminalBilateralContact &&
                frictional.graspNodeDisplacementM <
                    frictionless.graspNodeDisplacementM &&
                deterministicReplay;
        if (!retained) {
            std::ostringstream evidence;
            evidence << std::setprecision(12) << std::scientific
                << "PDO clamp did not establish bilateral frictional retention:"
                << " jaw_coordinate_rad=" << jawCoordinate
                << " jaw_center_separation_m=" << jaws.separation
                << " jaw_a_contact_steps="
                << frictional.jawAContactSteps
                << " jaw_b_contact_steps="
                << frictional.jawBContactSteps
                << " bilateral_pull_contact_steps="
                << frictional.bilateralPullContactSteps
                << " terminal_bilateral="
                << frictional.terminalBilateralContact
                << " normal_impulse="
                << frictional.maximumNormalImpulse
                << " frictional_tangent_impulse="
                << frictional.maximumTangentialImpulse
                << " frictionless_tangent_impulse="
                << frictionless.maximumTangentialImpulse
                << " attachment_load_n="
                << frictional.maximumAttachmentLoadN
                << " frictional_resistance_n="
                << frictional.meanPullResistanceN
                << " frictionless_resistance_n="
                << frictionless.meanPullResistanceN
                << " frictional_displacement_m="
                << frictional.graspNodeDisplacementM
                << " frictionless_displacement_m="
                << frictionless.graspNodeDisplacementM
                << " deterministic=" << deterministicReplay;
            throw std::runtime_error(evidence.str());
        }

        std::cout << std::setprecision(9)
            << "surgical_thread_grasp=ok"
            << " device=\"" << frictional.deviceName << "\""
            << " material=PDO_3_0"
            << " thread_diameter_mm="
            << 2000.0 * sutureSpec.threadRadiusM.value
            << " instrument=large_needle_driver"
            << " insert_patches=4"
            << " jaw_coordinate_rad=" << jawCoordinate
            << " jaw_center_separation_m=" << jaws.separation
            << " radial_preload_m=" << radialPreloadM
            << " jaw_a_contact_steps=" << frictional.jawAContactSteps
            << " jaw_b_contact_steps=" << frictional.jawBContactSteps
            << " maximum_normal_impulse_ns="
            << frictional.maximumNormalImpulse
            << " maximum_tangential_impulse_ns="
            << frictional.maximumTangentialImpulse
            << " maximum_attachment_load_n="
            << frictional.maximumAttachmentLoadN
            << " frictionless_mean_pull_resistance_n="
            << frictionless.meanPullResistanceN
            << " frictional_mean_pull_resistance_n="
            << frictional.meanPullResistanceN
            << " bilateral_pull_contact_steps="
            << frictional.bilateralPullContactSteps << "/40"
            << " first_missing_bilateral_pull_step="
            << frictional.firstMissingBilateralPullStep
            << " last_missing_bilateral_pull_step="
            << frictional.lastMissingBilateralPullStep
            << " terminal_bilateral=yes"
            << " frictionless_grasp_node_displacement_m="
            << frictionless.graspNodeDisplacementM
            << " frictional_grasp_node_displacement_m="
            << frictional.graspNodeDisplacementM
            << " deterministic=yes"
            << " failed_steps=" << frictional.failedSteps
            << " boundary=preseated_clamp_fixture_not_live_acquisition_or_knot"
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_thread_grasp=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
