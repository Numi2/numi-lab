#include "metalrobo/ArticulatedActuation.hpp"
#include "metalrobo/G1.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 2.0e-12;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(
    const double left,
    const double right,
    const double tolerance = kTolerance
) {
    return std::abs(left - right) <= tolerance;
}

std::vector<double> asDouble(const std::vector<float>& source) {
    return {source.begin(), source.end()};
}

bool bitwiseEqual(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(double)
         ) == 0);
}

bool bitwiseEqual(
    const metalrobo::ArticulatedActuationResult& left,
    const metalrobo::ArticulatedActuationResult& right
) {
    return bitwiseEqual(
               left.actuatorEffort,
               right.actuatorEffort
           ) &&
        bitwiseEqual(
            left.passiveFrictionEffort,
            right.passiveFrictionEffort
        ) &&
        bitwiseEqual(
            left.generalizedEffort,
            right.generalizedEffort
        );
}

metalrobo::EngineModel prependFreeBody(
    const metalrobo::EngineModel& source
) {
    metalrobo::EngineModel combined =
        metalrobo::makeFreeSphereEngineModel();
    const std::uint32_t bodyBase =
        static_cast<std::uint32_t>(combined.bodies.size());
    const std::uint32_t jointBase =
        static_cast<std::uint32_t>(combined.joints.size());
    const std::uint32_t qBase =
        static_cast<std::uint32_t>(combined.defaultQ.size());
    const std::uint32_t vBase =
        static_cast<std::uint32_t>(combined.defaultV.size());
    const std::uint32_t materialBase =
        static_cast<std::uint32_t>(combined.materials.size());
    const std::uint32_t articulationBase =
        static_cast<std::uint32_t>(
            combined.articulations.size()
        );

    for (MRArticulationGPU articulation :
         source.articulations) {
        articulation.rootBody += bodyBase;
        articulation.firstBody += bodyBase;
        articulation.firstJoint += jointBase;
        articulation.qOffset += qBase;
        articulation.vOffset += vBase;
        combined.articulations.push_back(articulation);
    }
    for (MRJointDescriptorGPU joint : source.joints) {
        joint.parentBody += bodyBase;
        joint.childBody += bodyBase;
        joint.qOffset += qBase;
        joint.vOffset += vBase;
        combined.joints.push_back(joint);
    }
    for (MRDofPropertiesGPU dof : source.dofs) {
        dof.articulationIndex += articulationBase;
        if (dof.jointIndex != MR_INVALID_INDEX) {
            dof.jointIndex += jointBase;
        }
        if (dof.qIndex != MR_INVALID_INDEX) {
            dof.qIndex += qBase;
        }
        dof.vIndex += vBase;
        combined.dofs.push_back(dof);
    }
    for (MRBodyPropertiesGPU body : source.bodies) {
        if (body.articulationIndex != MR_INVALID_INDEX) {
            body.articulationIndex += articulationBase;
        }
        if (body.parentBody != MR_INVALID_INDEX) {
            body.parentBody += bodyBase;
        }
        if (body.inboundJoint != MR_INVALID_INDEX) {
            body.inboundJoint += jointBase;
        }
        combined.bodies.push_back(body);
    }
    for (const MRMaterialGPU material : source.materials) {
        combined.materials.push_back(material);
    }
    for (MRShapeGPU shape : source.shapes) {
        shape.bodyIndex += bodyBase;
        shape.materialIndex += materialBase;
        combined.shapes.push_back(shape);
    }

    combined.defaultQ.insert(
        combined.defaultQ.end(),
        source.defaultQ.begin(),
        source.defaultQ.end()
    );
    combined.defaultV.insert(
        combined.defaultV.end(),
        source.defaultV.begin(),
        source.defaultV.end()
    );
    combined.world.bodyCount =
        static_cast<mr_u32>(combined.bodies.size());
    combined.world.articulationCount =
        static_cast<mr_u32>(combined.articulations.size());
    combined.world.jointCount =
        static_cast<mr_u32>(combined.joints.size());
    combined.world.shapeCount =
        static_cast<mr_u32>(combined.shapes.size());
    combined.world.materialCount =
        static_cast<mr_u32>(combined.materials.size());
    combined.world.nq =
        static_cast<mr_u32>(combined.defaultQ.size());
    combined.world.nv =
        static_cast<mr_u32>(combined.defaultV.size());
    combined.world.pairCapacity = 4096u;
    combined.world.contactCapacity = 4096u;
    combined.world.constraintCapacity = 4096u;
    combined.world.islandCapacity = 64u;
    combined.name = "free_body_then_" + source.name;
    return combined;
}

metalrobo::ArticulatedActuationResult sentinelResult() {
    return {
        .actuatorEffort = {91.0, -17.0},
        .passiveFrictionEffort = {6.0},
        .generalizedEffort = {-3.0, 4.0, 5.0},
    };
}

} // namespace

int main() {
    try {
        using metalrobo::ArticulatedActuationMode;
        using metalrobo::ArticulatedActuationResult;
        using metalrobo::ArticulatedActuationStatus;
        using metalrobo::ArticulatedDofCommand;

        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        std::string reason;
        require(
            model.valid(&reason),
            "canonical G1 model is invalid: " + reason
        );
        require(
            model.articulations.size() == 1u &&
                model.articulations[0].nq == 36u &&
                model.articulations[0].nv == 35u,
            "canonical G1 articulation dimensions changed"
        );

        std::vector<double> q = asDouble(model.defaultQ);
        std::vector<double> v = asDouble(model.defaultV);
        std::vector<ArticulatedDofCommand> commands(
            model.articulations[0].nv
        );

        const std::uint32_t modelPdDof = 6u;
        const MRDofPropertiesGPU& modelPdProperties =
            model.dofs[modelPdDof];
        commands[modelPdDof].mode =
            ArticulatedActuationMode::modelPD;
        commands[modelPdDof].feedForward = 1.25;
        commands[modelPdDof].desiredPosition =
            q[modelPdProperties.qIndex] + 0.125;
        commands[modelPdDof].desiredVelocity =
            v[modelPdDof] - 0.75;
        const double expectedModelPd =
            commands[modelPdDof].feedForward +
            static_cast<double>(modelPdProperties.drive.x) *
                (commands[modelPdDof].desiredPosition -
                 q[modelPdProperties.qIndex]) +
            static_cast<double>(modelPdProperties.drive.y) *
                (commands[modelPdDof].desiredVelocity -
                 v[modelPdDof]);

        const std::uint32_t customPdDof = 7u;
        const MRDofPropertiesGPU& customPdProperties =
            model.dofs[customPdDof];
        commands[customPdDof].mode =
            ArticulatedActuationMode::customPD;
        commands[customPdDof].feedForward = -0.75;
        commands[customPdDof].desiredPosition =
            q[customPdProperties.qIndex] + 0.2;
        commands[customPdDof].desiredVelocity =
            v[customPdDof] + 0.4;
        commands[customPdDof].stiffness = 9.5;
        commands[customPdDof].damping = 1.25;
        const double expectedCustomPd =
            commands[customPdDof].feedForward +
            commands[customPdDof].stiffness *
                (commands[customPdDof].desiredPosition -
                 q[customPdProperties.qIndex]) +
            commands[customPdDof].damping *
                (commands[customPdDof].desiredVelocity -
                 v[customPdDof]);

        const std::uint32_t effortDof = 8u;
        const double effortLimit =
            static_cast<double>(model.dofs[effortDof].limits.w);
        commands[effortDof].mode =
            ArticulatedActuationMode::effort;
        commands[effortDof].feedForward = 4.0 * effortLimit;

        ArticulatedActuationResult result;
        const auto diagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                q,
                v,
                commands,
                result
            );
        require(
            diagnostics.succeeded(),
            std::string("G1 actuation failed: ") +
                metalrobo::articulatedActuationStatusName(
                    diagnostics.status
                )
        );
        require(
            close(
                result.actuatorEffort[modelPdDof],
                expectedModelPd
            ),
            "model-PD equation was not evaluated exactly"
        );
        require(
            close(
                result.actuatorEffort[customPdDof],
                expectedCustomPd
            ),
            "custom-PD equation was not evaluated exactly"
        );
        require(
            result.actuatorEffort[effortDof] == effortLimit &&
                diagnostics.saturatedDofCount == 1u &&
                close(
                    diagnostics.maximumSaturationExcess,
                    3.0 * effortLimit
                ),
            "effort clamp or saturation diagnostics are incorrect"
        );
        require(
            std::all_of(
                result.passiveFrictionEffort.begin(),
                result.passiveFrictionEffort.end(),
                [](const double force) {
                    return force == 0.0;
                }
            ),
            "canonical G1 unexpectedly produced dry friction"
        );
        require(
            std::all_of(
                result.generalizedEffort.begin(),
                result.generalizedEffort.begin() + 6,
                [](const double force) {
                    return force == 0.0;
                }
            ),
            "floating-root force must stay zero"
        );

        metalrobo::EngineModel continuousModel = model;
        MRDofPropertiesGPU& continuousDof =
            continuousModel.dofs[modelPdDof];
        continuousModel.joints[continuousDof.jointIndex].jointType =
            MR_JOINT_CONTINUOUS;
        continuousDof.flags &= ~MR_DOF_FLAG_POSITION_LIMIT;
        continuousDof.limits.x = 0.0f;
        continuousDof.limits.y = 0.0f;
        require(
            continuousModel.valid(&reason),
            "continuous-joint actuation model is invalid: " + reason
        );
        std::vector<double> continuousQ = q;
        continuousQ[continuousDof.qIndex] =
            std::numbers::pi_v<double> - 0.01;
        std::vector<ArticulatedDofCommand> continuousCommands(
            model.articulations[0].nv
        );
        continuousCommands[modelPdDof].mode =
            ArticulatedActuationMode::customPD;
        continuousCommands[modelPdDof].desiredPosition =
            -std::numbers::pi_v<double> + 0.01;
        continuousCommands[modelPdDof].stiffness = 10.0;
        ArticulatedActuationResult continuousResult;
        const auto continuousDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                continuousModel,
                0u,
                continuousQ,
                v,
                continuousCommands,
                continuousResult
            );
        require(
            continuousDiagnostics.succeeded() &&
                close(
                    continuousResult.actuatorEffort[modelPdDof],
                    0.2
                ),
            "continuous-joint PD did not use the shortest angular error"
        );

        ArticulatedActuationResult replay;
        const auto replayDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                q,
                v,
                commands,
                replay
            );
        require(
            replayDiagnostics.succeeded() &&
                bitwiseEqual(result, replay),
            "identical actuation did not replay bitwise"
        );

        std::vector<ArticulatedDofCommand> rejectedCommands =
            commands;
        rejectedCommands[0].mode =
            ArticulatedActuationMode::effort;
        rejectedCommands[0].feedForward = 1.0;
        ArticulatedActuationResult rejected = sentinelResult();
        const ArticulatedActuationResult rejectedBefore = rejected;
        const auto rootDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                q,
                v,
                rejectedCommands,
                rejected
            );
        require(
            rootDiagnostics.status ==
                ArticulatedActuationStatus::
                    rootActuationForbidden &&
                rootDiagnostics.rejectedLocalDof == 0u &&
                bitwiseEqual(rejected, rejectedBefore),
            "root actuation rejection was not transactional"
        );

        rejectedCommands = commands;
        rejectedCommands.back().mode =
            ArticulatedActuationMode::customPD;
        rejectedCommands.back().stiffness = -1.0;
        rejected = sentinelResult();
        const auto semanticDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                q,
                v,
                rejectedCommands,
                rejected
            );
        require(
            semanticDiagnostics.status ==
                ArticulatedActuationStatus::
                    invalidCommandSemantics &&
                semanticDiagnostics.rejectedLocalDof ==
                    commands.size() - 1u &&
                bitwiseEqual(rejected, rejectedBefore),
            "late custom-PD rejection leaked partial output"
        );

        rejectedCommands = commands;
        rejectedCommands[modelPdDof].stiffness = 1.0;
        rejected = sentinelResult();
        const auto modelSemanticDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                q,
                v,
                rejectedCommands,
                rejected
            );
        require(
            modelSemanticDiagnostics.status ==
                ArticulatedActuationStatus::
                    invalidCommandSemantics &&
                bitwiseEqual(rejected, rejectedBefore),
            "model-PD accepted a custom gain"
        );

        rejectedCommands = commands;
        std::vector<double> nonfiniteQ = q;
        nonfiniteQ.back() =
            std::numeric_limits<double>::quiet_NaN();
        rejected = sentinelResult();
        const auto nonfiniteDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                nonfiniteQ,
                v,
                rejectedCommands,
                rejected
            );
        require(
            nonfiniteDiagnostics.status ==
                ArticulatedActuationStatus::nonfiniteInput &&
                bitwiseEqual(rejected, rejectedBefore),
            "non-finite state rejection was not transactional"
        );

        metalrobo::EngineModel frictionModel = model;
        constexpr double dryFriction = 3.0;
        frictionModel.dofs[modelPdDof].drive.w =
            static_cast<float>(dryFriction);
        require(
            frictionModel.valid(&reason),
            "G1 dry-friction copy is invalid: " + reason
        );
        std::vector<ArticulatedDofCommand> passiveCommands(
            model.articulations[0].nv
        );
        std::vector<double> movingV = v;
        movingV[modelPdDof] = 0.5;
        ArticulatedActuationResult moving;
        const auto movingDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                frictionModel,
                0u,
                q,
                movingV,
                passiveCommands,
                moving
            );
        require(
            movingDiagnostics.succeeded() &&
                moving.passiveFrictionEffort[modelPdDof] ==
                    -dryFriction &&
                moving.generalizedEffort[modelPdDof] ==
                    -dryFriction &&
                movingDiagnostics.movingFrictionDofCount == 1u &&
                movingDiagnostics.frictionActiveDofCount == 1u &&
                close(
                    movingDiagnostics.movingFrictionDissipation,
                    1.5
                ),
            "positive-velocity dry friction is not dissipative"
        );
        movingV[modelPdDof] = -0.5;
        const auto reverseDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                frictionModel,
                0u,
                q,
                movingV,
                passiveCommands,
                moving
            );
        require(
            reverseDiagnostics.succeeded() &&
                moving.passiveFrictionEffort[modelPdDof] ==
                    dryFriction &&
                close(
                    reverseDiagnostics.movingFrictionDissipation,
                    1.5
                ),
            "negative-velocity dry friction is not dissipative"
        );

        passiveCommands[modelPdDof].mode =
            ArticulatedActuationMode::effort;
        passiveCommands[modelPdDof].feedForward = 2.0;
        movingV[modelPdDof] = 0.0;
        ArticulatedActuationResult stiction;
        const auto stictionDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                frictionModel,
                0u,
                q,
                movingV,
                passiveCommands,
                stiction
            );
        require(
            stictionDiagnostics.succeeded() &&
                stiction.actuatorEffort[modelPdDof] == 2.0 &&
                stiction.passiveFrictionEffort[modelPdDof] ==
                    -2.0 &&
                stiction.generalizedEffort[modelPdDof] == 0.0 &&
                stictionDiagnostics.stictionDofCount == 1u,
            "near-zero dry friction did not cancel impending load"
        );

        const metalrobo::EngineModel offsetModel =
            prependFreeBody(model);
        require(
            offsetModel.valid(&reason),
            "nonzero-offset composed model is invalid: " + reason
        );
        require(
            offsetModel.articulations.size() == 2u &&
                offsetModel.articulations[1].qOffset == 7u &&
                offsetModel.articulations[1].vOffset == 6u,
            "composed G1 does not have the expected global offsets"
        );
        std::vector<ArticulatedDofCommand> offsetCommands(
            model.articulations[0].nv
        );
        offsetCommands[modelPdDof] = commands[modelPdDof];
        ArticulatedActuationResult offsetResult;
        const auto offsetDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                offsetModel,
                1u,
                q,
                v,
                offsetCommands,
                offsetResult
            );
        require(
            offsetDiagnostics.succeeded() &&
                close(
                    offsetResult.actuatorEffort[modelPdDof],
                    expectedModelPd
                ),
            "global q/v offsets corrupted articulation-local PD"
        );

        std::cout
            << std::setprecision(12)
            << "articulated_actuation=cpu_fp64"
            << " model_pd=" << result.actuatorEffort[modelPdDof]
            << " custom_pd=" << result.actuatorEffort[customPdDof]
            << " effort_clamped=" << result.actuatorEffort[effortDof]
            << " saturation_excess="
            << diagnostics.maximumSaturationExcess
            << " moving_friction="
            << -dryFriction
            << " stiction_generalized="
            << stiction.generalizedEffort[modelPdDof]
            << " continuous_seam="
            << continuousResult.actuatorEffort[modelPdDof]
            << " offset_q="
            << offsetModel.articulations[1].qOffset
            << " offset_v="
            << offsetModel.articulations[1].vOffset
            << " replay=bitwise"
            << " transaction=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_actuation status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
