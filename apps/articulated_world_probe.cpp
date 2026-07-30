#include "metalrobo/ArticulatedWorld.hpp"
#include "metalrobo/G1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

double maximumError(
    const std::span<const double> left,
    const std::span<const double> right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        result = std::max(
            result,
            std::abs(left[index] - right[index])
        );
    }
    return result;
}

metalrobo::ArticulatedWorldConfig makeConfig() {
    metalrobo::ArticulatedWorldConfig config;
    config.dynamics.gravity = {0.0, 0.0, 0.0};
    config.dynamics.timestep = 1.0 / 240.0;
    config.dynamics.applyBodyDamping = false;
    config.dynamics.integrator =
        metalrobo::ArticulatedIntegrator::symplecticEuler;
    config.constraintAdapter.timeConstant = 0.01F;
    config.constraintAdapter.dampingRatio = 1.0F;
    config.constraintAdapter.dissipation = 0.0F;
    config.constraintAdapter.stictionTransitionVelocity = 1.0e-3F;
    config.constraintEvaluation.timestep =
        config.dynamics.timestep;
    config.jointLimits.timestep = config.dynamics.timestep;
    config.constraintEvaluation.penetrationSlop = 1.0e-4;
    config.constraintEvaluation.maximumDepenetrationVelocity = 2.0;
    config.constraintEvaluation.minimumTimeConstantRatio = 2.0;
    config.constraintEvaluation.stictionTransitionVelocity =
        1.0e-3;
    // Eight nearly redundant G1 foot contacts require an explicit physical
    // regularization floor for a unique, solver-independent impulse.
    config.constraintEvaluation.minimumRegularization = 1.0e-6;
    config.constraintResidual.projectionStep = 1.0;
    config.constraintResidual.impulseTolerance = 2.0e-6;
    config.constraintResidual.residualTolerance = 2.0e-5;
    config.collision.capacities = {
        .pairCapacity = 128u,
        .rawContactCapacity = 128u,
        .manifoldCapacity = 128u,
    };
    config.constraintCapacity = 64u;
    config.quality.maximumIterations = 400u;
    config.quality.kktTolerance = 1.0e-10;
    config.reference.maximumIterations = 30000u;
    config.reference.optimalityTolerance = 1.0e-11;
    config.solverVelocityAgreementTolerance = 1.0e-9;
    return config;
}

MRBodyStateGPU makeEnvironmentBody(
    const std::uint32_t motionType,
    const std::array<double, 3> linearVelocity = {}
) {
    MRBodyStateGPU state{};
    state.position = f4(0.0, 0.0, 0.0, 1.0);
    state.orientation = f4(0.0, 0.0, 0.0, 1.0);
    state.linearVelocityAndInverseMass = f4(
        linearVelocity[0],
        linearVelocity[1],
        linearVelocity[2],
        0.0
    );
    state.angularVelocity = {};
    state.flagsAndIndices[0] = motionType;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = MR_INVALID_INDEX;
    return state;
}

MRShapeGPU makeGroundPlane() {
    constexpr double sineHalfQuarterTurn =
        0.7071067811865475244;
    MRShapeGPU plane{};
    // Environment-local body index. ArticulatedWorld remaps it.
    plane.bodyIndex = 0u;
    plane.shapeType = MR_SHAPE_PLANE;
    plane.materialIndex = 0u;
    plane.collisionGroup = 1u;
    plane.collisionMask = ~0u;
    plane.slotGeneration = 9001u;
    plane.localPosition = f4(0.0, 0.0, 0.0, 1.0);
    plane.localRotation = f4(
        sineHalfQuarterTurn,
        0.0,
        0.0,
        sineHalfQuarterTurn
    );
    return plane;
}

std::vector<double> loweredG1Configuration(
    const metalrobo::EngineModel& model,
    const metalrobo::ArticulatedDynamicsConfig& dynamics,
    const double desiredPenetration,
    const std::span<const double> initialConfiguration = {}
) {
    require(
        initialConfiguration.empty() ||
            initialConfiguration.size() == model.defaultQ.size(),
        "initial G1 configuration has wrong dimension"
    );
    std::vector<double> q;
    if (initialConfiguration.empty()) {
        q.assign(model.defaultQ.begin(), model.defaultQ.end());
    } else {
        q.assign(
            initialConfiguration.begin(),
            initialConfiguration.end()
        );
    }
    const std::vector<double> zeroVelocity(
        model.articulations[0].nv,
        0.0
    );
    std::vector<metalrobo::ArticulatedPointQuery> queries;
    std::vector<double> radii;
    for (const MRShapeGPU& shape : model.shapes) {
        if ((shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u &&
            shape.shapeType == MR_SHAPE_SPHERE) {
            queries.push_back({
                shape.bodyIndex,
                {
                    shape.localPosition.x,
                    shape.localPosition.y,
                    shape.localPosition.z,
                },
            });
            radii.push_back(shape.dimensions.x);
        }
    }
    require(
        queries.size() == metalrobo::kUnitreeG1FootSphereCount,
        "G1 executable foot sphere set changed"
    );
    std::vector<metalrobo::ArticulatedPointKinematics>
        points(queries.size());
    std::vector<double> jacobians(
        queries.size() * 3u * zeroVelocity.size(),
        0.0
    );
    const auto diagnostics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            zeroVelocity,
            queries,
            points,
            jacobians,
            dynamics
        );
    require(
        diagnostics.succeeded(),
        "G1 sole-clearance kinematics failed"
    );
    double minimumBottom =
        std::numeric_limits<double>::infinity();
    for (std::size_t index = 0u; index < points.size(); ++index) {
        minimumBottom = std::min(
            minimumBottom,
            points[index].position[2] - radii[index]
        );
    }
    require(
        std::isfinite(minimumBottom),
        "G1 sole clearance was nonfinite"
    );
    q[2] -= minimumBottom + desiredPenetration;
    return q;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        const std::size_t nv = model.articulations[0].nv;
        const std::vector<double> zeroForce(nv, 0.0);

        // The public world configuration must be executable as constructed.
        // This catches drift between the independently reusable dynamics and
        // semantic-evaluation defaults, as well as zero collision capacities.
        metalrobo::ArticulatedWorldConfig defaultConfig;
        std::vector<double> defaultQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        defaultQ[2] += 1.0;
        std::vector<double> defaultV(nv, 0.0);
        metalrobo::ArticulatedWorldCache defaultCache;
        const auto defaultDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                defaultQ,
                defaultV,
                zeroForce,
                {},
                {},
                {},
                defaultConfig,
                defaultCache
            );
        require(
            defaultDiagnostics.succeeded() &&
                defaultDiagnostics.contactCount == 0u &&
                defaultCache.step == 1u,
            "default-constructed articulated world is not executable"
        );

        // No-contact composition must be exactly the existing symplectic
        // articulated free-motion step, including generalized forces.
        metalrobo::ArticulatedWorldConfig freeConfig = makeConfig();
        freeConfig.dynamics.gravity = {0.2, -0.1, -9.7};
        std::vector<double> freeQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        freeQ[2] += 1.0;
        std::vector<double> freeV(nv, 0.0);
        freeV[0] = 0.15;
        freeV[1] = -0.07;
        freeV[2] = 0.11;
        freeV[3] = 0.03;
        freeV[4] = -0.02;
        freeV[5] = 0.01;
        freeV[6] = 0.2;
        std::vector<double> generalizedForce(nv, 0.0);
        generalizedForce[0] = 1.5;
        generalizedForce[6] = -0.4;
        std::vector<double> expectedQ = freeQ;
        std::vector<double> expectedV = freeV;
        const auto expectedDiagnostics =
            metalrobo::integrateArticulatedState(
                model,
                0u,
                expectedQ,
                expectedV,
                generalizedForce,
                {},
                freeConfig.dynamics
            );
        require(
            expectedDiagnostics.succeeded(),
            "standalone G1 free integration failed"
        );
        metalrobo::ArticulatedWorldCache freeCache;
        const auto freeDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                freeQ,
                freeV,
                generalizedForce,
                {},
                {},
                {},
                freeConfig,
                freeCache
            );
        const double freeQError = maximumError(freeQ, expectedQ);
        const double freeVError = maximumError(freeV, expectedV);
        require(
            freeDiagnostics.succeeded() &&
                freeDiagnostics.contactCount == 0u &&
                freeCache.step == 1u &&
                freeQError < 2.0e-14 &&
                freeVError < 2.0e-14,
            "composed no-contact step differs from free dynamics"
        );

        // The policy-facing controlled step must be exactly the manual
        // actuation -> generalized-force composition, including saturation
        // and passive loss, while retaining world transactionality.
        metalrobo::ArticulatedWorldConfig controlledConfig =
            makeConfig();
        std::vector<double> controlledInitialQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        controlledInitialQ[2] += 1.0;
        std::vector<double> controlledInitialV(nv, 0.0);
        controlledInitialV[6] = 0.05;
        std::vector<metalrobo::ArticulatedDofCommand> commands(nv);
        for (std::size_t localV = 6u; localV < nv; ++localV) {
            const MRArticulationGPU& articulation =
                model.articulations[0];
            const MRDofPropertiesGPU& dof =
                model.dofs[articulation.vOffset + localV];
            require(
                dof.qIndex != MR_INVALID_INDEX &&
                    dof.qIndex >= articulation.qOffset &&
                    dof.qIndex - articulation.qOffset <
                        controlledInitialQ.size(),
                "G1 actuator lost scalar configuration coordinate"
            );
            commands[localV].mode =
                metalrobo::ArticulatedActuationMode::modelPD;
            commands[localV].desiredPosition =
                controlledInitialQ[
                    dof.qIndex - articulation.qOffset
                ];
        }
        commands[6].feedForward = 2.0;

        metalrobo::ArticulatedActuationResult manualActuation;
        const auto manualActuationDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                controlledInitialQ,
                controlledInitialV,
                commands,
                manualActuation,
                controlledConfig.actuation
            );
        require(
            manualActuationDiagnostics.succeeded(),
            "manual G1 actuation evaluation failed"
        );

        std::vector<double> manualQ = controlledInitialQ;
        std::vector<double> manualV = controlledInitialV;
        std::vector<double> controlledQ = controlledInitialQ;
        std::vector<double> controlledV = controlledInitialV;
        metalrobo::ArticulatedWorldCache manualCache;
        metalrobo::ArticulatedWorldCache controlledCache;
        const auto manualWorldDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                manualQ,
                manualV,
                manualActuation.generalizedEffort,
                {},
                {},
                {},
                controlledConfig,
                manualCache
            );
        const auto controlledDiagnostics =
            metalrobo::stepControlledArticulatedWorldCpu(
                model,
                0u,
                controlledQ,
                controlledV,
                commands,
                {},
                {},
                {},
                controlledConfig,
                controlledCache
            );
        require(
            manualWorldDiagnostics.succeeded() &&
                controlledDiagnostics.succeeded() &&
                controlledDiagnostics.actuation.succeeded() &&
                controlledDiagnostics.actuation.dofCount == nv &&
                manualQ == controlledQ &&
                manualV == controlledV &&
                manualCache.step == controlledCache.step,
            "controlled world differs from manual actuation composition"
        );

        std::vector<double> rejectedQ = controlledInitialQ;
        std::vector<double> rejectedV = controlledInitialV;
        metalrobo::ArticulatedWorldCache rejectedCache;
        const std::vector<double> rejectedQBefore = rejectedQ;
        const std::vector<double> rejectedVBefore = rejectedV;
        std::vector<metalrobo::ArticulatedDofCommand>
            rejectedCommands = commands;
        rejectedCommands[0].mode =
            metalrobo::ArticulatedActuationMode::effort;
        rejectedCommands[0].feedForward = 1.0;
        const auto rejectedDiagnostics =
            metalrobo::stepControlledArticulatedWorldCpu(
                model,
                0u,
                rejectedQ,
                rejectedV,
                rejectedCommands,
                {},
                {},
                {},
                controlledConfig,
                rejectedCache
            );
        require(
            !rejectedDiagnostics.succeeded() &&
                rejectedDiagnostics.failure ==
                    metalrobo::ArticulatedWorldFailure::actuation &&
                rejectedDiagnostics.actuation.status ==
                    metalrobo::ArticulatedActuationStatus::
                        rootActuationForbidden &&
                rejectedQ == rejectedQBefore &&
                rejectedV == rejectedVBefore &&
                rejectedCache.step == 0u,
            "rejected controlled step published state"
        );

        // A policy step approaching a G1 position stop must generate a
        // unilateral impulse, not clamp q after integration. With no contact,
        // this also exercises the factor-only mixed-constraint path.
        metalrobo::ArticulatedWorldConfig limitConfig =
            makeConfig();
        std::vector<double> limitQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        limitQ[2] += 1.0;
        std::vector<double> limitV(nv, 0.0);
        const MRArticulationGPU& g1 = model.articulations[0];
        const std::size_t limitLocalV = 6u;
        const MRDofPropertiesGPU& limitDof =
            model.dofs[g1.vOffset + limitLocalV];
        require(
            (limitDof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                limitDof.qIndex >= g1.qOffset,
            "selected G1 DoF lost its position limit"
        );
        const std::size_t limitLocalQ =
            limitDof.qIndex - g1.qOffset;
        const double lowerLimit = limitDof.limits.x;
        limitQ[limitLocalQ] = lowerLimit + 5.0e-4;
        limitV[limitLocalV] = -1.0;
        const std::vector<double> limitQBefore = limitQ;
        metalrobo::ArticulatedWorldCache limitCache;
        const auto limitDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                limitQ,
                limitV,
                zeroForce,
                {},
                {},
                {},
                limitConfig,
                limitCache
            );
        require(
            limitDiagnostics.succeeded() &&
                limitDiagnostics.contactCount == 0u &&
                limitDiagnostics.jointLimitCount > 0u &&
                limitDiagnostics.mixedSolver.succeeded() &&
                limitDiagnostics.mixedSolver
                        .finalFactorApplications == 1u &&
                limitDiagnostics.maximumJointLimitImpulse > 0.0 &&
                limitQ[limitLocalQ] >= lowerLimit - 1.0e-10 &&
                limitQ[limitLocalQ] != lowerLimit &&
                limitQBefore[limitLocalQ] > lowerLimit,
            "G1 joint stop was not enforced by an impulse code=" +
                std::to_string(limitDiagnostics.code) +
                " failure=" +
                std::to_string(
                    static_cast<std::uint32_t>(
                        limitDiagnostics.failure
                    )
                ) +
                " rows=" +
                std::to_string(
                    limitDiagnostics.jointLimitCount
                ) +
                " impulse=" +
                std::to_string(
                    limitDiagnostics.maximumJointLimitImpulse
                ) +
                " q=" + std::to_string(limitQ[limitLocalQ]) +
                " lower=" + std::to_string(lowerLimit)
        );

        metalrobo::ArticulatedWorldConfig unsupportedLimitConfig =
            limitConfig;
        unsupportedLimitConfig.solverType =
            MR_SOLVER_REFERENCE_FP64;
        std::vector<double> unsupportedLimitQ = limitQBefore;
        std::vector<double> unsupportedLimitV(nv, 0.0);
        unsupportedLimitV[limitLocalV] = -1.0;
        const std::vector<double> unsupportedLimitQBefore =
            unsupportedLimitQ;
        const std::vector<double> unsupportedLimitVBefore =
            unsupportedLimitV;
        metalrobo::ArticulatedWorldCache unsupportedLimitCache;
        const auto unsupportedLimitDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                unsupportedLimitQ,
                unsupportedLimitV,
                zeroForce,
                {},
                {},
                {},
                unsupportedLimitConfig,
                unsupportedLimitCache
            );
        require(
            !unsupportedLimitDiagnostics.succeeded() &&
                unsupportedLimitDiagnostics.code ==
                    MR_STEP_UNSUPPORTED &&
                unsupportedLimitDiagnostics.failure ==
                    metalrobo::ArticulatedWorldFailure::
                        mixedConstraintSolver &&
                unsupportedLimitQ == unsupportedLimitQBefore &&
                unsupportedLimitV == unsupportedLimitVBefore &&
                unsupportedLimitCache.step == 0u,
            "unsupported reference limit solve published state"
        );

        // A late configuration-limit failure occurs after prediction and
        // collision. It must not publish q, v, or either cache.
        metalrobo::ArticulatedWorldConfig rollbackConfig =
            freeConfig;
        std::vector<double> limitLower(
            model.articulations[0].nq,
            -std::numeric_limits<double>::infinity()
        );
        std::vector<double> limitUpper(
            model.articulations[0].nq,
            std::numeric_limits<double>::infinity()
        );
        std::vector<double> rollbackQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        rollbackQ[2] += 1.0;
        std::vector<double> rollbackV(nv, 0.0);
        rollbackV[0] = 1.0;
        limitUpper[0] =
            rollbackQ[0] +
            0.5 * rollbackConfig.dynamics.timestep;
        rollbackConfig.dynamics.limits = {
            limitLower,
            limitUpper,
            0.0,
        };
        const std::vector<double> rollbackQBefore = rollbackQ;
        const std::vector<double> rollbackVBefore = rollbackV;
        metalrobo::ArticulatedWorldCache rollbackCache;
        const auto rollbackDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                rollbackQ,
                rollbackV,
                zeroForce,
                {},
                {},
                {},
                rollbackConfig,
                rollbackCache
            );
        require(
            !rollbackDiagnostics.succeeded() &&
                rollbackDiagnostics.failure ==
                    metalrobo::ArticulatedWorldFailure::
                        configurationIntegration &&
                rollbackDiagnostics.integration.status ==
                    metalrobo::ArticulatedDynamicsStatus::
                        jointLimitViolation &&
                rollbackQ == rollbackQBefore &&
                rollbackV == rollbackVBefore &&
                rollbackCache.step == 0u &&
                rollbackCache.manifolds.size() == 0u &&
                rollbackCache.impulses.size() == 0u,
            "late articulated-step failure was not transactional"
        );

        constexpr double desiredPenetration = 5.0e-4;
        const metalrobo::ArticulatedWorldConfig contactConfig =
            makeConfig();
        const std::vector<double> contactInitialQ =
            loweredG1Configuration(
                model,
                contactConfig.dynamics,
                desiredPenetration
            );
        std::vector<double> contactInitialV(nv, 0.0);
        contactInitialV[0] = 0.12;
        contactInitialV[2] = -0.25;
        const std::array<MRBodyStateGPU, 1> staticGround{
            makeEnvironmentBody(MR_MOTION_STATIC),
        };
        const std::array<MRShapeGPU, 1> groundShape{
            makeGroundPlane(),
        };

        // Exercise the actual composed monolithic path: collision contacts
        // and an active G1 stop share one Delassus matrix and one factor
        // application. The common contact residual is checked afterwards.
        std::vector<double> mixedSeed(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        mixedSeed[limitLocalQ] = lowerLimit + 5.0e-4;
        std::vector<double> mixedQ =
            loweredG1Configuration(
                model,
                contactConfig.dynamics,
                desiredPenetration,
                mixedSeed
            );
        std::vector<double> mixedV(nv, 0.0);
        mixedV[2] = -0.25;
        mixedV[limitLocalV] = -1.0;
        metalrobo::ArticulatedWorldCache mixedCache;
        const auto mixedDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                mixedQ,
                mixedV,
                zeroForce,
                {},
                staticGround,
                groundShape,
                contactConfig,
                mixedCache
            );
        require(
            mixedDiagnostics.succeeded() &&
                mixedDiagnostics.contactCount > 0u &&
                mixedDiagnostics.jointLimitCount > 0u &&
                mixedDiagnostics.mixedSolver.succeeded() &&
                mixedDiagnostics.mixedSolver.contactCount > 0u &&
                mixedDiagnostics.mixedSolver.limitRowCount > 0u &&
                mixedDiagnostics.mixedSolver
                        .finalFactorApplications == 1u &&
                mixedDiagnostics.constraintResidual.succeeded() &&
                mixedDiagnostics.constraintResidual.withinTolerance(
                    contactConfig.constraintResidual
                ),
            "monolithic G1 contact+limit world step failed"
        );

        // Identical initial states and caches must replay bit-for-bit.
        std::vector<double> qualityQ = contactInitialQ;
        std::vector<double> qualityV = contactInitialV;
        std::vector<double> replayQ = contactInitialQ;
        std::vector<double> replayV = contactInitialV;
        metalrobo::ArticulatedWorldCache qualityCache;
        metalrobo::ArticulatedWorldCache replayCache;
        const auto qualityDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                qualityQ,
                qualityV,
                zeroForce,
                {},
                staticGround,
                groundShape,
                contactConfig,
                qualityCache
            );
        const auto replayDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                replayQ,
                replayV,
                zeroForce,
                {},
                staticGround,
                groundShape,
                contactConfig,
                replayCache
            );
        require(
            qualityDiagnostics.succeeded() &&
                replayDiagnostics.succeeded() &&
                qualityDiagnostics.contactCount > 0u &&
                qualityDiagnostics.maximumNormalImpulse > 0.0 &&
                qualityDiagnostics.maximumVelocityCorrection > 0.0 &&
                qualityDiagnostics.semanticFingerprint != 0u &&
                qualityDiagnostics.adaptation.semanticFingerprint ==
                    qualityDiagnostics.semanticFingerprint &&
                qualityDiagnostics.constraintCompilation.succeeded() &&
                qualityDiagnostics.constraintKinematics.succeeded() &&
                qualityDiagnostics.constraintEvaluation.succeeded() &&
                qualityDiagnostics.constraintResidual.succeeded() &&
                qualityDiagnostics.constraintResidual.withinTolerance(
                    contactConfig.constraintResidual
                ) &&
                !qualityDiagnostics.contactOperator
                    .materializedDenseInverse &&
                qualityDiagnostics.contactSpaceAdapter.succeeded() &&
                qualityDiagnostics
                    .solverVelocityRelativeDisagreement <
                    contactConfig
                        .solverVelocityAgreementTolerance &&
                qualityQ == replayQ &&
                qualityV == replayV &&
                qualityV[2] > contactInitialV[2] &&
                qualityCache.step == 1u &&
                qualityCache.impulses.size() > 0u,
            "quality G1 ground-contact step failed"
        );

        // The independent reference solver must produce the same composed
        // state within its converged tolerance, while impulse application
        // still uses the retained factor.
        metalrobo::ArticulatedWorldConfig referenceConfig =
            contactConfig;
        referenceConfig.solverType = MR_SOLVER_REFERENCE_FP64;
        std::vector<double> referenceQ = contactInitialQ;
        std::vector<double> referenceV = contactInitialV;
        metalrobo::ArticulatedWorldCache referenceCache;
        const auto referenceDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                referenceQ,
                referenceV,
                zeroForce,
                {},
                staticGround,
                groundShape,
                referenceConfig,
                referenceCache
            );
        const double solverQError =
            maximumError(qualityQ, referenceQ);
        const double solverVError =
            maximumError(qualityV, referenceV);
        if (solverQError >= 2.0e-9 ||
            solverVError >= 5.0e-7) {
            std::cerr
                << std::setprecision(17)
                << "debug quality_v0=" << qualityV[0]
                << " quality_v2=" << qualityV[2]
                << " reference_v0=" << referenceV[0]
                << " reference_v2=" << referenceV[2]
                << " quality_objective="
                << qualityDiagnostics.qualitySolver.objective
                << " reference_objective="
                << referenceDiagnostics.referenceSolver.objective
                << " quality_impulse0="
                << qualityDiagnostics.qualitySolver.impulses[0]
                << " reference_impulse0="
                << referenceDiagnostics.referenceSolver.impulses[0]
                << '\n';
        }
        require(
            referenceDiagnostics.succeeded() &&
                referenceDiagnostics.referenceSolver.converged() &&
                referenceDiagnostics.contactOperator
                    .materializedDenseInverse &&
                solverQError < 2.0e-9 &&
                solverVError < 5.0e-7,
            "quality/reference composed G1 states disagree q=" +
                std::to_string(solverQError) +
                " v=" + std::to_string(solverVError) +
                " quality_kkt=" +
                std::to_string(
                    qualityDiagnostics.qualitySolver
                        .scaledKktCertificate
                ) +
                " reference_residual=" +
                std::to_string(
                    referenceDiagnostics.referenceSolver
                        .scaledOptimalityResidual
                )
        );

        // Prescribed kinematic motion must be compensated into the adapted
        // target rather than treated as a zero-velocity static world.
        const std::array<MRBodyStateGPU, 1> movingGround{
            makeEnvironmentBody(
                MR_MOTION_KINEMATIC,
                {0.2, 0.0, 0.0}
            ),
        };
        std::vector<double> kinematicQ = contactInitialQ;
        std::vector<double> kinematicV = contactInitialV;
        metalrobo::ArticulatedWorldCache kinematicCache;
        const auto kinematicDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                kinematicQ,
                kinematicV,
                zeroForce,
                {},
                movingGround,
                groundShape,
                contactConfig,
                kinematicCache
            );
        require(
            kinematicDiagnostics.succeeded() &&
                kinematicDiagnostics.adaptation
                    .maximumKinematicTargetCompensation >
                    0.199999,
            "kinematic contact velocity was not compensated"
        );

        // Force a collision-capacity failure from the exact same contacting
        // state and prove no state/cache publication.
        metalrobo::ArticulatedWorldConfig overflowConfig =
            contactConfig;
        overflowConfig.collision.capacities.pairCapacity = 0u;
        std::vector<double> overflowQ = contactInitialQ;
        std::vector<double> overflowV = contactInitialV;
        const std::vector<double> overflowQBefore = overflowQ;
        const std::vector<double> overflowVBefore = overflowV;
        metalrobo::ArticulatedWorldCache overflowCache;
        const auto overflowDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                overflowQ,
                overflowV,
                zeroForce,
                {},
                staticGround,
                groundShape,
                overflowConfig,
                overflowCache
            );
        require(
            !overflowDiagnostics.succeeded() &&
                overflowDiagnostics.code ==
                    MR_STEP_PAIR_CAPACITY_OVERFLOW &&
                overflowDiagnostics.failure ==
                    metalrobo::ArticulatedWorldFailure::collision &&
                overflowQ == overflowQBefore &&
                overflowV == overflowVBefore &&
                overflowCache.step == 0u &&
                overflowCache.manifolds.size() == 0u &&
                overflowCache.impulses.size() == 0u,
            "collision overflow was not transactional"
        );

        std::cout
            << std::setprecision(10)
            << "articulated_world=cpu_transactional"
            << " model=g1"
            << " free_q_error=" << freeQError
            << " free_v_error=" << freeVError
            << " contacts=" << qualityDiagnostics.contactCount
            << " max_normal_impulse="
            << qualityDiagnostics.maximumNormalImpulse
            << " velocity_correction="
            << qualityDiagnostics.maximumVelocityCorrection
            << " factor_contact_velocity_error="
            << qualityDiagnostics
                .solverVelocityRelativeDisagreement
            << " semantic_fingerprint="
            << qualityDiagnostics.semanticFingerprint
            << " common_residual="
            << qualityDiagnostics.constraintResidual
                .maximumNaturalResidual
            << " quality_reference_q_error=" << solverQError
            << " quality_reference_v_error=" << solverVError
            << " kinematic_compensation="
            << kinematicDiagnostics.adaptation
                .maximumKinematicTargetCompensation
            << " deterministic=yes"
            << " controlled_policy_step=yes"
            << " controlled_rollback=yes"
            << " joint_limit_impulse=yes"
            << " mixed_contact_limits=yes"
            << " late_rollback=yes"
            << " overflow_rollback=yes"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "articulated_world=status=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
