#include "metalrobo/ArticulatedWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct Mat3 {
    double m[3][3]{};
};

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finiteState(const MRBodyStateGPU& state) {
    return
        finite(state.position) &&
        finite(state.orientation) &&
        finite(state.linearVelocityAndInverseMass) &&
        finite(state.angularVelocity) &&
        finite(state.inverseInertiaWorldRow0) &&
        finite(state.inverseInertiaWorldRow1) &&
        finite(state.inverseInertiaWorldRow2);
}

bool representableAsFloat(const double value) {
    return
        finite(value) &&
        std::abs(value) <=
            static_cast<double>(std::numeric_limits<float>::max());
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

Mat3 matrix(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    return {{
        {row0.x, row0.y, row0.z},
        {row1.x, row1.y, row1.z},
        {row2.x, row2.y, row2.z},
    }};
}

Mat3 transpose(const Mat3& value) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] = value.m[column][row];
        }
    }
    return result;
}

Mat3 operator*(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            for (std::size_t inner = 0u; inner < 3u; ++inner) {
                result.m[row][column] +=
                    left.m[row][inner] * right.m[inner][column];
            }
        }
    }
    return result;
}

Mat3 rotationMatrix(const std::array<double, 4>& q) {
    const double xx = q[0] * q[0];
    const double yy = q[1] * q[1];
    const double zz = q[2] * q[2];
    const double xy = q[0] * q[1];
    const double xz = q[0] * q[2];
    const double yz = q[1] * q[2];
    const double xw = q[0] * q[3];
    const double yw = q[1] * q[3];
    const double zw = q[2] * q[3];
    return {{
        {1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),
         2.0 * (xz + yw)},
        {2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz),
         2.0 * (yz - xw)},
        {2.0 * (xz - yw), 2.0 * (yz + xw),
         1.0 - 2.0 * (xx + yy)},
    }};
}

MRStepStatusCode codeForDynamics(
    const ArticulatedDynamicsStatus status
) {
    switch (status) {
    case ArticulatedDynamicsStatus::success:
        return MR_STEP_SUCCESS;
    case ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite:
        return MR_STEP_FACTORIZATION_FAILED;
    case ArticulatedDynamicsStatus::nonlinearSolveFailed:
        return MR_STEP_DID_NOT_CONVERGE;
    case ArticulatedDynamicsStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case ArticulatedDynamicsStatus::unsupportedTopology:
        return MR_STEP_UNSUPPORTED;
    case ArticulatedDynamicsStatus::invalidModel:
    case ArticulatedDynamicsStatus::invalidDimensions:
    case ArticulatedDynamicsStatus::nonfiniteInput:
    case ArticulatedDynamicsStatus::invalidQuaternion:
    case ArticulatedDynamicsStatus::jointLimitViolation:
    case ArticulatedDynamicsStatus::bodySpeedLimitViolation:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

MRStepStatusCode codeForConstraintIR(
    const ConstraintIRStatus status
) {
    switch (status) {
    case ConstraintIRStatus::success:
        return MR_STEP_SUCCESS;
    case ConstraintIRStatus::unsupportedSemantics:
        return MR_STEP_UNSUPPORTED;
    case ConstraintIRStatus::invalidCount:
        return MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
    case ConstraintIRStatus::invalidAbiVersion:
    case ConstraintIRStatus::nonCanonicalOrder:
    case ConstraintIRStatus::invalidRange:
    case ConstraintIRStatus::invalidBlock:
    case ConstraintIRStatus::invalidEndpoint:
    case ConstraintIRStatus::nonfiniteData:
    case ConstraintIRStatus::invalidRow:
    case ConstraintIRStatus::invalidCone:
    case ConstraintIRStatus::infeasibleWarmStart:
    case ConstraintIRStatus::invalidEvaluationConfig:
    case ConstraintIRStatus::invalidEvaluationInput:
    case ConstraintIRStatus::invalidResidualInput:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

ArticulatedWorldStepDiagnostics fail(
    ArticulatedWorldStepDiagnostics diagnostics,
    const MRStepStatusCode code,
    const ArticulatedWorldFailure failure
) {
    diagnostics.code = code;
    diagnostics.failure = failure;
    return diagnostics;
}

bool validEnvironmentState(const MRBodyStateGPU& state) {
    return
        finiteState(state) &&
        (
            state.flagsAndIndices[0] == MR_MOTION_STATIC ||
            state.flagsAndIndices[0] == MR_MOTION_KINEMATIC
        ) &&
        state.flagsAndIndices[1] == MR_INVALID_INDEX &&
        state.flagsAndIndices[2] == MR_INVALID_INDEX &&
        state.linearVelocityAndInverseMass.w == 0.0f;
}

bool writeProjectedState(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const ArticulatedBodyKinematics& source,
    MRBodyStateGPU& destination
) {
    if (source.bodyIndex >= model.bodies.size()) {
        return false;
    }
    const MRBodyPropertiesGPU& properties =
        model.bodies[source.bodyIndex];
    if (properties.articulationIndex != articulationIndex ||
        properties.motionType != MR_MOTION_DYNAMIC ||
        !(properties.massAndInverseMass.y > 0.0f) ||
        !finite(properties.massAndInverseMass) ||
        !std::ranges::all_of(
            source.centerOfMassPosition,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.orientation,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.linearVelocity,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.angularVelocity,
            representableAsFloat
        )) {
        return false;
    }

    destination = {};
    destination.position = f4(
        source.centerOfMassPosition[0],
        source.centerOfMassPosition[1],
        source.centerOfMassPosition[2],
        1.0
    );
    destination.orientation = f4(
        source.orientation[0],
        source.orientation[1],
        source.orientation[2],
        source.orientation[3]
    );
    destination.linearVelocityAndInverseMass = f4(
        source.linearVelocity[0],
        source.linearVelocity[1],
        source.linearVelocity[2],
        properties.massAndInverseMass.y
    );
    destination.angularVelocity = f4(
        source.angularVelocity[0],
        source.angularVelocity[1],
        source.angularVelocity[2],
        0.0
    );

    const Mat3 rotation = rotationMatrix(source.orientation);
    const Mat3 inverseBody = matrix(
        properties.inverseInertiaRow0,
        properties.inverseInertiaRow1,
        properties.inverseInertiaRow2
    );
    const Mat3 inverseWorld =
        rotation * inverseBody * transpose(rotation);
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            if (!representableAsFloat(
                    inverseWorld.m[row][column]
                )) {
                return false;
            }
        }
    }
    destination.inverseInertiaWorldRow0 = f4(
        inverseWorld.m[0][0],
        inverseWorld.m[0][1],
        inverseWorld.m[0][2],
        0.0
    );
    destination.inverseInertiaWorldRow1 = f4(
        inverseWorld.m[1][0],
        inverseWorld.m[1][1],
        inverseWorld.m[1][2],
        0.0
    );
    destination.inverseInertiaWorldRow2 = f4(
        inverseWorld.m[2][0],
        inverseWorld.m[2][1],
        inverseWorld.m[2][2],
        0.0
    );
    destination.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    destination.flagsAndIndices[1] = articulationIndex;
    destination.flagsAndIndices[2] = source.bodyIndex;
    return finiteState(destination);
}

bool publishConstraintImpulses(
    const ArticulatedCollisionResult& adaptation,
    const std::span<const double> impulses,
    const std::span<const MRBodyStateGPU> states,
    const std::span<MRContactConstraintGPU> constraints,
    double& maximumNormalImpulse
) {
    if (impulses.size() != 3u * adaptation.contacts.size() ||
        adaptation.sourceConstraintIndices.size() !=
            adaptation.contacts.size()) {
        return false;
    }
    for (std::size_t adaptedIndex = 0u;
         adaptedIndex < adaptation.contacts.size();
         ++adaptedIndex) {
        const std::uint32_t sourceIndex =
            adaptation.sourceConstraintIndices[adaptedIndex];
        if (sourceIndex >= constraints.size()) {
            return false;
        }
        MRContactConstraintGPU& source = constraints[sourceIndex];
        if (source.bodyA >= states.size() ||
            source.bodyB >= states.size()) {
            return false;
        }
        const std::size_t impulseOffset = 3u * adaptedIndex;
        if (!finite(impulses[impulseOffset]) ||
            !finite(impulses[impulseOffset + 1u]) ||
            !finite(impulses[impulseOffset + 2u])) {
            return false;
        }
        const bool swapped =
            states[source.bodyA].flagsAndIndices[0] !=
                MR_MOTION_DYNAMIC;
        // The evaluated adapter's endpoint swap is exactly
        // diag(1, -1, 1) in [normal, u, v] coordinates. Publish those
        // fingerprinted coordinates directly; a world-vector round trip
        // would silently normalize/rebuild the frame and change semantics.
        const double normalImpulse =
            impulses[impulseOffset];
        const double tangentUImpulse =
            swapped
            ? -impulses[impulseOffset + 1u]
            : impulses[impulseOffset + 1u];
        const double tangentVImpulse =
            impulses[impulseOffset + 2u];
        if (!representableAsFloat(normalImpulse) ||
            !representableAsFloat(tangentUImpulse) ||
            !representableAsFloat(tangentVImpulse)) {
            return false;
        }
        source.impulses = f4(
            normalImpulse,
            tangentUImpulse,
            tangentVImpulse,
            0.0
        );
        maximumNormalImpulse = std::max(
            maximumNormalImpulse,
            std::abs(normalImpulse)
        );
    }
    return true;
}

double maximumAbsolute(
    const std::span<const double> values
) {
    double result = 0.0;
    for (const double value : values) {
        result = std::max(result, std::abs(value));
    }
    return result;
}

double maximumDifference(
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

} // namespace

ArticulatedWorldStepDiagnostics stepArticulatedWorldCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<double> q,
    const std::span<double> v,
    const std::span<const double> generalizedForce,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    const std::span<const MRBodyStateGPU> environmentBodies,
    const std::span<const MRShapeGPU> environmentShapes,
    const ArticulatedWorldConfig& config,
    ArticulatedWorldCache& cache,
    const std::span<const CollisionPairExclusion> exclusions
) {
    ArticulatedWorldStepDiagnostics diagnostics;
    diagnostics.adaptation.articulationIndex = articulationIndex;

    std::string modelReason;
    if (!model.valid(&modelReason) ||
        model.articulations.size() != 1u ||
        articulationIndex != 0u ||
        model.articulations[0].firstBody != 0u ||
        model.articulations[0].bodyCount != model.bodies.size() ||
        config.dynamics.integrator !=
            ArticulatedIntegrator::symplecticEuler ||
        config.dynamics.timestep !=
            config.constraintEvaluation.timestep ||
        !(
            config.constraintEvaluation.minimumRegularization >
            0.0
        ) ||
        !finite(
            config.constraintEvaluation.minimumRegularization
        ) ||
        !(config.solverVelocityAgreementTolerance > 0.0) ||
        !finite(config.solverVelocityAgreementTolerance) ||
        config.constraintCapacity >
            MR_MAX_CONTACTS_PER_SOLVER_BATCH ||
        (
            config.solverType != MR_SOLVER_REFERENCE_FP64 &&
            config.solverType != MR_SOLVER_QUALITY_NEWTON
        ) ||
        environmentBodies.size() >
            std::numeric_limits<std::uint32_t>::max() -
                model.bodies.size() ||
        environmentShapes.size() >
            std::numeric_limits<std::uint32_t>::max() -
                model.shapes.size() ||
        !std::ranges::all_of(
            environmentBodies,
            validEnvironmentState
        )) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedWorldFailure::invalidConfiguration
        );
    }

    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if (q.size() != articulation.nq ||
        v.size() != articulation.nv ||
        generalizedForce.size() != articulation.nv ||
        (!externalWrenches.empty() &&
         externalWrenches.size() != model.bodies.size())) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedWorldFailure::invalidConfiguration
        );
    }

    std::vector<double> acceleration(articulation.nv, 0.0);
    diagnostics.freeDynamics = computeArticulatedForwardDynamics(
        model,
        articulationIndex,
        q,
        v,
        generalizedForce,
        externalWrenches,
        acceleration,
        config.dynamics
    );
    if (!diagnostics.freeDynamics.succeeded()) {
        return fail(
            diagnostics,
            codeForDynamics(diagnostics.freeDynamics.status),
            ArticulatedWorldFailure::freeDynamics
        );
    }

    std::vector<double> freeVelocity(v.size(), 0.0);
    for (std::size_t dof = 0u; dof < v.size(); ++dof) {
        freeVelocity[dof] =
            v[dof] + config.dynamics.timestep * acceleration[dof];
        if (!finite(freeVelocity[dof])) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedWorldFailure::freeDynamics
            );
        }
    }

    std::vector<ArticulatedBodyKinematics> bodyKinematics(
        articulation.bodyCount
    );
    diagnostics.collisionKinematics =
        computeArticulatedBodyKinematics(
            model,
            articulationIndex,
            q,
            freeVelocity,
            bodyKinematics,
            config.dynamics
        );
    if (!diagnostics.collisionKinematics.succeeded()) {
        return fail(
            diagnostics,
            codeForDynamics(
                diagnostics.collisionKinematics.status
            ),
            ArticulatedWorldFailure::collisionStateProjection
        );
    }

    std::vector<MRBodyStateGPU> states(
        model.bodies.size() + environmentBodies.size()
    );
    for (const ArticulatedBodyKinematics& body : bodyKinematics) {
        if (!writeProjectedState(
                model,
                articulationIndex,
                body,
                states[body.bodyIndex]
            )) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedWorldFailure::collisionStateProjection
            );
        }
    }
    std::ranges::copy(
        environmentBodies,
        states.begin() + model.bodies.size()
    );

    std::vector<MRShapeGPU> shapes = model.shapes;
    shapes.reserve(model.shapes.size() + environmentShapes.size());
    for (const MRShapeGPU& environmentShape : environmentShapes) {
        if (environmentShape.bodyIndex >=
            environmentBodies.size()) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_INPUT,
                ArticulatedWorldFailure::invalidConfiguration
            );
        }
        MRShapeGPU remapped = environmentShape;
        remapped.bodyIndex +=
            static_cast<std::uint32_t>(model.bodies.size());
        shapes.push_back(remapped);
    }

    ArticulatedWorldCache workingCache = cache;
    const CollisionFrame collision = collideCpuReference(
        shapes,
        states,
        config.collision,
        workingCache.manifolds,
        exclusions
    );
    diagnostics.collision = collision.diagnostics;
    if (!collision.succeeded()) {
        return fail(
            diagnostics,
            collision.diagnostics.code,
            ArticulatedWorldFailure::collision
        );
    }

    ContactAssemblyResult assembly = assembleContactConstraints(
        collision,
        shapes,
        model.materials,
        states,
        config.constraintCapacity
    );
    diagnostics.assembly = assembly.diagnostics;
    diagnostics.contactCount =
        static_cast<std::uint32_t>(assembly.constraints.size());
    diagnostics.maximumPenetration =
        assembly.diagnostics.maximumPenetration;
    if (!assembly.diagnostics.succeeded()) {
        return fail(
            diagnostics,
            assembly.diagnostics.code,
            ArticulatedWorldFailure::contactAssembly
        );
    }

    ++workingCache.step;
    workingCache.impulses.beginStep(workingCache.step);
    workingCache.impulses.seed(assembly.constraints);

    std::vector<double> correctedVelocity = freeVelocity;
    if (!assembly.constraints.empty()) {
        const ConstraintIRV1AdapterResult compiled =
            adaptV1ContactsToConstraintIR(
                assembly.constraints,
                config.constraintAdapter
            );
        diagnostics.constraintCompilation =
            compiled.diagnostics;
        if (!compiled.succeeded()) {
            return fail(
                diagnostics,
                codeForConstraintIR(
                    compiled.diagnostics.status
                ),
                ArticulatedWorldFailure::constraintCompilation
            );
        }
        if (compiled.sourceConstraintIndices.size() !=
            compiled.ir.blocks.size()) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedWorldFailure::constraintCompilation
            );
        }

        const ConstraintIRVelocityResult freeConstraintVelocity =
            computeConstraintIRWorldPointVelocities(
                compiled.ir,
                states
            );
        diagnostics.constraintKinematics =
            freeConstraintVelocity.diagnostics;
        if (!freeConstraintVelocity.succeeded()) {
            return fail(
                diagnostics,
                codeForConstraintIR(
                    freeConstraintVelocity.diagnostics.status
                ),
                ArticulatedWorldFailure::constraintKinematics
            );
        }

        const ConstraintIREvaluationResult evaluation =
            evaluateConstraintIR(
                compiled.ir,
                {
                    freeConstraintVelocity.relativeVelocities,
                    compiled.preSolveVelocities,
                },
                config.constraintEvaluation
            );
        diagnostics.constraintEvaluation =
            evaluation.diagnostics;
        if (!evaluation.succeeded()) {
            return fail(
                diagnostics,
                codeForConstraintIR(
                    evaluation.diagnostics.status
                ),
                ArticulatedWorldFailure::constraintEvaluation
            );
        }
        const ConstraintIREvaluationView semantics =
            makeConstraintIREvaluationView(
                evaluation.evaluated,
                ConstraintIRConsumer::quality
            );
        diagnostics.semanticFingerprint =
            semantics.semanticFingerprint;

        ArticulatedCollisionResult adaptation =
            adaptEvaluatedArticulatedContacts(
                model,
                articulationIndex,
                semantics,
                states,
                config.constraintCapacity
            );
        adaptation.sourceConstraintIndices.reserve(
            adaptation.sourceBlockIndices.size()
        );
        for (const std::uint32_t sourceBlock :
             adaptation.sourceBlockIndices) {
            if (sourceBlock >=
                compiled.sourceConstraintIndices.size()) {
                return fail(
                    diagnostics,
                    MR_STEP_NONFINITE_RESULT,
                    ArticulatedWorldFailure::contactAdaptation
                );
            }
            adaptation.sourceConstraintIndices.push_back(
                compiled.sourceConstraintIndices[sourceBlock]
            );
        }
        diagnostics.adaptation = adaptation.diagnostics;
        if (!adaptation.succeeded()) {
            return fail(
                diagnostics,
                adaptation.diagnostics.code,
                ArticulatedWorldFailure::contactAdaptation
            );
        }

        ArticulatedContactProblem contactProblem;
        const bool needsDenseInverseOracle =
            config.solverType == MR_SOLVER_REFERENCE_FP64;
        diagnostics.contactOperator =
            buildArticulatedContactProblem(
                model,
                articulationIndex,
                q,
                freeVelocity,
                adaptation.contacts,
                contactProblem,
                config.dynamics,
                needsDenseInverseOracle
            );
        if (!diagnostics.contactOperator.succeeded()) {
            const MRStepStatusCode code =
                diagnostics.contactOperator.status ==
                    ArticulatedContactStatus::factorizationFailure
                ? MR_STEP_FACTORIZATION_FAILED
                : MR_STEP_NONFINITE_RESULT;
            return fail(
                diagnostics,
                code,
                ArticulatedWorldFailure::contactOperator
            );
        }

        std::span<const double> impulses;
        std::span<const double> solverVelocity;
        if (config.solverType == MR_SOLVER_QUALITY_NEWTON) {
            ContactSpaceConicProblem contactSpace;
            diagnostics.contactSpaceAdapter =
                buildArticulatedContactSpaceProblem(
                    contactProblem,
                    freeVelocity,
                    contactSpace
                );
            if (!diagnostics.contactSpaceAdapter.succeeded()) {
                return fail(
                    diagnostics,
                    MR_STEP_NONFINITE_RESULT,
                    ArticulatedWorldFailure::contactSpaceAdapter
                );
            }
            diagnostics.qualitySolver =
                solveQualityContactSpaceProblem(
                    contactSpace,
                    config.quality
                );
            if (!diagnostics.qualitySolver.converged()) {
                return fail(
                    diagnostics,
                    diagnostics.qualitySolver.code,
                    ArticulatedWorldFailure::contactSolver
                );
            }
            impulses = diagnostics.qualitySolver.impulses;
            solverVelocity = diagnostics.qualitySolver.velocity;
        } else {
            diagnostics.referenceSolver =
                solveReferenceConicProblem(
                    contactProblem.conic,
                    config.reference
                );
            if (!diagnostics.referenceSolver.converged()) {
                return fail(
                    diagnostics,
                    diagnostics.referenceSolver.code,
                    ArticulatedWorldFailure::contactSolver
                );
            }
            impulses = diagnostics.referenceSolver.impulses;
            solverVelocity = diagnostics.referenceSolver.velocity;
        }

        diagnostics.impulseApplication =
            applyArticulatedContactImpulses(
                contactProblem,
                impulses,
                correctedVelocity
            );
        if (!diagnostics.impulseApplication.succeeded()) {
            return fail(
                diagnostics,
                MR_STEP_FACTORIZATION_FAILED,
                ArticulatedWorldFailure::impulseApplication
            );
        }
        diagnostics.maximumVelocityCorrection =
            maximumDifference(correctedVelocity, freeVelocity);

        std::vector<double> solverCoordinateVelocity;
        if (config.solverType == MR_SOLVER_QUALITY_NEWTON) {
            solverCoordinateVelocity.assign(
                3u * contactProblem.contactCount,
                0.0
            );
            diagnostics.contactSpaceAdapter =
                applyArticulatedContactJacobian(
                    contactProblem,
                    correctedVelocity,
                    solverCoordinateVelocity
                );
            if (!diagnostics.contactSpaceAdapter.succeeded()) {
                return fail(
                    diagnostics,
                    MR_STEP_NONFINITE_RESULT,
                    ArticulatedWorldFailure::contactSpaceAdapter
                );
            }
        } else {
            solverCoordinateVelocity = correctedVelocity;
        }
        diagnostics.solverVelocityRelativeDisagreement =
            maximumDifference(
                solverCoordinateVelocity,
                solverVelocity
            ) /
            (1.0 + std::max(
                maximumAbsolute(solverCoordinateVelocity),
                maximumAbsolute(solverVelocity)
            ));
        if (!finite(
                diagnostics.solverVelocityRelativeDisagreement
            ) ||
            diagnostics.solverVelocityRelativeDisagreement >
                config.solverVelocityAgreementTolerance) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedWorldFailure::solverVelocityMismatch
            );
        }
        if (!publishConstraintImpulses(
                adaptation,
                impulses,
                states,
                assembly.constraints,
                diagnostics.maximumNormalImpulse
            )) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedWorldFailure::impulseApplication
            );
        }

        std::vector<ArticulatedBodyKinematics>
            correctedBodyKinematics(articulation.bodyCount);
        diagnostics.postSolveKinematics =
            computeArticulatedBodyKinematics(
                model,
                articulationIndex,
                q,
                correctedVelocity,
                correctedBodyKinematics,
                config.dynamics
            );
        if (!diagnostics.postSolveKinematics.succeeded()) {
            return fail(
                diagnostics,
                codeForDynamics(
                    diagnostics.postSolveKinematics.status
                ),
                ArticulatedWorldFailure::residualEvaluation
            );
        }
        std::vector<MRBodyStateGPU> postSolveStates = states;
        for (const ArticulatedBodyKinematics& body :
             correctedBodyKinematics) {
            if (!writeProjectedState(
                    model,
                    articulationIndex,
                    body,
                    postSolveStates[body.bodyIndex]
                )) {
                return fail(
                    diagnostics,
                    MR_STEP_NONFINITE_RESULT,
                    ArticulatedWorldFailure::residualEvaluation
                );
            }
        }
        const ConstraintIRVelocityResult postConstraintVelocity =
            computeConstraintIRWorldPointVelocities(
                compiled.ir,
                postSolveStates
            );
        if (!postConstraintVelocity.succeeded()) {
            diagnostics.constraintKinematics =
                postConstraintVelocity.diagnostics;
            return fail(
                diagnostics,
                codeForConstraintIR(
                    postConstraintVelocity.diagnostics.status
                ),
                ArticulatedWorldFailure::residualEvaluation
            );
        }
        std::vector<float> canonicalImpulses(
            compiled.ir.rows.size(),
            0.0F
        );
        for (std::uint32_t blockIndex = 0u;
             blockIndex < compiled.ir.blocks.size();
             ++blockIndex) {
            const ConstraintIRBlock& block =
                compiled.ir.blocks[blockIndex];
            const std::uint32_t sourceIndex =
                compiled.sourceConstraintIndices[blockIndex];
            if (sourceIndex >= assembly.constraints.size() ||
                block.dimension > 4u) {
                return fail(
                    diagnostics,
                    MR_STEP_NONFINITE_RESULT,
                    ArticulatedWorldFailure::residualEvaluation
                );
            }
            const mr_float4 sourceImpulse =
                assembly.constraints[sourceIndex].impulses;
            const std::array<float, 4> values{
                sourceImpulse.x,
                sourceImpulse.y,
                sourceImpulse.z,
                sourceImpulse.w,
            };
            for (std::uint32_t local = 0u;
                 local < block.dimension;
                 ++local) {
                canonicalImpulses[
                    block.impulseOffset + local
                ] = values[local];
            }
        }
        diagnostics.constraintResidual =
            evaluateConstraintIRResidual(
                semantics,
                postConstraintVelocity.relativeVelocities,
                canonicalImpulses,
                config.constraintResidual
            );
        if (!diagnostics.constraintResidual.succeeded() ||
            !diagnostics.constraintResidual.withinTolerance(
                config.constraintResidual
            )) {
            return fail(
                diagnostics,
                diagnostics.constraintResidual.succeeded()
                    ? MR_STEP_DID_NOT_CONVERGE
                    : codeForConstraintIR(
                        diagnostics.constraintResidual.status
                    ),
                ArticulatedWorldFailure::residualEvaluation
            );
        }
    }

    workingCache.impulses.commit(assembly.constraints);
    workingCache.impulses.prune(
        config.impulseCacheMaximumAge
    );

    std::vector<double> integratedQ(q.begin(), q.end());
    diagnostics.integration = integrateArticulatedConfiguration(
        model,
        articulationIndex,
        integratedQ,
        correctedVelocity,
        config.dynamics
    );
    if (!diagnostics.integration.succeeded()) {
        return fail(
            diagnostics,
            codeForDynamics(diagnostics.integration.status),
            ArticulatedWorldFailure::configurationIntegration
        );
    }

    std::ranges::copy(integratedQ, q.begin());
    std::ranges::copy(correctedVelocity, v.begin());
    cache = std::move(workingCache);
    diagnostics.code = MR_STEP_SUCCESS;
    diagnostics.failure = ArticulatedWorldFailure::none;
    return diagnostics;
}

} // namespace metalrobo
