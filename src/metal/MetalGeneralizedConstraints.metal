#include <metal_stdlib>

#include "metalrobo/constraint_ir_shared.h"
#include "metalrobo/engine_types.h"
#include "metalrobo/generalized_constraint_shared.h"

using namespace metal;

namespace {

constant uint kMaxRows = MR_GENERALIZED_CONSTRAINT_MAX_ROWS;

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline void publishFailure(
    device MRGeneralizedConstraintStatusGPU* statuses,
    const uint environment,
    const uint code,
    const uint failingRow,
    const uint failingInverseWork,
    const uint inverseMassCode
) {
    MRGeneralizedConstraintStatusGPU status = {};
    status.code = code;
    status.environment = environment;
    status.failingRow = failingRow;
    status.failingInverseWork = failingInverseWork;
    status.inverseMassCode = inverseMassCode;
    statuses[environment] = status;
}

} // namespace

// Forms J M^-1 J' directly from the response columns emitted by the
// multi-articulation inverse-mass grid. One GPU thread owns one matrix entry;
// no dense generalized mass matrix or inverse is ever materialized.
kernel void mr_generalized_constraint_delassus(
    device const MRGeneralizedConstraintDispatchGPU& dispatch
        [[buffer(0)]],
    device const float* jacobian [[buffer(1)]],
    device const float* responseColumns [[buffer(2)]],
    device float* delassus [[buffer(3)]],
    uint3 index [[thread_position_in_grid]]
) {
    const uint column = index.x;
    const uint row = index.y;
    const uint environment = index.z;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.rowCount ||
        column >= dispatch.rowCount) {
        return;
    }
    const uint jacobianBase = row * dispatch.nv;
    const uint responseBase =
        (environment * dispatch.rowCount + column) *
        dispatch.nv;
    float value = 0.0f;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        value = fma(
            jacobian[jacobianBase + dof],
            responseColumns[responseBase + dof],
            value
        );
    }
    delassus[
        (environment * dispatch.rowCount + row) *
            dispatch.rowCount +
        column
    ] = value;
}

// Deterministic scalar-block projected Gauss-Seidel for generalized
// ConstraintIR rows. Cross-articulation coupling enters through the global
// sparse Jacobian and block-diagonal response columns. This kernel is the
// throughput correctness path; the semismooth quality solver can consume the
// same Delassus operator without changing semantics.
kernel void mr_generalized_constraint_solve(
    device const MRGeneralizedConstraintDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRConstraintIRRowGPU* sourceRows [[buffer(1)]],
    device const float* warmImpulses [[buffer(2)]],
    device const float* jacobian [[buffer(3)]],
    device const float* freeVelocity [[buffer(4)]],
    device const float* responseColumns [[buffer(5)]],
    device const MRInverseMassStatusGPU* inverseStatuses
        [[buffer(6)]],
    device const float* delassus [[buffer(7)]],
    device float* outputImpulses [[buffer(8)]],
    device float* outputVelocity [[buffer(9)]],
    device MRGeneralizedConstraintStatusGPU* statuses
        [[buffer(10)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane != 0u || environment >= dispatch.environmentCount) {
        return;
    }
    if (dispatch.abiVersion !=
            MR_GENERALIZED_CONSTRAINT_ABI_VERSION ||
        dispatch.environmentCount == 0u ||
        dispatch.nv == 0u ||
        dispatch.rowCount == 0u ||
        dispatch.rowCount > kMaxRows ||
        dispatch.inverseWorkCount == 0u ||
        dispatch.solverIterations == 0u ||
        dispatch.reserved0 != 0u ||
        dispatch.reserved1 != 0u ||
        !finite4(dispatch.evaluation0) ||
        !finite4(dispatch.evaluation1) ||
        !(dispatch.evaluation0.x > 0.0f) ||
        dispatch.evaluation0.y < 0.0f ||
        dispatch.evaluation0.z < 0.0f ||
        dispatch.evaluation0.w < 0.0f ||
        dispatch.evaluation1.x < 0.0f ||
        !(dispatch.evaluation1.y > 0.0f) ||
        !(dispatch.evaluation1.z > 0.0f)) {
        publishFailure(
            statuses,
            environment,
            MR_GENERALIZED_CONSTRAINT_INVALID_DISPATCH,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVERSE_MASS_SUCCESS
        );
        return;
    }
    for (uint work = 0u;
         work < dispatch.inverseWorkCount;
         ++work) {
        const MRInverseMassStatusGPU inverse =
            inverseStatuses[
                work * dispatch.environmentCount + environment
            ];
        if (inverse.code != MR_INVERSE_MASS_SUCCESS) {
            publishFailure(
                statuses,
                environment,
                MR_GENERALIZED_CONSTRAINT_INVERSE_MASS_FAILED,
                MR_INVALID_INDEX,
                work,
                inverse.code
            );
            return;
        }
    }

    thread float impulse[kMaxRows];
    thread float physicalVelocity[kMaxRows];
    thread float targetVelocity[kMaxRows];
    thread float regularization[kMaxRows];
    thread float lowerBound[kMaxRows];
    thread float upperBound[kMaxRows];

    const float timestep = dispatch.evaluation0.x;
    const float maximumStabilization = dispatch.evaluation0.z;
    const float minimumTauRatio = dispatch.evaluation0.w;
    const float minimumRegularization = dispatch.evaluation1.x;
    const float tolerance = dispatch.evaluation1.y;
    const float diagonalFloor = dispatch.evaluation1.z;
    const uint velocityBase = environment * dispatch.nv;
    const uint delassusBase =
        environment * dispatch.rowCount * dispatch.rowCount;

    float minimumDiagonal = INFINITY;
    float maximumDiagonal = 0.0f;
    for (uint row = 0u; row < dispatch.rowCount; ++row) {
        const MRConstraintIRRowGPU source = sourceRows[row];
        if (!finite4(source.direction) ||
            !isfinite(source.positionError) ||
            !isfinite(source.targetVelocity) ||
            !isfinite(source.compliance) ||
            !isfinite(source.dissipation) ||
            !isfinite(source.timeConstant) ||
            !isfinite(source.dampingRatio) ||
            !isfinite(source.impulseLower) ||
            !isfinite(source.impulseUpper) ||
            source.compliance < 0.0f ||
            source.dissipation < 0.0f ||
            source.timeConstant < 0.0f ||
            source.dampingRatio < 0.0f ||
            source.impulseLower > source.impulseUpper) {
            publishFailure(
                statuses,
                environment,
                MR_GENERALIZED_CONSTRAINT_INVALID_ROW,
                row,
                MR_INVALID_INDEX,
                MR_INVERSE_MASS_SUCCESS
            );
            return;
        }

        float relative = 0.0f;
        const uint jacobianBase = row * dispatch.nv;
        for (uint dof = 0u; dof < dispatch.nv; ++dof) {
            const float velocity =
                freeVelocity[velocityBase + dof];
            const float coefficient =
                jacobian[jacobianBase + dof];
            if (!isfinite(velocity) || !isfinite(coefficient)) {
                publishFailure(
                    statuses,
                    environment,
                    MR_GENERALIZED_CONSTRAINT_NONFINITE_INPUT,
                    row,
                    MR_INVALID_INDEX,
                    MR_INVERSE_MASS_SUCCESS
                );
                return;
            }
            relative = fma(coefficient, velocity, relative);
        }

        float stabilization = 0.0f;
        if ((source.flags &
             MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED) != 0u) {
            const float tau = max(
                source.timeConstant,
                minimumTauRatio * timestep
            );
            if (!(tau > 0.0f) || !isfinite(tau)) {
                publishFailure(
                    statuses,
                    environment,
                    MR_GENERALIZED_CONSTRAINT_INVALID_ROW,
                    row,
                    MR_INVALID_INDEX,
                    MR_INVERSE_MASS_SUCCESS
                );
                return;
            }
            float positionError = source.positionError;
            const bool unilateral =
                (source.flags &
                 MR_CONSTRAINT_IR_ROW_UNILATERAL) != 0u;
            if (unilateral) {
                positionError = min(positionError, 0.0f);
            }
            const float ratio = timestep / tau;
            const float denominator =
                1.0f +
                2.0f * source.dampingRatio * ratio +
                ratio * ratio;
            if (unilateral) {
                stabilization = max(
                    -timestep * positionError /
                        (tau * tau * denominator),
                    0.0f
                );
            } else {
                stabilization =
                    (
                        relative - source.targetVelocity -
                        timestep * positionError / (tau * tau)
                    ) /
                    denominator;
            }
            stabilization = clamp(
                stabilization,
                unilateral ? 0.0f : -maximumStabilization,
                maximumStabilization
            );
        }
        targetVelocity[row] =
            source.targetVelocity + stabilization;
        regularization[row] = max(
            source.compliance / (timestep * timestep) +
                source.dissipation / timestep,
            minimumRegularization
        );
        lowerBound[row] = source.impulseLower;
        upperBound[row] = source.impulseUpper;
        impulse[row] = clamp(
            warmImpulses[row],
            lowerBound[row],
            upperBound[row]
        );
        physicalVelocity[row] = relative;

        const float diagonal =
            delassus[
                delassusBase + row * dispatch.rowCount + row
            ] + regularization[row];
        if (!(diagonal > diagonalFloor) ||
            !isfinite(diagonal) ||
            !isfinite(targetVelocity[row]) ||
            !isfinite(regularization[row]) ||
            !isfinite(impulse[row]) ||
            !isfinite(physicalVelocity[row])) {
            publishFailure(
                statuses,
                environment,
                MR_GENERALIZED_CONSTRAINT_SINGULAR_ROW,
                row,
                MR_INVALID_INDEX,
                MR_INVERSE_MASS_SUCCESS
            );
            return;
        }
        minimumDiagonal = min(minimumDiagonal, diagonal);
        maximumDiagonal = max(maximumDiagonal, diagonal);
    }

    // Warm starts are physical impulses, so first apply W*lambda to J*v.
    for (uint column = 0u;
         column < dispatch.rowCount;
         ++column) {
        const float value = impulse[column];
        if (value == 0.0f) {
            continue;
        }
        for (uint row = 0u; row < dispatch.rowCount; ++row) {
            physicalVelocity[row] = fma(
                delassus[
                    delassusBase +
                    row * dispatch.rowCount +
                    column
                ],
                value,
                physicalVelocity[row]
            );
        }
    }

    bool converged = false;
    uint completedIterations = 0u;
    float maximumDelta = INFINITY;
    for (uint iteration = 0u;
         iteration < dispatch.solverIterations;
         ++iteration) {
        maximumDelta = 0.0f;
        for (uint row = 0u; row < dispatch.rowCount; ++row) {
            const float diagonal =
                delassus[
                    delassusBase +
                    row * dispatch.rowCount +
                    row
                ] + regularization[row];
            const float oldImpulse = impulse[row];
            const float gradient =
                physicalVelocity[row] -
                targetVelocity[row] +
                regularization[row] * oldImpulse;
            const float candidate = clamp(
                oldImpulse - gradient / diagonal,
                lowerBound[row],
                upperBound[row]
            );
            const float delta = candidate - oldImpulse;
            if (!isfinite(candidate) || !isfinite(delta)) {
                publishFailure(
                    statuses,
                    environment,
                    MR_GENERALIZED_CONSTRAINT_NONFINITE_RESULT,
                    row,
                    MR_INVALID_INDEX,
                    MR_INVERSE_MASS_SUCCESS
                );
                return;
            }
            impulse[row] = candidate;
            maximumDelta = max(maximumDelta, abs(delta));
            for (uint affected = 0u;
                 affected < dispatch.rowCount;
                 ++affected) {
                physicalVelocity[affected] = fma(
                    delassus[
                        delassusBase +
                        affected * dispatch.rowCount +
                        row
                    ],
                    delta,
                    physicalVelocity[affected]
                );
            }
        }
        completedIterations = iteration + 1u;
        if (maximumDelta <= tolerance) {
            float iterationResidual = 0.0f;
            for (uint row = 0u;
                 row < dispatch.rowCount;
                 ++row) {
                const float gradient =
                    physicalVelocity[row] -
                    targetVelocity[row] +
                    regularization[row] * impulse[row];
                const float projected = clamp(
                    impulse[row] - gradient,
                    lowerBound[row],
                    upperBound[row]
                );
                iterationResidual = max(
                    iterationResidual,
                    abs(impulse[row] - projected)
                );
            }
            if (isfinite(iterationResidual) &&
                iterationResidual <= tolerance) {
                converged = true;
                break;
            }
        }
    }

    float naturalResidual = 0.0f;
    for (uint row = 0u; row < dispatch.rowCount; ++row) {
        const float gradient =
            physicalVelocity[row] -
            targetVelocity[row] +
            regularization[row] * impulse[row];
        const float projected = clamp(
            impulse[row] - gradient,
            lowerBound[row],
            upperBound[row]
        );
        naturalResidual = max(
            naturalResidual,
            abs(impulse[row] - projected)
        );
    }
    if (!converged ||
        !isfinite(naturalResidual) ||
        naturalResidual > tolerance) {
        MRGeneralizedConstraintStatusGPU status = {};
        status.code =
            MR_GENERALIZED_CONSTRAINT_DID_NOT_CONVERGE;
        status.environment = environment;
        status.iterations = completedIterations;
        status.failingRow = MR_INVALID_INDEX;
        status.failingInverseWork = MR_INVALID_INDEX;
        status.inverseMassCode = MR_INVERSE_MASS_SUCCESS;
        status.activeRows = dispatch.rowCount;
        status.diagnostics = float4(
            maximumDelta,
            naturalResidual,
            minimumDiagonal,
            maximumDiagonal
        );
        statuses[environment] = status;
        return;
    }

    for (uint row = 0u; row < dispatch.rowCount; ++row) {
        outputImpulses[
            environment * dispatch.rowCount + row
        ] = impulse[row];
    }
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        float velocity = freeVelocity[velocityBase + dof];
        for (uint row = 0u; row < dispatch.rowCount; ++row) {
            velocity = fma(
                responseColumns[
                    (environment * dispatch.rowCount + row) *
                        dispatch.nv +
                    dof
                ],
                impulse[row],
                velocity
            );
        }
        if (!isfinite(velocity)) {
            publishFailure(
                statuses,
                environment,
                MR_GENERALIZED_CONSTRAINT_NONFINITE_RESULT,
                dof,
                MR_INVALID_INDEX,
                MR_INVERSE_MASS_SUCCESS
            );
            return;
        }
        outputVelocity[velocityBase + dof] = velocity;
    }

    MRGeneralizedConstraintStatusGPU status = {};
    status.code = MR_GENERALIZED_CONSTRAINT_SUCCESS;
    status.environment = environment;
    status.iterations = completedIterations;
    status.failingRow = MR_INVALID_INDEX;
    status.failingInverseWork = MR_INVALID_INDEX;
    status.inverseMassCode = MR_INVERSE_MASS_SUCCESS;
    status.activeRows = dispatch.rowCount;
    status.diagnostics = float4(
        maximumDelta,
        naturalResidual,
        minimumDiagonal,
        maximumDiagonal
    );
    statuses[environment] = status;
}

namespace {

constant float kQualityArmijo = 1.0e-4f;
constant float kQualityMinimumCGDenominator = 1.0e-30f;

inline float qualityGroupDot(
    threadgroup const float* left,
    threadgroup const float* right,
    const uint count,
    const uint lane
) {
    float partial = 0.0f;
    for (uint index = lane; index < count; index += 32u) {
        partial = fma(left[index], right[index], partial);
    }
    return simd_sum(partial);
}

inline void qualityHessianAction(
    device const float* delassus,
    threadgroup const float* regularization,
    threadgroup const float* input,
    threadgroup float* output,
    const uint rowCount,
    const uint lane
) {
    for (uint row = lane; row < rowCount; row += 32u) {
        float value = regularization[row] * input[row];
        for (uint column = 0u;
             column < rowCount;
             ++column) {
            value = fma(
                delassus[row * rowCount + column],
                input[column],
                value
            );
        }
        output[row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// F(x) = x - P_[l,u](x - S*(H*x+c)), with positive diagonal S. The selected
// derivative is deterministic at bounds: only a strictly interior projection
// has D=1.
inline float qualityEvaluateNaturalMap(
    device const float* delassus,
    threadgroup const float* regularization,
    threadgroup const float* linear,
    threadgroup const float* lower,
    threadgroup const float* upper,
    threadgroup const float* stepScale,
    threadgroup const float* value,
    threadgroup float* gradient,
    threadgroup float* projection,
    threadgroup float* derivative,
    threadgroup float* residual,
    const uint rowCount,
    const uint lane
) {
    qualityHessianAction(
        delassus,
        regularization,
        value,
        gradient,
        rowCount,
        lane
    );
    float residualSquared = 0.0f;
    for (uint row = lane; row < rowCount; row += 32u) {
        gradient[row] += linear[row];
        const float projectionInput =
            value[row] - stepScale[row] * gradient[row];
        projection[row] = clamp(
            projectionInput,
            lower[row],
            upper[row]
        );
        derivative[row] =
            projectionInput > lower[row] &&
                projectionInput < upper[row]
            ? 1.0f
            : 0.0f;
        residual[row] = value[row] - projection[row];
        residualSquared = fma(
            residual[row],
            residual[row],
            residualSquared
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return 0.5f * simd_sum(residualSquared);
}

inline void qualityJacobianAction(
    device const float* delassus,
    threadgroup const float* regularization,
    threadgroup const float* derivative,
    threadgroup const float* stepScale,
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* hessianWork,
    const uint rowCount,
    const uint lane
) {
    qualityHessianAction(
        delassus,
        regularization,
        input,
        hessianWork,
        rowCount,
        lane
    );
    for (uint row = lane; row < rowCount; row += 32u) {
        output[row] =
            input[row] -
            derivative[row] *
                (
                    input[row] -
                    stepScale[row] * hessianWork[row]
                );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void qualityJacobianTransposeAction(
    device const float* delassus,
    threadgroup const float* regularization,
    threadgroup const float* derivative,
    threadgroup const float* stepScale,
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* derivativeWork,
    threadgroup float* hessianWork,
    const uint rowCount,
    const uint lane
) {
    for (uint row = lane; row < rowCount; row += 32u) {
        derivativeWork[row] =
            stepScale[row] * derivative[row] * input[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qualityHessianAction(
        delassus,
        regularization,
        derivativeWork,
        hessianWork,
        rowCount,
        lane
    );
    for (uint row = lane; row < rowCount; row += 32u) {
        output[row] =
            input[row] -
            derivative[row] * input[row] +
            hessianWork[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void qualityNormalEquationAction(
    device const float* delassus,
    threadgroup const float* regularization,
    threadgroup const float* derivative,
    threadgroup const float* stepScale,
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* jacobianWork,
    threadgroup float* hessianWorkA,
    threadgroup float* derivativeWork,
    threadgroup float* hessianWorkB,
    const uint rowCount,
    const float normalRegularization,
    const uint lane
) {
    qualityJacobianAction(
        delassus,
        regularization,
        derivative,
        stepScale,
        input,
        jacobianWork,
        hessianWorkA,
        rowCount,
        lane
    );
    qualityJacobianTransposeAction(
        delassus,
        regularization,
        derivative,
        stepScale,
        jacobianWork,
        output,
        derivativeWork,
        hessianWorkB,
        rowCount,
        lane
    );
    for (uint row = lane; row < rowCount; row += 32u) {
        output[row] = fma(
            normalRegularization,
            input[row],
            output[row]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

} // namespace

// Safeguarded semismooth Newton solve for scalar bilateral, unilateral, and
// bounded ConstraintIR rows. It consumes the same ABA-built Delassus matrix
// and response columns as the legacy projected reference. This is the quality path for
// generalized/equality constraints; exact friction-cone blocks remain in the
// contact quality kernel.
kernel void mr_generalized_constraint_quality_solve(
    device const MRGeneralizedConstraintDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRConstraintIRRowGPU* sourceRows [[buffer(1)]],
    device const float* warmImpulses [[buffer(2)]],
    device const float* jacobian [[buffer(3)]],
    device const float* freeVelocity [[buffer(4)]],
    device const float* responseColumns [[buffer(5)]],
    device const MRInverseMassStatusGPU* inverseStatuses
        [[buffer(6)]],
    device const float* delassusMatrices [[buffer(7)]],
    device float* outputImpulses [[buffer(8)]],
    device float* outputVelocity [[buffer(9)]],
    device MRGeneralizedConstraintStatusGPU* statuses
        [[buffer(10)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    threadgroup float value[kMaxRows];
    threadgroup float candidate[kMaxRows];
    threadgroup float residual[kMaxRows];
    threadgroup float candidateResidual[kMaxRows];
    threadgroup float gradient[kMaxRows];
    threadgroup float candidateGradient[kMaxRows];
    threadgroup float projection[kMaxRows];
    threadgroup float candidateProjection[kMaxRows];
    threadgroup float derivative[kMaxRows];
    threadgroup float candidateDerivative[kMaxRows];
    threadgroup float linear[kMaxRows];
    threadgroup float regularization[kMaxRows];
    threadgroup float lowerBound[kMaxRows];
    threadgroup float upperBound[kMaxRows];
    threadgroup float stepScale[kMaxRows];
    threadgroup float direction[kMaxRows];
    threadgroup float rightHandSide[kMaxRows];
    threadgroup float cgResidual[kMaxRows];
    threadgroup float cgDirection[kMaxRows];
    threadgroup float cgAction[kMaxRows];
    threadgroup float workA[kMaxRows];
    threadgroup float workB[kMaxRows];
    threadgroup float workC[kMaxRows];
    threadgroup float workD[kMaxRows];
    threadgroup float scalarWork[16u];
    threadgroup atomic_uint publishedSetupCode;
    threadgroup uint setupFailingRow;
    threadgroup uint setupFailingInverseWork;
    threadgroup uint setupInverseCode;

    if (lane == 0u) {
        uint setupCode = MR_GENERALIZED_CONSTRAINT_SUCCESS;
        setupFailingRow = MR_INVALID_INDEX;
        setupFailingInverseWork = MR_INVALID_INDEX;
        setupInverseCode = MR_INVERSE_MASS_SUCCESS;
        scalarWork[0] = INFINITY;
        scalarWork[1] = 0.0f;
        scalarWork[2] = 0.0f;
        scalarWork[3] = 0.0f;

        if (dispatch.abiVersion !=
                MR_GENERALIZED_CONSTRAINT_ABI_VERSION ||
            dispatch.environmentCount == 0u ||
            dispatch.nv == 0u ||
            dispatch.rowCount == 0u ||
            dispatch.rowCount > kMaxRows ||
            dispatch.inverseWorkCount == 0u ||
            dispatch.solverIterations == 0u ||
            dispatch.reserved0 == 0u ||
            dispatch.reserved1 == 0u ||
            !finite4(dispatch.evaluation0) ||
            !finite4(dispatch.evaluation1) ||
            !(dispatch.evaluation0.x > 0.0f) ||
            dispatch.evaluation0.y < 0.0f ||
            dispatch.evaluation0.z < 0.0f ||
            dispatch.evaluation0.w < 0.0f ||
            dispatch.evaluation1.x < 0.0f ||
            !(dispatch.evaluation1.y > 0.0f) ||
            !(dispatch.evaluation1.z > 0.0f) ||
            !(dispatch.evaluation1.w > 0.0f)) {
            setupCode =
                MR_GENERALIZED_CONSTRAINT_INVALID_DISPATCH;
        }

        for (uint work = 0u;
             setupCode == MR_GENERALIZED_CONSTRAINT_SUCCESS &&
                 work < dispatch.inverseWorkCount;
             ++work) {
            const MRInverseMassStatusGPU inverse =
                inverseStatuses[
                    work * dispatch.environmentCount +
                    environment
                ];
            if (inverse.code != MR_INVERSE_MASS_SUCCESS) {
                setupCode =
                    MR_GENERALIZED_CONSTRAINT_INVERSE_MASS_FAILED;
                setupFailingInverseWork = work;
                setupInverseCode = inverse.code;
            }
        }

        const float timestep = dispatch.evaluation0.x;
        const float maximumStabilization =
            dispatch.evaluation0.z;
        const float minimumTauRatio =
            dispatch.evaluation0.w;
        const float minimumRegularization =
            dispatch.evaluation1.x;
        const uint velocityBase = environment * dispatch.nv;
        device const float* delassus =
            delassusMatrices +
            environment * dispatch.rowCount *
                dispatch.rowCount;
        for (uint row = 0u;
             setupCode == MR_GENERALIZED_CONSTRAINT_SUCCESS &&
                 row < dispatch.rowCount;
             ++row) {
            const MRConstraintIRRowGPU source =
                sourceRows[row];
            if (!finite4(source.direction) ||
                !isfinite(source.positionError) ||
                !isfinite(source.targetVelocity) ||
                !isfinite(source.compliance) ||
                !isfinite(source.dissipation) ||
                !isfinite(source.timeConstant) ||
                !isfinite(source.dampingRatio) ||
                !isfinite(source.impulseLower) ||
                !isfinite(source.impulseUpper) ||
                source.compliance < 0.0f ||
                source.dissipation < 0.0f ||
                source.timeConstant < 0.0f ||
                source.dampingRatio < 0.0f ||
                source.impulseLower > source.impulseUpper) {
                setupCode =
                    MR_GENERALIZED_CONSTRAINT_INVALID_ROW;
                setupFailingRow = row;
                break;
            }
            float relative = 0.0f;
            const uint jacobianBase = row * dispatch.nv;
            for (uint dof = 0u;
                 dof < dispatch.nv;
                 ++dof) {
                const float velocity =
                    freeVelocity[velocityBase + dof];
                const float coefficient =
                    jacobian[jacobianBase + dof];
                if (!isfinite(velocity) ||
                    !isfinite(coefficient)) {
                    setupCode =
                        MR_GENERALIZED_CONSTRAINT_NONFINITE_INPUT;
                    setupFailingRow = row;
                    break;
                }
                relative = fma(
                    coefficient,
                    velocity,
                    relative
                );
            }
            if (setupCode !=
                    MR_GENERALIZED_CONSTRAINT_SUCCESS) {
                break;
            }
            float stabilization = 0.0f;
            if ((source.flags &
                 MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED) !=
                    0u) {
                const float tau = max(
                    source.timeConstant,
                    minimumTauRatio * timestep
                );
                if (!(tau > 0.0f) || !isfinite(tau)) {
                    setupCode =
                        MR_GENERALIZED_CONSTRAINT_INVALID_ROW;
                    setupFailingRow = row;
                    break;
                }
                float positionError = source.positionError;
                const bool unilateral =
                    (source.flags &
                     MR_CONSTRAINT_IR_ROW_UNILATERAL) != 0u;
                if (unilateral) {
                    positionError = min(positionError, 0.0f);
                }
                const float ratio = timestep / tau;
                const float denominator =
                    1.0f +
                    2.0f * source.dampingRatio * ratio +
                    ratio * ratio;
                if (unilateral) {
                    stabilization = max(
                        -timestep * positionError /
                            (tau * tau * denominator),
                        0.0f
                    );
                } else {
                    stabilization =
                        (
                            relative -
                            source.targetVelocity -
                            timestep * positionError /
                                (tau * tau)
                        ) / denominator;
                }
                stabilization = clamp(
                    stabilization,
                    unilateral
                        ? 0.0f
                        : -maximumStabilization,
                    maximumStabilization
                );
            }
            const float target =
                source.targetVelocity + stabilization;
            regularization[row] = max(
                source.compliance /
                    (timestep * timestep) +
                    source.dissipation / timestep,
                minimumRegularization
            );
            linear[row] = relative - target;
            lowerBound[row] = source.impulseLower;
            upperBound[row] = source.impulseUpper;
            value[row] = clamp(
                warmImpulses[row],
                lowerBound[row],
                upperBound[row]
            );
            const float diagonal =
                delassus[row * dispatch.rowCount + row] +
                regularization[row];
            float rowSum = regularization[row];
            for (uint column = 0u;
                 column < dispatch.rowCount;
                 ++column) {
                const float entry = delassus[
                    row * dispatch.rowCount + column
                ];
                if (!isfinite(entry)) {
                    setupCode =
                        MR_GENERALIZED_CONSTRAINT_NONFINITE_INPUT;
                    setupFailingRow = row;
                    break;
                }
                rowSum += abs(entry);
            }
            if (setupCode !=
                    MR_GENERALIZED_CONSTRAINT_SUCCESS) {
                break;
            }
            if (!(diagonal > dispatch.evaluation1.z) ||
                !isfinite(diagonal) ||
                !isfinite(rowSum) ||
                !(rowSum > 0.0f) ||
                !isfinite(linear[row]) ||
                !isfinite(regularization[row]) ||
                !isfinite(value[row])) {
                setupCode =
                    MR_GENERALIZED_CONSTRAINT_SINGULAR_ROW;
                setupFailingRow = row;
                break;
            }
            scalarWork[0] = min(scalarWork[0], diagonal);
            scalarWork[1] = max(scalarWork[1], diagonal);
            stepScale[row] = 1.0f / rowSum;
        }
        atomic_store_explicit(
            &publishedSetupCode,
            setupCode,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint setupCode = atomic_load_explicit(
        &publishedSetupCode,
        memory_order_relaxed
    );
    if (setupCode != MR_GENERALIZED_CONSTRAINT_SUCCESS) {
        if (lane == 0u) {
            publishFailure(
                statuses,
                environment,
                setupCode,
                setupFailingRow,
                setupFailingInverseWork,
                setupInverseCode
            );
        }
        return;
    }

    device const float* delassus =
        delassusMatrices +
        environment * dispatch.rowCount *
            dispatch.rowCount;
    const float tolerance = dispatch.evaluation1.y;
    const float normalRegularization =
        dispatch.evaluation1.w;
    bool converged = false;
    uint completedIterations = 0u;
    float lastMaximumDelta = INFINITY;
    float physicalNaturalResidual = INFINITY;

    for (uint iteration = 0u;
         iteration < dispatch.solverIterations;
         ++iteration) {
        const float merit = qualityEvaluateNaturalMap(
            delassus,
            regularization,
            linear,
            lowerBound,
            upperBound,
            stepScale,
            value,
            gradient,
            projection,
            derivative,
            residual,
            dispatch.rowCount,
            lane
        );
        float localPhysicalResidual = 0.0f;
        for (uint row = lane;
             row < dispatch.rowCount;
             row += 32u) {
            const float projected = clamp(
                value[row] - gradient[row],
                lowerBound[row],
                upperBound[row]
            );
            localPhysicalResidual = max(
                localPhysicalResidual,
                abs(value[row] - projected)
            );
        }
        physicalNaturalResidual =
            simd_max(localPhysicalResidual);
        completedIterations = iteration;
        if (isfinite(physicalNaturalResidual) &&
            physicalNaturalResidual <= tolerance) {
            converged = true;
            break;
        }

        qualityJacobianTransposeAction(
            delassus,
            regularization,
            derivative,
            stepScale,
            residual,
            rightHandSide,
            workA,
            workB,
            dispatch.rowCount,
            lane
        );
        for (uint row = lane;
             row < dispatch.rowCount;
             row += 32u) {
            rightHandSide[row] = -rightHandSide[row];
            direction[row] = 0.0f;
            cgResidual[row] = rightHandSide[row];
            cgDirection[row] = rightHandSide[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float residualSquared = qualityGroupDot(
            cgResidual,
            cgResidual,
            dispatch.rowCount,
            lane
        );
        const float cgTargetSquared = max(
            residualSquared * 1.0e-12f,
            1.0e-24f
        );
        bool cgValid = isfinite(residualSquared);
        for (uint cgIteration = 0u;
             cgValid &&
                 cgIteration < dispatch.reserved0 &&
                 residualSquared > cgTargetSquared;
             ++cgIteration) {
            qualityNormalEquationAction(
                delassus,
                regularization,
                derivative,
                stepScale,
                cgDirection,
                cgAction,
                workA,
                workB,
                workC,
                workD,
                dispatch.rowCount,
                normalRegularization,
                lane
            );
            const float denominator = qualityGroupDot(
                cgDirection,
                cgAction,
                dispatch.rowCount,
                lane
            );
            if (!isfinite(denominator) ||
                !(denominator >
                    kQualityMinimumCGDenominator)) {
                cgValid = false;
                break;
            }
            const float alpha =
                residualSquared / denominator;
            for (uint row = lane;
                 row < dispatch.rowCount;
                 row += 32u) {
                direction[row] = fma(
                    alpha,
                    cgDirection[row],
                    direction[row]
                );
                cgResidual[row] = fma(
                    -alpha,
                    cgAction[row],
                    cgResidual[row]
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float nextResidualSquared = qualityGroupDot(
                cgResidual,
                cgResidual,
                dispatch.rowCount,
                lane
            );
            if (!isfinite(nextResidualSquared)) {
                cgValid = false;
                break;
            }
            const float beta =
                nextResidualSquared /
                max(
                    residualSquared,
                    kQualityMinimumCGDenominator
                );
            for (uint row = lane;
                 row < dispatch.rowCount;
                 row += 32u) {
                cgDirection[row] = fma(
                    beta,
                    cgDirection[row],
                    cgResidual[row]
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            residualSquared = nextResidualSquared;
        }

        // rightHandSide is -J'F, so J'F dot p must be negative.
        float directionalDerivative = -qualityGroupDot(
            rightHandSide,
            direction,
            dispatch.rowCount,
            lane
        );
        if (!cgValid ||
            !isfinite(directionalDerivative) ||
            !(directionalDerivative < 0.0f)) {
            for (uint row = lane;
                 row < dispatch.rowCount;
                 row += 32u) {
                direction[row] = rightHandSide[row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            directionalDerivative = -qualityGroupDot(
                rightHandSide,
                rightHandSide,
                dispatch.rowCount,
                lane
            );
        }

        bool accepted = false;
        float acceptedDelta = INFINITY;
        float stepLength = 1.0f;
        for (uint lineSearch = 0u;
             lineSearch < dispatch.reserved1;
             ++lineSearch) {
            float localDelta = 0.0f;
            for (uint row = lane;
                 row < dispatch.rowCount;
                 row += 32u) {
                candidate[row] = clamp(
                    fma(
                        stepLength,
                        direction[row],
                        value[row]
                    ),
                    lowerBound[row],
                    upperBound[row]
                );
                localDelta = max(
                    localDelta,
                    abs(candidate[row] - value[row])
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float candidateMerit =
                qualityEvaluateNaturalMap(
                    delassus,
                    regularization,
                    linear,
                    lowerBound,
                    upperBound,
                    stepScale,
                    candidate,
                    candidateGradient,
                    candidateProjection,
                    candidateDerivative,
                    candidateResidual,
                    dispatch.rowCount,
                    lane
                );
            if (isfinite(candidateMerit) &&
                candidateMerit <=
                    merit +
                    kQualityArmijo *
                        stepLength *
                        directionalDerivative) {
                accepted = true;
                acceptedDelta = simd_max(localDelta);
                break;
            }
            stepLength *= 0.5f;
        }
        if (!accepted) {
            // Globally safe diagonally scaled projected-gradient fallback.
            float localDelta = 0.0f;
            for (uint row = lane;
                 row < dispatch.rowCount;
                 row += 32u) {
                candidate[row] = projection[row];
                localDelta = max(
                    localDelta,
                    abs(candidate[row] - value[row])
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float fallbackMerit =
                qualityEvaluateNaturalMap(
                    delassus,
                    regularization,
                    linear,
                    lowerBound,
                    upperBound,
                    stepScale,
                    candidate,
                    candidateGradient,
                    candidateProjection,
                    candidateDerivative,
                    candidateResidual,
                    dispatch.rowCount,
                    lane
                );
            if (!isfinite(fallbackMerit) ||
                !(fallbackMerit < merit)) {
                break;
            }
            accepted = true;
            acceptedDelta = simd_max(localDelta);
        }
        for (uint row = lane;
             row < dispatch.rowCount;
             row += 32u) {
            value[row] = candidate[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lastMaximumDelta = acceptedDelta;
        completedIterations = iteration + 1u;
    }

    // Certify the physical unit-step natural residual, independent of S.
    static_cast<void>(qualityEvaluateNaturalMap(
        delassus,
        regularization,
        linear,
        lowerBound,
        upperBound,
        stepScale,
        value,
        gradient,
        projection,
        derivative,
        residual,
        dispatch.rowCount,
        lane
    ));
    float localNatural = 0.0f;
    for (uint row = lane;
         row < dispatch.rowCount;
         row += 32u) {
        const float projected = clamp(
            value[row] - gradient[row],
            lowerBound[row],
            upperBound[row]
        );
        localNatural = max(
            localNatural,
            abs(value[row] - projected)
        );
    }
    physicalNaturalResidual = simd_max(localNatural);
    converged = converged ||
        (
            isfinite(physicalNaturalResidual) &&
            physicalNaturalResidual <= tolerance
        );
    if (!converged) {
        if (lane == 0u) {
            MRGeneralizedConstraintStatusGPU status = {};
            status.code =
                MR_GENERALIZED_CONSTRAINT_DID_NOT_CONVERGE;
            status.environment = environment;
            status.iterations = completedIterations;
            status.failingRow = MR_INVALID_INDEX;
            status.failingInverseWork = MR_INVALID_INDEX;
            status.inverseMassCode = MR_INVERSE_MASS_SUCCESS;
            status.activeRows = dispatch.rowCount;
            status.diagnostics = float4(
                lastMaximumDelta,
                physicalNaturalResidual,
                scalarWork[0],
                scalarWork[1]
            );
            statuses[environment] = status;
        }
        return;
    }

    for (uint row = lane;
         row < dispatch.rowCount;
         row += 32u) {
        outputImpulses[
            environment * dispatch.rowCount + row
        ] = value[row];
    }
    const uint velocityBase = environment * dispatch.nv;
    bool nonfiniteVelocity = false;
    for (uint dof = lane;
         dof < dispatch.nv;
         dof += 32u) {
        float velocity = freeVelocity[velocityBase + dof];
        for (uint row = 0u;
             row < dispatch.rowCount;
             ++row) {
            velocity = fma(
                responseColumns[
                    (environment * dispatch.rowCount + row) *
                        dispatch.nv +
                    dof
                ],
                value[row],
                velocity
            );
        }
        outputVelocity[velocityBase + dof] = velocity;
        nonfiniteVelocity =
            nonfiniteVelocity || !isfinite(velocity);
    }
    if (simd_any(nonfiniteVelocity)) {
        if (lane == 0u) {
            publishFailure(
                statuses,
                environment,
                MR_GENERALIZED_CONSTRAINT_NONFINITE_RESULT,
                MR_INVALID_INDEX,
                MR_INVALID_INDEX,
                MR_INVERSE_MASS_SUCCESS
            );
        }
        return;
    }
    if (lane == 0u) {
        MRGeneralizedConstraintStatusGPU status = {};
        status.code = MR_GENERALIZED_CONSTRAINT_SUCCESS;
        status.environment = environment;
        status.iterations = completedIterations;
        status.failingRow = MR_INVALID_INDEX;
        status.failingInverseWork = MR_INVALID_INDEX;
        status.inverseMassCode = MR_INVERSE_MASS_SUCCESS;
        status.activeRows = dispatch.rowCount;
        status.diagnostics = float4(
            lastMaximumDelta,
            physicalNaturalResidual,
            scalarWork[0],
            scalarWork[1]
        );
        statuses[environment] = status;
    }
}
