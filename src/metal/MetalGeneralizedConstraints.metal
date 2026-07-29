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
            converged = true;
            break;
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
