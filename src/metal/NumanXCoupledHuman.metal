#include <metal_stdlib>

#include "metalrobo/numanx_coupled_human_gpu.h"

using namespace metal;

namespace {

constant uint kMaxDofs = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
constant float kPivotFloor = 1.0e-12f;

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline bool knownOperation(const uint operation) {
    return operation <= MR_NUMANX_COUPLED_HUMAN_STAGED_PUBLISH;
}

inline bool validDispatch(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch
) {
    const bool projected =
        (dispatch.flags &
         MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR) != 0u;
    return dispatch.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        knownOperation(dispatch.operation) &&
        dispatch.environmentCount != 0u &&
        dispatch.nv != 0u && dispatch.nv <= kMaxDofs &&
        dispatch.nq != 0u &&
        dispatch.nq <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        dispatch.nq == dispatch.nv + 1u &&
        dispatch.sourceQStride == dispatch.nq &&
        dispatch.sourceVStride == dispatch.nv &&
        dispatch.generalizedVectorStride >= dispatch.nv &&
        dispatch.generalizedVectorStride <= kMaxDofs &&
        dispatch.factorStride == dispatch.nv * dispatch.nv &&
        dispatch.reactionStride >= dispatch.nv &&
        dispatch.reactionStride <= kMaxDofs &&
        dispatch.reserved0 == 0u &&
        dispatch.programFingerprint != 0u &&
        dispatch.transactionFingerprint != 0u &&
        dispatch.linearizationEpoch != 0u &&
        dispatch.slotGeneration != 0u &&
        (dispatch.flags &
         ~MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR) == 0u &&
        (!projected ||
         dispatch.projectorStride == dispatch.nv * dispatch.nv);
}

inline bool knownStage(const uint stage) {
    return stage <= MR_NUMANX_COUPLED_HUMAN_STAGE_RESOLVED;
}

inline bool matchingMetadata(
    device const MRNumanXCoupledHumanSlotMetadataGPU& metadata,
    constant MRNumanXCoupledHumanDispatchGPU& dispatch,
    const uint expectedStage,
    const uint requiredWitnesses
) {
    return metadata.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        metadata.structSize == sizeof(MRNumanXCoupledHumanSlotMetadataGPU) &&
        metadata.stage == expectedStage &&
        metadata.transactionSlot == dispatch.transactionSlot &&
        metadata.stepIndex == dispatch.stepIndex &&
        metadata.environmentCount == dispatch.environmentCount &&
        metadata.qCoordinateCount == dispatch.nq &&
        metadata.dofCount == dispatch.nv &&
        metadata.sourceQStride == dispatch.sourceQStride &&
        metadata.sourceVStride == dispatch.sourceVStride &&
        metadata.factorStride == dispatch.factorStride &&
        metadata.reactionStride == dispatch.reactionStride &&
        metadata.reserved0 == 0u && metadata.reserved1 == 0u &&
        metadata.reserved2 == 0u &&
        metadata.programFingerprint == dispatch.programFingerprint &&
        metadata.transactionFingerprint == dispatch.transactionFingerprint &&
        metadata.linearizationEpoch == dispatch.linearizationEpoch &&
        metadata.slotGeneration == dispatch.slotGeneration &&
        (metadata.operationWitnesses & requiredWitnesses) ==
            requiredWitnesses;
}

inline bool matchingMetadata(
    device const MRNumanXCoupledHumanSlotMetadataGPU& metadata,
    constant MRNumanXCoupledHumanResolveDispatchGPU& dispatch,
    const uint expectedStage,
    const uint requiredWitnesses
) {
    return metadata.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        metadata.structSize == sizeof(MRNumanXCoupledHumanSlotMetadataGPU) &&
        metadata.stage == expectedStage &&
        metadata.transactionSlot == dispatch.transactionSlot &&
        metadata.stepIndex == dispatch.stepIndex &&
        metadata.environmentCount == dispatch.environmentCount &&
        metadata.qCoordinateCount == dispatch.dofCount + 1u &&
        metadata.dofCount == dispatch.dofCount &&
        metadata.sourceQStride == dispatch.dofCount + 1u &&
        metadata.sourceVStride == dispatch.dofCount &&
        metadata.factorStride == dispatch.dofCount * dispatch.dofCount &&
        metadata.reactionStride == dispatch.reactionStride &&
        metadata.reserved0 == 0u && metadata.reserved1 == 0u &&
        metadata.reserved2 == 0u &&
        metadata.programFingerprint == dispatch.programFingerprint &&
        metadata.transactionFingerprint == dispatch.transactionFingerprint &&
        metadata.linearizationEpoch == dispatch.linearizationEpoch &&
        metadata.slotGeneration == dispatch.slotGeneration &&
        (metadata.operationWitnesses & requiredWitnesses) ==
            requiredWitnesses;
}

inline void rejectHuman(
    device MRNumanXCoupledHumanStatusGPU& status,
    const uint code
) {
    if (status.decision == MR_NUMANX_COUPLED_HUMAN_PENDING) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN;
        status.humanCode = code;
    }
}

inline void rejectMatter(
    device MRNumanXCoupledHumanStatusGPU& status,
    const uint code,
    const uint completedMicrosteps
) {
    if (status.decision == MR_NUMANX_COUPLED_HUMAN_PENDING) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
        status.matterCode = code;
        status.matterCompletedMicrosteps = completedMicrosteps;
    }
}

inline bool pending(
    device const MRNumanXCoupledHumanStatusGPU& status,
    constant MRNumanXCoupledHumanDispatchGPU& dispatch,
    const uint environment
) {
    return status.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        status.decision == MR_NUMANX_COUPLED_HUMAN_PENDING &&
        status.environment == environment &&
        status.stepIndex == dispatch.stepIndex;
}

inline bool applyProjector(
    device const float* projector,
    const uint projectorBase,
    thread const float* input,
    thread float* output,
    const uint count,
    const bool transpose,
    const bool useProjector
) {
    for (uint row = 0u; row < count; ++row) {
        float value = 0.0f;
        if (useProjector) {
            for (uint column = 0u; column < count; ++column) {
                const uint index = transpose
                    ? projectorBase + column * count + row
                    : projectorBase + row * count + column;
                const float coefficient = projector[index];
                if (!isfinite(coefficient)) return false;
                value += coefficient * input[column];
            }
        } else {
            value = input[row];
        }
        if (!isfinite(value)) return false;
        output[row] = value;
    }
    return true;
}

inline bool applyEffectiveTangent(
    device const float* factor,
    const uint factorBase,
    thread const float* input,
    thread float* intermediate,
    thread float* output,
    const uint count
) {
    // A0 = L0 L0^T where A0 is the Stand source effective tangent
    // M + armature + h*D. First evaluate L0^T*x, then L0*(L0^T*x).
    for (uint column = 0u; column < count; ++column) {
        float value = 0.0f;
        for (uint row = column; row < count; ++row) {
            const float coefficient =
                factor[factorBase + row * count + column];
            if (!isfinite(coefficient)) return false;
            value += coefficient * input[row];
        }
        if (!isfinite(value)) return false;
        intermediate[column] = value;
    }
    for (uint row = 0u; row < count; ++row) {
        float value = 0.0f;
        for (uint column = 0u; column <= row; ++column) {
            value += factor[factorBase + row * count + column] *
                intermediate[column];
        }
        if (!isfinite(value)) return false;
        output[row] = value;
    }
    return true;
}

inline bool solveEffectiveTangent(
    device const float* factor,
    const uint factorBase,
    thread const float* rightHandSide,
    thread float* workspace,
    thread float* output,
    const uint count,
    thread float& minimumPivot,
    thread float& maximumPivot
) {
    minimumPivot = INFINITY;
    maximumPivot = 0.0f;
    for (uint row = 0u; row < count; ++row) {
        float value = rightHandSide[row];
        for (uint column = 0u; column < row; ++column) {
            value -= factor[factorBase + row * count + column] *
                workspace[column];
        }
        const float diagonal = factor[factorBase + row * count + row];
        if (!(diagonal > kPivotFloor) || !isfinite(diagonal)) return false;
        minimumPivot = min(minimumPivot, diagonal);
        maximumPivot = max(maximumPivot, diagonal);
        workspace[row] = value / diagonal;
        if (!isfinite(workspace[row])) return false;
    }
    for (uint reverse = 0u; reverse < count; ++reverse) {
        const uint row = count - 1u - reverse;
        float value = workspace[row];
        for (uint column = row + 1u; column < count; ++column) {
            value -= factor[factorBase + column * count + row] *
                output[column];
        }
        const float diagonal = factor[factorBase + row * count + row];
        output[row] = value / diagonal;
        if (!isfinite(output[row])) return false;
    }
    return true;
}

inline MRInverseMassStatusGPU inverseFailure(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch,
    const uint environment,
    const uint code,
    const uint failingIndex
) {
    MRInverseMassStatusGPU result{};
    result.code = code;
    result.environment = environment;
    result.articulationIndex = dispatch.articulationIndex;
    result.failingIndex = failingIndex;
    result.bodyCount = dispatch.bodyCount;
    result.nq = dispatch.nq;
    result.nv = dispatch.nv;
    result.rhsCount = 1u;
    result.diagnostics = float4(0.0f);
    return result;
}

} // namespace

// Opens one slot.  Reaction and joint status are Human-owned storage; this
// kernel never touches live q/v or the MyoSim generalized-force arena.
kernel void numanx_coupled_human_begin(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch [[buffer(0)]],
    device float* matterReaction [[buffer(1)]],
    device MRNumanXCoupledHumanStatusGPU* statuses [[buffer(2)]],
    device MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !validDispatch(dispatch)) return;
    if (environment == 0u) {
        MRNumanXCoupledHumanSlotMetadataGPU identity{};
        identity.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
        identity.structSize = sizeof(MRNumanXCoupledHumanSlotMetadataGPU);
        identity.stage = MR_NUMANX_COUPLED_HUMAN_STAGE_BEGUN;
        identity.transactionSlot = dispatch.transactionSlot;
        identity.stepIndex = dispatch.stepIndex;
        identity.environmentCount = dispatch.environmentCount;
        identity.operationWitnesses = 0u;
        identity.qCoordinateCount = dispatch.nq;
        identity.dofCount = dispatch.nv;
        identity.sourceQStride = dispatch.sourceQStride;
        identity.sourceVStride = dispatch.sourceVStride;
        identity.factorStride = dispatch.factorStride;
        identity.reactionStride = dispatch.reactionStride;
        identity.programFingerprint = dispatch.programFingerprint;
        identity.transactionFingerprint = dispatch.transactionFingerprint;
        identity.linearizationEpoch = dispatch.linearizationEpoch;
        identity.slotGeneration = dispatch.slotGeneration;
        metadata[0] = identity;
    }
    const uint reactionBase = environment * dispatch.reactionStride;
    for (uint dof = 0u; dof < dispatch.nv; ++dof)
        matterReaction[reactionBase + dof] = 0.0f;

    MRNumanXCoupledHumanStatusGPU status{};
    status.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    status.decision = MR_NUMANX_COUPLED_HUMAN_PENDING;
    status.environment = environment;
    status.stepIndex = dispatch.stepIndex;
    status.humanCode = MR_NUMI_HUMAN_STAND_SUCCESS;
    status.matterCode = 0u;
    status.humanCompletedSteps = dispatch.stepIndex;
    status.matterCompletedMicrosteps = 0u;
    statuses[environment] = status;
}

// Candidate geometry itself is materialized by the owning Human callback.
// This guard validates its exact q/body/J outputs before Matter consumes them;
// there is intentionally no first-order kinematics fallback here.
kernel void numanx_coupled_human_validate_candidate(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch [[buffer(0)]],
    device const float* candidateQ [[buffer(1)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(2)]],
    device const float* pointJacobians [[buffer(3)]],
    device MRNumanXCoupledHumanStatusGPU* statuses [[buffer(4)]],
    device const MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !validDispatch(dispatch) ||
        dispatch.operation !=
            MR_NUMANX_COUPLED_HUMAN_CANDIDATE_KINEMATICS) return;
    const uint metadataStage = metadata[0].stage;
    if ((metadataStage != MR_NUMANX_COUPLED_HUMAN_STAGE_BEGUN &&
         metadataStage !=
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED) ||
        !matchingMetadata(metadata[0], dispatch, metadataStage,
            metadataStage ==
                    MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED
                ? MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE
                : 0u)) return;
    device MRNumanXCoupledHumanStatusGPU& status = statuses[environment];
    if (!pending(status, dispatch, environment)) return;
    if (dispatch.candidateQStride < dispatch.nq ||
        dispatch.candidateBodyStride < dispatch.bodyCount ||
        dispatch.pointStride < dispatch.pointCount ||
        dispatch.pointJacobianStride <
            dispatch.pointCount * 3u * dispatch.generalizedVectorStride) {
        rejectHuman(
            status,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH
        );
        return;
    }

    const uint qBase = environment * dispatch.candidateQStride;
    for (uint coordinate = 0u; coordinate < dispatch.nq; ++coordinate) {
        if (!isfinite(candidateQ[qBase + coordinate])) {
            rejectHuman(
                status,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_KINEMATICS_FAILED
            );
            return;
        }
    }
    const uint bodyBase = environment * dispatch.candidateBodyStride;
    for (uint body = 0u; body < dispatch.bodyCount; ++body) {
        const MRBodyStateGPU state = candidateBodies[bodyBase + body];
        if (!finite4(state.position) || !finite4(state.orientation) ||
            !finite4(state.linearVelocityAndInverseMass) ||
            !finite4(state.angularVelocity) ||
            !finite4(state.inverseInertiaWorldRow0) ||
            !finite4(state.inverseInertiaWorldRow1) ||
            !finite4(state.inverseInertiaWorldRow2)) {
            rejectHuman(
                status,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_KINEMATICS_FAILED
            );
            return;
        }
    }
    const uint jacobianBase =
        environment * dispatch.pointJacobianStride;
    for (uint point = 0u; point < dispatch.pointCount; ++point) {
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint rowBase = jacobianBase +
                (point * 3u + axis) * dispatch.generalizedVectorStride;
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                if (!isfinite(pointJacobians[rowBase + dof])) {
                    rejectHuman(
                        status,
                        MR_NUMANX_COUPLED_HUMAN_SERVICE_KINEMATICS_FAILED
                    );
                    return;
                }
            }
        }
    }
}

kernel void numanx_coupled_human_effective_tangent_action(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch [[buffer(0)]],
    device const float* factor [[buffer(1)]],
    device const float* projector [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    device MRNumanXCoupledHumanStatusGPU* statuses [[buffer(5)]],
    device const MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !validDispatch(dispatch) ||
        dispatch.operation != MR_NUMANX_COUPLED_HUMAN_MASS_ACTION) return;
    if (!matchingMetadata(
            metadata[0],
            dispatch,
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED,
            MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE)) return;
    device MRNumanXCoupledHumanStatusGPU& status = statuses[environment];
    const uint count = dispatch.nv;
    const uint vectorBase = environment * dispatch.generalizedVectorStride;
    for (uint dof = 0u; dof < count; ++dof)
        output[vectorBase + dof] = 0.0f;
    if (!pending(status, dispatch, environment)) return;

    const uint matrixBase = environment * dispatch.factorStride;
    const uint projectorBase = environment * dispatch.projectorStride;
    const bool useProjector =
        (dispatch.flags &
         MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR) != 0u;
    float source[kMaxDofs];
    float projected[kMaxDofs];
    float intermediate[kMaxDofs];
    float mass[kMaxDofs];
    float result[kMaxDofs];
    for (uint dof = 0u; dof < count; ++dof) {
        source[dof] = input[vectorBase + dof];
        if (!isfinite(source[dof])) {
            rejectHuman(
                status,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_INPUT
            );
            return;
        }
    }
    if (!applyProjector(
            projector, projectorBase, source, projected, count,
            false, useProjector) ||
        !applyEffectiveTangent(
            factor, matrixBase, projected, intermediate, mass, count) ||
        !applyProjector(
            projector, projectorBase, mass, result, count,
            true, useProjector)) {
        rejectHuman(
            status,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_FACTOR
        );
        return;
    }
    for (uint dof = 0u; dof < count; ++dof)
        output[vectorBase + dof] = result[dof];
}

kernel void numanx_coupled_human_effective_tangent_preconditioner(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch [[buffer(0)]],
    device const float* factor [[buffer(1)]],
    device const float* projector [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    device MRInverseMassStatusGPU* inverseStatuses [[buffer(5)]],
    device MRNumanXCoupledHumanStatusGPU* jointStatuses [[buffer(6)]],
    device const MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !validDispatch(dispatch) ||
        dispatch.operation !=
            MR_NUMANX_COUPLED_HUMAN_INVERSE_MASS_PRECONDITIONER) return;
    if (!matchingMetadata(
            metadata[0],
            dispatch,
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED,
            MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE)) return;
    device MRNumanXCoupledHumanStatusGPU& joint = jointStatuses[environment];
    const uint statusIndex =
        dispatch.articulationIndex * dispatch.statusStride + environment;
    const uint vectorBase = environment * dispatch.generalizedVectorStride;
    for (uint dof = 0u; dof < dispatch.nv; ++dof)
        output[vectorBase + dof] = 0.0f;
    if (!pending(joint, dispatch, environment)) {
        inverseStatuses[statusIndex] = inverseFailure(
            dispatch, environment,
            MR_INVERSE_MASS_INVALID_DISPATCH, MR_INVALID_INDEX
        );
        return;
    }

    const uint count = dispatch.nv;
    const uint matrixBase = environment * dispatch.factorStride;
    const uint projectorBase = environment * dispatch.projectorStride;
    // statusStride is the capacity between articulation blocks, matching
    // CoupledCandidateQuery.  This service writes only its Human articulation
    // block and leaves every other owner's records untouched.
    const bool useProjector =
        (dispatch.flags &
         MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR) != 0u;
    float source[kMaxDofs];
    float projectedRightHandSide[kMaxDofs];
    float workspace[kMaxDofs];
    float inverse[kMaxDofs];
    float result[kMaxDofs];
    float maximumInput = 0.0f;
    for (uint dof = 0u; dof < count; ++dof) {
        source[dof] = input[vectorBase + dof];
        maximumInput = max(maximumInput, abs(source[dof]));
        if (!isfinite(source[dof])) {
            inverseStatuses[statusIndex] = inverseFailure(
                dispatch, environment,
                MR_INVERSE_MASS_NONFINITE_INPUT, dof
            );
            rejectHuman(
                joint,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_INPUT
            );
            return;
        }
    }
    if (!applyProjector(
            projector, projectorBase, source, projectedRightHandSide,
            count, true, useProjector)) {
        inverseStatuses[statusIndex] = inverseFailure(
            dispatch, environment,
            MR_INVERSE_MASS_NONFINITE_INPUT, MR_INVALID_INDEX
        );
        rejectHuman(
            joint,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_INPUT
        );
        return;
    }
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    if (!solveEffectiveTangent(
            factor, matrixBase, projectedRightHandSide, workspace, inverse,
            count, minimumPivot, maximumPivot) ||
        !applyProjector(
            projector, projectorBase, inverse, result,
            count, false, useProjector)) {
        inverseStatuses[statusIndex] = inverseFailure(
            dispatch, environment,
            MR_INVERSE_MASS_FACTORIZATION_FAILED, MR_INVALID_INDEX
        );
        rejectHuman(
            joint,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_FACTOR
        );
        return;
    }

    float maximumOutput = 0.0f;
    for (uint dof = 0u; dof < count; ++dof) {
        output[vectorBase + dof] = result[dof];
        maximumOutput = max(maximumOutput, abs(result[dof]));
    }
    MRInverseMassStatusGPU status = inverseFailure(
        dispatch, environment,
        MR_INVERSE_MASS_SUCCESS, MR_INVALID_INDEX
    );
    status.diagnostics = float4(
        minimumPivot, maximumPivot, maximumOutput, maximumInput
    );
    inverseStatuses[statusIndex] = status;
}

// Stages the equal-and-opposite continuum reaction separately from MyoSim.
// The numerator is P^T*A0*P*deltaV and reaction=numerator/dt, where A0 is the
// source effective tangent, not pure M. It overwrites rather than accumulates,
// and the transaction witness permits exactly one publication.
kernel void numanx_coupled_human_stage_generalized_reaction(
    constant MRNumanXCoupledHumanDispatchGPU& dispatch [[buffer(0)]],
    device const float* factor [[buffer(1)]],
    device const float* projector [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* generalizedImpulse [[buffer(4)]],
    device float* matterReaction [[buffer(5)]],
    device const MRNumanXCoupledMatterOutcomeGPU* matterOutcomes
        [[buffer(6)]],
    constant uint& matterOutcomeStride [[buffer(7)]],
    constant uint& matterSuccessCode [[buffer(8)]],
    constant uint& expectedMatterCompletedMicrosteps [[buffer(9)]],
    device MRNumanXCoupledHumanStatusGPU* statuses [[buffer(10)]],
    device const MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(11)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !validDispatch(dispatch) ||
        dispatch.operation != MR_NUMANX_COUPLED_HUMAN_STAGED_PUBLISH) return;
    if (!matchingMetadata(
            metadata[0],
            dispatch,
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED,
            MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE)) return;
    device MRNumanXCoupledHumanStatusGPU& status = statuses[environment];
    const uint vectorBase = environment * dispatch.generalizedVectorStride;
    const uint reactionBase = environment * dispatch.reactionStride;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        generalizedImpulse[vectorBase + dof] = 0.0f;
        matterReaction[reactionBase + dof] = 0.0f;
    }
    if (!pending(status, dispatch, environment)) return;

    const MRNumanXCoupledMatterOutcomeGPU outcome =
        matterOutcomes[environment * matterOutcomeStride];
    if (outcome.environment != environment || outcome.reserved0 != 0u ||
        outcome.code != matterSuccessCode ||
        outcome.completedMicrosteps !=
            expectedMatterCompletedMicrosteps) {
        rejectMatter(
            status,
            outcome.environment == environment && outcome.reserved0 == 0u
                ? outcome.code
                : MR_INVALID_INDEX,
            outcome.completedMicrosteps
        );
        return;
    }

    const uint count = dispatch.nv;
    const uint matrixBase = environment * dispatch.factorStride;
    const uint projectorBase = environment * dispatch.projectorStride;
    const bool useProjector =
        (dispatch.flags &
         MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR) != 0u;
    float source[kMaxDofs];
    float projected[kMaxDofs];
    float intermediate[kMaxDofs];
    float mass[kMaxDofs];
    float impulse[kMaxDofs];
    for (uint dof = 0u; dof < count; ++dof) {
        source[dof] = input[vectorBase + dof];
        if (!isfinite(source[dof])) {
            rejectHuman(
                status,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_INPUT
            );
            return;
        }
    }
    if (!applyProjector(
            projector, projectorBase, source, projected, count,
            false, useProjector) ||
        !applyEffectiveTangent(
            factor, matrixBase, projected, intermediate, mass, count) ||
        !applyProjector(
            projector, projectorBase, mass, impulse, count,
            true, useProjector)) {
        rejectHuman(
            status,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_FACTOR
        );
        return;
    }
    const float inverseTimestep = dispatch.timestepAndInverse.y;
    if (!(inverseTimestep > 0.0f) || !isfinite(inverseTimestep)) {
        rejectHuman(
            status,
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH
        );
        return;
    }
    for (uint dof = 0u; dof < count; ++dof) {
        const float force = impulse[dof] * inverseTimestep;
        if (!isfinite(force)) {
            rejectHuman(
                status,
                MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_RESULT
            );
            return;
        }
        generalizedImpulse[vectorBase + dof] = impulse[dof];
        matterReaction[reactionBase + dof] = force;
    }
    status.matterCode = outcome.code;
    status.matterCompletedMicrosteps = outcome.completedMicrosteps;
}

kernel void numanx_coupled_human_resolve(
    constant MRNumanXCoupledHumanResolveDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanStandStatusGPU* standStatuses [[buffer(1)]],
    device const MRNumanXCoupledMatterOutcomeGPU* matterOutcomes
        [[buffer(2)]],
    device float* matterReaction [[buffer(3)]],
    device MRNumanXCoupledHumanStatusGPU* statuses [[buffer(4)]],
    device const MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
        dispatch.environmentCount == 0u || dispatch.statusStride == 0u ||
        dispatch.standStatusStride == 0u ||
        dispatch.matterOutcomeStride == 0u ||
        dispatch.dofCount == 0u || dispatch.dofCount > kMaxDofs ||
        dispatch.reactionStride < dispatch.dofCount ||
        dispatch.reactionStride > kMaxDofs ||
        dispatch.programFingerprint == 0u ||
        dispatch.transactionFingerprint == 0u ||
        dispatch.linearizationEpoch == 0u ||
        dispatch.slotGeneration == 0u ||
        !matchingMetadata(
            metadata[0],
            dispatch,
            MR_NUMANX_COUPLED_HUMAN_STAGE_PUBLISHED,
            MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE |
                MR_NUMANX_COUPLED_HUMAN_WITNESS_PUBLISH)) return;

    device MRNumanXCoupledHumanStatusGPU& status =
        statuses[environment * dispatch.statusStride];
    const MRNumiHumanStandStatusGPU human =
        standStatuses[environment * dispatch.standStatusStride];
    const MRNumanXCoupledMatterOutcomeGPU matter =
        matterOutcomes[environment * dispatch.matterOutcomeStride];

    const bool statusValid =
        status.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        status.environment == environment &&
        status.stepIndex == dispatch.stepIndex;
    const bool humanValid =
        human.environment == environment &&
        human.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        human.completedSteps == dispatch.expectedHumanCompletedSteps;
    const bool matterValid =
        matter.environment == environment && matter.reserved0 == 0u &&
        matter.code == dispatch.matterSuccessCode &&
        matter.completedMicrosteps ==
            dispatch.expectedMatterCompletedMicrosteps;
    const bool stagedPublishValid =
        status.matterCode == dispatch.matterSuccessCode &&
        status.matterCompletedMicrosteps == 0u;

    if (!statusValid) {
        status.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
        status.environment = environment;
        status.stepIndex = dispatch.stepIndex;
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN;
        status.humanCode =
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH;
    } else if (!humanValid) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN;
        status.humanCode = human.environment == environment
            ? (human.code == MR_NUMI_HUMAN_STAND_SUCCESS
                ? MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH
                : human.code)
            : MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH;
    } else if (status.decision ==
                   MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN) {
        // Preserve a deterministic service-side failure raised earlier.
    } else if (!matterValid || !stagedPublishValid || status.decision ==
                   MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
        status.matterCode = matter.environment == environment &&
                matter.reserved0 == 0u
            ? matter.code
            : MR_INVALID_INDEX;
    } else if (status.decision == MR_NUMANX_COUPLED_HUMAN_PENDING) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_ACCEPT;
        status.humanCode = human.code;
        status.matterCode = matter.code;
    } else if (status.decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT) {
        status.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN;
        status.humanCode =
            MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH;
    }
    status.humanCompletedSteps = human.completedSteps;
    status.matterCompletedMicrosteps = matter.completedMicrosteps;

    if (status.decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT) {
        const uint reactionBase = environment * dispatch.reactionStride;
        for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
            matterReaction[reactionBase + dof] = 0.0f;
        }
    }
}

// Single-thread private witness transition.  This kernel never mutates public
// reaction or status storage; a mismatched identity or illegal order leaves
// the previous transaction byte-for-byte intact.
kernel void numanx_coupled_human_advance(
    device MRNumanXCoupledHumanSlotMetadataGPU* metadata [[buffer(0)]],
    device const MRNumanXCoupledHumanStatusGPU* statuses [[buffer(1)]],
    constant MRNumanXCoupledHumanAdvanceDispatchGPU& dispatch [[buffer(2)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (threadIndex != 0u ||
        dispatch.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
        !knownStage(dispatch.expectedStage) ||
        !knownStage(dispatch.nextStage) ||
        dispatch.expectedStage ==
            MR_NUMANX_COUPLED_HUMAN_STAGE_UNINITIALIZED ||
        dispatch.nextStage == MR_NUMANX_COUPLED_HUMAN_STAGE_UNINITIALIZED ||
        dispatch.environmentCount == 0u ||
        dispatch.programFingerprint == 0u ||
        dispatch.transactionFingerprint == 0u ||
        dispatch.linearizationEpoch == 0u ||
        dispatch.slotGeneration == 0u ||
        dispatch.qCoordinateCount != dispatch.dofCount + 1u ||
        dispatch.dofCount == 0u || dispatch.dofCount > kMaxDofs ||
        dispatch.reactionStride < dispatch.dofCount ||
        dispatch.reactionStride > kMaxDofs || dispatch.reserved0 != 0u ||
        (dispatch.flags &
         ~MR_NUMANX_COUPLED_HUMAN_ADVANCE_REQUIRE_PENDING) != 0u) return;

    const bool candidateTransition =
        dispatch.witness == MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE &&
        dispatch.nextStage ==
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED &&
        (dispatch.expectedStage == MR_NUMANX_COUPLED_HUMAN_STAGE_BEGUN ||
         dispatch.expectedStage ==
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED) &&
        dispatch.flags ==
            MR_NUMANX_COUPLED_HUMAN_ADVANCE_REQUIRE_PENDING;
    const bool publishTransition =
        dispatch.witness == MR_NUMANX_COUPLED_HUMAN_WITNESS_PUBLISH &&
        dispatch.expectedStage ==
            MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED &&
        dispatch.nextStage == MR_NUMANX_COUPLED_HUMAN_STAGE_PUBLISHED &&
        dispatch.flags == 0u;
    const bool resolveTransition =
        dispatch.witness == MR_NUMANX_COUPLED_HUMAN_WITNESS_RESOLVE &&
        dispatch.expectedStage == MR_NUMANX_COUPLED_HUMAN_STAGE_PUBLISHED &&
        dispatch.nextStage == MR_NUMANX_COUPLED_HUMAN_STAGE_RESOLVED &&
        dispatch.flags == 0u;
    if (!candidateTransition && !publishTransition && !resolveTransition)
        return;

    device MRNumanXCoupledHumanSlotMetadataGPU& current = metadata[0];
    const uint requiredWitnesses = publishTransition
        ? MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE
        : resolveTransition
            ? MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE |
                MR_NUMANX_COUPLED_HUMAN_WITNESS_PUBLISH
            : dispatch.expectedStage ==
                    MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED
                ? MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE
                : 0u;
    if (current.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
        current.structSize != sizeof(MRNumanXCoupledHumanSlotMetadataGPU) ||
        current.stage != dispatch.expectedStage ||
        current.transactionSlot != dispatch.transactionSlot ||
        current.stepIndex != dispatch.stepIndex ||
        current.environmentCount != dispatch.environmentCount ||
        current.qCoordinateCount != dispatch.qCoordinateCount ||
        current.dofCount != dispatch.dofCount ||
        current.sourceQStride != dispatch.qCoordinateCount ||
        current.sourceVStride != dispatch.dofCount ||
        current.factorStride != dispatch.dofCount * dispatch.dofCount ||
        current.reactionStride != dispatch.reactionStride ||
        current.reserved0 != 0u || current.reserved1 != 0u ||
        current.reserved2 != 0u ||
        current.programFingerprint != dispatch.programFingerprint ||
        current.transactionFingerprint != dispatch.transactionFingerprint ||
        current.linearizationEpoch != dispatch.linearizationEpoch ||
        current.slotGeneration != dispatch.slotGeneration ||
        (current.operationWitnesses & requiredWitnesses) !=
            requiredWitnesses ||
        (!candidateTransition &&
         (current.operationWitnesses & dispatch.witness) != 0u)) return;

    if (candidateTransition) {
        for (uint environment = 0u;
             environment < dispatch.environmentCount;
             ++environment) {
            const MRNumanXCoupledHumanStatusGPU status =
                statuses[environment];
            if (status.abiVersion !=
                    MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
                status.decision != MR_NUMANX_COUPLED_HUMAN_PENDING ||
                status.environment != environment ||
                status.stepIndex != dispatch.stepIndex) return;
        }
    }

    current.stage = dispatch.nextStage;
    current.operationWitnesses |= dispatch.witness;
}
