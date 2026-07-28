#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

uint mapABAStatus(const uint code) {
    switch (code) {
    case MR_ABA_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ABA_INVALID_DISPATCH:
    case MR_ABA_INVALID_MODEL:
    case MR_ABA_INVALID_QUATERNION:
        return MR_STEP_UNSUPPORTED;
    case MR_ABA_NONFINITE_INPUT:
        return MR_STEP_NONFINITE_INPUT;
    case MR_ABA_FACTORIZATION_FAILED:
        return MR_STEP_FACTORIZATION_FAILED;
    case MR_ABA_NONFINITE_RESULT:
        return MR_STEP_NONFINITE_RESULT;
    case MR_ABA_UNSUPPORTED_TOPOLOGY:
        return MR_STEP_UNSUPPORTED;
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

} // namespace

// Zero-copy semantic adapter from the private world-family reset arena to
// MLX's explicit WorldState arrays. Sampling occurs before this dispatch;
// only the state layout changes here.
kernel void mr_mlx_import_world_family_state(
    constant MRMLXContactAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const float* resetQ [[buffer(1)]],
    device const float* resetV [[buffer(2)]],
    device const MRBodyStateGPU* resetSceneBodies [[buffer(3)]],
    device float* q [[buffer(4)]],
    device float* v [[buffer(5)]],
    device float4* positions [[buffer(6)]],
    device float4* orientations [[buffer(7)]],
    device float4* linearVelocities [[buffer(8)]],
    device float4* angularVelocities [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        q[environment * dispatch.nq + coordinate] =
            resetQ[environment * dispatch.nq + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        v[environment * dispatch.nv + coordinate] =
            resetV[environment * dispatch.nv + coordinate];
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyCount;
    for (uint body = 0u;
         body < dispatch.sceneBodyCount;
         ++body) {
        const MRBodyStateGPU state =
            resetSceneBodies[sceneBase + body];
        positions[sceneBase + body] = state.position;
        orientations[sceneBase + body] = state.orientation;
        linearVelocities[sceneBase + body] =
            state.linearVelocityAndInverseMass;
        angularVelocities[sceneBase + body] =
            state.angularVelocity;
    }
}

// Transactional publication adapter for the MLX custom primitive. q/v inputs
// are the immutable control-step checkpoint. A failed microstep restores that
// checkpoint, zeros acceleration, and leaves the failure latched for every
// remaining encoded microstep.
kernel void mr_mlx_world_commit_aba(
    constant MRMLXWorldStepDispatchGPU& dispatch [[buffer(0)]],
    device const float* checkpointQ [[buffer(1)]],
    device const float* checkpointV [[buffer(2)]],
    device const float* candidateQ [[buffer(3)]],
    device const float* candidateV [[buffer(4)]],
    device const float* candidateAcceleration [[buffer(5)]],
    device const MRABAStatusGPU* abaStatuses [[buffer(6)]],
    device float* destinationQ [[buffer(7)]],
    device float* destinationV [[buffer(8)]],
    device float* publishedAcceleration [[buffer(9)]],
    device MRMLXWorldStepStatusGPU* statuses [[buffer(10)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMLXWorldStepStatusGPU status{};
    if (dispatch.physicsSubstep != 0u) {
        status = statuses[environment];
    } else {
        status.code = MR_STEP_SUCCESS;
        status.abaCode = MR_ABA_SUCCESS;
        status.failingSubstep = MR_INVALID_INDEX;
        status.failingIndex = MR_INVALID_INDEX;
    }

    const bool validDispatch =
        dispatch.environmentCount > 0u &&
        dispatch.nq > 0u &&
        dispatch.nv > 0u &&
        dispatch.physicsSubsteps > 0u &&
        dispatch.physicsSubstep < dispatch.physicsSubsteps &&
        dispatch.reserved0 == 0u &&
        dispatch.reserved1 == 0u;
    const MRABAStatusGPU aba = abaStatuses[environment];
    const bool validABA =
        aba.environment == environment &&
        aba.articulationIndex == dispatch.articulationIndex &&
        aba.code <= MR_ABA_UNSUPPORTED_TOPOLOGY;
    if (status.code == MR_STEP_SUCCESS &&
        (!validDispatch || !validABA ||
         aba.code != MR_ABA_SUCCESS)) {
        status.code =
            validDispatch && validABA
            ? mapABAStatus(aba.code)
            : MR_STEP_UNSUPPORTED;
        status.abaCode =
            validABA ? aba.code : MR_ABA_INVALID_DISPATCH;
        status.failingSubstep = dispatch.physicsSubstep;
        status.failingIndex =
            validABA ? aba.failingIndex : MR_INVALID_INDEX;
    }

    const bool publishCandidate =
        status.code == MR_STEP_SUCCESS;
    const uint qBase = environment * dispatch.nq;
    const uint vBase = environment * dispatch.nv;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        destinationQ[qBase + coordinate] =
            publishCandidate
            ? candidateQ[qBase + coordinate]
            : checkpointQ[qBase + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        destinationV[vBase + coordinate] =
            publishCandidate
            ? candidateV[vBase + coordinate]
            : checkpointV[vBase + coordinate];
        publishedAcceleration[vBase + coordinate] =
            publishCandidate
            ? candidateAcceleration[vBase + coordinate]
            : 0.0f;
    }
    if (publishCandidate) {
        ++status.successfulSubsteps;
    }
    statuses[environment] = status;
}

// Initializes one pure MLX control-step transaction without mutating any
// input array. Reset selection has already been expressed with mx.where.
kernel void mr_mlx_prepare_contact_world(
    device const MRMetalWorldDispatchGPU& dispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* actions [[buffer(4)]],
    device const MRConvexQueryCacheGPU* pairCache [[buffer(5)]],
    device float* checkpointQ [[buffer(6)]],
    device float* checkpointV [[buffer(7)]],
    device float* workingEffort [[buffer(8)]],
    device MRMetalWorldStatusGPU* statuses [[buffer(9)]],
    device MRConvexQueryCacheGPU* nextPairCache [[buffer(10)]],
    device const MRWorldGPU& world [[buffer(11)]],
    device const MRArticulationGPU* articulations [[buffer(12)]],
    device const MRDofPropertiesGPU* dofs [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldStatusGPU status = {};
    status.code = MR_STEP_SUCCESS;
    status.environment = environment;
    status.controlStep = 0u;
    status.successfulSubsteps = 0u;
    status.abaCode = MR_ABA_SUCCESS;
    status.failingSubstep = MR_INVALID_INDEX;
    status.failingIndex = MR_INVALID_INDEX;
    status.flags = dispatch.flags;
    status.diagnostics =
        float4(3.402823466e+38f, 0.0f, 0.0f, 0.0f);
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
    for (uint coordinate = 0u; coordinate < dispatch.nq;
         ++coordinate) {
        checkpointQ[qBase + coordinate] = q[qBase + coordinate];
    }
    for (uint coordinate = 0u; coordinate < dispatch.nv;
         ++coordinate) {
        const float velocity = v[vBase + coordinate];
        checkpointV[vBase + coordinate] = velocity;
        float command = actions[vBase + coordinate];
        if ((dispatch.flags &
             MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES) != 0u) {
            device const MRDofPropertiesGPU& dof =
                dofs[articulation.vOffset + coordinate];
            command = 0.0f;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.qIndex >= articulation.qOffset &&
                dof.qIndex <
                    articulation.qOffset + articulation.nq) {
                const uint localQ =
                    dof.qIndex - articulation.qOffset;
                float target = actions[vBase + coordinate];
                if ((dof.flags &
                     MR_DOF_FLAG_POSITION_LIMIT) != 0u) {
                    target = clamp(
                        target,
                        dof.limits.x,
                        dof.limits.y
                    );
                }
                const float timestep =
                    world.gravityAndTimestep.w;
                command =
                    dof.drive.x *
                        (
                            target -
                            q[qBase + localQ] -
                            timestep * velocity
                        ) -
                    dof.drive.y * velocity;
                const float dryFriction = dof.drive.w;
                if (dryFriction > 0.0f) {
                    if (abs(velocity) > 1.0e-4f) {
                        command -=
                            copysign(dryFriction, velocity);
                    } else {
                        command -= clamp(
                            command,
                            -dryFriction,
                            dryFriction
                        );
                    }
                }
                if ((dof.flags &
                     MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                    dof.limits.w > 0.0f) {
                    command = clamp(
                        command,
                        -dof.limits.w,
                        dof.limits.w
                    );
                }
            }
        }
        workingEffort[vBase + coordinate] = command;
    }
    const uint cacheBase =
        environment * contactDispatch.convexCacheStride;
    for (uint pair = 0u;
         pair < contactDispatch.eligiblePairCount;
         ++pair) {
        nextPairCache[cacheBase + pair] =
            pairCache[cacheBase + pair];
    }
    statuses[environment] = status;
}

// The physics kernels update a candidate cache in-place. Restore the explicit
// input cache for failed environments so lazy MLX execution has the same
// all-or-nothing state transition as the standalone graph.
kernel void mr_mlx_commit_pair_cache(
    constant MRMLXContactAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const MRConvexQueryCacheGPU* checkpoint [[buffer(1)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    device MRConvexQueryCacheGPU* candidate [[buffer(3)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (globalIndex >= total) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.eligiblePairCount;
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        candidate[globalIndex] = checkpoint[globalIndex];
    }
}

kernel void mr_mlx_initialize_operator_dispatch(
    constant MRArticulatedOperatorDispatchGPU& source [[buffer(0)]],
    device MRArticulatedOperatorDispatchGPU* destination [[buffer(1)]],
    const uint index [[thread_position_in_grid]]
) {
    if (index == 0u) {
        destination[0] = source;
    }
}

kernel void mr_mlx_pack_scene_state(
    constant MRMLXContactAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(1)]],
    device const uint* sceneBodyIndices [[buffer(2)]],
    device const float4* positions [[buffer(3)]],
    device const float4* orientations [[buffer(4)]],
    device const float4* linearVelocities [[buffer(5)]],
    device const float4* angularVelocities [[buffer(6)]],
    device MRBodyStateGPU* states [[buffer(7)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.sceneBodyCount;
    if (index >= total || dispatch.sceneBodyCount == 0u) {
        return;
    }
    const uint localBody = index % dispatch.sceneBodyCount;
    const uint globalBody = sceneBodyIndices[localBody];
    MRBodyStateGPU state = {};
    state.position = positions[index];
    state.position.w = 1.0f;
    state.orientation = orientations[index];
    state.linearVelocityAndInverseMass =
        float4(linearVelocities[index].xyz, 0.0f);
    state.angularVelocity =
        float4(angularVelocities[index].xyz, 0.0f);
    state.flagsAndIndices[0] =
        bodyProperties[globalBody].motionType;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = globalBody;
    state.flagsAndIndices[3] = 0u;
    states[index] = state;
}

// Converts the canonical packed world state into fixed-shape semantic arrays
// and contact evidence. One environment owns its stable capacity segment, so
// no output append or scheduling atomic can affect ordering.
kernel void mr_mlx_unpack_scene_and_evidence(
    constant MRMLXContactAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyStateGPU* sceneStates [[buffer(1)]],
    device const MRContactConstraintGPU* contacts [[buffer(2)]],
    device const MRContactPointMetaGPU* metadata [[buffer(3)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses [[buffer(4)]],
    device const MRMetalWorldStatusGPU* worldStatuses [[buffer(5)]],
    device const float* candidateAcceleration [[buffer(6)]],
    device float4* positions [[buffer(7)]],
    device float4* orientations [[buffer(8)]],
    device float4* linearVelocities [[buffer(9)]],
    device float4* angularVelocities [[buffer(10)]],
    device float* contactEvidence [[buffer(11)]],
    device uint* contactIDs [[buffer(12)]],
    device uint* contactCounts [[buffer(13)]],
    device uint* contactMasks [[buffer(14)]],
    device float* acceleration [[buffer(15)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyCount;
    for (uint body = 0u; body < dispatch.sceneBodyCount; ++body) {
        const MRBodyStateGPU state = sceneStates[sceneBase + body];
        positions[sceneBase + body] = state.position;
        orientations[sceneBase + body] = state.orientation;
        linearVelocities[sceneBase + body] =
            float4(state.linearVelocityAndInverseMass.xyz, 0.0f);
        angularVelocities[sceneBase + body] =
            float4(state.angularVelocity.xyz, 0.0f);
    }

    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    const uint active = min(
        contactStatus.activeContacts,
        dispatch.contactCapacity
    );
    contactCounts[environment] = active;
    const uint contactBase =
        environment * dispatch.contactCapacity;
    for (uint localContact = 0u;
         localContact < dispatch.contactCapacity;
         ++localContact) {
        const uint flat = contactBase + localContact;
        const uint evidenceBase = flat * 16u;
        const uint idBase = flat * 4u;
        const bool valid =
            localContact < active &&
            contactStatus.code == MR_STEP_SUCCESS;
        contactMasks[flat] = valid ? 1u : 0u;
        for (uint component = 0u; component < 16u; ++component) {
            contactEvidence[evidenceBase + component] = 0.0f;
        }
        for (uint component = 0u; component < 4u; ++component) {
            contactIDs[idBase + component] = 0u;
        }
        if (!valid) {
            continue;
        }
        const MRContactConstraintGPU contact = contacts[flat];
        const MRContactPointMetaGPU meta = metadata[flat];
        contactEvidence[evidenceBase + 0u] =
            contact.pointAndSeparation.x;
        contactEvidence[evidenceBase + 1u] =
            contact.pointAndSeparation.y;
        contactEvidence[evidenceBase + 2u] =
            contact.pointAndSeparation.z;
        contactEvidence[evidenceBase + 3u] =
            contact.pointAndSeparation.w;
        contactEvidence[evidenceBase + 4u] = contact.normal.x;
        contactEvidence[evidenceBase + 5u] = contact.normal.y;
        contactEvidence[evidenceBase + 6u] = contact.normal.z;
        contactEvidence[evidenceBase + 7u] = contact.impulses.x;
        contactEvidence[evidenceBase + 8u] = contact.impulses.y;
        contactEvidence[evidenceBase + 9u] = contact.impulses.z;
        contactEvidence[evidenceBase + 10u] = contact.friction.x;
        contactEvidence[evidenceBase + 11u] = contact.friction.y;
        contactEvidence[evidenceBase + 12u] = contact.response.x;
        contactEvidence[evidenceBase + 13u] = contact.response.y;
        contactEvidence[evidenceBase + 14u] =
            contact.targetVelocityAndPreSolveNormal.w;
        contactEvidence[evidenceBase + 15u] =
            float(contact.flags);
        contactIDs[idBase + 0u] = meta.colliderA;
        contactIDs[idBase + 1u] = meta.colliderB;
        contactIDs[idBase + 2u] =
            uint(contact.featureKey & 0xfffffffful);
        contactIDs[idBase + 3u] =
            uint(contact.featureKey >> 32u);
    }
    const bool succeeded =
        worldStatuses[environment].code == MR_STEP_SUCCESS &&
        contactStatus.code == MR_STEP_SUCCESS;
    const uint velocityBase = environment * dispatch.nv;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        acceleration[velocityBase + dof] =
            succeeded
            ? candidateAcceleration[velocityBase + dof]
            : 0.0f;
    }
}
