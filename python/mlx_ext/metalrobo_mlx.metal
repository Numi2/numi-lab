#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/generalized_constraint_shared.h"
#include "metalrobo/parallel_aba_shared.h"
#include "metalrobo/r2s2r_types.h"
#include "metalrobo/tactile_types.h"
#include "metalrobo/world_compiler_types.h"

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

float3 tactileQuaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 doubledCross =
        2.0f * cross(quaternion.xyz, value);
    return value +
        quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

float3 tactileStableTangent(const float3 normal) {
    const float3 absolute = abs(normal);
    const float3 reference =
        absolute.x <= absolute.y && absolute.x <= absolute.z
        ? float3(1.0f, 0.0f, 0.0f)
        : absolute.y <= absolute.z
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(0.0f, 0.0f, 1.0f);
    return normalize(cross(reference, normal));
}

} // namespace

// Zero-copy semantic adapter from the private world-family reset arena to
// MLX's explicit WorldState arrays. Sampling occurs before this dispatch;
// only the state layout changes here.
kernel void mr_mlx_import_world_family_state(
    constant MRMLXWorldFamilyImportDispatchGPU&
        dispatch [[buffer(0)]],
    device const float* resetQ [[buffer(1)]],
    device const float* resetV [[buffer(2)]],
    device const MRBodyStateGPU* resetSceneBodies [[buffer(3)]],
    device const MRWorldScenarioHeaderGPU*
        scenarioHeaders [[buffer(4)]],
    device const MRWorldScenarioValueGPU*
        scenarioValues [[buffer(5)]],
    device const MRWorldBodyParametersGPU*
        bodyParameters [[buffer(6)]],
    device const MRWorldControllerParametersGPU*
        controllerParameters [[buffer(7)]],
    device float* q [[buffer(8)]],
    device float* v [[buffer(9)]],
    device float4* positions [[buffer(10)]],
    device float4* orientations [[buffer(11)]],
    device float4* linearVelocities [[buffer(12)]],
    device float4* angularVelocities [[buffer(13)]],
    device uint4* outputScenarioHeaders [[buffer(14)]],
    device float4* outputScenarioValues [[buffer(15)]],
    device uint4* outputScenarioIdentities [[buffer(16)]],
    device float4* outputBodyParameters [[buffer(17)]],
    device uint4* outputBodyIdentities [[buffer(18)]],
    device float4* outputControllerParameters [[buffer(19)]],
    device uint4* outputControllerIdentities [[buffer(20)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.state.x ||
        dispatch.topology.w != MR_R2S2R_ABI_VERSION) {
        return;
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.state.y;
         ++coordinate) {
        q[environment * dispatch.state.y + coordinate] =
            resetQ[environment * dispatch.state.y + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.state.z;
         ++coordinate) {
        v[environment * dispatch.state.z + coordinate] =
            resetV[environment * dispatch.state.z + coordinate];
    }
    const uint sceneBase =
        environment * dispatch.state.w;
    for (uint body = 0u;
         body < dispatch.state.w;
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
    const MRWorldScenarioHeaderGPU scenario =
        scenarioHeaders[environment];
    const uint headerBase = environment * 3u;
    outputScenarioHeaders[headerBase + 0u] = scenario.identity;
    outputScenarioHeaders[headerBase + 1u] = scenario.provenance;
    outputScenarioHeaders[headerBase + 2u] = scenario.sampling;
    const uint scenarioBase =
        environment * dispatch.topology.z;
    for (uint feature = 0u;
         feature < dispatch.topology.z;
         ++feature) {
        const MRWorldScenarioValueGPU value =
            scenarioValues[scenarioBase + feature];
        outputScenarioValues[scenarioBase + feature] = value.value;
        outputScenarioIdentities[scenarioBase + feature] =
            value.identity;
    }
    const uint bodyBase =
        environment * dispatch.topology.x;
    for (uint body = 0u; body < dispatch.topology.x; ++body) {
        const MRWorldBodyParametersGPU value =
            bodyParameters[bodyBase + body];
        outputBodyParameters[bodyBase + body] = value.physical;
        outputBodyIdentities[bodyBase + body] = value.identity;
    }
    const uint controllerBase =
        environment * dispatch.topology.y;
    for (uint articulation = 0u;
         articulation < dispatch.topology.y;
         ++articulation) {
        const MRWorldControllerParametersGPU value =
            controllerParameters[controllerBase + articulation];
        outputControllerParameters[
            controllerBase + articulation
        ] = value.controller;
        outputControllerIdentities[
            controllerBase + articulation
        ] = value.identity;
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

// Transactional publication for the generalized multi-articulation graph.
// A failed environment retains its explicit free velocity and publishes zero
// impulses, while the typed failure record remains visible to MLX.
kernel void mr_mlx_commit_generalized_constraints(
    constant MRGeneralizedConstraintDispatchGPU&
        dispatch [[buffer(0)]],
    device const float* freeVelocity [[buffer(1)]],
    device const float* candidateVelocity [[buffer(2)]],
    device const float* candidateImpulses [[buffer(3)]],
    device const MRGeneralizedConstraintStatusGPU*
        candidateStatuses [[buffer(4)]],
    device float* nextVelocity [[buffer(5)]],
    device float* impulses [[buffer(6)]],
    device MRGeneralizedConstraintStatusGPU*
        statuses [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRGeneralizedConstraintStatusGPU status =
        candidateStatuses[environment];
    const bool valid =
        status.environment == environment &&
        status.code <=
            MR_GENERALIZED_CONSTRAINT_NONFINITE_RESULT;
    if (!valid) {
        status = {};
        status.code =
            MR_GENERALIZED_CONSTRAINT_INVALID_DISPATCH;
        status.environment = environment;
        status.failingRow = MR_INVALID_INDEX;
        status.failingInverseWork = MR_INVALID_INDEX;
        status.inverseMassCode = MR_INVERSE_MASS_SUCCESS;
    }
    const bool publish =
        valid &&
        status.code == MR_GENERALIZED_CONSTRAINT_SUCCESS;
    const uint velocityBase = environment * dispatch.nv;
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        nextVelocity[velocityBase + coordinate] =
            publish
            ? candidateVelocity[velocityBase + coordinate]
            : freeVelocity[velocityBase + coordinate];
    }
    const uint impulseBase =
        environment * dispatch.rowCount;
    for (uint row = 0u;
         row < dispatch.rowCount;
         ++row) {
        impulses[impulseBase + row] =
            publish ? candidateImpulses[impulseBase + row] : 0.0f;
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
    device const float4* controllerParameters [[buffer(14)]],
    device uint* resetMasks [[buffer(15)]],
    device const float* actuatorProfileValues [[buffer(16)]],
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
    // Reset selection is already represented by mx.where before this
    // primitive. The native reset mask must nevertheless be initialized:
    // leaving invocation-local MLX storage undefined would make the
    // checkpoint kernels nondeterministic.
    resetMasks[environment] = 0u;
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
                dofs[coordinate];
            command = 0.0f;
            if (dof.articulationIndex <
                    world.articulationCount &&
                (dof.flags & MR_DOF_FLAG_DRIVE) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.qIndex < dispatch.nq) {
                const float4 controller =
                    controllerParameters[
                        environment *
                            world.articulationCount +
                        dof.articulationIndex
                    ];
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
                        dof.drive.x * controller.x *
                        (
                            target -
                            q[qBase + dof.qIndex] -
                            timestep * velocity
                        ) -
                    dof.drive.y * controller.y * velocity;
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
                // WorldProgram payload compensation is an estimate used by
                // the controller, not a mutation of the immutable robot
                // inertias. Apply it to the drive effort before the authored
                // effort limit; controller.w == 1 preserves the base model.
                command *= max(controller.w, 0.0f);
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
        const uint actuatorBase =
            (environment * dispatch.nv + coordinate) * 7u;
        const float noLoadSpeed =
            actuatorProfileValues[actuatorBase + 2u];
        const float efficiency =
            actuatorProfileValues[actuatorBase + 3u];
        const float stallTorque =
            actuatorProfileValues[actuatorBase + 6u];
        const float speedFraction = clamp(
            abs(velocity) /
                max(
                    noLoadSpeed,
                    1.175494351e-38f
                ),
            0.0f,
            1.0f
        );
        const float envelope = min(
            dofs[coordinate].limits.w,
            stallTorque *
                efficiency *
                (1.0f - speedFraction)
        );
        command = clamp(command, -envelope, envelope);
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

kernel void mr_mlx_apply_family_body_damping(
    constant MRMLXContactAdapterDispatchGPU& dispatch [[buffer(0)]],
    constant MRMetalWorldContactDispatchGPU&
        contactDispatch [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const float4* bodyParameters [[buffer(4)]],
    device const MRBodyStateGPU* currentBodies [[buffer(5)]],
    device MRBodyStateGPU* candidateBodies [[buffer(6)]],
    device const MRMetalWorldContactStatusGPU*
        statuses [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint bodyBase = environment * dispatch.bodyStateStride;
    const float timestep = contactDispatch.timestepAndBias.x;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint body = sceneBodyIndices[localScene];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[body];
        if (properties.motionType != MR_MOTION_DYNAMIC) {
            continue;
        }
        const float scale = max(
            bodyParameters[bodyBase + body].w,
            0.0f
        );
        const float baseLinear = exp(
            -timestep * properties.dampingAndSpeedLimits.x
        );
        const float variedLinear = exp(
            -timestep * properties.dampingAndSpeedLimits.x * scale
        );
        const float baseAngular = exp(
            -timestep * properties.dampingAndSpeedLimits.y
        );
        const float variedAngular = exp(
            -timestep * properties.dampingAndSpeedLimits.y * scale
        );
        device MRBodyStateGPU& candidate =
            candidateBodies[bodyBase + body];
        device const MRBodyStateGPU& current =
            currentBodies[bodyBase + body];
        candidate.linearVelocityAndInverseMass.xyz +=
            (variedLinear - baseLinear) *
            current.linearVelocityAndInverseMass.xyz;
        candidate.angularVelocity.xyz +=
            (variedAngular - baseAngular) *
            current.angularVelocity.xyz;
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

// Builds invocation-sized articulation packets on MLX's active encoder.
// The compiled topology supplies offsets while the lazy batch supplies the
// environment count, so no host upload or hidden synchronization is needed
// when the same immutable world is called with a smaller batch.
kernel void mr_mlx_initialize_world_articulation_dispatches(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(1)]],
    device const MRArticulationGPU* articulations [[buffer(2)]],
    constant uint& abaFlags [[buffer(3)]],
    device MRMultiABADispatchGPU* abaDispatches [[buffer(4)]],
    device MRArticulatedOperatorDispatchGPU*
        kinematicsDispatches [[buffer(5)]],
    device MRArticulatedOperatorDispatchGPU*
        factorDispatches [[buffer(6)]],
    const uint owner [[thread_position_in_grid]]
) {
    if (owner >= contactDispatch.articulationCount) {
        return;
    }
    const MRArticulationGPU articulation = articulations[owner];

    MRMultiABADispatchGPU aba = {};
    aba.dispatch.articulationIndex = owner;
    aba.dispatch.environmentCount = worldDispatch.environmentCount;
    aba.dispatch.flags = abaFlags;
    aba.dispatch.qStride = worldDispatch.qStride;
    aba.dispatch.vStride = worldDispatch.vStride;
    aba.dispatch.effortStride =
        worldDispatch.effortEnvironmentStride;
    aba.dispatch.wrenchStride = 0u;
    aba.dispatch.accelerationStride = worldDispatch.vStride;
    aba.dispatch.nextVStride = worldDispatch.vStride;
    aba.dispatch.nextQStride = worldDispatch.qStride;
    aba.qBase = articulation.qOffset;
    aba.vBase = articulation.vOffset;
    aba.effortBase = articulation.vOffset;
    aba.wrenchBase = 0u;
    aba.accelerationBase = articulation.vOffset;
    aba.nextVBase = articulation.vOffset;
    aba.nextQBase = articulation.qOffset;
    aba.statusBase =
        owner * worldDispatch.environmentCount;
    abaDispatches[owner] = aba;

    MRArticulatedOperatorDispatchGPU kinematics = {};
    kinematics.articulationIndex = owner;
    kinematics.environmentCount = worldDispatch.environmentCount;
    kinematics.flags =
        MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY;
    kinematics.qStride = worldDispatch.qStride;
    kinematics.bodyPoseStride = contactDispatch.bodyCount;
    kinematics.generalizedStride = worldDispatch.vStride;
    kinematicsDispatches[owner] = kinematics;

    MRArticulatedOperatorDispatchGPU factor = kinematics;
    factor.pointCount = contactDispatch.pointQueryStride;
    factor.flags =
        MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR |
        (
            (worldDispatch.flags &
             MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES) != 0u
            ? MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES
            : 0u
        );
    factor.pointStride = contactDispatch.pointQueryStride;
    factor.pointWorldStride =
        contactDispatch.pointQueryStride;
    factor.massMatrixStride =
        articulation.nv * articulation.nv;
    factor.pointJacobianStride =
        contactDispatch.pointQueryStride *
        3u * articulation.nv;
    factor.generalizedStride = worldDispatch.vStride;
    factorDispatches[owner] = factor;
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
        linearVelocities[index];
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
            state.linearVelocityAndInverseMass;
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

// Converts the final solver rows into the tactile subsystem's explicit
// wrench evidence. Geometry sampling remains independent from these impulses;
// force is never inferred from penetration depth.
kernel void mr_mlx_pack_tactile_contacts(
    constant MRMLXContactAdapterDispatchGPU& adapter [[buffer(0)]],
    constant MRTactileDispatchGPU& tactile [[buffer(1)]],
    device const MRShapeGPU* shapes [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const MRManifoldHeaderGPU* manifoldHeaders [[buffer(4)]],
    device const MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* metadata [[buffer(6)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    device MRTactileContactGPU* tactileContacts [[buffer(8)]],
    device uint* tactileContactCounts [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= tactile.counts.x) {
        return;
    }
    const uint capacity = tactile.queryCounts.w;
    const uint base = environment * capacity;
    const MRMetalWorldContactStatusGPU status = statuses[environment];
    const uint active =
        status.code == MR_STEP_SUCCESS
        ? min(status.activeContacts, capacity)
        : 0u;
    tactileContactCounts[environment] = active;
    for (uint local = 0u; local < capacity; ++local) {
        MRTactileContactGPU output = {};
        if (local < active) {
            const uint flat =
                environment * adapter.contactCapacity + local;
            const MRContactPointMetaGPU meta = metadata[flat];
            const MRContactConstraintGPU contact = contacts[flat];
            const MRShapeGPU shapeA = shapes[meta.colliderA];
            const MRManifoldHeaderGPU manifold =
                manifoldHeaders[
                    environment * adapter.manifoldCapacity +
                    meta.manifoldIndex
                ];
            const float3 normal = normalize(contact.normal.xyz);
            float3 tangent = tactileQuaternionRotate(
                bodies[
                    environment * adapter.bodyStateStride +
                    shapeA.bodyIndex
                ].orientation,
                manifold.tangentAndMetric.xyz
            );
            tangent -= normal * dot(tangent, normal);
            tangent =
                dot(tangent, tangent) > 1.0e-12f
                ? normalize(tangent)
                : tactileStableTangent(normal);
            const float3 bitangent = cross(normal, tangent);
            const float3 impulseOnA = -(
                normal * contact.impulses.x +
                tangent * contact.impulses.y +
                bitangent * contact.impulses.z
            );
            output.shapesAndFlags = uint4(
                meta.colliderA,
                meta.colliderB,
                MR_TACTILE_CONTACT_SOLVER_IMPULSE,
                0u
            );
            output.worldPoint = contact.pointAndSeparation;
            output.worldImpulseOnA = float4(impulseOnA, 0.0f);
            output.solverImpulseAndFriction = float4(
                abs(contact.impulses.x),
                length(contact.impulses.yz),
                contact.friction.x,
                contact.friction.y
            );
        }
        tactileContacts[base + local] = output;
    }
}

// Publishes semantically named MLX arrays from the mixed float/uint summary
// record. Python policy code never depends on packed channel ordinals.
kernel void mr_mlx_unpack_tactile_summary(
    constant MRTactileDispatchGPU& dispatch [[buffer(0)]],
    device const MRTactileSummaryGPU* summaries [[buffer(1)]],
    device const float* timestamps [[buffer(2)]],
    device float4* posePositionAndTimestamp [[buffer(3)]],
    device float4* poseOrientation [[buffer(4)]],
    device float4* netForceAndContactArea [[buffer(5)]],
    device float4* netTorqueAndMaximumDepth [[buffer(6)]],
    device float4* centroidLocalAndMeanDepth [[buffer(7)]],
    device float4* centroidWorldAndActiveCount [[buffer(8)]],
    device float4* centerOfPressureLocalAndForceWeight [[buffer(9)]],
    device float4* centerOfPressureWorldAndContactCount [[buffer(10)]],
    device float4* tangentialMotionAndFriction [[buffer(11)]],
    device uint4* statisticsAndIdentity [[buffer(12)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint total = dispatch.counts.x * dispatch.counts.z;
    if (index >= total) {
        return;
    }
    const uint environment = index / dispatch.counts.z;
    const MRTactileSummaryGPU summary = summaries[index];
    posePositionAndTimestamp[index] = float4(
        summary.posePositionAndTimestamp.xyz,
        timestamps[environment]
    );
    poseOrientation[index] = summary.poseOrientation;
    netForceAndContactArea[index] =
        summary.netForceAndContactArea;
    netTorqueAndMaximumDepth[index] =
        summary.netTorqueAndMaximumDepth;
    centroidLocalAndMeanDepth[index] =
        summary.centroidLocalAndMeanDepth;
    centroidWorldAndActiveCount[index] =
        summary.centroidWorldAndActiveCount;
    centerOfPressureLocalAndForceWeight[index] =
        summary.centerOfPressureLocalAndForceWeight;
    centerOfPressureWorldAndContactCount[index] =
        summary.centerOfPressureWorldAndContactCount;
    tangentialMotionAndFriction[index] =
        summary.tangentialMotionAndFriction;
    statisticsAndIdentity[index] =
        summary.statisticsAndIdentity;
}

// Fixed-capacity masked object-local labels for field-estimator training.
// This is sample evidence, not a fabricated force distribution; exact solver
// points and impulses remain in the separate contact-evidence output.
kernel void mr_mlx_publish_object_contact_points(
    constant MRTactileDispatchGPU& dispatch [[buffer(0)]],
    device const MRTactileHitGPU* hits [[buffer(1)]],
    device const MRShapeGPU* shapes [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device float4* objectLocalPoints [[buffer(4)]],
    device float4* objectLocalNormals [[buffer(5)]],
    device uint* mask [[buffer(6)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint total = dispatch.counts.x * dispatch.counts.w;
    if (index >= total) {
        return;
    }
    const uint environment = index / dispatch.counts.w;
    const MRTactileHitGPU hit = hits[index];
    const uint shapeIndex = hit.identityAndFlags.x;
    const bool active =
        (hit.identityAndFlags.z &
         MR_TACTILE_VALIDITY_CONTACT) != 0u &&
        shapeIndex < dispatch.geometryCounts.x;
    float4 localPoint = {};
    float4 localNormal = {};
    if (active) {
        const MRShapeGPU shape = shapes[shapeIndex];
        const MRBodyStateGPU body =
            bodies[
                environment * dispatch.counts.y +
                shape.bodyIndex
            ];
        const float4 inverseOrientation =
            float4(-body.orientation.xyz, body.orientation.w);
        localPoint = float4(
            tactileQuaternionRotate(
                inverseOrientation,
                hit.worldPointAndDepth.xyz -
                    body.position.xyz
            ),
            hit.worldPointAndDepth.w
        );
        localNormal = float4(
            tactileQuaternionRotate(
                inverseOrientation,
                hit.worldNormalAndRayParameter.xyz
            ),
            0.0f
        );
    }
    objectLocalPoints[index] = localPoint;
    objectLocalNormals[index] = localNormal;
    mask[index] = active ? 1u : 0u;
}
