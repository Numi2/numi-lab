#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

constant float kQuaternionMinimum = 1.0e-12f;
constant float kMatrixFloor = 1.0e-10f;
constant float kConeEpsilon = 1.0e-7f;

struct Mat3 {
    float3 row0;
    float3 row1;
    float3 row2;
};

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline float4 quaternionMultiply(
    const float4 left,
    const float4 right
) {
    return float4(
        left.w * right.x + right.w * left.x +
            left.y * right.z - left.z * right.y,
        left.w * right.y + right.w * left.y +
            left.z * right.x - left.x * right.z,
        left.w * right.z + right.w * left.z +
            left.x * right.y - left.y * right.x,
        left.w * right.w -
            left.x * right.x -
            left.y * right.y -
            left.z * right.z
    );
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& output
) {
    if (!finite4(input)) {
        return false;
    }
    const float normSquared = dot(input, input);
    if (!(normSquared > kQuaternionMinimum) ||
        !isfinite(normSquared)) {
        return false;
    }
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline Mat3 bodyInverseInertia(
    device const MRBodyPropertiesGPU& body
) {
    Mat3 result;
    result.row0 = body.inverseInertiaRow0.xyz;
    result.row1 = body.inverseInertiaRow1.xyz;
    result.row2 = body.inverseInertiaRow2.xyz;
    return result;
}

inline float3 multiply(
    const thread Mat3& matrix,
    const float3 vector
) {
    return float3(
        dot(matrix.row0, vector),
        dot(matrix.row1, vector),
        dot(matrix.row2, vector)
    );
}

inline Mat3 rotationMatrix(const float4 quaternion) {
    const float x = quaternion.x;
    const float y = quaternion.y;
    const float z = quaternion.z;
    const float w = quaternion.w;
    Mat3 result;
    result.row0 = float3(
        1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - z * w),
        2.0f * (x * z + y * w)
    );
    result.row1 = float3(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - x * w)
    );
    result.row2 = float3(
        2.0f * (x * z - y * w),
        2.0f * (y * z + x * w),
        1.0f - 2.0f * (x * x + y * y)
    );
    return result;
}

inline Mat3 transpose(const thread Mat3& matrix) {
    Mat3 result;
    result.row0 = float3(
        matrix.row0.x,
        matrix.row1.x,
        matrix.row2.x
    );
    result.row1 = float3(
        matrix.row0.y,
        matrix.row1.y,
        matrix.row2.y
    );
    result.row2 = float3(
        matrix.row0.z,
        matrix.row1.z,
        matrix.row2.z
    );
    return result;
}

inline Mat3 multiply(
    const thread Mat3& left,
    const thread Mat3& right
) {
    const Mat3 rightTranspose = transpose(right);
    Mat3 result;
    result.row0 = float3(
        dot(left.row0, rightTranspose.row0),
        dot(left.row0, rightTranspose.row1),
        dot(left.row0, rightTranspose.row2)
    );
    result.row1 = float3(
        dot(left.row1, rightTranspose.row0),
        dot(left.row1, rightTranspose.row1),
        dot(left.row1, rightTranspose.row2)
    );
    result.row2 = float3(
        dot(left.row2, rightTranspose.row0),
        dot(left.row2, rightTranspose.row1),
        dot(left.row2, rightTranspose.row2)
    );
    return result;
}

inline bool writeWorldInverseInertia(
    thread MRBodyStateGPU& state,
    device const MRBodyPropertiesGPU& body,
    const float4 orientation
) {
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 rotated = multiply(
        multiply(rotation, bodyInverseInertia(body)),
        transpose(rotation)
    );
    if (!finite3(rotated.row0) ||
        !finite3(rotated.row1) ||
        !finite3(rotated.row2)) {
        return false;
    }
    state.inverseInertiaWorldRow0 =
        float4(rotated.row0, 0.0f);
    state.inverseInertiaWorldRow1 =
        float4(rotated.row1, 0.0f);
    state.inverseInertiaWorldRow2 =
        float4(rotated.row2, 0.0f);
    return true;
}

inline Mat3 stateInverseInertia(
    device const MRBodyStateGPU& state
) {
    Mat3 result;
    result.row0 = state.inverseInertiaWorldRow0.xyz;
    result.row1 = state.inverseInertiaWorldRow1.xyz;
    result.row2 = state.inverseInertiaWorldRow2.xyz;
    return result;
}

inline bool validSceneState(
    device const MRBodyStateGPU& state,
    device const MRBodyPropertiesGPU& body,
    const uint globalBody
) {
    float4 normalized;
    return
        finite4(state.position) &&
        normalizedQuaternion(state.orientation, normalized) &&
        finite4(state.linearVelocityAndInverseMass) &&
        finite4(state.angularVelocity) &&
        state.flagsAndIndices[0] == body.motionType &&
        state.flagsAndIndices[1] == MR_INVALID_INDEX &&
        (
            state.flagsAndIndices[2] == globalBody ||
            state.flagsAndIndices[2] == MR_INVALID_INDEX
        );
}

inline uint mapOperatorStatus(const uint code) {
    switch (code) {
    case MR_ARTICULATED_OPERATOR_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ARTICULATED_OPERATOR_NONFINITE_INPUT:
        return MR_STEP_NONFINITE_INPUT;
    case MR_ARTICULATED_OPERATOR_FACTORIZATION_FAILED:
        return MR_STEP_FACTORIZATION_FAILED;
    case MR_ARTICULATED_OPERATOR_NONFINITE_RESULT:
    case MR_ARTICULATED_OPERATOR_ACCURACY_FAILED:
        return MR_STEP_NONFINITE_RESULT;
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

inline bool sceneEndpoint(
    device const MRBodyStateGPU& body,
    const uint articulationIndex
) {
    return body.flagsAndIndices[1] != articulationIndex;
}

inline bool dynamicSceneEndpoint(
    device const MRBodyStateGPU& body,
    const uint articulationIndex
) {
    return sceneEndpoint(body, articulationIndex) &&
        body.flagsAndIndices[0] == MR_MOTION_DYNAMIC;
}

inline float3 pointVelocity(
    device const MRBodyStateGPU& body,
    const float3 point
) {
    return
        body.linearVelocityAndInverseMass.xyz +
        cross(
            body.angularVelocity.xyz,
            point - body.position.xyz
        );
}

inline float3 articulatedPointVelocity(
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint query,
    const uint nv,
    device const float* velocity
) {
    float3 result = float3(0.0f);
    for (uint dof = 0u; dof < nv; ++dof) {
        result.x +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 0u) * nv + dof
            ] * velocity[dof];
        result.y +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 1u) * nv + dof
            ] * velocity[dof];
        result.z +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 2u) * nv + dof
            ] * velocity[dof];
    }
    return result;
}

inline float3 combinedJacobianColumn(
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint localConstraint,
    const uint dof,
    const uint nv,
    const bool articulatedA,
    const bool articulatedB
) {
    float3 result = float3(0.0f);
    const uint queryA = 2u * localConstraint;
    const uint queryB = queryA + 1u;
    if (articulatedA) {
        result -= float3(
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 0u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 1u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 2u) * nv + dof
            ]
        );
    }
    if (articulatedB) {
        result += float3(
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 0u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 1u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 2u) * nv + dof
            ]
        );
    }
    return result;
}

inline bool solveCholesky(
    device const float* factor,
    const uint factorBase,
    const uint nv,
    thread const float* rightHandSide,
    thread float* intermediate,
    thread float* solution
) {
    for (uint row = 0u; row < nv; ++row) {
        float value = rightHandSide[row];
        for (uint column = 0u; column < row; ++column) {
            value -=
                factor[factorBase + row * nv + column] *
                intermediate[column];
        }
        const float diagonal =
            factor[factorBase + row * nv + row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        intermediate[row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < nv; ++reverse) {
        const uint row = nv - 1u - reverse;
        float value = intermediate[row];
        for (uint column = row + 1u;
             column < nv;
             ++column) {
            value -=
                factor[factorBase + column * nv + row] *
                solution[column];
        }
        solution[row] =
            value / factor[factorBase + row * nv + row];
        if (!isfinite(solution[row])) {
            return false;
        }
    }
    return true;
}

inline float3 scenePointResponse(
    device const MRBodyStateGPU& body,
    const float3 point,
    const float3 impulse
) {
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return float3(0.0f);
    }
    const float3 lever = point - body.position.xyz;
    const float3 angularDelta = multiply(
        stateInverseInertia(body),
        cross(lever, impulse)
    );
    return
        body.linearVelocityAndInverseMass.w * impulse +
        cross(angularDelta, lever);
}

inline void applySceneImpulse(
    device MRBodyStateGPU& body,
    const float3 point,
    const float3 impulse
) {
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return;
    }
    const float3 lever = point - body.position.xyz;
    body.linearVelocityAndInverseMass.xyz +=
        body.linearVelocityAndInverseMass.w * impulse;
    body.angularVelocity.xyz += multiply(
        stateInverseInertia(body),
        cross(lever, impulse)
    );
}

inline float3 projectFrictionCone(
    const float3 impulse,
    device const MREvaluatedConstraintIRConeGPU& cone
) {
    float3 projected = impulse;
    projected.x = max(projected.x, 0.0f);
    if (cone.maximumNormalImpulse > 0.0f) {
        projected.x = min(
            projected.x,
            cone.maximumNormalImpulse
        );
    }
    const float limitU =
        cone.effectiveFrictionU * projected.x;
    const float limitV =
        cone.effectiveFrictionV * projected.x;
    if (!(limitU > 0.0f) || !(limitV > 0.0f)) {
        projected.yz = float2(0.0f);
        return projected;
    }
    const float normalizedSquared =
        (projected.y * projected.y) / (limitU * limitU) +
        (projected.z * projected.z) / (limitV * limitV);
    if (normalizedSquared > 1.0f) {
        projected.yz *= rsqrt(normalizedSquared);
    }
    return projected;
}

inline bool invert3x3(
    thread const float matrix[3][3],
    thread float inverse[3][3]
) {
    const float c00 =
        matrix[1][1] * matrix[2][2] -
        matrix[1][2] * matrix[2][1];
    const float c01 =
        matrix[1][2] * matrix[2][0] -
        matrix[1][0] * matrix[2][2];
    const float c02 =
        matrix[1][0] * matrix[2][1] -
        matrix[1][1] * matrix[2][0];
    const float determinant =
        matrix[0][0] * c00 +
        matrix[0][1] * c01 +
        matrix[0][2] * c02;
    if (!(determinant > kMatrixFloor) ||
        !isfinite(determinant)) {
        return false;
    }
    const float reciprocal = 1.0f / determinant;
    inverse[0][0] = c00 * reciprocal;
    inverse[0][1] =
        (matrix[0][2] * matrix[2][1] -
         matrix[0][1] * matrix[2][2]) * reciprocal;
    inverse[0][2] =
        (matrix[0][1] * matrix[1][2] -
         matrix[0][2] * matrix[1][1]) * reciprocal;
    inverse[1][0] = c01 * reciprocal;
    inverse[1][1] =
        (matrix[0][0] * matrix[2][2] -
         matrix[0][2] * matrix[2][0]) * reciprocal;
    inverse[1][2] =
        (matrix[0][2] * matrix[1][0] -
         matrix[0][0] * matrix[1][2]) * reciprocal;
    inverse[2][0] = c02 * reciprocal;
    inverse[2][1] =
        (matrix[0][1] * matrix[2][0] -
         matrix[0][0] * matrix[2][1]) * reciprocal;
    inverse[2][2] =
        (matrix[0][0] * matrix[1][1] -
         matrix[0][1] * matrix[1][0]) * reciprocal;
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            if (!isfinite(inverse[row][column])) {
                return false;
            }
        }
    }
    return true;
}

inline float3 relativePointVelocity(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    device const MRBodyStateGPU* bodies,
    const uint articulationIndex,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint nv,
    device const float* articulationVelocity
) {
    device const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    device const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
    const bool articulatedA =
        bodyA.flagsAndIndices[1] == articulationIndex;
    const bool articulatedB =
        bodyB.flagsAndIndices[1] == articulationIndex;
    float3 relative = float3(0.0f);
    if (articulatedA) {
        relative -= articulatedPointVelocity(
            pointJacobians,
            pointJacobianBase,
            2u * localConstraint,
            nv,
            articulationVelocity
        );
    } else {
        relative -= pointVelocity(
            bodyA,
            contact.pointAndSeparation.xyz
        );
    }
    if (articulatedB) {
        relative += articulatedPointVelocity(
            pointJacobians,
            pointJacobianBase,
            2u * localConstraint + 1u,
            nv,
            articulationVelocity
        );
    } else {
        relative += pointVelocity(
            bodyB,
            contact.pointAndSeparation.xyz
        );
    }
    return relative;
}

inline void applyContactDelta(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    const float3 delta,
    device const MREvaluatedConstraintIRRowGPU* rows,
    device float* articulationVelocity,
    device MRBodyStateGPU* bodies,
    device const float* responseColumns,
    const uint responseBase,
    const uint nv,
    const uint articulationIndex
) {
    const float3 impulse =
        rows[0].direction.xyz * delta.x +
        rows[1].direction.xyz * delta.y +
        rows[2].direction.xyz * delta.z;
    for (uint dof = 0u; dof < nv; ++dof) {
        articulationVelocity[dof] +=
            responseColumns[
                responseBase +
                (localConstraint * 3u + 0u) * nv + dof
            ] * delta.x +
            responseColumns[
                responseBase +
                (localConstraint * 3u + 1u) * nv + dof
            ] * delta.y +
            responseColumns[
                responseBase +
                (localConstraint * 3u + 2u) * nv + dof
            ] * delta.z;
    }
    device MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    device MRBodyStateGPU& bodyB = bodies[contact.bodyB];
    if (dynamicSceneEndpoint(bodyA, articulationIndex)) {
        applySceneImpulse(
            bodyA,
            contact.pointAndSeparation.xyz,
            -impulse
        );
    }
    if (dynamicSceneEndpoint(bodyB, articulationIndex)) {
        applySceneImpulse(
            bodyB,
            contact.pointAndSeparation.xyz,
            impulse
        );
    }
}

inline uint findRoot(thread uint* parents, uint node) {
    uint current = node;
    while (parents[current] != current) {
        current = parents[current];
    }
    while (parents[node] != node) {
        const uint next = parents[node];
        parents[node] = current;
        node = next;
    }
    return current;
}

inline void unionRoots(
    thread uint* parents,
    const uint left,
    const uint right
) {
    const uint leftRoot = findRoot(parents, left);
    const uint rightRoot = findRoot(parents, right);
    if (leftRoot == rightRoot) {
        return;
    }
    const uint minimum = min(leftRoot, rightRoot);
    const uint maximum = max(leftRoot, rightRoot);
    parents[maximum] = minimum;
}

} // namespace

// Applies resets/kinematic targets and checkpoints the complete contact state
// at the start of one control step.
kernel void mr_world_prepare_contact_step(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const MRBodyStateGPU* resetSceneBodies [[buffer(4)]],
    device const MRBodyStateGPU* kinematicTargets [[buffer(5)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(6)]],
    device const uint* sceneBodyIndices [[buffer(7)]],
    device MRBodyStateGPU* sceneState [[buffer(8)]],
    device MRBodyStateGPU* checkpointSceneState [[buffer(9)]],
    device MRManifoldHeaderGPU* manifoldHeaders [[buffer(10)]],
    device MRManifoldPointGPU* manifoldPoints [[buffer(11)]],
    device uint* manifoldCounts [[buffer(12)]],
    device MRManifoldHeaderGPU* checkpointHeaders [[buffer(13)]],
    device MRManifoldPointGPU* checkpointPoints [[buffer(14)]],
    device uint* checkpointCounts [[buffer(15)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(16)]],
    device MRConvexQueryCacheGPU* convexCaches [[buffer(17)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = {};
    status.code = MR_STEP_SUCCESS;
    status.environment = environment;
    status.controlStep = pass.controlStep;
    status.physicsSubstep = MR_INVALID_INDEX;
    status.firstFailingPair = MR_INVALID_INDEX;
    status.firstFailingConstraint = MR_INVALID_INDEX;
    status.firstFailingEventKeyLow = MR_INVALID_INDEX;
    status.firstFailingEventKeyHigh = MR_INVALID_INDEX;

    const bool hasResets =
        (worldDispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u;
    const bool applyReset =
        hasResets &&
        resetMasks[
            pass.controlStep *
                worldDispatch.resetMaskStepStride +
            environment
        ] != 0u;
    const bool hasTargets =
        (dispatch.flags &
         MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS) != 0u;
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint targetBase =
        (
            pass.controlStep * dispatch.environmentCount +
            environment
        ) * dispatch.sceneBodyStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        MRBodyStateGPU value = applyReset
            ? resetSceneBodies[sceneBase + localScene]
            : sceneState[sceneBase + localScene];
        if (hasTargets &&
            bodyProperties[globalBody].motionType ==
                MR_MOTION_KINEMATIC) {
            value = kinematicTargets[targetBase + localScene];
        }
        sceneState[sceneBase + localScene] = value;
        checkpointSceneState[sceneBase + localScene] = value;
    }

    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint oldCount = applyReset
        ? 0u
        : min(
              manifoldCounts[environment],
              dispatch.manifoldCapacity
          );
    checkpointCounts[environment] = oldCount;
    manifoldCounts[environment] = oldCount;
    if (applyReset) {
        const uint cacheBase =
            environment * dispatch.convexCacheStride;
        for (uint pair = 0u;
             pair < dispatch.eligiblePairCount;
             ++pair) {
            convexCaches[cacheBase + pair] = {};
        }
    }
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        if (manifold < oldCount) {
            checkpointHeaders[manifoldBase + manifold] =
                manifoldHeaders[manifoldBase + manifold];
            for (uint point = 0u;
                 point <
                     MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
                 ++point) {
                checkpointPoints[
                    pointBase +
                    manifold *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    point
                ] = manifoldPoints[
                    pointBase +
                    manifold *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    point
                ];
            }
        }
    }
    statuses[environment] = status;
}

// Projects articulation poses and scene states into one global body-state
// array consumed by collision. Inertia for scene bodies is rebuilt from the
// immutable body tensor so callers cannot forge inverse mass.
kernel void mr_world_build_body_states(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(4)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(5)]],
    device const MRBodyStateGPU* sceneStates [[buffer(6)]],
    device MRBodyStateGPU* bodyStates [[buffer(7)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    const MRArticulatedOperatorStatusGPU operatorStatus =
        operatorStatuses[environment];
    if (status.code != MR_STEP_SUCCESS ||
        operatorStatus.code !=
            MR_ARTICULATED_OPERATOR_SUCCESS) {
        if (status.code == MR_STEP_SUCCESS) {
            status.code = mapOperatorStatus(operatorStatus.code);
            status.firstFailingConstraint =
                operatorStatus.failingIndex;
            statuses[environment] = status;
        }
        return;
    }
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint poseBase =
        environment * articulation.bodyCount;
    for (uint localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const uint globalBody =
            articulation.firstBody + localBody;
        const MRArticulatedBodyPoseGPU pose =
            bodyPoses[poseBase + localBody];
        MRBodyStateGPU state = {};
        state.position = pose.position;
        state.orientation = pose.orientation;
        state.linearVelocityAndInverseMass.w = 0.0f;
        state.flagsAndIndices[0] =
            bodyProperties[globalBody].motionType;
        state.flagsAndIndices[1] = dispatch.articulationIndex;
        state.flagsAndIndices[2] = globalBody;
        state.flagsAndIndices[3] = 0u;
        bodyStates[bodyBase + globalBody] = state;
    }

    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        device const MRBodyStateGPU& input =
            sceneStates[sceneBase + localScene];
        if (globalBody >= dispatch.bodyCount ||
            properties.articulationIndex != MR_INVALID_INDEX ||
            !validSceneState(input, properties, globalBody)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        float4 orientation;
        if (!normalizedQuaternion(
                input.orientation,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        MRBodyStateGPU state = input;
        state.position.w = 1.0f;
        state.orientation = orientation;
        state.linearVelocityAndInverseMass.w =
            properties.motionType == MR_MOTION_DYNAMIC
            ? properties.massAndInverseMass.y
            : 0.0f;
        state.angularVelocity.w = 0.0f;
        state.flagsAndIndices[0] = properties.motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = globalBody;
        if (!writeWorldInverseInertia(
                state,
                properties,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        bodyStates[bodyBase + globalBody] = state;
    }
    statuses[environment] = status;
}

// Predicts unconstrained scene-body velocities while preserving current
// collision poses. Articulation entries are copied unchanged; ABA owns qdot.
kernel void mr_world_predict_scene(
    device const MRWorldGPU& world [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const MRBodyStateGPU* currentBodies [[buffer(4)]],
    device MRBodyStateGPU* candidateBodies [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint body = 0u; body < dispatch.bodyCount; ++body) {
        candidateBodies[bodyBase + body] =
            currentBodies[bodyBase + body];
    }
    const float timestep = dispatch.timestepAndBias.x;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device MRBodyStateGPU& state =
            candidateBodies[bodyBase + globalBody];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        if (properties.motionType != MR_MOTION_DYNAMIC) {
            continue;
        }
        // Exponential damping is invariant under event-time splitting:
        // exp(-d * h0) * exp(-d * h1) == exp(-d * (h0 + h1)).
        const float linearScale = exp(
            -timestep *
                properties.dampingAndSpeedLimits.x
        );
        const float angularScale = exp(
            -timestep *
                properties.dampingAndSpeedLimits.y
        );
        state.linearVelocityAndInverseMass.xyz =
            linearScale *
                state.linearVelocityAndInverseMass.xyz +
            timestep * world.gravityAndTimestep.xyz;
        state.angularVelocity.xyz *= angularScale;
        if (!finite4(state.linearVelocityAndInverseMass) ||
            !finite4(state.angularVelocity)) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
    }
}

// Reduces device-resident contact counts into the point-query prefix used by
// the articulated factor/Jacobian pass. Fixed strides remain unchanged and
// no count crosses the CPU boundary.
kernel void mr_world_finalize_factor_dispatch(
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(0)]],
    device const MRMetalWorldContactStatusGPU* statuses
        [[buffer(1)]],
    device MRArticulatedOperatorDispatchGPU* operatorDispatch
        [[buffer(2)]],
    device MRIndirectDispatchArgumentsGPU* indirectArguments
        [[buffer(3)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (threadIndex != 0u) {
        return;
    }
    uint maximumConstraints = 0u;
    for (uint environment = 0u;
         environment < contactDispatch.environmentCount;
         ++environment) {
        const MRMetalWorldContactStatusGPU status =
            statuses[environment];
        if (status.code == MR_STEP_SUCCESS) {
            maximumConstraints = max(
                maximumConstraints,
                min(
                    status.requiredConstraints,
                    contactDispatch.constraintCapacity
                )
            );
        }
    }
    const uint activePointCount = min(
        2u * maximumConstraints,
        contactDispatch.pointQueryStride
    );
    operatorDispatch[0].pointCount = activePointCount;
    MRIndirectDispatchArgumentsGPU arguments = {};
    arguments.threadgroupsX =
        activePointCount == 0u
        ? 0u
        : contactDispatch.environmentCount;
    arguments.threadgroupsY = 1u;
    arguments.threadgroupsZ = 1u;
    arguments.activeCount = activePointCount;
    indirectArguments[0] = arguments;
    MRIndirectDispatchArgumentsGPU contactArguments = {};
    contactArguments.threadgroupsX =
        activePointCount == 0u
        ? 0u
        : (
            contactDispatch.environmentCount + 63u
          ) / 64u;
    contactArguments.threadgroupsY = 1u;
    contactArguments.threadgroupsZ = 1u;
    contactArguments.activeCount = maximumConstraints;
    indirectArguments[1] = contactArguments;
}

// Environments can have different active contact counts while the articulated
// operator consumes one batch-wide point prefix. Only the short tail is
// initialized; active queries were emitted by the collision compiler.
kernel void mr_world_fill_point_query_tail(
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(0)]],
    device const MRArticulatedOperatorDispatchGPU& operatorDispatch
        [[buffer(1)]],
    device const MRArticulationGPU* articulations [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* statuses
        [[buffer(3)]],
    device MRArticulatedPointImpulseGPU* pointQueries
        [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= contactDispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint activePoints = min(
        2u * statuses[environment].requiredConstraints,
        operatorDispatch.pointCount
    );
    MRArticulatedPointImpulseGPU dummy = {};
    dummy.bodyIndex =
        articulations[contactDispatch.articulationIndex].rootBody;
    const uint base =
        environment * contactDispatch.pointQueryStride;
    for (uint point = activePoints;
         point < operatorDispatch.pointCount;
         ++point) {
        pointQueries[base + point] = dummy;
    }
}

// Evaluates the canonical IR at the current microstep after point Jacobians
// and free velocities are available.
kernel void mr_world_evaluate_constraint_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRContactConstraintGPU* contacts [[buffer(1)]],
    device MRContactConstraintGPU* mutableContacts [[buffer(2)]],
    device const MRConstraintIRBlockGPU* blocks [[buffer(3)]],
    device const MRConstraintIRRowGPU* rows [[buffer(4)]],
    device const MRConstraintIRConeGPU* cones [[buffer(5)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(6)]],
    device const float* candidateV [[buffer(7)]],
    device const float* pointJacobians [[buffer(8)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(9)]],
    device MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(10)]],
    device MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(11)]],
    device MRArticulationFactorCacheGPU* factorCaches [[buffer(12)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const MRArticulatedOperatorStatusGPU operatorStatus =
        operatorStatuses[environment];
    MRArticulationFactorCacheGPU cache = {};
    cache.environment = environment;
    cache.articulationIndex = dispatch.articulationIndex;
    cache.nv = dispatch.nv;
    cache.generation = status.physicsSubstep;
    cache.code = operatorStatus.code;
    cache.failingIndex = operatorStatus.failingIndex;
    cache.diagnostics = operatorStatus.diagnostics;
    factorCaches[environment] = cache;
    if (operatorStatus.code !=
        MR_ARTICULATED_OPERATOR_SUCCESS) {
        status.code = mapOperatorStatus(operatorStatus.code);
        status.firstFailingConstraint =
            operatorStatus.failingIndex;
        statuses[environment] = status;
        return;
    }
    status.diagnostics.z = operatorStatus.diagnostics.x;
    status.diagnostics.w = operatorStatus.diagnostics.y;
    status.residuals.w = operatorStatus.diagnostics.z;

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const float timestep = dispatch.timestepAndBias.x;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        const uint constraintIndex =
            constraintBase + localConstraint;
        device const MRContactConstraintGPU& contact =
            contacts[constraintIndex];
        const MRConstraintIRBlockGPU block =
            blocks[constraintIndex];
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            candidateBodies + bodyBase,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            candidateV + velocityBase
        );
        float relativeRows[3];
        for (uint localRow = 0u;
             localRow < 3u;
             ++localRow) {
            relativeRows[localRow] = dot(
                rows[
                    rowBase + block.rowOffset + localRow
                ].direction.xyz,
                relative
            );
        }
        const MRConstraintIRConeGPU sourceCone =
            cones[constraintIndex];
        const float slip = length(
            float2(relativeRows[1], relativeRows[2])
        );
        const bool staticRegion =
            slip <= max(
                1.0e-3f,
                sourceCone.stictionTransitionVelocity
            );
        MREvaluatedConstraintIRConeGPU evaluatedCone = {};
        evaluatedCone.effectiveFrictionU =
            staticRegion
            ? sourceCone.staticFrictionU
            : sourceCone.dynamicFrictionU;
        evaluatedCone.effectiveFrictionV =
            staticRegion
            ? sourceCone.staticFrictionV
            : sourceCone.dynamicFrictionV;
        evaluatedCone.staticFrictionU =
            sourceCone.staticFrictionU;
        evaluatedCone.staticFrictionV =
            sourceCone.staticFrictionV;
        evaluatedCone.dynamicFrictionU =
            sourceCone.dynamicFrictionU;
        evaluatedCone.dynamicFrictionV =
            sourceCone.dynamicFrictionV;
        evaluatedCone.rollingLength = sourceCone.rollingLength;
        evaluatedCone.torsionalLength =
            sourceCone.torsionalLength;
        evaluatedCone.restitutionThreshold =
            sourceCone.restitutionThreshold;
        evaluatedCone.adhesionImpulse =
            sourceCone.adhesionImpulse;
        evaluatedCone.maximumNormalImpulse =
            sourceCone.maximumNormalImpulse;

        float restitutionVelocity = 0.0f;
        for (uint localRow = 0u;
             localRow < 3u;
             ++localRow) {
            const MRConstraintIRRowGPU source =
                rows[
                    rowBase + block.rowOffset + localRow
                ];
            float stabilization = 0.0f;
            if ((source.flags &
                 MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED) !=
                0u) {
                const float tau = max(
                    source.timeConstant,
                    2.0f * timestep
                );
                float positionError = source.positionError;
                if ((source.flags &
                     MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL) !=
                    0u) {
                    positionError = min(
                        positionError +
                            dispatch.timestepAndBias.y,
                        0.0f
                    );
                } else if ((source.flags &
                            MR_CONSTRAINT_IR_ROW_UNILATERAL) !=
                           0u) {
                    positionError = min(positionError, 0.0f);
                }
                const float ratio = timestep / tau;
                const float denominator =
                    1.0f +
                    2.0f * source.dampingRatio * ratio +
                    ratio * ratio;
                stabilization = max(
                    -timestep * positionError /
                        (tau * tau * denominator),
                    0.0f
                );
                stabilization = clamp(
                    stabilization,
                    0.0f,
                    dispatch.timestepAndBias.z
                );
            }
            if (localRow == 0u &&
                (block.flags &
                 MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT) != 0u) {
                const float incoming =
                    relativeRows[0] - source.targetVelocity;
                if (incoming <
                    -sourceCone.restitutionThreshold) {
                    restitutionVelocity =
                        -sourceCone.restitution * incoming;
                    stabilization = max(
                        stabilization,
                        restitutionVelocity
                    );
                }
            }
            MREvaluatedConstraintIRRowGPU evaluated = {};
            evaluated.direction = source.direction;
            evaluated.targetVelocity =
                source.targetVelocity + stabilization;
            evaluated.regularization = max(
                source.compliance / (timestep * timestep) +
                    source.dissipation / timestep,
                0.0f
            );
            evaluated.impulseLower = source.impulseLower;
            evaluated.impulseUpper = source.impulseUpper;
            evaluated.sourcePositionError =
                source.positionError;
            evaluated.stabilizationVelocity = stabilization;
            evaluated.sourceTargetVelocity =
                source.targetVelocity;
            evaluated.relativeVelocity =
                relativeRows[localRow];
            evaluated.preSolveVelocity =
                relativeRows[localRow];
            if (!finite4(evaluated.direction) ||
                !isfinite(evaluated.targetVelocity) ||
                !isfinite(evaluated.regularization)) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }
            evaluatedRows[
                rowBase + block.rowOffset + localRow
            ] = evaluated;
        }
        evaluatedCone.restitutionVelocity =
            restitutionVelocity;
        evaluatedCones[constraintIndex] = evaluatedCone;
        MRContactConstraintGPU updated = contact;
        updated.targetVelocityAndPreSolveNormal.w =
            relativeRows[0];
        mutableContacts[constraintIndex] = updated;
    }
    statuses[environment] = status;
}

// Deterministic minimum-root union/find over the selected articulation and
// free dynamic bodies.
kernel void mr_world_build_contact_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyStateGPU* bodies [[buffer(1)]],
    device MRContactConstraintGPU* contacts [[buffer(2)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(3)]],
    device MRContactIslandGPU* islands [[buffer(4)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    if (dispatch.bodyCount >
        MR_ARTICULATED_OPERATOR_MAX_BODIES) {
        status.code = MR_STEP_ISLAND_CAPACITY_OVERFLOW;
        status.requiredIslands = dispatch.bodyCount;
        statuses[environment] = status;
        return;
    }
    if (status.requiredConstraints == 0u) {
        status.requiredIslands = 0u;
        status.islandCount = 0u;
        statuses[environment] = status;
        return;
    }

    uint parents[MR_ARTICULATED_OPERATOR_MAX_BODIES + 1u];
    bool active[MR_ARTICULATED_OPERATOR_MAX_BODIES + 1u];
    for (uint node = 0u; node <= dispatch.bodyCount; ++node) {
        parents[node] = node;
        active[node] = false;
    }
    active[0] = true;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint islandBase =
        environment * dispatch.islandStride;
    for (uint body = 0u; body < dispatch.bodyCount; ++body) {
        device const MRBodyStateGPU& state =
            bodies[bodyBase + body];
        if (state.flagsAndIndices[1] !=
                dispatch.articulationIndex &&
            state.flagsAndIndices[0] == MR_MOTION_DYNAMIC) {
            active[body + 1u] = true;
        }
    }

    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device const MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MRBodyStateGPU& bodyA =
            bodies[bodyBase + contact.bodyA];
        device const MRBodyStateGPU& bodyB =
            bodies[bodyBase + contact.bodyB];
        const uint nodeA =
            bodyA.flagsAndIndices[1] ==
                dispatch.articulationIndex
            ? 0u
            : bodyA.flagsAndIndices[0] == MR_MOTION_DYNAMIC
                ? contact.bodyA + 1u
                : MR_INVALID_INDEX;
        const uint nodeB =
            bodyB.flagsAndIndices[1] ==
                dispatch.articulationIndex
            ? 0u
            : bodyB.flagsAndIndices[0] == MR_MOTION_DYNAMIC
                ? contact.bodyB + 1u
                : MR_INVALID_INDEX;
        if (nodeA == MR_INVALID_INDEX &&
            nodeB == MR_INVALID_INDEX) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint =
                localConstraint;
            statuses[environment] = status;
            return;
        }
        if (nodeA != MR_INVALID_INDEX &&
            nodeB != MR_INVALID_INDEX) {
            unionRoots(parents, nodeA, nodeB);
        }
    }

    uint roots[MR_ARTICULATED_OPERATOR_MAX_BODIES + 1u];
    uint rootCount = 0u;
    for (uint node = 0u; node <= dispatch.bodyCount; ++node) {
        if (!active[node]) {
            continue;
        }
        const uint root = findRoot(parents, node);
        bool found = false;
        for (uint index = 0u; index < rootCount; ++index) {
            found = found || roots[index] == root;
        }
        if (!found) {
            roots[rootCount++] = root;
        }
    }
    status.requiredIslands = rootCount;
    status.islandCount = rootCount;
    if (rootCount > dispatch.islandCapacity) {
        status.code = MR_STEP_ISLAND_CAPACITY_OVERFLOW;
        statuses[environment] = status;
        return;
    }
    for (uint island = 0u; island < rootCount; ++island) {
        MRContactIslandGPU record = {};
        record.environment = environment;
        record.stableRoot = roots[island];
        record.firstConstraint = MR_INVALID_INDEX;
        islands[islandBase + island] = record;
    }
    for (uint node = 0u; node <= dispatch.bodyCount; ++node) {
        if (!active[node]) {
            continue;
        }
        const uint root = findRoot(parents, node);
        for (uint island = 0u; island < rootCount; ++island) {
            if (roots[island] == root) {
                ++islands[
                    islandBase + island
                ].dynamicNodeCount;
                break;
            }
        }
    }
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MRBodyStateGPU& bodyA =
            bodies[bodyBase + contact.bodyA];
        device const MRBodyStateGPU& bodyB =
            bodies[bodyBase + contact.bodyB];
        uint node =
            bodyA.flagsAndIndices[1] ==
                dispatch.articulationIndex
            ? 0u
            : bodyA.flagsAndIndices[0] == MR_MOTION_DYNAMIC
                ? contact.bodyA + 1u
                : bodyB.flagsAndIndices[1] ==
                      dispatch.articulationIndex
                    ? 0u
                    : contact.bodyB + 1u;
        const uint root = findRoot(parents, node);
        for (uint island = 0u; island < rootCount; ++island) {
            if (roots[island] != root) {
                continue;
            }
            contact.islandIndex = island;
            blocks[constraintBase + localConstraint]
                .islandIndex = island;
            device MRContactIslandGPU& record =
                islands[islandBase + island];
            record.firstConstraint = min(
                record.firstConstraint,
                localConstraint
            );
            ++record.constraintCount;
            break;
        }
    }
    statuses[environment] = status;
}

// Packs each deterministic island into contiguous 32-contact tiles. The
// contact-index arena is environment-major and preserves canonical constraint
// order even when constraints belonging to different islands were interleaved
// by the manifold compiler.
kernel void mr_world_build_contact_tiles(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyStateGPU* bodies [[buffer(1)]],
    device const MRContactConstraintGPU* contacts [[buffer(2)]],
    device const MRContactIslandGPU* islands [[buffer(3)]],
    device MRIslandWorkGPU* islandWork [[buffer(4)]],
    device MRContactTileGPU* tiles [[buffer(5)]],
    device uint* tileConstraintIndices [[buffer(6)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    device uint* islandWorkFlags [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    for (uint islandIndex = 0u;
         islandIndex < dispatch.islandCapacity;
         ++islandIndex) {
        islandWorkFlags[islandBase + islandIndex] = 0u;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    status.requiredSolverTiles = 0u;
    status.requiredSpillRows = 0u;
    status.solverTiles = 0u;
    status.spillRows = 0u;
    if (status.requiredConstraints == 0u) {
        statuses[environment] = status;
        return;
    }

    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint tileBase =
        environment * dispatch.solverTileCapacity;
    uint tileCursor = 0u;
    uint constraintIndexCursor = 0u;
    ulong requiredSpillRows = 0u;
    for (uint islandIndex = 0u;
         islandIndex < status.islandCount;
         ++islandIndex) {
        const MRContactIslandGPU island =
            islands[islandBase + islandIndex];
        const uint islandIndexStart = constraintIndexCursor;
        bool hasArticulation = false;
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            const MRContactConstraintGPU contact =
                contacts[constraintBase + localConstraint];
            if (contact.islandIndex != islandIndex) {
                continue;
            }
            if (constraintIndexCursor <
                dispatch.constraintCapacity) {
                tileConstraintIndices[
                    constraintBase + constraintIndexCursor
                ] = localConstraint;
            }
            ++constraintIndexCursor;
            hasArticulation =
                hasArticulation ||
                bodies[bodyBase + contact.bodyA]
                        .flagsAndIndices[1] ==
                    dispatch.articulationIndex ||
                bodies[bodyBase + contact.bodyB]
                        .flagsAndIndices[1] ==
                    dispatch.articulationIndex;
        }
        const uint packedCount =
            constraintIndexCursor - islandIndexStart;
        if (packedCount != island.constraintCount) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint =
                island.firstConstraint;
            statuses[environment] = status;
            return;
        }
        const uint tileCount =
            (packedCount + MR_WAVE32_CONTACTS_PER_TILE - 1u) /
            MR_WAVE32_CONTACTS_PER_TILE;
        MRIslandWorkGPU work = {};
        work.environment = environment;
        work.islandIndex = islandIndex;
        work.firstConstraint = island.firstConstraint;
        work.constraintCount = packedCount;
        work.firstTile = tileCursor;
        work.tileCount = tileCount;
        work.dofClass =
            dispatch.nv <= 8u ? 8u :
            dispatch.nv <= 16u ? 16u : 32u;
        work.flags = MR_ISLAND_WORK_VALID |
            (hasArticulation
                 ? MR_ISLAND_WORK_HAS_ARTICULATION
                 : 0u) |
            (packedCount > MR_WAVE32_CONTACTS_PER_TILE
                 ? MR_ISLAND_WORK_SPILL
                 : 0u) |
            (packedCount > 256u
                 ? MR_ISLAND_WORK_DISTRIBUTED
                 : 0u);
        if (packedCount > 256u) {
            status.queueFlags |=
                MR_ISLAND_WORK_DISTRIBUTED;
        }
        islandWork[islandBase + islandIndex] = work;
        islandWorkFlags[islandBase + islandIndex] = 1u;

        for (uint localTile = 0u;
             localTile < tileCount;
             ++localTile) {
            const uint tileConstraintOffset =
                islandIndexStart +
                localTile * MR_WAVE32_CONTACTS_PER_TILE;
            const uint tileConstraintCount = min(
                MR_WAVE32_CONTACTS_PER_TILE,
                packedCount -
                    localTile * MR_WAVE32_CONTACTS_PER_TILE
            );
            if (tileCursor < dispatch.solverTileCapacity) {
                MRContactTileGPU tile = {};
                tile.environment = environment;
                tile.islandIndex = islandIndex;
                tile.firstConstraint =
                    tileConstraintCount == 0u
                    ? MR_INVALID_INDEX
                    : tileConstraintIndices[
                          constraintBase +
                          tileConstraintOffset
                      ];
                tile.constraintCount = tileConstraintCount;
                tile.nextTile =
                    localTile + 1u < tileCount
                    ? tileCursor + 1u
                    : MR_INVALID_INDEX;
                tile.partialOffset = tileConstraintOffset;
                tile.flags = MR_ISLAND_WORK_VALID;
                tiles[tileBase + tileCursor] = tile;
            }
            ++tileCursor;
        }
        if (packedCount > MR_WAVE32_CONTACTS_PER_TILE) {
            requiredSpillRows +=
                static_cast<ulong>(
                    packedCount -
                    MR_WAVE32_CONTACTS_PER_TILE
                ) * 3u;
        }
    }
    status.requiredSolverTiles = tileCursor;
    status.solverTiles = min(
        tileCursor,
        dispatch.solverTileCapacity
    );
    status.requiredSpillRows = static_cast<uint>(
        min(
            requiredSpillRows,
            static_cast<ulong>(0xffffffffu)
        )
    );
    status.spillRows = min(
        status.requiredSpillRows,
        dispatch.spillRowCapacity
    );
    if (constraintIndexCursor != status.requiredConstraints ||
        tileCursor > dispatch.solverTileCapacity ||
        requiredSpillRows > dispatch.spillRowCapacity) {
        status.code = MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
        status.firstFailingStableKeyLow =
            status.islandCount == 0u
            ? MR_INVALID_INDEX
            : status.islandCount - 1u;
        status.firstFailingStableKeyHigh = environment;
    }
    statuses[environment] = status;
}

kernel void mr_world_scatter_island_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRIslandWorkGPU* denseWork [[buffer(3)]],
    device MRIslandWorkGPU* compactWork [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass = MR_WORLD_WORK_SOLVER;
        header.overflow = count > total ? 1u : 0u;
        // A following single-SIMDgroup reduction selects a homogeneous
        // 8/16-lane packet width without making any count host-visible.
        header.reserved0 = MR_WAVE32_CONTACTS_PER_TILE;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactWork[destination] = denseWork[globalIndex];
    }
}

kernel void mr_world_select_solver_cohort(
    device const MRIslandWorkGPU* compactWork [[buffer(0)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(1)]],
    device MRWaveWorkPacketGPU* packets [[buffer(2)]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    device MRWorkQueueHeaderGPU& header =
        headers[MR_WORLD_WORK_SOLVER];
    const uint count = header.count;
    uint requiredWidth = 8u;
    for (uint workSlot = lane;
         workSlot < count;
         workSlot += MR_WAVE32_CONTACTS_PER_TILE) {
        const MRIslandWorkGPU work = compactWork[workSlot];
        const bool distributed =
            (work.flags & MR_ISLAND_WORK_DISTRIBUTED) != 0u;
        const uint width =
            !distributed &&
            work.dofClass <= 8u &&
            work.constraintCount <= 8u
            ? 8u
            : !distributed &&
              work.dofClass <= 16u &&
              work.constraintCount <= 16u
            ? 16u
            : MR_WAVE32_CONTACTS_PER_TILE;
        requiredWidth = max(requiredWidth, width);
    }
    requiredWidth = simd_max(requiredWidth);
    if (lane == 0u) {
        const uint cohortWidth =
            count != 0u &&
            requiredWidth <= 8u &&
            (count & 3u) == 0u
            ? 8u
            : count != 0u &&
              requiredWidth <= 16u &&
              (count & 1u) == 0u
            ? 16u
            : MR_WAVE32_CONTACTS_PER_TILE;
        const uint cohortsPerGroup =
            MR_WAVE32_CONTACTS_PER_TILE / cohortWidth;
        header.reserved0 = cohortWidth;
        header.indirect.threadgroupsX =
            count / cohortsPerGroup;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Freeze the scan-ordered compact island stream into immutable packets
    // in the same one-SIMDgroup pass that chooses the homogeneous cohort.
    // Worker atomics may claim packets in any order without changing keys.
    const uint cohortWidth = header.reserved0;
    const uint cohortsPerGroup =
        MR_WAVE32_CONTACTS_PER_TILE / cohortWidth;
    for (uint packetSlot = lane;
         packetSlot < header.indirect.threadgroupsX;
         packetSlot += MR_WAVE32_CONTACTS_PER_TILE) {
        const uint firstWork = packetSlot * cohortsPerGroup;
        const uint validCohorts = min(
            cohortsPerGroup,
            header.count - firstWork
        );
        MRWaveWorkPacketGPU packet = {};
        packet.islandSlots = uint4(MR_INVALID_INDEX);
        packet.stableKeyLow = uint4(MR_INVALID_INDEX);
        packet.stableKeyHigh = uint4(MR_INVALID_INDEX);
        for (uint cohort = 0u;
             cohort < validCohorts;
             ++cohort) {
            const uint workSlot = firstWork + cohort;
            const MRIslandWorkGPU work =
                compactWork[workSlot];
            packet.islandSlots[cohort] = workSlot;
            packet.stableKeyLow[cohort] =
                work.islandIndex;
            packet.stableKeyHigh[cohort] =
                work.environment;
        }
        packet.metadata = uint4(
            cohortWidth,
            validCohorts,
            0u,
            0u
        );
        packets[packetSlot] = packet;
    }
}

kernel void mr_world_flag_distributed_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRIslandWorkGPU* denseWork [[buffer(1)]],
    device uint* flags [[buffer(2)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex >= total) {
        return;
    }
    const MRIslandWorkGPU work = denseWork[globalIndex];
    flags[globalIndex] =
        (work.flags &
         (
             MR_ISLAND_WORK_VALID |
             MR_ISLAND_WORK_DISTRIBUTED
         )) ==
            (
                MR_ISLAND_WORK_VALID |
                MR_ISLAND_WORK_DISTRIBUTED
            )
        ? 1u
        : 0u;
}

kernel void mr_world_scatter_distributed_island_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRIslandWorkGPU* denseWork [[buffer(3)]],
    device MRIslandWorkGPU* compactWork [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER_DISTRIBUTED];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass =
            MR_WORLD_WORK_SOLVER_DISTRIBUTED;
        header.overflow = count > total ? 1u : 0u;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
        header.reserved0 = total;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactWork[total + destination] =
            denseWork[globalIndex];
    }
}

kernel void mr_world_flag_distributed_tiles(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(1)]],
    device const MRIslandWorkGPU* denseWork [[buffer(2)]],
    device const MRContactTileGPU* denseTiles [[buffer(3)]],
    device uint* flags [[buffer(4)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount *
        dispatch.solverTileCapacity;
    if (globalIndex >= total) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.solverTileCapacity;
    const uint localTile =
        globalIndex -
        environment * dispatch.solverTileCapacity;
    const MRMetalWorldContactStatusGPU status =
        statuses[environment];
    if (status.code != MR_STEP_SUCCESS ||
        localTile >= status.solverTiles) {
        flags[globalIndex] = 0u;
        return;
    }
    const MRContactTileGPU tile = denseTiles[globalIndex];
    if ((tile.flags & MR_ISLAND_WORK_VALID) == 0u ||
        tile.environment != environment ||
        tile.islandIndex >= status.islandCount) {
        flags[globalIndex] = 0u;
        return;
    }
    const MRIslandWorkGPU work =
        denseWork[
            environment * dispatch.islandCapacity +
            tile.islandIndex
        ];
    flags[globalIndex] =
        (work.flags &
         MR_ISLAND_WORK_DISTRIBUTED) != 0u
        ? 1u
        : 0u;
}

kernel void mr_world_scatter_distributed_tile_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRContactTileGPU* denseTiles [[buffer(3)]],
    device MRContactTileGPU* compactTiles [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount *
        dispatch.solverTileCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER_SPILL];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass = MR_WORLD_WORK_SOLVER_SPILL;
        header.overflow = count > total ? 1u : 0u;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
        header.reserved0 = total;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactTiles[total + destination] =
            denseTiles[globalIndex];
    }
}

// Distributed islands use one SIMDgroup per compacted 32-contact tile for
// the expensive factor solves and 3x3 block construction. Per-contact output
// slots are unique; a later single-group segmented reducer applies them in
// canonical tile/contact order, so no atomic velocity accumulation is used.
kernel void mr_world_wave32_distributed_prepare(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(5)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(6)]],
    device float* responseColumns [[buffer(7)]],
    device float4* impulseDeltas [[buffer(8)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(9)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(10)]],
    device const MRIslandWorkGPU* denseIslandWork [[buffer(11)]],
    device const MRContactTileGPU* compactTiles [[buffer(12)]],
    device const uint* tileConstraintIndices [[buffer(13)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(14)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_SPILL];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRContactTileGPU tile =
            compactTiles[header.reserved0 + queueSlot];
        const uint environment = tile.environment;
        if (environment >= dispatch.environmentCount ||
            lane >= tile.constraintCount) {
            continue;
        }
        const MRMetalWorldContactStatusGPU environmentStatus =
            statuses[environment];
        if (environmentStatus.code != MR_STEP_SUCCESS ||
            tile.islandIndex >= environmentStatus.islandCount) {
            continue;
        }
        const MRIslandWorkGPU work =
            denseIslandWork[
                environment * dispatch.islandCapacity +
                tile.islandIndex
            ];
        if ((work.flags &
             (
                 MR_ISLAND_WORK_VALID |
                 MR_ISLAND_WORK_DISTRIBUTED
             )) !=
            (
                MR_ISLAND_WORK_VALID |
                MR_ISLAND_WORK_DISTRIBUTED
            )) {
            continue;
        }

        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint factorBase =
            environment * dispatch.factorStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint responseBase =
            environment *
            (dispatch.constraintStride * 3u * dispatch.nv);
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + lane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MRBodyStateGPU* bodies =
            candidateBodies + bodyBase;
        const bool articulatedA =
            bodies[contact.bodyA].flagsAndIndices[1] ==
                dispatch.articulationIndex;
        const bool articulatedB =
            bodies[contact.bodyB].flagsAndIndices[1] ==
                dispatch.articulationIndex;

        uint failure = MR_STEP_SUCCESS;
        float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
        float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
        float solution[MR_ARTICULATED_ABA_MAX_DOFS];
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 direction =
                evaluatedRows[
                    rowBase + 3u * localConstraint + axis
                ].direction.xyz;
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                rightHandSide[dof] = dot(
                    direction,
                    combinedJacobianColumn(
                        pointJacobians,
                        pointJacobianBase,
                        localConstraint,
                        dof,
                        dispatch.nv,
                        articulatedA,
                        articulatedB
                    )
                );
                intermediate[dof] = 0.0f;
                solution[dof] = 0.0f;
            }
            if (!solveCholesky(
                    factors,
                    factorBase,
                    dispatch.nv,
                    rightHandSide,
                    intermediate,
                    solution
                )) {
                failure = MR_STEP_FACTORIZATION_FAILED;
                break;
            }
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                responseColumns[
                    responseBase +
                    (localConstraint * 3u + axis) *
                        dispatch.nv +
                    dof
                ] = solution[dof];
            }
        }

        MRWave32PreconditionerGPU preconditioner = {};
        if (failure == MR_STEP_SUCCESS) {
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 directions[3] = {
                localRows[0].direction.xyz,
                localRows[1].direction.xyz,
                localRows[2].direction.xyz,
            };
            float effective[3][3];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    float value = 0.0f;
                    for (uint dof = 0u;
                         dof < dispatch.nv;
                         ++dof) {
                        const float rowJacobian = dot(
                            directions[row],
                            combinedJacobianColumn(
                                pointJacobians,
                                pointJacobianBase,
                                localConstraint,
                                dof,
                                dispatch.nv,
                                articulatedA,
                                articulatedB
                            )
                        );
                        value += rowJacobian *
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    column
                                ) * dispatch.nv +
                                dof
                            ];
                    }
                    if (dynamicSceneEndpoint(
                            bodies[contact.bodyA],
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            -scenePointResponse(
                                bodies[contact.bodyA],
                                contact.pointAndSeparation.xyz,
                                -directions[column]
                            )
                        );
                    }
                    if (dynamicSceneEndpoint(
                            bodies[contact.bodyB],
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            scenePointResponse(
                                bodies[contact.bodyB],
                                contact.pointAndSeparation.xyz,
                                directions[column]
                            )
                        );
                    }
                    if (row == column) {
                        value +=
                            localRows[row].regularization;
                    }
                    effective[row][column] = value;
                }
            }
            float inverse[3][3];
            if (!invert3x3(effective, inverse)) {
                failure = MR_STEP_FACTORIZATION_FAILED;
            } else {
                preconditioner.row0.xyz = float3(
                    inverse[0][0],
                    inverse[0][1],
                    inverse[0][2]
                );
                preconditioner.row1.xyz = float3(
                    inverse[1][0],
                    inverse[1][1],
                    inverse[1][2]
                );
                preconditioner.row2.xyz = float3(
                    inverse[2][0],
                    inverse[2][1],
                    inverse[2][2]
                );
            }
        }
        preconditioner.row0.w =
            static_cast<float>(failure);
        preconditioner.row1.w =
            static_cast<float>(localConstraint);
        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        if (failure == MR_STEP_SUCCESS) {
            const float3 warm = projectFrictionCone(
                contact.impulses.xyz,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            contact.impulses.xyz = warm;
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(warm, 0.0f);
        } else {
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(
                0.0f,
                0.0f,
                0.0f,
                static_cast<float>(failure)
            );
        }
    }
}

// All contacts in a distributed island evaluate their block-Jacobi cone
// update against the same accepted island velocity. Each compact tile is
// independent until the canonical segmented reducer applies these deltas.
kernel void mr_world_wave32_distributed_delta(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    device const float* candidateV [[buffer(2)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(5)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(6)]],
    device const MRWave32PreconditionerGPU* preconditioners [[buffer(7)]],
    device float4* impulseDeltas [[buffer(8)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    device const MRContactTileGPU* compactTiles [[buffer(10)]],
    device const uint* tileConstraintIndices [[buffer(11)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(12)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_SPILL];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRContactTileGPU tile =
            compactTiles[header.reserved0 + queueSlot];
        const uint environment = tile.environment;
        if (environment >= dispatch.environmentCount ||
            lane >= tile.constraintCount ||
            statuses[environment].code != MR_STEP_SUCCESS) {
            continue;
        }
        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + lane
            ];
        const MRWave32PreconditionerGPU preconditioner =
            preconditioners[
                constraintBase + localConstraint
            ];
        if (static_cast<uint>(
                preconditioner.row0.w
            ) != MR_STEP_SUCCESS) {
            continue;
        }
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            candidateBodies + bodyBase,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            candidateV + environment * dispatch.nv
        );
        const float3 previous = contact.impulses.xyz;
        const float3 rhs = float3(
            localRows[0].targetVelocity -
                dot(localRows[0].direction.xyz, relative) -
                localRows[0].regularization * previous.x,
            localRows[1].targetVelocity -
                dot(localRows[1].direction.xyz, relative) -
                localRows[1].regularization * previous.y,
            localRows[2].targetVelocity -
                dot(localRows[2].direction.xyz, relative) -
                localRows[2].regularization * previous.z
        );
        float3 candidate = previous + float3(
            dot(preconditioner.row0.xyz, rhs),
            dot(preconditioner.row1.xyz, rhs),
            dot(preconditioner.row2.xyz, rhs)
        );
        candidate = projectFrictionCone(
            candidate,
            evaluatedCones[
                constraintBase + localConstraint
            ]
        );
        const float3 delta = candidate - previous;
        if (!finite3(candidate) || !finite3(delta)) {
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(
                0.0f,
                0.0f,
                0.0f,
                static_cast<float>(MR_STEP_NONFINITE_RESULT)
            );
            continue;
        }
        contact.impulses.xyz = candidate;
        impulseDeltas[
            constraintBase + localConstraint
        ] = float4(delta, 0.0f);
    }
}

// One SIMDgroup deterministically reduces per-contact tile output across a
// distributed island. Lane ownership of articulation DoFs and scene bodies
// prevents write conflicts, and stable tile order makes the reduction
// bitwise replayable.
kernel void mr_world_wave32_distributed_reduce(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    device float* candidateV [[buffer(2)]],
    device MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(5)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(6)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(7)]],
    device const float* responseColumns [[buffer(8)]],
    device const float4* impulseDeltas [[buffer(9)]],
    device const MRWave32PreconditionerGPU* preconditioners [[buffer(10)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(11)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(12)]],
    device const MRIslandWorkGPU* compactIslandWork [[buffer(13)]],
    device const MRContactTileGPU* denseTiles [[buffer(14)]],
    device const uint* tileConstraintIndices [[buffer(15)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(16)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(17)]],
    constant MRMetalWorldPassGPU& pass [[buffer(18)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_DISTRIBUTED];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRIslandWorkGPU work =
            compactIslandWork[header.reserved0 + queueSlot];
        const uint environment = work.environment;
        const uint islandIndex = work.islandIndex;
        if (environment >= dispatch.environmentCount ||
            statuses[environment].code != MR_STEP_SUCCESS ||
            (work.flags &
             MR_ISLAND_WORK_DISTRIBUTED) == 0u) {
            continue;
        }
        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint responseBase =
            environment *
            (dispatch.constraintStride * 3u * dispatch.nv);
        const uint manifoldPointBase =
            environment * dispatch.manifoldStride *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
        const uint tileBase =
            environment * dispatch.solverTileCapacity;
        device float* articulationVelocity =
            candidateV + environment * dispatch.nv;
        device MRBodyStateGPU* bodies =
            candidateBodies + bodyBase;

        uint laneFailure = MR_STEP_SUCCESS;
        uint laneFailureConstraint = MR_INVALID_INDEX;
        float laneMaximumDelta = 0.0f;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                denseTiles[
                    tileBase + work.firstTile + localTile
                ];
            if (lane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset + lane
                ];
            const uint prepareFailure = static_cast<uint>(
                preconditioners[
                    constraintBase + localConstraint
                ].row0.w
            );
            const uint deltaFailure = static_cast<uint>(
                impulseDeltas[
                    constraintBase + localConstraint
                ].w
            );
            const uint failure = max(
                prepareFailure,
                deltaFailure
            );
            if (failure != MR_STEP_SUCCESS) {
                laneFailure = max(laneFailure, failure);
                laneFailureConstraint = min(
                    laneFailureConstraint,
                    localConstraint
                );
            }
            const float3 delta =
                impulseDeltas[
                    constraintBase + localConstraint
                ].xyz;
            laneMaximumDelta = max(
                laneMaximumDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
        }
        const uint maximumFailure = simd_max(laneFailure);
        const uint firstFailureConstraint =
            simd_min(laneFailureConstraint);
        if (maximumFailure != MR_STEP_SUCCESS) {
            if (lane == 0u) {
                MRWave32IslandStatusGPU failed = {};
                failed.code = maximumFailure;
                failed.environment = environment;
                failed.islandIndex = islandIndex;
                failed.residuals.w = static_cast<float>(
                    firstFailureConstraint
                );
                waveStatuses[
                    environment * dispatch.islandStride +
                    islandIndex
                ] = failed;
            }
            continue;
        }

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u &&
            lane < dispatch.nv) {
            float velocityDelta = 0.0f;
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    denseTiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    velocityDelta +=
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 0u) *
                                dispatch.nv +
                            lane
                        ] * delta.x +
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 1u) *
                                dispatch.nv +
                            lane
                        ] * delta.y +
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 2u) *
                                dispatch.nv +
                            lane
                        ] * delta.z;
                }
            }
            articulationVelocity[lane] += velocityDelta;
        }
        for (uint bodyIndex = lane;
             bodyIndex < dispatch.bodyCount;
             bodyIndex += MR_WAVE32_CONTACTS_PER_TILE) {
            device MRBodyStateGPU& body = bodies[bodyIndex];
            if (!dynamicSceneEndpoint(
                    body,
                    dispatch.articulationIndex
                )) {
                continue;
            }
            float3 linearImpulse = float3(0.0f);
            float3 angularImpulse = float3(0.0f);
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    denseTiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const MRContactConstraintGPU contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 impulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    if (contact.bodyA == bodyIndex) {
                        linearImpulse -= impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            -impulse
                        );
                    }
                    if (contact.bodyB == bodyIndex) {
                        linearImpulse += impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            impulse
                        );
                    }
                }
            }
            body.linearVelocityAndInverseMass.xyz +=
                body.linearVelocityAndInverseMass.w *
                linearImpulse;
            body.angularVelocity.xyz += multiply(
                stateInverseInertia(body),
                angularImpulse
            );
        }
        threadgroup_barrier(mem_flags::mem_device);

        // reserved1==2 marks the last distributed sweep for this
        // microstep. Only then publish manifold impulses and residuals.
        if (pass.reserved1 != 2u) {
            continue;
        }
        float laneNormalResidual = 0.0f;
        float laneConeViolation = 0.0f;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                denseTiles[
                    tileBase + work.firstTile + localTile
                ];
            if (lane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset + lane
                ];
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity
            );
            const float normalEquation =
                dot(
                    localRows[0].direction.xyz,
                    relative
                ) -
                localRows[0].targetVelocity +
                localRows[0].regularization *
                    contact.impulses.x;
            laneNormalResidual = max(
                laneNormalResidual,
                contact.impulses.x > kConeEpsilon
                ? abs(normalEquation)
                : max(-normalEquation, 0.0f)
            );
            const MREvaluatedConstraintIRConeGPU cone =
                evaluatedCones[
                    constraintBase + localConstraint
                ];
            const float limitU =
                cone.effectiveFrictionU *
                contact.impulses.x;
            const float limitV =
                cone.effectiveFrictionV *
                contact.impulses.x;
            float coneViolation = 0.0f;
            if (limitU > 0.0f && limitV > 0.0f) {
                coneViolation = max(
                    sqrt(
                        (
                            contact.impulses.y *
                            contact.impulses.y
                        ) / (limitU * limitU) +
                        (
                            contact.impulses.z *
                            contact.impulses.z
                        ) / (limitV * limitV)
                    ) - 1.0f,
                    0.0f
                );
            } else {
                coneViolation =
                    length(contact.impulses.yz);
            }
            laneConeViolation = max(
                laneConeViolation,
                coneViolation
            );
            const MRContactPointMetaGPU metadata =
                contactMetadata[
                    constraintBase + localConstraint
                ];
            if (metadata.manifoldIndex <
                    dispatch.manifoldCapacity &&
                metadata.pointIndex <
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
                candidateManifoldPoints[
                    manifoldPointBase +
                    metadata.manifoldIndex *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    metadata.pointIndex
                ].impulses = contact.impulses;
            }
        }
        const float maximumImpulseDelta =
            simd_max(laneMaximumDelta);
        const float maximumNormalResidual =
            simd_max(laneNormalResidual);
        const float maximumConeViolation =
            simd_max(laneConeViolation);
        if (lane == 0u) {
            MRWave32IslandStatusGPU result = {};
            result.code = MR_STEP_SUCCESS;
            result.environment = environment;
            result.islandIndex = islandIndex;
            result.iterations = pass.reserved0;
            result.residuals = float4(
                maximumImpulseDelta,
                maximumNormalResidual,
                maximumConeViolation,
                maximumNormalResidual > 1.0e-3f
                    ? 1.0f
                    : 0.0f
            );
            waveStatuses[
                environment * dispatch.islandStride +
                islandIndex
            ] = result;
        }
    }
}

float waveCohortMaximum(
    float value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = max(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

uint waveCohortMaximum(
    uint value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = max(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

uint waveCohortMinimum(
    uint value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = min(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

// Shared packet body. Standalone Metal obtains packet slots through indirect
// dispatch while MLX uses a fixed worker grid that repeatedly claims slots
// from the invocation-local queue cursor.
inline void mrWorldWave32SolvePacket(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    device const float* factors,
    device const float* pointJacobians,
    device float* candidateV,
    device MRBodyStateGPU* candidateBodies,
    device MRContactConstraintGPU* contacts,
    device const MRContactPointMetaGPU* contactMetadata,
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows,
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones,
    device float* responseColumns,
    device MRManifoldPointGPU* candidateManifoldPoints,
    device const MRMetalWorldContactStatusGPU* statuses,
    device const MRIslandWorkGPU* islandWork,
    device const MRContactTileGPU* tiles,
    device const uint* tileConstraintIndices,
    device float4* impulseDeltas,
    device MRWave32PreconditionerGPU* preconditioners,
    device MRWave32IslandStatusGPU* waveStatuses,
    device const MRWorkQueueHeaderGPU* workHeaders,
    device const MRWaveWorkPacketGPU* workPackets,
    constant MRMetalWorldPassGPU& pass,
    const uint packetSlot,
    const uint lane,
    threadgroup uint* failureCodes,
    threadgroup uint* failureConstraints,
    threadgroup uint* sharedFailure
) {
    const MRWorkQueueHeaderGPU workHeader =
        workHeaders[MR_WORLD_WORK_SOLVER];
    const MRWaveWorkPacketGPU packet =
        workPackets[packetSlot];
    const uint cohortWidth =
        packet.metadata.x == 8u ||
        packet.metadata.x == 16u
        ? packet.metadata.x
        : MR_WAVE32_CONTACTS_PER_TILE;
    const uint cohortIndex = lane / cohortWidth;
    const uint localLane =
        lane - cohortIndex * cohortWidth;
    if (cohortIndex >= packet.metadata.y) {
        return;
    }
    const uint workSlot = packet.islandSlots[cohortIndex];
    if (workSlot >= workHeader.count) {
        return;
    }
    const MRIslandWorkGPU work = islandWork[workSlot];
    const uint environment = work.environment;
    const uint islandIndex = work.islandIndex;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    const MRMetalWorldContactStatusGPU environmentStatus =
        statuses[environment];
    if (environmentStatus.code != MR_STEP_SUCCESS ||
        islandIndex >= environmentStatus.islandCount) {
        return;
    }
    if ((work.flags & MR_ISLAND_WORK_VALID) == 0u ||
        work.environment != environment ||
        work.islandIndex != islandIndex ||
        packet.stableKeyLow[cohortIndex] != islandIndex ||
        packet.stableKeyHigh[cohortIndex] != environment) {
        return;
    }
    if ((work.flags & MR_ISLAND_WORK_DISTRIBUTED) != 0u) {
        return;
    }

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const uint responseBase =
        environment *
        (dispatch.constraintStride * 3u * dispatch.nv);
    const uint manifoldPointBase =
        environment * dispatch.manifoldStride *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint tileBase =
        environment * dispatch.solverTileCapacity;
    device float* articulationVelocity =
        candidateV + velocityBase;
    device MRBodyStateGPU* bodies =
        candidateBodies + bodyBase;

    uint localFailure = MR_STEP_SUCCESS;
    uint localFailureConstraint = MR_INVALID_INDEX;
    float laneMaximumDelta = 0.0f;
    if (localLane == 0u) {
        sharedFailure[cohortIndex] = MR_STEP_SUCCESS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD-saturated response/preconditioner construction and warm-start
    // projection. Each lane solves its own three factorized RHS values.
    for (uint localTile = 0u;
         localTile < work.tileCount;
         ++localTile) {
        const MRContactTileGPU tile =
            tiles[tileBase + work.firstTile + localTile];
        if (localLane >= tile.constraintCount) {
            continue;
        }
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + localLane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        const bool articulatedA =
            bodies[contact.bodyA].flagsAndIndices[1] ==
                dispatch.articulationIndex;
        const bool articulatedB =
            bodies[contact.bodyB].flagsAndIndices[1] ==
                dispatch.articulationIndex;
        float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
        float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
        float solution[MR_ARTICULATED_ABA_MAX_DOFS];
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 direction =
                evaluatedRows[
                    rowBase + 3u * localConstraint + axis
                ].direction.xyz;
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                rightHandSide[dof] = dot(
                    direction,
                    combinedJacobianColumn(
                        pointJacobians,
                        pointJacobianBase,
                        localConstraint,
                        dof,
                        dispatch.nv,
                        articulatedA,
                        articulatedB
                    )
                );
                intermediate[dof] = 0.0f;
                solution[dof] = 0.0f;
            }
            if (!solveCholesky(
                    factors,
                    factorBase,
                    dispatch.nv,
                    rightHandSide,
                    intermediate,
                    solution
                )) {
                localFailure = MR_STEP_FACTORIZATION_FAILED;
                localFailureConstraint = min(
                    localFailureConstraint,
                    localConstraint
                );
                break;
            }
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                responseColumns[
                    responseBase +
                    (localConstraint * 3u + axis) *
                        dispatch.nv +
                    dof
                ] = solution[dof];
            }
        }
        if (localFailure != MR_STEP_SUCCESS) {
            continue;
        }

        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 directions[3] = {
            localRows[0].direction.xyz,
            localRows[1].direction.xyz,
            localRows[2].direction.xyz,
        };
        float effective[3][3];
        for (uint row = 0u; row < 3u; ++row) {
            for (uint column = 0u; column < 3u; ++column) {
                float value = 0.0f;
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    const float rowJacobian = dot(
                        directions[row],
                        combinedJacobianColumn(
                            pointJacobians,
                            pointJacobianBase,
                            localConstraint,
                            dof,
                            dispatch.nv,
                            articulatedA,
                            articulatedB
                        )
                    );
                    value += rowJacobian *
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + column) *
                                dispatch.nv +
                            dof
                        ];
                }
                if (dynamicSceneEndpoint(
                        bodies[contact.bodyA],
                        dispatch.articulationIndex
                    )) {
                    value += dot(
                        directions[row],
                        -scenePointResponse(
                            bodies[contact.bodyA],
                            contact.pointAndSeparation.xyz,
                            -directions[column]
                        )
                    );
                }
                if (dynamicSceneEndpoint(
                        bodies[contact.bodyB],
                        dispatch.articulationIndex
                    )) {
                    value += dot(
                        directions[row],
                        scenePointResponse(
                            bodies[contact.bodyB],
                            contact.pointAndSeparation.xyz,
                            directions[column]
                        )
                    );
                }
                if (row == column) {
                    value += localRows[row].regularization;
                }
                effective[row][column] = value;
            }
        }
        float inverse[3][3];
        if (!invert3x3(effective, inverse)) {
            localFailure = MR_STEP_FACTORIZATION_FAILED;
            localFailureConstraint = min(
                localFailureConstraint,
                localConstraint
            );
            continue;
        }
        MRWave32PreconditionerGPU preconditioner = {};
        preconditioner.row0.xyz = float3(
            inverse[0][0],
            inverse[0][1],
            inverse[0][2]
        );
        preconditioner.row1.xyz = float3(
            inverse[1][0],
            inverse[1][1],
            inverse[1][2]
        );
        preconditioner.row2.xyz = float3(
            inverse[2][0],
            inverse[2][1],
            inverse[2][2]
        );
        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        const float3 warm = projectFrictionCone(
            contact.impulses.xyz,
            evaluatedCones[constraintBase + localConstraint]
        );
        contact.impulses.xyz = warm;
        impulseDeltas[
            constraintBase + localConstraint
        ] = float4(warm, 0.0f);
    }

    failureCodes[lane] = localFailure;
    failureConstraints[lane] = localFailureConstraint;
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
    if (localLane == 0u) {
        sharedFailure[cohortIndex] = MR_STEP_SUCCESS;
        uint firstConstraint = MR_INVALID_INDEX;
        const uint firstCohortLane =
            cohortIndex * cohortWidth;
        for (uint sourceLane = firstCohortLane;
             sourceLane < firstCohortLane + cohortWidth;
             ++sourceLane) {
            if (failureCodes[sourceLane] == MR_STEP_SUCCESS) {
                continue;
            }
            sharedFailure[cohortIndex] =
                failureCodes[sourceLane];
            firstConstraint = min(
                firstConstraint,
                failureConstraints[sourceLane]
            );
        }
        if (sharedFailure[cohortIndex] != MR_STEP_SUCCESS) {
            MRWave32IslandStatusGPU failed = {};
            failed.code = sharedFailure[cohortIndex];
            failed.environment = environment;
            failed.islandIndex = islandIndex;
            failed.residuals.w =
                static_cast<float>(firstConstraint);
            waveStatuses[islandBase + islandIndex] = failed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint groupFailure = simd_max(localFailure);
    if (groupFailure != MR_STEP_SUCCESS) {
        if (localLane == 0u &&
            sharedFailure[cohortIndex] == MR_STEP_SUCCESS) {
            // A sibling cohort failed its factor solve. Route this healthy
            // island through the deterministic ordered replay so every lane
            // exits the packet uniformly without sacrificing per-environment
            // transactional isolation.
            MRWave32IslandStatusGPU replay = {};
            replay.code = MR_STEP_SUCCESS;
            replay.environment = environment;
            replay.islandIndex = islandIndex;
            replay.residuals.w = 1.0f;
            waveStatuses[islandBase + islandIndex] = replay;
        }
        return;
    }

    // Apply every warm-start impulse exactly once. Articulation DoFs and free
    // bodies have unique lane owners, eliminating conflicting atomic writes.
    if ((work.flags & MR_ISLAND_WORK_HAS_ARTICULATION) != 0u &&
        localLane < dispatch.nv) {
        float velocityDelta = 0.0f;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            for (uint slot = 0u;
                 slot < tile.constraintCount;
                 ++slot) {
                const uint localConstraint =
                    tileConstraintIndices[
                        constraintBase +
                        tile.partialOffset + slot
                    ];
                const float3 delta =
                    impulseDeltas[
                        constraintBase + localConstraint
                    ].xyz;
                velocityDelta +=
                    responseColumns[
                        responseBase +
                            (localConstraint * 3u + 0u) *
                                dispatch.nv +
                            localLane
                    ] * delta.x +
                    responseColumns[
                        responseBase +
                            (localConstraint * 3u + 1u) *
                                dispatch.nv +
                            localLane
                    ] * delta.y +
                    responseColumns[
                        responseBase +
                            (localConstraint * 3u + 2u) *
                                dispatch.nv +
                            localLane
                    ] * delta.z;
            }
        }
        articulationVelocity[localLane] += velocityDelta;
    }
    for (uint bodyIndex = localLane;
         bodyIndex < dispatch.bodyCount;
         bodyIndex += cohortWidth) {
        device MRBodyStateGPU& body = bodies[bodyIndex];
        if (!dynamicSceneEndpoint(
                body,
                dispatch.articulationIndex
            )) {
            continue;
        }
        float3 linearImpulse = float3(0.0f);
        float3 angularImpulse = float3(0.0f);
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            for (uint slot = 0u;
                 slot < tile.constraintCount;
                 ++slot) {
                const uint localConstraint =
                    tileConstraintIndices[
                        constraintBase +
                        tile.partialOffset + slot
                    ];
                const MRContactConstraintGPU contact =
                    contacts[constraintBase + localConstraint];
                const float3 delta =
                    impulseDeltas[
                        constraintBase + localConstraint
                    ].xyz;
                device const MREvaluatedConstraintIRRowGPU*
                    localRows =
                        evaluatedRows +
                        rowBase + 3u * localConstraint;
                const float3 impulse =
                    localRows[0].direction.xyz * delta.x +
                    localRows[1].direction.xyz * delta.y +
                    localRows[2].direction.xyz * delta.z;
                if (contact.bodyA == bodyIndex) {
                    linearImpulse -= impulse;
                    angularImpulse += cross(
                        contact.pointAndSeparation.xyz -
                            body.position.xyz,
                        -impulse
                    );
                }
                if (contact.bodyB == bodyIndex) {
                    linearImpulse += impulse;
                    angularImpulse += cross(
                        contact.pointAndSeparation.xyz -
                            body.position.xyz,
                        impulse
                    );
                }
            }
        }
        body.linearVelocityAndInverseMass.xyz +=
            body.linearVelocityAndInverseMass.w *
            linearImpulse;
        body.angularVelocity.xyz += multiply(
            stateInverseInertia(body),
            angularImpulse
        );
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint solverIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u
             ? dispatch.finalVelocityIterations
             : 0u);
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            if (localLane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset +
                    localLane
                ];
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity
            );
            const float3 previous = contact.impulses.xyz;
            const float3 rhs = float3(
                localRows[0].targetVelocity -
                    dot(localRows[0].direction.xyz, relative) -
                    localRows[0].regularization * previous.x,
                localRows[1].targetVelocity -
                    dot(localRows[1].direction.xyz, relative) -
                    localRows[1].regularization * previous.y,
                localRows[2].targetVelocity -
                    dot(localRows[2].direction.xyz, relative) -
                    localRows[2].regularization * previous.z
            );
            const MRWave32PreconditionerGPU preconditioner =
                preconditioners[
                    constraintBase + localConstraint
                ];
            float3 candidate = previous + float3(
                dot(preconditioner.row0.xyz, rhs),
                dot(preconditioner.row1.xyz, rhs),
                dot(preconditioner.row2.xyz, rhs)
            );
            candidate = projectFrictionCone(
                candidate,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            const float3 delta = candidate - previous;
            if (!finite3(candidate) || !finite3(delta)) {
                localFailure = MR_STEP_NONFINITE_RESULT;
                localFailureConstraint = min(
                    localFailureConstraint,
                    localConstraint
                );
                impulseDeltas[
                    constraintBase + localConstraint
                ] = float4(0.0f);
                continue;
            }
            contact.impulses.xyz = candidate;
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(delta, 0.0f);
            laneMaximumDelta = max(
                laneMaximumDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
        }
        threadgroup_barrier(mem_flags::mem_device);

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u &&
            localLane < dispatch.nv) {
            float velocityDelta = 0.0f;
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    velocityDelta +=
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 0u) *
                                    dispatch.nv +
                                localLane
                        ] * delta.x +
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 1u) *
                                    dispatch.nv +
                                localLane
                        ] * delta.y +
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 2u) *
                                    dispatch.nv +
                                localLane
                        ] * delta.z;
                }
            }
            articulationVelocity[localLane] += velocityDelta;
        }
        for (uint bodyIndex = localLane;
             bodyIndex < dispatch.bodyCount;
             bodyIndex += cohortWidth) {
            device MRBodyStateGPU& body = bodies[bodyIndex];
            if (!dynamicSceneEndpoint(
                    body,
                    dispatch.articulationIndex
                )) {
                continue;
            }
            float3 linearImpulse = float3(0.0f);
            float3 angularImpulse = float3(0.0f);
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const MRContactConstraintGPU contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 impulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    if (contact.bodyA == bodyIndex) {
                        linearImpulse -= impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            -impulse
                        );
                    }
                    if (contact.bodyB == bodyIndex) {
                        linearImpulse += impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            impulse
                        );
                    }
                }
            }
            body.linearVelocityAndInverseMass.xyz +=
                body.linearVelocityAndInverseMass.w *
                linearImpulse;
            body.angularVelocity.xyz += multiply(
                stateInverseInertia(body),
                angularImpulse
            );
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float laneNormalResidual = 0.0f;
    float laneConeViolation = 0.0f;
    for (uint localTile = 0u;
         localTile < work.tileCount;
         ++localTile) {
        const MRContactTileGPU tile =
            tiles[tileBase + work.firstTile + localTile];
        if (localLane >= tile.constraintCount) {
            continue;
        }
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase +
                tile.partialOffset +
                localLane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            bodies,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            articulationVelocity
        );
        const float normalEquation =
            dot(localRows[0].direction.xyz, relative) -
            localRows[0].targetVelocity +
            localRows[0].regularization *
                contact.impulses.x;
        laneNormalResidual = max(
            laneNormalResidual,
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f)
        );
        const MREvaluatedConstraintIRConeGPU cone =
            evaluatedCones[
                constraintBase + localConstraint
            ];
        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;
        const float limitV =
            cone.effectiveFrictionV * contact.impulses.x;
        float coneViolation = 0.0f;
        if (limitU > 0.0f && limitV > 0.0f) {
            coneViolation = max(
                sqrt(
                    (contact.impulses.y * contact.impulses.y) /
                        (limitU * limitU) +
                    (contact.impulses.z * contact.impulses.z) /
                        (limitV * limitV)
                ) - 1.0f,
                0.0f
            );
        } else {
            coneViolation = length(contact.impulses.yz);
        }
        laneConeViolation = max(
            laneConeViolation,
            coneViolation
        );

        const MRContactPointMetaGPU metadata =
            contactMetadata[constraintBase + localConstraint];
        if (metadata.manifoldIndex <
                dispatch.manifoldCapacity &&
            metadata.pointIndex <
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
            candidateManifoldPoints[
                manifoldPointBase +
                metadata.manifoldIndex *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                metadata.pointIndex
            ].impulses = contact.impulses;
        }
    }
    const float maximumImpulseDelta = waveCohortMaximum(
        laneMaximumDelta,
        cohortWidth
    );
    const float maximumNormalResidual = waveCohortMaximum(
        laneNormalResidual,
        cohortWidth
    );
    const float maximumConeViolation = waveCohortMaximum(
        laneConeViolation,
        cohortWidth
    );
    const uint maximumFailureCode = waveCohortMaximum(
        localFailure,
        cohortWidth
    );
    const uint firstFailureConstraint =
        waveCohortMinimum(
            localFailureConstraint,
            cohortWidth
        );
    if (localLane == 0u) {
        MRWave32IslandStatusGPU result = {};
        result.code =
            maximumFailureCode;
        result.environment = environment;
        result.islandIndex = islandIndex;
        result.iterations = solverIterations;
        result.residuals = float4(
            maximumImpulseDelta,
            maximumNormalResidual,
            maximumConeViolation,
            maximumFailureCode != MR_STEP_SUCCESS
                ? static_cast<float>(firstFailureConstraint)
                : maximumNormalResidual > 1.0e-3f
                ? 1.0f
                : 0.0f
        );
        waveStatuses[islandBase + islandIndex] = result;
    }
}

// One SIMD32 group owns one full mixed-contact island or a deterministic
// packet of two/four homogeneous compact islands. Within a packet, contiguous
// 16/8-lane cohorts own independent islands. A cohort lane owns one coupled
// normal/tangent block in each tile, then cooperatively assembles a single
// articulation/free-body velocity update without atomics.
kernel void mr_world_wave32_solve(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device const MRIslandWorkGPU* islandWork [[buffer(12)]],
    device const MRContactTileGPU* tiles [[buffer(13)]],
    device const uint* tileConstraintIndices [[buffer(14)]],
    device float4* impulseDeltas [[buffer(15)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(16)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(17)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(18)]],
    constant MRMetalWorldPassGPU& pass [[buffer(19)]],
    device const MRWaveWorkPacketGPU* workPackets [[buffer(20)]],
    const uint packetSlot [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint failureCodes[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint failureConstraints[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint sharedFailure[4u];
    mrWorldWave32SolvePacket(
        dispatch,
        factors,
        pointJacobians,
        candidateV,
        candidateBodies,
        contacts,
        contactMetadata,
        evaluatedRows,
        evaluatedCones,
        responseColumns,
        candidateManifoldPoints,
        statuses,
        islandWork,
        tiles,
        tileConstraintIndices,
        impulseDeltas,
        preconditioners,
        waveStatuses,
        workHeaders,
        workPackets,
        pass,
        packetSlot,
        lane,
        failureCodes,
        failureConstraints,
        sharedFailure
    );
}

// MLX cannot issue an indirect dispatch through its active encoder. A bounded
// occupancy-sized grid therefore remains resident and claims deterministic
// packet slots through one relaxed atomic per packet. Packet writes are
// disjoint, so claim order cannot affect physical ordering or replay hashes.
kernel void mr_world_wave32_solve_persistent(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device const MRIslandWorkGPU* islandWork [[buffer(12)]],
    device const MRContactTileGPU* tiles [[buffer(13)]],
    device const uint* tileConstraintIndices [[buffer(14)]],
    device float4* impulseDeltas [[buffer(15)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(16)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(17)]],
    device MRWorkQueueHeaderGPU* workHeaders [[buffer(18)]],
    constant MRMetalWorldPassGPU& pass [[buffer(19)]],
    device const MRWaveWorkPacketGPU* workPackets [[buffer(20)]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint claimedPacket;
    threadgroup uint failureCodes[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint failureConstraints[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint sharedFailure[4u];
    device MRWorkQueueHeaderGPU& header =
        workHeaders[MR_WORLD_WORK_SOLVER];
    device atomic_uint* cursor =
        reinterpret_cast<device atomic_uint*>(
            &header.workerCursor
        );
    if (lane == 0u) {
        device atomic_uint* flags =
            reinterpret_cast<device atomic_uint*>(
                &header.flags
            );
        atomic_fetch_or_explicit(
            flags,
            static_cast<uint>(
                MR_WORLD_QUEUE_PERSISTENT_WORKER
            ),
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while (true) {
        if (lane == 0u) {
            claimedPacket = atomic_fetch_add_explicit(
                cursor,
                1u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint packetSlot = claimedPacket;
        if (packetSlot >= header.indirect.threadgroupsX) {
            break;
        }
        mrWorldWave32SolvePacket(
            dispatch,
            factors,
            pointJacobians,
            candidateV,
            candidateBodies,
            contacts,
            contactMetadata,
            evaluatedRows,
            evaluatedCones,
            responseColumns,
            candidateManifoldPoints,
            statuses,
            islandWork,
            tiles,
            tileConstraintIndices,
            impulseDeltas,
            preconditioners,
            waveStatuses,
            workHeaders,
            workPackets,
            pass,
            packetSlot,
            lane,
            failureCodes,
            failureConstraints,
            sharedFailure
        );
        threadgroup_barrier(
            mem_flags::mem_device |
            mem_flags::mem_threadgroup
        );
    }
}

// Environment-level reduction is deliberately separate from the island
// kernels: islands never contend on status publication and failed
// environments remain transactionally isolated.
kernel void mr_world_reduce_wave32_status(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRWave32IslandStatusGPU* waveStatuses [[buffer(1)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    float4 residuals = float4(0.0f);
    uint iterations = 0u;
    for (uint island = 0u;
         island < status.islandCount;
         ++island) {
        const MRWave32IslandStatusGPU source =
            waveStatuses[islandBase + island];
        if (source.code != MR_STEP_SUCCESS) {
            status.code = source.code;
            status.firstFailingConstraint =
                static_cast<uint>(source.residuals.w);
            status.firstFailingStableKeyLow = island;
            status.firstFailingStableKeyHigh = environment;
            break;
        }
        residuals = max(residuals, source.residuals);
        iterations = max(iterations, source.iterations);
    }
    status.solverIterations = iterations;
    status.residuals = residuals;
    const uint cohortWidth =
        workHeaders[MR_WORLD_WORK_SOLVER].reserved0;
    const MRWorkQueueHeaderGPU solverHeader =
        workHeaders[MR_WORLD_WORK_SOLVER];
    status.queueFlags |=
        cohortWidth == 8u
        ? MR_WORLD_QUEUE_COHORT_8
        : cohortWidth == 16u
        ? MR_WORLD_QUEUE_COHORT_16
        : 0u;
    if ((solverHeader.flags &
         MR_WORLD_QUEUE_PERSISTENT_WORKER) != 0u) {
        status.queueFlags |=
            MR_WORLD_QUEUE_PERSISTENT_WORKER;
        status.workerPackets =
            solverHeader.indirect.threadgroupsX;
        status.workerHighWater = max(
            status.workerHighWater,
            solverHeader.indirect.threadgroupsX
        );
        status.workerEmptyPulls =
            solverHeader.workerCursor >
                    solverHeader.indirect.threadgroupsX
            ? solverHeader.workerCursor -
                solverHeader.indirect.threadgroupsX
            : 0u;
    }
    if (residuals.w > 0.0f) {
        status.queueFlags |=
            MR_ISLAND_WORK_STIFF_REPLAY;
    }
    statuses[environment] = status;
}

// Exact-cone block PGS/TGS velocity solve for one articulation plus arbitrary
// free rigid bodies. The articulation Cholesky factor is reused for all three
// RHS of every contact; no dense inverse is formed.
kernel void mr_world_solve_contact_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    constant MRMetalWorldPassGPU& pass [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const bool stiffReplay = pass.reserved1 != 0u;
    if (stiffReplay &&
        (status.queueFlags &
         MR_ISLAND_WORK_STIFF_REPLAY) == 0u) {
        return;
    }
    if (status.requiredConstraints == 0u) {
        status.solverIterations = 0u;
        status.residuals = float4(0.0f);
        statuses[environment] = status;
        return;
    }
    if (dispatch.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        pass.reserved0 > 1u ||
        pass.reserved1 > 1u) {
        status.code = MR_STEP_UNSUPPORTED;
        statuses[environment] = status;
        return;
    }

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const uint responseBase =
        environment *
        (dispatch.constraintStride * 3u * dispatch.nv);
    const uint manifoldPointBase =
        environment * dispatch.manifoldStride *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;

    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    device float* articulationVelocity =
        candidateV + velocityBase;
    device MRBodyStateGPU* bodies =
        candidateBodies + bodyBase;

    // Cache M^-1 J^T for normal/u/v.
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device const MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        const bool articulatedA =
            bodies[contact.bodyA].flagsAndIndices[1] ==
                dispatch.articulationIndex;
        const bool articulatedB =
            bodies[contact.bodyB].flagsAndIndices[1] ==
                dispatch.articulationIndex;
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 direction =
                evaluatedRows[
                    rowBase + 3u * localConstraint + axis
                ].direction.xyz;
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                rightHandSide[dof] = dot(
                    direction,
                    combinedJacobianColumn(
                        pointJacobians,
                        pointJacobianBase,
                        localConstraint,
                        dof,
                        dispatch.nv,
                        articulatedA,
                        articulatedB
                    )
                );
                intermediate[dof] = 0.0f;
                solution[dof] = 0.0f;
            }
            if (!solveCholesky(
                    factors,
                    factorBase,
                    dispatch.nv,
                    rightHandSide,
                    intermediate,
                    solution
                )) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                responseColumns[
                    responseBase +
                    (localConstraint * 3u + axis) *
                        dispatch.nv +
                    dof
                ] = solution[dof];
            }
        }
    }

    // Initial ordered solve warm-starts from persistent manifolds. A stiff
    // replay starts from the velocity/impulse state already accepted by the
    // Wave32 sweep, so applying the full warm impulse again would double it.
    if (!stiffReplay) {
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            const float3 warm = projectFrictionCone(
                contact.impulses.xyz,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            contact.impulses.xyz = warm;
            applyContactDelta(
                localConstraint,
                contact,
                warm,
                evaluatedRows +
                    rowBase + 3u * localConstraint,
                articulationVelocity,
                bodies,
                responseColumns,
                responseBase,
                dispatch.nv,
                dispatch.articulationIndex
            );
        }
    }

    float maximumImpulseDelta = 0.0f;
    const uint solverIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u
             ? dispatch.finalVelocityIterations
             : 0u);
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            device const MREvaluatedConstraintIRRowGPU* localRows =
                evaluatedRows +
                rowBase + 3u * localConstraint;
            const float3 directions[3] = {
                localRows[0].direction.xyz,
                localRows[1].direction.xyz,
                localRows[2].direction.xyz,
            };
            device const MRBodyStateGPU& bodyA =
                bodies[contact.bodyA];
            device const MRBodyStateGPU& bodyB =
                bodies[contact.bodyB];
            const bool articulatedA =
                bodyA.flagsAndIndices[1] ==
                    dispatch.articulationIndex;
            const bool articulatedB =
                bodyB.flagsAndIndices[1] ==
                    dispatch.articulationIndex;

            float effective[3][3];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    float value = 0.0f;
                    for (uint dof = 0u;
                         dof < dispatch.nv;
                         ++dof) {
                        const float rowJacobian = dot(
                            directions[row],
                            combinedJacobianColumn(
                                pointJacobians,
                                pointJacobianBase,
                                localConstraint,
                                dof,
                                dispatch.nv,
                                articulatedA,
                                articulatedB
                            )
                        );
                        value +=
                            rowJacobian *
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    column
                                ) * dispatch.nv +
                                dof
                            ];
                    }
                    if (dynamicSceneEndpoint(
                            bodyA,
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            -scenePointResponse(
                                bodyA,
                                contact.pointAndSeparation.xyz,
                                -directions[column]
                            )
                        );
                    }
                    if (dynamicSceneEndpoint(
                            bodyB,
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            scenePointResponse(
                                bodyB,
                                contact.pointAndSeparation.xyz,
                                directions[column]
                            )
                        );
                    }
                    if (row == column) {
                        value += localRows[row].regularization;
                    }
                    effective[row][column] = value;
                }
            }
            float inverse[3][3];
            if (!invert3x3(effective, inverse)) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }

            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity
            );
            const float3 previous = contact.impulses.xyz;
            float rhs[3];
            for (uint row = 0u; row < 3u; ++row) {
                rhs[row] =
                    localRows[row].targetVelocity -
                    dot(directions[row], relative) -
                    localRows[row].regularization *
                        previous[row];
            }
            float3 candidate = previous;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    candidate[row] +=
                        inverse[row][column] * rhs[column];
                }
            }
            candidate = projectFrictionCone(
                candidate,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            const float3 delta = candidate - previous;
            if (!finite3(candidate) || !finite3(delta)) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }
            maximumImpulseDelta = max(
                maximumImpulseDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
            contact.impulses.xyz = candidate;
            applyContactDelta(
                localConstraint,
                contact,
                delta,
                localRows,
                articulationVelocity,
                bodies,
                responseColumns,
                responseBase,
                dispatch.nv,
                dispatch.articulationIndex
            );
        }
    }

    float maximumNormalResidual = 0.0f;
    float maximumConeViolation = 0.0f;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows +
            rowBase + 3u * localConstraint;
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            bodies,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            articulationVelocity
        );
        const float normalEquation =
            dot(localRows[0].direction.xyz, relative) -
            localRows[0].targetVelocity +
            localRows[0].regularization *
                contact.impulses.x;
        const float normalResidual =
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f);
        maximumNormalResidual = max(
            maximumNormalResidual,
            normalResidual
        );
        device const MREvaluatedConstraintIRConeGPU& cone =
            evaluatedCones[constraintBase + localConstraint];
        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;
        const float limitV =
            cone.effectiveFrictionV * contact.impulses.x;
        float coneViolation = 0.0f;
        if (limitU > 0.0f && limitV > 0.0f) {
            coneViolation = max(
                sqrt(
                    (contact.impulses.y *
                     contact.impulses.y) /
                        (limitU * limitU) +
                    (contact.impulses.z *
                     contact.impulses.z) /
                        (limitV * limitV)
                ) - 1.0f,
                0.0f
            );
        } else {
            coneViolation = length(contact.impulses.yz);
        }
        maximumConeViolation = max(
            maximumConeViolation,
            coneViolation
        );

        const MRContactPointMetaGPU metadata =
            contactMetadata[constraintBase + localConstraint];
        if (metadata.manifoldIndex <
                dispatch.manifoldCapacity &&
            metadata.pointIndex <
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
            candidateManifoldPoints[
                manifoldPointBase +
                metadata.manifoldIndex *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                metadata.pointIndex
            ].impulses = contact.impulses;
        }
    }
    if (!isfinite(maximumImpulseDelta) ||
        !isfinite(maximumNormalResidual) ||
        !isfinite(maximumConeViolation)) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[environment] = status;
        return;
    }
    status.solverIterations = solverIterations;
    status.residuals.x = maximumImpulseDelta;
    status.residuals.y = maximumNormalResidual;
    status.residuals.z = maximumConeViolation;
    if (stiffReplay) {
        status.queueFlags &=
            ~MR_ISLAND_WORK_STIFF_REPLAY;
    }
    statuses[environment] = status;
}

// Integrates only after constrained velocities are available.
kernel void mr_world_integrate_contact_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(3)]],
    device const uint* sceneBodyIndices [[buffer(4)]],
    device const float* sourceQ [[buffer(5)]],
    device const float* candidateV [[buffer(6)]],
    device float* candidateQ [[buffer(7)]],
    device MRBodyStateGPU* candidateBodies [[buffer(8)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
    const float timestep = dispatch.timestepAndBias.x;
    const uint qBase = environment * articulation.nq;
    const uint vBase = environment * articulation.nv;
    for (uint q = 0u; q < articulation.nq; ++q) {
        candidateQ[qBase + q] = sourceQ[qBase + q];
    }
    if (articulation.rootType == MR_ROOT_FLOATING) {
        candidateQ[qBase + 0u] =
            sourceQ[qBase + 0u] +
            timestep * candidateV[vBase + 0u];
        candidateQ[qBase + 1u] =
            sourceQ[qBase + 1u] +
            timestep * candidateV[vBase + 1u];
        candidateQ[qBase + 2u] =
            sourceQ[qBase + 2u] +
            timestep * candidateV[vBase + 2u];
        const float3 rotationVector =
            timestep * float3(
                candidateV[vBase + 3u],
                candidateV[vBase + 4u],
                candidateV[vBase + 5u]
            );
        const float angle = length(rotationVector);
        const float halfAngle = 0.5f * angle;
        const float scale = angle > 1.0e-6f
            ? sin(halfAngle) / angle
            : 0.5f - angle * angle / 48.0f;
        const float4 increment = float4(
            rotationVector * scale,
            cos(halfAngle)
        );
        float4 orientation;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    increment,
                    float4(
                        sourceQ[qBase + 3u],
                        sourceQ[qBase + 4u],
                        sourceQ[qBase + 5u],
                        sourceQ[qBase + 6u]
                    )
                ),
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = 3u;
            statuses[environment] = status;
            return;
        }
        candidateQ[qBase + 3u] = orientation.x;
        candidateQ[qBase + 4u] = orientation.y;
        candidateQ[qBase + 5u] = orientation.z;
        candidateQ[qBase + 6u] = orientation.w;
    }
    for (uint localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const MRJointDescriptorGPU joint =
            joints[articulation.firstJoint + localJoint];
        if (joint.nv == 1u) {
            const uint localQ =
                joint.qOffset - articulation.qOffset;
            const uint localV =
                joint.vOffset - articulation.vOffset;
            candidateQ[qBase + localQ] =
                sourceQ[qBase + localQ] +
                timestep * candidateV[vBase + localV];
        }
    }

    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device MRBodyStateGPU& state =
            candidateBodies[bodyBase + globalBody];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        if (properties.motionType == MR_MOTION_STATIC) {
            continue;
        }
        const float maxLinear =
            properties.dampingAndSpeedLimits.z;
        const float maxAngular =
            properties.dampingAndSpeedLimits.w;
        float3 linear = state.linearVelocityAndInverseMass.xyz;
        float3 angular = state.angularVelocity.xyz;
        const float linearLength = length(linear);
        const float angularLength = length(angular);
        if (maxLinear > 0.0f && linearLength > maxLinear) {
            linear *= maxLinear / linearLength;
        }
        if (maxAngular > 0.0f && angularLength > maxAngular) {
            angular *= maxAngular / angularLength;
        }
        state.linearVelocityAndInverseMass.xyz = linear;
        state.angularVelocity.xyz = angular;
        state.position.xyz += timestep * linear;
        state.position.w = 1.0f;
        const float3 rotationVector = timestep * angular;
        const float angle = length(rotationVector);
        const float halfAngle = 0.5f * angle;
        const float scale = angle > 1.0e-6f
            ? sin(halfAngle) / angle
            : 0.5f - angle * angle / 48.0f;
        const float4 increment = float4(
            rotationVector * scale,
            cos(halfAngle)
        );
        float4 orientation;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    increment,
                    state.orientation
                ),
                orientation
            ) ||
            !finite4(state.position)) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        state.orientation = orientation;
        MRBodyStateGPU updatedState = state;
        if (!writeWorldInverseInertia(
                updatedState,
                properties,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        state = updatedState;
    }
    status.eventTimes.x = timestep;
    status.eventTimes.y = 0.0f;
    statuses[environment] = status;
}

// Converts a contact-stage failure into the existing world transaction latch.
kernel void mr_world_latch_contact_status(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses [[buffer(2)]],
    device MRMetalWorldStatusGPU* worldStatuses [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount) {
        return;
    }
    MRMetalWorldStatusGPU status = worldStatuses[environment];
    const MRMetalWorldContactStatusGPU contact =
        contactStatuses[environment];
    if (status.code == MR_STEP_SUCCESS &&
        (
            contact.code != MR_STEP_SUCCESS ||
            contact.environment != environment ||
            contact.controlStep != pass.controlStep ||
            contact.physicsSubstep != pass.physicsSubstep
        )) {
        status.code =
            contact.code == MR_STEP_SUCCESS
            ? MR_STEP_UNSUPPORTED
            : contact.code;
        status.failingSubstep = pass.physicsSubstep;
        status.failingIndex =
            contact.firstFailingConstraint != MR_INVALID_INDEX
            ? contact.firstFailingConstraint
            : contact.firstFailingPair;
        worldStatuses[environment] = status;
    }
}

// Publishes or rolls back free-body and manifold state after the q/v commit.
kernel void mr_world_commit_contact_state(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRMetalWorldStatusGPU* worldStatuses [[buffer(3)]],
    device const uint* sceneBodyIndices [[buffer(4)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(5)]],
    device const MRBodyStateGPU* checkpointScene [[buffer(6)]],
    device MRBodyStateGPU* destinationScene [[buffer(7)]],
    device const MRManifoldHeaderGPU* candidateHeaders [[buffer(8)]],
    device const MRManifoldPointGPU* candidatePoints [[buffer(9)]],
    device const uint* candidateCounts [[buffer(10)]],
    device const MRManifoldHeaderGPU* checkpointHeaders [[buffer(11)]],
    device const MRManifoldPointGPU* checkpointPoints [[buffer(12)]],
    device const uint* checkpointCounts [[buffer(13)]],
    device MRManifoldHeaderGPU* destinationHeaders [[buffer(14)]],
    device MRManifoldPointGPU* destinationPoints [[buffer(15)]],
    device uint* destinationCounts [[buffer(16)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const bool publish =
        pass.physicsSubstep < worldDispatch.physicsSubsteps &&
        worldStatuses[environment].code == MR_STEP_SUCCESS;
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        destinationScene[sceneBase + localScene] =
            publish
            ? candidateBodies[
                  bodyBase + sceneBodyIndices[localScene]
              ]
            : checkpointScene[sceneBase + localScene];
    }
    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    destinationCounts[environment] = publish
        ? candidateCounts[environment]
        : checkpointCounts[environment];
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        destinationHeaders[manifoldBase + manifold] =
            publish
            ? candidateHeaders[manifoldBase + manifold]
            : checkpointHeaders[manifoldBase + manifold];
        for (uint point = 0u;
             point <
                 MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
             ++point) {
            const uint index =
                pointBase +
                manifold *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                point;
            destinationPoints[index] = publish
                ? candidatePoints[index]
                : checkpointPoints[index];
        }
    }
}

// Convex/simplex caches are performance-semantic state: they influence query
// iteration order and therefore must obey the same transaction as physics.
// Only the final successful microstep publishes active convex/mesh entries.
kernel void mr_world_publish_convex_cache(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(1)]],
    device const uint* overlapFlags [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(3)]],
    device const MRConvexQueryCacheGPU* candidateCaches [[buffer(4)]],
    device MRConvexQueryCacheGPU* publishedCaches [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (globalIndex >= total) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.eligiblePairCount;
    const uint compiledPair =
        globalIndex -
        environment * dispatch.eligiblePairCount;
    const uint pairClass =
        eligiblePairs[compiledPair].pairClass;
    if (statuses[environment].code == MR_STEP_SUCCESS &&
        overlapFlags[globalIndex] == 1u &&
        (pairClass == MR_COLLISION_PAIR_CONVEX ||
         pairClass == MR_COLLISION_PAIR_MESH)) {
        publishedCaches[globalIndex] =
            candidateCaches[globalIndex];
    }
}

// Appends scene-body pose/velocity observations and captures typed contact
// evidence for the completed control step.
kernel void mr_world_capture_contact(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRBodyStateGPU* sceneState [[buffer(3)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(4)]],
    device float* observations [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* publicStatuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint observationBase =
        pass.controlStep * worldDispatch.observationStepStride +
        environment *
            worldDispatch.observationEnvironmentStride +
        worldDispatch.nq + worldDispatch.nv;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const MRBodyStateGPU state =
            sceneState[sceneBase + localScene];
        const uint output =
            observationBase + 13u * localScene;
        observations[output + 0u] = state.position.x;
        observations[output + 1u] = state.position.y;
        observations[output + 2u] = state.position.z;
        observations[output + 3u] = state.orientation.x;
        observations[output + 4u] = state.orientation.y;
        observations[output + 5u] = state.orientation.z;
        observations[output + 6u] = state.orientation.w;
        observations[output + 7u] =
            state.linearVelocityAndInverseMass.x;
        observations[output + 8u] =
            state.linearVelocityAndInverseMass.y;
        observations[output + 9u] =
            state.linearVelocityAndInverseMass.z;
        observations[output + 10u] =
            state.angularVelocity.x;
        observations[output + 11u] =
            state.angularVelocity.y;
        observations[output + 12u] =
            state.angularVelocity.z;
    }
    publicStatuses[
        pass.controlStep * dispatch.environmentCount +
        environment
    ] = statuses[environment];
}
