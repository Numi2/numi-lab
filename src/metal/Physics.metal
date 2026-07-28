#include <metal_stdlib>
#include "metalrobo/gpu_types.h"

using namespace metal;

namespace {

constant float kTiny = 1.0e-8f;
constant float kContactVelocityScale = 0.05f;

struct SpatialVector {
    float4 angular;
    float4 linear;
};

struct SpatialMatrix {
    float values[36];
};

// The complete per-environment articulation workspace is statically bounded.
// At MR_MAX_DOF this occupies less than 24 KiB, leaving headroom below the
// 32 KiB minimum Metal threadgroup-memory guarantee.
struct DynamicsScratch {
    float q[MR_MAX_DOF];
    float qd[MR_MAX_DOF];
    float qdd[MR_MAX_DOF];
    float torque[MR_MAX_DOF];
    float desiredPosition[MR_MAX_DOF];
    float articulatedD[MR_MAX_DOF];
    float articulatedUScalar[MR_MAX_DOF];

    float4 parentToChildRotation[MR_MAX_DOF];
    float4 parentToChildOffset[MR_MAX_DOF];
    float4 worldPosition[MR_MAX_LINKS];
    float4 worldRotation[MR_MAX_LINKS];

    SpatialVector velocity[MR_MAX_LINKS];
    SpatialVector biasAcceleration[MR_MAX_LINKS];
    SpatialVector acceleration[MR_MAX_LINKS];
    SpatialVector biasForce[MR_MAX_LINKS];
    SpatialVector articulatedU[MR_MAX_DOF];
    SpatialMatrix articulatedInertia[MR_MAX_LINKS];
};

inline float3 xyz(const mr_float4 value) {
    return float3(value.x, value.y, value.z);
}

inline float4 quaternionConjugate(const float4 q) {
    return float4(-q.xyz, q.w);
}

inline float4 quaternionNormalizeSafe(const float4 value) {
    return value * rsqrt(max(dot(value, value), kTiny));
}

inline float4 quaternionMultiply(const float4 lhs, const float4 rhs) {
    return float4(
        lhs.w * rhs.xyz + rhs.w * lhs.xyz + cross(lhs.xyz, rhs.xyz),
        lhs.w * rhs.w - dot(lhs.xyz, rhs.xyz)
    );
}

inline float3 quaternionRotate(const float4 q, const float3 value) {
    const float3 twiceCross = 2.0f * cross(q.xyz, value);
    return value + q.w * twiceCross + cross(q.xyz, twiceCross);
}

inline float4 axisAngleQuaternion(float3 axis, const float angle) {
    axis *= rsqrt(max(dot(axis, axis), kTiny));
    const float halfAngle = 0.5f * angle;
    return float4(axis * sin(halfAngle), cos(halfAngle));
}

inline SpatialVector spatialZero() {
    SpatialVector result;
    result.angular = 0.0f;
    result.linear = 0.0f;
    return result;
}

inline SpatialVector spatialAdd(
    const SpatialVector lhs,
    const SpatialVector rhs
) {
    SpatialVector result;
    result.angular = float4(lhs.angular.xyz + rhs.angular.xyz, 0.0f);
    result.linear = float4(lhs.linear.xyz + rhs.linear.xyz, 0.0f);
    return result;
}

inline SpatialVector spatialScale(
    const SpatialVector value,
    const float scale
) {
    SpatialVector result;
    result.angular = float4(value.angular.xyz * scale, 0.0f);
    result.linear = float4(value.linear.xyz * scale, 0.0f);
    return result;
}

inline float spatialDot(
    const SpatialVector lhs,
    const SpatialVector rhs
) {
    return dot(lhs.angular.xyz, rhs.angular.xyz) +
        dot(lhs.linear.xyz, rhs.linear.xyz);
}

inline SpatialVector motionCross(
    const SpatialVector lhs,
    const SpatialVector rhs
) {
    SpatialVector result;
    result.angular =
        float4(cross(lhs.angular.xyz, rhs.angular.xyz), 0.0f);
    result.linear = float4(
        cross(lhs.linear.xyz, rhs.angular.xyz) +
            cross(lhs.angular.xyz, rhs.linear.xyz),
        0.0f
    );
    return result;
}

inline SpatialVector forceCross(
    const SpatialVector motion,
    const SpatialVector force
) {
    SpatialVector result;
    result.angular = float4(
        cross(motion.angular.xyz, force.angular.xyz) +
            cross(motion.linear.xyz, force.linear.xyz),
        0.0f
    );
    result.linear =
        float4(cross(motion.angular.xyz, force.linear.xyz), 0.0f);
    return result;
}

inline SpatialVector motionParentToChild(
    const float4 childToParentRotation,
    const float3 childOffsetInParent,
    const SpatialVector parentMotion
) {
    const float4 parentToChild =
        quaternionConjugate(childToParentRotation);
    SpatialVector result;
    result.angular = float4(
        quaternionRotate(parentToChild, parentMotion.angular.xyz),
        0.0f
    );
    result.linear = float4(
        quaternionRotate(
            parentToChild,
            parentMotion.linear.xyz +
                cross(parentMotion.angular.xyz, childOffsetInParent)
        ),
        0.0f
    );
    return result;
}

inline SpatialVector forceChildToParent(
    const float4 childToParentRotation,
    const float3 childOffsetInParent,
    const SpatialVector childForce
) {
    const float3 linearParent =
        quaternionRotate(childToParentRotation, childForce.linear.xyz);
    SpatialVector result;
    result.angular = float4(
        quaternionRotate(childToParentRotation, childForce.angular.xyz) +
            cross(childOffsetInParent, linearParent),
        0.0f
    );
    result.linear = float4(linearParent, 0.0f);
    return result;
}

inline float spatialComponent(const SpatialVector value, const uint index) {
    return index < 3u ? value.angular[index] : value.linear[index - 3u];
}

inline void setSpatialComponent(
    thread SpatialVector& value,
    const uint index,
    const float component
) {
    if (index < 3u) {
        value.angular[index] = component;
    } else {
        value.linear[index - 3u] = component;
    }
}

inline SpatialVector matrixVector(
    threadgroup const SpatialMatrix& matrix,
    const SpatialVector vector
) {
    SpatialVector result = spatialZero();
    for (uint row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint column = 0u; column < 6u; ++column) {
            value += matrix.values[row * 6u + column] *
                spatialComponent(vector, column);
        }
        setSpatialComponent(result, row, value);
    }
    return result;
}

inline void matrixSubtractOuterProduct(
    threadgroup SpatialMatrix& matrix,
    const SpatialVector vector,
    const float inverseDenominator
) {
    for (uint row = 0u; row < 6u; ++row) {
        const float rowValue = spatialComponent(vector, row);
        for (uint column = 0u; column < 6u; ++column) {
            matrix.values[row * 6u + column] -=
                rowValue * spatialComponent(vector, column) *
                inverseDenominator;
        }
    }
}

inline SpatialVector unitSpatialVector(const uint index) {
    SpatialVector result = spatialZero();
    setSpatialComponent(result, index, 1.0f);
    return result;
}

inline void addTransformedInertia(
    threadgroup SpatialMatrix& parent,
    threadgroup const SpatialMatrix& child,
    const float4 childToParentRotation,
    const float3 childOffsetInParent
) {
    // Compute X^T I X column by column. This deliberately favors a compact,
    // auditable implementation; each threadgroup owns an environment and
    // parallelism is across environments.
    for (uint column = 0u; column < 6u; ++column) {
        const SpatialVector parentBasis = unitSpatialVector(column);
        const SpatialVector childMotion = motionParentToChild(
            childToParentRotation,
            childOffsetInParent,
            parentBasis
        );
        const SpatialVector childForce = matrixVector(child, childMotion);
        const SpatialVector parentForce = forceChildToParent(
            childToParentRotation,
            childOffsetInParent,
            childForce
        );
        for (uint row = 0u; row < 6u; ++row) {
            parent.values[row * 6u + column] +=
                spatialComponent(parentForce, row);
        }
    }
}

inline void initializeSpatialInertia(
    threadgroup SpatialMatrix& matrix,
    constant MRLinkGPU& link
) {
    for (uint index = 0u; index < 36u; ++index) {
        matrix.values[index] = 0.0f;
    }

    const float mass = max(link.massAndCOMX.x, 0.0f);
    const float3 com = float3(
        link.massAndCOMX.y,
        link.massAndCOMX.z,
        link.massAndCOMX.w
    );
    const float comSquared = dot(com, com);

    float inertia[9];
    inertia[0] = link.inertiaRow0.x;
    inertia[1] = link.inertiaRow0.y;
    inertia[2] = link.inertiaRow0.z;
    inertia[3] = link.inertiaRow1.x;
    inertia[4] = link.inertiaRow1.y;
    inertia[5] = link.inertiaRow1.z;
    inertia[6] = link.inertiaRow2.x;
    inertia[7] = link.inertiaRow2.y;
    inertia[8] = link.inertiaRow2.z;

    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            const float parallelAxis = mass * (
                (row == column ? comSquared : 0.0f) -
                com[row] * com[column]
            );
            matrix.values[row * 6u + column] =
                inertia[row * 3u + column] + parallelAxis;
        }
    }

    const float crossCom[9] = {
        0.0f, -com.z, com.y,
        com.z, 0.0f, -com.x,
        -com.y, com.x, 0.0f,
    };
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            matrix.values[row * 6u + column + 3u] =
                mass * crossCom[row * 3u + column];
            matrix.values[(row + 3u) * 6u + column] =
                -mass * crossCom[row * 3u + column];
            matrix.values[(row + 3u) * 6u + column + 3u] =
                row == column ? mass : 0.0f;
        }
    }
}

inline SpatialVector jointMotionSubspace(constant MRJointGPU& joint) {
    SpatialVector result = spatialZero();
    const float3 axis =
        xyz(joint.axis) * rsqrt(max(dot(xyz(joint.axis), xyz(joint.axis)), kTiny));
    if (joint.jointType == 1u) {
        result.linear = float4(axis, 0.0f);
    } else {
        result.angular = float4(axis, 0.0f);
    }
    return result;
}

inline void computeTransformsAndVelocity(
    constant MRModelGPU& model,
    constant MRJointGPU* joints,
    threadgroup DynamicsScratch& scratch
) {
    scratch.worldPosition[0] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    scratch.worldRotation[0] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    scratch.velocity[0] = spatialZero();

    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        constant MRJointGPU& joint = joints[jointIndex];
        const uint parent = uint(max(joint.parentLink, 0));
        const uint child = joint.childLink;
        const float4 originRotation = quaternionNormalizeSafe(float4(
            joint.parentRotation.x,
            joint.parentRotation.y,
            joint.parentRotation.z,
            joint.parentRotation.w
        ));
        const float3 axis =
            xyz(joint.axis) *
            rsqrt(max(dot(xyz(joint.axis), xyz(joint.axis)), kTiny));

        float4 childToParentRotation = originRotation;
        float3 childOffset = xyz(joint.parentOffset);
        if (joint.jointType == 1u) {
            childOffset += quaternionRotate(
                originRotation,
                axis * scratch.q[jointIndex]
            );
        } else {
            childToParentRotation = quaternionNormalizeSafe(
                quaternionMultiply(
                    originRotation,
                    axisAngleQuaternion(axis, scratch.q[jointIndex])
                )
            );
        }

        scratch.parentToChildRotation[jointIndex] =
            childToParentRotation;
        scratch.parentToChildOffset[jointIndex] =
            float4(childOffset, 0.0f);

        const float4 parentWorldRotation = scratch.worldRotation[parent];
        scratch.worldRotation[child] = quaternionNormalizeSafe(
            quaternionMultiply(
                parentWorldRotation,
                childToParentRotation
            )
        );
        scratch.worldPosition[child] = float4(
            scratch.worldPosition[parent].xyz +
                quaternionRotate(parentWorldRotation, childOffset),
            1.0f
        );

        const SpatialVector jointVelocity = spatialScale(
            jointMotionSubspace(joint),
            scratch.qd[jointIndex]
        );
        scratch.velocity[child] = spatialAdd(
            motionParentToChild(
                childToParentRotation,
                childOffset,
                scratch.velocity[parent]
            ),
            jointVelocity
        );
        scratch.biasAcceleration[child] =
            motionCross(scratch.velocity[child], jointVelocity);
    }
}

inline float3 pointVelocityWorld(
    threadgroup const DynamicsScratch& scratch,
    const uint linkIndex,
    const float3 localPoint
) {
    const SpatialVector bodyVelocity = scratch.velocity[linkIndex];
    const float3 localVelocity =
        bodyVelocity.linear.xyz +
        cross(bodyVelocity.angular.xyz, localPoint);
    return quaternionRotate(
        scratch.worldRotation[linkIndex],
        localVelocity
    );
}

inline SpatialVector computeContactForce(
    constant MRModelGPU& model,
    constant MRColliderGPU* colliders,
    threadgroup const DynamicsScratch& scratch,
    const uint linkIndex,
    thread float& normalImpulse,
    const float substepTimestep
) {
    SpatialVector accumulated = spatialZero();
    float3 planeNormal = xyz(model.groundPlane);
    planeNormal *= rsqrt(max(dot(planeNormal, planeNormal), kTiny));

    for (uint colliderIndex = 0u;
         colliderIndex < model.colliderCount;
         ++colliderIndex) {
        constant MRColliderGPU& collider = colliders[colliderIndex];
        if (collider.linkIndex != int(linkIndex)) {
            continue;
        }
        if (collider.shapeType != MR_SHAPE_SPHERE &&
            collider.shapeType != MR_SHAPE_CAPSULE) {
            continue;
        }

        float3 localCenter = xyz(collider.centerAndRadius);
        if (collider.shapeType == MR_SHAPE_CAPSULE) {
            const float3 endpointA = localCenter;
            const float3 endpointB = xyz(collider.extent);
            const float3 worldA =
                scratch.worldPosition[linkIndex].xyz +
                quaternionRotate(scratch.worldRotation[linkIndex], endpointA);
            const float3 worldB =
                scratch.worldPosition[linkIndex].xyz +
                quaternionRotate(scratch.worldRotation[linkIndex], endpointB);
            localCenter =
                dot(planeNormal, worldA) < dot(planeNormal, worldB)
                ? endpointA
                : endpointB;
        }

        const float radius = max(collider.centerAndRadius.w, 0.0f);
        const float3 worldCenter =
            scratch.worldPosition[linkIndex].xyz +
            quaternionRotate(
                scratch.worldRotation[linkIndex],
                localCenter
            );
        const float signedDistance =
            dot(planeNormal, worldCenter) -
            model.groundPlane.w -
            radius;
        if (signedDistance >= 0.0f) {
            continue;
        }

        const float3 normalInBody = quaternionRotate(
            quaternionConjugate(scratch.worldRotation[linkIndex]),
            planeNormal
        );
        const float3 localContact = localCenter - normalInBody * radius;
        const float3 pointVelocity =
            pointVelocityWorld(scratch, linkIndex, localContact);
        const float normalVelocity = dot(pointVelocity, planeNormal);
        const float penetration = -signedDistance;
        const float stiffness = max(collider.material.z, 0.0f);
        const float damping = max(collider.material.w, 0.0f);
        const float restitution = clamp(collider.material.y, 0.0f, 1.0f);
        float normalForce =
            stiffness * penetration -
            damping * (1.0f + restitution) * normalVelocity;
        normalForce = max(normalForce, 0.0f);

        const float3 tangentialVelocity =
            pointVelocity - planeNormal * normalVelocity;
        const float tangentialSpeed = length(tangentialVelocity);
        const float friction = max(collider.material.x, 0.0f);
        const float3 frictionForce =
            -friction * normalForce * tangentialVelocity /
            max(tangentialSpeed, kContactVelocityScale);
        const float3 worldForce =
            planeNormal * normalForce + frictionForce;
        const float3 bodyForce = quaternionRotate(
            quaternionConjugate(scratch.worldRotation[linkIndex]),
            worldForce
        );

        accumulated.linear.xyz += bodyForce;
        accumulated.angular.xyz += cross(localContact, bodyForce);
        normalImpulse += normalForce * substepTimestep;
    }
    return accumulated;
}

inline void runArticulatedBodyAlgorithm(
    constant MRModelGPU& model,
    constant MRJointGPU* joints,
    constant MRLinkGPU* links,
    constant MRColliderGPU* colliders,
    threadgroup DynamicsScratch& scratch,
    thread float& contactImpulse,
    const float substepTimestep
) {
    computeTransformsAndVelocity(model, joints, scratch);

    for (uint linkIndex = 0u;
         linkIndex < model.linkCount;
         ++linkIndex) {
        initializeSpatialInertia(
            scratch.articulatedInertia[linkIndex],
            links[linkIndex]
        );
        const SpatialVector momentum = matrixVector(
            scratch.articulatedInertia[linkIndex],
            scratch.velocity[linkIndex]
        );
        scratch.biasForce[linkIndex] =
            forceCross(scratch.velocity[linkIndex], momentum);
        if (linkIndex != 0u) {
            const SpatialVector externalForce = computeContactForce(
                model,
                colliders,
                scratch,
                linkIndex,
                contactImpulse,
                substepTimestep
            );
            scratch.biasForce[linkIndex] = spatialAdd(
                scratch.biasForce[linkIndex],
                spatialScale(externalForce, -1.0f)
            );
        }
    }

    for (int jointIndex = int(model.dofCount) - 1;
         jointIndex >= 0;
         --jointIndex) {
        constant MRJointGPU& joint = joints[uint(jointIndex)];
        const uint child = joint.childLink;
        const uint parent = uint(max(joint.parentLink, 0));
        const SpatialVector motionSubspace =
            jointMotionSubspace(joint);
        const SpatialVector articulatedU = matrixVector(
            scratch.articulatedInertia[child],
            motionSubspace
        );
        scratch.articulatedU[uint(jointIndex)] = articulatedU;
        const float denominator = max(
            spatialDot(motionSubspace, articulatedU) +
                max(joint.drive.w, 0.0f),
            1.0e-6f
        );
        const float generalizedForce =
            scratch.torque[uint(jointIndex)] -
            spatialDot(motionSubspace, scratch.biasForce[child]);
        scratch.articulatedD[uint(jointIndex)] = denominator;
        scratch.articulatedUScalar[uint(jointIndex)] =
            generalizedForce;

        const float inverseDenominator = 1.0f / denominator;
        matrixSubtractOuterProduct(
            scratch.articulatedInertia[child],
            articulatedU,
            inverseDenominator
        );
        SpatialVector propagatedBias = spatialAdd(
            scratch.biasForce[child],
            matrixVector(
                scratch.articulatedInertia[child],
                scratch.biasAcceleration[child]
            )
        );
        propagatedBias = spatialAdd(
            propagatedBias,
            spatialScale(
                articulatedU,
                generalizedForce * inverseDenominator
            )
        );

        addTransformedInertia(
            scratch.articulatedInertia[parent],
            scratch.articulatedInertia[child],
            scratch.parentToChildRotation[uint(jointIndex)],
            scratch.parentToChildOffset[uint(jointIndex)].xyz
        );
        scratch.biasForce[parent] = spatialAdd(
            scratch.biasForce[parent],
            forceChildToParent(
                scratch.parentToChildRotation[uint(jointIndex)],
                scratch.parentToChildOffset[uint(jointIndex)].xyz,
                propagatedBias
            )
        );
    }

    scratch.acceleration[0] = spatialZero();
    scratch.acceleration[0].linear = float4(
        -xyz(model.gravityAndTimestep),
        0.0f
    );
    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        constant MRJointGPU& joint = joints[jointIndex];
        const uint child = joint.childLink;
        const uint parent = uint(max(joint.parentLink, 0));
        SpatialVector acceleration = spatialAdd(
            motionParentToChild(
                scratch.parentToChildRotation[jointIndex],
                scratch.parentToChildOffset[jointIndex].xyz,
                scratch.acceleration[parent]
            ),
            scratch.biasAcceleration[child]
        );
        const float jointAcceleration = (
            scratch.articulatedUScalar[jointIndex] -
            spatialDot(scratch.articulatedU[jointIndex], acceleration)
        ) / scratch.articulatedD[jointIndex];
        scratch.qdd[jointIndex] = jointAcceleration;
        acceleration = spatialAdd(
            acceleration,
            spatialScale(
                jointMotionSubspace(joint),
                jointAcceleration
            )
        );
        scratch.acceleration[child] = acceleration;
    }
}

inline void updateDriveTorques(
    constant MRModelGPU& model,
    constant MRJointGPU* joints,
    threadgroup DynamicsScratch& scratch
) {
    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        constant MRJointGPU& joint = joints[jointIndex];
        float jointTorque =
            joint.drive.x *
                (scratch.desiredPosition[jointIndex] -
                 scratch.q[jointIndex]) -
            joint.drive.y * scratch.qd[jointIndex];

        // A stiff, damped soft stop begins just inside each hard limit.
        const float range =
            max(joint.limits.y - joint.limits.x, 1.0e-3f);
        const float margin = min(0.02f * range, 0.05f);
        const float lowerStop = joint.limits.x + margin;
        const float upperStop = joint.limits.y - margin;
        const float stopStiffness =
            max(5.0f * joint.drive.x, 100.0f);
        if (scratch.q[jointIndex] < lowerStop) {
            jointTorque +=
                stopStiffness *
                (lowerStop - scratch.q[jointIndex]) -
                joint.drive.y * min(scratch.qd[jointIndex], 0.0f);
        } else if (scratch.q[jointIndex] > upperStop) {
            jointTorque -=
                stopStiffness *
                (scratch.q[jointIndex] - upperStop) +
                joint.drive.y * max(scratch.qd[jointIndex], 0.0f);
        }
        scratch.torque[jointIndex] = clamp(
            jointTorque,
            -joint.limits.w,
            joint.limits.w
        );
    }
}

inline uint randomStep(thread uint& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

inline float randomUnit(thread uint& state) {
    return float(randomStep(state) >> 8u) * (1.0f / 16777216.0f);
}

inline uint makeRandomState(
    constant MRStepUniformsGPU& uniforms,
    const uint environmentIndex,
    const uint stream
) {
    uint value =
        uniforms.seedLo ^
        (uniforms.seedHi * 0x9e3779b9u) ^
        ((environmentIndex + 1u) * 0x85ebca6bu) ^
        ((stream + 1u) * 0xc2b2ae35u);
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value == 0u ? 1u : value;
}

inline float3 taskPointLocal(
    constant MRModelGPU& model,
    constant MRColliderGPU* colliders
) {
    float3 result = 0.0f;
    const int taskLink = int(model.linkCount) - 1;
    // The last collider on the terminal link is the task site. For Franka
    // this is the fixed flange at +0.107 m; future model builders can use the
    // same convention without widening the public ABI.
    for (uint index = 0u; index < model.colliderCount; ++index) {
        if (colliders[index].linkIndex == taskLink) {
            result = xyz(colliders[index].centerAndRadius);
        }
    }
    return result;
}

inline float3 taskPointWorld(
    constant MRModelGPU& model,
    constant MRColliderGPU* colliders,
    threadgroup const DynamicsScratch& scratch
) {
    const uint taskLink = model.linkCount - 1u;
    return scratch.worldPosition[taskLink].xyz +
        quaternionRotate(
            scratch.worldRotation[taskLink],
            taskPointLocal(model, colliders)
        );
}

inline void writeObservationsAndPoses(
    constant MRModelGPU& model,
    constant MRColliderGPU* colliders,
    constant MRStepUniformsGPU& uniforms,
    const uint environmentIndex,
    threadgroup const DynamicsScratch& scratch,
    device const float4* targets,
    device float* observations,
    device float4* bodyPositions,
    device float4* bodyRotations
) {
    const ulong observationBase =
        ulong(environmentIndex) * ulong(model.observationCount);
    for (uint index = 0u; index < model.observationCount; ++index) {
        observations[observationBase + index] = 0.0f;
    }
    for (uint index = 0u;
         index < model.dofCount && index < model.observationCount;
         ++index) {
        observations[observationBase + index] = scratch.q[index];
    }
    for (uint index = 0u;
         index < model.dofCount &&
         model.dofCount + index < model.observationCount;
         ++index) {
        observations[observationBase + model.dofCount + index] =
            scratch.qd[index];
    }

    const uint taskOffset = model.dofCount * 2u;
    const float3 target = targets[environmentIndex].xyz;
    const float3 taskDelta =
        taskPointWorld(model, colliders, scratch) - target;
    if (taskOffset + 2u < model.observationCount) {
        observations[observationBase + taskOffset] = taskDelta.x;
        observations[observationBase + taskOffset + 1u] = taskDelta.y;
        observations[observationBase + taskOffset + 2u] = taskDelta.z;
    }
    if (taskOffset + 5u < model.observationCount) {
        observations[observationBase + taskOffset + 3u] = target.x;
        observations[observationBase + taskOffset + 4u] = target.y;
        observations[observationBase + taskOffset + 5u] = target.z;
    }

    if (uniforms.captureBodyPoses != 0u) {
        const ulong poseBase =
            ulong(environmentIndex) * ulong(model.linkCount);
        for (uint linkIndex = 0u;
             linkIndex < model.linkCount;
             ++linkIndex) {
            bodyPositions[poseBase + linkIndex] =
                scratch.worldPosition[linkIndex];
            bodyRotations[poseBase + linkIndex] =
                scratch.worldRotation[linkIndex];
        }
    }
}

inline void resetEnvironment(
    constant MRModelGPU& model,
    constant MRJointGPU* joints,
    constant MRColliderGPU* colliders,
    constant float* homePosition,
    constant MRStepUniformsGPU& uniforms,
    const uint environmentIndex,
    const uint stream,
    threadgroup DynamicsScratch& scratch,
    device float* q,
    device float* qd,
    device float* qdd,
    device float* torque,
    device float4* targets,
    device uint* episodeSteps,
    device float* observations,
    device float* rewards,
    device uchar* terminated,
    device float4* bodyPositions,
    device float4* bodyRotations
) {
    uint randomState = makeRandomState(
        uniforms,
        environmentIndex,
        stream
    );
    const ulong stateBase =
        ulong(environmentIndex) * ulong(model.dofCount);
    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        const float lower = joints[jointIndex].limits.x;
        const float upper = joints[jointIndex].limits.y;
        const float perturbation =
            (2.0f * randomUnit(randomState) - 1.0f) * 0.01f;
        scratch.q[jointIndex] = clamp(
            homePosition[jointIndex] + perturbation,
            lower,
            upper
        );
        scratch.qd[jointIndex] = 0.0f;
        scratch.qdd[jointIndex] = 0.0f;
        scratch.torque[jointIndex] = 0.0f;
        scratch.desiredPosition[jointIndex] =
            scratch.q[jointIndex];
        q[stateBase + jointIndex] = scratch.q[jointIndex];
        qd[stateBase + jointIndex] = 0.0f;
        qdd[stateBase + jointIndex] = 0.0f;
        torque[stateBase + jointIndex] = 0.0f;
    }

    const float3 lower = xyz(model.targetLowerAndRadius);
    const float3 upper = xyz(model.targetUpperAndBonus);
    targets[environmentIndex] = float4(
        mix(lower.x, upper.x, randomUnit(randomState)),
        mix(lower.y, upper.y, randomUnit(randomState)),
        mix(lower.z, upper.z, randomUnit(randomState)),
        1.0f
    );
    episodeSteps[environmentIndex] = 0u;
    rewards[environmentIndex] = 0.0f;
    terminated[environmentIndex] = uchar(0);

    computeTransformsAndVelocity(model, joints, scratch);
    writeObservationsAndPoses(
        model,
        colliders,
        uniforms,
        environmentIndex,
        scratch,
        targets,
        observations,
        bodyPositions,
        bodyRotations
    );
}

} // namespace

kernel void metalrobo_reset(
    constant MRModelGPU& model [[buffer(0)]],
    constant MRJointGPU* joints [[buffer(1)]],
    constant MRLinkGPU* links [[buffer(2)]],
    constant MRColliderGPU* colliders [[buffer(3)]],
    constant float* homePosition [[buffer(4)]],
    constant MRStepUniformsGPU& uniforms [[buffer(5)]],
    device const float* actions [[buffer(6)]],
    device float* q [[buffer(7)]],
    device float* qd [[buffer(8)]],
    device float* qdd [[buffer(9)]],
    device float* torque [[buffer(10)]],
    device float4* targets [[buffer(11)]],
    device uint* episodeSteps [[buffer(12)]],
    device float* observations [[buffer(13)]],
    device float* rewards [[buffer(14)]],
    device uchar* terminated [[buffer(15)]],
    device float4* bodyPositions [[buffer(16)]],
    device float4* bodyRotations [[buffer(17)]],
    uint lane [[thread_position_in_threadgroup]],
    uint environmentIndex [[threadgroup_position_in_grid]]
) {
    (void)links;
    (void)actions;
    threadgroup DynamicsScratch scratch;
    if (environmentIndex >= uniforms.environmentCount || lane != 0u) {
        return;
    }
    resetEnvironment(
        model,
        joints,
        colliders,
        homePosition,
        uniforms,
        environmentIndex,
        0u,
        scratch,
        q,
        qd,
        qdd,
        torque,
        targets,
        episodeSteps,
        observations,
        rewards,
        terminated,
        bodyPositions,
        bodyRotations
    );
}

kernel void metalrobo_step(
    constant MRModelGPU& model [[buffer(0)]],
    constant MRJointGPU* joints [[buffer(1)]],
    constant MRLinkGPU* links [[buffer(2)]],
    constant MRColliderGPU* colliders [[buffer(3)]],
    constant float* homePosition [[buffer(4)]],
    constant MRStepUniformsGPU& uniforms [[buffer(5)]],
    device const float* actions [[buffer(6)]],
    device float* q [[buffer(7)]],
    device float* qd [[buffer(8)]],
    device float* qdd [[buffer(9)]],
    device float* torque [[buffer(10)]],
    device float4* targets [[buffer(11)]],
    device uint* episodeSteps [[buffer(12)]],
    device float* observations [[buffer(13)]],
    device float* rewards [[buffer(14)]],
    device uchar* terminated [[buffer(15)]],
    device float4* bodyPositions [[buffer(16)]],
    device float4* bodyRotations [[buffer(17)]],
    uint lane [[thread_position_in_threadgroup]],
    uint environmentIndex [[threadgroup_position_in_grid]]
) {
    threadgroup DynamicsScratch scratch;
    if (environmentIndex >= uniforms.environmentCount || lane != 0u) {
        return;
    }

    if (terminated[environmentIndex] != uchar(0)) {
        if (uniforms.autoReset == 0u) {
            rewards[environmentIndex] = 0.0f;
            return;
        }
        resetEnvironment(
            model,
            joints,
            colliders,
            homePosition,
            uniforms,
            environmentIndex,
            episodeSteps[environmentIndex] + 1u,
            scratch,
            q,
            qd,
            qdd,
            torque,
            targets,
            episodeSteps,
            observations,
            rewards,
            terminated,
            bodyPositions,
            bodyRotations
        );
    }

    const ulong stateBase =
        ulong(environmentIndex) * ulong(model.dofCount);
    const ulong actionBase =
        ulong(environmentIndex) * ulong(model.actionCount);
    float actionPenalty = 0.0f;
    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        scratch.q[jointIndex] = q[stateBase + jointIndex];
        scratch.qd[jointIndex] = qd[stateBase + jointIndex];
        const float action = jointIndex < model.actionCount
            ? clamp(actions[actionBase + jointIndex], -1.0f, 1.0f)
            : 0.0f;
        actionPenalty += action * action;
        const float desiredPosition = clamp(
            homePosition[jointIndex] +
                action * joints[jointIndex].drive.z,
            joints[jointIndex].limits.x,
            joints[jointIndex].limits.y
        );
        scratch.desiredPosition[jointIndex] = desiredPosition;
    }

    const uint substeps = max(model.substeps, 1u);
    const float substepTimestep =
        model.gravityAndTimestep.w / float(substeps);
    float contactImpulse = 0.0f;
    for (uint substep = 0u; substep < substeps; ++substep) {
        updateDriveTorques(model, joints, scratch);
        runArticulatedBodyAlgorithm(
            model,
            joints,
            links,
            colliders,
            scratch,
            contactImpulse,
            substepTimestep
        );
        for (uint jointIndex = 0u;
             jointIndex < model.dofCount;
             ++jointIndex) {
            const float maximumVelocity =
                max(joints[jointIndex].limits.z, 0.0f);
            scratch.qd[jointIndex] = clamp(
                scratch.qd[jointIndex] +
                    substepTimestep * scratch.qdd[jointIndex],
                -maximumVelocity,
                maximumVelocity
            );
            scratch.q[jointIndex] +=
                substepTimestep * scratch.qd[jointIndex];

            if (scratch.q[jointIndex] <= joints[jointIndex].limits.x) {
                scratch.q[jointIndex] = joints[jointIndex].limits.x;
                scratch.qd[jointIndex] =
                    max(scratch.qd[jointIndex], 0.0f);
            } else if (
                scratch.q[jointIndex] >= joints[jointIndex].limits.y
            ) {
                scratch.q[jointIndex] = joints[jointIndex].limits.y;
                scratch.qd[jointIndex] =
                    min(scratch.qd[jointIndex], 0.0f);
            }
        }
    }

    computeTransformsAndVelocity(model, joints, scratch);
    float velocityPenalty = 0.0f;
    bool finiteState = true;
    for (uint jointIndex = 0u;
         jointIndex < model.dofCount;
         ++jointIndex) {
        q[stateBase + jointIndex] = scratch.q[jointIndex];
        qd[stateBase + jointIndex] = scratch.qd[jointIndex];
        qdd[stateBase + jointIndex] = scratch.qdd[jointIndex];
        torque[stateBase + jointIndex] = scratch.torque[jointIndex];
        velocityPenalty +=
            scratch.qd[jointIndex] * scratch.qd[jointIndex];
        finiteState =
            finiteState &&
            isfinite(scratch.q[jointIndex]) &&
            isfinite(scratch.qd[jointIndex]);
    }

    const float distance = length(
        taskPointWorld(model, colliders, scratch) -
        targets[environmentIndex].xyz
    );
    const bool success =
        distance <= max(model.targetLowerAndRadius.w, 0.0f);
    const uint nextStep = episodeSteps[environmentIndex] + 1u;
    episodeSteps[environmentIndex] = nextStep;
    const bool horizonReached =
        model.episodeHorizon != 0u &&
        nextStep >= model.episodeHorizon;
    const bool done = success || horizonReached || !finiteState;
    terminated[environmentIndex] = done ? uchar(1) : uchar(0);

    float reward =
        -model.rewardScales.x * distance -
        model.rewardScales.y * actionPenalty -
        model.rewardScales.z * velocityPenalty -
        model.rewardScales.w * contactImpulse;
    if (success) {
        reward += model.targetUpperAndBonus.w;
    }
    if (!finiteState) {
        reward = -100.0f;
    }
    rewards[environmentIndex] = reward;

    writeObservationsAndPoses(
        model,
        colliders,
        uniforms,
        environmentIndex,
        scratch,
        targets,
        observations,
        bodyPositions,
        bodyRotations
    );
}
