#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/counter_rng.h"
#include "metalrobo/runtime_abi_generated.h"
#include "metalrobo/task_program_types.h"
#include "metalrobo/world_compiler_types.h"

using namespace metal;

namespace {

constant float kPi = 3.14159265358979323846f;
constant float kTwoPi = 2.0f * kPi;
template <typename T>
inline device const T* taskTable(
    device const uchar* arena,
    const uint byteOffset
) {
    return reinterpret_cast<device const T*>(
        arena + byteOffset
    );
}

inline device const MRTaskKinematicFrameGPU*
taskKinematicFrameTable(
    device const uchar* arena,
    device const MRTaskProgramHeaderGPU& program
) {
    return taskTable<MRTaskKinematicFrameGPU>(
        arena,
        program.offsets3.z +
            program.typedCounts.x * sizeof(MRTaskFrameGPU)
    );
}

inline float randomUnit(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel
) {
    return mr_task_counter_uniform(
        dispatch.seed,
        environment,
        episode,
        controlStep,
        channel
    );
}

inline float randomSigned(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel
) {
    return
        2.0f *
        randomUnit(
            dispatch,
            environment,
            episode,
            controlStep,
            channel
        ) -
        1.0f;
}

inline float randomRange(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel,
    const float lower,
    const float upper
) {
    return lower +
        (upper - lower) *
        randomUnit(
            dispatch,
            environment,
            episode,
            controlStep,
            channel
        );
}

inline float3 rotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 tangent =
        2.0f * cross(quaternion.xyz, value);
    return
        value +
        quaternion.w * tangent +
        cross(quaternion.xyz, tangent);
}

inline float3 rotateInverse(
    const float4 quaternion,
    const float3 value
) {
    const float3 tangent =
        2.0f * cross(quaternion.xyz, value);
    return
        value -
        quaternion.w * tangent +
        cross(quaternion.xyz, tangent);
}

inline float4 quaternionProduct(
    const float4 a,
    const float4 b
) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 normalizedQuaternionOrIdentity(
    const float4 value
) {
    const float normSquared = dot(value, value);
    return normSquared > 1.0e-12f && isfinite(normSquared)
        ? value * rsqrt(normSquared)
        : float4(0.0f, 0.0f, 0.0f, 1.0f);
}

struct TaskFramePose {
    float3 position;
    float4 orientation;
};

inline TaskFramePose taskFramePose(
    device const MRTaskFrameGPU& frame,
    const float4 bodyPosition,
    const float4 bodyOrientation
) {
    TaskFramePose result;
    result.position =
        bodyPosition.xyz +
        rotate(bodyOrientation, frame.localPosition.xyz);
    result.orientation = normalizedQuaternionOrIdentity(
        quaternionProduct(
            bodyOrientation,
            frame.localOrientation
        )
    );
    return result;
}

inline MRBodyStateGPU taskFrameBodyState(
    device const MRTaskFrameGPU& frame,
    device const MRBodyStateGPU* bodyStates,
    device const MRBodyStateGPU* sceneBodies
) {
    return frame.indices.y == MR_TASK_FRAME_SOURCE_SCENE_BODY
        ? sceneBodies[frame.indices.z]
        : bodyStates[frame.indices.x];
}

inline TaskFramePose taskResetBodyPose(
    device const MRTaskFrameGPU& frame,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const MRBodyStateGPU* sceneBodies
) {
    TaskFramePose result;
    if (frame.indices.y == MR_TASK_FRAME_SOURCE_SCENE_BODY) {
        device const MRBodyStateGPU& body =
            sceneBodies[frame.indices.z];
        result.position = body.position.xyz;
        result.orientation = body.orientation;
        return result;
    }
    device const MRArticulatedBodyPoseGPU& body =
        bodyPoses[frame.indices.z];
    result.position = body.position.xyz;
    result.orientation = body.orientation;
    return result;
}

inline float3 taskRotationVector(const float4 orientation) {
    float4 difference = normalizedQuaternionOrIdentity(
        orientation
    );
    if (difference.w < 0.0f) {
        difference = -difference;
    }
    const float sineHalf = length(difference.xyz);
    if (sineHalf < 1.0e-7f) {
        return 2.0f * difference.xyz;
    }
    const float angle = 2.0f * atan2(
        sineHalf,
        max(difference.w, 0.0f)
    );
    return difference.xyz * (angle / sineHalf);
}

inline float3 taskOrientationError(
    const float4 frameOrientation,
    const float4 goalOrientation
) {
    return taskRotationVector(
        quaternionProduct(
            goalOrientation,
            quaternionConjugate(frameOrientation)
        )
    );
}

struct TaskGoalPose {
    float3 position;
    float4 orientation;
};

inline ulong taskGoalRandomIdentity(
    device const MRTaskGoalGPU& goal
) {
    return static_cast<ulong>(goal.metadata.z) |
        (static_cast<ulong>(goal.metadata.w) << 32u);
}

inline float taskGoalRandomUnit(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskGoalGPU& goal,
    const uint environment,
    const uint episode,
    const uint channel
) {
    return mr_semantic_counter_uniform(
        dispatch.seed,
        environment,
        episode,
        taskGoalRandomIdentity(goal),
        0u,
        channel,
        MR_COUNTER_PURPOSE_TASK_GOAL
    );
}

inline float4 taskQuaternionExponential(
    const float3 rotationVector
) {
    const float angle = length(rotationVector);
    if (angle < 1.0e-7f) {
        return normalizedQuaternionOrIdentity(
            float4(0.5f * rotationVector, 1.0f)
        );
    }
    const float halfAngle = 0.5f * angle;
    return float4(
        rotationVector * (sin(halfAngle) / angle),
        cos(halfAngle)
    );
}

inline float4 taskQuaternionSlerp(
    const float4 startValue,
    const float4 endValue,
    const float progress
) {
    const float4 start = normalizedQuaternionOrIdentity(
        startValue
    );
    float4 end = normalizedQuaternionOrIdentity(endValue);
    float cosine = dot(start, end);
    if (cosine < 0.0f) {
        end = -end;
        cosine = -cosine;
    }
    cosine = clamp(cosine, 0.0f, 1.0f);
    if (cosine > 0.9995f) {
        return normalizedQuaternionOrIdentity(
            mix(start, end, progress)
        );
    }
    const float angle = acos(cosine);
    const float inverseSine = 1.0f / sin(angle);
    return normalizedQuaternionOrIdentity(
        sin((1.0f - progress) * angle) * inverseSine * start +
        sin(progress * angle) * inverseSine * end
    );
}

inline float taskGoalTrajectoryProgress(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskGoalGPU& goal,
    const uint episodeStep
) {
    const float unbounded =
        (
            float(episodeStep) * dispatch.timing.x +
            goal.timing.y
        ) /
        goal.timing.x;
    switch (goal.metadata.y) {
    case MR_TASK_GOAL_PLAYBACK_LOOP:
        return unbounded - floor(unbounded);
    case MR_TASK_GOAL_PLAYBACK_PING_PONG: {
        const float cycle = fmod(unbounded, 2.0f);
        return cycle <= 1.0f ? cycle : 2.0f - cycle;
    }
    case MR_TASK_GOAL_PLAYBACK_CLAMP:
    default:
        return clamp(unbounded, 0.0f, 1.0f);
    }
}

inline TaskGoalPose taskGoalPose(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskGoalGPU& goal,
    const uint environment,
    const uint episode,
    const uint episodeStep
) {
    TaskGoalPose result{
        goal.position.xyz,
        goal.orientation,
    };
    if (goal.metadata.x == MR_TASK_GOAL_SAMPLED_EPISODE) {
        float3 rotationVector;
        for (uint component = 0u; component < 3u; ++component) {
            const float positionUnit = taskGoalRandomUnit(
                dispatch,
                goal,
                environment,
                episode,
                component
            );
            const float rotationUnit = taskGoalRandomUnit(
                dispatch,
                goal,
                environment,
                episode,
                3u + component
            );
            result.position[component] += mix(
                goal.positionOffsetLower[component],
                goal.positionOffsetUpper[component],
                positionUnit
            );
            rotationVector[component] = mix(
                goal.rotationVectorLower[component],
                goal.rotationVectorUpper[component],
                rotationUnit
            );
        }
        result.orientation = normalizedQuaternionOrIdentity(
            quaternionProduct(
                goal.orientation,
                taskQuaternionExponential(rotationVector)
            )
        );
    } else if (goal.metadata.x == MR_TASK_GOAL_TRAJECTORY) {
        const float progress = taskGoalTrajectoryProgress(
            dispatch,
            goal,
            episodeStep
        );
        result.position = mix(
            goal.position.xyz,
            goal.targetPosition.xyz,
            progress
        );
        result.orientation = taskQuaternionSlerp(
            goal.orientation,
            goal.targetOrientation,
            progress
        );
    }
    return result;
}

inline bool relativeFrameObservationOpcode(const uint opcode) {
    return opcode == MR_TASK_OBSERVE_FRAME_RELATIVE_POSITION ||
        opcode == MR_TASK_OBSERVE_FRAME_RELATIVE_ORIENTATION ||
        opcode == MR_TASK_OBSERVE_FRAME_RELATIVE_LINEAR_VELOCITY ||
        opcode == MR_TASK_OBSERVE_FRAME_RELATIVE_ANGULAR_VELOCITY;
}

inline bool frameObservationOpcode(const uint opcode) {
    return
        (opcode >= MR_TASK_OBSERVE_FRAME_POSITION_WORLD &&
         opcode <= MR_TASK_OBSERVE_FRAME_GOAL_ORIENTATION_ERROR) ||
        relativeFrameObservationOpcode(opcode) ||
        opcode == MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_WORLD ||
        opcode == MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_HEADING ||
        opcode == MR_TASK_OBSERVE_FRAME_ANGULAR_VELOCITY_WORLD ||
        opcode == MR_TASK_OBSERVE_FRAME_LINEAR_JACOBIAN_WORLD ||
        opcode == MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD;
}

inline float taskFrameJacobianValue(
    device const MRTaskDispatchGPU& dispatch,
    const MRTaskObservationOperatorGPU operation,
    device const MRTaskKinematicFrameGPU* kinematicFrames,
    device const float* spatialJacobians,
    const uint environment
) {
    device const MRTaskKinematicFrameGPU& query =
        kinematicFrames[operation.source.y];
    if (query.layout.x == MR_INVALID_INDEX ||
        query.layout.y == MR_INVALID_INDEX ||
        query.coordinates.x == 0u ||
        operation.auxiliary.z < query.coordinates.y ||
        operation.auxiliary.z >=
            query.coordinates.y + query.coordinates.x) {
        // A generalized coordinate owned by another disconnected
        // articulation has an exact zero Jacobian entry.
        return operation.transform.y;
    }
    const uint row =
        operation.source.x ==
            MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD
        ? 3u + operation.source.z
        : operation.source.z;
    const uint localDof =
        operation.auxiliary.z - query.coordinates.y;
    const ulong ownerBase =
        static_cast<ulong>(dispatch.counts.x) *
            query.layout.z +
        static_cast<ulong>(environment) *
            query.layout.w;
    const ulong valueIndex =
        ownerBase +
        static_cast<ulong>(
            query.layout.y * 6u + row
        ) * query.coordinates.x +
        localDof;
    return
        operation.transform.x * spatialJacobians[valueIndex] +
        operation.transform.y;
}

inline bool sensorObservationOpcode(const uint opcode) {
    return opcode == MR_TASK_OBSERVE_SENSOR_VALUE ||
        opcode == MR_TASK_OBSERVE_SENSOR_VALIDITY;
}

inline ulong taskSensorFingerprint(
    device const MRTaskProgramHeaderGPU& program
) {
    return static_cast<ulong>(program.typedCounts.z) |
        (static_cast<ulong>(program.typedCounts.w) << 32u);
}

inline float taskSensorObservationValue(
    device const MRSensorProgramHeaderGPU& sensorProgram,
    const MRTaskObservationOperatorGPU operation,
    device const float* sensorOutputs,
    device const MRSensorSampleMetadataGPU* sensorMetadata,
    const uint environment
) {
    float value = 0.0f;
    if (operation.source.x == MR_TASK_OBSERVE_SENSOR_VALUE) {
        value = sensorOutputs[
            environment * sensorProgram.counts.y +
            operation.auxiliary.z + operation.source.z
        ];
    } else {
        const uint validity = sensorMetadata[
            environment * sensorProgram.counts.x +
            operation.source.y
        ].ageValidityAndLayout.y;
        value =
            (validity & (1u << operation.source.z)) != 0u
            ? 1.0f
            : 0.0f;
    }
    return operation.transform.x * value + operation.transform.y;
}

inline float taskFrameObservationValue(
    device const MRTaskDispatchGPU& dispatch,
    const MRTaskObservationOperatorGPU operation,
    device const MRTaskFrameGPU* frames,
    device const MRTaskKinematicFrameGPU* kinematicFrames,
    device const MRTaskGoalGPU* goals,
    device const float* spatialJacobians,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const float4 bodyPosition,
    const float4 bodyOrientation,
    const float3 bodyLinearVelocity,
    const float3 bodyAngularVelocity,
    const float4 referenceBodyPosition,
    const float4 referenceBodyOrientation,
    const float3 referenceBodyLinearVelocity,
    const float3 referenceBodyAngularVelocity
) {
    device const MRTaskFrameGPU& frame =
        frames[operation.source.y];
    const TaskFramePose pose = taskFramePose(
        frame,
        bodyPosition,
        bodyOrientation
    );
    if (relativeFrameObservationOpcode(operation.source.x)) {
        device const MRTaskFrameGPU& referenceFrame =
            frames[operation.auxiliary.z];
        const TaskFramePose referencePose = taskFramePose(
            referenceFrame,
            referenceBodyPosition,
            referenceBodyOrientation
        );
        float3 relative = float3(0.0f);
        if (operation.source.x ==
                MR_TASK_OBSERVE_FRAME_RELATIVE_POSITION) {
            relative = rotateInverse(
                referencePose.orientation,
                pose.position - referencePose.position
            );
        } else if (
            operation.source.x ==
                MR_TASK_OBSERVE_FRAME_RELATIVE_ORIENTATION
        ) {
            relative = taskRotationVector(
                quaternionProduct(
                    quaternionConjugate(referencePose.orientation),
                    pose.orientation
                )
            );
        } else if (
            operation.source.x ==
                MR_TASK_OBSERVE_FRAME_RELATIVE_LINEAR_VELOCITY
        ) {
            const float3 targetOffset = rotate(
                bodyOrientation,
                frame.localPosition.xyz
            );
            const float3 referenceOffset = rotate(
                referenceBodyOrientation,
                referenceFrame.localPosition.xyz
            );
            const float3 targetVelocity =
                bodyLinearVelocity +
                cross(bodyAngularVelocity, targetOffset);
            const float3 referenceVelocity =
                referenceBodyLinearVelocity +
                cross(
                    referenceBodyAngularVelocity,
                    referenceOffset
                );
            relative = rotateInverse(
                referencePose.orientation,
                targetVelocity - referenceVelocity -
                    cross(
                        referenceBodyAngularVelocity,
                        pose.position - referencePose.position
                    )
            );
        } else {
            relative = rotateInverse(
                referencePose.orientation,
                bodyAngularVelocity -
                    referenceBodyAngularVelocity
            );
        }
        const float value = relative[operation.source.z];
        return operation.transform.x * value +
            operation.transform.y;
    }
    float value = 0.0f;
    switch (operation.source.x) {
    case MR_TASK_OBSERVE_FRAME_LINEAR_JACOBIAN_WORLD:
    case MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD:
        return taskFrameJacobianValue(
            dispatch,
            operation,
            kinematicFrames,
            spatialJacobians,
            environment
        );
    case MR_TASK_OBSERVE_FRAME_POSITION_WORLD:
        value = pose.position[operation.source.z];
        break;
    case MR_TASK_OBSERVE_FRAME_ORIENTATION_WORLD:
        value = pose.orientation[operation.source.z];
        break;
    case MR_TASK_OBSERVE_FRAME_GOAL_POSITION_ERROR: {
        device const MRTaskGoalGPU& goal =
            goals[operation.auxiliary.z];
        const TaskGoalPose goalPose = taskGoalPose(
            dispatch,
            goal,
            environment,
            episode,
            episodeStep
        );
        value =
            (goalPose.position - pose.position)[
                operation.source.z
            ];
        break;
    }
    case MR_TASK_OBSERVE_FRAME_GOAL_ORIENTATION_ERROR: {
        device const MRTaskGoalGPU& goal =
            goals[operation.auxiliary.z];
        const TaskGoalPose goalPose = taskGoalPose(
            dispatch,
            goal,
            environment,
            episode,
            episodeStep
        );
        value = taskOrientationError(
            pose.orientation,
            goalPose.orientation
        )[operation.source.z];
        break;
    }
    case MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_WORLD: {
        const float3 offset = rotate(
            bodyOrientation,
            frame.localPosition.xyz
        );
        value = (
            bodyLinearVelocity +
            cross(bodyAngularVelocity, offset)
        )[operation.source.z];
        break;
    }
    case MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_HEADING: {
        const float3 offset = rotate(
            bodyOrientation,
            frame.localPosition.xyz
        );
        const float3 worldVelocity =
            bodyLinearVelocity +
            cross(bodyAngularVelocity, offset);
        float2 heading = rotate(
            pose.orientation,
            float3(1.0f, 0.0f, 0.0f)
        ).xy;
        heading *= rsqrt(max(dot(heading, heading), 1.0e-12f));
        const float2 headingVelocity = float2(
            dot(heading, worldVelocity.xy),
            dot(float2(-heading.y, heading.x), worldVelocity.xy)
        );
        value = headingVelocity[operation.source.z];
        break;
    }
    case MR_TASK_OBSERVE_FRAME_ANGULAR_VELOCITY_WORLD:
        value = bodyAngularVelocity[operation.source.z];
        break;
    default:
        break;
    }
    return operation.transform.x * value +
        operation.transform.y;
}

inline float3 normalizedOr(
    const float3 value,
    const float3 fallback
) {
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f &&
            isfinite(lengthSquared)
        ? value * rsqrt(lengthSquared)
        : fallback;
}

inline bool bodyMember(
    const uint body,
    device const MRTaskContactGroupGPU& group,
    device const uint* members
) {
    for (uint local = 0u;
         local < group.members.y;
         ++local) {
        if (members[group.members.x + local] == body) {
            return true;
        }
    }
    return false;
}

inline float3 stableContactTangent(const float3 normal) {
    const float3 absoluteNormal = abs(normal);
    const float3 reference =
        absoluteNormal.x <= absoluteNormal.y &&
        absoluteNormal.x <= absoluteNormal.z
        ? float3(1.0f, 0.0f, 0.0f)
        : absoluteNormal.y <= absoluteNormal.z
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(0.0f, 0.0f, 1.0f);
    return normalizedOr(
        cross(reference, normal),
        float3(1.0f, 0.0f, 0.0f)
    );
}

inline float surfaceHeight(
    device const MRTaskProgramHeaderGPU& program,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRBodyStateGPU* sceneBodies,
    const float2 worldPosition
) {
    if (program.terrain.x == MR_INVALID_INDEX ||
        program.terrain.y == MR_INVALID_INDEX) {
        return 0.0f;
    }
    const MRBodyStateGPU scene =
        sceneBodies[program.terrain.x];
    const MRShapeGPU shape = shapes[program.terrain.y];
    if (shape.shapeType != MR_SHAPE_HEIGHTFIELD ||
        program.terrain.z == MR_INVALID_INDEX) {
        return
            scene.position.z +
            shape.localPosition.z;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[program.terrain.z];
    if (geometry.kind != MR_GEOMETRY_HEIGHTFIELD ||
        geometry.vertexCount == 0u ||
        !(geometry.localLower.w > 0.0f)) {
        return scene.position.z;
    }

    const float3 sceneLocal = rotateInverse(
        scene.orientation,
        float3(worldPosition, scene.position.z) -
            scene.position.xyz
    );
    const float3 shapeLocal = rotateInverse(
        shape.localRotation,
        sceneLocal - shape.localPosition.xyz
    );
    const float spacing = geometry.localLower.w;
    const uint width = max(
        1u,
        uint(round(
            (geometry.localUpper.x -
             geometry.localLower.x) /
                spacing
        )) + 1u
    );
    const uint height = max(
        1u,
        uint(round(
            (geometry.localUpper.y -
             geometry.localLower.y) /
                spacing
        )) + 1u
    );
    if (width * height > geometry.vertexCount) {
        return scene.position.z;
    }
    const float gridX = clamp(
        (shapeLocal.x - geometry.localLower.x) /
            spacing,
        0.0f,
        float(width - 1u)
    );
    const float gridY = clamp(
        (shapeLocal.y - geometry.localLower.y) /
            spacing,
        0.0f,
        float(height - 1u)
    );
    const uint x0 = min(uint(floor(gridX)), width - 1u);
    const uint y0 = min(uint(floor(gridY)), height - 1u);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);
    const float tx = gridX - float(x0);
    const float ty = gridY - float(y0);
    const uint base = geometry.vertexOffset;
    const float h00 =
        geometryVertices[base + y0 * width + x0].z;
    const float h10 =
        geometryVertices[base + y0 * width + x1].z;
    const float h01 =
        geometryVertices[base + y1 * width + x0].z;
    const float h11 =
        geometryVertices[base + y1 * width + x1].z;
    const float localHeight = mix(
        mix(h00, h10, tx),
        mix(h01, h11, tx),
        ty
    );
    return
        scene.position.z +
        shape.localPosition.z +
        shape.dimensions.z * localHeight;
}

inline float4 rootOrientation(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) == 0u) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return float4(
        q[program.root.z + 3u],
        q[program.root.z + 4u],
        q[program.root.z + 5u],
        q[program.root.z + 6u]
    );
}

inline float3 rootReferenceOffset(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) == 0u) {
        return float3(0.0f);
    }
    return rotate(
        rootOrientation(program, q),
        program.rootReference.xyz
    );
}

inline float3 rootWorldPosition(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) == 0u) {
        return float3(0.0f);
    }
    return float3(
        q[program.root.z + 0u],
        q[program.root.z + 1u],
        q[program.root.z + 2u]
    ) + rootReferenceOffset(program, q);
}

inline float3 rootWorldLinearVelocity(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q,
    device const float* v
) {
    if ((program.schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) == 0u) {
        return float3(0.0f);
    }
    const float3 offset =
        rootReferenceOffset(program, q);
    return float3(
        v[program.root.w + 0u],
        v[program.root.w + 1u],
        v[program.root.w + 2u]
    ) + cross(
        float3(
            v[program.root.w + 3u],
            v[program.root.w + 4u],
            v[program.root.w + 5u]
        ),
        offset
    );
}

inline float rootHeight(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    return rootWorldPosition(program, q).z;
}

inline bool desiredSupportContact(
    const MRTaskContactGroupGPU group,
    const float phase
) {
    float normalized =
        fmod(phase + group.gait.x, kTwoPi);
    if (normalized < 0.0f) {
        normalized += kTwoPi;
    }
    normalized /= kTwoPi;
    return normalized < group.gait.y;
}

inline float cleanObservation(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskObservationOperatorGPU operation,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const MRTaskFrameGPU* frames,
    device const MRTaskKinematicFrameGPU* kinematicFrames,
    device const MRTaskGoalGPU* goals,
    device const float* spatialJacobians,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const MRBodyStateGPU* bodyStates,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* earlierAction,
    device const float* previousJointVelocity,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices
) {
    const float4 orientation = rootOrientation(program, q);
    float value = 0.0f;
    switch (operation.source.x) {
    case MR_TASK_OBSERVE_ROOT_ANGULAR_VELOCITY_LOCAL:
        if ((program.schedule.w &
             MR_TASK_PROGRAM_FLOATING_ROOT) == 0u) {
            break;
        }
        value = rotateInverse(
            orientation,
            float3(
                v[program.root.w + 3u],
                v[program.root.w + 4u],
                v[program.root.w + 5u]
            )
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_PROJECTED_GRAVITY:
        value = normalizedOr(
            rotateInverse(
                orientation,
                float3(0.0f, 0.0f, -1.0f)
            ),
            float3(0.0f, 0.0f, -1.0f)
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_COMMAND:
        value = state.commandAndPhase[operation.source.y];
        break;
    case MR_TASK_OBSERVE_JOINT_POSITION_ERROR: {
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        value =
            q[binding.indices.z] -
            defaultQ[binding.indices.z];
        break;
    }
    case MR_TASK_OBSERVE_JOINT_VELOCITY:
        value = v[
            actions[operation.source.y].indices.w
        ];
        break;
    case MR_TASK_OBSERVE_PREVIOUS_ACTION:
        value = previousAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_JOINT_ACCELERATION: {
        const uint action = operation.source.y;
        value = (
            v[actions[action].indices.w] -
            previousJointVelocity[action]
        ) / dispatch.timing.x;
        break;
    }
    case MR_TASK_OBSERVE_ACTION_DELTA:
        value = previousAction[operation.source.y] -
            earlierAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_JOINT_SOFT_LIMIT_VIOLATION: {
        const float position =
            q[actions[operation.source.y].indices.z];
        value =
            max(operation.parameters.x - position, 0.0f) +
            max(position - operation.parameters.y, 0.0f);
        break;
    }
    case MR_TASK_OBSERVE_MECHANICAL_POWER:
        value = state.powerReturnMetric.x;
        break;
    case MR_TASK_OBSERVE_DESIRED_SUPPORT_CONTACT: {
        const float phase = fmod(
            state.commandAndPhase.w +
                kTwoPi * dispatch.timing.x /
                    program.taskScalars.x,
            kTwoPi
        );
        value = float(desiredSupportContact(
            contactGroups[operation.source.y],
            phase
        ));
        break;
    }
    case MR_TASK_OBSERVE_ROOT_LINEAR_VELOCITY_LOCAL:
        value = rotateInverse(
            orientation,
            rootWorldLinearVelocity(program, q, v)
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_ROOT_HEIGHT:
        value = rootHeight(program, q);
        break;
    case MR_TASK_OBSERVE_CONTACT_METRIC: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = compactContact[
            group.members.w + operation.source.z
        ];
        break;
    }
    case MR_TASK_OBSERVE_TERRAIN_HEIGHT: {
        if ((program.schedule.w &
             MR_TASK_PROGRAM_TERRAIN) == 0u) {
            value = 0.0f;
            break;
        }
        const float3 offset = rotate(
            orientation,
            terrainSamples[operation.source.y].xyz
        );
        const float3 rootPosition =
            rootWorldPosition(program, q);
        value =
            surfaceHeight(
                program,
                shapes,
                geometryHeaders,
                geometryVertices,
                sceneBodies,
                rootPosition.xy + offset.xy
            ) -
            rootPosition.z;
        break;
    }
    case MR_TASK_OBSERVE_BODY_PARAMETER_MEAN: {
        float total = 0.0f;
        for (uint body = 0u;
             body < program.articulation.y;
             ++body) {
            total += bodyParameters[
                program.articulation.x + body
            ][operation.source.z];
        }
        value = total /
            max(float(program.articulation.y), 1.0f);
        break;
    }
    case MR_TASK_OBSERVE_BODY_PARAMETER:
        value = bodyParameters[
            operation.source.y
        ][operation.source.z];
        break;
    case MR_TASK_OBSERVE_CONTROLLER_PARAMETER:
        value =
            controllerParameters[0][operation.source.z];
        break;
    case MR_TASK_OBSERVE_CONTACT_WRENCH_LOCAL: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = compactContact[
            group.reference.y + operation.source.z
        ];
        break;
    }
    case MR_TASK_OBSERVE_FRAME_POSITION_WORLD:
    case MR_TASK_OBSERVE_FRAME_ORIENTATION_WORLD:
    case MR_TASK_OBSERVE_FRAME_GOAL_POSITION_ERROR:
    case MR_TASK_OBSERVE_FRAME_GOAL_ORIENTATION_ERROR:
    case MR_TASK_OBSERVE_FRAME_RELATIVE_POSITION:
    case MR_TASK_OBSERVE_FRAME_RELATIVE_ORIENTATION:
    case MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_WORLD:
    case MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_HEADING:
    case MR_TASK_OBSERVE_FRAME_ANGULAR_VELOCITY_WORLD:
    case MR_TASK_OBSERVE_FRAME_RELATIVE_LINEAR_VELOCITY:
    case MR_TASK_OBSERVE_FRAME_RELATIVE_ANGULAR_VELOCITY:
    case MR_TASK_OBSERVE_FRAME_LINEAR_JACOBIAN_WORLD:
    case MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD: {
        device const MRTaskFrameGPU& frame =
            frames[operation.source.y];
        const MRBodyStateGPU body = taskFrameBodyState(
            frame,
            bodyStates,
            sceneBodies
        );
        MRBodyStateGPU referenceBody = body;
        if (relativeFrameObservationOpcode(
                operation.source.x
            )) {
            device const MRTaskFrameGPU& referenceFrame =
                frames[operation.auxiliary.z];
            referenceBody = taskFrameBodyState(
                referenceFrame,
                bodyStates,
                sceneBodies
            );
        }
        return taskFrameObservationValue(
            dispatch,
            operation,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            episode,
            episodeStep,
            body.position,
            body.orientation,
            body.linearVelocityAndInverseMass.xyz,
            body.angularVelocity.xyz,
            referenceBody.position,
            referenceBody.orientation,
            referenceBody.linearVelocityAndInverseMass.xyz,
            referenceBody.angularVelocity.xyz
        );
    }
    default:
        value = 0.0f;
        break;
    }
    return
        operation.transform.x * value +
        operation.transform.y;
}

inline void writeFrame(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const MRTaskObservationOperatorGPU* actorOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const MRTaskFrameGPU* frames,
    device const MRTaskKinematicFrameGPU* kinematicFrames,
    device const MRTaskGoalGPU* goals,
    device const float* spatialJacobians,
    device const MRBodyStateGPU* bodyStates,
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* earlierAction,
    device const float* previousJointVelocity,
    device const float* sensorBias,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* actor,
    device float* clean
) {
    float3 normalizedGravity =
        float3(0.0f, 0.0f, -1.0f);
    bool haveNormalizedGravity = false;
    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        if (operation.source.x !=
                MR_TASK_OBSERVE_PROJECTED_GRAVITY ||
            (operation.source.w &
             MR_TASK_OBSERVATION_NORMALIZE_VECTOR3) == 0u) {
            continue;
        }
        const uint component = operation.source.z;
        float value = cleanObservation(
            dispatch,
            program,
            operation,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            episode,
            episodeStep,
            bodyStates,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            earlierAction,
            previousJointVelocity,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
        value +=
            operation.transform.z *
            randomSigned(
                dispatch,
                environment,
                episode,
                episodeStep,
                operation.auxiliary.y
            );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            value += sensorBias[operation.auxiliary.x];
        }
        normalizedGravity[component] = value;
        haveNormalizedGravity = true;
    }
    if (haveNormalizedGravity) {
        normalizedGravity = normalizedOr(
            normalizedGravity,
            float3(0.0f, 0.0f, -1.0f)
        );
    }

    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        if (sensorObservationOpcode(operation.source.x)) {
            // SensorIR owns this sample. The boundary refresh kernel writes
            // it after reset or after the accepted physics state.
            continue;
        }
        const float value = cleanObservation(
            dispatch,
            program,
            operation,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            episode,
            episodeStep,
            bodyStates,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            earlierAction,
            previousJointVelocity,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
        clean[index] = value;
        if (operation.source.x ==
                MR_TASK_OBSERVE_PROJECTED_GRAVITY &&
            (operation.source.w &
             MR_TASK_OBSERVATION_NORMALIZE_VECTOR3) != 0u) {
            actor[index] =
                normalizedGravity[operation.source.z];
            continue;
        }
        float corrupted =
            value +
            operation.transform.z *
                randomSigned(
                    dispatch,
                    environment,
                    episode,
                    episodeStep,
                    operation.auxiliary.y
                );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            corrupted += sensorBias[operation.auxiliary.x];
        }
        actor[index] = corrupted;
    }
}

inline void writeCriticFrame(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const MRTaskObservationOperatorGPU* criticOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const MRTaskFrameGPU* frames,
    device const MRTaskKinematicFrameGPU* kinematicFrames,
    device const MRTaskGoalGPU* goals,
    device const float* spatialJacobians,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const MRBodyStateGPU* bodyStates,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* earlierAction,
    device const float* previousJointVelocity,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* output
) {
    for (uint index = 0u;
         index < program.counts0.z;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            criticOperators[index];
        if (sensorObservationOpcode(operation.source.x)) {
            continue;
        }
        output[index] = cleanObservation(
            dispatch,
            program,
            operation,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            episode,
            episodeStep,
            bodyStates,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            earlierAction,
            previousJointVelocity,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
    }
}

inline void publishCritic(
    device const MRTaskProgramHeaderGPU& program,
    device const float* cleanHistory,
    device const float* criticHistory,
    device float* output
) {
    uint outputIndex = 0u;
    if ((program.schedule.w &
         MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u) {
        const uint historyElements =
            program.layout.x * program.layout.y;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            output[outputIndex++] = cleanHistory[index];
        }
    }
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    for (uint index = 0u;
         index < criticHistoryElements;
         ++index) {
        output[outputIndex++] = criticHistory[index];
    }
}

inline uint durationSteps(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const uint channel,
    const float lowerSeconds,
    const float upperSeconds
) {
    return max(
        1u,
        uint(floor(
            randomRange(
                dispatch,
                environment,
                episode,
                episodeStep,
                channel,
                lowerSeconds,
                upperSeconds
            ) /
            dispatch.timing.x
        ))
    );
}

inline float3 sampledCommand(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const MRTaskCommandOperatorGPU* commands,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const uint curriculum
) {
    float3 command = float3(0.0f);
    for (uint commandIndex = 0u;
         commandIndex < program.curriculum.w;
         ++commandIndex) {
        const MRTaskCommandOperatorGPU operation =
            commands[commandIndex];
        const float expansion =
            float(curriculum) * operation.curriculum.x;
        const float lower = max(
            operation.range.x - expansion,
            operation.range.z
        );
        const float upper = min(
            operation.range.y + expansion,
            operation.range.w
        );
        command[commandIndex] = randomRange(
            dispatch,
            environment,
            episode,
            episodeStep,
            16u + commandIndex,
            lower,
            upper
        );
    }
    if (randomUnit(
            dispatch,
            environment,
            episode,
            episodeStep,
            19u
        ) < program.commandSchedule.x) {
        command = float3(0.0f);
    }
    return command;
}

} // namespace

// Journals only task state that can be mutated before a physics transaction
// is known to have committed. Task state, action delay, and compact contact
// metrics change on every step. The larger observation/randomization state is
// copied only when this environment will reset.
kernel void mr_task_checkpoint_state(
    device const MRTaskDispatchGPU& dispatch
        [[buffer(MR_TASK_TRANSACTION_DISPATCH)]],
    device const MRTaskProgramHeaderGPU& program
        [[buffer(MR_TASK_TRANSACTION_PROGRAM)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_TASK_TRANSACTION_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_TASK_TRANSACTION_RESET_MASKS)]],
    device const MRTaskStateGPU* states
        [[buffer(MR_TASK_TRANSACTION_STATE)]],
    device const float* actionHistory
        [[buffer(MR_TASK_TRANSACTION_ACTION_HISTORY)]],
    device const float* actorHistory
        [[buffer(MR_TASK_TRANSACTION_ACTOR_HISTORY)]],
    device const float* cleanHistory
        [[buffer(MR_TASK_TRANSACTION_CLEAN_HISTORY)]],
    device const float* criticHistory
        [[buffer(MR_TASK_TRANSACTION_CRITIC_HISTORY)]],
    device const float* previousJointVelocity
        [[buffer(MR_TASK_TRANSACTION_PREVIOUS_JOINT_VELOCITY)]],
    device const float* encoderBias
        [[buffer(MR_TASK_TRANSACTION_ENCODER_BIAS)]],
    device const float4* bodyParameters
        [[buffer(MR_TASK_TRANSACTION_BODY_PARAMETERS)]],
    device const float4* controllerParameters
        [[buffer(MR_TASK_TRANSACTION_CONTROLLER_PARAMETERS)]],
    device const float* contactCompact
        [[buffer(MR_TASK_TRANSACTION_CONTACT_COMPACT)]],
    device MRTaskStateGPU* checkpointStates
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_STATE)]],
    device float* checkpointActionHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ACTION_HISTORY)]],
    device float* checkpointActorHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ACTOR_HISTORY)]],
    device float* checkpointCleanHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CLEAN_HISTORY)]],
    device float* checkpointCriticHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CRITIC_HISTORY)]],
    device float* checkpointPreviousJointVelocity
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_PREVIOUS_JOINT_VELOCITY)]],
    device float* checkpointEncoderBias
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ENCODER_BIAS)]],
    device float4* checkpointBodyParameters
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_BODY_PARAMETERS)]],
    device float4* checkpointControllerParameters
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CONTROLLER_PARAMETERS)]],
    device float* checkpointContactCompact
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CONTACT_COMPACT)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }

    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    const MRTaskStateGPU state = states[environment];
    const bool reset =
        state.status.x == 0u ||
        state.status.y != 0u ||
        resetMasks[maskIndex] != 0u;
    checkpointStates[environment] = state;

    const uint actionHistoryStride =
        program.layout.w * program.counts0.x;
    const uint actionHistoryBase =
        environment * actionHistoryStride;
    for (uint index = 0u;
         index < actionHistoryStride;
         ++index) {
        checkpointActionHistory[actionHistoryBase + index] =
            actionHistory[actionHistoryBase + index];
    }
    const uint contactBase = environment * program.layout.z;
    for (uint index = 0u;
         index < program.layout.z;
         ++index) {
        checkpointContactCompact[contactBase + index] =
            contactCompact[contactBase + index];
    }
    if (!reset) {
        return;
    }

    const uint historyStride =
        program.layout.x * program.layout.y;
    const uint historyBase = environment * historyStride;
    for (uint index = 0u; index < historyStride; ++index) {
        checkpointActorHistory[historyBase + index] =
            actorHistory[historyBase + index];
        checkpointCleanHistory[historyBase + index] =
            cleanHistory[historyBase + index];
    }
    const uint criticHistoryStride =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryStride;
    for (uint index = 0u;
         index < criticHistoryStride;
         ++index) {
        checkpointCriticHistory[criticHistoryBase + index] =
            criticHistory[criticHistoryBase + index];
    }
    const uint previousVelocityBase =
        environment * program.counts0.x;
    for (uint index = 0u;
         index < program.counts0.x;
         ++index) {
        checkpointPreviousJointVelocity[
            previousVelocityBase + index
        ] = previousJointVelocity[previousVelocityBase + index];
    }
    const uint biasBase = environment * program.counts2.z;
    for (uint index = 0u;
         index < program.counts2.z;
         ++index) {
        checkpointEncoderBias[biasBase + index] =
            encoderBias[biasBase + index];
    }
    const uint bodyBase = environment * dispatch.strides.y;
    for (uint index = 0u;
         index < dispatch.strides.y;
         ++index) {
        checkpointBodyParameters[bodyBase + index] =
            bodyParameters[bodyBase + index];
    }
    checkpointControllerParameters[environment] =
        controllerParameters[environment];
}

// Restores a failed environment after TaskIR completion and SensorIR binding
// have produced the typed failure outputs. Those outputs remain inspectable;
// persistent task state returns to the prior committed control boundary.
kernel void mr_task_restore_failed_state(
    device const MRTaskDispatchGPU& dispatch
        [[buffer(MR_TASK_TRANSACTION_DISPATCH)]],
    device const MRTaskProgramHeaderGPU& program
        [[buffer(MR_TASK_TRANSACTION_PROGRAM)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_TASK_TRANSACTION_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_TASK_TRANSACTION_RESET_MASKS)]],
    device const MRMetalWorldStatusGPU* worldStatuses
        [[buffer(MR_TASK_TRANSACTION_WORLD_STATUSES)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(MR_TASK_TRANSACTION_CONTACT_STATUSES)]],
    device MRTaskStateGPU* states
        [[buffer(MR_TASK_TRANSACTION_STATE)]],
    device float* actionHistory
        [[buffer(MR_TASK_TRANSACTION_ACTION_HISTORY)]],
    device float* actorHistory
        [[buffer(MR_TASK_TRANSACTION_ACTOR_HISTORY)]],
    device float* cleanHistory
        [[buffer(MR_TASK_TRANSACTION_CLEAN_HISTORY)]],
    device float* criticHistory
        [[buffer(MR_TASK_TRANSACTION_CRITIC_HISTORY)]],
    device float* previousJointVelocity
        [[buffer(MR_TASK_TRANSACTION_PREVIOUS_JOINT_VELOCITY)]],
    device float* encoderBias
        [[buffer(MR_TASK_TRANSACTION_ENCODER_BIAS)]],
    device float4* bodyParameters
        [[buffer(MR_TASK_TRANSACTION_BODY_PARAMETERS)]],
    device float4* controllerParameters
        [[buffer(MR_TASK_TRANSACTION_CONTROLLER_PARAMETERS)]],
    device float* contactCompact
        [[buffer(MR_TASK_TRANSACTION_CONTACT_COMPACT)]],
    device const MRTaskStateGPU* checkpointStates
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_STATE)]],
    device const float* checkpointActionHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ACTION_HISTORY)]],
    device const float* checkpointActorHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ACTOR_HISTORY)]],
    device const float* checkpointCleanHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CLEAN_HISTORY)]],
    device const float* checkpointCriticHistory
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CRITIC_HISTORY)]],
    device const float* checkpointPreviousJointVelocity
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_PREVIOUS_JOINT_VELOCITY)]],
    device const float* checkpointEncoderBias
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_ENCODER_BIAS)]],
    device const float4* checkpointBodyParameters
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_BODY_PARAMETERS)]],
    device const float4* checkpointControllerParameters
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CONTROLLER_PARAMETERS)]],
    device const float* checkpointContactCompact
        [[buffer(MR_TASK_TRANSACTION_CHECKPOINT_CONTACT_COMPACT)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        (worldStatuses[environment].code == MR_STEP_SUCCESS &&
         contactStatuses[environment].code == MR_STEP_SUCCESS)) {
        return;
    }

    states[environment] = checkpointStates[environment];
    const uint actionHistoryStride =
        program.layout.w * program.counts0.x;
    const uint actionHistoryBase =
        environment * actionHistoryStride;
    for (uint index = 0u;
         index < actionHistoryStride;
         ++index) {
        actionHistory[actionHistoryBase + index] =
            checkpointActionHistory[actionHistoryBase + index];
    }
    const uint contactBase = environment * program.layout.z;
    for (uint index = 0u;
         index < program.layout.z;
         ++index) {
        contactCompact[contactBase + index] =
            checkpointContactCompact[contactBase + index];
    }
    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    if (resetMasks[maskIndex] == 0u) {
        return;
    }

    const uint historyStride =
        program.layout.x * program.layout.y;
    const uint historyBase = environment * historyStride;
    for (uint index = 0u; index < historyStride; ++index) {
        actorHistory[historyBase + index] =
            checkpointActorHistory[historyBase + index];
        cleanHistory[historyBase + index] =
            checkpointCleanHistory[historyBase + index];
    }
    const uint criticHistoryStride =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryStride;
    for (uint index = 0u;
         index < criticHistoryStride;
         ++index) {
        criticHistory[criticHistoryBase + index] =
            checkpointCriticHistory[criticHistoryBase + index];
    }
    const uint previousVelocityBase =
        environment * program.counts0.x;
    for (uint index = 0u;
         index < program.counts0.x;
         ++index) {
        previousJointVelocity[previousVelocityBase + index] =
            checkpointPreviousJointVelocity[
                previousVelocityBase + index
            ];
    }
    const uint biasBase = environment * program.counts2.z;
    for (uint index = 0u;
         index < program.counts2.z;
         ++index) {
        encoderBias[biasBase + index] =
            checkpointEncoderBias[biasBase + index];
    }
    const uint bodyBase = environment * dispatch.strides.y;
    for (uint index = 0u;
         index < dispatch.strides.y;
         ++index) {
        bodyParameters[bodyBase + index] =
            checkpointBodyParameters[bodyBase + index];
    }
    controllerParameters[environment] =
        checkpointControllerParameters[environment];
}

kernel void mr_task_observe(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device uint* resetMasks [[buffer(6)]],
    device float* resetQ [[buffer(7)]],
    device float* resetV [[buffer(8)]],
    device MRBodyStateGPU* resetScene [[buffer(9)]],
    device float* sourceQ [[buffer(10)]],
    device float* sourceV [[buffer(11)]],
    device const MRBodyStateGPU* initialScene [[buffer(12)]],
    device const MRBodyStateGPU* bodyStates [[buffer(14)]],
    device MRBodyStateGPU* sourceScene [[buffer(15)]],
    device const float* defaultQ [[buffer(16)]],
    device MRTaskStateGPU* taskStates [[buffer(17)]],
    device float* actionHistory [[buffer(18)]],
    device float* actorHistory [[buffer(19)]],
    device float* cleanHistory [[buffer(20)]],
    device float* previousJointVelocity [[buffer(21)]],
    device float* sensorBias [[buffer(22)]],
    device float4* bodyParameters [[buffer(23)]],
    device float4* controllerParameters [[buffer(24)]],
    device float* actorObservations [[buffer(25)]],
    device float* criticObservations [[buffer(26)]],
    device float* compactContact [[buffer(27)]],
    device const MRShapeGPU* shapes [[buffer(28)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(29)]],
    device const float4* geometryVertices [[buffer(30)]],
    device const MRTaskCurriculumStateGPU* curriculumState
        [[buffer(5)]],
    device float* criticHistory [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.counts.x !=
            worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION ||
        program.counts0.x == 0u ||
        program.counts0.y == 0u ||
        program.layout.y == 0u ||
        program.articulation.w == 0u ||
        program.layout.w < 2u) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    device const MRTaskObservationOperatorGPU*
        actorOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.y
            );
    device const MRTaskObservationOperatorGPU*
        criticOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.z
            );
    device const MRTaskContactGroupGPU* contactGroups =
        taskTable<MRTaskContactGroupGPU>(
            arena,
            program.offsets0.w
        );
    device const uint* contactMembers =
        taskTable<uint>(arena, program.offsets1.x);
    device const MRTaskRandomizationOperatorGPU*
        randomization =
            taskTable<MRTaskRandomizationOperatorGPU>(
                arena,
                program.offsets2.y
            );
    device const MRTaskBiasSpecGPU* biasSpecs =
        taskTable<MRTaskBiasSpecGPU>(
            arena,
            program.offsets2.z
        );
    device const float4* terrainSamples =
        taskTable<float4>(arena, program.offsets2.w);
    device const float4* terrainProfiles =
        taskTable<float4>(arena, program.offsets3.x);
    device const MRTaskCommandOperatorGPU* commandOperators =
        taskTable<MRTaskCommandOperatorGPU>(
            arena,
            program.offsets3.y
        );
    device const MRTaskFrameGPU* frames =
        taskTable<MRTaskFrameGPU>(
            arena,
            program.offsets3.z
        );
    device const MRTaskKinematicFrameGPU* kinematicFrames =
        taskKinematicFrameTable(arena, program);
    device const MRTaskGoalGPU* goals =
        taskTable<MRTaskGoalGPU>(
            arena,
            program.offsets3.w
        );

    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase =
        environment * historyElements;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint previousVelocityBase =
        environment * program.counts0.x;
    const uint biasBase =
        environment * program.counts2.z;
    const uint bodyParameterBase =
        environment * dispatch.strides.y;
    const uint bodyBase =
        environment * dispatch.strides.y;
    device const float* spatialJacobians =
        reinterpret_cast<device const float*>(
            bodyStates +
            dispatch.counts.x * dispatch.strides.y
        );
    const uint contactBase =
        environment * program.layout.z;
    const uint sceneBase =
        environment * dispatch.strides.w;
    MRTaskStateGPU state = taskStates[environment];
    const uint globalCurriculum = min(
        curriculumState[0].commandLevel,
        program.curriculum.x - 1u
    );
    state.episode.z = globalCurriculum;
    const bool reset =
        state.status.x == 0u ||
        state.status.y != 0u ||
        resetMasks[maskIndex] != 0u;
    resetMasks[maskIndex] = reset ? 1u : 0u;

    if (reset) {
        const uint episode = state.episode.y + 1u;
        const uint curriculum = globalCurriculum;
        const uint terrainLevel =
            program.terrain.w == 0u
            ? 0u
            : min(
                  state.episode.w,
                  program.terrain.w - 1u
              );
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.z;
             ++coordinate) {
            resetQ[qBase + coordinate] =
                defaultQ[coordinate];
        }
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.w;
             ++coordinate) {
            resetV[vBase + coordinate] = 0.0f;
        }
        for (uint localScene = 0u;
             localScene < dispatch.strides.w;
             ++localScene) {
            MRBodyStateGPU scene =
                initialScene[sceneBase + localScene];
            scene.linearVelocityAndInverseMass.xyz =
                float3(0.0f);
            scene.angularVelocity = float4(0.0f);
            resetScene[sceneBase + localScene] = scene;
        }
        if (program.terrain.x != MR_INVALID_INDEX &&
            program.terrain.w != 0u) {
            resetScene[
                sceneBase + program.terrain.x
            ].position.xyz =
                terrainProfiles[terrainLevel].xyz;
        }
        for (uint body = 0u;
             body < dispatch.strides.y;
             ++body) {
            bodyParameters[
                bodyParameterBase + body
            ] = float4(1.0f);
        }
        controllerParameters[environment] =
            float4(1.0f, 1.0f, 0.0f, 1.0f);
        for (uint bias = 0u;
             bias < program.counts2.z;
             ++bias) {
            const MRTaskBiasSpecGPU spec =
                biasSpecs[bias];
            sensorBias[biasBase + bias] = randomRange(
                dispatch,
                environment,
                episode,
                0u,
                spec.metadata.x,
                spec.range.x,
                spec.range.y
            );
        }
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            previousJointVelocity[
                previousVelocityBase + action
            ] = 0.0f;
            for (uint delay = 0u;
                 delay < program.layout.w;
                 ++delay) {
                actionHistory[
                    delayBase +
                    delay * program.counts0.x +
                    action
                ] = 0.0f;
            }
        }
        for (uint index = 0u;
             index < program.layout.z;
             ++index) {
            compactContact[contactBase + index] = 0.0f;
        }

        uint actionDelay = 0u;
        uint observationDelay = 0u;
        for (uint index = 0u;
             index < program.counts2.y;
             ++index) {
            const MRTaskRandomizationOperatorGPU operation =
                randomization[index];
            if (curriculum < operation.target.w) {
                continue;
            }
            const uint channel = 2048u + index;
            switch (operation.target.x) {
            case MR_TASK_RANDOMIZE_ROOT_POSITION:
                for (uint component = 0u;
                     component < 3u;
                     ++component) {
                    resetQ[
                        qBase + program.root.z + component
                    ] +=
                        operation.parameters[component] *
                        randomSigned(
                            dispatch,
                            environment,
                            episode,
                            0u,
                            channel + component
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_ROOT_YAW: {
                const float yaw = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                const float halfYaw = 0.5f * yaw;
                const float4 authored = float4(
                    defaultQ[program.root.z + 3u],
                    defaultQ[program.root.z + 4u],
                    defaultQ[program.root.z + 5u],
                    defaultQ[program.root.z + 6u]
                );
                const float4 randomized =
                    quaternionProduct(
                        float4(
                            0.0f,
                            0.0f,
                            sin(halfYaw),
                            cos(halfYaw)
                        ),
                        authored
                    );
                resetQ[qBase + program.root.z + 3u] =
                    randomized.x;
                resetQ[qBase + program.root.z + 4u] =
                    randomized.y;
                resetQ[qBase + program.root.z + 5u] =
                    randomized.z;
                resetQ[qBase + program.root.z + 6u] =
                    randomized.w;
                break;
            }
            case MR_TASK_RANDOMIZE_ACTION_POSITION:
                for (uint action = 0u;
                     action < program.counts0.x;
                     ++action) {
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    resetQ[qBase + binding.indices.z] =
                        clamp(
                            defaultQ[binding.indices.z] +
                                randomRange(
                                    dispatch,
                                    environment,
                                    episode,
                                    0u,
                                    channel + action,
                                    operation.parameters.x,
                                    operation.parameters.y
                                ),
                            binding.parameters.y,
                            binding.parameters.z
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_VELOCITY:
                for (uint coordinate = 0u;
                     coordinate < dispatch.counts.w;
                     ++coordinate) {
                    resetV[vBase + coordinate] =
                        randomRange(
                            dispatch,
                            environment,
                            episode,
                            0u,
                            channel + coordinate,
                            operation.parameters.x,
                            operation.parameters.y
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_ACTION_VELOCITY:
                for (uint action = 0u;
                     action < program.counts0.x;
                     ++action) {
                    resetV[
                        vBase + actions[action].indices.w
                    ] = randomRange(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        channel + action,
                        operation.parameters.x,
                        operation.parameters.y
                    );
                }
                break;
            case MR_TASK_RANDOMIZE_BODY_PARAMETER: {
                const MRTaskContactGroupGPU group =
                    contactGroups[operation.target.y];
                const float sampled = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                for (uint local = 0u;
                     local < group.members.y;
                     ++local) {
                    const uint body =
                        contactMembers[
                            group.members.x + local
                        ];
                    bodyParameters[
                        bodyParameterBase + body
                    ][operation.target.z] = sampled;
                }
                break;
            }
            case MR_TASK_RANDOMIZE_BODY_PAYLOAD: {
                const uint body = operation.target.y;
                const float payload = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                bodyParameters[
                    bodyParameterBase + body
                ].x +=
                    payload * operation.parameters.z;
                bodyParameters[
                    bodyParameterBase + body
                ].x = max(
                    bodyParameters[
                        bodyParameterBase + body
                    ].x,
                    0.05f
                );
                break;
            }
            case MR_TASK_RANDOMIZE_CONTROLLER_PARAMETER:
                controllerParameters[environment][
                    operation.target.z
                ] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_ACTION_DELAY: {
                const uint lower =
                    uint(max(operation.parameters.x, 0.0f));
                const uint upper =
                    uint(max(
                        operation.parameters.y,
                        operation.parameters.x
                    ));
                actionDelay = min(
                    lower +
                        uint(floor(
                            float(upper - lower + 1u) *
                            randomUnit(
                                dispatch,
                                environment,
                                episode,
                                0u,
                                channel
                            )
                        )),
                    program.layout.w - 1u
                );
                break;
            }
            case MR_TASK_RANDOMIZE_OBSERVATION_DELAY: {
                const uint lower =
                    uint(max(operation.parameters.x, 0.0f));
                const uint upper =
                    uint(max(
                        operation.parameters.y,
                        operation.parameters.x
                    ));
                observationDelay = min(
                    lower +
                        uint(floor(
                            float(upper - lower + 1u) *
                            randomUnit(
                                dispatch,
                                environment,
                                episode,
                                0u,
                                channel
                            )
                        )),
                    program.schedule.y
                );
                break;
            }
            default:
                break;
            }
        }
        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;

        state.episode = uint4(
            0u,
            episode,
            curriculum,
            terrainLevel
        );
        state.schedule = uint4(
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                32u,
                program.commandSchedule.y,
                program.commandSchedule.z
            ),
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                33u,
                program.eventSchedule.y,
                program.eventSchedule.z
            ),
            actionDelay,
            observationDelay
        );
        state.status = uint4(
            1u,
            0u,
            MR_TASK_TERMINATION_CONTINUING,
            0u
        );
        state.commandAndPhase = float4(
            sampledCommand(
                dispatch,
                program,
                commandOperators,
                environment,
                episode,
                0u,
                curriculum
            ),
            0.0f
        );
        state.powerReturnMetric = float4(0.0f);

        // Reset is a transaction, not a side-band suggestion. Publish the
        // randomized state before the pre-policy kinematics pass so frame
        // observations and the subsequent physics step consume the same
        // state in this command buffer.
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.z;
             ++coordinate) {
            sourceQ[qBase + coordinate] =
                resetQ[qBase + coordinate];
        }
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.w;
             ++coordinate) {
            sourceV[vBase + coordinate] =
                resetV[vBase + coordinate];
        }
        for (uint localScene = 0u;
             localScene < dispatch.strides.w;
             ++localScene) {
            sourceScene[sceneBase + localScene] =
                resetScene[sceneBase + localScene];
        }

        device float* firstActor =
            actorHistory + historyBase;
        device float* firstClean =
            cleanHistory + historyBase;
        writeFrame(
            dispatch,
            program,
            actorOperators,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            bodyStates + bodyBase,
            terrainSamples,
            environment,
            episode,
            0u,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            actionHistory + delayBase,
            previousJointVelocity + previousVelocityBase,
            sensorBias + biasBase,
            compactContact + contactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            resetScene + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            firstActor,
            firstClean
        );
        for (uint history = 1u;
             history < program.layout.y;
             ++history) {
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = firstActor[index];
                cleanHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = firstClean[index];
            }
        }
        device float* firstCritic =
            criticHistory + criticHistoryBase;
        writeCriticFrame(
            dispatch,
            program,
            criticOperators,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            episode,
            0u,
            bodyStates + bodyBase,
            terrainSamples,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            actionHistory + delayBase,
            previousJointVelocity + previousVelocityBase,
            compactContact + contactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            resetScene + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            firstCritic
        );
        for (uint history = 1u;
             history < program.articulation.w;
             ++history) {
            for (uint index = 0u;
                 index < program.counts0.z;
                 ++index) {
                criticHistory[
                    criticHistoryBase +
                    history * program.counts0.z +
                    index
                ] = firstCritic[index];
            }
        }
    }

    taskStates[environment] = state;
    const uint actorOutputBase =
        pass.controlStep * dispatch.outputs.x +
        environment *
            program.layout.x * program.layout.y;
    const uint criticObservationSize =
        (
            (program.schedule.w &
             MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
            ? program.layout.x * program.layout.y
            : 0u
        ) +
        criticHistoryElements;
    const uint criticOutputBase =
        pass.controlStep * dispatch.outputs.y +
        environment * criticObservationSize;
    for (uint index = 0u;
         index < historyElements;
         ++index) {
        actorObservations[actorOutputBase + index] =
            actorHistory[historyBase + index];
    }
    publishCritic(
        program,
        cleanHistory + historyBase,
        criticHistory + criticHistoryBase,
        criticObservations + criticOutputBase
    );

    if (!reset && state.schedule.y == 0u &&
        (program.schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) != 0u &&
        program.eventSchedule.x > 0.0f) {
        const float progress = clamp(
            float(state.episode.z) /
                max(float(program.curriculum.x - 1u), 1.0f),
            0.0f,
            1.0f
        );
        sourceV[vBase + program.root.w + 0u] +=
            progress * program.eventSchedule.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                48u
            );
        sourceV[vBase + program.root.w + 1u] +=
            progress * program.eventSchedule.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                49u
            );
    }

}

// Reset observations must be generated from reset q, not from the body cache
// retained by the preceding episode. The host schedules generic articulated
// kinematics after mr_task_observe and before this kernel. Only reset
// environments are rewritten here: ordinary history and observation-delay
// semantics remain owned by mr_task_complete.
kernel void mr_task_refresh_frame_observations(
    device const MRTaskDispatchGPU& dispatch
        [[buffer(MR_TASK_FRAME_REFRESH_DISPATCH)]],
    device const MRTaskProgramHeaderGPU& program
        [[buffer(MR_TASK_FRAME_REFRESH_PROGRAM)]],
    device const uchar* arena
        [[buffer(MR_TASK_FRAME_REFRESH_ARENA)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_TASK_FRAME_REFRESH_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_TASK_FRAME_REFRESH_RESET_MASKS)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses
        [[buffer(MR_TASK_FRAME_REFRESH_BODY_POSES)]],
    device const MRBodyStateGPU* bodyStates
        [[buffer(MR_TASK_FRAME_REFRESH_BODY_STATES)]],
    device const MRBodyStateGPU* sceneBodies
        [[buffer(MR_TASK_FRAME_REFRESH_SCENE_BODIES)]],
    device const MRTaskStateGPU* taskStates
        [[buffer(MR_TASK_FRAME_REFRESH_TASK_STATES)]],
    device const float* sensorBias
        [[buffer(MR_TASK_FRAME_REFRESH_SENSOR_BIAS)]],
    device float* actorHistory
        [[buffer(MR_TASK_FRAME_REFRESH_ACTOR_HISTORY)]],
    device float* cleanHistory
        [[buffer(MR_TASK_FRAME_REFRESH_CLEAN_HISTORY)]],
    device float* criticHistory
        [[buffer(MR_TASK_FRAME_REFRESH_CRITIC_HISTORY)]],
    device float* actorObservations
        [[buffer(MR_TASK_FRAME_REFRESH_ACTOR_OBSERVATIONS)]],
    device float* criticObservations
        [[buffer(MR_TASK_FRAME_REFRESH_CRITIC_OBSERVATIONS)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.typedCounts.x == 0u ||
        program.layout.y == 0u ||
        program.articulation.w == 0u) {
        return;
    }
    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    if (resetMasks[maskIndex] == 0u) {
        return;
    }

    device const MRTaskObservationOperatorGPU* actorOperators =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets0.y
        );
    device const MRTaskObservationOperatorGPU* criticOperators =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets0.z
        );
    device const MRTaskFrameGPU* frames =
        taskTable<MRTaskFrameGPU>(arena, program.offsets3.z);
    device const MRTaskKinematicFrameGPU* kinematicFrames =
        taskKinematicFrameTable(arena, program);
    device const MRTaskGoalGPU* goals =
        taskTable<MRTaskGoalGPU>(arena, program.offsets3.w);

    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase = environment * historyElements;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint biasBase = environment * program.counts2.z;
    const uint bodyBase = environment * dispatch.strides.y;
    device const float* spatialJacobians =
        reinterpret_cast<device const float*>(
            bodyStates +
            dispatch.counts.x * dispatch.strides.y
        );
    const uint sceneBase = environment * dispatch.strides.w;
    const uint actorOutputBase =
        pass.controlStep * dispatch.outputs.x +
        environment * historyElements;
    const uint cleanOutputOffset =
        (program.schedule.w &
         MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
        ? 0u
        : MR_INVALID_INDEX;
    const uint criticObservationSize =
        (cleanOutputOffset != MR_INVALID_INDEX
             ? historyElements
             : 0u) +
        criticHistoryElements;
    const uint criticOutputBase =
        pass.controlStep * dispatch.outputs.y +
        environment * criticObservationSize;
    const MRTaskStateGPU state = taskStates[environment];

    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        if (!frameObservationOpcode(operation.source.x)) {
            continue;
        }
        device const MRTaskFrameGPU& frame =
            frames[operation.source.y];
        const TaskFramePose pose = taskResetBodyPose(
            frame,
            bodyPoses + bodyBase,
            sceneBodies + sceneBase
        );
        const MRBodyStateGPU velocityBody = taskFrameBodyState(
            frame,
            bodyStates + bodyBase,
            sceneBodies + sceneBase
        );
        MRBodyStateGPU referenceVelocityBody = velocityBody;
        TaskFramePose referencePose = pose;
        if (relativeFrameObservationOpcode(
                operation.source.x
            )) {
            device const MRTaskFrameGPU& referenceFrame =
                frames[operation.auxiliary.z];
            referencePose = taskResetBodyPose(
                referenceFrame,
                bodyPoses + bodyBase,
                sceneBodies + sceneBase
            );
            referenceVelocityBody = taskFrameBodyState(
                referenceFrame,
                bodyStates + bodyBase,
                sceneBodies + sceneBase
            );
        }
        const float clean = taskFrameObservationValue(
            dispatch,
            operation,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            state.episode.y,
            0u,
            float4(pose.position, 1.0f),
            pose.orientation,
            velocityBody.linearVelocityAndInverseMass.xyz,
            velocityBody.angularVelocity.xyz,
            float4(referencePose.position, 1.0f),
            referencePose.orientation,
            referenceVelocityBody.linearVelocityAndInverseMass.xyz,
            referenceVelocityBody.angularVelocity.xyz
        );
        float corrupted =
            clean +
            operation.transform.z *
                randomSigned(
                    dispatch,
                    environment,
                    state.episode.y,
                    0u,
                    operation.auxiliary.y
                );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            corrupted += sensorBias[
                biasBase + operation.auxiliary.x
            ];
        }
        for (uint history = 0u;
             history < program.layout.y;
             ++history) {
            const uint historyIndex =
                historyBase + history * program.layout.x + index;
            actorHistory[historyIndex] = corrupted;
            cleanHistory[historyIndex] = clean;
            actorObservations[
                actorOutputBase + history * program.layout.x + index
            ] = corrupted;
            if (cleanOutputOffset != MR_INVALID_INDEX) {
                criticObservations[
                    criticOutputBase +
                    history * program.layout.x + index
                ] = clean;
            }
        }
    }

    const uint criticOutputOffset =
        cleanOutputOffset != MR_INVALID_INDEX
        ? historyElements
        : 0u;
    for (uint index = 0u;
         index < program.counts0.z;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            criticOperators[index];
        if (!frameObservationOpcode(operation.source.x)) {
            continue;
        }
        device const MRTaskFrameGPU& frame =
            frames[operation.source.y];
        const TaskFramePose pose = taskResetBodyPose(
            frame,
            bodyPoses + bodyBase,
            sceneBodies + sceneBase
        );
        const MRBodyStateGPU velocityBody = taskFrameBodyState(
            frame,
            bodyStates + bodyBase,
            sceneBodies + sceneBase
        );
        MRBodyStateGPU referenceVelocityBody = velocityBody;
        TaskFramePose referencePose = pose;
        if (relativeFrameObservationOpcode(
                operation.source.x
            )) {
            device const MRTaskFrameGPU& referenceFrame =
                frames[operation.auxiliary.z];
            referencePose = taskResetBodyPose(
                referenceFrame,
                bodyPoses + bodyBase,
                sceneBodies + sceneBase
            );
            referenceVelocityBody = taskFrameBodyState(
                referenceFrame,
                bodyStates + bodyBase,
                sceneBodies + sceneBase
            );
        }
        const float clean = taskFrameObservationValue(
            dispatch,
            operation,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            state.episode.y,
            0u,
            float4(pose.position, 1.0f),
            pose.orientation,
            velocityBody.linearVelocityAndInverseMass.xyz,
            velocityBody.angularVelocity.xyz,
            float4(referencePose.position, 1.0f),
            referencePose.orientation,
            referenceVelocityBody.linearVelocityAndInverseMass.xyz,
            referenceVelocityBody.angularVelocity.xyz
        );
        for (uint history = 0u;
             history < program.articulation.w;
             ++history) {
            const uint historyIndex =
                criticHistoryBase +
                history * program.counts0.z + index;
            criticHistory[historyIndex] = clean;
            criticObservations[
                criticOutputBase + criticOutputOffset +
                history * program.counts0.z + index
            ] = clean;
        }
    }
}

// Materialize current accepted SensorIR truth into SignalIR scratch before
// reward and termination evaluation. The later history-refresh pass remains
// separate because it consumes TaskIR's updated episode/termination state.
kernel void mr_task_prepare_sensor_signals(
    device const MRTaskDispatchGPU& dispatch
        [[buffer(MR_TASK_SIGNAL_SENSOR_DISPATCH)]],
    device const MRTaskProgramHeaderGPU& program
        [[buffer(MR_TASK_SIGNAL_SENSOR_TASK_PROGRAM)]],
    device const uchar* arena
        [[buffer(MR_TASK_SIGNAL_SENSOR_TASK_ARENA)]],
    device const MRSensorProgramHeaderGPU& sensorProgram
        [[buffer(MR_TASK_SIGNAL_SENSOR_SENSOR_PROGRAM)]],
    device const float* sensorOutputs
        [[buffer(MR_TASK_SIGNAL_SENSOR_SENSOR_OUTPUTS)]],
    device const MRSensorSampleMetadataGPU* sensorMetadata
        [[buffer(MR_TASK_SIGNAL_SENSOR_SENSOR_METADATA)]],
    device MRBodyStateGPU* bodyStates
        [[buffer(MR_TASK_SIGNAL_SENSOR_BODY_STATES)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        sensorProgram.reserved.x != MR_SENSOR_PROGRAM_ABI_VERSION ||
        taskSensorFingerprint(program) == 0u ||
        taskSensorFingerprint(program) !=
            sensorProgram.sensorFingerprint ||
        sensorProgram.worldFingerprint != program.worldFingerprint) {
        return;
    }
    device const MRTaskObservationOperatorGPU* sources =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets4.x
        );
    device float* graphScratch =
        reinterpret_cast<device float*>(
            bodyStates +
            dispatch.counts.x * dispatch.strides.y
        ) + dispatch.counts.x * program.graphCounts.z;
    device float* sensorSourceValues =
        graphScratch +
        dispatch.counts.x * program.graphCounts.x +
        environment * program.graphCounts.w;
    for (uint sourceIndex = 0u;
         sourceIndex < program.graphCounts.y;
         ++sourceIndex) {
        const MRTaskObservationOperatorGPU source = sources[sourceIndex];
        if (!sensorObservationOpcode(source.source.x)) {
            continue;
        }
        sensorSourceValues[source.auxiliary.w] = taskSensorObservationValue(
            sensorProgram,
            source,
            sensorOutputs,
            sensorMetadata,
            environment
        );
    }
}

// Bind compiled SensorIR outputs into TaskIR histories without exposing
// simulator state to the learner. Reset-only mode seeds every history slot
// before the first action of an episode. Advance mode writes the tail that
// mr_task_complete already shifted, then republishes the terminal views used
// by value bootstrapping and optional final-policy evaluation.
kernel void mr_task_refresh_sensor_observations(
    device const MRTaskDispatchGPU& dispatch
        [[buffer(MR_TASK_SENSOR_REFRESH_DISPATCH)]],
    device const MRTaskProgramHeaderGPU& program
        [[buffer(MR_TASK_SENSOR_REFRESH_PROGRAM)]],
    device const uchar* arena
        [[buffer(MR_TASK_SENSOR_REFRESH_ARENA)]],
    device const MRSensorProgramHeaderGPU& sensorProgram
        [[buffer(MR_TASK_SENSOR_REFRESH_SENSOR_PROGRAM)]],
    constant MRMetalWorldPassGPU& pass
        [[buffer(MR_TASK_SENSOR_REFRESH_PASS)]],
    device const uint* resetMasks
        [[buffer(MR_TASK_SENSOR_REFRESH_RESET_MASKS)]],
    device const MRTaskStateGPU* taskStates
        [[buffer(MR_TASK_SENSOR_REFRESH_TASK_STATES)]],
    device const float* sensorOutputs
        [[buffer(MR_TASK_SENSOR_REFRESH_SENSOR_OUTPUTS)]],
    device const MRSensorSampleMetadataGPU* sensorMetadata
        [[buffer(MR_TASK_SENSOR_REFRESH_SENSOR_METADATA)]],
    device const float* sensorBias
        [[buffer(MR_TASK_SENSOR_REFRESH_SENSOR_BIAS)]],
    device float* actorHistory
        [[buffer(MR_TASK_SENSOR_REFRESH_ACTOR_HISTORY)]],
    device float* cleanHistory
        [[buffer(MR_TASK_SENSOR_REFRESH_CLEAN_HISTORY)]],
    device float* criticHistory
        [[buffer(MR_TASK_SENSOR_REFRESH_CRITIC_HISTORY)]],
    device float* actorObservations
        [[buffer(MR_TASK_SENSOR_REFRESH_ACTOR_OBSERVATIONS)]],
    device float* criticObservations
        [[buffer(MR_TASK_SENSOR_REFRESH_CRITIC_OBSERVATIONS)]],
    const uint environment [[thread_position_in_grid]]
) {
    const bool resetOnly =
        pass.reserved0 == MR_SENSOR_EXECUTION_RESET_ONLY;
    const bool advance =
        pass.reserved0 == MR_SENSOR_EXECUTION_ADVANCE;
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        (!resetOnly && !advance) ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        sensorProgram.reserved.x !=
            MR_SENSOR_PROGRAM_ABI_VERSION ||
        taskSensorFingerprint(program) == 0u ||
        taskSensorFingerprint(program) !=
            sensorProgram.sensorFingerprint ||
        sensorProgram.worldFingerprint !=
            program.worldFingerprint ||
        program.layout.y == 0u ||
        program.articulation.w == 0u) {
        return;
    }

    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    if (resetOnly && resetMasks[maskIndex] == 0u) {
        return;
    }
    const MRTaskStateGPU state = taskStates[environment];
    // mr_task_complete intentionally preserves the previous terminal frame
    // for failed (non-timeout) episodes. Match that transaction here.
    if (advance && state.status.y != 0u &&
        state.status.z != MR_TASK_TERMINATION_TIMEOUT) {
        return;
    }

    device const MRTaskObservationOperatorGPU* actorOperators =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets0.y
        );
    device const MRTaskObservationOperatorGPU* criticOperators =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets0.z
        );
    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase = environment * historyElements;
    const uint actorTailBase =
        historyBase +
        (program.layout.y - 1u) * program.layout.x;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint criticTailBase =
        criticHistoryBase +
        (program.articulation.w - 1u) * program.counts0.z;
    const uint biasBase = environment * program.counts2.z;
    const uint episodeStep = resetOnly ? 0u : state.episode.x;

    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        if (!sensorObservationOpcode(operation.source.x)) {
            continue;
        }
        const float clean = taskSensorObservationValue(
            sensorProgram,
            operation,
            sensorOutputs,
            sensorMetadata,
            environment
        );
        float corrupted =
            clean +
            operation.transform.z *
                randomSigned(
                    dispatch,
                    environment,
                    state.episode.y,
                    episodeStep,
                    operation.auxiliary.y
                );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            corrupted += sensorBias[
                biasBase + operation.auxiliary.x
            ];
        }
        if (resetOnly) {
            for (uint history = 0u;
                 history < program.layout.y;
                 ++history) {
                const uint destination =
                    historyBase + history * program.layout.x + index;
                actorHistory[destination] = corrupted;
                cleanHistory[destination] = clean;
            }
        } else {
            cleanHistory[actorTailBase + index] = clean;
            if (state.schedule.w != 0u &&
                program.layout.y > 1u) {
                const uint delay = min(
                    state.schedule.w,
                    program.layout.y - 1u
                );
                const uint sourceHistory =
                    program.layout.y - 1u - delay;
                actorHistory[actorTailBase + index] =
                    actorHistory[
                        historyBase +
                        sourceHistory * program.layout.x + index
                    ];
            } else {
                actorHistory[actorTailBase + index] = corrupted;
            }
        }
    }

    for (uint index = 0u;
         index < program.counts0.z;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            criticOperators[index];
        if (!sensorObservationOpcode(operation.source.x)) {
            continue;
        }
        const float clean = taskSensorObservationValue(
            sensorProgram,
            operation,
            sensorOutputs,
            sensorMetadata,
            environment
        );
        if (resetOnly) {
            for (uint history = 0u;
                 history < program.articulation.w;
                 ++history) {
                criticHistory[
                    criticHistoryBase +
                    history * program.counts0.z + index
                ] = clean;
            }
        } else {
            criticHistory[criticTailBase + index] = clean;
        }
    }

    const bool finalActor =
        advance && dispatch.timing.z != 0.0f &&
        pass.controlStep + 1u == dispatch.counts.y;
    if (resetOnly || finalActor) {
        const uint observationStep =
            resetOnly ? pass.controlStep : dispatch.counts.y;
        const uint actorOutputBase =
            observationStep * dispatch.outputs.x +
            environment * historyElements;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            actorObservations[actorOutputBase + index] =
                actorHistory[historyBase + index];
        }
    }

    if (resetOnly ||
        (advance && dispatch.timing.w != 0.0f)) {
        const uint observationStep =
            resetOnly ? pass.controlStep : dispatch.counts.y;
        const uint criticObservationSize =
            ((program.schedule.w &
              MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
                 ? historyElements
                 : 0u) +
            criticHistoryElements;
        const uint criticOutputBase =
            observationStep * dispatch.outputs.y +
            environment * criticObservationSize;
        publishCritic(
            program,
            cleanHistory + historyBase,
            criticHistory + criticHistoryBase,
            criticObservations + criticOutputBase
        );
    }
}

kernel void mr_task_apply_actions(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* actionStream [[buffer(5)]],
    device float* effortTrajectory [[buffer(6)]],
    device const float* defaultQ [[buffer(7)]],
    device const MRTaskStateGPU* taskStates [[buffer(8)]],
    device float* actionHistory [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.counts.x !=
            worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION ||
        program.counts0.x == 0u ||
        program.layout.w < 2u) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    const uint actionBase =
        pass.controlStep * dispatch.strides.x +
        environment * program.counts0.x;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    const MRTaskStateGPU state = taskStates[environment];

    for (uint coordinate = 0u;
         coordinate < dispatch.counts.w;
         ++coordinate) {
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            coordinate
        ] = 0.0f;
    }
    const uint lastSlot = program.layout.w - 1u;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        for (uint delay = 0u;
             delay + 1u < program.layout.w;
             ++delay) {
            actionHistory[
                delayBase +
                delay * program.counts0.x +
                action
            ] = actionHistory[
                delayBase +
                (delay + 1u) * program.counts0.x +
                action
            ];
        }
        actionHistory[
            delayBase +
            lastSlot * program.counts0.x +
            action
        ] = clamp(
            actionStream[actionBase + action],
            -1.0f,
            1.0f
        );
        const uint selected =
            lastSlot - min(state.schedule.z, lastSlot);
        const float delayed = actionHistory[
            delayBase +
            selected * program.counts0.x +
            action
        ];
        const MRTaskActionBindingGPU binding =
            actions[action];
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            binding.indices.w
        ] = clamp(
            defaultQ[binding.indices.z] +
                binding.parameters.x * delayed,
            binding.parameters.y,
            binding.parameters.z
        );
    }
}

kernel void mr_task_measure_effort(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    constant MRMetalWorldPassGPU& pass [[buffer(3)]],
    device const float* vState [[buffer(4)]],
    device const float* workingEffort [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    const uint vBase = environment * dispatch.counts.w;
    float mechanicalPower = 0.0f;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        const uint velocityIndex =
            actions[action].indices.w;
        mechanicalPower += abs(
            workingEffort[vBase + velocityIndex] *
            vState[vBase + velocityIndex]
        );
    }
    MRTaskStateGPU state = taskStates[environment];
    state.powerReturnMetric.x = mechanicalPower;
    taskStates[environment] = state;
}

kernel void mr_task_complete(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRTaskCurriculumStateGPU* curriculumState
        [[buffer(3)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(4)]],
    constant MRMetalWorldPassGPU& pass [[buffer(5)]],
    device const float* qState [[buffer(6)]],
    device const float* vState [[buffer(7)]],
    device MRBodyStateGPU* bodyStates [[buffer(8)]],
    device const MRContactConstraintGPU* contacts [[buffer(9)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(10)]],
    device const MRMetalWorldStatusGPU* worldStatuses
        [[buffer(11)]],
    device const MRBodyStateGPU* sceneState [[buffer(12)]],
    device const float* defaultQ [[buffer(14)]],
    device MRTaskStateGPU* taskStates [[buffer(15)]],
    device float* actionHistory [[buffer(16)]],
    device float* actorHistory [[buffer(17)]],
    device float* cleanHistory [[buffer(18)]],
    device float* previousJointVelocity [[buffer(19)]],
    device const float* sensorBias [[buffer(20)]],
    device const float4* bodyParameters [[buffer(21)]],
    device const float4* controllerParameters [[buffer(22)]],
    device float* compactContact [[buffer(23)]],
    device MRTaskTransitionGPU* transitions [[buffer(24)]],
    device const MRShapeGPU* shapes [[buffer(25)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(26)]],
    device const float4* geometryVertices [[buffer(27)]],
    device float* actorObservations [[buffer(28)]],
    device float* criticObservations [[buffer(29)]],
    device float* criticHistory [[buffer(30)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    device const MRTaskObservationOperatorGPU*
        actorOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.y
            );
    device const MRTaskObservationOperatorGPU*
        criticOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.z
            );
    device const MRTaskObservationOperatorGPU* signalSources =
        taskTable<MRTaskObservationOperatorGPU>(
            arena,
            program.offsets4.x
        );
    device const MRTaskSignalOperatorGPU* signalOperators =
        taskTable<MRTaskSignalOperatorGPU>(
            arena,
            program.offsets4.y
        );
    device const MRTaskContactGroupGPU* contactGroups =
        taskTable<MRTaskContactGroupGPU>(
            arena,
            program.offsets0.w
        );
    device const uint* contactMembers =
        taskTable<uint>(arena, program.offsets1.x);
    device const MRTaskRewardOperatorGPU* rewards =
        taskTable<MRTaskRewardOperatorGPU>(
            arena,
            program.offsets1.w
        );
    device const MRTaskRecorderOperatorGPU* recorders =
        taskTable<MRTaskRecorderOperatorGPU>(
            arena,
            program.offsets1.y
        );
    device const MRTaskTerminationOperatorGPU*
        terminations =
            taskTable<MRTaskTerminationOperatorGPU>(
                arena,
                program.offsets2.x
            );
    device const float4* terrainSamples =
        taskTable<float4>(arena, program.offsets2.w);
    device const MRTaskCommandOperatorGPU* commandOperators =
        taskTable<MRTaskCommandOperatorGPU>(
            arena,
            program.offsets3.y
        );
    device const MRTaskFrameGPU* frames =
        taskTable<MRTaskFrameGPU>(
            arena,
            program.offsets3.z
        );
    device const MRTaskKinematicFrameGPU* kinematicFrames =
        taskKinematicFrameTable(arena, program);
    device const MRTaskGoalGPU* goals =
        taskTable<MRTaskGoalGPU>(
            arena,
            program.offsets3.w
        );

    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint bodyBase =
        environment * dispatch.strides.y;
    device const float* spatialJacobians =
        reinterpret_cast<device const float*>(
            bodyStates +
            dispatch.counts.x * dispatch.strides.y
        );
    device float* graphScratch =
        reinterpret_cast<device float*>(
            bodyStates +
            dispatch.counts.x * dispatch.strides.y
        ) + dispatch.counts.x * program.graphCounts.z;
    device float* signalValues =
        graphScratch + environment * program.graphCounts.x;
    device const float* sensorSourceValues =
        graphScratch +
        dispatch.counts.x * program.graphCounts.x +
        environment * program.graphCounts.w;
    const uint sceneBase =
        environment * dispatch.strides.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase =
        environment * historyElements;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint previousVelocityBase =
        environment * program.counts0.x;
    const uint biasBase =
        environment * program.counts2.z;
    const uint bodyParameterBase =
        environment * dispatch.strides.y;
    const uint compactBase =
        environment * program.layout.z;
    const uint contactBase =
        environment * contactDispatch.constraintStride;
    MRTaskStateGPU state = taskStates[environment];
    const uint curriculum = min(
        curriculumState[0].commandLevel,
        program.curriculum.x - 1u
    );
    state.episode.z = curriculum;
    const MRMetalWorldStatusGPU worldStatus =
        worldStatuses[environment];
    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    const bool physicsError =
        worldStatus.code != MR_STEP_SUCCESS ||
        contactStatus.code != MR_STEP_SUCCESS;
    const uint publishedConstraints = physicsError
        ? 0u
        : min(
              contactStatus.requiredConstraints,
              contactDispatch.constraintCapacity
          );

    for (uint groupIndex = 0u;
         groupIndex < program.counts0.w;
         ++groupIndex) {
        const MRTaskContactGroupGPU group =
            contactGroups[groupIndex];
        const uint metric = compactBase + group.members.w;
        if ((group.members.z &
             MR_TASK_CONTACT_SUPPORT) != 0u) {
            // Preserve previous air time in slot 2. Other slots become
            // impulse/COP accumulators until the reduction is finalized.
            compactContact[metric + 0u] = 0.0f;
            compactContact[metric + 1u] = 0.0f;
            compactContact[metric + 3u] = 0.0f;
            compactContact[metric + 4u] = 0.0f;
            compactContact[metric + 5u] = 0.0f;
        } else if ((group.members.z &
                    MR_TASK_CONTACT_FORBIDDEN) != 0u) {
            compactContact[metric] = 0.0f;
        }
        const uint wrench =
            compactBase + group.reference.y;
        for (uint component = 0u;
             component < 6u;
             ++component) {
            compactContact[wrench + component] = 0.0f;
        }
    }

    const uint robotFirst = program.articulation.x;
    const uint robotEnd =
        robotFirst + program.articulation.y;
    for (uint constraintIndex =
             min(
                 contactDispatch.authoredConstraintCount,
                 publishedConstraints
             );
         constraintIndex < publishedConstraints;
         ++constraintIndex) {
        const MRContactConstraintGPU constraint =
            contacts[contactBase + constraintIndex];
        if ((constraint.flags &
             (MR_CONSTRAINT_FLAG_DISABLED |
              MR_CONSTRAINT_FLAG_GENERALIZED)) != 0u) {
            continue;
        }
        const bool robotA =
            constraint.bodyA >= robotFirst &&
            constraint.bodyA < robotEnd;
        const bool robotB =
            constraint.bodyB >= robotFirst &&
            constraint.bodyB < robotEnd;
        if (robotA == robotB) {
            continue;
        }
        const uint robotBody =
            robotA ? constraint.bodyA : constraint.bodyB;
        const float impulse = abs(constraint.impulses.x);
        const float3 normal = normalizedOr(
            constraint.normal.xyz,
            float3(0.0f, 0.0f, 1.0f)
        );
        const float3 authoredTangent =
            constraint.tangent.xyz -
            normal * dot(normal, constraint.tangent.xyz);
        const float3 tangent = normalizedOr(
            authoredTangent,
            stableContactTangent(normal)
        );
        const float3 bitangent = cross(normal, tangent);
        const float3 impulseOnA = -(
            normal * constraint.impulses.x +
            tangent * constraint.impulses.y +
            bitangent * constraint.impulses.z
        );
        const float3 robotImpulse =
            robotA ? impulseOnA : -impulseOnA;
        const float3 forceWorld =
            robotImpulse / dispatch.timing.y;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            device const MRTaskContactGroupGPU& group =
                contactGroups[groupIndex];
            if (!bodyMember(
                    robotBody,
                    group,
                    contactMembers
                )) {
                continue;
            }
            const uint metric =
                compactBase + group.members.w;
            const MRBodyStateGPU referenceBody =
                bodyStates[
                    bodyBase + group.reference.x
                ];
            const float3 referencePosition =
                referenceBody.position.xyz +
                rotate(
                    referenceBody.orientation,
                    group.localReference.xyz
                );
            const float torsionalSign =
                robotA ? -1.0f : 1.0f;
            const float3 torqueWorld =
                cross(
                    constraint.pointAndSeparation.xyz -
                        referencePosition,
                    forceWorld
                ) +
                torsionalSign *
                    normal *
                    constraint.impulses.w /
                    dispatch.timing.y;
            const float3 forceLocal = rotateInverse(
                referenceBody.orientation,
                forceWorld
            );
            const float3 torqueLocal = rotateInverse(
                referenceBody.orientation,
                torqueWorld
            );
            const uint wrench =
                compactBase + group.reference.y;
            compactContact[wrench + 0u] += forceLocal.x;
            compactContact[wrench + 1u] += forceLocal.y;
            compactContact[wrench + 2u] += forceLocal.z;
            compactContact[wrench + 3u] += torqueLocal.x;
            compactContact[wrench + 4u] += torqueLocal.y;
            compactContact[wrench + 5u] += torqueLocal.z;
            if ((group.members.z &
                 MR_TASK_CONTACT_SUPPORT) != 0u) {
                compactContact[metric + 0u] += impulse;
                compactContact[metric + 4u] +=
                    impulse *
                    constraint.pointAndSeparation.x;
                compactContact[metric + 5u] +=
                    impulse *
                    constraint.pointAndSeparation.y;
            }
            if ((group.members.z &
                 MR_TASK_CONTACT_FORBIDDEN) != 0u) {
                compactContact[metric] = max(
                    compactContact[metric],
                    float(
                        impulse / dispatch.timing.y >
                        program.taskScalars.w
                    )
                );
            }
        }
    }

    for (uint groupIndex = 0u;
         groupIndex < program.counts0.w;
         ++groupIndex) {
        const MRTaskContactGroupGPU group =
            contactGroups[groupIndex];
        if ((group.members.z &
             MR_TASK_CONTACT_SUPPORT) == 0u) {
            continue;
        }
        const uint metric =
            compactBase + group.members.w;
        const float impulse =
            compactContact[metric + 0u];
        const float previousAir =
            compactContact[metric + 2u];
        const MRBodyStateGPU body =
            bodyStates[
                bodyBase + group.reference.x
            ];
        const float3 offset = rotate(
            body.orientation,
            group.localReference.xyz
        );
        const float3 kinematicOffset = rotate(
            body.orientation,
            group.kinematicReference.xyz
        );
        const float3 position =
            body.position.xyz + offset;
        const float3 kinematicVelocity =
            body.linearVelocityAndInverseMass.xyz +
            cross(
                body.angularVelocity.xyz,
                kinematicOffset
            );
        const float force =
            impulse / dispatch.timing.y;
        const bool contact =
            force > program.taskScalars.w;
        const float slip =
            contact
            ? length(kinematicVelocity.xy)
            : 0.0f;
        const float airTime =
            contact
            ? 0.0f
            : previousAir + dispatch.timing.x;
        const float clearance =
            position.z -
            group.localReference.w -
            surfaceHeight(
                program,
                shapes,
                geometryHeaders,
                geometryVertices,
                sceneState + sceneBase,
                position.xy
            );
        const float2 cop =
            impulse > 1.0e-6f
            ? float2(
                  compactContact[metric + 4u],
                  compactContact[metric + 5u]
              ) /
                  impulse -
                  position.xy
            : float2(0.0f);
        compactContact[metric + 0u] = force;
        compactContact[metric + 1u] = slip;
        compactContact[metric + 2u] = airTime;
        compactContact[metric + 3u] = clearance;
        compactContact[metric + 4u] = cop.x;
        compactContact[metric + 5u] = cop.y;
    }

    device const float* q = qState + qBase;
    device const float* v = vState + vBase;

    const uint lastSlot = program.layout.w - 1u;
    const uint previousSlot = lastSlot - 1u;
    const uint episodeSteps = state.episode.x + 1u;
    device const float* signalAction =
        actionHistory +
        delayBase +
        (program.layout.w - 1u) * program.counts0.x;
    device const float* earlierSignalAction =
        actionHistory +
        delayBase +
        (program.layout.w - 2u) * program.counts0.x;
    for (uint signalIndex = 0u;
         signalIndex < program.graphCounts.x;
         ++signalIndex) {
        const MRTaskSignalOperatorGPU operation =
            signalOperators[signalIndex];
        const bool reductionNode =
            operation.inputs.x == MR_TASK_SIGNAL_REDUCTION;
        const float left =
            !reductionNode &&
            operation.inputs.z != MR_INVALID_INDEX
            ? signalValues[operation.inputs.z]
            : 0.0f;
        const float right =
            !reductionNode &&
            operation.inputs.w != MR_INVALID_INDEX
            ? signalValues[operation.inputs.w]
            : 0.0f;
        float value = 0.0f;
        switch (operation.inputs.x) {
        case MR_TASK_SIGNAL_SOURCE: {
            const MRTaskObservationOperatorGPU source =
                signalSources[operation.inputs.y];
            if (sensorObservationOpcode(source.source.x)) {
                // Prepared from the accepted SensorIR sample immediately
                // before this pass.
                value = sensorSourceValues[source.auxiliary.w];
            } else {
                value = cleanObservation(
                    dispatch,
                    program,
                    source,
                    actions,
                    contactGroups,
                    frames,
                    kinematicFrames,
                    goals,
                    spatialJacobians,
                    environment,
                    state.episode.y,
                    episodeSteps,
                    bodyStates + bodyBase,
                    terrainSamples,
                    q,
                    v,
                    defaultQ,
                    state,
                    signalAction,
                    earlierSignalAction,
                    previousJointVelocity + previousVelocityBase,
                    compactContact + compactBase,
                    bodyParameters + bodyParameterBase,
                    controllerParameters + environment,
                    sceneState + sceneBase,
                    shapes,
                    geometryHeaders,
                    geometryVertices
                );
            }
            break;
        }
        case MR_TASK_SIGNAL_REDUCTION: {
            const uint transform = operation.inputs.w & 0xffu;
            const uint reduction = operation.inputs.w >> 8u;
            for (uint local = 0u;
                 local < operation.inputs.z;
                 ++local) {
                const MRTaskObservationOperatorGPU source =
                    signalSources[operation.inputs.y + local];
                float element = sensorObservationOpcode(source.source.x)
                    ? sensorSourceValues[source.auxiliary.w]
                    : cleanObservation(
                          dispatch,
                          program,
                          source,
                          actions,
                          contactGroups,
                          frames,
                          kinematicFrames,
                          goals,
                          spatialJacobians,
                          environment,
                          state.episode.y,
                          episodeSteps,
                          bodyStates + bodyBase,
                          terrainSamples,
                          q,
                          v,
                          defaultQ,
                          state,
                          signalAction,
                          earlierSignalAction,
                          previousJointVelocity + previousVelocityBase,
                          compactContact + compactBase,
                          bodyParameters + bodyParameterBase,
                          controllerParameters + environment,
                          sceneState + sceneBase,
                          shapes,
                          geometryHeaders,
                          geometryVertices
                      );
                if (transform ==
                        MR_TASK_SIGNAL_TRANSFORM_ABSOLUTE) {
                    element = abs(element);
                } else if (transform ==
                               MR_TASK_SIGNAL_TRANSFORM_SQUARE) {
                    element *= element;
                }
                if (local == 0u &&
                    (reduction ==
                         MR_TASK_SIGNAL_REDUCE_MINIMUM ||
                     reduction ==
                         MR_TASK_SIGNAL_REDUCE_MAXIMUM)) {
                    value = element;
                } else if (reduction ==
                               MR_TASK_SIGNAL_REDUCE_MINIMUM) {
                    value = min(value, element);
                } else if (reduction ==
                               MR_TASK_SIGNAL_REDUCE_MAXIMUM) {
                    value = max(value, element);
                } else {
                    value += element;
                }
            }
            if (reduction == MR_TASK_SIGNAL_REDUCE_MEAN) {
                value /= float(operation.inputs.z);
            }
            break;
        }
        case MR_TASK_SIGNAL_CONSTANT:
            value = operation.parameters.x;
            break;
        case MR_TASK_SIGNAL_ADD:
            value = left + right;
            break;
        case MR_TASK_SIGNAL_SUBTRACT:
            value = left - right;
            break;
        case MR_TASK_SIGNAL_MULTIPLY:
            value = left * right;
            break;
        case MR_TASK_SIGNAL_MINIMUM:
            value = min(left, right);
            break;
        case MR_TASK_SIGNAL_MAXIMUM:
            value = max(left, right);
            break;
        case MR_TASK_SIGNAL_ABSOLUTE:
            value = abs(left);
            break;
        case MR_TASK_SIGNAL_SQUARE:
            value = left * left;
            break;
        case MR_TASK_SIGNAL_SQUARE_ROOT:
            value = sqrt(max(left, 0.0f));
            break;
        case MR_TASK_SIGNAL_SAFE_DIVIDE: {
            const float epsilon = operation.parameters.x;
            const float denominator =
                abs(right) >= epsilon
                ? right
                : copysign(epsilon, right == 0.0f ? 1.0f : right);
            value = left / denominator;
            break;
        }
        case MR_TASK_SIGNAL_CLAMP:
            value = clamp(
                left,
                operation.parameters.x,
                operation.parameters.y
            );
            break;
        case MR_TASK_SIGNAL_EXPONENTIAL_TRACKING: {
            const float normalized =
                (left - operation.parameters.x) /
                operation.parameters.y;
            value = exp(-(normalized * normalized));
            break;
        }
        case MR_TASK_SIGNAL_INSIDE_BOUNDS:
            value = float(
                left >= operation.parameters.x &&
                left <= operation.parameters.y
            );
            break;
        case MR_TASK_SIGNAL_EXPONENTIAL_DECAY:
            value = exp(-left / operation.parameters.x);
            break;
        case MR_TASK_SIGNAL_ATAN2:
            value = atan2(left, right);
            break;
        case MR_TASK_SIGNAL_TANH:
            value = tanh(left);
            break;
        case MR_TASK_SIGNAL_LESS_THAN:
            value = float(left < operation.parameters.x);
            break;
        case MR_TASK_SIGNAL_GREATER_THAN:
            value = float(left > operation.parameters.x);
            break;
        default:
            value = 0.0f;
            break;
        }
        signalValues[signalIndex] = value;
    }
    float3 recorderMetrics = float3(0.0f);
    for (uint recorderIndex = 0u;
         recorderIndex < program.counts1.y;
         ++recorderIndex) {
        const MRTaskRecorderOperatorGPU operation =
            recorders[recorderIndex];
        if (operation.source.y <
            MR_TASK_TRANSITION_METRIC_COUNT) {
            recorderMetrics[operation.source.y] =
                signalValues[operation.source.x];
        }
    }
    const float curriculumMetric =
        program.curriculum.z != MR_INVALID_INDEX
        ? signalValues[program.curriculum.z]
        : 0.0f;
    float phase =
        state.commandAndPhase.w +
        kTwoPi * dispatch.timing.x /
            program.taskScalars.x;
    phase = fmod(phase, kTwoPi);
    float reward = 0.0f;
    float4 rewardBreakdown0 = float4(0.0f);
    float4 rewardBreakdown1 = float4(0.0f);
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        const float value = signalValues[operation.source.x];
        const float contribution =
            operation.parameters.x * value;
        reward += contribution;
        const uint channel = operation.source.y;
        if (channel < 4u) {
            rewardBreakdown0[channel] += contribution;
        } else if (channel < MR_TASK_REWARD_CHANNEL_COUNT) {
            rewardBreakdown1[channel - 4u] += contribution;
        }
    }
    // TaskPack weights are rates. Integrating at the control boundary keeps
    // reward magnitude and PPO critic targets independent of control rate.
    reward *= dispatch.timing.x;
    rewardBreakdown0 *= dispatch.timing.x;
    rewardBreakdown1 *= dispatch.timing.x;

    const bool timeout =
        episodeSteps >= program.schedule.x;
    bool done = false;
    uint reason = MR_TASK_TERMINATION_CONTINUING;
    uint selectedPriority = 0u;
    float failurePenalty = 0.0f;
    for (uint index = 0u;
         index < program.counts2.x;
         ++index) {
        const MRTaskTerminationOperatorGPU operation =
            terminations[index];
        bool triggered = false;
        switch (operation.source.x) {
        case MR_TASK_TERMINATE_SIGNAL_BELOW:
            triggered =
                signalValues[operation.source.y] <
                operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_SIGNAL_ABOVE:
            triggered =
                signalValues[operation.source.y] >
                operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_SIGNAL_OUTSIDE: {
            const float signal =
                signalValues[operation.source.y];
            triggered = signal < operation.parameters.x ||
                signal > operation.parameters.z;
            break;
        }
        default:
            break;
        }
        if (triggered &&
            (!done ||
             operation.source.w >= selectedPriority)) {
            done = true;
            reason = operation.source.z;
            selectedPriority = operation.source.w;
            failurePenalty = operation.parameters.y;
        }
    }
    if (timeout) {
        done = true;
        reason = MR_TASK_TERMINATION_TIMEOUT;
    }
    if (physicsError) {
        done = true;
        reason = MR_TASK_TERMINATION_PHYSICS_ERROR;
    }
    if (done &&
        reason != MR_TASK_TERMINATION_TIMEOUT &&
        reason != MR_TASK_TERMINATION_PHYSICS_ERROR) {
        reward += failurePenalty;
        rewardBreakdown0.y += failurePenalty;
    }

    const float episodeReturn =
        state.powerReturnMetric.z + reward;
    const float episodeMetric =
        state.powerReturnMetric.w + curriculumMetric;
    const float episodeMetricMean =
        episodeMetric /
        max(float(episodeSteps), 1.0f);
    const bool successful =
        timeout &&
        !physicsError &&
        (
            program.curriculum.z == MR_INVALID_INDEX ||
            episodeMetricMean >= program.taskScalars.y
        );
    uint terrainLevel = state.episode.w;
    if (done) {
        if (successful) {
            if (program.terrain.w != 0u &&
                curriculum >= min(3u, program.curriculum.x - 1u)) {
                terrainLevel = min(
                    terrainLevel + 1u,
                    program.terrain.w - 1u
                );
            }
        } else if (program.terrain.w != 0u) {
            terrainLevel =
                terrainLevel == 0u
                ? 0u
                : terrainLevel - 1u;
        }
    }

    if (!done && state.schedule.x <= 1u) {
        state.commandAndPhase.xyz = sampledCommand(
            dispatch,
            program,
            commandOperators,
            environment,
            state.episode.y,
            episodeSteps,
            curriculum
        );
        state.schedule.x = durationSteps(
            dispatch,
            environment,
            state.episode.y,
            episodeSteps,
            64u,
            program.commandSchedule.y,
            program.commandSchedule.z
        );
    } else if (!done) {
        --state.schedule.x;
    }
    if (state.schedule.y == 0u || done) {
        state.schedule.y = durationSteps(
            dispatch,
            environment,
            state.episode.y,
            episodeSteps,
            65u,
            program.eventSchedule.y,
            program.eventSchedule.z
        );
    } else {
        --state.schedule.y;
    }

    state.episode = uint4(
        episodeSteps,
        state.episode.y,
        curriculum,
        terrainLevel
    );
    state.status.y = done ? 1u : 0u;
    state.status.z = reason;
    state.commandAndPhase.w = phase;
    state.powerReturnMetric = float4(
        0.0f,
        0.0f,
        done ? 0.0f : episodeReturn,
        done ? 0.0f : episodeMetric
    );

    if (!done || (timeout && !physicsError)) {
        for (uint history = 0u;
             history + 1u < program.layout.y;
             ++history) {
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = actorHistory[
                    historyBase +
                    (history + 1u) *
                        program.layout.x +
                    index
                ];
                cleanHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = cleanHistory[
                    historyBase +
                    (history + 1u) *
                        program.layout.x +
                    index
                ];
            }
        }
        device float* actorTail =
            actorHistory +
            historyBase +
            (program.layout.y - 1u) *
                program.layout.x;
        device float* cleanTail =
            cleanHistory +
            historyBase +
            (program.layout.y - 1u) *
                program.layout.x;
        device const float* currentAction =
            actionHistory +
            delayBase +
            lastSlot * program.counts0.x;
        writeFrame(
            dispatch,
            program,
            actorOperators,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            bodyStates + bodyBase,
            terrainSamples,
            environment,
            state.episode.y,
            episodeSteps,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            currentAction,
            actionHistory +
                delayBase +
                previousSlot * program.counts0.x,
            previousJointVelocity + previousVelocityBase,
            sensorBias + biasBase,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            actorTail,
            cleanTail
        );
        if (state.schedule.w != 0u &&
            program.layout.y > 1u) {
            const uint delay = min(
                state.schedule.w,
                program.layout.y - 1u
            );
            const uint sourceHistory =
                program.layout.y - 1u - delay;
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorTail[index] = actorHistory[
                    historyBase +
                    sourceHistory * program.layout.x +
                    index
                ];
            }
        }
        for (uint history = 0u;
             history + 1u < program.articulation.w;
             ++history) {
            for (uint index = 0u;
                 index < program.counts0.z;
                 ++index) {
                criticHistory[
                    criticHistoryBase +
                    history * program.counts0.z +
                    index
                ] = criticHistory[
                    criticHistoryBase +
                    (history + 1u) *
                        program.counts0.z +
                    index
                ];
            }
        }
        device float* criticTail =
            criticHistory +
            criticHistoryBase +
            (program.articulation.w - 1u) *
                program.counts0.z;
        writeCriticFrame(
            dispatch,
            program,
            criticOperators,
            actions,
            contactGroups,
            frames,
            kinematicFrames,
            goals,
            spatialJacobians,
            environment,
            state.episode.y,
            episodeSteps,
            bodyStates + bodyBase,
            terrainSamples,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            currentAction,
            actionHistory +
                delayBase +
                previousSlot * program.counts0.x,
            previousJointVelocity + previousVelocityBase,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            criticTail
        );
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            previousJointVelocity[
                previousVelocityBase + action
            ] = vState[
                vBase + actions[action].indices.w
            ];
        }
    }

    const bool finalPolicyObservation =
        dispatch.timing.z != 0.0f &&
        pass.controlStep + 1u == dispatch.counts.y;
    if (finalPolicyObservation) {
        const uint actorOutputBase =
            dispatch.counts.y * dispatch.outputs.x +
            environment * historyElements;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            actorObservations[actorOutputBase + index] =
                actorHistory[historyBase + index];
        }
    }
    if (dispatch.timing.w != 0.0f) {
        const uint criticObservationSize =
            (
                (program.schedule.w &
                 MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
                ? historyElements
                : 0u
            ) +
            criticHistoryElements;
        const uint criticOutputBase =
            dispatch.counts.y * dispatch.outputs.y +
            environment * criticObservationSize;
        publishCritic(
            program,
            cleanHistory + historyBase,
            criticHistory + criticHistoryBase,
            criticObservations + criticOutputBase
        );
    }
    taskStates[environment] = state;

    const uint transitionIndex =
        pass.controlStep * dispatch.outputs.z + environment;
    MRTaskTransitionGPU transition{};
    transition.rewardAndMetrics =
        float4(reward, recorderMetrics);
    transition.termination = uint4(
        done ? 1u : 0u,
        timeout ? 1u : 0u,
        physicsError ? 1u : 0u,
        reason
    );
    transition.rewardBreakdown0 = rewardBreakdown0;
    transition.rewardBreakdown1 = rewardBreakdown1;
    transition.policyRevision =
        dispatch.policyRevision;
    transition.episodeMetric =
        done && !physicsError
        ? episodeMetricMean
        : 0.0f;
    transition.taskProgress = uint4(
        curriculum,
        terrainLevel,
        0u,
        0u
    );
    transitions[transitionIndex] = transition;
}

// One native thread owns the global command curriculum. Episode outcomes are
// accumulated across the whole evaluation window, so early-reset environments
// rejoin the promotion evidence instead of becoming permanently phase-shifted.
kernel void mr_task_update_curriculum(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device MRTaskCurriculumStateGPU* curriculumState
        [[buffer(2)]],
    device MRTaskTransitionGPU* transitions [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    const uint gridIndex [[thread_position_in_grid]]
) {
    if (gridIndex != 0u ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.schedule.x == 0u ||
        program.curriculum.x == 0u) {
        return;
    }

    MRTaskCurriculumStateGPU state = curriculumState[0];
    const ulong completedSteps = state.controlSteps + 1ul;
    uint level = min(
        state.commandLevel,
        program.curriculum.x - 1u
    );
    const uint transitionBase =
        pass.controlStep * dispatch.outputs.z;
    for (uint environment = 0u;
         environment < dispatch.counts.x;
         ++environment) {
        const MRTaskTransitionGPU transition =
            transitions[transitionBase + environment];
        if (transition.termination.x != 0u &&
            transition.termination.z == 0u) {
            state.metricSum +=
                transition.episodeMetric;
            ++state.completedEpisodeCount;
            if (transition.termination.y != 0u) {
                ++state.timeoutEpisodeCount;
            }
        }
    }
    if (completedSteps % ulong(program.curriculum.y) == 0ul &&
        level + 1u < program.curriculum.x) {
        const float completed =
            float(state.completedEpisodeCount);
        const float meanMetric =
            state.metricSum / max(completed, 1.0f);
        const float survivalFraction =
            float(state.timeoutEpisodeCount) /
            max(completed, 1.0f);
        if (state.completedEpisodeCount != 0ul &&
            meanMetric > program.taskScalars.y &&
            survivalFraction >= program.taskScalars.z) {
            ++level;
        }
    }
    if (completedSteps % ulong(program.curriculum.y) == 0ul) {
        state.completedEpisodeCount = 0ul;
        state.timeoutEpisodeCount = 0ul;
        state.metricSum = 0.0f;
    }
    state.controlSteps = completedSteps;
    state.commandLevel = level;
    curriculumState[0] = state;
    for (uint environment = 0u;
         environment < dispatch.counts.x;
         ++environment) {
        device MRTaskTransitionGPU& transition =
            transitions[transitionBase + environment];
        transition.taskProgress = uint4(
            level,
            transition.taskProgress.y,
            0u,
            0u
        );
    }
}

inline void recordLearningPublicationFailure(
    threadgroup atomic_uint& invalidCount,
    threadgroup atomic_uint& firstInvalidOrdinal,
    const uint ordinal
) {
    atomic_fetch_add_explicit(
        &invalidCount,
        1u,
        memory_order_relaxed
    );
    atomic_fetch_min_explicit(
        &firstInvalidOrdinal,
        ordinal,
        memory_order_relaxed
    );
}

// One cooperative threadgroup validates the complete compact learning
// publication. The host waits for this fixed-size status record rather than
// rescanning the shared rollout payload. The command-buffer completion is the
// visibility boundary for both the status and the subsequent lease blit.
kernel void mr_task_validate_learning_publication(
    constant MRLearningPublicationDispatchGPU& dispatch
        [[buffer(MR_LEARNING_VALIDATE_DISPATCH)]],
    device const float* actorObservations
        [[buffer(MR_LEARNING_VALIDATE_ACTOR_OBSERVATIONS)]],
    device const float* criticObservations
        [[buffer(MR_LEARNING_VALIDATE_CRITIC_OBSERVATIONS)]],
    device const float* latents
        [[buffer(MR_LEARNING_VALIDATE_LATENTS)]],
    device const float* logProbabilities
        [[buffer(MR_LEARNING_VALIDATE_LOG_PROBABILITIES)]],
    device const float* values
        [[buffer(MR_LEARNING_VALIDATE_VALUES)]],
    device const MRTaskTransitionGPU* transitions
        [[buffer(MR_LEARNING_VALIDATE_TRANSITIONS)]],
    device MRLearningPublicationStatusGPU* status
        [[buffer(MR_LEARNING_VALIDATE_STATUS)]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint3 threadgroupSize [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint invalidCount;
    threadgroup atomic_uint firstInvalidOrdinal;
    if (lane == 0u) {
        atomic_store_explicit(
            &invalidCount,
            0u,
            memory_order_relaxed
        );
        atomic_store_explicit(
            &firstInvalidOrdinal,
            MR_INVALID_INDEX,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint actorCount = dispatch.floatCounts.x;
    const uint criticCount = dispatch.floatCounts.y;
    const uint latentCount = dispatch.floatCounts.z;
    const uint logProbabilityCount = dispatch.floatCounts.w;
    const uint valueCount = dispatch.recordCounts.x;
    const uint transitionCount = dispatch.recordCounts.y;
    const uint totalFloatCount = dispatch.recordCounts.z;
    const uint threads = max(threadgroupSize.x, 1u);

    uint streamBase = 0u;
    for (uint index = lane; index < actorCount; index += threads) {
        if (!isfinite(actorObservations[index])) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                streamBase + index
            );
        }
    }
    streamBase += actorCount;
    for (uint index = lane; index < criticCount; index += threads) {
        if (!isfinite(criticObservations[index])) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                streamBase + index
            );
        }
    }
    streamBase += criticCount;
    for (uint index = lane; index < latentCount; index += threads) {
        if (!isfinite(latents[index])) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                streamBase + index
            );
        }
    }
    streamBase += latentCount;
    for (uint index = lane;
         index < logProbabilityCount;
         index += threads) {
        if (!isfinite(logProbabilities[index])) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                streamBase + index
            );
        }
    }
    streamBase += logProbabilityCount;
    for (uint index = lane; index < valueCount; index += threads) {
        if (!isfinite(values[index])) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                streamBase + index
            );
        }
    }

    for (uint index = lane;
         index < transitionCount;
         index += threads) {
        const MRTaskTransitionGPU transition = transitions[index];
        const bool validTermination =
            transition.termination.x <= 1u &&
            transition.termination.y <= 1u &&
            transition.termination.z <= 1u &&
            transition.termination.w <=
                MR_TASK_TERMINATION_GOAL_ERROR &&
            (
                transition.termination.x != 0u ||
                transition.termination.w ==
                    MR_TASK_TERMINATION_CONTINUING
            );
        if (!all(isfinite(transition.rewardAndMetrics)) ||
            !all(isfinite(transition.rewardBreakdown0)) ||
            !all(isfinite(transition.rewardBreakdown1)) ||
            !isfinite(transition.timeoutBootstrapValue) ||
            !isfinite(transition.episodeMetric) ||
            transition.policyRevision != dispatch.policyRevision ||
            !validTermination) {
            recordLearningPublicationFailure(
                invalidCount,
                firstInvalidOrdinal,
                totalFloatCount + index
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane != 0u) {
        return;
    }
    const uint failures = atomic_load_explicit(
        &invalidCount,
        memory_order_relaxed
    );
    const uint first = atomic_load_explicit(
        &firstInvalidOrdinal,
        memory_order_relaxed
    );
    uint stream = MR_LEARNING_STREAM_NONE;
    uint streamIndex = MR_INVALID_INDEX;
    uint remaining = first;
    if (first != MR_INVALID_INDEX) {
        const uint counts[6] = {
            actorCount,
            criticCount,
            latentCount,
            logProbabilityCount,
            valueCount,
            transitionCount,
        };
        for (uint candidate = 0u; candidate < 6u; ++candidate) {
            if (remaining < counts[candidate]) {
                stream = candidate;
                streamIndex = remaining;
                break;
            }
            remaining -= counts[candidate];
        }
    }

    MRLearningPublicationStatusGPU publication{};
    publication.result = uint4(
        failures == 0u
            ? MR_LEARNING_PUBLICATION_SUCCESS
            : stream == MR_LEARNING_STREAM_TRANSITIONS
            ? MR_LEARNING_PUBLICATION_INVALID_TRANSITION
            : MR_LEARNING_PUBLICATION_INVALID_FLOAT,
        failures,
        stream,
        streamIndex
    );
    publication.checkedCounts = uint4(
        totalFloatCount,
        transitionCount,
        threadgroupSize.x,
        MR_RUNTIME_ABI_VERSION
    );
    publication.policyRevision = dispatch.policyRevision;
    publication.taskFingerprint = dispatch.taskFingerprint;
    publication.validationToken = dispatch.validationToken;
    publication.reserved = 0ul;
    status[0] = publication;
}
