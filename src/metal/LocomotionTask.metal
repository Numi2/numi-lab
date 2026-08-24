#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/task_program_types.h"

using namespace metal;

namespace {

constant float kPi = 3.14159265358979323846f;
constant float kTwoPi = 2.0f * kPi;
constant uint kImpactOrderMask = 0xffu;
constant uint kImpactSceneShift = 8u;
constant uint kImpactSceneMask = 0xffu << kImpactSceneShift;
constant uint kImpactEnabled = 1u << 16u;
constant uint kImpactOffsetShift = 17u;
constant uint kImpactOffsetMask = 0xffu << kImpactOffsetShift;
constant uint kImpactContactLatched = 1u << 25u;
constant uint kImpactContactPublished = 1u << 26u;

// ARDY_PHYSICS_GATED_REFERENCE_V4. status.w is available in ordinary
// interaction tracking and is compile-time excluded from impact-sequence tasks.
constant uint kInteractionPhaseFractionBits = 16u;
constant uint kInteractionPhaseScale = 1u << kInteractionPhaseFractionBits;
constant uint kInteractionPhaseMask = (1u << 29u) - 1u;
constant uint kInteractionRateShift = 29u;
constant uint kInteractionRateMask = 3u << kInteractionRateShift;
constant uint kInteractionFallLatched = 1u << 31u;

inline bool interactionPhysicsGated(
    device const MRTaskProgramHeaderGPU& program
) {
    return (program.schedule.w &
            MR_TASK_PROGRAM_INTERACTION_PHYSICS_GATED) != 0u;
}

inline float packedInteractionFramePosition(
    thread const MRTaskStateGPU& state
) {
    return float(state.status.w & kInteractionPhaseMask) /
        float(kInteractionPhaseScale);
}

inline uint interactionRateCode(thread const MRTaskStateGPU& state) {
    return (state.status.w & kInteractionRateMask) >>
        kInteractionRateShift;
}

inline float interactionPlaybackRate(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state
) {
    if (!interactionPhysicsGated(program)) {
        return 1.0f;
    }
    switch (interactionRateCode(state)) {
    case 1u: return 0.25f;
    case 2u: return 0.50f;
    case 3u: return 1.00f;
    default: return 0.0f;
    }
}

inline bool interactionFallIsLatched(
    thread const MRTaskStateGPU& state
) {
    return (state.status.w & kInteractionFallLatched) != 0u;
}

inline uint packInteractionClock(
    const float framePosition,
    const uint rateCode,
    const bool fallLatched
) {
    const float maximum =
        float(kInteractionPhaseMask) / float(kInteractionPhaseScale);
    const float bounded = clamp(framePosition, 0.0f, maximum);
    const uint fixed = min(
        uint(bounded * float(kInteractionPhaseScale) + 0.5f),
        kInteractionPhaseMask
    );
    return fixed |
        ((min(rateCode, 3u) << kInteractionRateShift) &
         kInteractionRateMask) |
        (fallLatched ? kInteractionFallLatched : 0u);
}


inline uint impactOrder(thread const MRTaskStateGPU& state) {
    return state.recoveryStats.w & kImpactOrderMask;
}

inline uint impactScene(thread const MRTaskStateGPU& state) {
    return
        (state.recoveryStats.w & kImpactSceneMask) >>
        kImpactSceneShift;
}

inline bool impactSequenceEnabled(
    thread const MRTaskStateGPU& state
) {
    return (state.recoveryStats.w & kImpactEnabled) != 0u;
}

inline uint impactOffset(thread const MRTaskStateGPU& state) {
    return
        (state.recoveryStats.w & kImpactOffsetMask) >>
        kImpactOffsetShift;
}

template <typename T>
inline device const T* taskTable(
    device const uchar* arena,
    const uint byteOffset
) {
    return reinterpret_cast<device const T*>(
        arena + byteOffset
    );
}

inline ulong mix64(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value =
        (value ^ (value >> 30u)) *
        0xbf58476d1ce4e5b9ul;
    value =
        (value ^ (value >> 27u)) *
        0x94d049bb133111ebul;
    return value ^ (value >> 31u);
}

inline float randomUnit(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel
) {
    ulong key = dispatch.seed;
    key ^= ulong(environment + 1u) *
        0xd2b74407b1ce6e93ul;
    key ^= ulong(episode + 1u) *
        0xca5a826395121157ul;
    key ^= ulong(controlStep + 1u) *
        0x9e3779b185ebca87ul;
    key ^= ulong(channel + 1u) *
        0x94d049bb133111ebul;
    return
        float(uint(mix64(key) >> 40u)) *
        (1.0f / 16777216.0f);
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

inline float4 yawQuaternion(const float4 orientation) {
    const float yaw = atan2(
        2.0f * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0f - 2.0f * (orientation.y * orientation.y + orientation.z * orientation.z)
    );
    return float4(0.0f, 0.0f, sin(0.5f * yaw), cos(0.5f * yaw));
}

inline float3 quaternionWorldAngularVelocity(
    const float4 first,
    const float4 second,
    const float framesPerSecond
) {
    const float4 firstUnit = normalize(first);
    const float4 secondUnit = normalize(second);
    float4 delta = quaternionProduct(
        secondUnit,
        float4(-firstUnit.xyz, firstUnit.w)
    );
    delta *= delta.w < 0.0f ? -1.0f : 1.0f;
    const float sineHalfAngle = length(delta.xyz);
    if (sineHalfAngle <= 1.0e-7f) {
        return 2.0f * framesPerSecond * delta.xyz;
    }
    const float angle = 2.0f * atan2(
        sineHalfAngle,
        clamp(delta.w, -1.0f, 1.0f)
    );
    return delta.xyz * (framesPerSecond * angle / sineHalfAngle);
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
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
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
    return rotate(
        rootOrientation(program, q),
        program.rootReference.xyz
    );
}

inline float3 rootWorldPosition(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
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
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
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

inline float3 rootWorldAngularVelocity(
    device const MRTaskProgramHeaderGPU& program,
    device const float* v
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
        return float3(0.0f);
    }
    return float3(
        v[program.root.w + 3u],
        v[program.root.w + 4u],
        v[program.root.w + 5u]
    );
}

inline float rootHeight(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    return rootWorldPosition(program, q).z;
}

inline float interactionFramePosition(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    if (program.interaction.x == 0u) {
        return 0.0f;
    }
    if (interactionPhysicsGated(program)) {
        const float position = packedInteractionFramePosition(state);
        return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
            ? fmod(position, float(program.interaction.x))
            : min(position, float(program.interaction.x - 1u));
    }
    float elapsedSeconds =
        float(state.episode.x) * controlStepSeconds;
    if (program.threat.y != 0u &&
        state.threatMetadata.x != MR_INVALID_INDEX) {
        // ARDY supplies imagined outcome timing. Align its final pose with
        // native closest approach rather than replaying from episode reset.
        elapsedSeconds = max(
            program.interactionTiming.y - state.threatGeometry.y,
            0.0f
        );
    } else if (program.threat.y != 0u) {
        elapsedSeconds = 0.0f;
    }
    const float unbounded =
        elapsedSeconds * program.interactionTiming.x;
    return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
        ? fmod(unbounded, float(program.interaction.x))
        : min(unbounded, float(program.interaction.x - 1u));
}

inline uint interactionFrame(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    return uint(floor(interactionFramePosition(
        program,
        state,
        controlStepSeconds
    )));
}

inline float interactionFrameBlend(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    const float position = interactionFramePosition(
        program,
        state,
        controlStepSeconds
    );
    return position - floor(position);
}

inline uint interactionNextFrame(
    device const MRTaskProgramHeaderGPU& program,
    const uint frame
) {
    return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
        ? (frame + 1u) % program.interaction.x
        : min(frame + 1u, program.interaction.x - 1u);
}

inline float4 quaternionInterpolate(
    const float4 first,
    float4 second,
    const float amount
) {
    second *= dot(first, second) < 0.0f ? -1.0f : 1.0f;
    return normalize(mix(first, second, amount));
}

inline float supportPatchFeature(
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskContactGroupGPU group,
    device const float* compactContact,
    const uint component
) {
    if (component < 6u) {
        return compactContact[group.reference.y + component];
    }
    if (component < 8u) {
        return compactContact[
            group.members.w + 4u + (component - 6u)
        ];
    }
    const float2 extent =
        group.supportPatchBounds.zw -
        group.supportPatchBounds.xy;
    const float cellArea =
        extent.x * extent.y /
        max(float(group.supportPatch.z), 1.0f);
    if (component == 8u) {
        uint occupied = 0u;
        for (uint cell = 0u;
             cell < group.supportPatch.z;
             ++cell) {
            occupied += compactContact[
                group.supportPatch.w + cell
            ] > program.dynamics.y ? 1u : 0u;
        }
        return float(occupied) * cellArea;
    }
    const uint cell = component - 9u;
    return compactContact[
        group.supportPatch.w + cell
    ] / max(cellArea, 1.0e-9f);
}

inline float cleanObservation(
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskObservationOperatorGPU operation,
    device const uchar* arena,
    const float controlStepSeconds,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
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
        value = rotateInverse(
            orientation,
            rootWorldAngularVelocity(program, v)
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
        value = operation.source.z < 3u
            ? state.commandAndPhase[operation.source.z]
            : state.commandExtension[operation.source.z - 3u];
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
    case MR_TASK_OBSERVE_JOINT_FINITE_DIFFERENCE_VELOCITY: {
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        value = (
            q[binding.indices.z] -
            previousJointPosition[operation.source.y]
        ) / controlStepSeconds;
        break;
    }
    case MR_TASK_OBSERVE_PREVIOUS_ACTION:
        value = previousAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_PREVIOUS_POLICY_ACTION:
        value = previousPolicyAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_DELAYED_ACTION:
        value = previousAction[
            operation.source.y -
            operation.source.z * program.counts0.x
        ];
        break;
    case MR_TASK_OBSERVE_INTERACTION_JOINT_POSITION_ERROR: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        const float reference = mix(
            targets[frame * program.interaction.y + operation.source.y],
            targets[nextFrame * program.interaction.y + operation.source.y],
            blend
        );
        value = q[binding.indices.z] - reference;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET:
    case MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET_VELOCITY: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const float current = targets[
            frame * program.interaction.y + operation.source.y
        ];
        const float next = targets[
            nextFrame * program.interaction.y + operation.source.y
        ];
        value = operation.source.x == MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET
            ? mix(current, next, blend)
            : (next - current) * program.interactionTiming.x;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_ANCHOR_ORIENTATION: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* rootTargets = taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
        device const float* jointTargets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const uint rootBase = frame * 7u;
        const uint nextRootBase = nextFrame * 7u;
        float4 referenceRoot = quaternionInterpolate(
            float4(
                rootTargets[rootBase + 3u], rootTargets[rootBase + 4u],
                rootTargets[rootBase + 5u], rootTargets[rootBase + 6u]
            ),
            float4(
                rootTargets[nextRootBase + 3u], rootTargets[nextRootBase + 4u],
                rootTargets[nextRootBase + 5u], rootTargets[nextRootBase + 6u]
            ),
            blend
        );
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_ALIGN_REFERENCE_YAW) != 0u) {
            const float4 initialReferenceRoot = float4(
                rootTargets[3u], rootTargets[4u], rootTargets[5u], rootTargets[6u]
            );
            const float4 referenceYaw = yawQuaternion(initialReferenceRoot);
            const float4 alignment = quaternionProduct(
                yawQuaternion(rootOrientation(program, defaultQ)),
                float4(-referenceYaw.xyz, referenceYaw.w)
            );
            referenceRoot = quaternionProduct(alignment, referenceRoot);
        }
        const float referenceWaist = mix(
            jointTargets[frame * program.interaction.y + operation.source.y],
            jointTargets[nextFrame * program.interaction.y + operation.source.y],
            blend
        );
        const MRTaskActionBindingGPU binding = actions[operation.source.y];
        const float liveWaist = q[binding.indices.z];
        const float4 referenceTorso = quaternionProduct(
            referenceRoot,
            float4(0.0f, 0.0f, sin(0.5f * referenceWaist),
                   cos(0.5f * referenceWaist))
        );
        const float4 liveTorso = quaternionProduct(
            orientation,
            float4(0.0f, 0.0f, sin(0.5f * liveWaist),
                   cos(0.5f * liveWaist))
        );
        // The source publishes the transpose of this relative rotation's
        // first two columns as its six-dimensional anchor representation.
        const float4 relative = quaternionProduct(
            float4(-referenceTorso.xyz, referenceTorso.w), liveTorso
        );
        const float x = relative.x;
        const float y = relative.y;
        const float z = relative.z;
        const float w = relative.w;
        const float values[6] = {
            1.0f - 2.0f * (y * y + z * z),
            2.0f * (x * y + z * w),
            2.0f * (x * y - z * w),
            1.0f - 2.0f * (x * x + z * z),
            2.0f * (x * z + y * w),
            2.0f * (y * z - x * w),
        };
        value = values[operation.source.z];
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        const MRTaskInteractionSampleGPU sample = samples[
            frame * program.interaction.z + operation.source.y
        ];
        if (operation.source.z == 0u) {
            value = sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_STICK ||
                    sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_ROLL ||
                    sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_SLIDE
                ? 1.0f
                : 0.0f;
        } else {
            value = sample.confidence.x;
        }
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint sampleIndex =
            frame * program.interaction.z + operation.source.y;
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        const uint feature = operation.source.z;
        if ((samples[sampleIndex].metadata.y &
             (1u << feature)) != 0u) {
            device const float* targets = taskTable<float>(
                arena,
                program.interactionOffsets1.x
            );
            value = targets[
                sampleIndex *
                    MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT +
                feature
            ];
        }
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint sampleIndex =
            frame * program.interaction.z + operation.source.y;
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        value = (samples[sampleIndex].metadata.y &
                 (1u << operation.source.z)) != 0u
            ? 1.0f
            : 0.0f;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_PHASE: {
        const float framePosition = interactionFramePosition(
            program,
            state,
            controlStepSeconds
        );
        const float progress = program.interaction.x > 1u
            ? framePosition / float(program.interaction.x - 1u)
            : 0.0f;
        value = operation.source.z == 0u
            ? sin(kTwoPi * progress)
            : operation.source.z == 1u
            ? cos(kTwoPi * progress)
            : progress;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_ROOT_TRACKING_ERROR: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
        const uint targetBase = frame * 7u;
        const uint nextBase = nextFrame * 7u;
        const float3 framePosition = float3(
            targets[targetBase + 0u],
            targets[targetBase + 1u],
            targets[targetBase + 2u]
        );
        const float4 frameOrientation = float4(
            targets[targetBase + 3u],
            targets[targetBase + 4u],
            targets[targetBase + 5u],
            targets[targetBase + 6u]
        );
        const float3 nextPosition = float3(
            targets[nextBase + 0u],
            targets[nextBase + 1u],
            targets[nextBase + 2u]
        );
        const float4 nextOrientation = float4(
            targets[nextBase + 3u],
            targets[nextBase + 4u],
            targets[nextBase + 5u],
            targets[nextBase + 6u]
        );
        const float3 targetPosition = mix(
            framePosition,
            nextPosition,
            blend
        );
        const float4 targetOrientation = quaternionInterpolate(
            frameOrientation,
            nextOrientation,
            blend
        );
        if (operation.source.z < 3u) {
            value = rotateInverse(
                orientation,
                targetPosition - rootWorldPosition(program, q)
            )[operation.source.z];
        } else if (operation.source.z < 6u) {
            float4 delta = quaternionProduct(
                float4(-orientation.xyz, orientation.w),
                targetOrientation
            );
            delta *= delta.w < 0.0f ? -1.0f : 1.0f;
            const float sineHalfAngle = length(delta.xyz);
            const float angle = sineHalfAngle > 1.0e-7f
                ? 2.0f * atan2(
                    sineHalfAngle,
                    clamp(delta.w, -1.0f, 1.0f)
                )
                : 2.0f * sineHalfAngle;
            const float3 orientationError = sineHalfAngle > 1.0e-7f
                ? delta.xyz * (angle / sineHalfAngle)
                : 2.0f * delta.xyz;
            value = orientationError[operation.source.z - 3u];
        } else if (operation.source.z < 9u) {
            const float3 targetVelocity =
                (nextPosition - framePosition) *
                program.interactionTiming.x;
            value = rotateInverse(
                orientation,
                targetVelocity - rootWorldLinearVelocity(program, q, v)
            )[operation.source.z - 6u];
        } else {
            const float3 targetAngularVelocity =
                quaternionWorldAngularVelocity(
                    frameOrientation,
                    nextOrientation,
                    program.interactionTiming.x
                );
            const float3 currentAngularVelocity = float3(
                v[program.root.w + 3u],
                v[program.root.w + 4u],
                v[program.root.w + 5u]
            );
            value = rotateInverse(
                orientation,
                targetAngularVelocity - currentAngularVelocity
            )[operation.source.z - 9u];
        }
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
    case MR_TASK_OBSERVE_SUPPORT_PATCH: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = supportPatchFeature(
            program,
            group,
            compactContact,
            operation.source.z
        );
        break;
    }
    case MR_TASK_OBSERVE_SUPPORT_SENSE: {
        float totalLoad = 0.0f;
        float signedLoad = 0.0f;
        float maximumSlip = 0.0f;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            const MRTaskContactGroupGPU group =
                contactGroups[groupIndex];
            if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                continue;
            }
            const uint metric = group.members.w;
            const float load = max(compactContact[metric], 0.0f);
            totalLoad += load;
            signedLoad += load * cos(group.gait.x);
            maximumSlip = max(
                maximumSlip,
                max(compactContact[metric + 1u], 0.0f)
            );
        }
        switch (operation.source.z) {
        case 0u:
            value = totalLoad;
            break;
        case 1u:
            value = totalLoad > 1.0e-6f
                ? signedLoad / totalLoad
                : 0.0f;
            break;
        default:
            value = maximumSlip;
            break;
        }
        break;
    }
    case MR_TASK_OBSERVE_GAIT_PHASE: {
        const float commandMagnitude = length(
            state.commandAndPhase.xyz
        );
        if (commandMagnitude >= 0.1f) {
            value = operation.source.z == 0u
                ? sin(state.commandAndPhase.w)
                : cos(state.commandAndPhase.w);
        }
        break;
    }
    case MR_TASK_OBSERVE_CYCLIC_PHASE:
        value = operation.source.z == 0u
            ? sin(state.commandAndPhase.w)
            : cos(state.commandAndPhase.w);
        break;
    case MR_TASK_OBSERVE_RECOVERY_EVENT:
        switch (operation.source.z) {
        case 0u:
            value = state.recovery.w;
            break;
        case 1u:
            value = state.recovery.x;
            break;
        case 2u:
            value = state.recovery.y;
            break;
        default:
            value = state.recovery.z;
            break;
        }
        break;
    case MR_TASK_OBSERVE_OBJECT_TRACK: {
        const MRBodyStateGPU object =
            sceneBodies[operation.source.y];
        const uint launchStep =
            (object.flagsAndIndices[3] &
             MR_BODY_STATE_LAUNCH_STEP_MASK) >>
            MR_BODY_STATE_LAUNCH_STEP_SHIFT;
        const bool visible = impactSequenceEnabled(state)
            ? impactScene(state) == operation.source.y + 1u
            : launchStep == 0u || state.episode.x >= launchStep;
        if (operation.source.z == 0u) {
            value = visible ? 1.0f : 0.0f;
            break;
        }
        if (!visible) {
            value = 0.0f;
            break;
        }
        const float3 relativePosition = rotateInverse(
            orientation,
            object.position.xyz - rootWorldPosition(program, q)
        );
        if (operation.source.z <= 3u) {
            value = relativePosition[operation.source.z - 1u];
            break;
        }
        const float3 relativeVelocity = rotateInverse(
            orientation,
            object.linearVelocityAndInverseMass.xyz -
                rootWorldLinearVelocity(program, q, v)
        );
        value = relativeVelocity[operation.source.z - 4u];
        break;
    }
    default:
        // Compilation rejects unknown sources. Device visual sources are a
        // separate direct suffix and never enter this physical-state reader.
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
    device const uchar* arena,
    device const MRTaskObservationOperatorGPU* actorOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
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
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
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
        const float value = cleanObservation(
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
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

inline void writeCurrentActor(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const uchar* arena,
    device const MRTaskObservationOperatorGPU* actorOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
    device const float* sensorBias,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* output
) {
    for (uint index = 0u; index < program.counts3.w; ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[program.layout.x + index];
        float value = cleanObservation(
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
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
        output[index] = value;
    }
}

inline void writeCriticFrame(
    device const MRTaskProgramHeaderGPU& program,
    device const uchar* arena,
    const float controlStepSeconds,
    device const MRTaskObservationOperatorGPU* criticOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
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
        output[index] = cleanObservation(
            program,
            operation,
            arena,
            controlStepSeconds,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
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

inline float4 scheduleSecondsForBand(
    device const MRTaskProgramHeaderGPU& program,
    const uint curriculum
) {
    float4 seconds = program.scheduleSeconds;
    if ((program.schedule.w & MR_TASK_PROGRAM_CLOCK_STRESS) == 0u ||
        program.schedule.z <= 1u) {
        return seconds;
    }
    const float progress = clamp(
        float(curriculum) /
            max(float(program.schedule.z - 1u), 1.0f),
        0.0f,
        1.0f
    );
    // Preserve the quiet adult rung, then progressively shorten both the
    // command horizon and disturbance interval. This models a finite
    // reaction budget without changing the physical impulse magnitude.
    const float commandScale = mix(1.0f, 0.45f, progress);
    const float disturbanceScale = mix(1.0f, 0.50f, progress);
    seconds.x = max(0.5f, seconds.x * commandScale);
    seconds.y = max(0.5f, seconds.y * commandScale);
    seconds.z = max(0.5f, seconds.z * disturbanceScale);
    seconds.w = max(0.5f, seconds.w * disturbanceScale);
    return seconds;
}

inline uint sampledDifficultyBand(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    const uint environment,
    const uint episode
) {
    const uint bandCount = max(program.schedule.z, 1u);
    const uint minimumBand = min(
        dispatch.sampling.x,
        bandCount - 1u
    );
    const uint requestedMaximum = dispatch.sampling.y == MR_INVALID_INDEX
        ? bandCount - 1u
        : dispatch.sampling.y;
    const uint maximumBand = max(
        minimumBand,
        min(requestedMaximum, bandCount - 1u)
    );
    const uint sampledBandCount = maximumBand - minimumBand + 1u;
    if (sampledBandCount == 1u) {
        return minimumBand;
    }
    const float exponent = max(program.commandUpper.w, 0.01f);
    const float sample = pow(
        randomUnit(dispatch, environment, episode, 0u, 15u),
        exponent
    );
    return minimumBand + min(
        uint(floor(sample * float(sampledBandCount))),
        sampledBandCount - 1u
    );
}

inline float3 sampledCommand(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const float4* curriculumRange,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const uint curriculum
) {
    const float3 lower = max(
        program.commandLower.xyz -
            float(curriculum) *
                curriculumRange[2].xyz,
        curriculumRange[0].xyz
    );
    const float3 upper = min(
        program.commandUpper.xyz +
            float(curriculum) *
                curriculumRange[2].xyz,
        curriculumRange[1].xyz
    );
    float3 command;
    for (uint component = 0u;
         component < 3u;
         ++component) {
        command[component] = randomRange(
            dispatch,
            environment,
            episode,
            episodeStep,
            16u + component,
            lower[component],
            upper[component]
        );
    }
    if (randomUnit(
            dispatch,
            environment,
            episode,
            episodeStep,
            19u
        ) < program.commandLower.w) {
        command = float3(0.0f);
    }
    return command;
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

} // namespace

kernel void mr_locomotion_task_latch_impact_contact(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    device const MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    constant MRMetalWorldPassGPU& pass [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }
    MRTaskStateGPU state = taskStates[environment];
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||
        impactOrder(state) >= program.counts3.x) {
        return;
    }
    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    if (contactStatus.code != MR_STEP_SUCCESS) {
        return;
    }
    device const MRTaskImpactEventGPU* impactEvents =
        reinterpret_cast<device const MRTaskImpactEventGPU*>(
            arena + program.offsets3.z
        );
    const uint activeImpactEvent =
        (impactOrder(state) + impactOffset(state)) %
        program.counts3.x;
    const uint projectileBody =
        impactEvents[activeImpactEvent].binding.w;
    const uint articulationBodyBegin = program.articulation.x;
    const uint articulationBodyEnd =
        articulationBodyBegin + program.articulation.y;
    const uint activeContacts = min(
        contactStatus.activeContacts,
        contactDispatch.constraintCapacity
    );
    const uint contactBase =
        environment * contactDispatch.constraintStride;
    for (uint contact = 0u; contact < activeContacts; ++contact) {
        const MRContactConstraintGPU constraint =
            contacts[contactBase + contact];
        const bool projectileA =
            constraint.bodyA == projectileBody;
        const bool projectileB =
            constraint.bodyB == projectileBody;
        if (projectileA == projectileB) {
            continue;
        }
        const uint other = projectileA
            ? constraint.bodyB
            : constraint.bodyA;
        if (other >= articulationBodyBegin &&
            other < articulationBodyEnd) {
            state.recoveryStats.w |= kImpactContactLatched;
            taskStates[environment] = state;
            return;
        }
    }
}

kernel void mr_locomotion_task_select_threat_query(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const MRBodyStateGPU* bodyStates [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.threat.y == 0u ||
        program.threat.x >= program.counts0.w ||
        contactDispatch.pointQueryStride == 0u) {
        return;
    }
    device const MRTaskContactGroupGPU* groups =
        taskTable<MRTaskContactGroupGPU>(arena, program.offsets0.w);
    device const uint* members =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* memberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(arena, program.offsets3.z);

    MRTaskStateGPU state = taskStates[environment];
    const uint queryBase =
        environment * contactDispatch.pointQueryStride;
    MRArticulatedPointImpulseGPU query = {};
    query.bodyIndex = program.root.y;
    query.localPoint = float4(0.0f);
    pointQueries[queryBase] = query;

    const uint order = impactOrder(state);
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||
        order >= program.counts3.x) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }
    const uint activeEvent =
        (order + impactOffset(state)) % program.counts3.x;
    const MRTaskImpactEventGPU event = impactEvents[activeEvent];
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU projectile =
        bodyStates[bodyBase + event.binding.w];
    const float3 velocity =
        projectile.linearVelocityAndInverseMass.xyz;
    const float horizontalSpeedSquared = dot(velocity.xy, velocity.xy);
    if (length(velocity) < program.threatTiming.x ||
        horizontalSpeedSquared < 1.0e-6f) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }

    const MRTaskContactGroupGPU protectedGroup = groups[program.threat.x];
    float bestClearance = INFINITY;
    float bestTime = 0.0f;
    float bestStrikeHeight = 0.0f;
    uint bestBody = MR_INVALID_INDEX;
    const float projectileRadius = event.projectile.x;
    for (uint local = 0u; local < protectedGroup.members.y; ++local) {
        const uint memberOffset = protectedGroup.members.x + local;
        const uint body = members[memberOffset];
        const float radius = memberRadii[memberOffset];
        if (!(radius > 0.0f)) {
            continue;
        }
        const float3 linkPosition = bodyStates[bodyBase + body].position.xyz;
        const float2 horizontalRelative =
            projectile.position.xy - linkPosition.xy;
        const float time = clamp(
            -dot(horizontalRelative, velocity.xy) /
                horizontalSpeedSquared,
            0.0f,
            program.threatTiming.y
        );
        const float3 predictedProjectile =
            projectile.position.xyz + velocity * time +
            0.5f * program.projectileGravity.xyz * time * time;
        const float safeRadius =
            radius + projectileRadius + program.threatTiming.z;
        const float clearance =
            length(predictedProjectile - linkPosition) - safeRadius;
        if (clearance < bestClearance) {
            bestClearance = clearance;
            bestTime = time;
            bestBody = body;
            bestStrikeHeight = predictedProjectile.z -
                bodyStates[bodyBase + program.root.y].position.z;
        }
    }
    if (bestBody == MR_INVALID_INDEX ||
        bestTime <= 0.0f ||
        bestTime >= program.threatTiming.y) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }

    uint threatClass = MR_TASK_THREAT_DUCK;
    if (bestStrikeHeight <= program.threatClassification.x) {
        threatClass = MR_TASK_THREAT_STEP_OVER;
    } else if (bestStrikeHeight <= program.threatClassification.y) {
        threatClass = MR_TASK_THREAT_SIDESTEP;
    } else if (bestStrikeHeight <= program.threatClassification.z) {
        threatClass = MR_TASK_THREAT_LEAN;
    }
    uint escape = state.threatMetadata.z;
    if (state.threatMetadata.w != activeEvent || escape > 2u ||
        escape == 1u) {
        const float2 rootDelta =
            bodyStates[bodyBase + program.root.y].position.xy -
            projectile.position.xy;
        const float crossTrack =
            velocity.x * rootDelta.y - velocity.y * rootDelta.x;
        const float sign = abs(crossTrack) > 1.0e-4f
            ? (crossTrack > 0.0f ? 1.0f : -1.0f)
            : ((environment & 1u) == 0u ? 1.0f : -1.0f);
        escape = sign > 0.0f ? 2u : 0u;
    }
    query.bodyIndex = bestBody;
    pointQueries[queryBase] = query;
    state.threatGeometry = float4(
        bestClearance,
        bestTime,
        bestStrikeHeight,
        bestClearance
    );
    state.threatTeacher = float4(0.0f);
    state.threatMetadata = uint4(
        bestBody,
        threatClass,
        escape,
        activeEvent
    );
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_joint_cbf_teacher(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const float* vState [[buffer(6)]],
    device float* actionStream [[buffer(7)]],
    device const float* defaultQ [[buffer(8)]],
    device const MRBodyStateGPU* bodyStates [[buffer(9)]],
    device const float* pointJacobians [[buffer(10)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(11)]],
    device MRTaskStateGPU* taskStates [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.threat.y == 0u) {
        return;
    }
    MRTaskStateGPU state = taskStates[environment];
    if (state.threatMetadata.x == MR_INVALID_INDEX ||
        state.threatMetadata.w == MR_INVALID_INDEX ||
        operatorStatuses[environment].code !=
            MR_ARTICULATED_OPERATOR_SUCCESS) {
        state.threatTeacher = float4(0.0f);
        taskStates[environment] = state;
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(arena, program.offsets0.x);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(arena, program.offsets3.z);
    device const MRTaskContactGroupGPU* groups =
        taskTable<MRTaskContactGroupGPU>(arena, program.offsets0.w);
    device const uint* members =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* memberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    const MRTaskImpactEventGPU event =
        impactEvents[state.threatMetadata.w];
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU projectile =
        bodyStates[bodyBase + event.binding.w];
    const MRBodyStateGPU threatened =
        bodyStates[bodyBase + state.threatMetadata.x];
    const float time = state.threatGeometry.y;
    const float3 predictedProjectile =
        projectile.position.xyz +
        projectile.linearVelocityAndInverseMass.xyz * time +
        0.5f * program.projectileGravity.xyz * time * time;
    float3 separation = threatened.position.xyz - predictedProjectile;
    const float escapeSign = state.threatMetadata.z == 2u ? 1.0f : -1.0f;
    if (length(separation.xy) < 1.0e-3f) {
        const float2 horizontalVelocity =
            projectile.linearVelocityAndInverseMass.xy;
        const float horizontalSpeed = length(horizontalVelocity);
        const float2 flight = horizontalSpeed > 1.0e-6f
            ? horizontalVelocity / horizontalSpeed
            : float2(1.0f, 0.0f);
        separation.xy =
            escapeSign * float2(-flight.y, flight.x) * 1.0e-3f;
    }
    const float distance = max(length(separation), 1.0e-6f);
    float linkRadius = 0.0f;
    const MRTaskContactGroupGPU group = groups[program.threat.x];
    for (uint local = 0u; local < group.members.y; ++local) {
        const uint offset = group.members.x + local;
        if (members[offset] == state.threatMetadata.x) {
            linkRadius = memberRadii[offset];
            break;
        }
    }
    const float safeRadius =
        linkRadius + event.projectile.x + program.threatTiming.z;
    const float h = distance * distance - safeRadius * safeRadius;
    const uint jacobianBase =
        environment * contactDispatch.pointQueryStride *
        3u * dispatch.counts.w;
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint actionBase =
        pass.controlStep * dispatch.strides.x +
        environment * program.counts0.x;
    float3 desiredPointVelocity = float3(0.0f);
    for (uint dof = 0u; dof < dispatch.counts.w; ++dof) {
        const float velocity = vState[vBase + dof];
        desiredPointVelocity += float3(
            pointJacobians[jacobianBase + 0u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 1u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 2u * dispatch.counts.w + dof]
        ) * velocity;
    }
    float gradientNormSquared = 0.0f;
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );
    for (uint action = 0u; action < program.counts0.x; ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        const uint dof = binding.indices.w - program.root.w;
        if (dof >= dispatch.counts.w) {
            continue;
        }
        const float reference =
            (program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u
            ? mix(
                  interactionJointTargets[
                      referenceFrame * program.interaction.y + action
                  ],
                  interactionJointTargets[
                      nextReferenceFrame * program.interaction.y + action
                  ],
                  referenceBlend
              )
            : defaultQ[binding.indices.z];
        const float target = clamp(
            reference +
                binding.parameters.x * clamp(
                    actionStream[actionBase + action],
                    -1.0f,
                    1.0f
                ),
            binding.parameters.y,
            binding.parameters.z
        );
        const float desiredVelocity =
            (target - qState[qBase + binding.indices.z]) /
            program.threatTeacher.y;
        const float currentVelocity = vState[vBase + binding.indices.w];
        const float3 column = float3(
            pointJacobians[jacobianBase + 0u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 1u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 2u * dispatch.counts.w + dof]
        );
        desiredPointVelocity += column *
            (desiredVelocity - currentVelocity);
        const float gradient = 2.0f * dot(separation, column);
        gradientNormSquared += gradient * gradient;
    }
    const float3 projectileVelocity =
        projectile.linearVelocityAndInverseMass.xyz +
        program.projectileGravity.xyz * time;
    const float barrierRate = 2.0f * dot(
        separation,
        desiredPointVelocity - projectileVelocity
    );
    const float urgencyFraction = clamp(
        (program.threatTeacher.x - time) /
            program.threatTeacher.x,
        0.0f,
        1.0f
    );
    const float urgency =
        urgencyFraction * safeRadius * safeRadius /
        program.threatTeacher.x;
    const float constraint =
        barrierRate + program.threatTiming.w * h;
    const float deficit = max(urgency - constraint, 0.0f);
    const float projectionScale = deficit /
        (gradientNormSquared + program.threatTeacher.z);
    const float correctionRms = projectionScale * sqrt(
        gradientNormSquared /
        max(float(program.counts0.x), 1.0f)
    );
    state.threatGeometry.w = constraint;
    state.threatTeacher = float4(
        correctionRms,
        max(safeRadius - distance, 0.0f),
        constraint + projectionScale * gradientNormSquared - urgency,
        urgency
    );
    if ((program.schedule.w &
         MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u) {
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            const MRTaskActionBindingGPU binding = actions[action];
            const uint dof = binding.indices.w - program.root.w;
            if (dof >= dispatch.counts.w ||
                !(binding.parameters.x > 0.0f)) {
                continue;
            }
            const float3 column = float3(
                pointJacobians[
                    jacobianBase + 0u * dispatch.counts.w + dof
                ],
                pointJacobians[
                    jacobianBase + 1u * dispatch.counts.w + dof
                ],
                pointJacobians[
                    jacobianBase + 2u * dispatch.counts.w + dof
                ]
            );
            const float gradient = 2.0f * dot(separation, column);
            const float reference = interactionJointTargets[
                referenceFrame * program.interaction.y + action
            ];
            const float requestedTarget = clamp(
                reference + binding.parameters.x * clamp(
                    actionStream[actionBase + action],
                    -1.0f,
                    1.0f
                ),
                binding.parameters.y,
                binding.parameters.z
            );
            const float correctedTarget = clamp(
                requestedTarget +
                    program.threatTeacher.y *
                    projectionScale * gradient,
                binding.parameters.y,
                binding.parameters.z
            );
            actionStream[actionBase + action] = clamp(
                (correctedTarget - reference) /
                    binding.parameters.x,
                -1.0f,
                1.0f
            );
        }
    }
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_motion_features(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const MRBodyStateGPU* bodyStates [[buffer(5)]],
    device float* features [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.motion.y == 0u ||
        program.motion.z != 9u * program.motion.y) {
        return;
    }
    device const uint* trackedBodies =
        taskTable<uint>(arena, program.motion.w);
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU anchor =
        bodyStates[bodyBase + program.motion.x];
    const float4 inverseAnchor = float4(
        -anchor.orientation.xyz,
        anchor.orientation.w
    );
    const uint outputBase =
        (pass.controlStep * dispatch.counts.x + environment) *
        program.motion.z;
    for (uint index = 0u; index < program.motion.y; ++index) {
        const MRBodyStateGPU body =
            bodyStates[bodyBase + trackedBodies[index]];
        const float3 position = rotateInverse(
            anchor.orientation,
            body.position.xyz - anchor.position.xyz
        );
        float4 orientation = quaternionProduct(
            inverseAnchor,
            body.orientation
        );
        orientation *= orientation.w < 0.0f ? -1.0f : 1.0f;
        const float x = orientation.x;
        const float y = orientation.y;
        const float z = orientation.z;
        const float w = orientation.w;
        const uint base = outputBase + 9u * index;
        features[base + 0u] = position.x;
        features[base + 1u] = position.y;
        features[base + 2u] = position.z;
        // First two rotation-matrix columns, row-major, matching PAC-MAN's
        // continuous 6D orientation representation.
        features[base + 3u] = 1.0f - 2.0f * (y * y + z * z);
        features[base + 4u] = 2.0f * (x * y - z * w);
        features[base + 5u] = 2.0f * (x * y + z * w);
        features[base + 6u] = 1.0f - 2.0f * (x * x + z * z);
        features[base + 7u] = 2.0f * (x * z - y * w);
        features[base + 8u] = 2.0f * (y * z + x * w);
    }
}

kernel void mr_locomotion_task_observe(
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
    device const MRTaskEvidenceStateGPU* evidenceState
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
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    device const float* initialActionPositions = taskTable<float>(
        arena,
        program.actuatorTerms.z
    );
    device const float* interactionRootTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.x
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
    device const float4* commandCurriculum =
        taskTable<float4>(arena, program.offsets3.y);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(
            arena,
            program.offsets3.z
        );
    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
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
    const uint contactBase =
        environment * program.layout.z;
    const uint sceneBase =
        environment * dispatch.strides.w;
    MRTaskStateGPU state = taskStates[environment];
    const bool reset =
        state.status.x == 0u ||
        state.status.y != 0u ||
        resetMasks[maskIndex] != 0u;
    resetMasks[maskIndex] = reset ? 1u : 0u;

    if (reset) {
        const uint episode = state.episode.y + 1u;
        const uint curriculum = sampledDifficultyBand(
            dispatch,
            program,
            environment,
            episode
        );
        state.episode.z = curriculum;
        const uint terrainLevel =
            program.terrain.w == 0u
            ? 0u
            : min(curriculum, program.terrain.w - 1u);
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.z;
             ++coordinate) {
            resetQ[qBase + coordinate] =
                defaultQ[coordinate];
        }
        if (program.actuatorTerms.w == program.counts0.x) {
            for (uint action = 0u; action < program.counts0.x; ++action) {
                resetQ[qBase + actions[action].indices.z] =
                    initialActionPositions[action];
            }
        }
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) != 0u) {
            for (uint action = 0u;
                 action < program.interaction.y;
                 ++action) {
                const MRTaskActionBindingGPU binding =
                    actions[action];
                resetQ[qBase + binding.indices.z] =
                    interactionJointTargets[action];
            }
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
            const uint launchStep =
                (scene.flagsAndIndices[3] &
                 MR_BODY_STATE_LAUNCH_STEP_MASK) >>
                MR_BODY_STATE_LAUNCH_STEP_SHIFT;
            if (launchStep != 0u ||
                (scene.flagsAndIndices[3] &
                 MR_BODY_STATE_PRESERVE_RESET_VELOCITY) == 0u) {
                scene.linearVelocityAndInverseMass.xyz =
                    float3(0.0f);
                scene.angularVelocity = float4(0.0f);
            }
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
            const uint qIndex = actions[action].indices.z;
            previousJointVelocity[
                previousVelocityBase + program.counts0.x + action
            ] = qIndex == MR_INVALID_INDEX
                ? 0.0f
                : resetQ[qBase + qIndex];
            for (uint delay = 0u;
                 delay < program.layout.w;
                 ++delay) {
                actionHistory[
                    delayBase +
                    delay * program.counts0.x +
                    action
                ] = 0.0f;
            }
            rawPolicyActions[
                environment * program.counts0.x + action
            ] = 0.0f;
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
            case MR_TASK_RANDOMIZE_ROOT_HEIGHT:
                resetQ[qBase + program.root.z + 2u] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_ROOT_ORIENTATION:
                for (uint component = 0u; component < 4u; ++component) {
                    resetQ[qBase + program.root.z + 3u + component] =
                        operation.parameters[component];
                }
                break;
            case MR_TASK_RANDOMIZE_JOINT_POSITION:
                resetQ[qBase + operation.target.y] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_POSITION:
                resetScene[
                    sceneBase + operation.target.y
                ].position[operation.target.z] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY:
                resetScene[
                    sceneBase + operation.target.y
                ].linearVelocityAndInverseMass[
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
            case MR_TASK_RANDOMIZE_SCENE_BODY_LAUNCH_STEP: {
                const uint lower = uint(operation.parameters.x);
                const uint upper = uint(operation.parameters.y);
                const uint launchStep = lower + uint(floor(
                    float(upper - lower + 1u) * randomUnit(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        channel
                    )
                ));
                device MRBodyStateGPU& scene = resetScene[
                    sceneBase + operation.target.y
                ];
                scene.flagsAndIndices[3] =
                    (scene.flagsAndIndices[3] &
                     ~MR_BODY_STATE_LAUNCH_STEP_MASK) |
                    ((launchStep << MR_BODY_STATE_LAUNCH_STEP_SHIFT) &
                     MR_BODY_STATE_LAUNCH_STEP_MASK);
                break;
            }
            case MR_TASK_RANDOMIZE_ACTION_POSITION:
                for (uint action = 0u;
                     action < program.counts0.x;
                     ++action) {
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference =
                        (program.schedule.w &
                         MR_TASK_PROGRAM_INTERACTION_RESET) != 0u
                        ? interactionJointTargets[action]
                        : defaultQ[binding.indices.z];
                    resetQ[qBase + binding.indices.z] =
                        clamp(
                            reference +
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
            case MR_TASK_RANDOMIZE_WORLD_BODY_PARAMETER:
                bodyParameters[
                    bodyParameterBase + operation.target.y
                ][operation.target.z] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
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
        uint interactionResetStep = 0u;
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) != 0u) {
            if (program.interactionCurriculum.x > 0.0f &&
                program.interactionCurriculum.y > 0.0f &&
                program.interaction.x > 1u &&
                program.interactionTiming.x > 0.0f &&
                dispatch.timing.x > 0.0f &&
                program.schedule.x > 1u) {
                const float clipControlSteps =
                    float(program.interaction.x - 1u) /
                    (program.interactionTiming.x * dispatch.timing.x);
                const uint maximumResetStep = uint(floor(
                    min(
                        float(program.schedule.x - 2u),
                        clipControlSteps
                    ) * program.interactionCurriculum.y
                ));
                const float curriculumSample = randomUnit(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    4094u
                );
                const float canonicalFraction =
                    1.0f - program.interactionCurriculum.x;
                if (curriculumSample >= canonicalFraction) {
                    const float phaseSample =
                        (curriculumSample - canonicalFraction) /
                        program.interactionCurriculum.x;
                    interactionResetStep = min(
                        uint(floor(
                            phaseSample *
                            float(maximumResetStep + 1u)
                        )),
                        maximumResetStep
                    );
                }
            }
            const float resetFramePosition = min(
                float(interactionResetStep) *
                    dispatch.timing.x *
                    program.interactionTiming.x,
                float(program.interaction.x - 1u)
            );
            const uint resetFrame = uint(floor(resetFramePosition));
            const uint nextResetFrame = interactionNextFrame(
                program,
                resetFrame
            );
            const float resetBlend =
                resetFramePosition - floor(resetFramePosition);
            const uint rootBase = resetFrame * 7u;
            const uint nextRootBase = nextResetFrame * 7u;
            const float4 frameOrientation = float4(
                interactionRootTargets[rootBase + 3u],
                interactionRootTargets[rootBase + 4u],
                interactionRootTargets[rootBase + 5u],
                interactionRootTargets[rootBase + 6u]
            );
            const float4 nextOrientation = float4(
                interactionRootTargets[nextRootBase + 3u],
                interactionRootTargets[nextRootBase + 4u],
                interactionRootTargets[nextRootBase + 5u],
                interactionRootTargets[nextRootBase + 6u]
            );
            const float4 targetOrientation = quaternionInterpolate(
                frameOrientation,
                nextOrientation,
                resetBlend
            );
            const float3 frameRootLinkPosition = float3(
                interactionRootTargets[rootBase + 0u],
                interactionRootTargets[rootBase + 1u],
                interactionRootTargets[rootBase + 2u]
            );
            const float3 nextRootLinkPosition = float3(
                interactionRootTargets[nextRootBase + 0u],
                interactionRootTargets[nextRootBase + 1u],
                interactionRootTargets[nextRootBase + 2u]
            );
            // Generalized floating-root translation is the root body's COM,
            // while InteractionPack and task observations author the root-link
            // origin. rootReference is link-origin minus COM in body space.
            const float3 frameRootCOMPosition =
                frameRootLinkPosition - rotate(
                    frameOrientation,
                    program.rootReference.xyz
                );
            const float3 nextRootCOMPosition =
                nextRootLinkPosition - rotate(
                    nextOrientation,
                    program.rootReference.xyz
                );
            const float3 targetRootCOMPosition = mix(
                frameRootCOMPosition,
                nextRootCOMPosition,
                resetBlend
            );
            resetQ[qBase + program.root.z + 0u] =
                targetRootCOMPosition.x;
            resetQ[qBase + program.root.z + 1u] =
                targetRootCOMPosition.y;
            resetQ[qBase + program.root.z + 2u] =
                targetRootCOMPosition.z;
            resetQ[qBase + program.root.z + 3u] = targetOrientation.x;
            resetQ[qBase + program.root.z + 4u] = targetOrientation.y;
            resetQ[qBase + program.root.z + 5u] = targetOrientation.z;
            resetQ[qBase + program.root.z + 6u] = targetOrientation.w;
            resetV[vBase + program.root.w + 0u] =
                (nextRootCOMPosition.x - frameRootCOMPosition.x) *
                program.interactionTiming.x;
            resetV[vBase + program.root.w + 1u] =
                (nextRootCOMPosition.y - frameRootCOMPosition.y) *
                program.interactionTiming.x;
            resetV[vBase + program.root.w + 2u] =
                (nextRootCOMPosition.z - frameRootCOMPosition.z) *
                program.interactionTiming.x;
            const float3 rootAngularVelocity =
                quaternionWorldAngularVelocity(
                    frameOrientation,
                    nextOrientation,
                    program.interactionTiming.x
                );
            resetV[vBase + program.root.w + 3u] =
                rootAngularVelocity.x;
            resetV[vBase + program.root.w + 4u] =
                rootAngularVelocity.y;
            resetV[vBase + program.root.w + 5u] =
                rootAngularVelocity.z;
            for (uint action = 0u;
                 action < program.interaction.y;
                 ++action) {
                const MRTaskActionBindingGPU binding = actions[action];
                const uint jointIndex =
                    resetFrame * program.interaction.y + action;
                const uint nextJointIndex =
                    nextResetFrame * program.interaction.y + action;
                resetQ[qBase + binding.indices.z] =
                    mix(
                        interactionJointTargets[jointIndex],
                        interactionJointTargets[nextJointIndex],
                        resetBlend
                    );
                resetV[vBase + binding.indices.w] =
                    (
                        interactionJointTargets[nextJointIndex] -
                        interactionJointTargets[jointIndex]
                    ) * program.interactionTiming.x;
            }
        }
        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;

        const float4 scheduleSeconds = scheduleSecondsForBand(
            program,
            curriculum
        );

        state.episode = uint4(
            interactionResetStep,
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
                scheduleSeconds.x,
                scheduleSeconds.y
            ),
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                33u,
                scheduleSeconds.z,
                scheduleSeconds.w
            ),
            actionDelay,
            observationDelay
        );
        uint interactionClock = 0u;
        if (interactionPhysicsGated(program)) {
            const float resetFramePosition = min(
                float(interactionResetStep) *
                    dispatch.timing.x *
                    program.interactionTiming.x,
                float(program.interaction.x - 1u)
            );
            interactionClock = packInteractionClock(
                resetFramePosition,
                0u,
                false
            );
        }
        state.status = uint4(
            1u,
            0u,
            MR_TASK_TERMINATION_CONTINUING,
            interactionClock
        );
        state.commandAndPhase = float4(
            sampledCommand(
                dispatch,
                program,
                commandCurriculum,
                environment,
                episode,
                0u,
                curriculum
            ),
            0.0f
        );
        state.commandExtension = float4(0.0f);
        state.airReturnTracking = float4(
            0.0f,
            rootHeight(program, resetQ + qBase),
            0.0f,
            0.0f
        );
        state.recovery = float4(0.0f);
        state.recoveryStats = uint4(0u);
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        const bool projectileEpisode = randomUnit(
            dispatch,
            environment,
            episode,
            0u,
            3069u
        ) >= program.dynamics.z;
        for (uint impact = 0u;
             impact < program.counts3.x;
             ++impact) {
            const MRTaskImpactEventGPU event =
                impactEvents[impact];
            if (curriculum < event.binding.z ||
                !projectileEpisode) {
                continue;
            }
            state.recoveryStats.w |= kImpactEnabled;
            device MRBodyStateGPU& held = resetScene[
                sceneBase + event.binding.x
            ];
            held.linearVelocityAndInverseMass.xyz =
                float3(0.0f);
            held.angularVelocity = float4(0.0f);
        }
        if (impactSequenceEnabled(state)) {
            const uint count = program.counts3.x;
            const uint offset = min(
                uint(floor(
                    float(count) * randomUnit(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        3072u
                    )
                )),
                count - 1u
            );
            state.recoveryStats.w |=
                offset << kImpactOffsetShift;
        }

        device float* firstActor =
            actorHistory + historyBase;
        device float* firstClean =
            cleanHistory + historyBase;
        writeFrame(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            episode,
            0u,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
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
            program,
            arena,
            dispatch.timing.x,
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
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

    if (!reset &&
        impactSequenceEnabled(state) &&
        program.counts3.x > 0u &&
        state.episode.z >= impactEvents[0].binding.z) {
        const uint order = impactOrder(state);
        uint activeScene = impactScene(state);
        bool newlyLaunched = false;
        if (order < program.counts3.x) {
            const uint eventIndex =
                (order + impactOffset(state)) %
                program.counts3.x;
            const MRTaskImpactEventGPU event =
                impactEvents[eventIndex];
            if (activeScene == 0u) {
                const bool stable =
                    state.recovery.w <= 0.5f &&
                    state.recovery.x <= event.gate.x &&
                    rootHeight(program, sourceQ + qBase) >=
                        event.gate.w;
                state.status.w = stable
                    ? state.status.w + 1u
                    : 0u;
                if (float(state.status.w) * dispatch.timing.x >=
                    event.gate.y) {
                    activeScene = event.binding.x + 1u;
                    state.recoveryStats.w =
                        (state.recoveryStats.w &
                         ~(kImpactSceneMask)) |
                        (activeScene << kImpactSceneShift);
                    state.status.w = 0u;
                    newlyLaunched = true;
                }
            } else {
                state.status.w += 1u;
            }
        }
        for (uint impact = 0u;
             impact < program.counts3.x;
             ++impact) {
            const MRTaskImpactEventGPU event =
                impactEvents[impact];
            const uint currentEvent =
                order < program.counts3.x
                ? (order + impactOffset(state)) %
                    program.counts3.x
                : MR_INVALID_INDEX;
            if (state.episode.z < event.binding.z ||
                (impact == currentEvent &&
                 activeScene != 0u &&
                 !newlyLaunched)) {
                continue;
            }
            MRBodyStateGPU scheduled = resetScene[
                sceneBase + event.binding.x
            ];
            scheduled.linearVelocityAndInverseMass.xyz =
                float3(0.0f);
            scheduled.angularVelocity = float4(0.0f);
            if (impact == currentEvent && newlyLaunched) {
                const MRBodyStateGPU initial = initialScene[
                    sceneBase + event.binding.x
                ];
                scheduled.linearVelocityAndInverseMass.xyz =
                    initial.linearVelocityAndInverseMass.xyz;
                scheduled.angularVelocity = initial.angularVelocity;
                for (uint index = 0u;
                     index < program.counts2.y;
                     ++index) {
                    const MRTaskRandomizationOperatorGPU operation =
                        randomization[index];
                    if (operation.target.x !=
                            MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY ||
                        operation.target.y != event.binding.x ||
                        state.episode.z < operation.target.w) {
                        continue;
                    }
                    scheduled.linearVelocityAndInverseMass[
                        operation.target.z
                    ] = randomRange(
                        dispatch,
                        environment,
                        state.episode.y,
                        0u,
                        2048u + index,
                        operation.parameters.x,
                        operation.parameters.y
                    );
                }
                if (program.projectile.y > 0.0f) {
                    const float targetRadius =
                        program.projectileGravity.w;
                    const float3 target = float3(
                        sourceQ[qBase + program.root.z] + randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4096u + impact * 4u,
                            -targetRadius,
                            targetRadius
                        ),
                        sourceQ[qBase + program.root.z + 1u] + randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4097u + impact * 4u,
                            -targetRadius,
                            targetRadius
                        ),
                        randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4098u + impact * 4u,
                            program.projectile.z,
                            program.projectile.w
                        )
                    );
                    // Scene-body velocity randomization owns the authored
                    // curriculum bands. Preserve its horizontal magnitude
                    // when retargeting the throw instead of replacing every
                    // level with the task-wide fallback range.
                    const float authoredHorizontalSpeed = length(
                        scheduled.linearVelocityAndInverseMass.xy
                    );
                    const float horizontalSpeed =
                        authoredHorizontalSpeed > 0.0f
                        ? authoredHorizontalSpeed
                        : randomRange(
                              dispatch,
                              environment,
                              state.episode.y,
                              0u,
                              4099u + impact * 4u,
                              program.projectile.x,
                              program.projectile.y
                          );
                    const float3 delta = target - scheduled.position.xyz;
                    const float flightSeconds = max(
                        length(delta.xy) / horizontalSpeed,
                        dispatch.timing.x
                    );
                    scheduled.linearVelocityAndInverseMass.xyz =
                        delta / flightSeconds -
                        0.5f * program.projectileGravity.xyz *
                            flightSeconds;
                }
            }
            sourceScene[
                sceneBase + event.binding.x
            ] = scheduled;
        }
    } else if (!reset) {
        for (uint localScene = 0u;
             localScene < dispatch.strides.w;
             ++localScene) {
            const MRBodyStateGPU authored =
                resetScene[sceneBase + localScene];
            const uint launchStep =
                (authored.flagsAndIndices[3] &
                 MR_BODY_STATE_LAUNCH_STEP_MASK) >>
                MR_BODY_STATE_LAUNCH_STEP_SHIFT;
            if (launchStep == 0u ||
                state.episode.x > launchStep) {
                continue;
            }
            MRBodyStateGPU scheduled = authored;
            if (state.episode.x < launchStep) {
                scheduled.linearVelocityAndInverseMass.xyz =
                    float3(0.0f);
                scheduled.angularVelocity = float4(0.0f);
            } else {
                // A delayed projectile is parked at reset, so resetScene
                // deliberately has zero velocity.  Restore the authored
                // launch state on its exact episode tick rather than letting
                // it fall vertically from its staging point.
                scheduled = initialScene[sceneBase + localScene];
            }
            sourceScene[sceneBase + localScene] = scheduled;
        }
    }

    taskStates[environment] = state;
    const uint actorObservationSize =
        dispatch.outputs.x / dispatch.counts.x;
    const uint actorOutputBase =
        pass.controlStep * dispatch.outputs.x +
        environment * actorObservationSize;
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
    device const float* observationQ = reset
        ? resetQ + qBase
        : sourceQ + qBase;
    device const float* observationV = reset
        ? resetV + vBase
        : sourceV + vBase;
    device const MRBodyStateGPU* observationScene = reset
        ? resetScene + sceneBase
        : sourceScene + sceneBase;
    writeCurrentActor(
        dispatch,
        program,
        arena,
        actorOperators,
        actions,
        contactGroups,
        terrainSamples,
        environment,
        state.episode.y,
        0u,
        observationQ,
        observationV,
        defaultQ,
        state,
        actionHistory + delayBase,
        rawPolicyActions + environment * program.counts0.x,
        previousJointVelocity + previousVelocityBase + program.counts0.x,
        sensorBias + biasBase,
        compactContact + contactBase,
        bodyParameters + bodyParameterBase,
        controllerParameters + environment,
        observationScene,
        shapes,
        geometryHeaders,
        geometryVertices,
        actorObservations + actorOutputBase + historyElements
    );
    publishCritic(
        program,
        cleanHistory + historyBase,
        criticHistory + criticHistoryBase,
        criticObservations + criticOutputBase
    );

    if (!reset && state.schedule.y == 0u &&
        program.dynamics.x > 0.0f) {
        const float progress = clamp(
            float(state.episode.z) /
                max(float(program.schedule.z - 1u), 1.0f),
            0.0f,
            1.0f
        );
        sourceV[vBase + program.root.w + 0u] +=
            progress * program.dynamics.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                48u
            );
        sourceV[vBase + program.root.w + 1u] +=
            progress * program.dynamics.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                49u
            );
    }

}

kernel void mr_locomotion_task_apply_actions(
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
    device float* teacherActions [[buffer(10)]],
    device const float* rawPolicyLatents [[buffer(11)]],
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
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    const uint actionBase =
        pass.controlStep * dispatch.strides.x +
        environment * program.counts0.x;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
    const MRTaskStateGPU state = taskStates[environment];
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );

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
    const uint filterSlot = program.layout.w - 1u;
    const uint rawLastSlot = filterSlot - 1u;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        for (uint delay = 0u;
             delay < rawLastSlot;
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
        // PolicyPack owns the action bound.  The task still clamps the
        // resulting position target to the joint range below, so policies
        // trained with residuals outside [-1, 1] retain their source action
        // semantics without weakening physical target safety.
        const float requested = actionStream[actionBase + action];
        rawPolicyActions[
            environment * program.counts0.x + action
        ] = rawPolicyLatents[actionBase + action];
        const float previous = actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ];
        const MRTaskActionBindingGPU binding =
            actions[action];
        const float responseTimeSeconds = binding.parameters.w;
        const float responseFraction =
            responseTimeSeconds > 0.0f
            ? clamp(
                  1.0f - exp(
                      -dispatch.timing.x /
                      responseTimeSeconds
                  ),
                  0.0f,
                  1.0f
              )
            : 1.0f;
        actionHistory[
            delayBase +
            rawLastSlot * program.counts0.x +
            action
        ] = requested;
        const uint selected =
            rawLastSlot -
            min(state.schedule.z, rawLastSlot);
        const float delayed = actionHistory[
            delayBase +
            selected * program.counts0.x +
            action
        ];
        actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ] = mix(
            previous,
            delayed,
            responseFraction
        );
        const float filtered = actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ];
        if (binding.actuator.x != MR_TASK_ACTUATOR_JOINT_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_GRIPPER_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_FLAPPING_POSITION) {
            // Velocity, effort, tendon, and body-wrench commands are
            // evaluated from live microstep state by the generic actuator
            // pass. Rotor mixers have their own compiled robot program. A
            // task without an executable TeacherPack has no teacher-action
            // allocation, so none of these branches may touch that stream.
            continue;
        }
        const bool interactionReference =
            (program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u;
        const bool avianActionSet =
            program.counts0.x >= 9u &&
            actions[0].actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION &&
            actions[1].actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION;
        const bool avianGroundCurriculum = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_GROUND_CURRICULUM) != 0u &&
            state.episode.z < 2u;
        const bool avianStandingCurriculum = avianGroundCurriculum &&
            state.episode.z == 0u;
        const bool avianCrowGroundGaitCarrier = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_GROUND_GAIT_CARRIER) != 0u &&
            state.episode.z == 1u;
        const bool avianCrowLiftoffTrimCarrier = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_LIFTOFF_TRIM_CARRIER) != 0u &&
            state.episode.z == 2u;
        // The live action sweep brackets the estimated hybrid's transition:
        // +0.100 remains ground-bound whereas +0.125 repeatedly reaches the
        // altitude boundary.  A positive stroke-plane tilt then supplies
        // forward authority, so trim on the previous accepted root height,
        // vertical rate, and yaw-frame forward speed together.  The lower
        // wing limit must remain below the static-liftoff threshold: after a
        // real climb or forward overspeed the carrier must be able to remove
        // thrust, not merely reduce an always-positive stroke.
        // This is a Metal-resident controller of the actual wing positions,
        // never an injected aerodynamic force or a prerecorded trajectory.
        float avianLiftoffWingCarrier = 0.0f;
        if (avianCrowLiftoffTrimCarrier &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION) {
            const float heightError = 0.85f - state.airReturnTracking.y;
            const float verticalRate = state.commandExtension.w;
            const float forwardSpeedError =
                state.commandExtension.z - 0.35f;
            avianLiftoffWingCarrier = clamp(
                -0.050f + 0.200f * heightError - 0.060f * verticalRate -
                    0.100f * forwardSpeedError,
                -0.750f,
                0.125f
            );
        }
        const float avianWingPolicyCommand = avianCrowLiftoffTrimCarrier
            ? clamp(
                avianLiftoffWingCarrier + 0.25f * filtered,
                -1.0f,
                1.0f
            )
            : filtered;
        const float avianWingAmplitude =
            avianGroundCurriculum &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f
            : clamp(
                binding.drive.z + binding.drive.w * avianWingPolicyCommand,
                0.0f,
                1.0f
            );
        float avianGroundGaitCarrier = 0.0f;
        if (avianCrowGroundGaitCarrier) {
            const float phase = kTwoPi *
                float(state.episode.x) * dispatch.timing.x / 0.50f;
            const float leftSwing = sin(phase);
            const float rightSwing = -leftSwing;
            switch (binding.indices.x) {
            case 3u:
                avianGroundGaitCarrier = -0.014f * leftSwing;
                break;
            case 4u:
                avianGroundGaitCarrier = 0.018f * max(leftSwing, 0.0f);
                break;
            case 5u:
                avianGroundGaitCarrier = -0.010f * max(leftSwing, 0.0f);
                break;
            case 6u:
                avianGroundGaitCarrier = -0.014f * rightSwing;
                break;
            case 7u:
                avianGroundGaitCarrier = 0.018f * max(rightSwing, 0.0f);
                break;
            case 8u:
                avianGroundGaitCarrier = -0.010f * max(rightSwing, 0.0f);
                break;
            default:
                break;
            }
        }
        // Stage 1 is a residual-learning problem around the qualified gait
        // carrier.  The learned action has a deliberately narrower authority
        // than the carrier: early policy updates can improve timing and trim
        // without trivially cancelling the stable walking cycle.  Later
        // liftoff and flight bands retain their full policy authority.
        const float avianGroundResidualScale =
            avianCrowGroundGaitCarrier ? 0.25f : 1.0f;
        const float avianLiftoffTailCarrier =
            avianCrowLiftoffTrimCarrier && binding.indices.x == 2u
            ? clamp(
                0.25f + 0.030f *
                    (state.commandExtension.z - 0.35f),
                0.0f,
                0.50f
            )
            : 0.0f;
        const float avianNonWingCommand =
            avianLiftoffTailCarrier != 0.0f
            ? clamp(
                avianLiftoffTailCarrier + 0.25f * filtered,
                -1.0f,
                1.0f
            )
            : filtered * avianGroundResidualScale + avianGroundGaitCarrier;
        const float studentTarget =
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? defaultQ[binding.indices.z] +
                binding.parameters.x *
                    // The clock is robot-owned, while each policy output is
                    // a bilateral stroke-amplitude residual. Resolved
                    // aerodynamic loads, not this kinematic carrier, decide
                    // the resulting height and attitude.
                    // The compiled flapping binding supplies its own
                    // zero-action trim and bounded residual span, so initial
                    // policy exploration begins in the viable wingbeat band.
                    avianWingAmplitude *
                    sin(state.commandAndPhase.w)
            : defaultQ[binding.indices.z] +
                binding.parameters.x *
                    avianNonWingCommand *
                    (avianStandingCurriculum ? 0.0f : 1.0f);
        float targetCandidate = studentTarget;
        if (interactionReference) {
            const float frameReference = interactionJointTargets[
                referenceFrame * program.interaction.y + action
            ];
            const float nextReference = interactionJointTargets[
                nextReferenceFrame * program.interaction.y + action
            ];
            const float reference = mix(
                frameReference,
                nextReference,
                referenceBlend
            );
            const float referenceVelocity =
                (nextReference - frameReference) *
                program.interactionTiming.x *
                interactionPlaybackRate(program, state);
            // MetalWorld's implicit drive evaluates position at q + h*v and
            // damps v toward zero. Lead the position target by
            // (h + kd/kp)*v_ref so the same physical drive instead tracks
            // both ARDY's q_ref and v_ref. Gravity, contact, effort limits,
            // and the articulated solve remain authoritative.
            const float velocityLeadSeconds =
                dispatch.timing.y +
                (binding.drive.x > 0.0f
                    ? binding.drive.y / binding.drive.x
                    : 0.0f);
            targetCandidate =
                reference +
                velocityLeadSeconds * referenceVelocity +
                (interactionPhysicsGated(program)
                    ? 1.0f
                    : program.interactionTiming.z) *
                    (studentTarget - defaultQ[binding.indices.z]);
        }
        const float target = clamp(
            targetCandidate,
            binding.parameters.y,
            binding.parameters.z
        );
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            binding.indices.w
        ] = target;
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u) {
            // With nonzero student authority the policy owns a residual on
            // top of ARDY's motion, so the neutral teacher residual is zero.
            // At zero authority this is pure teacher collection for an
            // autonomous student and the absolute interaction action remains
            // the correct distillation label.
            teacherActions[actionBase + action] =
                program.interactionTiming.z > 0.0f
                ? 0.0f
                : clamp(
                    (target - defaultQ[binding.indices.z]) /
                        binding.parameters.x,
                    -1.0f,
                    1.0f
                );
        }
    }
}

inline float actuatorEffortEnvelope(
    device const MRDofPropertiesGPU& dof,
    device const MRActuatorProfileGPU& actuator,
    const float velocity
) {
    const float speedFraction = clamp(
        abs(velocity) /
            max(actuator.motorAndSpeed.z, 1.175494351e-38f),
        0.0f,
        1.0f
    );
    const float dofLimit =
        (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
            dof.limits.w > 0.0f
        ? dof.limits.w
        : 3.402823466e+38f;
    return min(
        dofLimit,
        actuator.transmissionAndEnvelope.z *
            actuator.motorAndSpeed.w *
            (1.0f - speedFraction)
    );
}

inline float actuatorDryFriction(
    device const MRDofPropertiesGPU& dof,
    const float velocity,
    const float requested
) {
    const float friction = dof.drive.w;
    if (!(friction > 0.0f)) {
        return 0.0f;
    }
    return abs(velocity) > 1.0e-4f
        ? -copysign(friction, velocity)
        : -clamp(requested, -friction, friction);
}

// Executes every effort-producing RobotPack actuator from the live accepted
// microstep state. Position/gripper targets remain in the implicit-drive
// trajectory; rotor mixers remain in their robot-authored program. This pass
// adds no integration or kinematic override: it only writes generalized
// effort and world-frame body wrench consumed by the ordinary ABA solve.
kernel void mr_locomotion_task_apply_native_actuators(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const float* vState [[buffer(6)]],
    device const float* actionHistory [[buffer(7)]],
    device const MRDofPropertiesGPU* dofs [[buffer(8)]],
    device const MRActuatorProfileGPU* actuatorProfiles [[buffer(9)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(10)]],
    device float* workingEffort [[buffer(11)]],
    device MRABABodyWrenchGPU* bodyWrenches [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        pass.physicsSubstep >= worldDispatch.physicsSubsteps ||
        dispatch.counts.x != worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.layout.w < 2u) {
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(arena, program.offsets0.x);
    device const MRTaskActuatorTermGPU* terms =
        taskTable<MRTaskActuatorTermGPU>(arena, program.actuatorTerms.x);
    const uint bodyCount = dispatch.sampling.z;
    if ((worldDispatch.flags & MR_METAL_WORLD_HAS_BODY_WRENCHES) != 0u) {
        for (uint body = 0u; body < bodyCount; ++body) {
            bodyWrenches[environment * bodyCount + body] = {};
        }
    }
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint filterSlot = program.layout.w - 1u;
    const uint historyBase =
        environment * program.layout.w * program.counts0.x +
        filterSlot * program.counts0.x;
    for (uint action = 0u; action < program.counts0.x; ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        const uint kind = binding.actuator.x;
        if (kind == MR_TASK_ACTUATOR_JOINT_POSITION ||
            kind == MR_TASK_ACTUATOR_GRIPPER_POSITION ||
            kind == MR_TASK_ACTUATOR_FLAPPING_POSITION ||
            kind == MR_TASK_ACTUATOR_ROTOR_MIXER) {
            continue;
        }
        const float filtered = actionHistory[historyBase + action];
        if (kind == MR_TASK_ACTUATOR_JOINT_VELOCITY ||
            kind == MR_TASK_ACTUATOR_JOINT_EFFORT) {
            const uint dofIndex = binding.indices.y;
            const uint velocityIndex = binding.indices.w;
            if (dofIndex >= dispatch.counts.w ||
                velocityIndex >= dispatch.counts.w) {
                continue;
            }
            device const MRDofPropertiesGPU& dof = dofs[dofIndex];
            const float velocity = vState[vBase + velocityIndex];
            const float target = clamp(
                binding.parameters.x * filtered,
                binding.parameters.y,
                binding.parameters.z
            );
            float effort = kind == MR_TASK_ACTUATOR_JOINT_VELOCITY
                ? binding.drive.y * (target - velocity)
                : target;
            effort += actuatorDryFriction(dof, velocity, effort);
            const float envelope = actuatorEffortEnvelope(
                dof, actuatorProfiles[dofIndex], velocity);
            workingEffort[vBase + velocityIndex] =
                clamp(effort, -envelope, envelope);
            continue;
        }
        if (kind == MR_TASK_ACTUATOR_TENDON_POSITION) {
            const uint first = binding.indices.y;
            const uint count = binding.indices.z;
            if (first > program.actuatorTerms.y ||
                count > program.actuatorTerms.y - first) {
                continue;
            }
            float length = 0.0f;
            float rate = 0.0f;
            for (uint termIndex = 0u; termIndex < count; ++termIndex) {
                const MRTaskActuatorTermGPU term = terms[first + termIndex];
                length += term.coefficient.x *
                    qState[qBase + term.indices.y];
                rate += term.coefficient.x *
                    vState[vBase + term.indices.z];
            }
            const float target =
                binding.drive.w + binding.parameters.x * filtered;
            const float tension = clamp(
                binding.drive.x * (target - length) -
                    binding.drive.y * rate,
                -binding.drive.z,
                binding.drive.z
            );
            for (uint termIndex = 0u; termIndex < count; ++termIndex) {
                const MRTaskActuatorTermGPU term = terms[first + termIndex];
                const uint dofIndex = term.indices.x;
                const uint velocityIndex = term.indices.z;
                const float velocity = vState[vBase + velocityIndex];
                device const MRDofPropertiesGPU& dof = dofs[dofIndex];
                float effort = term.coefficient.x * tension;
                effort += actuatorDryFriction(dof, velocity, effort);
                effort += workingEffort[vBase + velocityIndex];
                const float envelope = actuatorEffortEnvelope(
                    dof, actuatorProfiles[dofIndex], velocity);
                workingEffort[vBase + velocityIndex] =
                    clamp(effort, -envelope, envelope);
            }
            continue;
        }
        if (kind == MR_TASK_ACTUATOR_BODY_WRENCH &&
            (worldDispatch.flags & MR_METAL_WORLD_HAS_BODY_WRENCHES) != 0u &&
            binding.actuator.y < bodyCount && binding.actuator.z < 6u) {
            const uint bodyIndex = binding.actuator.y;
            const uint wrenchIndex = environment * bodyCount + bodyIndex;
            const MRArticulatedBodyPoseGPU pose =
                bodyPoses[wrenchIndex];
            const uint component = binding.actuator.z;
            float3 local = float3(0.0f);
            local[component % 3u] = binding.parameters.x * filtered;
            const float3 world = rotate(pose.orientation, local);
            MRABABodyWrenchGPU wrench = bodyWrenches[wrenchIndex];
            if (component < 3u) {
                wrench.force.xyz += world;
            } else {
                wrench.torque.xyz += world;
            }
            bodyWrenches[wrenchIndex] = wrench;
        }
    }
}

kernel void mr_locomotion_task_measure_effort(
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
        const MRTaskActionBindingGPU binding = actions[action];
        if (binding.indices.w == MR_INVALID_INDEX) {
            continue;
        }
        const uint velocityIndex = binding.indices.w;
        mechanicalPower += abs(
            workingEffort[vBase + velocityIndex] *
            vState[vBase + velocityIndex]
        );
    }
    MRTaskStateGPU state = taskStates[environment];
    state.airReturnTracking.x = mechanicalPower;
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_complete(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRTaskEvidenceStateGPU* evidenceState
        [[buffer(3)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(4)]],
    constant MRMetalWorldPassGPU& pass [[buffer(5)]],
    device const float* qState [[buffer(6)]],
    device const float* vState [[buffer(7)]],
    device const MRBodyStateGPU* bodyStates [[buffer(8)]],
    device const MRContactConstraintGPU* contacts [[buffer(9)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(10)]],
    device const MRMetalWorldStatusGPU* worldStatuses
        [[buffer(11)]],
    device const MRBodyStateGPU* sceneState [[buffer(12)]],
    device const MRDofPropertiesGPU* dofs [[buffer(13)]],
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
    device const MRTaskContactGroupGPU* contactGroups =
        taskTable<MRTaskContactGroupGPU>(
            arena,
            program.offsets0.w
        );
    device const uint* contactMembers =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* contactMemberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const MRTaskIndexGroupGPU* jointGroups =
        taskTable<MRTaskIndexGroupGPU>(
            arena,
            program.offsets1.y
        );
    device const uint* jointMembers =
        taskTable<uint>(arena, program.offsets1.z);
    device const MRTaskRewardOperatorGPU* rewards =
        taskTable<MRTaskRewardOperatorGPU>(
            arena,
            program.offsets1.w
        );
    device const MRTaskTerminationOperatorGPU*
        terminations =
            taskTable<MRTaskTerminationOperatorGPU>(
                arena,
                program.offsets2.x
            );
    device const float4* terrainSamples =
        taskTable<float4>(arena, program.offsets2.w);
    device const float4* commandCurriculum =
        taskTable<float4>(arena, program.offsets3.y);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(
            arena,
            program.offsets3.z
        );
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    device const float* interactionRootTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
    device const MRTaskInteractionContactGPU*
        interactionContacts =
            taskTable<MRTaskInteractionContactGPU>(
                arena,
                program.interactionOffsets0.z
            );
    device const MRTaskInteractionSampleGPU*
        interactionSamples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
    device const float* interactionContactTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets1.x
        );
    device const float* interactionContactTolerances =
        taskTable<float>(
            arena,
            program.interactionOffsets1.y
        );
    device const uint* outcomeRewardOperations =
        taskTable<uint>(arena, program.interactionOffsets1.z);

    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint bodyBase =
        environment * dispatch.strides.y;
    const uint sceneBase =
        environment * dispatch.strides.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device const float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
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
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );
    const uint curriculum = min(
        state.episode.z,
        program.schedule.z - 1u
    );
    const MRMetalWorldStatusGPU worldStatus =
        worldStatuses[environment];
    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    const bool physicsError =
        worldStatus.code != MR_STEP_SUCCESS ||
        contactStatus.code != MR_STEP_SUCCESS;
    const uint activeContacts = physicsError
        ? 0u
        : min(
              contactStatus.activeContacts,
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
            for (uint cell = 0u;
                 cell < group.supportPatch.z;
                 ++cell) {
                compactContact[
                    compactBase + group.supportPatch.w + cell
                ] = 0.0f;
            }
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
    for (uint contact = 0u;
         contact < activeContacts;
         ++contact) {
        const MRContactConstraintGPU constraint =
            contacts[contactBase + contact];
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
            const float3 contactLocal = rotateInverse(
                referenceBody.orientation,
                constraint.pointAndSeparation.xyz -
                    referencePosition
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
                    impulse * contactLocal.x;
                compactContact[metric + 5u] +=
                    impulse * contactLocal.y;
                if (group.supportPatch.z != 0u &&
                    contactLocal.x >=
                        group.supportPatchBounds.x &&
                    contactLocal.x <=
                        group.supportPatchBounds.z &&
                    contactLocal.y >=
                        group.supportPatchBounds.y &&
                    contactLocal.y <=
                        group.supportPatchBounds.w) {
                    const float2 normalized = clamp(
                        (
                            contactLocal.xy -
                            group.supportPatchBounds.xy
                        ) /
                        (
                            group.supportPatchBounds.zw -
                            group.supportPatchBounds.xy
                        ),
                        float2(0.0f),
                        float2(0.999999f)
                    );
                    const uint column = min(
                        uint(normalized.x *
                            float(group.supportPatch.x)),
                        group.supportPatch.x - 1u
                    );
                    const uint row = min(
                        uint(normalized.y *
                            float(group.supportPatch.y)),
                        group.supportPatch.y - 1u
                    );
                    compactContact[
                        compactBase + group.supportPatch.w +
                        row * group.supportPatch.x + column
                    ] += impulse;
                }
            }
            if ((group.members.z &
                 MR_TASK_CONTACT_FORBIDDEN) != 0u) {
                compactContact[metric] = max(
                    compactContact[metric],
                    float(
                        impulse / dispatch.timing.y >
                        program.dynamics.y
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
            force > program.dynamics.y;
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
              ) / impulse
            : float2(0.0f);
        compactContact[metric + 0u] = force;
        compactContact[metric + 1u] = slip;
        compactContact[metric + 2u] = airTime;
        compactContact[metric + 3u] = clearance;
        compactContact[metric + 4u] = cop.x;
        compactContact[metric + 5u] = cop.y;
        for (uint cell = 0u;
             cell < group.supportPatch.z;
             ++cell) {
            compactContact[
                compactBase + group.supportPatch.w + cell
            ] /= dispatch.timing.y;
        }
    }

    device const float* q = qState + qBase;
    device const float* v = vState + vBase;
    const float4 orientation = rootOrientation(program, q);
    const float3 rootLinearVelocity =
        rootWorldLinearVelocity(program, q, v);
    const float3 baseLinear = rotateInverse(
        orientation,
        rootLinearVelocity
    );
    const float3 baseAngular = rotateInverse(
        orientation,
        rootWorldAngularVelocity(program, v)
    );
    const float3 gravity = normalizedOr(
        rotateInverse(
            orientation,
            float3(0.0f, 0.0f, -1.0f)
        ),
        float3(0.0f, 0.0f, -1.0f)
    );
    const float height = rootHeight(
        program,
        q
    );
    const float tilt = atan2(
        length(gravity.xy),
        max(-gravity.z, 1.0e-6f)
    );
    float2 yawBasis = float2(
        1.0f -
            2.0f *
                (
                    orientation.y * orientation.y +
                    orientation.z * orientation.z
                ),
        2.0f *
            (
                orientation.w * orientation.z +
                orientation.x * orientation.y
            )
    );
    yawBasis *= rsqrt(
        max(dot(yawBasis, yawBasis), 1.0e-12f)
    );
    bool figureEightConfigured = false;
    bool figureEightActive = false;
    float figureEightPathErrorSquared = 0.0f;
    const bool avianGroundCurriculum =
        (program.schedule.w & MR_TASK_PROGRAM_AVIAN_GROUND_CURRICULUM) != 0u;
    const uint avianCurriculumBand = state.episode.z;
    // The first two avian bands are supported standing and walking, not
    // flight.  The 0.1873 m target is the measured mean root height from the
    // held default-pose standing rollout on the imported hybrid; it prevents
    // an airborne height objective from competing with legged locomotion.
    constexpr float avianGroundRootHeightTarget = 0.1873f;
    const bool avianSupportedGroundStage =
        avianGroundCurriculum && avianCurriculumBand < 2u;
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation = rewards[rewardIndex];
        if (operation.source.x !=
                MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING) {
            continue;
        }
        figureEightConfigured = true;
        const float extentX = operation.parameters.y;
        const float extentY = operation.parameters.z;
        const float cycleSeconds = operation.parameters.w;
        const float takeoffSeconds = operation.auxiliary.x;
        const float episodeSeconds =
            float(state.episode.x + 1u) * dispatch.timing.x;
        if (avianGroundCurriculum && avianCurriculumBand == 0u) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            state.commandAndPhase.xyz = float3(0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 1u) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            // A modest target leaves forward progress, bilateral support,
            // and uprightness as the learning signal while wings remain
            // folded in the ground curriculum.
            state.commandAndPhase.xyz = float3(0.22f, 0.0f, 0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 2u) {
            const float verticalRate = clamp(
                (height - state.airReturnTracking.y) / dispatch.timing.x,
                -6.0f,
                6.0f
            );
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                verticalRate
            );
            // Learn the vertical transition and a modest forward airspeed
            // before asking the same policy to solve a curved flight path.
            // The 0.35 m/s target matches the authored launch-tracking scale;
            // the figure-eight command is reserved for the later flight band.
            state.commandAndPhase.xyz = float3(0.35f, 0.0f, 0.0f);
            break;
        }
        if (episodeSeconds <= takeoffSeconds) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            state.commandAndPhase.xyz = float3(0.9f, 0.0f, 0.0f);
            break;
        }
        figureEightActive = true;
        const float omega = kTwoPi / max(cycleSeconds, 1.0e-4f);
        const float theta = fmod(
            state.commandExtension.z + omega * dispatch.timing.x,
            kTwoPi
        );
        // Rotate the Gerono tangent so the first post-takeoff command is
        // forward in world X instead of diagonally across the crossing.
        const float initialAngle = atan2(2.0f * extentY, extentX);
        const float c = cos(-initialAngle);
        const float s = sin(-initialAngle);
        const float2 rawVelocity = float2(
            extentX * cos(theta) * omega,
            2.0f * extentY * cos(2.0f * theta) * omega
        );
        const float2 rawAcceleration = float2(
            -extentX * sin(theta) * omega * omega,
            -4.0f * extentY * sin(2.0f * theta) * omega * omega
        );
        const float2 targetVelocity = float2(
            c * rawVelocity.x - s * rawVelocity.y,
            s * rawVelocity.x + c * rawVelocity.y
        );
        const float2 targetAcceleration = float2(
            c * rawAcceleration.x - s * rawAcceleration.y,
            s * rawAcceleration.x + c * rawAcceleration.y
        );
        state.commandExtension.xy += targetVelocity * dispatch.timing.x;
        state.commandExtension.zw = float2(theta, 1.0f);
        const float2 pathError =
            state.commandExtension.xy - rootWorldPosition(program, q).xy;
        const float2 correctedVelocity =
            targetVelocity + clamp(pathError * 0.70f, -2.0f, 2.0f);
        state.commandAndPhase.xy = float2(
            yawBasis.x * correctedVelocity.x + yawBasis.y * correctedVelocity.y,
            -yawBasis.y * correctedVelocity.x + yawBasis.x * correctedVelocity.y
        );
        state.commandAndPhase.z = clamp(
            (targetVelocity.x * targetAcceleration.y -
             targetVelocity.y * targetAcceleration.x) /
                max(dot(targetVelocity, targetVelocity), 1.0e-4f),
            -2.0f,
            2.0f
        );
        figureEightPathErrorSquared = dot(pathError, pathError);
        break;
    }
    const float2 yawFrameLinear = float2(
        yawBasis.x *
            rootLinearVelocity.x +
            yawBasis.y *
                rootLinearVelocity.y,
        -yawBasis.y *
            rootLinearVelocity.x +
            yawBasis.x *
                rootLinearVelocity.y
    );
    if (avianGroundCurriculum && avianCurriculumBand == 2u) {
        // The action kernel consumes this accepted-state value on the next
        // control step.  Keeping the feedback inside task state avoids a
        // host readback loop and preserves transactional replay semantics.
        state.commandExtension.z = yawFrameLinear.x;
    }
    const float2 trackingDelta =
        yawFrameLinear - state.commandAndPhase.xy;
    const float trackingError =
        dot(trackingDelta, trackingDelta);
    const float yawDelta =
        baseAngular.z - state.commandAndPhase.z;
    const float yawError = yawDelta * yawDelta;

    float velocitySquared = 0.0f;
    float accelerationSquared = 0.0f;
    float actionRateSquared = 0.0f;
    float limitViolationSquared = 0.0f;
    const float mechanicalPower =
        state.airReturnTracking.x;
    const uint rawLastSlot = program.layout.w - 2u;
    const uint previousRawSlot = rawLastSlot - 1u;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        const MRTaskActionBindingGPU binding =
            actions[action];
        const float currentAction = actionHistory[
            delayBase +
            rawLastSlot * program.counts0.x +
            action
        ];
        const float previousAction = actionHistory[
            delayBase +
            previousRawSlot * program.counts0.x +
            action
        ];
        const float actionDelta = currentAction - previousAction;
        actionRateSquared += actionDelta * actionDelta;
        if (binding.indices.y == MR_INVALID_INDEX ||
            binding.indices.z == MR_INVALID_INDEX ||
            binding.indices.w == MR_INVALID_INDEX) {
            continue;
        }
        const MRDofPropertiesGPU dof =
            dofs[binding.indices.y];
        const float position =
            qState[qBase + binding.indices.z];
        const float velocity =
            vState[vBase + binding.indices.w];
        const float acceleration =
            (
                velocity -
                previousJointVelocity[
                    previousVelocityBase + action
                ]
            ) /
            dispatch.timing.x;
        const float lower =
            max(dof.limits.x - position, 0.0f);
        const float upper =
            max(position - dof.limits.y, 0.0f);
        velocitySquared += velocity * velocity;
        accelerationSquared +=
            acceleration * acceleration;
        limitViolationSquared +=
            lower * lower + upper * upper;
    }

    const float commandMagnitude = length(
        state.commandAndPhase.xyz
    );
    const bool moving = commandMagnitude > 0.1f;
    float phase =
        state.commandAndPhase.w +
        kTwoPi * dispatch.timing.x /
            program.locomotion.y;
    phase = fmod(phase, kTwoPi);

    bool recoveryConfigured = false;
    float recoveryActivationTilt = 0.0f;
    float recoveryStableTilt = 0.0f;
    float recoveryStableDuration = 0.0f;
    uint recoveryContactGroup = MR_INVALID_INDEX;
    bool standingConfigured = false;
    float standingHeight = program.locomotion.x;
    float standingCosine = 0.8f;
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        if (!recoveryConfigured &&
            (operation.source.x ==
                MR_TASK_REWARD_RECOVERY_TILT_PROGRESS ||
            operation.source.x ==
                MR_TASK_REWARD_RECOVERY_COMPLETION)) {
            recoveryConfigured = true;
            recoveryActivationTilt = operation.parameters.y;
            recoveryStableTilt = operation.parameters.z;
            recoveryStableDuration = operation.parameters.w;
            recoveryContactGroup = operation.source.y;
        }
        if (operation.source.x ==
                MR_TASK_REWARD_STANDING_COMPLETION) {
            standingConfigured = true;
            standingHeight = operation.parameters.y;
            standingCosine = operation.parameters.z;
        }
    }
    bool recoveryTouch = false;
    if (recoveryContactGroup != MR_INVALID_INDEX) {
        const MRTaskContactGroupGPU group =
            contactGroups[recoveryContactGroup];
        const uint wrench = compactBase + group.reference.y;
        recoveryTouch = length(float3(
            compactContact[wrench + 0u],
            compactContact[wrench + 1u],
            compactContact[wrench + 2u]
        )) > program.dynamics.y;
    }
    const bool recoveryActiveBefore = state.recovery.w > 0.5f;
    const bool recoveryTouchBefore = state.recoveryStats.z != 0u;
    const bool eventSequenceAvailable =
        impactSequenceEnabled(state) &&
        program.counts3.x > 0u &&
        curriculum >= impactEvents[0].binding.z;
    const bool recoveryActivated =
        recoveryConfigured &&
        !recoveryActiveBefore &&
        (eventSequenceAvailable
            ? impactScene(state) != 0u &&
                recoveryTouch && !recoveryTouchBefore
            : (recoveryTouch && !recoveryTouchBefore) ||
                (state.recovery.x < recoveryActivationTilt &&
                 tilt >= recoveryActivationTilt));
    const bool recoveryActive =
        recoveryActiveBefore || recoveryActivated;
    const float recoveryPeakTilt = recoveryActive
        ? max(
              recoveryActiveBefore ? state.recovery.y : tilt,
              tilt
          )
        : 0.0f;
    const float recoveryStableTime = recoveryActive
        ? (tilt <= recoveryStableTilt
            ? (recoveryActiveBefore ? state.recovery.z : 0.0f) +
                dispatch.timing.x
            : 0.0f)
        : 0.0f;
    const bool recoveryCompleted =
        recoveryActive &&
        recoveryStableTime >= recoveryStableDuration;
    const uint recoveryEventCount =
        state.recoveryStats.x + uint(recoveryActivated);
    const uint recoveryCompletionCount =
        state.recoveryStats.y + uint(recoveryCompleted);
    const uint activeImpactScene = impactScene(state);
    const uint activeImpactOrder = impactOrder(state);
    uint activeImpactEvent = MR_INVALID_INDEX;
    if (activeImpactScene != 0u &&
        activeImpactOrder < program.counts3.x) {
        activeImpactEvent =
            (activeImpactOrder + impactOffset(state)) %
            program.counts3.x;
    }
    const bool impactContactLatchedBefore =
        (state.recoveryStats.w & kImpactContactLatched) != 0u;
    const bool impactContactPublishedBefore =
        (state.recoveryStats.w & kImpactContactPublished) != 0u;
    bool projectileContact = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        activeImpactEvent != MR_INVALID_INDEX) {
        const uint projectileBody =
            impactEvents[activeImpactEvent].binding.w;
        const uint articulationBodyBegin = program.articulation.x;
        const uint articulationBodyEnd =
            articulationBodyBegin + program.articulation.y;
        for (uint contact = 0u;
             contact < activeContacts && !projectileContact;
             ++contact) {
            const MRContactConstraintGPU constraint =
                contacts[contactBase + contact];
            const bool projectileA =
                constraint.bodyA == projectileBody;
            const bool projectileB =
                constraint.bodyB == projectileBody;
            if (projectileA == projectileB) {
                continue;
            }
            const uint other = projectileA
                ? constraint.bodyB
                : constraint.bodyA;
            projectileContact =
                other >= articulationBodyBegin &&
                other < articulationBodyEnd;
        }
    }
    const bool impactContactLatched =
        impactContactLatchedBefore || projectileContact;
    const bool newProjectileContact =
        impactContactLatched && !impactContactPublishedBefore;
    uint impactTransitionFlags =
        eventSequenceAvailable
        ? MR_TASK_IMPACT_SEQUENCE_ENABLED
        : 0u;
    impactTransitionFlags |=
        activeImpactScene != 0u &&
        recoveryActivated
        ? MR_TASK_IMPACT_TOUCH
        : 0u;
    impactTransitionFlags |= newProjectileContact
        ? MR_TASK_IMPACT_CONTACT
        : 0u;
    bool impactWindowElapsed = false;
    bool missedImpact = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        impactOrder(state) < program.counts3.x) {
        const MRTaskImpactEventGPU event =
            impactEvents[activeImpactEvent];
        impactWindowElapsed =
            float(state.status.w) * dispatch.timing.x >=
            event.gate.z;
        missedImpact =
            !recoveryActive &&
            impactWindowElapsed &&
            !impactContactLatched;
    }
    if (recoveryCompleted) {
        impactTransitionFlags |= MR_TASK_IMPACT_RECOVERED;
    }
    if (missedImpact) {
        impactTransitionFlags |= MR_TASK_IMPACT_MISSED;
    }
    bool projectileThreat = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        activeImpactEvent != MR_INVALID_INDEX) {
        const MRBodyStateGPU projectile = sceneState[
            sceneBase + activeImpactScene - 1u
        ];
        const float3 rootPosition = float3(
            qState[qBase + program.root.z],
            qState[qBase + program.root.z + 1u],
            qState[qBase + program.root.z + 2u]
        );
        const float3 relativePosition =
            projectile.position.xyz - rootPosition;
        const float distance = max(
            length(relativePosition),
            1.0e-4f
        );
        const float3 relativeVelocity =
            projectile.linearVelocityAndInverseMass.xyz -
            rootLinearVelocity;
        const float closingSpeed =
            -dot(relativePosition, relativeVelocity) / distance;
        const float timeToClosestApproach =
            distance / max(closingSpeed, 1.0e-4f);
        projectileThreat =
            closingSpeed > 0.25f &&
            timeToClosestApproach <=
                impactEvents[activeImpactEvent].gate.z &&
            projectile.position.z > 0.10f;
    }
    float reward = 0.0f;
    float4 rewardBreakdown0 = float4(0.0f);
    float4 rewardBreakdown1 = float4(0.0f);
    float interactionTrackingSum = 0.0f;
    float interactionTrackingWeight = 0.0f;
    bool standingCompleted = false;
    bool restoredCompleted = false;
    uint recoveryOutcomeFlags = 0u;
    float4 outcomeChannels0 = float4(0.0f);
    float4 outcomeChannels1 = float4(0.0f);
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        float value = 0.0f;
        float interactionMetric = 0.0f;
        float interactionMetricWeight = 0.0f;
        switch (operation.source.x) {
        case MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING:
            value = figureEightActive
                ? exp(
                    -figureEightPathErrorSquared /
                    max(
                        0.0625f *
                            min(operation.parameters.y,
                                operation.parameters.z) *
                            min(operation.parameters.y,
                                operation.parameters.z),
                        0.25f
                    )
                )
                : 0.0f;
            break;
        case MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING:
            value = exp(
                -trackingError /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        case MR_TASK_REWARD_YAW_VELOCITY_TRACKING:
            value = exp(
                -yawError /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        case MR_TASK_REWARD_CONSTANT:
            value = 1.0f;
            break;
        case MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED:
            value = baseLinear.z * baseLinear.z;
            break;
        case MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED:
            value = dot(baseAngular.xy, baseAngular.xy);
            break;
        case MR_TASK_REWARD_TILT_SQUARED:
            // The isolated liftoff band is not yet a banking task.  Keep a
            // modest launch-pitch allowance, then penalize excess attitude so
            // height reward cannot be collected by an uncontrolled ballistic
            // climb.  The threshold comes from the held-out zero-policy mean
            // (about 0.29 rad) with margin for a physical push-off; later
            // figure-eight flight retains unconstrained learned banking.
            if (avianGroundCurriculum && avianCurriculumBand == 2u) {
                const float excessTilt = max(tilt - 0.35f, 0.0f);
                // The task's existing -0.50 reward weight yields a -3.0
                // quadratic coefficient beyond the launch envelope.
                value = 6.0f * excessTilt * excessTilt;
            } else {
                value = avianSupportedGroundStage ? tilt * tilt : 0.0f;
            }
            break;
        case MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED:
            value = dot(gravity.xy, gravity.xy);
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED: {
            const float error =
                height - (avianSupportedGroundStage
                    ? avianGroundRootHeightTarget
                    : program.locomotion.x);
            value = error * error;
            break;
        }
        case MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED:
            if (avianSupportedGroundStage) {
                const float error =
                    height - avianGroundRootHeightTarget;
                value = exp(-error * error / 0.0025f);
            } else {
                value = clamp(
                    height / max(program.locomotion.x, 1.0e-6f),
                    0.0f,
                    1.0f
                );
            }
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS:
            // Signed potential progress: rising earns exactly what settling
            // back down loses, preventing repeated bounce from manufacturing
            // height reward without a higher final physical state.
            value = avianSupportedGroundStage
                ? 0.0f
                : clamp(
                      (height - state.airReturnTracking.y) /
                          dispatch.timing.x,
                      -2.0f,
                      2.0f
                  );
            break;
        case MR_TASK_REWARD_UPRIGHTNESS:
            // Horizontal is not half-standing. Reward only the component of
            // body-up aligned with world-up; squaring retains a smooth slope
            // while concentrating value near an actually upright posture.
            value = pow(clamp(-gravity.z, 0.0f, 1.0f), 2.0f);
            break;
        case MR_TASK_REWARD_SUPPORT_CONTACT_COUNT: {
            float supportCount = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                supportCount += float(
                    compactContact[
                        compactBase + group.members.w
                    ] > program.dynamics.y
                );
                supportTotal += 1.0f;
            }
            value = supportCount / max(supportTotal, 1.0f);
            break;
        }
        case MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const MRBodyStateGPU body = bodyStates[
                bodyBase + group.reference.x
            ];
            const float bodyHeight =
                body.position.z +
                rotate(
                    body.orientation,
                    group.kinematicReference.xyz
                ).z;
            value = exp(
                clamp(
                    bodyHeight,
                    0.0f,
                    operation.parameters.y
                )
            ) - 1.0f;
            break;
        }
        case MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL: {
            float supportHeight = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const MRBodyStateGPU body = bodyStates[
                    bodyBase + group.reference.x
                ];
                supportHeight += max(
                    body.position.z +
                        rotate(
                            body.orientation,
                            group.kinematicReference.xyz
                        ).z,
                    0.0f
                );
                supportTotal += 1.0f;
            }
            value = exp(
                -operation.parameters.y *
                supportHeight / max(supportTotal, 1.0f)
            );
            break;
        }
        case MR_TASK_REWARD_BODY_UP_EXPONENTIAL:
            value = exp(-gravity.z);
            break;
        case MR_TASK_REWARD_STANDING_COMPLETION: {
            bool anySupported = false;
            uint supportTotal = 0u;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                anySupported = anySupported || compactContact[
                    compactBase + group.members.w
                ] > program.dynamics.y;
                ++supportTotal;
            }
            value = float(
                supportTotal > 0u &&
                anySupported &&
                height >= operation.parameters.y &&
                gravity.z <= -operation.parameters.z
            );
            standingCompleted = standingCompleted || value > 0.5f;
            break;
        }
        case MR_TASK_REWARD_RESTORATION: {
            bool anySupported = false;
            float supported = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                const bool contact = compactContact[
                    compactBase + group.members.w
                ] > program.dynamics.y;
                anySupported = anySupported || contact;
                supported += float(contact);
                supportTotal += 1.0f;
            }
            float jointErrorSquared = 0.0f;
            for (uint action = 0u;
                 action < program.counts0.x;
                 ++action) {
                const MRTaskActionBindingGPU binding = actions[action];
                const float error =
                    q[binding.indices.z] - defaultQ[binding.indices.z];
                jointErrorSquared += error * error;
            }
            const float jointRms = sqrt(
                jointErrorSquared /
                max(float(program.counts0.x), 1.0f)
            );
            const float3 position = rootWorldPosition(program, q);
            const float3 targetPosition =
                rootWorldPosition(program, defaultQ);
            const float rootError = length(
                position.xy - targetPosition.xy
            );
            const float4 targetOrientation =
                rootOrientation(program, defaultQ);
            const float orientationCosine = clamp(
                abs(dot(orientation, targetOrientation)),
                0.0f,
                1.0f
            );
            const float generalizedSpeedRms = sqrt(
                (
                    velocitySquared +
                    dot(baseLinear, baseLinear) +
                    dot(baseAngular, baseAngular)
                ) /
                max(float(program.counts0.x + 6u), 1.0f)
            );
            const float heightScore = smoothstep(
                0.15f,
                max(standingHeight, 0.1501f),
                height
            );
            const float uprightScore = smoothstep(
                0.0f,
                max(standingCosine, 1.0e-4f),
                -gravity.z
            );
            const float supportScore = supportTotal > 0.0f
                ? supported / supportTotal
                : 0.0f;
            const float postureScore = exp(
                -jointErrorSquared /
                max(
                    float(program.counts0.x) *
                        operation.parameters.y *
                        operation.parameters.y,
                    1.0e-8f
                )
            );
            const float positionScore = exp(
                -(rootError * rootError) /
                max(
                    operation.parameters.z * operation.parameters.z,
                    1.0e-8f
                )
            );
            const float orientationScore = smoothstep(
                0.0f,
                operation.parameters.w,
                orientationCosine
            );
            const float stillnessScore = exp(
                -(generalizedSpeedRms * generalizedSpeedRms) /
                max(
                    operation.auxiliary.x * operation.auxiliary.x,
                    1.0e-8f
                )
            );
            const float standingQuality =
                heightScore * uprightScore * supportScore;
            value =
                0.20f * heightScore +
                0.20f * uprightScore +
                0.20f * supportScore +
                standingQuality *
                    (
                        0.15f * postureScore +
                        0.10f * positionScore +
                        0.10f * orientationScore +
                        0.05f * stillnessScore
                    );
            restoredCompleted = restoredCompleted ||
                standingConfigured &&
                supportTotal > 0.0f &&
                anySupported &&
                height >= standingHeight &&
                gravity.z <= -standingCosine &&
                jointRms <= operation.parameters.y &&
                rootError <= operation.parameters.z &&
                orientationCosine >= operation.parameters.w &&
                generalizedSpeedRms <= operation.auxiliary.x;
            break;
        }
        case MR_TASK_REWARD_WHOLE_BODY_RECOVERY: {
            const MRTaskContactGroupGPU assistGroup =
                contactGroups[operation.source.y];
            const MRTaskContactGroupGPU trunkGroup =
                contactGroups[operation.source.z];
            const uint assistWrench =
                compactBase + assistGroup.reference.y;
            const uint trunkWrench =
                compactBase + trunkGroup.reference.y;
            const bool assistContact = length(float3(
                compactContact[assistWrench + 0u],
                compactContact[assistWrench + 1u],
                compactContact[assistWrench + 2u]
            )) > program.dynamics.y;
            const bool trunkContact = length(float3(
                compactContact[trunkWrench + 0u],
                compactContact[trunkWrench + 1u],
                compactContact[trunkWrench + 2u]
            )) > program.dynamics.y;

            float supported = 0.0f;
            float supportTotal = 0.0f;
            float2 supportCenter = float2(0.0f);
            float copMarginSum = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                supportTotal += 1.0f;
                const uint metric = compactBase + group.members.w;
                const bool contact = compactContact[metric] >
                    program.dynamics.y;
                if (!contact) {
                    continue;
                }
                supported += 1.0f;
                const MRBodyStateGPU body = bodyStates[
                    bodyBase + group.reference.x
                ];
                supportCenter += (
                    body.position.xyz +
                    rotate(body.orientation, group.localReference.xyz)
                ).xy;
                const float2 cop = float2(
                    compactContact[metric + 4u],
                    compactContact[metric + 5u]
                );
                const float margin = min(
                    min(
                        cop.x - group.supportPatchBounds.x,
                        group.supportPatchBounds.z - cop.x
                    ),
                    min(
                        cop.y - group.supportPatchBounds.y,
                        group.supportPatchBounds.w - cop.y
                    )
                );
                const float halfMinimumExtent = 0.5f * min(
                    group.supportPatchBounds.z -
                        group.supportPatchBounds.x,
                    group.supportPatchBounds.w -
                        group.supportPatchBounds.y
                );
                copMarginSum += clamp(
                    margin / max(halfMinimumExtent, 1.0e-5f),
                    0.0f,
                    1.0f
                );
            }

            const bool anyFootSupport = supported > 0.0f;
            const float supportScore = supportTotal > 0.0f
                ? supported / supportTotal
                : 0.0f;
            const float copMarginScore = supported > 0.0f
                ? copMarginSum / supported
                : 0.0f;
            const float2 rootXY = rootWorldPosition(program, q).xy;
            const float supportDistance = anyFootSupport
                ? length(rootXY - supportCenter / supported)
                : operation.parameters.w;
            const float baseOverSupportScore = anyFootSupport
                ? exp(
                      -(supportDistance * supportDistance) /
                      max(
                          operation.parameters.w *
                              operation.parameters.w,
                          1.0e-8f
                      )
                  )
                : 0.0f;
            // Continuous from the floor: a crouch at 0.14 m is not placed in
            // a zero-gradient dead zone merely because standing is 0.65 m.
            const float heightScore = clamp(
                height / max(operation.parameters.y, 1.0e-4f),
                0.0f,
                1.0f
            );
            const float uprightScore = clamp(
                -gravity.z / max(operation.parameters.z, 1.0e-4f),
                0.0f,
                1.0f
            );
            const float generalizedSpeedRms = sqrt(
                (
                    velocitySquared +
                    dot(baseLinear, baseLinear) +
                    dot(baseAngular, baseAngular)
                ) /
                max(float(program.counts0.x + 6u), 1.0f)
            );
            const float stillnessScore = exp(
                -(generalizedSpeedRms * generalizedSpeedRms) /
                max(
                    operation.auxiliary.x * operation.auxiliary.x,
                    1.0e-8f
                )
            );
            const float trunkClear = float(!trunkContact);
            const float transferQuality =
                trunkClear * supportScore * baseOverSupportScore;
            const float braceQuality = float(assistContact) *
                (1.0f - transferQuality);
            const float riseQuality = transferQuality *
                sqrt(max(heightScore * uprightScore, 0.0f));
            const float quietStandQuality = riseQuality *
                copMarginScore * stillnessScore;

            // Additive physical qualities expose partial progress. Bracing
            // fades continuously only after load transfer, preventing it
            // from becoming the final local optimum.
            value =
                0.05f * braceQuality +
                0.10f * trunkClear +
                0.15f * supportScore +
                0.15f * baseOverSupportScore +
                0.10f * copMarginScore +
                0.25f * riseQuality +
                0.20f * quietStandQuality;

            recoveryOutcomeFlags |= assistContact
                ? MR_TASK_OUTCOME_RECOVERY_BRACE : 0u;
            recoveryOutcomeFlags |= !trunkContact
                ? MR_TASK_OUTCOME_TRUNK_CLEAR : 0u;
            recoveryOutcomeFlags |= anyFootSupport
                ? MR_TASK_OUTCOME_FOOT_SUPPORT : 0u;
            recoveryOutcomeFlags |=
                !trunkContact && anyFootSupport &&
                    baseOverSupportScore >= 0.50f &&
                    copMarginScore > 0.0f
                ? MR_TASK_OUTCOME_SUPPORT_TRANSFER : 0u;
            recoveryOutcomeFlags |=
                riseQuality >= 0.35f
                ? MR_TASK_OUTCOME_RECOVERY_RISE : 0u;
            recoveryOutcomeFlags |=
                height >= operation.parameters.y &&
                    -gravity.z >= operation.parameters.z &&
                    supportScore >= 0.999f &&
                    baseOverSupportScore >= 0.50f &&
                    copMarginScore > 0.0f &&
                    generalizedSpeedRms <= operation.auxiliary.x
                ? MR_TASK_OUTCOME_QUIET_STAND : 0u;
            break;
        }
        case MR_TASK_REWARD_RECOVERY_TILT_PROGRESS:
            value = recoveryActiveBefore
                ? clamp(
                      (state.recovery.x - tilt) /
                          dispatch.timing.x,
                      -2.0f,
                      2.0f
                  )
                : 0.0f;
            break;
        case MR_TASK_REWARD_RECOVERY_COMPLETION:
            // Reward weights are rates. Dividing the one-shot event by dt
            // makes its authored weight the integrated completion bonus.
            value = recoveryCompleted
                ? 1.0f / dispatch.timing.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_LINK_CLEARANCE_BARRIER: {
            const uint projectileScene = operation.source.z;
            const bool selected = !impactSequenceEnabled(state) ||
                impactScene(state) == projectileScene + 1u;
            if (!selected) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU projectile = sceneState[
                sceneBase + projectileScene
            ];
            const float3 projectileVelocity =
                projectile.linearVelocityAndInverseMass.xyz;
            const bool live = length(projectileVelocity) > 0.5f &&
                projectile.position.z > operation.parameters.z;
            if (!live) {
                value = 0.0f;
                break;
            }
            const MRTaskContactGroupGPU protectedGroup =
                contactGroups[operation.source.y];
            float mostBinding = 0.0f;
            for (uint local = 0u;
                 local < protectedGroup.members.y;
                 ++local) {
                const uint bodyIndex = contactMembers[
                    protectedGroup.members.x + local
                ];
                const float linkRadius = contactMemberRadii[
                    protectedGroup.members.x + local
                ];
                if (!(linkRadius > 0.0f)) {
                    continue;
                }
                const MRBodyStateGPU link = bodyStates[
                    bodyBase + bodyIndex
                ];
                const float3 relative =
                    projectile.position.xyz - link.position.xyz;
                const float distance = max(length(relative), 1.0e-6f);
                const float clearance =
                    distance -
                    (linkRadius + operation.parameters.z);
                const float closingRate = dot(
                    relative,
                    projectileVelocity -
                        link.linearVelocityAndInverseMass.xyz
                ) / distance;
                const float constraint = clamp(
                    closingRate +
                        operation.parameters.y * clearance,
                    -operation.parameters.w,
                    0.0f
                );
                mostBinding = min(mostBinding, constraint);
            }
            value = mostBinding;
            break;
        }
        case MR_TASK_REWARD_PROJECTILE_MISS:
            // Convert the event into a one-shot integrated bonus despite the
            // continuous-time TaskPack reward convention.
            value = missedImpact
                ? 1.0f / dispatch.timing.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_EVASION: {
            const uint projectileScene = operation.source.z;
            const bool selected = !impactSequenceEnabled(state) ||
                impactScene(state) == projectileScene + 1u;
            if (!selected) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU projectile = sceneState[
                sceneBase + projectileScene
            ];
            const bool live =
                length(projectile.linearVelocityAndInverseMass.xyz) > 0.5f;
            if (!live) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU rootBody = bodyStates[
                bodyBase + program.root.y
            ];
            const float distance = length(
                projectile.position.xyz - rootBody.position.xyz
            );
            const float horizontalSpeedSquared = dot(
                rootBody.linearVelocityAndInverseMass.xy,
                rootBody.linearVelocityAndInverseMass.xy
            );
            const float positionBlend = operation.parameters.w;
            value =
                positionBlend *
                    (1.0f - exp(-operation.parameters.y * distance)) +
                (1.0f - positionBlend) *
                    exp(
                        -operation.parameters.z *
                            horizontalSpeedSquared
                    );
            break;
        }
        case MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS:
            value = !projectileThreat
                ? exp(
                      -operation.parameters.y *
                          dot(baseLinear.xy, baseLinear.xy)
                  )
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE:
            value = !projectileThreat ? actionRateSquared : 0.0f;
            break;
        case MR_TASK_REWARD_JOINT_CBF_CORRECTION:
            value = state.threatMetadata.x != MR_INVALID_INDEX
                ? state.threatTeacher.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_JOINT_CBF_BUFFER:
            value = state.threatMetadata.x != MR_INVALID_INDEX
                ? state.threatTeacher.y
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE:
            if (state.threatMetadata.x != MR_INVALID_INDEX &&
                isfinite(state.threatGeometry.x) &&
                isfinite(state.threatGeometry.y)) {
                const float urgency = 1.0f - clamp(
                    state.threatGeometry.y /
                        max(program.threatTiming.y, 1.0e-6f),
                    0.0f,
                    1.0f
                );
                value = clamp(
                    urgency * tanh(
                        state.threatGeometry.x /
                            max(operation.parameters.y, 1.0e-4f)
                    ),
                    -1.0f,
                    1.0f
                );
            }
            break;
        case MR_TASK_REWARD_JOINT_VELOCITY_SQUARED:
            value = velocitySquared;
            break;
        case MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED:
            value = accelerationSquared;
            break;
        case MR_TASK_REWARD_ACTION_RATE_SQUARED:
            value = actionRateSquared;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED:
            value = limitViolationSquared;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE: {
            const float softFactor = clamp(
                operation.parameters.y,
                1.0e-6f,
                1.0f
            );
            for (uint action = 0u;
                 action < program.counts0.x;
                 ++action) {
                const MRTaskActionBindingGPU binding =
                    actions[action];
                const MRDofPropertiesGPU dof =
                    dofs[binding.indices.y];
                const float center =
                    0.5f * (dof.limits.x + dof.limits.y);
                const float halfRange =
                    0.5f *
                    (dof.limits.y - dof.limits.x) *
                    softFactor;
                const float position =
                    qState[qBase + binding.indices.z];
                value +=
                    max(center - halfRange - position, 0.0f) +
                    max(position - center - halfRange, 0.0f);
            }
            break;
        }
        case MR_TASK_REWARD_MECHANICAL_POWER:
            value = mechanicalPower;
            break;
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED: {
            const MRTaskIndexGroupGPU group =
                jointGroups[operation.source.y];
            float sum = 0.0f;
            for (uint local = 0u;
                 local < group.members.y;
                 ++local) {
                const uint action =
                    jointMembers[
                        group.members.x + local
                    ];
                const MRTaskActionBindingGPU binding =
                    actions[action];
                const float error =
                    qState[qBase + binding.indices.z] -
                    defaultQ[binding.indices.z];
                sum += error * error;
            }
            value = sum /
                max(float(group.members.y), 1.0f);
            break;
        }
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE: {
            const MRTaskIndexGroupGPU group =
                jointGroups[operation.source.y];
            float sum = 0.0f;
            for (uint local = 0u;
                 local < group.members.y;
                 ++local) {
                const uint action =
                    jointMembers[
                        group.members.x + local
                    ];
                const MRTaskActionBindingGPU binding =
                    actions[action];
                sum += abs(
                    qState[qBase + binding.indices.z] -
                    defaultQ[binding.indices.z]
                );
            }
            value = sum;
            break;
        }
        case MR_TASK_REWARD_GAIT_CONTACT_MATCH: {
            if (!moving) {
                value = 0.0f;
                break;
            }
            float matched = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const bool desired =
                    desiredSupportContact(group, phase);
                const bool actual =
                    compactContact[
                        compactBase + group.members.w
                    ] > program.dynamics.y;
                matched += float(desired == actual);
            }
            value = matched;
            break;
        }
        case MR_TASK_REWARD_SWING_CLEARANCE: {
            if (!moving) {
                value = 0.0f;
                break;
            }
            float clearanceReward = 0.0f;
            float count = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    desiredSupportContact(group, phase) ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const float error =
                    compactContact[
                        compactBase +
                        group.members.w + 3u
                    ] -
                    program.locomotion.z;
                clearanceReward += exp(
                    -(error * error) /
                    max(
                        operation.parameters.y,
                        1.0e-8f
                    )
                );
                count += 1.0f;
            }
            value =
                clearanceReward / max(count, 1.0f);
            break;
        }
        case MR_TASK_REWARD_FOOT_CLEARANCE: {
            float errorVelocity = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const MRBodyStateGPU foot =
                    bodyStates[
                        bodyBase + group.reference.x
                    ];
                const float3 offset = rotate(
                    foot.orientation,
                    group.kinematicReference.xyz
                );
                const float3 position =
                    foot.position.xyz + offset;
                const float3 velocity =
                    foot.linearVelocityAndInverseMass.xyz +
                    cross(
                        foot.angularVelocity.xyz,
                        offset
                    );
                const float heightError =
                    position.z -
                    program.locomotion.z;
                const float velocityWeight = tanh(
                    operation.parameters.z *
                    length(velocity.xy)
                );
                errorVelocity +=
                    heightError * heightError *
                    velocityWeight;
            }
            value = exp(
                -errorVelocity /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        }
        case MR_TASK_REWARD_SUPPORT_SLIP:
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) != 0u &&
                    (operation.source.y == MR_INVALID_INDEX ||
                     operation.source.y == groupIndex)) {
                    value += compactContact[
                        compactBase +
                        group.members.w + 1u
                    ];
                }
            }
            break;
        case MR_TASK_REWARD_FORBIDDEN_CONTACT:
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_FORBIDDEN) != 0u &&
                    (operation.source.y == MR_INVALID_INDEX ||
                     operation.source.y == groupIndex)) {
                    value = max(
                        value,
                        compactContact[
                            compactBase +
                            group.members.w
                        ]
                    );
                }
            }
            break;
        case MR_TASK_REWARD_INTERACTION_JOINT_TRACKING: {
            float squaredError = 0.0f;
            float squaredVelocityError = 0.0f;
            float jointCount = 0.0f;
            if (operation.source.y == MR_INVALID_INDEX) {
                for (uint action = 0u;
                     action < program.interaction.y;
                     ++action) {
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference = mix(
                        interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ],
                        interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ],
                        referenceBlend
                    );
                    const float delta =
                        qState[qBase + binding.indices.z] -
                        reference;
                    const float referenceVelocity =
                        (interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ] - interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ]) * program.interactionTiming.x;
                    const float velocityDelta =
                        vState[vBase + binding.indices.w] -
                        referenceVelocity;
                    squaredError += delta * delta;
                    squaredVelocityError +=
                        velocityDelta * velocityDelta;
                    jointCount += 1.0f;
                }
            } else {
                const MRTaskIndexGroupGPU group =
                    jointGroups[operation.source.y];
                for (uint member = 0u;
                     member < group.members.y;
                     ++member) {
                    const uint action = jointMembers[
                        group.members.x + member
                    ];
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference = mix(
                        interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ],
                        interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ],
                        referenceBlend
                    );
                    const float delta =
                        qState[qBase + binding.indices.z] -
                        reference;
                    const float referenceVelocity =
                        (interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ] - interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ]) * program.interactionTiming.x;
                    const float velocityDelta =
                        vState[vBase + binding.indices.w] -
                        referenceVelocity;
                    squaredError += delta * delta;
                    squaredVelocityError +=
                        velocityDelta * velocityDelta;
                    jointCount += 1.0f;
                }
            }
            const float positionScore = exp(
                -(squaredError / max(jointCount, 1.0f)) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float velocityScore = operation.parameters.z > 0.0f
                ? exp(
                      -(squaredVelocityError /
                          max(jointCount, 1.0f)) /
                      operation.parameters.z
                  )
                : positionScore;
            value = 0.5f * (positionScore + velocityScore);
            interactionMetric = value;
            interactionMetricWeight = 1.0f;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_ROOT_TRACKING: {
            const uint targetBase = referenceFrame * 7u;
            const uint nextTargetBase = nextReferenceFrame * 7u;
            const float3 targetPosition = mix(
                float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                ),
                float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ),
                referenceBlend
            );
            const float3 positionDelta =
                rootWorldPosition(program, qState + qBase) -
                targetPosition;
            const float4 rawOrientation = float4(
                qState[qBase + program.root.z + 3u],
                qState[qBase + program.root.z + 4u],
                qState[qBase + program.root.z + 5u],
                qState[qBase + program.root.z + 6u]
            );
            const float4 orientation = rawOrientation * rsqrt(
                max(dot(rawOrientation, rawOrientation), 1.0e-12f)
            );
            const float4 rawTargetOrientation = float4(
                interactionRootTargets[targetBase + 3u],
                interactionRootTargets[targetBase + 4u],
                interactionRootTargets[targetBase + 5u],
                interactionRootTargets[targetBase + 6u]
            );
            const float4 rawNextTargetOrientation = float4(
                interactionRootTargets[nextTargetBase + 3u],
                interactionRootTargets[nextTargetBase + 4u],
                interactionRootTargets[nextTargetBase + 5u],
                interactionRootTargets[nextTargetBase + 6u]
            );
            const float4 targetOrientation = quaternionInterpolate(
                normalize(rawTargetOrientation),
                normalize(rawNextTargetOrientation),
                referenceBlend
            );
            const float orientationError =
                1.0f - abs(dot(orientation, targetOrientation));
            const float positionScore = exp(
                -dot(positionDelta, positionDelta) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float orientationScore = exp(
                -(orientationError * orientationError) /
                max(operation.parameters.z, 1.0e-8f)
            );
            const float3 targetLinearVelocity =
                (float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ) - float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                )) * program.interactionTiming.x;
            const float3 linearVelocityDelta =
                rootWorldLinearVelocity(
                    program,
                    qState + qBase,
                    vState + vBase
                ) - targetLinearVelocity;
            const float linearVelocityScore =
                operation.parameters.w > 0.0f
                ? exp(
                      -dot(linearVelocityDelta, linearVelocityDelta) /
                      operation.parameters.w
                  )
                : positionScore;
            const float3 targetAngularVelocity =
                quaternionWorldAngularVelocity(
                    rawTargetOrientation,
                    rawNextTargetOrientation,
                    program.interactionTiming.x
                );
            const float3 angularVelocityDelta = float3(
                vState[vBase + program.root.w + 3u],
                vState[vBase + program.root.w + 4u],
                vState[vBase + program.root.w + 5u]
            ) - targetAngularVelocity;
            const float angularVelocityScore =
                operation.auxiliary.x > 0.0f
                ? exp(
                      -dot(angularVelocityDelta, angularVelocityDelta) /
                      operation.auxiliary.x
                  )
                : orientationScore;
            value = 0.25f * (
                positionScore + orientationScore +
                linearVelocityScore + angularVelocityScore
            );
            interactionMetric = value;
            interactionMetricWeight = 1.0f;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_ROOT_LINEAR_VELOCITY_ERROR: {
            const uint targetBase = referenceFrame * 7u;
            const uint nextTargetBase = nextReferenceFrame * 7u;
            const float3 targetLinearVelocity =
                (float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ) - float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                )) * program.interactionTiming.x;
            const float3 linearVelocityDelta =
                rootWorldLinearVelocity(
                    program,
                    qState + qBase,
                    vState + vBase
                ) - targetLinearVelocity;
            const float scaledError = length(linearVelocityDelta) /
                max(operation.parameters.y, 1.0e-8f);
            // Pseudo-Huber growth stays informative far from a fast teacher
            // without the unbounded quadratic leverage of a squared penalty.
            value = sqrt(1.0f + scaledError * scaledError) - 1.0f;
            break;
        }
        case MR_TASK_REWARD_OBJECT_GRASP: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const uint objectBody = object.flagsAndIndices[2];
            uint distinctMembers = 0u;
            float normalForce = 0.0f;
            for (uint member = 0u;
                 member < group.members.y;
                 ++member) {
                const uint memberBody =
                    contactMembers[group.members.x + member];
                bool touching = false;
                for (uint contact = 0u;
                     contact < activeContacts;
                     ++contact) {
                    const MRContactConstraintGPU constraint =
                        contacts[contactBase + contact];
                    const bool matched =
                        (constraint.bodyA == memberBody &&
                         constraint.bodyB == objectBody) ||
                        (constraint.bodyB == memberBody &&
                         constraint.bodyA == objectBody);
                    if (!matched) {
                        continue;
                    }
                    touching = true;
                    normalForce += abs(constraint.impulses.x) /
                        dispatch.timing.y;
                }
                distinctMembers += uint(touching);
            }
            const float memberScore = smoothstep(
                0.0f,
                max(operation.parameters.y, 1.0f),
                float(distinctMembers)
            );
            const float forceScore = 1.0f - exp(
                -normalForce /
                max(operation.parameters.z, 1.0e-6f)
            );
            value = memberScore * forceScore;
            break;
        }
        case MR_TASK_REWARD_OBJECT_LIFT: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            value = smoothstep(
                operation.parameters.y,
                operation.parameters.z,
                object.position.z
            );
            break;
        }
        case MR_TASK_REWARD_OBJECT_POSITION: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const float3 delta = object.position.xyz - float3(
                operation.parameters.y,
                operation.parameters.z,
                operation.parameters.w
            );
            value = exp(
                -dot(delta, delta) /
                max(operation.auxiliary.x, 1.0e-8f)
            );
            break;
        }
        case MR_TASK_REWARD_OBJECT_PLACEMENT: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const MRBodyStateGPU goal =
                sceneState[sceneBase + operation.source.y];
            const float3 positionDelta =
                object.position.xyz - goal.position.xyz;
            const float positionScore = exp(
                -dot(positionDelta, positionDelta) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float quietScore = 0.5f * (
                exp(
                    -dot(
                        object.linearVelocityAndInverseMass.xyz,
                        object.linearVelocityAndInverseMass.xyz
                    ) / max(operation.parameters.z, 1.0e-8f)
                ) +
                exp(
                    -dot(
                        object.angularVelocity.xyz,
                        object.angularVelocity.xyz
                    ) / max(operation.parameters.w, 1.0e-8f)
                )
            );
            value = positionScore * quietScore;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING: {
            const uint trackIndex = operation.source.z;
            const MRTaskInteractionContactGPU track =
                interactionContacts[trackIndex];
            if (track.binding.x != operation.source.y) {
                value = 0.0f;
                break;
            }
            const uint sampleIndex =
                referenceFrame * program.interaction.z + trackIndex;
            const MRTaskInteractionSampleGPU sample =
                interactionSamples[sampleIndex];
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const bool expectedContact =
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_STICK ||
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_ROLL ||
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_SLIDE;
            const bool supportGroup =
                (group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u;
            const uint wrench = compactBase + group.reference.y;
            const bool actualContact = supportGroup
                ? compactContact[
                      compactBase + group.members.w
                  ] > program.dynamics.y
                : length(float3(
                      compactContact[wrench + 0u],
                      compactContact[wrench + 1u],
                      compactContact[wrench + 2u]
                  )) > program.dynamics.y;
            const float modeScore = float(
                expectedContact == actualContact
            );
            float normalizedSquaredError = 0.0f;
            float validFeatureCount = 0.0f;
            device const float* compact =
                compactContact + compactBase;
            for (uint feature = 0u;
                 feature <
                    MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT;
                 ++feature) {
                if ((sample.metadata.y & (1u << feature)) == 0u) {
                    continue;
                }
                const uint targetIndex =
                    sampleIndex *
                        MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT +
                    feature;
                const float tolerance = max(
                    interactionContactTolerances[targetIndex],
                    1.0e-8f
                );
                const float delta =
                    (
                        supportPatchFeature(
                            program,
                            group,
                            compact,
                            feature
                        ) -
                        interactionContactTargets[targetIndex]
                    ) /
                    tolerance;
                normalizedSquaredError += min(
                    delta * delta,
                    1.0e6f
                );
                validFeatureCount += 1.0f;
            }
            const float fieldScore = validFeatureCount > 0.0f
                ? exp(
                      -(normalizedSquaredError / validFeatureCount) /
                      max(operation.parameters.z, 1.0e-8f)
                  )
                : modeScore;
            const float confidence = clamp(
                sample.confidence.x,
                0.0f,
                1.0f
            );
            const float contactScore = mix(
                fieldScore,
                modeScore,
                clamp(operation.parameters.y, 0.0f, 1.0f)
            );
            value = confidence * contactScore;
            interactionMetric = value;
            interactionMetricWeight = confidence;
            break;
        }
        default:
            value = 0.0f;
            break;
        }
        const float contribution =
            operation.parameters.x * value;
        interactionTrackingSum += interactionMetric;
        interactionTrackingWeight += interactionMetricWeight;
        reward += contribution;
        for (uint channel = 0u;
             channel < program.interactionOffsets1.w;
             ++channel) {
            if (outcomeRewardOperations[channel] != operation.source.x) {
                continue;
            }
            if (channel < 4u) {
                outcomeChannels0[channel] += contribution;
            } else {
                outcomeChannels1[channel - 4u] += contribution;
            }
        }
        switch (operation.source.x) {
        case MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING:
        case MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING:
        case MR_TASK_REWARD_YAW_VELOCITY_TRACKING:
        case MR_TASK_REWARD_CONSTANT:
        case MR_TASK_REWARD_GAIT_CONTACT_MATCH:
        case MR_TASK_REWARD_SWING_CLEARANCE:
        case MR_TASK_REWARD_FOOT_CLEARANCE:
        case MR_TASK_REWARD_INTERACTION_JOINT_TRACKING:
            rewardBreakdown0.x += contribution;
            break;
        case MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED:
        case MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED:
        case MR_TASK_REWARD_TILT_SQUARED:
        case MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED:
        case MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED:
        case MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED:
        case MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS:
        case MR_TASK_REWARD_UPRIGHTNESS:
        case MR_TASK_REWARD_SUPPORT_CONTACT_COUNT:
        case MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL:
        case MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL:
        case MR_TASK_REWARD_BODY_UP_EXPONENTIAL:
        case MR_TASK_REWARD_STANDING_COMPLETION:
        case MR_TASK_REWARD_RESTORATION:
        case MR_TASK_REWARD_INTERACTION_ROOT_TRACKING:
        case MR_TASK_REWARD_INTERACTION_ROOT_LINEAR_VELOCITY_ERROR:
        case MR_TASK_REWARD_OBJECT_GRASP:
        case MR_TASK_REWARD_OBJECT_LIFT:
        case MR_TASK_REWARD_OBJECT_POSITION:
        case MR_TASK_REWARD_OBJECT_PLACEMENT:
        case MR_TASK_REWARD_RECOVERY_TILT_PROGRESS:
        case MR_TASK_REWARD_RECOVERY_COMPLETION:
        case MR_TASK_REWARD_WHOLE_BODY_RECOVERY:
        case MR_TASK_REWARD_LINK_CLEARANCE_BARRIER:
        case MR_TASK_REWARD_PROJECTILE_MISS:
        case MR_TASK_REWARD_PROJECTILE_EVASION:
        case MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS:
            rewardBreakdown0.y += contribution;
            break;
        case MR_TASK_REWARD_JOINT_VELOCITY_SQUARED:
            rewardBreakdown0.z += contribution;
            break;
        case MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED:
            rewardBreakdown0.w += contribution;
            break;
        case MR_TASK_REWARD_ACTION_RATE_SQUARED:
        case MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE:
        case MR_TASK_REWARD_JOINT_CBF_CORRECTION:
        case MR_TASK_REWARD_JOINT_CBF_BUFFER:
        case MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE:
            rewardBreakdown1.x += contribution;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED:
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE:
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED:
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE:
            rewardBreakdown1.y += contribution;
            break;
        case MR_TASK_REWARD_MECHANICAL_POWER:
            rewardBreakdown1.z += contribution;
            break;
        case MR_TASK_REWARD_SUPPORT_SLIP:
        case MR_TASK_REWARD_FORBIDDEN_CONTACT:
        case MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING:
            rewardBreakdown1.w += contribution;
            break;
        default:
            break;
        }
    }
    // TaskPack weights are rates. Integrating at the control boundary keeps
    // reward magnitude and PPO critic targets independent of control rate.
    reward *= dispatch.timing.x;
    rewardBreakdown0 *= dispatch.timing.x;
    rewardBreakdown1 *= dispatch.timing.x;
    outcomeChannels0 *= dispatch.timing.x;
    outcomeChannels1 *= dispatch.timing.x;

    const float tracking = exp(-trackingError / 0.25f);
    const float yawTracking = exp(-yawError / 0.25f);
    const uint episodeSteps = state.episode.x + 1u;
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
        const uint maximumBand = operation.schedule.y == MR_INVALID_INDEX
            ? program.schedule.z - 1u
            : operation.schedule.y;
        if (curriculum < operation.schedule.x ||
            curriculum > maximumBand) {
            continue;
        }
        bool triggered = false;
        switch (operation.source.x) {
        case MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT:
            triggered =
                height < operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_MAXIMUM_ROOT_HEIGHT:
            triggered =
                height > operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_MAXIMUM_TILT:
            triggered =
                tilt > operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_CONTACT_GROUP: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            triggered =
                compactContact[
                    compactBase + group.members.w
                ] > operation.parameters.x;
            break;
        }
        case MR_TASK_TERMINATE_PROJECTILE_CONTACT: {
            device const MRTaskContactGroupGPU& group =
                contactGroups[operation.source.y];
            if (!eventSequenceAvailable ||
                activeImpactEvent == MR_INVALID_INDEX ||
                state.status.w == 0u) {
                break;
            }
            const uint projectileBody =
                impactEvents[activeImpactEvent].binding.w;
            for (uint contact = 0u;
                 contact < activeContacts && !triggered;
                 ++contact) {
                const MRContactConstraintGPU constraint =
                    contacts[contactBase + contact];
                const bool projectileA =
                    constraint.bodyA == projectileBody;
                const bool projectileB =
                    constraint.bodyB == projectileBody;
                if (projectileA == projectileB) {
                    continue;
                }
                const uint other = projectileA
                    ? constraint.bodyB
                    : constraint.bodyA;
                triggered = bodyMember(
                    other,
                    group,
                    contactMembers
                ) && abs(constraint.impulses.x) /
                    dispatch.timing.y > operation.parameters.x;
            }
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

    // ARDY_CLOSED_LOOP_BALANCE_POLICY_V5. The reference clock is
    // subordinate to accepted physical support. The policy owns balance and
    // unloading; ARDY owns the whole-body motion reference.
    if (interactionPhysicsGated(program)) {
        float gatedFramePosition = interactionFramePosition(
            program,
            state,
            dispatch.timing.x
        );
        bool fallLatched = interactionFallIsLatched(state);
        bool forbiddenContact = false;
        uint actualSupportCount = 0u;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            const MRTaskContactGroupGPU group = contactGroups[groupIndex];
            if ((group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u &&
                compactContact[compactBase + group.members.w] >
                    program.dynamics.y) {
                ++actualSupportCount;
            }
            if ((group.members.z & MR_TASK_CONTACT_FORBIDDEN) != 0u &&
                compactContact[compactBase + group.members.w] > 0.5f) {
                forbiddenContact = true;
            }
        }

        uint comparedContacts = 0u;
        uint expectedContacts = 0u;
        uint strictContactMismatches = 0u;
        bool transitionMode = false;
        for (uint contactIndex = 0u;
             contactIndex < program.interaction.z;
             ++contactIndex) {
            const MRTaskInteractionContactGPU binding =
                interactionContacts[contactIndex];
            const uint sampleIndex =
                referenceFrame * program.interaction.z + contactIndex;
            const MRTaskInteractionSampleGPU sample =
                interactionSamples[sampleIndex];
            if (sample.confidence.x < 0.35f) {
                continue;
            }
            const uint mode = sample.metadata.x;
            const bool expectedContact =
                mode == MR_TASK_INTERACTION_CONTACT_STICK ||
                mode == MR_TASK_INTERACTION_CONTACT_ROLL ||
                mode == MR_TASK_INTERACTION_CONTACT_SLIDE;
            const bool transitional =
                mode == MR_TASK_INTERACTION_CONTACT_RELEASE ||
                mode == MR_TASK_INTERACTION_CONTACT_APPROACH;
            const MRTaskContactGroupGPU group =
                contactGroups[binding.binding.x];
            const uint wrench = compactBase + group.reference.y;
            const bool supportGroup =
                (group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u;
            const bool actualContact = supportGroup
                ? compactContact[compactBase + group.members.w] >
                    program.dynamics.y
                : length(float3(
                      compactContact[wrench + 0u],
                      compactContact[wrench + 1u],
                      compactContact[wrench + 2u]
                  )) > program.dynamics.y;
            ++comparedContacts;
            expectedContacts += expectedContact ? 1u : 0u;
            transitionMode = transitionMode || transitional;
            if (!transitional && expectedContact != actualContact) {
                ++strictContactMismatches;
            }
        }

        float jointErrorSquared = 0.0f;
        for (uint action = 0u;
             action < program.interaction.y;
             ++action) {
            const MRTaskActionBindingGPU binding = actions[action];
            const float reference = mix(
                interactionJointTargets[
                    referenceFrame * program.interaction.y + action
                ],
                interactionJointTargets[
                    nextReferenceFrame * program.interaction.y + action
                ],
                referenceBlend
            );
            const float error = qState[qBase + binding.indices.z] - reference;
            jointErrorSquared += error * error;
        }
        const float jointRms = sqrt(
            jointErrorSquared /
            max(float(program.interaction.y), 1.0f)
        );

        const bool physicalFall =
            height < 0.55f ||
            tilt > 0.50f ||
            forbiddenContact ||
            reason == MR_TASK_TERMINATION_HEIGHT ||
            reason == MR_TASK_TERMINATION_TILT ||
            reason == MR_TASK_TERMINATION_CONTACT;
        fallLatched = fallLatched || physicalFall;

        // Bootstrap only after 0.30 s of actual quiet bilateral support. The
        // spare training-only lane is unused by this non-threat task.
        const bool bootstrapFrame = gatedFramePosition <= 0.50f;
        float quietSupportSeconds = bootstrapFrame
            ? state.threatTeacher.w
            : 0.30f;
        const bool quietSupport =
            actualSupportCount >= 2u &&
            height > 0.68f &&
            tilt < 0.08f &&
            length(baseLinear.xy) < 0.08f &&
            abs(baseLinear.z) < 0.10f &&
            length(baseAngular) < 0.20f;
        if (bootstrapFrame) {
            quietSupportSeconds = quietSupport
                ? quietSupportSeconds + dispatch.timing.x
                : 0.0f;
        }
        state.threatTeacher.w = quietSupportSeconds;
        const bool bootstrapped =
            !bootstrapFrame || quietSupportSeconds >= 0.30f;

        uint nextRateCode = 0u;
        const bool atEnd =
            (program.interaction.w & MR_TASK_INTERACTION_LOOP) == 0u &&
            gatedFramePosition >=
                float(program.interaction.x - 1u) - 1.0e-4f;
        const bool contactsReady =
            comparedContacts == 0u || strictContactMismatches == 0u;
        const bool expectedFlight =
            comparedContacts > 0u && expectedContacts == 0u &&
            !transitionMode;
        const bool supportSafe =
            actualSupportCount > 0u &&
            height > 0.64f &&
            tilt < 0.25f;

        if (!fallLatched && !done && !atEnd && bootstrapped && contactsReady) {
            if (transitionMode) {
                // RELEASE must move far enough to unload; APPROACH must move
                // far enough to make contact. Requiring equality here would
                // deadlock both transitions on one static pose.
                if (supportSafe && jointRms <= 0.60f) {
                    nextRateCode = 1u;
                }
            } else if (expectedFlight) {
                if (actualSupportCount == 0u &&
                    height > 0.60f && tilt < 0.38f) {
                    nextRateCode = 3u;
                }
            } else if (supportSafe) {
                // The staged preload runs slowly. Once the ARDY stride begins,
                // tight tracking may use half rate; lagging support stays slow.
                const bool stagedPrefix = gatedFramePosition < 24.0f;
                nextRateCode = stagedPrefix
                    ? 1u
                    : jointRms <= 0.24f
                    ? 2u
                    : jointRms <= 0.55f
                    ? 1u
                    : 0u;
            }
        }

        const float nextRate = nextRateCode == 1u
            ? 0.25f
            : nextRateCode == 2u
            ? 0.50f
            : nextRateCode == 3u
            ? 1.00f
            : 0.0f;
        gatedFramePosition +=
            dispatch.timing.x * program.interactionTiming.x * nextRate;
        if ((program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u) {
            gatedFramePosition = fmod(
                gatedFramePosition,
                float(program.interaction.x)
            );
        } else {
            gatedFramePosition = min(
                gatedFramePosition,
                float(program.interaction.x - 1u)
            );
        }
        state.status.w = packInteractionClock(
            gatedFramePosition,
            nextRateCode,
            fallLatched
        );
    }

    const float episodeReturn =
        state.airReturnTracking.z + reward;
    // Report continuous episode evidence; it never decides whether another
    // difficulty band or learner update is allowed.
    const float interactionTrackingScore =
        interactionTrackingWeight > 0.0f
        ? interactionTrackingSum / interactionTrackingWeight
        : 0.0f;
    const bool interactionReference =
        (program.schedule.w &
         MR_TASK_PROGRAM_INTERACTION_RESET) != 0u;
    const float evidenceTracking = interactionReference
        ? interactionTrackingScore
        : tracking;
    const float episodeTracking =
        state.airReturnTracking.w + evidenceTracking;
    const float episodeMeanTrackingScore =
        episodeTracking /
        max(float(episodeSteps), 1.0f);
    const float episodeTrackingScore = episodeMeanTrackingScore;
    const float trackingScore = interactionReference
        ? interactionTrackingScore
        : 0.5f * (tracking + yawTracking);
    const uint terrainLevel = state.episode.w;
    const float4 scheduleSeconds = scheduleSecondsForBand(
        program,
        curriculum
    );

    if (!figureEightConfigured && !done && state.schedule.x <= 1u) {
        state.commandAndPhase.xyz = sampledCommand(
            dispatch,
            program,
            commandCurriculum,
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
            scheduleSeconds.x,
            scheduleSeconds.y
        );
    } else if (!figureEightConfigured && !done) {
        --state.schedule.x;
    }
    if (state.schedule.y == 0u || done) {
        state.schedule.y = durationSteps(
            dispatch,
            environment,
            state.episode.y,
            episodeSteps,
            65u,
            scheduleSeconds.z,
            scheduleSeconds.w
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
    state.airReturnTracking = float4(
        0.0f,
        height,
        done ? 0.0f : episodeReturn,
        done ? 0.0f : episodeTracking
    );
    state.recovery = done
        ? float4(0.0f)
        : float4(
              tilt,
              recoveryCompleted ? 0.0f : recoveryPeakTilt,
              recoveryCompleted ? 0.0f : recoveryStableTime,
              recoveryActive && !recoveryCompleted ? 1.0f : 0.0f
          );
    uint impactState = state.recoveryStats.w;
    if (impactContactLatched) {
        impactState |= kImpactContactLatched;
    }
    if (newProjectileContact) {
        impactState |= kImpactContactPublished;
    }
    const bool completedImpactWindow =
        recoveryCompleted ||
        (!recoveryActive && impactWindowElapsed);
    if (completedImpactWindow) {
        const uint nextOrder = min(
            impactOrder(state) + 1u,
            kImpactOrderMask
        );
        impactState =
            (impactState & kImpactOffsetMask) |
            kImpactEnabled |
            nextOrder;
        state.status.w = 0u;
    }
    state.recoveryStats = done
        ? uint4(0u)
        : uint4(
              recoveryEventCount,
              recoveryCompletionCount,
              recoveryTouch ? 1u : 0u,
              impactState
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
            (program.layout.w - 2u) * program.counts0.x;
        writeFrame(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            state.episode.y,
            episodeSteps,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            actionHistory + delayBase +
                (program.layout.w - 2u) * program.counts0.x,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
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
            program,
            arena,
            dispatch.timing.x,
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            currentAction,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
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
            const uint velocityIndex = actions[action].indices.w;
            previousJointVelocity[previousVelocityBase + action] =
                velocityIndex == MR_INVALID_INDEX
                ? 0.0f
                : vState[vBase + velocityIndex];
            const uint qIndex = actions[action].indices.z;
            previousJointVelocity[
                previousVelocityBase + program.counts0.x + action
            ] =
                qIndex == MR_INVALID_INDEX
                ? 0.0f
                : qState[qBase + qIndex];
        }
    }

    const bool finalPolicyObservation =
        dispatch.timing.z != 0.0f &&
        pass.controlStep + 1u == dispatch.counts.y;
    if (finalPolicyObservation) {
        const uint actorObservationSize =
            dispatch.outputs.x / dispatch.counts.x;
        const uint actorOutputBase =
            dispatch.counts.y * dispatch.outputs.x +
            environment * actorObservationSize;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            actorObservations[actorOutputBase + index] =
                actorHistory[historyBase + index];
        }
        writeCurrentActor(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            state.episode.y,
            episodeSteps,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            actionHistory + delayBase +
                (program.layout.w - 2u) * program.counts0.x,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            sensorBias + biasBase,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            actorObservations + actorOutputBase + historyElements
        );
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
    transition.rewardAndState =
        float4(reward, trackingScore, height, tilt);
    transition.termination = uint4(
        done ? 1u : 0u,
        timeout ? 1u : 0u,
        physicsError ? 1u : 0u,
        reason
    );
    transition.rewardBreakdown0 = rewardBreakdown0;
    transition.rewardBreakdown1 = rewardBreakdown1;
    transition.outcomeChannels0 = outcomeChannels0;
    transition.outcomeChannels1 = outcomeChannels1;
    transition.policyRevision =
        dispatch.policyRevision;
    transition.episodeTrackingScore =
        done && !physicsError
        ? episodeTrackingScore
        : 0.0f;
    transition.taskProgress = uint4(
        curriculum,
        terrainLevel,
        activeImpactEvent == MR_INVALID_INDEX
            ? 0u
            : activeImpactEvent + 1u,
        impactTransitionFlags |
            recoveryOutcomeFlags |
            (standingCompleted ? MR_TASK_OUTCOME_STANDING : 0u) |
            (restoredCompleted ? MR_TASK_OUTCOME_RESTORED : 0u) |
            (
                (program.schedule.w &
                 MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u &&
                    program.interactionTiming.z > 0.0f
                ? MR_TASK_OUTCOME_POLICY_RESIDUAL : 0u
            ) |
            (
                (program.schedule.w &
                 MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u &&
                    program.interactionTiming.z == 0.0f
                ? MR_TASK_OUTCOME_INTERACTION_TEACHER : 0u
            )
    );
    transitions[transitionIndex] = transition;
}

// One native thread reduces task-wide physical evidence. It never changes the
// episode sampling distribution or emits a pass/fail decision.
kernel void mr_locomotion_task_update_evidence(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device MRTaskEvidenceStateGPU* evidenceState
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
        program.schedule.z == 0u) {
        return;
    }

    MRTaskEvidenceStateGPU state = evidenceState[0];
    const ulong completedSteps = state.controlSteps + 1ul;
    const uint transitionBase =
        pass.controlStep * dispatch.outputs.z;
    for (uint environment = 0u;
         environment < dispatch.counts.x;
         ++environment) {
        const MRTaskTransitionGPU transition =
            transitions[transitionBase + environment];
        const uint impact = transition.taskProgress.w;
        state.impactContactCount +=
            (impact & MR_TASK_IMPACT_CONTACT) != 0u ? 1u : 0u;
        state.impactCleanMissCount +=
            (impact & MR_TASK_IMPACT_MISSED) != 0u ? 1u : 0u;
        state.balanceFailureCount +=
            transition.termination.x != 0u &&
            (transition.termination.w == MR_TASK_TERMINATION_HEIGHT ||
             transition.termination.w == MR_TASK_TERMINATION_TILT)
            ? 1u
            : 0u;
        if (transition.termination.x != 0u &&
            transition.termination.z == 0u) {
            state.trackingScoreSum +=
                transition.episodeTrackingScore;
            ++state.completedEpisodeCount;
            if (transition.termination.y != 0u) {
                ++state.timeoutEpisodeCount;
            }
        }
    }
    if (completedSteps % ulong(program.schedule.x) == 0ul) {
        const float completed =
            float(state.completedEpisodeCount);
        const ulong exposure =
            ulong(program.schedule.x) * ulong(dispatch.counts.x);
        const float rateScale = 1000000.0f /
            max(float(exposure), 1.0f);
        state.lastCompletedEpisodeCount = state.completedEpisodeCount;
        state.lastWindow = uint4(
            uint(round(float(state.impactContactCount) * rateScale)),
            uint(round(float(state.impactCleanMissCount) * rateScale)),
            uint(round(float(state.balanceFailureCount) * rateScale)),
            uint(round(
                state.trackingScoreSum /
                max(completed, 1.0f) * 1000000.0f
            ))
        );
        ++state.evidenceWindows;
        state.completedEpisodeCount = 0u;
        state.timeoutEpisodeCount = 0u;
        state.impactContactCount = 0u;
        state.impactCleanMissCount = 0u;
        state.balanceFailureCount = 0u;
        state.trackingScoreSum = 0.0f;
    }
    state.controlSteps = completedSteps;
    evidenceState[0] = state;
}
