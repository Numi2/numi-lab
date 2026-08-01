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

inline float cleanObservation(
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskObservationOperatorGPU operation,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
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
        value = state.commandAndPhase[operation.source.z];
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
    case MR_TASK_OBSERVE_MASKED_DEPTH:
        // Filled by the attached Visual Presentation device program after
        // the physics-authored observation pass and before policy inference.
        value = 0.0f;
        break;
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
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
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
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
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
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
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
    device const MRTaskProgramHeaderGPU& program,
    device const MRTaskObservationOperatorGPU* criticOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
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
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
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
    device const float* actionStream [[buffer(7)]],
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
    for (uint action = 0u; action < program.counts0.x; ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        const uint dof = binding.indices.w - program.root.w;
        if (dof >= dispatch.counts.w) {
            continue;
        }
        const float target = clamp(
            defaultQ[binding.indices.z] +
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
    const uint globalCurriculum = min(
        curriculumState[0].commandLevel,
        program.schedule.z - 1u
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
                program.scheduleSeconds.x,
                program.scheduleSeconds.y
            ),
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                33u,
                program.scheduleSeconds.z,
                program.scheduleSeconds.w
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
                commandCurriculum,
                environment,
                episode,
                0u,
                curriculum
            ),
            0.0f
        );
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
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
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
        globalCurriculum >= impactEvents[0].binding.z) {
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
                    const float horizontalSpeed = randomRange(
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
        const float requested = clamp(
            actionStream[actionBase + action],
            -1.0f,
            1.0f
        );
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
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            binding.indices.w
        ] = clamp(
            defaultQ[binding.indices.z] +
                binding.parameters.x * filtered,
            binding.parameters.y,
            binding.parameters.z
        );
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
        const uint velocityIndex =
            actions[action].indices.w;
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
    device const MRTaskCurriculumStateGPU* curriculumState
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
        program.schedule.z - 1u
    );
    state.episode.z = curriculum;
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
    const float4 orientation = rootOrientation(program, q);
    const float3 rootLinearVelocity =
        rootWorldLinearVelocity(program, q, v);
    const float3 baseLinear = rotateInverse(
        orientation,
        rootLinearVelocity
    );
    const float3 baseAngular = rotateInverse(
        orientation,
        float3(
            vState[vBase + program.root.w + 3u],
            vState[vBase + program.root.w + 4u],
            vState[vBase + program.root.w + 5u]
        )
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
        const float actionDelta =
            currentAction - previousAction;
        const float lower =
            max(dof.limits.x - position, 0.0f);
        const float upper =
            max(position - dof.limits.y, 0.0f);
        velocitySquared += velocity * velocity;
        accelerationSquared +=
            acceleration * acceleration;
        actionRateSquared +=
            actionDelta * actionDelta;
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
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        if (operation.source.x ==
                MR_TASK_REWARD_RECOVERY_TILT_PROGRESS ||
            operation.source.x ==
                MR_TASK_REWARD_RECOVERY_COMPLETION) {
            recoveryConfigured = true;
            recoveryActivationTilt = operation.parameters.y;
            recoveryStableTilt = operation.parameters.z;
            recoveryStableDuration = operation.parameters.w;
            recoveryContactGroup = operation.source.y;
            break;
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
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        float value = 0.0f;
        switch (operation.source.x) {
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
            value = tilt * tilt;
            break;
        case MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED:
            value = dot(gravity.xy, gravity.xy);
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED: {
            const float error =
                height - program.locomotion.x;
            value = error * error;
            break;
        }
        case MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED:
            value = clamp(
                height / max(program.locomotion.x, 1.0e-6f),
                0.0f,
                1.0f
            );
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS:
            value = min(
                max(
                    height - state.airReturnTracking.y,
                    0.0f
                ) / dispatch.timing.x,
                2.0f
            );
            break;
        case MR_TASK_REWARD_UPRIGHTNESS:
            value = clamp(0.5f * (1.0f - gravity.z), 0.0f, 1.0f);
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
            bool bothSupported = true;
            uint supportTotal = 0u;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                bothSupported = bothSupported &&
                    compactContact[
                        compactBase + group.members.w
                    ] > program.dynamics.y;
                ++supportTotal;
            }
            value = float(
                supportTotal > 0u &&
                bothSupported &&
                height >= operation.parameters.y &&
                gravity.z <= -operation.parameters.z
            );
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
        default:
            value = 0.0f;
            break;
        }
        const float contribution =
            operation.parameters.x * value;
        reward += contribution;
        switch (operation.source.x) {
        case MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING:
        case MR_TASK_REWARD_YAW_VELOCITY_TRACKING:
        case MR_TASK_REWARD_CONSTANT:
        case MR_TASK_REWARD_GAIT_CONTACT_MATCH:
        case MR_TASK_REWARD_SWING_CLEARANCE:
        case MR_TASK_REWARD_FOOT_CLEARANCE:
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
        case MR_TASK_REWARD_RECOVERY_TILT_PROGRESS:
        case MR_TASK_REWARD_RECOVERY_COMPLETION:
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
        bool triggered = false;
        switch (operation.source.x) {
        case MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT:
            triggered =
                height < operation.parameters.x;
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

    const float episodeReturn =
        state.airReturnTracking.z + reward;
    // Unitree's linear-command curriculum is gated by the linear tracking
    // reward alone. Keep the combined linear/yaw score below for rollout
    // reporting, but do not let a still-learning yaw controller prevent an
    // otherwise successful episode from expanding the x/y command range.
    const float episodeTracking =
        state.airReturnTracking.w + tracking;
    const float linearEpisodeTrackingScore =
        episodeTracking /
        max(float(episodeSteps), 1.0f);
    const bool recoveryCurriculum =
        (program.schedule.w &
         MR_TASK_PROGRAM_RECOVERY_CURRICULUM) != 0u;
    const float episodeRecoveryScore =
        recoveryEventCount == 0u
        ? 0.0f
        : float(recoveryCompletionCount) /
            float(recoveryEventCount);
    const float episodeTrackingScore = recoveryCurriculum
        ? episodeRecoveryScore
        : linearEpisodeTrackingScore;
    const float trackingScore =
        0.5f * (tracking + yawTracking);
    const bool successful =
        timeout &&
        !physicsError &&
        episodeTrackingScore >= program.locomotion.w;
    uint terrainLevel = state.episode.w;
    if (done) {
        if (successful) {
            if (program.terrain.w != 0u &&
                curriculum >= min(3u, program.schedule.z - 1u)) {
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
            program.scheduleSeconds.x,
            program.scheduleSeconds.y
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
            program.scheduleSeconds.z,
            program.scheduleSeconds.w
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
            currentAction,
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
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            currentAction,
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
        impactTransitionFlags
    );
    transitions[transitionIndex] = transition;
}

// One native thread owns the global command curriculum. Episode outcomes are
// accumulated across the whole evaluation window, so early-reset environments
// rejoin the promotion evidence instead of becoming permanently phase-shifted.
kernel void mr_locomotion_task_update_curriculum(
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
        program.schedule.z == 0u) {
        return;
    }

    MRTaskCurriculumStateGPU state = curriculumState[0];
    const ulong completedSteps = state.controlSteps + 1ul;
    uint level = min(
        state.commandLevel,
        program.schedule.z - 1u
    );
    const uint transitionBase =
        pass.controlStep * dispatch.outputs.z;
    const bool projectileCurriculum =
        (program.schedule.w &
         MR_TASK_PROGRAM_PROJECTILE_OUTCOME_CURRICULUM) != 0u;
    for (uint environment = 0u;
         environment < dispatch.counts.x;
         ++environment) {
        const MRTaskTransitionGPU transition =
            transitions[transitionBase + environment];
        if (projectileCurriculum) {
            const uint impact = transition.taskProgress.w;
            const bool cleanMiss =
                (impact & MR_TASK_IMPACT_MISSED) != 0u;
            const bool contact =
                (impact & MR_TASK_IMPACT_CONTACT) != 0u;
            if (cleanMiss || contact) {
                ++state.completedEpisodeCount;
                state.timeoutEpisodeCount += ulong(cleanMiss);
            }
            if (transition.termination.x != 0u &&
                (transition.termination.w ==
                     MR_TASK_TERMINATION_HEIGHT ||
                 transition.termination.w ==
                     MR_TASK_TERMINATION_TILT)) {
                state.trackingScoreSum += 1.0f;
            }
        } else if (transition.termination.x != 0u &&
            transition.termination.z == 0u) {
            state.trackingScoreSum +=
                transition.episodeTrackingScore;
            ++state.completedEpisodeCount;
            if (transition.termination.y != 0u) {
                ++state.timeoutEpisodeCount;
            }
        }
    }
    if (completedSteps % ulong(program.schedule.x) == 0ul &&
        level + 1u < program.schedule.z) {
        const float completed =
            float(state.completedEpisodeCount);
        const float meanTracking =
            state.trackingScoreSum / max(completed, 1.0f);
        const float survivalFraction =
            float(state.timeoutEpisodeCount) /
            max(completed, 1.0f);
        const float balanceFailureFraction =
            state.trackingScoreSum / max(completed, 1.0f);
        const bool advance = projectileCurriculum
            ? state.completedEpisodeCount != 0ul &&
                survivalFraction >= program.locomotion.w &&
                balanceFailureFraction <= program.commandUpper.w
            : state.completedEpisodeCount != 0ul &&
                meanTracking > program.locomotion.w &&
                survivalFraction >= program.commandUpper.w;
        if (advance) {
            ++level;
        }
    }
    if (completedSteps % ulong(program.schedule.x) == 0ul) {
        state.completedEpisodeCount = 0ul;
        state.timeoutEpisodeCount = 0ul;
        state.trackingScoreSum = 0.0f;
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
            transition.taskProgress.z,
            transition.taskProgress.w
        );
    }
}
