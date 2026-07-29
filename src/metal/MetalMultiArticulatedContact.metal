#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/multi_contact_shared.h"
#include "metalrobo/quality_solver_shared.h"

using namespace metal;

namespace {

inline float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 doubledCross =
        2.0f * cross(quaternion.xyz, value);
    return value +
        quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

inline float3 contactAxis(
    device const MRMultiContactGPU& contact,
    const uint localRow
) {
    return localRow == 0u
        ? contact.normal.xyz
        : (localRow == 1u
            ? contact.tangentU.xyz
            : contact.tangentV.xyz);
}

inline uint endpointKind(
    device const MRMultiContactEndpointsGPU& endpoint,
    const bool second
) {
    return second ? endpoint.kindB : endpoint.kindA;
}

inline uint endpointBody(
    device const MRMultiContactEndpointsGPU& endpoint,
    const bool second
) {
    return second ? endpoint.bodyB : endpoint.bodyA;
}

inline uint endpointSlice(
    device const MRMultiContactEndpointsGPU& endpoint,
    const bool second
) {
    return second ? endpoint.sliceB : endpoint.sliceA;
}

inline uint endpointQuery(
    device const MRMultiContactEndpointsGPU& endpoint,
    const bool second
) {
    return second ? endpoint.queryB : endpoint.queryA;
}

inline float3 endpointLocalPoint(
    device const MRMultiContactGPU& contact,
    const bool second
) {
    return second
        ? contact.localPointB.xyz
        : contact.localPointA.xyz;
}

inline float articulatedCoefficient(
    device const MRMultiContactEndpointsGPU& endpoint,
    device const MRMultiContactJacobianSliceGPU* slices,
    device const float* pointJacobians,
    const uint environment,
    const uint globalDof,
    const uint axis,
    const bool second
) {
    const uint sliceIndex = endpointSlice(endpoint, second);
    if (sliceIndex == MR_INVALID_INDEX) {
        return 0.0f;
    }
    const MRMultiContactJacobianSliceGPU slice =
        slices[sliceIndex];
    if (globalDof < slice.vOffset ||
        globalDof >= slice.vOffset + slice.nv) {
        return 0.0f;
    }
    const uint query = endpointQuery(endpoint, second);
    if (query >= slice.queryCount) {
        return NAN;
    }
    return pointJacobians[
        slice.jacobianOffset +
        environment * slice.jacobianEnvironmentStride +
        (query * 3u + axis) * slice.nv +
        (globalDof - slice.vOffset)
    ];
}

inline float sceneCoefficient(
    device const MRMultiContactGPU& contact,
    device const MRMultiContactEndpointsGPU& endpoint,
    device const MRBodyStateGPU* sceneBodies,
    device const uint* sceneVelocityOffsets,
    const uint environment,
    const uint sceneBodyCount,
    const uint globalDof,
    const float3 axis,
    const bool second
) {
    const uint bodyIndex = endpointBody(endpoint, second);
    if (bodyIndex >= sceneBodyCount) {
        return NAN;
    }
    const uint offset = sceneVelocityOffsets[bodyIndex];
    if (offset == MR_INVALID_INDEX ||
        globalDof < offset ||
        globalDof >= offset + 6u) {
        return 0.0f;
    }
    const MRBodyStateGPU state =
        sceneBodies[environment * sceneBodyCount + bodyIndex];
    const uint localDof = globalDof - offset;
    if (localDof < 3u) {
        return axis[localDof];
    }
    const float3 pointOffset = quaternionRotate(
        state.orientation,
        endpointLocalPoint(contact, second)
    );
    return cross(pointOffset, axis)[localDof - 3u];
}

inline float endpointCoefficient(
    device const MRMultiContactGPU& contact,
    device const MRMultiContactEndpointsGPU& endpoint,
    device const MRMultiContactJacobianSliceGPU* slices,
    device const float* pointJacobians,
    device const MRBodyStateGPU* sceneBodies,
    device const uint* sceneVelocityOffsets,
    const uint environment,
    const uint sceneBodyCount,
    const uint globalDof,
    const uint localAxis,
    const float3 axis,
    const bool second
) {
    const uint kind = endpointKind(endpoint, second);
    if (kind == MR_MULTI_CONTACT_ARTICULATED) {
        return articulatedCoefficient(
            endpoint,
            slices,
            pointJacobians,
            environment,
            globalDof,
            localAxis,
            second
        );
    }
    if (kind == MR_MULTI_CONTACT_SCENE_BODY) {
        return sceneCoefficient(
            contact,
            endpoint,
            sceneBodies,
            sceneVelocityOffsets,
            environment,
            sceneBodyCount,
            globalDof,
            axis,
            second
        );
    }
    return kind == MR_MULTI_CONTACT_STATIC_WORLD
        ? 0.0f
        : NAN;
}

inline float3 statePointVelocity(
    const MRBodyStateGPU state,
    const float3 localPoint
) {
    const float3 offset = quaternionRotate(
        state.orientation,
        localPoint
    );
    return
        state.linearVelocityAndInverseMass.xyz +
        cross(state.angularVelocity.xyz, offset);
}

} // namespace

kernel void mr_multi_contact_assemble_jacobian(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMultiContactGPU* contacts [[buffer(1)]],
    device const MRMultiContactEndpointsGPU* endpoints
        [[buffer(2)]],
    device const MRMultiContactJacobianSliceGPU* slices
        [[buffer(3)]],
    device const float* pointJacobians [[buffer(4)]],
    device const MRBodyStateGPU* sceneBodies [[buffer(5)]],
    device const uint* sceneVelocityOffsets [[buffer(6)]],
    device float* jacobian [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]
) {
    const uint dof = index.x;
    const uint row = index.y;
    const uint environment = index.z;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.rowCount ||
        dof >= dispatch.totalNv) {
        return;
    }
    const uint contactIndex = row / 3u;
    const uint localRow = row - 3u * contactIndex;
    device const MRMultiContactGPU& contact =
        contacts[
            environment * dispatch.contactCount +
            contactIndex
        ];
    device const MRMultiContactEndpointsGPU& topology =
        endpoints[contactIndex];
    const float3 axis = contactAxis(contact, localRow);
    const float fromA = endpointCoefficient(
        contact,
        topology,
        slices,
        pointJacobians,
        sceneBodies,
        sceneVelocityOffsets,
        environment,
        dispatch.sceneBodyCount,
        dof,
        localRow,
        axis,
        false
    );
    const float fromB = endpointCoefficient(
        contact,
        topology,
        slices,
        pointJacobians,
        sceneBodies,
        sceneVelocityOffsets,
        environment,
        dispatch.sceneBodyCount,
        dof,
        localRow,
        axis,
        true
    );
    jacobian[
        (environment * dispatch.rowCount + row) *
            dispatch.totalNv +
        dof
    ] = fromB - fromA;
}

kernel void mr_multi_contact_pack_free_velocity(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* articulatedVelocity [[buffer(1)]],
    device const MRBodyStateGPU* sceneBodies [[buffer(2)]],
    device const uint* sceneVelocityOffsets [[buffer(3)]],
    device float* packedVelocity [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]
) {
    const uint dof = index.x;
    const uint environment = index.y;
    if (environment >= dispatch.environmentCount ||
        dof >= dispatch.totalNv) {
        return;
    }
    if (dof < dispatch.articulatedNv) {
        packedVelocity[
            environment * dispatch.totalNv + dof
        ] = articulatedVelocity[
            environment * dispatch.articulatedNv + dof
        ];
        return;
    }
    float value = 0.0f;
    for (uint body = 0u;
         body < dispatch.sceneBodyCount;
         ++body) {
        const uint offset = sceneVelocityOffsets[body];
        if (offset == MR_INVALID_INDEX ||
            dof < offset || dof >= offset + 6u) {
            continue;
        }
        const MRBodyStateGPU state =
            sceneBodies[
                environment * dispatch.sceneBodyCount + body
            ];
        const uint local = dof - offset;
        value = local < 3u
            ? state.linearVelocityAndInverseMass[local]
            : state.angularVelocity[local - 3u];
        break;
    }
    packedVelocity[
        environment * dispatch.totalNv + dof
    ] = value;
}

kernel void mr_multi_contact_apply_scene_response(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* jacobian [[buffer(1)]],
    device const MRBodyStateGPU* sceneBodies [[buffer(2)]],
    device const uint* sceneVelocityOffsets [[buffer(3)]],
    device float* responseColumns [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]
) {
    const uint dof = index.x;
    const uint row = index.y;
    const uint environment = index.z;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.rowCount ||
        dof < dispatch.articulatedNv ||
        dof >= dispatch.totalNv) {
        return;
    }
    for (uint body = 0u;
         body < dispatch.sceneBodyCount;
         ++body) {
        const uint offset = sceneVelocityOffsets[body];
        if (offset == MR_INVALID_INDEX ||
            dof < offset || dof >= offset + 6u) {
            continue;
        }
        const MRBodyStateGPU state =
            sceneBodies[
                environment * dispatch.sceneBodyCount + body
            ];
        const uint base =
            (environment * dispatch.rowCount + row) *
                dispatch.totalNv;
        const uint local = dof - offset;
        float value = 0.0f;
        if (local < 3u) {
            value =
                state.linearVelocityAndInverseMass.w *
                jacobian[base + dof];
        } else {
            const float3 angular = float3(
                jacobian[base + offset + 3u],
                jacobian[base + offset + 4u],
                jacobian[base + offset + 5u]
            );
            const uint axis = local - 3u;
            const float3 inertiaRow =
                axis == 0u
                ? state.inverseInertiaWorldRow0.xyz
                : (axis == 1u
                    ? state.inverseInertiaWorldRow1.xyz
                    : state.inverseInertiaWorldRow2.xyz);
            value = dot(inertiaRow, angular);
        }
        responseColumns[base + dof] = value;
        return;
    }
}

kernel void mr_multi_contact_delassus(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
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
    const uint rowBase =
        (environment * dispatch.rowCount + row) *
            dispatch.totalNv;
    const uint columnBase =
        (environment * dispatch.rowCount + column) *
            dispatch.totalNv;
    float value = 0.0f;
    for (uint dof = 0u;
         dof < dispatch.totalNv;
         ++dof) {
        value = fma(
            jacobian[rowBase + dof],
            responseColumns[columnBase + dof],
            value
        );
    }
    delassus[
        (environment * dispatch.rowCount + row) *
            dispatch.rowCount +
        column
    ] = value;
}

kernel void mr_multi_contact_free_contact_velocity(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMultiContactGPU* contacts [[buffer(1)]],
    device const MRMultiContactEndpointsGPU* endpoints
        [[buffer(2)]],
    device const float* jacobian [[buffer(3)]],
    device const float* packedVelocity [[buffer(4)]],
    device const MRBodyStateGPU* sceneBodies [[buffer(5)]],
    device const uint* sceneVelocityOffsets [[buffer(6)]],
    device float* freeContactVelocity [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]
) {
    const uint row = index.x;
    const uint environment = index.y;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.rowCount) {
        return;
    }
    const uint contactIndex = row / 3u;
    const uint localRow = row - 3u * contactIndex;
    device const MRMultiContactGPU& contact =
        contacts[
            environment * dispatch.contactCount +
            contactIndex
        ];
    device const MRMultiContactEndpointsGPU& topology =
        endpoints[contactIndex];
    const float3 axis = contactAxis(contact, localRow);
    const uint jacobianBase =
        (environment * dispatch.rowCount + row) *
            dispatch.totalNv;
    const uint velocityBase =
        environment * dispatch.totalNv;
    float value = 0.0f;
    for (uint dof = 0u;
         dof < dispatch.totalNv;
         ++dof) {
        value = fma(
            jacobian[jacobianBase + dof],
            packedVelocity[velocityBase + dof],
            value
        );
    }
    for (uint endpointIndex = 0u;
         endpointIndex < 2u;
         ++endpointIndex) {
        const bool second = endpointIndex != 0u;
        if (endpointKind(topology, second) !=
            MR_MULTI_CONTACT_SCENE_BODY) {
            continue;
        }
        const uint body = endpointBody(topology, second);
        if (body >= dispatch.sceneBodyCount ||
            sceneVelocityOffsets[body] != MR_INVALID_INDEX) {
            continue;
        }
        const MRBodyStateGPU state =
            sceneBodies[
                environment * dispatch.sceneBodyCount + body
            ];
        const float prescribed = dot(
            axis,
            statePointVelocity(
                state,
                endpointLocalPoint(contact, second)
            )
        );
        value += second ? prescribed : -prescribed;
    }
    freeContactVelocity[
        environment * dispatch.rowCount + row
    ] = value;
}

kernel void mr_multi_contact_prepare_quality_matrix(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMultiContactGPU* contacts [[buffer(1)]],
    device const float* delassus [[buffer(2)]],
    device float* qualityMatrices [[buffer(3)]],
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
    const MRMultiContactGPU rowContact =
        contacts[
            environment * dispatch.contactCount + row / 3u
        ];
    const MRMultiContactGPU columnContact =
        contacts[
            environment * dispatch.contactCount + column / 3u
        ];
    const float rowScale = row % 3u == 0u
        ? 1.0f / rowContact.friction.x
        : 1.0f;
    const float columnScale = column % 3u == 0u
        ? 1.0f / columnContact.friction.x
        : 1.0f;
    const uint matrixIndex =
        (environment * dispatch.rowCount + row) *
            dispatch.rowCount +
        column;
    float value =
        rowScale * delassus[matrixIndex] * columnScale;
    if (row == column) {
        value +=
            rowScale * rowScale *
            rowContact.regularization[row % 3u];
    }
    qualityMatrices[matrixIndex] = value;
}

kernel void mr_multi_contact_prepare_quality_vector(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMultiContactGPU* contacts [[buffer(1)]],
    device const float* freeContactVelocity [[buffer(2)]],
    device float* linear [[buffer(3)]],
    device float* warm [[buffer(4)]],
    device float* scales [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]
) {
    const uint row = index.x;
    const uint environment = index.y;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.rowCount) {
        return;
    }
    const MRMultiContactGPU contact =
        contacts[
            environment * dispatch.contactCount + row / 3u
        ];
    const uint localRow = row % 3u;
    const float scale = localRow == 0u
        ? 1.0f / contact.friction.x
        : 1.0f;
    const uint output =
        environment * dispatch.rowCount + row;
    scales[output] = scale;
    linear[output] =
        scale * (
            freeContactVelocity[output] -
            contact.targetVelocity[localRow]
        );
    warm[output] =
        contact.warmImpulse[localRow] / scale;
}

kernel void mr_multi_contact_finalize(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulatedOperatorStatusGPU* pointStatuses
        [[buffer(1)]],
    device const MRInverseMassStatusGPU* inverseStatuses
        [[buffer(2)]],
    device const MRMetalQualityStatusGPU* qualityStatuses
        [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const float* compactImpulses [[buffer(5)]],
    device const float* packedFreeVelocity [[buffer(6)]],
    device const float* responseColumns [[buffer(7)]],
    device const float* delassus [[buffer(8)]],
    device float* physicalImpulses [[buffer(9)]],
    device float* nextVelocity [[buffer(10)]],
    device MRMultiContactStatusGPU* statuses [[buffer(11)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane != 0u ||
        environment >= dispatch.environmentCount) {
        return;
    }
    MRMultiContactStatusGPU status = {};
    status.code = MR_MULTI_CONTACT_SUCCESS;
    status.environment = environment;
    status.failingContact = MR_INVALID_INDEX;
    status.failingWork = MR_INVALID_INDEX;
    status.activeContacts = dispatch.contactCount;
    for (uint articulation = 0u;
         articulation < dispatch.articulationCount;
         ++articulation) {
        const MRArticulatedOperatorStatusGPU point =
            pointStatuses[
                articulation * dispatch.environmentCount +
                environment
            ];
        if (point.code != MR_ARTICULATED_OPERATOR_SUCCESS) {
            status.code =
                MR_MULTI_CONTACT_POINT_JACOBIAN_FAILED;
            status.failingWork = articulation;
            status.pointStatusCode = point.code;
            break;
        }
    }
    for (uint work = 0u;
         status.code == MR_MULTI_CONTACT_SUCCESS &&
             work < dispatch.inverseWorkCount;
         ++work) {
        const MRInverseMassStatusGPU inverse =
            inverseStatuses[
                work * dispatch.environmentCount +
                environment
            ];
        if (inverse.code != MR_INVERSE_MASS_SUCCESS) {
            status.code =
                MR_MULTI_CONTACT_INVERSE_MASS_FAILED;
            status.failingWork = work;
            status.inverseMassCode = inverse.code;
        }
    }
    const MRMetalQualityStatusGPU quality =
        qualityStatuses[environment];
    if (status.code == MR_MULTI_CONTACT_SUCCESS &&
        quality.code != MR_METAL_QUALITY_SUCCESS) {
        status.code = MR_MULTI_CONTACT_QUALITY_FAILED;
        status.qualityCode = quality.code;
    }

    const uint velocityBase =
        environment * dispatch.totalNv;
    const uint rowBase =
        environment * dispatch.rowCount;
    float maximumImpulse = 0.0f;
    float maximumVelocityChange = 0.0f;
    for (uint row = 0u;
         row < dispatch.rowCount;
         ++row) {
        const float impulse =
            status.code == MR_MULTI_CONTACT_SUCCESS
            ? scales[rowBase + row] *
                compactImpulses[rowBase + row]
            : 0.0f;
        if (!isfinite(impulse)) {
            status.code =
                MR_MULTI_CONTACT_NONFINITE_RESULT;
        }
        physicalImpulses[rowBase + row] =
            status.code == MR_MULTI_CONTACT_SUCCESS
            ? impulse
            : 0.0f;
        maximumImpulse = max(
            maximumImpulse,
            abs(impulse)
        );
    }
    if (status.code != MR_MULTI_CONTACT_SUCCESS) {
        for (uint row = 0u;
             row < dispatch.rowCount;
             ++row) {
            physicalImpulses[rowBase + row] = 0.0f;
        }
    }
    for (uint dof = 0u;
         dof < dispatch.totalNv;
         ++dof) {
        const float freeValue =
            packedFreeVelocity[velocityBase + dof];
        float candidate = freeValue;
        if (status.code == MR_MULTI_CONTACT_SUCCESS) {
            for (uint row = 0u;
                 row < dispatch.rowCount;
                 ++row) {
                candidate = fma(
                    responseColumns[
                        (environment * dispatch.rowCount + row) *
                            dispatch.totalNv +
                        dof
                    ],
                    physicalImpulses[rowBase + row],
                    candidate
                );
            }
        }
        if (!isfinite(candidate)) {
            status.code =
                MR_MULTI_CONTACT_NONFINITE_RESULT;
            candidate = freeValue;
        }
        nextVelocity[velocityBase + dof] =
            status.code == MR_MULTI_CONTACT_SUCCESS
            ? candidate
            : freeValue;
        maximumVelocityChange = max(
            maximumVelocityChange,
            abs(candidate - freeValue)
        );
    }

    float minimumDiagonal = INFINITY;
    float maximumAsymmetry = 0.0f;
    const uint matrixBase =
        environment * dispatch.rowCount * dispatch.rowCount;
    for (uint row = 0u;
         row < dispatch.rowCount;
         ++row) {
        minimumDiagonal = min(
            minimumDiagonal,
            delassus[
                matrixBase + row * dispatch.rowCount + row
            ]
        );
        for (uint column = row + 1u;
             column < dispatch.rowCount;
             ++column) {
            maximumAsymmetry = max(
                maximumAsymmetry,
                abs(
                    delassus[
                        matrixBase +
                        row * dispatch.rowCount + column
                    ] -
                    delassus[
                        matrixBase +
                        column * dispatch.rowCount + row
                    ]
                )
            );
        }
    }
    if (!isfinite(minimumDiagonal) ||
        !isfinite(maximumAsymmetry) ||
        minimumDiagonal < -dispatch.tolerances.y ||
        maximumAsymmetry > dispatch.tolerances.x) {
        status.code =
            MR_MULTI_CONTACT_NONFINITE_RESULT;
        for (uint dof = 0u;
             dof < dispatch.totalNv;
             ++dof) {
            nextVelocity[velocityBase + dof] =
                packedFreeVelocity[velocityBase + dof];
        }
        for (uint row = 0u;
             row < dispatch.rowCount;
             ++row) {
            physicalImpulses[rowBase + row] = 0.0f;
        }
    }
    status.diagnostics = float4(
        maximumImpulse,
        maximumVelocityChange,
        minimumDiagonal,
        maximumAsymmetry
    );
    statuses[environment] = status;
}
