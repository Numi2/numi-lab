#include <metal_stdlib>

#include "metalrobo/constraint_ir_shared.h"
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
    const float3 axis,
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
    const uint base =
        slice.jacobianOffset +
        environment * slice.jacobianEnvironmentStride +
        query * 3u * slice.nv +
        (globalDof - slice.vOffset);
    return
        axis.x * pointJacobians[base] +
        axis.y * pointJacobians[base + slice.nv] +
        axis.z * pointJacobians[base + 2u * slice.nv];
}

inline float articulatedAngularCoefficient(
    device const MRMultiContactEndpointsGPU& endpoint,
    device const MRMultiContactJacobianSliceGPU* slices,
    device const float* pointJacobians,
    device const MRArticulatedPointWorldGPU* pointWorld,
    const uint environment,
    const uint globalDof,
    const float3 axis,
    const bool second
) {
    const uint sliceIndex = endpointSlice(endpoint, second);
    if (sliceIndex == MR_INVALID_INDEX) {
        return 0.0f;
    }
    device const MRMultiContactJacobianSliceGPU& slice =
        slices[sliceIndex];
    if (globalDof < slice.vOffset ||
        globalDof >= slice.vOffset + slice.nv) {
        return 0.0f;
    }
    const uint query = endpointQuery(endpoint, second);
    if (query == MR_INVALID_INDEX ||
        query + 3u >= slice.queryCount) {
        return NAN;
    }
    const uint localDof = globalDof - slice.vOffset;
    const uint jacobianEnvironmentBase =
        slice.jacobianOffset +
        environment * slice.jacobianEnvironmentStride;
    const uint pointEnvironmentBase =
        slice.pointWorldOffset +
        environment * slice.queryCount;
    const float3 basePoint =
        pointWorld[pointEnvironmentBase + query]
            .position.xyz;
    const uint baseJacobian =
        jacobianEnvironmentBase +
        query * 3u * slice.nv + localDof;
    const float3 baseLinear{
        pointJacobians[baseJacobian],
        pointJacobians[baseJacobian + slice.nv],
        pointJacobians[baseJacobian + 2u * slice.nv],
    };
    float3 angular = float3(0.0f);
    for (uint basis = 0u; basis < 3u; ++basis) {
        const uint offsetQuery = query + basis + 1u;
        const float3 offset =
            pointWorld[
                pointEnvironmentBase + offsetQuery
            ].position.xyz - basePoint;
        const uint offsetJacobian =
            jacobianEnvironmentBase +
            offsetQuery * 3u * slice.nv + localDof;
        const float3 delta{
            pointJacobians[offsetJacobian],
            pointJacobians[
                offsetJacobian + slice.nv
            ],
            pointJacobians[
                offsetJacobian + 2u * slice.nv
            ],
        };
        angular += 0.5f * cross(
            offset,
            delta - baseLinear
        );
    }
    return dot(axis, angular);
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

inline float sceneAngularCoefficient(
    device const MRMultiContactEndpointsGPU& endpoint,
    device const uint* sceneVelocityOffsets,
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
    const uint localDof = globalDof - offset;
    return localDof < 3u
        ? 0.0f
        : axis[localDof - 3u];
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
            axis,
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

inline bool evaluateEqualityRow(
    const MRMultiContactDispatchGPU dispatch,
    const MRConstraintIRRowGPU source,
    const float relative,
    thread float& target,
    thread float& regularization
) {
    if (!all(isfinite(source.direction)) ||
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
        source.impulseLower >
            -0.5f * MR_CONSTRAINT_IR_UNBOUNDED ||
        source.impulseUpper <
            0.5f * MR_CONSTRAINT_IR_UNBOUNDED ||
        (source.flags &
         MR_CONSTRAINT_IR_ROW_UNILATERAL) != 0u) {
        return false;
    }
    float stabilization = 0.0f;
    if ((source.flags &
         MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED) != 0u) {
        const float timestep =
            dispatch.equalityEvaluation0.x;
        const float tau = max(
            source.timeConstant,
            dispatch.equalityEvaluation0.z * timestep
        );
        if (!(tau > 0.0f) || !isfinite(tau)) {
            return false;
        }
        const float ratio = timestep / tau;
        const float denominator =
            1.0f +
            2.0f * source.dampingRatio * ratio +
            ratio * ratio;
        stabilization =
            (
                relative - source.targetVelocity -
                timestep * source.positionError / (tau * tau)
            ) / denominator;
        stabilization = clamp(
            stabilization,
            -dispatch.equalityEvaluation0.y,
            dispatch.equalityEvaluation0.y
        );
    }
    target = source.targetVelocity + stabilization;
    regularization = max(
        source.compliance /
            (
                dispatch.equalityEvaluation0.x *
                dispatch.equalityEvaluation0.x
            ) +
            source.dissipation /
                dispatch.equalityEvaluation0.x,
        dispatch.equalityEvaluation0.w
    );
    return isfinite(target) &&
        isfinite(regularization) &&
        regularization >= 0.0f;
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
        axis,
        true
    );
    jacobian[
        (environment * dispatch.responseRowCount + row) *
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

kernel void mr_multi_contact_scatter_equality_jacobian(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* equalityJacobian [[buffer(1)]],
    device float* combinedJacobian [[buffer(2)]],
    uint3 index [[thread_position_in_grid]]
) {
    const uint dof = index.x;
    const uint row = index.y;
    const uint environment = index.z;
    if (environment >= dispatch.environmentCount ||
        row >= dispatch.staticEqualityRowCount ||
        dof >= dispatch.totalNv) {
        return;
    }
    const uint combinedRow = dispatch.rowCount + row;
    combinedJacobian[
        (environment * dispatch.responseRowCount +
         combinedRow) * dispatch.totalNv + dof
    ] = dof < dispatch.articulatedNv
        ? equalityJacobian[
              row * dispatch.articulatedNv + dof
          ]
        : 0.0f;
}

kernel void mr_multi_contact_assemble_point_equalities(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMultiContactGPU* equalities [[buffer(1)]],
    device const MRMultiContactEndpointsGPU* endpoints
        [[buffer(2)]],
    device const MRMultiContactJacobianSliceGPU* slices
        [[buffer(3)]],
    device const float* pointJacobians [[buffer(4)]],
    device const MRArticulatedPointWorldGPU* pointWorld
        [[buffer(5)]],
    device const uint* equalityKinds [[buffer(6)]],
    device const MRBodyStateGPU* sceneBodies [[buffer(7)]],
    device const uint* sceneVelocityOffsets [[buffer(8)]],
    device float* combinedJacobian [[buffer(9)]],
    uint3 index [[thread_position_in_grid]]
) {
    const uint dof = index.x;
    const uint pointRow = index.y;
    const uint environment = index.z;
    const uint dynamicRows =
        dispatch.equalityRowCount -
        dispatch.staticEqualityRowCount;
    if (environment >= dispatch.environmentCount ||
        pointRow >= dynamicRows ||
        dof >= dispatch.totalNv) {
        return;
    }
    const uint equality = pointRow / 3u;
    const uint localRow = pointRow % 3u;
    const uint equalityCount = dynamicRows / 3u;
    device const MRMultiContactGPU& geometry =
        equalities[
            environment * equalityCount + equality
        ];
    device const MRMultiContactEndpointsGPU& topology =
        endpoints[equality];
    const uint equalityKind = equalityKinds[equality];
    const float3 axis = contactAxis(geometry, localRow);
    float fromA = 0.0f;
    float fromB = 0.0f;
    if (topology.kindA == MR_MULTI_CONTACT_ARTICULATED) {
        fromA =
            equalityKind == MR_MULTI_EQUALITY_ANGULAR
            ? articulatedAngularCoefficient(
                  topology,
                  slices,
                  pointJacobians,
                  pointWorld,
                  environment,
                  dof,
                  axis,
                  false
              )
            : articulatedCoefficient(
                  topology,
                  slices,
                  pointJacobians,
                  environment,
                  dof,
                  axis,
                  false
              );
    } else if (topology.kindA ==
               MR_MULTI_CONTACT_SCENE_BODY) {
        fromA =
            equalityKind == MR_MULTI_EQUALITY_ANGULAR
            ? sceneAngularCoefficient(
                  topology,
                  sceneVelocityOffsets,
                  dispatch.sceneBodyCount,
                  dof,
                  axis,
                  false
              )
            : sceneCoefficient(
                  geometry,
                  topology,
                  sceneBodies,
                  sceneVelocityOffsets,
                  environment,
                  dispatch.sceneBodyCount,
                  dof,
                  axis,
                  false
              );
    } else if (topology.kindA !=
               MR_MULTI_CONTACT_STATIC_WORLD) {
        fromA = NAN;
    }
    if (topology.kindB == MR_MULTI_CONTACT_ARTICULATED) {
        fromB =
            equalityKind == MR_MULTI_EQUALITY_ANGULAR
            ? articulatedAngularCoefficient(
                  topology,
                  slices,
                  pointJacobians,
                  pointWorld,
                  environment,
                  dof,
                  axis,
                  true
              )
            : articulatedCoefficient(
                  topology,
                  slices,
                  pointJacobians,
                  environment,
                  dof,
                  axis,
                  true
              );
    } else if (topology.kindB ==
               MR_MULTI_CONTACT_SCENE_BODY) {
        fromB =
            equalityKind == MR_MULTI_EQUALITY_ANGULAR
            ? sceneAngularCoefficient(
                  topology,
                  sceneVelocityOffsets,
                  dispatch.sceneBodyCount,
                  dof,
                  axis,
                  true
              )
            : sceneCoefficient(
                  geometry,
                  topology,
                  sceneBodies,
                  sceneVelocityOffsets,
                  environment,
                  dispatch.sceneBodyCount,
                  dof,
                  axis,
                  true
              );
    } else if (topology.kindB !=
               MR_MULTI_CONTACT_STATIC_WORLD) {
        fromB = NAN;
    }
    const uint combinedRow =
        dispatch.rowCount +
        dispatch.staticEqualityRowCount +
        pointRow;
    combinedJacobian[
        (environment * dispatch.responseRowCount +
         combinedRow) * dispatch.totalNv + dof
    ] = fromB - fromA;
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
        row >= dispatch.responseRowCount ||
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
            (environment * dispatch.responseRowCount + row) *
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

// Exact bilateral reduction for the small generalized-equality frontier.
// Contact and equality rows share the same inverse-ABA response stream. One
// SIMD32 group owns an environment; lane zero performs deterministic
// Cholesky of G M^-1 G' + R while all lanes project velocity/response payloads.
kernel void mr_multi_contact_project_equalities(
    device const MRMultiContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRConstraintIRRowGPU* sourceRows [[buffer(1)]],
    device const float* combinedJacobian [[buffer(2)]],
    device const float* combinedResponse [[buffer(3)]],
    device const MRInverseMassStatusGPU* inverseStatuses
        [[buffer(4)]],
    device const float* packedFreeVelocity [[buffer(5)]],
    device float* projectedResponse [[buffer(6)]],
    device float* projectedFreeVelocity [[buffer(7)]],
    device float* equalityOperator [[buffer(8)]],
    device float* equalityCoupling [[buffer(9)]],
    device float* equalityFreeImpulses [[buffer(10)]],
    device float* equalityTargets [[buffer(11)]],
    device float* equalityRegularization [[buffer(12)]],
    device MRMultiContactEqualityStatusGPU* statuses
        [[buffer(13)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    threadgroup float factor[
        MR_MULTI_CONTACT_MAX_EQUALITY_ROWS *
        MR_MULTI_CONTACT_MAX_EQUALITY_ROWS
    ];
    threadgroup float right[
        MR_MULTI_CONTACT_MAX_EQUALITY_ROWS
    ];
    threadgroup float solution[
        MR_MULTI_CONTACT_MAX_EQUALITY_ROWS
    ];
    threadgroup float equilibration[
        MR_MULTI_CONTACT_MAX_EQUALITY_ROWS
    ];
    threadgroup atomic_uint sharedStatusCode;
    threadgroup uint failingRow;
    threadgroup float minimumPivot;

    const uint equalityRows = dispatch.equalityRowCount;
    const uint contactRows = dispatch.rowCount;
    const uint totalNv = dispatch.totalNv;
    const uint responseRows = dispatch.responseRowCount;
    const uint velocityBase = environment * totalNv;
    const uint combinedBase =
        environment * responseRows * totalNv;
    const uint projectedBase =
        environment * contactRows * totalNv;
    const uint operatorBase =
        environment * equalityRows * equalityRows;
    const uint couplingBase =
        environment * equalityRows * contactRows;
    const uint equalityBase = environment * equalityRows;

    if (lane == 0u) {
        uint statusCode =
            MR_MULTI_CONTACT_EQUALITY_SUCCESS;
        failingRow = MR_INVALID_INDEX;
        minimumPivot = INFINITY;
        if (dispatch.abiVersion !=
                MR_MULTI_CONTACT_ABI_VERSION ||
            equalityRows >
                MR_MULTI_CONTACT_MAX_EQUALITY_ROWS ||
            dispatch.staticEqualityRowCount >
                equalityRows ||
            (
                equalityRows -
                    dispatch.staticEqualityRowCount
            ) % 3u != 0u ||
            responseRows != contactRows + equalityRows ||
            !(dispatch.equalityEvaluation0.x > 0.0f) ||
            dispatch.equalityEvaluation0.y < 0.0f ||
            dispatch.equalityEvaluation0.z < 0.0f ||
            dispatch.equalityEvaluation0.w < 0.0f ||
            !(dispatch.equalityEvaluation1.x > 0.0f) ||
            !(dispatch.equalityEvaluation1.y > 0.0f) ||
            !all(isfinite(dispatch.equalityEvaluation0)) ||
            !all(isfinite(dispatch.equalityEvaluation1))) {
            statusCode =
                MR_MULTI_CONTACT_EQUALITY_INVALID_DISPATCH;
        }
        for (uint work = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 work < dispatch.inverseWorkCount;
             ++work) {
            const MRInverseMassStatusGPU inverse =
                inverseStatuses[
                    work * dispatch.environmentCount +
                    environment
                ];
            if (inverse.code != MR_INVERSE_MASS_SUCCESS) {
                statusCode =
                    MR_MULTI_CONTACT_EQUALITY_INVERSE_MASS_FAILED;
                failingRow = work;
            }
        }

        for (uint row = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 row < equalityRows;
             ++row) {
            const uint jacobianBase =
                combinedBase +
                (contactRows + row) * totalNv;
            float relative = 0.0f;
            for (uint dof = 0u; dof < totalNv; ++dof) {
                relative = fma(
                    combinedJacobian[jacobianBase + dof],
                    packedFreeVelocity[velocityBase + dof],
                    relative
                );
            }
            float target = 0.0f;
            float regularization = 0.0f;
            if (!isfinite(relative) ||
                !evaluateEqualityRow(
                    dispatch,
                    sourceRows[
                        environment * equalityRows + row
                    ],
                    relative,
                    target,
                    regularization
                )) {
                statusCode =
                    MR_MULTI_CONTACT_EQUALITY_INVALID_ROW;
                failingRow = row;
                break;
            }
            equalityTargets[equalityBase + row] = target;
            equalityRegularization[equalityBase + row] =
                regularization;
            right[row] = target - relative;
            for (uint column = 0u;
                 column < equalityRows;
                 ++column) {
                const uint responseBase =
                    combinedBase +
                    (contactRows + column) * totalNv;
                float value = 0.0f;
                for (uint dof = 0u;
                     dof < totalNv;
                     ++dof) {
                    value = fma(
                        combinedJacobian[
                            jacobianBase + dof
                        ],
                        combinedResponse[
                            responseBase + dof
                        ],
                        value
                    );
                }
                factor[row * equalityRows + column] = value;
            }
        }
        for (uint row = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 row < equalityRows;
             ++row) {
            for (uint column = row + 1u;
                 column < equalityRows;
                 ++column) {
                const float symmetric = 0.5f * (
                    factor[row * equalityRows + column] +
                    factor[column * equalityRows + row]
                );
                factor[row * equalityRows + column] =
                    symmetric;
                factor[column * equalityRows + row] =
                    symmetric;
            }
            factor[row * equalityRows + row] +=
                equalityRegularization[equalityBase + row];
        }
        for (uint index = 0u;
             index < equalityRows * equalityRows;
             ++index) {
            equalityOperator[operatorBase + index] =
                factor[index];
        }
        // Equality rows mix translational, angular, and generalized units.
        // A single dimensional maximum diagonal can therefore reject a
        // perfectly valid small-unit pivot merely because another row is
        // expressed in inverse radians. Symmetric Jacobi equilibration makes
        // the Cholesky acceptance test dimensionless while preserving the
        // physical operator published above. For S = diag(1/sqrt(A_ii)),
        // solve (S A S)y = S b and recover the physical impulse x = S y.
        for (uint row = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 row < equalityRows;
             ++row) {
            const float diagonal =
                factor[row * equalityRows + row];
            if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
                statusCode =
                    MR_MULTI_CONTACT_EQUALITY_FACTORIZATION_FAILED;
                failingRow = row;
                break;
            }
            equilibration[row] = rsqrt(diagonal);
        }
        for (uint row = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 row < equalityRows;
             ++row) {
            right[row] *= equilibration[row];
            for (uint column = 0u;
                 column < equalityRows;
                 ++column) {
                factor[row * equalityRows + column] *=
                    equilibration[row] * equilibration[column];
            }
        }
        const float pivotFloor = dispatch.equalityEvaluation1.x;
        for (uint row = 0u;
             statusCode ==
                     MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                 row < equalityRows;
             ++row) {
            for (uint column = 0u;
                 column <= row;
                 ++column) {
                float value =
                    factor[row * equalityRows + column];
                for (uint inner = 0u;
                     inner < column;
                     ++inner) {
                    value = fma(
                        -factor[
                            row * equalityRows + inner
                        ],
                        factor[
                            column * equalityRows + inner
                        ],
                        value
                    );
                }
                if (row == column) {
                    if (!(value > pivotFloor) ||
                        !isfinite(value)) {
                        statusCode =
                            MR_MULTI_CONTACT_EQUALITY_FACTORIZATION_FAILED;
                        failingRow = row;
                        break;
                    }
                    const float pivot = sqrt(value);
                    factor[row * equalityRows + column] =
                        pivot;
                    minimumPivot = min(minimumPivot, pivot);
                } else {
                    value /=
                        factor[
                            column * equalityRows + column
                        ];
                    if (!isfinite(value)) {
                        statusCode =
                            MR_MULTI_CONTACT_EQUALITY_NONFINITE_RESULT;
                        failingRow = row;
                        break;
                    }
                    factor[row * equalityRows + column] =
                        value;
                }
            }
        }
        if (statusCode ==
                MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
            equalityRows != 0u) {
            for (uint row = 0u;
                 row < equalityRows;
                 ++row) {
                float value = right[row];
                for (uint column = 0u;
                     column < row;
                     ++column) {
                    value = fma(
                        -factor[
                            row * equalityRows + column
                        ],
                        solution[column],
                        value
                    );
                }
                solution[row] =
                    value /
                    factor[row * equalityRows + row];
            }
            for (uint reverse = 0u;
                 reverse < equalityRows;
                 ++reverse) {
                const uint row =
                    equalityRows - 1u - reverse;
                float value = solution[row];
                for (uint column = row + 1u;
                     column < equalityRows;
                     ++column) {
                    value = fma(
                        -factor[
                            column * equalityRows + row
                        ],
                        right[column],
                        value
                    );
                }
                right[row] =
                    value /
                    factor[row * equalityRows + row];
            }
            for (uint row = 0u;
                 row < equalityRows;
                 ++row) {
                equalityFreeImpulses[
                    equalityBase + row
                ] = equilibration[row] * right[row];
            }

            for (uint contactRow = 0u;
                 contactRow < contactRows;
                 ++contactRow) {
                for (uint equalityRow = 0u;
                     equalityRow < equalityRows;
                     ++equalityRow) {
                    const uint jacobianBase =
                        combinedBase +
                        (contactRows + equalityRow) *
                            totalNv;
                    const uint responseBase =
                        combinedBase +
                        contactRow * totalNv;
                    float value = 0.0f;
                    for (uint dof = 0u;
                         dof < totalNv;
                         ++dof) {
                        value = fma(
                            combinedJacobian[
                                jacobianBase + dof
                            ],
                            combinedResponse[
                                responseBase + dof
                            ],
                            value
                        );
                    }
                    solution[equalityRow] =
                        equilibration[equalityRow] * value;
                }
                for (uint row = 0u;
                     row < equalityRows;
                     ++row) {
                    float value = solution[row];
                    for (uint column = 0u;
                         column < row;
                         ++column) {
                        value = fma(
                            -factor[
                                row * equalityRows + column
                            ],
                            right[column],
                            value
                        );
                    }
                    right[row] =
                        value /
                        factor[row * equalityRows + row];
                }
                for (uint reverse = 0u;
                     reverse < equalityRows;
                     ++reverse) {
                    const uint row =
                        equalityRows - 1u - reverse;
                    float value = right[row];
                    for (uint column = row + 1u;
                         column < equalityRows;
                         ++column) {
                        value = fma(
                            -factor[
                                column * equalityRows + row
                            ],
                            solution[column],
                            value
                        );
                    }
                    solution[row] =
                        value /
                        factor[row * equalityRows + row];
                }
                for (uint row = 0u;
                     row < equalityRows;
                     ++row) {
                    equalityCoupling[
                        couplingBase +
                        row * contactRows + contactRow
                    ] = equilibration[row] * solution[row];
                }
            }
        }
        atomic_store_explicit(
            &sharedStatusCode,
            statusCode,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint resolvedStatusCode =
        atomic_load_explicit(
            &sharedStatusCode,
            memory_order_relaxed
        );
    const bool succeeded =
        resolvedStatusCode ==
            MR_MULTI_CONTACT_EQUALITY_SUCCESS;
    for (uint dof = lane;
         dof < totalNv;
         dof += 32u) {
        float value =
            packedFreeVelocity[velocityBase + dof];
        if (succeeded) {
            for (uint row = 0u;
                 row < equalityRows;
                 ++row) {
                value = fma(
                    combinedResponse[
                        combinedBase +
                        (contactRows + row) * totalNv +
                        dof
                    ],
                    equalityFreeImpulses[
                        equalityBase + row
                    ],
                    value
                );
            }
        }
        projectedFreeVelocity[velocityBase + dof] =
            isfinite(value)
            ? value
            : packedFreeVelocity[velocityBase + dof];
    }
    for (uint linear = lane;
         linear < contactRows * totalNv;
         linear += 32u) {
        const uint contactRow = linear / totalNv;
        const uint dof = linear - contactRow * totalNv;
        float value =
            combinedResponse[
                combinedBase + contactRow * totalNv + dof
            ];
        if (succeeded) {
            for (uint row = 0u;
                 row < equalityRows;
                 ++row) {
                value = fma(
                    -combinedResponse[
                        combinedBase +
                        (contactRows + row) * totalNv +
                        dof
                    ],
                    equalityCoupling[
                        couplingBase +
                        row * contactRows + contactRow
                    ],
                    value
                );
            }
        }
        projectedResponse[projectedBase + linear] =
            isfinite(value) ? value : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);

    if (lane == 0u) {
        MRMultiContactEqualityStatusGPU status = {};
        status.code = resolvedStatusCode;
        status.environment = environment;
        status.failingRow = failingRow;
        status.rowCount = equalityRows;
        status.diagnostics.y =
            equalityRows == 0u ? 0.0f : minimumPivot;
        statuses[environment] = status;
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
        (environment * dispatch.responseRowCount + row) *
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
        (environment * dispatch.responseRowCount + row) *
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
    device const float* projectedFreeVelocity [[buffer(12)]],
    device const float* combinedJacobian [[buffer(13)]],
    device const float* equalityTargets [[buffer(14)]],
    device const float* equalityRegularization [[buffer(15)]],
    device const float* equalityFreeImpulses [[buffer(16)]],
    device const float* equalityCoupling [[buffer(17)]],
    device float* equalityImpulses [[buffer(18)]],
    device MRMultiContactEqualityStatusGPU* equalityStatuses
        [[buffer(19)]],
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
    MRMultiContactEqualityStatusGPU equality =
        equalityStatuses[environment];
    if (status.code == MR_MULTI_CONTACT_SUCCESS &&
        equality.code !=
            MR_MULTI_CONTACT_EQUALITY_SUCCESS) {
        status.code = MR_MULTI_CONTACT_EQUALITY_FAILED;
        status.failingWork = equality.failingRow;
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
    const uint equalityBase =
        environment * dispatch.equalityRowCount;
    const uint couplingBase =
        environment * dispatch.equalityRowCount *
            dispatch.rowCount;
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
        for (uint row = 0u;
             row < dispatch.equalityRowCount;
             ++row) {
            equalityImpulses[equalityBase + row] = 0.0f;
        }
    }
    for (uint dof = 0u;
         dof < dispatch.totalNv;
         ++dof) {
        const float publishedInput =
            packedFreeVelocity[velocityBase + dof];
        const float freeValue =
            projectedFreeVelocity[velocityBase + dof];
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
            candidate = publishedInput;
        }
        nextVelocity[velocityBase + dof] =
            status.code == MR_MULTI_CONTACT_SUCCESS
            ? candidate
            : publishedInput;
        maximumVelocityChange = max(
            maximumVelocityChange,
            abs(candidate - publishedInput)
        );
    }

    float maximumEqualityResidual = 0.0f;
    float maximumEqualityImpulse = 0.0f;
    float maximumNullSpaceLeakage = 0.0f;
    for (uint row = 0u;
         row < dispatch.equalityRowCount;
         ++row) {
        float impulse =
            equalityFreeImpulses[equalityBase + row];
        if (status.code == MR_MULTI_CONTACT_SUCCESS) {
            for (uint contactRow = 0u;
                 contactRow < dispatch.rowCount;
                 ++contactRow) {
                impulse = fma(
                    -equalityCoupling[
                        couplingBase +
                        row * dispatch.rowCount +
                        contactRow
                    ],
                    physicalImpulses[rowBase + contactRow],
                    impulse
                );
            }
        } else {
            impulse = 0.0f;
        }
        equalityImpulses[equalityBase + row] = impulse;
        maximumEqualityImpulse = max(
            maximumEqualityImpulse,
            abs(impulse)
        );
        float residual =
            -equalityTargets[equalityBase + row] +
            equalityRegularization[equalityBase + row] *
                impulse;
        const uint jacobianBase =
            (
                environment * dispatch.responseRowCount +
                dispatch.rowCount + row
            ) * dispatch.totalNv;
        for (uint dof = 0u;
             dof < dispatch.totalNv;
             ++dof) {
            residual = fma(
                combinedJacobian[jacobianBase + dof],
                nextVelocity[velocityBase + dof],
                residual
            );
        }
        maximumEqualityResidual = max(
            maximumEqualityResidual,
            abs(residual)
        );
        for (uint contactRow = 0u;
             contactRow < dispatch.rowCount;
             ++contactRow) {
            float leakage = 0.0f;
            for (uint dof = 0u;
                 dof < dispatch.totalNv;
                 ++dof) {
                leakage = fma(
                    combinedJacobian[
                        jacobianBase + dof
                    ],
                    responseColumns[
                        (environment * dispatch.rowCount +
                         contactRow) * dispatch.totalNv + dof
                    ],
                    leakage
                );
            }
            leakage +=
                equalityRegularization[
                    equalityBase + row
                ] *
                equalityCoupling[
                    couplingBase +
                    row * dispatch.rowCount + contactRow
                ];
            maximumNullSpaceLeakage = max(
                maximumNullSpaceLeakage,
                abs(leakage)
            );
        }
    }
    if (status.code == MR_MULTI_CONTACT_SUCCESS &&
        (
            !isfinite(maximumEqualityResidual) ||
            !isfinite(maximumEqualityImpulse) ||
            !isfinite(maximumNullSpaceLeakage) ||
            maximumEqualityResidual >
                dispatch.equalityEvaluation1.y ||
            maximumNullSpaceLeakage >
                dispatch.equalityEvaluation1.y
        )) {
        status.code = MR_MULTI_CONTACT_EQUALITY_FAILED;
        equality.code =
            MR_MULTI_CONTACT_EQUALITY_RESIDUAL_FAILED;
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
        for (uint row = 0u;
             row < dispatch.equalityRowCount;
             ++row) {
            equalityImpulses[equalityBase + row] = 0.0f;
        }
    }
    equality.code =
        status.code == MR_MULTI_CONTACT_EQUALITY_FAILED
        ? equality.code
        : MR_MULTI_CONTACT_EQUALITY_SUCCESS;
    equality.diagnostics.x = maximumEqualityResidual;
    equality.diagnostics.z = maximumEqualityImpulse;
    equality.diagnostics.w = maximumNullSpaceLeakage;
    equalityStatuses[environment] = equality;

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
