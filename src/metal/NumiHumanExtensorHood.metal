#include <metal_stdlib>

#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_extensor_hood_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"

using namespace metal;

namespace {

constant uint kMaxNodes = MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_NODES;
constant uint kMaxElements = MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_ELEMENTS;
constant uint kMaxDimension = MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_FREE_DIMENSION;

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

inline bool validBody(
    const uint bodyIndex,
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch
) {
    return bodyIndex >= dispatch.articulationFirstBody &&
        bodyIndex - dispatch.articulationFirstBody < dispatch.bodyPoseStride;
}

inline float3 worldPoint(
    const uint environment,
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    const uint bodyIndex,
    const float3 localPoint
) {
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride +
        bodyIndex - dispatch.articulationFirstBody];
    return pose.position.xyz + quaternionRotate(pose.orientation, localPoint);
}

inline float3 angularJacobian(
    const uint environment,
    const uint dof,
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float* pointJacobians,
    const uint bodyIndex
) {
    const uint localBody = bodyIndex - dispatch.articulationFirstBody;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody];
    const uint bodyPoint = dispatch.bodyJacobianPointOffset + 4u * localBody;
    const uint centerBase = environment * dispatch.pointJacobianStride +
        bodyPoint * 3u * dispatch.dofCount;
    const float3 center = float3(
        pointJacobians[centerBase + dof],
        pointJacobians[centerBase + dispatch.dofCount + dof],
        pointJacobians[centerBase + 2u * dispatch.dofCount + dof]);
    float3 angular = float3(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 localAxis = axis == 0u
            ? float3(1.0f, 0.0f, 0.0f)
            : (axis == 1u ? float3(0.0f, 1.0f, 0.0f)
                          : float3(0.0f, 0.0f, 1.0f));
        const float3 worldAxis = quaternionRotate(pose.orientation, localAxis);
        const uint axisBase = centerBase +
            (axis + 1u) * 3u * dispatch.dofCount;
        const float3 probe = float3(
            pointJacobians[axisBase + dof],
            pointJacobians[axisBase + dispatch.dofCount + dof],
            pointJacobians[axisBase + 2u * dispatch.dofCount + dof]);
        angular += 0.5f * cross(worldAxis, probe - center);
    }
    return angular;
}

inline float3 pointJacobian(
    const uint environment,
    const uint dof,
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float* pointJacobians,
    const uint bodyIndex,
    const float3 point
) {
    const uint localBody = bodyIndex - dispatch.articulationFirstBody;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody];
    const uint bodyPoint = dispatch.bodyJacobianPointOffset + 4u * localBody;
    const uint centerBase = environment * dispatch.pointJacobianStride +
        bodyPoint * 3u * dispatch.dofCount;
    const float3 center = float3(
        pointJacobians[centerBase + dof],
        pointJacobians[centerBase + dispatch.dofCount + dof],
        pointJacobians[centerBase + 2u * dispatch.dofCount + dof]);
    return center + cross(
        angularJacobian(environment, dof, dispatch, bodyPoses,
                        pointJacobians, bodyIndex),
        point - pose.position.xyz);
}

inline bool evaluate(
    const MRNumiHumanExtensorHoodRayGPU ray,
    device const MRNumiHumanExtensorHoodNodeGPU* nodes,
    device const MRNumiHumanExtensorHoodElementGPU* elements,
    thread const float3* position,
    thread const float3* initialPosition,
    thread const float3* appliedLoad,
    const float foundationStiffness,
    thread float3* residual,
    thread float* tension,
    thread float* stiffness,
    thread float& potential,
    thread float& strainEnergy,
    thread float& maximumResidual,
    thread uint& activeCount
) {
    potential = 0.0f;
    strainEnergy = 0.0f;
    maximumResidual = 0.0f;
    activeCount = 0u;
    for (uint node = 0u; node < kMaxNodes; ++node)
        residual[node] = float3(0.0f);
    for (uint element = 0u; element < kMaxElements; ++element) {
        tension[element] = 0.0f;
        stiffness[element] = 0.0f;
    }
    for (uint local = 0u; local < ray.elements.y; ++local) {
        const MRNumiHumanExtensorHoodElementGPU element =
            elements[ray.elements.x + local];
        const uint a = element.nodeA - ray.nodes.x;
        const uint b = element.nodeB - ray.nodes.x;
        if (a >= ray.nodes.y || b >= ray.nodes.y || a == b ||
            !finite4(element.material) || element.material.w != 0.0f ||
            !(element.material.x > 0.0f) || !(element.material.y > 0.0f) ||
            !(element.material.z > 0.0f)) return false;
        const float3 delta = position[b] - position[a];
        const float lengthValue = length(delta);
        if (!(lengthValue > 0.0f) || !isfinite(lengthValue)) return false;
        const float extension = lengthValue - element.material.x;
        if (!(extension > 0.0f)) continue;
        const float axial = element.material.y * element.material.z /
            element.material.x;
        const float forceMagnitude = axial * extension;
        const float3 force = forceMagnitude * delta / lengthValue;
        if (!all(isfinite(force)) || !isfinite(axial)) return false;
        residual[a] += force;
        residual[b] -= force;
        tension[local] = forceMagnitude;
        stiffness[local] = axial;
        strainEnergy += 0.5f * axial * extension * extension;
        ++activeCount;
    }
    potential = strainEnergy;
    for (uint local = 0u; local < ray.nodes.y; ++local) {
        residual[local] += appliedLoad[local];
        // Translation by initialPosition leaves the energy gradient unchanged
        // and avoids cancellation from metre-scale world coordinates.
        potential -= dot(
            appliedLoad[local], position[local] - initialPosition[local]);
        const MRNumiHumanExtensorHoodNodeGPU node = nodes[ray.nodes.x + local];
        if ((node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) == 0u) {
            const float3 displacement = position[local] - initialPosition[local];
            residual[local] -= foundationStiffness * displacement;
            potential += 0.5f * foundationStiffness *
                dot(displacement, displacement);
            maximumResidual = max(maximumResidual, length(residual[local]));
        }
    }
    return isfinite(potential) && isfinite(strainEnergy) &&
        isfinite(maximumResidual);
}

inline bool choleskySolve(
    thread float* matrix,
    thread float* rhs,
    thread float* solution,
    const uint dimension
) {
    float forward[kMaxDimension];
    for (uint row = 0u; row < dimension; ++row) {
        for (uint column = 0u; column <= row; ++column) {
            float value = matrix[row * kMaxDimension + column];
            for (uint inner = 0u; inner < column; ++inner) {
                value -= matrix[row * kMaxDimension + inner] *
                    matrix[column * kMaxDimension + inner];
            }
            if (row == column) {
                if (!(value > 0.0f) || !isfinite(value)) return false;
                matrix[row * kMaxDimension + column] = sqrt(value);
            } else {
                matrix[row * kMaxDimension + column] = value /
                    matrix[column * kMaxDimension + column];
            }
        }
    }
    for (uint row = 0u; row < dimension; ++row) {
        float value = rhs[row];
        for (uint column = 0u; column < row; ++column)
            value -= matrix[row * kMaxDimension + column] * forward[column];
        forward[row] = value / matrix[row * kMaxDimension + row];
    }
    for (uint index = 0u; index < dimension; ++index) solution[index] = 0.0f;
    for (uint reverse = 0u; reverse < dimension; ++reverse) {
        const uint row = dimension - 1u - reverse;
        float value = forward[row];
        for (uint column = row + 1u; column < dimension; ++column)
            value -= matrix[column * kMaxDimension + row] * solution[column];
        solution[row] = value / matrix[row * kMaxDimension + row];
        if (!isfinite(solution[row])) return false;
    }
    return true;
}

} // namespace

kernel void mr_numi_human_solve_extensor_hood(
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanExtensorHoodRayGPU* rays [[buffer(1)]],
    device const MRNumiHumanExtensorHoodNodeGPU* nodes [[buffer(2)]],
    device const MRNumiHumanExtensorHoodElementGPU* elements [[buffer(3)]],
    device const MRNumiHumanExtensorHoodInputGPU* inputs [[buffer(4)]],
    device const MRMujocoMuscleGPU* muscles [[buffer(5)]],
    device const MRMujocoMuscleSiteGPU* sites [[buffer(6)]],
    device const MRMujocoMuscleRouteNodeGPU* routeNodes [[buffer(7)]],
    device const MRMujocoMuscleResultGPU* muscleResults [[buffer(8)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(9)]],
    device MRNumiHumanExtensorHoodNodeResultGPU* nodeResults [[buffer(10)]],
    device MRNumiHumanExtensorHoodRayResultGPU* rayResults [[buffer(11)]],
    device const MRMujocoMuscleWrapGPU* wraps [[buffer(12)]],
    uint global [[thread_position_in_grid]]
) {
    if (dispatch.rayCount == 0u ||
        global >= dispatch.environmentCount * dispatch.rayCount) return;
    const uint environment = global / dispatch.rayCount;
    const uint rayIndex = global - environment * dispatch.rayCount;
    MRNumiHumanExtensorHoodRayResultGPU result{};
    result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_DISPATCH;
    result.environment = environment;
    result.rayIndex = rayIndex;
    rayResults[global] = result;
    if (dispatch.abiVersion != MR_NUMI_HUMAN_EXTENSOR_HOOD_GPU_ABI_VERSION ||
        dispatch.environmentCount == 0u || dispatch.rayCount != 8u ||
        dispatch.nodeCount == 0u || dispatch.elementCount == 0u ||
        dispatch.inputCount == 0u || dispatch.muscleCount == 0u ||
        dispatch.siteCount == 0u || dispatch.routeNodeCount == 0u ||
        dispatch.dofCount == 0u || dispatch.bodyPoseStride == 0u ||
        dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointOffset == MR_INVALID_INDEX ||
        dispatch.maximumIterations == 0u ||
        dispatch.maximumLineSearchSteps == 0u ||
        !finite4(dispatch.solver) || any(dispatch.solver <= float4(0.0f)) ||
        !finite4(dispatch.foundation) || !(dispatch.foundation.x > 0.0f) ||
        any(dispatch.foundation.yzw != float3(0.0f)) ||
        dispatch.wrapCount == 0u) return;

    const MRNumiHumanExtensorHoodRayGPU ray = rays[rayIndex];
    if (ray.nodes.y < 4u || ray.nodes.y > kMaxNodes ||
        ray.nodes.x > dispatch.nodeCount ||
        ray.nodes.y > dispatch.nodeCount - ray.nodes.x ||
        ray.elements.y == 0u || ray.elements.y > kMaxElements ||
        ray.elements.x > dispatch.elementCount ||
        ray.elements.y > dispatch.elementCount - ray.elements.x ||
        ray.elements.w == 0u ||
        ray.elements.w > MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_INPUTS ||
        ray.elements.z > dispatch.inputCount ||
        ray.elements.w > dispatch.inputCount - ray.elements.z ||
        ray.nodes.z > 1u || ray.nodes.w < 2u || ray.nodes.w > 5u) {
        result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_TOPOLOGY;
        rayResults[global] = result;
        return;
    }

    float3 position[kMaxNodes];
    float3 initialPosition[kMaxNodes];
    float3 candidate[kMaxNodes];
    float3 appliedLoad[kMaxNodes];
    float3 residual[kMaxNodes];
    float3 candidateResidual[kMaxNodes];
    float tension[kMaxElements];
    float stiffness[kMaxElements];
    float candidateTension[kMaxElements];
    float candidateStiffness[kMaxElements];
    uint freeIndex[kMaxNodes];
    uint freeCount = 0u;
    uint fixedCount = 0u;
    for (uint local = 0u; local < ray.nodes.y; ++local) {
        const MRNumiHumanExtensorHoodNodeGPU node = nodes[ray.nodes.x + local];
        if (!validBody(node.bodyIndex, dispatch) ||
            (node.flags & ~MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u ||
            node.role != local || !finite4(node.localPoint) ||
            node.localPoint.w != 0.0f ||
            (node.sourceSiteIndex != MR_INVALID_INDEX &&
             node.sourceSiteIndex >= dispatch.siteCount)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_TOPOLOGY;
            rayResults[global] = result;
            return;
        }
        const MRArticulatedBodyPoseGPU pose = bodyPoses[
            environment * dispatch.bodyPoseStride +
            node.bodyIndex - dispatch.articulationFirstBody];
        if (!finite4(pose.position) || !finite4(pose.orientation)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT;
            rayResults[global] = result;
            return;
        }
        position[local] = worldPoint(
            environment, dispatch, bodyPoses, node.bodyIndex,
            node.localPoint.xyz);
        initialPosition[local] = position[local];
        appliedLoad[local] = float3(0.0f);
        if ((node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u) {
            freeIndex[local] = MR_INVALID_INDEX;
            ++fixedCount;
        } else {
            freeIndex[local] = freeCount++;
        }
    }
    if (fixedCount < 3u || freeCount == 0u || 3u * freeCount > kMaxDimension) {
        result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_TOPOLOGY;
        rayResults[global] = result;
        return;
    }

    for (uint local = 0u; local < ray.elements.w; ++local) {
        const MRNumiHumanExtensorHoodInputGPU input =
            inputs[ray.elements.z + local];
        if (input.nodeIndex < ray.nodes.x ||
            input.nodeIndex >= ray.nodes.x + ray.nodes.y ||
            input.muscleIndex >= dispatch.muscleCount ||
            !validBody(input.proximalBodyIndex, dispatch) ||
            input.reserved0 != 0u || input.reserved1 != 0u ||
            input.reserved2 != 0u ||
            !finite4(input.proximalLocalPoint) ||
            input.proximalLocalPoint.w != 0.0f) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
            rayResults[global] = result;
            return;
        }
        const uint nodeLocal = input.nodeIndex - ray.nodes.x;
        if ((nodes[input.nodeIndex].flags &
             MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
            rayResults[global] = result;
            return;
        }
        const MRMujocoMuscleGPU muscle = muscles[input.muscleIndex];
        if (muscle.route.y < 2u || muscle.route.x > dispatch.routeNodeCount ||
            muscle.route.y > dispatch.routeNodeCount - muscle.route.x ||
            input.routeNodeOrdinal >= muscle.route.y - 1u ||
            input.targetRouteNodeOrdinal >= muscle.route.y ||
            input.routeNodeOrdinal >= input.targetRouteNodeOrdinal) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
            rayResults[global] = result;
            return;
        }
        const MRMujocoMuscleRouteNodeGPU cutRoute =
            routeNodes[muscle.route.x + input.targetRouteNodeOrdinal];
        if (cutRoute.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
            cutRoute.targetIndex != nodes[input.nodeIndex].sourceSiteIndex) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
            rayResults[global] = result;
            return;
        }
        uint cursor = input.routeNodeOrdinal;
        while (cursor + 1u < muscle.route.y) {
            const MRMujocoMuscleRouteNodeGPU first =
                routeNodes[muscle.route.x + cursor];
            const MRMujocoMuscleRouteNodeGPU next =
                routeNodes[muscle.route.x + cursor + 1u];
            if (first.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
                first.targetIndex >= dispatch.siteCount ||
                first.reserved0 != 0u ||
                !validBody(sites[first.targetIndex].bodyIndex, dispatch)) {
                result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
                rayResults[global] = result;
                return;
            }
            if (next.type == MR_MUJOCO_MUSCLE_ROUTE_SITE) {
                if (next.targetIndex >= dispatch.siteCount ||
                    next.reserved0 != 0u ||
                    !validBody(sites[next.targetIndex].bodyIndex, dispatch)) {
                    result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
                    rayResults[global] = result;
                    return;
                }
                ++cursor;
                continue;
            }
            if ((next.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE &&
                 next.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
                next.targetIndex >= dispatch.wrapCount ||
                next.reserved0 != 0u || cursor + 2u >= muscle.route.y) {
                result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
                rayResults[global] = result;
                return;
            }
            const MRMujocoMuscleWrapGPU wrap = wraps[next.targetIndex];
            const MRMujocoMuscleRouteNodeGPU last =
                routeNodes[muscle.route.x + cursor + 2u];
            if (wrap.type != next.type || !validBody(wrap.bodyIndex, dispatch) ||
                last.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
                last.targetIndex >= dispatch.siteCount ||
                !validBody(sites[last.targetIndex].bodyIndex, dispatch) ||
                (next.sideSiteIndex != MR_INVALID_INDEX &&
                 next.sideSiteIndex >= dispatch.siteCount)) {
                result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
                rayResults[global] = result;
                return;
            }
            cursor += 2u;
        }
        const MRMujocoMuscleResultGPU muscleResult = muscleResults[
            environment * dispatch.muscleCount + input.muscleIndex];
        if (muscleResult.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
            muscleResult.environment != environment ||
            muscleResult.muscleIndex != input.muscleIndex ||
            !finite4(muscleResult.activeForceAndReserved) ||
            any(muscleResult.activeForceAndReserved.yzw != float3(0.0f))) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_MUSCLE_RESULT;
            rayResults[global] = result;
            return;
        }
        const float3 proximal = worldPoint(
            environment, dispatch, bodyPoses, input.proximalBodyIndex,
            input.proximalLocalPoint.xyz);
        const float3 delta = proximal - position[nodeLocal];
        const float distanceValue = length(delta);
        if (!(distanceValue > dispatch.solver.y) || !isfinite(distanceValue)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT;
            rayResults[global] = result;
            return;
        }
        appliedLoad[nodeLocal] +=
            abs(muscleResult.activeForceAndReserved.x) * delta / distanceValue;
    }

    float potential = 0.0f;
    float strainEnergy = 0.0f;
    float maximumResidual = 0.0f;
    uint activeCount = 0u;
    if (!evaluate(ray, nodes, elements, position, initialPosition, appliedLoad,
                  dispatch.foundation.x,
                  residual, tension, stiffness, potential, strainEnergy,
                  maximumResidual, activeCount)) {
        result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT;
        rayResults[global] = result;
        return;
    }

    const uint dimension = 3u * freeCount;
    float matrix[kMaxDimension * kMaxDimension];
    float rhs[kMaxDimension];
    float step[kMaxDimension];
    uint completedIterations = 0u;
    for (; completedIterations < dispatch.maximumIterations;
         ++completedIterations) {
        if (maximumResidual <= dispatch.solver.x) break;
        for (uint row = 0u; row < dimension; ++row) {
            rhs[row] = 0.0f;
            for (uint column = 0u; column < dimension; ++column)
                matrix[row * kMaxDimension + column] = 0.0f;
            matrix[row * kMaxDimension + row] = dispatch.foundation.x;
        }
        for (uint local = 0u; local < ray.nodes.y; ++local) {
            if (freeIndex[local] == MR_INVALID_INDEX) continue;
            const uint base = 3u * freeIndex[local];
            rhs[base] = residual[local].x;
            rhs[base + 1u] = residual[local].y;
            rhs[base + 2u] = residual[local].z;
        }
        for (uint local = 0u; local < ray.elements.y; ++local) {
            if (!(tension[local] > 0.0f)) continue;
            const MRNumiHumanExtensorHoodElementGPU element =
                elements[ray.elements.x + local];
            const uint a = element.nodeA - ray.nodes.x;
            const uint b = element.nodeB - ray.nodes.x;
            const float3 delta = position[b] - position[a];
            const float lengthValue = length(delta);
            if (!(lengthValue >= dispatch.solver.y)) {
                result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT;
                rayResults[global] = result;
                return;
            }
            const float3 direction = delta / lengthValue;
            float block[9];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u; column < 3u; ++column) {
                    const float outer = direction[row] * direction[column];
                    block[3u * row + column] = stiffness[local] * outer +
                        tension[local] / lengthValue *
                        ((row == column ? 1.0f : 0.0f) - outer);
                }
            }
            const uint endpoint[2] = {a, b};
            for (uint rowNode = 0u; rowNode < 2u; ++rowNode) {
                if (freeIndex[endpoint[rowNode]] == MR_INVALID_INDEX) continue;
                for (uint columnNode = 0u; columnNode < 2u; ++columnNode) {
                    if (freeIndex[endpoint[columnNode]] == MR_INVALID_INDEX)
                        continue;
                    const float sign = rowNode == columnNode ? 1.0f : -1.0f;
                    const uint rowBase = 3u * freeIndex[endpoint[rowNode]];
                    const uint columnBase = 3u * freeIndex[endpoint[columnNode]];
                    for (uint row = 0u; row < 3u; ++row) {
                        for (uint column = 0u; column < 3u; ++column) {
                            matrix[(rowBase + row) * kMaxDimension +
                                   columnBase + column] +=
                                sign * block[3u * row + column];
                        }
                    }
                }
            }
        }
        float matrixScale = 1.0f;
        for (uint row = 0u; row < dimension; ++row)
            for (uint column = 0u; column < dimension; ++column)
                matrixScale = max(
                    matrixScale,
                    abs(matrix[row * kMaxDimension + column]));
        for (uint diagonal = 0u; diagonal < dimension; ++diagonal)
            matrix[diagonal * kMaxDimension + diagonal] +=
                dispatch.solver.z * matrixScale;
        if (!choleskySolve(matrix, rhs, step, dimension)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_SINGULAR_SYSTEM;
            rayResults[global] = result;
            return;
        }
        float descent = 0.0f;
        for (uint index = 0u; index < dimension; ++index)
            descent += rhs[index] * step[index];
        if (!(descent > 0.0f) || !isfinite(descent)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_SINGULAR_SYSTEM;
            rayResults[global] = result;
            return;
        }
        bool accepted = false;
        float alpha = 1.0f;
        float candidatePotential = 0.0f;
        float candidateStrainEnergy = 0.0f;
        float candidateMaximumResidual = 0.0f;
        uint candidateActiveCount = 0u;
        for (uint line = 0u; line < dispatch.maximumLineSearchSteps; ++line) {
            for (uint local = 0u; local < ray.nodes.y; ++local) {
                candidate[local] = position[local];
                if (freeIndex[local] == MR_INVALID_INDEX) continue;
                const uint base = 3u * freeIndex[local];
                candidate[local] += alpha * float3(
                    step[base], step[base + 1u], step[base + 2u]);
            }
            if (evaluate(
                    ray, nodes, elements, candidate, initialPosition,
                    appliedLoad, dispatch.foundation.x,
                    candidateResidual, candidateTension,
                    candidateStiffness, candidatePotential,
                    candidateStrainEnergy, candidateMaximumResidual,
                    candidateActiveCount) &&
                candidatePotential <= potential -
                    dispatch.solver.w * alpha * descent) {
                accepted = true;
                break;
            }
            alpha *= 0.5f;
        }
        if (!accepted) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_DID_NOT_CONVERGE;
            result.completedIterations = completedIterations;
            result.forceClosureAndMaximumResidual.w = maximumResidual;
            rayResults[global] = result;
            return;
        }
        for (uint local = 0u; local < ray.nodes.y; ++local) {
            position[local] = candidate[local];
            residual[local] = candidateResidual[local];
        }
        for (uint local = 0u; local < ray.elements.y; ++local) {
            tension[local] = candidateTension[local];
            stiffness[local] = candidateStiffness[local];
        }
        potential = candidatePotential;
        strainEnergy = candidateStrainEnergy;
        maximumResidual = candidateMaximumResidual;
        activeCount = candidateActiveCount;
    }
    if (maximumResidual > dispatch.solver.x) {
        result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_DID_NOT_CONVERGE;
        result.completedIterations = completedIterations;
        result.forceClosureAndMaximumResidual.w = maximumResidual;
        rayResults[global] = result;
        return;
    }

    float3 forceClosure = float3(0.0f);
    float3 momentClosure = float3(0.0f);
    float maximumTension = 0.0f;
    float maximumDisplacement = 0.0f;
    float maximumEngineeringStrain = 0.0f;
    const uint nodeBase = environment * dispatch.nodeCount + ray.nodes.x;
    for (uint local = 0u; local < ray.nodes.y; ++local) {
        const MRNumiHumanExtensorHoodNodeGPU node = nodes[ray.nodes.x + local];
        float3 bodyForce = float3(0.0f);
        if ((node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u)
            bodyForce += residual[local];
        else {
            const float3 displacement = position[local] - initialPosition[local];
            bodyForce += dispatch.foundation.x * displacement;
            maximumDisplacement = max(maximumDisplacement, length(displacement));
        }
        MRNumiHumanExtensorHoodNodeResultGPU nodeResult{};
        nodeResult.position = float4(position[local], 0.0f);
        nodeResult.bodyForce = float4(bodyForce, 0.0f);
        nodeResults[nodeBase + local] = nodeResult;
        forceClosure += bodyForce;
        const float3 bodyPoint =
            (node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u
                ? position[local] : initialPosition[local];
        momentClosure += cross(bodyPoint, bodyForce);
    }
    for (uint local = 0u; local < ray.elements.w; ++local) {
        const MRNumiHumanExtensorHoodInputGPU input =
            inputs[ray.elements.z + local];
        const float3 proximal = worldPoint(
            environment, dispatch, bodyPoses, input.proximalBodyIndex,
            input.proximalLocalPoint.xyz);
        const float3 distal = position[input.nodeIndex - ray.nodes.x];
        const float distanceValue = length(proximal - distal);
        if (!(distanceValue > dispatch.solver.y)) {
            result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT;
            rayResults[global] = result;
            return;
        }
        const float representedForce = abs(muscleResults[
            environment * dispatch.muscleCount + input.muscleIndex
        ].activeForceAndReserved.x);
        const float3 counterforce =
            -representedForce * (proximal - distal) / distanceValue;
        forceClosure += counterforce;
        momentClosure += cross(proximal, counterforce);
    }
    for (uint local = 0u; local < ray.elements.y; ++local) {
        maximumTension = max(maximumTension, tension[local]);
        if (!(tension[local] > 0.0f)) continue;
        const MRNumiHumanExtensorHoodElementGPU element =
            elements[ray.elements.x + local];
        const float currentLength = length(
            position[element.nodeB - ray.nodes.x] -
            position[element.nodeA - ray.nodes.x]);
        maximumEngineeringStrain = max(
            maximumEngineeringStrain,
            (currentLength - element.material.x) / element.material.x);
    }
    if (!finite4(float4(forceClosure, maximumResidual)) ||
        !finite4(float4(momentClosure, maximumTension)) ||
        !finite4(float4(strainEnergy, maximumDisplacement,
                        maximumEngineeringStrain, 0.0f))) {
        result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT;
        rayResults[global] = result;
        return;
    }
    result.status = MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS;
    result.completedIterations = completedIterations;
    result.forceClosureAndMaximumResidual =
        float4(forceClosure, maximumResidual);
    result.momentClosureAndMaximumTension =
        float4(momentClosure, maximumTension);
    result.energyAndCounts = float4(
        strainEnergy, float(activeCount), maximumDisplacement,
        maximumEngineeringStrain);
    rayResults[global] = result;
}

kernel void mr_numi_human_assemble_extensor_hood_correction(
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanExtensorHoodRayGPU* rays [[buffer(1)]],
    device const MRNumiHumanExtensorHoodNodeGPU* nodes [[buffer(2)]],
    device const MRNumiHumanExtensorHoodInputGPU* inputs [[buffer(3)]],
    device const MRMujocoMuscleGPU* muscles [[buffer(4)]],
    device const MRMujocoMuscleSiteGPU* sites [[buffer(5)]],
    device const MRMujocoMuscleRouteNodeGPU* routeNodes [[buffer(6)]],
    device const MRMujocoMuscleResultGPU* muscleResults [[buffer(7)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const MRNumiHumanExtensorHoodNodeResultGPU* nodeResults [[buffer(10)]],
    device const MRNumiHumanExtensorHoodRayResultGPU* rayResults [[buffer(11)]],
    device float* corrections [[buffer(12)]],
    device const float* suffixJacobians [[buffer(13)]],
    uint global [[thread_position_in_grid]]
) {
    if (global >= dispatch.environmentCount * dispatch.dofCount) return;
    const uint environment = global / dispatch.dofCount;
    const uint dof = global - environment * dispatch.dofCount;
    float correction = 0.0f;
    for (uint rayIndex = 0u; rayIndex < dispatch.rayCount; ++rayIndex) {
        const MRNumiHumanExtensorHoodRayResultGPU rayResult =
            rayResults[environment * dispatch.rayCount + rayIndex];
        if (rayResult.status != MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS) {
            corrections[global] = 0.0f;
            return;
        }
        const MRNumiHumanExtensorHoodRayGPU ray = rays[rayIndex];
        for (uint local = 0u; local < ray.nodes.y; ++local) {
            const uint nodeIndex = ray.nodes.x + local;
            const MRNumiHumanExtensorHoodNodeGPU node = nodes[nodeIndex];
            const MRNumiHumanExtensorHoodNodeResultGPU solved = nodeResults[
                environment * dispatch.nodeCount + nodeIndex];
            const float3 bodyPoint =
                (node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u
                    ? solved.position.xyz
                    : worldPoint(environment, dispatch, bodyPoses,
                                 node.bodyIndex, node.localPoint.xyz);
            correction += dot(
                solved.bodyForce.xyz,
                pointJacobian(environment, dof, dispatch, bodyPoses,
                              pointJacobians, node.bodyIndex,
                              bodyPoint));
        }
        for (uint local = 0u; local < ray.elements.w; ++local) {
            const MRNumiHumanExtensorHoodInputGPU input =
                inputs[ray.elements.z + local];
            const float representedForce = muscleResults[
                environment * dispatch.muscleCount + input.muscleIndex
            ].activeForceAndReserved.x;
            const float3 proximal = worldPoint(
                environment, dispatch, bodyPoses, input.proximalBodyIndex,
                input.proximalLocalPoint.xyz);
            const float3 distal = nodeResults[
                environment * dispatch.nodeCount + input.nodeIndex
            ].position.xyz;
            const float distanceValue = length(proximal - distal);
            if (!(distanceValue > dispatch.solver.y)) {
                corrections[global] = NAN;
                return;
            }
            const float3 counterforce =
                -abs(representedForce) * (proximal - distal) / distanceValue;
            correction += dot(
                counterforce,
                pointJacobian(environment, dof, dispatch, bodyPoses,
                              pointJacobians, input.proximalBodyIndex,
                              proximal));
            const uint inputIndex = ray.elements.z + local;
            const float derivative = suffixJacobians[
                (environment * dispatch.inputCount + inputIndex) *
                    dispatch.dofCount + dof];
            if (!isfinite(representedForce) || !isfinite(derivative)) {
                corrections[global] = derivative;
                return;
            }
            correction -= representedForce * derivative;
        }
    }
    corrections[global] = correction;
}

kernel void mr_numi_human_apply_extensor_hood_correction(
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch [[buffer(0)]],
    device const float* corrections [[buffer(1)]],
    device float* generalizedForces [[buffer(2)]],
    uint global [[thread_position_in_grid]]
) {
    if (global >= dispatch.environmentCount * dispatch.dofCount) return;
    const uint environment = global / dispatch.dofCount;
    const uint dof = global - environment * dispatch.dofCount;
    const float correction = corrections[global];
    const uint index =
        environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset + dof;
    if (!isfinite(correction)) {
        generalizedForces[index] = correction;
        return;
    }
    generalizedForces[index] += correction;
}

kernel void mr_numi_human_commit_extensor_hood_audit(
    constant MRNumiHumanExtensorHoodDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanStandStatusGPU* standStatuses [[buffer(1)]],
    device const MRNumiHumanExtensorHoodRayResultGPU* rayResults [[buffer(2)]],
    device MRNumiHumanExtensorHoodRayResultGPU* history [[buffer(3)]],
    uint global [[thread_position_in_grid]]
) {
    if (global >= dispatch.environmentCount * dispatch.rayCount ||
        dispatch.stepIndex >= MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS) return;
    const uint environment = global / dispatch.rayCount;
    const uint rayIndex = global - environment * dispatch.rayCount;
    MRNumiHumanExtensorHoodRayResultGPU result = rayResults[global];
    const MRNumiHumanStandStatusGPU stand = standStatuses[environment];
    result.transaction.x =
        result.status == MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS &&
        stand.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        stand.environment == environment &&
        stand.completedSteps == dispatch.stepIndex + 1u ? 1.0f : 0.0f;
    const uint historyIndex =
        (environment * MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS +
         dispatch.stepIndex) * dispatch.rayCount + rayIndex;
    history[historyIndex] = result;
}
