#include "ResidentArticulatedWorld.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace metalrobo::detail {
namespace {

void require(bool condition, const char* reason) {
    if (!condition) throw std::invalid_argument(reason);
}

static mr_float4 normalized(mr_float4 value) {
    const float scale = 1.0f / std::sqrt(value.x * value.x + value.y * value.y +
                                         value.z * value.z + value.w * value.w);
    return {value.x * scale, value.y * scale, value.z * scale, value.w * scale};
}

static mr_float4 scaled(mr_float4 v, float s) {
    return {v.x * s, v.y * s, v.z * s, v.w * s};
}

} // namespace

MRResidentProgramGPU compileResidentArticulation(
    const CompiledWorld& world, std::span<const MRBodyStateGPU> sceneBodies
) {
    const EngineModel& model = world.model();
    require(model.articulations.size() == 1 &&
            model.articulations.front().rootType == MR_ROOT_FIXED &&
            model.defaultQ.size() == model.defaultV.size() &&
            !model.defaultV.empty() && model.bodyNames.size() == model.bodies.size() &&
            model.defaultV.size() <= MR_RESIDENT_DOF_CAPACITY &&
            model.bodies.size() <= MR_RESIDENT_BODY_CAPACITY &&
            model.shapes.size() <= MR_RESIDENT_SHAPE_CAPACITY &&
            world.eligiblePairCount() <= MR_RESIDENT_PAIR_CAPACITY &&
            model.constraintProgram.blocks.empty() && world.rodCount() == 0u &&
            sceneBodies.size() == world.sceneBodyCount(),
            "the cooked world exceeds the resident operator's capabilities");

    MRResidentProgramGPU result{};
    result.sizes = {static_cast<unsigned>(model.bodies.size()), static_cast<unsigned>(model.dofs.size()),
                    world.eligiblePairCount(), 0u};
    result.gravityAndTimestep = model.world.gravityAndTimestep;
    result.stabilization = {model.world.solverScales.w, model.world.solverScales.z, 0, 0};
    std::copy(model.shapes.begin(), model.shapes.end(), result.shapes);
    const unsigned root = model.articulations.front().rootBody;
    for (unsigned i = 0; i < result.sizes.x; ++i) {
        const auto& src = model.bodies[i];
        auto& dst = result.bodies[i];
        auto& joint = result.joints[i];
        const bool articulated = src.articulationIndex != MR_INVALID_INDEX;
        dst.topology.x = articulated;
        dst.topology.y = MR_INVALID_INDEX;
        dst.topology.z = MR_INVALID_INDEX;
        dst.massAndDamping = {src.massAndInverseMass.x, src.massAndInverseMass.y,
                              src.dampingAndSpeedLimits.x, src.dampingAndSpeedLimits.y};
        dst.inertia[0] = src.inertiaRow0;
        dst.inertia[1] = src.inertiaRow1;
        dst.inertia[2] = src.inertiaRow2;
        dst.inverseInertia[0] = src.inverseInertiaRow0;
        dst.inverseInertia[1] = src.inverseInertiaRow1;
        dst.inverseInertia[2] = src.inverseInertiaRow2;
        dst.initialRotation.w = 1;
        joint.topology = {i, MR_INVALID_INDEX, MR_JOINT_FIXED, 0};
        if (articulated && i != root) {
            const auto& edge = model.joints.at(src.inboundJoint);
            require(edge.jointType == MR_JOINT_REVOLUTE || edge.jointType == MR_JOINT_PRISMATIC ||
                    edge.jointType == MR_JOINT_FIXED, "unsupported joint family");
            joint.topology.x = edge.parentBody;
            joint.topology.y = edge.nv == 0 ? MR_INVALID_INDEX : edge.vOffset;
            joint.topology.z = edge.jointType;
            joint.axis = edge.nv == 0 ? mr_float4{} : normalized(edge.axis0);
            joint.parentAnchor = edge.parentAnchor;
            joint.childAnchor = edge.childAnchor;
            joint.parentRotation = normalized(edge.parentRotation);
            joint.inverseChildRotation = normalized({-edge.childRotation.x, -edge.childRotation.y,
                                                     -edge.childRotation.z, edge.childRotation.w});
        }
    }

    for (unsigned n = 0; n < world.sceneBodyCount(); ++n) {
        const unsigned i = world.sceneBodyIndices()[n];
        const auto& state = sceneBodies[n];
        auto& dst = result.bodies[i];
        dst.topology.z = n;
        dst.initialPosition = state.position;
        dst.initialRotation = normalized(state.orientation);
        dst.initialLinear = state.linearVelocityAndInverseMass;
        dst.initialAngular = state.angularVelocity;
        if (state.flagsAndIndices[0] == MR_MOTION_DYNAMIC) {
            require(result.sizes.w < MR_RESIDENT_FREE_CAPACITY, "free-body capacity exceeded");
            require(state.linearVelocityAndInverseMass.w > 0.0f, "invalid inverse mass");
            dst.topology.y = result.sizes.w;
            if (result.sizes.w == 0) result.freeBody.x = i;
            else result.freeBody.y = i;
            ++result.sizes.w;
            const float originalInverseMass = dst.massAndDamping.y;
            const float inverseMass = state.linearVelocityAndInverseMass.w;
            const float scale = originalInverseMass / inverseMass;
            dst.massAndDamping.x *= scale;
            dst.massAndDamping.y = inverseMass;
            for (unsigned j = 0; j < 3; ++j) {
                dst.inertia[j] = scaled(dst.inertia[j], scale);
                dst.inverseInertia[j] = scaled(dst.inverseInertia[j], 1.0f / scale);
            }
        } else {
            require(state.flagsAndIndices[0] == MR_MOTION_STATIC, "unsupported scene body motion");
        }
    }

    for (unsigned i = 0; i < result.sizes.y; ++i) {
        const auto& src = model.dofs[i];
        auto& dst = result.dofs[i];
        require(src.qIndex == i && src.vIndex == i, "unsupported coordinate ordering");
        dst.topology = {model.joints[src.jointIndex].childBody, 0, src.flags, 0};
        dst.drive = src.drive;
        dst.limits = src.limits;
        dst.initial = {model.defaultQ[i], model.defaultV[i], 0, 0};
        if (!model.actuatorProfiles.empty()) {
            const auto& actuator = model.actuatorProfiles[i];
            const bool active = (actuator.identity.y & MR_ACTUATOR_PROFILE_ACTIVE) != 0;
            require(!active || (actuator.transmissionAndEnvelope.x == 0 &&
                                actuator.transmissionAndEnvelope.y == 0), "unsupported actuator delay or play");
            dst.initial.z = active;
            dst.motorAndSpeed = actuator.motorAndSpeed;
            dst.transmissionAndEnvelope = actuator.transmissionAndEnvelope;
        }
    }

    for (unsigned body = 0; body < result.sizes.x; ++body) {
        if (result.bodies[body].topology.x == 0) continue;
        unsigned cursor = body;
        for (unsigned depth = 0; cursor != root; ++depth) {
            require(depth < result.sizes.x, "cyclic articulation");
            const auto& joint = result.joints[cursor];
            if (joint.topology.y != MR_INVALID_INDEX) {
                result.joints[body].topology.w |= 1u << joint.topology.y;
                result.dofs[joint.topology.y].topology.y |= 1u << body;
            }
            cursor = joint.topology.x;
        }
    }

    unsigned index = 0;
    for (const auto& src : world.eligiblePairs()) {
        const auto& a = model.shapes[src.colliderA];
        const auto& b = model.shapes[src.colliderB];
        const auto allowed = [](const MRShapeGPU& shape) {
            return shape.shapeType <= MR_SHAPE_PLANE && shape.flags == 0 && shape.geometryCount == 0;
        };
        require(allowed(a) && allowed(b), "unsupported collision geometry");
        const auto& ma = model.materials[a.materialIndex];
        const auto& mb = model.materials[b.materialIndex];
        require(ma.geometry.y == 0 && mb.geometry.y == 0 &&
                ma.friction.z == 0 && mb.friction.z == 0 &&
                ma.friction.w == 0 && mb.friction.w == 0, "unsupported contact material");
        auto& dst = result.pairs[index++];
        dst.geometry = {src.colliderA, src.colliderB, src.pairClass, 0};
        dst.frictionAndRestitution = {
            std::sqrt(ma.friction.x * mb.friction.x), std::sqrt(ma.friction.y * mb.friction.y),
            std::max(ma.response.x, mb.response.x), std::max(ma.response.y, mb.response.y)
        };
        dst.response = {ma.response.z + mb.response.z, ma.response.w + mb.response.w,
                        a.contactRestAndBoundingRadius.y + b.contactRestAndBoundingRadius.y, 0};
    }
    std::stable_partition(result.pairs, result.pairs + result.sizes.z,
        [&](const MRResidentPairGPU& pair) {
            return result.shapes[pair.geometry.x].shapeType == MR_SHAPE_PLANE ||
                result.shapes[pair.geometry.y].shapeType == MR_SHAPE_PLANE;
        });
    return result;
}

void bindResidentArticulationActions(
    const CompiledTaskProgram& bindings, float controlSeconds, MRResidentProgramGPU& program
) {
    require(bindings.actionBindings().size() == program.sizes.y, "incompatible position-action binding");
    program.controls.x = bindings.actionBindings().size();
    unsigned mask = 0;
    for (const auto& binding : bindings.actionBindings()) {
        const auto& i = binding.indices;
        require((binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION ||
                 binding.actuator.x == MR_TASK_ACTUATOR_GRIPPER_POSITION) &&
                i.x < program.controls.x && i.z == i.w && i.w < program.sizes.y &&
                (mask & (1u << i.w)) == 0, "resident input requires distinct scalar position actions");
        mask |= 1u << i.w;
        auto& dof = program.dofs[i.w];
        dof.topology.w = i.x;
        dof.action = binding.parameters;
        if (binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION) {
            dof.drive.x = binding.drive.x;
            dof.drive.y = binding.drive.y;
            if (dof.drive.x > 0.0f || dof.drive.y > 0.0f) dof.topology.z |= MR_DOF_FLAG_DRIVE;
        }
        const float tau = dof.action.w;
        dof.action.w = tau > 0 ? std::clamp(1.0f - std::exp(-controlSeconds / tau), 0.0f, 1.0f) : 1.0f;
    }
}

} // namespace metalrobo::detail
