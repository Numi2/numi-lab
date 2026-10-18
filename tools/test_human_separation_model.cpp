#include "NumiHumanSeparationModel.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using metalrobo::EngineModel;
using metalrobo::separation::TranslationCarrierChain;
using metalrobo::separation::insertTranslationCarriers;

static MRBodyPropertiesGPU dynamicBody(std::uint32_t parent, std::uint32_t joint,
                                      float mass) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = parent;
    body.inboundJoint = joint;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass = {mass, 1.0f / mass, 0.0f, 0.0f};
    body.inertiaRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    body.inertiaRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    body.inertiaRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    body.inverseInertiaRow0 = body.inertiaRow0;
    body.inverseInertiaRow1 = body.inertiaRow1;
    body.inverseInertiaRow2 = body.inertiaRow2;
    return body;
}

static MRJointDescriptorGPU joint(std::uint32_t parent, std::uint32_t child,
                                  std::uint32_t type, std::uint32_t q,
                                  std::uint32_t v) {
    MRJointDescriptorGPU result{};
    result.parentBody = parent;
    result.childBody = child;
    result.jointType = type;
    result.qOffset = q;
    result.nq = type == MR_JOINT_FIXED ? 0u : 1u;
    result.vOffset = v;
    result.nv = type == MR_JOINT_FIXED ? 0u : 1u;
    result.axis0 = {0.0f, 0.0f, 1.0f, 0.0f};
    result.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    result.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    return result;
}

static EngineModel makeSourceModel() {
    EngineModel model{};
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 5u;
    model.world.jointCount = 4u;
    model.world.articulationCount = 1u;
    model.world.nq = 10u;
    model.world.nv = 9u;

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 5u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 4u;
    articulation.qOffset = 0u;
    articulation.nq = 10u;
    articulation.vOffset = 0u;
    articulation.nv = 9u;
    model.articulations.push_back(articulation);

    model.bodies.push_back(dynamicBody(MR_INVALID_INDEX, MR_INVALID_INDEX, 4.0f));
    model.bodies.push_back(dynamicBody(0u, 0u, 20.0f));
    MRBodyPropertiesGPU carrier{};
    carrier.articulationIndex = 0u;
    carrier.parentBody = 1u;
    carrier.inboundJoint = 1u;
    carrier.motionType = MR_MOTION_STATIC;
    model.bodies.push_back(carrier);
    carrier.parentBody = 2u;
    carrier.inboundJoint = 2u;
    model.bodies.push_back(carrier);
    model.bodies.push_back(dynamicBody(3u, 3u, 45.0f));

    model.joints.push_back(joint(0u, 1u, MR_JOINT_FIXED, 7u, 6u));
    model.joints.push_back(joint(1u, 2u, MR_JOINT_REVOLUTE, 7u, 6u));
    model.joints.push_back(joint(2u, 3u, MR_JOINT_REVOLUTE, 8u, 7u));
    model.joints.push_back(joint(3u, 4u, MR_JOINT_REVOLUTE, 9u, 8u));

    for (std::uint32_t v = 0u; v < 9u; ++v) {
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.vIndex = v;
        if (v < 6u) {
            dof.jointIndex = MR_INVALID_INDEX;
            dof.qIndex = v < 3u ? v : MR_INVALID_INDEX;
            dof.localDof = v;
            dof.flags = MR_DOF_FLAG_ROOT;
        } else {
            const std::uint32_t j = v - 5u;
            dof.jointIndex = j;
            dof.qIndex = v + 1u;
            dof.localDof = 0u;
        }
        model.dofs.push_back(dof);
    }
    model.defaultQ.assign(10u, 0.0f);
    model.defaultQ[3] = 1.0f;
    model.defaultQ[7] = 0.2f;
    model.defaultQ[8] = -0.1f;
    model.defaultQ[9] = 0.3f;
    model.defaultV = {0.1f, -0.2f, 0.3f, 0.01f, -0.02f, 0.03f,
                      0.04f, -0.05f, 0.06f};
    model.bodyNames = {"root", "lumbar2", "rot_carrier_a", "rot_carrier_b",
                       "lumbar1"};
    model.jointNames = {"root_weld", "L1_L2", "lumbar_rotation_b",
                        "lumbar_rotation_c"};
    model.dofNames = {"root_x", "root_y", "root_z", "root_rx", "root_ry",
                       "root_rz", "L1_L2_rz", "lumbar_rz_b", "lumbar_rz_c"};
    return model;
}

static bool connected(const EngineModel& model, std::uint32_t root,
                     std::uint32_t target) {
    std::vector<std::uint32_t> queue{root};
    std::vector<bool> seen(model.bodies.size(), false);
    seen[root] = true;
    for (std::size_t i = 0u; i < queue.size(); ++i) {
        const auto body = queue[i];
        if (body == target) return true;
        for (const auto& edge : model.joints) {
            std::uint32_t next = MR_INVALID_INDEX;
            if (edge.parentBody == body) next = edge.childBody;
            if (edge.childBody == body) next = edge.parentBody;
            if (next != MR_INVALID_INDEX && !seen[next]) {
                seen[next] = true;
                queue.push_back(next);
            }
        }
    }
    return false;
}

static void checkKinematicPrepAtZeroTranslation(
    const EngineModel& source, const EngineModel& model,
    const TranslationCarrierChain& chain) {
    assert(model.defaultQ.size() == model.world.nq);
    assert(model.defaultV.size() == model.world.nv);
    for (std::size_t q = 0u; q < source.defaultQ.size(); ++q) {
        assert(std::isfinite(model.defaultQ[chain.oldQToNew[q]]));
        assert(model.defaultQ[chain.oldQToNew[q]] == source.defaultQ[q]);
    }
    for (std::size_t v = 0u; v < source.defaultV.size(); ++v)
        assert(model.defaultV[chain.oldVToNew[v]] == source.defaultV[v]);
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        assert(model.defaultQ[chain.qIndices[axis]] == 0.0f);
        assert(model.defaultV[chain.vIndices[axis]] == 0.0f);
    }
}

int main() {
    const EngineModel source = makeSourceModel();
    EngineModel augmented{};
    TranslationCarrierChain chain{};
    std::string error;
    const bool ok = insertTranslationCarriers(
        source, 1u, "L1_L2_cut", augmented, chain, error);
    if (!ok) {
        std::cerr << error << "\n";
        return 1;
    }

    assert(source.world.bodyCount == 5u && source.world.jointCount == 4u);
    assert(source.world.nq == 10u && source.world.nv == 9u);
    assert(augmented.world.bodyCount == 8u && augmented.world.jointCount == 7u);
    assert(augmented.world.nq == 13u && augmented.world.nv == 12u);
    assert(augmented.articulations[0].bodyCount == 8u);
    assert(augmented.articulations[0].jointCount == 7u);
    assert(augmented.articulations[0].nq == 13u);
    assert(augmented.articulations[0].nv == 12u);

    assert(chain.oldBodyToNew.size() == source.bodies.size());
    assert(chain.oldJointToNew.size() == source.joints.size());
    assert(chain.oldQToNew.size() == source.defaultQ.size());
    assert(chain.oldVToNew.size() == source.defaultV.size());
    for (std::size_t oldBody = 0u; oldBody < source.bodies.size(); ++oldBody) {
        const auto newBody = chain.oldBodyToNew[oldBody];
        const auto& before = source.bodies[oldBody];
        const auto& after = augmented.bodies[newBody];
        const auto expectedParent = oldBody == 2u
            ? chain.bodyIndices[2]
            : (before.parentBody == MR_INVALID_INDEX
                ? MR_INVALID_INDEX : chain.oldBodyToNew[before.parentBody]);
        assert(after.parentBody == expectedParent);
        assert(after.inboundJoint == (before.inboundJoint == MR_INVALID_INDEX
            ? MR_INVALID_INDEX : chain.oldJointToNew[before.inboundJoint]));
        assert(std::memcmp(&after.massAndInverseMass, &before.massAndInverseMass,
                           sizeof(before.massAndInverseMass)) == 0);
        assert(std::memcmp(&after.inertiaRow0, &before.inertiaRow0,
                           sizeof(before.inertiaRow0)) == 0);
        assert(std::memcmp(&after.inertiaRow1, &before.inertiaRow1,
                           sizeof(before.inertiaRow1)) == 0);
        assert(std::memcmp(&after.inertiaRow2, &before.inertiaRow2,
                           sizeof(before.inertiaRow2)) == 0);
        assert(std::memcmp(&after.inverseInertiaRow0, &before.inverseInertiaRow0,
                           sizeof(before.inverseInertiaRow0)) == 0);
        assert(std::memcmp(&after.inverseInertiaRow1, &before.inverseInertiaRow1,
                           sizeof(before.inverseInertiaRow1)) == 0);
        assert(std::memcmp(&after.inverseInertiaRow2, &before.inverseInertiaRow2,
                           sizeof(before.inverseInertiaRow2)) == 0);
    }
    for (std::size_t oldJoint = 0u; oldJoint < source.joints.size(); ++oldJoint) {
        const auto newJoint = chain.oldJointToNew[oldJoint];
        const auto& before = source.joints[oldJoint];
        const auto& after = augmented.joints[newJoint];
        assert(after.parentBody == (oldJoint == 1u ? chain.bodyIndices[2]
            : chain.oldBodyToNew[before.parentBody]));
        assert(after.childBody == chain.oldBodyToNew[before.childBody]);
        assert(after.jointType == before.jointType);
        assert(after.nq == before.nq && after.nv == before.nv);
        const auto expectedQ = before.qOffset == MR_INVALID_INDEX ? before.qOffset
            : (oldJoint >= 1u ? before.qOffset + 3u : before.qOffset);
        const auto expectedV = before.vOffset == MR_INVALID_INDEX ? before.vOffset
            : (oldJoint >= 1u ? before.vOffset + 3u : before.vOffset);
        assert(after.qOffset == expectedQ && after.vOffset == expectedV);
    }
    for (std::size_t oldV = 0u; oldV < source.dofs.size(); ++oldV) {
        const auto newV = chain.oldVToNew[oldV];
        const auto& before = source.dofs[oldV];
        const auto& after = augmented.dofs[newV];
        assert(after.vIndex == newV);
        assert(after.qIndex == (before.qIndex == MR_INVALID_INDEX
            ? MR_INVALID_INDEX : chain.oldQToNew[before.qIndex]));
        assert(after.jointIndex == (before.jointIndex == MR_INVALID_INDEX
            ? MR_INVALID_INDEX : chain.oldJointToNew[before.jointIndex]));
        assert(augmented.defaultV[newV] == source.defaultV[oldV]);
    }
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const auto body = chain.bodyIndices[axis];
        const auto edge = chain.jointIndices[axis];
        assert(body == 2u + axis && edge == 1u + axis);
        assert(chain.qIndices[axis] == 7u + axis);
        assert(chain.vIndices[axis] == 6u + axis);
        assert(augmented.bodies[body].motionType == MR_MOTION_STATIC);
        assert(augmented.bodies[body].massAndInverseMass.x == 0.0f);
        assert(augmented.bodies[body].massAndInverseMass.y == 0.0f);
        assert(augmented.bodies[body].inertiaRow0.x == 0.0f);
        assert(augmented.joints[edge].jointType == MR_JOINT_PRISMATIC);
        assert(augmented.joints[edge].parentBody ==
               (axis == 0u ? 1u : chain.bodyIndices[axis - 1u]));
        assert(augmented.joints[edge].childBody == body);
        assert(augmented.joints[edge].qOffset == chain.qIndices[axis]);
        assert(augmented.joints[edge].vOffset == chain.vIndices[axis]);
        assert(augmented.joints[edge].axis0.w == 0.0f);
        assert(augmented.dofs[chain.vIndices[axis]].flags == 0u);
        assert(augmented.dofs[chain.vIndices[axis]].drive.x == 0.0f);
        assert(augmented.defaultQ[chain.qIndices[axis]] == 0.0f);
        assert(augmented.defaultV[chain.vIndices[axis]] == 0.0f);
    }
    assert(connected(augmented, chain.oldBodyToNew[0], chain.oldBodyToNew[4]));
    checkKinematicPrepAtZeroTranslation(source, augmented, chain);

    std::cout << "separation_model_cpu=PASS source_body_joint_q_v_maps=1 "
              << "old_mass_inertia_preserved=1 default_q_v_preserved=1 "
              << "zero_translation_state=PASS inserted_bodies=3 "
              << "inserted_prismatic_dofs=3\n";
    return 0;
}
