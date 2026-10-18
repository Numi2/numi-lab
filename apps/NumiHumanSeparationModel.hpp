#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::separation {

struct TranslationCarrierChain {
    std::array<std::uint32_t, 3> bodyIndices{};
    std::array<std::uint32_t, 3> jointIndices{};
    std::array<std::uint32_t, 3> qIndices{};
    std::array<std::uint32_t, 3> vIndices{};
    // Global source-index -> augmented-index maps. Bodies at/after the seam child shift.
    std::vector<std::uint32_t> oldBodyToNew;
    std::vector<std::uint32_t> oldJointToNew;
    std::vector<std::uint32_t> oldQToNew;
    std::vector<std::uint32_t> oldVToNew;
};

// Insert a massless XYZ prismatic carrier chain immediately before a selected
// authored tree edge. Joint and q/v arrays are inserted at that edge's source
// slots, preserving the engine's topological joint and generalized-coordinate
// ordering. Original body IDs at/after the seam child shift by three; use oldBodyToNew.
// The returned maps are required when callers remap programs outside
// EngineModel (for example source muscle routes or joint-equality payloads).
// This prepares topology only: the owning transaction must constrain the new
// translations at zero before the cut and release them at the event.
inline bool insertTranslationCarriers(
    const EngineModel& source,
    const std::uint32_t seamJointIndex,
    std::string_view stableTag,
    EngineModel& output,
    TranslationCarrierChain& chain,
    std::string& error
) {
    const auto reject = [&error](std::string message) {
        error = std::move(message);
        return false;
    };
    error.clear();

    if (source.world.bodyCount != source.bodies.size() ||
        source.world.jointCount != source.joints.size() ||
        source.world.articulationCount != source.articulations.size() ||
        source.world.nq != source.defaultQ.size() ||
        source.world.nv != source.defaultV.size() ||
        source.world.nv != source.dofs.size()) {
        return reject("EngineModel global counts do not match their owned arrays");
    }
    if ((!source.bodyNames.empty() &&
         source.bodyNames.size() != source.bodies.size()) ||
        (!source.jointNames.empty() &&
         source.jointNames.size() != source.joints.size()) ||
        (!source.dofNames.empty() &&
         source.dofNames.size() != source.dofs.size()) ||
        (!source.actuatorProfiles.empty() &&
         source.actuatorProfiles.size() != source.dofs.size())) {
        return reject("EngineModel semantic names or actuator profiles are misaligned");
    }
    if (!source.constraintProgram.empty()) {
        return reject("models with EngineModel ConstraintIR require explicit endpoint remapping");
    }
    if (seamJointIndex >= source.joints.size()) {
        return reject("seam joint index is outside EngineModel");
    }

    const MRJointDescriptorGPU& seam = source.joints[seamJointIndex];
    if (seam.parentBody >= source.bodies.size() ||
        seam.childBody >= source.bodies.size() ||
        seam.parentBody == seam.childBody ||
        seam.nq != 1u || seam.nv != 1u) {
        return reject("seam must be a distinct tree edge with one q and one v coordinate");
    }
    const std::uint32_t articulationIndex =
        source.bodies[seam.parentBody].articulationIndex;
    if (articulationIndex >= source.articulations.size() ||
        source.bodies[seam.childBody].articulationIndex != articulationIndex) {
        return reject("seam joint endpoints do not share one articulation");
    }

    const MRArticulationGPU& articulation =
        source.articulations[articulationIndex];
    const std::uint64_t bodyEnd =
        std::uint64_t(articulation.firstBody) + articulation.bodyCount;
    const std::uint64_t jointEnd =
        std::uint64_t(articulation.firstJoint) + articulation.jointCount;
    const std::uint64_t qEnd =
        std::uint64_t(articulation.qOffset) + articulation.nq;
    const std::uint64_t vEnd =
        std::uint64_t(articulation.vOffset) + articulation.nv;
    if (seamJointIndex < articulation.firstJoint ||
        seamJointIndex >= jointEnd ||
        seam.parentBody < articulation.firstBody ||
        seam.parentBody >= bodyEnd ||
        seam.childBody < articulation.firstBody ||
        seam.childBody >= bodyEnd ||
        source.bodies[seam.childBody].parentBody != seam.parentBody ||
        source.bodies[seam.childBody].inboundJoint != seamJointIndex ||
        seam.qOffset < articulation.qOffset ||
        seam.qOffset >= qEnd ||
        seam.vOffset < articulation.vOffset ||
        seam.vOffset >= vEnd) {
        return reject("seam joint is not the child body's authored tree edge");
    }
    if (articulationIndex + 1u != source.articulations.size() ||
        bodyEnd != source.bodies.size() ||
        jointEnd != source.joints.size() ||
        qEnd != source.world.nq ||
        vEnd != source.world.nv) {
        return reject("seam articulation must own the trailing global topology ranges");
    }
    if (!source.functionBasedJointPrograms.empty()) {
        for (const auto& program : source.functionBasedJointPrograms) {
            if (program.jointIndex >= source.joints.size()) {
                return reject("FunctionBased program joint index is out of range");
            }
        }
    }
    if (source.world.bodyCount >
            std::numeric_limits<std::uint32_t>::max() - 3u ||
        source.world.jointCount >
            std::numeric_limits<std::uint32_t>::max() - 3u ||
        source.world.nq > std::numeric_limits<std::uint32_t>::max() - 3u ||
        source.world.nv > std::numeric_limits<std::uint32_t>::max() - 3u ||
        articulation.bodyCount >
            std::numeric_limits<std::uint32_t>::max() - 3u ||
        articulation.jointCount >
            std::numeric_limits<std::uint32_t>::max() - 3u ||
        articulation.nq > std::numeric_limits<std::uint32_t>::max() - 3u ||
        articulation.nv > std::numeric_limits<std::uint32_t>::max() - 3u) {
        return reject("translation carrier insertion would overflow a 32-bit count");
    }

    const std::uint32_t qInsert = seam.qOffset;
    const std::uint32_t vInsert = seam.vOffset;
    if (qInsert > source.defaultQ.size() || vInsert > source.defaultV.size()) {
        return reject("seam coordinate insertion point is outside q/v storage");
    }

    TranslationCarrierChain candidateChain{};
    candidateChain.oldBodyToNew.resize(source.bodies.size());
    candidateChain.oldJointToNew.resize(source.joints.size());
    candidateChain.oldQToNew.resize(source.defaultQ.size());
    candidateChain.oldVToNew.resize(source.defaultV.size());
    const std::uint32_t bodyInsert = seam.childBody;
    for (std::uint32_t index = 0u; index < source.bodies.size(); ++index) {
        candidateChain.oldBodyToNew[index] =
            index < bodyInsert ? index : index + 3u;
    }
    for (std::uint32_t index = 0u; index < source.joints.size(); ++index) {
        candidateChain.oldJointToNew[index] =
            index < seamJointIndex ? index : index + 3u;
    }
    for (std::uint32_t index = 0u; index < source.defaultQ.size(); ++index) {
        candidateChain.oldQToNew[index] =
            index < qInsert ? index : index + 3u;
    }
    for (std::uint32_t index = 0u; index < source.defaultV.size(); ++index) {
        candidateChain.oldVToNew[index] =
            index < vInsert ? index : index + 3u;
    }

    EngineModel candidate = source;
    static constexpr std::array<mr_float4, 3> axes{{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    static constexpr std::array<const char*, 3> axisNames{{"tx", "ty", "tz"}};
    const std::string baseTag = stableTag.empty()
        ? (source.jointNames.empty()
            ? std::string("joint_") + std::to_string(seamJointIndex)
            : source.jointNames[seamJointIndex])
        : std::string(stableTag);
    std::vector<MRJointDescriptorGPU> newJoints;
    std::vector<MRDofPropertiesGPU> newDofs;
    std::vector<MRActuatorProfileGPU> newProfiles;
    std::vector<MRBodyPropertiesGPU> newBodies;
    newJoints.reserve(3u);
    newDofs.reserve(3u);
    newProfiles.reserve(3u);
    newBodies.reserve(3u);
    std::uint32_t parentBody = candidateChain.oldBodyToNew[seam.parentBody];

    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const std::uint32_t bodyIndex = bodyInsert + axis;
        const std::uint32_t jointIndex = seamJointIndex + axis;
        const std::uint32_t qIndex = qInsert + axis;
        const std::uint32_t vIndex = vInsert + axis;
        candidateChain.bodyIndices[axis] = bodyIndex;
        candidateChain.jointIndices[axis] = jointIndex;
        candidateChain.qIndices[axis] = qIndex;
        candidateChain.vIndices[axis] = vIndex;

        MRBodyPropertiesGPU carrier{};
        carrier.articulationIndex = articulationIndex;
        carrier.parentBody = parentBody;
        carrier.inboundJoint = jointIndex;
        carrier.motionType = MR_MOTION_STATIC;
        newBodies.push_back(carrier);

        MRJointDescriptorGPU translation{};
        translation.parentBody = parentBody;
        translation.childBody = bodyIndex;
        translation.jointType = MR_JOINT_PRISMATIC;
        translation.qOffset = qIndex;
        translation.nq = 1u;
        translation.vOffset = vIndex;
        translation.nv = 1u;
        translation.axis0 = axes[axis];
        translation.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        translation.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        newJoints.push_back(translation);

        MRDofPropertiesGPU dof{};
        dof.articulationIndex = articulationIndex;
        dof.jointIndex = jointIndex;
        dof.qIndex = qIndex;
        dof.vIndex = vIndex;
        dof.localDof = 0u;
        newDofs.push_back(dof);

        if (!candidate.actuatorProfiles.empty()) {
            MRActuatorProfileGPU profile{};
            profile.identity = {vIndex, 0u, 0u, 0u};
            newProfiles.push_back(profile);
        }

        parentBody = bodyIndex;
    }

    // Insert joint descriptors at the seam so the parent chain is visited
    // before the original seam edge and all descendants.
    candidate.bodies.insert(
        candidate.bodies.begin() + bodyInsert, newBodies.begin(), newBodies.end());
    for (std::uint32_t oldIndex = 0u; oldIndex < source.bodies.size(); ++oldIndex) {
        auto& body = candidate.bodies[candidateChain.oldBodyToNew[oldIndex]];
        if (source.bodies[oldIndex].parentBody != MR_INVALID_INDEX)
            body.parentBody = candidateChain.oldBodyToNew[source.bodies[oldIndex].parentBody];
        if (source.bodies[oldIndex].inboundJoint != MR_INVALID_INDEX)
            body.inboundJoint = candidateChain.oldJointToNew[source.bodies[oldIndex].inboundJoint];
    }
    candidate.joints.insert(
        candidate.joints.begin() + seamJointIndex,
        newJoints.begin(), newJoints.end());
    for (std::uint32_t oldIndex = 0u; oldIndex < source.joints.size(); ++oldIndex) {
        MRJointDescriptorGPU& joint =
            candidate.joints[candidateChain.oldJointToNew[oldIndex]];
        // Fixed joints carry the packed cursor in qOffset/vOffset even though
        // their nq/nv are zero, so shift every post-seam descriptor's cursor.
        if (oldIndex >= seamJointIndex) {
            if (joint.qOffset != MR_INVALID_INDEX) joint.qOffset += 3u;
            if (joint.vOffset != MR_INVALID_INDEX) joint.vOffset += 3u;
        }
    }
    for (auto& program : candidate.functionBasedJointPrograms) {
        program.jointIndex =
            candidateChain.oldJointToNew[program.jointIndex];
    }

    // Remap all original joint endpoints; the old seam now starts at the final
    // carrier. Other body identities keep their source semantics via oldBodyToNew.
    for (std::uint32_t oldIndex = 0u; oldIndex < source.joints.size(); ++oldIndex) {
        auto& joint = candidate.joints[candidateChain.oldJointToNew[oldIndex]];
        joint.parentBody = oldIndex == seamJointIndex
            ? parentBody : candidateChain.oldBodyToNew[source.joints[oldIndex].parentBody];
        joint.childBody = candidateChain.oldBodyToNew[source.joints[oldIndex].childBody];
    }
    candidate.bodies[candidateChain.oldBodyToNew[seam.childBody]].parentBody = parentBody;
    for (auto& articulationRecord : candidate.articulations) {
        if (articulationRecord.rootBody != MR_INVALID_INDEX)
            articulationRecord.rootBody = candidateChain.oldBodyToNew[articulationRecord.rootBody];
    }
    for (auto& shape : candidate.shapes) {
        if (shape.bodyIndex >= source.bodies.size())
            return reject("shape body index is outside source model");
        shape.bodyIndex = candidateChain.oldBodyToNew[shape.bodyIndex];
    }
    if (!candidate.bodyNames.empty()) {
        std::vector<std::string> names;
        names.reserve(source.bodyNames.size() + 3u);
        names.insert(names.end(), source.bodyNames.begin(), source.bodyNames.begin() + bodyInsert);
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            names.push_back(baseTag + "_separation_" + axisNames[axis] + "_carrier");
        names.insert(names.end(), source.bodyNames.begin() + bodyInsert, source.bodyNames.end());
        candidate.bodyNames = std::move(names);
    }

    // Remap existing generalized coordinates, then insert the new zero states
    // at the same source slot as the new seam coordinates.
    for (auto& dof : candidate.dofs) {
        if (dof.jointIndex != MR_INVALID_INDEX) {
            if (dof.jointIndex >= seamJointIndex) dof.jointIndex += 3u;
        }
        if (dof.qIndex != MR_INVALID_INDEX && dof.qIndex >= qInsert) {
            dof.qIndex += 3u;
        }
        if (dof.vIndex >= vInsert) dof.vIndex += 3u;
    }
    candidate.dofs.insert(
        candidate.dofs.begin() + vInsert,
        newDofs.begin(), newDofs.end());
    candidate.defaultQ.insert(
        candidate.defaultQ.begin() + qInsert, 3u, 0.0f);
    candidate.defaultV.insert(
        candidate.defaultV.begin() + vInsert, 3u, 0.0f);

    if (!candidate.actuatorProfiles.empty()) {
        for (std::uint32_t v = 0u; v < source.actuatorProfiles.size(); ++v) {
            auto& profile = candidate.actuatorProfiles[v];
            if (profile.identity.x >= vInsert) profile.identity.x += 3u;
        }
        candidate.actuatorProfiles.insert(
            candidate.actuatorProfiles.begin() + vInsert,
            newProfiles.begin(), newProfiles.end());
    }
    if (!candidate.jointNames.empty()) {
        std::vector<std::string> names;
        names.reserve(candidate.jointNames.size() + 3u);
        names.insert(names.end(), source.jointNames.begin(),
                     source.jointNames.begin() + seamJointIndex);
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            names.push_back(baseTag + "_separation_" + axisNames[axis] + "_joint");
        }
        names.insert(names.end(), source.jointNames.begin() + seamJointIndex,
                     source.jointNames.end());
        candidate.jointNames = std::move(names);
    }
    if (!candidate.dofNames.empty()) {
        std::vector<std::string> names;
        names.reserve(candidate.dofNames.size() + 3u);
        names.insert(names.end(), source.dofNames.begin(),
                     source.dofNames.begin() + vInsert);
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            names.push_back(baseTag + "_separation_" + axisNames[axis]);
        }
        names.insert(names.end(), source.dofNames.begin() + vInsert,
                     source.dofNames.end());
        candidate.dofNames = std::move(names);
    }

    MRArticulationGPU& augmented = candidate.articulations[articulationIndex];
    augmented.bodyCount += 3u;
    augmented.jointCount += 3u;
    augmented.nq += 3u;
    augmented.nv += 3u;
    candidate.world.bodyCount += 3u;
    candidate.world.jointCount += 3u;
    candidate.world.nq += 3u;
    candidate.world.nv += 3u;

    output = std::move(candidate);
    chain = std::move(candidateChain);
    return true;
}

} // namespace metalrobo::separation
