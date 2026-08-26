#include "metalrobo/OpenSimMobileRoot.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"

#include <algorithm>
#include <utility>

namespace metalrobo {
namespace {

FunctionBasedMobileRootDiagnostics fail(
    const FunctionBasedMobileRootStatus status,
    std::string message
) {
    return {status, std::move(message)};
}

bool hasFunctionProgram(
    const EngineModel& model,
    const std::uint32_t jointIndex
) {
    return std::ranges::any_of(
        model.functionBasedJointPrograms,
        [jointIndex](const FunctionBasedJointProgram& program) {
            return program.jointIndex == jointIndex;
        }
    );
}

} // namespace

const char* functionBasedMobileRootStatusName(
    const FunctionBasedMobileRootStatus status
) noexcept {
    switch (status) {
    case FunctionBasedMobileRootStatus::success:
        return "success";
    case FunctionBasedMobileRootStatus::invalidSourceModel:
        return "invalid_source_model";
    case FunctionBasedMobileRootStatus::unsupportedTopology:
        return "unsupported_topology";
    case FunctionBasedMobileRootStatus::sourceKinematicsFailed:
        return "source_kinematics_failed";
    case FunctionBasedMobileRootStatus::reducedModelInvalid:
        return "reduced_model_invalid";
    }
    return "unknown";
}

FunctionBasedMobileRootDiagnostics
reduceFixedFunctionBasedRootToMobileDefaultPose(
    const EngineModel& source,
    FunctionBasedMobileRootReduction& result
) {
    result = {};
    std::string sourceReason;
    if (!source.valid(&sourceReason)) {
        return fail(
            FunctionBasedMobileRootStatus::invalidSourceModel,
            "source model is invalid: " + sourceReason
        );
    }
    if (source.articulations.size() != 1u || source.bodies.size() < 2u ||
        source.joints.empty() || source.dofs.size() < 6u ||
        !source.shapes.empty() || !source.materials.empty() ||
        !source.geometryHeaders.empty() || !source.geometryVertices.empty() ||
        !source.geometryIndices.empty() || !source.convexFaces.empty() ||
        !source.convexHalfEdges.empty() || !source.meshBvhNodes.empty() ||
        !source.meshTriangles.empty() || !source.collisionExclusions.empty() ||
        !source.constraintProgram.empty() || !source.actuatorProfiles.empty()) {
        return fail(
            FunctionBasedMobileRootStatus::unsupportedTopology,
            "mobile-root reduction requires one unshaped, unconstrained source articulation without actuator profiles"
        );
    }
    const MRArticulationGPU& fixed = source.articulations.front();
    const MRJointDescriptorGPU& sourceRoot = source.joints.front();
    if (fixed.rootType != MR_ROOT_FIXED || fixed.rootBody != 0u ||
        fixed.firstBody != 0u || fixed.firstJoint != 0u ||
        sourceRoot.parentBody != 0u || sourceRoot.childBody != 1u ||
        sourceRoot.jointType != MR_JOINT_FUNCTION_BASED ||
        sourceRoot.qOffset != 0u || sourceRoot.vOffset != 0u ||
        sourceRoot.nq != 6u || sourceRoot.nv != 6u ||
        !hasFunctionProgram(source, 0u)) {
        return fail(
            FunctionBasedMobileRootStatus::unsupportedTopology,
            "source root must be the first six-coordinate FunctionBased joint below a synthetic fixed anchor"
        );
    }
    if (std::ranges::any_of(
            source.defaultV.begin(), source.defaultV.begin() + sourceRoot.nv,
            [](const float velocity) { return velocity != 0.0f; }
        )) {
        return fail(
            FunctionBasedMobileRootStatus::unsupportedTopology,
            "mobile-root reduction requires a stationary source default root"
        );
    }

    std::vector<double> sourceQ(source.defaultQ.begin(), source.defaultQ.end());
    std::vector<double> sourceV(source.defaultV.begin(), source.defaultV.end());
    std::vector<ArticulatedBodyKinematics> sourcePoses(fixed.bodyCount);
    const ArticulatedDynamicsDiagnostics poseDiagnostics =
        computeArticulatedBodyKinematics(
            source, 0u, sourceQ, sourceV, sourcePoses
        );
    if (!poseDiagnostics.succeeded()) {
        return fail(
            FunctionBasedMobileRootStatus::sourceKinematicsFailed,
            "source default kinematics failed with status=" +
                std::to_string(static_cast<std::uint32_t>(
                    poseDiagnostics.status
                ))
        );
    }
    const ArticulatedBodyKinematics& rootPose = sourcePoses.at(1u);

    EngineModel model = source;
    model.name = source.name.empty()
        ? "function_based_mobile_root_default_pose"
        : source.name + "_mobile_root_default_pose";
    model.bodies.erase(model.bodies.begin());
    for (std::size_t bodyIndex = 0u; bodyIndex < model.bodies.size();
         ++bodyIndex) {
        MRBodyPropertiesGPU& body = model.bodies[bodyIndex];
        if (bodyIndex == 0u) {
            body.parentBody = MR_INVALID_INDEX;
            body.inboundJoint = MR_INVALID_INDEX;
        } else if (body.parentBody == MR_INVALID_INDEX || body.parentBody == 0u ||
                   body.inboundJoint == MR_INVALID_INDEX ||
                   body.inboundJoint == 0u) {
            return fail(
                FunctionBasedMobileRootStatus::invalidSourceModel,
                "source body ownership does not descend from the removable anchor"
            );
        } else {
            --body.parentBody;
            --body.inboundJoint;
        }
    }
    model.joints.erase(model.joints.begin());
    for (MRJointDescriptorGPU& joint : model.joints) {
        if (joint.parentBody == MR_INVALID_INDEX || joint.childBody == MR_INVALID_INDEX ||
            joint.parentBody == 0u || joint.childBody == 0u ||
            joint.qOffset < sourceRoot.nq || joint.vOffset < sourceRoot.nv) {
            return fail(
                FunctionBasedMobileRootStatus::invalidSourceModel,
                "source joint ownership does not descend from the removable anchor"
            );
        }
        --joint.parentBody;
        --joint.childBody;
        ++joint.qOffset;
    }
    std::vector<FunctionBasedJointProgram> programs;
    programs.reserve(model.functionBasedJointPrograms.size() - 1u);
    for (FunctionBasedJointProgram program : model.functionBasedJointPrograms) {
        if (program.jointIndex == 0u) continue;
        if (program.jointIndex == MR_INVALID_INDEX) {
            return fail(
                FunctionBasedMobileRootStatus::invalidSourceModel,
                "source FunctionBased program has an invalid joint index"
            );
        }
        --program.jointIndex;
        programs.push_back(std::move(program));
    }
    model.functionBasedJointPrograms = std::move(programs);

    std::vector<MRDofPropertiesGPU> dofs;
    dofs.reserve(source.dofs.size());
    for (std::uint32_t localDof = 0u; localDof < 6u; ++localDof) {
        MRDofPropertiesGPU rootDof{};
        rootDof.articulationIndex = 0u;
        rootDof.jointIndex = MR_INVALID_INDEX;
        rootDof.qIndex = localDof < 3u ? localDof : MR_INVALID_INDEX;
        rootDof.vIndex = localDof;
        rootDof.localDof = localDof;
        rootDof.flags = MR_DOF_FLAG_ROOT;
        dofs.push_back(rootDof);
    }
    for (std::size_t sourceDof = sourceRoot.nv;
         sourceDof < source.dofs.size(); ++sourceDof) {
        MRDofPropertiesGPU dof = source.dofs[sourceDof];
        if (dof.jointIndex == MR_INVALID_INDEX || dof.jointIndex == 0u ||
            dof.qIndex == MR_INVALID_INDEX) {
            return fail(
                FunctionBasedMobileRootStatus::invalidSourceModel,
                "source DoF ownership does not descend from the removable anchor"
            );
        }
        --dof.jointIndex;
        ++dof.qIndex;
        dofs.push_back(dof);
    }
    model.dofs = std::move(dofs);

    model.defaultQ = {
        static_cast<float>(rootPose.centerOfMassPosition[0]),
        static_cast<float>(rootPose.centerOfMassPosition[1]),
        static_cast<float>(rootPose.centerOfMassPosition[2]),
        static_cast<float>(rootPose.orientation[0]),
        static_cast<float>(rootPose.orientation[1]),
        static_cast<float>(rootPose.orientation[2]),
        static_cast<float>(rootPose.orientation[3]),
    };
    model.defaultQ.insert(
        model.defaultQ.end(), source.defaultQ.begin() + sourceRoot.nq,
        source.defaultQ.end()
    );
    model.defaultV.assign(6u, 0.0f);
    model.defaultV.insert(
        model.defaultV.end(), source.defaultV.begin() + sourceRoot.nv,
        source.defaultV.end()
    );
    if (!model.bodyNames.empty()) model.bodyNames.erase(model.bodyNames.begin());
    if (!model.jointNames.empty()) model.jointNames.erase(model.jointNames.begin());
    if (!model.dofNames.empty()) {
        model.dofNames.erase(
            model.dofNames.begin(), model.dofNames.begin() + sourceRoot.nv
        );
        model.dofNames.insert(model.dofNames.begin(), {
            "root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz",
        });
    }

    MRArticulationGPU& floating = model.articulations.front();
    floating.rootBody = 0u;
    floating.rootType = MR_ROOT_FLOATING;
    floating.firstBody = 0u;
    floating.bodyCount = static_cast<mr_u32>(model.bodies.size());
    floating.firstJoint = 0u;
    floating.jointCount = static_cast<mr_u32>(model.joints.size());
    floating.qOffset = 0u;
    floating.nq = static_cast<mr_u32>(model.defaultQ.size());
    floating.vOffset = 0u;
    floating.nv = static_cast<mr_u32>(model.defaultV.size());
    model.world.bodyCount = static_cast<mr_u32>(model.bodies.size());
    model.world.jointCount = static_cast<mr_u32>(model.joints.size());
    model.world.nq = floating.nq;
    model.world.nv = floating.nv;

    std::string reducedReason;
    if (!model.valid(&reducedReason)) {
        return fail(
            FunctionBasedMobileRootStatus::reducedModelInvalid,
            "reduced mobile-root model is invalid: " + reducedReason
        );
    }
    result.model = std::move(model);
    result.sourceBodyToMobileBody.assign(source.bodies.size(), MR_INVALID_INDEX);
    result.sourceJointToMobileJoint.assign(source.joints.size(), MR_INVALID_INDEX);
    for (std::size_t index = 1u; index < source.bodies.size(); ++index) {
        result.sourceBodyToMobileBody[index] = static_cast<mr_u32>(index - 1u);
    }
    for (std::size_t index = 1u; index < source.joints.size(); ++index) {
        result.sourceJointToMobileJoint[index] = static_cast<mr_u32>(index - 1u);
    }
    return {};
}

} // namespace metalrobo
