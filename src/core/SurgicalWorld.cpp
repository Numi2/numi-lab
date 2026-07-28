#include "metalrobo/SurgicalWorld.hpp"

#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint32_t kRootQCount = 7u;
constexpr std::uint32_t kRootVCount = 6u;
constexpr std::uint32_t kJawA = 6u;
constexpr std::uint32_t kJawB = 7u;
constexpr std::uint32_t kBaseKey = 0x50534d42u;
constexpr std::uint32_t kJawKey = 0x50534d4au;

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool validPose(const SurgicalBasePose& pose) {
    if (!std::ranges::all_of(pose.position, finite) ||
        !std::ranges::all_of(pose.orientation, finite)) {
        return false;
    }
    double normSquared = 0.0;
    for (const float value : pose.orientation) {
        normSquared +=
            static_cast<double>(value) * value;
    }
    return
        std::isfinite(normSquared) &&
        std::abs(normSquared - 1.0) <= 1.0e-5;
}

void checkedAdd(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label,
    std::uint32_t& result
) {
    if (left + right >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            std::string(label) + " exceeds the 32-bit engine ABI"
        );
    }
    result = static_cast<std::uint32_t>(left + right);
}

struct OwnedConstraint {
    ConstraintIRStableKey key{};
    std::uint32_t type = MR_CONSTRAINT_BILATERAL;
    std::vector<ConstraintIREndpoint> endpoints;
    std::vector<ConstraintIRRow> rows;
    std::vector<float> warmImpulses;
};

ConstraintIRRow equalityRow() {
    ConstraintIRRow row{};
    row.direction = f4(1.0f, 0.0f, 0.0f);
    row.positionError = 0.0f;
    row.targetVelocity = 0.0f;
    row.compliance = 0.0f;
    row.dissipation = 0.0f;
    row.timeConstant = 0.01f;
    row.dampingRatio = 1.0f;
    row.impulseLower = -kConstraintIRUnbounded;
    row.impulseUpper = kConstraintIRUnbounded;
    return row;
}

void appendConstraintProgram(
    const DualPsmWorldConfig& config,
    const DualPsmWorldMetadata& metadata,
    EngineModel& model
) {
    std::vector<OwnedConstraint> owned;
    owned.reserve(
        (config.lockBases ? 12u : 0u) +
        (config.coupleJaws ? 2u : 0u)
    );
    for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
        if (config.lockBases) {
            for (std::uint32_t axis = 0u;
                 axis < kRootVCount;
                 ++axis) {
                OwnedConstraint constraint;
                constraint.key.words[0] = kBaseKey;
                constraint.key.words[1] = arm;
                constraint.key.words[2] = axis;
                constraint.key.words[3] = 0u;
                const std::uint32_t qIndex =
                    axis < 3u
                    ? metadata.qOffsets[arm] + axis
                    : kConstraintIRInvalidIndex;
                constraint.endpoints.push_back(
                    makeConstraintIRGeneralizedEndpoint(
                        metadata.articulationIndices[arm],
                        qIndex,
                        metadata.vOffsets[arm] + axis,
                        0u,
                        1.0f
                    )
                );
                constraint.rows.push_back(equalityRow());
                constraint.warmImpulses.push_back(0.0f);
                owned.push_back(std::move(constraint));
            }
        }
        if (config.coupleJaws) {
            OwnedConstraint constraint;
            constraint.key.words[0] = kJawKey;
            constraint.key.words[1] = arm;
            constraint.key.words[2] = 0u;
            constraint.key.words[3] = 0u;
            constraint.type = MR_CONSTRAINT_GEAR;
            const MRDofPropertiesGPU& jawA =
                model.dofs[
                    metadata.firstJawVelocity[arm]
                ];
            const MRDofPropertiesGPU& jawB =
                model.dofs[
                    metadata.secondJawVelocity[arm]
                ];
            constraint.endpoints.push_back(
                makeConstraintIRGeneralizedEndpoint(
                    metadata.articulationIndices[arm],
                    jawA.qIndex,
                    jawA.vIndex,
                    0u,
                    1.0f
                )
            );
            constraint.endpoints.push_back(
                makeConstraintIRGeneralizedEndpoint(
                    metadata.articulationIndices[arm],
                    jawB.qIndex,
                    jawB.vIndex,
                    0u,
                    1.0f
                )
            );
            constraint.rows.push_back(equalityRow());
            constraint.warmImpulses.push_back(0.0f);
            owned.push_back(std::move(constraint));
        }
    }

    std::vector<ConstraintIRSourceBlock> sources;
    sources.reserve(owned.size());
    for (const OwnedConstraint& constraint : owned) {
        sources.push_back({
            .key = constraint.key,
            .type = constraint.type,
            .flags = 0u,
            .islandIndex = 0u,
            .eventSlot = kConstraintIRInvalidIndex,
            .endpoints = constraint.endpoints,
            .rows = constraint.rows,
            .cone = std::nullopt,
            .warmImpulses = constraint.warmImpulses,
        });
    }
    ConstraintIRCompilationResult compiled =
        compileConstraintIR(sources);
    if (!compiled.succeeded()) {
        throw std::logic_error(
            "dual PSM ConstraintIR compilation failed: " +
            compiled.diagnostics.message
        );
    }
    model.constraintProgram = std::move(compiled.ir);
}

void appendFloatingPsm(
    const EngineModel& source,
    const SurgicalBasePose& pose,
    const std::uint32_t arm,
    DualPsmWorld& world
) {
    EngineModel& output = world.model;
    DualPsmWorldMetadata& metadata = world.metadata;
    if (source.articulations.size() != 1u ||
        source.articulations[0].rootType != MR_ROOT_FIXED ||
        source.world.nq != kSurgicalPSMJointCount ||
        source.world.nv != kSurgicalPSMJointCount ||
        !source.geometryHeaders.empty() ||
        !source.constraintProgram.empty()) {
        throw std::invalid_argument(
            "dual PSM composer requires the canonical primitive fixed-root PSM"
        );
    }

    const std::uint32_t articulationOffset =
        static_cast<std::uint32_t>(
            output.articulations.size()
        );
    const std::uint32_t bodyOffset =
        static_cast<std::uint32_t>(output.bodies.size());
    const std::uint32_t jointOffset =
        static_cast<std::uint32_t>(output.joints.size());
    const std::uint32_t materialOffset =
        static_cast<std::uint32_t>(output.materials.size());
    const std::uint32_t shapeOffset =
        static_cast<std::uint32_t>(output.shapes.size());
    const std::uint32_t qOffset =
        static_cast<std::uint32_t>(output.defaultQ.size());
    const std::uint32_t vOffset =
        static_cast<std::uint32_t>(output.defaultV.size());

    metadata.articulationIndices[arm] = articulationOffset;
    metadata.qOffsets[arm] = qOffset;
    metadata.vOffsets[arm] = vOffset;
    metadata.rootBodies[arm] =
        bodyOffset + source.articulations[0].rootBody;
    metadata.firstShapes[arm] = shapeOffset;

    MRArticulationGPU articulation =
        source.articulations[0];
    articulation.rootBody += bodyOffset;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody += bodyOffset;
    articulation.firstJoint += jointOffset;
    articulation.qOffset = qOffset;
    articulation.nq += kRootQCount;
    articulation.vOffset = vOffset;
    articulation.nv += kRootVCount;
    articulation.solverGroup = articulationOffset;
    output.articulations.push_back(articulation);

    output.defaultQ.insert(
        output.defaultQ.end(),
        pose.position.begin(),
        pose.position.end()
    );
    output.defaultQ.insert(
        output.defaultQ.end(),
        pose.orientation.begin(),
        pose.orientation.end()
    );
    output.defaultQ.insert(
        output.defaultQ.end(),
        source.defaultQ.begin(),
        source.defaultQ.end()
    );
    output.defaultV.insert(
        output.defaultV.end(),
        kRootVCount,
        0.0f
    );
    output.defaultV.insert(
        output.defaultV.end(),
        source.defaultV.begin(),
        source.defaultV.end()
    );

    for (std::uint32_t local = 0u;
         local < kRootVCount;
         ++local) {
        MRDofPropertiesGPU dof{};
        dof.articulationIndex = articulationOffset;
        dof.jointIndex = MR_INVALID_INDEX;
        dof.qIndex = local < 3u
            ? qOffset + local
            : MR_INVALID_INDEX;
        dof.vIndex = vOffset + local;
        dof.localDof = local;
        dof.flags = MR_DOF_FLAG_ROOT;
        output.dofs.push_back(dof);
    }

    for (const MRJointDescriptorGPU& sourceJoint :
         source.joints) {
        MRJointDescriptorGPU joint = sourceJoint;
        joint.parentBody += bodyOffset;
        joint.childBody += bodyOffset;
        joint.qOffset =
            qOffset + kRootQCount + sourceJoint.qOffset;
        joint.vOffset =
            vOffset + kRootVCount + sourceJoint.vOffset;
        output.joints.push_back(joint);
    }
    for (const MRDofPropertiesGPU& sourceDof :
         source.dofs) {
        MRDofPropertiesGPU dof = sourceDof;
        dof.articulationIndex = articulationOffset;
        dof.jointIndex =
            sourceDof.jointIndex == MR_INVALID_INDEX
            ? MR_INVALID_INDEX
            : jointOffset + sourceDof.jointIndex;
        dof.qIndex =
            sourceDof.qIndex == MR_INVALID_INDEX
            ? MR_INVALID_INDEX
            : qOffset + kRootQCount + sourceDof.qIndex;
        dof.vIndex =
            vOffset + kRootVCount + sourceDof.vIndex;
        output.dofs.push_back(dof);
    }
    metadata.firstJawVelocity[arm] =
        vOffset + kRootVCount + kJawA;
    metadata.secondJawVelocity[arm] =
        vOffset + kRootVCount + kJawB;

    for (const MRBodyPropertiesGPU& sourceBody :
         source.bodies) {
        MRBodyPropertiesGPU body = sourceBody;
        body.articulationIndex = articulationOffset;
        body.parentBody =
            sourceBody.parentBody == MR_INVALID_INDEX
            ? MR_INVALID_INDEX
            : bodyOffset + sourceBody.parentBody;
        body.inboundJoint =
            sourceBody.inboundJoint == MR_INVALID_INDEX
            ? MR_INVALID_INDEX
            : jointOffset + sourceBody.inboundJoint;
        output.bodies.push_back(body);
    }
    output.materials.insert(
        output.materials.end(),
        source.materials.begin(),
        source.materials.end()
    );
    for (const MRShapeGPU& sourceShape : source.shapes) {
        MRShapeGPU shape = sourceShape;
        shape.bodyIndex += bodyOffset;
        shape.materialIndex += materialOffset;
        shape.slotGeneration += arm << 16u;
        output.shapes.push_back(shape);
    }
    for (const CollisionPairExclusion& sourceExclusion :
         source.collisionExclusions) {
        output.collisionExclusions.push_back({
            .colliderA =
                shapeOffset + sourceExclusion.colliderA,
            .colliderB =
                shapeOffset + sourceExclusion.colliderB,
        });
    }
}

} // namespace

DualPsmWorld makeDualDvrkPsmWorld(
    const DualPsmWorldConfig& config
) {
    if (!validPose(config.leftBase) ||
        !validPose(config.rightBase) ||
        !std::ranges::all_of(config.gravity, finite) ||
        !finite(config.timestep) ||
        !(config.timestep > 0.0f)) {
        throw std::invalid_argument(
            "dual PSM world configuration is non-finite or invalid"
        );
    }

    const EngineModel source =
        makeDvrkPsmLargeNeedleDriverEngineModel();
    DualPsmWorld staged;
    staged.model.name = "dual_dvrk_psm_large_needle_driver";
    appendFloatingPsm(
        source,
        config.leftBase,
        0u,
        staged
    );
    appendFloatingPsm(
        source,
        config.rightBase,
        1u,
        staged
    );
    appendConstraintProgram(
        config,
        staged.metadata,
        staged.model
    );
    staged.metadata.baseLockBlockCount =
        config.lockBases ? 12u : 0u;
    staged.metadata.jawCouplingBlockCount =
        config.coupleJaws ? 2u : 0u;

    EngineModel& model = staged.model;
    MRWorldGPU& world = model.world;
    world.abiVersion = MR_ENGINE_ABI_VERSION;
    world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    world.articulationCount =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    world.jointCount =
        static_cast<std::uint32_t>(model.joints.size());
    world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    world.materialCount =
        static_cast<std::uint32_t>(
            model.materials.size()
        );
    world.nq =
        static_cast<std::uint32_t>(model.defaultQ.size());
    world.nv =
        static_cast<std::uint32_t>(model.defaultV.size());

    const std::uint64_t shapeCount = model.shapes.size();
    const std::uint64_t logicalPairs =
        shapeCount * (shapeCount - 1u) / 2u;
    std::uint32_t pairCapacity = 0u;
    std::uint32_t contactCapacity = 0u;
    std::uint32_t constraintCapacity = 0u;
    checkedAdd(logicalPairs, 0u, "pair capacity", pairCapacity);
    checkedAdd(
        4u * logicalPairs,
        0u,
        "contact capacity",
        contactCapacity
    );
    checkedAdd(
        contactCapacity,
        model.constraintProgram.blocks.size() + 64u,
        "constraint capacity",
        constraintCapacity
    );
    world.pairCapacity = std::max(pairCapacity, 1u);
    world.contactCapacity =
        std::max(contactCapacity, 8u);
    world.constraintCapacity =
        std::max(constraintCapacity, 8u);
    world.islandCapacity =
        std::max(world.bodyCount, 2u);
    world.solverType = MR_SOLVER_THROUGHPUT_TGS;
    world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    world.gravityAndTimestep = f4(
        config.gravity[0],
        config.gravity[1],
        config.gravity[2],
        config.timestep
    );
    world.solverScales =
        f4(1.0e-7f, 1.0e-9f, 2.0f, 1.0e-5f);

    std::ranges::sort(
        model.collisionExclusions,
        [](const CollisionPairExclusion& left,
           const CollisionPairExclusion& right) {
            return
                std::pair{
                    left.colliderA,
                    left.colliderB,
                } <
                std::pair{
                    right.colliderA,
                    right.colliderB,
                };
        }
    );

    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "internal dual PSM model is invalid: " + reason
        );
    }
    return staged;
}

} // namespace metalrobo
