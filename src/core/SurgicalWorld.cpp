#include "metalrobo/SurgicalWorld.hpp"

#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint32_t kJawA = 6u;
constexpr std::uint32_t kJawB = 7u;
constexpr std::uint32_t kBaseKey = 0x50534d42u;
constexpr std::uint32_t kJawKey = 0x50534d4au;
constexpr std::uint32_t kFirstPsmCollisionGroup = 1u << 8u;

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

bool validContactMaterial(const MRMaterialGPU& material) {
    const auto finite4 = [](const mr_float4 value) {
        return
            std::isfinite(value.x) &&
            std::isfinite(value.y) &&
            std::isfinite(value.z) &&
            std::isfinite(value.w);
    };
    return
        finite4(material.friction) &&
        finite4(material.response) &&
        finite4(material.geometry) &&
        material.friction.x >= material.friction.y &&
        material.friction.y >= 0.0f &&
        material.friction.z >= 0.0f &&
        material.friction.w >= 0.0f &&
        material.response.x >= 0.0f &&
        material.response.x <= 1.0f &&
        material.response.y >= 0.0f &&
        material.response.z >= 0.0f &&
        material.response.w >= 0.0f &&
        material.geometry.x >= 0.0f &&
        material.geometry.y >= 0.0f;
}

using DVec3 = std::array<double, 3>;
using DMat3 = std::array<double, 9>;

DVec3 add(const DVec3& left, const DVec3& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

DVec3 subtract(const DVec3& left, const DVec3& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

double dot(const DVec3& left, const DVec3& right) {
    return
        left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

DVec3 multiply(const DVec3& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double length(const DVec3& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

DVec3 leastAlignedDirector(const DVec3& tangent) {
    const std::array<DVec3, 3> axes{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    std::size_t selected = 0u;
    for (std::size_t axis = 1u; axis < axes.size(); ++axis) {
        if (std::abs(dot(tangent, axes[axis])) <
            std::abs(dot(tangent, axes[selected]))) {
            selected = axis;
        }
    }
    const DVec3 projected = subtract(
        axes[selected],
        multiply(tangent, dot(axes[selected], tangent))
    );
    const double projectedLength = length(projected);
    if (!(projectedLength > 1.0e-12) ||
        !std::isfinite(projectedLength)) {
        throw std::invalid_argument(
            "thread material-frame reference is degenerate"
        );
    }
    return multiply(projected, 1.0 / projectedLength);
}

DVec3 rotate(
    const std::array<float, 4>& quaternion,
    const DVec3& value
) {
    const DVec3 imaginary{
        quaternion[0],
        quaternion[1],
        quaternion[2],
    };
    const auto cross = [](const DVec3& a, const DVec3& b) {
        return DVec3{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    };
    return add(
        value,
        multiply(
            cross(
                imaginary,
                add(
                    cross(imaginary, value),
                    multiply(value, quaternion[3])
                )
            ),
            2.0
        )
    );
}

DMat3 rotationMatrix(
    const std::array<float, 4>& q
) {
    const double x = q[0];
    const double y = q[1];
    const double z = q[2];
    const double w = q[3];
    return {
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    };
}

DMat3 rotateTensor(
    const DMat3& body,
    const std::array<float, 4>& quaternion
) {
    const DMat3 rotation = rotationMatrix(quaternion);
    DMat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            double value = 0.0;
            for (std::size_t left = 0u; left < 3u; ++left) {
                for (std::size_t right = 0u;
                     right < 3u;
                     ++right) {
                    value +=
                        rotation[row * 3u + left] *
                        body[left * 3u + right] *
                        rotation[column * 3u + right];
                }
            }
            result[row * 3u + column] = value;
        }
    }
    return result;
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
                 axis < kSurgicalFloatingRootVCount;
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
        !source.geometryHeaders.empty()) {
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
    articulation.nq += kSurgicalFloatingRootQCount;
    articulation.vOffset = vOffset;
    articulation.nv += kSurgicalFloatingRootVCount;
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
        kSurgicalFloatingRootVCount,
        0.0f
    );
    output.defaultV.insert(
        output.defaultV.end(),
        source.defaultV.begin(),
        source.defaultV.end()
    );

    for (std::uint32_t local = 0u;
         local < kSurgicalFloatingRootVCount;
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
            qOffset + kSurgicalFloatingRootQCount + sourceJoint.qOffset;
        joint.vOffset =
            vOffset + kSurgicalFloatingRootVCount + sourceJoint.vOffset;
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
            : qOffset + kSurgicalFloatingRootQCount + sourceDof.qIndex;
        dof.vIndex =
            vOffset + kSurgicalFloatingRootVCount + sourceDof.vIndex;
        output.dofs.push_back(dof);
    }
    metadata.firstJawVelocity[arm] =
        vOffset + kSurgicalFloatingRootVCount + kJawA;
    metadata.secondJawVelocity[arm] =
        vOffset + kSurgicalFloatingRootVCount + kJawB;

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
        // The fixed-root source excludes one robot-internal group to suppress
        // self collision. Duplicating that bit for both arms would also
        // suppress arm-to-arm contact. Give each PSM a distinct self bit,
        // remove only that bit from its mask, and restore the source bit so
        // the other arm remains an ordinary collision partner.
        const std::uint32_t armGroup =
            kFirstPsmCollisionGroup << arm;
        shape.collisionGroup = armGroup;
        shape.collisionMask =
            (sourceShape.collisionMask |
             sourceShape.collisionGroup) & ~armGroup;
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
    world.solverType = MR_SOLVER_TEMPORAL_CONE;
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

DualPsmNeedleThreadWorld
makeDualDvrkPsmNeedleThreadWorld(
    const DualPsmNeedleThreadWorldConfig& config
) {
    if (!validPose(config.needlePose) ||
        !validContactMaterial(config.threadContactMaterial) ||
        config.threadNodeCount < 3u ||
        !(config.threadLengthM > 0.0) ||
        !std::isfinite(config.threadLengthM) ||
        !(config.attachmentCompliance >= 0.0) ||
        !std::isfinite(config.attachmentCompliance) ||
        !(config.tangentAttachmentComplianceRadPerNm >= 0.0) ||
        !std::isfinite(
            config.tangentAttachmentComplianceRadPerNm
        ) ||
        !(config.torsionalAttachmentComplianceRadPerNm >= 0.0) ||
        !std::isfinite(
            config.torsionalAttachmentComplianceRadPerNm
        ) ||
        !std::ranges::all_of(
            config.threadExitDirectionLocal,
            [](const double value) {
                return std::isfinite(value);
            }
        )) {
        throw std::invalid_argument(
            "dual PSM needle-thread configuration is invalid"
        );
    }
    const double directionLength =
        length(config.threadExitDirectionLocal);
    if (!(directionLength > 1.0e-12) ||
        !std::isfinite(directionLength)) {
        throw std::invalid_argument(
            "thread exit direction is degenerate"
        );
    }
    const DVec3 normalizedExitLocal = multiply(
        config.threadExitDirectionLocal,
        1.0 / directionLength
    );

    DualPsmNeedleThreadWorld staged;
    staged.robots = makeDualDvrkPsmWorld(config.robots);
    staged.needle = makeCurvedSutureNeedleAsset(
        SurgicalAssetIds{
            .bodyIndex = 0u,
            .materialIndex = 0u,
            .slotGenerationBase = 1u,
            .collisionGroup = 1u,
            .collisionMask = ~0u,
            .motionType = MR_MOTION_DYNAMIC,
        },
        config.needle
    );

    const double centerlineRadius =
        config.needle.arcLengthM.value /
        config.needle.arcAngleRad.value;
    const double startAngle =
        -0.5 * config.needle.arcAngleRad.value;
    staged.metadata.swageAnchorLocal = {
        centerlineRadius * std::cos(startAngle) -
            staged.needle.rigid.geometryCenterOfMassM[0],
        centerlineRadius * std::sin(startAngle) -
            staged.needle.rigid.geometryCenterOfMassM[1],
        -staged.needle.rigid.geometryCenterOfMassM[2],
    };
    const DVec3 needlePosition{
        config.needlePose.position[0],
        config.needlePose.position[1],
        config.needlePose.position[2],
    };
    staged.metadata.swageAnchorWorld = add(
        needlePosition,
        rotate(
            config.needlePose.orientation,
            staged.metadata.swageAnchorLocal
        )
    );
    staged.metadata.initialThreadDirectionWorld = rotate(
        config.needlePose.orientation,
        normalizedExitLocal
    );
    const DVec3 initialMaterialDirectorWorld =
        leastAlignedDirector(
            staged.metadata.initialThreadDirectionWorld
        );
    const std::array<float, 4> inverseNeedleOrientation{
        -config.needlePose.orientation[0],
        -config.needlePose.orientation[1],
        -config.needlePose.orientation[2],
        config.needlePose.orientation[3],
    };
    staged.metadata.swageMaterialDirectorLocal = rotate(
        inverseNeedleOrientation,
        initialMaterialDirectorWorld
    );

    const MRBodyPropertiesGPU& needleBody =
        staged.needle.rigid.body;
    MRBodyStateGPU& needleState = staged.needleState;
    needleState.position = f4(
        config.needlePose.position[0],
        config.needlePose.position[1],
        config.needlePose.position[2],
        1.0f
    );
    needleState.orientation = f4(
        config.needlePose.orientation[0],
        config.needlePose.orientation[1],
        config.needlePose.orientation[2],
        config.needlePose.orientation[3]
    );
    needleState.linearVelocityAndInverseMass =
        f4(0.0f, 0.0f, 0.0f, needleBody.massAndInverseMass.y);
    needleState.angularVelocity = f4(0.0f, 0.0f, 0.0f);
    const DMat3 inverseBody{
        needleBody.inverseInertiaRow0.x,
        needleBody.inverseInertiaRow0.y,
        needleBody.inverseInertiaRow0.z,
        needleBody.inverseInertiaRow1.x,
        needleBody.inverseInertiaRow1.y,
        needleBody.inverseInertiaRow1.z,
        needleBody.inverseInertiaRow2.x,
        needleBody.inverseInertiaRow2.y,
        needleBody.inverseInertiaRow2.z,
    };
    const DMat3 inverseWorld = rotateTensor(
        inverseBody,
        config.needlePose.orientation
    );
    needleState.inverseInertiaWorldRow0 = f4(
        static_cast<float>(inverseWorld[0]),
        static_cast<float>(inverseWorld[1]),
        static_cast<float>(inverseWorld[2])
    );
    needleState.inverseInertiaWorldRow1 = f4(
        static_cast<float>(inverseWorld[3]),
        static_cast<float>(inverseWorld[4]),
        static_cast<float>(inverseWorld[5])
    );
    needleState.inverseInertiaWorldRow2 = f4(
        static_cast<float>(inverseWorld[6]),
        static_cast<float>(inverseWorld[7]),
        static_cast<float>(inverseWorld[8])
    );
    needleState.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    needleState.flagsAndIndices[1] = MR_INVALID_INDEX;
    needleState.flagsAndIndices[2] = 0u;

    staged.threadModel = makeStraightSutureRod(
        config.threadNodeCount,
        config.threadLengthM,
        config.threadMaterial
    );
    staged.threadModel.name =
        "dual_psm_needle_swage_thread";
    staged.threadModel.fidelityBoundary =
        "DER research material; hard swage root, two-axis clamped tangent, "
        "and material-frame weld; not package-calibrated or clinical";
    const double restLength =
        config.threadLengthM /
        static_cast<double>(config.threadNodeCount - 1u);
    staged.metadata.swageTangentComplianceRadPerNm =
        config.tangentAttachmentComplianceRadPerNm;
    for (std::uint32_t node = 0u;
         node < config.threadNodeCount;
         ++node) {
        staged.threadModel.restPositions[node] = add(
            staged.metadata.swageAnchorWorld,
            multiply(
                staged.metadata.initialThreadDirectionWorld,
                restLength * static_cast<double>(node)
            )
        );
    }
    std::string rodReason;
    if (!staged.threadModel.valid(&rodReason)) {
        throw std::logic_error(
            "internal surgical thread is invalid: " + rodReason
        );
    }
    staged.threadState =
        makeDiscreteElasticRodDefaultState(
            staged.threadModel
        );
    staged.threadContactMaterial = config.threadContactMaterial;
    staged.attachments[0] = {
        .nodeIndex = 0u,
        .targetPosition = staged.metadata.swageAnchorWorld,
        .targetVelocity = {0.0, 0.0, 0.0},
        .compliance = config.attachmentCompliance,
    };
    staged.rigidBindings[0] = {
        .bodyIndex = staged.metadata.needleSceneBodyIndex,
        .localAnchor = staged.metadata.swageAnchorLocal,
    };
    staged.tangentBindings[0] = {
        .edgeIndex = 0u,
        .bodyIndex = staged.metadata.needleSceneBodyIndex,
        .localAnchor = add(
            staged.metadata.swageAnchorLocal,
            multiply(normalizedExitLocal, restLength)
        ),
        .localTangent = normalizedExitLocal,
        .localDirector = staged.metadata.swageMaterialDirectorLocal,
        .complianceRadPerNm =
            config.tangentAttachmentComplianceRadPerNm,
    };
    staged.twistBindings[0] = {
        .edgeIndex = 0u,
        .bodyIndex = staged.metadata.needleSceneBodyIndex,
        .localTangent = normalizedExitLocal,
        .localMaterialDirector =
            staged.metadata.swageMaterialDirectorLocal,
        .referenceTangentWorld =
            staged.metadata.initialThreadDirectionWorld,
        .referenceMaterialDirectorWorld =
            initialMaterialDirectorWorld,
        .complianceRadPerNm =
            config.torsionalAttachmentComplianceRadPerNm,
    };
    return staged;
}

BowelAnastomosisSutureSpec
makeBowelAnastomosisSutureSpec() noexcept {
    BowelAnastomosisSutureSpec result;
    result.needle = makeBowelAnastomosisNeedleSpec();
    return result;
}

DualPsmNeedleThreadWorldConfig
makeBowelAnastomosisNeedleThreadWorldConfig(
    const BowelAnastomosisSutureSpec& spec
) {
    const auto validPositive = [](const SurgicalScalar& value) {
        return value.value > 0.0 && std::isfinite(value.value);
    };
    if (!validPositive(spec.threadLengthM) ||
        !validPositive(spec.threadRadiusM) ||
        !validPositive(spec.threadDensityKgPerM3) ||
        !validPositive(spec.threadYoungModulusPa) ||
        !(spec.threadPoissonRatio.value > -1.0) ||
        !(spec.threadPoissonRatio.value < 0.5) ||
        !std::isfinite(spec.threadPoissonRatio.value) ||
        spec.threadNodeCount < 2u) {
        throw std::invalid_argument(
            "bowel-anastomosis suture specification is invalid"
        );
    }
    DualPsmNeedleThreadWorldConfig result;
    result.needle = spec.needle;
    result.threadMaterial = {
        .radius = spec.threadRadiusM.value,
        .density = spec.threadDensityKgPerM3.value,
        .youngModulus = spec.threadYoungModulusPa.value,
        .poissonRatio = spec.threadPoissonRatio.value,
    };
    result.threadNodeCount = spec.threadNodeCount;
    result.threadLengthM = spec.threadLengthM.value;
    return result;
}

} // namespace metalrobo
