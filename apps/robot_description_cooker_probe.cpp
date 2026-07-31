#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/ConstraintIR.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr std::string_view kUrdf = R"(
<robot name="cooked_two_link">
  <link name="base">
    <inertial>
      <origin xyz="0.01 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0"
               iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><box size="0.4 0.2 0.2"/></geometry>
    </collision>
  </link>
  <link name="tool">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0"
               iyy="0.06" iyz="0" izz="0.07"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.2"/></geometry>
    </collision>
  </link>
  <link name="finger">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.004" ixy="0" ixz="0"
               iyy="0.005" iyz="0" izz="0.006"/>
    </inertial>
    <collision>
      <geometry><capsule radius="0.015" length="0.08"/></geometry>
    </collision>
  </link>
  <joint name="base_to_tool" type="revolute">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.2" upper="1.4"
           effort="40" velocity="2.5"/>
    <dynamics damping="0.2" friction="0.1"/>
  </joint>
  <joint name="tool_to_finger" type="revolute">
    <parent link="tool"/>
    <child link="finger"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="3"/>
    <mimic joint="base_to_tool" multiplier="-0.5" offset="0.2"/>
  </joint>
  <transmission name="base_drive">
    <joint name="base_to_tool"/>
    <actuator name="base_motor"/>
  </transmission>
</robot>
)";

constexpr std::string_view kSrdf = R"(
<robot name="cooked_two_link">
  <disable_collisions link1="base" link2="tool"
                      reason="Adjacent"/>
  <passive_joint name="tool_to_finger"/>
</robot>
)";

} // namespace

int main() {
    try {
        metalrobo::EngineModel model;
        const auto diagnostics =
            metalrobo::cookRobotDescription(
                kUrdf,
                kSrdf,
                model,
                {},
                "inline://cooked_two_link.urdf"
            );
        require(
            diagnostics.succeeded(),
            std::string("URDF/SRDF cook failed: ") +
                metalrobo::robotDescriptionStatusName(
                    diagnostics.status
                ) +
                " " + diagnostics.message
        );
        std::string reason;
        require(
            model.valid(&reason),
            "cooked model invalid: " + reason
        );
        require(
            model.name == "cooked_two_link" &&
            model.world.bodyCount == 3u &&
            model.world.jointCount == 2u &&
            model.world.nq == 2u &&
            model.world.nv == 2u &&
            model.shapes.size() == 3u &&
            model.collisionExclusions.size() == 1u,
            "cooked model counts are wrong"
        );
        require(
            model.joints[0].parentBody == 0u &&
            model.joints[0].childBody == 1u &&
            std::abs(
                model.joints[0].parentAnchor.z - 0.3F
            ) < 1.0e-6F &&
            std::abs(
                model.joints[0].parentAnchor.x + 0.01F
            ) < 1.0e-6F &&
            std::abs(
                model.joints[0].childAnchor.z + 0.05F
            ) < 1.0e-6F,
            "COM-centred joint anchors are wrong"
        );
        require(
            model.dofs[0].limits.x == -1.2F &&
            model.dofs[0].limits.y == 1.4F &&
            model.dofs[0].limits.z == 2.5F &&
            model.dofs[0].limits.w == 40.0F &&
            model.dofs[0].drive.y == 0.2F &&
            model.dofs[0].drive.w == 0.1F,
            "joint limits/dynamics were not preserved"
        );
        require(
            model.shapes[0].shapeType == MR_SHAPE_BOX &&
            model.shapes[1].shapeType ==
                MR_SHAPE_CYLINDER &&
            model.shapes[2].shapeType ==
                MR_SHAPE_CAPSULE &&
            std::abs(model.shapes[0].dimensions.x - 0.2F) <
                1.0e-6F &&
            std::abs(model.shapes[1].dimensions.y - 0.1F) <
                1.0e-6F &&
            std::abs(model.shapes[2].dimensions.x - 0.015F) <
                1.0e-6F &&
            std::abs(model.shapes[2].dimensions.y - 0.04F) <
                1.0e-6F,
            "primitive geometry was not cooked"
        );
        require(
            diagnostics.mimicConstraintCount == 1u &&
            diagnostics.transmissionJointCount == 1u &&
            diagnostics.passiveJointCount == 1u &&
            model.constraintProgram.blocks.size() == 1u &&
            model.constraintProgram.rows.size() == 1u &&
            model.constraintProgram.blocks[0].type ==
                MR_CONSTRAINT_GEAR &&
            std::abs(model.defaultQ[1] - 0.2F) < 1.0e-6F &&
            (model.dofs[0].flags & MR_DOF_FLAG_ACTUATED) != 0u &&
            (model.dofs[1].flags & MR_DOF_FLAG_ACTUATED) == 0u,
            "mimic, transmission, or SRDF passive semantics were lost"
        );
        require(
            metalrobo::validateConstraintIR(
                model.constraintProgram
            ).succeeded(),
            "cooked mimic gear did not produce canonical ConstraintIR"
        );

        const metalrobo::EngineModel unchanged = model;
        const auto failed =
            metalrobo::cookRobotDescription(
                "<robot name=\"bad\"><link name=\"x\"/></robot>",
                {},
                model
            );
        require(
            !failed.succeeded() &&
            model.name == unchanged.name &&
            model.world.bodyCount ==
                unchanged.world.bodyCount &&
            model.defaultQ == unchanged.defaultQ,
            "failed URDF cook mutated the accepted model"
        );

        metalrobo::RobotDescriptionCookOptions floatingOptions;
        floatingOptions.rootMode =
            metalrobo::RobotDescriptionRootMode::floating;
        metalrobo::EngineModel floating;
        const auto floatingDiagnostics =
            metalrobo::cookRobotDescription(
                kUrdf,
                kSrdf,
                floating,
                floatingOptions
            );
        require(
            floatingDiagnostics.succeeded() &&
            floating.articulations[0].rootType ==
                MR_ROOT_FLOATING &&
            floating.world.nq == 9u &&
            floating.world.nv == 8u,
            "floating-root URDF cook failed"
        );

        const std::filesystem::path sourceRoot{
            METALROBO_SOURCE_DIR
        };
        metalrobo::EngineModel meshModel;
        const auto meshDiagnostics =
            metalrobo::cookRobotDescriptionFiles(
                sourceRoot /
                    "assets/probes/mesh_robot.urdf",
                {},
                meshModel
            );
        require(
            meshDiagnostics.succeeded(),
            std::string("mesh URDF cook failed: ") +
                metalrobo::robotDescriptionStatusName(
                    meshDiagnostics.status
                ) +
                " " + meshDiagnostics.message
        );
        require(
            meshModel.valid(&reason),
            "mesh model invalid: " + reason
        );
        require(
            meshModel.shapes.size() == 3u &&
            meshModel.geometryHeaders.size() == 2u &&
            meshModel.shapes[0].shapeType ==
                MR_SHAPE_CONVEX &&
            meshModel.shapes[1].shapeType ==
                MR_SHAPE_CONVEX &&
            meshModel.shapes[2].shapeType ==
                MR_SHAPE_CONVEX &&
            meshModel.shapes[0].geometryOffset ==
                meshModel.shapes[1].geometryOffset &&
            meshModel.shapes[0].geometryCount == 1u &&
            meshModel.shapes[0].dimensions.x == 2.0F &&
            meshModel.shapes[0].dimensions.y == 4.0F &&
            meshModel.shapes[0].dimensions.z == 6.0F &&
            std::abs(
                meshModel.shapes[0]
                        .contactRestAndBoundingRadius.z -
                    std::sqrt(14.0F)
            ) < 1.0e-6F,
            "mesh scale, deduplication, or bounding radius was lost"
        );
        require(
            meshDiagnostics.meshAssetCount == 2u &&
            meshDiagnostics.meshVertexCount == 20u &&
            meshDiagnostics.meshTriangleCount == 16u,
            "mesh diagnostics are wrong"
        );
        metalrobo::EngineModel replayMesh;
        const auto replayMeshDiagnostics =
            metalrobo::cookRobotDescriptionFiles(
                sourceRoot /
                    "assets/probes/mesh_robot.urdf",
                {},
                replayMesh
            );
        require(
            replayMeshDiagnostics.succeeded() &&
            replayMeshDiagnostics.sourceFingerprint ==
                meshDiagnostics.sourceFingerprint &&
            replayMesh.geometryVertices.size() ==
                meshModel.geometryVertices.size() &&
            std::memcmp(
                replayMesh.geometryVertices.data(),
                meshModel.geometryVertices.data(),
                meshModel.geometryVertices.size() *
                    sizeof(mr_float4)
            ) == 0 &&
            replayMesh.geometryIndices ==
                meshModel.geometryIndices,
            "mesh cook or content fingerprint is nondeterministic"
        );

        std::cout
            << "robot_description_cooker=ok"
            << " links=" << diagnostics.linkCount
            << " joints=" << diagnostics.jointCount
            << " dofs=" << diagnostics.dofCount
            << " colliders=" << diagnostics.colliderCount
            << " exclusions=" << diagnostics.exclusionCount
            << " mimic_gears="
            << diagnostics.mimicConstraintCount
            << " transmission_joints="
            << diagnostics.transmissionJointCount
            << " passive_joints="
            << diagnostics.passiveJointCount
            << " mesh_assets="
            << meshDiagnostics.meshAssetCount
            << " mesh_vertices="
            << meshDiagnostics.meshVertexCount
            << " mesh_triangles="
            << meshDiagnostics.meshTriangleCount
            << " fingerprint="
            << diagnostics.sourceFingerprint
            << " mesh_fingerprint="
            << meshDiagnostics.sourceFingerprint
            << " convex_dedup=yes"
            << " fixed_and_floating=yes"
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "robot description cooker probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
