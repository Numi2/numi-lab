#include "metalrobo/RobotDescriptionCooker.hpp"

#include <cmath>
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
  <joint name="base_to_tool" type="revolute">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.2" upper="1.4"
           effort="40" velocity="2.5"/>
    <dynamics damping="0.2" friction="0.1"/>
  </joint>
</robot>
)";

constexpr std::string_view kSrdf = R"(
<robot name="cooked_two_link">
  <disable_collisions link1="base" link2="tool"
                      reason="Adjacent"/>
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
            model.world.bodyCount == 2u &&
            model.world.jointCount == 1u &&
            model.world.nq == 1u &&
            model.world.nv == 1u &&
            model.shapes.size() == 2u &&
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
            std::abs(model.shapes[0].dimensions.x - 0.2F) <
                1.0e-6F &&
            std::abs(model.shapes[1].dimensions.y - 0.1F) <
                1.0e-6F,
            "primitive geometry was not cooked"
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
            floating.world.nq == 8u &&
            floating.world.nv == 7u,
            "floating-root URDF cook failed"
        );

        std::cout
            << "robot_description_cooker=ok"
            << " links=" << diagnostics.linkCount
            << " joints=" << diagnostics.jointCount
            << " dofs=" << diagnostics.dofCount
            << " colliders=" << diagnostics.colliderCount
            << " exclusions=" << diagnostics.exclusionCount
            << " fingerprint="
            << diagnostics.sourceFingerprint
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
