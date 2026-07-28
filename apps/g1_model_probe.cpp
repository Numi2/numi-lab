#include "metalrobo/G1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr double kTolerance = 2.0e-6;

bool close(const double left, const double right, const double tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double determinant(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    return
        static_cast<double>(row0.x) *
            (static_cast<double>(row1.y) * row2.z -
             static_cast<double>(row1.z) * row2.y) -
        static_cast<double>(row0.y) *
            (static_cast<double>(row1.x) * row2.z -
             static_cast<double>(row1.z) * row2.x) +
        static_cast<double>(row0.z) *
            (static_cast<double>(row1.x) * row2.y -
             static_cast<double>(row1.y) * row2.x);
}

double inverseProductError(const MRBodyPropertiesGPU& body) {
    const double inertia[3][3]{
        {
            body.inertiaRow0.x,
            body.inertiaRow0.y,
            body.inertiaRow0.z,
        },
        {
            body.inertiaRow1.x,
            body.inertiaRow1.y,
            body.inertiaRow1.z,
        },
        {
            body.inertiaRow2.x,
            body.inertiaRow2.y,
            body.inertiaRow2.z,
        },
    };
    const double inverse[3][3]{
        {
            body.inverseInertiaRow0.x,
            body.inverseInertiaRow0.y,
            body.inverseInertiaRow0.z,
        },
        {
            body.inverseInertiaRow1.x,
            body.inverseInertiaRow1.y,
            body.inverseInertiaRow1.z,
        },
        {
            body.inverseInertiaRow2.x,
            body.inverseInertiaRow2.y,
            body.inverseInertiaRow2.z,
        },
    };

    double maximumError = 0.0;
    for (std::size_t row = 0; row < 3u; ++row) {
        for (std::size_t column = 0; column < 3u; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < 3u; ++inner) {
                value += inertia[row][inner] * inverse[inner][column];
            }
            const double expected = row == column ? 1.0 : 0.0;
            maximumError =
                std::max(maximumError, std::abs(value - expected));
        }
    }
    return maximumError;
}

constexpr std::array<std::string_view, metalrobo::kUnitreeG1JointCount>
    kExpectedJointNames{{
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }};

constexpr std::array<float, metalrobo::kUnitreeG1JointCount>
    kExpectedResetQ{{
        -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
        -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.3f, 0.25f, 0.0f, 0.97f, 0.15f, 0.0f, 0.0f,
        0.3f, -0.25f, 0.0f, 0.97f, -0.15f, 0.0f, 0.0f,
    }};

constexpr std::array<std::array<float, 3>, 4> kExpectedFootCenters{{
    {-0.05f, 0.025f, -0.03f},
    {-0.05f, -0.025f, -0.03f},
    {0.12f, 0.03f, -0.03f},
    {0.12f, -0.03f, -0.03f},
}};

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        const metalrobo::G1ModelMetadata& metadata =
            metalrobo::unitreeG1Metadata();

        std::string reason;
        require(
            model.valid(&reason),
            "EngineModel::valid rejected G1: " + reason
        );
        require(
            model.world.bodyCount == 30u &&
                model.world.jointCount == 29u &&
                model.world.nq == 36u &&
                model.world.nv == 35u,
            "canonical G1 counts are incorrect"
        );
        require(
            model.articulations.size() == 1u &&
                model.articulations[0].rootBody == 0u &&
                model.articulations[0].rootType == MR_ROOT_FLOATING &&
                model.articulations[0].bodyCount == 30u &&
                model.articulations[0].jointCount == 29u &&
                model.articulations[0].nq == 36u &&
                model.articulations[0].nv == 35u,
            "floating articulation descriptor is incorrect"
        );
        require(
            metadata.modeMachine == 5u && metadata.modePr == 0u,
            "G1 hardware mode metadata is incorrect"
        );
        require(
            metadata.sourceCommit ==
                "aa0f5c68b5aba347bad409e71b6430407da758d7" &&
                metadata.sourceLicense == "BSD-3-Clause",
            "pinned Unitree provenance is missing"
        );

        require(
            model.defaultQ.size() == 36u &&
                model.defaultV.size() == 35u,
            "reset-state dimensions are incorrect"
        );
        require(
                close(model.defaultQ[0], 0.0) &&
                close(model.defaultQ[1], 0.0) &&
                close(model.defaultQ[2], 0.723969939696) &&
                close(model.defaultQ[3], 0.0) &&
                close(model.defaultQ[4], 0.0) &&
                close(model.defaultQ[5], 0.0) &&
                close(model.defaultQ[6], 1.0),
            "floating pelvis COM reset pose is incorrect"
        );
        for (std::size_t index = 0; index < kExpectedResetQ.size(); ++index) {
            require(
                close(model.defaultQ[7u + index], kExpectedResetQ[index]),
                "RL Lab reset joint vector is out of SDK order"
            );
        }
        require(
            std::ranges::all_of(model.defaultV, [](const float value) {
                return value == 0.0f;
            }),
            "G1 reset velocity is not zero"
        );

        for (std::size_t index = 0; index < model.joints.size(); ++index) {
            const MRJointDescriptorGPU& joint = model.joints[index];
            const metalrobo::G1JointLimit& limit =
                metadata.jointLimits[index];
            require(
                limit.name == kExpectedJointNames[index],
                "joint name does not match Unitree SDK order"
            );
            require(
                metadata.rlLabDrives[index].name == limit.name,
                "RL drive order diverges from SDK joint order"
            );
            require(
                joint.parentBody == limit.parentBody &&
                    joint.childBody == limit.childBody &&
                    joint.childBody == index + 1u &&
                    joint.jointType == MR_JOINT_REVOLUTE &&
                    joint.qOffset == 7u + index &&
                    joint.vOffset == 6u + index,
                "joint topology or generalized offset is incorrect"
            );
            require(
                model.bodies[joint.childBody].parentBody == joint.parentBody &&
                    model.bodies[joint.childBody].inboundJoint == index,
                "body parent/inbound-joint metadata is inconsistent"
            );
            // The runtime state origin is the COM, so a joint's child anchor
            // plus the factual child COM must recover the URDF link origin.
            require(
                close(
                    joint.childAnchor.x +
                        model.bodies[joint.childBody].centerOfMass.x,
                    0.0
                ) &&
                    close(
                        joint.childAnchor.y +
                            model.bodies[joint.childBody].centerOfMass.y,
                        0.0
                    ) &&
                    close(
                        joint.childAnchor.z +
                            model.bodies[joint.childBody].centerOfMass.z,
                        0.0
                    ),
                "joint child anchor is not COM-centred"
            );
            require(
                model.defaultQ[joint.qOffset] >=
                        limit.lowerPosition - 1.0e-6f &&
                    model.defaultQ[joint.qOffset] <=
                        limit.upperPosition + 1.0e-6f &&
                    limit.maximumEffort > 0.0f &&
                    limit.maximumVelocity > 0.0f,
                "reset or hardware joint limit is invalid"
            );
            require(
                metadata.rlLabDrives[index].stiffness > 0.0f &&
                    metadata.rlLabDrives[index].damping > 0.0f &&
                    close(metadata.rlLabDrives[index].armature, 0.01),
                "named RL Lab drive metadata is invalid"
            );
        }
        require(
            close(metadata.jointLimits[4].maximumEffort, 35.0) &&
                close(metadata.jointLimits[4].maximumVelocity, 30.0) &&
                close(metadata.jointLimits[13].maximumEffort, 35.0) &&
                close(metadata.jointLimits[13].maximumVelocity, 30.0),
            "canonical URDF ankle/waist limits were replaced by a preset"
        );
        require(
            close(metadata.rlLabDrives[3].stiffness, 150.0) &&
                close(metadata.rlLabDrives[3].damping, 4.0) &&
                close(metadata.rlLabDrives[12].stiffness, 200.0) &&
                close(metadata.rlLabDrives[12].damping, 5.0),
            "RL Lab knee or waist-yaw drive is incorrect"
        );

        double totalMass = 0.0;
        double maximumInverseError = 0.0;
        std::size_t fullTensorCount = 0u;
        for (const MRBodyPropertiesGPU& body : model.bodies) {
            totalMass += body.massAndInverseMass.x;
            require(
                body.massAndInverseMass.x > 0.0f &&
                    body.massAndInverseMass.y > 0.0f &&
                    determinant(
                        body.inertiaRow0,
                        body.inertiaRow1,
                        body.inertiaRow2
                    ) > 0.0 &&
                    determinant(
                        body.inverseInertiaRow0,
                        body.inverseInertiaRow1,
                        body.inverseInertiaRow2
                    ) > 0.0,
                "G1 body inertia is not positive definite"
            );
            maximumInverseError = std::max(
                maximumInverseError,
                inverseProductError(body)
            );
            if (std::abs(body.inertiaRow0.y) > 0.0f ||
                std::abs(body.inertiaRow0.z) > 0.0f ||
                std::abs(body.inertiaRow1.z) > 0.0f) {
                ++fullTensorCount;
            }
        }
        require(
            std::abs(totalMass - metadata.canonicalMassKg) <= 5.0e-6,
            "folded G1 mass is not 33.34114202 kg"
        );
        require(
            maximumInverseError <= 2.0e-4 && fullTensorCount >= 20u,
            "full inertia tensors or their inverses regressed"
        );
        require(
            close(model.bodies[0].massAndInverseMass.x, 3.814) &&
                close(model.bodies[15].massAndInverseMass.x, 7.817) &&
                close(model.bodies[22].massAndInverseMass.x, 0.25457647) &&
                close(model.bodies[29].massAndInverseMass.x, 0.25457647),
            "fixed-link inertial folding regressed"
        );

        require(
            model.shapes.size() ==
                metalrobo::kUnitreeG1PrimitiveShapeCount,
            "official primitive collider count is incorrect"
        );
        for (std::size_t footIndex = 0;
             footIndex < metadata.feet.size();
             ++footIndex) {
            const metalrobo::G1FootFrame& foot = metadata.feet[footIndex];
            const std::uint32_t expectedBody =
                footIndex == 0u ? 6u : 12u;
            require(
                foot.bodyIndex == expectedBody &&
                    foot.bodyName ==
                        metadata.bodyNames[expectedBody] &&
                    close(
                        foot.solePosition.x +
                            model.bodies[expectedBody].centerOfMass.x,
                        0.035
                    ) &&
                    close(
                        foot.solePosition.y +
                            model.bodies[expectedBody].centerOfMass.y,
                        0.0
                    ) &&
                    close(
                        foot.solePosition.z +
                            model.bodies[expectedBody].centerOfMass.z,
                        -0.035
                    ) &&
                    close(foot.soleRotation.w, 1.0),
                "derived G1 sole frame is incorrect"
            );
            for (std::size_t point = 0; point < 4u; ++point) {
                const MRShapeGPU& shape =
                    model.shapes[foot.sphereShapeIndices[point]];
                require(
                    shape.bodyIndex == expectedBody &&
                        shape.shapeType == MR_SHAPE_SPHERE &&
                        close(shape.dimensions.x, 0.005) &&
                        close(
                            shape.localPosition.x +
                                model.bodies[expectedBody]
                                    .centerOfMass.x,
                            kExpectedFootCenters[point][0]
                        ) &&
                        close(
                            shape.localPosition.y +
                                model.bodies[expectedBody]
                                    .centerOfMass.y,
                            kExpectedFootCenters[point][1]
                        ) &&
                        close(
                            shape.localPosition.z +
                                model.bodies[expectedBody]
                                    .centerOfMass.z,
                            kExpectedFootCenters[point][2]
                        ),
                    "official foot contact sphere is incorrect"
                );
            }
        }
        for (std::size_t shape = 8u; shape < 12u; ++shape) {
            require(
                model.shapes[shape].shapeType == MR_SHAPE_CYLINDER &&
                    (
                        model.shapes[shape].flags &
                        MR_SHAPE_FLAG_SIMULATION_DISABLED
                    ) != 0u,
                "unsupported shoulder cylinder was not retained and disabled"
            );
        }

        require(
            metadata.imus[0].bodyIndex == 0u &&
                metadata.imus[0].name == "imu_in_pelvis" &&
                metadata.imus[0].transport == "LowState.imu_state" &&
                close(
                    metadata.imus[0].localPosition.x +
                        model.bodies[0].centerOfMass.x,
                    0.04525
                ) &&
                close(
                    metadata.imus[0].localPosition.z +
                        model.bodies[0].centerOfMass.z,
                    -0.08339
                ),
            "pelvis IMU metadata is incorrect"
        );
        require(
            metadata.imus[1].bodyIndex == 15u &&
                metadata.imus[1].name == "imu_in_torso" &&
                metadata.imus[1].transport == "rt/secondary_imu" &&
                close(
                    metadata.imus[1].localPosition.x +
                        model.bodies[15].centerOfMass.x,
                    -0.03959
                ) &&
                close(
                    metadata.imus[1].localPosition.y +
                        model.bodies[15].centerOfMass.y,
                    -0.00224
                ) &&
                close(
                    metadata.imus[1].localPosition.z +
                        model.bodies[15].centerOfMass.z,
                    0.14792
                ),
            "torso IMU metadata is incorrect"
        );
        for (const metalrobo::G1ImuFrame& imu : metadata.imus) {
            require(
                close(imu.localRotation.x, 0.0) &&
                    close(imu.localRotation.y, 0.0) &&
                    close(imu.localRotation.z, 0.0) &&
                    close(imu.localRotation.w, 1.0) &&
                    close(imu.gyroscopeNoise, 5.0e-4) &&
                    close(imu.gyroscopeCutoff, 34.9) &&
                    close(imu.accelerometerNoise, 1.0e-2) &&
                    close(imu.accelerometerCutoff, 157.0),
                "IMU transform or named MJCF noise preset is incorrect"
            );
        }

        std::cout << std::fixed << std::setprecision(8)
                  << "model=\"" << model.name << "\""
                  << " mode_machine=" << metadata.modeMachine
                  << " mode_pr=" << metadata.modePr
                  << " bodies=" << model.world.bodyCount
                  << " joints=" << model.world.jointCount
                  << " nq=" << model.world.nq
                  << " nv=" << model.world.nv
                  << " root_com_z=" << model.defaultQ[2]
                  << " mass_kg=" << totalMass
                  << " primitive_shapes=" << model.world.shapeCount
                  << " executable_shapes="
                  << metalrobo::kUnitreeG1ExecutableShapeCount
                  << " foot_spheres=8"
                  << " imus=" << metadata.imus.size()
                  << " max_inverse_error=" << std::scientific
                  << maximumInverseError
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_g1_model_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
