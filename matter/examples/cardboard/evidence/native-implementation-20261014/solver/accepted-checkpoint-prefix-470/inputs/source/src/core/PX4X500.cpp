#include "metalrobo/PX4X500.hpp"

#include <stdexcept>

namespace metalrobo {
namespace { mr_float4 f4(float x, float y, float z, float w = 0.0f) { return {x, y, z, w}; } }

MRMulticopterModelGPU makePX4X500MulticopterModel(const float physicsTimestep) {
    if (!(physicsTimestep > 0.0f)) throw std::invalid_argument("PX4 X500 timestep must be positive");
    MRMulticopterModelGPU model{};
    model.rotorCount = 4u;
    model.coefficients = f4(8.54858e-06f, 0.016f, 8.06428e-05f, 1.0e-06f);
    model.motorAndTimestep = f4(0.0125f, 0.025f, 1000.0f, physicsTimestep);
    return model;
}

std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS> makePX4X500Rotors() {
    std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS> rotors{};
    // SDF link poses and ccw/ccw/cw/cw motor declarations at the pinned PX4
    // revision. Gazebo applies reaction torque as -turningDirection * thrust
    // * momentConstant, therefore ccw uses -1 and cw uses +1 here.
    rotors[0].positionAndReactionSign = f4(0.174f, -0.174f, 0.058130869f, -1.0f);
    rotors[1].positionAndReactionSign = f4(-0.174f, 0.174f, 0.058130869f, -1.0f);
    rotors[2].positionAndReactionSign = f4(0.174f, 0.174f, 0.058130869f, 1.0f);
    rotors[3].positionAndReactionSign = f4(-0.174f, -0.174f, 0.058130869f, 1.0f);
    return rotors;
}

MRBodyPropertiesGPU makePX4X500BodyProperties() {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = MR_MOTION_DYNAMIC;
    // Composite about the source-model total COM: base link plus all four
    // rotor link masses/inertias transformed from their SDF poses. Rotor spin
    // angular momentum remains a later coupling term; this preserves the
    // source rigid airframe inertia rather than adding only rotor mass.
    constexpr float mass = 2.0643077f;
    body.massAndInverseMass = f4(mass, 1.0f / mass, 0.0f);
    body.inertiaRow0 = f4(0.023839481f, 0.0f, 0.0f);
    body.inertiaRow1 = f4(0.0f, 0.023942405f, 0.0f);
    body.inertiaRow2 = f4(0.0f, 0.0f, 0.043999955f);
    body.inverseInertiaRow0 = f4(1.0f / body.inertiaRow0.x, 0.0f, 0.0f);
    body.inverseInertiaRow1 = f4(0.0f, 1.0f / body.inertiaRow1.y, 0.0f);
    body.inverseInertiaRow2 = f4(0.0f, 0.0f, 1.0f / body.inertiaRow2.z);
    body.dampingAndSpeedLimits = f4(0.0f, 0.0f, 100.0f, 100.0f);
    return body;
}
} // namespace metalrobo
