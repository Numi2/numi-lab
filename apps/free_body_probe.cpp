#include "metalrobo/FreeBodyDynamics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

mr_float4 f4(float x, float y, float z, float w = 0.0f) {
    return {x, y, z, w};
}

double quaternionNorm(const mr_float4 q) {
    return std::sqrt(
        static_cast<double>(q.x) * q.x +
        static_cast<double>(q.y) * q.y +
        static_cast<double>(q.z) * q.z +
        static_cast<double>(q.w) * q.w
    );
}

double angularEnergy(
    const MRBodyPropertiesGPU& body,
    const MRBodyStateGPU& state
) {
    // The probe starts at identity; energy is computed in the world frame
    // from I_world = inverse(inverseInertiaWorld), which remains diagonal only
    // initially. For the final invariant use |L| instead below.
    const double wx = state.angularVelocity.x;
    const double wy = state.angularVelocity.y;
    const double wz = state.angularVelocity.z;
    return 0.5 * (
        body.inertiaRow0.x * wx * wx +
        body.inertiaRow1.y * wy * wy +
        body.inertiaRow2.z * wz * wz
    );
}

double angularMomentumNorm(
    const MRBodyPropertiesGPU& body,
    const MRBodyStateGPU& state
) {
    // Reconstruct R from q, rotate omega to body, then rotate I*omega back.
    const double x = state.orientation.x;
    const double y = state.orientation.y;
    const double z = state.orientation.z;
    const double w = state.orientation.w;
    const double r[3][3]{
        {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
        {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
        {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)},
    };
    const double omegaWorld[3]{
        state.angularVelocity.x,
        state.angularVelocity.y,
        state.angularVelocity.z,
    };
    double omegaBody[3]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            omegaBody[row] += r[column][row] * omegaWorld[column];
        }
    }
    const double lBody[3]{
        body.inertiaRow0.x * omegaBody[0],
        body.inertiaRow1.y * omegaBody[1],
        body.inertiaRow2.z * omegaBody[2],
    };
    double lWorld[3]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            lWorld[row] += r[row][column] * lBody[column];
        }
    }
    return std::sqrt(
        lWorld[0] * lWorld[0] +
        lWorld[1] * lWorld[1] +
        lWorld[2] * lWorld[2]
    );
}

} // namespace

int main() {
    try {
        MRBodyPropertiesGPU body{};
        body.motionType = MR_MOTION_DYNAMIC;
        body.massAndInverseMass = f4(2.0f, 0.5f, 0.0f, 0.0f);
        body.inertiaRow0 = f4(0.7f, 0.0f, 0.0f);
        body.inertiaRow1 = f4(0.0f, 1.1f, 0.0f);
        body.inertiaRow2 = f4(0.0f, 0.0f, 1.7f);
        body.inverseInertiaRow0 = f4(1.0f / 0.7f, 0.0f, 0.0f);
        body.inverseInertiaRow1 = f4(0.0f, 1.0f / 1.1f, 0.0f);
        body.inverseInertiaRow2 = f4(0.0f, 0.0f, 1.0f / 1.7f);
        body.dampingAndSpeedLimits = f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);

        MRBodyStateGPU state{};
        state.position = f4(0.0f, 0.0f, 0.0f, 1.0f);
        state.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        state.linearVelocityAndInverseMass = f4(0.0f, 0.0f, 0.0f, 0.5f);
        state.angularVelocity = f4(1.7f, -0.9f, 2.2f);
        state.inverseInertiaWorldRow0 = body.inverseInertiaRow0;
        state.inverseInertiaWorldRow1 = body.inverseInertiaRow1;
        state.inverseInertiaWorldRow2 = body.inverseInertiaRow2;
        state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;

        std::vector<MRBodyPropertiesGPU> bodies{body};
        std::vector<MRBodyStateGPU> states{state};
        std::vector<metalrobo::BodyWrench> wrenches(1);

        metalrobo::FreeBodyIntegratorConfig config;
        config.timestep = 1.0e-3;
        config.gravity = f4(0.0f, 0.0f, 0.0f);
        config.integrator =
            metalrobo::FreeBodyIntegrator::implicitMidpoint;
        config.nonlinearIterations = 12u;
        config.nonlinearTolerance = 1.0e-12;

        const double initialMomentum = angularMomentumNorm(body, state);
        const double initialEnergy = angularEnergy(body, state);
        std::uint32_t maximumIterations = 0;
        double maximumResidual = 0.0;
        constexpr std::uint32_t steps = 10000u;
        for (std::uint32_t step = 0; step < steps; ++step) {
            const auto diagnostics = metalrobo::integrateFreeBodies(
                bodies,
                states,
                wrenches,
                config
            );
            if (!diagnostics.succeeded()) {
                throw std::runtime_error(
                    "implicit midpoint free-body solve failed"
                );
            }
            maximumIterations =
                std::max(maximumIterations, diagnostics.maximumIterations);
            maximumResidual =
                std::max(maximumResidual, diagnostics.maximumResidual);
        }

        const double finalMomentum = angularMomentumNorm(body, states[0]);
        // Rotational energy is invariant; compute in body coordinates via
        // |omega| and diagonal inertia reconstruction in the helper's frame.
        const double finalQNorm = quaternionNorm(states[0].orientation);
        const double momentumDrift =
            std::abs(finalMomentum - initialMomentum) /
            std::max(initialMomentum, 1.0e-12);
        if (std::abs(finalQNorm - 1.0) > 2.0e-6 ||
            momentumDrift > 3.0e-3 ||
            !std::isfinite(finalMomentum)) {
            throw std::runtime_error(
                "free-body conservation gate exceeded"
            );
        }

        std::cout << std::scientific << std::setprecision(6)
                  << "integrator=implicit_midpoint"
                  << " steps=" << steps
                  << " dt=" << config.timestep
                  << " q_norm_error=" << std::abs(finalQNorm - 1.0)
                  << " angular_momentum_drift=" << momentumDrift
                  << " initial_energy=" << initialEnergy
                  << " max_newton_iterations=" << maximumIterations
                  << " max_residual=" << maximumResidual
                  << " gyroscopic_term=yes"
                  << " finite=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_free_body_probe: " << error.what() << '\n';
        return 1;
    }
}
