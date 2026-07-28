#include "metalrobo/Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Vec3 {
    double x;
    double y;
    double z;
};

Vec3 rotate(const mr_float4 q, const Vec3 v) {
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 uv{
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x,
    };
    const Vec3 uuv{
        u.y * uv.z - u.z * uv.y,
        u.z * uv.x - u.x * uv.z,
        u.x * uv.y - u.y * uv.x,
    };
    return {
        v.x + 2.0 * (q.w * uv.x + uuv.x),
        v.y + 2.0 * (q.w * uv.y + uuv.y),
        v.z + 2.0 * (q.w * uv.z + uuv.z),
    };
}

Vec3 endEffector(const metalrobo::CpuState& state) {
    const mr_float4 position = state.bodyPositions.back();
    const Vec3 flangeOffset =
        rotate(state.bodyRotations.back(), {0.0, 0.0, 0.107});
    return {
        position.x + flangeOffset.x,
        position.y + flangeOffset.y,
        position.z + flangeOffset.z,
    };
}

double distance(const Vec3 a, const mr_float4 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool sameReset(
    const metalrobo::CpuState& left,
    const metalrobo::CpuState& right
) {
    return left.q == right.q && left.qd == right.qd &&
        left.target.x == right.target.x && left.target.y == right.target.y &&
        left.target.z == right.target.z;
}

} // namespace

int main() {
    try {
        const metalrobo::Model model = metalrobo::makeFrankaPandaModel();
        std::string reason;
        if (!model.valid(&reason)) {
            throw std::runtime_error("model validation failed: " + reason);
        }

        metalrobo::CpuState state;
        metalrobo::CpuState repeated;
        constexpr std::uint64_t seed = 0x4d4554414cULL;
        metalrobo::resetCpuState(model, state, seed);
        metalrobo::resetCpuState(model, repeated, seed);
        if (!sameReset(state, repeated)) {
            throw std::runtime_error("same-seed reset was not deterministic");
        }

        double movingMass = 0.0;
        for (std::size_t index = 1; index < model.links.size(); ++index) {
            movingMass += model.links[index].massAndCOMX.x;
        }
        const Vec3 initialEndEffector = endEffector(state);
        const double initialError = distance(initialEndEffector, state.target);
        double maximumSpeed = 0.0;
        double maximumTorque = 0.0;
        double maximumAcceleration = 0.0;
        std::vector<float> minimumQ = state.q;
        std::vector<float> maximumQ = state.q;
        std::vector<float> actions(model.gpu.actionCount, 0.0f);

        constexpr std::uint32_t steps = 600;
        for (std::uint32_t step = 0; step < steps; ++step) {
            for (std::size_t joint = 0; joint < actions.size(); ++joint) {
                const double phase =
                    static_cast<double>(step) * 0.012 +
                    static_cast<double>(joint) * 0.61;
                actions[joint] = static_cast<float>(0.30 * std::sin(phase));
            }
            metalrobo::stepCpu(model, state, actions);
            for (std::size_t joint = 0; joint < state.q.size(); ++joint) {
                if (!std::isfinite(state.q[joint]) ||
                    !std::isfinite(state.qd[joint]) ||
                    !std::isfinite(state.qdd[joint]) ||
                    !std::isfinite(state.torque[joint])) {
                    throw std::runtime_error(
                        "non-finite state at control step " +
                        std::to_string(step)
                    );
                }
                if (state.q[joint] < model.joints[joint].limits.x - 1.0e-5f ||
                    state.q[joint] > model.joints[joint].limits.y + 1.0e-5f) {
                    throw std::runtime_error("joint limit escaped");
                }
                minimumQ[joint] = std::min(minimumQ[joint], state.q[joint]);
                maximumQ[joint] = std::max(maximumQ[joint], state.q[joint]);
                maximumSpeed =
                    std::max(
                        maximumSpeed,
                        std::abs(static_cast<double>(state.qd[joint]))
                    );
                maximumTorque =
                    std::max(
                        maximumTorque,
                        std::abs(static_cast<double>(state.torque[joint]))
                    );
                maximumAcceleration =
                    std::max(
                        maximumAcceleration,
                        std::abs(static_cast<double>(state.qdd[joint]))
                    );
            }
        }

        const Vec3 finalEndEffector = endEffector(state);
        const double finalError = distance(finalEndEffector, state.target);
        std::cout << std::fixed << std::setprecision(4)
                  << "model=\"" << model.name << "\""
                  << " dof=" << model.gpu.dofCount
                  << " links=" << model.gpu.linkCount
                  << " spheres=" << model.gpu.colliderCount
                  << " moving_mass_kg=" << movingMass
                  << " control_hz="
                  << 1.0 / model.gpu.gravityAndTimestep.w
                  << " physics_hz="
                  << model.gpu.substeps /
                         model.gpu.gravityAndTimestep.w
                  << '\n'
                  << "deterministic_reset=yes"
                  << " steps=" << state.step
                  << " finite=yes"
                  << " max_speed_rad_s=" << maximumSpeed
                  << " max_torque_nm=" << maximumTorque
                  << " max_accel_rad_s2=" << maximumAcceleration
                  << '\n'
                  << "target=(" << state.target.x << ',' << state.target.y
                  << ',' << state.target.z << ")"
                  << " ee_initial=(" << initialEndEffector.x << ','
                  << initialEndEffector.y << ',' << initialEndEffector.z
                  << ") error_initial_m=" << initialError
                  << " ee_final=(" << finalEndEffector.x << ','
                  << finalEndEffector.y << ',' << finalEndEffector.z
                  << ") error_final_m=" << finalError << '\n'
                  << "joint_ranges_rad=";
        for (std::size_t joint = 0; joint < minimumQ.size(); ++joint) {
            if (joint != 0) {
                std::cout << ',';
            }
            std::cout << '[' << minimumQ[joint] << ',' << maximumQ[joint]
                      << ']';
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_cpu_probe: " << error.what() << '\n';
        return 1;
    }
}
