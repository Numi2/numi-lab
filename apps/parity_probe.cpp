#include "metalrobo/Model.hpp"
#include "metalrobo/Runtime.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        metalrobo::Model model = metalrobo::makeFrankaPandaModel();
        metalrobo::RuntimeDescriptor descriptor;
        descriptor.environmentCount = 1;
        descriptor.seed = 0x504152495459ULL;
        descriptor.autoReset = false;
        descriptor.captureBodyPoses = true;
        std::unique_ptr<metalrobo::Runtime> gpu =
            metalrobo::makeMetalRuntime(model, descriptor);

        metalrobo::CpuState cpu;
        metalrobo::resetCpuState(model, cpu, descriptor.seed);
        const std::span<const float> initial = gpu->observations();
        for (std::size_t joint = 0; joint < model.joints.size(); ++joint) {
            cpu.q[joint] = initial[joint];
            cpu.qd[joint] = initial[model.gpu.dofCount + joint];
            cpu.qdd[joint] = 0.0f;
            cpu.torque[joint] = 0.0f;
        }

        std::vector<float> actions(model.gpu.actionCount);
        for (std::size_t joint = 0; joint < actions.size(); ++joint) {
            actions[joint] =
                0.2f * std::sin(0.37f * static_cast<float>(joint + 1));
        }

        gpu->step(actions);
        metalrobo::stepCpu(model, cpu, actions);
        const std::span<const float> gpuObservation = gpu->observations();

        double maximumPositionError = 0.0;
        double maximumVelocityError = 0.0;
        for (std::size_t joint = 0; joint < model.joints.size(); ++joint) {
            maximumPositionError = std::max(
                maximumPositionError,
                std::abs(
                    static_cast<double>(gpuObservation[joint]) -
                    static_cast<double>(cpu.q[joint])
                )
            );
            maximumVelocityError = std::max(
                maximumVelocityError,
                std::abs(
                    static_cast<double>(
                        gpuObservation[model.gpu.dofCount + joint]
                    ) -
                    static_cast<double>(cpu.qd[joint])
                )
            );
        }

        if (!std::isfinite(maximumPositionError) ||
            !std::isfinite(maximumVelocityError)) {
            throw std::runtime_error(
                "CPU/Metal comparison produced a non-finite error"
            );
        }

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << gpu->deviceName() << "\" "
                  << "control_steps=1 "
                  << "physics_substeps=" << model.gpu.substeps << ' '
                  << "max_q_error_rad=" << maximumPositionError << ' '
                  << "max_qd_error_rad_s=" << maximumVelocityError << '\n';

        // The paths use different precision and algebraic factorizations.
        // These bounds are a smoke-level convention/equation check, not a
        // claim of bitwise equivalence.
        if (maximumPositionError > 5.0e-3 ||
            maximumVelocityError > 5.0e-1) {
            throw std::runtime_error(
                "CPU/Metal one-step trajectory disagreement is too large"
            );
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_parity_probe: " << error.what() << '\n';
        return 1;
    }
}
