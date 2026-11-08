// Generate a zero-action policy bound to the test runtime's reported contract.
// This is an inference-path regression fixture, not a trained policy.
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: policy OUTPUT WORLD TASK OBS ACTION INPUTS OUTPUTS\n";
        return 2;
    }
    metalrobo::PolicyPack policy;
    policy.id = "clean_horizon_zero_policy";
    policy.contract = {1u, std::strtoull(argv[2], nullptr, 10),
        std::strtoull(argv[3], nullptr, 10), std::strtoull(argv[4], nullptr, 10),
        std::strtoull(argv[5], nullptr, 10)};
    const auto inputs = static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const auto outputs = static_cast<std::uint32_t>(std::strtoul(argv[7], nullptr, 10));
    policy.layers.push_back({.inputCount = inputs, .outputCount = outputs,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(static_cast<std::size_t>(inputs) * outputs, 0.0f),
        .bias = std::vector<float>(outputs, 0.0f)});
    const auto result = metalrobo::writePolicyPack(policy, argv[1]);
    if (!result.succeeded()) {
        std::cerr << result.message << "\n";
        return 1;
    }
    return 0;
}
