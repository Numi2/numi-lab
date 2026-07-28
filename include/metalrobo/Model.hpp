#pragma once

#include "metalrobo/gpu_types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct Model {
    MRModelGPU gpu{};
    std::vector<MRJointGPU> joints;
    std::vector<MRLinkGPU> links;
    std::vector<MRColliderGPU> colliders;
    std::vector<float> homePosition;
    std::string name;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

[[nodiscard]] Model makeFrankaPandaModel();

struct CpuState {
    std::vector<float> q;
    std::vector<float> qd;
    std::vector<float> qdd;
    std::vector<float> torque;
    std::vector<mr_float4> bodyPositions;
    std::vector<mr_float4> bodyRotations;
    mr_float4 target{};
    std::uint32_t step = 0;
};

void resetCpuState(const Model& model, CpuState& state, std::uint64_t seed);
void stepCpu(
    const Model& model,
    CpuState& state,
    std::span<const float> normalizedActions
);

} // namespace metalrobo
