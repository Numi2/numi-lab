#pragma once

#include "metalrobo/engine_types.h"

#include <string>
#include <vector>

namespace metalrobo {

// Canonical, compiled engine model. Unlike the v0 Franka Model, generalized
// configuration and velocity storage are explicitly different (nq != nv).
// Runtime buffers are separate; this object contains immutable topology and a
// complete reset state.
struct EngineModel {
    MRWorldGPU world{};
    std::vector<MRArticulationGPU> articulations;
    std::vector<MRJointDescriptorGPU> joints;
    // Exactly one authoritative record per generalized velocity coordinate,
    // stored in global v order.
    std::vector<MRDofPropertiesGPU> dofs;
    std::vector<MRBodyPropertiesGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    std::vector<MRMaterialGPU> materials;
    std::vector<float> defaultQ;
    std::vector<float> defaultV;
    std::string name;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

// Minimal floating/free-body scene used to prove the canonical nq=7, nv=6
// path independently from the fixed-base Franka compatibility runtime.
[[nodiscard]] EngineModel makeFreeSphereEngineModel();

} // namespace metalrobo
