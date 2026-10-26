#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/EngineModel.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

// Compiler inputs are the final cooked FEM nodal masses and positions,
// expressed in the donor's original COM frame. Never use source mesh volume
// or the pre-cook payload mass as a substitute for this discrete mass measure.
struct NumiHumanTissueMassNode {
    // One unique FEM-node row. In the legacy API donorBody/localPosition also
    // describe the mass source; with explicit contributions they instead name
    // the node's single registered frame owner and anchor point.
    std::uint32_t nodeIndex = MR_INVALID_INDEX;
    std::uint32_t donorBody = MR_INVALID_INDEX;
    double massKg = 0.0;
    std::array<double, 3> localPosition{};
};

// A piece of a FEM node's authoritative lumped mass, attributed to its actual
// source rigid body and source-local point for rigid mass/inertia debit. Several
// contributions may share a nodeIndex while the node/frame table stays unique.
struct NumiHumanTissueMassContribution {
    std::uint32_t nodeIndex = MR_INVALID_INDEX;
    std::uint32_t donorBody = MR_INVALID_INDEX;
    double massKg = 0.0;
    std::array<double, 3> localPosition{};
};

struct NumiHumanTissueMassPartition {
    std::uint32_t donorBody = MR_INVALID_INDEX;
    std::uint32_t nodeCount = 0u;
    double tissueMassKg = 0.0;
    std::array<double, 3> tissueFirstMomentKgM{};
    // Integral x*x^T dm about the ORIGINAL donor COM, row-major.
    std::array<double, 9> tissueSecondMomentKgM2{};
    // New rigid COM in the old COM frame. Every joint anchor, muscle site,
    // wrap, contact, attachment and sensor point must subtract this offset.
    std::array<double, 3> remainingCOMOffsetM{};
    MRBodyPropertiesGPU sourceBody{};
    MRBodyPropertiesGPU remainingBody{};
    double packedMomentRelativeError = 0.0;
};

struct NumiHumanTissueMassResult {
    std::vector<NumiHumanTissueMassPartition> partitions;
    std::string error;
    [[nodiscard]] bool succeeded() const noexcept { return error.empty(); }
};

// Atomic compiler operation: no partial partitions are returned on failure.
// This computes mass ownership, not force ownership or anatomical validity.
// Body axes are retained; a non-diagonal residual COM inertia is intentional.
[[nodiscard]] NumiHumanTissueMassResult compileNumiHumanTissueMassPartition(
    std::span<const MRBodyPropertiesGPU> bodies,
    std::span<const NumiHumanTissueMassNode> nodes);

// Explicit source-mass path. The node span is the unique physical FEM-node/frame
// table; contributions may split one node's mass across original donors and
// source-local points. Their mass must sum to each node's mass.
[[nodiscard]] NumiHumanTissueMassResult compileNumiHumanTissueMassPartition(
    std::span<const MRBodyPropertiesGPU> bodies,
    std::span<const NumiHumanTissueMassNode> nodes,
    std::span<const NumiHumanTissueMassContribution> contributions);

// Rebase canonical mechanics atomically. External muscle/sensor/attachment
// packs must consume the same offsets before composing a CompiledRun. Spatial
// ConstraintIR rows need their own recompilation and are rejected here.
[[nodiscard]] bool rebaseNumiHumanTissueMassPartition(
    const EngineModel& source, std::span<const NumiHumanTissueMassPartition> partitions,
    EngineModel& result, std::string& error);

} // namespace metalrobo
