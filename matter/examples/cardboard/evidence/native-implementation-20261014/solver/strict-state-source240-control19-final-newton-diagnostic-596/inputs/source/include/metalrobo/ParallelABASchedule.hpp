#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/parallel_aba_shared.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

enum class ParallelABAScheduleStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    capacityOverflow,
    invalidCompiledSchedule,
    allocationFailure,
};

struct ParallelABAScheduleDiagnostics {
    ParallelABAScheduleStatus status =
        ParallelABAScheduleStatus::success;
    std::uint32_t articulationCount = 0u;
    std::uint32_t maximumDepth = 0u;
    std::uint32_t maximumLevelWidth = 0u;
    std::uint32_t firstFailingArticulation = MR_INVALID_INDEX;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ParallelABAScheduleStatus::success;
    }
};

// Pointer-free immutable topology for multi-kernel Metal ABA. Forward levels
// expose parent-complete bodies; reverse levels expose child-complete bodies
// and parent-owned deterministic reductions. This is deliberately state-free
// so every environment and MLX stream can share one private GPU copy.
struct ParallelABASchedule {
    std::vector<MRParallelABAArticulationGPU> articulations;
    std::vector<MRParallelABALevelGPU> levels;
    std::vector<MRParallelABAParentReductionGPU> parentReductions;
    std::vector<std::uint32_t> levelBodies;
    std::vector<std::uint32_t> bodyOrder;
    std::vector<std::uint32_t> parentLocal;
    std::vector<std::uint32_t> inboundJoint;
    std::vector<std::uint32_t> childOffsets;
    std::vector<std::uint32_t> childIndices;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

[[nodiscard]] ParallelABAScheduleDiagnostics
compileParallelABASchedule(
    const EngineModel& model,
    ParallelABASchedule& output
);

[[nodiscard]] const char* parallelABAScheduleStatusName(
    ParallelABAScheduleStatus status
) noexcept;

} // namespace metalrobo
