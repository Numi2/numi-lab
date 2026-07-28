#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo {

struct EngineModelComponent {
    const EngineModel* model = nullptr;
    // Stable instance identity participates in composed ConstraintIR keys.
    std::string_view instanceId;
};

struct EngineModelComposeConfig {
    std::string name = "composed_world";
    mr_float4 gravityAndTimestep{
        0.0f,
        0.0f,
        -9.81f,
        1.0f / 1000.0f,
    };
    mr_float4 solverScales{
        1.0e-7f,
        1.0e-9f,
        2.0f,
        1.0e-5f,
    };
    std::uint32_t solverType = MR_SOLVER_THROUGHPUT_TGS;
    std::uint32_t frictionConeType =
        MR_FRICTION_CONE_ELLIPTIC;
    // Deterministic raw-contact budget per eligible pair.
    std::uint32_t contactsPerPair = 4u;
    std::uint32_t constraintHeadroom = 64u;
};

enum class EngineModelComposeStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidComponent,
    duplicateInstanceId,
    capacityOverflow,
    constraintCompilationFailure,
    invalidComposedModel,
    allocationFailure,
};

struct EngineModelComposeDiagnostics {
    EngineModelComposeStatus status =
        EngineModelComposeStatus::success;
    std::uint32_t componentCount = 0u;
    std::uint32_t articulationCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t shapeCount = 0u;
    std::uint32_t geometryCount = 0u;
    std::uint32_t constraintBlockCount = 0u;
    std::uint32_t firstFailingComponent = MR_INVALID_INDEX;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == EngineModelComposeStatus::success;
    }
};

// Transactionally concatenates heterogeneous canonical models. Every global
// index and nested geometry-arena reference is rebased; ConstraintIR blocks
// are recompiled under stable instance namespaces. Configuration state is
// concatenated without hidden transforms, so each component's defaultQ keeps
// its authored world pose (including floating roots).
[[nodiscard]] EngineModelComposeDiagnostics composeEngineModels(
    std::span<const EngineModelComponent> components,
    EngineModel& output,
    const EngineModelComposeConfig& config = {}
);

[[nodiscard]] const char* engineModelComposeStatusName(
    EngineModelComposeStatus status
) noexcept;

} // namespace metalrobo
