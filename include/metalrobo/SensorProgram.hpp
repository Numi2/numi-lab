#pragma once

#include "metalrobo/Tactile.hpp"
#include "metalrobo/WorldCompiler.hpp"
#include "metalrobo/runtime_abi_generated.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo {

class CompiledWorld;

enum class SensorExecutionDomain : std::uint32_t {
    nativeState = 0u,
    presentation = 1u,
    tactile = 2u,
};

enum class SensorCompileStatus : std::uint32_t {
    success = 0u,
    invalidWorld,
    invalidSpec,
    unresolvedSemantic,
    duplicateSemantic,
    invalidTactileSystem,
    arithmeticOverflow,
    internalFailure,
};

struct SensorCompileDiagnostics {
    SensorCompileStatus status = SensorCompileStatus::success;
    std::uint64_t fingerprint = 0u;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SensorCompileStatus::success;
    }
};

struct SensorProgramLayout {
    std::uint32_t sensorCount = 0u;
    std::uint32_t outputElementCount = 0u;
    std::uint32_t historyElementCount = 0u;
    std::uint32_t tactileSampleCount = 0u;
    std::uint32_t nativeStateSensorCount = 0u;
    std::uint32_t presentationSensorCount = 0u;
    std::uint32_t tactileSensorCount = 0u;
    std::uint32_t maximumHistoryLength = 0u;
};

// Immutable, pointer-free sensor schedule compiled against one exact world.
// Authored names are retained only for diagnostics and policy/task contract
// compilation; the GPU sees generated descriptors and stable indices.
class CompiledSensorProgram {
public:
    CompiledSensorProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t worldFingerprint() const noexcept;
    [[nodiscard]] const SensorProgramLayout& layout() const noexcept;
    [[nodiscard]] const MRSensorProgramHeaderGPU& header() const noexcept;
    [[nodiscard]] std::span<const std::string> sensorIds() const noexcept;
    [[nodiscard]] std::span<const MRSensorDescriptorGPU>
    descriptors() const noexcept;
    [[nodiscard]] const CookedTactileSystem& tactileSystem() const noexcept;
    [[nodiscard]] std::uint32_t sensorIndex(
        std::string_view id
    ) const noexcept;

private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;

    friend SensorCompileDiagnostics compileSensorProgram(
        std::span<const SensorSpec>,
        const CookedTactileSystem&,
        const CompiledWorld&,
        CompiledSensorProgram&
    );
};

// Transactional: output is unchanged on every failure.
[[nodiscard]] SensorCompileDiagnostics compileSensorProgram(
    std::span<const SensorSpec> sensors,
    const CookedTactileSystem& tactile,
    const CompiledWorld& world,
    CompiledSensorProgram& output
);

[[nodiscard]] const char* sensorCompileStatusName(
    SensorCompileStatus status
) noexcept;

} // namespace metalrobo
