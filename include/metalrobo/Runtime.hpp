#pragma once

#include "metalrobo/Model.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace metalrobo {

struct RuntimeStats {
    double lastGpuMilliseconds = 0.0;
    double totalGpuMilliseconds = 0.0;
    std::uint64_t controlSteps = 0;
    std::uint64_t physicsSteps = 0;
};

struct RuntimeDescriptor {
    std::uint32_t environmentCount = 1024;
    std::uint64_t seed = 1;
    bool autoReset = true;
    bool captureBodyPoses = true;
    std::string metallibPath;
};

class Runtime {
public:
    virtual ~Runtime() = default;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] virtual const Model& model() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t environmentCount() const noexcept = 0;
    [[nodiscard]] virtual std::span<const float> observations() const noexcept = 0;
    [[nodiscard]] virtual std::span<const float> rewards() const noexcept = 0;
    [[nodiscard]] virtual std::span<const std::uint8_t> terminated() const noexcept = 0;
    [[nodiscard]] virtual std::span<const float> bodyPositions() const noexcept = 0;
    [[nodiscard]] virtual std::span<const float> bodyRotations() const noexcept = 0;
    [[nodiscard]] virtual RuntimeStats stats() const noexcept = 0;
    [[nodiscard]] virtual std::string deviceName() const = 0;

    virtual void reset(std::uint64_t seed) = 0;
    virtual void step(std::span<const float> normalizedActions) = 0;

protected:
    Runtime() = default;
};

[[nodiscard]] std::unique_ptr<Runtime> makeMetalRuntime(
    Model model,
    const RuntimeDescriptor& descriptor
);

} // namespace metalrobo
