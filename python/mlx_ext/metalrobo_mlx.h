#pragma once

#include "metalrobo/MetalWorld.hpp"

#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace metalrobo::mlx_ext {

struct MetalResources;

class MLXCompiledWorld {
public:
    MLXCompiledWorld(
        CompiledWorld world,
        float controlTimestep,
        std::uint32_t physicsSubsteps,
        bool applyBodyDamping,
        std::string metallibPath
    );
    ~MLXCompiledWorld();

    MLXCompiledWorld(const MLXCompiledWorld&) = delete;
    MLXCompiledWorld& operator=(const MLXCompiledWorld&) = delete;

    [[nodiscard]] const CompiledWorld& world() const noexcept;
    [[nodiscard]] float controlTimestep() const noexcept;
    [[nodiscard]] std::uint32_t physicsSubsteps() const noexcept;
    [[nodiscard]] bool applyBodyDamping() const noexcept;
    [[nodiscard]] const std::string& metallibPath() const noexcept;
    [[nodiscard]] std::vector<float> defaultQ() const;
    [[nodiscard]] std::vector<float> defaultV() const;
    [[nodiscard]] std::vector<float> effortLimits() const;

    MetalResources& resources(mx::metal::Device& device);

private:
    CompiledWorld world_;
    float controlTimestep_ = 0.0f;
    std::uint32_t physicsSubsteps_ = 0u;
    bool applyBodyDamping_ = true;
    std::string metallibPath_;
    std::mutex resourceMutex_;
    std::unique_ptr<MetalResources> resources_;
};

[[nodiscard]] std::shared_ptr<MLXCompiledWorld> compileWorld(
    const std::string& model,
    float controlTimestep,
    std::uint32_t physicsSubsteps,
    bool applyBodyDamping,
    const std::string& metallibPath
);

[[nodiscard]] std::vector<mx::array> abaStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& effort,
    mx::StreamOrDevice stream = {}
);

// Synchronous FP64 validation oracle. It is intentionally separate from the
// custom primitive and is never reachable from the MLX execution path.
[[nodiscard]] std::vector<float> debugCPUStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const std::vector<float>& q,
    const std::vector<float>& v,
    const std::vector<float>& effort
);

class ABAWorldStepPrimitive final : public mx::Primitive {
public:
    ABAWorldStepPrimitive(
        mx::Stream stream,
        std::shared_ptr<MLXCompiledWorld> world
    );

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs
    ) override;

    std::vector<mx::array> jvp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums
    ) override;
    std::vector<mx::array> vjp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs
    ) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes
    ) override;

    [[nodiscard]] const char* name() const override;
    [[nodiscard]] bool is_equivalent(
        const mx::Primitive& other
    ) const override;

private:
    std::shared_ptr<MLXCompiledWorld> world_;
};

} // namespace metalrobo::mlx_ext
