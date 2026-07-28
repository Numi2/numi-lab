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
        std::uint32_t environmentCapacity,
        MetalWorldActuationMode actuationMode,
        MetalWorldSolverMode solverMode,
        std::uint32_t velocityIterations,
        std::uint32_t finalVelocityIterations,
        MetalWorldCCDMode ccdMode,
        std::uint32_t maxCCDAdvanceSolvePasses,
        std::uint32_t maxCCDZeroTimeReplays,
        float ccdSimultaneousTolerance,
        std::vector<MRBodyStateGPU> defaultSceneBodies,
        std::string metallibPath
    );
    ~MLXCompiledWorld();

    MLXCompiledWorld(const MLXCompiledWorld&) = delete;
    MLXCompiledWorld& operator=(const MLXCompiledWorld&) = delete;

    [[nodiscard]] const CompiledWorld& world() const noexcept;
    [[nodiscard]] float controlTimestep() const noexcept;
    [[nodiscard]] std::uint32_t physicsSubsteps() const noexcept;
    [[nodiscard]] bool applyBodyDamping() const noexcept;
    [[nodiscard]] std::uint32_t environmentCapacity() const noexcept;
    [[nodiscard]] MetalWorldActuationMode actuationMode() const noexcept;
    [[nodiscard]] MetalWorldSolverMode solverMode() const noexcept;
    [[nodiscard]] std::uint32_t velocityIterations() const noexcept;
    [[nodiscard]] std::uint32_t
    finalVelocityIterations() const noexcept;
    [[nodiscard]] MetalWorldCCDMode ccdMode() const noexcept;
    [[nodiscard]] std::uint32_t
    maxCCDAdvanceSolvePasses() const noexcept;
    [[nodiscard]] std::uint32_t
    maxCCDZeroTimeReplays() const noexcept;
    [[nodiscard]] float ccdSimultaneousTolerance() const noexcept;
    [[nodiscard]] const std::vector<MRBodyStateGPU>&
    defaultSceneBodies() const noexcept;
    [[nodiscard]] const std::string& metallibPath() const noexcept;
    [[nodiscard]] std::vector<float> defaultQ() const;
    [[nodiscard]] std::vector<float> defaultV() const;
    [[nodiscard]] std::vector<float> effortLimits() const;

    void prepareStream(mx::StreamOrDevice stream = {});
    MetalResources& resources(mx::metal::Device& device);

private:
    CompiledWorld world_;
    float controlTimestep_ = 0.0f;
    std::uint32_t physicsSubsteps_ = 0u;
    bool applyBodyDamping_ = true;
    std::uint32_t environmentCapacity_ = 0u;
    MetalWorldActuationMode actuationMode_ =
        MetalWorldActuationMode::effort;
    MetalWorldSolverMode solverMode_ =
        MetalWorldSolverMode::freeMotionABA;
    std::uint32_t velocityIterations_ = 1u;
    std::uint32_t finalVelocityIterations_ = 1u;
    MetalWorldCCDMode ccdMode_ = MetalWorldCCDMode::disabled;
    std::uint32_t maxCCDAdvanceSolvePasses_ =
        MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES;
    std::uint32_t maxCCDZeroTimeReplays_ =
        MR_CCD_DEFAULT_ZERO_TIME_REPLAYS;
    float ccdSimultaneousTolerance_ = 1.0e-5f;
    std::vector<MRBodyStateGPU> defaultSceneBodies_;
    std::string metallibPath_;
    std::mutex resourceMutex_;
    std::unique_ptr<MetalResources> resources_;
};

[[nodiscard]] std::shared_ptr<MLXCompiledWorld> compileWorld(
    const std::string& model,
    const std::string& scene,
    std::uint32_t environmentCapacity,
    MetalWorldCapacityProfile capacityProfile,
    float controlTimestep,
    std::uint32_t physicsSubsteps,
    bool applyBodyDamping,
    const std::string& actuationMode,
    const std::string& solverMode,
    std::uint32_t velocityIterations,
    std::uint32_t finalVelocityIterations,
    const std::string& ccdMode,
    std::uint32_t maxCCDAdvanceSolvePasses,
    std::uint32_t maxCCDZeroTimeReplays,
    float ccdSimultaneousTolerance,
    const std::string& metallibPath,
    mx::StreamOrDevice stream = {}
);

[[nodiscard]] std::vector<mx::array> abaStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& effort,
    mx::StreamOrDevice stream = {}
);

[[nodiscard]] std::vector<mx::array> worldStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& effort,
    const mx::array& scenePosition,
    const mx::array& sceneOrientation,
    const mx::array& sceneLinearVelocity,
    const mx::array& sceneAngularVelocity,
    const mx::array& manifoldHeaders,
    const mx::array& manifoldPoints,
    const mx::array& manifoldCounts,
    const mx::array& pairCache,
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

class WorldStepPrimitive final : public mx::Primitive {
public:
    WorldStepPrimitive(
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
