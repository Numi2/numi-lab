#pragma once

#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
#include "metalrobo/Tactile.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/c_api.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/r2s2r_types.h"
#include "metalrobo/scene_query_types.h"
#include "metalrobo/world_compiler_types.h"

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
struct MetalGeneralizedResources;

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
        std::uint32_t waveWorkerGroups,
        std::vector<MRBodyStateGPU> defaultSceneBodies,
        CookedTactileSystem tactile,
        std::uint64_t authoredPackHash,
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
    [[nodiscard]] std::uint32_t waveWorkerGroups() const noexcept;
    [[nodiscard]] const std::vector<MRBodyStateGPU>&
    defaultSceneBodies() const noexcept;
    [[nodiscard]] const CookedTactileSystem& tactile()
        const noexcept;
    [[nodiscard]] bool hasTactile() const noexcept;
    [[nodiscard]] std::uint64_t authoredPackHash() const noexcept;
    [[nodiscard]] const std::string& metallibPath() const noexcept;
    [[nodiscard]] std::vector<float> defaultQ() const;
    [[nodiscard]] std::vector<float> defaultV() const;
    [[nodiscard]] std::vector<float> effortLimits() const;
    [[nodiscard]] std::vector<float>
    defaultActuatorTargets() const;
    [[nodiscard]] std::vector<float>
    actuatorProfileValues() const;
    [[nodiscard]] std::vector<std::uint32_t>
    actuatorProfileFlags() const;

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
    std::uint32_t waveWorkerGroups_ = 0u;
    std::vector<MRBodyStateGPU> defaultSceneBodies_;
    CookedTactileSystem tactile_;
    std::uint64_t authoredPackHash_ = 0u;
    std::string metallibPath_;
    std::mutex resourceMutex_;
    std::unique_ptr<MetalResources> resources_;
};

class MLXCompiledMultiArticulatedProgram {
public:
    MLXCompiledMultiArticulatedProgram(
        CompiledMetalMultiArticulatedProgram program,
        std::uint32_t environmentCapacity,
        MetalMultiArticulatedConstraintConfig config,
        std::string metallibPath
    );
    ~MLXCompiledMultiArticulatedProgram();

    MLXCompiledMultiArticulatedProgram(
        const MLXCompiledMultiArticulatedProgram&
    ) = delete;
    MLXCompiledMultiArticulatedProgram& operator=(
        const MLXCompiledMultiArticulatedProgram&
    ) = delete;

    [[nodiscard]] const CompiledMetalMultiArticulatedProgram&
    program() const noexcept;
    [[nodiscard]] std::uint32_t environmentCapacity()
        const noexcept;
    [[nodiscard]] const MetalMultiArticulatedConstraintConfig&
    config() const noexcept;
    [[nodiscard]] const std::string& metallibPath()
        const noexcept;
    [[nodiscard]] std::vector<float> defaultQ() const;
    [[nodiscard]] std::vector<float> defaultV() const;

    void prepareStream(mx::StreamOrDevice stream = {});
    MetalGeneralizedResources& resources(
        mx::metal::Device& device
    );

private:
    CompiledMetalMultiArticulatedProgram program_;
    std::uint32_t environmentCapacity_ = 0u;
    MetalMultiArticulatedConstraintConfig config_{};
    std::string metallibPath_;
    std::mutex resourceMutex_;
    std::unique_ptr<MetalGeneralizedResources> resources_;
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
    std::uint32_t waveWorkerGroups,
    const std::string& metallibPath,
    mx::StreamOrDevice stream = {}
);

[[nodiscard]] std::shared_ptr<MLXCompiledWorld> compileWorldPack(
    const std::string& path,
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
    std::uint32_t waveWorkerGroups,
    const std::string& metallibPath,
    mx::StreamOrDevice stream = {}
);

[[nodiscard]] std::shared_ptr<
    MLXCompiledMultiArticulatedProgram
> compileMultiArticulatedProgram(
    const std::string& model,
    std::uint32_t environmentCapacity,
    const std::string& solverMode,
    std::uint32_t solverIterations,
    float convergenceTolerance,
    float timestep,
    const std::string& metallibPath,
    mx::StreamOrDevice stream = {}
);

[[nodiscard]] std::vector<mx::array>
generalizedConstraintStep(
    const std::shared_ptr<
        MLXCompiledMultiArticulatedProgram
    >& program,
    const mx::array& q,
    const mx::array& freeVelocity,
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
    const mx::array& rodPositions,
    const mx::array& rodVelocities,
    const mx::array& rodTwists,
    const mx::array& rodTwistRates,
    const mx::array& rodWitnessCache,
    const mx::array& bodyParameters,
    const mx::array& controllerParameters,
    const mx::array& tactilePreviousDepth,
    const mx::array& tactilePreviousValidity,
    const mx::array& tactilePreviousObject,
    const mx::array& tactilePreviousMotion,
    const mx::array& tactileTargetAnchor,
    const mx::array& tactileFrameIndex,
    const mx::array& tactileTimestamp,
    const mx::array& resetMask,
    const mx::array& actuatorProfileValues,
    mx::StreamOrDevice stream = {}
);

// Imports the GPU-private reset state produced by MetalWorldFamilyContext
// directly into MLX arrays on the active Metal command encoder.
[[nodiscard]] std::vector<mx::array> worldFamilyState(
    const std::shared_ptr<MLXCompiledWorld>& world,
    std::uintptr_t resetQBuffer,
    std::uintptr_t resetVBuffer,
    std::uintptr_t resetSceneBodiesBuffer,
    std::uintptr_t scenarioHeadersBuffer,
    std::uintptr_t scenarioValuesBuffer,
    std::uintptr_t bodyParametersBuffer,
    std::uintptr_t controllerParametersBuffer,
    std::uint32_t environmentCount,
    std::uint32_t variationCount,
    std::uint32_t bodyCount,
    std::uint32_t articulationCount,
    std::uint64_t generation,
    std::uint64_t authoredPackHash,
    mx::StreamOrDevice stream = {}
);

// Writes policy-ready visual modalities directly into MLX-owned arrays on
// MLX's active Metal compute encoder. Body records use the native
// environment-major MRBodyStateGPU layout.
[[nodiscard]] std::vector<mx::array> visualObservation(
    const std::shared_ptr<MLXCompiledWorld>& world,
    std::uintptr_t rendererHandle,
    std::uintptr_t worldFamilyHandle,
    const mx::array& currentBodyStates,
    const mx::array& previousBodyStates,
    std::uint64_t frameIndex,
    std::uint32_t sensorSequence,
    std::uint32_t cameraIndex,
    std::uint32_t outputMask,
    mx::StreamOrDevice stream = {}
);

// Materializes the complete authoritative rigid state without publishing it
// through host memory. The resulting uint32 view is the native
// environment-major MRBodyStateGPU record layout shared by visual and scene
// query primitives.
[[nodiscard]] std::vector<mx::array> materializeBodyStates(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& scenePosition,
    const mx::array& sceneOrientation,
    const mx::array& sceneLinearVelocity,
    const mx::array& sceneAngularVelocity,
    mx::StreamOrDevice stream = {}
);

// Casts arbitrary world-space rays against the cooked physics geometry and
// returns metric distance, world point/normal, stable identities, and
// validity directly as MLX arrays.
[[nodiscard]] std::vector<mx::array> sceneRaycast(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& bodyStates,
    const mx::array& origins,
    const mx::array& directions,
    const mx::array& maximumDistances,
    const mx::array& options,
    mx::StreamOrDevice stream = {}
);

// Casts a graph-static ray pattern whose origins and directions are expressed
// in each ray's parent-body frame. The Metal kernel composes body motion and
// traversal without materializing environment-major world-space ray arrays.
[[nodiscard]] std::vector<mx::array> sceneRaycastPattern(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& bodyStates,
    const mx::array& parentBodies,
    const mx::array& localOrigins,
    const mx::array& localDirections,
    const mx::array& maximumDistances,
    const mx::array& options,
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

class GeneralizedConstraintStepPrimitive final
    : public mx::Primitive {
public:
    GeneralizedConstraintStepPrimitive(
        mx::Stream stream,
        std::shared_ptr<
            MLXCompiledMultiArticulatedProgram
        > program
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
    std::shared_ptr<
        MLXCompiledMultiArticulatedProgram
    > program_;
};

class WorldFamilyStatePrimitive final : public mx::Primitive {
public:
    WorldFamilyStatePrimitive(
        mx::Stream stream,
        std::shared_ptr<MLXCompiledWorld> world,
        MTL::Buffer* resetQ,
        MTL::Buffer* resetV,
        MTL::Buffer* resetSceneBodies,
        MTL::Buffer* scenarioHeaders,
        MTL::Buffer* scenarioValues,
        MTL::Buffer* bodyParameters,
        MTL::Buffer* controllerParameters,
        std::uint32_t environmentCount,
        std::uint32_t variationCount,
        std::uint32_t bodyCount,
        std::uint32_t articulationCount,
        std::uint64_t generation
    );
    ~WorldFamilyStatePrimitive() override;

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
    MTL::Buffer* resetQ_ = nullptr;
    MTL::Buffer* resetV_ = nullptr;
    MTL::Buffer* resetSceneBodies_ = nullptr;
    MTL::Buffer* scenarioHeaders_ = nullptr;
    MTL::Buffer* scenarioValues_ = nullptr;
    MTL::Buffer* bodyParameters_ = nullptr;
    MTL::Buffer* controllerParameters_ = nullptr;
    std::uint32_t environmentCount_ = 0u;
    std::uint32_t variationCount_ = 0u;
    std::uint32_t bodyCount_ = 0u;
    std::uint32_t articulationCount_ = 0u;
    std::uint64_t generation_ = 0u;
};

class VisualObservationPrimitive final : public mx::Primitive {
public:
    VisualObservationPrimitive(
        mx::Stream stream,
        std::shared_ptr<MLXCompiledWorld> world,
        MRHybridRendererHandle* renderer,
        MRWorldFamilyHandle* worlds,
        std::uint32_t environmentCount,
        std::uint32_t bodyCount,
        std::uint32_t width,
        std::uint32_t height,
        std::uint64_t frameIndex,
        std::uint32_t sensorSequence,
        std::uint32_t cameraIndex,
        std::uint32_t outputMask
    );
    ~VisualObservationPrimitive() override;

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
    MRHybridRendererHandle* renderer_ = nullptr;
    MRWorldFamilyHandle* worlds_ = nullptr;
    std::uint32_t environmentCount_ = 0u;
    std::uint32_t bodyCount_ = 0u;
    std::uint32_t width_ = 0u;
    std::uint32_t height_ = 0u;
    std::uint64_t frameIndex_ = 0u;
    std::uint32_t sensorSequence_ = 0u;
    std::uint32_t cameraIndex_ = 0u;
    std::uint32_t outputMask_ = MR_HYBRID_OUTPUT_ALL_TRUTH;
};

class BodyStatePrimitive final : public mx::Primitive {
public:
    BodyStatePrimitive(
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

class SceneRaycastPrimitive final : public mx::Primitive {
public:
    SceneRaycastPrimitive(
        mx::Stream stream,
        std::shared_ptr<MLXCompiledWorld> world,
        std::uint32_t environmentCount,
        std::uint32_t rayCount,
        bool mountedPattern
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
    std::uint32_t environmentCount_ = 0u;
    std::uint32_t rayCount_ = 0u;
    bool mountedPattern_ = false;
};

} // namespace metalrobo::mlx_ext
