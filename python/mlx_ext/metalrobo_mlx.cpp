#include "metalrobo_mlx.h"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/GeometryCooker.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"
#include "metalrobo/SurgicalWorld.hpp"
#include "metalrobo/WorldPack.hpp"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo::mlx_ext {

namespace {

constexpr std::uint32_t kABAThreads = 32u;
constexpr std::uint32_t kStatusWords =
    sizeof(MRMLXWorldStepStatusGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kABAStatusWords =
    sizeof(MRABAStatusGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kContactStatusWords =
    sizeof(MRMetalWorldContactStatusGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kGeneralizedStatusWords =
    sizeof(MRGeneralizedConstraintStatusGPU) /
    sizeof(std::uint32_t);
constexpr std::uint32_t kInverseMassStatusWords =
    sizeof(MRInverseMassStatusGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kManifoldHeaderWords =
    sizeof(MRManifoldHeaderGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kManifoldPointWords =
    sizeof(MRManifoldPointGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kConvexCacheWords =
    sizeof(MRConvexQueryCacheGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kRodWitnessWords =
    sizeof(MRRodToolWitnessGPU) / sizeof(std::uint32_t);
constexpr std::uint32_t kEvidenceFloatWidth = 16u;
constexpr std::uint32_t kEvidenceIDWidth = 4u;
constexpr std::uint32_t kWorldThreads = 256u;
constexpr std::uint32_t kOperatorThreads = 128u;

enum ImmutableBufferIndex : std::size_t {
    kImmutableWorld = 0u,
    kImmutableArticulations = 1u,
    kImmutableJoints = 2u,
    kImmutableDofs = 3u,
    kImmutableBodies = 4u,
    kImmutableEmptyWrench = 5u,
    kImmutableShapes = 6u,
    kImmutableMaterials = 7u,
    kImmutableSceneBodyIndices = 8u,
    kImmutableEligiblePairs = 9u,
    kImmutableGeometryHeaders = 10u,
    kImmutableGeometryVertices = 11u,
    kImmutableMeshNodes = 12u,
    kImmutableMeshTriangles = 13u,
    kImmutableDynamicNodes = 14u,
    kImmutableBodyDynamicNodes = 15u,
    kImmutableRodColliders = 16u,
    kImmutableRodShapeSources = 17u,
    kImmutableRodToolPairs = 18u,
    kImmutableRodRestLengths = 19u,
    kImmutableRodRestTwists = 20u,
    kImmutableRodRestCurvatures = 21u,
    kImmutableRodInverseMasses = 22u,
    kImmutableRodInverseTwistInertias = 23u,
    kImmutableRodStretchStiffness = 24u,
    kImmutableRodBendStiffness = 25u,
    kImmutableRodTwistStiffness = 26u,
    kImmutableAuthoredIRBlocks = 27u,
    kImmutableAuthoredIREndpoints = 28u,
    kImmutableAuthoredIRRows = 29u,
    kImmutableAuthoredIRCones = 30u,
    kImmutableAuthoredIRWarmImpulses = 31u,
    kImmutableConvexFaces = 32u,
    kImmutableActuatorProfiles = 33u,
    kImmutableTactileSensors = 34u,
    kImmutableTactileSamples = 35u,
    kImmutableTactileTargets = 36u,
    kImmutableTactileShapeToSensor = 37u,
};

enum GeneralizedImmutableBufferIndex : std::size_t {
    kGeneralizedWorld = 0u,
    kGeneralizedArticulations = 1u,
    kGeneralizedJoints = 2u,
    kGeneralizedDofs = 3u,
    kGeneralizedBodies = 4u,
    kGeneralizedRows = 5u,
    kGeneralizedWarmImpulses = 6u,
    kGeneralizedJacobian = 7u,
    kGeneralizedInverseDispatches = 8u,
    kGeneralizedRhs = 9u,
    kGeneralizedScheduleArticulations = 10u,
    kGeneralizedScheduleLevels = 11u,
    kGeneralizedScheduleReductions = 12u,
    kGeneralizedScheduleLevelBodies = 13u,
    kGeneralizedScheduleParents = 14u,
    kGeneralizedScheduleInboundJoints = 15u,
    kGeneralizedScheduleChildOffsets = 16u,
    kGeneralizedScheduleChildIndices = 17u,
};

std::string currentBinaryDirectory() {
    static const std::string directory = [] {
        Dl_info information{};
        if (dladdr(
                reinterpret_cast<void*>(
                    &currentBinaryDirectory
                ),
                &information
            ) == 0 ||
            information.dli_fname == nullptr) {
            throw std::runtime_error(
                "could not locate the MetalRobo MLX extension"
            );
        }
        return std::filesystem::path(
            information.dli_fname
        ).parent_path().string();
    }();
    return directory;
}

template <typename T>
mx::allocator::Buffer immutableBuffer(
    const T* data,
    const std::size_t count
) {
    const std::size_t logicalBytes = count * sizeof(T);
    const std::size_t allocationBytes =
        std::max<std::size_t>(logicalBytes, sizeof(T));
    auto buffer = mx::allocator::malloc(allocationBytes);
    auto* metalBuffer =
        static_cast<MTL::Buffer*>(buffer.ptr());
    if (metalBuffer == nullptr ||
        metalBuffer->contents() == nullptr) {
        throw std::runtime_error(
            "MLXCompiledWorld could not allocate immutable Metal storage"
        );
    }
    std::memset(
        metalBuffer->contents(),
        0,
        allocationBytes
    );
    if (logicalBytes != 0u) {
        std::memcpy(
            metalBuffer->contents(),
            data,
            logicalBytes
        );
    }
    return buffer;
}

mx::array temporary(
    const mx::Shape& shape,
    const mx::Dtype dtype
) {
    std::size_t count = 1u;
    for (const auto dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return mx::array(
        mx::allocator::malloc(count * mx::size_of(dtype)),
        shape,
        dtype
    );
}

void validateInput(
    const mx::array& value,
    const mx::Shape& expected,
    const char* label
) {
    if (value.dtype() != mx::float32 ||
        value.shape() != expected) {
        throw std::invalid_argument(
            std::string(label) +
            " must be a float32 MLX array with the compiled shape"
        );
    }
}

using MlxRodVec3 = std::array<double, 3u>;

MlxRodVec3 mlxRodSub(
    const MlxRodVec3& a,
    const MlxRodVec3& b
) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

MlxRodVec3 mlxRodAdd(
    const MlxRodVec3& a,
    const MlxRodVec3& b
) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

MlxRodVec3 mlxRodScale(
    const MlxRodVec3& value,
    const double scale
) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double mlxRodDot(
    const MlxRodVec3& a,
    const MlxRodVec3& b
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

MlxRodVec3 mlxRodCross(
    const MlxRodVec3& a,
    const MlxRodVec3& b
) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

bool mlxRodNormalize(
    const MlxRodVec3& input,
    MlxRodVec3& output
) {
    const double squared = mlxRodDot(input, input);
    if (!(squared > 1.0e-28) || !std::isfinite(squared)) {
        return false;
    }
    output = mlxRodScale(input, 1.0 / std::sqrt(squared));
    return std::all_of(
        output.begin(),
        output.end(),
        [](const double value) {
            return std::isfinite(value);
        }
    );
}

MlxRodVec3 mlxRodRotate(
    const MlxRodVec3& vector,
    const MlxRodVec3& axis,
    const double angle
) {
    return mlxRodAdd(
        mlxRodAdd(
            mlxRodScale(vector, std::cos(angle)),
            mlxRodScale(
                mlxRodCross(axis, vector),
                std::sin(angle)
            )
        ),
        mlxRodScale(
            axis,
            mlxRodDot(axis, vector) *
                (1.0 - std::cos(angle))
        )
    );
}

MlxRodVec3 mlxRodLeastAligned(
    const MlxRodVec3& tangent
) {
    const std::array<MlxRodVec3, 3u> axes{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    std::size_t selected = 0u;
    for (std::size_t axis = 1u; axis < axes.size(); ++axis) {
        if (std::abs(mlxRodDot(tangent, axes[axis])) <
            std::abs(mlxRodDot(tangent, axes[selected]))) {
            selected = axis;
        }
    }
    MlxRodVec3 result = mlxRodSub(
        axes[selected],
        mlxRodScale(
            tangent,
            mlxRodDot(axes[selected], tangent)
        )
    );
    static_cast<void>(mlxRodNormalize(result, result));
    return result;
}

bool mlxRodTransport(
    const MlxRodVec3& director,
    const MlxRodVec3& from,
    const MlxRodVec3& to,
    MlxRodVec3& output
) {
    const MlxRodVec3 axis = mlxRodCross(from, to);
    const double sine = std::sqrt(mlxRodDot(axis, axis));
    const double cosine =
        std::clamp(mlxRodDot(from, to), -1.0, 1.0);
    if (sine <= 1.0e-14) {
        if (cosine < 0.0) {
            return false;
        }
        output = director;
        return true;
    }
    output = mlxRodRotate(
        director,
        mlxRodScale(axis, 1.0 / sine),
        std::atan2(sine, cosine)
    );
    output = mlxRodSub(
        output,
        mlxRodScale(to, mlxRodDot(output, to))
    );
    return mlxRodNormalize(output, output);
}

bool mlxRodRestCurvature(
    const DiscreteElasticRodModel& model,
    const std::size_t vertex,
    mr_float4& output
) {
    MlxRodVec3 left;
    MlxRodVec3 right;
    if (!mlxRodNormalize(
            mlxRodSub(
                model.restPositions[vertex + 1u],
                model.restPositions[vertex]
            ),
            left
        ) ||
        !mlxRodNormalize(
            mlxRodSub(
                model.restPositions[vertex + 2u],
                model.restPositions[vertex + 1u]
            ),
            right
        )) {
        return false;
    }
    const MlxRodVec3 referenceLeft =
        mlxRodLeastAligned(left);
    MlxRodVec3 referenceRight;
    if (!mlxRodTransport(
            referenceLeft,
            left,
            right,
            referenceRight
        )) {
        return false;
    }
    const MlxRodVec3 directorLeft = mlxRodRotate(
        referenceLeft,
        left,
        model.restTwists[vertex]
    );
    const MlxRodVec3 directorRight = mlxRodRotate(
        referenceRight,
        right,
        model.restTwists[vertex + 1u]
    );
    const MlxRodVec3 secondLeft =
        mlxRodCross(left, directorLeft);
    const MlxRodVec3 secondRight =
        mlxRodCross(right, directorRight);
    const double denominator = 1.0 + mlxRodDot(left, right);
    if (!(denominator > 1.0e-8) ||
        !std::isfinite(denominator)) {
        return false;
    }
    const MlxRodVec3 binormal = mlxRodScale(
        mlxRodCross(left, right),
        2.0 / denominator
    );
    output = {
        static_cast<float>(
            0.5 * mlxRodDot(
                binormal,
                mlxRodAdd(secondLeft, secondRight)
            )
        ),
        static_cast<float>(
            -0.5 * mlxRodDot(
                binormal,
                mlxRodAdd(directorLeft, directorRight)
            )
        ),
        0.0f,
        0.0f,
    };
    return std::isfinite(output.x) &&
        std::isfinite(output.y);
}

std::vector<MRMultiInverseMassDispatchGPU>
generalizedInverseDispatches(
    const CompiledMetalMultiArticulatedProgram& program,
    const std::uint32_t environments
) {
    const EngineModel& model = program.model();
    const std::size_t rowCount = program.rowCount();
    std::vector<MRMultiInverseMassDispatchGPU> result;
    result.reserve(
        program.rowChunkOffsets().size() *
        model.articulations.size()
    );
    for (std::size_t chunk = 0u;
         chunk < program.rowChunkOffsets().size();
         ++chunk) {
        const std::uint32_t rowBegin =
            program.rowChunkOffsets()[chunk];
        const std::uint32_t rowCountInChunk =
            program.rowChunkCounts()[chunk];
        for (std::uint32_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            MRMultiInverseMassDispatchGPU work{};
            work.dispatch.articulationIndex =
                articulationIndex;
            work.dispatch.environmentCount = environments;
            work.dispatch.rhsCount = rowCountInChunk;
            work.dispatch.qStride = model.world.nq;
            work.dispatch.rhsEnvironmentStride =
                static_cast<std::uint32_t>(
                    rowCount * model.world.nv
                );
            work.dispatch.rhsVectorStride = model.world.nv;
            work.dispatch.outputEnvironmentStride =
                work.dispatch.rhsEnvironmentStride;
            work.dispatch.outputVectorStride = model.world.nv;
            work.qBase = articulation.qOffset;
            work.rhsBase =
                rowBegin * model.world.nv +
                articulation.vOffset;
            work.outputBase = work.rhsBase;
            work.statusBase =
                static_cast<std::uint32_t>(
                    result.size() * environments
                );
            result.push_back(work);
        }
    }
    return result;
}

std::vector<float> generalizedRhs(
    const CompiledMetalMultiArticulatedProgram& program,
    const std::uint32_t environments
) {
    const auto jacobian = program.generalizedJacobian();
    std::vector<float> result(
        static_cast<std::size_t>(environments) *
        jacobian.size()
    );
    for (std::uint32_t environment = 0u;
         environment < environments;
         ++environment) {
        std::copy(
            jacobian.begin(),
            jacobian.end(),
            result.begin() +
                static_cast<std::size_t>(environment) *
                    jacobian.size()
        );
    }
    return result;
}

struct AuthoredPhysicsBase {
    std::uint32_t primaryArticulation = MR_INVALID_INDEX;
    std::vector<MRBodyStateGPU> sceneBodies;
};

AuthoredPhysicsBase authoredPhysicsBase(
    const WorldTemplate& worldTemplate
) {
    const std::uint32_t robotIndex =
        worldTemplate.assetIndex(
            worldTemplate.task.robotAssetId
        );
    if (robotIndex == MR_INVALID_INDEX) {
        throw std::invalid_argument(
            "authored world pack has no task robot asset"
        );
    }
    const WorldAsset& robot =
        worldTemplate.assets[robotIndex];
    if (robot.articulationIndex == MR_INVALID_INDEX ||
        robot.articulationIndex >=
            worldTemplate.engineModel.articulations.size()) {
        throw std::invalid_argument(
            "authored task robot has no executable articulation"
        );
    }

    AuthoredPhysicsBase result;
    result.primaryArticulation = robot.articulationIndex;
    std::vector<std::uint32_t> bodyToScene(
        worldTemplate.engineModel.bodies.size(),
        MR_INVALID_INDEX
    );
    for (std::uint32_t body = 0u;
         body < worldTemplate.engineModel.bodies.size();
         ++body) {
        const MRBodyPropertiesGPU& properties =
            worldTemplate.engineModel.bodies[body];
        if (properties.articulationIndex != MR_INVALID_INDEX) {
            continue;
        }
        bodyToScene[body] =
            static_cast<std::uint32_t>(
                result.sceneBodies.size()
            );
        MRBodyStateGPU state{};
        state.position.w = 1.0f;
        state.orientation.w = 1.0f;
        state.flagsAndIndices[0] = properties.motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = body;
        result.sceneBodies.push_back(state);
    }
    for (const WorldAsset& asset : worldTemplate.assets) {
        for (const std::uint32_t body : asset.bodyIndices) {
            if (body >= bodyToScene.size()) {
                throw std::invalid_argument(
                    "authored asset body exceeds the engine topology"
                );
            }
            const std::uint32_t scene = bodyToScene[body];
            if (scene == MR_INVALID_INDEX) {
                continue;
            }
            MRBodyStateGPU& state = result.sceneBodies[scene];
            state.position = {
                asset.initialPose.position.x,
                asset.initialPose.position.y,
                asset.initialPose.position.z,
                1.0f,
            };
            state.orientation = asset.initialPose.orientation;
        }
    }
    return result;
}

std::vector<MRActuatorProfileGPU> mlxExecutionActuatorProfiles(
    const EngineModel& model
) {
    std::vector<MRActuatorProfileGPU> profiles =
        model.actuatorProfiles.empty()
        ? std::vector<MRActuatorProfileGPU>(model.dofs.size())
        : model.actuatorProfiles;
    for (std::size_t index = 0u;
         index < profiles.size();
         ++index) {
        const MRDofPropertiesGPU& dof = model.dofs[index];
        MRActuatorProfileGPU& profile = profiles[index];
        if ((profile.identity.y &
             MR_ACTUATOR_PROFILE_ACTIVE) != 0u) {
            continue;
        }
        profile.motorAndSpeed = {
            0.0f,
            0.0f,
            std::numeric_limits<float>::max(),
            1.0f,
        };
        profile.transmissionAndEnvelope = {};
        profile.transmissionAndEnvelope.z =
            (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                dof.limits.w > 0.0f
            ? dof.limits.w
            : std::numeric_limits<float>::max();
        profile.identity = {
            static_cast<mr_u32>(index),
            0u,
            0u,
            0u,
        };
    }
    return profiles;
}

} // namespace

struct MetalDeviceTuningProfile {
    std::uint64_t registryID = 0u;
    std::uint32_t appleGPUFamily = 0u;
    std::uint32_t waveWorkerGroupCount = 32u;
};

MetalDeviceTuningProfile tuningProfile(
    mx::metal::Device& device
) {
    MetalDeviceTuningProfile profile{};
    MTL::Device* metalDevice = device.mtl_device();
    if (metalDevice == nullptr) {
        return profile;
    }
    profile.registryID = metalDevice->registryID();
    for (std::uint32_t family = 10u;
         family != 0u;
         --family) {
        const auto candidate = static_cast<MTL::GPUFamily>(
            static_cast<std::uint32_t>(
                MTL::GPUFamilyApple1
            ) +
            family - 1u
        );
        if (metalDevice->supportsFamily(candidate)) {
            profile.appleGPUFamily = family;
            break;
        }
    }
    // The local 10-core Apple M4 was measured explicitly across
    // 32/64/96/128 groups on 2026-07-28; 96 won the 1,024-environment
    // Franka contact workload. Other Apple9/10 devices retain the
    // conservative profile until measured. The grid is fixed before lazy
    // evaluation; no rollout-time autotuning or synchronization is allowed.
    constexpr std::uint64_t measuredM4RegistryID = 4294968246ull;
    profile.waveWorkerGroupCount =
        profile.registryID == measuredM4RegistryID
        ? 96u
        : profile.appleGPUFamily >= 9u ? 64u : 32u;
    return profile;
}

struct MetalResources {
    mx::metal::Device* device = nullptr;
    MetalDeviceTuningProfile tuning{};
    std::vector<mx::allocator::Buffer> buffers;
    MTL::ComputePipelineState* abaKernel = nullptr;
    MTL::ComputePipelineState* multiABAKernel = nullptr;
    MTL::ComputePipelineState* commitKernel = nullptr;
    std::unordered_map<
        std::string,
        MTL::ComputePipelineState*
    > kernels;
    std::size_t bodyToSceneBufferIndex = 0u;

    ~MetalResources() {
        for (const auto buffer : buffers) {
            mx::allocator::free(buffer);
        }
    }

    [[nodiscard]] MTL::Buffer* buffer(
        const std::size_t index
    ) const {
        return static_cast<MTL::Buffer*>(
            const_cast<void*>(buffers.at(index).ptr())
        );
    }

    [[nodiscard]] MTL::ComputePipelineState* kernel(
        const std::string& name
    ) const {
        const auto found = kernels.find(name);
        if (found == kernels.end() || found->second == nullptr) {
            throw std::runtime_error(
                "MLX MetalRobo kernel is unavailable: " + name
            );
        }
        return found->second;
    }
};

struct MetalGeneralizedResources {
    mx::metal::Device* device = nullptr;
    std::vector<mx::allocator::Buffer> buffers;
    MTL::ComputePipelineState* inverseKernel = nullptr;
    MTL::ComputePipelineState* delassusKernel = nullptr;
    MTL::ComputePipelineState* solveKernel = nullptr;
    MTL::ComputePipelineState* commitKernel = nullptr;
    std::uint32_t inverseWorkCount = 0u;

    ~MetalGeneralizedResources() {
        for (const auto buffer : buffers) {
            mx::allocator::free(buffer);
        }
    }

    [[nodiscard]] MTL::Buffer* buffer(
        const std::size_t index
    ) const {
        return static_cast<MTL::Buffer*>(
            const_cast<void*>(buffers.at(index).ptr())
        );
    }
};

MLXCompiledWorld::MLXCompiledWorld(
    CompiledWorld world,
    const float controlTimestep,
    const std::uint32_t physicsSubsteps,
    const bool applyBodyDamping,
    const std::uint32_t environmentCapacity,
    const MetalWorldActuationMode actuationMode,
    const MetalWorldSolverMode solverMode,
    const std::uint32_t velocityIterations,
    const std::uint32_t finalVelocityIterations,
    const MetalWorldCCDMode ccdMode,
    const std::uint32_t maxCCDAdvanceSolvePasses,
    const std::uint32_t maxCCDZeroTimeReplays,
    const float ccdSimultaneousTolerance,
    const std::uint32_t waveWorkerGroups,
    std::vector<MRBodyStateGPU> defaultSceneBodies,
    CookedTactileSystem tactile,
    const std::uint64_t authoredPackHash,
    std::string metallibPath
)
    : world_(std::move(world)),
      controlTimestep_(controlTimestep),
      physicsSubsteps_(physicsSubsteps),
      applyBodyDamping_(applyBodyDamping),
      environmentCapacity_(environmentCapacity),
      actuationMode_(actuationMode),
      solverMode_(solverMode),
      velocityIterations_(velocityIterations),
      finalVelocityIterations_(finalVelocityIterations),
      ccdMode_(ccdMode),
      maxCCDAdvanceSolvePasses_(maxCCDAdvanceSolvePasses),
      maxCCDZeroTimeReplays_(maxCCDZeroTimeReplays),
      ccdSimultaneousTolerance_(ccdSimultaneousTolerance),
      waveWorkerGroups_(waveWorkerGroups),
      defaultSceneBodies_(std::move(defaultSceneBodies)),
      tactile_(std::move(tactile)),
      authoredPackHash_(authoredPackHash),
      metallibPath_(std::move(metallibPath)) {}

MLXCompiledWorld::~MLXCompiledWorld() = default;

const CompiledWorld& MLXCompiledWorld::world() const noexcept {
    return world_;
}

float MLXCompiledWorld::controlTimestep() const noexcept {
    return controlTimestep_;
}

std::uint32_t MLXCompiledWorld::physicsSubsteps() const noexcept {
    return physicsSubsteps_;
}

bool MLXCompiledWorld::applyBodyDamping() const noexcept {
    return applyBodyDamping_;
}

std::uint32_t MLXCompiledWorld::environmentCapacity() const noexcept {
    return environmentCapacity_;
}

MetalWorldActuationMode
MLXCompiledWorld::actuationMode() const noexcept {
    return actuationMode_;
}

MetalWorldSolverMode MLXCompiledWorld::solverMode() const noexcept {
    return solverMode_;
}

std::uint32_t MLXCompiledWorld::velocityIterations() const noexcept {
    return velocityIterations_;
}

std::uint32_t
MLXCompiledWorld::finalVelocityIterations() const noexcept {
    return finalVelocityIterations_;
}

MetalWorldCCDMode MLXCompiledWorld::ccdMode() const noexcept {
    return ccdMode_;
}

std::uint32_t
MLXCompiledWorld::maxCCDAdvanceSolvePasses() const noexcept {
    return maxCCDAdvanceSolvePasses_;
}

std::uint32_t
MLXCompiledWorld::maxCCDZeroTimeReplays() const noexcept {
    return maxCCDZeroTimeReplays_;
}

float MLXCompiledWorld::ccdSimultaneousTolerance() const noexcept {
    return ccdSimultaneousTolerance_;
}

std::uint32_t
MLXCompiledWorld::waveWorkerGroups() const noexcept {
    return waveWorkerGroups_;
}

const std::vector<MRBodyStateGPU>&
MLXCompiledWorld::defaultSceneBodies() const noexcept {
    return defaultSceneBodies_;
}

const CookedTactileSystem&
MLXCompiledWorld::tactile() const noexcept {
    return tactile_;
}

bool MLXCompiledWorld::hasTactile() const noexcept {
    return !tactile_.sensors.empty();
}

std::uint64_t
MLXCompiledWorld::authoredPackHash() const noexcept {
    return authoredPackHash_;
}

const std::string& MLXCompiledWorld::metallibPath() const noexcept {
    return metallibPath_;
}

std::vector<float> MLXCompiledWorld::defaultQ() const {
    return world_.model().defaultQ;
}

std::vector<float> MLXCompiledWorld::defaultV() const {
    return world_.model().defaultV;
}

std::vector<float> MLXCompiledWorld::effortLimits() const {
    const EngineModel& model = world_.model();
    const MRArticulationGPU& articulation =
        model.articulations[world_.articulationIndex()];
    std::vector<float> limits(articulation.nv, 0.0f);
    for (const MRDofPropertiesGPU& dof : model.dofs) {
        if (dof.articulationIndex !=
                world_.articulationIndex() ||
            dof.vIndex < articulation.vOffset ||
            dof.vIndex >=
                articulation.vOffset + articulation.nv) {
            continue;
        }
        limits[dof.vIndex - articulation.vOffset] =
            dof.limits.w;
    }
    return limits;
}

std::vector<float>
MLXCompiledWorld::defaultActuatorTargets() const {
    const EngineModel& model = world_.model();
    const MRArticulationGPU& articulation =
        model.articulations[world_.articulationIndex()];
    std::vector<float> targets(articulation.nv, 0.0f);
    for (std::uint32_t local = 0u;
         local < articulation.nv;
         ++local) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + local];
        if (dof.qIndex != MR_INVALID_INDEX &&
            dof.qIndex < model.defaultQ.size()) {
            targets[local] = model.defaultQ[dof.qIndex];
        }
    }
    return targets;
}

std::vector<float>
MLXCompiledWorld::actuatorProfileValues() const {
    const EngineModel& model = world_.model();
    const MRArticulationGPU& articulation =
        model.articulations[world_.articulationIndex()];
    std::vector<float> values(
        static_cast<std::size_t>(articulation.nv) * 7u,
        0.0f
    );
    const std::vector<MRActuatorProfileGPU> profiles =
        mlxExecutionActuatorProfiles(model);
    for (std::uint32_t local = 0u;
         local < articulation.nv;
         ++local) {
        const MRActuatorProfileGPU& profile =
            profiles[
                articulation.vOffset + local
            ];
        const std::size_t base =
            static_cast<std::size_t>(local) * 7u;
        values[base + 0u] = profile.motorAndSpeed.x;
        values[base + 1u] = profile.motorAndSpeed.y;
        values[base + 2u] = profile.motorAndSpeed.z;
        values[base + 3u] = profile.motorAndSpeed.w;
        values[base + 4u] =
            profile.transmissionAndEnvelope.x;
        values[base + 5u] =
            profile.transmissionAndEnvelope.y;
        values[base + 6u] =
            profile.transmissionAndEnvelope.z;
    }
    return values;
}

std::vector<std::uint32_t>
MLXCompiledWorld::actuatorProfileFlags() const {
    const EngineModel& model = world_.model();
    const MRArticulationGPU& articulation =
        model.articulations[world_.articulationIndex()];
    std::vector<std::uint32_t> flags(
        articulation.nv,
        0u
    );
    if (model.actuatorProfiles.empty()) {
        return flags;
    }
    for (std::uint32_t local = 0u;
         local < articulation.nv;
         ++local) {
        flags[local] =
            model.actuatorProfiles[
                articulation.vOffset + local
            ].identity.y;
    }
    return flags;
}

void MLXCompiledWorld::prepareStream(
    mx::StreamOrDevice stream
) {
    const auto selectedStream = mx::to_stream(stream);
    auto& device =
        mx::metal::device(selectedStream.device);
    static_cast<void>(resources(device));
}

MetalResources& MLXCompiledWorld::resources(
    mx::metal::Device& device
) {
    const std::lock_guard lock(resourceMutex_);
    if (resources_ != nullptr) {
        if (resources_->device != &device) {
            throw std::runtime_error(
                "MLXCompiledWorld cannot migrate between Metal devices"
            );
        }
        return *resources_;
    }

    const EngineModel& model = world_.model();
    const MRArticulationGPU& articulation =
        model.articulations[world_.articulationIndex()];
    MRWorldGPU worldRecord = model.world;
    worldRecord.gravityAndTimestep.w =
        controlTimestep_ /
        static_cast<float>(physicsSubsteps_);
    MRJointDescriptorGPU emptyJoint{};
    MRDofPropertiesGPU emptyDof{};
    MRBodyPropertiesGPU emptyBody{};
    MRABABodyWrenchGPU emptyWrench{};
    MRShapeGPU emptyShape{};
    MRMaterialGPU emptyMaterial{};
    std::uint32_t emptyIndex = 0u;
    MRCompiledCollisionPairGPU emptyPair{};
    MRGeometryHeaderGPU emptyGeometry{};
    mr_float4 emptyVertex{};
    MRMeshBVHNodeGPU emptyMeshNode{};
    MRMeshTriangleGPU emptyTriangle{};
    MRConvexFaceGPU emptyConvexFace{};
    MRTactileSensorGPU emptyTactileSensor{};
    MRTactileSampleGPU emptyTactileSample{};
    MRWorldDynamicNodeGPU emptyDynamicNode{};
    MRRodColliderGPU emptyRodCollider{};
    MRRodToolPairGPU emptyRodToolPair{};
    MRConstraintIRBlockGPU emptyAuthoredBlock{};
    MRConstraintIREndpointGPU emptyAuthoredEndpoint{};
    MRConstraintIRRowGPU emptyAuthoredRow{};
    MRConstraintIRConeGPU emptyAuthoredCone{};
    mr_float4 emptyRodCurvature{};
    float emptyRodScalar = 0.0f;

    auto staged = std::make_unique<MetalResources>();
    staged->device = &device;
    staged->tuning = tuningProfile(device);
    if (waveWorkerGroups_ != 0u) {
        staged->tuning.waveWorkerGroupCount =
            waveWorkerGroups_;
    }
    staged->buffers.reserve(hasTactile() ? 39u : 35u);
    staged->buffers.push_back(
        immutableBuffer(&worldRecord, 1u)
    );
    staged->buffers.push_back(immutableBuffer(
        model.articulations.data(),
        model.articulations.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        model.joints.empty()
            ? &emptyJoint
            : model.joints.data(),
        std::max<std::size_t>(model.joints.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        model.dofs.empty() ? &emptyDof : model.dofs.data(),
        std::max<std::size_t>(model.dofs.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        model.bodies.empty() ? &emptyBody : model.bodies.data(),
        std::max<std::size_t>(model.bodies.size(), 1u)
    ));
    staged->buffers.push_back(
        immutableBuffer(&emptyWrench, 1u)
    );
    staged->buffers.push_back(immutableBuffer(
        model.shapes.empty() ? &emptyShape : model.shapes.data(),
        std::max<std::size_t>(model.shapes.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        model.materials.empty()
            ? &emptyMaterial
            : model.materials.data(),
        std::max<std::size_t>(model.materials.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.sceneBodyIndices().empty()
            ? &emptyIndex
            : world_.sceneBodyIndices().data(),
        std::max<std::size_t>(
            world_.sceneBodyIndices().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.eligiblePairs().empty()
            ? &emptyPair
            : world_.eligiblePairs().data(),
        std::max<std::size_t>(
            world_.eligiblePairs().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.geometryHeaders.empty()
            ? &emptyGeometry
            : model.geometryHeaders.data(),
        std::max<std::size_t>(
            model.geometryHeaders.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.geometryVertices.empty()
            ? &emptyVertex
            : model.geometryVertices.data(),
        std::max<std::size_t>(
            model.geometryVertices.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.meshBvhNodes.empty()
            ? &emptyMeshNode
            : model.meshBvhNodes.data(),
        std::max<std::size_t>(
            model.meshBvhNodes.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.meshTriangles.empty()
            ? &emptyTriangle
            : model.meshTriangles.data(),
        std::max<std::size_t>(
            model.meshTriangles.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.dynamicNodes().empty()
            ? &emptyDynamicNode
            : world_.dynamicNodes().data(),
        std::max<std::size_t>(
            world_.dynamicNodes().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.bodyDynamicNodes().empty()
            ? &emptyIndex
            : world_.bodyDynamicNodes().data(),
        std::max<std::size_t>(
            world_.bodyDynamicNodes().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.rodColliders().empty()
            ? &emptyRodCollider
            : world_.rodColliders().data(),
        std::max<std::size_t>(
            world_.rodColliders().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.rodShapeSources().empty()
            ? &emptyShape
            : world_.rodShapeSources().data(),
        std::max<std::size_t>(
            world_.rodShapeSources().size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        world_.rodToolPairs().empty()
            ? &emptyRodToolPair
            : world_.rodToolPairs().data(),
        std::max<std::size_t>(
            world_.rodToolPairs().size(),
            1u
        )
    ));

    std::vector<float> rodRestLengths;
    std::vector<float> rodRestTwists;
    std::vector<mr_float4> rodRestCurvatures;
    std::vector<float> rodInverseMasses;
    std::vector<float> rodInverseTwistInertias;
    std::vector<float> rodStretchStiffness;
    std::vector<float> rodBendStiffness;
    std::vector<float> rodTwistStiffness;
    rodRestLengths.reserve(world_.rodEdgeCount());
    rodRestTwists.reserve(world_.rodEdgeCount());
    rodInverseMasses.reserve(world_.rodNodeCount());
    rodInverseTwistInertias.reserve(world_.rodEdgeCount());
    rodStretchStiffness.reserve(world_.rodEdgeCount());
    for (const HeterogeneousRodProgram& program :
         world_.rodPrograms()) {
        for (const double value : program.model.restLengths) {
            rodRestLengths.push_back(
                static_cast<float>(value)
            );
        }
        for (const double value : program.model.restTwists) {
            rodRestTwists.push_back(
                static_cast<float>(value)
            );
        }
        for (const double value : program.model.nodeMasses) {
            rodInverseMasses.push_back(
                static_cast<float>(1.0 / value)
            );
        }
        for (const double value :
             program.model.edgeRotationalInertias) {
            rodInverseTwistInertias.push_back(
                static_cast<float>(1.0 / value)
            );
        }
        for (const double value :
             program.model.stretchStiffness) {
            rodStretchStiffness.push_back(
                static_cast<float>(value)
            );
        }
        for (std::size_t bend = 0u;
             bend + 1u <
                 program.model.restLengths.size();
             ++bend) {
            mr_float4 curvature{};
            if (!mlxRodRestCurvature(
                    program.model,
                    bend,
                    curvature
                )) {
                throw std::runtime_error(
                    "compiled MLX rod has degenerate rest curvature"
                );
            }
            rodRestCurvatures.push_back(curvature);
            rodBendStiffness.push_back(
                static_cast<float>(
                    program.model.bendStiffness[bend]
                )
            );
            rodTwistStiffness.push_back(
                static_cast<float>(
                    program.model.twistStiffness[bend]
                )
            );
        }
    }
    staged->buffers.push_back(immutableBuffer(
        rodRestLengths.empty()
            ? &emptyRodScalar
            : rodRestLengths.data(),
        std::max<std::size_t>(rodRestLengths.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        rodRestTwists.empty()
            ? &emptyRodScalar
            : rodRestTwists.data(),
        std::max<std::size_t>(rodRestTwists.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        rodRestCurvatures.empty()
            ? &emptyRodCurvature
            : rodRestCurvatures.data(),
        std::max<std::size_t>(
            rodRestCurvatures.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        rodInverseMasses.empty()
            ? &emptyRodScalar
            : rodInverseMasses.data(),
        std::max<std::size_t>(rodInverseMasses.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        rodInverseTwistInertias.empty()
            ? &emptyRodScalar
            : rodInverseTwistInertias.data(),
        std::max<std::size_t>(
            rodInverseTwistInertias.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        rodStretchStiffness.empty()
            ? &emptyRodScalar
            : rodStretchStiffness.data(),
        std::max<std::size_t>(
            rodStretchStiffness.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        rodBendStiffness.empty()
            ? &emptyRodScalar
            : rodBendStiffness.data(),
        std::max<std::size_t>(
            rodBendStiffness.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        rodTwistStiffness.empty()
            ? &emptyRodScalar
            : rodTwistStiffness.data(),
        std::max<std::size_t>(
            rodTwistStiffness.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.blocks.empty()
            ? &emptyAuthoredBlock
            : model.constraintProgram.blocks.data(),
        std::max<std::size_t>(
            model.constraintProgram.blocks.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.endpoints.empty()
            ? &emptyAuthoredEndpoint
            : model.constraintProgram.endpoints.data(),
        std::max<std::size_t>(
            model.constraintProgram.endpoints.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.rows.empty()
            ? &emptyAuthoredRow
            : model.constraintProgram.rows.data(),
        std::max<std::size_t>(
            model.constraintProgram.rows.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.cones.empty()
            ? &emptyAuthoredCone
            : model.constraintProgram.cones.data(),
        std::max<std::size_t>(
            model.constraintProgram.cones.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.warmImpulses.empty()
            ? &emptyRodScalar
            : model.constraintProgram.warmImpulses.data(),
        std::max<std::size_t>(
            model.constraintProgram.warmImpulses.size(),
            1u
        )
    ));
    staged->buffers.push_back(immutableBuffer(
        model.convexFaces.empty()
            ? &emptyConvexFace
            : model.convexFaces.data(),
        std::max<std::size_t>(
            model.convexFaces.size(),
            1u
        )
    ));
    const std::vector<MRActuatorProfileGPU>
        actuatorProfiles =
            mlxExecutionActuatorProfiles(model);
    staged->buffers.push_back(immutableBuffer(
        actuatorProfiles.data(),
        actuatorProfiles.size()
    ));
    if (hasTactile()) {
        staged->buffers.push_back(immutableBuffer(
            tactile_.sensors.empty()
                ? &emptyTactileSensor
                : tactile_.sensors.data(),
            std::max<std::size_t>(
                tactile_.sensors.size(),
                1u
            )
        ));
        staged->buffers.push_back(immutableBuffer(
            tactile_.samples.empty()
                ? &emptyTactileSample
                : tactile_.samples.data(),
            std::max<std::size_t>(
                tactile_.samples.size(),
                1u
            )
        ));
        staged->buffers.push_back(immutableBuffer(
            tactile_.targetShapeIndices.empty()
                ? &emptyIndex
                : tactile_.targetShapeIndices.data(),
            std::max<std::size_t>(
                tactile_.targetShapeIndices.size(),
                1u
            )
        ));
        staged->buffers.push_back(immutableBuffer(
            tactile_.shapeToSensor.empty()
                ? &emptyIndex
                : tactile_.shapeToSensor.data(),
            std::max<std::size_t>(
                tactile_.shapeToSensor.size(),
                1u
            )
        ));
    }
    std::vector<std::uint32_t> bodyToScene(
        model.bodies.size(),
        MR_INVALID_INDEX
    );
    for (std::uint32_t local = 0u;
         local < world_.sceneBodyIndices().size();
         ++local) {
        const std::uint32_t body =
            world_.sceneBodyIndices()[local];
        if (body >= bodyToScene.size()) {
            throw std::runtime_error(
                "MLXCompiledWorld scene-body mapping is invalid"
            );
        }
        bodyToScene[body] = local;
    }
    staged->bodyToSceneBufferIndex = staged->buffers.size();
    staged->buffers.push_back(immutableBuffer(
        bodyToScene.empty() ? &emptyIndex : bodyToScene.data(),
        std::max<std::size_t>(bodyToScene.size(), 1u)
    ));

    auto* physicsLibrary =
        device.get_library("MetalRobo", metallibPath_);
    const std::string abaName =
        articulation.bodyCount <= 12u &&
            articulation.nv <= 16u &&
            articulation.nq <= 17u
        ? "mr_articulated_aba_step_small"
        : "mr_articulated_aba_step";
    staged->abaKernel =
        device.get_kernel(abaName, physicsLibrary);
    staged->multiABAKernel = device.get_kernel(
        "mr_multi_articulated_aba_step",
        physicsLibrary
    );
    auto* adapterLibrary =
        device.get_library(
            "MetalRoboMLX",
            currentBinaryDirectory()
        );
    staged->commitKernel = device.get_kernel(
        "mr_mlx_world_commit_aba",
        adapterLibrary
    );
    std::vector<std::string> physicsKernels{
        "mr_articulated_operator",
        "mr_articulated_materialize_body_velocities",
        "mr_metal_world_commit",
        "mr_world_prepare_contact_step",
        "mr_world_build_body_states",
        "mr_world_predict_scene",
        "mr_world_project_swept_colliders",
        "mr_world_initialize_ccd_event_state",
        "mr_world_prepare_ccd_event_pass",
        "mr_world_materialize_event_articulation",
        "mr_world_predict_scene_event",
        "mr_world_overlay_event_articulation_bodies",
        "mr_world_project_joint_limits",
        "mr_world_project_event_colliders",
        "mr_world_restore_inactive_event_candidate",
        "mr_world_publish_event_segment",
        "mr_world_flag_eligible_pairs",
        "mr_world_resolve_ccd",
        "mr_world_select_ccd_event_state",
        "mr_world_finalize_ccd_event_state",
        "mr_world_scan_blocks",
        "mr_world_scan_add_block_offsets",
        "mr_world_flag_pair_work_class",
        "mr_world_scatter_pair_queue",
        "mr_world_narrowphase_pair_queue",
        "mr_world_narrowphase_convex_queue",
        "mr_world_narrowphase_mesh_queue",
        "mr_world_collide_compile",
        "mr_world_finalize_pair_manifold",
        "mr_world_scan_manifold_ir",
        "mr_world_scatter_manifold_records",
        "mr_world_scatter_manifold_ir",
        "mr_world_seed_authored_constraint_ir",
        "mr_world_initialize_multi_articulation_queries",
        "mr_world_compose_multi_articulation_operator",
        "mr_world_finalize_factor_dispatch",
        "mr_world_fill_point_query_tail",
        "mr_world_evaluate_constraint_ir",
        "mr_world_build_contact_islands",
        "mr_world_solve_contact_islands",
        "mr_world_build_contact_tiles",
        "mr_world_scatter_island_queue",
        "mr_world_select_solver_cohort",
        "mr_world_flag_distributed_islands",
        "mr_world_scatter_distributed_island_queue",
        "mr_world_flag_distributed_tiles",
        "mr_world_scatter_distributed_tile_queue",
        "mr_world_wave32_solve",
        "mr_world_wave32_solve_persistent",
        "mr_world_wave32_distributed_prepare",
        "mr_world_wave32_distributed_delta",
        "mr_world_wave32_distributed_reduce",
        "mr_world_reduce_wave32_status",
        "mr_world_prepare_unified_quality",
        "mr_world_reconstruct_unified_quality_warm_start",
        "mr_world_build_unified_quality_queue",
        "mr_unified_quality_solve",
        "mr_unified_quality_solve_persistent",
        "mr_world_apply_unified_quality",
        "mr_world_publish_unified_quality_queue_status",
        "mr_world_prepare_rod_state",
        "mr_world_prepare_rod_contact_cache",
        "mr_world_initialize_rod_event_state",
        "mr_world_restore_inactive_rod_event_candidate",
        "mr_world_publish_rod_event_segment",
        "mr_world_pack_rod_state",
        "mr_discrete_elastic_rod_step",
        "mr_world_unpack_rod_state",
        "mr_world_latch_rod_status",
        "mr_world_latch_rod_contact_status",
        "mr_world_factor_rod_operator",
        "mr_world_project_swept_rod_colliders",
        "mr_world_resolve_rod_ccd",
        "mr_world_tag_rod_ccd_witnesses",
        "mr_rod_tool_narrowphase",
        "mr_world_scan_rod_contact_ir",
        "mr_world_scatter_rod_contact_ir",
        "mr_world_solve_rod_contact_constraints",
        "mr_world_solve_generalized_constraints",
        "mr_world_commit_rod_state",
        "mr_world_commit_rod_contact_cache",
        "mr_world_integrate_contact_state",
        "mr_world_latch_contact_status",
        "mr_world_commit_contact_state",
        "mr_scene_query_pack_body_states",
        "mr_scene_raycast",
        "mr_scene_raycast_pattern",
    };
    if (hasTactile()) {
        physicsKernels.insert(
            physicsKernels.end(),
            {
                "mr_tactile_reduce",
            }
        );
    }
    for (const std::string& name : physicsKernels) {
        staged->kernels.emplace(
            name,
            device.get_kernel(name, physicsLibrary)
        );
    }
    if (hasTactile()) {
        const bool debugHits = false;
        staged->kernels.emplace(
            "mr_tactile_sample",
            device.get_kernel(
                "mr_tactile_sample",
                physicsLibrary,
                "mr_tactile_sample_nondebug",
                {{
                    &debugHits,
                    MTL::DataTypeBool,
                    0u,
                }}
            )
        );
    }
    std::vector<std::string> adapterKernels{
        "mr_mlx_import_world_family_state",
        "mr_mlx_prepare_contact_world",
        "mr_mlx_commit_pair_cache",
        "mr_mlx_apply_family_contact_parameters",
        "mr_mlx_apply_family_body_damping",
        "mr_mlx_initialize_operator_dispatch",
        "mr_mlx_initialize_world_articulation_dispatches",
        "mr_mlx_pack_scene_state",
        "mr_mlx_unpack_scene_and_evidence",
    };
    if (hasTactile()) {
        adapterKernels.insert(
            adapterKernels.end(),
            {
                "mr_mlx_pack_tactile_contacts",
                "mr_mlx_unpack_tactile_summary",
                "mr_mlx_publish_object_contact_points",
            }
        );
    }
    for (const std::string& name : adapterKernels) {
        staged->kernels.emplace(
            name,
            device.get_kernel(name, adapterLibrary)
        );
    }
    if (staged->abaKernel == nullptr ||
        staged->multiABAKernel == nullptr ||
        staged->commitKernel == nullptr ||
        std::any_of(
            staged->kernels.begin(),
            staged->kernels.end(),
            [](const auto& entry) {
                return entry.second == nullptr;
            }
        )) {
        throw std::runtime_error(
            "MLXCompiledWorld could not create Metal pipelines"
        );
    }
    const auto* pairNarrowphase =
        staged->kernels.at(
            "mr_world_narrowphase_pair_queue"
        );
    const auto* wave32Solve = staged->kernels.at(
        "mr_world_wave32_solve_persistent"
    );
    const auto* manifoldScan = staged->kernels.at(
        "mr_world_scan_manifold_ir"
    );
    const auto* solverCohort =
        staged->kernels.at("mr_world_select_solver_cohort");
    if (pairNarrowphase->threadExecutionWidth() !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        solverCohort->threadExecutionWidth() !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        manifoldScan->threadExecutionWidth() !=
            MR_WAVE32_CONTACTS_PER_TILE ||
        wave32Solve->threadExecutionWidth() !=
            MR_WAVE32_CONTACTS_PER_TILE) {
        throw std::runtime_error(
            "MLX contact physics requires a queried Metal SIMD "
            "execution width of 32"
        );
    }
    resources_ = std::move(staged);
    return *resources_;
}

MLXCompiledMultiArticulatedProgram::
    MLXCompiledMultiArticulatedProgram(
        CompiledMetalMultiArticulatedProgram program,
        const std::uint32_t environmentCapacity,
        MetalMultiArticulatedConstraintConfig config,
        std::string metallibPath
    )
    : program_(std::move(program)),
      environmentCapacity_(environmentCapacity),
      config_(std::move(config)),
      metallibPath_(std::move(metallibPath)) {}

MLXCompiledMultiArticulatedProgram::
    ~MLXCompiledMultiArticulatedProgram() = default;

const CompiledMetalMultiArticulatedProgram&
MLXCompiledMultiArticulatedProgram::program() const noexcept {
    return program_;
}

std::uint32_t
MLXCompiledMultiArticulatedProgram::environmentCapacity()
    const noexcept {
    return environmentCapacity_;
}

const MetalMultiArticulatedConstraintConfig&
MLXCompiledMultiArticulatedProgram::config() const noexcept {
    return config_;
}

const std::string&
MLXCompiledMultiArticulatedProgram::metallibPath()
    const noexcept {
    return metallibPath_;
}

std::vector<float>
MLXCompiledMultiArticulatedProgram::defaultQ() const {
    return program_.model().defaultQ;
}

std::vector<float>
MLXCompiledMultiArticulatedProgram::defaultV() const {
    return program_.model().defaultV;
}

void MLXCompiledMultiArticulatedProgram::prepareStream(
    mx::StreamOrDevice stream
) {
    const auto selectedStream = mx::to_stream(stream);
    auto& device = mx::metal::device(selectedStream.device);
    static_cast<void>(resources(device));
}

MetalGeneralizedResources&
MLXCompiledMultiArticulatedProgram::resources(
    mx::metal::Device& device
) {
    const std::lock_guard lock(resourceMutex_);
    if (resources_ != nullptr) {
        if (resources_->device != &device) {
            throw std::runtime_error(
                "compiled multi-articulation program cannot "
                "migrate between Metal devices"
            );
        }
        return *resources_;
    }
    if (!program_.valid() || environmentCapacity_ == 0u) {
        throw std::runtime_error(
            "compiled multi-articulation MLX program is invalid"
        );
    }

    const EngineModel& model = program_.model();
    const auto inverseDispatches =
        generalizedInverseDispatches(
            program_,
            environmentCapacity_
        );
    const auto rhs = generalizedRhs(
        program_,
        environmentCapacity_
    );
    const ParallelABASchedule& schedule =
        program_.abaSchedule();
    MRJointDescriptorGPU emptyJoint{};

    auto staged =
        std::make_unique<MetalGeneralizedResources>();
    staged->device = &device;
    staged->inverseWorkCount =
        static_cast<std::uint32_t>(
            inverseDispatches.size()
        );
    staged->buffers.reserve(18u);
    staged->buffers.push_back(
        immutableBuffer(&model.world, 1u)
    );
    staged->buffers.push_back(immutableBuffer(
        model.articulations.data(),
        model.articulations.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        model.joints.empty()
            ? &emptyJoint
            : model.joints.data(),
        std::max<std::size_t>(model.joints.size(), 1u)
    ));
    staged->buffers.push_back(immutableBuffer(
        model.dofs.data(),
        model.dofs.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        model.bodies.data(),
        model.bodies.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.rows.data(),
        program_.rowCount()
    ));
    staged->buffers.push_back(immutableBuffer(
        model.constraintProgram.warmImpulses.data(),
        program_.rowCount()
    ));
    staged->buffers.push_back(immutableBuffer(
        program_.generalizedJacobian().data(),
        program_.generalizedJacobian().size()
    ));
    staged->buffers.push_back(immutableBuffer(
        inverseDispatches.data(),
        inverseDispatches.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        rhs.data(),
        rhs.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.articulations.data(),
        schedule.articulations.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.levels.data(),
        schedule.levels.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.parentReductions.data(),
        schedule.parentReductions.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.levelBodies.data(),
        schedule.levelBodies.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.parentLocal.data(),
        schedule.parentLocal.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.inboundJoint.data(),
        schedule.inboundJoint.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.childOffsets.data(),
        schedule.childOffsets.size()
    ));
    staged->buffers.push_back(immutableBuffer(
        schedule.childIndices.data(),
        schedule.childIndices.size()
    ));

    auto* physicsLibrary =
        device.get_library("MetalRobo", metallibPath_);
    staged->inverseKernel = device.get_kernel(
        "mr_parallel_multi_articulated_inverse_mass",
        physicsLibrary
    );
    staged->delassusKernel = device.get_kernel(
        "mr_generalized_constraint_delassus",
        physicsLibrary
    );
    staged->solveKernel = device.get_kernel(
        config_.solverMode ==
                MetalGeneralizedConstraintSolverMode::
                    qualitySemismoothNewton
            ? "mr_generalized_constraint_quality_solve"
            : "mr_generalized_constraint_solve",
        physicsLibrary
    );
    auto* adapterLibrary = device.get_library(
        "MetalRoboMLX",
        currentBinaryDirectory()
    );
    staged->commitKernel = device.get_kernel(
        "mr_mlx_commit_generalized_constraints",
        adapterLibrary
    );
    if (staged->inverseKernel == nullptr ||
        staged->delassusKernel == nullptr ||
        staged->solveKernel == nullptr ||
        staged->commitKernel == nullptr ||
        staged->inverseKernel->threadExecutionWidth() != 32u ||
        staged->solveKernel->threadExecutionWidth() != 32u) {
        throw std::runtime_error(
            "MLX generalized constraints require SIMD32 pipelines"
        );
    }
    resources_ = std::move(staged);
    return *resources_;
}

std::shared_ptr<MLXCompiledWorld> compileWorld(
    const std::string& modelName,
    const std::string& scene,
    const std::uint32_t environmentCapacity,
    MetalWorldCapacityProfile capacityProfile,
    const float controlTimestep,
    const std::uint32_t physicsSubsteps,
    const bool applyBodyDamping,
    const std::string& requestedActuationMode,
    const std::string& requestedSolverMode,
    const std::uint32_t velocityIterations,
    const std::uint32_t finalVelocityIterations,
    const std::string& requestedCCDMode,
    const std::uint32_t maxCCDAdvanceSolvePasses,
    const std::uint32_t maxCCDZeroTimeReplays,
    const float ccdSimultaneousTolerance,
    const std::uint32_t waveWorkerGroups,
    const std::string& requestedMetallibPath,
    mx::StreamOrDevice stream
) {
    if (!std::isfinite(controlTimestep) ||
        !(controlTimestep > 0.0f) ||
        physicsSubsteps == 0u ||
        physicsSubsteps > 128u ||
        environmentCapacity == 0u ||
        velocityIterations == 0u ||
        velocityIterations > 128u ||
        finalVelocityIterations > 128u) {
        throw std::invalid_argument(
            "control_timestep must be finite and positive and "
            "physics_substeps must be in [1, 128], with a positive "
            "environment_capacity and valid solver iterations"
        );
    }
    if (maxCCDAdvanceSolvePasses == 0u ||
        maxCCDAdvanceSolvePasses >
            MR_CCD_MAX_ADVANCE_SOLVE_PASSES ||
        maxCCDZeroTimeReplays >
            MR_CCD_MAX_ZERO_TIME_REPLAYS ||
        !std::isfinite(ccdSimultaneousTolerance) ||
        !(ccdSimultaneousTolerance > 0.0f)) {
        throw std::invalid_argument(
            "max_ccd_advance_solve_passes and "
            "max_ccd_zero_time_replays exceed the compiled limits, "
            "or ccd_simultaneous_tolerance is not positive"
        );
    }
    if (waveWorkerGroups != 0u &&
        waveWorkerGroups != 32u &&
        waveWorkerGroups != 64u &&
        waveWorkerGroups != 96u &&
        waveWorkerGroups != 128u) {
        throw std::invalid_argument(
            "wave_worker_groups must be 0, 32, 64, 96, or 128"
        );
    }
    const bool addPickPlace = scene == "pick_place";
    const bool addNeedleThread = scene == "needle_thread";
    EngineModel model;
    HeterogeneousWorld heterogeneousWorld;
    bool useHeterogeneousWorld = false;
    if (modelName == "dual_psm" && addNeedleThread) {
        const auto diagnostics =
            makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                heterogeneousWorld
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                "could not cook the dual-PSM needle/thread "
                "world: " + diagnostics.message
            );
        }
        model = heterogeneousWorld.model;
        useHeterogeneousWorld = true;
    } else if (modelName == "franka") {
        model = addPickPlace
            ? makeFrankaPickPlaceEngineModel()
            : makeFrankaPandaEngineModel();
    } else if (modelName == "g1") {
        model = makeUnitreeG1EngineModel();
    } else if (modelName == "psm") {
        model = makeDvrkPsmLargeNeedleDriverEngineModel();
    } else {
        throw std::invalid_argument(
            "model must be 'franka', 'g1', 'psm', or "
            "'dual_psm' with scene='needle_thread'"
        );
    }

    MetalWorldActuationMode actuationMode;
    if (requestedActuationMode == "effort") {
        actuationMode = MetalWorldActuationMode::effort;
    } else if (requestedActuationMode == "implicit_position") {
        actuationMode =
            MetalWorldActuationMode::implicitPositionDrive;
    } else {
        throw std::invalid_argument(
            "actuation_mode must be 'effort' or "
            "'implicit_position'"
        );
    }

    MetalWorldSolverMode solverMode;
    if (requestedSolverMode == "free_motion_aba") {
        solverMode = MetalWorldSolverMode::freeMotionABA;
    } else if (requestedSolverMode == "throughput_pgs") {
        solverMode = MetalWorldSolverMode::throughputPGS;
    } else if (requestedSolverMode == "throughput_tgs") {
        solverMode = MetalWorldSolverMode::throughputTGS;
    } else if (requestedSolverMode == "quality_newton") {
        solverMode = MetalWorldSolverMode::qualityNewton;
    } else {
        throw std::invalid_argument(
            "solver_mode must be 'free_motion_aba', "
            "'throughput_pgs', 'throughput_tgs', or "
            "'quality_newton'"
        );
    }
    MetalWorldCCDMode ccdMode;
    if (requestedCCDMode == "disabled") {
        ccdMode = MetalWorldCCDMode::disabled;
    } else if (requestedCCDMode == "speculative") {
        ccdMode = MetalWorldCCDMode::speculative;
    } else if (requestedCCDMode == "hybrid") {
        ccdMode = MetalWorldCCDMode::hybrid;
    } else {
        throw std::invalid_argument(
            "ccd_mode must be 'disabled', 'speculative', or 'hybrid'"
        );
    }

    const bool addCube =
        scene == "cube" ||
        scene == "cube_ground" ||
        scene == "cube_terrain";
    const bool addGround =
        scene == "ground" || scene == "cube_ground";
    const bool addTerrain =
        scene == "terrain" || scene == "cube_terrain";
    const bool addNeedle = scene == "needle";
    if (!scene.empty() &&
        scene != "none" &&
        !addCube &&
        !addGround &&
        !addTerrain &&
        !addNeedle &&
        !addNeedleThread &&
        !addPickPlace) {
        throw std::invalid_argument(
            "scene must be 'none', 'cube', 'ground', 'terrain', "
            "'needle', 'needle_thread', 'pick_place', "
            "'cube_ground', or 'cube_terrain'"
        );
    }
    if (addPickPlace && modelName != "franka") {
        throw std::invalid_argument(
            "the pick_place scene requires model='franka'"
        );
    }
    if (addNeedle && modelName != "psm") {
        throw std::invalid_argument(
            "the needle scene requires model='psm'"
        );
    }
    if (addNeedleThread && modelName != "dual_psm") {
        throw std::invalid_argument(
            "the needle_thread scene requires model='dual_psm'"
        );
    }
    if ((addCube || addGround || addTerrain || addNeedle ||
         addNeedleThread || addPickPlace) &&
        solverMode == MetalWorldSolverMode::freeMotionABA) {
        throw std::invalid_argument(
            "contact scenes require a contact solver mode"
        );
    }
    if (solverMode == MetalWorldSolverMode::qualityNewton &&
        ccdMode == MetalWorldCCDMode::hybrid) {
        throw std::invalid_argument(
            "quality_newton currently requires disabled or "
            "speculative CCD; MLX never routes event-time quality "
            "solves through TGS"
        );
    }
    std::vector<MRBodyStateGPU> defaultSceneBodies =
        useHeterogeneousWorld
        ? heterogeneousWorld.defaultSceneBodies
        : std::vector<MRBodyStateGPU>{};
    if (addPickPlace) {
        defaultSceneBodies = makeFrankaPickPlaceSceneState();
    }
    if (addCube) {
        constexpr float inertia = 1.0f / 600.0f;
        constexpr float inverseInertia = 600.0f;
        MRBodyPropertiesGPU body{};
        body.articulationIndex = MR_INVALID_INDEX;
        body.parentBody = MR_INVALID_INDEX;
        body.inboundJoint = MR_INVALID_INDEX;
        body.motionType = MR_MOTION_DYNAMIC;
        body.massAndInverseMass = {1.0f, 1.0f, 0.0f, 0.0f};
        body.inertiaRow0 = {inertia, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, inertia, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, inertia, 0.0f};
        body.inverseInertiaRow0 =
            {inverseInertia, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaRow1 =
            {0.0f, inverseInertia, 0.0f, 0.0f};
        body.inverseInertiaRow2 =
            {0.0f, 0.0f, inverseInertia, 0.0f};
        body.dampingAndSpeedLimits =
            {0.01f, 0.01f, 100.0f, 100.0f};
        const std::uint32_t cubeBody =
            static_cast<std::uint32_t>(model.bodies.size());
        model.bodies.push_back(body);

        MRShapeGPU cube{};
        cube.bodyIndex = cubeBody;
        cube.shapeType = MR_SHAPE_BOX;
        cube.materialIndex = 0u;
        cube.flags =
            ccdMode == MetalWorldCCDMode::hybrid
            ? MR_SHAPE_FLAG_ENABLE_CCD
            : 0u;
        cube.collisionGroup = 1u;
        cube.collisionMask = ~0u;
        cube.slotGeneration = 1u;
        cube.localPosition.w = 1.0f;
        cube.localRotation.w = 1.0f;
        cube.dimensions = {0.05f, 0.05f, 0.05f, 0.0f};
        cube.contactRestAndBoundingRadius =
            {0.002f, 0.0f, 0.08660254f, 0.0f};
        model.shapes.push_back(cube);
        model.world.bodyCount =
            static_cast<std::uint32_t>(model.bodies.size());
        model.world.shapeCount =
            static_cast<std::uint32_t>(model.shapes.size());

        MRBodyStateGPU cubeState{};
        cubeState.position.w = 1.0f;
        cubeState.orientation.w = 1.0f;
        cubeState.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        cubeState.flagsAndIndices[1] = MR_INVALID_INDEX;
        cubeState.flagsAndIndices[2] = cubeBody;
        if (modelName == "franka" && model.shapes.size() > 1u) {
            const MRShapeGPU witnessShape =
                model.shapes[model.shapes.size() - 2u];
            std::vector<double> q64(
                model.defaultQ.begin(),
                model.defaultQ.end()
            );
            std::vector<double> v64(
                model.defaultV.begin(),
                model.defaultV.end()
            );
            const ArticulatedPointQuery query{
                .bodyIndex = witnessShape.bodyIndex,
                .localPoint = {
                    witnessShape.localPosition.x,
                    witnessShape.localPosition.y,
                    witnessShape.localPosition.z,
                },
            };
            ArticulatedPointKinematics witness{};
            std::vector<double> jacobian(
                3u * model.articulations[0].nv
            );
            const auto witnessStatus =
                computeArticulatedPointJacobians(
                    model,
                    0u,
                    q64,
                    v64,
                    std::span{&query, 1u},
                    std::span{&witness, 1u},
                    jacobian
                );
            if (!witnessStatus.succeeded()) {
                throw std::runtime_error(
                    "could not place the MLX Franka cube scene"
                );
            }
            cubeState.position = {
                static_cast<float>(witness.position[0]) +
                    witnessShape.dimensions.x + 0.047f,
                static_cast<float>(witness.position[1]),
                static_cast<float>(witness.position[2]),
                1.0f,
            };
        } else {
            cubeState.position = {0.0f, 0.0f, 0.1f, 1.0f};
        }
        defaultSceneBodies.push_back(cubeState);
    }
    if (addGround) {
        MRBodyPropertiesGPU ground{};
        ground.articulationIndex = MR_INVALID_INDEX;
        ground.parentBody = MR_INVALID_INDEX;
        ground.inboundJoint = MR_INVALID_INDEX;
        ground.motionType = MR_MOTION_STATIC;
        ground.dampingAndSpeedLimits =
            {0.0f, 0.0f, 1.0e6f, 1.0e6f};
        const std::uint32_t groundBody =
            static_cast<std::uint32_t>(model.bodies.size());
        model.bodies.push_back(ground);

        MRShapeGPU plane{};
        plane.bodyIndex = groundBody;
        plane.shapeType = MR_SHAPE_PLANE;
        plane.materialIndex = 0u;
        plane.collisionGroup = 1u;
        plane.collisionMask = ~0u;
        plane.slotGeneration = 1u;
        plane.localPosition.w = 1.0f;
        // The collision plane's authored normal is +Y. Robotics models are
        // Z-up, so rotate +Y onto +Z.
        constexpr float kSqrtHalf = 0.7071067811865476f;
        plane.localRotation = {
            kSqrtHalf,
            0.0f,
            0.0f,
            kSqrtHalf,
        };
        model.shapes.push_back(plane);
        model.world.bodyCount =
            static_cast<std::uint32_t>(model.bodies.size());
        model.world.shapeCount =
            static_cast<std::uint32_t>(model.shapes.size());

        MRBodyStateGPU groundState{};
        groundState.position.w = 1.0f;
        groundState.orientation.w = 1.0f;
        groundState.flagsAndIndices[0] = MR_MOTION_STATIC;
        groundState.flagsAndIndices[1] = MR_INVALID_INDEX;
        groundState.flagsAndIndices[2] = groundBody;
        defaultSceneBodies.push_back(groundState);
    }
    if (addTerrain) {
        constexpr std::uint32_t side = 17u;
        constexpr float spacing = 0.25f;
        constexpr float halfWidth =
            0.5f * spacing * static_cast<float>(side - 1u);
        std::vector<mr_float4> vertices;
        vertices.reserve(side * side);
        for (std::uint32_t y = 0u; y < side; ++y) {
            for (std::uint32_t x = 0u; x < side; ++x) {
                const float px =
                    -halfWidth + spacing * static_cast<float>(x);
                const float py =
                    -halfWidth + spacing * static_cast<float>(y);
                const float height =
                    0.035f *
                        std::sin(1.7f * px) *
                        std::sin(1.3f * py) +
                    0.015f *
                        std::sin(3.1f * px) *
                        std::sin(2.3f * py);
                vertices.push_back({px, py, height, 1.0f});
            }
        }
        std::vector<std::uint32_t> indices;
        indices.reserve(
            6u * (side - 1u) * (side - 1u)
        );
        for (std::uint32_t y = 0u; y + 1u < side; ++y) {
            for (std::uint32_t x = 0u; x + 1u < side; ++x) {
                const std::uint32_t v00 = y * side + x;
                const std::uint32_t v10 = v00 + 1u;
                const std::uint32_t v01 = v00 + side;
                const std::uint32_t v11 = v01 + 1u;
                indices.insert(
                    indices.end(),
                    {v00, v10, v11, v00, v11, v01}
                );
            }
        }
        const GeometryCookResult cooked =
            cookTriangleMeshGeometry(
                model,
                vertices,
                indices
            );
        if (!cooked.succeeded()) {
            throw std::runtime_error(
                "could not cook MLX terrain: " + cooked.message
            );
        }

        MRBodyPropertiesGPU terrain{};
        terrain.articulationIndex = MR_INVALID_INDEX;
        terrain.parentBody = MR_INVALID_INDEX;
        terrain.inboundJoint = MR_INVALID_INDEX;
        terrain.motionType = MR_MOTION_STATIC;
        terrain.dampingAndSpeedLimits =
            {0.0f, 0.0f, 1.0e6f, 1.0e6f};
        const std::uint32_t terrainBody =
            static_cast<std::uint32_t>(model.bodies.size());
        model.bodies.push_back(terrain);

        MRShapeGPU mesh{};
        mesh.bodyIndex = terrainBody;
        mesh.shapeType = MR_SHAPE_TRIANGLE_MESH;
        mesh.materialIndex = 0u;
        mesh.collisionGroup = 1u;
        mesh.collisionMask = ~0u;
        mesh.slotGeneration = 1u;
        mesh.localPosition.w = 1.0f;
        mesh.localRotation.w = 1.0f;
        mesh.dimensions = {1.0f, 1.0f, 1.0f, 0.0f};
        mesh.contactRestAndBoundingRadius =
            {0.002f, 0.0f, 3.0f, 0.0f};
        mesh.geometryOffset = cooked.geometryIndex;
        mesh.geometryCount = 1u;
        model.shapes.push_back(mesh);
        model.world.bodyCount =
            static_cast<std::uint32_t>(model.bodies.size());
        model.world.shapeCount =
            static_cast<std::uint32_t>(model.shapes.size());

        MRBodyStateGPU terrainState{};
        terrainState.position.w = 1.0f;
        terrainState.orientation.w = 1.0f;
        terrainState.flagsAndIndices[0] = MR_MOTION_STATIC;
        terrainState.flagsAndIndices[1] = MR_INVALID_INDEX;
        terrainState.flagsAndIndices[2] = terrainBody;
        defaultSceneBodies.push_back(terrainState);
    }
    if (addNeedle) {
        const std::uint32_t needleBody =
            static_cast<std::uint32_t>(model.bodies.size());
        const std::uint32_t needleMaterial =
            static_cast<std::uint32_t>(model.materials.size());
        CurvedSutureNeedleAsset needle =
            makeCurvedSutureNeedleAsset({
                .bodyIndex = needleBody,
                .materialIndex = needleMaterial,
                .slotGenerationBase = 700001u,
                .collisionGroup = 1u,
                .collisionMask = ~0u,
                .motionType = MR_MOTION_DYNAMIC,
            });
        if (ccdMode == MetalWorldCCDMode::hybrid) {
            for (MRShapeGPU& shape : needle.rigid.shapes) {
                shape.flags |= MR_SHAPE_FLAG_ENABLE_CCD;
            }
        }

        constexpr std::uint32_t jawShapeIndex = 15u;
        if (jawShapeIndex >= model.shapes.size()) {
            throw std::runtime_error(
                "PSM jaw collision geometry is incomplete"
            );
        }
        const MRShapeGPU jawShape = model.shapes[jawShapeIndex];
        const ArticulatedPointQuery jawQuery{
            .bodyIndex = jawShape.bodyIndex,
            .localPoint = {
                jawShape.localPosition.x,
                jawShape.localPosition.y,
                jawShape.localPosition.z,
            },
        };
        std::vector<double> q64(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> v64(
            model.defaultV.begin(),
            model.defaultV.end()
        );
        ArticulatedPointKinematics jawKinematics{};
        std::vector<double> jawJacobian(
            3u * model.articulations.front().nv
        );
        const auto jawStatus =
            computeArticulatedPointJacobians(
                model,
                0u,
                q64,
                v64,
                std::span{&jawQuery, 1u},
                std::span{&jawKinematics, 1u},
                jawJacobian
            );
        if (!jawStatus.succeeded()) {
            throw std::runtime_error(
                "could not place the MLX surgical needle"
            );
        }
        const std::uint32_t graspShapeIndex =
            (
                needle.metadata.graspShapeBegin +
                needle.metadata.graspShapeEnd
            ) / 2u;
        if (graspShapeIndex >= needle.rigid.shapes.size()) {
            throw std::runtime_error(
                "surgical needle grasp geometry is incomplete"
            );
        }
        const MRShapeGPU graspShape =
            needle.rigid.shapes[graspShapeIndex];
        const float centerDistance =
            jawShape.dimensions.x +
            graspShape.dimensions.x +
            2.0e-4f;

        model.bodies.push_back(needle.rigid.body);
        model.materials.push_back(needle.rigid.material);
        model.shapes.insert(
            model.shapes.end(),
            needle.rigid.shapes.begin(),
            needle.rigid.shapes.end()
        );
        model.world.bodyCount =
            static_cast<std::uint32_t>(model.bodies.size());
        model.world.materialCount =
            static_cast<std::uint32_t>(model.materials.size());
        model.world.shapeCount =
            static_cast<std::uint32_t>(model.shapes.size());

        MRBodyStateGPU needleState{};
        needleState.position = {
            static_cast<float>(jawKinematics.position[0]) +
                centerDistance -
                graspShape.localPosition.x,
            static_cast<float>(jawKinematics.position[1]) -
                graspShape.localPosition.y,
            static_cast<float>(jawKinematics.position[2]) -
                graspShape.localPosition.z,
            1.0f,
        };
        needleState.orientation.w = 1.0f;
        needleState.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        needleState.flagsAndIndices[1] = MR_INVALID_INDEX;
        needleState.flagsAndIndices[2] = needleBody;
        defaultSceneBodies.push_back(needleState);
    }
    if (actuationMode ==
            MetalWorldActuationMode::implicitPositionDrive &&
        defaultSceneBodies.empty()) {
        throw std::invalid_argument(
            "MLX implicit_position currently requires a contact "
            "scene so it executes through WorldStepPrimitive"
        );
    }
    CompiledWorld compiled;
    const MetalWorldCapacityProfile requestedCapacityProfile =
        capacityProfile;
    if (!defaultSceneBodies.empty() &&
        !useHeterogeneousWorld) {
        const std::uint32_t contactCapacity =
            addPickPlace
            ? 128u
            : modelName == "franka"
            ? 32u
            : 64u;
        const std::uint32_t pairCapacity =
            addPickPlace
            ? 256u
            : modelName == "franka"
            ? 64u
            : modelName == "psm"
            ? 256u
            : 128u;
        const auto fill = [](
            std::uint32_t& selected,
            const std::uint32_t fallback
        ) {
            if (selected == 0u) {
                selected = fallback;
            }
        };
        fill(capacityProfile.candidatePairs, pairCapacity);
        fill(
            capacityProfile.rawContacts,
            2u * contactCapacity
        );
        fill(capacityProfile.manifolds, contactCapacity);
        fill(
            capacityProfile.constraintBlocks,
            contactCapacity
        );
        fill(
            capacityProfile.constraintRows,
            3u * capacityProfile.constraintBlocks
        );
        fill(
            capacityProfile.islands,
            addPickPlace ? 8u : 2u
        );
        fill(capacityProfile.hardConvexPairs, pairCapacity);
        fill(
            capacityProfile.meshTriangleCandidates,
            4u * contactCapacity
        );
        // Solver-tile and spill capacities remain zero here so the
        // authoritative CompiledWorld graph derives them from the actual
        // articulation/free-body topology and authored-constraint prefix.
        // Duplicating that inference in the MLX wrapper previously lagged
        // pick-place's second dynamic object and under-reserved its arena.
        fill(capacityProfile.ccdCandidates, pairCapacity);
        fill(
            capacityProfile.ccdEvents,
            MR_CCD_DEFAULT_MAX_EVENTS
        );
        if (addTerrain) {
            capacityProfile.meshTriangleCandidates =
                std::max<std::uint32_t>(
                    capacityProfile.meshTriangleCandidates,
                    4096u
                );
        }
    }
    const auto compileSelectedWorld = [&]() {
        return useHeterogeneousWorld
            ? compileMetalWorld(
                  heterogeneousWorld,
                  compiled,
                  capacityProfile
              )
            : compileMetalWorld(
                  model,
                  0u,
                  compiled,
                  capacityProfile
              );
    };
    auto diagnostics = compileSelectedWorld();
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            "could not compile MLX world: " +
            diagnostics.message
        );
    }
    if (!useHeterogeneousWorld &&
        !defaultSceneBodies.empty()) {
        // First compile discovers the canonical eligible-pair and
        // dynamic-node topology. Specialize the fixed MLX arenas from that
        // immutable graph while retaining the deliberately bounded raw and
        // solved-contact activity budgets above. This avoids multiplying
        // generic 128/256-slot reservations across thousands of worlds.
        const std::uint32_t eligiblePairs =
            std::max(compiled.eligiblePairCount(), 1u);
        const std::uint32_t dynamicNodes =
            std::max<std::uint32_t>(
                static_cast<std::uint32_t>(
                    compiled.dynamicNodes().size()
                ),
                1u
            );
        const bool hasMeshPairs = std::any_of(
            compiled.eligiblePairs().begin(),
            compiled.eligiblePairs().end(),
            [](const MRCompiledCollisionPairGPU& pair) {
                return
                    pair.pairClass == MR_COLLISION_PAIR_MESH;
            }
        );
        if (requestedCapacityProfile.candidatePairs == 0u) {
            capacityProfile.candidatePairs = eligiblePairs;
        }
        if (requestedCapacityProfile.manifolds == 0u) {
            capacityProfile.manifolds = std::min(
                capacityProfile.manifolds,
                eligiblePairs
            );
        }
        if (requestedCapacityProfile.hardConvexPairs == 0u) {
            capacityProfile.hardConvexPairs = std::min(
                capacityProfile.hardConvexPairs,
                eligiblePairs
            );
        }
        if (!hasMeshPairs &&
            requestedCapacityProfile
                    .meshTriangleCandidates == 0u) {
            capacityProfile.meshTriangleCandidates = 1u;
        }
        if (requestedCapacityProfile.ccdCandidates == 0u) {
            capacityProfile.ccdCandidates = std::min(
                capacityProfile.ccdCandidates,
                eligiblePairs
            );
        }
        if (requestedCapacityProfile.ccdEvents == 0u) {
            capacityProfile.ccdEvents = std::min(
                capacityProfile.ccdEvents,
                eligiblePairs
            );
        }
        if (requestedCapacityProfile.dynamicNodes == 0u) {
            capacityProfile.dynamicNodes = dynamicNodes;
        }
        if (requestedCapacityProfile
                .islandNodeReferences == 0u) {
            capacityProfile.islandNodeReferences =
                dynamicNodes;
        }
        diagnostics = compileSelectedWorld();
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                "could not compile topology-specialized MLX world: " +
                diagnostics.message
            );
        }
    }
    const std::size_t qualityNv =
        static_cast<std::size_t>(compiled.nv()) +
        6u * compiled.sceneBodyCount() +
        3u * compiled.rodNodeCount() +
        compiled.rodEdgeCount();
    if (solverMode == MetalWorldSolverMode::qualityNewton &&
        qualityNv >
            MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES) {
        throw std::invalid_argument(
            "quality_newton exceeds the compiled generalized "
            "velocity bucket"
        );
    }

    std::string metallibPath = requestedMetallibPath;
    if (metallibPath.empty()) {
        metallibPath = METALROBO_DEFAULT_METALLIB;
    }
    if (metallibPath.empty() ||
        !std::filesystem::is_regular_file(metallibPath)) {
        throw std::invalid_argument(
            "MetalRobo.metallib was not found; pass metallib_path"
        );
    }
    auto world = std::make_shared<MLXCompiledWorld>(
        std::move(compiled),
        controlTimestep,
        physicsSubsteps,
        applyBodyDamping,
        environmentCapacity,
        actuationMode,
        solverMode,
        velocityIterations,
        finalVelocityIterations,
        ccdMode,
        maxCCDAdvanceSolvePasses,
        maxCCDZeroTimeReplays,
        ccdSimultaneousTolerance,
        waveWorkerGroups,
        std::move(defaultSceneBodies),
        CookedTactileSystem{},
        0u,
        std::move(metallibPath)
    );
    world->prepareStream(stream);
    return world;
}

std::shared_ptr<MLXCompiledWorld> compileWorldPack(
    const std::string& path,
    const std::uint32_t environmentCapacity,
    MetalWorldCapacityProfile capacityProfile,
    float controlTimestep,
    const std::uint32_t physicsSubsteps,
    const bool applyBodyDamping,
    const std::string& requestedActuationMode,
    const std::string& requestedSolverMode,
    const std::uint32_t velocityIterations,
    const std::uint32_t finalVelocityIterations,
    const std::string& requestedCCDMode,
    const std::uint32_t maxCCDAdvanceSolvePasses,
    const std::uint32_t maxCCDZeroTimeReplays,
    const float ccdSimultaneousTolerance,
    const std::uint32_t waveWorkerGroups,
    const std::string& requestedMetallibPath,
    mx::StreamOrDevice stream
) {
    if (path.empty() || environmentCapacity == 0u ||
        physicsSubsteps == 0u || physicsSubsteps > 128u ||
        velocityIterations == 0u ||
        velocityIterations > 128u ||
        finalVelocityIterations > 128u ||
        maxCCDAdvanceSolvePasses == 0u ||
        maxCCDAdvanceSolvePasses >
            MR_CCD_MAX_ADVANCE_SOLVE_PASSES ||
        maxCCDZeroTimeReplays >
            MR_CCD_MAX_ZERO_TIME_REPLAYS ||
        !std::isfinite(ccdSimultaneousTolerance) ||
        !(ccdSimultaneousTolerance > 0.0f) ||
        (waveWorkerGroups != 0u &&
         waveWorkerGroups != 32u &&
         waveWorkerGroups != 64u &&
         waveWorkerGroups != 96u &&
         waveWorkerGroups != 128u)) {
        throw std::invalid_argument(
            "authored world compile options are invalid"
        );
    }

    MRWorldPack pack;
    const WorldPackResult loaded = readWorldPack(path, pack);
    if (!loaded.succeeded()) {
        throw std::invalid_argument(
            "authored world-pack load failed [" +
            std::string(worldPackStatusName(loaded.status)) +
            "]: " + loaded.message
        );
    }
    const WorldTemplate& authored = pack.family.worldTemplate;
    std::string reason;
    if (!authored.valid(&reason)) {
        throw std::invalid_argument(
            "authored world pack is incomplete: " + reason
        );
    }
    const AuthoredPhysicsBase base =
        authoredPhysicsBase(authored);
    if (controlTimestep == 0.0f) {
        controlTimestep =
            static_cast<float>(
                authored.task.controlPeriodSeconds
            );
    }
    if (!std::isfinite(controlTimestep) ||
        !(controlTimestep > 0.0f)) {
        throw std::invalid_argument(
            "control_timestep must be positive, or zero to use "
            "the authored task control period"
        );
    }

    MetalWorldActuationMode actuationMode;
    if (requestedActuationMode == "effort") {
        actuationMode = MetalWorldActuationMode::effort;
    } else if (requestedActuationMode == "implicit_position") {
        actuationMode =
            MetalWorldActuationMode::implicitPositionDrive;
    } else {
        throw std::invalid_argument(
            "actuation_mode must be 'effort' or "
            "'implicit_position'"
        );
    }
    MetalWorldSolverMode solverMode;
    if (requestedSolverMode == "free_motion_aba") {
        solverMode = MetalWorldSolverMode::freeMotionABA;
    } else if (requestedSolverMode == "throughput_pgs") {
        solverMode = MetalWorldSolverMode::throughputPGS;
    } else if (requestedSolverMode == "throughput_tgs") {
        solverMode = MetalWorldSolverMode::throughputTGS;
    } else if (requestedSolverMode == "quality_newton") {
        solverMode = MetalWorldSolverMode::qualityNewton;
    } else {
        throw std::invalid_argument(
            "solver_mode must be 'free_motion_aba', "
            "'throughput_pgs', 'throughput_tgs', or "
            "'quality_newton'"
        );
    }
    MetalWorldCCDMode ccdMode;
    if (requestedCCDMode == "disabled") {
        ccdMode = MetalWorldCCDMode::disabled;
    } else if (requestedCCDMode == "speculative") {
        ccdMode = MetalWorldCCDMode::speculative;
    } else if (requestedCCDMode == "hybrid") {
        ccdMode = MetalWorldCCDMode::hybrid;
    } else {
        throw std::invalid_argument(
            "ccd_mode must be 'disabled', 'speculative', or 'hybrid'"
        );
    }
    if (solverMode == MetalWorldSolverMode::freeMotionABA &&
        (!base.sceneBodies.empty() ||
         !authored.tactileSystem.sensors.empty())) {
        throw std::invalid_argument(
            "authored scene bodies and tactile sensors require a "
            "contact-capable solver"
        );
    }
    if (!authored.tactileSystem.sensors.empty() &&
        base.sceneBodies.empty()) {
        throw std::invalid_argument(
            "authored tactile world has no explicit scene body"
        );
    }
    if (solverMode == MetalWorldSolverMode::qualityNewton &&
        ccdMode == MetalWorldCCDMode::hybrid) {
        throw std::invalid_argument(
            "quality_newton requires disabled or speculative CCD"
        );
    }

    CompiledWorld compiled;
    const MetalWorldCompileDiagnostics diagnostics =
        compileMetalWorld(
            authored.engineModel,
            base.primaryArticulation,
            compiled,
            capacityProfile
        );
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            "could not compile authored MLX world: " +
            diagnostics.message
        );
    }
    const std::size_t qualityNv =
        static_cast<std::size_t>(compiled.nv()) +
        6u * compiled.sceneBodyCount() +
        3u * compiled.rodNodeCount() +
        compiled.rodEdgeCount();
    if (solverMode == MetalWorldSolverMode::qualityNewton &&
        qualityNv >
            MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES) {
        throw std::invalid_argument(
            "quality_newton exceeds the compiled generalized "
            "velocity bucket"
        );
    }

    std::string metallibPath = requestedMetallibPath;
    if (metallibPath.empty()) {
        metallibPath = METALROBO_DEFAULT_METALLIB;
    }
    if (metallibPath.empty() ||
        !std::filesystem::is_regular_file(metallibPath)) {
        throw std::invalid_argument(
            "MetalRobo.metallib was not found; pass metallib_path"
        );
    }
    auto world = std::make_shared<MLXCompiledWorld>(
        std::move(compiled),
        controlTimestep,
        physicsSubsteps,
        applyBodyDamping,
        environmentCapacity,
        actuationMode,
        solverMode,
        velocityIterations,
        finalVelocityIterations,
        ccdMode,
        maxCCDAdvanceSolvePasses,
        maxCCDZeroTimeReplays,
        ccdSimultaneousTolerance,
        waveWorkerGroups,
        base.sceneBodies,
        authored.tactileSystem,
        pack.contentHash,
        std::move(metallibPath)
    );
    world->prepareStream(stream);
    return world;
}

std::shared_ptr<MLXCompiledMultiArticulatedProgram>
compileMultiArticulatedProgram(
    const std::string& modelName,
    const std::uint32_t environmentCapacity,
    const std::string& requestedSolverMode,
    const std::uint32_t solverIterations,
    const float convergenceTolerance,
    const float timestep,
    const std::string& requestedMetallibPath,
    mx::StreamOrDevice stream
) {
    if (environmentCapacity == 0u ||
        solverIterations == 0u ||
        !std::isfinite(convergenceTolerance) ||
        !(convergenceTolerance > 0.0f) ||
        !std::isfinite(timestep) ||
        !(timestep > 0.0f)) {
        throw std::invalid_argument(
            "multi-articulation MLX capacities and solver "
            "configuration must be positive and finite"
        );
    }

    const DualPsmWorld dual = makeDualDvrkPsmWorld();
    EngineModel model;
    if (modelName == "dual_psm") {
        model = dual.model;
    } else if (modelName == "dual_psm_g1") {
        const EngineModel g1 = makeUnitreeG1EngineModel();
        const std::array components{
            EngineModelComponent{
                .model = &dual.model,
                .instanceId = "dual_psm_cell",
            },
            EngineModelComponent{
                .model = &g1,
                .instanceId = "g1_observer",
            },
        };
        const auto composed = composeEngineModels(
            components,
            model
        );
        if (!composed.succeeded()) {
            throw std::runtime_error(
                "could not compose MLX multi-articulation model: " +
                composed.message
            );
        }
    } else {
        throw std::invalid_argument(
            "multi-articulation model must be 'dual_psm' or "
            "'dual_psm_g1'"
        );
    }

    CompiledMetalMultiArticulatedProgram compiled;
    const auto compiledDiagnostics =
        compileMetalMultiArticulatedProgram(model, compiled);
    if (!compiledDiagnostics.succeeded()) {
        throw std::runtime_error(
            "could not compile MLX multi-articulation program: " +
            compiledDiagnostics.message
        );
    }
    MetalMultiArticulatedConstraintConfig config;
    if (requestedSolverMode == "throughput_pgs") {
        config.solverMode =
            MetalGeneralizedConstraintSolverMode::throughputPGS;
    } else if (requestedSolverMode ==
               "quality_semismooth_newton") {
        config.solverMode =
            MetalGeneralizedConstraintSolverMode::
                qualitySemismoothNewton;
    } else {
        throw std::invalid_argument(
            "multi-articulation solver_mode must be "
            "'throughput_pgs' or 'quality_semismooth_newton'"
        );
    }
    config.solverIterations = solverIterations;
    config.convergenceTolerance = convergenceTolerance;
    config.evaluation.timestep = timestep;
    config.evaluation.minimumRegularization = 1.0e-7;

    std::string metallibPath = requestedMetallibPath;
    if (metallibPath.empty()) {
        metallibPath = METALROBO_DEFAULT_METALLIB;
    }
    if (metallibPath.empty() ||
        !std::filesystem::is_regular_file(metallibPath)) {
        throw std::invalid_argument(
            "MetalRobo.metallib was not found; pass metallib_path"
        );
    }
    auto program = std::make_shared<
        MLXCompiledMultiArticulatedProgram
    >(
        std::move(compiled),
        environmentCapacity,
        std::move(config),
        std::move(metallibPath)
    );
    program->prepareStream(stream);
    return program;
}

std::vector<mx::array> generalizedConstraintStep(
    const std::shared_ptr<
        MLXCompiledMultiArticulatedProgram
    >& program,
    const mx::array& q,
    const mx::array& freeVelocity,
    mx::StreamOrDevice stream
) {
    if (program == nullptr) {
        throw std::invalid_argument(
            "program must be an MLXCompiledMultiArticulatedProgram"
        );
    }
    const auto& compiled = program->program();
    const auto environments =
        static_cast<mx::ShapeElem>(
            program->environmentCapacity()
        );
    const mx::Shape qShape{
        environments,
        static_cast<mx::ShapeElem>(
            compiled.model().world.nq
        ),
    };
    const mx::Shape vShape{
        environments,
        static_cast<mx::ShapeElem>(
            compiled.model().world.nv
        ),
    };
    validateInput(q, qShape, "q");
    validateInput(
        freeVelocity,
        vShape,
        "free_velocity"
    );

    const auto selectedStream = mx::to_stream(stream);
    const std::vector<mx::array> inputs{
        mx::contiguous(q, false, selectedStream),
        mx::contiguous(
            freeVelocity,
            false,
            selectedStream
        ),
    };
    const auto primitive = std::make_shared<
        GeneralizedConstraintStepPrimitive
    >(selectedStream, program);
    return mx::array::make_arrays(
        {
            vShape,
            {
                environments,
                static_cast<mx::ShapeElem>(
                    compiled.rowCount()
                ),
            },
            {
                environments,
                static_cast<mx::ShapeElem>(
                    kGeneralizedStatusWords
                ),
            },
        },
        {
            mx::float32,
            mx::float32,
            mx::uint32,
        },
        primitive,
        inputs
    );
}

std::vector<mx::array> abaStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& effort,
    mx::StreamOrDevice stream
) {
    if (world == nullptr) {
        throw std::invalid_argument(
            "world must be an MLXCompiledWorld"
        );
    }
    if (q.ndim() != 2u || q.shape(0) <= 0) {
        throw std::invalid_argument(
            "q must have shape [environment, nq]"
        );
    }
    const auto environmentCount = q.shape(0);
    if (static_cast<std::uint64_t>(environmentCount) >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "environment count exceeds the Metal ABI"
        );
    }
    const mx::Shape qShape{
        environmentCount,
        static_cast<mx::ShapeElem>(world->world().nq()),
    };
    const mx::Shape vShape{
        environmentCount,
        static_cast<mx::ShapeElem>(world->world().nv()),
    };
    validateInput(q, qShape, "q");
    validateInput(v, vShape, "v");
    validateInput(effort, vShape, "actions");

    const auto selectedStream = mx::to_stream(stream);
    const std::vector<mx::array> inputs{
        mx::contiguous(q, false, selectedStream),
        mx::contiguous(v, false, selectedStream),
        mx::contiguous(effort, false, selectedStream),
    };
    const auto primitive = std::make_shared<
        ABAWorldStepPrimitive
    >(selectedStream, world);
    return mx::array::make_arrays(
        {
            qShape,
            vShape,
            vShape,
            {
                environmentCount,
                static_cast<mx::ShapeElem>(kStatusWords),
            },
        },
        {
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
        },
        primitive,
        inputs
    );
}

std::vector<mx::array> worldStep(
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
    mx::StreamOrDevice stream
) {
    if (world == nullptr ||
        world->solverMode() ==
            MetalWorldSolverMode::freeMotionABA) {
        throw std::invalid_argument(
            "world_step requires a contact-capable MLXCompiledWorld"
        );
    }
    if (q.ndim() != 2u || q.shape(0) <= 0) {
        throw std::invalid_argument(
            "q must have shape [environment, nq]"
        );
    }
    const auto environments = q.shape(0);
    if (static_cast<std::uint64_t>(environments) >
        world->environmentCapacity()) {
        throw std::invalid_argument(
            "environment batch exceeds compile_world environment_capacity"
        );
    }
    const auto sceneBodies =
        static_cast<mx::ShapeElem>(
            world->world().sceneBodyCount()
        );
    if (sceneBodies == 0) {
        throw std::invalid_argument(
            "contact MLX primitive requires an explicit scene state"
        );
    }
    const auto manifolds =
        static_cast<mx::ShapeElem>(
            world->world().capacities().manifolds
        );
    const auto pairs =
        static_cast<mx::ShapeElem>(
            world->world().eligiblePairCount()
        );
    const auto contacts =
        static_cast<mx::ShapeElem>(
            world->world().capacities().constraintBlocks
        );
    const auto rodNodes =
        static_cast<mx::ShapeElem>(
            world->world().rodNodeCount()
        );
    const auto rodEdges =
        static_cast<mx::ShapeElem>(
            world->world().rodEdgeCount()
        );
    const auto rodWitnesses =
        static_cast<mx::ShapeElem>(
            world->world().rodToolPairs().size() *
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR
        );
    const mx::Shape qShape{
        environments,
        static_cast<mx::ShapeElem>(world->world().nq()),
    };
    const mx::Shape vShape{
        environments,
        static_cast<mx::ShapeElem>(world->world().nv()),
    };
    const mx::Shape sceneShape{
        environments,
        sceneBodies,
        4,
    };
    const mx::Shape headerShape{
        environments,
        manifolds,
        static_cast<mx::ShapeElem>(kManifoldHeaderWords),
    };
    const mx::Shape pointShape{
        environments,
        manifolds,
        static_cast<mx::ShapeElem>(
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        ),
        static_cast<mx::ShapeElem>(kManifoldPointWords),
    };
    const mx::Shape countShape{environments};
    const mx::Shape cacheShape{
        environments,
        pairs,
        static_cast<mx::ShapeElem>(kConvexCacheWords),
    };
    const mx::Shape rodNodeShape{
        environments,
        rodNodes,
        4,
    };
    const mx::Shape rodEdgeShape{
        environments,
        rodEdges,
    };
    const mx::Shape rodWitnessShape{
        environments,
        rodWitnesses,
        static_cast<mx::ShapeElem>(kRodWitnessWords),
    };
    const mx::Shape controllerShape{
        environments,
        static_cast<mx::ShapeElem>(
            world->world().model().articulations.size()
        ),
        4,
    };
    const mx::Shape bodyParameterShape{
        environments,
        static_cast<mx::ShapeElem>(
            world->world().model().bodies.size()
        ),
        4,
    };
    const auto tactileSamples = static_cast<mx::ShapeElem>(
        world->tactile().samples.size()
    );
    const auto tactileSensors = static_cast<mx::ShapeElem>(
        world->tactile().sensors.size()
    );
    const mx::Shape tactileDenseShape{
        environments,
        tactileSamples,
    };
    const mx::Shape tactileMotionShape{
        environments,
        tactileSamples,
        4,
    };
    const mx::Shape tactileClockShape =
        world->hasTactile()
        ? mx::Shape{environments}
        : mx::Shape{environments, 0};
    const mx::Shape tactileSummaryShape{
        environments,
        tactileSensors,
        4,
    };
    const mx::Shape actuatorProfileShape{
        environments,
        static_cast<mx::ShapeElem>(world->world().nv()),
        7,
    };
    const mx::Shape tactileStatusShape{
        environments,
        tactileSensors,
        static_cast<mx::ShapeElem>(
            sizeof(MRTactileStatusGPU) /
            sizeof(std::uint32_t)
        ),
    };
    constexpr mx::ShapeElem bodyRecordWords =
        sizeof(MRBodyStateGPU) / sizeof(std::uint32_t);
    const mx::Shape bodyStateShape = world->hasTactile()
        ? mx::Shape{
              environments,
              static_cast<mx::ShapeElem>(
                  world->world().model().bodies.size()
              ),
              bodyRecordWords,
          }
        : mx::Shape{
              environments,
              0,
              bodyRecordWords,
          };
    validateInput(q, qShape, "q");
    validateInput(v, vShape, "v");
    validateInput(effort, vShape, "actions");
    validateInput(scenePosition, sceneShape, "scene_position");
    validateInput(
        sceneOrientation,
        sceneShape,
        "scene_orientation"
    );
    validateInput(
        sceneLinearVelocity,
        sceneShape,
        "scene_linear_velocity"
    );
    validateInput(
        sceneAngularVelocity,
        sceneShape,
        "scene_angular_velocity"
    );
    const auto validateU32 = [](
        const mx::array& value,
        const mx::Shape& shape,
        const char* label
    ) {
        if (value.dtype() != mx::uint32 ||
            value.shape() != shape) {
            throw std::invalid_argument(
                std::string(label) +
                " must be a uint32 MLX array with the compiled shape"
            );
        }
    };
    validateU32(manifoldHeaders, headerShape, "manifold_headers");
    validateU32(manifoldPoints, pointShape, "manifold_points");
    validateU32(manifoldCounts, countShape, "manifold_counts");
    validateU32(pairCache, cacheShape, "pair_cache");
    validateInput(
        rodPositions,
        rodNodeShape,
        "rod_positions"
    );
    validateInput(
        rodVelocities,
        rodNodeShape,
        "rod_velocities"
    );
    validateInput(rodTwists, rodEdgeShape, "rod_twists");
    validateInput(
        rodTwistRates,
        rodEdgeShape,
        "rod_twist_rates"
    );
    validateU32(
        rodWitnessCache,
        rodWitnessShape,
        "rod_witness_cache"
    );
    validateInput(
        bodyParameters,
        bodyParameterShape,
        "body_parameters"
    );
    validateInput(
        controllerParameters,
        controllerShape,
        "controller_parameters"
    );
    validateInput(
        tactilePreviousDepth,
        tactileDenseShape,
        "tactile_previous_depth"
    );
    validateInput(
        tactilePreviousMotion,
        tactileMotionShape,
        "tactile_previous_motion"
    );
    validateInput(
        tactileTargetAnchor,
        tactileMotionShape,
        "tactile_target_anchor"
    );
    validateInput(
        tactileTimestamp,
        tactileClockShape,
        "tactile_timestamp"
    );
    validateU32(
        tactilePreviousValidity,
        tactileDenseShape,
        "tactile_previous_validity"
    );
    validateU32(
        tactilePreviousObject,
        tactileDenseShape,
        "tactile_previous_object"
    );
    validateU32(resetMask, countShape, "reset_mask");
    validateInput(
        actuatorProfileValues,
        actuatorProfileShape,
        "actuator_profile_values"
    );
    if (tactileFrameIndex.dtype() != mx::uint64 ||
        tactileFrameIndex.shape() != tactileClockShape) {
        throw std::invalid_argument(
            "tactile_frame_index must be a uint64 MLX array "
            "with the compiled shape"
        );
    }

    const auto selectedStream = mx::to_stream(stream);
    std::vector<mx::array> inputs{
        mx::contiguous(q, false, selectedStream),
        mx::contiguous(v, false, selectedStream),
        mx::contiguous(effort, false, selectedStream),
        mx::contiguous(scenePosition, false, selectedStream),
        mx::contiguous(sceneOrientation, false, selectedStream),
        mx::contiguous(sceneLinearVelocity, false, selectedStream),
        mx::contiguous(sceneAngularVelocity, false, selectedStream),
        mx::contiguous(manifoldHeaders, false, selectedStream),
        mx::contiguous(manifoldPoints, false, selectedStream),
        mx::contiguous(manifoldCounts, false, selectedStream),
        mx::contiguous(pairCache, false, selectedStream),
        mx::contiguous(rodPositions, false, selectedStream),
        mx::contiguous(rodVelocities, false, selectedStream),
        mx::contiguous(rodTwists, false, selectedStream),
        mx::contiguous(rodTwistRates, false, selectedStream),
        mx::contiguous(
            rodWitnessCache,
            false,
            selectedStream
        ),
        mx::contiguous(bodyParameters, false, selectedStream),
        mx::contiguous(
            controllerParameters,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactilePreviousDepth,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactilePreviousValidity,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactilePreviousObject,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactilePreviousMotion,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactileTargetAnchor,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactileFrameIndex,
            false,
            selectedStream
        ),
        mx::contiguous(
            tactileTimestamp,
            false,
            selectedStream
        ),
        mx::contiguous(resetMask, false, selectedStream),
        mx::contiguous(
            actuatorProfileValues,
            false,
            selectedStream
        ),
    };
    const auto primitive =
        std::make_shared<WorldStepPrimitive>(
            selectedStream,
            world
        );
    return mx::array::make_arrays(
        {
            qShape,
            vShape,
            sceneShape,
            sceneShape,
            sceneShape,
            sceneShape,
            headerShape,
            pointShape,
            countShape,
            cacheShape,
            rodNodeShape,
            rodNodeShape,
            rodEdgeShape,
            rodEdgeShape,
            rodWitnessShape,
            vShape,
            {
                environments,
                static_cast<mx::ShapeElem>(kContactStatusWords),
            },
            {
                environments,
                contacts,
                static_cast<mx::ShapeElem>(kEvidenceFloatWidth),
            },
            {
                environments,
                contacts,
                static_cast<mx::ShapeElem>(kEvidenceIDWidth),
            },
            countShape,
            {environments, contacts},
            tactileDenseShape,
            tactileDenseShape,
            tactileMotionShape,
            tactileDenseShape,
            tactileDenseShape,
            tactileMotionShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileSummaryShape,
            tactileStatusShape,
            tactileMotionShape,
            tactileMotionShape,
            tactileDenseShape,
            bodyStateShape,
        },
        {
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
            mx::uint32,
            mx::uint32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::float32,
            mx::uint32,
            mx::float32,
            mx::uint32,
            mx::uint32,
            mx::uint32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
        },
        primitive,
        inputs
    );
}

std::vector<mx::array> worldFamilyState(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const std::uintptr_t resetQBuffer,
    const std::uintptr_t resetVBuffer,
    const std::uintptr_t resetSceneBodiesBuffer,
    const std::uintptr_t scenarioHeadersBuffer,
    const std::uintptr_t scenarioValuesBuffer,
    const std::uintptr_t bodyParametersBuffer,
    const std::uintptr_t controllerParametersBuffer,
    const std::uint32_t environmentCount,
    const std::uint32_t variationCount,
    const std::uint32_t bodyCount,
    const std::uint32_t articulationCount,
    const std::uint64_t generation,
    const std::uint64_t authoredPackHash,
    mx::StreamOrDevice stream
) {
    if (world == nullptr ||
        resetQBuffer == 0u ||
        resetVBuffer == 0u ||
        resetSceneBodiesBuffer == 0u ||
        scenarioHeadersBuffer == 0u ||
        scenarioValuesBuffer == 0u ||
        bodyParametersBuffer == 0u ||
        controllerParametersBuffer == 0u) {
        throw std::invalid_argument(
            "world-family state requires a compiled world and "
            "non-null Metal reset/scenario/parameter buffers"
        );
    }
    if (world->authoredPackHash() != authoredPackHash) {
        throw std::invalid_argument(
            "sampled world family and compiled MLX physics must use "
            "the same authored pack"
        );
    }
    if (environmentCount == 0u ||
        environmentCount > world->environmentCapacity()) {
        throw std::invalid_argument(
            "world-family environment count exceeds the compiled "
            "MLX world capacity"
        );
    }
    const auto sceneBodyCount =
        static_cast<mx::ShapeElem>(
            world->world().sceneBodyCount()
        );
    if (sceneBodyCount == 0) {
        throw std::invalid_argument(
            "world-family state requires a contact-capable scene"
        );
    }
    if (variationCount == 0u ||
        bodyCount != world->world().model().bodies.size() ||
        articulationCount !=
            world->world().model().articulations.size()) {
        throw std::invalid_argument(
            "world-family scenario or parameter topology does not "
            "match the compiled MLX world"
        );
    }
    const mx::Shape qShape{
        static_cast<mx::ShapeElem>(environmentCount),
        static_cast<mx::ShapeElem>(world->world().nq()),
    };
    const mx::Shape vShape{
        static_cast<mx::ShapeElem>(environmentCount),
        static_cast<mx::ShapeElem>(world->world().nv()),
    };
    const mx::Shape sceneShape{
        static_cast<mx::ShapeElem>(environmentCount),
        sceneBodyCount,
        4,
    };
    const auto selectedStream = mx::to_stream(stream);
    const auto primitive =
        std::make_shared<WorldFamilyStatePrimitive>(
            selectedStream,
            world,
            reinterpret_cast<MTL::Buffer*>(resetQBuffer),
            reinterpret_cast<MTL::Buffer*>(resetVBuffer),
            reinterpret_cast<MTL::Buffer*>(
                resetSceneBodiesBuffer
            ),
            reinterpret_cast<MTL::Buffer*>(
                scenarioHeadersBuffer
            ),
            reinterpret_cast<MTL::Buffer*>(
                scenarioValuesBuffer
            ),
            reinterpret_cast<MTL::Buffer*>(
                bodyParametersBuffer
            ),
            reinterpret_cast<MTL::Buffer*>(
                controllerParametersBuffer
            ),
            environmentCount,
            variationCount,
            bodyCount,
            articulationCount,
            generation
        );
    return mx::array::make_arrays(
        {
            qShape,
            vShape,
            sceneShape,
            sceneShape,
            sceneShape,
            sceneShape,
            {
                static_cast<mx::ShapeElem>(environmentCount),
                3,
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(variationCount),
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(variationCount),
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(bodyCount),
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(bodyCount),
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(articulationCount),
                4,
            },
            {
                static_cast<mx::ShapeElem>(environmentCount),
                static_cast<mx::ShapeElem>(articulationCount),
                4,
            },
        },
        {
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::float32,
            mx::uint32,
            mx::float32,
            mx::uint32,
            mx::float32,
            mx::uint32,
        },
        primitive,
        {}
    );
}

std::vector<mx::array> visualObservation(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const std::uintptr_t rendererHandle,
    const std::uintptr_t worldFamilyHandle,
    const mx::array& currentBodyStates,
    const mx::array& previousBodyStates,
    const std::uint64_t frameIndex,
    const std::uint32_t sensorSequence,
    const std::uint32_t cameraIndex,
    const std::uint32_t outputMask,
    mx::StreamOrDevice stream
) {
    auto* renderer = reinterpret_cast<MRHybridRendererHandle*>(
        rendererHandle
    );
    auto* worlds = reinterpret_cast<MRWorldFamilyHandle*>(
        worldFamilyHandle
    );
    if (world == nullptr || renderer == nullptr || worlds == nullptr) {
        throw std::invalid_argument(
            "visual observation requires a compiled authored world plus "
            "live renderer and world-family handles"
        );
    }
    const std::uint64_t compiledPackHash =
        world->authoredPackHash();
    const std::uint64_t sampledPackHash =
        mr_world_family_authored_pack_hash(worlds);
    if (compiledPackHash == 0u ||
        sampledPackHash == 0u ||
        compiledPackHash != sampledPackHash) {
        throw std::invalid_argument(
            "visual observation physics and sampled worlds must come "
            "from the same explicit authored pack"
        );
    }
    if ((outputMask &
         ~static_cast<std::uint32_t>(
             MR_HYBRID_OUTPUT_ALL_TRUTH
         )) != 0u) {
        throw std::invalid_argument(
            "visual observation output selection is invalid"
        );
    }
    const MRHybridRendererLayoutC rendererLayout =
        mr_hybrid_renderer_layout(renderer);
    const MRWorldFamilyLayoutC worldLayout =
        mr_world_family_layout(worlds);
    const std::uint32_t environments =
        worldLayout.active_instance_count;
    constexpr mx::ShapeElem bodyRecordWords =
        sizeof(MRBodyStateGPU) / sizeof(std::uint32_t);
    const mx::Shape bodyShape{
        static_cast<mx::ShapeElem>(environments),
        static_cast<mx::ShapeElem>(rendererLayout.body_count),
        bodyRecordWords,
    };
    if (environments == 0u ||
        environments > rendererLayout.capacity ||
        environments > world->environmentCapacity() ||
        rendererLayout.body_count == 0u ||
        rendererLayout.body_count !=
            world->world().model().bodies.size() ||
        cameraIndex >= worldLayout.sensor_count_per_instance ||
        currentBodyStates.dtype() != mx::uint32 ||
        previousBodyStates.dtype() != mx::uint32 ||
        currentBodyStates.shape() != bodyShape ||
        previousBodyStates.shape() != bodyShape) {
        throw std::invalid_argument(
            "visual body records, active worlds, or camera do not match "
            "the compiled renderer"
        );
    }
    const auto selectedStream = mx::to_stream(stream);
    const auto primitive =
        std::make_shared<VisualObservationPrimitive>(
            selectedStream,
            world,
            renderer,
            worlds,
            environments,
            rendererLayout.body_count,
            rendererLayout.width,
            rendererLayout.height,
            frameIndex,
            sensorSequence,
            cameraIndex,
            outputMask
        );
    const mx::Shape scalarShape{
        static_cast<mx::ShapeElem>(environments),
        static_cast<mx::ShapeElem>(rendererLayout.height),
        static_cast<mx::ShapeElem>(rendererLayout.width),
    };
    const mx::Shape vectorShape{
        static_cast<mx::ShapeElem>(environments),
        static_cast<mx::ShapeElem>(rendererLayout.height),
        static_cast<mx::ShapeElem>(rendererLayout.width),
        4,
    };
    const mx::Shape emptyScalarShape{
        static_cast<mx::ShapeElem>(environments),
        0,
        0,
    };
    const mx::Shape emptyVectorShape{
        static_cast<mx::ShapeElem>(environments),
        0,
        0,
        4,
    };
    return mx::array::make_arrays(
        {
            vectorShape,
            scalarShape,
            (outputMask & MR_HYBRID_OUTPUT_SEGMENTATION) != 0u
                ? scalarShape
                : emptyScalarShape,
            (outputMask & MR_HYBRID_OUTPUT_IDENTITIES) != 0u
                ? vectorShape
                : emptyVectorShape,
            (outputMask & MR_HYBRID_OUTPUT_NORMALS) != 0u
                ? vectorShape
                : emptyVectorShape,
            (outputMask & MR_HYBRID_OUTPUT_MOTION) != 0u
                ? vectorShape
                : emptyVectorShape,
            scalarShape,
        },
        {
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
            mx::float32,
            mx::float32,
            mx::uint32,
        },
        primitive,
        {
            mx::contiguous(
                currentBodyStates,
                false,
                selectedStream
            ),
            mx::contiguous(
                previousBodyStates,
                false,
                selectedStream
            ),
        }
    );
}

std::vector<mx::array> materializeBodyStates(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& q,
    const mx::array& v,
    const mx::array& scenePosition,
    const mx::array& sceneOrientation,
    const mx::array& sceneLinearVelocity,
    const mx::array& sceneAngularVelocity,
    mx::StreamOrDevice stream
) {
    if (world == nullptr || q.ndim() != 2u ||
        q.shape(0) <= 0) {
        throw std::invalid_argument(
            "body-state materialization requires a compiled world "
            "and environment-major q"
        );
    }
    const auto environments = q.shape(0);
    if (static_cast<std::uint64_t>(environments) >
        world->environmentCapacity()) {
        throw std::invalid_argument(
            "body-state environment count exceeds compiled capacity"
        );
    }
    const auto bodies = static_cast<mx::ShapeElem>(
        world->world().model().bodies.size()
    );
    const auto scenes = static_cast<mx::ShapeElem>(
        world->world().sceneBodyCount()
    );
    const mx::Shape qShape{
        environments,
        static_cast<mx::ShapeElem>(world->world().nq()),
    };
    const mx::Shape vShape{
        environments,
        static_cast<mx::ShapeElem>(world->world().nv()),
    };
    const mx::Shape sceneShape{environments, scenes, 4};
    validateInput(q, qShape, "q");
    validateInput(v, vShape, "v");
    validateInput(
        scenePosition,
        sceneShape,
        "scene_position"
    );
    validateInput(
        sceneOrientation,
        sceneShape,
        "scene_orientation"
    );
    validateInput(
        sceneLinearVelocity,
        sceneShape,
        "scene_linear_velocity"
    );
    validateInput(
        sceneAngularVelocity,
        sceneShape,
        "scene_angular_velocity"
    );
    constexpr mx::ShapeElem bodyWords =
        sizeof(MRBodyStateGPU) / sizeof(std::uint32_t);
    const auto selectedStream = mx::to_stream(stream);
    const auto primitive = std::make_shared<BodyStatePrimitive>(
        selectedStream,
        world
    );
    return mx::array::make_arrays(
        {{environments, bodies, bodyWords}},
        {mx::uint32},
        primitive,
        {
            mx::contiguous(q, false, selectedStream),
            mx::contiguous(v, false, selectedStream),
            mx::contiguous(
                scenePosition,
                false,
                selectedStream
            ),
            mx::contiguous(
                sceneOrientation,
                false,
                selectedStream
            ),
            mx::contiguous(
                sceneLinearVelocity,
                false,
                selectedStream
            ),
            mx::contiguous(
                sceneAngularVelocity,
                false,
                selectedStream
            ),
        }
    );
}

std::vector<mx::array> sceneRaycast(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& bodyStates,
    const mx::array& origins,
    const mx::array& directions,
    const mx::array& maximumDistances,
    const mx::array& options,
    mx::StreamOrDevice stream
) {
    if (world == nullptr ||
        origins.ndim() != 3u ||
        origins.shape(0) <= 0 ||
        origins.shape(1) <= 0) {
        throw std::invalid_argument(
            "scene raycast requires a compiled world and "
            "environment-major rays"
        );
    }
    const auto environments = origins.shape(0);
    const auto rays = origins.shape(1);
    if (static_cast<std::uint64_t>(environments) >
            world->environmentCapacity() ||
        static_cast<std::uint64_t>(rays) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "scene raycast dimensions exceed compiled capacity"
        );
    }
    constexpr mx::ShapeElem bodyWords =
        sizeof(MRBodyStateGPU) / sizeof(std::uint32_t);
    const mx::Shape bodyShape{
        environments,
        static_cast<mx::ShapeElem>(
            world->world().model().bodies.size()
        ),
        bodyWords,
    };
    const mx::Shape rayShape{environments, rays, 4};
    const mx::Shape scalarShape{environments, rays};
    if (bodyStates.dtype() != mx::uint32 ||
        bodyStates.shape() != bodyShape ||
        origins.dtype() != mx::float32 ||
        origins.shape() != rayShape ||
        directions.dtype() != mx::float32 ||
        directions.shape() != rayShape ||
        maximumDistances.dtype() != mx::float32 ||
        maximumDistances.shape() != scalarShape ||
        options.dtype() != mx::uint32 ||
        options.shape() != rayShape) {
        throw std::invalid_argument(
            "scene raycast arrays do not match the compiled "
            "environment/ray layout"
        );
    }
    const auto selectedStream = mx::to_stream(stream);
    const auto primitive =
        std::make_shared<SceneRaycastPrimitive>(
            selectedStream,
            world,
            static_cast<std::uint32_t>(environments),
            static_cast<std::uint32_t>(rays),
            false
        );
    return mx::array::make_arrays(
        {
            scalarShape,
            rayShape,
            rayShape,
            rayShape,
            scalarShape,
        },
        {
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
        },
        primitive,
        {
            mx::contiguous(
                bodyStates,
                false,
                selectedStream
            ),
            mx::contiguous(origins, false, selectedStream),
            mx::contiguous(
                directions,
                false,
                selectedStream
            ),
            mx::contiguous(
                maximumDistances,
                false,
                selectedStream
            ),
            mx::contiguous(options, false, selectedStream),
        }
    );
}

std::vector<mx::array> sceneRaycastPattern(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const mx::array& bodyStates,
    const mx::array& parentBodies,
    const mx::array& localOrigins,
    const mx::array& localDirections,
    const mx::array& maximumDistances,
    const mx::array& options,
    mx::StreamOrDevice stream
) {
    if (world == nullptr ||
        parentBodies.ndim() != 1u ||
        parentBodies.shape(0) <= 0 ||
        bodyStates.ndim() != 3u ||
        bodyStates.shape(0) <= 0) {
        throw std::invalid_argument(
            "mounted scene raycast requires a compiled world, "
            "body states, and a nonempty shared ray pattern"
        );
    }
    const auto environments = bodyStates.shape(0);
    const auto rays = parentBodies.shape(0);
    if (static_cast<std::uint64_t>(environments) >
            world->environmentCapacity() ||
        static_cast<std::uint64_t>(rays) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "mounted scene raycast dimensions exceed compiled capacity"
        );
    }
    constexpr mx::ShapeElem bodyWords =
        sizeof(MRBodyStateGPU) / sizeof(std::uint32_t);
    const mx::Shape bodyShape{
        environments,
        static_cast<mx::ShapeElem>(
            world->world().model().bodies.size()
        ),
        bodyWords,
    };
    const mx::Shape patternShape{rays, 4};
    const mx::Shape rayShape{environments, rays, 4};
    const mx::Shape scalarShape{environments, rays};
    if (bodyStates.dtype() != mx::uint32 ||
        bodyStates.shape() != bodyShape ||
        parentBodies.dtype() != mx::uint32 ||
        localOrigins.dtype() != mx::float32 ||
        localOrigins.shape() != patternShape ||
        localDirections.dtype() != mx::float32 ||
        localDirections.shape() != patternShape ||
        maximumDistances.dtype() != mx::float32 ||
        maximumDistances.shape() != scalarShape ||
        options.dtype() != mx::uint32 ||
        options.shape() != rayShape) {
        throw std::invalid_argument(
            "mounted scene raycast arrays do not match the compiled "
            "body/pattern layout"
        );
    }
    const auto selectedStream = mx::to_stream(stream);
    const auto primitive =
        std::make_shared<SceneRaycastPrimitive>(
            selectedStream,
            world,
            static_cast<std::uint32_t>(environments),
            static_cast<std::uint32_t>(rays),
            true
        );
    return mx::array::make_arrays(
        {
            scalarShape,
            rayShape,
            rayShape,
            rayShape,
            scalarShape,
        },
        {
            mx::float32,
            mx::float32,
            mx::float32,
            mx::uint32,
            mx::uint32,
        },
        primitive,
        {
            mx::contiguous(
                bodyStates,
                false,
                selectedStream
            ),
            mx::contiguous(
                parentBodies,
                false,
                selectedStream
            ),
            mx::contiguous(
                localOrigins,
                false,
                selectedStream
            ),
            mx::contiguous(
                localDirections,
                false,
                selectedStream
            ),
            mx::contiguous(
                maximumDistances,
                false,
                selectedStream
            ),
            mx::contiguous(options, false, selectedStream),
        }
    );
}

std::vector<float> debugCPUStep(
    const std::shared_ptr<MLXCompiledWorld>& world,
    const std::vector<float>& q,
    const std::vector<float>& v,
    const std::vector<float>& effort
) {
    if (world == nullptr) {
        throw std::invalid_argument(
            "world must be an MLXCompiledWorld"
        );
    }
    const CompiledWorld& compiled = world->world();
    if (q.size() != compiled.nq() ||
        v.size() != compiled.nv() ||
        effort.size() != compiled.nv()) {
        throw std::invalid_argument(
            "debug CPU state does not match the compiled world"
        );
    }
    std::vector<double> q64(q.begin(), q.end());
    std::vector<double> v64(v.begin(), v.end());
    std::vector<double> effort64(
        effort.begin(),
        effort.end()
    );
    const MRWorldGPU& record = compiled.model().world;
    ArticulatedDynamicsConfig config{};
    config.gravity = {
        record.gravityAndTimestep.x,
        record.gravityAndTimestep.y,
        record.gravityAndTimestep.z,
    };
    config.timestep =
        static_cast<double>(world->controlTimestep()) /
        static_cast<double>(world->physicsSubsteps());
    config.applyBodyDamping = world->applyBodyDamping();
    for (std::uint32_t substep = 0u;
         substep < world->physicsSubsteps();
         ++substep) {
        const auto diagnostics = integrateArticulatedState(
            compiled.model(),
            compiled.articulationIndex(),
            q64,
            v64,
            effort64,
            {},
            config
        );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                "FP64 debug oracle rejected the step at substep " +
                std::to_string(substep)
            );
        }
    }
    std::vector<float> result;
    result.reserve(q64.size() + v64.size());
    for (const double value : q64) {
        result.push_back(static_cast<float>(value));
    }
    for (const double value : v64) {
        result.push_back(static_cast<float>(value));
    }
    return result;
}

ABAWorldStepPrimitive::ABAWorldStepPrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world
)
    : mx::Primitive(stream), world_(std::move(world)) {}

void ABAWorldStepPrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo physics has no MLX CPU fallback"
    );
}

void ABAWorldStepPrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (inputs.size() != 3u || outputs.size() != 4u) {
        throw std::runtime_error(
            "MetalRobo MLX primitive received an invalid graph"
        );
    }
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalResources& resources = world_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);

    for (mx::array& output : outputs) {
        output.set_data(
            mx::allocator::malloc(output.nbytes())
        );
    }

    const auto environmentCount =
        static_cast<std::uint32_t>(inputs[0].shape(0));
    const auto nq = world_->world().nq();
    const auto nv = world_->world().nv();
    const auto articulationIndex =
        world_->world().articulationIndex();
    const mx::Shape qShape{
        static_cast<mx::ShapeElem>(environmentCount),
        static_cast<mx::ShapeElem>(nq),
    };
    const mx::Shape vShape{
        static_cast<mx::ShapeElem>(environmentCount),
        static_cast<mx::ShapeElem>(nv),
    };
    mx::array candidateQ =
        temporary(qShape, mx::float32);
    mx::array candidateV =
        temporary(vShape, mx::float32);
    mx::array candidateAcceleration =
        temporary(vShape, mx::float32);
    mx::array abaStatuses = temporary(
        {
            static_cast<mx::ShapeElem>(environmentCount),
            static_cast<mx::ShapeElem>(kABAStatusWords),
        },
        mx::uint32
    );
    mx::array intermediateQ =
        temporary(qShape, mx::float32);
    mx::array intermediateV =
        temporary(vShape, mx::float32);
    encoder.add_temporaries({
        candidateQ,
        candidateV,
        candidateAcceleration,
        abaStatuses,
        intermediateQ,
        intermediateV,
    });

    const mx::array* sourceQ = &inputs[0];
    const mx::array* sourceV = &inputs[1];
    for (std::uint32_t physicsSubstep = 0u;
         physicsSubstep < world_->physicsSubsteps();
         ++physicsSubstep) {
        MRABADispatchGPU abaDispatch{};
        abaDispatch.articulationIndex = articulationIndex;
        abaDispatch.environmentCount = environmentCount;
        abaDispatch.flags =
            world_->applyBodyDamping()
            ? MR_ABA_APPLY_BODY_DAMPING
            : 0u;
        abaDispatch.qStride = nq;
        abaDispatch.vStride = nv;
        abaDispatch.effortStride = nv;
        abaDispatch.accelerationStride = nv;
        abaDispatch.nextVStride = nv;
        abaDispatch.nextQStride = nq;

        encoder.set_compute_pipeline_state(
            resources.abaKernel
        );
        encoder.set_buffer(resources.buffer(0u), 0);
        encoder.set_buffer(resources.buffer(1u), 1);
        encoder.set_buffer(resources.buffer(2u), 2);
        encoder.set_buffer(resources.buffer(3u), 3);
        encoder.set_buffer(resources.buffer(4u), 4);
        encoder.set_bytes(abaDispatch, 5);
        encoder.set_input_array(*sourceQ, 6);
        encoder.set_input_array(*sourceV, 7);
        encoder.set_input_array(inputs[2], 8);
        encoder.set_buffer(resources.buffer(5u), 9);
        encoder.set_output_array(
            candidateAcceleration,
            10
        );
        encoder.set_output_array(candidateV, 11);
        encoder.set_output_array(candidateQ, 12);
        encoder.set_output_array(abaStatuses, 13);
        encoder.dispatch_threadgroups(
            MTL::Size(environmentCount, 1u, 1u),
            MTL::Size(kABAThreads, 1u, 1u)
        );

        const bool finalSubstep =
            physicsSubstep + 1u ==
            world_->physicsSubsteps();
        mx::array& destinationQ =
            finalSubstep ? outputs[0] : intermediateQ;
        mx::array& destinationV =
            finalSubstep ? outputs[1] : intermediateV;
        MRMLXWorldStepDispatchGPU stepDispatch{};
        stepDispatch.environmentCount = environmentCount;
        stepDispatch.nq = nq;
        stepDispatch.nv = nv;
        stepDispatch.physicsSubstep = physicsSubstep;
        stepDispatch.physicsSubsteps =
            world_->physicsSubsteps();
        stepDispatch.articulationIndex =
            articulationIndex;

        encoder.set_compute_pipeline_state(
            resources.commitKernel
        );
        encoder.set_bytes(stepDispatch, 0);
        encoder.set_input_array(inputs[0], 1);
        encoder.set_input_array(inputs[1], 2);
        encoder.set_input_array(candidateQ, 3);
        encoder.set_input_array(candidateV, 4);
        encoder.set_input_array(
            candidateAcceleration,
            5
        );
        encoder.set_input_array(abaStatuses, 6);
        encoder.set_output_array(destinationQ, 7);
        encoder.set_output_array(destinationV, 8);
        encoder.set_output_array(outputs[2], 9);
        encoder.set_output_array(outputs[3], 10);
        const auto threadgroupSize = std::min<
            std::uint32_t
        >(
            environmentCount,
            static_cast<std::uint32_t>(
                resources.commitKernel
                    ->maxTotalThreadsPerThreadgroup()
            )
        );
        encoder.dispatch_threads(
            MTL::Size(environmentCount, 1u, 1u),
            MTL::Size(threadgroupSize, 1u, 1u)
        );
        sourceQ = &destinationQ;
        sourceV = &destinationV;
    }
}

std::vector<mx::array> ABAWorldStepPrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo physics does not implement JVP"
    );
}

std::vector<mx::array> ABAWorldStepPrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo physics does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
ABAWorldStepPrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo environments use the native batch axis; "
        "vmap is not supported"
    );
}

const char* ABAWorldStepPrimitive::name() const {
    return "MetalRoboABAWorldStep";
}

bool ABAWorldStepPrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const ABAWorldStepPrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get();
}

GeneralizedConstraintStepPrimitive::
    GeneralizedConstraintStepPrimitive(
        mx::Stream stream,
        std::shared_ptr<
            MLXCompiledMultiArticulatedProgram
        > program
    )
    : mx::Primitive(stream),
      program_(std::move(program)) {}

void GeneralizedConstraintStepPrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo generalized constraints have no MLX CPU fallback"
    );
}

void GeneralizedConstraintStepPrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (inputs.size() != 2u || outputs.size() != 3u) {
        throw std::runtime_error(
            "generalized constraint primitive received an "
            "invalid graph"
        );
    }
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalGeneralizedResources& resources =
        program_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);
    for (mx::array& output : outputs) {
        output.set_data(mx::allocator::malloc(output.nbytes()));
    }

    const auto& compiled = program_->program();
    const EngineModel& model = compiled.model();
    const std::uint32_t environments =
        program_->environmentCapacity();
    const std::uint32_t rows = compiled.rowCount();
    const std::uint32_t nv = model.world.nv;
    const std::size_t inverseStatusCount =
        static_cast<std::size_t>(
            resources.inverseWorkCount
        ) * environments;
    mx::array response = temporary(
        {
            static_cast<mx::ShapeElem>(environments),
            static_cast<mx::ShapeElem>(rows),
            static_cast<mx::ShapeElem>(nv),
        },
        mx::float32
    );
    mx::array inverseStatuses = temporary(
        {
            static_cast<mx::ShapeElem>(
                inverseStatusCount
            ),
            static_cast<mx::ShapeElem>(
                kInverseMassStatusWords
            ),
        },
        mx::uint32
    );
    mx::array delassus = temporary(
        {
            static_cast<mx::ShapeElem>(environments),
            static_cast<mx::ShapeElem>(rows),
            static_cast<mx::ShapeElem>(rows),
        },
        mx::float32
    );
    mx::array candidateVelocity = temporary(
        outputs[0].shape(),
        mx::float32
    );
    mx::array candidateImpulses = temporary(
        outputs[1].shape(),
        mx::float32
    );
    mx::array candidateStatuses = temporary(
        outputs[2].shape(),
        mx::uint32
    );
    encoder.add_temporaries({
        response,
        inverseStatuses,
        delassus,
        candidateVelocity,
        candidateImpulses,
        candidateStatuses,
    });

    MRGeneralizedConstraintDispatchGPU dispatch{};
    dispatch.abiVersion =
        MR_GENERALIZED_CONSTRAINT_ABI_VERSION;
    dispatch.environmentCount = environments;
    dispatch.nv = nv;
    dispatch.rowCount = rows;
    dispatch.inverseWorkCount =
        resources.inverseWorkCount;
    dispatch.solverIterations =
        program_->config().solverIterations;
    if (program_->config().solverMode ==
        MetalGeneralizedConstraintSolverMode::
            qualitySemismoothNewton) {
        dispatch.reserved0 =
            program_->config().qualityCGIterations;
        dispatch.reserved1 =
            program_->config().qualityLineSearchIterations;
    }
    dispatch.evaluation0 = {
        static_cast<float>(
            program_->config().evaluation.timestep
        ),
        static_cast<float>(
            program_->config().evaluation.penetrationSlop
        ),
        static_cast<float>(
            program_->config().evaluation
                .maximumDepenetrationVelocity
        ),
        static_cast<float>(
            program_->config().evaluation
                .minimumTimeConstantRatio
        ),
    };
    dispatch.evaluation1 = {
        static_cast<float>(
            program_->config().evaluation
                .minimumRegularization
        ),
        program_->config().convergenceTolerance,
        program_->config().diagonalFloor,
        program_->config().
            qualityNormalEquationRegularization,
    };

    encoder.set_compute_pipeline_state(
        resources.inverseKernel
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedWorld),
        0
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedArticulations),
        1
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedJoints),
        2
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedDofs),
        3
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedBodies),
        4
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedInverseDispatches),
        5
    );
    encoder.set_input_array(inputs[0], 6);
    encoder.set_buffer(
        resources.buffer(kGeneralizedRhs),
        7
    );
    encoder.set_output_array(response, 8);
    encoder.set_output_array(inverseStatuses, 9);
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleArticulations),
        10
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleLevels),
        11
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleReductions),
        12
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleLevelBodies),
        13
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleParents),
        14
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleInboundJoints),
        15
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleChildOffsets),
        16
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedScheduleChildIndices),
        17
    );
    encoder.dispatch_threadgroups(
        MTL::Size(
            environments,
            resources.inverseWorkCount,
            1u
        ),
        MTL::Size(32u, 1u, 1u)
    );
    encoder.barrier();

    encoder.set_compute_pipeline_state(
        resources.delassusKernel
    );
    encoder.set_bytes(dispatch, 0);
    encoder.set_buffer(
        resources.buffer(kGeneralizedJacobian),
        1
    );
    encoder.set_input_array(response, 2);
    encoder.set_output_array(delassus, 3);
    encoder.dispatch_threads(
        MTL::Size(rows, rows, environments),
        MTL::Size(8u, 8u, 1u)
    );
    encoder.barrier();

    encoder.set_compute_pipeline_state(
        resources.solveKernel
    );
    encoder.set_bytes(dispatch, 0);
    encoder.set_buffer(
        resources.buffer(kGeneralizedRows),
        1
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedWarmImpulses),
        2
    );
    encoder.set_buffer(
        resources.buffer(kGeneralizedJacobian),
        3
    );
    encoder.set_input_array(inputs[1], 4);
    encoder.set_input_array(response, 5);
    encoder.set_input_array(inverseStatuses, 6);
    encoder.set_input_array(delassus, 7);
    encoder.set_output_array(candidateImpulses, 8);
    encoder.set_output_array(candidateVelocity, 9);
    encoder.set_output_array(candidateStatuses, 10);
    encoder.dispatch_threadgroups(
        MTL::Size(environments, 1u, 1u),
        MTL::Size(32u, 1u, 1u)
    );
    encoder.barrier();

    encoder.set_compute_pipeline_state(
        resources.commitKernel
    );
    encoder.set_bytes(dispatch, 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(candidateVelocity, 2);
    encoder.set_input_array(candidateImpulses, 3);
    encoder.set_input_array(candidateStatuses, 4);
    encoder.set_output_array(outputs[0], 5);
    encoder.set_output_array(outputs[1], 6);
    encoder.set_output_array(outputs[2], 7);
    const auto commitWidth = std::min<std::uint32_t>(
        environments,
        static_cast<std::uint32_t>(
            resources.commitKernel
                ->maxTotalThreadsPerThreadgroup()
        )
    );
    encoder.dispatch_threads(
        MTL::Size(environments, 1u, 1u),
        MTL::Size(commitWidth, 1u, 1u)
    );
}

std::vector<mx::array>
GeneralizedConstraintStepPrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo generalized constraints do not implement JVP"
    );
}

std::vector<mx::array>
GeneralizedConstraintStepPrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo generalized constraints do not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
GeneralizedConstraintStepPrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo generalized constraints use the native batch "
        "axis; vmap is unsupported"
    );
}

const char* GeneralizedConstraintStepPrimitive::name() const {
    return "MetalRoboGeneralizedConstraintStep";
}

bool GeneralizedConstraintStepPrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed = dynamic_cast<
        const GeneralizedConstraintStepPrimitive*
    >(&other);
    return typed != nullptr &&
        typed->program_.get() == program_.get();
}

WorldStepPrimitive::WorldStepPrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world
)
    : mx::Primitive(stream), world_(std::move(world)) {}

void WorldStepPrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo contact physics has no MLX CPU fallback"
    );
}

void WorldStepPrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (inputs.size() != 27u || outputs.size() != 42u) {
        throw std::runtime_error(
            "MetalRobo contact primitive received an invalid graph"
        );
    }
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalResources& resources = world_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);
    for (mx::array& output : outputs) {
        output.set_data(mx::allocator::malloc(output.nbytes()));
    }

    const CompiledWorld& compiled = world_->world();
    const EngineModel& model = compiled.model();
    const MRArticulationGPU articulation =
        model.articulations[compiled.articulationIndex()];
    const std::uint32_t articulationCount =
        compiled.articulationCount();
    const auto environments =
        static_cast<std::uint32_t>(inputs[0].shape(0));
    const std::uint32_t nq = compiled.nq();
    const std::uint32_t nv = compiled.nv();
    const std::uint32_t sceneCount = compiled.sceneBodyCount();
    // CompiledWorld::bodyCount() is the articulation-local count. Contact
    // dispatches address both articulation and free/kinematic scene bodies.
    const std::uint32_t bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    const std::uint32_t shapeCount = compiled.colliderCount();
    const std::uint32_t pairCount = compiled.eligiblePairCount();
    const MetalWorldCapacityProfile capacity =
        compiled.capacities();
    const std::uint32_t pairCapacity = capacity.candidatePairs;
    const std::uint32_t rawCapacity = capacity.rawContacts;
    const std::uint32_t manifoldCapacity = capacity.manifolds;
    const std::uint32_t constraintCapacity =
        capacity.constraintBlocks;
    const std::uint32_t rowCapacity = capacity.constraintRows;
    const std::uint32_t islandCapacity = capacity.islands;
    const std::uint32_t islandConstraintReferenceCapacity =
        std::min(
            capacity.islandConstraintReferences,
            constraintCapacity
        );
    const std::uint32_t tileCapacity = capacity.solverTiles;
    const std::uint32_t rodNodeCount =
        compiled.rodNodeCount();
    const std::uint32_t rodEdgeCount =
        compiled.rodEdgeCount();
    const std::uint32_t rodCount =
        compiled.rodCount();
    const bool multipleArticulations = articulationCount > 1u;
    const bool hybridCCD =
        world_->ccdMode() == MetalWorldCCDMode::hybrid;
    const bool futureKinematics =
        hybridCCD &&
        std::any_of(
            model.shapes.begin(),
            model.shapes.end(),
            [](const MRShapeGPU& shape) {
                return
                    (shape.flags & MR_SHAPE_FLAG_ENABLE_CCD) !=
                    0u;
            }
        );
    const bool qualityMode =
        world_->solverMode() ==
        MetalWorldSolverMode::qualityNewton;
    const bool throughputTGS =
        world_->solverMode() ==
        MetalWorldSolverMode::throughputTGS;
    const MetalWorldQualityConfig qualityConfig{};
    const std::uint32_t qualityNv =
        nv +
        6u * sceneCount +
        3u * rodNodeCount +
        rodEdgeCount;
    const std::uint32_t requestedQualityBlocks =
        capacity.qualityRows / 3u;
    const std::uint32_t qualityBlockCount =
        std::min(
            constraintCapacity,
            std::min(
                static_cast<std::uint32_t>(
                    MR_UNIFIED_QUALITY_MAX_BLOCKS
                ),
                requestedQualityBlocks == 0u
                ? static_cast<std::uint32_t>(
                      MR_UNIFIED_QUALITY_MAX_BLOCKS
                  )
                : requestedQualityBlocks
            )
        );
    const std::uint32_t qualityRows =
        3u * qualityBlockCount;
    const bool qualityDirectPath =
        qualityNv <=
            qualityConfig.directMaximumGeneralizedVelocities &&
        qualityRows <= qualityConfig.directMaximumRows;
    const std::uint32_t qualityHessianStride =
        qualityDirectPath ? qualityNv * qualityNv : 1u;
    const std::uint32_t qualityVectorStride =
        std::max(qualityNv, qualityRows);
    const std::uint32_t pointStride =
        2u * constraintCapacity;
    const std::uint32_t rodToolPairCount =
        static_cast<std::uint32_t>(
            compiled.rodToolPairs().size()
        );
    const std::size_t rodWitnessCount =
        static_cast<std::size_t>(environments) *
        rodToolPairCount *
        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    const std::uint32_t queueStride = std::max(
        pairCapacity,
        std::max(islandCapacity, tileCapacity)
    );
    const std::size_t pairFlagCount =
        static_cast<std::size_t>(environments) * pairCount;
    const std::size_t islandWorkCount =
        static_cast<std::size_t>(environments) *
        islandCapacity;
    const std::size_t tileWorkCount =
        static_cast<std::size_t>(environments) *
        tileCapacity;
    const std::size_t compactionCount = std::max(
        std::max(
            std::max<std::size_t>(
                pairFlagCount,
                islandWorkCount
            ),
            tileWorkCount
        ),
        std::size_t{1u}
    );

    std::vector<mx::array> retainedTemporaries;
    retainedTemporaries.reserve(96u);
    const auto makeTemporary = [&](
        const mx::Shape& shape,
        const mx::Dtype dtype
    ) {
        mx::array value = temporary(shape, dtype);
        retainedTemporaries.push_back(value);
        return value;
    };
    const auto rawTemporary = [&](
        const std::size_t bytes
    ) {
        const std::size_t words =
            std::max<std::size_t>(
                (bytes + sizeof(std::uint32_t) - 1u) /
                    sizeof(std::uint32_t),
                1u
            );
        return makeTemporary(
            {static_cast<mx::ShapeElem>(words)},
            mx::uint32
        );
    };
    const auto rawRecords = [&]<typename T>(
        const std::size_t count
    ) {
        return rawTemporary(
            std::max<std::size_t>(count, 1u) * sizeof(T)
        );
    };
    const mx::Shape qShape{
        static_cast<mx::ShapeElem>(environments),
        static_cast<mx::ShapeElem>(nq),
    };
    const mx::Shape vShape{
        static_cast<mx::ShapeElem>(environments),
        static_cast<mx::ShapeElem>(nv),
    };

    mx::array checkpointQ =
        makeTemporary(qShape, mx::float32);
    mx::array checkpointV =
        makeTemporary(vShape, mx::float32);
    mx::array workingEffort =
        makeTemporary(vShape, mx::float32);
    mx::array worldStatuses =
        rawRecords.template operator()<MRMetalWorldStatusGPU>(
            environments
        );
    mx::array resetMasks =
        rawRecords.template operator()<std::uint32_t>(
            environments
        );
    mx::array scenePackedA =
        rawRecords.template operator()<MRBodyStateGPU>(
            static_cast<std::size_t>(environments) *
            sceneCount
        );
    mx::array scenePackedB =
        rawRecords.template operator()<MRBodyStateGPU>(
            static_cast<std::size_t>(environments) *
            sceneCount
        );
    mx::array scenePackedC =
        rawRecords.template operator()<MRBodyStateGPU>(
            static_cast<std::size_t>(environments) *
            sceneCount
        );
    mx::array checkpointScene =
        rawRecords.template operator()<MRBodyStateGPU>(
            static_cast<std::size_t>(environments) *
            sceneCount
        );
    mx::array qPingA = makeTemporary(qShape, mx::float32);
    mx::array qPingB = makeTemporary(qShape, mx::float32);
    mx::array vPingA = makeTemporary(vShape, mx::float32);
    mx::array vPingB = makeTemporary(vShape, mx::float32);
    // CCD event replay is a separate topology cohort. Do not reserve its
    // complete state ping-pong arena for the overwhelmingly common rigid
    // speculative/discrete cohorts.
    mx::array eventQPingA = hybridCCD
        ? makeTemporary(qShape, mx::float32)
        : qPingA;
    mx::array eventQPingB = hybridCCD
        ? makeTemporary(qShape, mx::float32)
        : qPingB;
    mx::array eventVPingA = hybridCCD
        ? makeTemporary(vShape, mx::float32)
        : vPingA;
    mx::array eventVPingB = hybridCCD
        ? makeTemporary(vShape, mx::float32)
        : vPingB;
    mx::array eventScenePingA =
        hybridCCD
        ? rawRecords.template operator()<MRBodyStateGPU>(
              static_cast<std::size_t>(environments) *
              sceneCount
          )
        : scenePackedA;
    mx::array eventScenePingB =
        hybridCCD
        ? rawRecords.template operator()<MRBodyStateGPU>(
              static_cast<std::size_t>(environments) *
              sceneCount
          )
        : scenePackedB;
    mx::array eventManifoldHeadersA =
        hybridCCD
        ? rawRecords.template operator()<MRManifoldHeaderGPU>(
              static_cast<std::size_t>(environments) *
              manifoldCapacity
          )
        : inputs[7];
    mx::array eventManifoldHeadersB =
        hybridCCD
        ? rawRecords.template operator()<MRManifoldHeaderGPU>(
              static_cast<std::size_t>(environments) *
              manifoldCapacity
          )
        : inputs[7];
    mx::array eventManifoldPointsA =
        hybridCCD
        ? rawRecords.template operator()<MRManifoldPointGPU>(
              static_cast<std::size_t>(environments) *
              manifoldCapacity *
              MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
          )
        : inputs[8];
    mx::array eventManifoldPointsB =
        hybridCCD
        ? rawRecords.template operator()<MRManifoldPointGPU>(
              static_cast<std::size_t>(environments) *
              manifoldCapacity *
              MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
          )
        : inputs[8];
    mx::array eventManifoldCountsA =
        hybridCCD
        ? rawRecords.template operator()<std::uint32_t>(
              environments
          )
        : inputs[9];
    mx::array eventManifoldCountsB =
        hybridCCD
        ? rawRecords.template operator()<std::uint32_t>(
              environments
          )
        : inputs[9];
    mx::array manifoldHeadersA =
        rawRecords.template operator()<MRManifoldHeaderGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity
        );
    mx::array manifoldHeadersB =
        rawRecords.template operator()<MRManifoldHeaderGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity
        );
    mx::array manifoldPointsA =
        rawRecords.template operator()<MRManifoldPointGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        );
    mx::array manifoldPointsB =
        rawRecords.template operator()<MRManifoldPointGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        );
    mx::array manifoldCountsA =
        rawRecords.template operator()<std::uint32_t>(
            environments
        );
    mx::array manifoldCountsB =
        rawRecords.template operator()<std::uint32_t>(
            environments
        );
    mx::array checkpointManifoldHeaders =
        rawRecords.template operator()<MRManifoldHeaderGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity
        );
    mx::array checkpointManifoldPoints =
        rawRecords.template operator()<MRManifoldPointGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        );
    mx::array checkpointManifoldCounts =
        rawRecords.template operator()<std::uint32_t>(
            environments
        );
    mx::array candidateManifoldHeaders =
        rawRecords.template operator()<MRManifoldHeaderGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity
        );
    mx::array candidateManifoldPoints =
        rawRecords.template operator()<MRManifoldPointGPU>(
            static_cast<std::size_t>(environments) *
            manifoldCapacity *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        );
    mx::array candidateManifoldCounts =
        rawRecords.template operator()<std::uint32_t>(
            environments
        );
    mx::array candidateAcceleration =
        makeTemporary(vShape, mx::float32);
    mx::array candidateV =
        makeTemporary(vShape, mx::float32);
    mx::array candidateQ =
        makeTemporary(qShape, mx::float32);
    mx::array abaStatuses =
        rawRecords.template operator()<MRABAStatusGPU>(
            static_cast<std::size_t>(environments) *
            articulationCount
        );
    mx::array bodyPoses =
        rawRecords.template operator()<MRArticulatedBodyPoseGPU>(
            static_cast<std::size_t>(environments) *
            bodyCount
        );
    mx::array futureBodyPoses =
        futureKinematics
        ? rawRecords.template operator()<MRArticulatedBodyPoseGPU>(
              static_cast<std::size_t>(environments) *
              bodyCount
          )
        : bodyPoses;
    // Tactile sampling already requires the final global body records.
    // Publish that same arena as an MLX result so a visual policy stage can
    // consume it without another kinematics/materialization pass. Worlds
    // without tactile sensing retain the internal scratch path and expose an
    // empty tensor, so their step graph gains no observation work.
    mx::array currentBodies = world_->hasTactile()
        ? outputs[41]
        : rawRecords.template operator()<MRBodyStateGPU>(
              static_cast<std::size_t>(environments) * bodyCount
          );
    mx::array candidateBodies =
        rawRecords.template operator()<MRBodyStateGPU>(
            static_cast<std::size_t>(environments) * bodyCount
        );
    mx::array projected =
        rawRecords.template operator()<MRProjectedColliderGPU>(
            static_cast<std::size_t>(environments) * shapeCount
        );
    mx::array futureProjected =
        rawRecords.template operator()<MRProjectedColliderGPU>(
            static_cast<std::size_t>(environments) * shapeCount
        );
    mx::array pairFlags =
        rawRecords.template operator()<std::uint32_t>(
            std::max<std::size_t>(pairFlagCount, 1u)
        );
    mx::array scanOffsets =
        rawRecords.template operator()<std::uint32_t>(
            compactionCount
        );
    mx::array scanScratch =
        rawRecords.template operator()<std::uint32_t>(
            2u * compactionCount
        );
    mx::array compactionFlags =
        rawRecords.template operator()<std::uint32_t>(
            compactionCount
        );
    mx::array workHeaders =
        rawRecords.template operator()<MRWorkQueueHeaderGPU>(
            MR_WORLD_WORK_CLASS_COUNT
        );
    mx::array pairWork =
        rawRecords.template operator()<MRPairWorkGPU>(
            static_cast<std::size_t>(environments) *
            pairCapacity
        );
    mx::array pairRawCounts =
        rawRecords.template operator()<std::uint32_t>(
            std::max<std::size_t>(pairFlagCount, 1u)
        );
    mx::array pairRawStaging =
        rawRecords.template operator()<MRRawContactGPU>(
            std::max<std::size_t>(pairFlagCount, 1u) *
            MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR
        );
    mx::array pairManifoldHeaders =
        rawRecords.template operator()<MRManifoldHeaderGPU>(
            std::max<std::size_t>(pairFlagCount, 1u)
        );
    mx::array pairManifoldPoints =
        rawRecords.template operator()<MRManifoldPointGPU>(
            std::max<std::size_t>(pairFlagCount, 1u) *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY
        );
    mx::array manifoldScatter =
        rawRecords.template operator()<MRManifoldIRScatterGPU>(
            std::max<std::size_t>(pairFlagCount, 1u)
        );
    mx::array candidatePairs =
        rawRecords.template operator()<MRCandidatePairGPU>(
            static_cast<std::size_t>(environments) *
            pairCapacity
        );
    mx::array rawContacts =
        rawRecords.template operator()<MRRawContactGPU>(
            static_cast<std::size_t>(environments) *
            rawCapacity
        );
    mx::array rawPairIndices =
        rawRecords.template operator()<std::uint32_t>(
            static_cast<std::size_t>(environments) *
            rawCapacity
        );
    mx::array contacts =
        rawRecords.template operator()<MRContactConstraintGPU>(
            static_cast<std::size_t>(environments) *
            constraintCapacity
        );
    mx::array contactMetadata =
        rawRecords.template operator()<MRContactPointMetaGPU>(
            static_cast<std::size_t>(environments) *
            constraintCapacity
        );
    mx::array irBlocks =
        rawRecords.template operator()<MRConstraintIRBlockGPU>(
            static_cast<std::size_t>(environments) *
            constraintCapacity
        );
    mx::array irEndpoints =
        rawRecords.template operator()<MRConstraintIREndpointGPU>(
            static_cast<std::size_t>(environments) *
            2u * constraintCapacity
        );
    mx::array endpointRuntime =
        rawRecords.template operator()<
            MRConstraintEndpointRuntimeGPU>(
                static_cast<std::size_t>(environments) *
                2u * constraintCapacity
            );
    mx::array irRows =
        rawRecords.template operator()<MRConstraintIRRowGPU>(
            static_cast<std::size_t>(environments) *
            rowCapacity
        );
    mx::array irCones =
        rawRecords.template operator()<MRConstraintIRConeGPU>(
            static_cast<std::size_t>(environments) *
            constraintCapacity
        );
    mx::array pointQueries =
        rawRecords.template operator()<MRArticulatedPointImpulseGPU>(
            static_cast<std::size_t>(environments) *
            articulationCount * pointStride
        );
    mx::array multiABADispatches =
        rawRecords.template operator()<MRMultiABADispatchGPU>(
            articulationCount
        );
    mx::array kinematicsDispatches =
        rawRecords.template operator()<
            MRArticulatedOperatorDispatchGPU>(
                articulationCount
            );
    mx::array operatorDispatch =
        rawRecords.template operator()<
            MRArticulatedOperatorDispatchGPU>(
                articulationCount
            );
    mx::array activeIndirect =
        rawRecords.template operator()<
            MRIndirectDispatchArgumentsGPU>(2u);
    mx::array pointWorld =
        rawRecords.template operator()<MRArticulatedPointWorldGPU>(
            static_cast<std::size_t>(environments) *
            pointStride
        );
    mx::array factor =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) * nv * nv
        );
    mx::array factorStaging =
        multipleArticulations
        ? rawRecords.template operator()<float>(
              static_cast<std::size_t>(environments) * nv * nv
          )
        : factor;
    mx::array pointJacobians =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            pointStride * 3u * nv
        );
    mx::array pointJacobiansStaging =
        multipleArticulations
        ? rawRecords.template operator()<float>(
              static_cast<std::size_t>(environments) *
              pointStride * 3u * nv
          )
        : pointJacobians;
    mx::array generalizedImpulse =
        makeTemporary(vShape, mx::float32);
    mx::array deltaVelocity =
        makeTemporary(vShape, mx::float32);
    mx::array operatorStatuses =
        rawRecords.template operator()<
            MRArticulatedOperatorStatusGPU>(
                static_cast<std::size_t>(environments) *
                articulationCount
            );
    mx::array evaluatedRows =
        rawRecords.template operator()<
            MREvaluatedConstraintIRRowGPU>(
                static_cast<std::size_t>(environments) *
                rowCapacity
            );
    mx::array evaluatedCones =
        rawRecords.template operator()<
            MREvaluatedConstraintIRConeGPU>(
                static_cast<std::size_t>(environments) *
                constraintCapacity
            );
    mx::array factorCaches =
        rawRecords.template operator()<
            MRArticulationFactorCacheGPU>(
                static_cast<std::size_t>(environments) *
                articulationCount
            );
    mx::array islands =
        rawRecords.template operator()<MRContactIslandGPU>(
            islandWorkCount
        );
    mx::array islandNodeReferences =
        rawRecords.template operator()<MRIslandNodeRefGPU>(
            static_cast<std::size_t>(environments) *
            capacity.islandNodeReferences
        );
    mx::array islandConstraintReferences =
        rawRecords.template operator()<
            MRIslandConstraintRefGPU>(
                static_cast<std::size_t>(environments) *
                islandConstraintReferenceCapacity
            );
    mx::array denseIslandWork =
        rawRecords.template operator()<MRIslandWorkGPU>(
            throughputTGS ? islandWorkCount : 1u
        );
    mx::array compactIslandWork =
        rawRecords.template operator()<MRIslandWorkGPU>(
            throughputTGS ? 2u * islandWorkCount : 1u
        );
    mx::array waveWorkPackets =
        rawRecords.template operator()<MRWaveWorkPacketGPU>(
            throughputTGS ? islandWorkCount : 1u
        );
    mx::array contactTiles =
        rawRecords.template operator()<MRContactTileGPU>(
            throughputTGS
            ? 2u * static_cast<std::size_t>(environments) *
                  tileCapacity
            : 1u
        );
    mx::array tileConstraintIndices =
        rawRecords.template operator()<std::uint32_t>(
            throughputTGS
            ? static_cast<std::size_t>(environments) *
                  constraintCapacity
            : 1u
        );
    mx::array impulseDeltas =
        rawRecords.template operator()<mr_float4>(
            throughputTGS
            ? static_cast<std::size_t>(environments) *
                  constraintCapacity
            : 1u
        );
    mx::array preconditioners =
        rawRecords.template operator()<
            MRWave32PreconditionerGPU>(
                throughputTGS
                ? static_cast<std::size_t>(environments) *
                      constraintCapacity
                : 1u
            );
    mx::array waveStatuses =
        rawRecords.template operator()<MRWave32IslandStatusGPU>(
            throughputTGS ? islandWorkCount : 1u
        );
    mx::array responseColumns =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            constraintCapacity * 3u * nv
        );
    mx::array ccdPairs =
        rawRecords.template operator()<MRCCDPairGPU>(
            hybridCCD
            ? static_cast<std::size_t>(environments) *
                  capacity.ccdCandidates
            : 1u
        );
    mx::array ccdEventStatesA =
        rawRecords.template operator()<MRCCDEventStateGPU>(
            environments
        );
    mx::array ccdEventStatesB =
        rawRecords.template operator()<MRCCDEventStateGPU>(
            environments
        );
    mx::array ccdImpactClusters =
        rawRecords.template operator()<MRCCDImpactClusterGPU>(
            environments
        );
    mx::array eventRodNodesA =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array eventRodNodesB =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array eventRodEdgesA =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array eventRodEdgesB =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array eventRodWitnessesA =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array eventRodWitnessesB =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array projectedRodColliders =
        rawRecords.template operator()<MRProjectedColliderGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array futureProjectedRodColliders =
        rawRecords.template operator()<MRProjectedColliderGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodNodesA =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodNodesB =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array candidateRodNodes =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array checkpointRodNodes =
        rawRecords.template operator()<MRRodNodeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodEdgesA =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodEdgesB =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array candidateRodEdges =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array checkpointRodEdges =
        rawRecords.template operator()<MRRodEdgeStateGPU>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodInputPositions =
        rawRecords.template operator()<mr_float4>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodInputVelocities =
        rawRecords.template operator()<mr_float4>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodInputTwists =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodInputTwistRates =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodOutputPositions =
        rawRecords.template operator()<mr_float4>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodOutputVelocities =
        rawRecords.template operator()<mr_float4>(
            static_cast<std::size_t>(environments) *
            rodNodeCount
        );
    mx::array rodOutputTwists =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodOutputTwistRates =
        rawRecords.template operator()<float>(
            static_cast<std::size_t>(environments) *
            rodEdgeCount
        );
    mx::array rodStatuses =
        rawRecords.template operator()<MRRodGPUStatus>(
            static_cast<std::size_t>(environments) *
            rodCount
        );
    mx::array checkpointRodWitnesses =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array candidateRodWitnesses =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array rodWitnessesA =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array rodWitnessesB =
        rawRecords.template operator()<MRRodToolWitnessGPU>(
            rodWitnessCount
        );
    mx::array rodWitnessCounts =
        rawRecords.template operator()<std::uint32_t>(
            static_cast<std::size_t>(environments) *
            rodToolPairCount
        );
    mx::array rodContactScratch =
        rawRecords.template operator()<std::uint32_t>(
            rodWitnessCount
        );
    mx::array rodConstraintWitnessIndices =
        rawRecords.template operator()<std::uint32_t>(
            rodCount != 0u
            ? static_cast<std::size_t>(environments) *
                  constraintCapacity
            : 1u
        );
    mx::array rodFactorCaches =
        rawRecords.template operator()<MRRodFactorCacheGPU>(
            std::max<std::size_t>(
                static_cast<std::size_t>(environments) *
                    rodCount,
                1u
            )
        );
    mx::array rodOperatorArena =
        rawRecords.template operator()<float>(
            rodCount != 0u
            ? std::max<std::size_t>(
                  static_cast<std::size_t>(environments) *
                      capacity.operatorVelocityElements,
                  1u
              )
            : 1u
        );
    mx::array qualityBlocks =
        rawRecords.template operator()<MRUnifiedQualityBlockGPU>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityBlockCount
            : 1u
        );
    mx::array qualityDynamics =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityNv * qualityNv
            : 1u
        );
    mx::array qualityJacobian =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityRows * qualityNv
            : 1u
        );
    mx::array qualityBias =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityFreeVelocity =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityWarmVelocity =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityWarmImpulses =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityOutputVelocity =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityOutputImpulses =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityVectorStride
            : 1u
        );
    mx::array qualityDerivatives =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityBlockCount * 36u
            : 1u
        );
    mx::array qualityHessian =
        rawRecords.template operator()<float>(
            qualityMode
            ? static_cast<std::size_t>(environments) *
                  qualityHessianStride
            : 1u
        );
    mx::array qualityStatuses =
        rawRecords.template operator()<MRUnifiedQualityStatusGPU>(
            qualityMode ? environments : 1u
        );
    mx::array qualityWorkQueue =
        rawRecords.template operator()<
            MRUnifiedQualityWorkQueueGPU>(1u);
    mx::array qualityWorkPackets =
        rawRecords.template operator()<
            MRUnifiedQualityWorkPacketGPU>(
                qualityMode ? environments : 1u
            );

    // Tactile temporaries exist only for authored tactile worlds. The
    // tactile-free graph aliases already-retained arrays and never reaches
    // the corresponding dispatch block below.
    mx::array tactileContacts = contacts;
    mx::array tactileContactCounts = manifoldCountsA;
    mx::array tactileHits = contactMetadata;
    mx::array tactileSummaries = contactMetadata;
    if (world_->hasTactile()) {
        tactileContacts =
            rawRecords.template operator()<MRTactileContactGPU>(
                static_cast<std::size_t>(environments) *
                constraintCapacity
            );
        tactileContactCounts =
            rawRecords.template operator()<std::uint32_t>(
                environments
            );
        tactileHits =
            rawRecords.template operator()<MRTactileHitGPU>(
                static_cast<std::size_t>(environments) *
                world_->tactile().samples.size()
            );
        tactileSummaries =
            rawRecords.template operator()<MRTactileSummaryGPU>(
                static_cast<std::size_t>(environments) *
                world_->tactile().sensors.size()
            );
    }

    encoder.add_temporaries(std::move(retainedTemporaries));

    MRMetalWorldDispatchGPU worldDispatch{};
    worldDispatch.abiVersion = MR_METAL_WORLD_ABI_VERSION;
    worldDispatch.articulationIndex =
        compiled.articulationIndex();
    worldDispatch.environmentCount = environments;
    worldDispatch.controlStepCount = 1u;
    worldDispatch.physicsSubsteps = world_->physicsSubsteps();
    worldDispatch.flags =
        MR_METAL_WORLD_CONTACTS |
        MR_METAL_WORLD_DETERMINISTIC |
        (
            world_->applyBodyDamping()
            ? MR_METAL_WORLD_APPLY_BODY_DAMPING
            : 0u
        ) |
        (
            world_->actuationMode() ==
                MetalWorldActuationMode::
                    implicitPositionDrive
            ? MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES
            : 0u
        );
    worldDispatch.nq = nq;
    worldDispatch.nv = nv;
    worldDispatch.qStride = nq;
    worldDispatch.vStride = nv;
    worldDispatch.effortEnvironmentStride = nv;
    worldDispatch.observationEnvironmentStride = nq + nv;
    worldDispatch.effortStepStride = environments * nv;
    worldDispatch.resetMaskStepStride = environments;
    worldDispatch.observationStepStride =
        environments * (nq + nv);
    worldDispatch.accelerationStepStride = environments * nv;

    MRMetalWorldContactDispatchGPU contactDispatch{};
    contactDispatch.abiVersion =
        MR_METAL_WORLD_CONTACT_ABI_VERSION;
    contactDispatch.environmentCount = environments;
    contactDispatch.articulationIndex =
        compiled.articulationIndex();
    contactDispatch.solverType =
        world_->solverMode() ==
            MetalWorldSolverMode::throughputPGS
        ? MR_SOLVER_THROUGHPUT_PGS
        : qualityMode
        ? MR_SOLVER_QUALITY_NEWTON
        : MR_SOLVER_THROUGHPUT_TGS;
    contactDispatch.bodyCount = bodyCount;
    contactDispatch.sceneBodyCount = sceneCount;
    contactDispatch.shapeCount = shapeCount;
    contactDispatch.eligiblePairCount = pairCount;
    contactDispatch.pairCapacity = pairCapacity;
    contactDispatch.rawContactCapacity = rawCapacity;
    contactDispatch.manifoldCapacity = manifoldCapacity;
    contactDispatch.constraintCapacity = constraintCapacity;
    contactDispatch.rowCapacity = rowCapacity;
    contactDispatch.islandCapacity = islandCapacity;
    contactDispatch.sceneBodyStride = sceneCount;
    contactDispatch.bodyStateStride = bodyCount;
    contactDispatch.pairStride = pairCapacity;
    contactDispatch.rawContactStride = rawCapacity;
    contactDispatch.manifoldStride = manifoldCapacity;
    contactDispatch.constraintStride = constraintCapacity;
    contactDispatch.rowStride = rowCapacity;
    contactDispatch.islandStride = islandCapacity;
    contactDispatch.pointQueryStride = pointStride;
    contactDispatch.factorStride = nv * nv;
    contactDispatch.nv = nv;
    contactDispatch.flags =
        MR_METAL_WORLD_CONTACT_DETERMINISTIC |
        MR_METAL_WORLD_CONTACT_WARM_START |
        MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE;
    if (world_->solverMode() ==
        MetalWorldSolverMode::throughputTGS) {
        contactDispatch.flags |=
            MR_METAL_WORLD_CONTACT_WAVE32;
    } else if (qualityMode) {
        contactDispatch.flags |=
            MR_METAL_WORLD_CONTACT_QUALITY;
    }
    contactDispatch.velocityIterations =
        world_->velocityIterations();
    contactDispatch.finalVelocityIterations =
        world_->finalVelocityIterations();
    contactDispatch.hardConvexCapacity =
        capacity.hardConvexPairs;
    contactDispatch.meshCandidateCapacity =
        capacity.meshTriangleCandidates;
    contactDispatch.solverTileCapacity = tileCapacity;
    contactDispatch.spillRowCapacity = capacity.spillRows;
    contactDispatch.ccdCandidateCapacity =
        capacity.ccdCandidates;
    contactDispatch.ccdEventCapacity = capacity.ccdEvents;
    contactDispatch.ccdMode =
        static_cast<std::uint32_t>(world_->ccdMode());
    contactDispatch.maxCCDEvents = MR_CCD_DEFAULT_MAX_EVENTS;
    contactDispatch.maxConservativeAdvancementIterations = 16u;
    contactDispatch.workQueueClassCount =
        MR_WORLD_WORK_CLASS_COUNT;
    contactDispatch.queueStride = queueStride;
    contactDispatch.convexCacheStride = pairCount;
    contactDispatch.maxCCDAdvanceSolvePasses =
        world_->maxCCDAdvanceSolvePasses();
    contactDispatch.maxCCDZeroTimeReplays =
        world_->maxCCDZeroTimeReplays();
    contactDispatch.waveWorkerGroupCount =
        resources.tuning.waveWorkerGroupCount;
    contactDispatch.rodToolPairCount = rodToolPairCount;
    contactDispatch.articulationCount = articulationCount;
    contactDispatch.dynamicNodeCount =
        static_cast<std::uint32_t>(
            compiled.dynamicNodes().size()
        );
    contactDispatch.islandNodeReferenceCapacity =
        capacity.islandNodeReferences;
    contactDispatch.islandConstraintReferenceCapacity =
        islandConstraintReferenceCapacity;
    contactDispatch.rodCount = rodCount;
    contactDispatch.rodNodeCount = rodNodeCount;
    contactDispatch.rodEdgeCount = rodEdgeCount;
    contactDispatch.operatorVelocityCapacity =
        capacity.operatorVelocityElements;
    contactDispatch.nq = nq;
    contactDispatch.qStride = nq;
    contactDispatch.vStride = nv;
    contactDispatch.rodContactOuterIterations = 2u;
    contactDispatch.authoredConstraintCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.blocks.size()
        );
    contactDispatch.authoredEndpointCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.endpoints.size()
        );
    contactDispatch.authoredRowCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.rows.size()
        );
    contactDispatch.authoredConeCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.cones.size()
        );
    if (world_->ccdMode() != MetalWorldCCDMode::disabled) {
        contactDispatch.flags |= MR_METAL_WORLD_CONTACT_CCD;
    }
    if (futureKinematics) {
        contactDispatch.flags |=
            MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS;
    }
    const float substepTimestep =
        world_->controlTimestep() /
        static_cast<float>(world_->physicsSubsteps());
    contactDispatch.timestepAndBias = {
        substepTimestep,
        model.world.solverScales.w,
        model.world.solverScales.z,
        1.0f,
    };
    contactDispatch.manifoldThresholds =
        {0.02f, 0.02f, 0.002f, 0.95f};
    contactDispatch.ccdParameters =
        {1.0e-5f, 1.0e-5f, 1.0f, 1.0e4f};
    contactDispatch.ccdEventParameters = {
        world_->ccdSimultaneousTolerance(),
        1.0e-5f,
        0.0f,
        0.0f,
    };

    std::vector<MRRodGPUDispatch> rodDispatches;
    std::vector<MRRodGPUDispatch> rodCollisionDispatches;
    rodDispatches.reserve(rodCount);
    rodCollisionDispatches.reserve(rodCount);
    std::size_t rodToolPairCursor = 0u;
    for (std::size_t rod = 0u;
         rod < compiled.rodPrograms().size();
         ++rod) {
        const HeterogeneousRodProgram& program =
            compiled.rodPrograms()[rod];
        const std::uint32_t nodeBase =
            compiled.rodNodeOffsets()[rod];
        const std::uint32_t edgeBase =
            compiled.rodEdgeOffsets()[rod];
        const std::uint32_t localNodes =
            compiled.rodNodeOffsets()[rod + 1u] - nodeBase;
        const std::uint32_t localEdges =
            compiled.rodEdgeOffsets()[rod + 1u] - edgeBase;
        while (rodToolPairCursor <
                   compiled.rodToolPairs().size() &&
               compiled.rodToolPairs()[rodToolPairCursor]
                       .rodCollider < edgeBase) {
            ++rodToolPairCursor;
        }
        const std::size_t pairBase = rodToolPairCursor;
        while (rodToolPairCursor <
                   compiled.rodToolPairs().size() &&
               compiled.rodToolPairs()[rodToolPairCursor]
                       .rodCollider <
                   edgeBase + localEdges) {
            ++rodToolPairCursor;
        }
        const std::uint32_t localPairs =
            static_cast<std::uint32_t>(
                rodToolPairCursor - pairBase
            );

        MRRodGPUDispatch dispatch{};
        dispatch.abiVersion = MR_ROD_GPU_ABI_VERSION;
        dispatch.environmentCount = environments;
        dispatch.nodeCount = localNodes;
        dispatch.edgeCount = localEdges;
        dispatch.attachmentCount = 0u;
        dispatch.solverIterations =
            program.stepConfig.solverIterations;
        dispatch.stateNodeStride = rodNodeCount;
        dispatch.stateEdgeStride = rodEdgeCount;
        dispatch.rigidBodyCount = bodyCount;
        dispatch.stateBodyStride = bodyCount;
        dispatch.flags =
            program.stepConfig.enableSelfCollision
            ? MR_ROD_GPU_FLAG_SELF_COLLISION
            : 0u;
        dispatch.toolShapeCount = shapeCount;
        dispatch.toolPairCount = localPairs;
        dispatch.toolContactStride =
            rodToolPairCount *
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
        dispatch.toolContactIterations =
            localPairs == 0u ? 0u : 2u;
        dispatch.rodMaterialIndex =
            program.collision.materialIndex;
        dispatch.rodNodeBase = nodeBase;
        dispatch.rodEdgeBase = edgeBase;
        dispatch.toolPairBase =
            static_cast<std::uint32_t>(pairBase);
        dispatch.toolPairWorldStride = rodToolPairCount;
        dispatch.gravityAndTimestep = {
            static_cast<float>(program.stepConfig.gravity[0]),
            static_cast<float>(program.stepConfig.gravity[1]),
            static_cast<float>(program.stepConfig.gravity[2]),
            substepTimestep,
        };
        dispatch.dampingDerivativeTolerance = {
            static_cast<float>(
                program.stepConfig.linearDamping
            ),
            static_cast<float>(
                program.stepConfig.twistDamping
            ),
            static_cast<float>(std::max(
                program.stepConfig.derivativeStep,
                3.5e-4
            )),
            static_cast<float>(
                program.stepConfig.constraintTolerance
            ),
        };
        dispatch.selfCollision = {
            static_cast<float>(program.model.radius),
            static_cast<float>(
                program.stepConfig.selfCollisionMargin
            ),
            static_cast<float>(
                program.stepConfig.selfCollisionCompliance
            ),
            0.0f,
        };
        dispatch.toolContact = {
            static_cast<float>(
                program.collision.contactOffset
            ),
            static_cast<float>(
                program.collision.restOffset
            ),
            0.0f,
            0.0f,
        };
        dispatch.toolResponse =
            {0.0f, 1.0f, 1.0f, 10.0f};
        rodDispatches.push_back(dispatch);

        MRRodGPUDispatch collision = dispatch;
        if (localPairs != 0u) {
            collision.flags |=
                MR_ROD_GPU_FLAG_TOOL_COLLISION |
                MR_ROD_GPU_FLAG_TOOL_WARM_START;
            if (std::any_of(
                    compiled.rodToolPairs().begin() +
                        static_cast<std::ptrdiff_t>(pairBase),
                    compiled.rodToolPairs().begin() +
                        static_cast<std::ptrdiff_t>(
                            rodToolPairCursor
                        ),
                    [](const MRRodToolPairGPU& pair) {
                        return
                            (pair.flags &
                             MR_ROD_TOOL_PAIR_ENABLE_CCD) !=
                            0u;
                    }
                )) {
                collision.flags |=
                    MR_ROD_GPU_FLAG_ENABLE_CCD;
            }
        }
        rodCollisionDispatches.push_back(collision);
    }

    MRUnifiedQualityDispatchGPU qualityDispatch{};
    if (qualityMode) {
        qualityDispatch.abiVersion =
            MR_UNIFIED_QUALITY_ABI_VERSION;
        qualityDispatch.problemCount = environments;
        qualityDispatch.generalizedVelocityCount = qualityNv;
        qualityDispatch.rowCount = qualityRows;
        qualityDispatch.blockCount = qualityBlockCount;
        qualityDispatch.dynamicsStride = qualityNv * qualityNv;
        qualityDispatch.jacobianStride = qualityRows * qualityNv;
        qualityDispatch.vectorStride = qualityVectorStride;
        qualityDispatch.maximumNewtonIterations =
            qualityConfig.maximumNewtonIterations;
        qualityDispatch.maximumPCGIterations =
            qualityConfig.maximumPCGIterations;
        qualityDispatch.maximumLineSearchIterations =
            qualityConfig.maximumLineSearchIterations;
        qualityDispatch.directMaximumGeneralizedVelocities =
            qualityConfig.directMaximumGeneralizedVelocities;
        qualityDispatch.directMaximumRows =
            qualityConfig.directMaximumRows;
        qualityDispatch.derivativeStride =
            qualityBlockCount * 36u;
        qualityDispatch.hessianStride = qualityHessianStride;
        qualityDispatch.blockStride = qualityBlockCount;
        qualityDispatch.tolerances = {
            qualityConfig.optimalityTolerance,
            qualityConfig.feasibilityTolerance,
            qualityConfig.armijoConstant,
            qualityConfig.lineSearchContraction,
        };
        qualityDispatch.numerics = {
            qualityConfig.complianceFloorMultiplier,
            1.0e-10f,
            1.0e-20f,
            64.0f,
        };
    }

    MRMLXContactAdapterDispatchGPU adapterDispatch{};
    adapterDispatch.environmentCount = environments;
    adapterDispatch.sceneBodyCount = sceneCount;
    adapterDispatch.bodyStateStride = bodyCount;
    adapterDispatch.contactCapacity = constraintCapacity;
    adapterDispatch.manifoldCapacity = manifoldCapacity;
    adapterDispatch.eligiblePairCount = pairCount;
    adapterDispatch.nq = nq;
    adapterDispatch.nv = nv;

    MRABADispatchGPU abaDispatch{};
    abaDispatch.articulationIndex = compiled.articulationIndex();
    abaDispatch.environmentCount = environments;
    abaDispatch.flags =
        (
            world_->applyBodyDamping()
            ? MR_ABA_APPLY_BODY_DAMPING
            : 0u
        ) |
        (
            world_->actuationMode() ==
                MetalWorldActuationMode::
                    implicitPositionDrive
            ? MR_ABA_IMPLICIT_DRIVES
            : 0u
        );
    abaDispatch.qStride = nq;
    abaDispatch.vStride = nv;
    abaDispatch.effortStride = nv;
    abaDispatch.accelerationStride = nv;
    abaDispatch.nextVStride = nv;
    abaDispatch.nextQStride = nq;

    const auto setPhysicsKernel = [&](
        const char* name
    ) {
        encoder.set_compute_pipeline_state(
            resources.kernel(name)
        );
    };
    const auto inputArray = [&](
        const mx::array& array,
        const int index,
        const std::int64_t offset = 0
    ) {
        encoder.set_input_array(array, index, offset);
    };
    const auto outputArray = [&](
        mx::array& array,
        const int index,
        const std::int64_t offset = 0
    ) {
        encoder.set_output_array(array, index, offset);
    };
    const auto immutable = [&](
        const std::size_t buffer,
        const int index
    ) {
        encoder.set_buffer(resources.buffer(buffer), index);
    };
    const auto dispatchThreads = [&](
        const std::size_t count,
        MTL::ComputePipelineState* pipeline
    ) {
        const std::size_t width = std::max<std::size_t>(
            1u,
            std::min<std::size_t>(
                kWorldThreads,
                pipeline->maxTotalThreadsPerThreadgroup()
            )
        );
        encoder.dispatch_threads(
            MTL::Size(std::max<std::size_t>(count, 1u), 1u, 1u),
            MTL::Size(width, 1u, 1u)
        );
        encoder.barrier();
    };
    const auto encodeAllArticulationKinematics = [&](
        const mx::array& qState,
        mx::array& poses
    ) {
        setPhysicsKernel("mr_articulated_operator");
        for (std::uint32_t owner = 0u;
             owner < articulationCount;
             ++owner) {
            const MRArticulationGPU& owned =
                model.articulations[owner];
            immutable(kImmutableWorld, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            immutable(kImmutableDofs, 3);
            immutable(kImmutableBodies, 4);
            inputArray(
                kinematicsDispatches,
                5,
                owner *
                    sizeof(MRArticulatedOperatorDispatchGPU)
            );
            inputArray(
                qState,
                6,
                owned.qOffset * sizeof(float)
            );
            inputArray(pointQueries, 7);
            outputArray(
                poses,
                8,
                owned.firstBody *
                    sizeof(MRArticulatedBodyPoseGPU)
            );
            outputArray(pointWorld, 9);
            outputArray(factor, 10);
            outputArray(pointJacobians, 11);
            outputArray(
                generalizedImpulse,
                12,
                owned.vOffset * sizeof(float)
            );
            outputArray(
                deltaVelocity,
                13,
                owned.vOffset * sizeof(float)
            );
            outputArray(
                operatorStatuses,
                14,
                owner * environments *
                    sizeof(MRArticulatedOperatorStatusGPU)
            );
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(kOperatorThreads, 1u, 1u)
            );
        }
        encoder.barrier();
    };
    const auto encodeAllArticulationFactors = [&](
        const mx::array& qState,
        mx::array& poses
    ) {
        setPhysicsKernel("mr_articulated_operator");
        std::size_t factorPrefix = 0u;
        std::size_t jacobianPrefix = 0u;
        for (std::uint32_t owner = 0u;
             owner < articulationCount;
             ++owner) {
            const MRArticulationGPU& owned =
                model.articulations[owner];
            immutable(kImmutableWorld, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            immutable(kImmutableDofs, 3);
            immutable(kImmutableBodies, 4);
            inputArray(
                operatorDispatch,
                5,
                owner *
                    sizeof(MRArticulatedOperatorDispatchGPU)
            );
            inputArray(
                qState,
                6,
                owned.qOffset * sizeof(float)
            );
            inputArray(
                pointQueries,
                7,
                owner * environments *
                    pointStride *
                    sizeof(MRArticulatedPointImpulseGPU)
            );
            outputArray(
                poses,
                8,
                owned.firstBody *
                    sizeof(MRArticulatedBodyPoseGPU)
            );
            outputArray(pointWorld, 9);
            outputArray(
                articulationCount > 1u
                    ? factorStaging
                    : factor,
                10,
                articulationCount > 1u
                    ? factorPrefix * sizeof(float)
                    : 0u
            );
            outputArray(
                articulationCount > 1u
                    ? pointJacobiansStaging
                    : pointJacobians,
                11,
                articulationCount > 1u
                    ? jacobianPrefix * sizeof(float)
                    : 0u
            );
            outputArray(
                generalizedImpulse,
                12,
                owned.vOffset * sizeof(float)
            );
            outputArray(
                deltaVelocity,
                13,
                owned.vOffset * sizeof(float)
            );
            outputArray(
                operatorStatuses,
                14,
                owner * environments *
                    sizeof(MRArticulatedOperatorStatusGPU)
            );
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(kOperatorThreads, 1u, 1u)
            );
            factorPrefix +=
                static_cast<std::size_t>(environments) *
                owned.nv * owned.nv;
            jacobianPrefix +=
                static_cast<std::size_t>(environments) *
                pointStride * 3u * owned.nv;
        }
        encoder.barrier();
        if (articulationCount > 1u) {
            setPhysicsKernel(
                "mr_world_compose_multi_articulation_operator"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            inputArray(operatorDispatch, 2);
            inputArray(operatorStatuses, 3);
            inputArray(factorStaging, 4);
            inputArray(pointJacobiansStaging, 5);
            outputArray(factor, 6);
            outputArray(pointJacobians, 7);
            outputArray(outputs[16], 8);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_compose_multi_articulation_operator"
                )
            );
        }
    };

    setPhysicsKernel("mr_mlx_prepare_contact_world");
    encoder.set_bytes(worldDispatch, 0);
    encoder.set_bytes(contactDispatch, 1);
    inputArray(inputs[0], 2);
    inputArray(inputs[1], 3);
    inputArray(inputs[2], 4);
    inputArray(inputs[10], 5);
    outputArray(checkpointQ, 6);
    outputArray(checkpointV, 7);
    outputArray(workingEffort, 8);
    outputArray(worldStatuses, 9);
    outputArray(outputs[9], 10);
    immutable(kImmutableWorld, 11);
    immutable(kImmutableArticulations, 12);
    immutable(kImmutableDofs, 13);
    inputArray(inputs[17], 14);
    outputArray(resetMasks, 15);
    inputArray(inputs[26], 16);
    dispatchThreads(
        environments,
        resources.kernel("mr_mlx_prepare_contact_world")
    );

    setPhysicsKernel("mr_mlx_pack_scene_state");
    encoder.set_bytes(adapterDispatch, 0);
    immutable(kImmutableBodies, 1);
    immutable(kImmutableSceneBodyIndices, 2);
    inputArray(inputs[3], 3);
    inputArray(inputs[4], 4);
    inputArray(inputs[5], 5);
    inputArray(inputs[6], 6);
    outputArray(scenePackedA, 7);
    dispatchThreads(
        static_cast<std::size_t>(environments) * sceneCount,
        resources.kernel("mr_mlx_pack_scene_state")
    );

    setPhysicsKernel(
        "mr_mlx_initialize_world_articulation_dispatches"
    );
    encoder.set_bytes(worldDispatch, 0);
    encoder.set_bytes(contactDispatch, 1);
    immutable(kImmutableArticulations, 2);
    encoder.set_bytes(abaDispatch.flags, 3);
    outputArray(multiABADispatches, 4);
    outputArray(kinematicsDispatches, 5);
    outputArray(operatorDispatch, 6);
    dispatchThreads(
        articulationCount,
        resources.kernel(
            "mr_mlx_initialize_world_articulation_dispatches"
        )
    );

    if (rodCount != 0u) {
        setPhysicsKernel("mr_world_unpack_rod_state");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(inputs[11], 1);
        inputArray(inputs[12], 2);
        inputArray(inputs[13], 3);
        inputArray(inputs[14], 4);
        outputArray(rodNodesA, 5);
        outputArray(rodEdgesA, 6);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_unpack_rod_state")
        );
    }

    MRMetalWorldPassGPU controlPass{};
    controlPass.controlStep = 0u;
    controlPass.physicsSubstep = MR_INVALID_INDEX;
    setPhysicsKernel("mr_world_prepare_contact_step");
    encoder.set_bytes(worldDispatch, 0);
    encoder.set_bytes(contactDispatch, 1);
    encoder.set_bytes(controlPass, 2);
    inputArray(resetMasks, 3);
    inputArray(scenePackedA, 4);
    inputArray(scenePackedA, 5);
    immutable(kImmutableBodies, 6);
    immutable(kImmutableSceneBodyIndices, 7);
    outputArray(scenePackedA, 8);
    outputArray(checkpointScene, 9);
    inputArray(inputs[7], 10);
    inputArray(inputs[8], 11);
    inputArray(inputs[9], 12);
    outputArray(checkpointManifoldHeaders, 13);
    outputArray(checkpointManifoldPoints, 14);
    outputArray(checkpointManifoldCounts, 15);
    outputArray(outputs[16], 16);
    outputArray(outputs[9], 17);
    dispatchThreads(
        environments,
        resources.kernel("mr_world_prepare_contact_step")
    );

    if (rodCount != 0u) {
        setPhysicsKernel("mr_world_prepare_rod_state");
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(contactDispatch, 1);
        encoder.set_bytes(controlPass, 2);
        inputArray(resetMasks, 3);
        inputArray(rodNodesA, 4);
        inputArray(rodEdgesA, 5);
        outputArray(rodNodesA, 6);
        outputArray(rodEdgesA, 7);
        outputArray(checkpointRodNodes, 8);
        outputArray(checkpointRodEdges, 9);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_prepare_rod_state")
        );
    }
    if (rodWitnessCount != 0u) {
        setPhysicsKernel(
            "mr_world_prepare_rod_contact_cache"
        );
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(contactDispatch, 1);
        encoder.set_bytes(controlPass, 2);
        inputArray(resetMasks, 3);
        inputArray(inputs[15], 4);
        outputArray(checkpointRodWitnesses, 5);
        outputArray(candidateRodWitnesses, 6);
        dispatchThreads(
            rodWitnessCount,
            resources.kernel(
                "mr_world_prepare_rod_contact_cache"
            )
        );
    }

    std::uint32_t activePairClassMask = 0u;
    for (const MRCompiledCollisionPairGPU& pair :
         compiled.eligiblePairs()) {
        std::uint32_t workClass = MR_WORLD_WORK_ANALYTIC;
        if (pair.pairClass == MR_COLLISION_PAIR_BOX_BOX) {
            workClass = MR_WORLD_WORK_SAT_CLIP;
        } else if (pair.pairClass ==
                   MR_COLLISION_PAIR_MESH) {
            workClass = MR_WORLD_WORK_MESH;
        } else if (pair.pairClass ==
                   MR_COLLISION_PAIR_CONVEX) {
            workClass =
                model.shapes[pair.colliderA].shapeType ==
                        MR_SHAPE_CONVEX ||
                    model.shapes[pair.colliderB].shapeType ==
                        MR_SHAPE_CONVEX
                ? MR_WORLD_WORK_HULL_GJK
                : MR_WORLD_WORK_PRIMITIVE_GJK;
        }
        activePairClassMask |= 1u << workClass;
    }

    const auto encodeScan = [&](
        mx::array& flags,
        const std::size_t elementCount,
        const std::uint32_t workClass
    ) {
        std::vector<MRScanLevelGPU> levels;
        std::size_t count = elementCount;
        std::size_t scratchCursor = 0u;
        std::size_t inputOffset = 0u;
        while (count != 0u) {
            const std::size_t blockCount =
                (count + MR_BROADPHASE_SCAN_BLOCK_SIZE - 1u) /
                MR_BROADPHASE_SCAN_BLOCK_SIZE;
            MRScanLevelGPU level{};
            level.elementCount =
                static_cast<std::uint32_t>(count);
            level.blockCount =
                static_cast<std::uint32_t>(blockCount);
            level.inputOffset =
                static_cast<std::uint32_t>(inputOffset);
            level.outputOffset =
                levels.empty()
                ? 0u
                : static_cast<std::uint32_t>(scratchCursor);
            if (!levels.empty()) {
                scratchCursor += count;
            }
            level.blockSumOffset =
                static_cast<std::uint32_t>(scratchCursor);
            level.workClass = workClass;
            level.flags = levels.empty()
                ? MR_SCAN_BOOLEAN_INPUT
                : 0u;
            scratchCursor += blockCount;
            levels.push_back(level);

            setPhysicsKernel("mr_world_scan_blocks");
            if (levels.size() == 1u) {
                inputArray(flags, 0);
                outputArray(scanOffsets, 1);
            } else {
                inputArray(scanScratch, 0);
                outputArray(scanScratch, 1);
            }
            outputArray(scanScratch, 2);
            encoder.set_bytes(level, 3);
            encoder.dispatch_threadgroups(
                MTL::Size(blockCount, 1u, 1u),
                MTL::Size(
                    MR_BROADPHASE_SCAN_BLOCK_SIZE,
                    1u,
                    1u
                )
            );
            encoder.barrier();
            if (blockCount <= 1u) {
                break;
            }
            inputOffset = level.blockSumOffset;
            count = blockCount;
        }
        for (std::size_t levelIndex = levels.size();
             levelIndex-- > 1u;) {
            MRScanLevelGPU child = levels[levelIndex - 1u];
            child.parentOffset =
                levels[levelIndex].outputOffset;
            setPhysicsKernel(
                "mr_world_scan_add_block_offsets"
            );
            if (levelIndex == 1u) {
                outputArray(scanOffsets, 0);
            } else {
                outputArray(scanScratch, 0);
            }
            inputArray(scanScratch, 1);
            encoder.set_bytes(child, 2);
            dispatchThreads(
                child.elementCount,
                resources.kernel(
                    "mr_world_scan_add_block_offsets"
                )
            );
        }
    };

    const mx::array* sourceQ = &inputs[0];
    const mx::array* sourceV = &inputs[1];
    mx::array* sourceScene = &scenePackedA;
    const mx::array* sourceHeaders = &inputs[7];
    const mx::array* sourcePoints = &inputs[8];
    const mx::array* sourceCounts = &inputs[9];
    mx::array* sourceRodNodes = &rodNodesA;
    mx::array* sourceRodEdges = &rodEdgesA;
    mx::array* destinationRodNodes = &rodNodesB;
    mx::array* destinationRodEdges = &rodEdgesB;
    const mx::array* sourceRodWitnesses = &inputs[15];
    mx::array* destinationRodWitnesses = &rodWitnessesA;

    const auto encodeRodSubstep = [&](
        const MRMetalWorldPassGPU& rodPass,
        const mx::array& sourceNodes,
        const mx::array& sourceEdges,
        mx::array& destinationNodes,
        mx::array& destinationEdges,
        const mx::array& eventStates,
        const std::uint32_t eventSegmentMode
    ) {
        if (rodCount == 0u) {
            return;
        }
        setPhysicsKernel("mr_world_pack_rod_state");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(sourceNodes, 1);
        inputArray(sourceEdges, 2);
        outputArray(rodInputPositions, 3);
        outputArray(rodInputVelocities, 4);
        outputArray(rodInputTwists, 5);
        outputArray(rodInputTwistRates, 6);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_pack_rod_state")
        );

        setPhysicsKernel("mr_discrete_elastic_rod_step");
        for (std::size_t rod = 0u;
             rod < rodDispatches.size();
             ++rod) {
            const std::size_t nodeOffset =
                compiled.rodNodeOffsets()[rod];
            const std::size_t edgeOffset =
                compiled.rodEdgeOffsets()[rod];
            const std::size_t bendOffset = edgeOffset - rod;
            encoder.set_bytes(rodDispatches[rod], 0);
            encoder.set_buffer(
                resources.buffer(kImmutableRodRestLengths),
                1,
                edgeOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodRestTwists),
                2,
                edgeOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodRestCurvatures),
                3,
                bendOffset * sizeof(mr_float4)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodInverseMasses),
                4,
                nodeOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(
                    kImmutableRodInverseTwistInertias
                ),
                5,
                edgeOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodStretchStiffness),
                6,
                edgeOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodBendStiffness),
                7,
                bendOffset * sizeof(float)
            );
            encoder.set_buffer(
                resources.buffer(kImmutableRodTwistStiffness),
                8,
                bendOffset * sizeof(float)
            );
            inputArray(
                rodInputPositions,
                9,
                nodeOffset * sizeof(mr_float4)
            );
            inputArray(
                rodInputVelocities,
                10,
                nodeOffset * sizeof(mr_float4)
            );
            inputArray(
                rodInputTwists,
                11,
                edgeOffset * sizeof(float)
            );
            inputArray(
                rodInputTwistRates,
                12,
                edgeOffset * sizeof(float)
            );
            // Attachments are represented through the typed world graph.
            immutable(kImmutableRodInverseMasses, 13);
            outputArray(
                rodOutputPositions,
                14,
                nodeOffset * sizeof(mr_float4)
            );
            outputArray(
                rodOutputVelocities,
                15,
                nodeOffset * sizeof(mr_float4)
            );
            outputArray(
                rodOutputTwists,
                16,
                edgeOffset * sizeof(float)
            );
            outputArray(
                rodOutputTwistRates,
                17,
                edgeOffset * sizeof(float)
            );
            outputArray(
                rodStatuses,
                18,
                rod * environments * sizeof(MRRodGPUStatus)
            );
            immutable(kImmutableRodInverseMasses, 19);
            inputArray(eventStates, 20);
            encoder.set_bytes(eventSegmentMode, 21);
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(MR_ROD_GPU_MAX_NODES, 1u, 1u)
            );
        }
        encoder.barrier();

        setPhysicsKernel("mr_world_unpack_rod_state");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(rodOutputPositions, 1);
        inputArray(rodOutputVelocities, 2);
        inputArray(rodOutputTwists, 3);
        inputArray(rodOutputTwistRates, 4);
        outputArray(destinationNodes, 5);
        outputArray(destinationEdges, 6);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_unpack_rod_state")
        );

        setPhysicsKernel("mr_world_latch_rod_status");
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(contactDispatch, 1);
        encoder.set_bytes(rodPass, 2);
        inputArray(rodStatuses, 3);
        outputArray(worldStatuses, 4);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_latch_rod_status")
        );
        setPhysicsKernel(
            "mr_world_latch_rod_contact_status"
        );
        encoder.set_bytes(contactDispatch, 0);
        encoder.set_bytes(rodPass, 1);
        inputArray(rodStatuses, 2);
        outputArray(outputs[16], 3);
        dispatchThreads(
            environments,
            resources.kernel(
                "mr_world_latch_rod_contact_status"
            )
        );

        setPhysicsKernel("mr_world_factor_rod_operator");
        for (std::size_t rod = 0u;
             rod < rodDispatches.size();
             ++rod) {
            const std::uint32_t rodIndex =
                static_cast<std::uint32_t>(rod);
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(rodDispatches[rod], 1);
            inputArray(destinationNodes, 2);
            immutable(kImmutableRodInverseMasses, 3);
            immutable(kImmutableRodInverseTwistInertias, 4);
            immutable(kImmutableRodColliders, 5);
            immutable(kImmutableRodRestLengths, 6);
            immutable(kImmutableRodStretchStiffness, 7);
            immutable(kImmutableRodBendStiffness, 8);
            immutable(kImmutableRodTwistStiffness, 9);
            outputArray(rodFactorCaches, 10);
            outputArray(rodOperatorArena, 11);
            encoder.set_bytes(rodIndex, 12);
            encoder.set_bytes(rodPass, 13);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_factor_rod_operator"
                )
            );
        }
        encoder.barrier();
    };

    for (std::uint32_t substep = 0u;
         substep < world_->physicsSubsteps();
         ++substep) {
        const bool finalSubstep =
            substep + 1u == world_->physicsSubsteps();
        mx::array* destinationQ = finalSubstep
            ? &outputs[0]
            : (substep % 2u == 0u ? &qPingA : &qPingB);
        mx::array* destinationV = finalSubstep
            ? &outputs[1]
            : (substep % 2u == 0u ? &vPingA : &vPingB);
        mx::array* destinationScene = finalSubstep
            ? &scenePackedC
            : (substep % 2u == 0u
                   ? &scenePackedB
                   : &scenePackedA);
        mx::array* destinationHeaders = finalSubstep
            ? &outputs[6]
            : (substep % 2u == 0u
                   ? &manifoldHeadersA
                   : &manifoldHeadersB);
        mx::array* destinationPoints = finalSubstep
            ? &outputs[7]
            : (substep % 2u == 0u
                   ? &manifoldPointsA
                   : &manifoldPointsB);
        mx::array* destinationCounts = finalSubstep
            ? &outputs[8]
            : (substep % 2u == 0u
                   ? &manifoldCountsA
                   : &manifoldCountsB);
        destinationRodWitnesses = finalSubstep
            ? &outputs[14]
            : (substep % 2u == 0u
                   ? &rodWitnessesA
                   : &rodWitnessesB);
        destinationRodNodes =
            substep % 2u == 0u
            ? &rodNodesB
            : &rodNodesA;
        destinationRodEdges =
            substep % 2u == 0u
            ? &rodEdgesB
            : &rodEdgesA;

        MRMetalWorldPassGPU pass{};
        pass.controlStep = 0u;
        pass.physicsSubstep = substep;
        MRMetalWorldPassGPU solverPass = pass;
        solverPass.reserved0 = finalSubstep ? 1u : 0u;
        const std::uint32_t eventPassCount =
            hybridCCD
            ? world_->maxCCDAdvanceSolvePasses()
            : 1u;
        const mx::array* eventSourceQ = sourceQ;
        const mx::array* eventSourceV = sourceV;
        const mx::array* eventSourceScene = sourceScene;
        const mx::array* eventSourceHeaders = sourceHeaders;
        const mx::array* eventSourcePoints = sourcePoints;
        const mx::array* eventSourceCounts = sourceCounts;
        mx::array* eventDestinationQ = &eventQPingA;
        mx::array* eventDestinationV = &eventVPingA;
        mx::array* eventDestinationScene = &eventScenePingA;
        mx::array* eventDestinationHeaders =
            &eventManifoldHeadersA;
        mx::array* eventDestinationPoints =
            &eventManifoldPointsA;
        mx::array* eventDestinationCounts =
            &eventManifoldCountsA;
        mx::array* eventStateIn = &ccdEventStatesA;
        mx::array* eventStateOut = &ccdEventStatesB;
        const mx::array* eventSourceRodNodes =
            &eventRodNodesA;
        const mx::array* eventSourceRodEdges =
            &eventRodEdgesA;
        const mx::array* eventSourceRodWitnesses =
            &eventRodWitnessesA;
        mx::array* eventDestinationRodNodes =
            &eventRodNodesB;
        mx::array* eventDestinationRodEdges =
            &eventRodEdgesB;
        mx::array* eventDestinationRodWitnesses =
            &eventRodWitnessesB;
        if (hybridCCD) {
            setPhysicsKernel(
                "mr_world_initialize_ccd_event_state"
            );
            encoder.set_bytes(contactDispatch, 0);
            outputArray(ccdEventStatesA, 1);
            outputArray(ccdEventStatesB, 2);
            outputArray(ccdImpactClusters, 3);
            outputArray(outputs[16], 4);
            encoder.set_bytes(pass, 5);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_initialize_ccd_event_state"
                )
            );
            if (rodCount != 0u) {
                setPhysicsKernel(
                    "mr_world_initialize_rod_event_state"
                );
                encoder.set_bytes(contactDispatch, 0);
                inputArray(*sourceRodNodes, 1);
                inputArray(*sourceRodEdges, 2);
                inputArray(*sourceRodWitnesses, 3);
                outputArray(eventRodNodesA, 4);
                outputArray(eventRodEdgesA, 5);
                outputArray(eventRodWitnessesA, 6);
                outputArray(eventRodNodesB, 7);
                outputArray(eventRodEdgesB, 8);
                outputArray(eventRodWitnessesB, 9);
                dispatchThreads(
                    environments,
                    resources.kernel(
                        "mr_world_initialize_rod_event_state"
                    )
                );
            }
        }

        if (!hybridCCD && rodCount != 0u) {
            setPhysicsKernel("mr_world_pack_rod_state");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(*sourceRodNodes, 1);
            inputArray(*sourceRodEdges, 2);
            outputArray(rodInputPositions, 3);
            outputArray(rodInputVelocities, 4);
            outputArray(rodInputTwists, 5);
            outputArray(rodInputTwistRates, 6);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_pack_rod_state")
            );

            setPhysicsKernel("mr_discrete_elastic_rod_step");
            for (std::size_t rod = 0u;
                 rod < rodDispatches.size();
                 ++rod) {
                const std::size_t nodeOffset =
                    compiled.rodNodeOffsets()[rod];
                const std::size_t edgeOffset =
                    compiled.rodEdgeOffsets()[rod];
                const std::size_t bendOffset =
                    edgeOffset - rod;
                encoder.set_bytes(rodDispatches[rod], 0);
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodRestLengths
                    ),
                    1,
                    edgeOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodRestTwists
                    ),
                    2,
                    edgeOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodRestCurvatures
                    ),
                    3,
                    bendOffset * sizeof(mr_float4)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodInverseMasses
                    ),
                    4,
                    nodeOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodInverseTwistInertias
                    ),
                    5,
                    edgeOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodStretchStiffness
                    ),
                    6,
                    edgeOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodBendStiffness
                    ),
                    7,
                    bendOffset * sizeof(float)
                );
                encoder.set_buffer(
                    resources.buffer(
                        kImmutableRodTwistStiffness
                    ),
                    8,
                    bendOffset * sizeof(float)
                );
                inputArray(
                    rodInputPositions,
                    9,
                    nodeOffset * sizeof(mr_float4)
                );
                inputArray(
                    rodInputVelocities,
                    10,
                    nodeOffset * sizeof(mr_float4)
                );
                inputArray(
                    rodInputTwists,
                    11,
                    edgeOffset * sizeof(float)
                );
                inputArray(
                    rodInputTwistRates,
                    12,
                    edgeOffset * sizeof(float)
                );
                // Attachments are represented through the typed world
                // graph. This inert placeholder is never read because the
                // mechanics dispatch carries attachmentCount == 0.
                immutable(kImmutableRodInverseMasses, 13);
                outputArray(
                    rodOutputPositions,
                    14,
                    nodeOffset * sizeof(mr_float4)
                );
                outputArray(
                    rodOutputVelocities,
                    15,
                    nodeOffset * sizeof(mr_float4)
                );
                outputArray(
                    rodOutputTwists,
                    16,
                    edgeOffset * sizeof(float)
                );
                outputArray(
                    rodOutputTwistRates,
                    17,
                    edgeOffset * sizeof(float)
                );
                outputArray(
                    rodStatuses,
                    18,
                    rod * environments *
                        sizeof(MRRodGPUStatus)
                );
                immutable(kImmutableRodInverseMasses, 19);
                inputArray(ccdEventStatesA, 20);
                constexpr std::uint32_t fullRodMicrostep =
                    MR_CCD_SEGMENT_FULL_MICROSTEP;
                encoder.set_bytes(fullRodMicrostep, 21);
                encoder.dispatch_threadgroups(
                    MTL::Size(environments, 1u, 1u),
                    MTL::Size(
                        MR_ROD_GPU_MAX_NODES,
                        1u,
                        1u
                    )
                );
            }
            encoder.barrier();

            setPhysicsKernel("mr_world_unpack_rod_state");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(rodOutputPositions, 1);
            inputArray(rodOutputVelocities, 2);
            inputArray(rodOutputTwists, 3);
            inputArray(rodOutputTwistRates, 4);
            outputArray(candidateRodNodes, 5);
            outputArray(candidateRodEdges, 6);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_unpack_rod_state")
            );

            setPhysicsKernel("mr_world_latch_rod_status");
            encoder.set_bytes(worldDispatch, 0);
            encoder.set_bytes(contactDispatch, 1);
            encoder.set_bytes(pass, 2);
            inputArray(rodStatuses, 3);
            outputArray(worldStatuses, 4);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_latch_rod_status")
            );
            setPhysicsKernel(
                "mr_world_latch_rod_contact_status"
            );
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(pass, 1);
            inputArray(rodStatuses, 2);
            outputArray(outputs[16], 3);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_latch_rod_contact_status"
                )
            );

            setPhysicsKernel("mr_world_factor_rod_operator");
            for (std::size_t rod = 0u;
                 rod < rodDispatches.size();
                 ++rod) {
                const std::uint32_t rodIndex =
                    static_cast<std::uint32_t>(rod);
                encoder.set_bytes(contactDispatch, 0);
                encoder.set_bytes(rodDispatches[rod], 1);
                inputArray(candidateRodNodes, 2);
                immutable(kImmutableRodInverseMasses, 3);
                immutable(
                    kImmutableRodInverseTwistInertias,
                    4
                );
                immutable(kImmutableRodColliders, 5);
                immutable(kImmutableRodRestLengths, 6);
                immutable(kImmutableRodStretchStiffness, 7);
                immutable(kImmutableRodBendStiffness, 8);
                immutable(kImmutableRodTwistStiffness, 9);
                outputArray(rodFactorCaches, 10);
                outputArray(rodOperatorArena, 11);
                encoder.set_bytes(rodIndex, 12);
                encoder.set_bytes(pass, 13);
                dispatchThreads(
                    environments,
                    resources.kernel(
                        "mr_world_factor_rod_operator"
                    )
                );
            }
            encoder.barrier();
        }

        for (std::uint32_t eventPass = 0u;
             eventPass < eventPassCount;
             ++eventPass) {
        if (hybridCCD && eventPass != 0u) {
            setPhysicsKernel(
                "mr_world_prepare_ccd_event_pass"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(*eventStateIn, 1);
            outputArray(outputs[16], 2);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_prepare_ccd_event_pass"
                )
            );
        }
        encoder.set_compute_pipeline_state(
            articulationCount > 1u
            ? resources.multiABAKernel
            : resources.abaKernel
        );
        immutable(kImmutableWorld, 0);
        immutable(kImmutableArticulations, 1);
        immutable(kImmutableJoints, 2);
        immutable(kImmutableDofs, 3);
        immutable(kImmutableBodies, 4);
        if (articulationCount > 1u) {
            inputArray(multiABADispatches, 5);
        } else {
            encoder.set_bytes(abaDispatch, 5);
        }
        inputArray(*eventSourceQ, 6);
        inputArray(*eventSourceV, 7);
        inputArray(workingEffort, 8);
        immutable(kImmutableEmptyWrench, 9);
        outputArray(candidateAcceleration, 10);
        outputArray(candidateV, 11);
        outputArray(candidateQ, 12);
        outputArray(abaStatuses, 13);
        encoder.dispatch_threadgroups(
            MTL::Size(
                environments,
                articulationCount > 1u
                ? articulationCount
                : 1u,
                1u
            ),
            MTL::Size(kABAThreads, 1u, 1u)
        );
        encoder.barrier();

        encodeAllArticulationKinematics(
            *eventSourceQ,
            bodyPoses
        );

        if (hybridCCD) {
            const std::uint32_t remainingMode =
                MR_CCD_SEGMENT_REMAINING;
            setPhysicsKernel(
                "mr_world_materialize_event_articulation"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            inputArray(*eventSourceQ, 3);
            inputArray(*eventSourceV, 4);
            inputArray(candidateAcceleration, 5);
            inputArray(*eventStateIn, 6);
            outputArray(candidateQ, 7);
            outputArray(candidateV, 8);
            outputArray(abaStatuses, 9);
            outputArray(outputs[16], 10);
            encoder.set_bytes(remainingMode, 11);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_materialize_event_articulation"
                )
            );
        }

        if (futureKinematics) {
            encodeAllArticulationKinematics(
                candidateQ,
                futureBodyPoses
            );
        }

        setPhysicsKernel("mr_world_build_body_states");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableArticulations, 1);
        immutable(kImmutableBodies, 2);
        immutable(kImmutableSceneBodyIndices, 3);
        inputArray(bodyPoses, 4);
        inputArray(operatorStatuses, 5);
        inputArray(*eventSourceScene, 6);
        outputArray(currentBodies, 7);
        outputArray(outputs[16], 8);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_build_body_states")
        );

        if (hybridCCD) {
            const std::uint32_t remainingMode =
                MR_CCD_SEGMENT_REMAINING;
            setPhysicsKernel("mr_world_predict_scene_event");
            immutable(kImmutableWorld, 0);
            encoder.set_bytes(contactDispatch, 1);
            immutable(kImmutableBodies, 2);
            immutable(kImmutableSceneBodyIndices, 3);
            inputArray(currentBodies, 4);
            inputArray(*eventStateIn, 5);
            outputArray(candidateBodies, 6);
            outputArray(outputs[16], 7);
            encoder.set_bytes(remainingMode, 8);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_predict_scene_event"
                )
            );
        } else {
            setPhysicsKernel("mr_world_predict_scene");
            immutable(kImmutableWorld, 0);
            encoder.set_bytes(contactDispatch, 1);
            immutable(kImmutableBodies, 2);
            immutable(kImmutableSceneBodyIndices, 3);
            inputArray(currentBodies, 4);
            outputArray(candidateBodies, 5);
            outputArray(outputs[16], 6);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_predict_scene")
            );
            setPhysicsKernel(
                "mr_mlx_apply_family_body_damping"
            );
            encoder.set_bytes(adapterDispatch, 0);
            encoder.set_bytes(contactDispatch, 1);
            immutable(kImmutableBodies, 2);
            immutable(kImmutableSceneBodyIndices, 3);
            inputArray(inputs[16], 4);
            inputArray(currentBodies, 5);
            outputArray(candidateBodies, 6);
            inputArray(outputs[16], 7);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_mlx_apply_family_body_damping"
                )
            );
        }

        if (hybridCCD && rodCount != 0u) {
            encodeRodSubstep(
                pass,
                *eventSourceRodNodes,
                *eventSourceRodEdges,
                candidateRodNodes,
                candidateRodEdges,
                *eventStateIn,
                MR_CCD_SEGMENT_REMAINING
            );
        }

        if (!hybridCCD && rodToolPairCount != 0u) {
            setPhysicsKernel("mr_rod_tool_narrowphase");
            for (std::size_t rod = 0u;
                 rod < rodCollisionDispatches.size();
                 ++rod) {
                const MRRodGPUDispatch& rodDispatch =
                    rodCollisionDispatches[rod];
                if (rodDispatch.toolPairCount == 0u) {
                    continue;
                }
                encoder.set_bytes(rodDispatch, 0);
                immutable(kImmutableRodColliders, 1);
                immutable(kImmutableRodShapeSources, 2);
                immutable(kImmutableRodToolPairs, 3);
                immutable(kImmutableShapes, 4);
                inputArray(candidateBodies, 5);
                immutable(kImmutableGeometryHeaders, 6);
                immutable(kImmutableGeometryVertices, 7);
                immutable(kImmutableMeshNodes, 8);
                immutable(kImmutableMeshTriangles, 9);
                inputArray(rodOutputPositions, 10);
                inputArray(rodOutputVelocities, 11);
                inputArray(rodOutputTwistRates, 12);
                inputArray(*sourceRodWitnesses, 13);
                outputArray(rodWitnessCounts, 14);
                outputArray(candidateRodWitnesses, 15);
                outputArray(
                    rodStatuses,
                    16,
                    rod * environments *
                        sizeof(MRRodGPUStatus)
                );
                dispatchThreads(
                    static_cast<std::size_t>(environments) *
                        rodDispatch.toolPairCount,
                    resources.kernel(
                        "mr_rod_tool_narrowphase"
                    )
                );
            }
        }

        setPhysicsKernel("mr_world_project_swept_colliders");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableShapes, 1);
        immutable(kImmutableArticulations, 2);
        inputArray(currentBodies, 3);
        inputArray(candidateBodies, 4);
        inputArray(futureBodyPoses, 5);
        inputArray(candidateV, 6);
        immutable(kImmutableGeometryHeaders, 7);
        outputArray(projected, 8);
        outputArray(futureProjected, 9);
        outputArray(outputs[16], 10);
        inputArray(*eventStateIn, 11);
        dispatchThreads(
            static_cast<std::size_t>(environments) *
                shapeCount,
            resources.kernel(
                "mr_world_project_swept_colliders"
            )
        );
        if (hybridCCD && rodCount != 0u) {
            setPhysicsKernel(
                "mr_world_project_swept_rod_colliders"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableRodColliders, 1);
            inputArray(*eventSourceRodNodes, 2);
            inputArray(candidateRodNodes, 3);
            outputArray(projectedRodColliders, 4);
            outputArray(futureProjectedRodColliders, 5);
            outputArray(outputs[16], 6);
            dispatchThreads(
                static_cast<std::size_t>(environments) *
                    rodEdgeCount,
                resources.kernel(
                    "mr_world_project_swept_rod_colliders"
                )
            );
        }

        setPhysicsKernel("mr_world_flag_eligible_pairs");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableShapes, 1);
        immutable(kImmutableEligiblePairs, 2);
        inputArray(projected, 3);
        outputArray(pairFlags, 4);
        dispatchThreads(
            std::max<std::size_t>(pairFlagCount, 1u),
            resources.kernel("mr_world_flag_eligible_pairs")
        );

        if (world_->ccdMode() == MetalWorldCCDMode::hybrid) {
            setPhysicsKernel("mr_world_resolve_ccd");
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableShapes, 1);
            immutable(kImmutableEligiblePairs, 2);
            inputArray(pairFlags, 3);
            inputArray(projected, 4);
            inputArray(futureProjected, 5);
            immutable(kImmutableGeometryHeaders, 6);
            immutable(kImmutableGeometryVertices, 7);
            immutable(kImmutableMeshNodes, 8);
            immutable(kImmutableMeshTriangles, 9);
            outputArray(ccdPairs, 10);
            outputArray(outputs[16], 11);
            inputArray(*eventStateIn, 12);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_resolve_ccd")
            );
            if (rodCount != 0u) {
                setPhysicsKernel("mr_world_resolve_rod_ccd");
                encoder.set_bytes(contactDispatch, 0);
                immutable(kImmutableRodColliders, 1);
                immutable(kImmutableRodShapeSources, 2);
                immutable(kImmutableRodToolPairs, 3);
                immutable(kImmutableShapes, 4);
                inputArray(projectedRodColliders, 5);
                inputArray(futureProjectedRodColliders, 6);
                inputArray(projected, 7);
                inputArray(futureProjected, 8);
                immutable(kImmutableGeometryHeaders, 9);
                immutable(kImmutableGeometryVertices, 10);
                immutable(kImmutableMeshNodes, 11);
                immutable(kImmutableMeshTriangles, 12);
                outputArray(ccdPairs, 13);
                outputArray(outputs[16], 14);
                inputArray(*eventStateIn, 15);
                dispatchThreads(
                    environments,
                    resources.kernel("mr_world_resolve_rod_ccd")
                );
            }

            setPhysicsKernel(
                "mr_world_select_ccd_event_state"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(ccdPairs, 1);
            inputArray(*eventStateIn, 2);
            outputArray(*eventStateOut, 3);
            outputArray(ccdImpactClusters, 4);
            outputArray(outputs[16], 5);
            encoder.set_bytes(pass, 6);
            encoder.set_bytes(eventPass, 7);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_select_ccd_event_state"
                )
            );

            const std::uint32_t selectedMode =
                MR_CCD_SEGMENT_SELECTED;
            setPhysicsKernel(
                "mr_world_materialize_event_articulation"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            inputArray(*eventSourceQ, 3);
            inputArray(*eventSourceV, 4);
            inputArray(candidateAcceleration, 5);
            inputArray(*eventStateOut, 6);
            outputArray(candidateQ, 7);
            outputArray(candidateV, 8);
            outputArray(abaStatuses, 9);
            outputArray(outputs[16], 10);
            encoder.set_bytes(selectedMode, 11);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_materialize_event_articulation"
                )
            );

            encodeAllArticulationKinematics(
                candidateQ,
                bodyPoses
            );

            setPhysicsKernel("mr_world_predict_scene_event");
            immutable(kImmutableWorld, 0);
            encoder.set_bytes(contactDispatch, 1);
            immutable(kImmutableBodies, 2);
            immutable(kImmutableSceneBodyIndices, 3);
            inputArray(currentBodies, 4);
            inputArray(*eventStateOut, 5);
            outputArray(candidateBodies, 6);
            outputArray(outputs[16], 7);
            encoder.set_bytes(selectedMode, 8);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_predict_scene_event"
                )
            );
            if (rodCount != 0u) {
                encodeRodSubstep(
                    pass,
                    *eventSourceRodNodes,
                    *eventSourceRodEdges,
                    candidateRodNodes,
                    candidateRodEdges,
                    *eventStateOut,
                    selectedMode
                );
            }

            setPhysicsKernel(
                "mr_world_overlay_event_articulation_bodies"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableBodies, 2);
            inputArray(bodyPoses, 3);
            inputArray(operatorStatuses, 4);
            outputArray(candidateBodies, 5);
            outputArray(outputs[16], 6);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_overlay_event_articulation_bodies"
                )
            );

            setPhysicsKernel(
                "mr_world_project_event_colliders"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableShapes, 1);
            immutable(kImmutableGeometryHeaders, 2);
            inputArray(candidateBodies, 3);
            outputArray(projected, 4);
            outputArray(outputs[16], 5);
            dispatchThreads(
                static_cast<std::size_t>(environments) *
                    shapeCount,
                resources.kernel(
                    "mr_world_project_event_colliders"
                )
            );

            setPhysicsKernel("mr_world_flag_eligible_pairs");
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableShapes, 1);
            immutable(kImmutableEligiblePairs, 2);
            inputArray(projected, 3);
            outputArray(pairFlags, 4);
            dispatchThreads(
                std::max<std::size_t>(pairFlagCount, 1u),
                resources.kernel(
                    "mr_world_flag_eligible_pairs"
                )
            );
            if (rodToolPairCount != 0u) {
                setPhysicsKernel("mr_rod_tool_narrowphase");
                for (std::size_t rod = 0u;
                     rod < rodCollisionDispatches.size();
                     ++rod) {
                    const MRRodGPUDispatch& rodDispatch =
                        rodCollisionDispatches[rod];
                    if (rodDispatch.toolPairCount == 0u) {
                        continue;
                    }
                    encoder.set_bytes(rodDispatch, 0);
                    immutable(kImmutableRodColliders, 1);
                    immutable(kImmutableRodShapeSources, 2);
                    immutable(kImmutableRodToolPairs, 3);
                    immutable(kImmutableShapes, 4);
                    inputArray(candidateBodies, 5);
                    immutable(kImmutableGeometryHeaders, 6);
                    immutable(kImmutableGeometryVertices, 7);
                    immutable(kImmutableMeshNodes, 8);
                    immutable(kImmutableMeshTriangles, 9);
                    inputArray(rodOutputPositions, 10);
                    inputArray(rodOutputVelocities, 11);
                    inputArray(rodOutputTwistRates, 12);
                    inputArray(*eventSourceRodWitnesses, 13);
                    outputArray(rodWitnessCounts, 14);
                    outputArray(candidateRodWitnesses, 15);
                    outputArray(
                        rodStatuses,
                        16,
                        rod * environments *
                            sizeof(MRRodGPUStatus)
                    );
                    dispatchThreads(
                        static_cast<std::size_t>(environments) *
                            rodDispatch.toolPairCount,
                        resources.kernel(
                            "mr_rod_tool_narrowphase"
                        )
                    );
                }

                setPhysicsKernel(
                    "mr_world_tag_rod_ccd_witnesses"
                );
                encoder.set_bytes(contactDispatch, 0);
                inputArray(ccdPairs, 1);
                inputArray(outputs[16], 2);
                outputArray(candidateRodWitnesses, 3);
                dispatchThreads(
                    rodWitnessCount,
                    resources.kernel(
                        "mr_world_tag_rod_ccd_witnesses"
                    )
                );
            }
        }

        const std::size_t pairWorkers =
            static_cast<std::size_t>(environments) *
            pairCapacity;
        const std::size_t persistentPairWorkers = std::min(
            pairWorkers,
            static_cast<std::size_t>(
                MR_WORLD_QUEUE_THREADS_PER_THREADGROUP
            ) * 64u
        );
        for (std::uint32_t workClass =
                 MR_WORLD_WORK_ANALYTIC;
             workClass <= MR_WORLD_WORK_MESH;
             ++workClass) {
            if ((activePairClassMask &
                 (1u << workClass)) == 0u) {
                continue;
            }
            const mr_uint4 classConfig{
                workClass,
                activePairClassMask,
                1u,
                0u,
            };
            setPhysicsKernel(
                "mr_world_flag_pair_work_class"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableShapes, 1);
            immutable(kImmutableEligiblePairs, 2);
            inputArray(pairFlags, 3);
            outputArray(compactionFlags, 4);
            encoder.set_bytes(classConfig, 5);
            dispatchThreads(
                std::max<std::size_t>(pairFlagCount, 1u),
                resources.kernel(
                    "mr_world_flag_pair_work_class"
                )
            );
            encodeScan(
                compactionFlags,
                std::max<std::size_t>(pairFlagCount, 1u),
                workClass
            );

            setPhysicsKernel("mr_world_scatter_pair_queue");
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableShapes, 1);
            immutable(kImmutableEligiblePairs, 2);
            inputArray(compactionFlags, 3);
            inputArray(scanOffsets, 4);
            outputArray(pairWork, 5);
            outputArray(workHeaders, 6);
            encoder.set_bytes(classConfig, 7);
            dispatchThreads(
                std::max<std::size_t>(pairFlagCount, 1u),
                resources.kernel(
                    "mr_world_scatter_pair_queue"
                )
            );
        }

        for (std::uint32_t workClass =
                 MR_WORLD_WORK_ANALYTIC;
             workClass <= MR_WORLD_WORK_MESH;
             ++workClass) {
            if ((activePairClassMask &
                 (1u << workClass)) == 0u) {
                continue;
            }
            const mr_uint4 classConfig{
                workClass,
                activePairClassMask,
                1u,
                0u,
            };
            const char* kernelName = nullptr;
            if (workClass == MR_WORLD_WORK_ANALYTIC ||
                workClass == MR_WORLD_WORK_SAT_CLIP) {
                kernelName =
                    "mr_world_narrowphase_pair_queue";
                setPhysicsKernel(kernelName);
                encoder.set_bytes(contactDispatch, 0);
                immutable(kImmutableShapes, 1);
                immutable(kImmutableEligiblePairs, 2);
                inputArray(projected, 3);
                inputArray(pairWork, 4);
                inputArray(workHeaders, 5);
                outputArray(pairRawStaging, 6);
                outputArray(pairRawCounts, 7);
                encoder.set_bytes(classConfig, 8);
            } else if (
                workClass == MR_WORLD_WORK_PRIMITIVE_GJK ||
                workClass == MR_WORLD_WORK_HULL_GJK ||
                workClass == MR_WORLD_WORK_HARD_CONVEX
            ) {
                kernelName =
                    "mr_world_narrowphase_convex_queue";
                setPhysicsKernel(kernelName);
                encoder.set_bytes(contactDispatch, 0);
                immutable(kImmutableShapes, 1);
                immutable(kImmutableEligiblePairs, 2);
                inputArray(projected, 3);
                inputArray(pairWork, 4);
                inputArray(workHeaders, 5);
                immutable(kImmutableGeometryHeaders, 6);
                immutable(kImmutableGeometryVertices, 7);
                outputArray(pairRawStaging, 8);
                outputArray(pairRawCounts, 9);
                inputArray(inputs[10], 10);
                outputArray(outputs[9], 11);
                encoder.set_bytes(classConfig, 12);
            } else {
                kernelName =
                    "mr_world_narrowphase_mesh_queue";
                setPhysicsKernel(kernelName);
                encoder.set_bytes(contactDispatch, 0);
                immutable(kImmutableShapes, 1);
                immutable(kImmutableEligiblePairs, 2);
                inputArray(projected, 3);
                inputArray(pairWork, 4);
                inputArray(workHeaders, 5);
                immutable(kImmutableGeometryHeaders, 6);
                immutable(kImmutableGeometryVertices, 7);
                immutable(kImmutableMeshNodes, 8);
                immutable(kImmutableMeshTriangles, 9);
                outputArray(pairRawStaging, 10);
                outputArray(pairRawCounts, 11);
                outputArray(outputs[9], 12);
                encoder.set_bytes(classConfig, 13);
            }
            dispatchThreads(
                persistentPairWorkers,
                resources.kernel(kernelName)
            );
        }

        setPhysicsKernel(
            "mr_world_initialize_multi_articulation_queries"
        );
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableArticulations, 1);
        outputArray(pointQueries, 2);
        inputArray(outputs[16], 3);
        dispatchThreads(
            environments,
            resources.kernel(
                "mr_world_initialize_multi_articulation_queries"
            )
        );

        setPhysicsKernel(
            "mr_world_seed_authored_constraint_ir"
        );
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableAuthoredIRBlocks, 1);
        immutable(kImmutableAuthoredIREndpoints, 2);
        immutable(kImmutableAuthoredIRRows, 3);
        immutable(kImmutableAuthoredIRCones, 4);
        immutable(kImmutableAuthoredIRWarmImpulses, 5);
        immutable(kImmutableDynamicNodes, 6);
        outputArray(contacts, 7);
        outputArray(contactMetadata, 8);
        outputArray(irBlocks, 9);
        outputArray(irEndpoints, 10);
        outputArray(endpointRuntime, 11);
        outputArray(irRows, 12);
        outputArray(irCones, 13);
        outputArray(outputs[16], 14);
        immutable(kImmutableBodyDynamicNodes, 15);
        inputArray(candidateBodies, 16);
        inputArray(candidateRodNodes, 17);
        dispatchThreads(
            environments,
            resources.kernel(
                "mr_world_seed_authored_constraint_ir"
            )
        );

        setPhysicsKernel("mr_world_finalize_pair_manifold");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableShapes, 1);
        inputArray(
            hybridCCD ? candidateBodies : currentBodies,
            2
        );
        immutable(kImmutableEligiblePairs, 3);
        inputArray(*eventSourceCounts, 4);
        inputArray(*eventSourceHeaders, 5);
        inputArray(*eventSourcePoints, 6);
        inputArray(pairFlags, 7);
        inputArray(pairRawCounts, 8);
        inputArray(pairRawStaging, 9);
        inputArray(outputs[9], 10);
        inputArray(ccdPairs, 11);
        outputArray(pairManifoldHeaders, 12);
        outputArray(pairManifoldPoints, 13);
        outputArray(manifoldScatter, 14);
        inputArray(outputs[16], 15);
        encoder.set_bytes(pass, 16);
        dispatchThreads(
            pairFlagCount,
            resources.kernel(
                "mr_world_finalize_pair_manifold"
            )
        );

        setPhysicsKernel("mr_world_scan_manifold_ir");
        encoder.set_bytes(contactDispatch, 0);
        outputArray(manifoldScatter, 1);
        outputArray(candidateManifoldCounts, 2);
        outputArray(outputs[16], 3);
        encoder.set_bytes(pass, 4);
        encoder.dispatch_threads(
            MTL::Size(
                static_cast<std::size_t>(environments) *
                    MR_SIMD_WIDTH,
                1u,
                1u
            ),
            MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
        );
        encoder.barrier();

        if (rodWitnessCount != 0u) {
            setPhysicsKernel("mr_world_scan_rod_contact_ir");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(rodWitnessCounts, 1);
            inputArray(candidateRodWitnesses, 2);
            outputArray(rodContactScratch, 3);
            outputArray(outputs[16], 4);
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
            );
            encoder.barrier();
        }

        setPhysicsKernel(
            "mr_world_scatter_manifold_records"
        );
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableEligiblePairs, 1);
        inputArray(pairRawCounts, 2);
        inputArray(pairRawStaging, 3);
        inputArray(pairManifoldHeaders, 4);
        inputArray(pairManifoldPoints, 5);
        inputArray(manifoldScatter, 6);
        inputArray(outputs[16], 7);
        outputArray(candidatePairs, 8);
        outputArray(rawContacts, 9);
        outputArray(rawPairIndices, 10);
        outputArray(candidateManifoldHeaders, 11);
        outputArray(candidateManifoldPoints, 12);
        dispatchThreads(
            pairFlagCount,
            resources.kernel(
                "mr_world_scatter_manifold_records"
            )
        );

        setPhysicsKernel("mr_world_scatter_manifold_ir");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableShapes, 1);
        immutable(kImmutableMaterials, 2);
        inputArray(
            hybridCCD ? candidateBodies : currentBodies,
            3
        );
        immutable(kImmutableArticulations, 4);
        immutable(kImmutableEligiblePairs, 5);
        inputArray(pairManifoldHeaders, 6);
        inputArray(pairManifoldPoints, 7);
        inputArray(manifoldScatter, 8);
        inputArray(outputs[16], 9);
        outputArray(contacts, 10);
        outputArray(contactMetadata, 11);
        outputArray(irBlocks, 12);
        outputArray(irEndpoints, 13);
        outputArray(endpointRuntime, 14);
        outputArray(irRows, 15);
        outputArray(irCones, 16);
        outputArray(pointQueries, 17);
        immutable(kImmutableBodyDynamicNodes, 18);
        dispatchThreads(
            pairFlagCount *
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY,
            resources.kernel(
                "mr_world_scatter_manifold_ir"
            )
        );

        if (rodWitnessCount != 0u) {
            setPhysicsKernel("mr_world_scatter_rod_contact_ir");
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableRodColliders, 1);
            immutable(kImmutableRodToolPairs, 2);
            immutable(kImmutableShapes, 3);
            immutable(kImmutableMaterials, 4);
            inputArray(candidateBodies, 5);
            inputArray(rodWitnessCounts, 6);
            inputArray(candidateRodWitnesses, 7);
            inputArray(rodContactScratch, 8);
            inputArray(outputs[16], 9);
            outputArray(contacts, 10);
            outputArray(contactMetadata, 11);
            outputArray(irBlocks, 12);
            outputArray(irEndpoints, 13);
            outputArray(endpointRuntime, 14);
            outputArray(irRows, 15);
            outputArray(irCones, 16);
            outputArray(pointQueries, 17);
            immutable(kImmutableBodyDynamicNodes, 18);
            outputArray(rodConstraintWitnessIndices, 19);
            dispatchThreads(
                rodWitnessCount,
                resources.kernel(
                    "mr_world_scatter_rod_contact_ir"
                )
            );
        }

        setPhysicsKernel(
            "mr_mlx_apply_family_contact_parameters"
        );
        encoder.set_bytes(adapterDispatch, 0);
        encoder.set_bytes(contactDispatch, 1);
        inputArray(inputs[16], 2);
        inputArray(outputs[16], 3);
        outputArray(contacts, 4);
        dispatchThreads(
            environments,
            resources.kernel(
                "mr_mlx_apply_family_contact_parameters"
            )
        );

        setPhysicsKernel("mr_world_finalize_factor_dispatch");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(outputs[16], 1);
        outputArray(operatorDispatch, 2);
        outputArray(activeIndirect, 3);
        dispatchThreads(
            1u,
            resources.kernel(
                "mr_world_finalize_factor_dispatch"
            )
        );

        setPhysicsKernel("mr_world_fill_point_query_tail");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(operatorDispatch, 1);
        immutable(kImmutableArticulations, 2);
        inputArray(outputs[16], 3);
        outputArray(pointQueries, 4);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_fill_point_query_tail")
        );

        encodeAllArticulationFactors(
            hybridCCD ? candidateQ : *eventSourceQ,
            bodyPoses
        );

        setPhysicsKernel("mr_world_evaluate_constraint_ir");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(contacts, 1);
        outputArray(contacts, 2);
        inputArray(irBlocks, 3);
        inputArray(irRows, 4);
        inputArray(irCones, 5);
        inputArray(candidateBodies, 6);
        inputArray(candidateV, 7);
        inputArray(pointJacobians, 8);
        inputArray(operatorStatuses, 9);
        outputArray(evaluatedRows, 10);
        outputArray(evaluatedCones, 11);
        outputArray(factorCaches, 12);
        outputArray(outputs[16], 13);
        inputArray(irEndpoints, 14);
        inputArray(candidateRodNodes, 15);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_evaluate_constraint_ir")
        );

        setPhysicsKernel("mr_world_build_contact_islands");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(candidateBodies, 1);
        outputArray(contacts, 2);
        outputArray(irBlocks, 3);
        outputArray(islands, 4);
        outputArray(outputs[16], 5);
        immutable(kImmutableDynamicNodes, 6);
        immutable(kImmutableBodyDynamicNodes, 7);
        inputArray(endpointRuntime, 8);
        outputArray(islandNodeReferences, 9);
        outputArray(islandConstraintReferences, 10);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_build_contact_islands")
        );

        if (!qualityMode &&
            contactDispatch.authoredConstraintCount != 0u) {
            setPhysicsKernel(
                "mr_world_solve_generalized_constraints"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            outputArray(candidateV, 2);
            outputArray(contacts, 3);
            inputArray(irBlocks, 4);
            inputArray(irEndpoints, 5);
            inputArray(evaluatedRows, 6);
            outputArray(outputs[16], 7);
            encoder.set_bytes(solverPass, 8);
            outputArray(candidateBodies, 9);
            outputArray(candidateRodNodes, 10);
            immutable(kImmutableRodInverseMasses, 11);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_solve_generalized_constraints"
                )
            );
        }

        if (qualityMode) {
            setPhysicsKernel("mr_world_prepare_unified_quality");
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(qualityDispatch, 1);
            immutable(kImmutableSceneBodyIndices, 2);
            inputArray(factor, 3);
            inputArray(pointJacobians, 4);
            inputArray(candidateV, 5);
            inputArray(candidateBodies, 6);
            inputArray(contacts, 7);
            inputArray(irEndpoints, 8);
            inputArray(irBlocks, 9);
            inputArray(evaluatedRows, 10);
            inputArray(evaluatedCones, 11);
            outputArray(outputs[16], 12);
            outputArray(qualityBlocks, 13);
            outputArray(qualityDynamics, 14);
            outputArray(qualityJacobian, 15);
            outputArray(qualityBias, 16);
            outputArray(qualityFreeVelocity, 17);
            outputArray(qualityWarmVelocity, 18);
            outputArray(qualityWarmImpulses, 19);
            inputArray(candidateRodNodes, 20);
            inputArray(candidateRodEdges, 21);
            immutable(kImmutableRodInverseMasses, 22);
            immutable(
                kImmutableRodInverseTwistInertias,
                23
            );
            immutable(kImmutableRodColliders, 24);
            inputArray(candidateRodWitnesses, 25);
            inputArray(rodConstraintWitnessIndices, 26);
            inputArray(rodFactorCaches, 27);
            inputArray(rodOperatorArena, 28);
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
            );
            encoder.barrier();

            setPhysicsKernel(
                "mr_world_reconstruct_unified_quality_warm_start"
            );
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(qualityDispatch, 1);
            inputArray(factor, 2);
            inputArray(qualityDynamics, 3);
            inputArray(qualityJacobian, 4);
            inputArray(qualityFreeVelocity, 5);
            inputArray(qualityWarmImpulses, 6);
            outputArray(qualityWarmVelocity, 7);
            outputArray(outputs[16], 8);
            inputArray(rodFactorCaches, 9);
            inputArray(rodOperatorArena, 10);
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
            );
            encoder.barrier();

            setPhysicsKernel(
                "mr_world_build_unified_quality_queue"
            );
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(qualityDispatch, 1);
            inputArray(outputs[16], 2);
            outputArray(qualityWorkQueue, 3);
            outputArray(qualityWorkPackets, 4);
            encoder.dispatch_threadgroups(
                MTL::Size(1u, 1u, 1u),
                MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
            );
            encoder.barrier();

            setPhysicsKernel(
                "mr_unified_quality_solve_persistent"
            );
            encoder.set_bytes(qualityDispatch, 0);
            inputArray(qualityBlocks, 1);
            inputArray(qualityDynamics, 2);
            inputArray(qualityJacobian, 3);
            inputArray(qualityBias, 4);
            inputArray(qualityFreeVelocity, 5);
            inputArray(qualityWarmVelocity, 6);
            inputArray(qualityWarmImpulses, 7);
            outputArray(qualityOutputVelocity, 8);
            outputArray(qualityOutputImpulses, 9);
            outputArray(qualityDerivatives, 10);
            outputArray(qualityHessian, 11);
            outputArray(qualityStatuses, 12);
            outputArray(qualityWorkQueue, 13);
            inputArray(qualityWorkPackets, 14);
            encoder.dispatch_threadgroups(
                MTL::Size(
                    std::max(
                        contactDispatch.waveWorkerGroupCount,
                        1u
                    ),
                    1u,
                    1u
                ),
                MTL::Size(MR_SIMD_WIDTH, 1u, 1u)
            );
            encoder.barrier();

            setPhysicsKernel("mr_world_apply_unified_quality");
            encoder.set_bytes(contactDispatch, 0);
            encoder.set_bytes(qualityDispatch, 1);
            immutable(kImmutableSceneBodyIndices, 2);
            inputArray(qualityStatuses, 3);
            inputArray(qualityOutputVelocity, 4);
            inputArray(qualityOutputImpulses, 5);
            outputArray(candidateV, 6);
            outputArray(candidateBodies, 7);
            outputArray(contacts, 8);
            inputArray(contactMetadata, 9);
            outputArray(candidateManifoldPoints, 10);
            outputArray(outputs[16], 11);
            outputArray(candidateRodNodes, 12);
            outputArray(candidateRodEdges, 13);
            outputArray(candidateRodWitnesses, 14);
            inputArray(rodConstraintWitnessIndices, 15);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_apply_unified_quality"
                )
            );
            encoder.barrier();

            setPhysicsKernel(
                "mr_world_publish_unified_quality_queue_status"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(qualityWorkQueue, 1);
            outputArray(outputs[16], 2);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_publish_unified_quality_queue_status"
                )
            );
            encoder.barrier();
        } else if (
            world_->solverMode() ==
                MetalWorldSolverMode::throughputTGS
        ) {
            setPhysicsKernel("mr_world_build_contact_tiles");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(candidateBodies, 1);
            inputArray(contacts, 2);
            inputArray(islands, 3);
            outputArray(denseIslandWork, 4);
            outputArray(contactTiles, 5);
            outputArray(tileConstraintIndices, 6);
            outputArray(outputs[16], 7);
            outputArray(compactionFlags, 8);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_build_contact_tiles"
                )
            );

            encodeScan(
                compactionFlags,
                islandWorkCount,
                MR_WORLD_WORK_SOLVER
            );

            setPhysicsKernel("mr_world_scatter_island_queue");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(compactionFlags, 1);
            inputArray(scanOffsets, 2);
            inputArray(denseIslandWork, 3);
            outputArray(compactIslandWork, 4);
            outputArray(workHeaders, 5);
            dispatchThreads(
                islandWorkCount,
                resources.kernel(
                    "mr_world_scatter_island_queue"
                )
            );

            setPhysicsKernel("mr_world_select_solver_cohort");
            inputArray(compactIslandWork, 0);
            outputArray(workHeaders, 1);
            outputArray(waveWorkPackets, 2);
            encoder.dispatch_threadgroups(
                MTL::Size(1u, 1u, 1u),
                MTL::Size(
                    MR_WAVE32_CONTACTS_PER_TILE,
                    1u,
                    1u
                )
            );
            encoder.barrier();

            if (constraintCapacity > 256u) {
                setPhysicsKernel(
                    "mr_world_flag_distributed_islands"
                );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(denseIslandWork, 1);
            outputArray(compactionFlags, 2);
            dispatchThreads(
                islandWorkCount,
                resources.kernel(
                    "mr_world_flag_distributed_islands"
                )
            );
            encodeScan(
                compactionFlags,
                islandWorkCount,
                MR_WORLD_WORK_SOLVER_DISTRIBUTED
            );
            setPhysicsKernel(
                "mr_world_scatter_distributed_island_queue"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(compactionFlags, 1);
            inputArray(scanOffsets, 2);
            inputArray(denseIslandWork, 3);
            outputArray(compactIslandWork, 4);
            outputArray(workHeaders, 5);
            dispatchThreads(
                islandWorkCount,
                resources.kernel(
                    "mr_world_scatter_distributed_island_queue"
                )
            );

            setPhysicsKernel(
                "mr_world_flag_distributed_tiles"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(outputs[16], 1);
            inputArray(denseIslandWork, 2);
            inputArray(contactTiles, 3);
            outputArray(compactionFlags, 4);
            dispatchThreads(
                tileWorkCount,
                resources.kernel(
                    "mr_world_flag_distributed_tiles"
                )
            );
            encodeScan(
                compactionFlags,
                tileWorkCount,
                MR_WORLD_WORK_SOLVER_SPILL
            );
            setPhysicsKernel(
                "mr_world_scatter_distributed_tile_queue"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(compactionFlags, 1);
            inputArray(scanOffsets, 2);
            inputArray(contactTiles, 3);
            outputArray(contactTiles, 4);
            outputArray(workHeaders, 5);
            dispatchThreads(
                tileWorkCount,
                resources.kernel(
                    "mr_world_scatter_distributed_tile_queue"
                )
            );

            const std::size_t distributedTileWorkers =
                std::min<std::size_t>(
                    std::max<std::size_t>(
                        tileWorkCount,
                        1u
                    ),
                    resources.tuning.waveWorkerGroupCount
                );
            const std::size_t distributedIslandWorkers =
                std::min<std::size_t>(
                    std::max<std::size_t>(
                        islandWorkCount,
                        1u
                    ),
                    64u
                );
            const auto dispatchDistributedTiles = [&]() {
                encoder.dispatch_threadgroups(
                    MTL::Size(
                        distributedTileWorkers,
                        1u,
                        1u
                    ),
                    MTL::Size(
                        MR_WAVE32_CONTACTS_PER_TILE,
                        1u,
                        1u
                    )
                );
                encoder.barrier();
            };
            const auto dispatchDistributedIslands = [&]() {
                encoder.dispatch_threadgroups(
                    MTL::Size(
                        distributedIslandWorkers,
                        1u,
                        1u
                    ),
                    MTL::Size(
                        MR_WAVE32_CONTACTS_PER_TILE,
                        1u,
                        1u
                    )
                );
                encoder.barrier();
            };

            setPhysicsKernel(
                "mr_world_wave32_distributed_prepare"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            inputArray(pointJacobians, 2);
            inputArray(candidateBodies, 3);
            outputArray(contacts, 4);
            inputArray(evaluatedRows, 5);
            inputArray(evaluatedCones, 6);
            outputArray(responseColumns, 7);
            outputArray(impulseDeltas, 8);
            outputArray(preconditioners, 9);
            inputArray(outputs[16], 10);
            inputArray(denseIslandWork, 11);
            inputArray(contactTiles, 12);
            inputArray(tileConstraintIndices, 13);
            inputArray(workHeaders, 14);
            dispatchDistributedTiles();

            const std::uint32_t solverIterations =
                contactDispatch.velocityIterations +
                (
                    solverPass.reserved0 != 0u
                    ? contactDispatch.finalVelocityIterations
                    : 0u
                );
            const auto encodeDistributedReduce = [&](
                const std::uint32_t mode
            ) {
                setPhysicsKernel(
                    "mr_world_wave32_distributed_reduce"
                );
                encoder.set_bytes(contactDispatch, 0);
                inputArray(pointJacobians, 1);
                outputArray(candidateV, 2);
                outputArray(candidateBodies, 3);
                outputArray(contacts, 4);
                inputArray(contactMetadata, 5);
                inputArray(evaluatedRows, 6);
                inputArray(evaluatedCones, 7);
                inputArray(responseColumns, 8);
                inputArray(impulseDeltas, 9);
                inputArray(preconditioners, 10);
                outputArray(candidateManifoldPoints, 11);
                inputArray(outputs[16], 12);
                inputArray(compactIslandWork, 13);
                inputArray(contactTiles, 14);
                inputArray(tileConstraintIndices, 15);
                outputArray(waveStatuses, 16);
                inputArray(workHeaders, 17);
                MRMetalWorldPassGPU distributedPass =
                    solverPass;
                distributedPass.reserved0 =
                    solverIterations;
                distributedPass.reserved1 = mode;
                encoder.set_bytes(distributedPass, 18);
                dispatchDistributedIslands();
            };
            encodeDistributedReduce(0u);
                for (std::uint32_t iteration = 0u;
                     iteration < solverIterations;
                     ++iteration) {
                setPhysicsKernel(
                    "mr_world_wave32_distributed_delta"
                );
                encoder.set_bytes(contactDispatch, 0);
                inputArray(pointJacobians, 1);
                inputArray(candidateV, 2);
                inputArray(candidateBodies, 3);
                outputArray(contacts, 4);
                inputArray(evaluatedRows, 5);
                inputArray(evaluatedCones, 6);
                inputArray(preconditioners, 7);
                outputArray(impulseDeltas, 8);
                inputArray(outputs[16], 9);
                inputArray(contactTiles, 10);
                inputArray(tileConstraintIndices, 11);
                inputArray(workHeaders, 12);
                dispatchDistributedTiles();
                    encodeDistributedReduce(
                        iteration + 1u == solverIterations
                        ? 2u
                        : 1u
                    );
                }
            }

            setPhysicsKernel(
                "mr_world_wave32_solve_persistent"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            inputArray(pointJacobians, 2);
            outputArray(candidateV, 3);
            outputArray(candidateBodies, 4);
            outputArray(contacts, 5);
            inputArray(contactMetadata, 6);
            inputArray(evaluatedRows, 7);
            inputArray(evaluatedCones, 8);
            outputArray(responseColumns, 9);
            outputArray(candidateManifoldPoints, 10);
            inputArray(outputs[16], 11);
            inputArray(compactIslandWork, 12);
            inputArray(contactTiles, 13);
            inputArray(tileConstraintIndices, 14);
            outputArray(impulseDeltas, 15);
            outputArray(preconditioners, 16);
            outputArray(waveStatuses, 17);
            inputArray(workHeaders, 18);
            encoder.set_bytes(solverPass, 19);
            inputArray(waveWorkPackets, 20);
            outputArray(candidateRodNodes, 21);
            outputArray(candidateRodEdges, 22);
            immutable(kImmutableRodInverseMasses, 23);
            immutable(
                kImmutableRodInverseTwistInertias,
                24
            );
            immutable(kImmutableRodColliders, 25);
            outputArray(candidateRodWitnesses, 26);
            inputArray(rodConstraintWitnessIndices, 27);
            inputArray(rodFactorCaches, 28);
            outputArray(rodOperatorArena, 29);
            // Persistent workers claim multiple packets, so an
            // occupancy-sized pool is sufficient. Launching one worker per
            // possible island needlessly saturated the integrated GPU for
            // small replay batches and competed with WindowServer.
            const std::size_t persistentWaveWorkerBudget =
                std::max<std::size_t>(
                    1u,
                    std::min<std::size_t>(
                        resources.tuning.waveWorkerGroupCount,
                        32u
                    )
                );
            const std::size_t persistentWaveWorkers =
                std::min<std::size_t>(
                    std::max<std::size_t>(
                        2u *
                            static_cast<std::size_t>(
                                environments
                            ),
                        1u
                    ),
                    persistentWaveWorkerBudget
                );
            encoder.dispatch_threadgroups(
                MTL::Size(
                    persistentWaveWorkers,
                    1u,
                    1u
                ),
                MTL::Size(
                    MR_WAVE32_CONTACTS_PER_TILE,
                    1u,
                    1u
                )
            );
            encoder.barrier();

            setPhysicsKernel("mr_world_reduce_wave32_status");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(waveStatuses, 1);
            outputArray(outputs[16], 2);
            inputArray(workHeaders, 3);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_reduce_wave32_status"
                )
            );
            MRMetalWorldPassGPU replayPass = solverPass;
            replayPass.reserved1 = 1u;
            setPhysicsKernel(
                "mr_world_solve_contact_islands"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            inputArray(pointJacobians, 2);
            outputArray(candidateV, 3);
            outputArray(candidateBodies, 4);
            outputArray(contacts, 5);
            inputArray(contactMetadata, 6);
            inputArray(evaluatedRows, 7);
            inputArray(evaluatedCones, 8);
            outputArray(responseColumns, 9);
            outputArray(candidateManifoldPoints, 10);
            outputArray(outputs[16], 11);
            encoder.set_bytes(replayPass, 12);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_solve_contact_islands"
                )
            );
        } else {
            setPhysicsKernel("mr_world_solve_contact_islands");
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            inputArray(pointJacobians, 2);
            outputArray(candidateV, 3);
            outputArray(candidateBodies, 4);
            outputArray(contacts, 5);
            inputArray(contactMetadata, 6);
            inputArray(evaluatedRows, 7);
            inputArray(evaluatedCones, 8);
            outputArray(responseColumns, 9);
            outputArray(candidateManifoldPoints, 10);
            outputArray(outputs[16], 11);
            encoder.set_bytes(solverPass, 12);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_solve_contact_islands"
                )
            );
        }

        if (rodWitnessCount != 0u &&
            !qualityMode &&
            world_->solverMode() !=
                MetalWorldSolverMode::throughputTGS) {
            setPhysicsKernel(
                "mr_world_solve_rod_contact_constraints"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            inputArray(pointJacobians, 2);
            outputArray(candidateV, 3);
            outputArray(candidateBodies, 4);
            outputArray(candidateRodNodes, 5);
            outputArray(candidateRodEdges, 6);
            immutable(kImmutableRodInverseMasses, 7);
            immutable(
                kImmutableRodInverseTwistInertias,
                8
            );
            immutable(kImmutableRodColliders, 9);
            outputArray(contacts, 10);
            inputArray(irBlocks, 11);
            inputArray(evaluatedRows, 12);
            inputArray(evaluatedCones, 13);
            outputArray(responseColumns, 14);
            outputArray(candidateRodWitnesses, 15);
            outputArray(outputs[16], 16);
            encoder.set_bytes(solverPass, 17);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_solve_rod_contact_constraints"
                )
            );
        }

        if (!qualityMode &&
            contactDispatch.authoredConstraintCount != 0u) {
            setPhysicsKernel(
                "mr_world_solve_generalized_constraints"
            );
            encoder.set_bytes(contactDispatch, 0);
            inputArray(factor, 1);
            outputArray(candidateV, 2);
            outputArray(contacts, 3);
            inputArray(irBlocks, 4);
            inputArray(irEndpoints, 5);
            inputArray(evaluatedRows, 6);
            outputArray(outputs[16], 7);
            encoder.set_bytes(solverPass, 8);
            outputArray(candidateBodies, 9);
            outputArray(candidateRodNodes, 10);
            immutable(kImmutableRodInverseMasses, 11);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_solve_generalized_constraints"
                )
            );
        }

        const std::uint32_t stateAlreadyIntegrated =
            hybridCCD ? 1u : 0u;
        setPhysicsKernel("mr_world_project_joint_limits");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableArticulations, 1);
        immutable(kImmutableDofs, 2);
        inputArray(*eventSourceQ, 3);
        outputArray(candidateQ, 4);
        outputArray(candidateV, 5);
        outputArray(outputs[16], 6);
        encoder.set_bytes(stateAlreadyIntegrated, 7);
        dispatchThreads(
            environments,
            resources.kernel(
                "mr_world_project_joint_limits"
            )
        );

        if (!hybridCCD) {
            setPhysicsKernel("mr_world_integrate_contact_state");
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            immutable(kImmutableBodies, 3);
            immutable(kImmutableSceneBodyIndices, 4);
            inputArray(*eventSourceQ, 5);
            inputArray(candidateV, 6);
            outputArray(candidateQ, 7);
            outputArray(candidateBodies, 8);
            outputArray(outputs[16], 9);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_integrate_contact_state"
                )
            );
        } else {
            setPhysicsKernel(
                "mr_world_restore_inactive_event_candidate"
            );
            encoder.set_bytes(contactDispatch, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableSceneBodyIndices, 2);
            inputArray(*eventSourceQ, 3);
            inputArray(*eventSourceV, 4);
            inputArray(*eventSourceScene, 5);
            inputArray(*eventSourceHeaders, 6);
            inputArray(*eventSourcePoints, 7);
            inputArray(*eventSourceCounts, 8);
            outputArray(candidateQ, 9);
            outputArray(candidateV, 10);
            outputArray(candidateBodies, 11);
            outputArray(candidateManifoldHeaders, 12);
            outputArray(candidateManifoldPoints, 13);
            outputArray(candidateManifoldCounts, 14);
            inputArray(outputs[16], 15);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_restore_inactive_event_candidate"
                )
            );
            if (rodCount != 0u) {
                setPhysicsKernel(
                    "mr_world_restore_inactive_rod_event_candidate"
                );
                encoder.set_bytes(contactDispatch, 0);
                inputArray(*eventSourceRodNodes, 1);
                inputArray(*eventSourceRodEdges, 2);
                inputArray(*eventSourceRodWitnesses, 3);
                outputArray(candidateRodNodes, 4);
                outputArray(candidateRodEdges, 5);
                outputArray(candidateRodWitnesses, 6);
                inputArray(outputs[16], 7);
                dispatchThreads(
                    environments,
                    resources.kernel(
                        "mr_world_restore_inactive_rod_event_candidate"
                    )
                );
            }

            setPhysicsKernel(
                "mr_world_finalize_ccd_event_state"
            );
            encoder.set_bytes(contactDispatch, 0);
            outputArray(*eventStateOut, 1);
            outputArray(outputs[16], 2);
            encoder.set_bytes(eventPass, 3);
            dispatchThreads(
                environments,
                resources.kernel(
                    "mr_world_finalize_ccd_event_state"
                )
            );

            if (eventPass + 1u < eventPassCount) {
                setPhysicsKernel(
                    "mr_world_publish_event_segment"
                );
                encoder.set_bytes(contactDispatch, 0);
                immutable(kImmutableArticulations, 1);
                immutable(kImmutableSceneBodyIndices, 2);
                inputArray(*eventSourceQ, 3);
                inputArray(*eventSourceV, 4);
                inputArray(*eventSourceScene, 5);
                inputArray(*eventSourceHeaders, 6);
                inputArray(*eventSourcePoints, 7);
                inputArray(*eventSourceCounts, 8);
                inputArray(candidateQ, 9);
                inputArray(candidateV, 10);
                inputArray(candidateBodies, 11);
                inputArray(candidateManifoldHeaders, 12);
                inputArray(candidateManifoldPoints, 13);
                inputArray(candidateManifoldCounts, 14);
                inputArray(outputs[16], 15);
                outputArray(*eventDestinationQ, 16);
                outputArray(*eventDestinationV, 17);
                outputArray(*eventDestinationScene, 18);
                outputArray(*eventDestinationHeaders, 19);
                outputArray(*eventDestinationPoints, 20);
                outputArray(*eventDestinationCounts, 21);
                dispatchThreads(
                    environments,
                    resources.kernel(
                        "mr_world_publish_event_segment"
                    )
                );
                if (rodCount != 0u) {
                    setPhysicsKernel(
                        "mr_world_publish_rod_event_segment"
                    );
                    encoder.set_bytes(contactDispatch, 0);
                    inputArray(*eventSourceRodNodes, 1);
                    inputArray(*eventSourceRodEdges, 2);
                    inputArray(*eventSourceRodWitnesses, 3);
                    inputArray(candidateRodNodes, 4);
                    inputArray(candidateRodEdges, 5);
                    inputArray(candidateRodWitnesses, 6);
                    inputArray(outputs[16], 7);
                    outputArray(*eventDestinationRodNodes, 8);
                    outputArray(*eventDestinationRodEdges, 9);
                    outputArray(
                        *eventDestinationRodWitnesses,
                        10
                    );
                    dispatchThreads(
                        environments,
                        resources.kernel(
                            "mr_world_publish_rod_event_segment"
                        )
                    );
                }
                eventSourceQ = eventDestinationQ;
                eventSourceV = eventDestinationV;
                eventSourceScene = eventDestinationScene;
                eventSourceHeaders = eventDestinationHeaders;
                eventSourcePoints = eventDestinationPoints;
                eventSourceCounts = eventDestinationCounts;
                const bool destinationWasA =
                    eventDestinationQ == &eventQPingA;
                eventDestinationQ = destinationWasA
                    ? &eventQPingB
                    : &eventQPingA;
                eventDestinationV = destinationWasA
                    ? &eventVPingB
                    : &eventVPingA;
                eventDestinationScene = destinationWasA
                    ? &eventScenePingB
                    : &eventScenePingA;
                eventDestinationHeaders = destinationWasA
                    ? &eventManifoldHeadersB
                    : &eventManifoldHeadersA;
                eventDestinationPoints = destinationWasA
                    ? &eventManifoldPointsB
                    : &eventManifoldPointsA;
                eventDestinationCounts = destinationWasA
                    ? &eventManifoldCountsB
                    : &eventManifoldCountsA;
                eventSourceRodNodes =
                    eventDestinationRodNodes;
                eventSourceRodEdges =
                    eventDestinationRodEdges;
                eventSourceRodWitnesses =
                    eventDestinationRodWitnesses;
                const bool rodDestinationWasA =
                    eventDestinationRodNodes ==
                    &eventRodNodesA;
                eventDestinationRodNodes =
                    rodDestinationWasA
                    ? &eventRodNodesB
                    : &eventRodNodesA;
                eventDestinationRodEdges =
                    rodDestinationWasA
                    ? &eventRodEdgesB
                    : &eventRodEdgesA;
                eventDestinationRodWitnesses =
                    rodDestinationWasA
                    ? &eventRodWitnessesB
                    : &eventRodWitnessesA;
                std::swap(eventStateIn, eventStateOut);
            }
        }
        }

        setPhysicsKernel("mr_world_latch_contact_status");
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(pass, 1);
        inputArray(outputs[16], 2);
        outputArray(worldStatuses, 3);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_latch_contact_status")
        );

        setPhysicsKernel("mr_metal_world_commit");
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(pass, 1);
        inputArray(abaStatuses, 2);
        inputArray(candidateQ, 3);
        inputArray(candidateV, 4);
        outputArray(*destinationQ, 5);
        outputArray(*destinationV, 6);
        outputArray(worldStatuses, 7);
        inputArray(checkpointQ, 8);
        inputArray(checkpointV, 9);
        immutable(kImmutableWorld, 10);
        dispatchThreads(
            environments,
            resources.kernel("mr_metal_world_commit")
        );

        setPhysicsKernel("mr_world_commit_contact_state");
        encoder.set_bytes(worldDispatch, 0);
        encoder.set_bytes(contactDispatch, 1);
        encoder.set_bytes(pass, 2);
        inputArray(worldStatuses, 3);
        immutable(kImmutableSceneBodyIndices, 4);
        inputArray(candidateBodies, 5);
        inputArray(checkpointScene, 6);
        outputArray(*destinationScene, 7);
        inputArray(candidateManifoldHeaders, 8);
        inputArray(candidateManifoldPoints, 9);
        inputArray(candidateManifoldCounts, 10);
        inputArray(checkpointManifoldHeaders, 11);
        inputArray(checkpointManifoldPoints, 12);
        inputArray(checkpointManifoldCounts, 13);
        outputArray(*destinationHeaders, 14);
        outputArray(*destinationPoints, 15);
        outputArray(*destinationCounts, 16);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_commit_contact_state")
        );

        if (rodCount != 0u) {
            setPhysicsKernel("mr_world_commit_rod_state");
            encoder.set_bytes(worldDispatch, 0);
            encoder.set_bytes(contactDispatch, 1);
            encoder.set_bytes(pass, 2);
            inputArray(worldStatuses, 3);
            inputArray(candidateRodNodes, 4);
            inputArray(candidateRodEdges, 5);
            inputArray(checkpointRodNodes, 6);
            inputArray(checkpointRodEdges, 7);
            outputArray(*destinationRodNodes, 8);
            outputArray(*destinationRodEdges, 9);
            dispatchThreads(
                environments,
                resources.kernel("mr_world_commit_rod_state")
            );
        }
        if (rodWitnessCount != 0u) {
            setPhysicsKernel(
                "mr_world_commit_rod_contact_cache"
            );
            encoder.set_bytes(worldDispatch, 0);
            encoder.set_bytes(contactDispatch, 1);
            encoder.set_bytes(pass, 2);
            inputArray(worldStatuses, 3);
            inputArray(candidateRodWitnesses, 4);
            inputArray(checkpointRodWitnesses, 5);
            outputArray(*destinationRodWitnesses, 6);
            dispatchThreads(
                rodWitnessCount,
                resources.kernel(
                    "mr_world_commit_rod_contact_cache"
                )
            );
        }

        sourceQ = destinationQ;
        sourceV = destinationV;
        sourceScene = destinationScene;
        sourceHeaders = destinationHeaders;
        sourcePoints = destinationPoints;
        sourceCounts = destinationCounts;
        sourceRodNodes = destinationRodNodes;
        sourceRodEdges = destinationRodEdges;
        sourceRodWitnesses = destinationRodWitnesses;
    }

    if (rodCount != 0u) {
        setPhysicsKernel("mr_world_pack_rod_state");
        encoder.set_bytes(contactDispatch, 0);
        inputArray(*sourceRodNodes, 1);
        inputArray(*sourceRodEdges, 2);
        outputArray(outputs[10], 3);
        outputArray(outputs[11], 4);
        outputArray(outputs[12], 5);
        outputArray(outputs[13], 6);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_pack_rod_state")
        );
    }

    setPhysicsKernel("mr_mlx_commit_pair_cache");
    encoder.set_bytes(adapterDispatch, 0);
    inputArray(inputs[10], 1);
    inputArray(outputs[16], 2);
    outputArray(outputs[9], 3);
    dispatchThreads(
        std::max<std::size_t>(pairFlagCount, 1u),
        resources.kernel("mr_mlx_commit_pair_cache")
    );

    if (world_->hasTactile()) {
        // Re-materialize the committed state. The contact loop's body arena
        // may describe an intermediate microstep and articulation entries do
        // not carry generalized velocities, both of which would corrupt the
        // physical meaning of surface-relative motion.
        encodeAllArticulationKinematics(*sourceQ, bodyPoses);
        setPhysicsKernel("mr_world_build_body_states");
        encoder.set_bytes(contactDispatch, 0);
        immutable(kImmutableArticulations, 1);
        immutable(kImmutableBodies, 2);
        immutable(kImmutableSceneBodyIndices, 3);
        inputArray(bodyPoses, 4);
        inputArray(operatorStatuses, 5);
        inputArray(*sourceScene, 6);
        outputArray(currentBodies, 7);
        outputArray(outputs[16], 8);
        dispatchThreads(
            environments,
            resources.kernel("mr_world_build_body_states")
        );

        setPhysicsKernel(
            "mr_articulated_materialize_body_velocities"
        );
        for (std::uint32_t owner = 0u;
             owner < articulationCount;
             ++owner) {
            const MRArticulationGPU& owned =
                model.articulations[owner];
            immutable(kImmutableWorld, 0);
            immutable(kImmutableArticulations, 1);
            immutable(kImmutableJoints, 2);
            immutable(kImmutableDofs, 3);
            immutable(kImmutableBodies, 4);
            inputArray(
                kinematicsDispatches,
                5,
                owner *
                    sizeof(MRArticulatedOperatorDispatchGPU)
            );
            inputArray(
                *sourceQ,
                6,
                owned.qOffset * sizeof(float)
            );
            inputArray(
                *sourceV,
                7,
                owned.vOffset * sizeof(float)
            );
            outputArray(
                currentBodies,
                8,
                owned.firstBody * sizeof(MRBodyStateGPU)
            );
            outputArray(
                operatorStatuses,
                9,
                owner * environments *
                    sizeof(MRArticulatedOperatorStatusGPU)
            );
            encoder.dispatch_threadgroups(
                MTL::Size(environments, 1u, 1u),
                MTL::Size(kOperatorThreads, 1u, 1u)
            );
        }
        encoder.barrier();

        MRTactileDispatchGPU tactileDispatch{};
        tactileDispatch.counts = {
            environments,
            bodyCount,
            static_cast<std::uint32_t>(
                world_->tactile().sensors.size()
            ),
            static_cast<std::uint32_t>(
                world_->tactile().samples.size()
            ),
        };
        tactileDispatch.geometryCounts = {
            shapeCount,
            static_cast<std::uint32_t>(
                model.geometryHeaders.size()
            ),
            static_cast<std::uint32_t>(
                model.geometryVertices.size()
            ),
            static_cast<std::uint32_t>(
                model.convexFaces.size()
            ),
        };
        tactileDispatch.queryCounts = {
            static_cast<std::uint32_t>(
                model.meshBvhNodes.size()
            ),
            static_cast<std::uint32_t>(
                model.meshTriangles.size()
            ),
            static_cast<std::uint32_t>(
                world_->tactile().targetShapeIndices.size()
            ),
            constraintCapacity,
        };
        tactileDispatch.frameAndAbi = {
            0u,
            0u,
            MR_TACTILE_ABI_VERSION,
            0u,
        };
        tactileDispatch.timing = {
            world_->controlTimestep(),
            1.0f / world_->controlTimestep(),
            0.0f,
            static_cast<float>(world_->physicsSubsteps()) /
                world_->controlTimestep(),
        };

        setPhysicsKernel("mr_mlx_pack_tactile_contacts");
        encoder.set_bytes(adapterDispatch, 0);
        encoder.set_bytes(tactileDispatch, 1);
        immutable(kImmutableShapes, 2);
        inputArray(currentBodies, 3);
        inputArray(*sourceHeaders, 4);
        inputArray(contacts, 5);
        inputArray(contactMetadata, 6);
        inputArray(outputs[16], 7);
        outputArray(tactileContacts, 8);
        outputArray(tactileContactCounts, 9);
        dispatchThreads(
            environments,
            resources.kernel("mr_mlx_pack_tactile_contacts")
        );

        setPhysicsKernel("mr_tactile_sample");
        encoder.set_bytes(tactileDispatch, 0);
        immutable(kImmutableTactileSensors, 1);
        immutable(kImmutableTactileSamples, 2);
        immutable(kImmutableTactileTargets, 3);
        immutable(kImmutableShapes, 4);
        immutable(kImmutableGeometryHeaders, 5);
        immutable(kImmutableGeometryVertices, 6);
        immutable(kImmutableConvexFaces, 7);
        immutable(kImmutableMeshNodes, 8);
        immutable(kImmutableMeshTriangles, 9);
        inputArray(currentBodies, 10);
        inputArray(inputs[25], 11);
        inputArray(inputs[18], 12);
        inputArray(inputs[19], 13);
        inputArray(inputs[20], 14);
        inputArray(tactileHits, 15);
        inputArray(inputs[21], 16);
        inputArray(inputs[22], 17);
        outputArray(outputs[21], 18);
        outputArray(outputs[22], 19);
        outputArray(outputs[23], 20);
        outputArray(outputs[26], 21);
        outputArray(outputs[24], 22);
        outputArray(outputs[25], 23);
        outputArray(tactileHits, 24);
        inputArray(inputs[23], 25);
        dispatchThreads(
            static_cast<std::size_t>(environments) *
                world_->tactile().samples.size(),
            resources.kernel("mr_tactile_sample")
        );

        setPhysicsKernel("mr_tactile_reduce");
        encoder.set_bytes(tactileDispatch, 0);
        immutable(kImmutableTactileSensors, 1);
        immutable(kImmutableTactileSamples, 2);
        inputArray(currentBodies, 3);
        inputArray(tactileContacts, 4);
        inputArray(tactileContactCounts, 5);
        inputArray(inputs[25], 6);
        inputArray(outputs[21], 7);
        inputArray(outputs[24], 8);
        inputArray(outputs[25], 9);
        inputArray(outputs[23], 10);
        outputArray(tactileSummaries, 11);
        outputArray(outputs[37], 12);
        inputArray(inputs[23], 13);
        immutable(kImmutableTactileShapeToSensor, 14);
        dispatchThreads(
            static_cast<std::size_t>(environments) *
                world_->tactile().sensors.size(),
            resources.kernel("mr_tactile_reduce")
        );

        setPhysicsKernel("mr_mlx_unpack_tactile_summary");
        encoder.set_bytes(tactileDispatch, 0);
        inputArray(tactileSummaries, 1);
        inputArray(inputs[24], 2);
        outputArray(outputs[27], 3);
        outputArray(outputs[28], 4);
        outputArray(outputs[29], 5);
        outputArray(outputs[30], 6);
        outputArray(outputs[31], 7);
        outputArray(outputs[32], 8);
        outputArray(outputs[33], 9);
        outputArray(outputs[34], 10);
        outputArray(outputs[35], 11);
        outputArray(outputs[36], 12);
        dispatchThreads(
            static_cast<std::size_t>(environments) *
                world_->tactile().sensors.size(),
            resources.kernel(
                "mr_mlx_unpack_tactile_summary"
            )
        );

        setPhysicsKernel(
            "mr_mlx_publish_object_contact_points"
        );
        encoder.set_bytes(tactileDispatch, 0);
        inputArray(tactileHits, 1);
        immutable(kImmutableShapes, 2);
        inputArray(currentBodies, 3);
        outputArray(outputs[38], 4);
        outputArray(outputs[39], 5);
        outputArray(outputs[40], 6);
        dispatchThreads(
            static_cast<std::size_t>(environments) *
                world_->tactile().samples.size(),
            resources.kernel(
                "mr_mlx_publish_object_contact_points"
            )
        );
    }

    setPhysicsKernel("mr_mlx_unpack_scene_and_evidence");
    encoder.set_bytes(adapterDispatch, 0);
    inputArray(*sourceScene, 1);
    inputArray(contacts, 2);
    inputArray(contactMetadata, 3);
    inputArray(outputs[16], 4);
    inputArray(worldStatuses, 5);
    inputArray(candidateAcceleration, 6);
    outputArray(outputs[2], 7);
    outputArray(outputs[3], 8);
    outputArray(outputs[4], 9);
    outputArray(outputs[5], 10);
    outputArray(outputs[17], 11);
    outputArray(outputs[18], 12);
    outputArray(outputs[19], 13);
    outputArray(outputs[20], 14);
    outputArray(outputs[15], 15);
    dispatchThreads(
        environments,
        resources.kernel(
            "mr_mlx_unpack_scene_and_evidence"
        )
    );
}

std::vector<mx::array> WorldStepPrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo contact physics does not implement JVP"
    );
}

std::vector<mx::array> WorldStepPrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo contact physics does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
WorldStepPrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo environments use the native batch axis; "
        "vmap is not supported"
    );
}

const char* WorldStepPrimitive::name() const {
    return "MetalRoboWorldStep";
}

bool WorldStepPrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const WorldStepPrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get();
}

WorldFamilyStatePrimitive::WorldFamilyStatePrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world,
    MTL::Buffer* resetQ,
    MTL::Buffer* resetV,
    MTL::Buffer* resetSceneBodies,
    MTL::Buffer* scenarioHeaders,
    MTL::Buffer* scenarioValues,
    MTL::Buffer* bodyParameters,
    MTL::Buffer* controllerParameters,
    const std::uint32_t environmentCount,
    const std::uint32_t variationCount,
    const std::uint32_t bodyCount,
    const std::uint32_t articulationCount,
    const std::uint64_t generation
)
    : mx::Primitive(stream),
      world_(std::move(world)),
      resetQ_(resetQ),
      resetV_(resetV),
      resetSceneBodies_(resetSceneBodies),
      scenarioHeaders_(scenarioHeaders),
      scenarioValues_(scenarioValues),
      bodyParameters_(bodyParameters),
      controllerParameters_(controllerParameters),
      environmentCount_(environmentCount),
      variationCount_(variationCount),
      bodyCount_(bodyCount),
      articulationCount_(articulationCount),
      generation_(generation) {
    resetQ_->retain();
    resetV_->retain();
    resetSceneBodies_->retain();
    scenarioHeaders_->retain();
    scenarioValues_->retain();
    bodyParameters_->retain();
    controllerParameters_->retain();
}

WorldFamilyStatePrimitive::~WorldFamilyStatePrimitive() {
    resetQ_->release();
    resetV_->release();
    resetSceneBodies_->release();
    scenarioHeaders_->release();
    scenarioValues_->release();
    bodyParameters_->release();
    controllerParameters_->release();
}

void WorldFamilyStatePrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo world-family state has no MLX CPU fallback"
    );
}

void WorldFamilyStatePrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (!inputs.empty() || outputs.size() != 13u) {
        throw std::runtime_error(
            "MetalRobo world-family primitive received an invalid graph"
        );
    }
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalResources& resources = world_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);
    for (mx::array& output : outputs) {
        output.set_data(mx::allocator::malloc(output.nbytes()));
    }

    MRMLXWorldFamilyImportDispatchGPU dispatch{};
    dispatch.state = {
        environmentCount_,
        world_->world().nq(),
        world_->world().nv(),
        world_->world().sceneBodyCount(),
    };
    dispatch.topology = {
        bodyCount_,
        articulationCount_,
        variationCount_,
        MR_R2S2R_ABI_VERSION,
    };
    dispatch.generation = {
        static_cast<std::uint32_t>(generation_),
        static_cast<std::uint32_t>(generation_ >> 32u),
        0u,
        0u,
    };

    encoder.set_compute_pipeline_state(
        resources.kernel("mr_mlx_import_world_family_state")
    );
    encoder.set_bytes(dispatch, 0);
    encoder.set_buffer(resetQ_, 1);
    encoder.set_buffer(resetV_, 2);
    encoder.set_buffer(resetSceneBodies_, 3);
    encoder.set_buffer(scenarioHeaders_, 4);
    encoder.set_buffer(scenarioValues_, 5);
    encoder.set_buffer(bodyParameters_, 6);
    encoder.set_buffer(controllerParameters_, 7);
    for (std::size_t output = 0u;
         output < outputs.size();
         ++output) {
        encoder.set_output_array(
            outputs[output],
            static_cast<int>(output + 8u)
        );
    }
    const auto threadgroupSize = std::min<std::uint32_t>(
        kWorldThreads,
        resources.kernel(
            "mr_mlx_import_world_family_state"
        )->maxTotalThreadsPerThreadgroup()
    );
    encoder.dispatch_threads(
        MTL::Size(environmentCount_, 1u, 1u),
        MTL::Size(threadgroupSize, 1u, 1u)
    );
}

std::vector<mx::array> WorldFamilyStatePrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo world-family state does not implement JVP"
    );
}

std::vector<mx::array> WorldFamilyStatePrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo world-family state does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
WorldFamilyStatePrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo world families use the native batch axis; "
        "vmap is not supported"
    );
}

const char* WorldFamilyStatePrimitive::name() const {
    return "MetalRoboWorldFamilyState";
}

bool WorldFamilyStatePrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const WorldFamilyStatePrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get() &&
        typed->resetQ_ == resetQ_ &&
        typed->resetV_ == resetV_ &&
        typed->resetSceneBodies_ == resetSceneBodies_ &&
        typed->scenarioHeaders_ == scenarioHeaders_ &&
        typed->scenarioValues_ == scenarioValues_ &&
        typed->bodyParameters_ == bodyParameters_ &&
        typed->controllerParameters_ == controllerParameters_ &&
        typed->environmentCount_ == environmentCount_ &&
        typed->variationCount_ == variationCount_ &&
        typed->bodyCount_ == bodyCount_ &&
        typed->articulationCount_ == articulationCount_ &&
        typed->generation_ == generation_;
}

namespace {

struct MLXVisualEncoderContext {
    mx::metal::CommandEncoder* encoder = nullptr;
    bool hasEncodedDispatch = false;
};

void mlxVisualSetLabel(void*, const char*) {}

void mlxVisualUseResidencySet(
    void* opaque,
    void* residencySet
) {
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    auto* typedSet =
        reinterpret_cast<MTL::ResidencySet*>(residencySet);
    if (typedSet == nullptr) {
        throw std::runtime_error(
            "MLX visual renderer received no resource residency set"
        );
    }
    context.encoder->get_command_buffer()->useResidencySet(
        typedSet
    );
}

void mlxVisualSetPipeline(void* opaque, void* pipeline) {
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    context.encoder->set_compute_pipeline_state(
        reinterpret_cast<MTL::ComputePipelineState*>(pipeline)
    );
}

void mlxVisualSetBuffer(
    void* opaque,
    void* buffer,
    const std::size_t offset,
    const std::uint32_t index
) {
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    context.encoder->set_buffer(
        reinterpret_cast<MTL::Buffer*>(buffer),
        static_cast<int>(index),
        static_cast<std::int64_t>(offset)
    );
}

void mlxVisualSetBytes(
    void* opaque,
    const void* bytes,
    const std::size_t length,
    const std::uint32_t index
) {
    if (length != sizeof(MRHybridRenderUniformsGPU)) {
        throw std::runtime_error(
            "MLX visual encoder received an unsupported immediate "
            "constant layout"
        );
    }
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    context.encoder->set_bytes(
        *static_cast<const MRHybridRenderUniformsGPU*>(bytes),
        static_cast<int>(index)
    );
}

void mlxVisualDispatchThreads(
    void* opaque,
    const std::size_t threadCount,
    const std::size_t threadsPerThreadgroup
) {
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    // Renderer-owned scratch resources are not MLX arrays, so MLX cannot
    // infer their producer/consumer hazards. Order only adjacent renderer
    // passes; the first dispatch and the completed primitive need no extra
    // synchronization.
    if (context.hasEncodedDispatch) {
        context.encoder->barrier();
    }
    context.encoder->dispatch_threads(
        MTL::Size(threadCount, 1u, 1u),
        MTL::Size(threadsPerThreadgroup, 1u, 1u)
    );
    context.hasEncodedDispatch = true;
}

void mlxVisualDispatchThreadgroups(
    void* opaque,
    const std::size_t threadgroupCount,
    const std::size_t threadsPerThreadgroup
) {
    auto& context =
        *static_cast<MLXVisualEncoderContext*>(opaque);
    if (context.hasEncodedDispatch) {
        context.encoder->barrier();
    }
    context.encoder->dispatch_threadgroups(
        MTL::Size(threadgroupCount, 1u, 1u),
        MTL::Size(threadsPerThreadgroup, 1u, 1u)
    );
    context.hasEncodedDispatch = true;
}

} // namespace

VisualObservationPrimitive::VisualObservationPrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world,
    MRHybridRendererHandle* renderer,
    MRWorldFamilyHandle* worlds,
    const std::uint32_t environmentCount,
    const std::uint32_t bodyCount,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint64_t frameIndex,
    const std::uint32_t sensorSequence,
    const std::uint32_t cameraIndex,
    const std::uint32_t outputMask
)
    : mx::Primitive(stream),
      world_(std::move(world)),
      renderer_(renderer),
      worlds_(worlds),
      environmentCount_(environmentCount),
      bodyCount_(bodyCount),
      width_(width),
      height_(height),
      frameIndex_(frameIndex),
      sensorSequence_(sensorSequence),
      cameraIndex_(cameraIndex),
      outputMask_(outputMask) {
    mr_hybrid_renderer_retain(renderer_);
    mr_world_family_retain(worlds_);
}

VisualObservationPrimitive::~VisualObservationPrimitive() {
    mr_hybrid_renderer_destroy(renderer_);
    mr_world_family_destroy(worlds_);
}

void VisualObservationPrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo visual observation has no MLX CPU fallback"
    );
}

void VisualObservationPrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (inputs.size() != 2u || outputs.size() != 7u) {
        throw std::runtime_error(
            "MetalRobo visual observation received an invalid graph"
        );
    }
    for (mx::array& output : outputs) {
        output.set_data(mx::allocator::malloc(output.nbytes()));
    }

    auto& streamValue = stream();
    auto& commandEncoder =
        mx::metal::get_command_encoder(streamValue);
    // These bindings register MLX's data dependencies. Renderer kernels
    // replace lower argument slots before every dispatch.
    commandEncoder.set_input_array(inputs[0], 30);
    commandEncoder.set_input_array(inputs[1], 31);
    for (const mx::array& output : outputs) {
        commandEncoder.register_output_array(output);
    }

    MLXVisualEncoderContext context{
        &commandEncoder,
        false,
    };
    MRMetalComputeEncoderCallbacksC callbacks{};
    callbacks.context = &context;
    callbacks.set_label = mlxVisualSetLabel;
    callbacks.use_residency_set =
        mlxVisualUseResidencySet;
    callbacks.set_pipeline = mlxVisualSetPipeline;
    callbacks.set_buffer = mlxVisualSetBuffer;
    callbacks.set_bytes = mlxVisualSetBytes;
    callbacks.dispatch_threads = mlxVisualDispatchThreads;
    callbacks.dispatch_threadgroups =
        mlxVisualDispatchThreadgroups;

    MRHybridObservationBuffersC destination{};
    destination.rgb = outputs[0].buffer().ptr();
    destination.depth = outputs[1].buffer().ptr();
    destination.segmentation =
        (outputMask_ & MR_HYBRID_OUTPUT_SEGMENTATION) != 0u
        ? outputs[2].buffer().ptr()
        : nullptr;
    destination.identities =
        (outputMask_ & MR_HYBRID_OUTPUT_IDENTITIES) != 0u
        ? outputs[3].buffer().ptr()
        : nullptr;
    destination.normals =
        (outputMask_ & MR_HYBRID_OUTPUT_NORMALS) != 0u
        ? outputs[4].buffer().ptr()
        : nullptr;
    destination.motion =
        (outputMask_ & MR_HYBRID_OUTPUT_MOTION) != 0u
        ? outputs[5].buffer().ptr()
        : nullptr;
    destination.validity = outputs[6].buffer().ptr();
    destination.output_mask = outputMask_;

    if (mr_hybrid_renderer_encode_graph(
            renderer_,
            worlds_,
            inputs[0].buffer().ptr(),
            inputs[1].buffer().ptr(),
            static_cast<std::size_t>(inputs[0].offset()),
            static_cast<std::size_t>(inputs[1].offset()),
            environmentCount_,
            bodyCount_,
            frameIndex_,
            sensorSequence_,
            cameraIndex_,
            &callbacks,
            &destination
        ) != 0) {
        throw std::runtime_error(
            std::string{"MetalRobo MLX visual encode failed: "} +
            mr_last_error()
        );
    }
    // MLX may release the primitive immediately after encoding, before its
    // command buffer reaches the GPU. Keep native heaps and sampled-world
    // buffers alive until Metal signals completion.
    mr_hybrid_renderer_retain(renderer_);
    mr_world_family_retain(worlds_);
    commandEncoder.get_command_buffer()->addCompletedHandler(
        MTL::HandlerFunction{
            [renderer = renderer_, worlds = worlds_](
                MTL::CommandBuffer*
            ) {
                mr_hybrid_renderer_destroy(renderer);
                mr_world_family_destroy(worlds);
            }
        }
    );
}

std::vector<mx::array> VisualObservationPrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo visual observation does not implement JVP"
    );
}

std::vector<mx::array> VisualObservationPrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo visual observation does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
VisualObservationPrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo visual observations use the native batch axis; "
        "vmap is not supported"
    );
}

const char* VisualObservationPrimitive::name() const {
    return "MetalRoboVisualObservation";
}

bool VisualObservationPrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const VisualObservationPrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get() &&
        typed->renderer_ == renderer_ &&
        typed->worlds_ == worlds_ &&
        typed->environmentCount_ == environmentCount_ &&
        typed->bodyCount_ == bodyCount_ &&
        typed->width_ == width_ &&
        typed->height_ == height_ &&
        typed->frameIndex_ == frameIndex_ &&
        typed->sensorSequence_ == sensorSequence_ &&
        typed->cameraIndex_ == cameraIndex_ &&
        typed->outputMask_ == outputMask_;
}

BodyStatePrimitive::BodyStatePrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world
)
    : mx::Primitive(stream), world_(std::move(world)) {}

void BodyStatePrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo body-state materialization has no MLX CPU fallback"
    );
}

void BodyStatePrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    if (inputs.size() != 6u || outputs.size() != 1u) {
        throw std::runtime_error(
            "MetalRobo body-state primitive received an invalid graph"
        );
    }
    outputs[0].set_data(
        mx::allocator::malloc(outputs[0].nbytes())
    );
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalResources& resources = world_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);

    const CompiledWorld& compiled = world_->world();
    const EngineModel& model = compiled.model();
    const std::uint32_t environments =
        static_cast<std::uint32_t>(inputs[0].shape(0));
    const std::uint32_t bodies =
        static_cast<std::uint32_t>(model.bodies.size());
    const std::uint32_t articulations =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    const std::uint32_t scenes =
        compiled.sceneBodyCount();
    const auto rawTemporary = [](
        const std::size_t bytes
    ) {
        return temporary(
            {
                static_cast<mx::ShapeElem>(
                    std::max<std::size_t>(
                        (bytes + sizeof(std::uint32_t) - 1u) /
                            sizeof(std::uint32_t),
                        1u
                    )
                ),
            },
            mx::uint32
        );
    };
    mx::array bodyPoses = rawTemporary(
        static_cast<std::size_t>(environments) *
        bodies * sizeof(MRArticulatedBodyPoseGPU)
    );
    mx::array operatorStatuses = rawTemporary(
        static_cast<std::size_t>(environments) *
        articulations *
        sizeof(MRArticulatedOperatorStatusGPU)
    );
    mx::array dummyRecords = rawTemporary(
        sizeof(MRArticulatedPointImpulseGPU)
    );
    mx::array dummyFloat = temporary({1}, mx::float32);
    std::vector<mx::array> temporaries{
        bodyPoses,
        operatorStatuses,
        dummyRecords,
        dummyFloat,
    };
    encoder.add_temporaries(std::move(temporaries));

    auto* operatorKernel =
        resources.kernel("mr_articulated_operator");
    encoder.set_compute_pipeline_state(operatorKernel);
    for (std::uint32_t owner = 0u;
         owner < articulations;
         ++owner) {
        const MRArticulationGPU& articulation =
            model.articulations[owner];
        MRArticulatedOperatorDispatchGPU dispatch{};
        dispatch.articulationIndex = owner;
        dispatch.environmentCount = environments;
        dispatch.pointCount = 0u;
        dispatch.flags =
            MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY;
        dispatch.qStride = model.world.nq;
        dispatch.pointStride = 1u;
        dispatch.bodyPoseStride = bodies;
        dispatch.pointWorldStride = 1u;
        dispatch.massMatrixStride = 1u;
        dispatch.pointJacobianStride = 1u;
        dispatch.generalizedStride = model.world.nv;
        encoder.set_buffer(
            resources.buffer(kImmutableWorld),
            0
        );
        encoder.set_buffer(
            resources.buffer(kImmutableArticulations),
            1
        );
        encoder.set_buffer(
            resources.buffer(kImmutableJoints),
            2
        );
        encoder.set_buffer(
            resources.buffer(kImmutableDofs),
            3
        );
        encoder.set_buffer(
            resources.buffer(kImmutableBodies),
            4
        );
        encoder.set_bytes(dispatch, 5);
        encoder.set_input_array(
            inputs[0],
            6,
            articulation.qOffset * sizeof(float)
        );
        encoder.set_input_array(dummyRecords, 7);
        encoder.set_output_array(
            bodyPoses,
            8,
            articulation.firstBody *
                sizeof(MRArticulatedBodyPoseGPU)
        );
        encoder.set_output_array(dummyRecords, 9);
        encoder.set_output_array(dummyFloat, 10);
        encoder.set_output_array(dummyFloat, 11);
        encoder.set_output_array(dummyFloat, 12);
        encoder.set_output_array(dummyFloat, 13);
        encoder.set_output_array(
            operatorStatuses,
            14,
            static_cast<std::int64_t>(owner) *
                environments *
                sizeof(MRArticulatedOperatorStatusGPU)
        );
        encoder.dispatch_threadgroups(
            MTL::Size(environments, 1u, 1u),
            MTL::Size(kOperatorThreads, 1u, 1u)
        );
    }
    encoder.barrier();

    MRBodyStateMaterializeDispatchGPU materialize{};
    materialize.abiVersion = MR_SCENE_QUERY_ABI_VERSION;
    materialize.environmentCount = environments;
    materialize.bodyCount = bodies;
    materialize.articulationCount = articulations;
    materialize.sceneBodyCount = scenes;
    materialize.qStride = model.world.nq;
    materialize.vStride = model.world.nv;
    materialize.bodyStride = bodies;
    auto* packKernel =
        resources.kernel("mr_scene_query_pack_body_states");
    encoder.set_compute_pipeline_state(packKernel);
    encoder.set_bytes(materialize, 0);
    encoder.set_buffer(
        resources.buffer(kImmutableBodies),
        1
    );
    encoder.set_buffer(
        resources.buffer(resources.bodyToSceneBufferIndex),
        2
    );
    encoder.set_input_array(bodyPoses, 3);
    encoder.set_input_array(operatorStatuses, 4);
    encoder.set_input_array(inputs[2], 5);
    encoder.set_input_array(inputs[3], 6);
    encoder.set_input_array(inputs[4], 7);
    encoder.set_input_array(inputs[5], 8);
    encoder.set_output_array(outputs[0], 9);
    const std::size_t packWidth = std::min<std::size_t>(
        kWorldThreads,
        packKernel->maxTotalThreadsPerThreadgroup()
    );
    encoder.dispatch_threads(
        MTL::Size(
            static_cast<std::size_t>(environments) * bodies,
            1u,
            1u
        ),
        MTL::Size(packWidth, 1u, 1u)
    );
    encoder.barrier();

    auto* velocityKernel = resources.kernel(
        "mr_articulated_materialize_body_velocities"
    );
    encoder.set_compute_pipeline_state(velocityKernel);
    for (std::uint32_t owner = 0u;
         owner < articulations;
         ++owner) {
        const MRArticulationGPU& articulation =
            model.articulations[owner];
        MRArticulatedOperatorDispatchGPU dispatch{};
        dispatch.articulationIndex = owner;
        dispatch.environmentCount = environments;
        dispatch.pointCount = 0u;
        dispatch.flags =
            MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY;
        dispatch.qStride = model.world.nq;
        dispatch.pointStride = 1u;
        dispatch.bodyPoseStride = bodies;
        dispatch.pointWorldStride = 1u;
        dispatch.massMatrixStride = 1u;
        dispatch.pointJacobianStride = 1u;
        dispatch.generalizedStride = model.world.nv;
        encoder.set_buffer(
            resources.buffer(kImmutableWorld),
            0
        );
        encoder.set_buffer(
            resources.buffer(kImmutableArticulations),
            1
        );
        encoder.set_buffer(
            resources.buffer(kImmutableJoints),
            2
        );
        encoder.set_buffer(
            resources.buffer(kImmutableDofs),
            3
        );
        encoder.set_buffer(
            resources.buffer(kImmutableBodies),
            4
        );
        encoder.set_bytes(dispatch, 5);
        encoder.set_input_array(
            inputs[0],
            6,
            articulation.qOffset * sizeof(float)
        );
        encoder.set_input_array(
            inputs[1],
            7,
            articulation.vOffset * sizeof(float)
        );
        encoder.set_output_array(
            outputs[0],
            8,
            articulation.firstBody * sizeof(MRBodyStateGPU)
        );
        encoder.set_output_array(
            operatorStatuses,
            9,
            static_cast<std::int64_t>(owner) *
                environments *
                sizeof(MRArticulatedOperatorStatusGPU)
        );
        encoder.dispatch_threadgroups(
            MTL::Size(environments, 1u, 1u),
            MTL::Size(kOperatorThreads, 1u, 1u)
        );
    }
}

std::vector<mx::array> BodyStatePrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo body-state materialization does not implement JVP"
    );
}

std::vector<mx::array> BodyStatePrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo body-state materialization does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
BodyStatePrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo body states use the native environment batch axis"
    );
}

const char* BodyStatePrimitive::name() const {
    return "MetalRoboBodyState";
}

bool BodyStatePrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const BodyStatePrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get();
}

SceneRaycastPrimitive::SceneRaycastPrimitive(
    mx::Stream stream,
    std::shared_ptr<MLXCompiledWorld> world,
    const std::uint32_t environmentCount,
    const std::uint32_t rayCount,
    const bool mountedPattern
)
    : mx::Primitive(stream),
      world_(std::move(world)),
      environmentCount_(environmentCount),
      rayCount_(rayCount),
      mountedPattern_(mountedPattern) {}

void SceneRaycastPrimitive::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo scene raycast has no MLX CPU fallback"
    );
}

void SceneRaycastPrimitive::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    const std::size_t expectedInputs =
        mountedPattern_ ? 6u : 5u;
    if (inputs.size() != expectedInputs ||
        outputs.size() != 5u) {
        throw std::runtime_error(
            "MetalRobo scene raycast received an invalid graph"
        );
    }
    for (mx::array& output : outputs) {
        output.set_data(mx::allocator::malloc(output.nbytes()));
    }
    auto& streamValue = stream();
    auto& device = mx::metal::device(streamValue.device);
    MetalResources& resources = world_->resources(device);
    auto& encoder =
        mx::metal::get_command_encoder(streamValue);
    const EngineModel& model = world_->world().model();

    MRSceneQueryDispatchGPU dispatch{};
    dispatch.abiVersion = MR_SCENE_QUERY_ABI_VERSION;
    dispatch.environmentCount = environmentCount_;
    dispatch.rayCount = rayCount_;
    dispatch.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    dispatch.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    dispatch.geometryCount = static_cast<std::uint32_t>(
        model.geometryHeaders.size()
    );
    dispatch.vertexCount = static_cast<std::uint32_t>(
        model.geometryVertices.size()
    );
    dispatch.meshNodeCount = static_cast<std::uint32_t>(
        model.meshBvhNodes.size()
    );
    dispatch.meshTriangleCount =
        static_cast<std::uint32_t>(
            model.meshTriangles.size()
        );
    dispatch.convexFaceCount =
        static_cast<std::uint32_t>(
            model.convexFaces.size()
        );
    dispatch.rayStride = rayCount_;
    dispatch.bodyStride = dispatch.bodyCount;

    auto* kernel = resources.kernel(
        mountedPattern_
        ? "mr_scene_raycast_pattern"
        : "mr_scene_raycast"
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_bytes(dispatch, 0);
    encoder.set_buffer(
        resources.buffer(kImmutableShapes),
        1
    );
    encoder.set_buffer(
        resources.buffer(kImmutableGeometryHeaders),
        2
    );
    encoder.set_buffer(
        resources.buffer(kImmutableGeometryVertices),
        3
    );
    encoder.set_buffer(
        resources.buffer(kImmutableMeshNodes),
        4
    );
    encoder.set_buffer(
        resources.buffer(kImmutableMeshTriangles),
        5
    );
    encoder.set_buffer(
        resources.buffer(kImmutableConvexFaces),
        6
    );
    encoder.set_input_array(inputs[0], 7);
    if (mountedPattern_) {
        encoder.set_input_array(inputs[1], 8);
        encoder.set_input_array(inputs[2], 9);
        encoder.set_input_array(inputs[3], 10);
        encoder.set_input_array(inputs[4], 11);
        encoder.set_input_array(inputs[5], 12);
        encoder.set_output_array(outputs[0], 13);
        encoder.set_output_array(outputs[1], 14);
        encoder.set_output_array(outputs[2], 15);
        encoder.set_output_array(outputs[3], 16);
        encoder.set_output_array(outputs[4], 17);
    } else {
        encoder.set_input_array(inputs[1], 8);
        encoder.set_input_array(inputs[2], 9);
        encoder.set_input_array(inputs[3], 10);
        encoder.set_input_array(inputs[4], 11);
        encoder.set_output_array(outputs[0], 12);
        encoder.set_output_array(outputs[1], 13);
        encoder.set_output_array(outputs[2], 14);
        encoder.set_output_array(outputs[3], 15);
        encoder.set_output_array(outputs[4], 16);
    }
    const std::size_t width = std::min<std::size_t>(
        kWorldThreads,
        kernel->maxTotalThreadsPerThreadgroup()
    );
    encoder.dispatch_threads(
        MTL::Size(
            static_cast<std::size_t>(environmentCount_) *
                rayCount_,
            1u,
            1u
        ),
        MTL::Size(width, 1u, 1u)
    );
}

std::vector<mx::array> SceneRaycastPrimitive::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo scene raycast does not implement JVP"
    );
}

std::vector<mx::array> SceneRaycastPrimitive::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&
) {
    throw std::runtime_error(
        "MetalRobo scene raycast does not implement VJP"
    );
}

std::pair<std::vector<mx::array>, std::vector<int>>
SceneRaycastPrimitive::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&
) {
    throw std::runtime_error(
        "MetalRobo scene rays use the native environment batch axis"
    );
}

const char* SceneRaycastPrimitive::name() const {
    return "MetalRoboSceneRaycast";
}

bool SceneRaycastPrimitive::is_equivalent(
    const mx::Primitive& other
) const {
    const auto* typed =
        dynamic_cast<const SceneRaycastPrimitive*>(&other);
    return typed != nullptr &&
        typed->world_.get() == world_.get() &&
        typed->environmentCount_ == environmentCount_ &&
        typed->rayCount_ == rayCount_ &&
        typed->mountedPattern_ == mountedPattern_;
}

} // namespace metalrobo::mlx_ext
