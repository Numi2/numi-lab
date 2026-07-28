#include "metalrobo_mlx.h"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/G1.hpp"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
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

} // namespace

struct MetalResources {
    mx::metal::Device* device = nullptr;
    std::vector<mx::allocator::Buffer> buffers;
    MTL::ComputePipelineState* abaKernel = nullptr;
    MTL::ComputePipelineState* commitKernel = nullptr;

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
};

MLXCompiledWorld::MLXCompiledWorld(
    CompiledWorld world,
    const float controlTimestep,
    const std::uint32_t physicsSubsteps,
    const bool applyBodyDamping,
    std::string metallibPath
)
    : world_(std::move(world)),
      controlTimestep_(controlTimestep),
      physicsSubsteps_(physicsSubsteps),
      applyBodyDamping_(applyBodyDamping),
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

    auto staged = std::make_unique<MetalResources>();
    staged->device = &device;
    staged->buffers.reserve(6u);
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
    auto* adapterLibrary =
        device.get_library(
            "MetalRoboMLX",
            currentBinaryDirectory()
        );
    staged->commitKernel = device.get_kernel(
        "mr_mlx_world_commit_aba",
        adapterLibrary
    );
    if (staged->abaKernel == nullptr ||
        staged->commitKernel == nullptr) {
        throw std::runtime_error(
            "MLXCompiledWorld could not create Metal pipelines"
        );
    }
    resources_ = std::move(staged);
    return *resources_;
}

std::shared_ptr<MLXCompiledWorld> compileWorld(
    const std::string& modelName,
    const float controlTimestep,
    const std::uint32_t physicsSubsteps,
    const bool applyBodyDamping,
    const std::string& requestedMetallibPath
) {
    if (!std::isfinite(controlTimestep) ||
        !(controlTimestep > 0.0f) ||
        physicsSubsteps == 0u ||
        physicsSubsteps > 128u) {
        throw std::invalid_argument(
            "control_timestep must be finite and positive and "
            "physics_substeps must be in [1, 128]"
        );
    }
    EngineModel model;
    if (modelName == "franka") {
        model = makeFrankaPandaEngineModel();
    } else if (modelName == "g1") {
        model = makeUnitreeG1EngineModel();
    } else {
        throw std::invalid_argument(
            "model must be 'franka' or 'g1'"
        );
    }
    CompiledWorld compiled;
    const auto diagnostics =
        compileMetalWorld(model, 0u, compiled);
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            "could not compile MLX world: " +
            diagnostics.message
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
    return std::make_shared<MLXCompiledWorld>(
        std::move(compiled),
        controlTimestep,
        physicsSubsteps,
        applyBodyDamping,
        std::move(metallibPath)
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

} // namespace metalrobo::mlx_ext
