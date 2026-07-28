#include "metalrobo/c_api.h"

#include "metalrobo/EpisodeTwinCompiler.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/Model.hpp"
#include "metalrobo/Runtime.hpp"
#include "metalrobo/WorldPack.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

struct MRRuntimeHandle {
    std::unique_ptr<metalrobo::Runtime> runtime;
    std::string deviceName;
};

struct MRWorldFamilyHandle {
    metalrobo::MetalWorldFamilyContext context;
    metalrobo::WorldFamily family;
    metalrobo::WorldInstanceBatch readback;
    std::string deviceName;
    double lastSampleMilliseconds = 0.0;
};

struct MRHybridRendererHandle {
    metalrobo::MetalHybridRenderer renderer;
    metalrobo::HybridObservationBatch readback;
    std::string deviceName;
    std::uint32_t activeEnvironmentCount = 0u;
    double lastRenderMilliseconds = 0.0;
};

static_assert(sizeof(MRHybridGaussianC) == 80u);
static_assert(sizeof(MRHybridGaussianGPU) == 80u);

namespace {

thread_local std::string gLastError;

template <typename Function>
int translateErrors(Function&& function) noexcept {
    try {
        function();
        gLastError.clear();
        return 0;
    } catch (const std::exception& error) {
        gLastError = error.what();
    } catch (...) {
        gLastError = "MetalRobo failed with an unknown native exception.";
    }
    return -1;
}

bool requireHandle(const MRRuntimeHandle* handle) {
    if (handle != nullptr && handle->runtime != nullptr) {
        return true;
    }
    gLastError = "MetalRobo runtime handle is null.";
    return false;
}

bool requireWorldFamilyHandle(const MRWorldFamilyHandle* handle) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo world-family handle is null.";
    return false;
}

bool requireHybridRendererHandle(
    const MRHybridRendererHandle* handle
) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo hybrid renderer handle is null.";
    return false;
}

std::runtime_error worldFamilyError(
    const char* operation,
    const metalrobo::MetalWorldFamilyDiagnostics& diagnostics
) {
    return std::runtime_error(
        std::string{operation} + " failed [" +
        metalrobo::metalWorldFamilyStatusName(diagnostics.status) +
        "]: " + diagnostics.message
    );
}

} // namespace

extern "C" {

const char* mr_version(void) {
    return "0.4.0";
}

const char* mr_last_error(void) {
    return gLastError.c_str();
}

int mr_compile_episode_manifest(
    const char* manifest_path,
    const char* output_pack_path,
    const char* artifact_store_path
) {
    if (manifest_path == nullptr || manifest_path[0] == '\0' ||
        output_pack_path == nullptr || output_pack_path[0] == '\0') {
        gLastError =
            "manifest_path and output_pack_path must be nonempty.";
        return -1;
    }
    return translateErrors([&] {
        metalrobo::CaptureManifest manifest;
        const metalrobo::EpisodeTwinCompilerResult loaded =
            metalrobo::loadCaptureManifestJSON(
                manifest_path,
                manifest
            );
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"capture manifest load failed ["} +
                metalrobo::episodeTwinCompilerStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        if (manifest.engineModelId != "franka_pick_place" ||
            manifest.worldProgramId != "franka_pick_place") {
            throw std::runtime_error(
                "capture manifest references an unregistered engine "
                "model or world program"
            );
        }

        const std::filesystem::path outputPath{output_pack_path};
        metalrobo::EpisodeTwinCompilerConfig config;
        config.artifactStore =
            artifact_store_path != nullptr &&
                artifact_store_path[0] != '\0'
            ? std::filesystem::path{artifact_store_path}
            : outputPath.parent_path() /
                (outputPath.stem().string() + ".artifacts");
        metalrobo::EpisodeTwinCompiler compiler{std::move(config)};
        metalrobo::CompiledEpisodeTwin compiled;
        const metalrobo::EpisodeTwinCompilerResult result =
            compiler.compile(
                manifest,
                metalrobo::makeFrankaPickPlaceEngineModel(),
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                compiled
            );
        if (!result.succeeded()) {
            throw std::runtime_error(
                std::string{"episode twin compilation failed ["} +
                metalrobo::episodeTwinCompilerStatusName(result.status) +
                "]: " + result.message
            );
        }
        const metalrobo::WorldPackResult written =
            metalrobo::writeWorldPack(
                compiled.worldPack,
                outputPath
            );
        if (!written.succeeded()) {
            throw std::runtime_error(
                std::string{"world-pack write failed ["} +
                metalrobo::worldPackStatusName(written.status) +
                "]: " + written.message
            );
        }
    });
}

MRRuntimeHandle* mr_create_franka(
    const uint32_t environment_count,
    const uint64_t seed,
    const char* metallib_path
) {
    if (environment_count == 0) {
        gLastError = "environment_count must be greater than zero.";
        return nullptr;
    }

    MRRuntimeHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::RuntimeDescriptor descriptor;
        descriptor.environmentCount = environment_count;
        descriptor.seed = seed;
        descriptor.autoReset = true;
        descriptor.captureBodyPoses = true;
        if (metallib_path != nullptr) {
            descriptor.metallibPath = metallib_path;
        }

        auto handle = std::make_unique<MRRuntimeHandle>();
        handle->runtime = metalrobo::makeMetalRuntime(
            metalrobo::makeFrankaPandaModel(),
            descriptor
        );
        handle->deviceName = handle->runtime->deviceName();
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_destroy(MRRuntimeHandle* handle) {
    delete handle;
}

int mr_reset(MRRuntimeHandle* handle, const uint64_t seed) {
    if (!requireHandle(handle)) {
        return -1;
    }
    return translateErrors([&] { handle->runtime->reset(seed); });
}

int mr_step(
    MRRuntimeHandle* handle,
    const float* normalized_actions,
    const size_t action_count
) {
    if (!requireHandle(handle)) {
        return -1;
    }
    const std::size_t required =
        static_cast<std::size_t>(handle->runtime->environmentCount()) *
        handle->runtime->model().gpu.actionCount;
    if (normalized_actions == nullptr) {
        gLastError = "normalized_actions is null.";
        return -1;
    }
    if (action_count != required) {
        gLastError =
            "action_count does not match environment_count * action_count.";
        return -1;
    }
    return translateErrors([&] {
        handle->runtime->step(
            std::span<const float>(normalized_actions, action_count)
        );
    });
}

uint32_t mr_environment_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->environmentCount() : 0;
}

uint32_t mr_action_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->model().gpu.actionCount : 0;
}

uint32_t mr_observation_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle)
        ? handle->runtime->model().gpu.observationCount
        : 0;
}

uint32_t mr_link_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->model().gpu.linkCount : 0;
}

const float* mr_observations(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->observations().data();
}

const float* mr_rewards(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->rewards().data();
}

const uint8_t* mr_terminated(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->terminated().data();
}

const float* mr_body_positions(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->bodyPositions().data();
}

const float* mr_body_rotations(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->bodyRotations().data();
}

MRRuntimeStatsC mr_stats(const MRRuntimeHandle* handle) {
    MRRuntimeStatsC result{};
    if (!requireHandle(handle)) {
        return result;
    }
    const metalrobo::RuntimeStats stats = handle->runtime->stats();
    result.last_gpu_milliseconds = stats.lastGpuMilliseconds;
    result.total_gpu_milliseconds = stats.totalGpuMilliseconds;
    result.control_steps = stats.controlSteps;
    result.physics_steps = stats.physicsSteps;
    return result;
}

const char* mr_device_name(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

MRWorldFamilyHandle* mr_create_franka_pick_place_world_family(
    const uint32_t capacity,
    const char* metallib_path
) {
    if (capacity == 0u) {
        gLastError = "world-family capacity must be greater than zero.";
        return nullptr;
    }

    MRWorldFamilyHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::WorldTemplate worldTemplate;
        const metalrobo::WorldCompileResult twin =
            metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                metalrobo::makeFrankaPickPlaceEngineModel(),
                worldTemplate
            );
        if (!twin.succeeded()) {
            throw std::runtime_error(
                "Franka episode compilation failed: " + twin.message
            );
        }

        metalrobo::WorldFamily family;
        const metalrobo::WorldCompileResult compiled =
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            );
        if (!compiled.succeeded()) {
            throw std::runtime_error(
                "Franka world-family compilation failed: " +
                compiled.message
            );
        }

        metalrobo::MetalWorldFamilyConfig config;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRWorldFamilyHandle>();
        handle->context = metalrobo::MetalWorldFamilyContext{
            std::move(config)
        };
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.compile(family, capacity);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family compile", diagnostics);
        }
        handle->family = std::move(family);
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

MRWorldFamilyHandle* mr_load_world_family_pack(
    const char* pack_path,
    const uint32_t capacity,
    const char* metallib_path
) {
    if (pack_path == nullptr || pack_path[0] == '\0') {
        gLastError = "world-pack path must be nonempty.";
        return nullptr;
    }
    if (capacity == 0u) {
        gLastError = "world-family capacity must be greater than zero.";
        return nullptr;
    }

    MRWorldFamilyHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::MRWorldPack pack;
        const metalrobo::WorldPackResult loaded =
            metalrobo::readWorldPack(pack_path, pack);
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"world-pack load failed ["} +
                metalrobo::worldPackStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        metalrobo::MetalWorldFamilyConfig config;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRWorldFamilyHandle>();
        handle->context = metalrobo::MetalWorldFamilyContext{
            std::move(config)
        };
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.compile(pack, capacity);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError(
                "packed world-family compile",
                diagnostics
            );
        }
        handle->family = std::move(pack.family);
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_world_family_destroy(MRWorldFamilyHandle* handle) {
    delete handle;
}

int mr_world_family_sample(
    MRWorldFamilyHandle* handle,
    const uint32_t instance_count,
    const uint64_t seed
) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.sample(instance_count, seed);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family sample", diagnostics);
        }
        handle->lastSampleMilliseconds =
            diagnostics.elapsedMilliseconds;
        handle->readback = {};
    });
}

int mr_world_family_readback(MRWorldFamilyHandle* handle) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::WorldInstanceBatch staged;
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.readback(staged);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family readback", diagnostics);
        }
        handle->readback = std::move(staged);
    });
}

MRWorldFamilyLayoutC mr_world_family_layout(
    const MRWorldFamilyHandle* handle
) {
    MRWorldFamilyLayoutC result{};
    if (!requireWorldFamilyHandle(handle)) {
        return result;
    }
    const metalrobo::MetalWorldFamilyLayout layout =
        handle->context.layout();
    result.capacity = layout.capacity;
    result.active_instance_count = layout.activeInstanceCount;
    result.asset_count_per_instance =
        layout.assetCountPerInstance;
    result.sensor_count_per_instance =
        layout.sensorCountPerInstance;
    result.appearance_count_per_instance =
        layout.appearanceCountPerInstance;
    result.variation_count = layout.variationCount;
    result.categorical_value_count =
        layout.categoricalValueCount;
    result.asset_binding_count = layout.assetBindingCount;
    result.binding_index_count = layout.bindingIndexCount;
    result.primary_articulation_index =
        layout.primaryArticulationIndex;
    result.nq = layout.nq;
    result.nv = layout.nv;
    result.body_count = layout.bodyCount;
    result.scene_body_count = layout.sceneBodyCount;
    result.articulation_count = layout.articulationCount;
    result.retained_private_bytes = layout.totalPrivateBytes();
    return result;
}

MRWorldFamilyStatsC mr_world_family_stats(
    const MRWorldFamilyHandle* handle
) {
    MRWorldFamilyStatsC result{};
    if (!requireWorldFamilyHandle(handle)) {
        return result;
    }
    const metalrobo::MetalWorldFamilyStats stats =
        handle->context.stats();
    result.compile_count = stats.compileCount;
    result.sample_count = stats.sampleCount;
    result.readback_count = stats.readbackCount;
    result.last_sample_milliseconds =
        handle->lastSampleMilliseconds;
    return result;
}

const char* mr_world_family_device_name(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

void* mr_world_family_native_buffer(
    const MRWorldFamilyHandle* handle,
    const uint32_t buffer_kind
) {
    if (!requireWorldFamilyHandle(handle) || buffer_kind > 10u) {
        if (buffer_kind > 10u) {
            gLastError = "world-family buffer kind is invalid.";
        }
        return nullptr;
    }
    return handle->context.nativeBuffer(
        static_cast<metalrobo::MetalWorldFamilyBuffer>(buffer_kind)
    );
}

const MRWorldInstanceHeaderGPU* mr_world_family_instance_headers(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.instances.empty()) {
        return nullptr;
    }
    return handle->readback.instances.data();
}

const MRWorldAssetInstanceGPU* mr_world_family_asset_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.assets.empty()) {
        return nullptr;
    }
    return handle->readback.assets.data();
}

const MRWorldSensorInstanceGPU* mr_world_family_sensor_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.sensors.empty()) {
        return nullptr;
    }
    return handle->readback.sensors.data();
}

const MRWorldAppearanceInstanceGPU*
mr_world_family_appearance_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.appearances.empty()) {
        return nullptr;
    }
    return handle->readback.appearances.data();
}

MRHybridRendererHandle* mr_hybrid_renderer_create(
    const MRHybridGaussianC* gaussians,
    const size_t gaussian_count,
    const uint32_t asset_count,
    const uint32_t capacity,
    const uint32_t width,
    const uint32_t height,
    const char* metallib_path
) {
    if (gaussians == nullptr || gaussian_count == 0u ||
        gaussian_count >
            std::numeric_limits<std::uint32_t>::max() ||
        asset_count == 0u || capacity == 0u ||
        width == 0u || height == 0u) {
        gLastError =
            "hybrid renderer scene, dimensions, and capacity "
            "must be nonempty.";
        return nullptr;
    }
    MRHybridRendererHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::HybridGaussianScene scene;
        scene.id = "c_api_hybrid_scene";
        scene.assetCount = asset_count;
        scene.gaussians.resize(gaussian_count);
        for (std::size_t index = 0u;
             index < gaussian_count;
             ++index) {
            const MRHybridGaussianC& source = gaussians[index];
            MRHybridGaussianGPU& destination =
                scene.gaussians[index];
            std::memcpy(
                &destination.meanAndOpacity,
                source.mean_and_opacity,
                sizeof(source.mean_and_opacity)
            );
            std::memcpy(
                &destination.scaleAndImportance,
                source.scale_and_importance,
                sizeof(source.scale_and_importance)
            );
            std::memcpy(
                &destination.orientation,
                source.orientation,
                sizeof(source.orientation)
            );
            std::memcpy(
                &destination.colorAndEmission,
                source.color_and_emission,
                sizeof(source.color_and_emission)
            );
            std::memcpy(
                &destination.binding,
                source.binding,
                sizeof(source.binding)
            );
        }
        metalrobo::MetalHybridRendererConfig config;
        config.width = width;
        config.height = height;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRHybridRendererHandle>();
        handle->renderer = metalrobo::MetalHybridRenderer{
            std::move(config)
        };
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.compile(scene, capacity);
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid renderer compile failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_hybrid_renderer_destroy(MRHybridRendererHandle* handle) {
    delete handle;
}

int mr_hybrid_renderer_render(
    MRHybridRendererHandle* handle,
    const MRWorldFamilyHandle* worlds,
    const uint32_t environment_count,
    const uint32_t camera_index
) {
    if (!requireHybridRendererHandle(handle) ||
        !requireWorldFamilyHandle(worlds)) {
        return -1;
    }
    return translateErrors([&] {
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.render(
                worlds->context,
                environment_count,
                camera_index
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid render failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->activeEnvironmentCount = environment_count;
        handle->lastRenderMilliseconds =
            diagnostics.elapsedMilliseconds;
    });
}

int mr_hybrid_renderer_readback(
    MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::HybridObservationBatch candidate;
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.readback(candidate);
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid readback failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->readback = std::move(candidate);
    });
}

MRHybridRendererLayoutC mr_hybrid_renderer_layout(
    const MRHybridRendererHandle* handle
) {
    MRHybridRendererLayoutC result{};
    if (!requireHybridRendererHandle(handle)) {
        return result;
    }
    const metalrobo::MetalHybridRendererLayout layout =
        handle->renderer.layout();
    result.capacity = layout.capacity;
    result.active_environment_count =
        handle->activeEnvironmentCount;
    result.width = layout.width;
    result.height = layout.height;
    result.tile_count_x = layout.tileCountX;
    result.tile_count_y = layout.tileCountY;
    result.gaussian_count = layout.gaussianCount;
    result.maximum_gaussians_per_tile =
        layout.maximumGaussiansPerTile;
    result.retained_private_bytes = layout.retainedPrivateBytes;
    result.last_render_milliseconds =
        handle->lastRenderMilliseconds;
    return result;
}

const char* mr_hybrid_renderer_device_name(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

void* mr_hybrid_renderer_native_buffer(
    const MRHybridRendererHandle* handle,
    const uint32_t buffer_kind
) {
    if (!requireHybridRendererHandle(handle) ||
        buffer_kind >
            static_cast<std::uint32_t>(
                metalrobo::MetalHybridRendererBuffer::
                    tileOverflowCounts
            )) {
        if (handle != nullptr) {
            gLastError =
                "hybrid renderer buffer kind is invalid.";
        }
        return nullptr;
    }
    return handle->renderer.nativeBuffer(
        static_cast<metalrobo::MetalHybridRendererBuffer>(
            buffer_kind
        )
    );
}

const float* mr_hybrid_renderer_rgb(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.rgb.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(
        handle->readback.rgb.data()
    );
}

const float* mr_hybrid_renderer_depth(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.depth.empty()) {
        return nullptr;
    }
    return handle->readback.depth.data();
}

const uint32_t* mr_hybrid_renderer_segmentation(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.segmentation.empty()) {
        return nullptr;
    }
    return handle->readback.segmentation.data();
}

} // extern "C"
