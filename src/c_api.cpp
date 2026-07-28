#include "metalrobo/c_api.h"

#include "metalrobo/Model.hpp"
#include "metalrobo/Runtime.hpp"

#include <exception>
#include <memory>
#include <span>
#include <string>

struct MRRuntimeHandle {
    std::unique_ptr<metalrobo::Runtime> runtime;
    std::string deviceName;
};

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

} // namespace

extern "C" {

const char* mr_version(void) {
    return "0.4.0";
}

const char* mr_last_error(void) {
    return gLastError.c_str();
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

} // extern "C"
