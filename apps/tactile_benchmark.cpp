#include "metalrobo/MetalTactile.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint32_t environments = 256u;
    std::uint32_t width = 32u;
    std::uint32_t height = 32u;
    std::uint32_t sensors = 2u;
    std::uint32_t updatePeriod = 1u;
    std::uint32_t warmupIterations = 3u;
    std::uint32_t measuredIterations = 11u;
};

template <typename Result>
requires requires(const Result& value) {
    value.succeeded();
    value.message;
}
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + result.message
        );
    }
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint32_t number(const std::string_view value) {
    std::uint32_t result = 0u;
    const auto parsed = std::from_chars(
        value.data(),
        value.data() + value.size(),
        result
    );
    require(
        parsed.ec == std::errc{} &&
        parsed.ptr == value.data() + value.size() &&
        result > 0u,
        "benchmark option must be a positive integer"
    );
    return result;
}

Options options(const int argc, const char* const* argv) {
    Options result;
    require(
        (argc - 1) % 2 == 0,
        "usage: metalrobo_tactile_benchmark "
        "[--environments N] [--width N] [--height N] [--sensors N] "
        "[--update-period N] [--warmup N] [--iterations N]"
    );
    for (int argument = 1; argument < argc; argument += 2) {
        const std::string_view option{argv[argument]};
        const std::uint32_t value = number(argv[argument + 1]);
        if (option == "--environments") {
            result.environments = value;
        } else if (option == "--width") {
            result.width = value;
        } else if (option == "--height") {
            result.height = value;
        } else if (option == "--sensors") {
            result.sensors = value;
        } else if (option == "--update-period") {
            result.updatePeriod = value;
        } else if (option == "--warmup") {
            result.warmupIterations = value;
        } else if (option == "--iterations") {
            result.measuredIterations = value;
        } else {
            throw std::runtime_error(
                "unknown tactile benchmark option: " +
                std::string{option}
            );
        }
    }
    require(
        result.width <= 512u &&
        result.height <= 512u &&
        result.sensors <= 64u,
        "benchmark atlas or sensor count exceeds the bounded harness"
    );
    return result;
}

metalrobo::EngineModel makeModel() {
    metalrobo::EngineModel model;
    model.name = "tactile_benchmark_primitives";
    model.bodies.resize(2u);
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        model.bodies[body].articulationIndex = MR_INVALID_INDEX;
        model.bodies[body].parentBody = MR_INVALID_INDEX;
        model.bodies[body].inboundJoint = MR_INVALID_INDEX;
        model.bodies[body].motionType =
            body == 0u ? MR_MOTION_STATIC : MR_MOTION_KINEMATIC;
    }
    model.shapes.resize(2u);
    MRShapeGPU& backing = model.shapes[0u];
    backing.bodyIndex = 0u;
    backing.shapeType = MR_SHAPE_BOX;
    backing.collisionGroup = 1u;
    backing.collisionMask = 1u;
    backing.slotGeneration = 1u;
    backing.localPosition = {0.0f, 0.0f, -0.005f, 1.0f};
    backing.localRotation.w = 1.0f;
    backing.dimensions = {0.03f, 0.03f, 0.005f, 0.0f};
    backing.contactRestAndBoundingRadius = {
        0.004f,
        0.004f,
        0.05f,
        0.0f,
    };
    MRShapeGPU& target = model.shapes[1u];
    target.bodyIndex = 1u;
    target.shapeType = MR_SHAPE_SPHERE;
    target.collisionGroup = 1u;
    target.collisionMask = 1u;
    target.slotGeneration = 2u;
    target.localPosition.w = 1.0f;
    target.localRotation.w = 1.0f;
    target.dimensions = {0.012f, 0.0f, 0.0f, 0.0f};
    target.contactRestAndBoundingRadius = {
        0.001f,
        0.0f,
        0.013f,
        0.0f,
    };
    return model;
}

MRBodyStateGPU body(
    const std::uint32_t index,
    const mr_float4 position,
    const std::uint32_t motion
) {
    MRBodyStateGPU result{};
    result.position = position;
    result.orientation.w = 1.0f;
    result.flagsAndIndices[0] = motion;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = index;
    return result;
}

double percentile(
    const std::vector<double>& sorted,
    const double fraction
) {
    const double coordinate =
        fraction * static_cast<double>(sorted.size() - 1u);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(coordinate));
    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(coordinate));
    const double blend =
        coordinate - static_cast<double>(lower);
    return sorted[lower] * (1.0 - blend) +
        sorted[upper] * blend;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Options config = options(argc, argv);
        const metalrobo::EngineModel model = makeModel();
        std::vector<metalrobo::TactileSensorSpec> authored;
        authored.reserve(config.sensors);
        for (std::uint32_t sensor = 0u;
             sensor < config.sensors;
             ++sensor) {
            metalrobo::TactilePose pose;
            pose.position = {0.0f, 0.0f, 0.004f, 0.0f};
            auto specification =
                metalrobo::makeFlatTactileSensor(
                    "benchmark_" + std::to_string(sensor),
                    0u,
                    0u,
                    pose,
                    config.width,
                    config.height,
                    0.03f,
                    0.03f,
                    0.004f
                );
            specification.targetShapeIndices = {1u};
            specification.updatePeriodSteps =
                config.updatePeriod;
            authored.push_back(std::move(specification));
        }
        metalrobo::CookedTactileSystem tactile;
        require(
            metalrobo::cookTactileSystem(
                authored,
                model,
                tactile
            ),
            "cook benchmark tactile system"
        );
        std::vector<MRBodyStateGPU> bodies;
        bodies.reserve(
            static_cast<std::size_t>(config.environments) * 2u
        );
        for (std::uint32_t environment = 0u;
             environment < config.environments;
             ++environment) {
            bodies.push_back(body(
                0u,
                {0.0f, 0.0f, 0.0f, 1.0f},
                MR_MOTION_STATIC
            ));
            bodies.push_back(body(
                1u,
                {
                    0.0f,
                    0.0f,
                    0.004f + 0.012f - 0.002f,
                    1.0f,
                },
                MR_MOTION_KINEMATIC
            ));
        }

        metalrobo::MetalTactileConfig runtimeConfig;
        runtimeConfig.contactCapacityPerEnvironment = 1u;
        runtimeConfig.enableDebugHits = false;
        metalrobo::MetalTactileContext runtime(runtimeConfig);
        const auto compileStart =
            std::chrono::steady_clock::now();
        const auto compiled = runtime.compile(
            tactile,
            model,
            config.environments
        );
        const auto compileEnd =
            std::chrono::steady_clock::now();
        require(compiled, "compile benchmark Metal tactile context");
        const double compileMilliseconds =
            std::chrono::duration<double, std::milli>(
                compileEnd - compileStart
            ).count();

        metalrobo::MetalTactileHostFrame frame;
        frame.environmentCount = config.environments;
        frame.bodies = bodies;
        frame.observationTimestepSeconds = 1.0f / 60.0f;
        for (std::uint32_t iteration = 0u;
             iteration < config.warmupIterations;
             ++iteration) {
            frame.frameIndex = iteration;
            frame.timestampSeconds =
                iteration / 60.0;
            require(
                runtime.observe(frame),
                "warm up tactile benchmark"
            );
        }
        std::vector<double> samples;
        samples.reserve(config.measuredIterations);
        for (std::uint32_t iteration = 0u;
             iteration < config.measuredIterations;
             ++iteration) {
            frame.frameIndex =
                config.warmupIterations + iteration;
            frame.timestampSeconds =
                frame.frameIndex / 60.0;
            const auto measured = runtime.observe(frame);
            require(measured, "run tactile benchmark");
            samples.push_back(measured.elapsedMilliseconds);
        }
        std::ranges::sort(samples);
        const double medianMilliseconds =
            percentile(samples, 0.5);
        const double p95Milliseconds =
            percentile(samples, 0.95);
        metalrobo::TactileObservationBatch readback;
        const auto readbackStart =
            std::chrono::steady_clock::now();
        require(
            runtime.readback(config.environments, readback),
            "benchmark tactile diagnostic readback"
        );
        const auto readbackEnd =
            std::chrono::steady_clock::now();
        const double readbackMilliseconds =
            std::chrono::duration<double, std::milli>(
                readbackEnd - readbackStart
            ).count();
        const std::uint64_t samplesPerFrame =
            static_cast<std::uint64_t>(config.environments) *
            config.sensors *
            config.width *
            config.height;
        const double seconds =
            medianMilliseconds * 0.001;
        const double tactileFramesPerSecond =
            static_cast<double>(
                static_cast<std::uint64_t>(config.environments) *
                config.sensors
            ) / seconds;
        const double samplesPerSecond =
            static_cast<double>(samplesPerFrame) / seconds;
        const auto layout = runtime.layout();

        std::cout
            << std::setprecision(9)
            << "{\"schema\":\"metalrobo.tactile_benchmark\","
            << "\"device\":\"" << compiled.deviceName << "\","
            << "\"backend\":"
            << static_cast<std::uint32_t>(layout.queryBackend)
            << ",\"hardware_ray_queries_available\":"
            << (
                layout.hardwareRayQueriesAvailable
                ? "true"
                : "false"
            )
            << ",\"environments\":" << config.environments
            << ",\"sensors_per_environment\":"
            << config.sensors
            << ",\"width\":" << config.width
            << ",\"height\":" << config.height
            << ",\"update_period_steps\":"
            << config.updatePeriod
            << ",\"compile_ms\":" << compileMilliseconds
            << ",\"median_observe_ms\":"
            << medianMilliseconds
            << ",\"p95_observe_ms\":" << p95Milliseconds
            << ",\"min_observe_ms\":" << samples.front()
            << ",\"max_observe_ms\":" << samples.back()
            << ",\"tactile_frames_per_second\":"
            << tactileFramesPerSecond
            << ",\"samples_per_second\":"
            << samplesPerSecond
            << ",\"retained_bytes\":" << layout.retainedBytes
            << ",\"bytes_per_environment\":"
            << layout.bytesPerEnvironment
            << ",\"native_publication_zero_copy\":true"
            << ",\"diagnostic_readback_ms\":"
            << readbackMilliseconds
            << ",\"iterations\":"
            << config.measuredIterations
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_tactile_benchmark: "
                  << error.what() << '\n';
        return 1;
    }
}
