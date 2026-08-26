#import <Metal/Metal.h>

#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPlatform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Result>
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + result.message
        );
    }
}

metalrobo::WorldPose cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const float x = target.x - position.x;
    const float y = target.y - position.y;
    const float z = target.z - position.z;
    const float inverseLength = 1.0f / std::sqrt(x * x + y * y + z * z);
    const mr_float4 forward{x * inverseLength, y * inverseLength, z * inverseLength, 0.0f};
    // Renderer camera +z is forward and screen-up is local -y.  Preserve a
    // world-z horizon rather than accepting the shortest-arc camera rotation,
    // whose roll changes when an inspection camera crosses an axis.
    const mr_float4 preferredUp{0.0f, 0.0f, 1.0f, 0.0f};
    mr_float4 localY{-preferredUp.x, -preferredUp.y, -preferredUp.z, 0.0f};
    mr_float4 localX{
        localY.y * forward.z - localY.z * forward.y,
        localY.z * forward.x - localY.x * forward.z,
        localY.x * forward.y - localY.y * forward.x,
        0.0f,
    };
    const float localXLength = std::sqrt(
        localX.x * localX.x + localX.y * localX.y + localX.z * localX.z
    );
    if (localXLength < 1.0e-5f) {
        throw std::runtime_error("BodyParts3D inspection camera is parallel to world up");
    }
    localX.x /= localXLength;
    localX.y /= localXLength;
    localX.z /= localXLength;
    localY = {
        forward.y * localX.z - forward.z * localX.y,
        forward.z * localX.x - forward.x * localX.z,
        forward.x * localX.y - forward.y * localX.x,
        0.0f,
    };
    const float m00 = localX.x;
    const float m01 = localY.x;
    const float m02 = forward.x;
    const float m10 = localX.y;
    const float m11 = localY.y;
    const float m12 = forward.y;
    const float m20 = localX.z;
    const float m21 = localY.z;
    const float m22 = forward.z;
    mr_float4 orientation{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        orientation = {(m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
        orientation = {0.25f * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
        orientation = {(m01 + m10) / scale, 0.25f * scale, (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
        orientation = {(m02 + m20) / scale, (m12 + m21) / scale, 0.25f * scale, (m10 - m01) / scale};
    }
    return {position, orientation};
}

std::pair<mr_float4, float> packCentreAndDistance(
    const metalrobo::VisualAssetPackV2& pack
) {
    if (pack.vertices.empty()) {
        throw std::runtime_error("BodyParts3D visual pack has no vertices");
    }
    mr_float4 minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0.0f,
    };
    mr_float4 maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
    };
    for (const MRVisualVertexGPUV2& vertex : pack.vertices) {
        minimum.x = std::min(minimum.x, vertex.position.x);
        minimum.y = std::min(minimum.y, vertex.position.y);
        minimum.z = std::min(minimum.z, vertex.position.z);
        maximum.x = std::max(maximum.x, vertex.position.x);
        maximum.y = std::max(maximum.y, vertex.position.y);
        maximum.z = std::max(maximum.z, vertex.position.z);
    }
    const mr_float4 centre{
        0.5f * (minimum.x + maximum.x),
        0.5f * (minimum.y + maximum.y),
        0.5f * (minimum.z + maximum.z),
        0.0f,
    };
    const float extent = std::max({
        maximum.x - minimum.x,
        maximum.y - minimum.y,
        maximum.z - minimum.z,
    });
    return {centre, std::max(1.35f * extent, 2.25f)};
}

metalrobo::SensorSpec makeCamera(
    const std::string& id,
    const mr_float4 position,
    const mr_float4 target
) {
    constexpr std::uint32_t dimension = 512u;
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "workspace";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ASSET;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = dimension;
    camera.height = dimension;
    camera.intrinsics = {360.0f, 360.0f, 256.0f, 256.0f};
    camera.maximumDepthMeters = 12.0f;
    return camera;
}

metalrobo::VisualMotionSampleBatchV1 staticMotion(
    const std::uint32_t bodyCount,
    const std::uint64_t frameIndex
) {
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = 1u;
    motion.bodyCount = bodyCount;
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 120.0;
    motion.timestampsSeconds = {motion.exposureOpenSeconds, motion.exposureCloseSeconds};
    motion.bodyStates.resize(2u * bodyCount);
    for (MRBodyStateGPU& body : motion.bodyStates) {
        body.orientation.w = 1.0f;
    }
    motion.scenarioIdentity = 8601u;
    motion.sensorIdentity = 8601u;
    motion.frameIndex = frameIndex;
    motion.sensorSequence = static_cast<std::uint32_t>(frameIndex);
    return motion;
}

std::uint8_t toSrgb(const float value) {
    const float linear = std::clamp(value, 0.0f, 1.0f);
    const float encoded = linear <= 0.0031308f
        ? 12.92f * linear
        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::lround(255.0f * std::clamp(encoded, 0.0f, 1.0f)));
}

void writePpm(
    const std::filesystem::path& path,
    const metalrobo::HybridObservationBatch& observation
) {
    const std::size_t pixels = static_cast<std::size_t>(observation.width) * observation.height;
    if (observation.environmentCount != 1u || observation.rgb.size() != pixels) {
        throw std::runtime_error("BodyParts3D preview RGB readback has an unexpected layout");
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not create visual preview frame: " + path.string());
    }
    stream << "P6\n" << observation.width << " " << observation.height << "\n255\n";
    for (const mr_float4 pixel : observation.rgb) {
        const std::array<std::uint8_t, 3u> encoded{
            toSrgb(pixel.x),
            toSrgb(pixel.y),
            toSrgb(pixel.z),
        };
        stream.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    }
    if (!stream) {
        throw std::runtime_error("could not write visual preview frame: " + path.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 3) {
                std::cerr << "usage: metalrobo_bodyparts3d_visual_probe PACK.mrvpack OUTPUT_DIRECTORY\n";
                return 2;
            }
            const std::filesystem::path packPath{argv[1]};
            const std::filesystem::path outputDirectory{argv[2]};
            metalrobo::VisualAssetPackV2 pack;
            std::string reason;
            if (!metalrobo::readVisualAssetPack(packPath, pack, &reason)) {
                throw std::runtime_error("could not read visual pack: " + reason);
            }
            const auto [centre, distance] = packCentreAndDistance(pack);
            const std::array cameraDefinitions{
                std::pair{"axis_negative_y", mr_float4{centre.x, centre.y - distance, centre.z + 0.08f, 0.0f}},
                std::pair{"oblique_positive_x_negative_y", mr_float4{centre.x + 0.82f * distance, centre.y - 0.82f * distance, centre.z + 0.24f * distance, 0.0f}},
                std::pair{"axis_positive_y", mr_float4{centre.x, centre.y + distance, centre.z + 0.08f, 0.0f}},
            };

            const metalrobo::EngineModel model = metalrobo::makeFrankaPickPlaceEngineModel();
            metalrobo::EpisodeTwin episode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
            episode.id = "bodyparts3d_source_static_preview_v1";
            episode.sensors.clear();
            for (const auto& [id, position] : cameraDefinitions) {
                episode.sensors.push_back(makeCamera(id, position, centre));
            }
            metalrobo::WorldTemplate world;
            require(metalrobo::compileEpisodeTwin(episode, model, world), "preview episode compile");
            metalrobo::WorldProgram program;
            program.id = "bodyparts3d_source_static_preview_world_v1";
            metalrobo::WorldFamily family;
            require(metalrobo::compileWorldFamily(world, program, family), "preview world compile");
            metalrobo::MetalWorldFamilyContext worlds;
            require(worlds.compile(family, 1u), "preview world device compile");
            require(worlds.sample(1u, 8601u), "preview world sample");

            const std::array assets{
                metalrobo::VisualAssetReferenceV3{packPath, pack.contentHash, 0u, 8601u, 1u},
            };
            metalrobo::VisualSceneManifestV3 manifest;
            if (!metalrobo::compileVisualSceneManifestV3(
                    world,
                    assets,
                    metalrobo::makeNeutralStudioEnvironmentV2(),
                    metalrobo::makeIndoorAreaLightRigV1(),
                    manifest,
                    &reason
                )) {
                throw std::runtime_error("preview scene compile: " + reason);
            }
            metalrobo::MetalHybridRendererConfig configuration;
            configuration.width = 512u;
            configuration.height = 512u;
            configuration.maximumReferenceFramesInFlight = 1u;
            metalrobo::MetalHybridRenderer renderer(configuration);
            require(
                renderer.compile(
                    std::move(manifest.renderScene),
                    metalrobo::VisualRendererProfileV1::sensorReference(),
                    1u
                ),
                "preview renderer compile"
            );
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (device == nil || queue == nil) {
                throw std::runtime_error("could not create an Apple Metal command queue");
            }
            std::filesystem::create_directories(outputDirectory);
            for (std::uint32_t camera = 0u; camera < cameraDefinitions.size(); ++camera) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                metalrobo::MetalHybridFrameCommandContext context;
                context.commandBuffer = (__bridge void*)command;
                require(
                    renderer.encodeFrame(
                        worlds,
                        staticMotion(static_cast<std::uint32_t>(model.bodies.size()), camera + 1u),
                        camera,
                        context
                    ),
                    "preview frame encode"
                );
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) {
                    throw std::runtime_error(
                        "preview command failed: " +
                        std::string{command.error.localizedDescription.UTF8String}
                    );
                }
                metalrobo::HybridObservationBatch observation;
                require(renderer.readback(observation), "preview frame readback");
                const std::size_t covered = static_cast<std::size_t>(std::count(
                    observation.segmentation.begin(), observation.segmentation.end(), 8601u
                ));
                if (covered == 0u) {
                    throw std::runtime_error("preview frame contains no BodyParts3D source pixels");
                }
                const std::filesystem::path frame = outputDirectory /
                    (std::string{cameraDefinitions[camera].first} + ".ppm");
                writePpm(frame, observation);
                std::cout << "view=" << cameraDefinitions[camera].first
                          << " covered_pixels=" << covered
                          << " total_pixels=" << observation.segmentation.size()
                          << " frame=" << frame.string() << '\n';
            }
            std::cout << "device=\"" << device.name.UTF8String << "\""
                      << " source_pack=" << pack.contentHash
                      << " vertices=" << pack.vertices.size()
                      << " triangles=" << pack.indices.size() / 3u
                      << " static_source_preview=ok\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "metalrobo_bodyparts3d_visual_probe: " << error.what() << '\n';
            return 1;
        }
    }
}
