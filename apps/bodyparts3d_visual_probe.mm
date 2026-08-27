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

struct PackFraming {
    mr_float4 centre{};
    mr_float4 minimum{};
    mr_float4 maximum{};
    float distance = 0.0f;
};

PackFraming packFraming(
    const metalrobo::VisualAssetPackV2& pack,
    const bool tightFrame,
    const bool focusLowerThird,
    const bool fillFrame
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
    const float verticalExtent = maximum.z - minimum.z;
    const float distance = focusLowerThird
        ? std::max(0.65f * verticalExtent, 0.30f)
        : (fillFrame
            ? std::max(0.95f * verticalExtent, 0.30f)
            : std::max(1.35f * extent, tightFrame ? 0.0f : 2.25f));
    return {centre, minimum, maximum, distance};
}

metalrobo::SensorSpec makeCamera(
    const std::string& id,
    const mr_float4 position,
    const mr_float4 target,
    const std::uint32_t dimension
) {
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "workspace";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ASSET;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = dimension;
    camera.height = dimension;
    const float halfDimension = 0.5f * static_cast<float>(dimension);
    const float focalLength = 0.703125f * static_cast<float>(dimension);
    camera.intrinsics = {focalLength, focalLength, halfDimension, halfDimension};
    camera.maximumDepthMeters = 12.0f;
    return camera;
}

metalrobo::VisualLightRigV1 makeAnatomyStudioLightRig() {
    metalrobo::VisualLightRigV1 rig = metalrobo::makeIndoorAreaLightRigV1();
    rig.id = "anatomy_studio_three_point";
    rig.contentHash = "builtin:anatomy-studio-three-point-v1";
    MRVisualLightGPUV1 key = rig.lights.front();
    key.positionAndRange = {0.38f, -0.72f, 1.35f, 12.0f};
    key.directionAndSpot = {-0.16f, 0.34f, -0.93f, -1.0f};
    key.colorAndIntensity = {1.0f, 0.94f, 0.86f, 900.0f};
    key.shape = {0.75f, 0.55f, -1.0f, 0.08f};
    key.identity.w = 1u;
    MRVisualLightGPUV1 fill = key;
    fill.positionAndRange = {-0.64f, -0.34f, 0.84f, 12.0f};
    fill.directionAndSpot = {0.56f, 0.28f, -0.78f, -1.0f};
    fill.colorAndIntensity = {0.62f, 0.73f, 1.0f, 260.0f};
    fill.shape = {0.65f, 0.55f, -1.0f, 0.08f};
    fill.identity.w = 2u;
    fill.shadow = {0u, 0u, 4u, 0u};
    MRVisualLightGPUV1 rim = key;
    rim.positionAndRange = {-0.05f, 0.72f, 1.18f, 12.0f};
    rim.directionAndSpot = {0.04f, -0.51f, -0.86f, -1.0f};
    rim.colorAndIntensity = {0.85f, 0.92f, 1.0f, 480.0f};
    rim.shape = {0.55f, 0.40f, -1.0f, 0.08f};
    rim.identity.w = 3u;
    rig.lights = {key, fill, rim};
    return rig;
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
            if (argc < 3) {
                std::cerr << "usage: metalrobo_bodyparts3d_visual_probe PACK.mrvpack OUTPUT_DIRECTORY [--dimension <128..4096>] [--tight-frame] [--fill-frame] [--focus-lower-third]\n";
                return 2;
            }
            std::uint32_t dimension = 512u;
            bool tightFrame = false;
            bool focusLowerThird = false;
            bool fillFrame = false;
            for (int argument = 3; argument < argc; ++argument) {
                const std::string option{argv[argument]};
                if (option == "--tight-frame") {
                    if (tightFrame) {
                        throw std::runtime_error("--tight-frame may only be provided once");
                    }
                    tightFrame = true;
                    continue;
                }
                if (option == "--focus-lower-third") {
                    if (focusLowerThird || fillFrame) {
                        throw std::runtime_error("--focus-lower-third may only be provided once and cannot be combined with --fill-frame");
                    }
                    focusLowerThird = true;
                    continue;
                }
                if (option == "--fill-frame") {
                    if (fillFrame || focusLowerThird) {
                        throw std::runtime_error("--fill-frame may only be provided once and cannot be combined with --focus-lower-third");
                    }
                    fillFrame = true;
                    continue;
                }
                if (option != "--dimension" || ++argument >= argc) {
                    throw std::runtime_error("expected --dimension <128..4096>, --tight-frame, --fill-frame, or --focus-lower-third");
                }
                const std::string dimensionText{argv[argument]};
                std::size_t parsed = 0u;
                const unsigned long candidate = std::stoul(dimensionText, &parsed);
                if (parsed != dimensionText.size() || candidate < 128u || candidate > 4096u) {
                    throw std::runtime_error("visual preview dimension must be an integer in [128, 4096]");
                }
                dimension = static_cast<std::uint32_t>(candidate);
            }
            const std::filesystem::path packPath{argv[1]};
            const std::filesystem::path outputDirectory{argv[2]};
            metalrobo::VisualAssetPackV2 pack;
            std::string reason;
            if (!metalrobo::readVisualAssetPack(packPath, pack, &reason)) {
                throw std::runtime_error("could not read visual pack: " + reason);
            }
            const PackFraming framing = packFraming(pack, tightFrame, focusLowerThird, fillFrame);
            mr_float4 target = framing.centre;
            if (focusLowerThird) {
                target.z = framing.minimum.z + 0.32f * (framing.maximum.z - framing.minimum.z);
            }
            const std::array cameraDefinitions{
                std::pair{"axis_negative_y", mr_float4{target.x, target.y - framing.distance, target.z + 0.08f, 0.0f}},
                std::pair{"oblique_positive_x_negative_y", mr_float4{target.x + 0.82f * framing.distance, target.y - 0.82f * framing.distance, target.z + 0.24f * framing.distance, 0.0f}},
                std::pair{"axis_positive_y", mr_float4{target.x, target.y + framing.distance, target.z + 0.08f, 0.0f}},
            };

            const metalrobo::EngineModel model = metalrobo::makeFrankaPickPlaceEngineModel();
            metalrobo::EpisodeTwin episode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
            episode.id = "bodyparts3d_source_static_preview_v1";
            episode.sensors.clear();
            for (const auto& [id, position] : cameraDefinitions) {
                episode.sensors.push_back(makeCamera(id, position, target, dimension));
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
                    makeAnatomyStudioLightRig(),
                    manifest,
                    &reason
                )) {
                throw std::runtime_error("preview scene compile: " + reason);
            }
            metalrobo::MetalHybridRendererConfig configuration;
            configuration.width = dimension;
            configuration.height = dimension;
            // Keep one reference workspace per fixed camera.  At high
            // resolution, reusing the first camera's workspace can leave a
            // later multi-angle readback without its source segmentation.
            configuration.maximumReferenceFramesInFlight = 3u;
            configuration.clearColorAndDepth = {0.012f, 0.018f, 0.028f, 1.0e30f};
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
