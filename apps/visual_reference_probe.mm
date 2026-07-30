#import <Metal/Metal.h>

#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPlatform.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

metalrobo::VisualAssetPackV2 makePack() {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = "reference_cube";
    pack.sourceUri = "probe://reference_cube";
    pack.sourceContentHash = "sha256:reference-cube-source";
    pack.license = "CC0-1.0";
    pack.preprocessingProvenance =
        "metalrobo_visual_reference_probe/v3";

    constexpr std::array positions{
        std::array{-0.06f, -0.06f, -0.06f},
        std::array{ 0.06f, -0.06f, -0.06f},
        std::array{ 0.06f,  0.06f, -0.06f},
        std::array{-0.06f,  0.06f, -0.06f},
        std::array{-0.06f, -0.06f,  0.06f},
        std::array{ 0.06f, -0.06f,  0.06f},
        std::array{ 0.06f,  0.06f,  0.06f},
        std::array{-0.06f,  0.06f,  0.06f},
    };
    pack.vertices.reserve(positions.size());
    for (const auto& position : positions) {
        const float length = std::sqrt(
            position[0] * position[0] +
            position[1] * position[1] +
            position[2] * position[2]
        );
        const std::array<float, 3u> normal{
            position[0] / length,
            position[1] / length,
            position[2] / length,
        };
        const std::array<float, 3u> reference =
            std::abs(normal[2]) < 0.999f
            ? std::array<float, 3u>{0.0f, 0.0f, 1.0f}
            : std::array<float, 3u>{0.0f, 1.0f, 0.0f};
        std::array<float, 3u> tangent{
            reference[1] * normal[2] -
                reference[2] * normal[1],
            reference[2] * normal[0] -
                reference[0] * normal[2],
            reference[0] * normal[1] -
                reference[1] * normal[0],
        };
        const float tangentLength = std::sqrt(
            tangent[0] * tangent[0] +
            tangent[1] * tangent[1] +
            tangent[2] * tangent[2]
        );
        for (float& component : tangent) {
            component /= tangentLength;
        }
        pack.vertices.push_back({
            {position[0], position[1], position[2], 1.0f},
            {
                normal[0],
                normal[1],
                normal[2],
                1.0f,
            },
            {tangent[0], tangent[1], tangent[2], 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
    }
    pack.indices = {
        0u, 2u, 1u, 0u, 3u, 2u,
        4u, 5u, 6u, 4u, 6u, 7u,
        0u, 1u, 5u, 0u, 5u, 4u,
        2u, 3u, 7u, 2u, 7u, 6u,
        1u, 2u, 6u, 1u, 6u, 5u,
        3u, 0u, 4u, 3u, 4u, 7u,
    };

    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = {0.85f, 0.08f, 0.03f, 1.0f};
    material.emissionAndStrength = {0.0f, 0.0f, 0.0f, 0.0f};
    material.surface = {0.32f, 0.15f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff =
        {0.25f, 0.12f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    material.textureIndices1 = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED,
        0u,
        1u,
    };
    material.reserved = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    pack.materials.push_back(material);

    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {
        0u,
        static_cast<std::uint32_t>(pack.indices.size()),
        0u,
        0u,
    };
    primitive.identity = {77u, 7001u, 11u, 1u};
    primitive.boundsMinimum = {-0.06f, -0.06f, -0.06f, 1.0f};
    primitive.boundsMaximum = { 0.06f,  0.06f,  0.06f, 1.0f};
    pack.primitives.push_back(primitive);

    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {
        0u,
        11u,
        MR_VISUAL_BINDING_RIGID_BODY,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
    };
    instance.identity = {77u, 7001u, 11u, 1u};
    instance.geometry = {0u, 1u, 0u, 0u};
    pack.instances.push_back(instance);
    pack.symbolicBindings.push_back({
        "pick_object",
        "pick_object",
        0u,
        11u,
        MR_VISUAL_BINDING_RIGID_BODY,
    });
    pack.contentHash =
        metalrobo::computeVisualAssetPackContentHash(pack);
    return pack;
}

metalrobo::VisualMotionSampleBatchV1 makeMotion(
    const std::uint32_t bodyCount,
    const std::uint64_t frameIndex
) {
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = 1u;
    motion.bodyCount = bodyCount;
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 80.0;
    motion.timestampsSeconds = {
        motion.exposureOpenSeconds,
        motion.exposureCloseSeconds,
    };
    motion.bodyStates.resize(
        static_cast<std::size_t>(motion.sampleCount) * bodyCount
    );
    for (MRBodyStateGPU& body : motion.bodyStates) {
        body.orientation.w = 1.0f;
    }
    motion.bodyStates[11u].position =
        {0.42f, -0.08f, 0.08f, 0.0f};
    motion.bodyStates[
        static_cast<std::size_t>(bodyCount) + 11u
    ].position = {0.52f, 0.08f, 0.08f, 0.0f};
    motion.scenarioIdentity = 29u;
    motion.sensorIdentity = 1u;
    motion.frameIndex = frameIndex;
    motion.sensorSequence =
        static_cast<std::uint32_t>(frameIndex);
    return motion;
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const metalrobo::EngineModel model =
                metalrobo::makeFrankaPickPlaceEngineModel();
            metalrobo::WorldTemplate worldTemplate;
            require(
                metalrobo::compileEpisodeTwin(
                    metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                    model,
                    worldTemplate
                ),
                "episode compile"
            );
            metalrobo::WorldFamily family;
            require(
                metalrobo::compileWorldFamily(
                    worldTemplate,
                    metalrobo::makeFrankaPickPlaceWorldProgram(),
                    family
                ),
                "family compile"
            );
            metalrobo::MetalWorldFamilyContext worlds;
            require(worlds.compile(family, 1u), "world compile");
            require(worlds.sample(1u, 29u), "world sample");

            metalrobo::MetalHybridRendererConfig config;
            config.width = 160u;
            config.height = 120u;
            config.maximumReferenceFramesInFlight = 2u;
            metalrobo::MetalHybridRenderer renderer(config);
            std::string manifestReason;
            const std::filesystem::path packPath =
                std::filesystem::temp_directory_path() /
                "metalrobo-reference-cube.mrvpack";
            const metalrobo::VisualAssetPackV2 pack =
                makePack();
            if (!metalrobo::writeVisualAssetPack(
                    pack,
                    packPath,
                    &manifestReason
                )) {
                throw std::runtime_error(
                    "authored pack write: " + manifestReason
                );
            }
            const std::array authoredAssets{
                metalrobo::VisualAssetReferenceV3{
                    packPath,
                    pack.contentHash,
                    2u,
                    77u,
                    7001u,
                },
            };
            metalrobo::VisualSceneManifestV3 manifest;
            if (!metalrobo::compileVisualSceneManifestV3(
                    worldTemplate,
                    authoredAssets,
                    metalrobo::makeNeutralStudioEnvironmentV2(),
                    metalrobo::makeIndoorAreaLightRigV1(),
                    manifest,
                    &manifestReason
                )) {
                throw std::runtime_error(
                    "authored-only V3 composition: " +
                    manifestReason
                );
            }
            std::error_code ignored;
            for (MRVisualSensorBindingGPU& sensor :
                 manifest.renderScene.sensorBindings) {
                sensor.timing.y = 1.0f / 240.0f;
                sensor.timing.z = 1.0f / 120.0f;
                sensor.shutter.x = MR_VISUAL_SHUTTER_ROLLING;
                sensor.shutter.y =
                    MR_VISUAL_SHUTTER_TOP_TO_BOTTOM;
            }
            manifest.renderScene.fingerprint =
                metalrobo::computeVisualRenderSceneV3Fingerprint(
                    manifest.renderScene
                );
            manifest.fingerprint =
                metalrobo::computeVisualSceneManifestV3Fingerprint(
                    manifest
                );
            const std::filesystem::path manifestPath =
                std::filesystem::temp_directory_path() /
                (
                    "metalrobo-visual-v3-" +
                    std::to_string(manifest.fingerprint) +
                    ".json"
                );
            if (!metalrobo::writeVisualSceneManifestV3(
                    manifest,
                    manifestPath,
                    &manifestReason
                ) ||
                std::filesystem::file_size(manifestPath) == 0u) {
                throw std::runtime_error(
                    "V3 scene manifest: " + manifestReason
                );
            }
            std::filesystem::remove(manifestPath, ignored);
            auto referenceScene =
                std::move(manifest.renderScene);
            auto fastScene = referenceScene;
            require(
                renderer.compile(
                    std::move(referenceScene),
                    metalrobo::VisualRendererProfileV1::
                        sensorReference(),
                    1u
                ),
                "reference compile"
            );

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (device == nil || queue == nil) {
                throw std::runtime_error(
                    "reference probe could not create a Metal queue"
                );
            }
            const auto referenceStart =
                std::chrono::steady_clock::now();
            for (std::uint64_t frame = 1u; frame <= 2u; ++frame) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                const auto motion = makeMotion(
                    static_cast<std::uint32_t>(model.bodies.size()),
                    frame
                );
                metalrobo::MetalHybridFrameCommandContext context;
                context.commandBuffer = (__bridge void*)command;
                require(
                    renderer.encodeFrame(
                        worlds,
                        motion,
                        0u,
                        context
                    ),
                    "reference encode"
                );
                [command commit];
                [command waitUntilCompleted];
                if (command.status !=
                    MTLCommandBufferStatusCompleted) {
                    throw std::runtime_error(
                        "reference command failed: " +
                        std::string{
                            command.error.localizedDescription.UTF8String
                        }
                    );
                }
            }
            const double referenceMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    referenceStart
                ).count() / 2.0;

            metalrobo::HybridObservationBatch observations;
            require(renderer.readback(observations), "reference readback");
            const auto visible = std::find(
                observations.segmentation.begin(),
                observations.segmentation.end(),
                77u
            );
            if (visible == observations.segmentation.end()) {
                throw std::runtime_error(
                    "reference ray visibility did not render the "
                    "authored cube"
                );
            }
            const std::size_t pixel = static_cast<std::size_t>(
                visible - observations.segmentation.begin()
            );
            if ((observations.validity[pixel] &
                 MR_VISUAL_VALIDITY_GEOMETRY) == 0u ||
                !(observations.depth[pixel] > 0.05f) ||
                !(observations.depth[pixel] < 10.0f)) {
                throw std::runtime_error(
                    "reference ray truth buffers are invalid"
                );
            }

            metalrobo::MetalHybridRenderer fastRenderer(config);
            fastScene.sensorBindings[0].shutter.y =
                MR_VISUAL_SHUTTER_RIGHT_TO_LEFT;
            fastScene.fingerprint =
                metalrobo::computeVisualRenderSceneV3Fingerprint(
                    fastScene
                );
            require(
                fastRenderer.compile(
                    std::move(fastScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    1u
                ),
                "fast presentation compile"
            );
            const auto fastStart =
                std::chrono::steady_clock::now();
            for (std::uint64_t frame = 3u; frame <= 4u; ++frame) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                const auto motion = makeMotion(
                    static_cast<std::uint32_t>(model.bodies.size()),
                    frame
                );
                metalrobo::MetalHybridFrameCommandContext context;
                context.commandBuffer = (__bridge void*)command;
                require(
                    fastRenderer.encodeFrame(
                        worlds,
                        motion,
                        0u,
                        context
                    ),
                    "fast rolling-shutter encode"
                );
                [command commit];
                [command waitUntilCompleted];
                if (command.status !=
                    MTLCommandBufferStatusCompleted) {
                    throw std::runtime_error(
                        "fast rolling-shutter command failed"
                    );
                }
            }
            const double fastMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    fastStart
                ).count() / 2.0;
            metalrobo::HybridObservationBatch fastObservations;
            require(
                fastRenderer.readback(fastObservations),
                "fast rolling-shutter readback"
            );
            if (std::find(
                    fastObservations.segmentation.begin(),
                    fastObservations.segmentation.end(),
                    77u
                ) == fastObservations.segmentation.end()) {
                throw std::runtime_error(
                    "banded fast renderer lost authored geometry"
                );
            }
            const auto layout = renderer.layout();
            std::cout
                << "device=\"" << device.name.UTF8String << "\""
                << " ray_instances=" << layout.rayInstanceCount
                << " compact_blas_bytes="
                << layout.accelerationStructureBytes
                << " shadow_workspace_bytes="
                << layout.shadowWorkspaceBytes
                << " semantic=" << observations.segmentation[pixel]
                << " depth=" << observations.depth[pixel]
                << " fast_bytes="
                << fastRenderer.layout().retainedPrivateBytes
                << " reference_ms=" << referenceMilliseconds
                << " fast_ms=" << fastMilliseconds
                << '\n';
            std::filesystem::remove(packPath, ignored);
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "metalrobo_visual_reference_probe: "
                << error.what() << '\n';
            return 1;
        }
    }
}
