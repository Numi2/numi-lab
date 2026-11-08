#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/VisualPlatform.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
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

void require(
    const bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require(const bool condition, const char* message) {
    require(condition, std::string{message});
}

std::size_t validGeometryPixels(
    const metalrobo::HybridObservationBatch& observations
) {
    std::size_t result = 0u;
    for (const std::uint32_t validity : observations.validity) {
        result +=
            (validity & MR_VISUAL_VALIDITY_GEOMETRY) != 0u
            ? 1u
            : 0u;
    }
    return result;
}

metalrobo::VisualAssetPackV2 makeAuthoredObjectPack(
    const std::uint32_t bodyIndex
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = "pick_object_authored";
    pack.sourceUri = "probe://pick_object_authored";
    pack.sourceContentHash = "sha256:pick-object-probe";
    pack.license = "CC0-1.0";
    pack.preprocessingProvenance =
        "metalrobo_visual_platform_probe/v2";
    constexpr std::array positions{
        std::array{-0.07f, -0.07f, -0.07f},
        std::array{ 0.07f, -0.07f, -0.07f},
        std::array{ 0.00f,  0.08f, -0.07f},
        std::array{ 0.00f,  0.00f,  0.09f},
    };
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
        0u, 2u, 1u,
        0u, 1u, 3u,
        1u, 2u, 3u,
        2u, 0u, 3u,
    };
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = {
        0.82f,
        0.12f,
        0.03f,
        1.0f,
    };
    material.surface = {0.3f, 0.1f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff =
        {0.2f, 0.15f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    material.textureIndices1 = material.textureIndices0;
    material.reserved = material.textureIndices0;
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED,
        0u,
        1u,
    };
    pack.materials = {material};
    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {
        0u,
        static_cast<std::uint32_t>(pack.indices.size()),
        0u,
        0u,
    };
    primitive.identity = {1u, 1u, bodyIndex, 1u};
    primitive.boundsMinimum = {-0.07f, -0.07f, -0.07f, 1.0f};
    primitive.boundsMaximum = {0.07f, 0.08f, 0.09f, 1.0f};
    pack.primitives = {primitive};
    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {
        0u,
        bodyIndex,
        MR_VISUAL_BINDING_RIGID_BODY,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
    };
    instance.identity = {1u, 1u, bodyIndex, 1u};
    instance.geometry = {0u, 1u, 0u, 0u};
    pack.instances = {instance};
    pack.symbolicBindings = {{
        "pick_object",
        "pick_object",
        0u,
        bodyIndex,
        MR_VISUAL_BINDING_RIGID_BODY,
    }};
    pack.contentHash =
        metalrobo::computeVisualAssetPackContentHash(pack);
    return pack;
}

} // namespace

int main() {
    try {
        metalrobo::WorldTemplate worldTemplate;
        require(
            metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                metalrobo::makeFrankaPickPlaceEngineModel(),
                worldTemplate
            ),
            "episode compile"
        );
        std::string reason;
        metalrobo::WorldFamily family;
        require(
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            ),
            "family compile"
        );
        constexpr std::uint32_t kManipulatedBody = 11u;
        constexpr std::uint32_t kManipulatedSemantic = 2u;
        const std::uint32_t manipulatedAsset =
            worldTemplate.assetIndex("pick_object");
        const metalrobo::VisualAssetPackV2 authoredPack =
            makeAuthoredObjectPack(kManipulatedBody);
        const std::filesystem::path authoredPackPath =
            std::filesystem::temp_directory_path() /
            "metalrobo-pick-object-probe.mrvpack";
        require(
            metalrobo::writeVisualAssetPack(
                authoredPack,
                authoredPackPath,
                &reason
            ),
            "authored V2 pack write: " + reason
        );
        const std::array authoredReferences{
            metalrobo::VisualAssetReferenceV3{
                authoredPackPath,
                authoredPack.contentHash,
                manipulatedAsset,
                kManipulatedSemantic,
                manipulatedAsset + 1u,
            },
        };
        metalrobo::VisualSceneManifestV3 authoredVisualScene;
        require(
            metalrobo::compileVisualSceneManifestV3(
                worldTemplate,
                authoredReferences,
                metalrobo::makeNeutralStudioEnvironmentV2(),
                metalrobo::makeIndoorAreaLightRigV1(),
                authoredVisualScene,
                &reason
            ),
            "authored-only V3 visual compile: " + reason
        );
        const std::array<MRHybridGaussianGPU, 1> capturedAppearance{{
            {
                {0.45f, 0.0f, 0.08f, 0.65f},
                {0.025f, 0.025f, 0.025f, 1.0f},
                {0.0f, 0.0f, 0.0f, 1.0f},
                {0.15f, 0.45f, 0.95f, 0.0f},
                {
                    worldTemplate.assetIndex("workspace"),
                    MR_INVALID_INDEX,
                    3u,
                    MR_HYBRID_GAUSSIAN_ASSET_LOCAL,
                },
            },
        }};
        authoredVisualScene.renderScene.gaussians.assign(
            capturedAppearance.begin(),
            capturedAppearance.end()
        );
        authoredVisualScene.renderScene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(
                authoredVisualScene.renderScene
            );
        authoredVisualScene.fingerprint =
            metalrobo::computeVisualSceneManifestV3Fingerprint(
                authoredVisualScene
            );
        std::error_code ignoredAuthoredPackRemoval;
        require(
            authoredVisualScene.valid(&reason) &&
                authoredVisualScene.visualPackHashes.size() == 1u &&
                authoredVisualScene.renderScene.visualPacks.size() == 1u &&
                authoredVisualScene.renderScene.sensorBindings.size() == 2u,
            "authored V3 scene contains unexpected references: " + reason
        );
        const std::filesystem::path visualManifestPath =
            std::filesystem::temp_directory_path() /
            (
                "metalrobo-visual-scene-" +
                std::to_string(authoredVisualScene.fingerprint) +
                ".json"
            );
        require(
            metalrobo::writeVisualSceneManifestV3(
                authoredVisualScene,
                visualManifestPath,
                &reason
            ) &&
                std::filesystem::file_size(
                    visualManifestPath
                ) > 0u,
            "visual scene manifest write failed: " + reason
        );
        std::error_code ignoredManifestRemoval;
        std::filesystem::remove(
            visualManifestPath,
            ignoredManifestRemoval
        );

        constexpr std::uint32_t kEnvironmentCount = 2u;
        metalrobo::MetalWorldFamilyContext worlds;
        require(
            worlds.compile(family, kEnvironmentCount),
            "world-family Metal compile"
        );
        require(
            worlds.sample(kEnvironmentCount, 0x5eed1234ull),
            "world-family sample"
        );
        metalrobo::WorldInstanceBatch sampledWorlds;
        require(
            worlds.readback(sampledWorlds),
            "world-family readback"
        );
        metalrobo::MetalWorldFamilyPhysicsBatch physics;
        require(
            worlds.readbackPhysics(physics),
            "world-family physics readback"
        );

        std::vector<MRBodyStateGPU> bodyStates;
        require(
            metalrobo::composeVisualBodyStates(
                worldTemplate.engineModel,
                kEnvironmentCount,
                physics.resetQ,
                physics.resetV,
                physics.resetSceneBodies,
                bodyStates,
                &reason
            ),
            "visual body composition: " + reason
        );
        require(
            bodyStates.size() ==
                kEnvironmentCount *
                    worldTemplate.engineModel.bodies.size(),
            "visual body state does not match global body indexing"
        );
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount;
             ++environment) {
            bodyStates[
                static_cast<std::size_t>(environment) *
                    worldTemplate.engineModel.bodies.size() +
                kManipulatedBody
            ].position.x += 0.075f;
        }

        const std::uint64_t visualSceneFingerprint =
            authoredVisualScene.renderScene.fingerprint;
        const std::uint64_t visualManifestFingerprint =
            authoredVisualScene.fingerprint;
        const std::uint64_t environmentFingerprint =
            authoredVisualScene.renderScene.environment.fingerprint;
        const std::uint64_t lightRigFingerprint =
            authoredVisualScene.renderScene.lightRig.fingerprint;
        auto directScene = authoredVisualScene.renderScene;
        directScene.gaussians.clear();
        auto boundObjectScene = directScene;
        directScene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(directScene);
        metalrobo::MetalHybridRenderer renderer;
        require(
            renderer.compile(
                std::move(authoredVisualScene.renderScene),
                metalrobo::VisualRendererProfileV1::sensorFast(),
                kEnvironmentCount
            ),
            "visual sensor runtime compile"
        );
        metalrobo::HybridLiveStateBatch liveState;
        liveState.environmentCount = kEnvironmentCount;
        liveState.bodyCount = static_cast<std::uint32_t>(
            worldTemplate.engineModel.bodies.size()
        );
        liveState.currentBodies =
            std::span<const MRBodyStateGPU>{bodyStates};
        liveState.previousBodies =
            std::span<const MRBodyStateGPU>{bodyStates};
        liveState.frameIndex = 23u;
        liveState.source = MR_VISUAL_SOURCE_SIMULATION;
        liveState.captureTimestampSeconds = 1.5;
        liveState.frameAgeSeconds = 0.0;

        std::array<metalrobo::HybridObservationBatch, 2>
            observations;
        metalrobo::MetalHybridRendererDiagnostics lastRender;
        for (std::uint32_t camera = 0u; camera < 2u; ++camera) {
            liveState.sensorSequence = 100u + camera;
            lastRender = renderer.renderLive(
                worlds,
                liveState,
                camera
            );
            require(lastRender, "live camera render");
            require(
                renderer.readback(observations[camera]),
                "live camera readback"
            );
        }
        const std::size_t fixedVisible =
            validGeometryPixels(observations[0]);
        require(
            fixedVisible > 100u,
            "fixed camera did not see physics-bound visual geometry"
        );
        const auto firstVisible = std::find_if(
            observations[0].identities.begin(),
            observations[0].identities.end(),
            [](const mr_uint4 value) {
                return value.x != MR_INVALID_INDEX &&
                    value.y != MR_INVALID_INDEX &&
                    value.z != MR_INVALID_INDEX;
            }
        );
        const std::size_t visibleIndex =
            static_cast<std::size_t>(
                firstVisible -
                observations[0].identities.begin()
            );
        require(
            visibleIndex < observations[0].identities.size() &&
                observations[0].identities[visibleIndex].x !=
                    MR_INVALID_INDEX &&
                observations[0].identities[visibleIndex].y !=
                    MR_INVALID_INDEX &&
                observations[0].identities[visibleIndex].z !=
                    MR_INVALID_INDEX &&
                std::isfinite(
                    observations[0].normals[visibleIndex].x
                ),
            "dense semantic, instance, link, or normal truth is missing"
        );

        metalrobo::MetalHybridObjectTracker tracker;
        const std::uint32_t trackedBody =
            observations[0].identities[visibleIndex].z;
        metalrobo::MetalHybridObjectTrackerConfig trackerConfig;
        trackerConfig.capacity = kEnvironmentCount;
        trackerConfig.cameraIndex = 0u;
        trackerConfig.rootBodyIndex =
            worldTemplate.engineModel.articulations.front().firstBody;
        constexpr std::uint32_t kActorFrameSize = 7u;
        constexpr std::uint32_t kActorHistoryLength = 5u;
        trackerConfig.maximumActorHistoryLength = kActorHistoryLength;
        trackerConfig.timestepSeconds = 0.02f;
        trackerConfig.bindings.push_back({
            .instanceId = observations[0].identities[visibleIndex].y,
            .actorFrameOffset = 0u,
            .positionScale = 1.0f,
            .velocityScale = 1.0f,
            .minimumVisiblePixels = 1u,
        });
        const metalrobo::MetalHybridRendererLayout rendererLayout =
            renderer.layout();
        const std::uint32_t visualPixels =
            rendererLayout.width * rendererLayout.height;
        trackerConfig.maskedDepthInstanceIds = {
            observations[0].identities[visibleIndex].y,
        };
        trackerConfig.maskedDepthWidth = rendererLayout.width;
        trackerConfig.maskedDepthHeight = rendererLayout.height;
        trackerConfig.maskedDepthActorFrameOffset =
            kActorFrameSize * kActorHistoryLength;
        trackerConfig.maskedDepthFrameOffsets = {0u, 1u};
        trackerConfig.maskedDepthFeatureCount =
            MR_TASK_MASKED_DEPTH_FEATURE_COUNT;
        trackerConfig.maskedDepthNearMeters = 0.1f;
        trackerConfig.maskedDepthFarMeters = 5.0f;
        trackerConfig.maskedDepthPixelDropoutProbability = 0.1f;
        trackerConfig.maskedDepthJitterMeters = 0.15f;
        trackerConfig.maskedDepthNoiseSigmaMeters = 0.03f;
        trackerConfig.maskedDepthEdgeFlickerProbability = 0.15f;
        require(
            tracker.compile(renderer, worlds, std::move(trackerConfig)),
            "device object tracker compile"
        );
        const metalrobo::MetalWorldDeviceObservationProgram
            trackerProgram = tracker.observationProgram();
        require(trackerProgram.valid(), "device object tracker program");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        const std::size_t bodyBytes =
            bodyStates.size() * sizeof(MRBodyStateGPU);
        id<MTLBuffer> bodyBuffer = [device
            newBufferWithBytes:bodyStates.data()
                         length:bodyBytes
                        options:MTLResourceStorageModeShared];
        const std::uint32_t kTrackWidth =
            kActorFrameSize * kActorHistoryLength +
            2u * visualPixels + MR_TASK_MASKED_DEPTH_FEATURE_COUNT;
        const std::size_t trackElements =
            kEnvironmentCount * kTrackWidth;
        id<MTLBuffer> resetBuffer = [device
            newBufferWithLength:2u * kEnvironmentCount *
                sizeof(std::uint32_t)
                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> historyBuffer = [device
            newBufferWithLength:trackElements * sizeof(float)
                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> observationBuffer = [device
            newBufferWithLength:trackElements * sizeof(float)
                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> taskStateBuffer = [device
            newBufferWithLength:
                kEnvironmentCount * sizeof(MRTaskStateGPU)
                         options:MTLResourceStorageModeShared];
        require(
            device != nil && queue != nil && command != nil &&
                bodyBuffer != nil && resetBuffer != nil &&
                historyBuffer != nil && observationBuffer != nil &&
                taskStateBuffer != nil,
            "device object tracker probe allocation"
        );
        std::memset(resetBuffer.contents, 0, resetBuffer.length);
        std::fill_n(
            static_cast<std::uint32_t*>(resetBuffer.contents),
            kEnvironmentCount,
            1u
        );
        std::memset(historyBuffer.contents, 0, historyBuffer.length);
        std::memset(
            observationBuffer.contents,
            0,
            observationBuffer.length
        );
        std::memset(
            taskStateBuffer.contents,
            0,
            taskStateBuffer.length
        );
        metalrobo::MetalWorldDeviceObservationPass trackerPass{
            .commandBuffer = (__bridge void*)command,
            .currentBodies = (__bridge void*)bodyBuffer,
            .resetMasks = (__bridge void*)resetBuffer,
            .taskStates = (__bridge void*)taskStateBuffer,
            .actorHistory = (__bridge void*)historyBuffer,
            .actorObservations = (__bridge void*)observationBuffer,
            .actorObservationOffsetElements = 0u,
            .seed = 0x8d19c4a57b236ef0ull,
            .controlStep = 0u,
            .environmentCount = kEnvironmentCount,
            .bodyCount = static_cast<std::uint32_t>(
                worldTemplate.engineModel.bodies.size()
            ),
            .actorFrameSize = kActorFrameSize,
            .actorHistoryLength = kActorHistoryLength,
            .actorObservationSize = kTrackWidth,
        };
        require(
            trackerProgram.encode(trackerProgram.context, trackerPass),
            "device object tracker encode"
        );
        [command commit];
        [command waitUntilCompleted];
        require(
            command.status == MTLCommandBufferStatusCompleted,
            "device object tracker command completion"
        );
        const auto* tracked = static_cast<const float*>(
            observationBuffer.contents
        );
        require(
            tracked[0] > 0.5f &&
                std::all_of(
                    tracked,
                    tracked + trackElements,
                    [](const float value) {
                        return std::isfinite(value);
                    }
                ),
            "device object tracker did not publish a finite visible track"
        );
        const float* currentDepth =
            tracked + kActorFrameSize * kActorHistoryLength;
        const float* previousDepth = currentDepth + visualPixels;
        const float* depthFeatures = previousDepth + visualPixels;
        require(
            std::all_of(
                currentDepth,
                previousDepth + visualPixels,
                [](const float value) {
                    return std::isfinite(value) &&
                        value >= 0.0f && value <= 1.0f;
                }
            ) &&
            std::any_of(
                currentDepth,
                currentDepth + visualPixels,
                [](const float value) { return value < 1.0f; }
            ) &&
            std::equal(
                currentDepth,
                currentDepth + visualPixels,
                previousDepth
            ) &&
            std::all_of(
                depthFeatures,
                depthFeatures + MR_TASK_MASKED_DEPTH_FEATURE_COUNT,
                [](const float value) {
                    return std::isfinite(value) &&
                        value >= -1.0f && value <= 1.0f;
                }
            ) &&
            depthFeatures[0u] > 0.0f &&
            depthFeatures[0u] == depthFeatures[5u] &&
            depthFeatures[3u] == depthFeatures[8u] &&
            depthFeatures[20u] == 0.0f &&
            depthFeatures[21u] == 0.0f &&
            depthFeatures[22u] == 0.0f &&
            depthFeatures[23u] == 0.0f,
            "device masked-depth reset history is invalid"
        );
        const std::vector<float> firstDepth(
            currentDepth,
            currentDepth + visualPixels
        );
        auto* deviceBodies = static_cast<MRBodyStateGPU*>(
            bodyBuffer.contents
        );
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount;
             ++environment) {
            deviceBodies[
                static_cast<std::size_t>(environment) *
                    worldTemplate.engineModel.bodies.size() +
                trackedBody
            ].position.x += 0.02f;
        }
        metalrobo::HybridObservationBatch beforeRejectedEncode;
        require(renderer.readback(beforeRejectedEncode), "pre-rejection readback");
        require(beforeRejectedEncode.metadata.identity.z == 0u,
                "first sensor sequence is not zero");
        id<MTLCommandBuffer> rejectedCommand = [queue commandBuffer];
        auto rejectedPass = trackerPass;
        rejectedPass.commandBuffer = (__bridge void*)rejectedCommand;
        rejectedPass.bodyCount += 1u;
        require(!trackerProgram.encode(trackerProgram.context, rejectedPass),
                "invalid body count must reject tracker encode");
        metalrobo::HybridObservationBatch afterRejectedEncode;
        require(renderer.readback(afterRejectedEncode), "post-rejection readback");
        require(std::memcmp(&beforeRejectedEncode.metadata, &afterRejectedEncode.metadata,
                            sizeof(beforeRejectedEncode.metadata)) == 0,
                "rejected encode changed frame metadata");
        id<MTLCommandBuffer> secondCommand = [queue commandBuffer];
        std::fill_n(
            static_cast<std::uint32_t*>(resetBuffer.contents),
            kEnvironmentCount,
            0u
        );
        auto* taskStates = static_cast<MRTaskStateGPU*>(
            taskStateBuffer.contents
        );
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount;
             ++environment) {
            taskStates[environment].episode.x = 1u;
        }
        trackerPass.commandBuffer = (__bridge void*)secondCommand;
        // A one-step Swift submission begins again at local slot zero. The
        // resident episode step must still advance visual temporal history.
        trackerPass.controlStep = 0u;
        require(
            trackerProgram.encode(trackerProgram.context, trackerPass),
            "device object tracker temporal encode"
        );
        [secondCommand commit];
        [secondCommand waitUntilCompleted];
        require(secondCommand.status == MTLCommandBufferStatusCompleted,
                "retry command failed");
        metalrobo::HybridObservationBatch retryFrame;
        require(renderer.readback(retryFrame), "retry frame readback");
        std::cout << "retry_sensor_sequence=" << retryFrame.metadata.identity.z << '\n';
        require(retryFrame.metadata.identity.z == 1u,
                "rejected encode advanced the retry sensor sequence");
        const std::uint32_t newestTrack =
            (kActorHistoryLength - 1u) * kActorFrameSize;
        require(
            secondCommand.status == MTLCommandBufferStatusCompleted &&
                tracked[newestTrack] > 0.5f &&
                std::abs(tracked[newestTrack + 4u]) > 0.1f &&
                std::equal(
                    previousDepth,
                    previousDepth + visualPixels,
                    firstDepth.begin()
                ),
            "device object tracker did not retain visible temporal motion"
        );

        // Graph-only geometric rendering has no materialized observation
        // planes, so this masked-depth tracker must use direct winners.
        metalrobo::MetalHybridRendererConfig directRendererConfig;
        directRendererConfig.width = 16u;
        directRendererConfig.height = 9u;
        directRendererConfig.geometricObservationsOnly = true;
        directRendererConfig.retainObservationBuffers = false;
        metalrobo::MetalHybridRenderer directRenderer(directRendererConfig);
        require(directRenderer.compile(
                    std::move(directScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    kEnvironmentCount),
                "direct-winner graph-only renderer compile");

        metalrobo::MetalHybridObjectTrackerConfig directConfig;
        directConfig.capacity = kEnvironmentCount;
        directConfig.cameraIndex = 0u;
        directConfig.rootBodyIndex =
            worldTemplate.engineModel.articulations.front().firstBody;
        directConfig.maximumActorHistoryLength = kActorHistoryLength;
        directConfig.timestepSeconds = 0.02f;
        directConfig.maskedDepthInstanceIds = {
            observations[0].identities[visibleIndex].y};
        directConfig.maskedDepthWidth = 16u;
        directConfig.maskedDepthHeight = 9u;
        directConfig.maskedDepthActorFrameOffset =
            kActorFrameSize * kActorHistoryLength;
        directConfig.maskedDepthFrameOffsets = {0u, 1u};
        directConfig.maskedDepthNearMeters = 0.1f;
        directConfig.maskedDepthFarMeters = 5.0f;
        metalrobo::MetalHybridObjectTracker directTracker;
        require(directTracker.compile(directRenderer, worlds, std::move(directConfig)),
                "direct-winner tracker compile");
        const auto directProgram = directTracker.observationProgram();
        require(directProgram.valid(), "direct-winner tracker program");
        std::memset(observationBuffer.contents, 0, observationBuffer.length);
        std::fill_n(static_cast<std::uint32_t*>(resetBuffer.contents),
                    kEnvironmentCount, 1u);
        auto directPass = trackerPass;
        id<MTLCommandBuffer> directFirst = [queue commandBuffer];
        directPass.commandBuffer = (__bridge void*)directFirst;
        require(directProgram.encode(directProgram.context, directPass),
                "direct-winner first encode");
        [directFirst commit];
        [directFirst waitUntilCompleted];
        require(directFirst.status == MTLCommandBufferStatusCompleted,
                "direct-winner first completion");
        id<MTLCommandBuffer> directRejected = [queue commandBuffer];
        auto directRejectedPass = directPass;
        directRejectedPass.commandBuffer = (__bridge void*)directRejected;
        directRejectedPass.bodyCount += 1u;
        require(!directProgram.encode(directProgram.context, directRejectedPass),
                "direct-winner invalid body count must reject");
        std::fill_n(static_cast<std::uint32_t*>(resetBuffer.contents),
                    kEnvironmentCount, 0u);
        id<MTLCommandBuffer> directRetry = [queue commandBuffer];
        directPass.commandBuffer = (__bridge void*)directRetry;
        require(directProgram.encode(directProgram.context, directPass),
                "direct-winner retry encode");
        [directRetry commit];
        [directRetry waitUntilCompleted];
        require(directRetry.status == MTLCommandBufferStatusCompleted,
                "direct-winner retry completion");
        const auto* directOutput = static_cast<const float*>(observationBuffer.contents);
        for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
            const float* first = directOutput + env * kTrackWidth +
                kActorFrameSize * kActorHistoryLength;
            require(std::all_of(first, first + 2u * 16u * 9u,
                                [](float value) {
                                    return std::isfinite(value) &&
                                        value >= 0.0f && value <= 1.0f;
                                }),
                    "direct-winner retry produced invalid depth");
        }
        std::cout << "direct_winner_retry=pass\n";

        // Bound object tracks need materialized depth, identity and validity.
        // A small geometric sensor must not select the winner-only shortcut.
        boundObjectScene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(boundObjectScene);
        metalrobo::MetalHybridRendererConfig boundRendererConfig;
        boundRendererConfig.width = 16u;
        boundRendererConfig.height = 9u;
        boundRendererConfig.geometricObservationsOnly = true;
        auto smallEpisode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
        smallEpisode.sensors[0].width = 16u;
        smallEpisode.sensors[0].height = 9u;
        smallEpisode.sensors[0].intrinsics = {14.0f, 10.5f, 8.0f, 4.5f};
        metalrobo::WorldTemplate smallTemplate;
        require(metalrobo::compileEpisodeTwin(
                    smallEpisode, metalrobo::makeFrankaPickPlaceEngineModel(),
                    smallTemplate), "small camera episode compile");
        metalrobo::WorldFamily smallFamily;
        require(metalrobo::compileWorldFamily(
                    smallTemplate, metalrobo::makeFrankaPickPlaceWorldProgram(),
                    smallFamily), "small camera family compile");
        metalrobo::MetalWorldFamilyContext smallWorlds;
        require(smallWorlds.compile(smallFamily, kEnvironmentCount),
                "small camera world compile");
        require(smallWorlds.sample(kEnvironmentCount, 0x5eed1234ull),
                "small camera world sample");
        auto boundReferenceScene = boundObjectScene;
        metalrobo::MetalHybridRenderer referenceRenderer(boundRendererConfig);
        require(referenceRenderer.compile(std::move(boundReferenceScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    kEnvironmentCount), "bound-object reference renderer compile");
        auto referenceLive = liveState;
        referenceLive.currentBodies = std::span<const MRBodyStateGPU>{
            deviceBodies, bodyStates.size()};
        referenceLive.previousBodies = referenceLive.currentBodies;
        require(referenceRenderer.renderLive(smallWorlds, referenceLive, 0u),
                "bound-object independent reference render");
        metalrobo::HybridObservationBatch boundReference;
        require(referenceRenderer.readback(boundReference), "bound-object reference readback");
        metalrobo::MetalHybridRenderer boundRenderer(boundRendererConfig);
        require(boundRenderer.compile(
                    std::move(boundObjectScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    kEnvironmentCount),
                "bound-object geometric renderer compile");
        metalrobo::MetalHybridObjectTrackerConfig boundConfig;
        boundConfig.capacity = kEnvironmentCount;
        boundConfig.cameraIndex = 0u;
        boundConfig.rootBodyIndex =
            worldTemplate.engineModel.articulations.front().firstBody;
        boundConfig.maximumActorHistoryLength = kActorHistoryLength;
        boundConfig.timestepSeconds = 0.02f;
        const std::uint32_t boundInstance = observations[0].identities[visibleIndex].y;
        boundConfig.bindings.push_back({
            .instanceId = boundInstance,
            .actorFrameOffset = 0u,
            .positionScale = 1.0f,
            .velocityScale = 1.0f,
            .minimumVisiblePixels = 1u,
        });
        boundConfig.maskedDepthInstanceIds = {boundInstance};
        boundConfig.maskedDepthWidth = 16u;
        boundConfig.maskedDepthHeight = 9u;
        boundConfig.maskedDepthActorFrameOffset =
            kActorFrameSize * kActorHistoryLength;
        boundConfig.maskedDepthFrameOffsets = {0u, 1u};
        boundConfig.maskedDepthNearMeters = 0.1f;
        boundConfig.maskedDepthFarMeters = 5.0f;
        metalrobo::MetalHybridObjectTracker boundTracker;
        require(boundTracker.compile(boundRenderer, smallWorlds, std::move(boundConfig)),
                "bound-object geometric tracker compile");
        const auto boundProgram = boundTracker.observationProgram();
        require(boundProgram.valid(), "bound-object geometric tracker program");
        constexpr std::size_t kBoundOffset = 13u;
        id<MTLBuffer> boundObservations = [device
            newBufferWithLength:(kBoundOffset + trackElements) * sizeof(float)
                        options:MTLResourceStorageModeShared];
        require(boundObservations != nil, "bound-object observation allocation");
        std::fill_n(static_cast<float*>(boundObservations.contents),
                    kBoundOffset + trackElements, -5.0f);
        std::memset(historyBuffer.contents, 0, historyBuffer.length);
        std::fill_n(static_cast<std::uint32_t*>(resetBuffer.contents),
                    kEnvironmentCount, 1u);
        id<MTLCommandBuffer> boundCommand = [queue commandBuffer];
        auto boundPass = trackerPass;
        boundPass.commandBuffer = (__bridge void*)boundCommand;
        boundPass.actorObservations = (__bridge void*)boundObservations;
        boundPass.actorObservationOffsetElements = kBoundOffset;
        require(boundProgram.encode(boundProgram.context, boundPass),
                "bound-object geometric tracker encode");
        [boundCommand commit];
        [boundCommand waitUntilCompleted];
        require(boundCommand.status == MTLCommandBufferStatusCompleted,
                "bound-object geometric tracker completion");
        metalrobo::HybridObservationBatch boundFrame;
        require(boundRenderer.readback(boundFrame), "bound-object frame readback");
        const auto* boundOutputBase = static_cast<const float*>(boundObservations.contents);
        require(std::all_of(boundOutputBase, boundOutputBase + kBoundOffset,
                            [](float value) { return value == -5.0f; }),
                "bound-object output prefix changed");
        const auto* boundOutput = boundOutputBase + kBoundOffset;
        const auto* boundHistory = static_cast<const float*>(historyBuffer.contents);
        constexpr std::size_t kSmallPixels = 16u * 9u;
        for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
            std::size_t referenceVisible = 0u;
            std::size_t actualVisible = 0u;
            for (std::size_t pixel = 0u; pixel < kSmallPixels; ++pixel) {
                const auto index = env * kSmallPixels + pixel;
                const bool expected =
                    boundReference.identities[index].y == boundInstance &&
                    (boundReference.validity[index] & MR_VISUAL_VALIDITY_DEPTH) != 0u;
                const bool actual =
                    boundFrame.identities[index].y == boundInstance &&
                    (boundFrame.validity[index] & MR_VISUAL_VALIDITY_DEPTH) != 0u;
                referenceVisible += expected ? 1u : 0u;
                actualVisible += actual ? 1u : 0u;
                require(expected == actual, "bound-object selected-pixel mismatch");
            }
            const float* frame = boundOutput + env * kTrackWidth;
            const float* depth = frame + kActorFrameSize * kActorHistoryLength;
            const auto selectedPixels = std::count_if(
                depth, depth + kSmallPixels,
                [](float value) { return std::isfinite(value) && value < 1.0f; });
            for (std::size_t pixel = 0u; pixel < kSmallPixels; ++pixel) {
                const auto index = env * kSmallPixels + pixel;
                const bool selected = boundFrame.identities[index].y == boundInstance &&
                    (boundFrame.validity[index] &
                     (MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_DEPTH)) ==
                        (MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_DEPTH);
                const float metric = selected
                    ? std::clamp(boundFrame.depth[index], 0.1f, 5.0f) : 5.0f;
                const float expected = (metric - 0.1f) / 4.9f;
                require(std::abs(depth[pixel] - expected) < 2.0e-6f &&
                            std::abs(depth[kSmallPixels + pixel] - expected) < 2.0e-6f,
                        "bound-object depth differs from materialized image");
            }
            std::cout << "bound_env=" << env << " reference_visible=" << referenceVisible
                      << " actual_visible=" << actualVisible << " track=" << frame[0]
                      << " history=" << boundHistory[env * kActorFrameSize * kActorHistoryLength]
                      << " depth_pixels=" << selectedPixels << '\n';
            require(referenceVisible > 0u && frame[0] > 0.5f &&
                    boundHistory[env * kActorFrameSize * kActorHistoryLength] > 0.5f &&
                    selectedPixels > 0,
                    "bound-object tracker requires materialized visible planes");
        }
        std::filesystem::remove(authoredPackPath, ignoredAuthoredPackRemoval);

        metalrobo::VisualSensorProfileV2 sensorProfile;
        sensorProfile.id = "franka.rgbd.reference";
        sensorProfile.frameJitterSeconds = 0.0005;
        sensorProfile.fingerprint =
            metalrobo::computeVisualSensorProfileFingerprint(
                sensorProfile
            );
        const std::array<std::uint32_t, 2> cameras{0u, 1u};
        metalrobo::VisualBatchAssemblyV1 assembly;
        assembly.provenance.source =
            MR_VISUAL_SOURCE_SIMULATION;
        assembly.provenance.episodeTwinFingerprint =
            worldTemplate.fingerprint;
        assembly.provenance.scenarioFingerprint =
            family.program.fingerprint;
        assembly.provenance.rendererFingerprint =
            visualSceneFingerprint;
        assembly.provenance.visualPackFingerprint =
            visualManifestFingerprint ^ visualSceneFingerprint;
        assembly.provenance.environmentMapFingerprint =
            environmentFingerprint;
        assembly.provenance.lightRigFingerprint =
            lightRigFingerprint;
        assembly.provenance.rendererProfileFingerprint =
            metalrobo::VisualRendererProfileV1::sensorFast()
                .fingerprint;
        assembly.provenance.shutterProfileFingerprint =
            sensorProfile.fingerprint;
        assembly.provenance.sensorProfileFingerprint =
            sensorProfile.fingerprint;
        assembly.provenance.calibrationFingerprint =
            family.fingerprint ^ 0x5aa55aa55aa55aa5ull;
        assembly.cameraIndices = cameras;
        assembly.observations = observations;
        assembly.currentBodyStates = bodyStates;
        metalrobo::VisualFrameBatchV1 frames;
        metalrobo::VisualTruthBatchV1 truth;
        require(
            metalrobo::assembleVisualBatches(
                worldTemplate,
                sampledWorlds,
                assembly,
                frames,
                truth,
                &reason
            ),
            "visual contract assembly: " + reason
        );
        require(
            frames.valid(&reason) && truth.valid(&reason) &&
                frames.viewCount == 2u &&
                frames.cameras[1].sensorId == "wrist_rgbd" &&
                truth.objectPoses.size() ==
                    kEnvironmentCount *
                        worldTemplate.assets.size() &&
                truth.linkPoses.size() ==
                    bodyStates.size(),
            "deployable frame or supervisory truth is incomplete: " +
                reason
        );
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount;
             ++environment) {
            const MRVisualPoseGPU& objectPose =
                truth.objectPoses[
                    static_cast<std::size_t>(environment) *
                        worldTemplate.assets.size() +
                    manipulatedAsset
                ];
            const MRVisualPoseGPU& liveLinkPose =
                truth.linkPoses[
                    static_cast<std::size_t>(environment) *
                        worldTemplate.engineModel.bodies.size() +
                    kManipulatedBody
                ];
            require(
                std::abs(
                    objectPose.position.x -
                    liveLinkPose.position.x
                ) < 1.0e-5f &&
                    std::abs(
                        objectPose.position.y -
                        liveLinkPose.position.y
                    ) < 1.0e-5f &&
                    std::abs(
                        objectPose.position.z -
                        liveLinkPose.position.z
                    ) < 1.0e-5f,
                "object pose supervision did not follow live physics"
            );
        }
        auto independentlyTimedObservations = observations;
        independentlyTimedObservations[1].metadata.timing.x = 1.45f;
        independentlyTimedObservations[1].metadata.timing.y = 0.05f;
        assembly.observations = independentlyTimedObservations;
        metalrobo::VisualFrameBatchV1 independentlyTimedFrames;
        metalrobo::VisualTruthBatchV1 independentlyTimedTruth;
        require(
            metalrobo::assembleVisualBatches(
                worldTemplate,
                sampledWorlds,
                assembly,
                independentlyTimedFrames,
                independentlyTimedTruth,
                &reason
            ) &&
                independentlyTimedFrames
                        .cameras[1]
                        .captureTimestampSeconds !=
                    independentlyTimedFrames
                        .cameras[0]
                        .captureTimestampSeconds,
            "independently timed visual sensors were rejected: " +
                reason
        );
        auto unsynchronizedObservations = observations;
        ++unsynchronizedObservations[1].metadata.identity.x;
        assembly.observations = unsynchronizedObservations;
        metalrobo::VisualFrameBatchV1 rejectedFrames;
        metalrobo::VisualTruthBatchV1 rejectedTruth;
        require(
            !metalrobo::assembleVisualBatches(
                worldTemplate,
                sampledWorlds,
                assembly,
                rejectedFrames,
                rejectedTruth,
                &reason
            ),
            "unsynchronized visual views were accepted"
        );
        assembly.observations = observations;

        const std::array<float, 4> proprioception{
            0.1f,
            0.2f,
            0.3f,
            0.4f,
        };
        const std::array<float, 2> previousActions{0.0f, 0.0f};
        const std::array<float, 2> commands{1.0f, 1.0f};
        const std::array<float, 6> privileged{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
        };
        metalrobo::PolicyObservationRequestV1 request;
        request.profile =
            metalrobo::ObservationProfileV1::rgbXYZ;
        request.environmentCount = kEnvironmentCount;
        request.proprioception = proprioception;
        request.proprioceptionWidth = 2u;
        request.previousActions = previousActions;
        request.previousActionWidth = 1u;
        request.taskCommands = commands;
        request.taskCommandWidth = 1u;
        request.privilegedState = privileged;
        request.privilegedStateWidth = 3u;
        metalrobo::PolicyObservationBatchV1 policy;
        require(
            metalrobo::PolicyObservationAssemblerV1{}.assemble(
                frames,
                nullptr,
                request,
                policy,
                &reason
            ),
            "policy observation assembly: " + reason
        );
        require(
            policy.valid(&reason) &&
                policy.privilegedWidth == 3u,
            "actor/critic observation groups were not preserved"
        );

        constexpr float kIdealX = 0.2f;
        constexpr float kIdealY = -0.1f;
        const mr_float4 distortion{
            0.1f,
            -0.01f,
            0.005f,
            -0.004f,
        };
        const float radiusSquared =
            kIdealX * kIdealX + kIdealY * kIdealY;
        const float radial =
            1.0f + distortion.x * radiusSquared +
            distortion.y * radiusSquared * radiusSquared;
        const float distortedX =
            kIdealX * radial +
            2.0f * distortion.z * kIdealX * kIdealY +
            distortion.w *
                (radiusSquared + 2.0f * kIdealX * kIdealX);
        const float distortedY =
            kIdealY * radial +
            distortion.z *
                (radiusSquared + 2.0f * kIdealY * kIdealY) +
            2.0f * distortion.w * kIdealX * kIdealY;
        metalrobo::VisualFrameBatchV1 calibratedFrame;
        calibratedFrame.source = MR_VISUAL_SOURCE_SIMULATION;
        calibratedFrame.environmentCount = 1u;
        calibratedFrame.viewCount = 1u;
        calibratedFrame.width = 1u;
        calibratedFrame.height = 1u;
        calibratedFrame.modalities =
            MR_VISUAL_MODALITY_RGB |
            MR_VISUAL_MODALITY_DEPTH |
            MR_VISUAL_MODALITY_DEPTH_VALIDITY;
        calibratedFrame.episodeTwinFingerprint = 1u;
        calibratedFrame.scenarioFingerprint = 2u;
        calibratedFrame.rendererFingerprint = 3u;
        calibratedFrame.sensorProfileFingerprint = 4u;
        calibratedFrame.calibrationFingerprint = 5u;
        metalrobo::VisualCameraFrameV1 calibratedCamera;
        calibratedCamera.sensorId = "distorted_rgbd";
        calibratedCamera.intrinsics = {
            100.0f,
            100.0f,
            0.5f - 100.0f * distortedX,
            0.5f - 100.0f * distortedY,
        };
        calibratedCamera.distortion = distortion;
        calibratedCamera.frameIndex = 9u;
        calibratedCamera.captureTimestampSeconds = 0.75;
        calibratedCamera.valid = true;
        calibratedFrame.cameras = {calibratedCamera};
        calibratedFrame.rgbLinear = {
            {0.1f, 0.2f, 0.3f, 1.0f},
        };
        calibratedFrame.depthMeters = {2.0f};
        calibratedFrame.depthValidity = {1u};
        metalrobo::PolicyObservationRequestV1 calibratedRequest;
        calibratedRequest.profile =
            metalrobo::ObservationProfileV1::rgbXYZ;
        calibratedRequest.environmentCount = 1u;
        metalrobo::PolicyObservationBatchV1 calibratedPolicy;
        require(
            metalrobo::PolicyObservationAssemblerV1{}.assemble(
                calibratedFrame,
                nullptr,
                calibratedRequest,
                calibratedPolicy,
                &reason
            ),
            "distortion-aware RGB-XYZ assembly: " + reason
        );
        require(
            calibratedPolicy.deployable.size() == 7u &&
                std::abs(
                    calibratedPolicy.deployable[3] - 0.4f
                ) < 1.0e-4f &&
                std::abs(
                    calibratedPolicy.deployable[4] + 0.2f
                ) < 1.0e-4f &&
                std::abs(
                    calibratedPolicy.deployable[5] - 2.0f
                ) < 1.0e-5f &&
                calibratedPolicy.deployable[6] == 1.0f,
            "RGB-XYZ ignored calibrated lens distortion"
        );

        metalrobo::PerceptionTensorV1 objectTensor;
        objectTensor.id = "objects";
        objectTensor.modality =
            MR_VISUAL_MODALITY_OBJECT_POSE;
        objectTensor.coordinateFrame =
            MR_VISUAL_FRAME_ROBOT_BASE;
        objectTensor.shape = {kEnvironmentCount, 1u};
        objectTensor.floatValues = {0.25f, 0.75f};
        objectTensor.timestampSeconds =
            frames.cameras.front().captureTimestampSeconds;
        metalrobo::PerceptionResultBatchV1 perception;
        perception.providerId = "probe.replaceable-provider";
        perception.providerContentHash = "sha256:probe";
        perception.frameIndex =
            frames.cameras.front().frameIndex;
        perception.timestampSeconds =
            frames.cameras.front().captureTimestampSeconds;
        perception.tensors = {objectTensor};
        metalrobo::PolicyObservationRequestV1 objectRequest =
            request;
        objectRequest.profile =
            metalrobo::ObservationProfileV1::objectCentric;
        auto stalePerception = perception;
        ++stalePerception.frameIndex;
        require(
            !metalrobo::PolicyObservationAssemblerV1{}.assemble(
                frames,
                &stalePerception,
                objectRequest,
                policy,
                &reason
            ),
            "stale perception result entered a policy observation"
        );
        require(
            metalrobo::PolicyObservationAssemblerV1{}.assemble(
                frames,
                &perception,
                objectRequest,
                policy,
                &reason
            ),
            "synchronized perception assembly: " + reason
        );
        metalrobo::PerceptionTensorV1 noDetections =
            objectTensor;
        noDetections.shape = {kEnvironmentCount, 0u};
        noDetections.floatValues.clear();
        perception.tensors = {noDetections};
        require(
            noDetections.validContract(&reason) &&
                metalrobo::PolicyObservationAssemblerV1{}.assemble(
                    frames,
                    &perception,
                    objectRequest,
                    policy,
                    &reason
                ),
            "zero-detection perception result was not representable: " +
                reason
        );

        metalrobo::PerceptionProviderDescriptorV1 descriptor;
        descriptor.id = "probe.replaceable-provider";
        descriptor.contentHash = "sha256:probe";
        descriptor.inputModalities =
            MR_VISUAL_MODALITY_RGB |
            MR_VISUAL_MODALITY_DEPTH;
        descriptor.capabilities =
            MR_PERCEPTION_CAP_DENSE_FEATURE |
            MR_PERCEPTION_CAP_OBJECT_POSE;
        descriptor.acceptsDeviceBuffers = true;
        require(
            descriptor.valid(&reason),
            "replaceable perception provider contract is invalid: " +
                reason
        );

        const auto captureA =
            metalrobo::makeVisualSensorCapture(
                sensorProfile,
                sampledWorlds.instances.front().identity.x,
                0u,
                23u
            );
        const auto captureB =
            metalrobo::makeVisualSensorCapture(
                sensorProfile,
                sampledWorlds.instances.front().identity.x,
                0u,
                23u
            );
        require(
            captureA.valid(&reason) &&
                captureA.publishTimestampSeconds ==
                    captureB.publishTimestampSeconds,
            "sensor timing is not deterministic: " + reason
        );

        metalrobo::VisualEpisodeStreamV1 episode;
        episode.id = "franka.visual.probe";
        episode.episodeTwinFingerprint =
            worldTemplate.fingerprint;
        episode.worldFamilyFingerprint = family.fingerprint;
        episode.scenarioFingerprint =
            assembly.provenance.scenarioFingerprint;
        episode.rendererFingerprint =
            authoredVisualScene.renderScene.fingerprint;
        episode.visualSceneFingerprint =
            authoredVisualScene.fingerprint;
        episode.visualPackFingerprint =
            assembly.provenance.visualPackFingerprint;
        episode.environmentMapFingerprint =
            assembly.provenance.environmentMapFingerprint;
        episode.lightRigFingerprint =
            assembly.provenance.lightRigFingerprint;
        episode.rendererProfileFingerprint =
            assembly.provenance.rendererProfileFingerprint;
        episode.shutterProfileFingerprint =
            assembly.provenance.shutterProfileFingerprint;
        episode.sensorProfileFingerprint =
            assembly.provenance.sensorProfileFingerprint;
        episode.calibrationFingerprint =
            assembly.provenance.calibrationFingerprint;
        episode.physicsFingerprint =
            worldTemplate.fingerprint;
        metalrobo::VisualEpisodeStepV1 step;
        step.frameIndex = 23u;
        step.scenarioKey =
            sampledWorlds.instances.front().identity.x;
        step.timestampSeconds = 1.5;
        step.reward = 0.25;
        step.proprioception = {0.1f, 0.2f};
        step.action = {0.0f};
        step.taskCommand = {1.0f};
        step.privilegedState = {1.0f, 2.0f, 3.0f};
        step.frameContentHash = "sha256:frame-probe";
        step.truthContentHash = "sha256:truth-probe";
        require(
            episode.append(std::move(step), &reason) &&
                episode.finalize(&reason),
            "visual episode stream is invalid: " + reason
        );
        const std::filesystem::path manifestPath =
            std::filesystem::temp_directory_path() /
            (
                "metalrobo-visual-" +
                std::to_string(episode.fingerprint) +
                ".json"
            );
        require(
            metalrobo::writeVisualEpisodeManifest(
                episode,
                manifestPath,
                &reason
            ) &&
                std::filesystem::file_size(manifestPath) > 0u,
            "visual episode manifest write failed: " + reason
        );
        std::error_code ignored;
        std::filesystem::remove(manifestPath, ignored);

        std::cout
            << "device=\"" << lastRender.deviceName << "\""
            << " environments=" << kEnvironmentCount
            << " views=" << frames.viewCount
            << " triangles="
            << lastRender.layout.meshTriangleCount
            << " clusters="
            << lastRender.layout.meshClusterCount
            << " fixed_visible_pixels=" << fixedVisible
            << " wrist_visible_pixels="
            << validGeometryPixels(observations[1])
            << " deployable_width=" << policy.deployableWidth
            << " truth_objects=" << truth.objectPoses.size()
            << " episode_fingerprint=" << episode.fingerprint
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_visual_platform_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
