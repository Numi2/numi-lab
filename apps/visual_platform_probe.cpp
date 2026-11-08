#include <fstream>
#include "metalrobo/RunProgram.hpp"
#include <cstdlib>
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/VisualPlatform.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <numeric>
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
        if (std::getenv("NUMI_TEST_DEPTH_HOLD") != nullptr)
            trackerConfig.nominalSensorRateHz = 25.0f;
        if (std::getenv("NUMI_TEST_RESET_FRAME") != nullptr)
            trackerConfig.nominalSensorRateHz = std::getenv("NUMI_TEST_LONG_RATE") != nullptr ? 10.0f :
                (std::getenv("NUMI_TEST_RESET_SAMPLED_FAILURE") != nullptr ? 25.0f : 50.0f);
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
        if (std::getenv("NUMI_TEST_WORLD_TRANSACTION") != nullptr) {
            trackerConfig.maskedDepthInstanceIds.clear();
            trackerConfig.maskedDepthFrameOffsets.clear();
            trackerConfig.maximumActorHistoryLength = 32u;
        }
        require(
            tracker.compile(renderer, worlds, std::move(trackerConfig)),
            "device object tracker compile"
        );
        const metalrobo::MetalWorldDeviceObservationProgram
            trackerProgram = tracker.observationProgram();
        require(trackerProgram.valid(), "device object tracker program");
        if (std::getenv("NUMI_TEST_TRACKER_LEASE") != nullptr) {
            id<MTLDevice> testDevice = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> testQueue = [testDevice newCommandQueue];
            id<MTLCommandBuffer> first = [testQueue commandBuffer];
            id<MTLCommandBuffer> second = [testQueue commandBuffer];
            require(first != nil && second != nil && trackerProgram.begin && trackerProgram.finish,
                    "tracker lease setup");
            auto incomplete = trackerProgram;
            incomplete.begin = nullptr;
            require(incomplete.configured() && !incomplete.valid(), "incomplete lease hooks admitted");
            auto unleasedResult = trackerProgram;
            unleasedResult.begin = nullptr;
            unleasedResult.finish = nullptr;
            require(unleasedResult.configured() && !unleasedResult.valid(), "unleased physical-step callback admitted");
            metalrobo::MetalHybridObjectTracker otherTracker;
            metalrobo::MetalHybridObjectTrackerConfig otherConfig;
            otherConfig.capacity = kEnvironmentCount;
            otherConfig.cameraIndex = 0u;
            otherConfig.rootBodyIndex = worldTemplate.engineModel.articulations.front().firstBody;
            otherConfig.maximumActorHistoryLength = kActorHistoryLength;
            otherConfig.bindings.push_back({
                .instanceId = observations[0].identities[visibleIndex].y,
                .actorFrameOffset = 0u,
                .positionScale = 1.0f,
                .velocityScale = 1.0f,
                .minimumVisiblePixels = 1u,
            });
            require(otherTracker.compile(renderer, worlds, std::move(otherConfig)), "second tracker compile");
            const auto otherProgram = otherTracker.observationProgram();
            id<MTLBuffer> foreignBodies = [testDevice newBufferWithBytes:bodyStates.data()
                length:bodyStates.size()*sizeof(MRBodyStateGPU) options:MTLResourceStorageModeShared];
            id<MTLBuffer> foreignReset = [testDevice newBufferWithLength:kEnvironmentCount*sizeof(std::uint32_t)
                options:MTLResourceStorageModeShared];
            constexpr std::size_t foreignValues = kEnvironmentCount*kActorFrameSize*kActorHistoryLength;
            id<MTLBuffer> foreignHistory = [testDevice newBufferWithLength:foreignValues*sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> foreignOutput = [testDevice newBufferWithLength:foreignValues*sizeof(float)
                options:MTLResourceStorageModeShared];
            require(foreignBodies && foreignReset && foreignHistory && foreignOutput, "foreign buffers");
            std::memset(foreignReset.contents, 0, foreignReset.length);
            std::memset(foreignHistory.contents, 0, foreignHistory.length);
            std::memset(foreignOutput.contents, 0, foreignOutput.length);
            metalrobo::MetalWorldDeviceObservationPass foreignPass{
                .commandBuffer=(__bridge void*)second,
                .currentBodies=(__bridge void*)foreignBodies,
                .resetMasks=(__bridge void*)foreignReset,
                .actorHistory=(__bridge void*)foreignHistory,
                .actorObservations=(__bridge void*)foreignOutput,
                .environmentCount=kEnvironmentCount,
                .bodyCount=static_cast<std::uint32_t>(worldTemplate.engineModel.bodies.size()),
                .actorFrameSize=kActorFrameSize,
                .actorHistoryLength=kActorHistoryLength,
                .actorObservationSize=kActorFrameSize*kActorHistoryLength,
            };
            metalrobo::HybridObservationBatch before, after;
            require(renderer.readback(before), "lease initial readback");
            require(trackerProgram.begin(trackerProgram.context, (__bridge void*)first), "lease begin");
            require(!trackerProgram.begin(trackerProgram.context, (__bridge void*)second), "overlapping tracker lease admitted");
            require(!otherProgram.begin(otherProgram.context, (__bridge void*)second), "shared renderer lease admitted");
            require(!otherProgram.encode(otherProgram.context, foreignPass), "foreign unleased encode admitted");
            require(!renderer.readback(after).succeeded(), "provisional renderer metadata was readable");
            require(!tracker.reset().succeeded(), "tracker reset during lease was admitted");
            require(!renderer.renderLive(worlds, liveState, 0u).succeeded(), "competing renderer admitted");
            trackerProgram.finish(trackerProgram.context, (__bridge void*)second, false);
            require(!tracker.reset().succeeded(), "wrong command released tracker lease");
            trackerProgram.finish(trackerProgram.context, (__bridge void*)first, false);
            require(renderer.readback(after), "lease post-abort readback");
            require(std::memcmp(&before.metadata, &after.metadata, sizeof(before.metadata)) == 0,
                    "lease abort changed frame metadata");
            require(trackerProgram.begin(trackerProgram.context, (__bridge void*)second), "tracker lease not released");
            trackerProgram.finish(trackerProgram.context, (__bridge void*)second, false);
            require(otherProgram.encode(otherProgram.context, foreignPass), "foreign encode still rejected after release");
            std::cout << "observation_lease_exclusion_and_abort=pass\n";
            return 0;
        }
        if (std::getenv("NUMI_TEST_WORLD_TRANSACTION") != nullptr) {
            using namespace metalrobo;
            RunManifest manifest;
            manifest.id = "tracker_world_rejection_probe";
            manifest.robot = *builtinRobotPack("franka_panda");
            manifest.scene = makeFrankaPickPlaceScenePack();
            manifest.sensors.id = "franka_default_sensors";
            manifest.task = makeFrankaPickPlaceTaskPack(manifest.sensors.observation, manifest.reality.reset);
            manifest.reality.id = "nominal_reality";
            manifest.profile.id = "franka_probe";
            manifest.profile.environmentCount = kEnvironmentCount;
            manifest.profile.controlSteps = 1u;
            manifest.profile.physicsSubsteps = 4u;
            manifest.profile.controlTimestepSeconds = 0.02f;
            manifest.teacher.id = "no_teacher";
            CompiledRun run;
            auto compile = compileRun(manifest, run);
            require(compile.succeeded(), "Franka run compile: " + compile.message);
            require(run.model().bodies.size() == worldTemplate.engineModel.bodies.size() &&
                    run.model().bodyNames == worldTemplate.engineModel.bodyNames, "body identity mismatch");
            for (std::size_t i=0;i<run.model().bodyNames.size();++i) {
                std::cout << "body_binding " << i << " run=" << run.model().bodyNames[i]
                          << " visual=" << worldTemplate.engineModel.bodyNames[i] << '\n';
            }

            const auto& layout=run.task().layout();
            std::cout << "world_task_frame=" << layout.actorFrameSize << " history=" << layout.actorHistoryLength << " bodies=" << run.model().bodies.size() << '\n';
            require(layout.actorFrameSize >= 7u && layout.actorHistoryLength <= 32u, "incompatible tracker layout");
            std::vector<float> q(kEnvironmentCount * run.world().nq()), v(kEnvironmentCount * run.world().nv(), 0.0f);
            std::vector<MRBodyStateGPU> scene;
            for (std::uint32_t e=0; e<kEnvironmentCount; ++e) {
                std::copy(run.model().defaultQ.begin(), run.model().defaultQ.end(), q.begin()+e*run.world().nq());
                scene.insert(scene.end(), run.defaultSceneBodies().begin(), run.defaultSceneBodies().end());
            }
            std::vector<std::uint32_t> resetMasks(2u*kEnvironmentCount, 0u);
            MetalWorldBatch batch{.environmentCount=kEnvironmentCount, .controlStepCount=1u,
                .initialQ=q, .initialV=v, .policyRevision=1u,
                .resetMasks=std::span<const std::uint32_t>{resetMasks.data(), kEnvironmentCount},
                .initialSceneBodies=scene};
            const char* rejection = std::getenv("NUMI_TEST_REJECT_ACTION");
            struct ActionOwner {
                int rejectStep;
                MetalHybridRenderer* renderer;
                MetalHybridObjectTracker* tracker;
                bool checkedLease = false;
            } action{rejection == nullptr ? -1 : (std::strcmp(rejection, "second") == 0 ? 1 : 0),
                     &renderer, &tracker};
            MetalWorldStepConfig config;
            config.timestepSeconds=0.02f; config.physicsSubsteps=4u;
            config.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
            config.taskProgram=run.task(); config.deviceObservationProgram=trackerProgram;
            struct StepOwner {
                MetalWorldDeviceObservationProgram tracker;
                bool reject = false;
                std::uint32_t calls = 0u;
            } stepOwner{trackerProgram, std::getenv("NUMI_TEST_REJECT_AFTER_STEP") != nullptr};
            const bool checkStep = std::getenv("NUMI_TEST_WORLD_STEP_CALLBACK") != nullptr || stepOwner.reject;
            if (checkStep) {
                config.deviceObservationProgram = {
                    .context = &stepOwner,
                    .encode = [](void* opaque, const MetalWorldDeviceObservationPass& pass) -> bool {
                        auto* owner = static_cast<StepOwner*>(opaque);
                        return owner->tracker.encode(owner->tracker.context, pass);
                    },
                    .begin = [](void* opaque, void* command) noexcept -> bool {
                        auto* owner = static_cast<StepOwner*>(opaque);
                        return owner->tracker.begin(owner->tracker.context, command);
                    },
                    .finish = [](void* opaque, void* command, bool committed) noexcept {
                        auto* owner = static_cast<StepOwner*>(opaque);
                        owner->tracker.finish(owner->tracker.context, command, committed);
                    },
                    .afterStep = [](void* opaque, const MetalWorldDeviceObservationStepResult& pass) -> bool {
                        auto* owner = static_cast<StepOwner*>(opaque);
                        ++owner->calls;
                        return owner->tracker.afterStep(owner->tracker.context, pass) && !owner->reject;
                    },
                };
            }
            config.deviceActionProgram={.context=&action,
                .encode=[](void* opaque, const MetalWorldDeviceActionPass& pass) noexcept -> bool {
                    auto* owner = static_cast<ActionOwner*>(opaque);
                    HybridObservationBatch blocked;
                    if (owner->renderer->readback(blocked).succeeded() || owner->tracker->reset().succeeded()) return false;
                    owner->checkedLease = true;
                    if (owner->rejectStep == static_cast<int>(pass.controlStep)) return false;
                    id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
                    id<MTLBuffer> actions=(__bridge id<MTLBuffer>)pass.taskActions;
                    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
                    if (!blit || !actions) return false;
                    [blit fillBuffer:actions range:NSMakeRange(0, actions.length) value:0];
                    [blit endEncoding]; return true;
                }, .policyRevision=1u, .taskFingerprint=run.task().fingerprint(), .actionCount=layout.actionCount};
            config.ccdMode=MetalWorldCCDMode::disabled;
            MetalWorldContext context;
            MetalWorldResidentState resident;
            const bool residentMode = std::getenv("NUMI_TEST_RESIDENT_REJECT") != nullptr;
            if (residentMode) {
                const int desiredRejection = action.rejectStep;
                action.rejectStep = -1;
                MetalWorldSubmission first;
                auto firstStatus = context.initializeResidentState(run.world(), batch, config, resident, first);
                require(firstStatus.succeeded() && first.valid(), "resident setup submit");
                MetalWorldResult firstResult;
                firstStatus = first.wait(firstResult);
                require(firstStatus.succeeded() && firstStatus.failedStepCount == 0u, "resident setup step");
                action.rejectStep = desiredRejection;
                batch.initialQ = {}; batch.initialV = {}; batch.initialSceneBodies = {};
            }
            if (action.rejectStep >= 0) {
                MetalWorldResidentState discarded; MetalWorldSubmission rejected;
                MetalWorldBatch rejectedBatch = batch;
                if (action.rejectStep == 1) {
                    rejectedBatch.controlStepCount = 2u;
                    rejectedBatch.resetMasks = resetMasks;
                }
                auto r = residentMode
                    ? context.submitResident(run.world(),rejectedBatch,config,resident,rejected)
                    : context.initializeResidentState(run.world(),rejectedBatch,config,discarded,rejected);
                std::cout << "rejected_submit=" << !r.succeeded() << " ticket=" << rejected.valid() << " message=" << r.message << '\n';
                require(!r.succeeded() && !rejected.valid() && action.checkedLease &&
                        r.message.find("device action program rejected") != std::string::npos,
                        "expected later callback rejection with active lease");
                action.rejectStep=-1;
            }
            if (stepOwner.reject) {
                MetalWorldResidentState discarded; MetalWorldSubmission rejected;
                const auto failure = context.initializeResidentState(run.world(), batch, config, discarded, rejected);
                require(!failure.succeeded() && !rejected.valid() && stepOwner.calls == 1u &&
                        failure.message.find("physical-step outcome") != std::string::npos,
                        "expected observation step-result rejection");
                stepOwner.reject = false;
            }
            const std::uint32_t priorStepCalls = stepOwner.calls;
            MetalWorldSubmission submission;
            auto r = residentMode
                ? context.submitResident(run.world(),batch,config,resident,submission)
                : context.initializeResidentState(run.world(),batch,config,resident,submission);
            require(r.succeeded() && submission.valid(), "retry submit: " + r.message);
            MetalWorldResult result; r=submission.wait(result);
            std::cout << "retry_wait=" << r.succeeded() << " failed_steps=" << r.failedStepCount << " message=" << r.message << '\n';
            require(r.succeeded() && r.failedStepCount==0u && action.checkedLease, "retry physical step failed");
            if (checkStep) require(stepOwner.calls == priorStepCalls + 1u, "physical-step callback did not execute");
            HybridObservationBatch rendered; require(renderer.readback(rendered), "retry frame readback");
            std::cout << "world_retry_sensor_sequence=" << rendered.metadata.identity.z << '\n';
            require(rendered.metadata.identity.z==(residentMode ? 1u : 0u) &&
                    rendered.metadata.identity.x==(residentMode ? 1u : 0u) &&
                    rendered.metadata.identity.y==0u &&
                    std::abs(rendered.metadata.timing.x-(residentMode ? 0.02f : 0.0f)) < 1.0e-7f,
                    "rejected world submission advanced sensor sequence, frame or time");
            if (const char* prefix = std::getenv("NUMI_TEST_RESULT_PREFIX")) {
                auto save = [prefix](const char* suffix, const auto& data) {
                    std::ofstream file(std::string(prefix) + suffix, std::ios::binary | std::ios::trunc);
                    require(file.good(), "result file open failed");
                    file.write(reinterpret_cast<const char*>(data.data()),
                               static_cast<std::streamsize>(data.size()*sizeof(data[0])));
                    file.close(); require(file.good(), "result file write failed");
                };
                save(".q", result.finalQ); save(".v", result.finalV);
                save(".scene", result.finalSceneBodies); save(".actor", result.actorObservations);
                save(".status", result.statuses);
                std::cout << "saved_actor_values=" << result.actorObservations.size() << '\n';
                require(!result.actorObservations.empty(), "missing actor observations");
            }
            return 0;
        }

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
        if (std::getenv("NUMI_TEST_DEPTH_HOLD") != nullptr) {
            auto* tasks = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            auto* bodies = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            std::fill_n(masks, kEnvironmentCount, 0u);
            id<MTLBuffer> statuses = [device newBufferWithLength:
                kEnvironmentCount * sizeof(MRMetalWorldStatusGPU)
                options:MTLResourceStorageModeShared];
            require(statuses != nil, "hold statuses");
            std::memset(statuses.contents, 0, statuses.length);
            const auto pixels = [&]() {
                const auto* out = static_cast<const float*>(observationBuffer.contents);
                std::vector<float> values;
                for (std::uint32_t e=0; e<kEnvironmentCount; ++e) {
                    const auto* begin = out + e*kTrackWidth + kActorFrameSize*kActorHistoryLength;
                    values.insert(values.end(), begin, begin + visualPixels);
                }
                return values;
            };
            auto previous = pixels();
            bool held = true, sampledChange = false;
            for (std::uint32_t step=1; step<=3; ++step) {
                for (std::uint32_t e=0; e<kEnvironmentCount; ++e) tasks[e].episode.x=step;
                if (step == 2) bodies[trackedBody].position.x += 0.20f;
                id<MTLCommandBuffer> cb=[queue commandBuffer]; trackerPass.commandBuffer=(__bridge void*)cb;
                require(trackerProgram.begin(trackerProgram.context,trackerPass.commandBuffer),"hold begin");
                require(trackerProgram.encode(trackerProgram.context,trackerPass),"hold encode");
                const metalrobo::MetalWorldDeviceObservationStepResult result{
                    .commandBuffer=trackerPass.commandBuffer,.worldStatuses=(__bridge void*)statuses,
                    .environmentCount=kEnvironmentCount,.actorFrameSize=kActorFrameSize,
                    .actorHistoryLength=kActorHistoryLength,.controlStep=0u,.contactMode=false};
                require(trackerProgram.afterStep(trackerProgram.context,result),"hold afterStep");
                [cb commit];trackerProgram.finish(trackerProgram.context,trackerPass.commandBuffer,true);
                [cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"hold completion");
                metalrobo::HybridObservationBatch capture;require(renderer.readback(capture),"hold metadata");
                auto current=pixels();
                const auto changed = static_cast<std::size_t>(std::inner_product(
                    previous.begin(),previous.end(),current.begin(),std::size_t{0},std::plus<>(),
                    [](float x,float y){return std::memcmp(&x,&y,sizeof(float)) != 0 ? std::size_t{1} : std::size_t{0};}));
                std::cout << "HOLD step=" << step << " sensor=" << capture.metadata.identity.z << " changed_pixels=" << changed << '\n';
                if (step == 2) sampledChange = changed > 0u;
                else held = held && changed == 0u;
                previous=std::move(current);
            }
            require(sampledChange, "fresh sample did not change");
            require(held, "unsampled masked depth changed");
            return 0;
        }
        if (std::getenv("NUMI_TEST_RESET_FRAME") != nullptr) {
            auto* bodies = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            auto* tasks = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            const std::size_t last=(kActorHistoryLength-1u)*kActorFrameSize;
            std::cout.precision(9);
            auto report=[&](int step) {
                metalrobo::HybridObservationBatch capture; require(renderer.readback(capture), "readback");
                const auto* output=static_cast<const float*>(observationBuffer.contents);
                const auto* first=output+kActorFrameSize*kActorHistoryLength;
                const std::size_t env0Far=std::count(first,first+visualPixels,1.0f);
                const auto* features=first+2u*visualPixels;
                float featureAbs=0; for(unsigned i=0;i<MR_TASK_MASKED_DEPTH_FEATURE_COUNT;i++) featureAbs+=std::abs(features[i]);
                std::cout << "CHECK step=" << step << " frame=" << capture.metadata.identity.z
                          << " env0_visible=" << output[last] << " env0_x=" << output[last+1u]
                          << " env0_far=" << env0Far << " pixels=" << visualPixels << " features_abs=" << featureAbs
                          << " env1_visible=" << output[kTrackWidth+last] << " env1_x=" << output[kTrackWidth+last+1u] << '\n';
                require(output[kTrackWidth+last]>0.5f && std::isfinite(output[kTrackWidth+last+1u]),"env1 invalid");
                const bool failedReset = std::getenv("NUMI_TEST_RESET_ROLLBACK") != nullptr;
                bool stale=(std::getenv("NUMI_TEST_LONG_RATE") && step>0 && step<5 && !(failedReset && step==2)) ||
                    (std::getenv("NUMI_TEST_SLOW_RATE") && !std::getenv("NUMI_TEST_LONG_RATE") && step==1) ||
                    (std::getenv("NUMI_TEST_RESET_SAMPLED_FAILURE") && (step==2 || step==3));
                if (stale) {
                    require(output[last]==0.0f && output[last+1u]==0.0f && env0Far==visualPixels && featureAbs==0.0f,"stale reset frame leaked");
                    for (std::uint32_t h=0;h<kActorHistoryLength;++h)
                        require(std::all_of(output+h*kActorFrameSize,output+h*kActorFrameSize+7u,[](float x){return x==0.0f;}),"stale track history leaked");
                    require(std::all_of(first,first+2u*visualPixels,[](float x){return x==1.0f;}),"stale depth history leaked");
                }
                else require(output[last]>0.5f && std::isfinite(output[last+1u]),"fresh env0 track invalid");
            };
            report(0);
            std::fill_n(masks,kEnvironmentCount,0u);
            bodies[trackedBody].position.x += 0.20f;
            id<MTLBuffer> statuses=[device newBufferWithLength:kEnvironmentCount*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses!=nil,"statuses"); std::memset(statuses.contents,0,statuses.length);
            for (std::uint32_t step=1; step<=5; ++step) {
                const bool failedReset = std::getenv("NUMI_TEST_RESET_ROLLBACK") != nullptr;
                const bool sampledFailure = std::getenv("NUMI_TEST_RESET_SAMPLED_FAILURE") != nullptr;
                masks[0]=(sampledFailure ? step==2 : (step==1 || (failedReset && step==3))) ? 1u : 0u;
                auto* codes = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
                codes[0].code = ((failedReset && step==1) || (sampledFailure && step==2))
                    ? MR_STEP_FACTORIZATION_FAILED : MR_STEP_SUCCESS;
                tasks[0].episode.x=step-1u; tasks[0].episode.y=1u; tasks[1].episode.x=step;
                id<MTLCommandBuffer> cb=[queue commandBuffer]; trackerPass.commandBuffer=(__bridge void*)cb;
                require(trackerProgram.begin(trackerProgram.context,trackerPass.commandBuffer),"begin");
                require(trackerProgram.encode(trackerProgram.context,trackerPass),"encode");
                const metalrobo::MetalWorldDeviceObservationStepResult outcome{
                    .commandBuffer=trackerPass.commandBuffer,.worldStatuses=(__bridge void*)statuses,
                    .environmentCount=kEnvironmentCount,.actorFrameSize=kActorFrameSize,
                    .actorHistoryLength=kActorHistoryLength,.controlStep=0u,.contactMode=false};
                require(trackerProgram.afterStep(trackerProgram.context,outcome),"afterStep");
                [cb commit];trackerProgram.finish(trackerProgram.context,trackerPass.commandBuffer,true);
                [cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"completion");
                report(step);
            }
            return 0;
        }
        if (std::getenv("NUMI_TEST_TRACKER_STEP_ROLLBACK") != nullptr) {
            require(trackerProgram.begin && trackerProgram.finish && trackerProgram.afterStep,
                    "tracker physical-step hooks missing");
            const std::vector<float> initial(tracked, tracked + trackElements);
            constexpr std::size_t kOutputPrefix = 13u;
            constexpr std::size_t kOutputSuffix = 17u;
            id<MTLBuffer> offsetOutput = [device newBufferWithLength:
                (kOutputPrefix + trackElements + kOutputSuffix) * sizeof(float)
                options:MTLResourceStorageModeShared];
            require(offsetOutput != nil, "offset output allocation");
            auto* offsetValues = static_cast<float*>(offsetOutput.contents);
            std::fill_n(offsetValues, kOutputPrefix + trackElements + kOutputSuffix, -73.0f);
            std::copy(initial.begin(), initial.end(), offsetValues + kOutputPrefix);
            trackerPass.actorObservations = (__bridge void*)offsetOutput;
            trackerPass.actorObservationOffsetElements = kOutputPrefix;
            const auto* actorHistoryValues = static_cast<const float*>(historyBuffer.contents);
            const std::size_t actorStride = kActorFrameSize * kActorHistoryLength;
            const std::vector<float> initialHistory(actorHistoryValues, actorHistoryValues + kEnvironmentCount * actorStride);
            auto* bodyValues = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            auto* taskValues = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            std::fill_n(masks, kEnvironmentCount, 0u);
            id<MTLBuffer> statuses = [device newBufferWithLength:
                kEnvironmentCount * sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses != nil, "step status allocation");
            auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
            std::memset(statusValues, 0, statuses.length);
            statusValues[0].code = MR_STEP_FACTORIZATION_FAILED;
            auto observeStep = [&](std::uint32_t episodeStep) {
                for (std::uint32_t e=0u; e<kEnvironmentCount; ++e) {
                    bodyValues[static_cast<std::size_t>(e)*worldTemplate.engineModel.bodies.size()+trackedBody].position.x += 0.02f;
                    taskValues[e].episode.x = episodeStep;
                }
                id<MTLCommandBuffer> stepCommand = [queue commandBuffer];
                require(stepCommand != nil, "step command allocation");
                trackerPass.commandBuffer = (__bridge void*)stepCommand;
                require(trackerProgram.begin(trackerProgram.context, trackerPass.commandBuffer), "step begin");
                auto outOfRange = trackerPass;
                outOfRange.actorObservationOffsetElements = kOutputPrefix + trackElements;
                require(!trackerProgram.encode(trackerProgram.context, outOfRange), "out-of-range actor output admitted");
                outOfRange.actorObservationOffsetElements = static_cast<std::size_t>(-1);
                require(!trackerProgram.encode(trackerProgram.context, outOfRange), "overflow actor output admitted");
                outOfRange = trackerPass;
                outOfRange.controlStep = 2u;
                require(!trackerProgram.encode(trackerProgram.context, outOfRange), "out-of-range reset mask admitted");
                require(trackerProgram.encode(trackerProgram.context, trackerPass), "step observation");
                const metalrobo::MetalWorldDeviceObservationStepResult outcome{
                    .commandBuffer = trackerPass.commandBuffer,
                    .worldStatuses = (__bridge void*)statuses,
                    .contactStatuses = nullptr,
                    .environmentCount = kEnvironmentCount,
                    .actorFrameSize = kActorFrameSize,
                    .actorHistoryLength = kActorHistoryLength,
                    .controlStep = 0u,
                    .contactMode = false,
                };
                auto invalidOutcome = outcome;
                invalidOutcome.actorFrameSize += 1u;
                require(!trackerProgram.afterStep(trackerProgram.context, invalidOutcome), "mismatched history layout admitted");
                invalidOutcome = outcome;
                invalidOutcome.controlStep += 1u;
                require(!trackerProgram.afterStep(trackerProgram.context, invalidOutcome), "mismatched step admitted");
                require(trackerProgram.afterStep(trackerProgram.context, outcome), "step result");
                require(!trackerProgram.afterStep(trackerProgram.context, outcome), "duplicate step result admitted");
                [stepCommand commit];
                trackerProgram.finish(trackerProgram.context, trackerPass.commandBuffer, true);
                [stepCommand waitUntilCompleted];
                require(stepCommand.status == MTLCommandBufferStatusCompleted, "step command completion");
                require(std::all_of(offsetValues, offsetValues + kOutputPrefix, [](float x){return x == -73.0f;}) &&
                        std::all_of(offsetValues + kOutputPrefix + trackElements,
                                    offsetValues + kOutputPrefix + trackElements + kOutputSuffix,
                                    [](float x){return x == -73.0f;}), "actor output canary changed");
                const auto* values = offsetValues + kOutputPrefix;
                return std::vector<float>(values, values + trackElements);
            };
            const auto middle = observeStep(1u);
            require(std::equal(initial.begin(), initial.begin()+kTrackWidth, middle.begin()),
                    "failed environment actor observation was not restored");
            require(std::equal(initialHistory.begin(), initialHistory.begin()+actorStride, actorHistoryValues),
                    "failed environment actor history was not restored");
            require(!std::equal(initialHistory.begin()+actorStride, initialHistory.end(), actorHistoryValues+actorStride),
                    "successful environment actor history did not advance");
            std::memset(statusValues, 0, statuses.length);
            const auto next = observeStep(2u);
            for (std::uint32_t e=0u; e<kEnvironmentCount; ++e) {
                const std::size_t base = static_cast<std::size_t>(e) * kTrackWidth;
                const std::size_t initialLast = base + (kActorHistoryLength-1u)*kActorFrameSize;
                const std::size_t previousLast = base + (kActorHistoryLength-2u)*kActorFrameSize;
                const auto& expected = e == 0u ? initial : middle;
                require(std::equal(next.begin()+previousLast, next.begin()+previousLast+4u,
                                   expected.begin()+initialLast), "object history did not follow per-environment physical status");
                const std::size_t depthCurrent = base + kActorFrameSize*kActorHistoryLength;
                const std::size_t depthPrevious = depthCurrent + visualPixels;
                require(std::equal(next.begin()+depthPrevious, next.begin()+depthPrevious+visualPixels,
                                   expected.begin()+depthCurrent), "masked-depth history did not follow per-environment physical status");
            }
            const std::size_t acceptedLast = kTrackWidth+(kActorHistoryLength-1u)*kActorFrameSize;
            require(!std::equal(initial.begin()+acceptedLast, initial.begin()+acceptedLast+4u,
                                middle.begin()+acceptedLast), "accepted object history did not change");
            const std::size_t acceptedDepth = kTrackWidth+kActorFrameSize*kActorHistoryLength;
            require(!std::equal(initial.begin()+acceptedDepth, initial.begin()+acceptedDepth+visualPixels,
                                middle.begin()+acceptedDepth), "accepted depth history did not change");
            std::cout << "per_environment_actor_object_and_masked_depth_rollback=pass\n";
            return 0;
        }
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
        require(beforeRejectedEncode.metadata.identity.z == 0u &&
                    beforeRejectedEncode.metadata.identity.x == 0u &&
                    beforeRejectedEncode.metadata.identity.y == 0u &&
                    beforeRejectedEncode.metadata.timing.x == 0.0f,
                "first tracker frame identity or time is not zero");
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
        std::cout << "retry_frame=" << retryFrame.metadata.identity.x
                  << " retry_time=" << retryFrame.metadata.timing.x << std::endl;
        require(retryFrame.metadata.identity.x == 1u &&
                    retryFrame.metadata.identity.y == 0u &&
                    std::abs(retryFrame.metadata.timing.x - 0.02f) < 1.0e-7f,
                "tracker frame identity or time reset at submission boundary");
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
        if (std::getenv("NUMI_TEST_DIRECT_RESET_FRAME") != nullptr) directConfig.nominalSensorRateHz = 10.0f;
        directConfig.maskedDepthInstanceIds = {
            observations[0].identities[visibleIndex].y};
        directConfig.maskedDepthWidth = 16u;
        directConfig.maskedDepthHeight = 9u;
        directConfig.maskedDepthActorFrameOffset =
            kActorFrameSize * kActorHistoryLength;
        directConfig.maskedDepthFrameOffsets = {0u, 1u};
        directConfig.maskedDepthNearMeters = 0.1f;
        directConfig.maskedDepthFarMeters = 5.0f;
        metalrobo::MetalWorldFamilyContext directSmallWorlds;
        if (std::getenv("NUMI_TEST_DIRECT_RESET_FRAME") != nullptr) {
            auto directEpisode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
            directEpisode.sensors[0].width = 16u;
            directEpisode.sensors[0].height = 9u;
            directEpisode.sensors[0].intrinsics = {14.0f, 10.5f, 8.0f, 4.5f};
            metalrobo::WorldTemplate directTemplate;
            require(metalrobo::compileEpisodeTwin(directEpisode, metalrobo::makeFrankaPickPlaceEngineModel(),
                        directTemplate), "direct small camera episode");
            metalrobo::WorldFamily directFamily;
            require(metalrobo::compileWorldFamily(directTemplate, metalrobo::makeFrankaPickPlaceWorldProgram(),
                        directFamily), "direct small camera family");
            require(directSmallWorlds.compile(directFamily,kEnvironmentCount), "direct small camera world");
            require(directSmallWorlds.sample(kEnvironmentCount,0x5eed1234ull), "direct small camera sample");
        }
        metalrobo::MetalHybridObjectTracker directTracker;
        const auto& directTrackerWorlds = std::getenv("NUMI_TEST_DIRECT_RESET_FRAME") != nullptr
            ? directSmallWorlds : worlds;
        require(directTracker.compile(directRenderer, directTrackerWorlds, std::move(directConfig)),
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
        if (std::getenv("NUMI_TEST_DIRECT_RESET_FRAME") != nullptr) {
            auto* directMasks=static_cast<std::uint32_t*>(resetBuffer.contents);
            auto* directTasks=static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* directValues=static_cast<float*>(observationBuffer.contents);
            constexpr std::size_t pixels=16u*9u;
            const std::size_t offset=kActorFrameSize*kActorHistoryLength;
            const auto old0=std::vector<float>(directValues+offset,directValues+offset+pixels);
            const auto old1=std::vector<float>(directValues+kTrackWidth+offset,directValues+kTrackWidth+offset+pixels);
            require(std::any_of(old0.begin(),old0.end(),[](float x){return x<1.0f;}), "direct initial scene has no returns");
            std::fill_n(directMasks,kEnvironmentCount,0u);
            deviceBodies[trackedBody].position.x += 0.2f;
            id<MTLBuffer> statuses=[device newBufferWithLength:kEnvironmentCount*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses!=nil,"direct status buffer"); std::memset(statuses.contents,0,statuses.length);
            for (std::uint32_t step=1;step<=5;++step) {
                directMasks[0]=step==1 ? 1u : 0u;
                directTasks[0].episode.x=step-1;directTasks[0].episode.y=1u;directTasks[1].episode.x=step;
                id<MTLCommandBuffer> cb=[queue commandBuffer];directPass.commandBuffer=(__bridge void*)cb;
                require(directProgram.begin(directProgram.context,directPass.commandBuffer),"direct reset begin");
                require(directProgram.encode(directProgram.context,directPass),"direct reset encode");
                const metalrobo::MetalWorldDeviceObservationStepResult outcome{
                    .commandBuffer=directPass.commandBuffer,.worldStatuses=(__bridge void*)statuses,
                    .environmentCount=kEnvironmentCount,.actorFrameSize=kActorFrameSize,
                    .actorHistoryLength=kActorHistoryLength,.controlStep=0u,.contactMode=false};
                require(directProgram.afterStep(directProgram.context,outcome),"direct reset result");
                [cb commit];directProgram.finish(directProgram.context,directPass.commandBuffer,true);
                [cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"direct reset completion");
                const float* d0=directValues+offset;
                const float* d1=directValues+kTrackWidth+offset;
                if (step<5) {
                    require(std::all_of(d0,d0+pixels,[](float x){return x==1.0f;}),"direct stale reset depth leaked");
                    require(std::equal(old1.begin(),old1.end(),d1),"direct other environment changed without sampling");
                } else {
                    require(std::any_of(d0,d0+pixels,[](float x){return x<1.0f;}),"direct fresh depth absent");
                    require(!std::equal(old0.begin(),old0.end(),d0),"direct reset recovery reused old frame");
                }
            }
            std::cout << "direct_reset_stale_depth_suppressed=pass\n";
            return 0;
        }
        id<MTLCommandBuffer> directDiscarded = [queue commandBuffer];
        auto directDiscardedPass = directPass;
        directDiscardedPass.commandBuffer = (__bridge void*)directDiscarded;
        require(directProgram.begin(directProgram.context, (__bridge void*)directDiscarded),
                "direct-winner lease begin");
        require(directProgram.encode(directProgram.context, directDiscardedPass),
                "direct-winner provisional encode");
        directProgram.finish(directProgram.context, (__bridge void*)directDiscarded, false);
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
        auto dropoutScene = boundObjectScene;
        auto dropoutReferenceScene = boundObjectScene;
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
        auto dropoutConfig = boundConfig;
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
        // Keep the same geometry and transforms, but force camera depth
        // dropout. Geometry is still present while no depth is measurable.
        auto dropoutEpisode = smallEpisode;
        dropoutEpisode.sensors[0].depthDropout = 1.0f;
        metalrobo::WorldTemplate dropoutTemplate;
        require(metalrobo::compileEpisodeTwin(
                    dropoutEpisode, metalrobo::makeFrankaPickPlaceEngineModel(),
                    dropoutTemplate), "dropout episode compile");
        metalrobo::WorldFamily dropoutFamily;
        require(metalrobo::compileWorldFamily(
                    dropoutTemplate, metalrobo::makeFrankaPickPlaceWorldProgram(),
                    dropoutFamily), "dropout family compile");
        metalrobo::MetalWorldFamilyContext dropoutWorlds;
        require(dropoutWorlds.compile(dropoutFamily, kEnvironmentCount),
                "dropout world compile");
        require(dropoutWorlds.sample(kEnvironmentCount, 0x5eed1234ull),
                "dropout world sample");
        metalrobo::MetalHybridRenderer dropoutReferenceRenderer(boundRendererConfig);
        require(dropoutReferenceRenderer.compile(std::move(dropoutReferenceScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    kEnvironmentCount), "dropout reference compile");
        require(dropoutReferenceRenderer.renderLive(dropoutWorlds, referenceLive, 0u),
                "dropout reference render");
        metalrobo::HybridObservationBatch dropoutReference;
        require(dropoutReferenceRenderer.readback(dropoutReference),
                "dropout reference readback");
        for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
            std::size_t geometry = 0u, validDepth = 0u;
            for (std::size_t pixel = 0u; pixel < kSmallPixels; ++pixel) {
                const auto index = env * kSmallPixels + pixel;
                if (dropoutReference.identities[index].y == boundInstance &&
                    (dropoutReference.validity[index] & MR_VISUAL_VALIDITY_GEOMETRY)) {
                    ++geometry;
                    validDepth += (dropoutReference.validity[index] &
                                   MR_VISUAL_VALIDITY_DEPTH) != 0u;
                }
            }
            std::cout << "dropout_reference_env=" << env << " geometry=" << geometry
                      << " valid_depth=" << validDepth << '\n';
            require(geometry > 0u && validDepth == 0u,
                    "dropout reference must have geometry without depth");
        }
        metalrobo::MetalHybridRenderer dropoutRenderer(boundRendererConfig);
        require(dropoutRenderer.compile(std::move(dropoutScene),
                    metalrobo::VisualRendererProfileV1::sensorFast(),
                    kEnvironmentCount), "dropout tracker renderer compile");
        metalrobo::MetalHybridObjectTracker dropoutTracker;
        require(dropoutTracker.compile(dropoutRenderer, dropoutWorlds,
                                       std::move(dropoutConfig)),
                "dropout tracker compile");
        const auto dropoutProgram = dropoutTracker.observationProgram();
        std::fill_n(static_cast<float*>(boundObservations.contents),
                    kBoundOffset + trackElements, -5.0f);
        std::memset(historyBuffer.contents, 0, historyBuffer.length);
        std::fill_n(static_cast<std::uint32_t*>(resetBuffer.contents),
                    kEnvironmentCount, 1u);
        id<MTLCommandBuffer> dropoutCommand = [queue commandBuffer];
        auto dropoutPass = boundPass;
        dropoutPass.commandBuffer = (__bridge void*)dropoutCommand;
        require(dropoutProgram.valid() &&
                    dropoutProgram.encode(dropoutProgram.context, dropoutPass),
                "dropout tracker encode");
        [dropoutCommand commit];
        [dropoutCommand waitUntilCompleted];
        require(dropoutCommand.status == MTLCommandBufferStatusCompleted,
                "dropout tracker completion");
        const float* dropoutOutput =
            static_cast<const float*>(boundObservations.contents) + kBoundOffset;
        const float* dropoutHistory = static_cast<const float*>(historyBuffer.contents);
        for (std::uint32_t env = 0u; env < kEnvironmentCount; ++env) {
            const float* frame = dropoutOutput + env * kTrackWidth;
            const float* history = dropoutHistory +
                env * kActorFrameSize * kActorHistoryLength;
            const float* depth = frame + kActorFrameSize * kActorHistoryLength;
            std::cout << "dropout_env=" << env << " confidence=" << frame[0]
                      << " history_confidence=" << history[0] << '\n';
            require(std::all_of(frame, frame + kActorFrameSize * kActorHistoryLength,
                                [](float value) { return value == 0.0f; }) &&
                    std::all_of(history, history + kActorFrameSize * kActorHistoryLength,
                                [](float value) { return value == 0.0f; }) &&
                    std::all_of(depth, depth + 2u * kSmallPixels,
                                [](float value) { return value == 1.0f; }),
                    "dropout produced a spurious object track or depth");
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
