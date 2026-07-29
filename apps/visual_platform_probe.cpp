#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/VisualPlatform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
        result += (validity & 4u) != 0u ? 1u : 0u;
    }
    return result;
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
        metalrobo::WorldFamily family;
        require(
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            ),
            "family compile"
        );

        metalrobo::VisualSceneManifestV1 visualScene;
        std::string reason;
        require(
            metalrobo::compileVisualSceneManifest(
                worldTemplate,
                visualScene,
                &reason
            ),
            "visual scene compile: " + reason
        );
        require(
            visualScene.valid(&reason) &&
                !visualScene.renderScene.meshVertices.empty() &&
                !visualScene.renderScene.meshTriangles.empty() &&
                visualScene.renderScene.sensorBindings.size() == 2u &&
                visualScene.bodyCount ==
                    worldTemplate.engineModel.bodies.size(),
            "compiled visual scene is incomplete: " + reason
        );
        const std::array<MRHybridGaussianGPU, 1> capturedAppearance{{
            {
                {0.45f, 0.0f, 0.08f, 0.65f},
                {0.025f, 0.025f, 0.025f, 1.0f},
                {0.0f, 0.0f, 0.0f, 1.0f},
                {0.15f, 0.45f, 0.95f, 0.0f},
                {
                    0u,
                    MR_INVALID_INDEX,
                    0u,
                    MR_HYBRID_GAUSSIAN_ASSET_LOCAL,
                },
            },
        }};
        require(
            metalrobo::attachGaussianField(
                visualScene,
                "workspace",
                capturedAppearance,
                "capture://probe/workspace.gsplat",
                "sha256:probe-gaussian-layer",
                "probe-only",
                "gaussian-capture-probe-v1",
                &reason
            ),
            "Gaussian appearance attachment: " + reason
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
                kEnvironmentCount * visualScene.bodyCount,
            "visual body state does not match global body indexing"
        );

        metalrobo::MetalHybridRenderer renderer;
        require(
            renderer.compile(
                visualScene.renderScene,
                kEnvironmentCount
            ),
            "visual sensor runtime compile"
        );
        metalrobo::HybridLiveStateBatch liveState;
        liveState.environmentCount = kEnvironmentCount;
        liveState.bodyCount = visualScene.bodyCount;
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
            observations[0].validity.begin(),
            observations[0].validity.end(),
            [](const std::uint32_t value) {
                return (value & 4u) != 0u;
            }
        );
        const std::size_t visibleIndex =
            static_cast<std::size_t>(
                firstVisible -
                observations[0].validity.begin()
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

        metalrobo::VisualSensorProfileV1 sensorProfile;
        sensorProfile.id = "franka.rgbd.reference";
        sensorProfile.frameJitterSeconds = 0.0005;
        sensorProfile.fingerprint =
            metalrobo::computeVisualSensorProfileFingerprint(
                sensorProfile
            );
        const std::array<std::uint32_t, 2> cameras{0u, 1u};
        metalrobo::VisualBatchAssemblyV1 assembly;
        assembly.provenance = {
            MR_VISUAL_SOURCE_SIMULATION,
            worldTemplate.fingerprint,
            family.program.fingerprint,
            visualScene.fingerprint,
            sensorProfile.fingerprint,
            family.fingerprint ^ 0x5aa55aa55aa55aa5ull,
        };
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
        episode.rendererFingerprint = visualScene.fingerprint;
        episode.visualSceneFingerprint =
            visualScene.fingerprint;
        episode.sensorProfileFingerprint =
            assembly.provenance.sensorProfileFingerprint;
        episode.calibrationFingerprint =
            assembly.provenance.calibrationFingerprint;
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
            << visualScene.renderScene.meshTriangles.size()
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
