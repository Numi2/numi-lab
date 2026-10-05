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
#include <system_error>
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

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::VisualAssetPackV2 makeOffscreenCubePack() {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = "human_resting_deformation_probe_cube";
    pack.sourceUri = "probe://human_resting/deformation_cube";
    pack.sourceContentHash = "sha256:human-resting-deformation-probe";
    pack.license = "CC0-1.0";
    pack.preprocessingProvenance =
        "human_resting_mesh_deformation_probe/v1";

    constexpr std::array positions{
        std::array{-0.06f, -1.06f, -0.06f},
        std::array{ 0.06f, -1.06f, -0.06f},
        std::array{ 0.06f, -0.94f, -0.06f},
        std::array{-0.06f, -0.94f, -0.06f},
        std::array{-0.06f, -1.06f,  0.06f},
        std::array{ 0.06f, -1.06f,  0.06f},
        std::array{ 0.06f, -0.94f,  0.06f},
        std::array{-0.06f, -0.94f,  0.06f},
    };
    for (const auto& position : positions) {
        pack.vertices.push_back({
            {position[0], position[1], position[2], 1.0f},
            {0.0f, 0.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
            {0.8f, 0.12f, 0.04f, 1.0f},
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
    material.baseColorAndOpacity = {0.8f, 0.12f, 0.04f, 1.0f};
    material.surface = {0.4f, 0.0f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {0.0f, 0.0f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX, MR_INVALID_INDEX,
        MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    material.textureIndices1 = material.textureIndices0;
    material.reserved = material.textureIndices0;
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED,
        0u,
        1u,
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
    primitive.boundsMinimum = {-0.06f, -1.06f, -0.06f, 1.0f};
    primitive.boundsMaximum = { 0.06f, -0.94f,  0.06f, 1.0f};
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

constexpr char kDeformationShader[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ProbeVertex {
    float4 position;
    float4 normalAndTangentSign;
    float4 tangent;
    float4 texcoord01;
    float4 color;
};

struct ProbeInstance {
    float4 translationAndScale;
    float4 orientation;
    uint4 binding;
    uint4 identity;
    uint4 geometry;
};

struct AcceptedDeformationState {
    float lateralDisplacementMeters;
    float reserved;
    uint visibleFlagMask;
    uint visibleFlagValue;
};

kernel void deform_accepted_human_mesh(
    device ProbeVertex* vertices [[buffer(0)]],
    device ProbeInstance* instances [[buffer(1)]],
    device const AcceptedDeformationState* acceptedState [[buffer(2)]],
    constant uint& vertexCount [[buffer(3)]],
    const uint index [[thread_position_in_grid]]) {
    if (index < vertexCount) {
        ProbeVertex record = vertices[index];
        record.position.y += acceptedState[0].lateralDisplacementMeters;
        record.normalAndTangentSign = float4(
            0.0f, 0.0f, 1.0f, record.normalAndTangentSign.w);
        vertices[index] = record;
    }
    if (index == 0u) {
        const uint oldFlags = instances[0].binding.w;
        instances[0].binding.w =
            (oldFlags & ~acceptedState[0].visibleFlagMask) |
            (acceptedState[0].visibleFlagValue &
             acceptedState[0].visibleFlagMask);
    }
}
)METAL";

struct AcceptedDeformationState {
    float lateralDisplacementMeters = 0.0f;
    float reserved = 0.0f;
    std::uint32_t visibleFlagMask = 0u;
    std::uint32_t visibleFlagValue = 0u;
};

struct CallbackContext {
    __strong id<MTLComputePipelineState> pipeline = nil;
    std::uint32_t invocationCount = 0u;
};

bool encodeDeformation(
    void* opaqueContext,
    const metalrobo::MetalHybridMeshDeformationLease& lease
) {
    auto* context = static_cast<CallbackContext*>(opaqueContext);
    if (context == nullptr || context->pipeline == nil ||
        lease.encoder == nullptr || !lease.encoder->valid() ||
        lease.acceptedStateBuffer == nullptr ||
        lease.acceptedStateByteCount < 16u ||
        lease.meshVertexCount == 0u || lease.meshInstanceCount == 0u) {
        return false;
    }
    const auto& encoder = *lease.encoder;
    encoder.setPipeline(
        encoder.context,
        (__bridge void*)context->pipeline
    );
    encoder.setBuffer(encoder.context, lease.meshVertices, 0u, 0u);
    encoder.setBuffer(encoder.context, lease.meshInstances, 0u, 1u);
    encoder.setBuffer(
        encoder.context,
        lease.acceptedStateBuffer,
        lease.acceptedStateOffset,
        2u
    );
    encoder.setBytes(
        encoder.context,
        &lease.meshVertexCount,
        sizeof(lease.meshVertexCount),
        3u
    );
    encoder.dispatchThreads(
        encoder.context,
        std::max(lease.meshVertexCount, lease.meshInstanceCount),
        32u
    );
    ++context->invocationCount;
    return true;
}

metalrobo::MetalHybridRendererDiagnostics encodeFrame(
    metalrobo::MetalHybridRenderer& renderer,
    const metalrobo::MetalWorldFamilyContext& worlds,
    const metalrobo::HybridDeviceStateBatch& state,
    const std::uint32_t cameraIndex,
    id<MTLCommandQueue> queue
) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command computeCommandEncoder];
    require(command != nil && encoder != nil,
        "could not create borrowed renderer command");
    auto diagnostics = renderer.encode(
        worlds,
        state,
        cameraIndex,
        (__bridge void*)encoder
    );
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (diagnostics.succeeded() &&
        command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("renderer command did not complete");
    }
    return diagnostics;
}

std::size_t semanticPixelCount(
    const metalrobo::HybridObservationBatch& observations,
    const std::uint32_t semantic
) {
    return static_cast<std::size_t>(std::count(
        observations.segmentation.begin(),
        observations.segmentation.end(),
        semantic
    ));
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr std::uint32_t environmentCount = 1u;
            constexpr std::uint32_t semantic = 77u;
            std::string reason;
            const metalrobo::EngineModel model =
                metalrobo::makeFrankaPickPlaceEngineModel();
            metalrobo::WorldTemplate worldTemplate;
            require(metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                model,
                worldTemplate
            ), "episode compile");
            metalrobo::WorldFamily family;
            require(metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            ), "family compile");
            metalrobo::MetalWorldFamilyContext worlds;
            require(worlds.compile(family, environmentCount),
                "world compile");
            require(worlds.sample(environmentCount, 0x5eed1234ull),
                "world sample");

            const std::filesystem::path packPath =
                std::filesystem::temp_directory_path() /
                ("numi-human-resting-deformation-" +
                    std::to_string(
                        std::chrono::steady_clock::now()
                            .time_since_epoch().count()
                    ) + ".mrvpack");
            const auto pack = makeOffscreenCubePack();
            require(metalrobo::writeVisualAssetPack(
                pack,
                packPath,
                &reason
            ), "visual asset pack write");
            const std::array references{
                metalrobo::VisualAssetReferenceV3{
                    packPath,
                    pack.contentHash,
                    worldTemplate.assetIndex("pick_object"),
                    semantic,
                    7001u,
                },
            };
            metalrobo::VisualSceneManifestV3 manifest;
            require(metalrobo::compileVisualSceneManifestV3(
                worldTemplate,
                references,
                metalrobo::makeNeutralStudioEnvironmentV2(),
                metalrobo::makeIndoorAreaLightRigV1(),
                manifest,
                &reason
            ), "visual scene compile");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (device == nil || queue == nil) {
                throw std::runtime_error("Metal device or queue unavailable");
            }
            NSError* shaderError = nil;
            NSString* shaderSource = [NSString
                stringWithUTF8String:kDeformationShader];
            id<MTLLibrary> testLibrary = [device
                newLibraryWithSource:shaderSource
                options:nil
                error:&shaderError];
            if (testLibrary == nil) {
                NSString* detail = shaderError == nil
                    ? @"unknown Metal compiler error"
                    : shaderError.localizedDescription;
                throw std::runtime_error(
                    std::string{"deformation validation shader failed: "} +
                    detail.UTF8String
                );
            }
            id<MTLFunction> deformFunction = [testLibrary
                newFunctionWithName:@"deform_accepted_human_mesh"];
            if (deformFunction == nil) {
                throw std::runtime_error(
                    "deformation validation shader function is missing"
                );
            }
            CallbackContext callbackContext;
            callbackContext.pipeline = [device
                newComputePipelineStateWithFunction:deformFunction
                error:&shaderError];
            if (callbackContext.pipeline == nil) {
                NSString* detail = shaderError == nil
                    ? @"unknown Metal pipeline error"
                    : shaderError.localizedDescription;
                throw std::runtime_error(
                    std::string{"deformation validation pipeline failed: "} +
                    detail.UTF8String
                );
            }

            metalrobo::MetalHybridRendererConfig config;
            config.metallibPath = METALROBO_METALLIB;
            config.width = 96u;
            config.height = 72u;
            metalrobo::MetalHybridRenderer renderer(config);
            const auto rendererCompile = renderer.compile(
                std::move(manifest.renderScene),
                metalrobo::VisualRendererProfileV1::sensorFast(),
                environmentCount
            );
            require(rendererCompile, "renderer compile");
            std::error_code ignored;
            std::filesystem::remove(packPath, ignored);

            std::vector<MRBodyStateGPU> bodyStates(model.bodies.size());
            for (MRBodyStateGPU& body : bodyStates) {
                body.orientation.w = 1.0f;
            }
            require(bodyStates.size() > 11u,
                "probe body model lacks the authored object body");
            bodyStates[11u].position = {0.42f, -0.08f, 0.08f, 0.0f};
            id<MTLBuffer> bodyBuffer = [device
                newBufferWithBytes:bodyStates.data()
                length:bodyStates.size() * sizeof(MRBodyStateGPU)
                options:MTLResourceStorageModeShared];
            AcceptedDeformationState acceptedState{
                0.92f,
                0.0f,
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
            };
            id<MTLBuffer> acceptedStateBuffer = [device
                newBufferWithBytes:&acceptedState
                length:sizeof(acceptedState)
                options:MTLResourceStorageModeShared];
            require(bodyBuffer != nil && acceptedStateBuffer != nil,
                "probe input-buffer allocation failed");

            metalrobo::HybridDeviceStateBatch state;
            state.currentBodyStates = (__bridge void*)bodyBuffer;
            state.previousBodyStates = (__bridge void*)bodyBuffer;
            state.environmentCount = environmentCount;
            state.bodyCount = static_cast<std::uint32_t>(bodyStates.size());
            state.frameIndex = 1u;
            state.sensorSequence = 1u;
            state.source = MR_VISUAL_SOURCE_SIMULATION;
            state.acceptedRootFingerprint = 0x11u;
            state.acceptedTransactionFingerprint = 0x22u;
            state.acceptedTimestampMicroseconds = 1'000'000u;

            metalrobo::HybridObservationBatch unchangedBefore;
            require(encodeFrame(renderer, worlds, state, 0u, queue),
                "disabled baseline encode");
            require(renderer.readback(unchangedBefore),
                "disabled baseline readback");
            const std::size_t baselinePixels =
                semanticPixelCount(unchangedBefore, semantic);
            require(baselinePixels == 0u,
                "off-screen authored mesh unexpectedly rendered");

            metalrobo::MetalHybridMeshDeformationRequest request;
            request.encode = encodeDeformation;
            request.context = &callbackContext;
            request.acceptedStateIsCommitted = true;
            request.acceptedStateBuffer =
                (__bridge void*)acceptedStateBuffer;
            request.acceptedStateByteCount = sizeof(acceptedState);
            request.acceptedRootFingerprint = 0x99u;
            request.acceptedTransactionFingerprint = 0x22u;
            request.acceptedTimestampMicroseconds = 1'000'000u;
            request.expectedEnvironmentCount = environmentCount;
            const auto layout = renderer.layout();
            request.expectedMeshVertexCount = layout.meshVertexCount;
            request.expectedMeshIndexCount = layout.meshIndexCount;
            request.expectedMeshTriangleCount = layout.meshTriangleCount;
            request.expectedMeshPrimitiveCount = layout.meshPrimitiveCount;
            request.expectedMeshInstanceCount = layout.meshInstanceCount;
            state.meshDeformation = &request;
            const auto stale = encodeFrame(renderer, worlds, state, 0u, queue);
            require(
                stale.status ==
                    metalrobo::MetalHybridRendererStatus::invalidConfiguration &&
                    callbackContext.invocationCount == 0u,
                "stale accepted-root deformation was not rejected before callback"
            );

            state.meshDeformation = nullptr;
            ++state.frameIndex;
            ++state.sensorSequence;
            metalrobo::HybridObservationBatch unchangedAfter;
            require(encodeFrame(renderer, worlds, state, 0u, queue),
                "disabled post-rejection encode");
            require(renderer.readback(unchangedAfter),
                "disabled post-rejection readback");
            require(
                unchangedAfter.segmentation == unchangedBefore.segmentation &&
                    semanticPixelCount(unchangedAfter, semantic) ==
                        baselinePixels,
                "disabled deformation changed accepted mesh geometry"
            );

            request.acceptedRootFingerprint = state.acceptedRootFingerprint;
            request.acceptedTransactionFingerprint =
                state.acceptedTransactionFingerprint;
            request.acceptedTimestampMicroseconds =
                state.acceptedTimestampMicroseconds;
            state.meshDeformation = &request;
            ++state.frameIndex;
            ++state.sensorSequence;
            require(encodeFrame(renderer, worlds, state, 0u, queue),
                "accepted deformation encode");
            require(callbackContext.invocationCount == 1u,
                "accepted deformation callback was not invoked exactly once");
            metalrobo::HybridObservationBatch deformed;
            require(renderer.readback(deformed),
                "deformed geometry readback");
            const std::size_t deformedPixels =
                semanticPixelCount(deformed, semantic);
            require(deformedPixels > 0u,
                "accepted GPU deformation did not move mesh into view; "
                "cluster bounds were not rebuilt");

            std::cout << "device=\"" << rendererCompile.deviceName
                      << "\" profile=sensor_fast capacity="
                      << renderer.layout().capacity
                      << " vertices=" << layout.meshVertexCount
                      << " triangles=" << layout.meshTriangleCount
                      << " rejected_stale=1 disabled_pixels="
                      << semanticPixelCount(unchangedAfter, semantic)
                      << " deformed_pixels=" << deformedPixels
                      << " callback_invocations="
                      << callbackContext.invocationCount << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "human_resting_mesh_deformation_probe: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
