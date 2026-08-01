#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalHybridRenderer.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <ranges>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kMinimumAllocationBytes = 16u;
constexpr NSUInteger kCameraThreads = 64u;
constexpr NSUInteger kPixelThreads = 256u;
constexpr std::uint32_t kLiveCurrent = 1u << 0u;
constexpr std::uint32_t kLivePrevious = 1u << 1u;
const char kMetalHybridRendererImageAnchor = 0;

struct RendererBuffers {
    __strong id<MTLBuffer> gaussians = nil;
    __strong id<MTLBuffer> meshVertices = nil;
    __strong id<MTLBuffer> meshIndices = nil;
    __strong id<MTLBuffer> meshTriangles = nil;
    __strong id<MTLBuffer> meshClusters = nil;
    __strong id<MTLBuffer> meshTriangleClusters = nil;
    __strong id<MTLBuffer> meshPrimitives = nil;
    __strong id<MTLBuffer> meshInstances = nil;
    __strong id<MTLBuffer> materials = nil;
    __strong id<MTLBuffer> textureBindings = nil;
    __strong id<MTLBuffer> resourceArgumentBuffer = nil;
    __strong id<MTLBuffer> lights = nil;
    __strong id<MTLBuffer> environmentData = nil;
    __strong id<MTLBuffer> rayVisibleInstances = nil;
    __strong id<MTLBuffer> rayBlasIndices = nil;
    __strong id<MTLBuffer> sensorBindings = nil;
    __strong id<MTLBuffer> currentBodies = nil;
    __strong id<MTLBuffer> previousBodies = nil;
    __strong id<MTLBuffer> cameraStates = nil;
    __strong id<MTLBuffer> visualInstanceStates = nil;
    __strong id<MTLBuffer> nearClippedTriangles = nil;
    __strong id<MTLBuffer> nearClippedTriangleCounts = nil;
    __strong id<MTLBuffer> nearClippedDispatchArguments = nil;
    __strong id<MTLBuffer> projected = nil;
    __strong id<MTLBuffer> tileCounts = nil;
    __strong id<MTLBuffer> tileIndices = nil;
    __strong id<MTLBuffer> tileOverflowCounts = nil;
    __strong id<MTLBuffer> meshTileCounts = nil;
    __strong id<MTLBuffer> meshTileRecords = nil;
    __strong id<MTLBuffer> meshTileOverflowCounts = nil;
    __strong id<MTLBuffer> meshClusterVisibility = nil;
    __strong id<MTLBuffer> meshWinners = nil;
    __strong id<MTLBuffer> rgb = nil;
    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> segmentation = nil;
    __strong id<MTLBuffer> identities = nil;
    __strong id<MTLBuffer> normals = nil;
    __strong id<MTLBuffer> motion = nil;
    __strong id<MTLBuffer> validity = nil;
    __strong id<MTLBuffer> shadowAtlas = nil;
    __strong id<MTLBuffer> temporalAccumulation = nil;
};

struct RuntimeVisualScene {
    struct GeometrySource {
        std::filesystem::path path;
        std::uint64_t vertexFileOffset = 0u;
        std::uint64_t vertexByteCount = 0u;
        std::uint64_t indexFileOffset = 0u;
        std::uint64_t indexByteCount = 0u;
        std::uint64_t materialFileOffset = 0u;
        std::uint64_t materialByteCount = 0u;
        std::uint32_t vertexBase = 0u;
        std::uint32_t indexBase = 0u;
        std::uint32_t materialBase = 0u;
        std::uint32_t bindingBase = 0u;
    };

    struct TextureSource {
        std::filesystem::path path;
        std::uint64_t payloadFileOffset = 0u;
    };

    std::vector<MRHybridGaussianGPU> gaussians;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::vector<GeometrySource> geometrySources;
    std::vector<MRHybridMeshClusterGPU> meshClusters;
    std::vector<std::uint32_t> meshTriangleClusters;
    std::vector<MRVisualPrimitiveGPUV2> primitives;
    std::vector<MRVisualInstanceGPUV2> instances;
    std::vector<MRVisualMaterialGPUV2> materials;
    std::vector<MRVisualTextureBindingGPUV2> textureBindings;
    std::vector<VisualTextureImageV2> textures;
    std::vector<TextureSource> textureSources;
    std::vector<MRVisualLightGPUV1> lights;
    VisualEnvironmentPackV2 environmentPack;
    std::array<TextureSource, 3u> environmentTextureSources{};
    MRVisualEnvironmentGPUV2 environmentData{};
    std::vector<MRVisualSensorBindingGPU> sensors;
};

using RayGeometryKey =
    std::vector<std::array<std::uint32_t, 3u>>;

bool matchesRayGeometry(
    const RayGeometryKey& key,
    const std::span<const MRVisualPrimitiveGPUV2> primitives
) {
    if (key.size() != primitives.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < key.size(); ++index) {
        const MRVisualPrimitiveGPUV2& primitive =
            primitives[index];
        if (key[index] != std::array{
                primitive.geometry.x,
                primitive.geometry.y,
                primitive.geometry.z,
            }) {
            return false;
        }
    }
    return true;
}

struct CompactedPrimitiveAccelerationStructures {
    __strong NSArray<id<MTLAccelerationStructure>>* structures = nil;
    std::size_t retainedBytes = 0u;
};

constexpr std::uint32_t kReferenceEnvironmentsPerTLAS =
    MR_HYBRID_REFERENCE_ENVIRONMENTS_PER_TLAS;
constexpr std::size_t kAccelerationStructureScratchAlignment = 256u;

std::string nsString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return std::string{value.UTF8String};
}

std::string describeError(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string description =
        nsString(error.localizedDescription);
    return description.empty()
        ? nsString(error.description)
        : description;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalHybridRendererImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }
    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalHybridRendererDiagnostics reject(
    MetalHybridRendererDiagnostics diagnostics,
    const MetalHybridRendererStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool appendAlignedReadbackRegion(
    const std::size_t bytes,
    std::size_t& cursor,
    std::size_t& offset
) {
    constexpr std::size_t kAlignment = 256u;
    const std::size_t remainder = cursor % kAlignment;
    const std::size_t padding =
        remainder == 0u ? 0u : kAlignment - remainder;
    if (!checkedAdd(cursor, padding, offset) ||
        !checkedAdd(offset, bytes, cursor)) {
        return false;
    }
    return cursor <= std::numeric_limits<NSUInteger>::max();
}

template <typename Value>
bool checkedBytes(
    const std::size_t count,
    std::size_t& result
) {
    return checkedMultiply(count, sizeof(Value), result) &&
        result <= std::numeric_limits<NSUInteger>::max();
}

id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    const MTLResourceOptions options,
    NSString* label
) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(
            logicalBytes,
            kMinimumAllocationBytes
        )
    );
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:options];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> makePrivateBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModePrivate,
        label
    );
}

id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModeShared,
        label
    );
}

bool finite4(const mr_float4& value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

mr_float4 interpolate4(
    const mr_float4 a,
    const mr_float4 b,
    const float fraction
) {
    return {
        a.x + (b.x - a.x) * fraction,
        a.y + (b.y - a.y) * fraction,
        a.z + (b.z - a.z) * fraction,
        a.w + (b.w - a.w) * fraction,
    };
}

mr_float4 interpolateQuaternion(
    mr_float4 a,
    mr_float4 b,
    const float fraction
) {
    const float dot =
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (dot < 0.0f) {
        b = {-b.x, -b.y, -b.z, -b.w};
    }
    mr_float4 result = interpolate4(a, b, fraction);
    const float length = std::sqrt(
        result.x * result.x + result.y * result.y +
        result.z * result.z + result.w * result.w
    );
    if (!(length > 1.0e-12f)) {
        return a;
    }
    result.x /= length;
    result.y /= length;
    result.z /= length;
    result.w /= length;
    return result;
}

void writeInterpolatedBodyPose(
    MRBodyStateGPU& result,
    const MRBodyStateGPU& a,
    const MRBodyStateGPU& b,
    const float fraction
) {
    result.position = interpolate4(a.position, b.position, fraction);
    result.orientation =
        interpolateQuaternion(a.orientation, b.orientation, fraction);
}

void copyBodyPoses(
    const std::span<const MRBodyStateGPU> input,
    const std::span<MRBodyStateGPU> output
) {
    for (std::size_t index = 0u; index < input.size(); ++index) {
        output[index].position = input[index].position;
        output[index].orientation = input[index].orientation;
    }
}

bool sampleMotionStates(
    const VisualMotionSampleBatchV1& motion,
    const double timestamp,
    std::span<MRBodyStateGPU> output
) {
    const std::size_t perSample =
        static_cast<std::size_t>(motion.environmentCount) *
        motion.bodyCount;
    if (output.size() != perSample ||
        motion.timestampsSeconds.empty()) {
        return false;
    }
    const auto upper = std::upper_bound(
        motion.timestampsSeconds.begin(),
        motion.timestampsSeconds.end(),
        timestamp
    );
    if (upper == motion.timestampsSeconds.begin()) {
        copyBodyPoses(motion.sample(0u), output);
        return true;
    }
    if (upper == motion.timestampsSeconds.end()) {
        copyBodyPoses(
            motion.sample(motion.sampleCount - 1u),
            output
        );
        return true;
    }
    const std::size_t right =
        static_cast<std::size_t>(
            upper - motion.timestampsSeconds.begin()
        );
    const std::size_t left = right - 1u;
    const double duration =
        motion.timestampsSeconds[right] -
        motion.timestampsSeconds[left];
    const float fraction = duration <= 0.0
        ? 0.0f
        : static_cast<float>(
              (timestamp - motion.timestampsSeconds[left]) /
              duration
          );
    const auto leftStates =
        motion.sample(static_cast<std::uint32_t>(left));
    const auto rightStates =
        motion.sample(static_cast<std::uint32_t>(right));
    for (std::size_t index = 0u; index < perSample; ++index) {
        writeInterpolatedBodyPose(
            output[index],
            leftStates[index],
            rightStates[index],
            fraction
        );
    }
    return true;
}

VisualTextureImageV2 makeNeutralCubeTexture(
    const std::string& id
) {
    VisualTextureImageV2 texture{};
    texture.id = id;
    texture.contentHash = "builtin:" + id;
    texture.width = 1u;
    texture.height = 1u;
    texture.mipCount = 1u;
    texture.arrayLength = 6u;
    texture.pixelFormat =
        VisualTexturePixelFormatV2::rgba16Float;
    texture.dimension = VisualTextureDimensionV2::cube;
    constexpr std::uint32_t kRowBytes = 256u;
    texture.data.resize(6u * kRowBytes, 0u);
    for (std::uint32_t slice = 0u; slice < 6u; ++slice) {
        auto* values = reinterpret_cast<std::uint16_t*>(
            texture.data.data() + slice * kRowBytes
        );
        values[0] = 0x3c00u;
        values[1] = 0x3c00u;
        values[2] = 0x3c00u;
        values[3] = 0x3c00u;
        texture.subresources.push_back({
            0u,
            slice,
            1u,
            1u,
            static_cast<std::uint64_t>(slice) * kRowBytes,
            kRowBytes,
            kRowBytes,
            kRowBytes,
        });
    }
    return texture;
}

VisualTextureImageV2 makeNeutralBrdfTexture() {
    VisualTextureImageV2 texture{};
    texture.id = "neutral_dfg";
    texture.contentHash = "builtin:neutral-dfg";
    texture.width = 1u;
    texture.height = 1u;
    texture.mipCount = 1u;
    texture.arrayLength = 1u;
    texture.pixelFormat = VisualTexturePixelFormatV2::rg16Float;
    texture.dimension = VisualTextureDimensionV2::texture2D;
    texture.data.resize(256u, 0u);
    auto* values =
        reinterpret_cast<std::uint16_t*>(texture.data.data());
    values[0] = 0x3c00u;
    values[1] = 0u;
    texture.subresources.push_back({
        0u,
        0u,
        1u,
        1u,
        0u,
        256u,
        256u,
        256u,
    });
    return texture;
}

VisualEnvironmentPackV2 makeNeutralEnvironmentPack() {
    VisualEnvironmentPackV2 pack{};
    pack.id = "neutral_studio";
    pack.sourceUri = "builtin:neutral-studio-v3";
    pack.sourceContentHash = "builtin:neutral-studio-v3";
    pack.sourceColorSpace = "linear-rec709";
    pack.preprocessingProvenance =
        "metalrobo builtin neutral IBL v3";
    pack.specularFaceSize = 1u;
    pack.diffuseFaceSize = 1u;
    pack.brdfLutSize = 1u;
    pack.diffuseIrradiance =
        makeNeutralCubeTexture("neutral_irradiance");
    pack.prefilteredSpecular =
        makeNeutralCubeTexture("neutral_specular");
    pack.brdfLut = makeNeutralBrdfTexture();
    pack.contentHash =
        computeVisualEnvironmentPackContentHash(pack);
    return pack;
}

void rebaseMaterialBindings(
    MRVisualMaterialGPUV2& material,
    const std::uint32_t bindingBase
) {
    const auto rebase = [bindingBase](std::uint32_t& value) {
        if (value != MR_INVALID_INDEX) {
            value += bindingBase;
        }
    };
    rebase(material.textureIndices0.x);
    rebase(material.textureIndices0.y);
    rebase(material.textureIndices0.z);
    rebase(material.textureIndices0.w);
    rebase(material.textureIndices1.x);
    rebase(material.textureIndices1.y);
    rebase(material.textureIndices1.z);
    // Direct USD keeps scalar bindings separate from glTF's packed maps.
    rebase(material.reserved.x);
    rebase(material.reserved.y);
    rebase(material.reserved.z);
    rebase(material.reserved.w);
}

const VisualPackSectionV2* packSection(
    const std::span<const VisualPackSectionV2> sections,
    const VisualAssetSectionKindV2 kind,
    const std::uint32_t index = 0u
) {
    const auto found = std::ranges::find_if(
        sections,
        [kind, index](const VisualPackSectionV2& section) {
            return section.kind == kind &&
                section.index == index;
        }
    );
    return found == sections.end() ? nullptr : &*found;
}

bool flattenScene(
    VisualRenderSceneV3&& scene,
    RuntimeVisualScene& output,
    std::string& reason
) {
    output.gaussians = std::move(scene.gaussians);
    output.lights = std::move(scene.lightRig.lights);
    output.sensors = std::move(scene.sensorBindings);

    for (const VisualAssetReferenceV3& reference :
         scene.visualPacks) {
        VisualAssetPackV2 pack{};
        if (!readVisualAssetPackIndex(
                reference.packPath,
                pack,
                &reason
            ) ||
            pack.contentHash != reference.contentHash) {
            if (reason.empty()) {
                reason =
                    "visual pack reference hash does not match";
            }
            return false;
        }
        const VisualPackSectionV2* vertexSection = packSection(
            pack.sections,
            VisualAssetSectionKindV2::vertices
        );
        const VisualPackSectionV2* indexSection = packSection(
            pack.sections,
            VisualAssetSectionKindV2::indices
        );
        const VisualPackSectionV2* materialSection = packSection(
            pack.sections,
            VisualAssetSectionKindV2::materials
        );
        if (vertexSection == nullptr || indexSection == nullptr ||
            materialSection == nullptr ||
            vertexSection->elementCount >
                std::numeric_limits<std::uint32_t>::max() -
                    output.vertexCount ||
            indexSection->elementCount >
                std::numeric_limits<std::uint32_t>::max() -
                    output.indexCount ||
            output.primitives.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            output.instances.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            output.materials.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            output.textures.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            output.textureBindings.size() >
                std::numeric_limits<std::uint32_t>::max()) {
            reason = "visual scene pack bases exceed uint32";
            return false;
        }
        const std::uint32_t vertexBase =
            output.vertexCount;
        const std::uint32_t indexBase =
            output.indexCount;
        const std::uint32_t primitiveBase =
            static_cast<std::uint32_t>(output.primitives.size());
        const std::uint32_t instanceBase =
            static_cast<std::uint32_t>(output.instances.size());
        const std::uint32_t materialBase =
            static_cast<std::uint32_t>(output.materials.size());
        const std::uint32_t textureBase =
            static_cast<std::uint32_t>(output.textures.size());
        const std::uint32_t bindingBase =
            static_cast<std::uint32_t>(
                output.textureBindings.size()
            );

        output.vertexCount += static_cast<std::uint32_t>(
            vertexSection->elementCount
        );
        output.indexCount += static_cast<std::uint32_t>(
            indexSection->elementCount
        );
        output.geometrySources.push_back({
            reference.packPath,
            vertexSection->fileOffset,
            vertexSection->byteCount,
            indexSection->fileOffset,
            indexSection->byteCount,
            materialSection->fileOffset,
            materialSection->byteCount,
            vertexBase,
            indexBase,
            materialBase,
            bindingBase,
        });
        for (MRVisualTextureBindingGPUV2 binding :
             pack.textureBindings) {
            binding.resource.x += textureBase;
            output.textureBindings.push_back(binding);
        }
        for (MRVisualMaterialGPUV2 material : pack.materials) {
            rebaseMaterialBindings(material, bindingBase);
            output.materials.push_back(material);
        }
        for (MRVisualInstanceGPUV2 instance : pack.instances) {
            instance.binding.x = reference.assetIndex;
            instance.geometry.x += primitiveBase;
            if (instance.identity.x == 0u) {
                instance.identity.x = reference.semanticId;
            }
            if (instance.identity.y == 0u) {
                instance.identity.y = reference.instanceId;
            }
            output.instances.push_back(instance);
        }
        for (MRVisualPrimitiveGPUV2 primitive :
             pack.primitives) {
            primitive.geometry.x += indexBase;
            primitive.geometry.z += materialBase;
            primitive.geometry.w += instanceBase;
            if (primitive.identity.x == 0u) {
                primitive.identity.x = reference.semanticId;
            }
            if (primitive.identity.y == 0u) {
                primitive.identity.y = reference.instanceId;
            }
            output.primitives.push_back(primitive);
        }
        for (std::uint32_t textureIndex = 0u;
             textureIndex < pack.textures.size();
             ++textureIndex) {
            const VisualPackSectionV2* payload = packSection(
                pack.sections,
                VisualAssetSectionKindV2::texturePayload,
                textureIndex
            );
            if (payload == nullptr) {
                reason = "visual texture payload section is absent";
                return false;
            }
            output.textureSources.push_back({
                reference.packPath,
                payload->fileOffset,
            });
        }
        output.textures.insert(
            output.textures.end(),
            std::make_move_iterator(pack.textures.begin()),
            std::make_move_iterator(pack.textures.end())
        );
    }

    if (scene.environment.packPath.empty()) {
        output.environmentPack = makeNeutralEnvironmentPack();
    } else if (!readVisualEnvironmentPackIndex(
                   scene.environment.packPath,
                   output.environmentPack,
                   &reason
               ) ||
               output.environmentPack.contentHash !=
                   scene.environment.contentHash) {
        if (reason.empty()) {
            reason =
                "environment pack reference hash does not match";
        }
        return false;
    } else {
        for (std::uint32_t index = 0u;
             index < output.environmentTextureSources.size();
             ++index) {
            const VisualPackSectionV2* payload = packSection(
                output.environmentPack.sections,
                VisualAssetSectionKindV2::texturePayload,
                index
            );
            if (payload == nullptr) {
                reason =
                    "environment texture payload section is absent";
                return false;
            }
            output.environmentTextureSources[index] = {
                scene.environment.packPath,
                payload->fileOffset,
            };
        }
    }
    output.environmentData.dimensions = {
        output.environmentPack.prefilteredSpecular.mipCount,
        output.environmentPack.diffuseFaceSize,
        output.environmentPack.specularFaceSize,
        output.environmentPack.brdfLutSize,
    };
    output.environmentData.parameters = {
        scene.environment.intensity,
        scene.environment.rotationRadians,
        scene.environment.packPath.empty() ? 0.0f : 1.0f,
        0.0f,
    };

    if (!output.primitives.empty() && output.materials.empty()) {
        reason = "V3 render scene has no materials";
        return false;
    }
    return output.indexCount % 3u == 0u;
}

bool buildMeshClusters(
    RuntimeVisualScene& scene,
    std::string& reason
) {
    scene.meshClusters.clear();
    scene.meshTriangleClusters.assign(
        scene.indexCount / 3u,
        MR_INVALID_INDEX
    );
    scene.meshClusters.reserve(
        (
            static_cast<std::size_t>(scene.indexCount / 3u) +
            MR_HYBRID_MESH_CLUSTER_TRIANGLES - 1u
        ) /
        MR_HYBRID_MESH_CLUSTER_TRIANGLES
    );
    for (std::uint32_t primitiveIndex = 0u;
         primitiveIndex < scene.primitives.size();
         ++primitiveIndex) {
        const MRVisualPrimitiveGPUV2& primitive =
            scene.primitives[primitiveIndex];
        if (primitive.geometry.x % 3u != 0u ||
            primitive.geometry.y % 3u != 0u ||
            primitive.geometry.x > scene.indexCount ||
            primitive.geometry.y >
                scene.indexCount - primitive.geometry.x) {
            reason =
                "visual primitive index range cannot form mesh clusters";
            return false;
        }
        const std::uint32_t firstTriangle =
            primitive.geometry.x / 3u;
        const std::uint32_t triangleCount =
            primitive.geometry.y / 3u;
        for (std::uint32_t first = 0u;
             first < triangleCount;
             first += MR_HYBRID_MESH_CLUSTER_TRIANGLES) {
            const std::uint32_t count = std::min(
                std::uint32_t(MR_HYBRID_MESH_CLUSTER_TRIANGLES),
                triangleCount - first
            );
            MRHybridMeshClusterGPU cluster{};
            cluster.range = {
                firstTriangle + first,
                count,
                primitiveIndex,
                0u,
            };
            const std::uint32_t clusterIndex =
                static_cast<std::uint32_t>(
                    scene.meshClusters.size()
                );
            scene.meshClusters.push_back(cluster);
            std::fill_n(
                scene.meshTriangleClusters.begin() +
                    cluster.range.x,
                cluster.range.y,
                clusterIndex
            );
        }
    }
    if (scene.meshClusters.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        reason = "visual mesh cluster count exceeds uint32";
        return false;
    }
    if (std::ranges::find(
            scene.meshTriangleClusters,
            MR_INVALID_INDEX
        ) != scene.meshTriangleClusters.end()) {
        reason =
            "visual mesh triangles are not covered by a primitive";
        return false;
    }
    return true;
}

} // namespace

namespace detail {

struct ReferenceFrameWorkspace {
    std::atomic_bool inUse{false};
    __strong id<MTLBuffer> motionBodies = nil;
    __strong id<MTLBuffer> instanceDescriptors = nil;
    __strong id<MTLBuffer> motionTransforms = nil;
    __strong id<MTLBuffer> buildScratch = nil;
    __strong NSArray<MTLInstanceAccelerationStructureDescriptor*>*
        tlasDescriptors = nil;
    __strong NSArray<id<MTLAccelerationStructure>>*
        instanceStructures = nil;
    std::vector<std::size_t> buildScratchOffsets;
    std::size_t retainedBytes = 0u;
    std::size_t motionBodyBytes = 0u;
    std::size_t structureBytes = 0u;
    std::size_t scratchBytes = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t instancesPerEnvironment = 0u;
    std::uint32_t instanceCount = 0u;
    std::uint32_t keyframeCount = 0u;
    bool built = false;
    std::uint32_t refitCount = 0u;
};

struct ExposureFrameWorkspace {
    std::atomic_bool inUse{false};
    __strong id<MTLBuffer> motionBodies = nil;
    std::size_t retainedBytes = 0u;
};

struct MetalHybridRendererState {
    explicit MetalHybridRendererState(
        MetalHybridRendererConfig configured
    )
        : config(std::move(configured)) {}

    MetalHybridRendererConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool compiled = false;
    bool requiresLiveState = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    std::string deviceName;
    __strong id<MTLComputePipelineState> clearPipeline = nil;
    __strong id<MTLComputePipelineState> rebaseIndicesPipeline = nil;
    __strong id<MTLComputePipelineState>
        rebaseMaterialBindingsPipeline = nil;
    __strong id<MTLComputePipelineState> expandTrianglesPipeline = nil;
    __strong id<MTLComputePipelineState> buildMeshClustersPipeline = nil;
    __strong id<MTLComputePipelineState> prepareCameraPipeline = nil;
    __strong id<MTLComputePipelineState> resolveNearClippedPipeline = nil;
    __strong id<MTLComputePipelineState> clearObservationPipeline = nil;
    __strong id<MTLComputePipelineState> binPipeline = nil;
    __strong id<MTLComputePipelineState> renderPipeline = nil;
    __strong id<MTLComputePipelineState> cullMeshClustersPipeline = nil;
    __strong id<MTLComputePipelineState> binMeshPipeline = nil;
    __strong id<MTLComputePipelineState> resolveMeshTilesPipeline = nil;
    __strong id<MTLComputePipelineState> compositeMeshPipeline = nil;
    __strong id<MTLComputePipelineState> clearShadowPipeline = nil;
    __strong id<MTLComputePipelineState> rasterShadowPipeline = nil;
    __strong id<MTLComputePipelineState> clearAccumulationPipeline = nil;
    __strong id<MTLComputePipelineState> accumulatePipeline = nil;
    __strong id<MTLComputePipelineState> resolveAccumulationPipeline = nil;
    __strong id<MTLComputePipelineState> applySensorPipeline = nil;
    __strong id<MTLComputePipelineState> prepareRayInstancesPipeline = nil;
    __strong id<MTLComputePipelineState> referenceRenderPipeline = nil;
    __strong id<MTLArgumentEncoder> resourceArgumentEncoder = nil;
    __strong id<MTLHeap> geometryHeap = nil;
    __strong id<MTLHeap> visualResourceHeap = nil;
    __strong id<MTLResidencySet> visualResourceResidencySet = nil;
    __strong NSArray<id<MTLTexture>>* materialTextures = nil;
    __strong NSArray<id<MTLSamplerState>>* materialSamplers = nil;
    __strong id<MTLTexture> diffuseIrradiance = nil;
    __strong id<MTLTexture> prefilteredSpecular = nil;
    __strong id<MTLTexture> brdfLut = nil;
    __strong NSArray<id<MTLAccelerationStructure>>*
        primitiveAccelerationStructures = nil;
    RendererBuffers buffers;
    MetalHybridRendererLayout layout;
    MRVisualFrameMetadataGPU activeMetadata{};
    std::vector<MRVisualSensorBindingGPU> sensorProfiles;
    VisualRendererProfileV1 rendererProfile =
        VisualRendererProfileV1::sensorFast();
    VisualEnvironmentReferenceV2 environment;
    std::uint64_t renderSceneFingerprint = 0u;
    std::uint32_t shadowLightIndex = MR_INVALID_INDEX;
    std::uint32_t nearClippedTriangleCapacity = 0u;
    std::vector<std::uint32_t> rayVisibleInstances;
    std::vector<std::uint32_t> rayBlasIndices;
    std::vector<std::shared_ptr<ReferenceFrameWorkspace>>
        referenceWorkspaces;
    std::vector<std::shared_ptr<ExposureFrameWorkspace>>
        exposureWorkspaces;
    std::uint32_t assetCount = 0u;
    std::uint32_t textureBindingCount = 0u;
    std::uint32_t activeEnvironmentCount = 0u;
};

} // namespace detail

namespace {

MetalHybridRendererDiagnostics initialize(
    detail::MetalHybridRendererState& state,
    MetalHybridRendererDiagnostics diagnostics
) {
    if (state.initialized) {
        diagnostics.deviceName = state.deviceName;
        return diagnostics;
    }
    state.device = MTLCreateSystemDefaultDevice();
    if (state.device == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalDeviceUnavailable,
            "no Metal device is available"
        );
    }
    state.queue = [state.device newCommandQueue];
    if (state.queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalDeviceUnavailable,
            "could not create the hybrid-render command queue"
        );
    }
    const std::string path = state.config.metallibPath.empty()
        ? defaultMetallibPath()
        : state.config.metallibPath;
    if (path.empty() || !regularFile(path)) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metallibUnavailable,
            "MetalRobo.metallib is unavailable"
        );
    }
    NSError* error = nil;
    state.library = [state.device
        newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                    error:&error];
    if (state.library == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalLibraryFailure,
            "could not load MetalRobo.metallib: " +
                describeError(error)
        );
    }

    const auto pipeline =
        [&state, &error](
            NSString* name
        ) -> id<MTLComputePipelineState> {
        id<MTLFunction> function =
            [state.library newFunctionWithName:name];
        if (function == nil) {
            return nil;
        }
        return [state.device
            newComputePipelineStateWithFunction:function
                                          error:&error];
    };
    state.clearPipeline =
        pipeline(@"mr_hybrid_clear_tiles");
    state.rebaseIndicesPipeline =
        pipeline(@"mr_visual_rebase_indices_v3");
    state.rebaseMaterialBindingsPipeline =
        pipeline(@"mr_visual_rebase_material_bindings_v3");
    state.expandTrianglesPipeline =
        pipeline(@"mr_visual_expand_triangles_v3");
    state.buildMeshClustersPipeline =
        pipeline(@"mr_visual_build_mesh_clusters");
    state.prepareCameraPipeline =
        pipeline(@"mr_hybrid_prepare_cameras");
    state.resolveNearClippedPipeline =
        pipeline(@"mr_hybrid_resolve_near_clipped_mesh");
    state.clearObservationPipeline =
        pipeline(@"mr_hybrid_clear_observations");
    state.binPipeline =
        pipeline(@"mr_hybrid_bin_gaussians");
    state.renderPipeline =
        pipeline(@"mr_hybrid_render_tiles");
    state.cullMeshClustersPipeline =
        pipeline(@"mr_hybrid_cull_mesh_clusters");
    state.binMeshPipeline =
        pipeline(@"mr_hybrid_bin_mesh");
    state.resolveMeshTilesPipeline =
        pipeline(@"mr_hybrid_resolve_mesh_tiles");
    state.compositeMeshPipeline =
        pipeline(@"mr_hybrid_composite_mesh");
    state.clearShadowPipeline =
        pipeline(@"mr_hybrid_clear_shadow_atlas");
    state.rasterShadowPipeline =
        pipeline(@"mr_hybrid_rasterize_shadow_clusters");
    state.clearAccumulationPipeline =
        pipeline(@"mr_hybrid_clear_temporal_accumulation");
    state.accumulatePipeline =
        pipeline(@"mr_hybrid_accumulate_temporal_sample");
    state.resolveAccumulationPipeline =
        pipeline(@"mr_hybrid_resolve_temporal_accumulation");
    state.applySensorPipeline =
        pipeline(@"mr_hybrid_apply_sensor");
    state.prepareRayInstancesPipeline =
        pipeline(@"mr_hybrid_prepare_ray_instances");
    state.referenceRenderPipeline =
        pipeline(@"mr_hybrid_render_reference");
    id<MTLFunction> compositeFunction =
        [state.library
            newFunctionWithName:@"mr_hybrid_composite_mesh"];
    state.resourceArgumentEncoder =
        [compositeFunction newArgumentEncoderWithBufferIndex:21u];
    if (state.clearPipeline == nil ||
        state.rebaseIndicesPipeline == nil ||
        state.rebaseMaterialBindingsPipeline == nil ||
        state.expandTrianglesPipeline == nil ||
        state.buildMeshClustersPipeline == nil ||
        state.prepareCameraPipeline == nil ||
        state.resolveNearClippedPipeline == nil ||
        state.clearObservationPipeline == nil ||
        state.binPipeline == nil ||
        state.renderPipeline == nil ||
        state.cullMeshClustersPipeline == nil ||
        state.binMeshPipeline == nil ||
        state.resolveMeshTilesPipeline == nil ||
        state.compositeMeshPipeline == nil ||
        state.clearShadowPipeline == nil ||
        state.rasterShadowPipeline == nil ||
        state.clearAccumulationPipeline == nil ||
        state.accumulatePipeline == nil ||
        state.resolveAccumulationPipeline == nil ||
        state.applySensorPipeline == nil ||
        state.resourceArgumentEncoder == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalPipelineFailure,
            "could not create visual sensor pipelines: " +
                describeError(error)
        );
    }
    if (state.renderPipeline.maxTotalThreadsPerThreadgroup <
            MR_HYBRID_MAX_GAUSSIANS_PER_TILE ||
        state.rasterShadowPipeline.maxTotalThreadsPerThreadgroup <
            MR_HYBRID_MESH_CLUSTER_TRIANGLES ||
        state.resolveMeshTilesPipeline
                .maxTotalThreadsPerThreadgroup <
            MR_HYBRID_TILE_SIZE * MR_HYBRID_TILE_SIZE) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalPipelineFailure,
            "device cannot dispatch the 16x16 visual tile kernels"
        );
    }
    state.deviceName = nsString(state.device.name);
    state.initialized = true;
    diagnostics.deviceName = state.deviceName;
    return diagnostics;
}

bool upload(
    id<MTLDevice> device,
    id<MTLBlitCommandEncoder> blit,
    const void* source,
    const std::size_t bytes,
    id<MTLBuffer> destination,
    NSString* label,
    std::vector<id<MTLBuffer>>& staging
) {
    if (bytes == 0u) {
        return true;
    }
    id<MTLBuffer> buffer =
        makeSharedBuffer(device, bytes, label);
    if (buffer == nil) {
        return false;
    }
    std::memcpy(buffer.contents, source, bytes);
    [blit copyFromBuffer:buffer
            sourceOffset:0u
                toBuffer:destination
       destinationOffset:0u
                    size:bytes];
    staging.push_back(buffer);
    return true;
}

constexpr std::uint32_t kMaximumSceneTexturesV3 = 2048u;
constexpr std::uint32_t kSceneSamplerCountV3 = 108u;
constexpr std::uint32_t kEnvironmentArgumentBaseV3 =
    kMaximumSceneTexturesV3 + kSceneSamplerCountV3;

MTLPixelFormat metalPixelFormat(
    const VisualTexturePixelFormatV2 format
) {
    switch (format) {
    case VisualTexturePixelFormatV2::rgba8Unorm:
        return MTLPixelFormatRGBA8Unorm;
    case VisualTexturePixelFormatV2::rgba8UnormSrgb:
        return MTLPixelFormatRGBA8Unorm_sRGB;
    case VisualTexturePixelFormatV2::rg11b10Float:
        return MTLPixelFormatRG11B10Float;
    case VisualTexturePixelFormatV2::rg16Float:
        return MTLPixelFormatRG16Float;
    case VisualTexturePixelFormatV2::rgba16Float:
        return MTLPixelFormatRGBA16Float;
    }
    return MTLPixelFormatInvalid;
}

MTLTextureDescriptor* textureDescriptor(
    const VisualTextureImageV2& texture
) {
    MTLTextureDescriptor* descriptor =
        [[MTLTextureDescriptor alloc] init];
    descriptor.textureType =
        texture.dimension == VisualTextureDimensionV2::cube
        ? MTLTextureTypeCube
        : MTLTextureType2D;
    descriptor.pixelFormat =
        metalPixelFormat(texture.pixelFormat);
    descriptor.width = texture.width;
    descriptor.height = texture.height;
    descriptor.depth = 1u;
    descriptor.mipmapLevelCount = texture.mipCount;
    descriptor.arrayLength = 1u;
    descriptor.sampleCount = 1u;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    descriptor.hazardTrackingMode =
        MTLHazardTrackingModeUntracked;
    descriptor.usage = MTLTextureUsageShaderRead;
    return descriptor;
}

std::size_t alignResourceOffset(
    const std::size_t value,
    const std::size_t alignment
) {
    if (alignment <= 1u) {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

struct NativeVisualResources {
    __strong id<MTLHeap> heap = nil;
    __strong NSArray<id<MTLTexture>>* materialTextures = nil;
    __strong NSArray<id<MTLSamplerState>>* samplers = nil;
    __strong id<MTLTexture> diffuseIrradiance = nil;
    __strong id<MTLTexture> prefilteredSpecular = nil;
    __strong id<MTLTexture> brdfLut = nil;
    std::size_t retainedBytes = 0u;
};

struct NativeGeometryResources {
    __strong id<MTLHeap> heap = nil;
    std::size_t retainedBytes = 0u;
};

bool createNativeGeometryResources(
    id<MTLDevice> device,
    const std::array<std::size_t, 9u>& byteCounts,
    RendererBuffers& buffers,
    NativeGeometryResources& output,
    std::string& reason
) {
    const MTLResourceOptions options =
        MTLResourceStorageModePrivate |
        MTLResourceHazardTrackingModeUntracked;
    std::array<std::size_t, 9u> lengths{};
    std::array<std::size_t, 9u> offsets{};
    constexpr std::array<std::size_t, 9u> typedMinimums{
        sizeof(MRVisualVertexGPUV2),
        sizeof(std::uint32_t),
        sizeof(MRVisualTriangleGPUV2),
        sizeof(MRHybridMeshClusterGPU),
        sizeof(std::uint32_t),
        sizeof(MRVisualPrimitiveGPUV2),
        sizeof(MRVisualInstanceGPUV2),
        sizeof(MRVisualMaterialGPUV2),
        sizeof(MRVisualTextureBindingGPUV2),
    };
    std::size_t heapBytes = 0u;
    for (std::size_t index = 0u;
         index < byteCounts.size();
         ++index) {
        lengths[index] = std::max<std::size_t>(
            byteCounts[index],
            typedMinimums[index]
        );
        const MTLSizeAndAlign sizeAndAlign =
            [device
                heapBufferSizeAndAlignWithLength:lengths[index]
                                         options:options];
        if (sizeAndAlign.size == 0u ||
            sizeAndAlign.align == 0u ||
            heapBytes >
                std::numeric_limits<std::size_t>::max() -
                    sizeAndAlign.align) {
            reason = "visual geometry heap sizing failed";
            return false;
        }
        heapBytes = alignResourceOffset(
            heapBytes,
            sizeAndAlign.align
        );
        offsets[index] = heapBytes;
        if (sizeAndAlign.size >
            std::numeric_limits<std::size_t>::max() - heapBytes) {
            reason = "visual geometry heap size overflows";
            return false;
        }
        heapBytes += sizeAndAlign.size;
    }
    MTLHeapDescriptor* descriptor =
        [[MTLHeapDescriptor alloc] init];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    descriptor.hazardTrackingMode =
        MTLHazardTrackingModeUntracked;
    descriptor.size = heapBytes;
    output.heap = [device newHeapWithDescriptor:descriptor];
    if (output.heap == nil) {
        reason = "could not allocate visual geometry placement heap";
        return false;
    }
    output.heap.label = @"MetalRobo V3 streamed geometry heap";
    const auto buffer = [&](
        const std::size_t index,
        NSString* label
    ) -> id<MTLBuffer> {
        id<MTLBuffer> result =
            [output.heap
                newBufferWithLength:lengths[index]
                           options:options
                            offset:offsets[index]];
        result.label = label;
        return result;
    };
    buffers.meshVertices =
        buffer(0u, @"MetalRobo streamed vertices");
    buffers.meshIndices =
        buffer(1u, @"MetalRobo streamed indices");
    buffers.meshTriangles =
        buffer(2u, @"MetalRobo expanded triangles");
    buffers.meshClusters =
        buffer(3u, @"MetalRobo mesh clusters");
    buffers.meshTriangleClusters =
        buffer(4u, @"MetalRobo triangle cluster indices");
    buffers.meshPrimitives =
        buffer(5u, @"MetalRobo visual primitives");
    buffers.meshInstances =
        buffer(6u, @"MetalRobo visual instances");
    buffers.materials =
        buffer(7u, @"MetalRobo visual materials");
    buffers.textureBindings =
        buffer(8u, @"MetalRobo visual texture bindings");
    const std::array created{
        buffers.meshVertices,
        buffers.meshIndices,
        buffers.meshTriangles,
        buffers.meshClusters,
        buffers.meshTriangleClusters,
        buffers.meshPrimitives,
        buffers.meshInstances,
        buffers.materials,
        buffers.textureBindings,
    };
    if (std::ranges::any_of(
            created,
            [](id<MTLBuffer> value) {
                return value == nil;
            }
        )) {
        reason =
            "could not place visual geometry buffers in the heap";
        return false;
    }
    output.retainedBytes = output.heap.size;
    return true;
}

bool createNativeVisualResources(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLArgumentEncoder> argumentEncoder,
    RuntimeVisualScene& scene,
    __strong id<MTLBuffer>& argumentBuffer,
    NativeVisualResources& output,
    std::string& reason
) {
    @autoreleasepool {
    if (scene.textures.size() > kMaximumSceneTexturesV3) {
        reason =
            "visual scene exceeds the tier-2 native texture table";
        return false;
    }
    std::vector<const VisualTextureImageV2*> images;
    std::vector<const RuntimeVisualScene::TextureSource*> sources;
    images.reserve(scene.textures.size() + 3u);
    sources.reserve(scene.textures.size() + 3u);
    if (scene.textureSources.size() != scene.textures.size()) {
        reason = "visual texture source index is inconsistent";
        return false;
    }
    for (std::size_t index = 0u;
         index < scene.textures.size();
         ++index) {
        images.push_back(&scene.textures[index]);
        sources.push_back(&scene.textureSources[index]);
    }
    images.push_back(&scene.environmentPack.diffuseIrradiance);
    sources.push_back(&scene.environmentTextureSources[0]);
    images.push_back(&scene.environmentPack.prefilteredSpecular);
    sources.push_back(&scene.environmentTextureSources[1]);
    images.push_back(&scene.environmentPack.brdfLut);
    sources.push_back(&scene.environmentTextureSources[2]);

    std::size_t heapBytes = 0u;
    std::vector<std::size_t> offsets;
    offsets.reserve(images.size());
    for (const VisualTextureImageV2* image : images) {
        MTLTextureDescriptor* descriptor =
            textureDescriptor(*image);
        if (descriptor.pixelFormat == MTLPixelFormatInvalid) {
            reason = "visual texture pixel format is unsupported";
            return false;
        }
        const MTLSizeAndAlign sizeAndAlign =
            [device heapTextureSizeAndAlignWithDescriptor:descriptor];
        if (sizeAndAlign.size == 0u ||
            sizeAndAlign.align == 0u ||
            heapBytes >
                std::numeric_limits<std::size_t>::max() -
                    sizeAndAlign.align) {
            reason = "visual texture heap sizing failed";
            return false;
        }
        heapBytes = alignResourceOffset(
            heapBytes,
            sizeAndAlign.align
        );
        offsets.push_back(heapBytes);
        if (sizeAndAlign.size >
            std::numeric_limits<std::size_t>::max() - heapBytes) {
            reason = "visual texture heap size overflows";
            return false;
        }
        heapBytes += sizeAndAlign.size;
    }

    MTLHeapDescriptor* heapDescriptor =
        [[MTLHeapDescriptor alloc] init];
    heapDescriptor.type = MTLHeapTypePlacement;
    heapDescriptor.storageMode = MTLStorageModePrivate;
    heapDescriptor.cpuCacheMode =
        MTLCPUCacheModeDefaultCache;
    heapDescriptor.hazardTrackingMode =
        MTLHazardTrackingModeUntracked;
    heapDescriptor.size = std::max<std::size_t>(
        heapBytes,
        kMinimumAllocationBytes
    );
    output.heap = [device newHeapWithDescriptor:heapDescriptor];
    if (output.heap == nil) {
        reason = "could not allocate the visual texture placement heap";
        return false;
    }
    output.heap.label = @"MetalRobo V3 native visual resources";

    NSMutableArray<id<MTLTexture>>* textures =
        [[NSMutableArray alloc]
            initWithCapacity:images.size()];
    for (std::size_t index = 0u;
         index < images.size();
         ++index) {
        MTLTextureDescriptor* descriptor =
            textureDescriptor(*images[index]);
        id<MTLTexture> texture =
            [output.heap
                newTextureWithDescriptor:descriptor
                                  offset:offsets[index]];
        if (texture == nil) {
            reason =
                "could not place a native visual texture in the heap";
            return false;
        }
        texture.label = @(images[index]->id.c_str());
        [textures addObject:texture];
    }

    struct StagingUploadTask {
        __unsafe_unretained id<MTLTexture> texture = nil;
        const VisualTextureImageV2* image = nullptr;
        const VisualTextureSubresourceV2* subresource = nullptr;
    };
    std::vector<StagingUploadTask> stagingTasks;
    std::size_t maximumSubresourceBytes = 0u;
    bool needsMetalIO = false;
    for (std::size_t imageIndex = 0u;
         imageIndex < images.size();
         ++imageIndex) {
        const VisualTextureImageV2& image = *images[imageIndex];
        const bool streamed =
            !sources[imageIndex]->path.empty();
        needsMetalIO = needsMetalIO || streamed;
        for (const VisualTextureSubresourceV2& subresource :
             image.subresources) {
            if (!streamed &&
                (subresource.dataOffset > image.data.size() ||
                 subresource.dataSize >
                    image.data.size() - subresource.dataOffset)) {
                reason =
                    "visual texture subresource points outside its "
                    "cooked payload";
                return false;
            }
            if (!streamed) {
                maximumSubresourceBytes =
                    std::max<std::size_t>(
                        maximumSubresourceBytes,
                        subresource.dataSize
                    );
                stagingTasks.push_back({
                    textures[imageIndex],
                    &image,
                    &subresource,
                });
            }
        }
    }

    if (needsMetalIO) {
        MTLIOCommandQueueDescriptor* descriptor =
            [[MTLIOCommandQueueDescriptor alloc] init];
        descriptor.type = MTLIOCommandQueueTypeSerial;
        descriptor.priority = MTLIOPriorityHigh;
        descriptor.maxCommandBufferCount = 2u;
        descriptor.maxCommandsInFlight = 64u;
        NSError* ioError = nil;
        id<MTLIOCommandQueue> ioQueue =
            [device
                newIOCommandQueueWithDescriptor:descriptor
                                          error:&ioError];
        if (ioQueue == nil) {
            reason =
                "could not create Metal I/O texture queue: " +
                describeError(ioError);
            return false;
        }
        NSMutableDictionary<NSString*, id<MTLIOFileHandle>>*
            handles = [[NSMutableDictionary alloc] init];
        id<MTLIOCommandBuffer> command = nil;
        std::uint32_t loadCount = 0u;
        const auto finishBatch = [&]() {
            if (command == nil) {
                return true;
            }
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLIOStatusComplete) {
                reason =
                    "Metal I/O texture load failed: " +
                    describeError(command.error);
                return false;
            }
            command = nil;
            loadCount = 0u;
            return true;
        };
        for (std::size_t imageIndex = 0u;
             imageIndex < images.size();
             ++imageIndex) {
            const auto& source = *sources[imageIndex];
            if (source.path.empty()) {
                continue;
            }
            NSString* path = @(source.path.string().c_str());
            id<MTLIOFileHandle> handle = handles[path];
            if (handle == nil) {
                NSURL* url = [NSURL fileURLWithPath:path];
                handle = [device
                    newIOFileHandleWithURL:url
                                    error:&ioError];
                if (handle == nil) {
                    reason =
                        "could not open texture pack through Metal I/O: " +
                        describeError(ioError);
                    return false;
                }
                handles[path] = handle;
            }
            const VisualTextureImageV2& image =
                *images[imageIndex];
            for (const VisualTextureSubresourceV2& subresource :
                 image.subresources) {
                if (command == nil) {
                    command = [ioQueue commandBuffer];
                }
                [command
                    loadTexture:textures[imageIndex]
                           slice:subresource.arraySlice
                           level:subresource.mipLevel
                            size:MTLSizeMake(
                                subresource.width,
                                subresource.height,
                                1u
                            )
               sourceBytesPerRow:subresource.bytesPerRow
             sourceBytesPerImage:subresource.bytesPerImage
               destinationOrigin:MTLOriginMake(0u, 0u, 0u)
                    sourceHandle:handle
              sourceHandleOffset:
                    source.payloadFileOffset +
                    subresource.dataOffset];
                if (++loadCount == 64u && !finishBatch()) {
                    return false;
                }
            }
        }
        if (!finishBatch()) {
            return false;
        }
    }

    if (!stagingTasks.empty()) {
        const std::array<id<MTLBuffer>, 2u> staging{
            makeSharedBuffer(
                device,
                maximumSubresourceBytes,
                @"MetalRobo texture staging 0"
            ),
            makeSharedBuffer(
                device,
                maximumSubresourceBytes,
                @"MetalRobo texture staging 1"
            ),
        };
        if (staging[0] == nil || staging[1] == nil) {
            reason = "could not allocate bounded texture staging";
            return false;
        }
        for (std::size_t first = 0u;
             first < stagingTasks.size();
             first += staging.size()) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit =
                [command blitCommandEncoder];
            const std::size_t count = std::min<std::size_t>(
                staging.size(),
                stagingTasks.size() - first
            );
            for (std::size_t slot = 0u;
                 slot < count;
                 ++slot) {
                const StagingUploadTask& task =
                    stagingTasks[first + slot];
                std::memcpy(
                    staging[slot].contents,
                    task.image->data.data() +
                        task.subresource->dataOffset,
                    task.subresource->dataSize
                );
                [blit
                    copyFromBuffer:staging[slot]
                        sourceOffset:0u
                   sourceBytesPerRow:
                        task.subresource->bytesPerRow
                 sourceBytesPerImage:
                        task.subresource->bytesPerImage
                          sourceSize:MTLSizeMake(
                              task.subresource->width,
                              task.subresource->height,
                              1u
                          )
                           toTexture:task.texture
                    destinationSlice:
                        task.subresource->arraySlice
                    destinationLevel:
                        task.subresource->mipLevel
                   destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
            }
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                reason =
                    "native visual texture upload failed: " +
                    describeError(command.error);
                return false;
            }
        }
    }

    const std::size_t materialTextureCount =
        scene.textures.size();
    output.materialTextures = [textures
        subarrayWithRange:NSMakeRange(0u, materialTextureCount)];
    output.diffuseIrradiance =
        textures[materialTextureCount];
    output.prefilteredSpecular =
        textures[materialTextureCount + 1u];
    output.brdfLut =
        textures[materialTextureCount + 2u];

    std::array<std::uint32_t, kSceneSamplerCountV3>
        samplerRemap{};
    samplerRemap.fill(MR_INVALID_INDEX);
    NSMutableArray<id<MTLSamplerState>>* samplers =
        [[NSMutableArray alloc]
            initWithCapacity:std::min<std::size_t>(
                scene.textureBindings.size(),
                kSceneSamplerCountV3
            )];
    for (MRVisualTextureBindingGPUV2& binding :
         scene.textureBindings) {
        const std::uint32_t authoredIndex =
            binding.resource.y;
        if (authoredIndex >= kSceneSamplerCountV3) {
            reason = "visual sampler encoding is invalid";
            return false;
        }
        if (samplerRemap[authoredIndex] != MR_INVALID_INDEX) {
            binding.resource.y = samplerRemap[authoredIndex];
            continue;
        }
        const std::uint32_t addressIndex =
            authoredIndex % 9u;
        const std::uint32_t filterIndex =
            authoredIndex / 9u;
        MTLSamplerDescriptor* descriptor =
            [[MTLSamplerDescriptor alloc] init];
        descriptor.minFilter =
            (filterIndex & 1u) != 0u
            ? MTLSamplerMinMagFilterNearest
            : MTLSamplerMinMagFilterLinear;
        descriptor.magFilter =
            (filterIndex & 2u) != 0u
            ? MTLSamplerMinMagFilterNearest
            : MTLSamplerMinMagFilterLinear;
        const std::uint32_t mipClass = filterIndex >> 2u;
        descriptor.mipFilter =
            mipClass == 0u
            ? MTLSamplerMipFilterNotMipmapped
            : mipClass == 1u
                ? MTLSamplerMipFilterNearest
                : MTLSamplerMipFilterLinear;
        const auto address = [](
            const std::uint32_t addressClass
        ) {
            return addressClass == 1u
            ? MTLSamplerAddressModeClampToEdge
            : addressClass == 2u
                ? MTLSamplerAddressModeMirrorRepeat
                : MTLSamplerAddressModeRepeat;
        };
        descriptor.sAddressMode =
            address(addressIndex / 3u);
        descriptor.tAddressMode =
            address(addressIndex % 3u);
        descriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.normalizedCoordinates = YES;
        id<MTLSamplerState> sampler =
            [device newSamplerStateWithDescriptor:descriptor];
        if (sampler == nil) {
            reason = "could not create visual texture sampler table";
            return false;
        }
        samplerRemap[authoredIndex] =
            static_cast<std::uint32_t>(samplers.count);
        binding.resource.y = samplerRemap[authoredIndex];
        [samplers addObject:sampler];
    }
    output.samplers = samplers;

    argumentBuffer = makeSharedBuffer(
        device,
        argumentEncoder.encodedLength,
        @"MetalRobo V3 visual resource argument buffer"
    );
    if (argumentBuffer == nil) {
        reason = "could not allocate visual resource argument buffer";
        return false;
    }
    [argumentEncoder setArgumentBuffer:argumentBuffer offset:0u];
    for (std::uint32_t index = 0u;
         index < materialTextureCount;
         ++index) {
        [argumentEncoder
            setTexture:output.materialTextures[index]
               atIndex:index];
    }
    for (std::uint32_t index = 0u;
         index < output.samplers.count;
         ++index) {
        [argumentEncoder
            setSamplerState:output.samplers[index]
                    atIndex:kMaximumSceneTexturesV3 + index];
    }
    [argumentEncoder
        setTexture:output.diffuseIrradiance
           atIndex:kEnvironmentArgumentBaseV3];
    [argumentEncoder
        setTexture:output.prefilteredSpecular
           atIndex:kEnvironmentArgumentBaseV3 + 1u];
    [argumentEncoder
        setTexture:output.brdfLut
           atIndex:kEnvironmentArgumentBaseV3 + 2u];

    output.retainedBytes =
        output.heap.size + argumentBuffer.length;
    for (VisualTextureImageV2& texture : scene.textures) {
        std::vector<std::uint8_t>{}.swap(texture.data);
    }
    std::vector<std::uint8_t>{}.swap(
        scene.environmentPack.diffuseIrradiance.data
    );
    std::vector<std::uint8_t>{}.swap(
        scene.environmentPack.prefilteredSpecular.data
    );
    std::vector<std::uint8_t>{}.swap(
        scene.environmentPack.brdfLut.data
    );
    return true;
    }
}

bool loadAndPrepareStreamedGeometry(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> rebasePipeline,
    id<MTLComputePipelineState> rebaseMaterialPipeline,
    id<MTLComputePipelineState> expandPipeline,
    id<MTLComputePipelineState> buildClustersPipeline,
    const RuntimeVisualScene& scene,
    const RendererBuffers& buffers,
    std::string& reason
) {
    if (scene.geometrySources.empty()) {
        return true;
    }
    MTLIOCommandQueueDescriptor* ioDescriptor =
        [[MTLIOCommandQueueDescriptor alloc] init];
    ioDescriptor.type = MTLIOCommandQueueTypeSerial;
    ioDescriptor.priority = MTLIOPriorityHigh;
    ioDescriptor.maxCommandBufferCount = 2u;
    ioDescriptor.maxCommandsInFlight = 16u;
    NSError* error = nil;
    id<MTLIOCommandQueue> ioQueue =
        [device
            newIOCommandQueueWithDescriptor:ioDescriptor
                                      error:&error];
    if (ioQueue == nil) {
        reason =
            "could not create Metal I/O geometry queue: " +
            describeError(error);
        return false;
    }
    NSMutableDictionary<NSString*, id<MTLIOFileHandle>>* handles =
        [[NSMutableDictionary alloc] init];
    id<MTLIOCommandBuffer> ioCommand = nil;
    std::uint32_t ioLoadCount = 0u;
    const auto finishBatch = [&]() {
        if (ioCommand == nil) {
            return true;
        }
        [ioCommand commit];
        [ioCommand waitUntilCompleted];
        if (ioCommand.status != MTLIOStatusComplete) {
            reason =
                "Metal I/O geometry load failed: " +
                describeError(ioCommand.error);
            return false;
        }
        ioCommand = nil;
        ioLoadCount = 0u;
        return true;
    };
    for (const RuntimeVisualScene::GeometrySource& source :
         scene.geometrySources) {
        NSString* path = @(source.path.string().c_str());
        id<MTLIOFileHandle> handle = handles[path];
        if (handle == nil) {
            NSURL* url = [NSURL fileURLWithPath:path];
            handle = [device
                newIOFileHandleWithURL:url
                                error:&error];
            if (handle == nil) {
                reason =
                    "could not open geometry pack through Metal I/O: " +
                    describeError(error);
                return false;
            }
            handles[path] = handle;
        }
        if (ioCommand == nil) {
            ioCommand = [ioQueue commandBuffer];
        }
        [ioCommand
            loadBuffer:buffers.meshVertices
                 offset:
                    static_cast<std::size_t>(source.vertexBase) *
                    sizeof(MRVisualVertexGPUV2)
                   size:source.vertexByteCount
           sourceHandle:handle
     sourceHandleOffset:source.vertexFileOffset];
        [ioCommand
            loadBuffer:buffers.meshIndices
                 offset:
                    static_cast<std::size_t>(source.indexBase) *
                    sizeof(std::uint32_t)
                   size:source.indexByteCount
           sourceHandle:handle
     sourceHandleOffset:source.indexFileOffset];
        [ioCommand
            loadBuffer:buffers.materials
                 offset:
                    static_cast<std::size_t>(source.materialBase) *
                    sizeof(MRVisualMaterialGPUV2)
                   size:source.materialByteCount
           sourceHandle:handle
     sourceHandleOffset:source.materialFileOffset];
        ioLoadCount += 3u;
        if (ioLoadCount >= 64u && !finishBatch()) {
            return false;
        }
    }
    if (!finishBatch()) {
        return false;
    }

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
        reason =
            "could not create streamed geometry preparation command";
        return false;
    }
    [encoder setComputePipelineState:rebasePipeline];
    [encoder setBuffer:buffers.meshIndices offset:0u atIndex:0u];
    const NSUInteger rebaseThreads = std::min<NSUInteger>(
        rebasePipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    for (const RuntimeVisualScene::GeometrySource& source :
         scene.geometrySources) {
        const mr_uint4 range{
            source.indexBase,
            static_cast<std::uint32_t>(
                source.indexByteCount / sizeof(std::uint32_t)
            ),
            source.vertexBase,
            0u,
        };
        [encoder setBytes:&range
                   length:sizeof(range)
                  atIndex:1u];
        [encoder dispatchThreads:MTLSizeMake(
                                     range.y,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      rebaseThreads,
                                      1u,
                                      1u
                                  )];
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:rebaseMaterialPipeline];
    [encoder setBuffer:buffers.materials offset:0u atIndex:0u];
    const NSUInteger materialThreads = std::min<NSUInteger>(
        rebaseMaterialPipeline.maxTotalThreadsPerThreadgroup,
        128u
    );
    for (const RuntimeVisualScene::GeometrySource& source :
         scene.geometrySources) {
        const mr_uint4 range{
            source.materialBase,
            static_cast<std::uint32_t>(
                source.materialByteCount /
                sizeof(MRVisualMaterialGPUV2)
            ),
            source.bindingBase,
            0u,
        };
        [encoder setBytes:&range
                   length:sizeof(range)
                  atIndex:1u];
        [encoder dispatchThreads:MTLSizeMake(
                                     range.y,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      materialThreads,
                                      1u,
                                      1u
                                  )];
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:expandPipeline];
    [encoder setBuffer:buffers.meshIndices offset:0u atIndex:0u];
    [encoder setBuffer:buffers.meshPrimitives offset:0u atIndex:1u];
    [encoder setBuffer:buffers.meshTriangles offset:0u atIndex:2u];
    const std::uint32_t primitiveCount =
        static_cast<std::uint32_t>(scene.primitives.size());
    [encoder setBytes:&primitiveCount
               length:sizeof(primitiveCount)
              atIndex:3u];
    const NSUInteger expandThreads = std::min<NSUInteger>(
        expandPipeline.maxTotalThreadsPerThreadgroup,
        128u
    );
    [encoder dispatchThreads:MTLSizeMake(
                                 primitiveCount,
                                 1u,
                                 1u
                             )
        threadsPerThreadgroup:MTLSizeMake(
                                  expandThreads,
                                  1u,
                                  1u
                              )];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:buildClustersPipeline];
    [encoder setBuffer:buffers.meshTriangles offset:0u atIndex:0u];
    [encoder setBuffer:buffers.meshVertices offset:0u atIndex:1u];
    [encoder setBuffer:buffers.meshClusters offset:0u atIndex:2u];
    const std::uint32_t clusterCount =
        static_cast<std::uint32_t>(scene.meshClusters.size());
    [encoder setBytes:&clusterCount
               length:sizeof(clusterCount)
              atIndex:3u];
    const NSUInteger clusterThreads = std::min<NSUInteger>(
        buildClustersPipeline.maxTotalThreadsPerThreadgroup,
        128u
    );
    if (clusterCount != 0u) {
        [encoder dispatchThreads:MTLSizeMake(
                                     clusterCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      clusterThreads,
                                      1u,
                                      1u
                                  )];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        reason =
            "streamed geometry preparation failed: " +
            describeError(command.error);
        return false;
    }
    return true;
}

bool buildCompactedPrimitiveAccelerationStructures(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLBuffer> vertices,
    id<MTLBuffer> indices,
    const std::span<const MRVisualMaterialGPUV2> materials,
    const std::span<const RayGeometryKey> geometryKeys,
    const std::size_t maximumRetainedBytes,
    CompactedPrimitiveAccelerationStructures& output,
    std::string& reason
) {
    @autoreleasepool {
    output = {};
    if (geometryKeys.empty()) {
        output.structures = @[];
        return true;
    }
    if (![device supportsRaytracing]) {
        reason =
            "sensor_reference requires Metal ray-tracing support";
        return false;
    }

    NSMutableArray<MTLPrimitiveAccelerationStructureDescriptor*>*
        descriptors = [NSMutableArray
            arrayWithCapacity:geometryKeys.size()];
    std::vector<MTLAccelerationStructureSizes> sizes;
    sizes.reserve(geometryKeys.size());
    NSUInteger maximumScratchBytes = 0u;
    for (const RayGeometryKey& key : geometryKeys) {
        NSMutableArray<MTLAccelerationStructureGeometryDescriptor*>*
            geometryDescriptors = [NSMutableArray
                arrayWithCapacity:key.size()];
        for (const std::array<std::uint32_t, 3u>& geometry : key) {
            MTLAccelerationStructureTriangleGeometryDescriptor*
                triangle =
                    [MTLAccelerationStructureTriangleGeometryDescriptor
                        descriptor];
            triangle.vertexBuffer = vertices;
            triangle.vertexBufferOffset = 0u;
            // Positions are the first xyz values in the authored V2 vertex;
            // ray hardware walks the shared interleaved stream directly.
            triangle.vertexFormat = MTLAttributeFormatFloat3;
            triangle.vertexStride = sizeof(MRVisualVertexGPUV2);
            triangle.indexBuffer = indices;
            triangle.indexBufferOffset =
                static_cast<NSUInteger>(geometry[0]) *
                sizeof(std::uint32_t);
            triangle.indexType = MTLIndexTypeUInt32;
            triangle.triangleCount = geometry[1] / 3u;
            triangle.opaque =
                geometry[2] >= materials.size() ||
                materials[geometry[2]].flags.x ==
                    MR_VISUAL_ALPHA_OPAQUE;
            [geometryDescriptors addObject:triangle];
        }
        MTLPrimitiveAccelerationStructureDescriptor* descriptor =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        descriptor.geometryDescriptors = geometryDescriptors;
        descriptor.usage =
            MTLAccelerationStructureUsagePreferFastIntersection;
        [descriptors addObject:descriptor];
        const MTLAccelerationStructureSizes measured =
            [device accelerationStructureSizesWithDescriptor:descriptor];
        if (measured.accelerationStructureSize == 0u ||
            measured.buildScratchBufferSize == 0u) {
            reason =
                "Metal returned an invalid primitive acceleration "
                "structure size";
            return false;
        }
        sizes.push_back(measured);
        maximumScratchBytes = std::max(
            maximumScratchBytes,
            measured.buildScratchBufferSize
        );
    }

    id<MTLBuffer> scratch = makePrivateBuffer(
        device,
        maximumScratchBytes,
        @"MetalRobo reusable BLAS build scratch"
    );
    if (scratch == nil) {
        reason =
            "could not allocate the reusable BLAS build workspace";
        return false;
    }

    // Conservative BLAS builds are transient and can be substantially larger
    // than their compacted result. Build only a bounded group at a time,
    // reusing one scratch allocation across the entire scene.
    constexpr std::size_t kMaximumTransientBuildBatchBytes =
        256ull * 1024ull * 1024ull;
    const std::size_t transientBatchBudget = std::max<std::size_t>(
        1u,
        std::min(
            kMaximumTransientBuildBatchBytes,
            maximumRetainedBytes / 4u
        )
    );
    NSMutableArray<id<MTLAccelerationStructure>>* compacted =
        [NSMutableArray arrayWithCapacity:geometryKeys.size()];
    std::size_t retainedBytes = 0u;
    std::size_t first = 0u;
    while (first < sizes.size()) {
        @autoreleasepool {
        std::size_t end = first;
        std::size_t batchBytes = 0u;
        while (end < sizes.size()) {
            const std::size_t candidate =
                sizes[end].accelerationStructureSize;
            if (end != first &&
                candidate > transientBatchBudget - std::min(
                    transientBatchBudget,
                    batchBytes
                )) {
                break;
            }
            if (!checkedAdd(batchBytes, candidate, batchBytes)) {
                reason =
                    "primitive acceleration structure size overflows";
                return false;
            }
            ++end;
            if (batchBytes >= transientBatchBudget) {
                break;
            }
        }

        NSMutableArray<id<MTLAccelerationStructure>>* conservative =
            [NSMutableArray arrayWithCapacity:end - first];
        id<MTLBuffer> compactedSizes = makeSharedBuffer(
            device,
            (end - first) * sizeof(std::uint64_t),
            @"MetalRobo BLAS compacted sizes"
        );
        id<MTLCommandBuffer> buildCommand = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEncoder =
            [buildCommand accelerationStructureCommandEncoder];
        if (compactedSizes == nil ||
            buildCommand == nil ||
            buildEncoder == nil) {
            reason =
                "could not allocate a primitive acceleration "
                "structure build batch";
            return false;
        }
        for (std::size_t index = first; index < end; ++index) {
            id<MTLAccelerationStructure> structure =
                [device newAccelerationStructureWithSize:
                    sizes[index].accelerationStructureSize];
            if (structure == nil) {
                reason =
                    "could not allocate a primitive acceleration "
                    "structure";
                return false;
            }
            [conservative addObject:structure];
            [buildEncoder
                buildAccelerationStructure:structure
                descriptor:descriptors[index]
                scratchBuffer:scratch
                scratchBufferOffset:0u];
            [buildEncoder
                writeCompactedAccelerationStructureSize:structure
                toBuffer:compactedSizes
                offset:(index - first) * sizeof(std::uint64_t)
                sizeDataType:MTLDataTypeULong];
        }
        [buildEncoder endEncoding];
        [buildCommand commit];
        [buildCommand waitUntilCompleted];
        if (buildCommand.status != MTLCommandBufferStatusCompleted) {
            reason =
                "primitive acceleration structure build failed: " +
                describeError(buildCommand.error);
            return false;
        }

        const auto* compactedByteCounts =
            static_cast<const std::uint64_t*>(
                compactedSizes.contents
            );
        NSMutableArray<id<MTLAccelerationStructure>>*
            compactedBatch =
                [NSMutableArray arrayWithCapacity:end - first];
        for (std::size_t index = first; index < end; ++index) {
            const std::uint64_t byteCount =
                compactedByteCounts[index - first];
            if (byteCount == 0u ||
                byteCount >
                    std::numeric_limits<NSUInteger>::max() ||
                !checkedAdd(
                    retainedBytes,
                    static_cast<std::size_t>(byteCount),
                    retainedBytes
                ) ||
                retainedBytes > maximumRetainedBytes) {
                reason =
                    "compacted acceleration structures exceed the "
                    "configured retained-memory budget";
                return false;
            }
            id<MTLAccelerationStructure> structure =
                [device newAccelerationStructureWithSize:
                    static_cast<NSUInteger>(byteCount)];
            if (structure == nil) {
                reason =
                    "could not allocate a compacted primitive "
                    "acceleration structure";
                return false;
            }
            [compactedBatch addObject:structure];
        }

        id<MTLCommandBuffer> compactCommand = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> compactEncoder =
            [compactCommand accelerationStructureCommandEncoder];
        if (compactCommand == nil || compactEncoder == nil) {
            reason =
                "could not create the BLAS compaction command";
            return false;
        }
        for (std::size_t local = 0u;
             local < compactedBatch.count;
             ++local) {
            [compactEncoder
                copyAndCompactAccelerationStructure:
                    conservative[local]
                toAccelerationStructure:compactedBatch[local]];
        }
        [compactEncoder endEncoding];
        [compactCommand commit];
        [compactCommand waitUntilCompleted];
        if (compactCommand.status != MTLCommandBufferStatusCompleted) {
            reason =
                "primitive acceleration structure compaction failed: " +
                describeError(compactCommand.error);
            return false;
        }
        [compacted addObjectsFromArray:compactedBatch];
        first = end;
        }
    }

    output.structures = [compacted copy];
    output.retainedBytes = retainedBytes;
    return true;
    }
}

struct AcquiredReferenceWorkspace {
    std::shared_ptr<detail::ReferenceFrameWorkspace> workspace;
    __strong NSArray<MTLInstanceAccelerationStructureDescriptor*>*
        descriptors = nil;
};

AcquiredReferenceWorkspace acquireReferenceWorkspace(
    detail::MetalHybridRendererState& state,
    const std::size_t motionBodyBytes,
    const std::uint32_t environmentCount,
    const std::uint32_t instancesPerEnvironment,
    const std::uint32_t keyframeCount,
    std::string& reason
) {
    AcquiredReferenceWorkspace result;
    std::size_t instanceCountValue = 0u;
    std::size_t transformCount = 0u;
    std::size_t descriptorBytes = 0u;
    std::size_t transformBytes = 0u;
    if (environmentCount == 0u ||
        instancesPerEnvironment == 0u ||
        keyframeCount < 2u ||
        !checkedMultiply(
            environmentCount,
            instancesPerEnvironment,
            instanceCountValue
        ) ||
        instanceCountValue >
            std::numeric_limits<std::uint32_t>::max() ||
        !checkedMultiply(
            instanceCountValue,
            keyframeCount,
            transformCount
        ) ||
        !checkedBytes<MTLAccelerationStructureMotionInstanceDescriptor>(
            instanceCountValue,
            descriptorBytes
        ) ||
        !checkedBytes<MTLComponentTransform>(
            transformCount,
            transformBytes
        )) {
        reason =
            "reference instance or motion-transform count overflows";
        return result;
    }
    const std::uint32_t instanceCount =
        static_cast<std::uint32_t>(instanceCountValue);
    const std::uint32_t groupCount =
        (environmentCount - 1u) /
            kReferenceEnvironmentsPerTLAS +
        1u;

    for (const auto& candidate : state.referenceWorkspaces) {
        bool expected = false;
        if (candidate != nullptr &&
            candidate->inUse.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel
            )) {
            result.workspace = candidate;
            break;
        }
    }
    if (result.workspace == nullptr) {
        if (state.referenceWorkspaces.size() >=
            state.config.maximumReferenceFramesInFlight) {
            reason =
                "all recycled sensor_reference frame workspaces are "
                "still in flight";
            return result;
        }
        result.workspace =
            std::make_shared<detail::ReferenceFrameWorkspace>();
        result.workspace->inUse.store(
            true,
            std::memory_order_release
        );
        state.referenceWorkspaces.push_back(result.workspace);
    }
    const auto releaseOnFailure = [&result]() {
        result.workspace->inUse.store(
            false,
            std::memory_order_release
        );
        result.workspace.reset();
    };

    const bool topologyChanged =
        result.workspace->environmentCount != environmentCount ||
        result.workspace->instancesPerEnvironment !=
            instancesPerEnvironment ||
        result.workspace->instanceCount != instanceCount ||
        result.workspace->keyframeCount != keyframeCount;
    bool topologyStorageChanged = false;
    if (result.workspace->motionBodies == nil ||
        result.workspace->motionBodies.length < motionBodyBytes) {
        result.workspace->motionBodies = makeSharedBuffer(
            state.device,
            motionBodyBytes,
            @"MetalRobo recycled reference body history"
        );
    }
    if (result.workspace->instanceDescriptors == nil ||
        result.workspace->instanceDescriptors.length <
            descriptorBytes) {
        result.workspace->instanceDescriptors = makePrivateBuffer(
            state.device,
            descriptorBytes,
            @"MetalRobo recycled ray instance descriptors"
        );
        topologyStorageChanged = true;
    }
    if (result.workspace->motionTransforms == nil ||
        result.workspace->motionTransforms.length <
            transformBytes) {
        result.workspace->motionTransforms = makePrivateBuffer(
            state.device,
            transformBytes,
            @"MetalRobo recycled component motion transforms"
        );
        topologyStorageChanged = true;
    }
    if (result.workspace->motionBodies == nil ||
        result.workspace->instanceDescriptors == nil ||
        result.workspace->motionTransforms == nil) {
        reason =
            "could not allocate recycled reference frame buffers";
        releaseOnFailure();
        return result;
    }

    const bool rebuildTopologyResources =
        topologyChanged ||
        topologyStorageChanged ||
        result.workspace->tlasDescriptors == nil ||
        result.workspace->tlasDescriptors.count != groupCount ||
        result.workspace->instanceStructures == nil ||
        result.workspace->instanceStructures.count != groupCount ||
        result.workspace->buildScratch == nil ||
        result.workspace->buildScratchOffsets.size() != groupCount;
    if (rebuildTopologyResources) {
        NSMutableArray<MTLInstanceAccelerationStructureDescriptor*>*
            descriptors = [NSMutableArray arrayWithCapacity:groupCount];
        std::vector<MTLAccelerationStructureSizes> groupSizes;
        groupSizes.reserve(groupCount);
        std::vector<std::size_t> scratchOffsets;
        scratchOffsets.reserve(groupCount);
        std::size_t totalScratchBytes = 0u;
        for (std::uint32_t group = 0u;
             group < groupCount;
             ++group) {
            const std::uint32_t firstEnvironment =
                group * kReferenceEnvironmentsPerTLAS;
            const std::uint32_t environmentsInGroup = std::min(
                kReferenceEnvironmentsPerTLAS,
                environmentCount - firstEnvironment
            );
            const std::size_t firstInstance =
                static_cast<std::size_t>(firstEnvironment) *
                instancesPerEnvironment;
            const std::size_t groupInstances =
                static_cast<std::size_t>(environmentsInGroup) *
                instancesPerEnvironment;
            const std::size_t firstTransform =
                firstInstance * keyframeCount;
            const std::size_t groupTransforms =
                groupInstances * keyframeCount;

            MTLInstanceAccelerationStructureDescriptor* descriptor =
                [MTLInstanceAccelerationStructureDescriptor descriptor];
            descriptor.instancedAccelerationStructures =
                state.primitiveAccelerationStructures;
            descriptor.instanceDescriptorBuffer =
                result.workspace->instanceDescriptors;
            descriptor.instanceDescriptorBufferOffset =
                firstInstance *
                sizeof(MTLAccelerationStructureMotionInstanceDescriptor);
            descriptor.instanceDescriptorStride =
                sizeof(MTLAccelerationStructureMotionInstanceDescriptor);
            descriptor.instanceCount = groupInstances;
            descriptor.instanceDescriptorType =
                MTLAccelerationStructureInstanceDescriptorTypeMotion;
            descriptor.motionTransformBuffer =
                result.workspace->motionTransforms;
            descriptor.motionTransformBufferOffset =
                firstTransform * sizeof(MTLComponentTransform);
            descriptor.motionTransformCount = groupTransforms;
            descriptor.motionTransformType = MTLTransformTypeComponent;
            descriptor.motionTransformStride =
                sizeof(MTLComponentTransform);
            descriptor.usage =
                MTLAccelerationStructureUsageRefit |
                MTLAccelerationStructureUsagePreferFastIntersection;
            const MTLAccelerationStructureSizes sizes =
                [state.device
                    accelerationStructureSizesWithDescriptor:descriptor];
            const std::size_t scratchBytes = std::max(
                sizes.buildScratchBufferSize,
                sizes.refitScratchBufferSize
            );
            if (sizes.accelerationStructureSize == 0u ||
                scratchBytes == 0u) {
                reason =
                    "Metal returned an invalid grouped reference "
                    "acceleration-structure size";
                releaseOnFailure();
                return result;
            }
            const std::size_t scratchOffset = alignResourceOffset(
                totalScratchBytes,
                kAccelerationStructureScratchAlignment
            );
            if (scratchOffset < totalScratchBytes ||
                !checkedAdd(
                    scratchOffset,
                    scratchBytes,
                    totalScratchBytes
                )) {
                reason =
                    "grouped reference TLAS scratch size overflows";
                releaseOnFailure();
                return result;
            }
            scratchOffsets.push_back(scratchOffset);
            groupSizes.push_back(sizes);
            [descriptors addObject:descriptor];
        }

        bool allocateStructures =
            topologyChanged ||
            result.workspace->instanceStructures == nil ||
            result.workspace->instanceStructures.count != groupCount;
        if (!allocateStructures) {
            for (std::uint32_t group = 0u;
                 group < groupCount;
                 ++group) {
                if (result.workspace->instanceStructures[group].size <
                    groupSizes[group].accelerationStructureSize) {
                    allocateStructures = true;
                    break;
                }
            }
        }
        if (allocateStructures) {
            NSMutableArray<id<MTLAccelerationStructure>>* structures =
                [NSMutableArray arrayWithCapacity:groupCount];
            std::size_t structureBytes = 0u;
            for (const MTLAccelerationStructureSizes sizes :
                 groupSizes) {
                id<MTLAccelerationStructure> structure =
                    [state.device newAccelerationStructureWithSize:
                        sizes.accelerationStructureSize];
                if (structure == nil ||
                    !checkedAdd(
                        structureBytes,
                        sizes.accelerationStructureSize,
                        structureBytes
                    )) {
                    reason =
                        "could not allocate grouped reference TLASes";
                    releaseOnFailure();
                    return result;
                }
                [structures addObject:structure];
            }
            result.workspace->instanceStructures = [structures copy];
            result.workspace->structureBytes = structureBytes;
            result.workspace->built = false;
            result.workspace->refitCount = 0u;
        }
        if (result.workspace->buildScratch == nil ||
            result.workspace->scratchBytes < totalScratchBytes) {
            result.workspace->buildScratch = makePrivateBuffer(
                state.device,
                totalScratchBytes,
                @"MetalRobo grouped TLAS build scratch"
            );
            result.workspace->scratchBytes = totalScratchBytes;
        }
        if (result.workspace->instanceStructures == nil ||
            result.workspace->instanceStructures.count != groupCount ||
            result.workspace->buildScratch == nil) {
            reason =
                "could not allocate the grouped reference TLAS workspace";
            releaseOnFailure();
            return result;
        }
        result.workspace->tlasDescriptors = [descriptors copy];
        result.workspace->buildScratchOffsets =
            std::move(scratchOffsets);
        result.workspace->built = false;
        result.workspace->refitCount = 0u;
    }
    result.workspace->motionBodyBytes = motionBodyBytes;
    result.workspace->environmentCount = environmentCount;
    result.workspace->instancesPerEnvironment =
        instancesPerEnvironment;
    result.workspace->instanceCount = instanceCount;
    result.workspace->keyframeCount = keyframeCount;
    result.workspace->retainedBytes =
        result.workspace->motionBodies.length +
        result.workspace->instanceDescriptors.length +
        result.workspace->motionTransforms.length +
        result.workspace->scratchBytes +
        result.workspace->structureBytes;

    std::size_t totalRetained = state.layout.retainedPrivateBytes;
    for (const auto& workspace : state.referenceWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "reference workspace retained byte count overflows";
            releaseOnFailure();
            return result;
        }
    }
    for (const auto& workspace : state.exposureWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "visual frame workspace retained byte count "
                "overflows";
            releaseOnFailure();
            return result;
        }
    }
    if (totalRetained > state.config.maximumRetainedBytes) {
        reason =
            "recycled reference workspaces exceed the configured "
            "unified-memory budget";
        releaseOnFailure();
        return result;
    }
    result.descriptors = result.workspace->tlasDescriptors;
    return result;
}

std::shared_ptr<detail::ExposureFrameWorkspace>
acquireExposureWorkspace(
    detail::MetalHybridRendererState& state,
    const std::size_t motionBodyBytes,
    std::string& reason
) {
    std::shared_ptr<detail::ExposureFrameWorkspace> result;
    for (const auto& candidate : state.exposureWorkspaces) {
        bool expected = false;
        if (candidate != nullptr &&
            candidate->inUse.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel
            )) {
            result = candidate;
            break;
        }
    }
    if (result == nullptr) {
        if (state.exposureWorkspaces.size() >=
            state.config.maximumReferenceFramesInFlight) {
            reason =
                "all recycled physical-exposure workspaces are "
                "still in flight";
            return nullptr;
        }
        result =
            std::make_shared<detail::ExposureFrameWorkspace>();
        result->inUse.store(true, std::memory_order_release);
        state.exposureWorkspaces.push_back(result);
    }
    if (result->motionBodies == nil ||
        result->motionBodies.length < motionBodyBytes) {
        result->motionBodies = makeSharedBuffer(
            state.device,
            motionBodyBytes,
            @"MetalRobo recycled fast-shutter body history"
        );
    }
    if (result->motionBodies == nil) {
        reason =
            "could not allocate the recycled physical-exposure "
            "body history";
        result->inUse.store(false, std::memory_order_release);
        return nullptr;
    }
    result->retainedBytes = result->motionBodies.length;
    std::size_t totalRetained = state.layout.retainedPrivateBytes;
    for (const auto& workspace : state.referenceWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "visual frame workspace retained byte count "
                "overflows";
            result->inUse.store(false, std::memory_order_release);
            return nullptr;
        }
    }
    for (const auto& workspace : state.exposureWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "physical-exposure workspace retained byte count "
                "overflows";
            result->inUse.store(false, std::memory_order_release);
            return nullptr;
        }
    }
    if (totalRetained > state.config.maximumRetainedBytes) {
        reason =
            "recycled physical-exposure workspaces exceed the "
            "configured unified-memory budget";
        result->inUse.store(false, std::memory_order_release);
        return nullptr;
    }
    return result;
}

struct EncodePassOptions {
    NSUInteger currentBodyOffset = 0u;
    NSUInteger previousBodyOffset = 0u;
    std::uint32_t temporalSample = 0u;
    std::uint32_t temporalSampleCount = 1u;
    bool physicalExposure = false;
    bool clearAccumulation = false;
    bool accumulateRadiance = false;
    bool resolveAccumulation = false;
    bool applySensor = true;
    bool renderMeshes = true;
    bool truthOnly = false;
    bool updateShadows = true;
    MRHybridRenderUniformsGPU* encodedUniforms = nullptr;
    std::uint32_t bandFirst = 0u;
    std::uint32_t bandCount = 0u;
    std::uint32_t bandAxis = 0u;
    std::uint32_t motionKeyframes = 0u;
    float exposureFraction = -1.0f;
    float previousExposureFraction = 0.0f;
    float shutterWindowSeconds = 0.0f;
    bool interpolateMotion = false;
    bool resourcesResident = false;
    bool bodyBuffersValidated = false;
    const HybridDeviceObservationBuffers* outputs = nullptr;
};

class HybridComputeEncoder {
public:
    explicit HybridComputeEncoder(
        id<MTLComputeCommandEncoder> encoder
    )
        : native_(encoder) {}

    explicit HybridComputeEncoder(
        const MetalHybridComputeEncoderCallbacks& callbacks
    )
        : callbacks_(&callbacks) {}

    [[nodiscard]] bool valid() const noexcept {
        return native_ != nil ||
            (callbacks_ != nullptr && callbacks_->valid());
    }

    void setLabel(const char* label) {
        if (native_ != nil) {
            native_.label = @(label);
        } else if (callbacks_->setLabel != nullptr) {
            callbacks_->setLabel(callbacks_->context, label);
        }
    }

    void useResources(
        id<MTLHeap> heap,
        id<MTLResidencySet> residencySet
    ) {
        if (native_ != nil) {
            [native_ useHeap:heap];
        } else if (
            residencySet != nil &&
            callbacks_->useResidencySet != nullptr
        ) {
            callbacks_->useResidencySet(
                callbacks_->context,
                (__bridge void*)residencySet
            );
        } else if (callbacks_->useHeap != nullptr) {
            callbacks_->useHeap(
                callbacks_->context,
                (__bridge void*)heap
            );
        }
    }

    void setPipeline(id<MTLComputePipelineState> pipeline) {
        if (native_ != nil) {
            [native_ setComputePipelineState:pipeline];
        } else {
            callbacks_->setPipeline(
                callbacks_->context,
                (__bridge void*)pipeline
            );
        }
    }

    void setBuffer(
        id<MTLBuffer> buffer,
        const NSUInteger offset,
        const NSUInteger index
    ) {
        if (native_ != nil) {
            [native_ setBuffer:buffer offset:offset atIndex:index];
        } else {
            callbacks_->setBuffer(
                callbacks_->context,
                (__bridge void*)buffer,
                offset,
                static_cast<std::uint32_t>(index)
            );
        }
    }

    template <std::size_t Count>
    void setBuffers(
        id<MTLBuffer> __unsafe_unretained (&buffers)[Count],
        const NSUInteger (&offsets)[Count],
        const NSUInteger first
    ) {
        if (native_ != nil) {
            [native_ setBuffers:buffers
                        offsets:offsets
                      withRange:NSMakeRange(first, Count)];
            return;
        }
        for (std::size_t index = 0u; index < Count; ++index) {
            setBuffer(buffers[index], offsets[index], first + index);
        }
    }

    void setBytes(
        const void* bytes,
        const std::size_t length,
        const NSUInteger index
    ) {
        if (native_ != nil) {
            [native_ setBytes:bytes length:length atIndex:index];
        } else {
            callbacks_->setBytes(
                callbacks_->context,
                bytes,
                length,
                static_cast<std::uint32_t>(index)
            );
        }
    }

    void dispatchThreads(
        const NSUInteger count,
        const NSUInteger threadsPerThreadgroup
    ) {
        if (native_ != nil) {
            [native_ dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    threadsPerThreadgroup,
                    1u,
                    1u
                )];
        } else {
            callbacks_->dispatchThreads(
                callbacks_->context,
                count,
                threadsPerThreadgroup
            );
        }
    }

    void dispatchThreadgroups(
        const NSUInteger count,
        const NSUInteger threadsPerThreadgroup
    ) {
        if (native_ != nil) {
            [native_ dispatchThreadgroups:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    threadsPerThreadgroup,
                    1u,
                    1u
                )];
        } else {
            callbacks_->dispatchThreadgroups(
                callbacks_->context,
                count,
                threadsPerThreadgroup
            );
        }
    }

    [[nodiscard]] bool dispatchThreadgroupsIndirect(
        id<MTLBuffer> arguments,
        const NSUInteger offset,
        const NSUInteger threadsPerThreadgroup
    ) {
        if (native_ != nil) {
            [native_
                dispatchThreadgroupsWithIndirectBuffer:arguments
                indirectBufferOffset:offset
                threadsPerThreadgroup:MTLSizeMake(
                    threadsPerThreadgroup,
                    1u,
                    1u
                )];
            return true;
        }
        if (callbacks_->dispatchThreadgroupsIndirect == nullptr) {
            return false;
        }
        callbacks_->dispatchThreadgroupsIndirect(
            callbacks_->context,
            (__bridge void*)arguments,
            offset,
            threadsPerThreadgroup
        );
        return true;
    }

private:
    __unsafe_unretained id<MTLComputeCommandEncoder> native_ = nil;
    const MetalHybridComputeEncoderCallbacks* callbacks_ = nullptr;
};

struct EncodeWorldResources {
    MetalWorldFamilyLayout layout{};
    __strong id<MTLBuffer> instances = nil;
    __strong id<MTLBuffer> assets = nil;
    __strong id<MTLBuffer> sensors = nil;
    __strong id<MTLBuffer> appearances = nil;
};

bool resolveEncodeWorldResources(
    const detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const std::uint32_t environmentCount,
    const std::uint32_t cameraIndex,
    EncodeWorldResources& resources,
    std::string& reason
) {
    resources.layout = worlds.layout();
    if (environmentCount == 0u ||
        environmentCount > state.layout.capacity ||
        environmentCount > resources.layout.activeInstanceCount ||
        resources.layout.assetCountPerInstance < state.assetCount ||
        cameraIndex >= resources.layout.sensorCountPerInstance ||
        (state.layout.sensorBindingCount != 0u &&
         cameraIndex >= state.layout.sensorBindingCount)) {
        reason =
            "sampled world count, assets, sensor bindings, or camera "
            "are incompatible with the visual scene";
        return false;
    }
    resources.instances =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::instanceHeaders
        );
    resources.assets =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::assetInstances
        );
    resources.sensors =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::sensorInstances
        );
    resources.appearances =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::appearanceInstances
        );
    if (resources.instances == nil ||
        resources.assets == nil ||
        resources.sensors == nil ||
        resources.appearances == nil ||
        resources.instances.device != state.device ||
        resources.assets.device != state.device ||
        resources.sensors.device != state.device ||
        resources.appearances.device != state.device) {
        reason =
            "world-family Metal buffers are unavailable or on a "
            "different device";
        return false;
    }
    return true;
}

MetalHybridRendererDiagnostics encodeLocked(
    detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    HybridComputeEncoder& encoder,
    const EncodePassOptions options = {},
    const EncodeWorldResources* resolvedWorld = nullptr
) {
    MetalHybridRendererDiagnostics diagnostics;
    diagnostics.layout = state.layout;
    diagnostics.deviceName = state.deviceName;
    if (!state.compiled) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::notCompiled,
            "compile the visual sensor runtime before rendering"
        );
    }
    if (options.outputs == nullptr &&
        !state.config.retainObservationBuffers) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "this graph-only visual sensor retains no standalone "
            "observation planes"
        );
    }
    if (!encoder.valid()) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "visual sensor encoding requires a Metal compute encoder"
        );
    }
    const std::uint32_t environmentCount =
        liveState.environmentCount;
    EncodeWorldResources localWorld;
    if (resolvedWorld == nullptr) {
        std::string reason;
        if (!resolveEncodeWorldResources(
                state,
                worlds,
                environmentCount,
                cameraIndex,
                localWorld,
                reason
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::incompatibleWorldFamily,
                std::move(reason)
            );
        }
        resolvedWorld = &localWorld;
    }
    const MetalWorldFamilyLayout& worldLayout =
        resolvedWorld->layout;
    id<MTLBuffer> instances = resolvedWorld->instances;
    id<MTLBuffer> assets = resolvedWorld->assets;
    id<MTLBuffer> sensors = resolvedWorld->sensors;
    id<MTLBuffer> appearances = resolvedWorld->appearances;
    if (environmentCount == 0u ||
        environmentCount > worldLayout.activeInstanceCount) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            "the resolved world batch no longer covers the render"
        );
    }

    id<MTLBuffer> currentBodies =
        liveState.currentBodyStates == nullptr
        ? state.buffers.currentBodies
        : (__bridge id<MTLBuffer>)liveState.currentBodyStates;
    id<MTLBuffer> previousBodies =
        liveState.previousBodyStates == nullptr
        ? currentBodies
        : (__bridge id<MTLBuffer>)liveState.previousBodyStates;
    std::uint32_t liveFlags = 0u;
    if (liveState.currentBodyStates != nullptr) {
        liveFlags |= kLiveCurrent;
    }
    if (liveState.previousBodyStates != nullptr) {
        liveFlags |= kLivePrevious;
    }
    if (state.requiresLiveState &&
        (liveFlags & kLiveCurrent) == 0u) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "body/link-bound visual geometry requires live body state"
        );
    }
    if ((liveFlags & kLiveCurrent) != 0u &&
        liveState.bodyCount != state.layout.bodyCount) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "live visual body count does not match the compiled scene"
        );
    }
    std::size_t liveBodyCount = 0u;
    std::size_t liveBodyBytes = 0u;
    if (!checkedMultiply(
            environmentCount,
            state.layout.bodyCount,
            liveBodyCount
        ) ||
        !checkedBytes<MRBodyStateGPU>(
            liveBodyCount,
            liveBodyBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::capacityOverflow,
            "live visual body-state size overflows"
        );
    }
    if (!options.bodyBuffersValidated &&
        (currentBodies == nil || previousBodies == nil ||
         currentBodies.device != state.device ||
         previousBodies.device != state.device ||
         ((liveFlags & kLiveCurrent) != 0u &&
          (options.currentBodyOffset >
               currentBodies.length ||
           liveBodyBytes >
               currentBodies.length -
                   options.currentBodyOffset ||
           options.previousBodyOffset >
               previousBodies.length ||
           liveBodyBytes >
               previousBodies.length -
                   options.previousBodyOffset)))) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "live visual body buffers are unavailable, undersized, or "
            "on another device"
        );
    }

    MRHybridRenderUniformsGPU uniforms{};
    uniforms.counts = {
        environmentCount,
        state.layout.gaussianCount,
        worldLayout.assetCountPerInstance,
        worldLayout.sensorCountPerInstance,
    };
    uniforms.image = {
        state.layout.width,
        state.layout.height,
        state.layout.tileCountX,
        state.layout.tileCountY,
    };
    uniforms.render = {
        cameraIndex,
        state.layout.maximumGaussiansPerTile,
        MR_HYBRID_TILE_SIZE,
        MR_HYBRID_RENDERER_ABI_VERSION,
    };
    uniforms.live = {
        state.layout.bodyCount,
        liveFlags,
        state.layout.sensorBindingCount,
        state.layout.meshTriangleCount,
    };
    uniforms.timing = {
        static_cast<std::uint32_t>(liveState.frameIndex),
        static_cast<std::uint32_t>(
            liveState.frameIndex >> 32u
        ),
        liveState.sensorSequence,
        static_cast<std::uint32_t>(liveState.source),
    };
    uniforms.clearColorAndDepth =
        state.config.clearColorAndDepth;
    const MRVisualSensorBindingGPU profile =
        cameraIndex < state.sensorProfiles.size()
        ? state.sensorProfiles[cameraIndex]
        : MRVisualSensorBindingGPU{
              {},
              {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
              {0.05f, 10.0f, 0.001f, 0.0f},
              {
                  MR_VISUAL_SHUTTER_GLOBAL,
                  MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                  0u,
                  0u,
              },
          };
    uniforms.sensorTiming = profile.timing;
    uniforms.sensorRangeAndResponse =
        profile.rangeAndResponse;
    if (options.physicalExposure) {
        uniforms.sensorRangeAndResponse.w = 0.0f;
    }
    uniforms.presentation = {
        state.textureBindingCount,
        state.layout.lightCount,
        static_cast<std::uint32_t>(
            state.rendererProfile.kind
        ),
        state.nearClippedTriangleCapacity,
    };
    uniforms.meshTiling = {
        state.layout.maximumMeshTrianglesPerTile,
        state.layout.meshClusterCount,
        MR_HYBRID_MESH_MICRO_TRIANGLE_PIXELS,
        options.outputs == nullptr
            ? static_cast<std::uint32_t>(
                  MR_HYBRID_OUTPUT_ALL_TRUTH
              )
            : options.outputs->outputMask,
    };
    uniforms.shutter = {
        profile.shutter.x,
        profile.shutter.y,
        options.temporalSample,
        options.temporalSampleCount,
    };
    const float temporalFraction =
        options.temporalSampleCount <= 1u
        ? 0.5f
        : (
              static_cast<float>(options.temporalSample) + 0.5f
          ) /
              static_cast<float>(options.temporalSampleCount);
    uniforms.exposure = {
        temporalFraction,
        static_cast<float>(
            state.rendererProfile.rollingShutterBands
        ),
        state.environment.intensity,
        state.environment.rotationRadians,
    };
    uniforms.shadow = {
        state.rendererProfile.shadowMapResolution,
        state.rendererProfile.shadowMapResolution,
        state.rendererProfile.kind ==
                MR_VISUAL_RENDERER_SENSOR_REFERENCE
            ? 2u
            : 1u,
        state.shadowLightIndex,
    };
    uniforms.shadowBatch = {
        0u,
        environmentCount,
        state.layout.shadowLayerCapacity,
        0u,
    };
    uniforms.band = {
        options.bandFirst,
        options.bandCount == 0u
            ? (
                  options.bandAxis == 0u
                  ? state.layout.height
                  : state.layout.width
              )
            : options.bandCount,
        options.bandAxis,
        options.truthOnly ? 1u : 0u,
    };
    uniforms.ray = {
        state.layout.rayInstanceCount,
        options.motionKeyframes,
        state.rendererProfile.areaLightSamples,
        options.renderMeshes &&
                state.layout.meshTriangleCount != 0u
            ? state.layout.meshInstanceCount
            : 0u,
    };
    uniforms.rayBatch = {
        0u,
        environmentCount,
        0u,
        0u,
    };
    uniforms.rayTiming = {
        options.shutterWindowSeconds > 0.0f
            ? options.shutterWindowSeconds
            : profile.timing.y + profile.timing.z,
        0.5f,
        options.previousExposureFraction,
        options.interpolateMotion ? 1.0f : 0.0f,
    };
    if (options.exposureFraction >= 0.0f) {
        uniforms.exposure.x = options.exposureFraction;
    }
    if (options.encodedUniforms != nullptr) {
        *options.encodedUniforms = uniforms;
    }

    const NSUInteger bandPixelsPerEnvironment =
        static_cast<NSUInteger>(uniforms.band.y) *
        (
            uniforms.band.z == 0u
            ? state.layout.width
            : state.layout.height
        );
    const NSUInteger bandPixelCount =
        static_cast<NSUInteger>(environmentCount) *
        bandPixelsPerEnvironment;
    const std::uint32_t firstBandTile =
        uniforms.band.x / MR_HYBRID_TILE_SIZE;
    const std::uint32_t lastBandTile =
        (
            uniforms.band.x + uniforms.band.y - 1u
        ) / MR_HYBRID_TILE_SIZE;
    const std::uint32_t bandTilesOnAxis =
        lastBandTile - firstBandTile + 1u;
    const NSUInteger bandTilesPerEnvironment =
        uniforms.band.z == 0u
        ? static_cast<NSUInteger>(state.layout.tileCountX) *
            bandTilesOnAxis
        : static_cast<NSUInteger>(state.layout.tileCountY) *
            bandTilesOnAxis;
    const NSUInteger bandTileCount =
        static_cast<NSUInteger>(environmentCount) *
        bandTilesPerEnvironment;
    const NSUInteger projectedCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.gaussianCount;
    const NSUInteger triangleCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.meshTriangleCount;
    const NSUInteger meshClusterGroupCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.meshClusterCount;
    const bool sensorFusedIntoComposite =
        options.applySensor &&
        !options.resolveAccumulation &&
        options.renderMeshes &&
        triangleCount != 0u;
    const HybridDeviceObservationBuffers* graphOutputs =
        options.outputs;
    id<MTLBuffer> rgbOutput = graphOutputs == nullptr
        ? state.buffers.rgb
        : (__bridge id<MTLBuffer>)graphOutputs->rgb;
    id<MTLBuffer> depthOutput = graphOutputs == nullptr
        ? state.buffers.depth
        : (__bridge id<MTLBuffer>)graphOutputs->depth;
    id<MTLBuffer> segmentationOutput = graphOutputs == nullptr
        ? state.buffers.segmentation
        : (__bridge id<MTLBuffer>)graphOutputs->segmentation;
    id<MTLBuffer> identitiesOutput = graphOutputs == nullptr
        ? state.buffers.identities
        : (__bridge id<MTLBuffer>)graphOutputs->identities;
    id<MTLBuffer> normalsOutput = graphOutputs == nullptr
        ? state.buffers.normals
        : (__bridge id<MTLBuffer>)graphOutputs->normals;
    id<MTLBuffer> motionOutput = graphOutputs == nullptr
        ? state.buffers.motion
        : (__bridge id<MTLBuffer>)graphOutputs->motion;
    id<MTLBuffer> validityOutput = graphOutputs == nullptr
        ? state.buffers.validity
        : (__bridge id<MTLBuffer>)graphOutputs->validity;

    encoder.setLabel("MetalRobo visual sensor runtime");
    if (!options.resourcesResident) {
        encoder.useResources(
            state.visualResourceHeap,
            state.visualResourceResidencySet
        );
    }
    if (options.clearAccumulation) {
        encoder.setPipeline(state.clearAccumulationPipeline);
        encoder.setBuffer(
            state.buffers.temporalAccumulation,
            0u,
            0u
        );
        encoder.setBytes(&uniforms, sizeof(uniforms), 1u);
        constexpr NSUInteger accumulationThreads = kPixelThreads;
        encoder.dispatchThreads(
            bandPixelCount,
            accumulationThreads
        );
    }
    encoder.setPipeline(state.prepareCameraPipeline);
    id<MTLBuffer> __unsafe_unretained prepareBuffers[] = {
        instances,
        assets,
        sensors,
        state.buffers.sensorBindings,
        currentBodies,
        previousBodies,
        state.buffers.meshInstances,
        state.buffers.cameraStates,
        state.buffers.visualInstanceStates,
        state.buffers.nearClippedTriangleCounts,
    };
    const NSUInteger prepareOffsets[] = {
        0u,
        0u,
        0u,
        0u,
        options.currentBodyOffset,
        options.previousBodyOffset,
        0u,
        0u,
        0u,
        0u,
    };
    encoder.setBuffers(prepareBuffers, prepareOffsets, 0u);
    encoder.setBytes(&uniforms, sizeof(uniforms), 10u);
    constexpr NSUInteger cameraThreads = kCameraThreads;
    encoder.dispatchThreads(environmentCount, cameraThreads);

    if (projectedCount != 0u) {
        encoder.setPipeline(state.clearPipeline);
        encoder.setBuffer(state.buffers.tileCounts, 0u, 0u);
        encoder.setBuffer(
            state.buffers.tileOverflowCounts,
            0u,
            1u
        );
        encoder.setBytes(&uniforms, sizeof(uniforms), 2u);
        encoder.setBuffer(
            state.buffers.meshTileCounts,
            0u,
            3u
        );
        encoder.setBuffer(
            state.buffers.meshTileOverflowCounts,
            0u,
            4u
        );
        constexpr NSUInteger clearThreads = kPixelThreads;
        encoder.dispatchThreads(
            std::max<NSUInteger>(
                bandTileCount,
                environmentCount
            ),
            clearThreads
        );
    } else {
        encoder.setPipeline(state.clearObservationPipeline);
        id<MTLBuffer> __unsafe_unretained observationBuffers[] = {
            instances,
            sensors,
            appearances,
            state.buffers.cameraStates,
            state.buffers.resourceArgumentBuffer,
            state.buffers.environmentData,
            rgbOutput,
            depthOutput,
            segmentationOutput,
            identitiesOutput,
            normalsOutput,
            motionOutput,
            validityOutput,
        };
        const NSUInteger observationOffsets[
            std::size(observationBuffers)
        ] = {};
        encoder.setBuffers(
            observationBuffers,
            observationOffsets,
            0u
        );
        encoder.setBytes(&uniforms, sizeof(uniforms), 13u);
        id<MTLBuffer> __unsafe_unretained observationAuxiliary[] = {
            state.buffers.meshWinners,
            state.buffers.nearClippedDispatchArguments,
            state.buffers.meshTileCounts,
            state.buffers.meshTileOverflowCounts,
        };
        const NSUInteger observationAuxiliaryOffsets[] = {
            0u,
            0u,
            0u,
            0u,
        };
        encoder.setBuffers(
            observationAuxiliary,
            observationAuxiliaryOffsets,
            14u
        );
        constexpr NSUInteger clearObservationThreads =
            kPixelThreads;
        encoder.dispatchThreads(
            bandPixelCount,
            clearObservationThreads
        );
    }

    if (projectedCount != 0u) {
        encoder.setPipeline(state.binPipeline);
        id<MTLBuffer> __unsafe_unretained binBuffers[] = {
            state.buffers.gaussians,
            instances,
            assets,
            sensors,
            state.buffers.cameraStates,
            currentBodies,
            previousBodies,
            state.buffers.projected,
            state.buffers.tileCounts,
            state.buffers.tileIndices,
            state.buffers.tileOverflowCounts,
        };
        const NSUInteger binOffsets[] = {
            0u,
            0u,
            0u,
            0u,
            0u,
            options.currentBodyOffset,
            options.previousBodyOffset,
            0u,
            0u,
            0u,
            0u,
        };
        encoder.setBuffers(binBuffers, binOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 11u);
        constexpr NSUInteger binThreads = kPixelThreads;
        encoder.dispatchThreads(projectedCount, binThreads);
    }

    if (projectedCount != 0u) {
        encoder.setPipeline(state.renderPipeline);
        id<MTLBuffer> __unsafe_unretained renderBuffers[] = {
            state.buffers.projected,
            state.buffers.tileCounts,
            state.buffers.tileIndices,
            instances,
            sensors,
            appearances,
            state.buffers.cameraStates,
            state.buffers.resourceArgumentBuffer,
            state.buffers.environmentData,
            rgbOutput,
            depthOutput,
            segmentationOutput,
            identitiesOutput,
            normalsOutput,
            motionOutput,
            validityOutput,
        };
        const NSUInteger renderOffsets[
            std::size(renderBuffers)
        ] = {};
        encoder.setBuffers(renderBuffers, renderOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 16u);
        id<MTLBuffer> __unsafe_unretained renderAuxiliary[] = {
            state.buffers.meshWinners,
            state.buffers.nearClippedDispatchArguments,
        };
        const NSUInteger renderAuxiliaryOffsets[] = {0u, 0u};
        encoder.setBuffers(
            renderAuxiliary,
            renderAuxiliaryOffsets,
            17u
        );
        encoder.dispatchThreadgroups(
            bandTileCount,
            MR_HYBRID_MAX_GAUSSIANS_PER_TILE
        );
    }

    if (options.renderMeshes && triangleCount != 0u) {
        encoder.setPipeline(state.cullMeshClustersPipeline);
        id<MTLBuffer> __unsafe_unretained cullBuffers[] = {
            state.buffers.meshClusters,
            state.buffers.meshPrimitives,
            state.buffers.meshInstances,
            instances,
            sensors,
            state.buffers.cameraStates,
            state.buffers.visualInstanceStates,
            state.buffers.meshClusterVisibility,
        };
        const NSUInteger cullOffsets[
            std::size(cullBuffers)
        ] = {};
        encoder.setBuffers(cullBuffers, cullOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 8u);
        encoder.dispatchThreads(
            meshClusterGroupCount,
            kPixelThreads
        );

        encoder.setPipeline(state.binMeshPipeline);
        id<MTLBuffer> __unsafe_unretained rasterBuffers[] = {
            state.buffers.meshVertices,
            state.buffers.meshTriangles,
            state.buffers.meshPrimitives,
            state.buffers.meshInstances,
            instances,
            nil,
            sensors,
            state.buffers.cameraStates,
            nil,
            state.buffers.meshWinners,
            state.buffers.nearClippedTriangles,
            state.buffers.nearClippedTriangleCounts,
            state.buffers.nearClippedDispatchArguments,
            nil,
            state.buffers.visualInstanceStates,
            state.buffers.meshTileCounts,
            state.buffers.meshTileRecords,
            state.buffers.meshTileOverflowCounts,
            state.buffers.meshTriangleClusters,
            state.buffers.meshClusterVisibility,
        };
        const NSUInteger rasterOffsets[
            std::size(rasterBuffers)
        ] = {};
        encoder.setBuffers(rasterBuffers, rasterOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 13u);
        encoder.dispatchThreads(
            triangleCount,
            kPixelThreads
        );

        encoder.setPipeline(state.resolveMeshTilesPipeline);
        id<MTLBuffer> __unsafe_unretained resolveTileBuffers[] = {
            state.buffers.meshTileCounts,
            state.buffers.meshTileRecords,
            state.buffers.meshWinners,
        };
        const NSUInteger resolveTileOffsets[
            std::size(resolveTileBuffers)
        ] = {};
        encoder.setBuffers(
            resolveTileBuffers,
            resolveTileOffsets,
            0u
        );
        encoder.setBytes(&uniforms, sizeof(uniforms), 3u);
        encoder.dispatchThreadgroups(
            bandTileCount,
            MR_HYBRID_TILE_SIZE * MR_HYBRID_TILE_SIZE
        );

        encoder.setPipeline(state.resolveNearClippedPipeline);
        id<MTLBuffer> __unsafe_unretained clippedBuffers[] = {
            state.buffers.nearClippedTriangles,
            state.buffers.nearClippedTriangleCounts,
            instances,
            sensors,
            state.buffers.cameraStates,
            state.buffers.meshWinners,
        };
        const NSUInteger clippedOffsets[
            std::size(clippedBuffers)
        ] = {};
        encoder.setBuffers(clippedBuffers, clippedOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 6u);
        if (!encoder.dispatchThreadgroupsIndirect(
                state.buffers.nearClippedDispatchArguments,
                0u,
                MR_HYBRID_NEAR_CLIPPED_RESOLVE_THREADS
            )) {
            encoder.dispatchThreads(
                bandPixelCount,
                MR_HYBRID_NEAR_CLIPPED_RESOLVE_THREADS
            );
        }

        const bool shadowResourcesAvailable =
            !options.truthOnly &&
            state.shadowLightIndex != MR_INVALID_INDEX &&
            state.layout.shadowLayerCapacity != 0u;
        const std::uint32_t batchCapacity =
            shadowResourcesAvailable
            ? state.layout.shadowLayerCapacity
            : environmentCount;
        const bool updateShadowAtlas =
            shadowResourcesAvailable &&
            (
                options.updateShadows ||
                environmentCount >
                    state.layout.shadowLayerCapacity
            );
        constexpr NSUInteger clearShadowThreads = kPixelThreads;
        constexpr NSUInteger compositeThreads = kPixelThreads;
        for (std::uint32_t environmentStart = 0u;
             environmentStart < environmentCount;
             environmentStart += batchCapacity) {
            const std::uint32_t batchCount = std::min(
                batchCapacity,
                environmentCount - environmentStart
            );
            uniforms.shadowBatch = {
                environmentStart,
                batchCount,
                state.layout.shadowLayerCapacity,
                sensorFusedIntoComposite ? 1u : 0u,
            };
            if (updateShadowAtlas) {
                const NSUInteger shadowPixelCount =
                    static_cast<NSUInteger>(batchCount) *
                    uniforms.shadow.x * uniforms.shadow.y;
                encoder.setPipeline(state.clearShadowPipeline);
                encoder.setBuffer(
                    state.buffers.shadowAtlas,
                    0u,
                    0u
                );
                encoder.setBytes(&uniforms, sizeof(uniforms), 1u);
                encoder.dispatchThreads(
                    shadowPixelCount,
                    clearShadowThreads
                );

                const NSUInteger batchClusterCount =
                    static_cast<NSUInteger>(batchCount) *
                    state.layout.meshClusterCount;
                encoder.setPipeline(state.rasterShadowPipeline);
                id<MTLBuffer> __unsafe_unretained shadowBuffers[] = {
                    state.buffers.meshVertices,
                    state.buffers.meshTriangles,
                    state.buffers.meshPrimitives,
                    state.buffers.meshInstances,
                    nil,
                    nil,
                    nil,
                    state.buffers.lights,
                    state.buffers.shadowAtlas,
                };
                const NSUInteger shadowOffsets[
                    std::size(shadowBuffers)
                ] = {};
                encoder.setBuffers(
                    shadowBuffers,
                    shadowOffsets,
                    0u
                );
                encoder.setBytes(&uniforms, sizeof(uniforms), 9u);
                encoder.setBuffer(
                    state.buffers.visualInstanceStates,
                    0u,
                    10u
                );
                encoder.setBuffer(
                    state.buffers.meshClusters,
                    0u,
                    11u
                );
                encoder.dispatchThreadgroups(
                    batchClusterCount,
                    MR_HYBRID_MESH_CLUSTER_TRIANGLES
                );
            }

            encoder.setPipeline(state.compositeMeshPipeline);
            id<MTLBuffer> __unsafe_unretained compositeBuffers[] = {
                state.buffers.meshVertices,
                state.buffers.meshTriangles,
                state.buffers.meshPrimitives,
                state.buffers.meshInstances,
                state.buffers.materials,
                instances,
                nil,
                sensors,
                appearances,
                state.buffers.cameraStates,
                nil,
                nil,
                state.buffers.meshWinners,
                rgbOutput,
                depthOutput,
                segmentationOutput,
                identitiesOutput,
                normalsOutput,
                motionOutput,
                validityOutput,
                state.buffers.textureBindings,
                state.buffers.resourceArgumentBuffer,
                state.buffers.lights,
                state.buffers.environmentData,
                state.buffers.shadowAtlas,
            };
            const NSUInteger compositeOffsets[
                std::size(compositeBuffers)
            ] = {};
            encoder.setBuffers(
                compositeBuffers,
                compositeOffsets,
                0u
            );
            encoder.setBytes(&uniforms, sizeof(uniforms), 25u);
            encoder.setBuffer(
                state.buffers.visualInstanceStates,
                0u,
                26u
            );
            encoder.dispatchThreads(
                static_cast<NSUInteger>(batchCount) *
                    bandPixelsPerEnvironment,
                compositeThreads
            );
        }
    }

    if (options.accumulateRadiance) {
        encoder.setPipeline(state.accumulatePipeline);
        id<MTLBuffer> __unsafe_unretained accumulationBuffers[] = {
            rgbOutput,
            state.buffers.temporalAccumulation,
        };
        const NSUInteger accumulationOffsets[] = {0u, 0u};
        encoder.setBuffers(
            accumulationBuffers,
            accumulationOffsets,
            0u
        );
        encoder.setBytes(&uniforms, sizeof(uniforms), 2u);
        constexpr NSUInteger accumulateThreads = kPixelThreads;
        encoder.dispatchThreads(
            bandPixelCount,
            accumulateThreads
        );
    }
    if (options.resolveAccumulation) {
        encoder.setPipeline(state.resolveAccumulationPipeline);
        id<MTLBuffer> __unsafe_unretained resolveBuffers[] = {
            state.buffers.temporalAccumulation,
            instances,
            sensors,
            rgbOutput,
            depthOutput,
            validityOutput,
        };
        const NSUInteger resolveOffsets[
            std::size(resolveBuffers)
        ] = {};
        encoder.setBuffers(resolveBuffers, resolveOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 6u);
        constexpr NSUInteger resolveThreads = kPixelThreads;
        encoder.dispatchThreads(bandPixelCount, resolveThreads);
    }
    if (options.applySensor &&
        !options.resolveAccumulation &&
        !sensorFusedIntoComposite) {
        encoder.setPipeline(state.applySensorPipeline);
        id<MTLBuffer> __unsafe_unretained sensorBuffers[] = {
            instances,
            sensors,
            rgbOutput,
            depthOutput,
            validityOutput,
        };
        const NSUInteger sensorOffsets[
            std::size(sensorBuffers)
        ] = {};
        encoder.setBuffers(sensorBuffers, sensorOffsets, 0u);
        encoder.setBytes(&uniforms, sizeof(uniforms), 5u);
        constexpr NSUInteger sensorThreads = kPixelThreads;
        encoder.dispatchThreads(bandPixelCount, sensorThreads);
    }

    state.activeEnvironmentCount = environmentCount;
    state.activeMetadata.dimensions = {
        environmentCount,
        1u,
        state.layout.width,
        state.layout.height,
    };
    state.activeMetadata.identity = uniforms.timing;
    state.activeMetadata.timing = {
        static_cast<float>(
            liveState.captureTimestampSeconds
        ),
        static_cast<float>(liveState.frameAgeSeconds),
        profile.timing.y,
        profile.timing.z,
    };
    state.activeMetadata.contract = {
        MR_VISUAL_MODALITY_RGB |
            MR_VISUAL_MODALITY_DEPTH |
            MR_VISUAL_MODALITY_DEPTH_VALIDITY |
            MR_VISUAL_MODALITY_NORMAL |
            MR_VISUAL_MODALITY_MOTION |
            MR_VISUAL_MODALITY_SEMANTIC |
            MR_VISUAL_MODALITY_INSTANCE |
            MR_VISUAL_MODALITY_LINK,
        MR_VISUAL_FRAME_CAMERA,
        MR_VISUAL_PLATFORM_ABI_VERSION,
        0u,
    };
    return diagnostics;
}

MetalHybridRendererDiagnostics encodeReferenceFrameLocked(
    detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const VisualMotionSampleBatchV1& motion,
    const std::uint32_t cameraIndex,
    id<MTLCommandBuffer> command
) {
    MetalHybridRendererDiagnostics diagnostics;
    diagnostics.layout = state.layout;
    diagnostics.deviceName = state.deviceName;
    if (!state.compiled ||
        !state.rendererProfile.rayQueryVisibility ||
        state.primitiveAccelerationStructures == nil ||
        state.prepareRayInstancesPipeline == nil ||
        state.referenceRenderPipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::notCompiled,
            "sensor_reference was not compiled for ray-query rendering"
        );
    }
    EncodeWorldResources worldResources;
    std::string worldReason;
    if (!resolveEncodeWorldResources(
            state,
            worlds,
            motion.environmentCount,
            cameraIndex,
            worldResources,
            worldReason
        )) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            std::move(worldReason)
        );
    }
    id<MTLBuffer> instances = worldResources.instances;
    id<MTLBuffer> assets = worldResources.assets;
    id<MTLBuffer> sensors = worldResources.sensors;
    id<MTLBuffer> appearances = worldResources.appearances;

    const std::uint32_t keyframeCount =
        state.rendererProfile.temporalSamples;
    std::size_t statesPerKeyframe = 0u;
    std::size_t retainedStateCount = 0u;
    std::size_t retainedStateBytes = 0u;
    std::size_t descriptorCountValue = 0u;
    if (keyframeCount < 2u ||
        !checkedMultiply(
            motion.environmentCount,
            motion.bodyCount,
            statesPerKeyframe
        ) ||
        !checkedMultiply(
            statesPerKeyframe,
            static_cast<std::size_t>(keyframeCount) + 1u,
            retainedStateCount
        ) ||
        !checkedBytes<MRBodyStateGPU>(
            retainedStateCount,
            retainedStateBytes
        ) ||
        !checkedMultiply(
            motion.environmentCount,
            state.layout.rayInstanceCount,
            descriptorCountValue
        ) ||
        descriptorCountValue >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::capacityOverflow,
            "reference motion or instance count overflows"
        );
    }
    const std::uint32_t descriptorCount =
        static_cast<std::uint32_t>(descriptorCountValue);
    if (descriptorCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::invalidScene,
            "sensor_reference has no ray-visible or shadow-casting "
            "mesh instances"
        );
    }

    std::string reason;
    AcquiredReferenceWorkspace acquired =
        acquireReferenceWorkspace(
            state,
            retainedStateBytes,
            motion.environmentCount,
            state.layout.rayInstanceCount,
            keyframeCount,
            reason
        );
    if (acquired.workspace == nullptr ||
        acquired.descriptors == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            std::move(reason)
        );
    }
    const auto releaseWorkspace = [&acquired]() {
        acquired.workspace->inUse.store(
            false,
            std::memory_order_release
        );
    };

    std::span<MRBodyStateGPU> sampledStates{
        static_cast<MRBodyStateGPU*>(
            acquired.workspace->motionBodies.contents
        ),
        retainedStateCount,
    };
    const double shutterOpen = motion.exposureOpenSeconds;
    const double shutterClose = motion.exposureCloseSeconds;
    for (std::uint32_t keyframe = 0u;
         keyframe < keyframeCount;
         ++keyframe) {
        const double fraction =
            static_cast<double>(keyframe) /
            static_cast<double>(keyframeCount - 1u);
        const double timestamp =
            shutterOpen +
            fraction * (shutterClose - shutterOpen);
        if (!sampleMotionStates(
                motion,
                timestamp,
                {
                    sampledStates.data() +
                        statesPerKeyframe * keyframe,
                    statesPerKeyframe,
                }
            )) {
            releaseWorkspace();
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::missingLiveState,
                "could not interpolate reference motion keyframes"
            );
        }
    }
    const double truthTimestamp =
        0.5 * (shutterOpen + shutterClose);
    if (!sampleMotionStates(
            motion,
            truthTimestamp,
            {
                sampledStates.data() +
                    statesPerKeyframe * keyframeCount,
                statesPerKeyframe,
            }
        )) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "could not interpolate center-exposure reference truth"
        );
    }

    HybridDeviceStateBatch live;
    live.currentBodyStates =
        (__bridge void*)acquired.workspace->motionBodies;
    live.previousBodyStates =
        (__bridge void*)acquired.workspace->motionBodies;
    live.environmentCount = motion.environmentCount;
    live.bodyCount = motion.bodyCount;
    live.frameIndex = motion.frameIndex;
    live.sensorSequence = motion.sensorSequence;
    live.source = motion.source;
    live.captureTimestampSeconds = truthTimestamp;
    live.frameAgeSeconds = 0.0;

    id<MTLComputeCommandEncoder> baseEncoder =
        [command computeCommandEncoder];
    if (baseEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference Gaussian-base encoder"
        );
    }
    MRHybridRenderUniformsGPU uniforms{};
    EncodePassOptions baseOptions;
    baseOptions.currentBodyOffset =
        statesPerKeyframe * keyframeCount *
        sizeof(MRBodyStateGPU);
    baseOptions.previousBodyOffset =
        baseOptions.currentBodyOffset;
    baseOptions.physicalExposure = true;
    baseOptions.applySensor = false;
    baseOptions.renderMeshes = false;
    baseOptions.bodyBuffersValidated = true;
    baseOptions.encodedUniforms = &uniforms;
    HybridComputeEncoder baseCommandEncoder{baseEncoder};
    diagnostics = encodeLocked(
        state,
        worlds,
        live,
        cameraIndex,
        baseCommandEncoder,
        baseOptions,
        &worldResources
    );
    [baseEncoder endEncoding];
    if (!diagnostics.succeeded()) {
        releaseWorkspace();
        return diagnostics;
    }

    uniforms.ray = {
        state.layout.rayInstanceCount,
        keyframeCount,
        state.rendererProfile.areaLightSamples,
        0u,
    };
    uniforms.shutter = {
        cameraIndex < state.sensorProfiles.size()
            ? state.sensorProfiles[cameraIndex].shutter.x
            : MR_VISUAL_SHUTTER_GLOBAL,
        cameraIndex < state.sensorProfiles.size()
            ? state.sensorProfiles[cameraIndex].shutter.y
            : MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
        0u,
        state.rendererProfile.temporalSamples,
    };
    uniforms.rayTiming = {
        static_cast<float>(
            std::max(0.0, shutterClose - shutterOpen)
        ),
        0.5f,
        1.0f / static_cast<float>(
            state.rendererProfile.temporalSamples
        ),
        0.0f,
    };

    id<MTLComputeCommandEncoder> prepareEncoder =
        [command computeCommandEncoder];
    if (prepareEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the ray-instance preparation encoder"
        );
    }
    prepareEncoder.label =
        @"MetalRobo reference motion-instance preparation";
    [prepareEncoder
        setComputePipelineState:state.prepareRayInstancesPipeline];
    [prepareEncoder setBuffer:state.buffers.meshInstances
                       offset:0u
                      atIndex:0u];
    [prepareEncoder setBuffer:state.buffers.rayVisibleInstances
                       offset:0u
                      atIndex:1u];
    [prepareEncoder setBuffer:state.buffers.rayBlasIndices
                       offset:0u
                      atIndex:2u];
    [prepareEncoder setBuffer:instances offset:0u atIndex:3u];
    [prepareEncoder setBuffer:assets offset:0u atIndex:4u];
    [prepareEncoder setBuffer:acquired.workspace->motionBodies
                       offset:0u
                      atIndex:5u];
    [prepareEncoder
        setBuffer:acquired.workspace->instanceDescriptors
        offset:0u
        atIndex:6u];
    [prepareEncoder setBuffer:acquired.workspace->motionTransforms
                       offset:0u
                      atIndex:7u];
    [prepareEncoder setBytes:&uniforms
                      length:sizeof(uniforms)
                     atIndex:8u];
    const NSUInteger prepareThreads =
        std::min<NSUInteger>(
            state.prepareRayInstancesPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
    [prepareEncoder dispatchThreads:MTLSizeMake(
                                        descriptorCount,
                                        1u,
                                        1u
                                    )
        threadsPerThreadgroup:MTLSizeMake(
                                  prepareThreads,
                                  1u,
                                  1u
                              )];
    [prepareEncoder endEncoding];

    id<MTLAccelerationStructureCommandEncoder> buildEncoder =
        [command accelerationStructureCommandEncoder];
    if (buildEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference TLAS encoder"
        );
    }
    buildEncoder.label = @"MetalRobo grouped reference motion TLASes";
    // Refit recycles the existing TLAS for stable topology. Periodic rebuilds
    // restore traversal quality after large accumulated object motion.
    const bool refit =
        acquired.workspace->built &&
        acquired.workspace->refitCount < 32u;
    for (NSUInteger group = 0u;
         group < acquired.descriptors.count;
         ++group) {
        if (refit) {
            [buildEncoder
                refitAccelerationStructure:
                    acquired.workspace->instanceStructures[group]
                descriptor:acquired.descriptors[group]
                destination:nil
                scratchBuffer:acquired.workspace->buildScratch
                scratchBufferOffset:
                    acquired.workspace->buildScratchOffsets[group]];
        } else {
            [buildEncoder
                buildAccelerationStructure:
                    acquired.workspace->instanceStructures[group]
                descriptor:acquired.descriptors[group]
                scratchBuffer:acquired.workspace->buildScratch
                scratchBufferOffset:
                    acquired.workspace->buildScratchOffsets[group]];
        }
    }
    if (refit) {
        ++acquired.workspace->refitCount;
    } else {
        acquired.workspace->built = true;
        acquired.workspace->refitCount = 0u;
    }
    [buildEncoder endEncoding];

    id<MTLComputeCommandEncoder> referenceEncoder =
        [command computeCommandEncoder];
    if (referenceEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference ray-query encoder"
        );
    }
    referenceEncoder.label =
        @"MetalRobo exact-shutter reference renderer";
    [referenceEncoder
        setComputePipelineState:state.referenceRenderPipeline];
    [referenceEncoder setBuffer:state.buffers.meshVertices
                         offset:0u
                        atIndex:0u];
    [referenceEncoder setBuffer:state.buffers.meshIndices
                         offset:0u
                        atIndex:1u];
    [referenceEncoder setBuffer:state.buffers.meshPrimitives
                         offset:0u
                        atIndex:2u];
    [referenceEncoder setBuffer:state.buffers.meshInstances
                         offset:0u
                        atIndex:3u];
    [referenceEncoder setBuffer:state.buffers.rayVisibleInstances
                         offset:0u
                        atIndex:4u];
    [referenceEncoder setBuffer:state.buffers.materials
                         offset:0u
                        atIndex:5u];
    [referenceEncoder setBuffer:state.buffers.textureBindings
                         offset:0u
                        atIndex:6u];
    [referenceEncoder setBuffer:state.buffers.resourceArgumentBuffer
                         offset:0u
                        atIndex:7u];
    [referenceEncoder setBuffer:state.buffers.lights
                         offset:0u
                        atIndex:8u];
    [referenceEncoder setBuffer:state.buffers.environmentData
                         offset:0u
                        atIndex:9u];
    [referenceEncoder setBuffer:instances offset:0u atIndex:10u];
    [referenceEncoder setBuffer:assets offset:0u atIndex:11u];
    [referenceEncoder setBuffer:sensors offset:0u atIndex:12u];
    [referenceEncoder setBuffer:appearances offset:0u atIndex:13u];
    [referenceEncoder setBuffer:state.buffers.sensorBindings
                         offset:0u
                        atIndex:14u];
    [referenceEncoder setBuffer:acquired.workspace->motionBodies
                         offset:0u
                        atIndex:15u];
    [referenceEncoder setBuffer:state.buffers.rgb
                         offset:0u
                        atIndex:16u];
    [referenceEncoder setBuffer:state.buffers.depth
                         offset:0u
                        atIndex:17u];
    [referenceEncoder setBuffer:state.buffers.segmentation
                         offset:0u
                        atIndex:18u];
    [referenceEncoder setBuffer:state.buffers.identities
                         offset:0u
                        atIndex:19u];
    [referenceEncoder setBuffer:state.buffers.normals
                         offset:0u
                        atIndex:20u];
    [referenceEncoder setBuffer:state.buffers.motion
                         offset:0u
                        atIndex:21u];
    [referenceEncoder setBuffer:state.buffers.validity
                         offset:0u
                        atIndex:22u];
    [referenceEncoder useHeap:state.visualResourceHeap];
    for (id<MTLAccelerationStructure> primitive :
         state.primitiveAccelerationStructures) {
        [referenceEncoder
            useResource:primitive
            usage:MTLResourceUsageRead];
    }
    const NSUInteger pixelsPerEnvironment =
        static_cast<NSUInteger>(state.layout.width) *
        state.layout.height;
    const NSUInteger pixelCount =
        static_cast<NSUInteger>(motion.environmentCount) *
        pixelsPerEnvironment;
    const NSUInteger referenceThreads =
        std::min<NSUInteger>(
            state.referenceRenderPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
    for (NSUInteger group = 0u;
         group < acquired.workspace->instanceStructures.count;
         ++group) {
        const std::uint32_t firstEnvironment =
            static_cast<std::uint32_t>(group) *
            kReferenceEnvironmentsPerTLAS;
        const std::uint32_t environmentsInGroup = std::min(
            kReferenceEnvironmentsPerTLAS,
            motion.environmentCount - firstEnvironment
        );
        uniforms.rayBatch = {
            firstEnvironment,
            environmentsInGroup,
            0u,
            0u,
        };
        [referenceEncoder setBytes:&uniforms
                            length:sizeof(uniforms)
                           atIndex:23u];
        [referenceEncoder
            setAccelerationStructure:
                acquired.workspace->instanceStructures[group]
            atBufferIndex:24u];
        [referenceEncoder dispatchThreads:MTLSizeMake(
                                              static_cast<NSUInteger>(
                                                  environmentsInGroup
                                              ) *
                                                  pixelsPerEnvironment,
                                              1u,
                                              1u
                                          )
            threadsPerThreadgroup:MTLSizeMake(
                                      referenceThreads,
                                      1u,
                                      1u
                                  )];
    }

    [referenceEncoder
        setComputePipelineState:state.applySensorPipeline];
    [referenceEncoder setBuffer:instances offset:0u atIndex:0u];
    [referenceEncoder setBuffer:sensors offset:0u atIndex:1u];
    [referenceEncoder setBuffer:state.buffers.rgb
                         offset:0u
                        atIndex:2u];
    [referenceEncoder setBuffer:state.buffers.depth
                         offset:0u
                        atIndex:3u];
    [referenceEncoder setBuffer:state.buffers.validity
                         offset:0u
                        atIndex:4u];
    uniforms.rayBatch = {
        0u,
        motion.environmentCount,
        0u,
        0u,
    };
    [referenceEncoder setBytes:&uniforms
                        length:sizeof(uniforms)
                       atIndex:5u];
    const NSUInteger sensorThreads =
        std::min<NSUInteger>(
            state.applySensorPipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
    [referenceEncoder dispatchThreads:MTLSizeMake(
                                          pixelCount,
                                          1u,
                                          1u
                                      )
        threadsPerThreadgroup:MTLSizeMake(
                                  sensorThreads,
                                  1u,
                                  1u
                              )];
    [referenceEncoder endEncoding];

    const auto retainedWorkspace = acquired.workspace;
    [command addCompletedHandler:^(
        id<MTLCommandBuffer> completed
    ) {
        if (completed.status != MTLCommandBufferStatusCompleted) {
            retainedWorkspace->built = false;
            retainedWorkspace->refitCount = 0u;
        }
        retainedWorkspace->inUse.store(
            false,
            std::memory_order_release
        );
    }];
    return diagnostics;
}

} // namespace

MetalHybridRenderer::MetalHybridRenderer(
    MetalHybridRendererConfig config
)
    : state_(
          std::make_shared<detail::MetalHybridRendererState>(
              std::move(config)
          )
      ) {}

MetalHybridRenderer::~MetalHybridRenderer() = default;

MetalHybridRenderer::MetalHybridRenderer(
    MetalHybridRenderer&& other
) noexcept = default;

MetalHybridRenderer& MetalHybridRenderer::operator=(
    MetalHybridRenderer&& other
) noexcept = default;

MetalHybridRendererDiagnostics MetalHybridRenderer::compile(
    VisualRenderSceneV3&& scene,
    const VisualRendererProfileV1& profile,
    const std::uint32_t capacity
) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        std::string reason;
        if (!scene.valid(&reason) || !profile.valid(&reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        const std::uint32_t sceneAssetCount = scene.assetCount;
        const std::uint32_t sceneBodyCount = scene.bodyCount;
        VisualEnvironmentReferenceV2 sceneEnvironment = scene.environment;
        const std::uint64_t sceneFingerprint =
            scene.fingerprint != 0u
            ? scene.fingerprint
            : computeVisualRenderSceneV3Fingerprint(scene);
        RuntimeVisualScene runtime;
        if (!flattenScene(std::move(scene), runtime, reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        if (!profile.rayQueryVisibility &&
            !buildMeshClusters(runtime, reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        std::uint32_t shadowLightIndex = MR_INVALID_INDEX;
        std::uint32_t shadowLightPriority = 0u;
        for (std::uint32_t lightIndex = 0u;
             lightIndex < runtime.lights.size();
             ++lightIndex) {
            const MRVisualLightGPUV1& light =
                runtime.lights[lightIndex];
            if (light.shadow.x == 0u) {
                continue;
            }
            if (shadowLightIndex == MR_INVALID_INDEX ||
                light.identity.z > shadowLightPriority) {
                shadowLightIndex = lightIndex;
                shadowLightPriority = light.identity.z;
            }
        }
        std::vector<RayGeometryKey> rayGeometryKeys;
        std::vector<std::uint32_t> rayVisibleInstances;
        std::vector<std::uint32_t> rayBlasIndices;
        if (profile.rayQueryVisibility) {
            rayGeometryKeys.reserve(runtime.instances.size());
            rayVisibleInstances.reserve(runtime.instances.size());
            rayBlasIndices.reserve(runtime.instances.size());
            for (std::uint32_t instanceIndex = 0u;
                 instanceIndex < runtime.instances.size();
                 ++instanceIndex) {
                const MRVisualInstanceGPUV2& instance =
                    runtime.instances[instanceIndex];
                if ((instance.binding.w & (
                         MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR |
                         MR_VISUAL_INSTANCE_CASTS_SHADOW
                     )) == 0u ||
                    instance.geometry.y == 0u) {
                    continue;
                }
                const auto primitives =
                    std::span<const MRVisualPrimitiveGPUV2>{
                        runtime.primitives
                    }.subspan(
                        instance.geometry.x,
                        instance.geometry.y
                    );
                const auto found = std::ranges::find_if(
                    rayGeometryKeys,
                    [&](const RayGeometryKey& key) {
                        return matchesRayGeometry(
                            key,
                            primitives
                        );
                    }
                );
                std::uint32_t blasIndex = 0u;
                if (found == rayGeometryKeys.end()) {
                    blasIndex = static_cast<std::uint32_t>(
                        rayGeometryKeys.size()
                    );
                    RayGeometryKey key;
                    key.reserve(primitives.size());
                    for (const MRVisualPrimitiveGPUV2& primitive :
                         primitives) {
                        key.push_back({
                            primitive.geometry.x,
                            primitive.geometry.y,
                            primitive.geometry.z,
                        });
                    }
                    rayGeometryKeys.push_back(std::move(key));
                } else {
                    blasIndex = static_cast<std::uint32_t>(
                        found - rayGeometryKeys.begin()
                    );
                }
                rayVisibleInstances.push_back(instanceIndex);
                rayBlasIndices.push_back(blasIndex);
            }
        }
        if (capacity == 0u ||
            state_->config.width == 0u ||
            state_->config.height == 0u ||
            state_->config.maximumGaussiansPerTile == 0u ||
            state_->config.maximumGaussiansPerTile >
                MR_HYBRID_MAX_GAUSSIANS_PER_TILE ||
            state_->config.maximumMeshTrianglesPerTile == 0u ||
            state_->config.maximumMeshTrianglesPerTile >
                MR_HYBRID_MAX_MESH_TRIANGLES_PER_TILE ||
            state_->config.shadowLayerBatchSize == 0u ||
            state_->config.maximumReferenceFramesInFlight == 0u ||
            state_->config.maximumRetainedBytes == 0u ||
            state_->config.maximumShadowAtlasBytes == 0u ||
            (profile.rayQueryVisibility &&
             !state_->config.retainObservationBuffers) ||
            !finite4(state_->config.clearColorAndDepth) ||
            state_->config.clearColorAndDepth.w <= 0.0f) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidConfiguration,
                "visual sensor dimensions, tile capacity, clear "
                "values, memory budget, or world capacity are invalid"
            );
        }
        diagnostics = initialize(
            *state_,
            std::move(diagnostics)
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        if (profile.rayQueryVisibility &&
            (state_->prepareRayInstancesPipeline == nil ||
             state_->referenceRenderPipeline == nil)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalPipelineFailure,
                "sensor_reference ray-query pipelines are unavailable"
            );
        }

        MetalHybridRendererLayout layout;
        layout.capacity = capacity;
        layout.width = state_->config.width;
        layout.height = state_->config.height;
        layout.tileCountX =
            (layout.width + MR_HYBRID_TILE_SIZE - 1u) /
            MR_HYBRID_TILE_SIZE;
        layout.tileCountY =
            (layout.height + MR_HYBRID_TILE_SIZE - 1u) /
            MR_HYBRID_TILE_SIZE;
        const auto fitsUint32 = [](const std::size_t value) {
            return value <=
                std::numeric_limits<std::uint32_t>::max();
        };
        if (!fitsUint32(runtime.gaussians.size()) ||
            !fitsUint32(runtime.meshClusters.size()) ||
            !fitsUint32(runtime.primitives.size()) ||
            !fitsUint32(runtime.instances.size()) ||
            !fitsUint32(runtime.materials.size()) ||
            !fitsUint32(runtime.textureBindings.size()) ||
            !fitsUint32(runtime.textures.size()) ||
            !fitsUint32(runtime.lights.size()) ||
            !fitsUint32(runtime.sensors.size())) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual scene element count exceeds uint32"
            );
        }
        layout.gaussianCount =
            static_cast<std::uint32_t>(
                runtime.gaussians.size()
        );
        layout.meshVertexCount =
            runtime.vertexCount;
        layout.meshTriangleCount =
            runtime.indexCount / 3u;
        layout.meshClusterCount =
            static_cast<std::uint32_t>(
                runtime.meshClusters.size()
            );
        layout.meshPrimitiveCount =
            static_cast<std::uint32_t>(runtime.primitives.size());
        layout.meshInstanceCount =
            static_cast<std::uint32_t>(runtime.instances.size());
        layout.meshIndexCount =
            runtime.indexCount;
        layout.materialCount =
            static_cast<std::uint32_t>(
                runtime.materials.size()
            );
        layout.textureCount =
            static_cast<std::uint32_t>(
                runtime.textures.size()
            );
        layout.lightCount =
            static_cast<std::uint32_t>(
                runtime.lights.size()
            );
        layout.bodyCount = sceneBodyCount;
        layout.sensorBindingCount =
            static_cast<std::uint32_t>(
                runtime.sensors.size()
            );
        const std::uint32_t nearClippedTriangleCapacity =
            profile.rayQueryVisibility
            ? 0u
            : std::min(
                  layout.meshTriangleCount,
                  std::uint32_t(
                      MR_HYBRID_MAX_NEAR_CLIPPED_TRIANGLES
                  )
              );
        layout.rayInstanceCount =
            static_cast<std::uint32_t>(
                rayVisibleInstances.size()
            );
        layout.maximumGaussiansPerTile =
            state_->config.maximumGaussiansPerTile;
        layout.maximumMeshTrianglesPerTile = std::min(
            state_->config.maximumMeshTrianglesPerTile,
            std::max(layout.meshTriangleCount, 1u)
        );

        std::size_t pixelCount = 0u;
        std::size_t tileCount = 0u;
        std::size_t projectedCount = 0u;
        std::size_t tileIndexCount = 0u;
        std::size_t meshTileRecordCount = 0u;
        std::size_t meshClusterVisibilityCount = 0u;
        std::size_t bodyStateCount = 0u;
        std::size_t visualInstanceStateCount = 0u;
        std::size_t nearClippedTriangleCount = 0u;
        if (!checkedMultiply(
                layout.width,
                layout.height,
                pixelCount
            ) ||
            !checkedMultiply(pixelCount, capacity, pixelCount) ||
            !checkedMultiply(
                layout.tileCountX,
                layout.tileCountY,
                tileCount
            ) ||
            !checkedMultiply(tileCount, capacity, tileCount) ||
            !checkedMultiply(
                runtime.gaussians.size(),
                capacity,
                projectedCount
            ) ||
            !checkedMultiply(
                tileCount,
                layout.maximumGaussiansPerTile,
                tileIndexCount
            ) ||
            !checkedMultiply(
                tileCount,
                layout.maximumMeshTrianglesPerTile,
                meshTileRecordCount
            ) ||
            !checkedMultiply(
                capacity,
                layout.meshClusterCount,
                meshClusterVisibilityCount
            ) ||
            !checkedMultiply(
                capacity,
                layout.bodyCount,
                bodyStateCount
            ) ||
            !checkedMultiply(
                capacity,
                layout.meshInstanceCount,
                visualInstanceStateCount
            ) ||
            !checkedMultiply(
                capacity,
                nearClippedTriangleCapacity,
                nearClippedTriangleCount
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual sensor element counts overflow"
            );
        }

        std::size_t gaussianBytes = 0u;
        std::size_t meshVertexBytes = 0u;
        std::size_t meshIndexBytes = 0u;
        std::size_t meshTriangleBytes = 0u;
        std::size_t meshClusterBytes = 0u;
        std::size_t meshTriangleClusterBytes = 0u;
        std::size_t meshPrimitiveBytes = 0u;
        std::size_t meshInstanceBytes = 0u;
        std::size_t materialBytes = 0u;
        std::size_t textureBindingBytes = 0u;
        std::size_t lightBytes = 0u;
        std::size_t environmentBytes = 0u;
        std::size_t rayVisibleInstanceBytes = 0u;
        std::size_t rayBlasIndexBytes = 0u;
        std::size_t sensorBindingBytes = 0u;
        std::size_t bodyStateBytes = 0u;
        std::size_t cameraStateBytes = 0u;
        std::size_t visualInstanceStateBytes = 0u;
        std::size_t nearClippedTriangleBytes = 0u;
        std::size_t nearClippedCountBytes = 0u;
        const std::size_t nearClippedDispatchBytes =
            sizeof(MTLDispatchThreadgroupsIndirectArguments);
        std::size_t projectedBytes = 0u;
        std::size_t tileCountBytes = 0u;
        std::size_t tileIndexBytes = 0u;
        std::size_t tileOverflowBytes = 0u;
        std::size_t meshTileCountBytes = 0u;
        std::size_t meshTileRecordBytes = 0u;
        std::size_t meshTileOverflowBytes = 0u;
        std::size_t meshClusterVisibilityBytes = 0u;
        std::size_t meshWinnerBytes = 0u;
        std::size_t rgbBytes = 0u;
        std::size_t temporalAccumulationBytes = 0u;
        std::size_t depthBytes = 0u;
        std::size_t uintBytes = 0u;
        std::size_t float4Bytes = 0u;
        std::size_t uint4Bytes = 0u;
        std::size_t shadowPixelsPerLayer = 0u;
        std::size_t shadowBytesPerLayer = 0u;
        std::size_t shadowPixelCount = 0u;
        std::size_t shadowBytes = 0u;
        if (!checkedMultiply(
                profile.shadowMapResolution,
                profile.shadowMapResolution,
                shadowPixelsPerLayer
            ) ||
            !checkedBytes<std::uint32_t>(
                shadowPixelsPerLayer,
                shadowBytesPerLayer
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "one shadow layer exceeds Metal limits"
            );
        }
        const bool usesShadowAtlas =
            !profile.rayQueryVisibility &&
            runtime.indexCount != 0u &&
            shadowLightIndex != MR_INVALID_INDEX &&
            profile.shadowMapResolution != 0u;
        if (usesShadowAtlas) {
            const std::size_t budgetLayers =
                state_->config.maximumShadowAtlasBytes /
                shadowBytesPerLayer;
            if (budgetLayers == 0u) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "one shadow layer exceeds the configured shadow "
                    "workspace budget"
                );
            }
            layout.shadowLayerCapacity =
                static_cast<std::uint32_t>(std::min<std::size_t>({
                    capacity,
                    state_->config.shadowLayerBatchSize,
                    budgetLayers,
                }));
        }
        if (!checkedBytes<MRHybridGaussianGPU>(
                runtime.gaussians.size(),
                gaussianBytes
            ) ||
            !checkedBytes<MRVisualVertexGPUV2>(
                runtime.vertexCount,
                meshVertexBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.indexCount,
                meshIndexBytes
            ) ||
            !checkedBytes<MRVisualTriangleGPUV2>(
                runtime.indexCount / 3u,
                meshTriangleBytes
            ) ||
            !checkedBytes<MRHybridMeshClusterGPU>(
                runtime.meshClusters.size(),
                meshClusterBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.meshTriangleClusters.size(),
                meshTriangleClusterBytes
            ) ||
            !checkedBytes<MRVisualPrimitiveGPUV2>(
                runtime.primitives.size(),
                meshPrimitiveBytes
            ) ||
            !checkedBytes<MRVisualInstanceGPUV2>(
                runtime.instances.size(),
                meshInstanceBytes
            ) ||
            !checkedBytes<MRVisualMaterialGPUV2>(
                runtime.materials.size(),
                materialBytes
            ) ||
            !checkedBytes<MRVisualTextureBindingGPUV2>(
                runtime.textureBindings.size(),
                textureBindingBytes
            ) ||
            !checkedBytes<MRVisualLightGPUV1>(
                runtime.lights.size(),
                lightBytes
            ) ||
            !checkedBytes<MRVisualEnvironmentGPUV2>(
                1u,
                environmentBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                rayVisibleInstances.size(),
                rayVisibleInstanceBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                rayBlasIndices.size(),
                rayBlasIndexBytes
            ) ||
            !checkedBytes<MRVisualSensorBindingGPU>(
                runtime.sensors.size(),
                sensorBindingBytes
            ) ||
            !checkedBytes<MRBodyStateGPU>(
                bodyStateCount,
                bodyStateBytes
            ) ||
            !checkedBytes<MRHybridCameraStateGPU>(
                capacity,
                cameraStateBytes
            ) ||
            !checkedBytes<MRHybridVisualInstanceStateGPU>(
                visualInstanceStateCount,
                visualInstanceStateBytes
            ) ||
            !checkedBytes<MRHybridNearClippedTriangleGPU>(
                nearClippedTriangleCount,
                nearClippedTriangleBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                capacity,
                nearClippedCountBytes
            ) ||
            !checkedBytes<MRHybridProjectedGaussianGPU>(
                projectedCount,
                projectedBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.gaussians.empty() ? 0u : tileCount,
                tileCountBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.gaussians.empty() ? 0u : tileIndexCount,
                tileIndexBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.gaussians.empty() ? 0u : capacity,
                tileOverflowBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                profile.rayQueryVisibility ||
                        runtime.indexCount == 0u
                    ? 0u
                    : tileCount,
                meshTileCountBytes
            ) ||
            !checkedBytes<MRHybridMeshTileRecordGPU>(
                profile.rayQueryVisibility ||
                        runtime.indexCount == 0u
                    ? 0u
                    : meshTileRecordCount,
                meshTileRecordBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                profile.rayQueryVisibility ||
                        runtime.indexCount == 0u
                    ? 0u
                    : capacity,
                meshTileOverflowBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                profile.rayQueryVisibility
                    ? 0u
                    : meshClusterVisibilityCount,
                meshClusterVisibilityBytes
            ) ||
            !checkedMultiply(
                pixelCount,
                2u * sizeof(std::uint32_t),
                meshWinnerBytes
            ) ||
            meshWinnerBytes >
                std::numeric_limits<NSUInteger>::max() ||
            !checkedBytes<mr_float4>(
                pixelCount,
                rgbBytes
            ) ||
            !checkedMultiply(
                profile.rayQueryVisibility
                    ? 0u
                    : pixelCount,
                4u * sizeof(std::uint16_t),
                temporalAccumulationBytes
            ) ||
            temporalAccumulationBytes >
                std::numeric_limits<NSUInteger>::max() ||
            !checkedBytes<float>(
                pixelCount,
                depthBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                pixelCount,
                uintBytes
            ) ||
            !checkedBytes<mr_float4>(
                pixelCount,
                float4Bytes
            ) ||
            !checkedBytes<mr_uint4>(
                pixelCount,
                uint4Bytes
            ) ||
            !checkedMultiply(
                shadowPixelsPerLayer,
                layout.shadowLayerCapacity,
                shadowPixelCount
            ) ||
            !checkedBytes<std::uint32_t>(
                shadowPixelCount,
                shadowBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual sensor buffers exceed Metal limits"
            );
        }
        layout.shadowWorkspaceBytes = shadowBytes;

        std::size_t requestedRetention = 0u;
        for (const std::size_t bytes : {
                 gaussianBytes,
                 lightBytes,
                 environmentBytes,
                 rayVisibleInstanceBytes,
                 rayBlasIndexBytes,
                 sensorBindingBytes,
                 2u * bodyStateBytes,
                 cameraStateBytes,
                 visualInstanceStateBytes,
                 nearClippedTriangleBytes,
                 nearClippedCountBytes,
                 nearClippedDispatchBytes,
                 projectedBytes,
                 tileCountBytes,
                 tileIndexBytes,
                 tileOverflowBytes,
                 meshTileCountBytes,
                 meshTileRecordBytes,
                 meshTileOverflowBytes,
                 meshClusterVisibilityBytes,
                 meshWinnerBytes,
                 state_->config.retainObservationBuffers
                     ? rgbBytes
                     : 0u,
                 temporalAccumulationBytes,
                 state_->config.retainObservationBuffers
                     ? depthBytes
                     : 0u,
                 state_->config.retainObservationBuffers
                     ? 2u * uintBytes
                     : 0u,
                 state_->config.retainObservationBuffers
                     ? 2u * float4Bytes
                     : 0u,
                 state_->config.retainObservationBuffers
                     ? uint4Bytes
                     : 0u,
                 shadowBytes,
             }) {
            if (!checkedAdd(
                    requestedRetention,
                    bytes,
                    requestedRetention
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "visual sensor retained byte count overflows"
                );
            }
        }
        if (shadowBytes >
                state_->config.maximumShadowAtlasBytes ||
            requestedRetention >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual profile exceeds the configured unified-memory "
                "budget; lower shadow resolution or compile capacity"
            );
        }

        RendererBuffers buffers;
        buffers.gaussians = makePrivateBuffer(
            state_->device,
            gaussianBytes,
            @"MetalRobo visual Gaussians"
        );
        NativeGeometryResources geometryResources;
        if (!createNativeGeometryResources(
                state_->device,
                {
                    meshVertexBytes,
                    meshIndexBytes,
                    meshTriangleBytes,
                    meshClusterBytes,
                    meshTriangleClusterBytes,
                    meshPrimitiveBytes,
                    meshInstanceBytes,
                    materialBytes,
                    textureBindingBytes,
                },
                buffers,
                geometryResources,
                reason
            ) ||
            !checkedAdd(
                requestedRetention,
                geometryResources.retainedBytes,
                requestedRetention
            ) ||
            requestedRetention >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                reason.empty()
                    ? "streamed geometry exceeds retained memory"
                    : std::move(reason)
            );
        }
        buffers.lights = makePrivateBuffer(
            state_->device,
            std::max<std::size_t>(
                lightBytes,
                sizeof(MRVisualLightGPUV1)
            ),
            @"MetalRobo visual light rig"
        );
        buffers.environmentData = makePrivateBuffer(
            state_->device,
            environmentBytes,
            @"MetalRobo visual environment parameters"
        );
        buffers.rayVisibleInstances = makePrivateBuffer(
            state_->device,
            rayVisibleInstanceBytes,
            @"MetalRobo ray-visible instance indices"
        );
        buffers.rayBlasIndices = makePrivateBuffer(
            state_->device,
            rayBlasIndexBytes,
            @"MetalRobo ray BLAS indices"
        );
        buffers.sensorBindings = makePrivateBuffer(
            state_->device,
            std::max<std::size_t>(
                sensorBindingBytes,
                sizeof(MRVisualSensorBindingGPU)
            ),
            @"MetalRobo visual sensor bindings"
        );
        buffers.currentBodies = makeSharedBuffer(
            state_->device,
            std::max<std::size_t>(
                bodyStateBytes,
                sizeof(MRBodyStateGPU)
            ),
            @"MetalRobo visual current bodies"
        );
        buffers.previousBodies = makeSharedBuffer(
            state_->device,
            std::max<std::size_t>(
                bodyStateBytes,
                sizeof(MRBodyStateGPU)
            ),
            @"MetalRobo visual previous bodies"
        );
        buffers.cameraStates = makePrivateBuffer(
            state_->device,
            cameraStateBytes,
            @"MetalRobo visual camera states"
        );
        buffers.visualInstanceStates = makePrivateBuffer(
            state_->device,
            std::max<std::size_t>(
                visualInstanceStateBytes,
                sizeof(MRHybridVisualInstanceStateGPU)
            ),
            @"MetalRobo visual instance states"
        );
        buffers.nearClippedTriangles = makePrivateBuffer(
            state_->device,
            nearClippedTriangleBytes,
            @"MetalRobo near-clipped mesh records"
        );
        buffers.nearClippedTriangleCounts = makePrivateBuffer(
            state_->device,
            nearClippedCountBytes,
            @"MetalRobo near-clipped mesh counts"
        );
        buffers.nearClippedDispatchArguments = makePrivateBuffer(
            state_->device,
            nearClippedDispatchBytes,
            @"MetalRobo near-clipped indirect dispatch"
        );
        buffers.projected = makePrivateBuffer(
            state_->device,
            projectedBytes,
            @"MetalRobo projected Gaussians"
        );
        buffers.tileCounts = makePrivateBuffer(
            state_->device,
            tileCountBytes,
            @"MetalRobo Gaussian tile counts"
        );
        buffers.tileIndices = makePrivateBuffer(
            state_->device,
            tileIndexBytes,
            @"MetalRobo Gaussian tile indices"
        );
        buffers.tileOverflowCounts = makePrivateBuffer(
            state_->device,
            tileOverflowBytes,
            @"MetalRobo Gaussian tile overflows"
        );
        buffers.meshTileCounts = makePrivateBuffer(
            state_->device,
            meshTileCountBytes,
            @"MetalRobo mesh tile counts"
        );
        buffers.meshTileRecords = makePrivateBuffer(
            state_->device,
            meshTileRecordBytes,
            @"MetalRobo mesh tile records"
        );
        buffers.meshTileOverflowCounts = makePrivateBuffer(
            state_->device,
            meshTileOverflowBytes,
            @"MetalRobo mesh tile overflows"
        );
        buffers.meshClusterVisibility = makePrivateBuffer(
            state_->device,
            meshClusterVisibilityBytes,
            @"MetalRobo mesh cluster visibility"
        );
        buffers.meshWinners = makePrivateBuffer(
            state_->device,
            meshWinnerBytes,
            @"MetalRobo mesh pixel winners"
        );
        if (state_->config.retainObservationBuffers) {
            buffers.rgb = makePrivateBuffer(
                state_->device,
                rgbBytes,
                @"MetalRobo visual RGB"
            );
            buffers.depth = makePrivateBuffer(
                state_->device,
                depthBytes,
                @"MetalRobo visual depth"
            );
            buffers.segmentation = makePrivateBuffer(
                state_->device,
                uintBytes,
                @"MetalRobo visual semantics"
            );
            buffers.identities = makePrivateBuffer(
                state_->device,
                uint4Bytes,
                @"MetalRobo visual identities"
            );
            buffers.normals = makePrivateBuffer(
                state_->device,
                float4Bytes,
                @"MetalRobo visual normals"
            );
            buffers.motion = makePrivateBuffer(
                state_->device,
                float4Bytes,
                @"MetalRobo visual motion"
            );
            buffers.validity = makePrivateBuffer(
                state_->device,
                uintBytes,
                @"MetalRobo visual validity"
            );
        }
        buffers.shadowAtlas = makePrivateBuffer(
            state_->device,
            shadowBytes,
            @"MetalRobo environment-major shadow atlas"
        );
        buffers.temporalAccumulation = makePrivateBuffer(
            state_->device,
            temporalAccumulationBytes,
            @"MetalRobo temporal radiance accumulation"
        );
        NativeVisualResources nativeResources;
        if (!createNativeVisualResources(
                state_->device,
                state_->queue,
                state_->resourceArgumentEncoder,
                runtime,
                buffers.resourceArgumentBuffer,
                nativeResources,
                reason
            ) ||
            !checkedAdd(
                requestedRetention,
                nativeResources.retainedBytes,
                requestedRetention
            ) ||
            requestedRetention >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                reason.empty()
                    ? "native visual resources exceed retained memory"
                    : std::move(reason)
            );
        }
        MTLResidencySetDescriptor* residencyDescriptor =
            [[MTLResidencySetDescriptor alloc] init];
        residencyDescriptor.label =
            @"MetalRobo immutable visual-resource residency";
        residencyDescriptor.initialCapacity = 1u;
        NSError* residencyError = nil;
        id<MTLResidencySet> visualResourceResidencySet =
            [state_->device
                newResidencySetWithDescriptor:residencyDescriptor
                error:&residencyError];
        if (visualResourceResidencySet == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not create visual-resource residency set: " +
                    describeError(residencyError)
            );
        }
        [visualResourceResidencySet
            addAllocation:nativeResources.heap];
        [visualResourceResidencySet commit];
        const std::array allBuffers{
            buffers.gaussians,
            buffers.meshVertices,
            buffers.meshIndices,
            buffers.meshTriangles,
            buffers.meshClusters,
            buffers.meshTriangleClusters,
            buffers.meshPrimitives,
            buffers.meshInstances,
            buffers.materials,
            buffers.textureBindings,
            buffers.resourceArgumentBuffer,
            buffers.lights,
            buffers.environmentData,
            buffers.rayVisibleInstances,
            buffers.rayBlasIndices,
            buffers.sensorBindings,
            buffers.currentBodies,
            buffers.previousBodies,
            buffers.cameraStates,
            buffers.visualInstanceStates,
            buffers.nearClippedTriangles,
            buffers.nearClippedTriangleCounts,
            buffers.nearClippedDispatchArguments,
            buffers.projected,
            buffers.tileCounts,
            buffers.tileIndices,
            buffers.tileOverflowCounts,
            buffers.meshTileCounts,
            buffers.meshTileRecords,
            buffers.meshTileOverflowCounts,
            buffers.meshClusterVisibility,
            buffers.meshWinners,
            buffers.shadowAtlas,
            buffers.temporalAccumulation,
        };
        if (std::ranges::any_of(
                allBuffers,
                [](id<MTLBuffer> buffer) {
                    return buffer == nil;
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual sensor buffers"
            );
        }
        if (state_->config.retainObservationBuffers) {
            const std::array observationBuffers{
                buffers.rgb,
                buffers.depth,
                buffers.segmentation,
                buffers.identities,
                buffers.normals,
                buffers.motion,
                buffers.validity,
            };
            if (std::ranges::any_of(
                    observationBuffers,
                    [](id<MTLBuffer> buffer) {
                        return buffer == nil;
                    }
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::metalBufferFailure,
                    "could not allocate retained visual observation "
                    "buffers"
                );
            }
        }
        std::memset(
            buffers.currentBodies.contents,
            0,
            buffers.currentBodies.length
        );
        std::memset(
            buffers.previousBodies.contents,
            0,
            buffers.previousBodies.length
        );

        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual scene upload command"
            );
        }
        std::vector<id<MTLBuffer>> staging;
        if (!upload(
                state_->device,
                blit,
                runtime.gaussians.data(),
                gaussianBytes,
                buffers.gaussians,
                @"MetalRobo Gaussian upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.meshClusters.data(),
                meshClusterBytes,
                buffers.meshClusters,
                @"MetalRobo mesh cluster upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.meshTriangleClusters.data(),
                meshTriangleClusterBytes,
                buffers.meshTriangleClusters,
                @"MetalRobo triangle cluster-index upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.primitives.data(),
                meshPrimitiveBytes,
                buffers.meshPrimitives,
                @"MetalRobo mesh primitive upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.instances.data(),
                meshInstanceBytes,
                buffers.meshInstances,
                @"MetalRobo mesh instance upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.textureBindings.data(),
                textureBindingBytes,
                buffers.textureBindings,
                @"MetalRobo texture binding upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.lights.data(),
                lightBytes,
                buffers.lights,
                @"MetalRobo light rig upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                &runtime.environmentData,
                environmentBytes,
                buffers.environmentData,
                @"MetalRobo environment parameter upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                rayVisibleInstances.data(),
                rayVisibleInstanceBytes,
                buffers.rayVisibleInstances,
                @"MetalRobo ray-visible instance upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                rayBlasIndices.data(),
                rayBlasIndexBytes,
                buffers.rayBlasIndices,
                @"MetalRobo ray BLAS-index upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.sensors.data(),
                sensorBindingBytes,
                buffers.sensorBindings,
                @"MetalRobo sensor binding upload",
                staging
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual scene upload buffers"
            );
        }
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual scene upload failed: " +
                    describeError(command.error)
            );
        }
        // Upload staging is no longer referenced once the blit completes.
        // Release it before BLAS construction so unified-memory peaks at one
        // scene representation instead of source + staging + acceleration
        // structures.
        std::vector<id<MTLBuffer>>{}.swap(staging);
        if (!loadAndPrepareStreamedGeometry(
                state_->device,
                state_->queue,
                state_->rebaseIndicesPipeline,
                state_->rebaseMaterialBindingsPipeline,
                state_->expandTrianglesPipeline,
                state_->buildMeshClustersPipeline,
                runtime,
                buffers,
                reason
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                std::move(reason)
            );
        }
        const bool requiresLiveState =
            std::ranges::any_of(
                runtime.gaussians,
                [](const MRHybridGaussianGPU& gaussian) {
                    return gaussian.binding.w ==
                        MR_HYBRID_GAUSSIAN_BODY_LOCAL;
                }
            ) ||
            std::ranges::any_of(
                runtime.instances,
                [](const MRVisualInstanceGPUV2& instance) {
                    return instance.binding.z ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        instance.binding.z ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            ) ||
            std::ranges::any_of(
                runtime.sensors,
                [](const MRVisualSensorBindingGPU& binding) {
                    return binding.identity.x ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        binding.identity.x ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            );
        auto sensorProfiles = std::move(runtime.sensors);
        std::vector<MRHybridGaussianGPU>{}.swap(runtime.gaussians);
        std::vector<RuntimeVisualScene::GeometrySource>{}.swap(
            runtime.geometrySources
        );
        std::vector<MRHybridMeshClusterGPU>{}.swap(
            runtime.meshClusters
        );
        std::vector<std::uint32_t>{}.swap(
            runtime.meshTriangleClusters
        );
        std::vector<MRVisualPrimitiveGPUV2>{}.swap(runtime.primitives);
        std::vector<MRVisualInstanceGPUV2>{}.swap(runtime.instances);
        std::vector<MRVisualTextureBindingGPUV2>{}.swap(
            runtime.textureBindings
        );
        std::vector<VisualTextureImageV2>{}.swap(runtime.textures);
        std::vector<MRVisualLightGPUV1>{}.swap(runtime.lights);

        CompactedPrimitiveAccelerationStructures
            primitiveAccelerationStructures;
        const std::size_t accelerationStructureBudget =
            state_->config.maximumRetainedBytes -
            requestedRetention;
        if (!buildCompactedPrimitiveAccelerationStructures(
                state_->device,
                state_->queue,
                buffers.meshVertices,
                buffers.meshIndices,
                runtime.materials,
                rayGeometryKeys,
                accelerationStructureBudget,
                primitiveAccelerationStructures,
                reason
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                std::move(reason)
            );
        }
        std::vector<MRVisualMaterialGPUV2>{}.swap(runtime.materials);
        layout.accelerationStructureBytes =
            primitiveAccelerationStructures.retainedBytes;
        if (!checkedAdd(
                requestedRetention,
                layout.accelerationStructureBytes,
                layout.retainedPrivateBytes
            ) ||
            layout.retainedPrivateBytes >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual scene plus compacted acceleration structures "
                "exceed the configured retained-memory budget"
            );
        }
        state_->buffers = std::move(buffers);
        state_->geometryHeap = geometryResources.heap;
        state_->visualResourceHeap = nativeResources.heap;
        state_->visualResourceResidencySet =
            visualResourceResidencySet;
        state_->materialTextures =
            nativeResources.materialTextures;
        state_->materialSamplers = nativeResources.samplers;
        state_->diffuseIrradiance =
            nativeResources.diffuseIrradiance;
        state_->prefilteredSpecular =
            nativeResources.prefilteredSpecular;
        state_->brdfLut = nativeResources.brdfLut;
        state_->primitiveAccelerationStructures =
            primitiveAccelerationStructures.structures;
        state_->referenceWorkspaces.clear();
        state_->exposureWorkspaces.clear();
        state_->layout = layout;
        state_->assetCount = sceneAssetCount;
        state_->textureBindingCount =
            static_cast<std::uint32_t>(
                runtime.textureBindings.size()
            );
        state_->sensorProfiles = std::move(sensorProfiles);
        state_->rendererProfile = profile;
        state_->environment = std::move(sceneEnvironment);
        state_->renderSceneFingerprint = sceneFingerprint;
        state_->shadowLightIndex = shadowLightIndex;
        state_->nearClippedTriangleCapacity =
            nearClippedTriangleCapacity;
        state_->rayVisibleInstances =
            std::move(rayVisibleInstances);
        state_->rayBlasIndices = std::move(rayBlasIndices);
        state_->requiresLiveState = requiresLiveState;
        state_->activeEnvironmentCount = 0u;
        state_->activeMetadata = {};
        state_->compiled = true;
        diagnostics.layout = layout;
        diagnostics.deviceName = state_->deviceName;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed while compiling visual sensor runtime"
        );
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::render(
    const MetalWorldFamilyContext& worlds,
    const std::uint32_t environmentCount,
    const std::uint32_t cameraIndex
) {
    HybridDeviceStateBatch state;
    state.environmentCount = environmentCount;
    state.source = MR_VISUAL_SOURCE_SIMULATION;
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        if (command == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual sensor command"
            );
        }
        const auto start = std::chrono::steady_clock::now();
        if (state_->rendererProfile.rayQueryVisibility) {
            if (state_->requiresLiveState) {
                return reject(
                    {},
                    MetalHybridRendererStatus::missingLiveState,
                    "body/link-bound sensor_reference scenes require "
                    "encodeFrame motion samples"
                );
            }
            const MRVisualSensorBindingGPU sensor =
                cameraIndex < state_->sensorProfiles.size()
                ? state_->sensorProfiles[cameraIndex]
                : MRVisualSensorBindingGPU{
                      {},
                      {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
                      {0.05f, 10.0f, 0.001f, 0.0f},
                      {
                          MR_VISUAL_SHUTTER_GLOBAL,
                          MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                          0u,
                          0u,
                      },
                  };
            const double shutterWindow = std::max(
                1.0e-9,
                static_cast<double>(sensor.timing.y) +
                    (
                        sensor.shutter.x ==
                            MR_VISUAL_SHUTTER_ROLLING
                        ? static_cast<double>(sensor.timing.z)
                        : 0.0
                    )
            );
            VisualMotionSampleBatchV1 motion;
            motion.environmentCount = environmentCount;
            motion.bodyCount = state_->layout.bodyCount;
            motion.sampleCount = 2u;
            motion.exposureOpenSeconds = 0.0;
            motion.exposureCloseSeconds = shutterWindow;
            motion.timestampsSeconds = {0.0, shutterWindow};
            std::size_t motionBodyCount = 0u;
            if (motion.bodyCount == 0u ||
                !checkedMultiply(
                    static_cast<std::size_t>(environmentCount),
                    motion.bodyCount,
                    motionBodyCount
                ) ||
                !checkedMultiply(
                    motionBodyCount,
                    motion.sampleCount,
                    motionBodyCount
                )) {
                return reject(
                    {},
                    MetalHybridRendererStatus::capacityOverflow,
                    "static reference motion dimensions overflow"
                );
            }
            motion.bodyStates.resize(motionBodyCount);
            for (MRBodyStateGPU& body : motion.bodyStates) {
                body.orientation.w = 1.0f;
            }
            motion.scenarioIdentity = 1u;
            motion.sensorIdentity =
                static_cast<std::uint64_t>(cameraIndex) + 1u;
            motion.frameIndex = 1u;
            MetalHybridRendererDiagnostics diagnostics =
                encodeReferenceFrameLocked(
                    *state_,
                    worlds,
                    motion,
                    cameraIndex,
                    command
                );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            [command commit];
            [command waitUntilCompleted];
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start
                ).count();
            if (command.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::metalCommandFailure,
                    "reference visual sensor render failed: " +
                        describeError(command.error)
                );
            }
            return diagnostics;
        }
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual sensor command"
            );
        }
        HybridComputeEncoder commandEncoder{encoder};
        MetalHybridRendererDiagnostics diagnostics =
            encodeLocked(
                *state_,
                worlds,
                state,
                cameraIndex,
                commandEncoder
            );
        if (!diagnostics.succeeded()) {
            [encoder endEncoding];
            return diagnostics;
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual sensor render failed: " +
                    describeError(command.error)
            );
        }
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::renderLive(
    const MetalWorldFamilyContext& worlds,
    const HybridLiveStateBatch& liveState,
    const std::uint32_t cameraIndex
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        const std::size_t expected =
            static_cast<std::size_t>(
                liveState.environmentCount
            ) * liveState.bodyCount;
        if (liveState.environmentCount == 0u ||
            liveState.bodyCount != state_->layout.bodyCount ||
            liveState.currentBodies.size() != expected ||
            (!liveState.previousBodies.empty() &&
             liveState.previousBodies.size() != expected)) {
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                "host live-state dimensions do not match the visual scene"
            );
        }
        const std::size_t bytes =
            expected * sizeof(MRBodyStateGPU);
        if (bytes > state_->buffers.currentBodies.length ||
            bytes > state_->buffers.previousBodies.length) {
            return reject(
                {},
                MetalHybridRendererStatus::capacityOverflow,
                "host live-state batch exceeds compiled capacity"
            );
        }
        std::memcpy(
            state_->buffers.currentBodies.contents,
            liveState.currentBodies.data(),
            bytes
        );
        if (liveState.previousBodies.empty()) {
            std::memcpy(
                state_->buffers.previousBodies.contents,
                liveState.currentBodies.data(),
                bytes
            );
        } else {
            std::memcpy(
                state_->buffers.previousBodies.contents,
                liveState.previousBodies.data(),
                bytes
            );
        }
        HybridDeviceStateBatch deviceState;
        deviceState.currentBodyStates =
            (__bridge void*)state_->buffers.currentBodies;
        deviceState.previousBodyStates =
            liveState.previousBodies.empty()
            ? nullptr
            : (__bridge void*)state_->buffers.previousBodies;
        deviceState.environmentCount =
            liveState.environmentCount;
        deviceState.bodyCount = liveState.bodyCount;
        deviceState.frameIndex = liveState.frameIndex;
        deviceState.sensorSequence =
            liveState.sensorSequence;
        deviceState.source = liveState.source;
        deviceState.captureTimestampSeconds =
            liveState.captureTimestampSeconds;
        deviceState.frameAgeSeconds =
            liveState.frameAgeSeconds;

        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create live visual sensor command"
            );
        }
        const auto start = std::chrono::steady_clock::now();
        HybridComputeEncoder commandEncoder{encoder};
        MetalHybridRendererDiagnostics diagnostics =
            encodeLocked(
                *state_,
                worlds,
                deviceState,
                cameraIndex,
                commandEncoder
            );
        if (!diagnostics.succeeded()) {
            [encoder endEncoding];
            return diagnostics;
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "live visual sensor render failed: " +
                    describeError(command.error)
            );
        }
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::renderFrame(
    const MetalWorldFamilyContext& worlds,
    const VisualMotionSampleBatchV1& motion,
    const std::uint32_t cameraIndex
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    id<MTLCommandBuffer> command = [state_->queue commandBuffer];
    if (command == nil) {
        return reject(
            {},
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create presentation command buffer"
        );
    }
    MetalHybridFrameCommandContext context;
    context.commandBuffer = (__bridge void*)command;
    auto diagnostics = encodeFrame(
        worlds,
        motion,
        cameraIndex,
        context
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "presentation render failed: " + describeError(command.error)
        );
    }
    return diagnostics;
}

MetalHybridRendererDiagnostics MetalHybridRenderer::encode(
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    void* metalComputeCommandEncoder
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        HybridComputeEncoder encoder{
            (__bridge id<MTLComputeCommandEncoder>)
                metalComputeCommandEncoder
        };
        EncodePassOptions options;
        options.currentBodyOffset = liveState.currentBodyOffset;
        options.previousBodyOffset = liveState.previousBodyOffset;
        return encodeLocked(
            *state_,
            worlds,
            liveState,
            cameraIndex,
            encoder,
            options
        );
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::encodeGraph(
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    const MetalHybridComputeEncoderCallbacks& callbacks,
    const HybridDeviceObservationBuffers& outputs
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!callbacks.valid() ||
            state_->rendererProfile.rayQueryVisibility) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "graph encoding requires sensor_fast and a complete "
                "active-compute callback surface"
            );
        }
        if ((outputs.outputMask &
             ~static_cast<std::uint32_t>(
                 MR_HYBRID_OUTPUT_ALL_TRUTH
             )) != 0u) {
            return reject(
                {},
                MetalHybridRendererStatus::metalBufferFailure,
                "graph observation output selection is invalid"
            );
        }
        const std::array<void*, 7u> outputPointers{
            outputs.rgb,
            outputs.depth,
            outputs.segmentation,
            outputs.identities,
            outputs.normals,
            outputs.motion,
            outputs.validity,
        };
        if (outputs.rgb == nullptr ||
            outputs.depth == nullptr ||
            outputs.validity == nullptr ||
            (
                (outputs.outputMask &
                 MR_HYBRID_OUTPUT_SEGMENTATION) != 0u &&
                outputs.segmentation == nullptr
            ) ||
            (
                (outputs.outputMask &
                 MR_HYBRID_OUTPUT_IDENTITIES) != 0u &&
                outputs.identities == nullptr
            ) ||
            (
                (outputs.outputMask &
                 MR_HYBRID_OUTPUT_NORMALS) != 0u &&
                outputs.normals == nullptr
            ) ||
            (
                (outputs.outputMask &
                 MR_HYBRID_OUTPUT_MOTION) != 0u &&
                outputs.motion == nullptr
            )) {
            return reject(
                {},
                MetalHybridRendererStatus::metalBufferFailure,
                "graph observation outputs must all be supplied"
            );
        }
        std::size_t pixelCount = 0u;
        if (!checkedMultiply(
                liveState.environmentCount,
                state_->layout.width,
                pixelCount
            ) ||
            !checkedMultiply(
                pixelCount,
                state_->layout.height,
                pixelCount
            )) {
            return reject(
                {},
                MetalHybridRendererStatus::capacityOverflow,
                "graph observation dimensions overflow"
            );
        }
        const std::array<std::size_t, 7u> requiredBytes{
            pixelCount * sizeof(mr_float4),
            pixelCount * sizeof(float),
            pixelCount * sizeof(std::uint32_t),
            pixelCount * sizeof(mr_uint4),
            pixelCount * sizeof(mr_float4),
            pixelCount * sizeof(mr_float4),
            pixelCount * sizeof(std::uint32_t),
        };
        const std::array<bool, 7u> requiredOutputs{
            true,
            true,
            (outputs.outputMask &
             MR_HYBRID_OUTPUT_SEGMENTATION) != 0u,
            (outputs.outputMask &
             MR_HYBRID_OUTPUT_IDENTITIES) != 0u,
            (outputs.outputMask &
             MR_HYBRID_OUTPUT_NORMALS) != 0u,
            (outputs.outputMask &
             MR_HYBRID_OUTPUT_MOTION) != 0u,
            true,
        };
        for (std::size_t index = 0u;
             index < outputPointers.size();
             ++index) {
            if (!requiredOutputs[index]) {
                continue;
            }
            id<MTLBuffer> buffer =
                (__bridge id<MTLBuffer>)outputPointers[index];
            if (buffer == nil ||
                buffer.device != state_->device ||
                buffer.length < requiredBytes[index]) {
                return reject(
                    {},
                    MetalHybridRendererStatus::metalBufferFailure,
                    "graph observation output is undersized or on a "
                    "different Metal device"
                );
            }
        }

        HybridComputeEncoder encoder{callbacks};
        EncodePassOptions options;
        options.currentBodyOffset = liveState.currentBodyOffset;
        options.previousBodyOffset = liveState.previousBodyOffset;
        options.outputs = &outputs;
        return encodeLocked(
            *state_,
            worlds,
            liveState,
            cameraIndex,
            encoder,
            options
        );
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::encodeFrame(
    const MetalWorldFamilyContext& worlds,
    const VisualMotionSampleBatchV1& motion,
    const std::uint32_t cameraIndex,
    const MetalHybridFrameCommandContext& commandContext
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        std::string reason;
        if (!motion.valid(&reason) ||
            !state_->compiled ||
            motion.bodyCount != state_->layout.bodyCount ||
            motion.environmentCount > state_->layout.capacity) {
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                reason.empty()
                    ? "motion samples do not match the compiled scene"
                    : std::move(reason)
            );
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)
                commandContext.commandBuffer;
        if (command == nil ||
            command.commandQueue.device != state_->device) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "encodeFrame requires a caller-owned Metal command "
                "buffer on the renderer device"
            );
        }
        if (state_->rendererProfile.rayQueryVisibility) {
            if (commandContext.activeComputeCommandEncoder != nullptr) {
                return reject(
                    {},
                    MetalHybridRendererStatus::metalCommandFailure,
                    "sensor_reference requires a command buffer "
                    "without a caller-owned active encoder"
                );
            }
            return encodeReferenceFrameLocked(
                *state_,
                worlds,
                motion,
                cameraIndex,
                command
            );
        }
        id<MTLComputeCommandEncoder> encoder =
            (__bridge id<MTLComputeCommandEncoder>)
                commandContext.activeComputeCommandEncoder;
        const bool ownsEncoder = encoder == nil;
        if (ownsEncoder) {
            encoder = [command computeCommandEncoder];
        }
        if (encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "encodeFrame could not create a compute encoder"
            );
        }
        EncodeWorldResources worldResources;
        std::string worldReason;
        if (!resolveEncodeWorldResources(
                *state_,
                worlds,
                motion.environmentCount,
                cameraIndex,
                worldResources,
                worldReason
            )) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            return reject(
                {},
                MetalHybridRendererStatus::incompatibleWorldFamily,
                std::move(worldReason)
            );
        }
        [encoder useHeap:state_->visualResourceHeap];
        HybridComputeEncoder commandEncoder{encoder};

        const std::uint32_t temporalSamples =
            state_->rendererProfile.temporalSamples;
        const std::size_t statesPerSample =
            static_cast<std::size_t>(motion.environmentCount) *
            motion.bodyCount;
        std::size_t retainedStateCount = 0u;
        std::size_t retainedStateBytes = 0u;
        if (!checkedMultiply(
                statesPerSample,
                temporalSamples,
                retainedStateCount
            ) ||
            !checkedBytes<MRBodyStateGPU>(
                retainedStateCount,
                retainedStateBytes
            ) ||
            retainedStateBytes >
                state_->config.maximumRetainedBytes / 2u) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            return reject(
                {},
                MetalHybridRendererStatus::capacityOverflow,
                "physical exposure samples exceed the unified-memory "
                "budget"
            );
        }
        std::string workspaceReason;
        const auto exposureWorkspace = acquireExposureWorkspace(
            *state_,
            retainedStateBytes,
            workspaceReason
        );
        if (exposureWorkspace == nullptr) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            return reject(
                {},
                MetalHybridRendererStatus::metalBufferFailure,
                std::move(workspaceReason)
            );
        }
        const auto releaseExposureWorkspace =
            [&exposureWorkspace]() {
                exposureWorkspace->inUse.store(
                    false,
                    std::memory_order_release
                );
            };
        id<MTLBuffer> motionBuffer =
            exposureWorkspace->motionBodies;
        std::span<MRBodyStateGPU> sampledStates{
            static_cast<MRBodyStateGPU*>(motionBuffer.contents),
            retainedStateCount,
        };
        const MRVisualSensorBindingGPU sensorProfile =
            cameraIndex < state_->sensorProfiles.size()
            ? state_->sensorProfiles[cameraIndex]
            : MRVisualSensorBindingGPU{
                  {},
                  {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
                  {0.05f, 10.0f, 0.001f, 0.0f},
                  {
                      MR_VISUAL_SHUTTER_GLOBAL,
                      MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                      0u,
                      0u,
                  },
              };
        const double exposureDuration = std::max(
            0.0,
            static_cast<double>(sensorProfile.timing.y)
        );
        const bool rollingShutter =
            sensorProfile.shutter.x ==
                MR_VISUAL_SHUTTER_ROLLING &&
            sensorProfile.timing.z > 0.0f;
        const double readoutDuration = rollingShutter
            ? static_cast<double>(sensorProfile.timing.z)
            : 0.0;
        const double shutterWindow =
            exposureDuration + readoutDuration;
        const double availableWindow =
            motion.exposureCloseSeconds -
            motion.exposureOpenSeconds;
        if (!(shutterWindow > 0.0) ||
            availableWindow + 1.0e-9 < shutterWindow) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            releaseExposureWorkspace();
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                "motion history does not cover the complete shutter "
                "exposure and readout interval"
            );
        }
        for (std::uint32_t keyframe = 0u;
             keyframe < temporalSamples;
             ++keyframe) {
            const double fraction =
                temporalSamples <= 1u
                ? 0.5
                : static_cast<double>(keyframe) /
                    static_cast<double>(temporalSamples - 1u);
            const double timestamp =
                motion.exposureOpenSeconds +
                fraction * shutterWindow;
            if (!sampleMotionStates(
                    motion,
                    timestamp,
                    {
                        sampledStates.data() +
                            statesPerSample * keyframe,
                        statesPerSample,
                    }
                )) {
                if (ownsEncoder) {
                    [encoder endEncoding];
                }
                releaseExposureWorkspace();
                return reject(
                    {},
                    MetalHybridRendererStatus::missingLiveState,
                    "could not interpolate physical exposure state"
                );
            }
        }
        const double truthTimestamp =
            motion.exposureOpenSeconds +
            0.5 * shutterWindow;
        HybridDeviceStateBatch live;
        live.currentBodyStates = (__bridge void*)motionBuffer;
        live.previousBodyStates = (__bridge void*)motionBuffer;
        live.environmentCount = motion.environmentCount;
        live.bodyCount = motion.bodyCount;
        live.frameIndex = motion.frameIndex;
        live.sensorSequence = motion.sensorSequence;
        live.source = motion.source;
        live.captureTimestampSeconds = truthTimestamp;
        live.frameAgeSeconds = 0.0;

        const bool horizontalScan =
            sensorProfile.shutter.y ==
                MR_VISUAL_SHUTTER_LEFT_TO_RIGHT ||
            sensorProfile.shutter.y ==
                MR_VISUAL_SHUTTER_RIGHT_TO_LEFT;
        const std::uint32_t bandAxis =
            horizontalScan ? 1u : 0u;
        const std::uint32_t bandDimension =
            horizontalScan
            ? state_->layout.width
            : state_->layout.height;
        const std::uint32_t bandCount = rollingShutter
            ? std::max<std::uint32_t>(
                  1u,
                  std::min(
                      state_->rendererProfile.rollingShutterBands,
                      bandDimension
                  )
              )
            : 1u;

        MetalHybridRendererDiagnostics diagnostics;
        for (std::uint32_t band = 0u;
             band < bandCount;
             ++band) {
            const std::uint32_t first =
                static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(band) *
                    bandDimension / bandCount
                );
            const std::uint32_t end =
                static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(band + 1u) *
                    bandDimension / bandCount
                );
            const std::uint32_t pixelsInBand = end - first;
            float scanFraction =
                (
                    static_cast<float>(first) +
                    0.5f * static_cast<float>(pixelsInBand)
                ) /
                static_cast<float>(bandDimension);
            if (sensorProfile.shutter.y ==
                    MR_VISUAL_SHUTTER_BOTTOM_TO_TOP ||
                sensorProfile.shutter.y ==
                    MR_VISUAL_SHUTTER_RIGHT_TO_LEFT) {
                scanFraction = 1.0f - scanFraction;
            }

            float previousFraction = std::clamp(
                static_cast<float>(
                    (
                        scanFraction * readoutDuration +
                        0.5 * exposureDuration /
                            std::max<std::uint32_t>(
                                temporalSamples,
                                1u
                            )
                    ) / shutterWindow
                ),
                0.0f,
                1.0f
            );
            for (std::uint32_t sample = 0u;
                 sample < temporalSamples;
                 ++sample) {
                const double sampleFraction =
                    (
                        static_cast<double>(sample) + 0.5
                    ) /
                    temporalSamples;
                const float exposureFraction = std::clamp(
                    static_cast<float>(
                        (
                            scanFraction * readoutDuration +
                            sampleFraction * exposureDuration
                        ) / shutterWindow
                    ),
                    0.0f,
                    1.0f
                );
                EncodePassOptions options;
                options.temporalSample = sample;
                options.temporalSampleCount = temporalSamples;
                options.physicalExposure = true;
                options.clearAccumulation = sample == 0u;
                options.accumulateRadiance = true;
                options.applySensor = false;
                options.updateShadows = sample == 0u;
                options.resourcesResident = true;
                options.bodyBuffersValidated = true;
                options.bandFirst = first;
                options.bandCount = pixelsInBand;
                options.bandAxis = bandAxis;
                options.motionKeyframes = temporalSamples;
                options.exposureFraction = exposureFraction;
                options.previousExposureFraction =
                    sample == 0u
                    ? exposureFraction
                    : previousFraction;
                options.shutterWindowSeconds =
                    static_cast<float>(shutterWindow);
                options.interpolateMotion =
                    temporalSamples >= 2u;
                diagnostics = encodeLocked(
                    *state_,
                    worlds,
                    live,
                    cameraIndex,
                    commandEncoder,
                    options,
                    &worldResources
                );
                if (!diagnostics.succeeded()) {
                    if (ownsEncoder) {
                        [encoder endEncoding];
                    }
                    releaseExposureWorkspace();
                    return diagnostics;
                }
                previousFraction = exposureFraction;
            }

            // Geometry channels are evaluated at the band's row/column
            // exposure midpoint while radiance resolves from the two
            // temporal samples accumulated above.
            const float truthFraction = std::clamp(
                static_cast<float>(
                    (
                        scanFraction * readoutDuration +
                        0.5 * exposureDuration
                    ) / shutterWindow
                ),
                0.0f,
                1.0f
            );
            EncodePassOptions truthOptions;
            truthOptions.temporalSample = temporalSamples / 2u;
            truthOptions.temporalSampleCount = temporalSamples;
            truthOptions.physicalExposure = true;
            truthOptions.resolveAccumulation = true;
            truthOptions.applySensor = true;
            truthOptions.truthOnly = true;
            truthOptions.updateShadows = false;
            truthOptions.resourcesResident = true;
            truthOptions.bodyBuffersValidated = true;
            truthOptions.bandFirst = first;
            truthOptions.bandCount = pixelsInBand;
            truthOptions.bandAxis = bandAxis;
            truthOptions.motionKeyframes = temporalSamples;
            truthOptions.exposureFraction = truthFraction;
            truthOptions.previousExposureFraction =
                previousFraction;
            truthOptions.shutterWindowSeconds =
                static_cast<float>(shutterWindow);
            truthOptions.interpolateMotion =
                temporalSamples >= 2u;
            diagnostics = encodeLocked(
                *state_,
                worlds,
                live,
                cameraIndex,
                commandEncoder,
                truthOptions,
                &worldResources
            );
            if (!diagnostics.succeeded()) {
                if (ownsEncoder) {
                    [encoder endEncoding];
                }
                releaseExposureWorkspace();
                return diagnostics;
            }
        }
        if (ownsEncoder) {
            [encoder endEncoding];
        }
        if (!diagnostics.succeeded()) {
            releaseExposureWorkspace();
            return diagnostics;
        }
        [command addCompletedHandler:^(
            id<MTLCommandBuffer> completed
        ) {
            (void)completed;
            exposureWorkspace->inUse.store(
                false,
                std::memory_order_release
            );
        }];
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            {},
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed for physical exposure samples"
        );
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::readback(
    HybridObservationBatch& output
) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.layout = state_->layout;
        diagnostics.deviceName = state_->deviceName;
        if (!state_->config.retainObservationBuffers) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::notCompiled,
                "graph-only visual sensors retain no readback planes"
            );
        }
        if (!state_->compiled ||
            state_->activeEnvironmentCount == 0u) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::notCompiled,
                "render observations before reading them back"
            );
        }
        std::size_t pixelCount = 0u;
        if (!checkedMultiply(
                state_->layout.width,
                state_->layout.height,
                pixelCount
            ) ||
            !checkedMultiply(
                pixelCount,
                state_->activeEnvironmentCount,
                pixelCount
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual readback pixel count overflows"
            );
        }
        std::size_t rgbBytes = 0u;
        std::size_t depthBytes = 0u;
        std::size_t uintBytes = 0u;
        std::size_t uint4Bytes = 0u;
        if (!checkedBytes<mr_float4>(pixelCount, rgbBytes) ||
            !checkedBytes<float>(pixelCount, depthBytes) ||
            !checkedBytes<std::uint32_t>(pixelCount, uintBytes) ||
            !checkedBytes<mr_uint4>(pixelCount, uint4Bytes)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual readback byte count overflows"
            );
        }
        enum ReadbackPlane : std::size_t {
            rgbPlane,
            depthPlane,
            segmentationPlane,
            identityPlane,
            normalPlane,
            motionPlane,
            validityPlane,
            planeCount,
        };
        const std::array<std::size_t, planeCount> planeBytes{
            rgbBytes,
            depthBytes,
            uintBytes,
            uint4Bytes,
            rgbBytes,
            rgbBytes,
            uintBytes,
        };
        std::array<std::size_t, planeCount> planeOffsets{};
        std::size_t readbackBytes = 0u;
        for (std::size_t plane = 0u;
             plane < planeCount;
             ++plane) {
            if (!appendAlignedReadbackRegion(
                    planeBytes[plane],
                    readbackBytes,
                    planeOffsets[plane]
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "visual readback layout overflows"
                );
            }
        }
        output.rgb.reserve(pixelCount);
        output.depth.reserve(pixelCount);
        output.segmentation.reserve(pixelCount);
        output.identities.reserve(pixelCount);
        output.normals.reserve(pixelCount);
        output.motion.reserve(pixelCount);
        output.validity.reserve(pixelCount);
        id<MTLBuffer> readback = makeSharedBuffer(
            state_->device,
            readbackBytes,
            @"MetalRobo visual observation readback"
        );
        if (readback == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual observation readback"
            );
        }
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual observation readback command"
            );
        }
        const auto copy =
            [blit, readback](
                id<MTLBuffer> source,
                const std::size_t destinationOffset,
                const std::size_t bytes
            ) {
            [blit copyFromBuffer:source
                    sourceOffset:0u
                        toBuffer:readback
               destinationOffset:destinationOffset
                            size:bytes];
        };
        copy(
            state_->buffers.rgb,
            planeOffsets[rgbPlane],
            rgbBytes
        );
        copy(
            state_->buffers.depth,
            planeOffsets[depthPlane],
            depthBytes
        );
        copy(
            state_->buffers.segmentation,
            planeOffsets[segmentationPlane],
            uintBytes
        );
        copy(
            state_->buffers.identities,
            planeOffsets[identityPlane],
            uint4Bytes
        );
        copy(
            state_->buffers.normals,
            planeOffsets[normalPlane],
            rgbBytes
        );
        copy(
            state_->buffers.motion,
            planeOffsets[motionPlane],
            rgbBytes
        );
        copy(
            state_->buffers.validity,
            planeOffsets[validityPlane],
            uintBytes
        );
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual observation readback failed: " +
                    describeError(command.error)
            );
        }
        output.rgb.resize(pixelCount);
        output.depth.resize(pixelCount);
        output.segmentation.resize(pixelCount);
        output.identities.resize(pixelCount);
        output.normals.resize(pixelCount);
        output.motion.resize(pixelCount);
        output.validity.resize(pixelCount);
        const auto* contents =
            static_cast<const std::byte*>(readback.contents);
        std::memcpy(
            output.rgb.data(),
            contents + planeOffsets[rgbPlane],
            rgbBytes
        );
        std::memcpy(
            output.depth.data(),
            contents + planeOffsets[depthPlane],
            depthBytes
        );
        std::memcpy(
            output.segmentation.data(),
            contents + planeOffsets[segmentationPlane],
            uintBytes
        );
        std::memcpy(
            output.identities.data(),
            contents + planeOffsets[identityPlane],
            uint4Bytes
        );
        std::memcpy(
            output.normals.data(),
            contents + planeOffsets[normalPlane],
            rgbBytes
        );
        std::memcpy(
            output.motion.data(),
            contents + planeOffsets[motionPlane],
            rgbBytes
        );
        std::memcpy(
            output.validity.data(),
            contents + planeOffsets[validityPlane],
            uintBytes
        );
        output.environmentCount = state_->activeEnvironmentCount;
        output.width = state_->layout.width;
        output.height = state_->layout.height;
        output.metadata = state_->activeMetadata;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed during visual readback"
        );
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererLayout MetalHybridRenderer::layout() const noexcept {
    if (state_ == nullptr) {
        return {};
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->layout;
    } catch (...) {
        return {};
    }
}

void* MetalHybridRenderer::nativeBuffer(
    const MetalHybridRendererBuffer buffer
) const noexcept {
    if (state_ == nullptr) {
        return nullptr;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->compiled) {
            return nullptr;
        }
        id<MTLBuffer> selected = nil;
        switch (buffer) {
        case MetalHybridRendererBuffer::rgb:
            selected = state_->buffers.rgb;
            break;
        case MetalHybridRendererBuffer::depth:
            selected = state_->buffers.depth;
            break;
        case MetalHybridRendererBuffer::segmentation:
            selected = state_->buffers.segmentation;
            break;
        case MetalHybridRendererBuffer::projectedGaussians:
            selected = state_->buffers.projected;
            break;
        case MetalHybridRendererBuffer::tileOverflowCounts:
            selected = state_->buffers.tileOverflowCounts;
            break;
        case MetalHybridRendererBuffer::identities:
            selected = state_->buffers.identities;
            break;
        case MetalHybridRendererBuffer::normals:
            selected = state_->buffers.normals;
            break;
        case MetalHybridRendererBuffer::motion:
            selected = state_->buffers.motion;
            break;
        case MetalHybridRendererBuffer::validity:
            selected = state_->buffers.validity;
            break;
        case MetalHybridRendererBuffer::meshWinners:
            selected = state_->buffers.meshWinners;
            break;
        case MetalHybridRendererBuffer::shadowAtlas:
            selected = state_->buffers.shadowAtlas;
            break;
        case MetalHybridRendererBuffer::temporalAccumulation:
            selected = state_->buffers.temporalAccumulation;
            break;
        case MetalHybridRendererBuffer::meshTileOverflowCounts:
            selected = state_->buffers.meshTileOverflowCounts;
            break;
        case MetalHybridRendererBuffer::cameraStates:
            selected = state_->buffers.cameraStates;
            break;
    }
        return (__bridge void*)selected;
    } catch (...) {
        return nullptr;
    }
}

struct MetalHybridObjectTracker::State {
    MetalHybridRenderer* renderer = nullptr;
    const MetalWorldFamilyContext* worlds = nullptr;
    MetalHybridObjectTrackerConfig config;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    __strong id<MTLComputePipelineState> maskedDepthPipeline = nil;
    __strong id<MTLBuffer> bindings = nil;
    __strong id<MTLBuffer> history = nil;
    __strong id<MTLBuffer> maskedDepthInstances = nil;
    __strong id<MTLBuffer> maskedDepthHistory = nil;
    MetalHybridRendererLayout rendererLayout{};
    bool compiled = false;
};

MetalHybridObjectTracker::MetalHybridObjectTracker()
    : state_(std::make_unique<State>()) {}

MetalHybridObjectTracker::~MetalHybridObjectTracker() = default;

MetalHybridObjectTracker::MetalHybridObjectTracker(
    MetalHybridObjectTracker&&
) noexcept = default;

MetalHybridObjectTracker& MetalHybridObjectTracker::operator=(
    MetalHybridObjectTracker&&
) noexcept = default;

MetalHybridRendererDiagnostics MetalHybridObjectTracker::compile(
    MetalHybridRenderer& renderer,
    const MetalWorldFamilyContext& worlds,
    MetalHybridObjectTrackerConfig config
) {
    MetalHybridRendererDiagnostics diagnostics{};
    try {
        if (state_ == nullptr) {
            state_ = std::make_unique<State>();
        }
        const MetalHybridRendererLayout layout = renderer.layout();
        if (renderer.state_ == nullptr ||
            !renderer.state_->initialized || layout.capacity == 0u ||
            config.capacity == 0u || config.capacity > layout.capacity ||
            config.maximumActorHistoryLength == 0u ||
            config.cameraIndex >= layout.sensorBindingCount ||
            config.rootBodyIndex >= layout.bodyCount ||
            (config.bindings.empty() &&
             config.maskedDepthInstanceIds.empty()) ||
            !std::isfinite(config.timestepSeconds) ||
            !(config.timestepSeconds > 0.0f) ||
            !std::isfinite(
                config.maximumTrackSpeedMetersPerSecond
            ) ||
            !(config.maximumTrackSpeedMetersPerSecond > 0.0f)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidConfiguration,
                "object tracker configuration or compiled renderer is invalid"
            );
        }
        const bool maskedDepth =
            !config.maskedDepthInstanceIds.empty();
        const auto probability = [](const float value) {
            return std::isfinite(value) &&
                value >= 0.0f && value <= 1.0f;
        };
        if (maskedDepth &&
            (config.maskedDepthWidth != layout.width ||
             config.maskedDepthHeight != layout.height ||
             config.maskedDepthFrameOffsets.empty() ||
             config.maskedDepthFrameOffsets.size() > 4u ||
             config.maskedDepthFrameOffsets.front() != 0u ||
             !std::ranges::is_sorted(
                 config.maskedDepthFrameOffsets
             ) ||
             std::adjacent_find(
                 config.maskedDepthFrameOffsets.begin(),
                 config.maskedDepthFrameOffsets.end()
             ) != config.maskedDepthFrameOffsets.end() ||
             !std::isfinite(config.maskedDepthNearMeters) ||
             !std::isfinite(config.maskedDepthFarMeters) ||
             !(config.maskedDepthNearMeters > 0.0f) ||
             !(config.maskedDepthFarMeters >
               config.maskedDepthNearMeters) ||
             !probability(
                 config.maskedDepthFullDropoutProbability
             ) ||
             !probability(
                 config.maskedDepthPixelDropoutProbability
             ) ||
             !probability(
                 config.maskedDepthEdgeFlickerProbability
             ) ||
             !std::isfinite(config.maskedDepthJitterMeters) ||
             config.maskedDepthJitterMeters < 0.0f ||
             !std::isfinite(config.maskedDepthNoiseSigmaMeters) ||
             config.maskedDepthNoiseSigmaMeters < 0.0f)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidConfiguration,
                "masked-depth dimensions, sparse offsets, or range are invalid"
            );
        }
        std::vector<MRHybridObjectTrackBindingGPU> bindings;
        bindings.reserve(config.bindings.size());
        for (const MetalHybridObjectTrackBinding& binding :
             config.bindings) {
            if (binding.instanceId == MR_INVALID_INDEX ||
                binding.minimumVisiblePixels == 0u ||
                !std::isfinite(binding.positionScale) ||
                !std::isfinite(binding.velocityScale)) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::invalidConfiguration,
                    "object track binding is invalid"
                );
            }
            bindings.push_back({
                {
                    binding.instanceId,
                    binding.actorFrameOffset,
                    0u,
                    0u,
                },
                {
                    binding.positionScale,
                    binding.velocityScale,
                    static_cast<float>(binding.minimumVisiblePixels),
                    0.0f,
                },
            });
        }
        id<MTLDevice> device = renderer.state_->device;
        id<MTLFunction> function = [renderer.state_->library
            newFunctionWithName:@"mr_hybrid_reduce_object_tracks"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            function == nil
            ? nil
            : [device newComputePipelineStateWithFunction:function
                                                    error:&error];
        if (pipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalPipelineFailure,
                "could not create object-track reduction pipeline: " +
                    describeError(error)
            );
        }
        id<MTLFunction> maskedDepthFunction =
            [renderer.state_->library
                newFunctionWithName:@"mr_hybrid_masked_depth_history"];
        id<MTLComputePipelineState> maskedDepthPipeline =
            !maskedDepth
            ? nil
            : maskedDepthFunction == nil
                ? nil
                : [device
                    newComputePipelineStateWithFunction:maskedDepthFunction
                                                  error:&error];
        if (maskedDepth && maskedDepthPipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalPipelineFailure,
                "could not create masked-depth history pipeline: " +
                    describeError(error)
            );
        }
        id<MTLBuffer> bindingBuffer = [device
            newBufferWithLength:std::max<std::size_t>(
                bindings.size() * sizeof(MRHybridObjectTrackBindingGPU),
                kMinimumAllocationBytes
            ) options:MTLResourceStorageModeShared];
        if (!bindings.empty() && bindingBuffer != nil) {
            std::memcpy(
                bindingBuffer.contents,
                bindings.data(),
                bindings.size() * sizeof(bindings.front())
            );
        }
        const std::size_t historyRecords =
            static_cast<std::size_t>(config.capacity) *
            config.maximumActorHistoryLength * bindings.size() * 2u;
        id<MTLBuffer> history = [device
            newBufferWithLength:std::max<std::size_t>(
                                    historyRecords * sizeof(mr_float4),
                                    kMinimumAllocationBytes
                                )
                         options:MTLResourceStorageModePrivate];
        id<MTLBuffer> maskedDepthInstances = [device
            newBufferWithLength:std::max<std::size_t>(
                config.maskedDepthInstanceIds.size() * sizeof(std::uint32_t),
                kMinimumAllocationBytes
            ) options:MTLResourceStorageModeShared];
        if (maskedDepth && maskedDepthInstances != nil) {
            std::memcpy(
                maskedDepthInstances.contents,
                config.maskedDepthInstanceIds.data(),
                config.maskedDepthInstanceIds.size() * sizeof(std::uint32_t)
            );
        }
        const std::size_t maskedDepthValues = maskedDepth
            ? static_cast<std::size_t>(config.capacity) *
                config.maskedDepthWidth * config.maskedDepthHeight *
                (config.maskedDepthFrameOffsets.back() + 1u)
            : 0u;
        id<MTLBuffer> maskedDepthHistory = [device
            newBufferWithLength:std::max<std::size_t>(
                maskedDepthValues * sizeof(float),
                kMinimumAllocationBytes
            ) options:MTLResourceStorageModePrivate];
        if (bindingBuffer == nil || history == nil ||
            maskedDepthInstances == nil || maskedDepthHistory == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate object-track tables or history"
            );
        }
        id<MTLCommandBuffer> command =
            [renderer.state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> clear =
            [command blitCommandEncoder];
        if (command == nil || clear == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not clear object-track history"
            );
        }
        [clear fillBuffer:history
                    range:NSMakeRange(0u, history.length)
                    value:0u];
        [clear fillBuffer:maskedDepthHistory
                    range:NSMakeRange(0u, maskedDepthHistory.length)
                    value:0u];
        [clear endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "object-track history initialization failed"
            );
        }
        state_->renderer = &renderer;
        state_->worlds = &worlds;
        state_->config = std::move(config);
        state_->device = device;
        state_->pipeline = pipeline;
        state_->maskedDepthPipeline = maskedDepthPipeline;
        state_->bindings = bindingBuffer;
        state_->history = history;
        state_->maskedDepthInstances = maskedDepthInstances;
        state_->maskedDepthHistory = maskedDepthHistory;
        state_->rendererLayout = layout;
        state_->compiled = true;
        diagnostics.layout = layout;
        diagnostics.deviceName = renderer.state_->deviceName;
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    } catch (...) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "unknown object tracker compilation failure"
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridObjectTracker::reset() {
    MetalHybridRendererDiagnostics diagnostics{};
    try {
        if (state_ == nullptr || !state_->compiled ||
            state_->renderer == nullptr || state_->device == nil ||
            state_->history == nil ||
            state_->maskedDepthHistory == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::notCompiled,
                "object tracker is not compiled"
            );
        }
        id<MTLCommandQueue> queue =
            state_->renderer->state_->queue;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> clear =
            [command blitCommandEncoder];
        if (queue == nil || command == nil || clear == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not begin object-track reset"
            );
        }
        [clear fillBuffer:state_->history
                    range:NSMakeRange(0u, state_->history.length)
                    value:0u];
        [clear fillBuffer:state_->maskedDepthHistory
                    range:NSMakeRange(
                        0u,
                        state_->maskedDepthHistory.length
                    )
                    value:0u];
        [clear endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "object-track reset command failed"
            );
        }
        diagnostics.layout = state_->rendererLayout;
        diagnostics.deviceName =
            state_->renderer->state_->deviceName;
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    } catch (...) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "unknown object-track reset failure"
        );
    }
}

bool MetalHybridObjectTracker::encodeObservation(
    void* context,
    const MetalWorldDeviceObservationPass& pass
) {
    auto* state = static_cast<State*>(context);
    if (state == nullptr || !state->compiled ||
        state->renderer == nullptr || state->worlds == nullptr ||
        pass.commandBuffer == nullptr || pass.currentBodies == nullptr ||
        pass.resetMasks == nullptr || pass.actorHistory == nullptr ||
        pass.actorObservations == nullptr ||
        pass.environmentCount == 0u ||
        pass.environmentCount > state->config.capacity ||
        pass.actorHistoryLength == 0u ||
        pass.actorHistoryLength >
            state->config.maximumActorHistoryLength) {
        return false;
    }
    for (const MetalHybridObjectTrackBinding& binding :
         state->config.bindings) {
        if (binding.actorFrameOffset + 7u > pass.actorFrameSize) {
            return false;
        }
    }
    const bool maskedDepth =
        !state->config.maskedDepthInstanceIds.empty();
    const std::uint64_t maskedDepthValues =
        static_cast<std::uint64_t>(
            state->config.maskedDepthWidth
        ) * state->config.maskedDepthHeight *
        state->config.maskedDepthFrameOffsets.size();
    if (maskedDepth &&
        (state->config.maskedDepthActorFrameOffset +
             maskedDepthValues > pass.actorObservationSize ||
         pass.taskStates == nullptr ||
         state->maskedDepthPipeline == nil ||
         state->maskedDepthInstances == nil ||
         state->maskedDepthHistory == nil)) {
        return false;
    }
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLComputeCommandEncoder> renderEncoder =
        [command computeCommandEncoder];
    if (renderEncoder == nil) {
        return false;
    }
    HybridDeviceStateBatch liveState{
        .currentBodyStates = pass.currentBodies,
        .previousBodyStates = pass.currentBodies,
        .environmentCount = pass.environmentCount,
        .bodyCount = pass.bodyCount,
        .frameIndex = pass.controlStep,
        .sensorSequence = pass.controlStep,
        .source = MR_VISUAL_SOURCE_SIMULATION,
        .captureTimestampSeconds =
            double(pass.controlStep) * state->config.timestepSeconds,
        .frameAgeSeconds = 0.0,
    };
    const MetalHybridRendererDiagnostics rendered =
        state->renderer->encode(
            *state->worlds,
            liveState,
            state->config.cameraIndex,
            (__bridge void*)renderEncoder
        );
    [renderEncoder endEncoding];
    if (!rendered.succeeded()) {
        return false;
    }
    id<MTLBuffer> depth =
        (__bridge id<MTLBuffer>)state->renderer->nativeBuffer(
            MetalHybridRendererBuffer::depth
        );
    id<MTLBuffer> identities =
        (__bridge id<MTLBuffer>)state->renderer->nativeBuffer(
            MetalHybridRendererBuffer::identities
        );
    id<MTLBuffer> validity =
        (__bridge id<MTLBuffer>)state->renderer->nativeBuffer(
            MetalHybridRendererBuffer::validity
        );
    id<MTLBuffer> cameraStates =
        (__bridge id<MTLBuffer>)state->renderer->nativeBuffer(
            MetalHybridRendererBuffer::cameraStates
        );
    id<MTLBuffer> instanceHeaders =
        (__bridge id<MTLBuffer>)state->worlds->nativeBuffer(
            MetalWorldFamilyBuffer::instanceHeaders
        );
    id<MTLBuffer> sensorInstances =
        (__bridge id<MTLBuffer>)state->worlds->nativeBuffer(
            MetalWorldFamilyBuffer::sensorInstances
        );
    if (depth == nil || identities == nil || validity == nil ||
        cameraStates == nil || instanceHeaders == nil ||
        sensorInstances == nil) {
        return false;
    }
    const std::uint64_t outputOffset =
        pass.actorObservationOffsetElements;
    const MRHybridObjectTrackUniformsGPU uniforms{
        {
            pass.environmentCount,
            static_cast<std::uint32_t>(state->config.bindings.size()),
            state->rendererLayout.width,
            state->rendererLayout.height,
        },
        {
            pass.actorFrameSize,
            pass.actorHistoryLength,
            pass.controlStep,
            pass.environmentCount,
        },
        {
            pass.bodyCount,
            state->config.rootBodyIndex,
            state->config.cameraIndex,
            0u,
        },
        {
            static_cast<std::uint32_t>(outputOffset),
            static_cast<std::uint32_t>(outputOffset >> 32u),
            0u,
            0u,
        },
        {
            state->config.timestepSeconds,
            state->config.maximumTrackSpeedMetersPerSecond,
            0.0f,
            0.0f,
        },
    };
    if (!state->config.bindings.empty()) {
        id<MTLComputeCommandEncoder> reduce =
            [command computeCommandEncoder];
        if (reduce == nil) {
            return false;
        }
        reduce.label = @"MetalRobo RGB-D object track reduction";
        [reduce setComputePipelineState:state->pipeline];
        const std::array<id<MTLBuffer>, 12u> buffers{{
            depth,
            identities,
            validity,
            state->bindings,
            state->history,
            (__bridge id<MTLBuffer>)pass.resetMasks,
            (__bridge id<MTLBuffer>)pass.actorHistory,
            (__bridge id<MTLBuffer>)pass.actorObservations,
            (__bridge id<MTLBuffer>)pass.currentBodies,
            cameraStates,
            instanceHeaders,
            sensorInstances,
        }};
        for (NSUInteger index = 0u; index < buffers.size(); ++index) {
            [reduce setBuffer:buffers[index] offset:0u atIndex:index];
        }
        [reduce setBytes:&uniforms length:sizeof(uniforms) atIndex:12u];
        const NSUInteger threads =
            pass.environmentCount * state->config.bindings.size();
        const NSUInteger width = std::min<NSUInteger>(
            state->pipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
        [reduce dispatchThreadgroups:MTLSizeMake(threads, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        [reduce endEncoding];
    }
    if (maskedDepth) {
        const auto& offsets =
            state->config.maskedDepthFrameOffsets;
        const std::uint32_t ringCapacity = offsets.back() + 1u;
        const MRHybridMaskedDepthUniformsGPU depthUniforms{
            {
                pass.environmentCount,
                state->config.maskedDepthWidth,
                state->config.maskedDepthHeight,
                static_cast<std::uint32_t>(
                    state->config.maskedDepthInstanceIds.size()
                ),
            },
            {
                pass.actorObservationSize,
                1u,
                pass.controlStep,
                pass.environmentCount,
            },
            {
                static_cast<std::uint32_t>(outputOffset),
                static_cast<std::uint32_t>(outputOffset >> 32u),
                static_cast<std::uint32_t>(pass.seed),
                static_cast<std::uint32_t>(pass.seed >> 32u),
            },
            {
                ringCapacity,
                static_cast<std::uint32_t>(offsets.size()),
                state->config.maskedDepthActorFrameOffset,
                state->config.maskedDepthCurriculumLevelCount,
            },
            {
                offsets.size() > 0u ? offsets[0u] : 0u,
                offsets.size() > 1u ? offsets[1u] : 0u,
                offsets.size() > 2u ? offsets[2u] : 0u,
                offsets.size() > 3u ? offsets[3u] : 0u,
            },
            {
                state->config.maskedDepthNearMeters,
                state->config.maskedDepthFarMeters,
                1.0f /
                    (state->config.maskedDepthFarMeters -
                     state->config.maskedDepthNearMeters),
                state->config.maskedDepthEdgeFlickerProbability,
            },
            {
                state->config.maskedDepthFullDropoutProbability,
                state->config.maskedDepthPixelDropoutProbability,
                state->config.maskedDepthJitterMeters,
                state->config.maskedDepthNoiseSigmaMeters,
            },
            {
                state->config.maskedDepthCurriculumCorruptionGain,
                0.0f,
                0.0f,
                0.0f,
            },
        };
        id<MTLComputeCommandEncoder> mask =
            [command computeCommandEncoder];
        if (mask == nil) {
            return false;
        }
        mask.label = @"MetalRobo masked depth history";
        [mask setComputePipelineState:state->maskedDepthPipeline];
        const std::array<id<MTLBuffer>, 9u> buffers{{
            depth,
            identities,
            validity,
            state->maskedDepthInstances,
            state->maskedDepthHistory,
            (__bridge id<MTLBuffer>)pass.resetMasks,
            (__bridge id<MTLBuffer>)pass.actorHistory,
            (__bridge id<MTLBuffer>)pass.actorObservations,
            (__bridge id<MTLBuffer>)pass.taskStates,
        }};
        for (NSUInteger index = 0u; index < buffers.size(); ++index) {
            [mask setBuffer:buffers[index] offset:0u atIndex:index];
        }
        [mask setBytes:&depthUniforms
                length:sizeof(depthUniforms)
               atIndex:9u];
        const NSUInteger width = std::min<NSUInteger>(
            state->maskedDepthPipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
        [mask dispatchThreadgroups:
                MTLSizeMake(pass.environmentCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        [mask endEncoding];
    }
    return true;
}

MetalWorldDeviceObservationProgram
MetalHybridObjectTracker::observationProgram() noexcept {
    return state_ != nullptr && state_->compiled
        ? MetalWorldDeviceObservationProgram{
              .context = state_.get(),
              .encode = &MetalHybridObjectTracker::encodeObservation,
          }
        : MetalWorldDeviceObservationProgram{};
}

const char* metalHybridRendererStatusName(
    const MetalHybridRendererStatus status
) noexcept {
    switch (status) {
    case MetalHybridRendererStatus::success:
        return "success";
    case MetalHybridRendererStatus::invalidScene:
        return "invalid_scene";
    case MetalHybridRendererStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalHybridRendererStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalHybridRendererStatus::missingLiveState:
        return "missing_live_state";
    case MetalHybridRendererStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalHybridRendererStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalHybridRendererStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalHybridRendererStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalHybridRendererStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalHybridRendererStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalHybridRendererStatus::incompatibleWorldFamily:
        return "incompatible_world_family";
    case MetalHybridRendererStatus::notCompiled:
        return "not_compiled";
    case MetalHybridRendererStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
