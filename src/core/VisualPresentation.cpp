#include "metalrobo/VisualPresentation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kPackMagic{
    'M', 'R', 'V', 'P', 'A', 'C', 'K', '1',
};
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (character < 0x20u) {
                result += "\\u00";
                result.push_back(hex[character >> 4u]);
                result.push_back(hex[character & 0x0fu]);
            } else {
                result.push_back(static_cast<char>(character));
            }
            break;
        }
    }
    return result;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite4(const mr_float4& value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

bool unitQuaternion(const mr_float4& value) {
    if (!finite4(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    return std::abs(squared - 1.0) <= 1.0e-3;
}

class HashBuilder {
public:
    void append(const void* data, const std::size_t size) {
        const auto* bytes =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= kFnvPrime;
        }
    }

    template <typename Value>
    void scalar(const Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        append(&value, sizeof(value));
    }

    template <typename Value>
    void values(const std::span<const Value> values) {
        const std::uint64_t count = values.size();
        scalar(count);
        if (!values.empty()) {
            append(values.data(), values.size_bytes());
        }
    }

    void string(const std::string& value) {
        values<char>(value);
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFnvOffset;
};

class BinaryWriter {
public:
    template <typename Value>
    void scalar(const Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const auto* first =
            reinterpret_cast<const std::uint8_t*>(&value);
        bytes_.insert(bytes_.end(), first, first + sizeof(value));
    }

    template <typename Value>
    void values(const std::vector<Value>& values) {
        static_assert(std::is_trivially_copyable_v<Value>);
        scalar(static_cast<std::uint64_t>(values.size()));
        const auto* first =
            reinterpret_cast<const std::uint8_t*>(values.data());
        bytes_.insert(
            bytes_.end(),
            first,
            first + values.size() * sizeof(Value)
        );
    }

    void string(const std::string& value) {
        scalar(static_cast<std::uint64_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const {
        return bytes_;
    }

private:
    std::vector<std::uint8_t> bytes_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::span<const std::uint8_t> bytes)
        : bytes_(bytes) {}

    template <typename Value>
    bool scalar(Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        if (sizeof(Value) > bytes_.size() - offset_) {
            return false;
        }
        std::memcpy(&value, bytes_.data() + offset_, sizeof(Value));
        offset_ += sizeof(Value);
        return true;
    }

    template <typename Value>
    bool values(std::vector<Value>& values) {
        static_assert(std::is_trivially_copyable_v<Value>);
        std::uint64_t count = 0u;
        if (!scalar(count) ||
            count > std::numeric_limits<std::size_t>::max() /
                sizeof(Value)) {
            return false;
        }
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(Value);
        if (bytes > bytes_.size() - offset_) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        if (bytes != 0u) {
            std::memcpy(values.data(), bytes_.data() + offset_, bytes);
        }
        offset_ += bytes;
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t size = 0u;
        if (!scalar(size) ||
            size > bytes_.size() - offset_) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            static_cast<std::size_t>(size)
        );
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    [[nodiscard]] bool complete() const noexcept {
        return offset_ == bytes_.size();
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0u;
};

void writeTexture(BinaryWriter& writer, const VisualTextureImageV1& texture) {
    writer.string(texture.id);
    writer.string(texture.contentHash);
    writer.scalar(texture.width);
    writer.scalar(texture.height);
    writer.scalar(texture.flags);
    writer.values(texture.mipTexelOffsets);
    writer.values(texture.rgba8);
}

bool readTexture(BinaryReader& reader, VisualTextureImageV1& texture) {
    return reader.string(texture.id) &&
        reader.string(texture.contentHash) &&
        reader.scalar(texture.width) &&
        reader.scalar(texture.height) &&
        reader.scalar(texture.flags) &&
        reader.values(texture.mipTexelOffsets) &&
        reader.values(texture.rgba8);
}

void writeBinding(
    BinaryWriter& writer,
    const VisualSymbolicBindingV1& binding
) {
    writer.string(binding.node);
    writer.string(binding.link);
    writer.scalar(binding.instanceIndex);
    writer.scalar(binding.bodyIndex);
    writer.scalar(binding.binding);
}

bool readBinding(
    BinaryReader& reader,
    VisualSymbolicBindingV1& binding
) {
    return reader.string(binding.node) &&
        reader.string(binding.link) &&
        reader.scalar(binding.instanceIndex) &&
        reader.scalar(binding.bodyIndex) &&
        reader.scalar(binding.binding);
}

std::vector<std::uint8_t> serializePack(
    const VisualAssetPackV1& pack,
    const bool includeContentHash
) {
    BinaryWriter writer;
    writer.scalar(pack.schemaVersion);
    writer.string(pack.id);
    writer.string(pack.sourceUri);
    writer.string(pack.sourceContentHash);
    writer.string(includeContentHash ? pack.contentHash : std::string{});
    writer.string(pack.license);
    writer.string(pack.preprocessingProvenance);
    writer.string(pack.coordinateConvention);
    writer.values(pack.vertices);
    writer.values(pack.indices);
    writer.values(pack.primitives);
    writer.values(pack.instances);
    writer.values(pack.materials);
    writer.scalar(static_cast<std::uint64_t>(pack.textures.size()));
    for (const VisualTextureImageV1& texture : pack.textures) {
        writeTexture(writer, texture);
    }
    writer.scalar(
        static_cast<std::uint64_t>(pack.symbolicBindings.size())
    );
    for (const VisualSymbolicBindingV1& binding :
         pack.symbolicBindings) {
        writeBinding(writer, binding);
    }
    return writer.bytes();
}

std::uint64_t hashBytes(
    const std::span<const std::uint8_t> bytes
) {
    HashBuilder hash;
    hash.values<std::uint8_t>(bytes);
    return hash.finish();
}

std::string fnvContentHash(const VisualAssetPackV1& pack) {
    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::nouppercase
           << hashBytes(serializePack(pack, false));
    return stream.str();
}

bool validMaterial(const MRVisualMaterialGPUV2& material) {
    return finite4(material.baseColorAndOpacity) &&
        finite4(material.emissionAndStrength) &&
        finite4(material.surface) &&
        finite4(material.coatingAndAlphaCutoff) &&
        material.baseColorAndOpacity.w >= 0.0f &&
        material.baseColorAndOpacity.w <= 1.0f &&
        material.surface.x >= 0.0f &&
        material.surface.x <= 1.0f &&
        material.surface.y >= 0.0f &&
        material.surface.y <= 1.0f &&
        material.coatingAndAlphaCutoff.x >= 0.0f &&
        material.coatingAndAlphaCutoff.x <= 1.0f &&
        material.coatingAndAlphaCutoff.y >= 0.0f &&
        material.coatingAndAlphaCutoff.y <= 1.0f &&
        material.flags.x <= MR_VISUAL_ALPHA_BLEND;
}

MRVisualLightGPUV1 defaultLight() {
    MRVisualLightGPUV1 light{};
    light.positionAndRange = {0.5f, -0.5f, 2.5f, 20.0f};
    light.directionAndSpot = {
        -0.35f, 0.35f, -0.87f, -1.0f,
    };
    light.colorAndIntensity = {1.0f, 0.97f, 0.92f, 1200.0f};
    light.shape = {1.0f, 1.0f, -1.0f, 0.08f};
    light.identity = {
        MR_VISUAL_LIGHT_DIRECTIONAL,
        MR_VISUAL_LIGHT_UNIT_LUX,
        100u,
        1u,
    };
    light.shadow = {1u, 0u, 4u, 0u};
    return light;
}

std::uint32_t remapTexture(
    const std::uint32_t texture,
    const std::uint32_t offset
) {
    return texture == MR_INVALID_INDEX
        ? MR_INVALID_INDEX
        : texture + offset;
}

} // namespace

bool VisualTextureImageV1::valid(std::string* reason) const {
    if (id.empty() || contentHash.empty() ||
        width == 0u || height == 0u ||
        mipTexelOffsets.empty() ||
        mipTexelOffsets.front() != 0u ||
        rgba8.empty() || rgba8.size() % 4u != 0u ||
        (flags & ~(MR_VISUAL_TEXTURE_SRGB |
                   MR_VISUAL_TEXTURE_CLAMP_U |
                   MR_VISUAL_TEXTURE_CLAMP_V)) != 0u) {
        return fail(reason, "visual texture metadata is invalid");
    }
    std::uint64_t texelCount = 0u;
    std::uint32_t levelWidth = width;
    std::uint32_t levelHeight = height;
    for (std::size_t level = 0u;
         level < mipTexelOffsets.size();
         ++level) {
        if (mipTexelOffsets[level] != texelCount) {
            return fail(reason, "visual texture mip offsets are not packed");
        }
        texelCount +=
            static_cast<std::uint64_t>(levelWidth) * levelHeight;
        levelWidth = std::max(1u, levelWidth / 2u);
        levelHeight = std::max(1u, levelHeight / 2u);
    }
    if (texelCount * 4u != rgba8.size()) {
        return fail(reason, "visual texture mip payload size is invalid");
    }
    return true;
}

bool VisualSymbolicBindingV1::valid(
    const std::uint32_t instanceCount,
    std::string* reason
) const {
    const bool bodyBinding =
        binding == MR_VISUAL_BINDING_RIGID_BODY ||
        binding == MR_VISUAL_BINDING_ARTICULATED_LINK;
    if (node.empty() || instanceIndex >= instanceCount ||
        binding > MR_VISUAL_BINDING_ARTICULATED_LINK ||
        (bodyBinding && (link.empty() ||
                         bodyIndex == MR_INVALID_INDEX)) ||
        (!bodyBinding && bodyIndex != MR_INVALID_INDEX)) {
        return fail(reason, "symbolic visual binding is invalid");
    }
    return true;
}

bool VisualAssetPackV1::valid(std::string* reason) const {
    if (schemaVersion != kVisualAssetPackVersion ||
        id.empty() || sourceUri.empty() ||
        sourceContentHash.empty() || contentHash.empty() ||
        license.empty() || preprocessingProvenance.empty() ||
        coordinateConvention.empty() || vertices.empty() ||
        indices.empty() || primitives.empty() ||
        instances.empty() || materials.empty()) {
        return fail(reason, "visual asset pack metadata is incomplete");
    }
    for (const MRVisualVertexGPUV2& vertex : vertices) {
        if (!finite4(vertex.position) ||
            !finite4(vertex.normalAndTangentSign) ||
            !finite4(vertex.tangent) ||
            !finite4(vertex.texcoord01) ||
            !finite4(vertex.color) ||
            vertex.position.w != 1.0f ||
            std::abs(vertex.normalAndTangentSign.w) != 1.0f) {
            return fail(reason, "visual asset pack has an invalid vertex");
        }
    }
    if (std::ranges::any_of(
            indices,
            [this](const std::uint32_t index) {
                return index >= vertices.size();
            }
        )) {
        return fail(reason, "visual asset pack index is out of range");
    }
    if (!std::ranges::all_of(materials, validMaterial)) {
        return fail(reason, "visual asset pack has an invalid material");
    }
    for (const VisualTextureImageV1& texture : textures) {
        if (!texture.valid(reason)) {
            return false;
        }
    }
    const auto textureValid = [this](const std::uint32_t texture) {
        return texture == MR_INVALID_INDEX ||
            texture < textures.size();
    };
    for (const MRVisualMaterialGPUV2& material : materials) {
        const std::array textureIndices{
            material.textureIndices0.x,
            material.textureIndices0.y,
            material.textureIndices0.z,
            material.textureIndices0.w,
            material.textureIndices1.x,
            material.textureIndices1.y,
            material.textureIndices1.z,
        };
        if (!std::ranges::all_of(textureIndices, textureValid)) {
            return fail(reason, "visual material texture is out of range");
        }
    }
    for (std::size_t index = 0u; index < instances.size(); ++index) {
        const MRVisualInstanceGPUV2& instance = instances[index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(instance.geometry.x) +
            instance.geometry.y;
        if (!finite4(instance.translationAndScale) ||
            !unitQuaternion(instance.orientation) ||
            !(instance.translationAndScale.w > 0.0f) ||
            instance.binding.z >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            end > primitives.size()) {
            return fail(reason, "visual asset pack has an invalid instance");
        }
    }
    for (std::size_t index = 0u; index < primitives.size(); ++index) {
        const MRVisualPrimitiveGPUV2& primitive = primitives[index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(primitive.geometry.x) +
            primitive.geometry.y;
        if (primitive.geometry.y == 0u ||
            primitive.geometry.y % 3u != 0u ||
            end > indices.size() ||
            primitive.geometry.z >= materials.size() ||
            primitive.geometry.w >= instances.size() ||
            !finite4(primitive.boundsMinimum) ||
            !finite4(primitive.boundsMaximum) ||
            primitive.boundsMinimum.x > primitive.boundsMaximum.x ||
            primitive.boundsMinimum.y > primitive.boundsMaximum.y ||
            primitive.boundsMinimum.z > primitive.boundsMaximum.z) {
            return fail(reason, "visual asset pack has an invalid primitive");
        }
        const MRVisualInstanceGPUV2& instance =
            instances[primitive.geometry.w];
        if (index < instance.geometry.x ||
            index >= instance.geometry.x + instance.geometry.y) {
            return fail(
                reason,
                "visual primitive is outside its instance range"
            );
        }
    }
    for (const VisualSymbolicBindingV1& binding : symbolicBindings) {
        if (!binding.valid(
                static_cast<std::uint32_t>(instances.size()),
                reason
            )) {
            return false;
        }
    }
    if (contentHash != fnvContentHash(*this) &&
        !contentHash.starts_with("sha256:")) {
        return fail(reason, "visual asset pack content hash is invalid");
    }
    return true;
}

bool VisualEnvironmentV1::valid(
    const std::size_t textureCount,
    std::string* reason
) const {
    if (id.empty() || contentHash.empty() ||
        !finite(intensity) || intensity < 0.0f ||
        !finite(rotationRadians) ||
        (textureIndex != MR_INVALID_INDEX &&
         textureIndex >= textureCount) ||
        !std::ranges::all_of(diffuseSH, finite4)) {
        return fail(reason, "visual environment is invalid");
    }
    return true;
}

bool VisualLightRigV1::valid(std::string* reason) const {
    if (id.empty() || contentHash.empty() || lights.empty()) {
        return fail(reason, "visual light rig is incomplete");
    }
    std::set<std::uint32_t> identities;
    for (const MRVisualLightGPUV1& light : lights) {
        if (!finite4(light.positionAndRange) ||
            !finite4(light.directionAndSpot) ||
            !finite4(light.colorAndIntensity) ||
            !finite4(light.shape) ||
            light.identity.x > MR_VISUAL_LIGHT_RECTANGLE ||
            light.identity.y > MR_VISUAL_LIGHT_UNIT_NIT ||
            light.identity.w == 0u ||
            !identities.insert(light.identity.w).second ||
            light.positionAndRange.w < 0.0f ||
            light.colorAndIntensity.x < 0.0f ||
            light.colorAndIntensity.y < 0.0f ||
            light.colorAndIntensity.z < 0.0f ||
            light.colorAndIntensity.w < 0.0f) {
            return fail(reason, "visual light rig has an invalid light");
        }
    }
    return true;
}

bool VisualRenderSceneV2::valid(std::string* reason) const {
    if (id.empty() || assetCount == 0u ||
        assetCount >= MR_INVALID_INDEX ||
        (gaussians.empty() && primitives.empty())) {
        return fail(reason, "V2 visual render scene is incomplete");
    }
    VisualAssetPackV1 view;
    view.id = id;
    view.sourceUri = "runtime:" + id;
    view.sourceContentHash = "runtime";
    view.contentHash = "sha256:runtime";
    view.license = "runtime";
    view.preprocessingProvenance = "runtime";
    view.vertices = vertices;
    view.indices = indices;
    view.primitives = primitives;
    view.instances = instances;
    view.materials = materials;
    view.textures = textures;
    if (!primitives.empty() && !view.valid(reason)) {
        return false;
    }
    for (const MRVisualInstanceGPUV2& instance : instances) {
        const bool bodyBound =
            instance.binding.z == MR_VISUAL_BINDING_RIGID_BODY ||
            instance.binding.z ==
                MR_VISUAL_BINDING_ARTICULATED_LINK;
        if (instance.binding.x >= assetCount ||
            (bodyBound && instance.binding.y >= bodyCount)) {
            return fail(reason, "V2 visual instance binding is invalid");
        }
    }
    for (const MRHybridGaussianGPU& gaussian : gaussians) {
        if (!finite4(gaussian.meanAndOpacity) ||
            !finite4(gaussian.scaleAndImportance) ||
            !unitQuaternion(gaussian.orientation) ||
            !finite4(gaussian.colorAndEmission) ||
            gaussian.binding.x >= assetCount ||
            gaussian.binding.z == 0u ||
            gaussian.binding.z == MR_INVALID_INDEX ||
            gaussian.binding.w > MR_HYBRID_GAUSSIAN_WORLD ||
            (gaussian.binding.w ==
                 MR_HYBRID_GAUSSIAN_BODY_LOCAL &&
             gaussian.binding.y >= bodyCount)) {
            return fail(reason, "V2 visual scene has an invalid Gaussian");
        }
    }
    for (const MRVisualSensorBindingGPU& sensor : sensorBindings) {
        const bool bodyBound =
            sensor.identity.x == MR_VISUAL_BINDING_RIGID_BODY ||
            sensor.identity.x ==
                MR_VISUAL_BINDING_ARTICULATED_LINK;
        if (sensor.identity.x >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            (bodyBound && sensor.identity.y >= bodyCount) ||
            !finite4(sensor.timing) ||
            !finite4(sensor.rangeAndResponse) ||
            !(sensor.timing.x > 0.0f) ||
            !(sensor.rangeAndResponse.y >
              sensor.rangeAndResponse.x)) {
            return fail(reason, "V2 visual sensor binding is invalid");
        }
    }
    if (!environment.valid(textures.size(), reason) ||
        !lightRig.valid(reason)) {
        return false;
    }
    if (fingerprint != 0u &&
        fingerprint != computeVisualRenderSceneV2Fingerprint(*this)) {
        return fail(reason, "V2 visual scene fingerprint does not match");
    }
    return true;
}

bool VisualSceneManifestV2::valid(std::string* reason) const {
    const std::set<std::string> uniquePacks{
        visualPackHashes.begin(),
        visualPackHashes.end(),
    };
    if (schemaVersion != kVisualSceneManifestV2Version ||
        id.empty() ||
        coordinateConvention != "x-forward,y-left,z-up" ||
        worldFingerprint == 0u ||
        fingerprint == 0u ||
        visualPackHashes.empty() ||
        uniquePacks.size() != visualPackHashes.size() ||
        std::ranges::any_of(
            visualPackHashes,
            [](const std::string& value) {
                return value.empty();
            }
        ) ||
        environmentMapHash.empty() ||
        lightRigHash.empty() ||
        preprocessingProvenance.empty() ||
        !renderScene.valid(reason)) {
        return reason != nullptr && !reason->empty()
            ? false
            : fail(reason, "V2 visual scene manifest is incomplete");
    }
    if (renderScene.fingerprint == 0u ||
        environmentMapHash != renderScene.environment.contentHash ||
        lightRigHash != renderScene.lightRig.contentHash ||
        fingerprint != computeVisualSceneManifestV2Fingerprint(*this)) {
        return fail(
            reason,
            "V2 visual scene manifest provenance does not match its "
            "render scene"
        );
    }
    return true;
}

VisualRendererProfileV1 VisualRendererProfileV1::sensorFast() {
    VisualRendererProfileV1 result;
    result.id = "sensor_fast";
    result.kind = MR_VISUAL_RENDERER_SENSOR_FAST;
    result.temporalSamples = 2u;
    result.rollingShutterBands = 16u;
    result.shadowMapResolution = 128u;
    result.areaLightSamples = 4u;
    result.rayQueryVisibility = false;
    result.retainObservation = false;
    result.fingerprint =
        computeVisualRendererProfileFingerprint(result);
    return result;
}

VisualRendererProfileV1 VisualRendererProfileV1::sensorReference() {
    VisualRendererProfileV1 result;
    result.id = "sensor_reference";
    result.kind = MR_VISUAL_RENDERER_SENSOR_REFERENCE;
    result.temporalSamples = 8u;
    result.rollingShutterBands = 0u;
    result.shadowMapResolution = 512u;
    result.areaLightSamples = 8u;
    result.rayQueryVisibility = true;
    result.retainObservation = true;
    result.fingerprint =
        computeVisualRendererProfileFingerprint(result);
    return result;
}

bool VisualRendererProfileV1::valid(std::string* reason) const {
    if (id.empty() || kind > MR_VISUAL_RENDERER_SENSOR_REFERENCE ||
        temporalSamples == 0u || temporalSamples > 64u ||
        rollingShutterBands > 4096u ||
        shadowMapResolution == 0u ||
        shadowMapResolution > 8192u ||
        areaLightSamples == 0u || areaLightSamples > 64u ||
        (kind == MR_VISUAL_RENDERER_SENSOR_FAST &&
         rayQueryVisibility) ||
        (kind == MR_VISUAL_RENDERER_SENSOR_REFERENCE &&
         !rayQueryVisibility)) {
        return fail(reason, "visual renderer profile is invalid");
    }
    if (fingerprint != 0u &&
        fingerprint !=
            computeVisualRendererProfileFingerprint(*this)) {
        return fail(
            reason,
            "visual renderer profile fingerprint does not match"
        );
    }
    return true;
}

bool VisualSensorProfileV2::valid(std::string* reason) const {
    if (id.empty() || !finite(nominalRateHz) ||
        !(nominalRateHz > 0.0) ||
        !finite(exposureSeconds) || exposureSeconds < 0.0 ||
        !finite(shutterReadoutSeconds) ||
        shutterReadoutSeconds < 0.0 ||
        !finite(frameJitterSeconds) ||
        frameJitterSeconds < 0.0 ||
        !finite(minimumDepthMeters) ||
        minimumDepthMeters < 0.0 ||
        !finite(maximumDepthMeters) ||
        !(maximumDepthMeters > minimumDepthMeters) ||
        !finite(depthQuantumMeters) ||
        !(depthQuantumMeters > 0.0) ||
        !finite(latencySeconds) || latencySeconds < 0.0 ||
        shutterModel > MR_VISUAL_SHUTTER_ROLLING ||
        shutterDirection > MR_VISUAL_SHUTTER_RIGHT_TO_LEFT ||
        (shutterModel == MR_VISUAL_SHUTTER_GLOBAL &&
         shutterReadoutSeconds != 0.0)) {
        return fail(reason, "V2 visual sensor profile is invalid");
    }
    return true;
}

bool VisualMotionSampleBatchV1::valid(std::string* reason) const {
    std::size_t perSample = 0u;
    if (schemaVersion != kVisualMotionSampleBatchVersion ||
        environmentCount == 0u || bodyCount == 0u ||
        sampleCount < 2u ||
        !finite(exposureOpenSeconds) ||
        !finite(exposureCloseSeconds) ||
        exposureOpenSeconds > exposureCloseSeconds ||
        timestampsSeconds.size() != sampleCount ||
        environmentCount >
            std::numeric_limits<std::size_t>::max() / bodyCount) {
        return fail(reason, "visual motion sample metadata is invalid");
    }
    perSample =
        static_cast<std::size_t>(environmentCount) * bodyCount;
    if (sampleCount >
            std::numeric_limits<std::size_t>::max() / perSample ||
        bodyStates.size() != perSample * sampleCount ||
        timestampsSeconds.front() > exposureOpenSeconds ||
        timestampsSeconds.back() < exposureCloseSeconds ||
        !std::ranges::is_sorted(timestampsSeconds)) {
        return fail(reason, "visual motion sample coverage is invalid");
    }
    for (const double timestamp : timestampsSeconds) {
        if (!finite(timestamp)) {
            return fail(reason, "visual motion timestamp is invalid");
        }
    }
    for (const MRBodyStateGPU& body : bodyStates) {
        if (!finite4(body.position) ||
            !unitQuaternion(body.orientation) ||
            !finite4(body.linearVelocityAndInverseMass) ||
            !finite4(body.angularVelocity)) {
            return fail(reason, "visual motion body state is invalid");
        }
    }
    return true;
}

std::span<const MRBodyStateGPU> VisualMotionSampleBatchV1::sample(
    const std::uint32_t sampleIndex
) const noexcept {
    if (sampleIndex >= sampleCount) {
        return {};
    }
    const std::size_t count =
        static_cast<std::size_t>(environmentCount) * bodyCount;
    return {
        bodyStates.data() + count * sampleIndex,
        count,
    };
}

bool writeVisualAssetPack(
    const VisualAssetPackV1& pack,
    const std::filesystem::path& path,
    std::string* reason
) {
    if (!pack.valid(reason)) {
        return false;
    }
    const std::vector<std::uint8_t> payload =
        serializePack(pack, true);
    BinaryWriter file;
    for (const char value : kPackMagic) {
        file.scalar(value);
    }
    file.scalar(static_cast<std::uint64_t>(payload.size()));
    file.scalar(hashBytes(payload));
    for (const std::uint8_t value : payload) {
        file.scalar(value);
    }
    std::error_code error;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            return fail(reason, "could not create visual pack directory");
        }
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            return fail(reason, "could not open visual pack output");
        }
        const auto& bytes = file.bytes();
        stream.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        stream.flush();
        if (!stream) {
            return fail(reason, "could not write visual asset pack");
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        return fail(reason, "could not publish visual asset pack");
    }
    return true;
}

bool readVisualAssetPack(
    const std::filesystem::path& path,
    VisualAssetPackV1& output,
    std::string* reason
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail(reason, "could not open visual asset pack");
    }
    const std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>(),
    };
    const std::size_t headerBytes =
        kPackMagic.size() + 2u * sizeof(std::uint64_t);
    if (bytes.size() < headerBytes ||
        !std::equal(
            kPackMagic.begin(),
            kPackMagic.end(),
            bytes.begin()
        )) {
        return fail(reason, "visual asset pack header is invalid");
    }
    BinaryReader header{
        std::span<const std::uint8_t>{bytes}.subspan(kPackMagic.size())
    };
    std::uint64_t payloadSize = 0u;
    std::uint64_t checksum = 0u;
    if (!header.scalar(payloadSize) ||
        !header.scalar(checksum) ||
        payloadSize != bytes.size() - headerBytes) {
        return fail(reason, "visual asset pack size is invalid");
    }
    const std::span payload =
        std::span<const std::uint8_t>{bytes}.subspan(headerBytes);
    if (hashBytes(payload) != checksum) {
        return fail(reason, "visual asset pack checksum failed");
    }
    BinaryReader reader{payload};
    VisualAssetPackV1 candidate;
    std::uint64_t textureCount = 0u;
    std::uint64_t bindingCount = 0u;
    if (!reader.scalar(candidate.schemaVersion) ||
        !reader.string(candidate.id) ||
        !reader.string(candidate.sourceUri) ||
        !reader.string(candidate.sourceContentHash) ||
        !reader.string(candidate.contentHash) ||
        !reader.string(candidate.license) ||
        !reader.string(candidate.preprocessingProvenance) ||
        !reader.string(candidate.coordinateConvention) ||
        !reader.values(candidate.vertices) ||
        !reader.values(candidate.indices) ||
        !reader.values(candidate.primitives) ||
        !reader.values(candidate.instances) ||
        !reader.values(candidate.materials) ||
        !reader.scalar(textureCount) ||
        textureCount > std::numeric_limits<std::uint32_t>::max()) {
        return fail(reason, "visual asset pack payload is truncated");
    }
    candidate.textures.resize(
        static_cast<std::size_t>(textureCount)
    );
    for (VisualTextureImageV1& texture : candidate.textures) {
        if (!readTexture(reader, texture)) {
            return fail(reason, "visual texture payload is truncated");
        }
    }
    if (!reader.scalar(bindingCount) ||
        bindingCount > std::numeric_limits<std::uint32_t>::max()) {
        return fail(reason, "visual binding payload is truncated");
    }
    candidate.symbolicBindings.resize(
        static_cast<std::size_t>(bindingCount)
    );
    for (VisualSymbolicBindingV1& binding :
         candidate.symbolicBindings) {
        if (!readBinding(reader, binding)) {
            return fail(reason, "visual binding payload is truncated");
        }
    }
    if (!reader.complete() || !candidate.valid(reason)) {
        return false;
    }
    output = std::move(candidate);
    return true;
}

std::string computeVisualAssetPackContentHash(
    const VisualAssetPackV1& pack
) {
    return fnvContentHash(pack);
}

bool appendVisualAssetPack(
    VisualAssetPackV1&& pack,
    const std::uint32_t assetIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    VisualRenderSceneV2& scene,
    std::string* reason
) {
    if (!pack.valid(reason) || assetIndex >= scene.assetCount ||
        semanticId == 0u || semanticId == MR_INVALID_INDEX ||
        instanceId == 0u || instanceId == MR_INVALID_INDEX) {
        return reason != nullptr && !reason->empty()
            ? false
            : fail(reason, "visual pack scene binding is invalid");
    }
    if (scene.vertices.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.vertices.size() ||
        scene.indices.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.indices.size() ||
        scene.primitives.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.primitives.size() ||
        scene.instances.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.instances.size() ||
        scene.materials.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.materials.size() ||
        scene.textures.size() >
            std::numeric_limits<std::uint32_t>::max() -
                pack.textures.size()) {
        return fail(reason, "visual pack append exceeds uint32 capacity");
    }
    const std::uint32_t vertexOffset =
        static_cast<std::uint32_t>(scene.vertices.size());
    const std::uint32_t indexOffset =
        static_cast<std::uint32_t>(scene.indices.size());
    const std::uint32_t primitiveOffset =
        static_cast<std::uint32_t>(scene.primitives.size());
    const std::uint32_t instanceOffset =
        static_cast<std::uint32_t>(scene.instances.size());
    const std::uint32_t materialOffset =
        static_cast<std::uint32_t>(scene.materials.size());
    const std::uint32_t textureOffset =
        static_cast<std::uint32_t>(scene.textures.size());

    for (std::uint32_t& index : pack.indices) {
        index += vertexOffset;
    }
    for (MRVisualMaterialGPUV2& material : pack.materials) {
        material.textureIndices0 = {
            remapTexture(material.textureIndices0.x, textureOffset),
            remapTexture(material.textureIndices0.y, textureOffset),
            remapTexture(material.textureIndices0.z, textureOffset),
            remapTexture(material.textureIndices0.w, textureOffset),
        };
        material.textureIndices1 = {
            remapTexture(material.textureIndices1.x, textureOffset),
            remapTexture(material.textureIndices1.y, textureOffset),
            remapTexture(material.textureIndices1.z, textureOffset),
            remapTexture(material.textureIndices1.w, textureOffset),
        };
    }
    for (MRVisualInstanceGPUV2& instance : pack.instances) {
        instance.binding.x = assetIndex;
        instance.identity.x = semanticId;
        instance.identity.y = instanceId;
        instance.geometry.x += primitiveOffset;
    }
    for (const VisualSymbolicBindingV1& binding :
         pack.symbolicBindings) {
        MRVisualInstanceGPUV2& instance =
            pack.instances[binding.instanceIndex];
        instance.binding.y = binding.bodyIndex;
        instance.binding.z = binding.binding;
    }
    for (MRVisualPrimitiveGPUV2& primitive : pack.primitives) {
        primitive.geometry.x += indexOffset;
        primitive.geometry.z += materialOffset;
        primitive.geometry.w += instanceOffset;
        primitive.identity.x = semanticId;
        primitive.identity.y = instanceId;
    }
    const auto moveArena = []<typename Value>(
        std::vector<Value>& destination,
        std::vector<Value>& source
    ) {
        if (destination.empty()) {
            destination = std::move(source);
            return;
        }
        destination.reserve(destination.size() + source.size());
        destination.insert(
            destination.end(),
            std::make_move_iterator(source.begin()),
            std::make_move_iterator(source.end())
        );
        source.clear();
    };
    moveArena(scene.vertices, pack.vertices);
    moveArena(scene.indices, pack.indices);
    moveArena(scene.materials, pack.materials);
    moveArena(scene.textures, pack.textures);
    moveArena(scene.instances, pack.instances);
    moveArena(scene.primitives, pack.primitives);
    scene.fingerprint = 0u;
    return true;
}

VisualEnvironmentV1 makeNeutralStudioEnvironmentV1() {
    VisualEnvironmentV1 result;
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.scalar(result.textureIndex);
    hash.scalar(result.intensity);
    hash.scalar(result.rotationRadians);
    hash.values<mr_float4>(result.diffuseSH);
    result.fingerprint = hash.finish();
    return result;
}

VisualLightRigV1 makeStudioKeyLightRigV1() {
    VisualLightRigV1 result;
    result.lights = {defaultLight()};
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.values<MRVisualLightGPUV1>(result.lights);
    result.fingerprint = hash.finish();
    return result;
}

VisualLightRigV1 makeIndoorAreaLightRigV1() {
    VisualLightRigV1 result;
    result.id = "indoor_area";
    result.contentHash = "builtin:indoor-area-v1";
    MRVisualLightGPUV1 light = defaultLight();
    light.positionAndRange = {0.6f, -0.4f, 2.6f, 20.0f};
    light.directionAndSpot = {-0.25f, 0.20f, -0.95f, -1.0f};
    light.colorAndIntensity = {1.0f, 0.94f, 0.86f, 950.0f};
    light.shape = {0.9f, 0.7f, -1.0f, 0.08f};
    light.identity.x = MR_VISUAL_LIGHT_RECTANGLE;
    light.identity.y = MR_VISUAL_LIGHT_UNIT_NIT;
    light.shadow.x = 1u;
    result.lights = {light};
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.values<MRVisualLightGPUV1>(result.lights);
    result.fingerprint = hash.finish();
    return result;
}

std::uint64_t computeVisualRenderSceneV2Fingerprint(
    const VisualRenderSceneV2& scene
) {
    HashBuilder hash;
    hash.string(scene.id);
    hash.scalar(scene.assetCount);
    hash.scalar(scene.bodyCount);
    hash.values<MRHybridGaussianGPU>(scene.gaussians);
    hash.values<MRVisualVertexGPUV2>(scene.vertices);
    hash.values<std::uint32_t>(scene.indices);
    hash.values<MRVisualPrimitiveGPUV2>(scene.primitives);
    hash.values<MRVisualInstanceGPUV2>(scene.instances);
    hash.values<MRVisualMaterialGPUV2>(scene.materials);
    for (const VisualTextureImageV1& texture : scene.textures) {
        hash.string(texture.id);
        hash.string(texture.contentHash);
        hash.scalar(texture.width);
        hash.scalar(texture.height);
        hash.scalar(texture.flags);
        hash.values<std::uint32_t>(texture.mipTexelOffsets);
        hash.values<std::uint8_t>(texture.rgba8);
    }
    hash.values<MRVisualSensorBindingGPU>(scene.sensorBindings);
    hash.string(scene.environment.id);
    hash.string(scene.environment.contentHash);
    hash.scalar(scene.environment.textureIndex);
    hash.scalar(scene.environment.intensity);
    hash.scalar(scene.environment.rotationRadians);
    hash.values<mr_float4>(scene.environment.diffuseSH);
    hash.string(scene.lightRig.id);
    hash.string(scene.lightRig.contentHash);
    hash.values<MRVisualLightGPUV1>(scene.lightRig.lights);
    return hash.finish();
}

std::uint64_t computeVisualSceneManifestV2Fingerprint(
    const VisualSceneManifestV2& manifest
) {
    HashBuilder hash;
    hash.scalar(manifest.schemaVersion);
    hash.string(manifest.id);
    hash.string(manifest.coordinateConvention);
    hash.scalar(manifest.worldFingerprint);
    for (const std::string& packHash : manifest.visualPackHashes) {
        hash.string(packHash);
    }
    hash.string(manifest.environmentMapHash);
    hash.string(manifest.lightRigHash);
    hash.string(manifest.preprocessingProvenance);
    hash.scalar(manifest.renderScene.fingerprint);
    return hash.finish();
}

bool writeVisualSceneManifestV2(
    const VisualSceneManifestV2& manifest,
    const std::filesystem::path& path,
    std::string* reason
) {
    std::string manifestReason;
    if (path.empty() || !manifest.valid(&manifestReason)) {
        return fail(
            reason,
            "V2 visual scene manifest input is invalid: " +
                manifestReason
        );
    }
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(
            path.parent_path(),
            error
        );
        if (error) {
            return fail(
                reason,
                "could not create V2 visual manifest directory"
            );
        }
    }
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(manifest.fingerprint);
    std::ofstream output(
        temporary,
        std::ios::binary | std::ios::trunc
    );
    if (!output) {
        return fail(reason, "could not open V2 visual scene manifest");
    }
    output
        << "{\n"
        << "  \"schema_version\": 2,\n"
        << "  \"id\": \"" << jsonEscape(manifest.id) << "\",\n"
        << "  \"coordinate_convention\": \""
        << jsonEscape(manifest.coordinateConvention) << "\",\n"
        << "  \"world_fingerprint\": "
        << manifest.worldFingerprint << ",\n"
        << "  \"fingerprint\": " << manifest.fingerprint << ",\n"
        << "  \"visual_pack_hashes\": [";
    for (std::size_t index = 0u;
         index < manifest.visualPackHashes.size();
         ++index) {
        if (index != 0u) {
            output << ',';
        }
        output << '"' << jsonEscape(manifest.visualPackHashes[index])
               << '"';
    }
    const VisualRenderSceneV2& scene = manifest.renderScene;
    output
        << "],\n"
        << "  \"environment_map_hash\": \""
        << jsonEscape(manifest.environmentMapHash) << "\",\n"
        << "  \"light_rig_hash\": \""
        << jsonEscape(manifest.lightRigHash) << "\",\n"
        << "  \"preprocessing_provenance\": \""
        << jsonEscape(manifest.preprocessingProvenance) << "\",\n"
        << "  \"render_scene\": {\n"
        << "    \"id\": \"" << jsonEscape(scene.id) << "\",\n"
        << "    \"asset_count\": " << scene.assetCount << ",\n"
        << "    \"body_count\": " << scene.bodyCount << ",\n"
        << "    \"gaussian_count\": " << scene.gaussians.size() << ",\n"
        << "    \"vertex_count\": " << scene.vertices.size() << ",\n"
        << "    \"index_count\": " << scene.indices.size() << ",\n"
        << "    \"primitive_count\": " << scene.primitives.size() << ",\n"
        << "    \"instance_count\": " << scene.instances.size() << ",\n"
        << "    \"material_count\": " << scene.materials.size() << ",\n"
        << "    \"texture_count\": " << scene.textures.size() << ",\n"
        << "    \"sensor_binding_count\": "
        << scene.sensorBindings.size() << ",\n"
        << "    \"light_count\": " << scene.lightRig.lights.size() << ",\n"
        << "    \"environment_id\": \""
        << jsonEscape(scene.environment.id) << "\",\n"
        << "    \"light_rig_id\": \""
        << jsonEscape(scene.lightRig.id) << "\",\n"
        << "    \"fingerprint\": " << scene.fingerprint << "\n"
        << "  }\n"
        << "}\n";
    output.close();
    if (!output) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not write V2 visual scene manifest");
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not publish V2 visual scene manifest");
    }
    return true;
}

std::uint64_t computeVisualRendererProfileFingerprint(
    const VisualRendererProfileV1& profile
) {
    HashBuilder hash;
    hash.string(profile.id);
    hash.scalar(profile.kind);
    hash.scalar(profile.temporalSamples);
    hash.scalar(profile.rollingShutterBands);
    hash.scalar(profile.shadowMapResolution);
    hash.scalar(profile.areaLightSamples);
    hash.scalar(profile.rayQueryVisibility);
    hash.scalar(profile.retainObservation);
    return hash.finish();
}

const char* visualAssetCookStatusName(
    const VisualAssetCookStatus status
) noexcept {
    switch (status) {
    case VisualAssetCookStatus::success:
        return "success";
    case VisualAssetCookStatus::ioFailure:
        return "io_failure";
    case VisualAssetCookStatus::unsupportedFormat:
        return "unsupported_format";
    case VisualAssetCookStatus::malformedAsset:
        return "malformed_asset";
    case VisualAssetCookStatus::unsupportedFeature:
        return "unsupported_feature";
    case VisualAssetCookStatus::invalidGeometry:
        return "invalid_geometry";
    case VisualAssetCookStatus::invalidMaterial:
        return "invalid_material";
    case VisualAssetCookStatus::invalidTexture:
        return "invalid_texture";
    case VisualAssetCookStatus::invalidBinding:
        return "invalid_binding";
    case VisualAssetCookStatus::capacityOverflow:
        return "capacity_overflow";
    case VisualAssetCookStatus::writeFailure:
        return "write_failure";
    case VisualAssetCookStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
