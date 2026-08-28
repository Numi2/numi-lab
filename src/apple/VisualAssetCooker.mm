#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <MetalKit/MetalKit.h>
#import <ModelIO/ModelIO.h>

#import <simd/simd.h>

#include "metalrobo/VisualPresentation.hpp"
#include "VisualKernelHashes.h"

#include <CommonCrypto/CommonDigest.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr std::uint32_t kGlbMagic = 0x46546c67u;
constexpr std::uint32_t kGlbJsonChunk = 0x4e4f534au;
constexpr std::uint32_t kGlbBinaryChunk = 0x004e4942u;

struct Matrix4 {
    double value[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
};

struct GltfDocument {
    NSDictionary* root = nil;
    std::filesystem::path baseDirectory;
    std::vector<std::vector<std::uint8_t>> buffers;
};

struct PrimitiveTemplate {
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t materialIndex = 0u;
    mr_float4 boundsMinimum{};
    mr_float4 boundsMaximum{};
};

struct ImportedMesh {
    std::vector<PrimitiveTemplate> primitives;
};

struct AuthoredVisualBodyBinding {
    std::uint32_t body = MR_INVALID_INDEX;
    std::uint32_t kind = MR_VISUAL_BINDING_ASSET;
};

bool resolveVisualBodyBinding(
    const VisualAssetCookOptions& options,
    const std::string& node,
    AuthoredVisualBodyBinding& output,
    std::string& message
) {
    const auto articulated = options.linkBodyIndices.find(node);
    const auto rigid = options.rigidBodyIndices.find(node);
    if (articulated != options.linkBodyIndices.end() &&
        rigid != options.rigidBodyIndices.end()) {
        message = "visual node has both rigid and articulated bindings";
        return false;
    }
    if (articulated != options.linkBodyIndices.end()) {
        output = {
            articulated->second,
            MR_VISUAL_BINDING_ARTICULATED_LINK,
        };
    } else if (rigid != options.rigidBodyIndices.end()) {
        output = {rigid->second, MR_VISUAL_BINDING_RIGID_BODY};
    }
    return true;
}

VisualAssetCookDiagnostics reject(
    VisualAssetCookDiagnostics diagnostics,
    const VisualAssetCookStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

std::optional<std::vector<std::uint8_t>> readBytes(
    const std::filesystem::path& path
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::nullopt;
    }
    return std::vector<std::uint8_t>{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>(),
    };
}

void updateSha256(
    CC_SHA256_CTX& context,
    const void* data,
    const std::size_t size
) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t offset = 0u;
    while (offset < size) {
        const std::size_t count = std::min<std::size_t>(
            size - offset,
            std::numeric_limits<CC_LONG>::max()
        );
        CC_SHA256_Update(
            &context,
            bytes + offset,
            static_cast<CC_LONG>(count)
        );
        offset += count;
    }
}

std::string finishSha256(CC_SHA256_CTX& context) {
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    constexpr std::array digits{
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    };
    std::string result = "sha256:";
    result.reserve(result.size() + digest.size() * 2u);
    for (const std::uint8_t byte : digest) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 15u]);
    }
    return result;
}

std::string sha256(const std::span<const std::uint8_t> bytes) {
    CC_SHA256_CTX context{};
    CC_SHA256_Init(&context);
    updateSha256(context, bytes.data(), bytes.size());
    return finishSha256(context);
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return {};
    }
    CC_SHA256_CTX context{};
    CC_SHA256_Init(&context);
    std::array<char, 1u << 20u> buffer{};
    while (stream) {
        stream.read(
            buffer.data(),
            static_cast<std::streamsize>(buffer.size())
        );
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            updateSha256(
                context,
                buffer.data(),
                static_cast<std::size_t>(count)
            );
        }
    }
    if (!stream.eof()) {
        return {};
    }
    return finishSha256(context);
}

std::uint32_t littleU32(
    const std::span<const std::uint8_t> bytes,
    const std::size_t offset
) {
    return static_cast<std::uint32_t>(bytes[offset]) |
        static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u |
        static_cast<std::uint32_t>(bytes[offset + 2u]) << 16u |
        static_cast<std::uint32_t>(bytes[offset + 3u]) << 24u;
}

NSArray* arrayValue(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    return [value isKindOfClass:NSArray.class]
        ? static_cast<NSArray*>(value)
        : nil;
}

NSDictionary* dictionaryValue(
    NSDictionary* dictionary,
    NSString* key
) {
    id value = dictionary[key];
    return [value isKindOfClass:NSDictionary.class]
        ? static_cast<NSDictionary*>(value)
        : nil;
}

NSNumber* numberValue(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    return [value isKindOfClass:NSNumber.class]
        ? static_cast<NSNumber*>(value)
        : nil;
}

NSString* stringValue(NSDictionary* dictionary, NSString* key) {
    id value = dictionary[key];
    return [value isKindOfClass:NSString.class]
        ? static_cast<NSString*>(value)
        : nil;
}

std::string utf8(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return value.UTF8String;
}

std::string frameworkVersion(Class frameworkClass) {
    NSBundle* bundle = [NSBundle bundleForClass:frameworkClass];
    NSString* version =
        bundle.infoDictionary[@"CFBundleVersion"];
    if (version.length == 0u) {
        version =
            bundle.infoDictionary[
                @"CFBundleShortVersionString"
            ];
    }
    return utf8(version);
}

std::uint32_t uintValue(
    NSDictionary* dictionary,
    NSString* key,
    const std::uint32_t fallback = 0u
) {
    NSNumber* value = numberValue(dictionary, key);
    return value == nil
        ? fallback
        : static_cast<std::uint32_t>(value.unsignedLongLongValue);
}

double doubleValue(
    NSDictionary* dictionary,
    NSString* key,
    const double fallback
) {
    NSNumber* value = numberValue(dictionary, key);
    return value == nil ? fallback : value.doubleValue;
}

bool boolValue(
    NSDictionary* dictionary,
    NSString* key,
    const bool fallback
) {
    NSNumber* value = numberValue(dictionary, key);
    return value == nil ? fallback : value.boolValue;
}

bool loadUri(
    NSString* uri,
    const std::filesystem::path& base,
    std::vector<std::uint8_t>& output
) {
    if ([uri hasPrefix:@"data:"]) {
        NSRange separator = [uri rangeOfString:@","];
        if (separator.location == NSNotFound) {
            return false;
        }
        NSString* metadata =
            [uri substringToIndex:separator.location];
        NSString* payload =
            [uri substringFromIndex:separator.location + 1u];
        NSData* data = nil;
        if ([metadata hasSuffix:@";base64"]) {
            data = [[NSData alloc]
                initWithBase64EncodedString:payload
                                    options:0u];
        } else {
            NSString* decoded =
                [payload stringByRemovingPercentEncoding];
            data = [decoded dataUsingEncoding:NSUTF8StringEncoding];
        }
        if (data == nil) {
            return false;
        }
        output.assign(
            static_cast<const std::uint8_t*>(data.bytes),
            static_cast<const std::uint8_t*>(data.bytes) + data.length
        );
        return true;
    }
    NSString* decoded = [uri stringByRemovingPercentEncoding];
    const std::filesystem::path path =
        base / utf8(decoded == nil ? uri : decoded);
    const auto bytes = readBytes(path);
    if (!bytes.has_value()) {
        return false;
    }
    output = *bytes;
    return true;
}

bool parseGltf(
    const std::filesystem::path& source,
    const std::span<const std::uint8_t> sourceBytes,
    GltfDocument& document,
    std::string& message
) {
    std::vector<std::uint8_t> jsonBytes;
    std::vector<std::uint8_t> binaryChunk;
    const bool isGlb = source.extension() == ".glb";
    if (isGlb) {
        if (sourceBytes.size() < 20u ||
            littleU32(sourceBytes, 0u) != kGlbMagic ||
            littleU32(sourceBytes, 4u) != 2u ||
            littleU32(sourceBytes, 8u) != sourceBytes.size()) {
            message = "GLB header is invalid";
            return false;
        }
        std::size_t offset = 12u;
        while (offset + 8u <= sourceBytes.size()) {
            const std::uint32_t size =
                littleU32(sourceBytes, offset);
            const std::uint32_t type =
                littleU32(sourceBytes, offset + 4u);
            offset += 8u;
            if (size > sourceBytes.size() - offset) {
                message = "GLB chunk exceeds the file";
                return false;
            }
            const auto chunk =
                sourceBytes.subspan(offset, size);
            if (type == kGlbJsonChunk && jsonBytes.empty()) {
                jsonBytes.assign(chunk.begin(), chunk.end());
            } else if (
                type == kGlbBinaryChunk &&
                binaryChunk.empty()
            ) {
                binaryChunk.assign(chunk.begin(), chunk.end());
            }
            offset += size;
        }
        if (jsonBytes.empty()) {
            message = "GLB has no JSON chunk";
            return false;
        }
    } else {
        jsonBytes.assign(sourceBytes.begin(), sourceBytes.end());
    }
    NSData* data = [NSData
        dataWithBytes:jsonBytes.data()
               length:jsonBytes.size()];
    NSError* error = nil;
    id object = [NSJSONSerialization
        JSONObjectWithData:data
                   options:0u
                     error:&error];
    if (![object isKindOfClass:NSDictionary.class]) {
        message = "glTF JSON is malformed: " +
            utf8(error.localizedDescription);
        return false;
    }
    document.root = static_cast<NSDictionary*>(object);
    document.baseDirectory = source.parent_path();
    NSArray* buffers = arrayValue(document.root, @"buffers");
    if (buffers == nil || buffers.count == 0u) {
        message = "glTF has no buffers";
        return false;
    }
    document.buffers.reserve(buffers.count);
    for (NSUInteger index = 0u; index < buffers.count; ++index) {
        NSDictionary* buffer = [buffers[index]
            isKindOfClass:NSDictionary.class]
            ? static_cast<NSDictionary*>(buffers[index])
            : nil;
        if (buffer == nil) {
            message = "glTF buffer record is invalid";
            return false;
        }
        std::vector<std::uint8_t> bytes;
        NSString* uri = stringValue(buffer, @"uri");
        if (uri == nil && index == 0u && isGlb) {
            bytes = binaryChunk;
        } else if (uri == nil ||
                   !loadUri(
                       uri,
                       document.baseDirectory,
                       bytes
                   )) {
            message = "glTF buffer URI could not be read";
            return false;
        }
        const std::uint64_t declared =
            numberValue(buffer, @"byteLength").unsignedLongLongValue;
        if (declared > bytes.size()) {
            message = "glTF buffer is shorter than declared";
            return false;
        }
        document.buffers.push_back(std::move(bytes));
    }
    return true;
}

std::size_t componentSize(const std::uint32_t componentType) {
    switch (componentType) {
    case 5120u:
    case 5121u:
        return 1u;
    case 5122u:
    case 5123u:
        return 2u;
    case 5125u:
    case 5126u:
        return 4u;
    default:
        return 0u;
    }
}

std::size_t componentCount(NSString* type) {
    if ([type isEqualToString:@"SCALAR"]) {
        return 1u;
    }
    if ([type isEqualToString:@"VEC2"]) {
        return 2u;
    }
    if ([type isEqualToString:@"VEC3"]) {
        return 3u;
    }
    if ([type isEqualToString:@"VEC4"]) {
        return 4u;
    }
    return 0u;
}

double readComponent(
    const std::uint8_t* source,
    const std::uint32_t type,
    const bool normalized
) {
    switch (type) {
    case 5120u: {
        std::int8_t value = 0;
        std::memcpy(&value, source, sizeof(value));
        return normalized
            ? std::max(-1.0, static_cast<double>(value) / 127.0)
            : value;
    }
    case 5121u: {
        const std::uint8_t value = *source;
        return normalized
            ? static_cast<double>(value) / 255.0
            : value;
    }
    case 5122u: {
        std::int16_t value = 0;
        std::memcpy(&value, source, sizeof(value));
        return normalized
            ? std::max(-1.0, static_cast<double>(value) / 32767.0)
            : value;
    }
    case 5123u: {
        std::uint16_t value = 0u;
        std::memcpy(&value, source, sizeof(value));
        return normalized
            ? static_cast<double>(value) / 65535.0
            : value;
    }
    case 5125u: {
        std::uint32_t value = 0u;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }
    case 5126u: {
        float value = 0.0f;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }
    default:
        return 0.0;
    }
}

bool accessorBytes(
    const GltfDocument& document,
    const std::uint32_t accessorIndex,
    const std::uint8_t*& first,
    std::size_t& count,
    std::size_t& components,
    std::size_t& stride,
    std::uint32_t& type,
    bool& normalized,
    std::string& message
) {
    NSArray* accessors = arrayValue(document.root, @"accessors");
    NSArray* views = arrayValue(document.root, @"bufferViews");
    if (accessors == nil || views == nil ||
        accessorIndex >= accessors.count ||
        ![accessors[accessorIndex]
            isKindOfClass:NSDictionary.class]) {
        message = "glTF accessor index is invalid";
        return false;
    }
    NSDictionary* accessor = static_cast<NSDictionary*>(
        accessors[accessorIndex]
    );
    if (dictionaryValue(accessor, @"sparse") != nil) {
        message = "sparse glTF accessors are not supported";
        return false;
    }
    const std::uint32_t viewIndex =
        uintValue(accessor, @"bufferView", MR_INVALID_INDEX);
    if (viewIndex >= views.count ||
        ![views[viewIndex] isKindOfClass:NSDictionary.class]) {
        message = "glTF accessor has no valid buffer view";
        return false;
    }
    NSDictionary* view =
        static_cast<NSDictionary*>(views[viewIndex]);
    const std::uint32_t bufferIndex =
        uintValue(view, @"buffer", MR_INVALID_INDEX);
    if (bufferIndex >= document.buffers.size()) {
        message = "glTF buffer view references an invalid buffer";
        return false;
    }
    type = uintValue(accessor, @"componentType");
    const std::size_t bytesPerComponent = componentSize(type);
    components = componentCount(stringValue(accessor, @"type"));
    count = static_cast<std::size_t>(
        uintValue(accessor, @"count")
    );
    normalized = boolValue(accessor, @"normalized", false);
    if (bytesPerComponent == 0u || components == 0u ||
        count == 0u) {
        message = "glTF accessor shape is unsupported";
        return false;
    }
    const std::size_t elementBytes =
        bytesPerComponent * components;
    stride = static_cast<std::size_t>(
        uintValue(
            view,
            @"byteStride",
            static_cast<std::uint32_t>(elementBytes)
        )
    );
    if (stride < elementBytes) {
        message = "glTF accessor stride is too small";
        return false;
    }
    const std::size_t offset =
        static_cast<std::size_t>(uintValue(view, @"byteOffset")) +
        static_cast<std::size_t>(uintValue(accessor, @"byteOffset"));
    const auto& buffer = document.buffers[bufferIndex];
    if (offset > buffer.size() ||
        count > (buffer.size() - offset + stride - elementBytes) /
            stride) {
        message = "glTF accessor exceeds its buffer";
        return false;
    }
    first = buffer.data() + offset;
    return true;
}

bool readFloatAccessor(
    const GltfDocument& document,
    const std::uint32_t accessorIndex,
    const std::size_t expectedComponents,
    std::vector<float>& values,
    std::string& message
) {
    const std::uint8_t* first = nullptr;
    std::size_t count = 0u;
    std::size_t components = 0u;
    std::size_t stride = 0u;
    std::uint32_t type = 0u;
    bool normalized = false;
    if (!accessorBytes(
            document,
            accessorIndex,
            first,
            count,
            components,
            stride,
            type,
            normalized,
            message
        ) ||
        components != expectedComponents) {
        if (message.empty()) {
            message = "glTF accessor component count is invalid";
        }
        return false;
    }
    const std::size_t bytesPerComponent = componentSize(type);
    values.resize(count * components);
    for (std::size_t item = 0u; item < count; ++item) {
        for (std::size_t component = 0u;
             component < components;
             ++component) {
            const double value = readComponent(
                first + item * stride +
                    component * bytesPerComponent,
                type,
                normalized
            );
            if (!std::isfinite(value) ||
                std::abs(value) >
                    std::numeric_limits<float>::max()) {
                message = "glTF accessor contains a non-finite value";
                return false;
            }
            values[item * components + component] =
                static_cast<float>(value);
        }
    }
    return true;
}

bool readIndexAccessor(
    const GltfDocument& document,
    const std::uint32_t accessorIndex,
    std::vector<std::uint32_t>& values,
    std::string& message
) {
    const std::uint8_t* first = nullptr;
    std::size_t count = 0u;
    std::size_t components = 0u;
    std::size_t stride = 0u;
    std::uint32_t type = 0u;
    bool normalized = false;
    if (!accessorBytes(
            document,
            accessorIndex,
            first,
            count,
            components,
            stride,
            type,
            normalized,
            message
        ) ||
        components != 1u ||
        normalized ||
        (type != 5121u && type != 5123u && type != 5125u)) {
        message = "glTF index accessor is unsupported";
        return false;
    }
    values.resize(count);
    for (std::size_t index = 0u; index < count; ++index) {
        values[index] = static_cast<std::uint32_t>(
            readComponent(first + index * stride, type, false)
        );
    }
    return true;
}

std::vector<std::uint8_t> imagePayload(
    const GltfDocument& document,
    NSDictionary* image
) {
    std::vector<std::uint8_t> result;
    if (NSString* uri = stringValue(image, @"uri"); uri != nil) {
        loadUri(uri, document.baseDirectory, result);
        return result;
    }
    NSArray* views = arrayValue(document.root, @"bufferViews");
    const std::uint32_t viewIndex =
        uintValue(image, @"bufferView", MR_INVALID_INDEX);
    if (views == nil || viewIndex >= views.count ||
        ![views[viewIndex] isKindOfClass:NSDictionary.class]) {
        return result;
    }
    NSDictionary* view =
        static_cast<NSDictionary*>(views[viewIndex]);
    const std::uint32_t bufferIndex =
        uintValue(view, @"buffer", MR_INVALID_INDEX);
    if (bufferIndex >= document.buffers.size()) {
        return result;
    }
    const std::size_t offset = uintValue(view, @"byteOffset");
    const std::size_t length = uintValue(view, @"byteLength");
    const auto& buffer = document.buffers[bufferIndex];
    if (offset > buffer.size() ||
        length > buffer.size() - offset) {
        return {};
    }
    result.assign(
        buffer.begin() + static_cast<std::ptrdiff_t>(offset),
        buffer.begin() +
            static_cast<std::ptrdiff_t>(offset + length)
    );
    return result;
}

float srgbToLinear(const float value) {
    return value <= 0.04045f
        ? value / 12.92f
        : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb(const float value) {
    return value <= 0.0031308f
        ? value * 12.92f
        : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

bool cookRgba8Texture(
    std::vector<std::uint8_t> level,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool srgb,
    const bool mipmaps,
    const std::string& id,
    const std::string& contentHash,
    VisualTextureImageV2& output
) {
    output = {};
    output.id = id;
    output.contentHash = contentHash;
    output.width = width;
    output.height = height;
    output.flags = srgb ? MR_VISUAL_TEXTURE_SRGB : 0u;
    output.pixelFormat = srgb
        ? VisualTexturePixelFormatV2::rgba8UnormSrgb
        : VisualTexturePixelFormatV2::rgba8Unorm;
    output.dimension = VisualTextureDimensionV2::texture2D;
    output.arrayLength = 1u;
    std::uint32_t levelWidth = width;
    std::uint32_t levelHeight = height;
    std::uint32_t mipLevel = 0u;
    while (true) {
        if (levelWidth >
            (std::numeric_limits<std::uint32_t>::max() - 255u) /
                4u) {
            return false;
        }
        const std::uint32_t bytesPerRow =
            (levelWidth * 4u + 255u) & ~255u;
        if (levelHeight != 0u &&
            bytesPerRow >
                std::numeric_limits<std::uint32_t>::max() /
                    levelHeight) {
            return false;
        }
        const std::uint32_t bytesPerImage =
            bytesPerRow * levelHeight;
        const std::uint64_t offset = output.data.size();
        if (bytesPerImage >
            std::numeric_limits<std::size_t>::max() -
                output.data.size()) {
            return false;
        }
        output.data.resize(
            output.data.size() + bytesPerImage,
            0u
        );
        for (std::uint32_t row = 0u;
             row < levelHeight;
             ++row) {
            std::memcpy(
                output.data.data() + offset +
                    static_cast<std::uint64_t>(row) * bytesPerRow,
                level.data() +
                    static_cast<std::size_t>(row) *
                        levelWidth * 4u,
                static_cast<std::size_t>(levelWidth) * 4u
            );
        }
        output.subresources.push_back({
            mipLevel,
            0u,
            levelWidth,
            levelHeight,
            offset,
            bytesPerImage,
            bytesPerRow,
            bytesPerImage,
        });
        ++mipLevel;
        if (!mipmaps ||
            (levelWidth == 1u && levelHeight == 1u)) {
            break;
        }
        const std::uint32_t nextWidth =
            std::max(1u, levelWidth / 2u);
        const std::uint32_t nextHeight =
            std::max(1u, levelHeight / 2u);
        std::vector<std::uint8_t> next(
            static_cast<std::size_t>(nextWidth) *
                nextHeight * 4u
        );
        for (std::uint32_t y = 0u; y < nextHeight; ++y) {
            for (std::uint32_t x = 0u; x < nextWidth; ++x) {
                std::array<float, 4u> sum{};
                for (std::uint32_t dy = 0u; dy < 2u; ++dy) {
                    for (std::uint32_t dx = 0u; dx < 2u; ++dx) {
                        const std::uint32_t sourceX =
                            std::min(levelWidth - 1u, 2u * x + dx);
                        const std::uint32_t sourceY =
                            std::min(levelHeight - 1u, 2u * y + dy);
                        const std::size_t sourcePixel =
                            static_cast<std::size_t>(sourceY) *
                                levelWidth +
                            sourceX;
                        for (std::size_t channel = 0u;
                             channel < 4u;
                             ++channel) {
                            float value =
                                level[sourcePixel * 4u + channel] /
                                255.0f;
                            if (srgb && channel < 3u) {
                                value = srgbToLinear(value);
                            }
                            sum[channel] += value;
                        }
                    }
                }
                const std::size_t destination =
                    static_cast<std::size_t>(y) * nextWidth + x;
                for (std::size_t channel = 0u;
                     channel < 4u;
                     ++channel) {
                    float value = 0.25f * sum[channel];
                    if (srgb && channel < 3u) {
                        value = linearToSrgb(value);
                    }
                    next[destination * 4u + channel] =
                        static_cast<std::uint8_t>(std::clamp(
                            std::lround(value * 255.0f),
                            0l,
                            255l
                        ));
                }
            }
        }
        level = std::move(next);
        levelWidth = nextWidth;
        levelHeight = nextHeight;
    }
    output.mipCount = mipLevel;
    return true;
}

bool decodeTexture(
    const std::vector<std::uint8_t>& encoded,
    const bool srgb,
    const bool mipmaps,
    const std::string& id,
    VisualTextureImageV2& output
) {
    if (encoded.empty()) {
        return false;
    }
    NSData* data = [NSData
        dataWithBytes:encoded.data()
               length:encoded.size()];
    CGImageSourceRef source = CGImageSourceCreateWithData(
        (__bridge CFDataRef)data,
        nullptr
    );
    if (source == nullptr) {
        return false;
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(
        source,
        0u,
        nullptr
    );
    CFRelease(source);
    if (image == nullptr) {
        return false;
    }
    const std::size_t width = CGImageGetWidth(image);
    const std::size_t height = CGImageGetHeight(image);
    if (width == 0u || height == 0u ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max() ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height >
            std::numeric_limits<std::size_t>::max() / 4u) {
        CGImageRelease(image);
        return false;
    }
    std::vector<std::uint8_t> level(width * height * 4u);
    CGColorSpaceRef colorSpace =
        CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef context = CGBitmapContextCreate(
        level.data(),
        width,
        height,
        8u,
        width * 4u,
        colorSpace,
        static_cast<CGBitmapInfo>(
            static_cast<std::uint32_t>(
                kCGImageAlphaPremultipliedLast
            ) |
            static_cast<std::uint32_t>(
                kCGBitmapByteOrder32Big
            )
        )
    );
    CGColorSpaceRelease(colorSpace);
    if (context == nullptr) {
        CGImageRelease(image);
        return false;
    }
    CGContextDrawImage(
        context,
        CGRectMake(0.0, 0.0, width, height),
        image
    );
    CGContextRelease(context);
    CGImageRelease(image);
    // CoreGraphics returns premultiplied color; restore straight alpha for
    // glTF's material equations.
    for (std::size_t pixel = 0u;
         pixel < width * height;
         ++pixel) {
        const float alpha = level[pixel * 4u + 3u] / 255.0f;
        if (alpha > 0.0f && alpha < 1.0f) {
            for (std::size_t channel = 0u; channel < 3u; ++channel) {
                level[pixel * 4u + channel] =
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(
                            level[pixel * 4u + channel] / alpha
                        ),
                        0l,
                        255l
                    ));
            }
        }
    }
    return cookRgba8Texture(
        std::move(level),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        srgb,
        mipmaps,
        id,
        sha256(encoded),
        output
    );
}

bool decodeModelTexture(
    MTKTextureLoader* loader,
    MDLTexture* texture,
    const bool srgb,
    const bool mipmaps,
    const std::string& textureId,
    VisualTextureImageV2& output
) {
    if (loader == nil || texture == nil) {
        return false;
    }
    NSError* error = nil;
    id<MTLTexture> native = [loader
        newTextureWithMDLTexture:texture
                         options:@{
                             MTKTextureLoaderOptionSRGB:
                                 @(srgb),
                             MTKTextureLoaderOptionAllocateMipmaps:
                                 @NO,
                             MTKTextureLoaderOptionTextureStorageMode:
                                 @(MTLStorageModeShared),
                             MTKTextureLoaderOptionTextureUsage:
                                 @(MTLTextureUsageShaderRead),
                         }
                           error:&error];
    if (native == nil ||
        native.textureType != MTLTextureType2D ||
        native.depth != 1u ||
        native.width == 0u ||
        native.height == 0u ||
        native.width >
            std::numeric_limits<std::uint32_t>::max() ||
        native.height >
            std::numeric_limits<std::uint32_t>::max() ||
        native.width >
            std::numeric_limits<std::size_t>::max() /
                native.height ||
        native.width * native.height >
            std::numeric_limits<std::size_t>::max() / 4u) {
        return false;
    }
    const std::size_t width = native.width;
    const std::size_t height = native.height;
    const bool rgba =
        native.pixelFormat == MTLPixelFormatRGBA8Unorm ||
        native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB;
    const bool bgra =
        native.pixelFormat == MTLPixelFormatBGRA8Unorm ||
        native.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!rgba && !bgra) {
        return false;
    }
    std::vector<std::uint8_t> pixels(width * height * 4u);
    [native
        getBytes:pixels.data()
     bytesPerRow:width * 4u
      fromRegion:MTLRegionMake2D(
                     0u,
                     0u,
                     width,
                     height
                 )
     mipmapLevel:0u];
    if (bgra) {
        for (std::size_t pixel = 0u;
             pixel < width * height;
             ++pixel) {
            std::swap(
                pixels[pixel * 4u],
                pixels[pixel * 4u + 2u]
            );
        }
    }
    const std::string contentHash = sha256(pixels);
    return cookRgba8Texture(
        std::move(pixels),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        srgb,
        mipmaps,
        textureId,
        contentHash,
        output
    );
}

void normalize3(float& x, float& y, float& z) {
    const float length =
        std::sqrt(x * x + y * y + z * z);
    if (length > 1.0e-12f) {
        x /= length;
        y /= length;
        z /= length;
    }
}

void generateNormals(
    std::span<MRVisualVertexGPUV2> vertices,
    const std::span<const std::uint32_t> indices
) {
    for (MRVisualVertexGPUV2& vertex : vertices) {
        vertex.normalAndTangentSign =
            {0.0f, 0.0f, 0.0f, 1.0f};
    }
    for (std::size_t index = 0u;
         index + 2u < indices.size();
         index += 3u) {
        const mr_float4 p0 = vertices[indices[index]].position;
        const mr_float4 p1 = vertices[indices[index + 1u]].position;
        const mr_float4 p2 = vertices[indices[index + 2u]].position;
        const float ax = p1.x - p0.x;
        const float ay = p1.y - p0.y;
        const float az = p1.z - p0.z;
        const float bx = p2.x - p0.x;
        const float by = p2.y - p0.y;
        const float bz = p2.z - p0.z;
        const float nx = ay * bz - az * by;
        const float ny = az * bx - ax * bz;
        const float nz = ax * by - ay * bx;
        for (std::size_t corner = 0u; corner < 3u; ++corner) {
            mr_float4& normal =
                vertices[indices[index + corner]]
                    .normalAndTangentSign;
            normal.x += nx;
            normal.y += ny;
            normal.z += nz;
        }
    }
    for (MRVisualVertexGPUV2& vertex : vertices) {
        normalize3(
            vertex.normalAndTangentSign.x,
            vertex.normalAndTangentSign.y,
            vertex.normalAndTangentSign.z
        );
    }
}

void generateTangents(
    std::span<MRVisualVertexGPUV2> vertices,
    const std::span<const std::uint32_t> indices
) {
    std::vector<std::array<float, 3u>> tangents(vertices.size());
    std::vector<std::array<float, 3u>> bitangents(vertices.size());
    for (std::size_t index = 0u;
         index + 2u < indices.size();
         index += 3u) {
        const std::uint32_t i0 = indices[index];
        const std::uint32_t i1 = indices[index + 1u];
        const std::uint32_t i2 = indices[index + 2u];
        const auto& v0 = vertices[i0];
        const auto& v1 = vertices[i1];
        const auto& v2 = vertices[i2];
        const float x1 = v1.position.x - v0.position.x;
        const float y1 = v1.position.y - v0.position.y;
        const float z1 = v1.position.z - v0.position.z;
        const float x2 = v2.position.x - v0.position.x;
        const float y2 = v2.position.y - v0.position.y;
        const float z2 = v2.position.z - v0.position.z;
        const float s1 = v1.texcoord01.x - v0.texcoord01.x;
        const float t1 = v1.texcoord01.y - v0.texcoord01.y;
        const float s2 = v2.texcoord01.x - v0.texcoord01.x;
        const float t2 = v2.texcoord01.y - v0.texcoord01.y;
        const float determinant = s1 * t2 - s2 * t1;
        if (std::abs(determinant) <= 1.0e-12f) {
            continue;
        }
        const float inverse = 1.0f / determinant;
        const std::array tangent{
            (t2 * x1 - t1 * x2) * inverse,
            (t2 * y1 - t1 * y2) * inverse,
            (t2 * z1 - t1 * z2) * inverse,
        };
        const std::array bitangent{
            (s1 * x2 - s2 * x1) * inverse,
            (s1 * y2 - s2 * y1) * inverse,
            (s1 * z2 - s2 * z1) * inverse,
        };
        for (const std::uint32_t vertex : {i0, i1, i2}) {
            for (std::size_t component = 0u;
                 component < 3u;
                 ++component) {
                tangents[vertex][component] += tangent[component];
                bitangents[vertex][component] +=
                    bitangent[component];
            }
        }
    }
    for (std::size_t index = 0u;
         index < vertices.size();
         ++index) {
        MRVisualVertexGPUV2& vertex = vertices[index];
        const auto& normal = vertex.normalAndTangentSign;
        auto tangent = tangents[index];
        const float projection =
            normal.x * tangent[0] +
            normal.y * tangent[1] +
            normal.z * tangent[2];
        tangent[0] -= normal.x * projection;
        tangent[1] -= normal.y * projection;
        tangent[2] -= normal.z * projection;
        normalize3(tangent[0], tangent[1], tangent[2]);
        const float tangentLengthSquared =
            tangent[0] * tangent[0] +
            tangent[1] * tangent[1] +
            tangent[2] * tangent[2];
        if (!(tangentLengthSquared > 1.0e-12f) ||
            !std::isfinite(tangentLengthSquared)) {
            // Pick the coordinate axis least aligned with the normal, then
            // cross it into an exactly orthogonal fallback. This remains
            // stable for UV-less geometry near every pole; selecting a fixed
            // X tangent near +Z can leave a projection large enough to break
            // the packed vertex contract.
            const std::array axis =
                std::abs(normal.x) <= std::abs(normal.y) &&
                        std::abs(normal.x) <= std::abs(normal.z)
                    ? std::array{1.0f, 0.0f, 0.0f}
                    : std::abs(normal.y) <= std::abs(normal.z)
                        ? std::array{0.0f, 1.0f, 0.0f}
                        : std::array{0.0f, 0.0f, 1.0f};
            tangent = {
                axis[1] * normal.z - axis[2] * normal.y,
                axis[2] * normal.x - axis[0] * normal.z,
                axis[0] * normal.y - axis[1] * normal.x,
            };
            normalize3(tangent[0], tangent[1], tangent[2]);
        }
        const float crossX =
            normal.y * tangent[2] - normal.z * tangent[1];
        const float crossY =
            normal.z * tangent[0] - normal.x * tangent[2];
        const float crossZ =
            normal.x * tangent[1] - normal.y * tangent[0];
        const float handedness =
            crossX * bitangents[index][0] +
                    crossY * bitangents[index][1] +
                    crossZ * bitangents[index][2] <
                0.0f
            ? -1.0f
            : 1.0f;
        vertex.tangent = {
            tangent[0], tangent[1], tangent[2], 0.0f,
        };
        vertex.normalAndTangentSign.w = handedness;
    }
}

Matrix4 multiply(const Matrix4& left, const Matrix4& right) {
    Matrix4 result{};
    for (std::size_t row = 0u; row < 4u; ++row) {
        for (std::size_t column = 0u; column < 4u; ++column) {
            result.value[row][column] = 0.0;
            for (std::size_t inner = 0u; inner < 4u; ++inner) {
                result.value[row][column] +=
                    left.value[row][inner] *
                    right.value[inner][column];
            }
        }
    }
    return result;
}

Matrix4 nodeTransform(NSDictionary* node) {
    Matrix4 result;
    if (NSArray* matrix = arrayValue(node, @"matrix");
        matrix != nil && matrix.count == 16u) {
        for (std::size_t column = 0u; column < 4u; ++column) {
            for (std::size_t row = 0u; row < 4u; ++row) {
                result.value[row][column] =
                    [matrix[column * 4u + row] doubleValue];
            }
        }
        return result;
    }
    std::array<double, 3u> translation{};
    std::array<double, 4u> rotation{0.0, 0.0, 0.0, 1.0};
    std::array<double, 3u> scale{1.0, 1.0, 1.0};
    if (NSArray* values = arrayValue(node, @"translation");
        values != nil && values.count == 3u) {
        for (std::size_t index = 0u; index < 3u; ++index) {
            translation[index] = [values[index] doubleValue];
        }
    }
    if (NSArray* values = arrayValue(node, @"rotation");
        values != nil && values.count == 4u) {
        for (std::size_t index = 0u; index < 4u; ++index) {
            rotation[index] = [values[index] doubleValue];
        }
    }
    if (NSArray* values = arrayValue(node, @"scale");
        values != nil && values.count == 3u) {
        for (std::size_t index = 0u; index < 3u; ++index) {
            scale[index] = [values[index] doubleValue];
        }
    }
    const double x = rotation[0];
    const double y = rotation[1];
    const double z = rotation[2];
    const double w = rotation[3];
    const std::array<std::array<double, 3u>, 3u> matrix{{
        {
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        },
        {
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        },
        {
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        },
    }};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.value[row][column] =
                matrix[row][column] * scale[column];
        }
        result.value[row][3] = translation[row];
    }
    return result;
}

mr_float4 rotationQuaternion(
    const Matrix4& matrix,
    const double scale
) {
    const double m00 = matrix.value[0][0] / scale;
    const double m11 = matrix.value[1][1] / scale;
    const double m22 = matrix.value[2][2] / scale;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
    const double trace = m00 + m11 + m22;
    if (trace > 0.0) {
        const double s = 2.0 * std::sqrt(trace + 1.0);
        w = 0.25 * s;
        x = (matrix.value[2][1] - matrix.value[1][2]) /
            (scale * s);
        y = (matrix.value[0][2] - matrix.value[2][0]) /
            (scale * s);
        z = (matrix.value[1][0] - matrix.value[0][1]) /
            (scale * s);
    } else if (m00 > m11 && m00 > m22) {
        const double s =
            2.0 * std::sqrt(1.0 + m00 - m11 - m22);
        w = (matrix.value[2][1] - matrix.value[1][2]) /
            (scale * s);
        x = 0.25 * s;
        y = (matrix.value[0][1] + matrix.value[1][0]) /
            (scale * s);
        z = (matrix.value[0][2] + matrix.value[2][0]) /
            (scale * s);
    } else if (m11 > m22) {
        const double s =
            2.0 * std::sqrt(1.0 + m11 - m00 - m22);
        w = (matrix.value[0][2] - matrix.value[2][0]) /
            (scale * s);
        x = (matrix.value[0][1] + matrix.value[1][0]) /
            (scale * s);
        y = 0.25 * s;
        z = (matrix.value[1][2] + matrix.value[2][1]) /
            (scale * s);
    } else {
        const double s =
            2.0 * std::sqrt(1.0 + m22 - m00 - m11);
        w = (matrix.value[1][0] - matrix.value[0][1]) /
            (scale * s);
        x = (matrix.value[0][2] + matrix.value[2][0]) /
            (scale * s);
        y = (matrix.value[1][2] + matrix.value[2][1]) /
            (scale * s);
        z = 0.25 * s;
    }
    const double length = std::sqrt(x * x + y * y + z * z + w * w);
    return {
        static_cast<float>(x / length),
        static_cast<float>(y / length),
        static_cast<float>(z / length),
        static_cast<float>(w / length),
    };
}

bool decomposeUniform(
    const Matrix4& matrix,
    mr_float4& translationAndScale,
    mr_float4& orientation
) {
    std::array<double, 3u> scale{};
    for (std::size_t column = 0u; column < 3u; ++column) {
        scale[column] = std::sqrt(
            matrix.value[0][column] * matrix.value[0][column] +
            matrix.value[1][column] * matrix.value[1][column] +
            matrix.value[2][column] * matrix.value[2][column]
        );
    }
    const double average = (scale[0] + scale[1] + scale[2]) / 3.0;
    if (!(average > 0.0) ||
        std::abs(scale[0] - average) > 1.0e-5 * average ||
        std::abs(scale[1] - average) > 1.0e-5 * average ||
        std::abs(scale[2] - average) > 1.0e-5 * average) {
        return false;
    }
    translationAndScale = {
        static_cast<float>(matrix.value[0][3]),
        static_cast<float>(matrix.value[1][3]),
        static_cast<float>(matrix.value[2][3]),
        static_cast<float>(average),
    };
    orientation = rotationQuaternion(matrix, average);
    return true;
}

bool decomposeRigidResidual(
    const Matrix4& matrix,
    mr_float4& translationAndScale,
    mr_float4& orientation,
    Matrix4& residual
) {
    simd_double3x3 linear{
        {
            {
                matrix.value[0][0],
                matrix.value[1][0],
                matrix.value[2][0],
            },
            {
                matrix.value[0][1],
                matrix.value[1][1],
                matrix.value[2][1],
            },
            {
                matrix.value[0][2],
                matrix.value[1][2],
                matrix.value[2][2],
            },
        }
    };
    const double determinant = simd_determinant(linear);
    const double scale = std::cbrt(std::abs(determinant));
    if (!(scale > 1.0e-12) || !std::isfinite(scale)) {
        return false;
    }

    simd_double3x3 rotation = linear;
    for (std::size_t column = 0u; column < 3u; ++column) {
        rotation.columns[column] /= scale;
    }
    for (std::uint32_t iteration = 0u;
         iteration < 12u;
         ++iteration) {
        const simd_double3x3 inverseTranspose =
            simd_transpose(simd_inverse(rotation));
        simd_double3x3 next = rotation;
        for (std::size_t column = 0u;
             column < 3u;
             ++column) {
            next.columns[column] =
                0.5 * (
                    rotation.columns[column] +
                    inverseTranspose.columns[column]
                );
        }
        const double delta =
            simd_length(next.columns[0] - rotation.columns[0]) +
            simd_length(next.columns[1] - rotation.columns[1]) +
            simd_length(next.columns[2] - rotation.columns[2]);
        rotation = next;
        if (delta < 1.0e-12) {
            break;
        }
    }
    if (simd_determinant(rotation) < 0.0) {
        rotation.columns[0] = -rotation.columns[0];
    }
    simd_double3x3 scaledRotation = rotation;
    for (std::size_t column = 0u; column < 3u; ++column) {
        scaledRotation.columns[column] *= scale;
    }
    const simd_double3x3 baked = simd_mul(
        simd_inverse(scaledRotation),
        linear
    );
    if (!std::isfinite(simd_determinant(baked))) {
        return false;
    }

    Matrix4 rigid{};
    Matrix4 bakedMatrix{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            rigid.value[row][column] =
                rotation.columns[column][row];
            bakedMatrix.value[row][column] =
                baked.columns[column][row];
        }
        rigid.value[row][3] = matrix.value[row][3];
        bakedMatrix.value[row][3] = 0.0;
    }
    translationAndScale = {
        static_cast<float>(matrix.value[0][3]),
        static_cast<float>(matrix.value[1][3]),
        static_cast<float>(matrix.value[2][3]),
        static_cast<float>(scale),
    };
    orientation = rotationQuaternion(rigid, 1.0);
    residual = bakedMatrix;
    return std::isfinite(translationAndScale.x) &&
        std::isfinite(translationAndScale.y) &&
        std::isfinite(translationAndScale.z) &&
        std::isfinite(translationAndScale.w) &&
        std::isfinite(orientation.x) &&
        std::isfinite(orientation.y) &&
        std::isfinite(orientation.z) &&
        std::isfinite(orientation.w);
}

std::uint32_t textureForMaterial(
    const GltfDocument& document,
    const std::uint32_t textureIndex,
    const bool srgb,
    const VisualAssetCookOptions& options,
    std::map<std::pair<std::uint32_t, bool>, std::uint32_t>& textureMap,
    VisualAssetPackV2& pack,
    std::string& message
) {
    if (textureIndex == MR_INVALID_INDEX) {
        return MR_INVALID_INDEX;
    }
    const auto key = std::pair{textureIndex, srgb};
    if (const auto found = textureMap.find(key);
        found != textureMap.end()) {
        return found->second;
    }
    NSArray* textures = arrayValue(document.root, @"textures");
    NSArray* images = arrayValue(document.root, @"images");
    if (textures == nil || images == nil ||
        textureIndex >= textures.count ||
        ![textures[textureIndex] isKindOfClass:NSDictionary.class]) {
        message = "glTF material references an invalid texture";
        return MR_INVALID_INDEX;
    }
    NSDictionary* texture =
        static_cast<NSDictionary*>(textures[textureIndex]);
    const std::uint32_t imageIndex =
        uintValue(texture, @"source", MR_INVALID_INDEX);
    if (imageIndex >= images.count ||
        ![images[imageIndex] isKindOfClass:NSDictionary.class]) {
        message = "glTF texture references an invalid image";
        return MR_INVALID_INDEX;
    }
    NSDictionary* image =
        static_cast<NSDictionary*>(images[imageIndex]);
    const std::vector<std::uint8_t> encoded =
        imagePayload(document, image);
    VisualTextureImageV2 decoded;
    const std::string name = utf8(stringValue(image, @"name"));
    if (!decodeTexture(
            encoded,
            srgb,
            options.generateMipmaps,
            name.empty()
                ? "texture_" + std::to_string(textureIndex)
                : name,
            decoded
        )) {
        message = "glTF texture image could not be decoded";
        return MR_INVALID_INDEX;
    }
    const std::uint32_t result =
        static_cast<std::uint32_t>(pack.textures.size());
    pack.textures.push_back(std::move(decoded));
    textureMap.emplace(key, result);
    return result;
}

std::uint32_t textureReference(
    NSDictionary* record
) {
    return record == nil
        ? MR_INVALID_INDEX
        : uintValue(record, @"index", MR_INVALID_INDEX);
}

std::uint32_t textureBindingForMaterial(
    const GltfDocument& document,
    NSDictionary* record,
    const bool srgb,
    const VisualAssetCookOptions& options,
    std::map<std::pair<std::uint32_t, bool>, std::uint32_t>&
        textureMap,
    VisualAssetPackV2& pack,
    std::string& message
) {
    const std::uint32_t source = textureReference(record);
    if (source == MR_INVALID_INDEX) {
        return MR_INVALID_INDEX;
    }
    const std::uint32_t textureIndex = textureForMaterial(
        document,
        source,
        srgb,
        options,
        textureMap,
        pack,
        message
    );
    if (textureIndex == MR_INVALID_INDEX) {
        if (message.empty()) {
            message = "glTF material texture is invalid";
        }
        return MR_INVALID_INDEX;
    }
    NSArray* textures = arrayValue(document.root, @"textures");
    NSDictionary* texture =
        textures != nil && source < textures.count &&
            [textures[source] isKindOfClass:NSDictionary.class]
        ? static_cast<NSDictionary*>(textures[source])
        : nil;
    NSArray* samplers = arrayValue(document.root, @"samplers");
    const std::uint32_t samplerSource =
        uintValue(texture, @"sampler", MR_INVALID_INDEX);
    NSDictionary* sampler =
        samplers != nil && samplerSource < samplers.count &&
            [samplers[samplerSource]
                isKindOfClass:NSDictionary.class]
        ? static_cast<NSDictionary*>(samplers[samplerSource])
        : nil;
    const std::uint32_t wrapS =
        uintValue(sampler, @"wrapS", 10497u);
    const std::uint32_t wrapT =
        uintValue(sampler, @"wrapT", 10497u);
    const std::uint32_t minFilter =
        uintValue(sampler, @"minFilter", 9987u);
    const std::uint32_t magFilter =
        uintValue(sampler, @"magFilter", 9729u);
    const bool minNearest =
        minFilter == 9728u ||
        minFilter == 9984u ||
        minFilter == 9986u;
    const bool magNearest = magFilter == 9728u;
    const std::uint32_t mipClass =
        minFilter == 9728u || minFilter == 9729u
        ? 0u
        : minFilter == 9984u || minFilter == 9985u
            ? 1u
            : 2u;
    const auto addressClass = [](
        const std::uint32_t wrap
    ) {
        return wrap == 33071u
            ? 1u
            : wrap == 33648u
                ? 2u
                : 0u;
    };
    const std::uint32_t samplerIndex =
        3u * addressClass(wrapS) +
        addressClass(wrapT) +
        (minNearest ? 9u : 0u) +
        (magNearest ? 18u : 0u) +
        36u * mipClass;
    std::uint32_t texcoordSet =
        uintValue(record, @"texCoord", 0u);
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float rotation = 0.0f;
    NSDictionary* extensions =
        dictionaryValue(record, @"extensions");
    NSDictionary* transform = dictionaryValue(
        extensions,
        @"KHR_texture_transform"
    );
    if (transform != nil) {
        texcoordSet =
            uintValue(transform, @"texCoord", texcoordSet);
        if (NSArray* offset = arrayValue(transform, @"offset");
            offset != nil && offset.count == 2u) {
            offsetX = [offset[0] floatValue];
            offsetY = [offset[1] floatValue];
        }
        if (NSArray* scale = arrayValue(transform, @"scale");
            scale != nil && scale.count == 2u) {
            scaleX = [scale[0] floatValue];
            scaleY = [scale[1] floatValue];
        }
        rotation =
            static_cast<float>(
                doubleValue(transform, @"rotation", 0.0)
            );
    }
    const float sine = std::sin(rotation);
    const float cosine = std::cos(rotation);
    MRVisualTextureBindingGPUV2 binding{};
    binding.resource = {
        textureIndex,
        samplerIndex,
        std::min(texcoordSet, 1u),
        (wrapS == 33071u ? MR_VISUAL_TEXTURE_CLAMP_U : 0u) |
            (wrapT == 33071u
                 ? MR_VISUAL_TEXTURE_CLAMP_V
                 : 0u),
    };
    binding.uvTransform0 = {
        cosine * scaleX,
        -sine * scaleY,
        offsetX,
        0.0f,
    };
    binding.uvTransform1 = {
        sine * scaleX,
        cosine * scaleY,
        offsetY,
        0.0f,
    };
    if (pack.textureBindings.size() ==
        std::numeric_limits<std::uint32_t>::max()) {
        message = "visual texture binding count exceeds uint32";
        return MR_INVALID_INDEX;
    }
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            pack.textureBindings.size()
        );
    pack.textureBindings.push_back(binding);
    return result;
}

bool importMaterials(
    const GltfDocument& document,
    const VisualAssetCookOptions& options,
    VisualAssetPackV2& pack,
    std::string& message
) {
    NSArray* materials = arrayValue(document.root, @"materials");
    std::map<std::pair<std::uint32_t, bool>, std::uint32_t>
        textureMap;
    const NSUInteger count = materials == nil ? 0u : materials.count;
    pack.materials.reserve(std::max<NSUInteger>(1u, count));
    for (NSUInteger index = 0u; index < std::max<NSUInteger>(1u, count);
         ++index) {
        NSDictionary* material =
            materials != nil && index < materials.count &&
                [materials[index] isKindOfClass:NSDictionary.class]
            ? static_cast<NSDictionary*>(materials[index])
            : @{};
        NSDictionary* pbr =
            dictionaryValue(material, @"pbrMetallicRoughness");
        MRVisualMaterialGPUV2 gpu{};
        gpu.baseColorAndOpacity = {1.0f, 1.0f, 1.0f, 1.0f};
        gpu.emissionAndStrength = {0.0f, 0.0f, 0.0f, 1.0f};
        gpu.surface = {
            static_cast<float>(doubleValue(pbr, @"roughnessFactor", 1.0)),
            static_cast<float>(doubleValue(pbr, @"metallicFactor", 1.0)),
            static_cast<float>(
                doubleValue(
                    dictionaryValue(material, @"normalTexture"),
                    @"scale",
                    1.0
                )
            ),
            static_cast<float>(
                doubleValue(
                    dictionaryValue(material, @"occlusionTexture"),
                    @"strength",
                    1.0
                )
            ),
        };
        gpu.coatingAndAlphaCutoff = {
            0.0f,
            0.0f,
            1.0f,
            static_cast<float>(
                doubleValue(material, @"alphaCutoff", 0.5)
            ),
        };
        if (NSArray* base = arrayValue(pbr, @"baseColorFactor");
            base != nil && base.count == 4u) {
            gpu.baseColorAndOpacity = {
                [base[0] floatValue],
                [base[1] floatValue],
                [base[2] floatValue],
                [base[3] floatValue],
            };
        }
        if (NSArray* emissive = arrayValue(material, @"emissiveFactor");
            emissive != nil && emissive.count == 3u) {
            gpu.emissionAndStrength = {
                [emissive[0] floatValue],
                [emissive[1] floatValue],
                [emissive[2] floatValue],
                1.0f,
            };
        }
        NSDictionary* extensions =
            dictionaryValue(material, @"extensions");
        NSDictionary* emissiveStrength = dictionaryValue(
            extensions,
            @"KHR_materials_emissive_strength"
        );
        gpu.emissionAndStrength.w = static_cast<float>(
            doubleValue(
                emissiveStrength,
                @"emissiveStrength",
                1.0
            )
        );
        NSDictionary* clearcoat = dictionaryValue(
            extensions,
            @"KHR_materials_clearcoat"
        );
        gpu.coatingAndAlphaCutoff.x = static_cast<float>(
            doubleValue(clearcoat, @"clearcoatFactor", 0.0)
        );
        gpu.coatingAndAlphaCutoff.y = static_cast<float>(
            doubleValue(
                clearcoat,
                @"clearcoatRoughnessFactor",
                0.0
            )
        );
        NSString* alphaMode = stringValue(material, @"alphaMode");
        const std::uint32_t mode =
            [alphaMode isEqualToString:@"BLEND"]
            ? MR_VISUAL_ALPHA_BLEND
            : [alphaMode isEqualToString:@"MASK"]
                ? MR_VISUAL_ALPHA_MASK
                : MR_VISUAL_ALPHA_OPAQUE;
        NSDictionary* unlit =
            dictionaryValue(extensions, @"KHR_materials_unlit");
        gpu.flags = {
            mode,
            (boolValue(material, @"doubleSided", false)
                 ? MR_VISUAL_MATERIAL_DOUBLE_SIDED
                 : 0u) |
                (unlit != nil ? MR_VISUAL_MATERIAL_UNLIT : 0u),
            0u,
            static_cast<std::uint32_t>(index + 1u),
        };
        const auto texture = [&](
            NSDictionary* record,
            const bool isSrgb
        ) -> std::uint32_t {
            return textureBindingForMaterial(
                document,
                record,
                isSrgb,
                options,
                textureMap,
                pack,
                message
            );
        };
        gpu.textureIndices0 = {
            texture(
                dictionaryValue(pbr, @"baseColorTexture"),
                true
            ),
            texture(
                dictionaryValue(
                    pbr,
                    @"metallicRoughnessTexture"
                ),
                false
            ),
            texture(
                dictionaryValue(material, @"normalTexture"),
                false
            ),
            texture(
                dictionaryValue(material, @"occlusionTexture"),
                false
            ),
        };
        gpu.textureIndices1 = {
            texture(
                dictionaryValue(material, @"emissiveTexture"),
                true
            ),
            texture(
                dictionaryValue(clearcoat, @"clearcoatTexture"),
                false
            ),
            texture(
                dictionaryValue(
                    clearcoat,
                    @"clearcoatRoughnessTexture"
                ),
                false
            ),
            MR_INVALID_INDEX,
        };
        gpu.reserved = {
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
        };
        if (!message.empty()) {
            return false;
        }
        pack.materials.push_back(gpu);
    }
    return true;
}

bool importMeshes(
    const GltfDocument& document,
    const VisualAssetCookOptions& options,
    VisualAssetPackV2& pack,
    std::vector<ImportedMesh>& imported,
    std::string& message
) {
    NSArray* meshes = arrayValue(document.root, @"meshes");
    if (meshes == nil || meshes.count == 0u) {
        message = "glTF has no meshes";
        return false;
    }
    imported.resize(meshes.count);
    for (NSUInteger meshIndex = 0u;
         meshIndex < meshes.count;
         ++meshIndex) {
        if (![meshes[meshIndex] isKindOfClass:NSDictionary.class]) {
            message = "glTF mesh record is invalid";
            return false;
        }
        NSDictionary* mesh =
            static_cast<NSDictionary*>(meshes[meshIndex]);
        NSArray* primitives = arrayValue(mesh, @"primitives");
        if (primitives == nil || primitives.count == 0u) {
            message = "glTF mesh has no primitives";
            return false;
        }
        for (id primitiveObject in primitives) {
            if (![primitiveObject isKindOfClass:NSDictionary.class]) {
                message = "glTF primitive record is invalid";
                return false;
            }
            NSDictionary* primitive =
                static_cast<NSDictionary*>(primitiveObject);
            if (uintValue(primitive, @"mode", 4u) != 4u) {
                message =
                    "only glTF triangle-list primitives are supported";
                return false;
            }
            NSDictionary* attributes =
                dictionaryValue(primitive, @"attributes");
            const std::uint32_t positionAccessor =
                uintValue(
                    attributes,
                    @"POSITION",
                    MR_INVALID_INDEX
                );
            std::vector<float> positions;
            if (positionAccessor == MR_INVALID_INDEX ||
                !readFloatAccessor(
                    document,
                    positionAccessor,
                    3u,
                    positions,
                    message
                )) {
                message = message.empty()
                    ? "glTF primitive has no positions"
                    : message;
                return false;
            }
            const std::size_t vertexCount = positions.size() / 3u;
            if (pack.vertices.size() >
                    std::numeric_limits<std::uint32_t>::max() -
                        vertexCount) {
                message = "glTF vertex count exceeds uint32";
                return false;
            }
            std::vector<float> normals;
            std::vector<float> tangents;
            std::vector<float> uv0;
            std::vector<float> uv1;
            std::vector<float> colors;
            const auto optionalAttribute = [&](
                NSString* name,
                const std::size_t components,
                std::vector<float>& values
            ) -> bool {
                const std::uint32_t accessor =
                    uintValue(attributes, name, MR_INVALID_INDEX);
                return accessor == MR_INVALID_INDEX ||
                    (readFloatAccessor(
                         document,
                         accessor,
                         components,
                         values,
                         message
                     ) &&
                     values.size() / components == vertexCount);
            };
            if (!optionalAttribute(@"NORMAL", 3u, normals) ||
                !optionalAttribute(@"TANGENT", 4u, tangents) ||
                !optionalAttribute(@"TEXCOORD_0", 2u, uv0) ||
                !optionalAttribute(@"TEXCOORD_1", 2u, uv1)) {
                message = message.empty()
                    ? "glTF vertex attribute count does not match"
                    : message;
                return false;
            }
            const std::uint32_t colorAccessor =
                uintValue(attributes, @"COLOR_0", MR_INVALID_INDEX);
            std::size_t colorComponents = 0u;
            if (colorAccessor != MR_INVALID_INDEX) {
                const std::uint8_t* first = nullptr;
                std::size_t count = 0u;
                std::size_t stride = 0u;
                std::uint32_t type = 0u;
                bool normalized = false;
                if (!accessorBytes(
                        document,
                        colorAccessor,
                        first,
                        count,
                        colorComponents,
                        stride,
                        type,
                        normalized,
                        message
                    ) ||
                    (colorComponents != 3u &&
                     colorComponents != 4u) ||
                    !readFloatAccessor(
                        document,
                        colorAccessor,
                        colorComponents,
                        colors,
                        message
                    ) ||
                    count != vertexCount) {
                    message = "glTF vertex colors are invalid";
                    return false;
                }
            }
            const std::uint32_t vertexOffset =
                static_cast<std::uint32_t>(pack.vertices.size());
            for (std::size_t vertex = 0u;
                 vertex < vertexCount;
                 ++vertex) {
                MRVisualVertexGPUV2 gpu{};
                gpu.position = {
                    positions[3u * vertex],
                    positions[3u * vertex + 1u],
                    positions[3u * vertex + 2u],
                    1.0f,
                };
                gpu.normalAndTangentSign = normals.empty()
                    ? mr_float4{0.0f, 0.0f, 0.0f, 1.0f}
                    : mr_float4{
                          normals[3u * vertex],
                          normals[3u * vertex + 1u],
                          normals[3u * vertex + 2u],
                          tangents.empty()
                              ? 1.0f
                              : tangents[4u * vertex + 3u],
                      };
                gpu.tangent = tangents.empty()
                    ? mr_float4{}
                    : mr_float4{
                          tangents[4u * vertex],
                          tangents[4u * vertex + 1u],
                          tangents[4u * vertex + 2u],
                          0.0f,
                      };
                gpu.texcoord01 = {
                    uv0.empty() ? 0.0f : uv0[2u * vertex],
                    uv0.empty() ? 0.0f : uv0[2u * vertex + 1u],
                    uv1.empty() ? 0.0f : uv1[2u * vertex],
                    uv1.empty() ? 0.0f : uv1[2u * vertex + 1u],
                };
                gpu.color = {
                    colors.empty() ? 1.0f
                                   : colors[colorComponents * vertex],
                    colors.empty()
                        ? 1.0f
                        : colors[colorComponents * vertex + 1u],
                    colors.empty()
                        ? 1.0f
                        : colors[colorComponents * vertex + 2u],
                    colors.empty() || colorComponents == 3u
                        ? 1.0f
                        : colors[colorComponents * vertex + 3u],
                };
                pack.vertices.push_back(gpu);
            }
            std::vector<std::uint32_t> localIndices;
            const std::uint32_t indexAccessor =
                uintValue(primitive, @"indices", MR_INVALID_INDEX);
            if (indexAccessor == MR_INVALID_INDEX) {
                localIndices.resize(vertexCount);
                std::ranges::iota_view<std::uint32_t, std::uint32_t>
                    range{0u, static_cast<std::uint32_t>(vertexCount)};
                std::ranges::copy(range, localIndices.begin());
            } else if (!readIndexAccessor(
                           document,
                           indexAccessor,
                           localIndices,
                           message
                       )) {
                return false;
            }
            if (localIndices.empty() ||
                localIndices.size() % 3u != 0u ||
                std::ranges::any_of(
                    localIndices,
                    [vertexCount](const std::uint32_t index) {
                        return index >= vertexCount;
                    }
                )) {
                message = "glTF triangle indices are invalid";
                return false;
            }
            const std::uint32_t firstIndex =
                static_cast<std::uint32_t>(pack.indices.size());
            for (const std::uint32_t index : localIndices) {
                pack.indices.push_back(vertexOffset + index);
            }
            auto vertexSpan = std::span{
                pack.vertices.data() + vertexOffset,
                vertexCount,
            };
            if (normals.empty()) {
                if (!options.generateNormals) {
                    message =
                        "glTF normals are absent and generation is disabled";
                    return false;
                }
                generateNormals(vertexSpan, localIndices);
            } else {
                for (MRVisualVertexGPUV2& vertex : vertexSpan) {
                    normalize3(
                        vertex.normalAndTangentSign.x,
                        vertex.normalAndTangentSign.y,
                        vertex.normalAndTangentSign.z
                    );
                }
            }
            if (tangents.empty()) {
                if (!options.generateTangents) {
                    message =
                        "glTF tangents are absent and generation is disabled";
                    return false;
                }
                generateTangents(vertexSpan, localIndices);
            }
            PrimitiveTemplate result;
            result.firstIndex = firstIndex;
            result.indexCount =
                static_cast<std::uint32_t>(localIndices.size());
            result.materialIndex = uintValue(primitive, @"material");
            if (result.materialIndex >= pack.materials.size()) {
                message = "glTF primitive material is out of range";
                return false;
            }
            result.boundsMinimum = {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                1.0f,
            };
            result.boundsMaximum = {
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                1.0f,
            };
            for (const MRVisualVertexGPUV2& vertex : vertexSpan) {
                result.boundsMinimum.x =
                    std::min(result.boundsMinimum.x, vertex.position.x);
                result.boundsMinimum.y =
                    std::min(result.boundsMinimum.y, vertex.position.y);
                result.boundsMinimum.z =
                    std::min(result.boundsMinimum.z, vertex.position.z);
                result.boundsMaximum.x =
                    std::max(result.boundsMaximum.x, vertex.position.x);
                result.boundsMaximum.y =
                    std::max(result.boundsMaximum.y, vertex.position.y);
                result.boundsMaximum.z =
                    std::max(result.boundsMaximum.z, vertex.position.z);
            }
            imported[meshIndex].primitives.push_back(result);
        }
    }
    return true;
}

bool appendNode(
    const GltfDocument& document,
    const std::vector<ImportedMesh>& meshes,
    const VisualAssetCookOptions& options,
    const std::uint32_t nodeIndex,
    const Matrix4& parent,
    std::set<std::uint32_t>& active,
    VisualAssetPackV2& pack,
    std::string& message
) {
    NSArray* nodes = arrayValue(document.root, @"nodes");
    if (nodes == nil || nodeIndex >= nodes.count ||
        ![nodes[nodeIndex] isKindOfClass:NSDictionary.class] ||
        !active.insert(nodeIndex).second) {
        message = "glTF node hierarchy is invalid or cyclic";
        return false;
    }
    NSDictionary* node =
        static_cast<NSDictionary*>(nodes[nodeIndex]);
    const Matrix4 world = multiply(parent, nodeTransform(node));
    const std::uint32_t meshIndex =
        uintValue(node, @"mesh", MR_INVALID_INDEX);
    if (meshIndex != MR_INVALID_INDEX) {
        if (meshIndex >= meshes.size()) {
            message = "glTF node mesh is out of range";
            return false;
        }
        MRVisualInstanceGPUV2 instance{};
        if (!decomposeUniform(
                world,
                instance.translationAndScale,
                instance.orientation
            )) {
            message =
                "glTF node has non-uniform scale; bake it before cooking";
            return false;
        }
        const std::string nodeName =
            utf8(stringValue(node, @"name"));
        AuthoredVisualBodyBinding body;
        if (!resolveVisualBodyBinding(
                options,
                nodeName,
                body,
                message
            )) {
            return false;
        }
        instance.binding = {
            0u,
            body.body,
            body.kind,
            MR_VISUAL_INSTANCE_CASTS_SHADOW |
                MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
        };
        instance.identity = {
            0u,
            0u,
            body.body,
            static_cast<std::uint32_t>(pack.instances.size() + 1u),
        };
        instance.geometry = {
            static_cast<std::uint32_t>(pack.primitives.size()),
            static_cast<std::uint32_t>(
                meshes[meshIndex].primitives.size()
            ),
            0u,
            0u,
        };
        const std::uint32_t instanceIndex =
            static_cast<std::uint32_t>(pack.instances.size());
        pack.instances.push_back(instance);
        for (const PrimitiveTemplate& source :
             meshes[meshIndex].primitives) {
            MRVisualPrimitiveGPUV2 primitive{};
            primitive.geometry = {
                source.firstIndex,
                source.indexCount,
                source.materialIndex,
                instanceIndex,
            };
            primitive.identity = {
                0u,
                0u,
                instance.identity.z,
                static_cast<std::uint32_t>(
                    pack.primitives.size() + 1u
                ),
            };
            primitive.boundsMinimum = source.boundsMinimum;
            primitive.boundsMaximum = source.boundsMaximum;
            pack.primitives.push_back(primitive);
        }
        VisualSymbolicBindingV2 binding;
        binding.node = nodeName.empty()
            ? "node_" + std::to_string(nodeIndex)
            : nodeName;
        binding.instanceIndex = instanceIndex;
        if (body.kind != MR_VISUAL_BINDING_ASSET) {
            binding.link = nodeName;
            binding.bodyIndex = body.body;
            binding.binding =
                static_cast<MRVisualBindingKind>(body.kind);
        }
        pack.symbolicBindings.push_back(std::move(binding));
    }
    if (NSArray* children = arrayValue(node, @"children");
        children != nil) {
        for (NSNumber* child in children) {
            if (![child isKindOfClass:NSNumber.class] ||
                !appendNode(
                    document,
                    meshes,
                    options,
                    child.unsignedIntValue,
                    world,
                    active,
                    pack,
                    message
                )) {
                return false;
            }
        }
    }
    active.erase(nodeIndex);
    return true;
}

bool importNodes(
    const GltfDocument& document,
    const std::vector<ImportedMesh>& meshes,
    const VisualAssetCookOptions& options,
    VisualAssetPackV2& pack,
    std::string& message
) {
    NSArray* scenes = arrayValue(document.root, @"scenes");
    NSArray* nodes = arrayValue(document.root, @"nodes");
    if (nodes == nil || nodes.count == 0u) {
        message = "glTF has no nodes";
        return false;
    }
    NSArray* roots = nil;
    if (scenes != nil && scenes.count != 0u) {
        const std::uint32_t sceneIndex =
            uintValue(document.root, @"scene", 0u);
        if (sceneIndex >= scenes.count ||
            ![scenes[sceneIndex] isKindOfClass:NSDictionary.class]) {
            message = "glTF default scene is invalid";
            return false;
        }
        roots = arrayValue(
            static_cast<NSDictionary*>(scenes[sceneIndex]),
            @"nodes"
        );
    }
    NSMutableArray<NSNumber*>* synthesized = nil;
    if (roots == nil) {
        std::set<std::uint32_t> children;
        for (id object in nodes) {
            if ([object isKindOfClass:NSDictionary.class]) {
                NSArray* nodeChildren = arrayValue(
                    static_cast<NSDictionary*>(object),
                    @"children"
                );
                if (nodeChildren != nil) {
                    for (NSNumber* child in nodeChildren) {
                        children.insert(child.unsignedIntValue);
                    }
                }
            }
        }
        synthesized = [NSMutableArray array];
        for (std::uint32_t index = 0u; index < nodes.count; ++index) {
            if (!children.contains(index)) {
                [synthesized addObject:@(index)];
            }
        }
        roots = synthesized;
    }
    std::set<std::uint32_t> active;
    const Matrix4 identity;
    for (NSNumber* root in roots) {
        if (![root isKindOfClass:NSNumber.class] ||
            !appendNode(
                document,
                meshes,
                options,
                root.unsignedIntValue,
                identity,
                active,
                pack,
                message
            )) {
            return false;
        }
    }
    if (pack.instances.empty()) {
        message = "glTF scene contains no mesh instances";
        return false;
    }
    return true;
}

VisualAssetCookDiagnostics cookGltf(
    const std::filesystem::path& source,
    const std::span<const std::uint8_t> sourceBytes,
    VisualAssetPackV2& output,
    const VisualAssetCookOptions& options
) {
    VisualAssetCookDiagnostics diagnostics;
    GltfDocument document;
    std::string message;
    if (!parseGltf(source, sourceBytes, document, message)) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            std::move(message)
        );
    }
    VisualAssetPackV2 candidate;
    candidate.id = options.id.empty()
        ? source.stem().string()
        : options.id;
    candidate.sourceUri = source.string();
    candidate.sourceContentHash = sha256(sourceBytes);
    candidate.license = options.license;
    candidate.preprocessingProvenance =
        options.preprocessingProvenance +
        ";input=gltf2;normals=" +
        (options.generateNormals ? "generate-if-missing" : "authored") +
        ";tangents=" +
        (options.generateTangents ? "generate-if-missing" : "authored") +
        ";mips=" +
        (options.generateMipmaps ? "linear-box-v1" : "none");
    if (!importMaterials(document, options, candidate, message)) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidMaterial,
            std::move(message)
        );
    }
    std::vector<ImportedMesh> meshes;
    if (!importMeshes(
            document,
            options,
            candidate,
            meshes,
            message
        )) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidGeometry,
            std::move(message)
        );
    }
    if (!importNodes(
            document,
            meshes,
            options,
            candidate,
            message
        )) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidBinding,
            std::move(message)
        );
    }
    candidate.contentHash =
        computeVisualAssetPackContentHash(candidate);
    if (!candidate.valid(&message)) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            std::move(message)
        );
    }
    diagnostics.vertexCount =
        static_cast<std::uint32_t>(candidate.vertices.size());
    diagnostics.indexCount =
        static_cast<std::uint32_t>(candidate.indices.size());
    diagnostics.primitiveCount =
        static_cast<std::uint32_t>(candidate.primitives.size());
    diagnostics.instanceCount =
        static_cast<std::uint32_t>(candidate.instances.size());
    diagnostics.materialCount =
        static_cast<std::uint32_t>(candidate.materials.size());
    diagnostics.textureCount =
        static_cast<std::uint32_t>(candidate.textures.size());
    diagnostics.sourceHash = candidate.sourceContentHash;
    diagnostics.packHash = candidate.contentHash;
    output = std::move(candidate);
    return diagnostics;
}

std::optional<std::string> xmlProperty(
    xmlNode* node,
    const char* name
) {
    xmlChar* value = xmlGetProp(
        node,
        reinterpret_cast<const xmlChar*>(name)
    );
    if (value == nullptr) {
        return std::nullopt;
    }
    std::string result{reinterpret_cast<const char*>(value)};
    xmlFree(value);
    return result;
}

xmlNode* xmlChild(xmlNode* parent, const char* name) {
    for (xmlNode* child = parent == nullptr ? nullptr : parent->children;
         child != nullptr;
         child = child->next) {
        if (child->type == XML_ELEMENT_NODE &&
            std::strcmp(
                reinterpret_cast<const char*>(child->name),
                name
            ) == 0) {
            return child;
        }
    }
    return nullptr;
}

std::vector<xmlNode*> xmlChildren(
    xmlNode* parent,
    const char* name
) {
    std::vector<xmlNode*> result;
    for (xmlNode* child = parent == nullptr ? nullptr : parent->children;
         child != nullptr;
         child = child->next) {
        if (child->type == XML_ELEMENT_NODE &&
            std::strcmp(
                reinterpret_cast<const char*>(child->name),
                name
            ) == 0) {
            result.push_back(child);
        }
    }
    return result;
}

bool parseUrdfVector3(
    const std::optional<std::string>& text,
    const std::array<double, 3u>& fallback,
    std::array<double, 3u>& output
) {
    if (!text.has_value()) {
        output = fallback;
        return true;
    }
    std::istringstream stream{*text};
    return bool(stream >> output[0] >> output[1] >> output[2]) &&
        (stream >> std::ws).eof() &&
        std::ranges::all_of(output, [](const double value) {
            return std::isfinite(value);
        });
}

bool urdfVisualTransform(
    xmlNode* visual,
    xmlNode* mesh,
    Matrix4& output
) {
    std::array<double, 3u> xyz{};
    std::array<double, 3u> rpy{};
    std::array<double, 3u> scale{};
    xmlNode* origin = xmlChild(visual, "origin");
    if (!parseUrdfVector3(
            origin == nullptr
                ? std::nullopt
                : xmlProperty(origin, "xyz"),
            {0.0, 0.0, 0.0},
            xyz
        ) ||
        !parseUrdfVector3(
            origin == nullptr
                ? std::nullopt
                : xmlProperty(origin, "rpy"),
            {0.0, 0.0, 0.0},
            rpy
        ) ||
        !parseUrdfVector3(
            xmlProperty(mesh, "scale"),
            {1.0, 1.0, 1.0},
            scale
        )) {
        return false;
    }
    const double cr = std::cos(rpy[0]);
    const double sr = std::sin(rpy[0]);
    const double cp = std::cos(rpy[1]);
    const double sp = std::sin(rpy[1]);
    const double cy = std::cos(rpy[2]);
    const double sy = std::sin(rpy[2]);
    output = {};
    output.value[0][0] = cy * cp * scale[0];
    output.value[0][1] =
        (cy * sp * sr - sy * cr) * scale[1];
    output.value[0][2] =
        (cy * sp * cr + sy * sr) * scale[2];
    output.value[1][0] = sy * cp * scale[0];
    output.value[1][1] =
        (sy * sp * sr + cy * cr) * scale[1];
    output.value[1][2] =
        (sy * sp * cr - cy * sr) * scale[2];
    output.value[2][0] = -sp * scale[0];
    output.value[2][1] = cp * sr * scale[1];
    output.value[2][2] = cp * cr * scale[2];
    output.value[0][3] = xyz[0];
    output.value[1][3] = xyz[1];
    output.value[2][3] = xyz[2];
    return true;
}

Matrix4 visualInstanceTransform(
    const MRVisualInstanceGPUV2& instance
) {
    const double x = instance.orientation.x;
    const double y = instance.orientation.y;
    const double z = instance.orientation.z;
    const double w = instance.orientation.w;
    const double scale = instance.translationAndScale.w;
    Matrix4 result{};
    result.value[0][0] =
        scale * (1.0 - 2.0 * (y * y + z * z));
    result.value[0][1] = scale * 2.0 * (x * y - z * w);
    result.value[0][2] = scale * 2.0 * (x * z + y * w);
    result.value[1][0] = scale * 2.0 * (x * y + z * w);
    result.value[1][1] =
        scale * (1.0 - 2.0 * (x * x + z * z));
    result.value[1][2] = scale * 2.0 * (y * z - x * w);
    result.value[2][0] = scale * 2.0 * (x * z - y * w);
    result.value[2][1] = scale * 2.0 * (y * z + x * w);
    result.value[2][2] =
        scale * (1.0 - 2.0 * (x * x + y * y));
    result.value[0][3] = instance.translationAndScale.x;
    result.value[1][3] = instance.translationAndScale.y;
    result.value[2][3] = instance.translationAndScale.z;
    return result;
}

struct UsdStageMetadata {
    double metersPerUnit = 0.01;
    char upAxis = 'Y';
    std::string defaultPrim;
};

bool readUsdStageMetadata(
    const std::filesystem::path& source,
    UsdStageMetadata& output,
    std::string& message
) {
    NSTask* task = [[NSTask alloc] init];
    task.executableURL =
        [NSURL fileURLWithPath:@"/usr/bin/usdcat"];
    task.arguments = @[
        @"--layerMetadata",
        @(source.string().c_str()),
    ];
    NSPipe* pipe = [NSPipe pipe];
    task.standardOutput = pipe;
    task.standardError = pipe;
    NSError* error = nil;
    if (![task launchAndReturnError:&error]) {
        message =
            "Apple OpenUSD metadata query could not launch: " +
            utf8(error.localizedDescription);
        return false;
    }
    NSData* bytes =
        [pipe.fileHandleForReading readDataToEndOfFile];
    [task waitUntilExit];
    if (task.terminationStatus != 0 || bytes.length == 0u) {
        message =
            "Apple OpenUSD could not read stage metadata";
        return false;
    }
    NSString* text = [[NSString alloc]
        initWithData:bytes
            encoding:NSUTF8StringEncoding];
    const std::string metadata = utf8(text);
    std::smatch match;
    if (std::regex_search(
            metadata,
            match,
            std::regex{
                R"(metersPerUnit\s*=\s*([0-9eE+\-.]+))"
            }
        )) {
        try {
            output.metersPerUnit = std::stod(match[1].str());
        } catch (...) {
            message = "USD metersPerUnit metadata is malformed";
            return false;
        }
    }
    if (std::regex_search(
            metadata,
            match,
            std::regex{R"usd(upAxis\s*=\s*"([XYZxyz])")usd"}
        )) {
        output.upAxis = static_cast<char>(
            std::toupper(
                static_cast<unsigned char>(match[1].str()[0])
            )
        );
    }
    if (std::regex_search(
            metadata,
            match,
            std::regex{R"usd(defaultPrim\s*=\s*"([^"]+)")usd"}
        )) {
        output.defaultPrim = match[1].str();
    }
    if (!(output.metersPerUnit > 0.0) ||
        !std::isfinite(output.metersPerUnit)) {
        message = "USD metersPerUnit must be finite and positive";
        return false;
    }
    return true;
}

Matrix4 usdCoordinateTransform(
    const UsdStageMetadata& metadata
) {
    Matrix4 result{};
    for (auto& row : result.value) {
        std::fill(std::begin(row), std::end(row), 0.0);
    }
    result.value[3][3] = 1.0;
    const double unit = metadata.metersPerUnit;
    if (metadata.upAxis == 'Z') {
        // USD +X right, +Y forward, +Z up -> MetalRobo
        // +X forward, +Y left, +Z up.
        result.value[0][1] = unit;
        result.value[1][0] = -unit;
        result.value[2][2] = unit;
    } else if (metadata.upAxis == 'X') {
        result.value[0][1] = unit;
        result.value[1][2] = -unit;
        result.value[2][0] = unit;
    } else {
        // USD +X right, +Y up, +Z back -> MetalRobo
        // +X forward, +Y left, +Z up.
        result.value[0][2] = -unit;
        result.value[1][0] = -unit;
        result.value[2][1] = unit;
    }
    return result;
}

Matrix4 matrixFromSimd(const matrix_float4x4& value) {
    Matrix4 result;
    for (std::size_t column = 0u; column < 4u; ++column) {
        for (std::size_t row = 0u; row < 4u; ++row) {
            result.value[row][column] =
                value.columns[column][row];
        }
    }
    return result;
}

double determinant3(const Matrix4& matrix) {
    return
        matrix.value[0][0] *
            (
                matrix.value[1][1] * matrix.value[2][2] -
                matrix.value[1][2] * matrix.value[2][1]
            ) -
        matrix.value[0][1] *
            (
                matrix.value[1][0] * matrix.value[2][2] -
                matrix.value[1][2] * matrix.value[2][0]
            ) +
        matrix.value[0][2] *
            (
                matrix.value[1][0] * matrix.value[2][1] -
                matrix.value[1][1] * matrix.value[2][0]
            );
}

float materialScalar(
    MDLMaterial* material,
    const MDLMaterialSemantic semantic,
    const float fallback
) {
    if (material == nil) {
        return fallback;
    }
    for (MDLMaterialProperty* property in
         [material propertiesWithSemantic:semantic]) {
        switch (property.type) {
        case MDLMaterialPropertyTypeFloat:
            return property.floatValue;
        case MDLMaterialPropertyTypeFloat2:
            return property.float2Value.x;
        case MDLMaterialPropertyTypeFloat3:
            return property.float3Value.x;
        case MDLMaterialPropertyTypeFloat4:
            return property.float4Value.x;
        case MDLMaterialPropertyTypeColor:
            return property.luminance;
        default:
            break;
        }
    }
    return fallback;
}

mr_float4 materialColor(
    MDLMaterial* material,
    const MDLMaterialSemantic semantic,
    const mr_float4 fallback
) {
    if (material == nil) {
        return fallback;
    }
    for (MDLMaterialProperty* property in
         [material propertiesWithSemantic:semantic]) {
        switch (property.type) {
        case MDLMaterialPropertyTypeFloat3: {
            const vector_float3 value = property.float3Value;
            return {value.x, value.y, value.z, fallback.w};
        }
        case MDLMaterialPropertyTypeFloat4: {
            const vector_float4 value = property.float4Value;
            return {value.x, value.y, value.z, value.w};
        }
        case MDLMaterialPropertyTypeColor: {
            CGColorRef color = property.color;
            if (color == nullptr) {
                break;
            }
            const CGFloat* components = CGColorGetComponents(color);
            const std::size_t count =
                CGColorGetNumberOfComponents(color);
            if (count >= 3u) {
                return {
                    static_cast<float>(components[0]),
                    static_cast<float>(components[1]),
                    static_cast<float>(components[2]),
                    count >= 4u
                        ? static_cast<float>(components[3])
                        : fallback.w,
                };
            }
            if (count >= 1u) {
                const float value =
                    static_cast<float>(components[0]);
                return {
                    value,
                    value,
                    value,
                    count >= 2u
                        ? static_cast<float>(components[1])
                        : fallback.w,
                };
            }
            break;
        }
        default:
            break;
        }
    }
    return fallback;
}

MDLMaterialProperty* materialTextureProperty(
    MDLMaterial* material,
    const MDLMaterialSemantic semantic
) {
    if (material == nil) {
        return nil;
    }
    for (MDLMaterialProperty* property in
         [material propertiesWithSemantic:semantic]) {
        if (property.type == MDLMaterialPropertyTypeTexture ||
            property.type == MDLMaterialPropertyTypeURL ||
            property.type == MDLMaterialPropertyTypeString) {
            return property;
        }
    }
    return nil;
}

MDLMaterialProperty* materialPropertyContaining(
    MDLMaterial* material,
    const std::string_view token
) {
    if (material == nil) {
        return nil;
    }
    for (MDLMaterialProperty* property in material) {
        std::string name = utf8(property.name);
        std::ranges::transform(
            name,
            name.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            }
        );
        if (name.find(token) != std::string::npos) {
            return property;
        }
    }
    return nil;
}

MDLTextureSampler* textureSampler(
    MDLMaterialProperty* property
) {
    if (property == nil) {
        return nil;
    }
    if (property.type == MDLMaterialPropertyTypeTexture) {
        return property.textureSamplerValue;
    }
    NSURL* url = property.URLValue;
    if (url == nil && property.stringValue.length != 0u) {
        url = [NSURL
            fileURLWithPath:property.stringValue];
    }
    if (url == nil) {
        return nil;
    }
    MDLTextureSampler* sampler =
        [[MDLTextureSampler alloc] init];
    sampler.texture = [[MDLURLTexture alloc]
        initWithURL:url
              name:property.name];
    return sampler;
}

struct ModelTextureCache {
    std::map<std::pair<std::uintptr_t, bool>, std::uint32_t>
        objects;
    std::unordered_map<std::string, std::uint32_t> content;
};

std::uint32_t appendModelTextureBinding(
    MTKTextureLoader* textureLoader,
    MDLMaterialProperty* property,
    const bool srgb,
    const VisualAssetCookOptions& options,
    ModelTextureCache& textureCache,
    VisualAssetPackV2& pack,
    std::string& message
) {
    MDLTextureSampler* sampler = textureSampler(property);
    MDLTexture* texture = sampler.texture;
    if (texture == nil) {
        return MR_INVALID_INDEX;
    }
    const auto key = std::pair{
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)texture
        ),
        srgb,
    };
    std::uint32_t textureIndex = MR_INVALID_INDEX;
    if (const auto found = textureCache.objects.find(key);
        found != textureCache.objects.end()) {
        textureIndex = found->second;
    } else {
        VisualTextureImageV2 decoded;
        std::string id = utf8(texture.name);
        if (id.empty()) {
            id = utf8(property.name);
        }
        if (id.empty()) {
            id = "usd_texture_" +
                std::to_string(pack.textures.size());
        }
        bool decodedTexture = false;
        // Loose USD stages commonly expose relative texture URLs. Loading
        // them through MDLAsset::loadTextures can replace an otherwise valid
        // PNG/JPEG with a black placeholder on some Model I/O versions. Read
        // the authored file directly first; keep MTKTextureLoader as the
        // fallback for resolved USDZ/package and procedural textures.
        NSURL* authoredURL = property.URLValue;
        if (authoredURL == nil &&
            [texture isKindOfClass:MDLURLTexture.class]) {
            authoredURL = static_cast<MDLURLTexture*>(texture).URL;
        }
        if (authoredURL != nil && authoredURL.isFileURL) {
            std::filesystem::path texturePath{
                utf8(authoredURL.path)
            };
            if (texturePath.is_relative()) {
                texturePath = std::filesystem::path{pack.sourceUri}
                                  .parent_path() / texturePath;
            }
            if (const auto encoded = readBytes(texturePath)) {
                decodedTexture = decodeTexture(
                    *encoded,
                    srgb,
                    options.generateMipmaps,
                    id,
                    decoded
                );
            }
        }
        @autoreleasepool {
            if (!decodedTexture) {
                decodedTexture = decodeModelTexture(
                    textureLoader,
                    texture,
                    srgb,
                    options.generateMipmaps,
                    id,
                    decoded
                );
            }
        }
        if (!decodedTexture) {
            message =
                "Model I/O material texture could not be decoded";
            return MR_INVALID_INDEX;
        }
        const std::string contentKey =
            decoded.contentHash +
            (
                decoded.pixelFormat ==
                    VisualTexturePixelFormatV2::rgba8UnormSrgb
                ? ":srgb"
                : ":linear"
            );
        if (const auto existing =
                textureCache.content.find(contentKey);
            existing != textureCache.content.end()) {
            textureIndex = existing->second;
        } else {
            textureIndex = static_cast<std::uint32_t>(
                pack.textures.size()
            );
            pack.textures.push_back(std::move(decoded));
            textureCache.content.emplace(
                contentKey,
                textureIndex
            );
        }
        textureCache.objects.emplace(key, textureIndex);
    }

    const bool minNearest =
        sampler.hardwareFilter != nil &&
        sampler.hardwareFilter.minFilter ==
            MDLMaterialTextureFilterModeNearest;
    const bool magNearest =
        sampler.hardwareFilter != nil &&
        sampler.hardwareFilter.magFilter ==
            MDLMaterialTextureFilterModeNearest;
    const auto addressClass = [](
        const MDLMaterialTextureWrapMode mode
    ) {
        return mode == MDLMaterialTextureWrapModeClamp
            ? 1u
            : mode == MDLMaterialTextureWrapModeMirror
                ? 2u
                : 0u;
    };
    const std::uint32_t samplerIndex =
        (
            sampler.hardwareFilter == nil
            ? 0u
            : 3u * addressClass(
                    sampler.hardwareFilter.sWrapMode
                ) +
                addressClass(
                    sampler.hardwareFilter.tWrapMode
                )
        ) +
        (minNearest ? 9u : 0u) +
        (magNearest ? 18u : 0u) +
        (options.generateMipmaps ? 72u : 0u);
    MRVisualTextureBindingGPUV2 binding{};
    binding.resource = {
        textureIndex,
        samplerIndex,
        0u,
        0u,
    };
    binding.uvTransform0 = {1.0f, 0.0f, 0.0f, 0.0f};
    binding.uvTransform1 = {0.0f, 1.0f, 0.0f, 0.0f};
    if (sampler.transform != nil) {
        const matrix_float4x4 matrix =
            [sampler.transform localTransformAtTime:0.0];
        binding.uvTransform0 = {
            matrix.columns[0].x,
            matrix.columns[1].x,
            matrix.columns[3].x,
            0.0f,
        };
        binding.uvTransform1 = {
            matrix.columns[0].y,
            matrix.columns[1].y,
            matrix.columns[3].y,
            0.0f,
        };
    }
    if (pack.textureBindings.size() ==
        std::numeric_limits<std::uint32_t>::max()) {
        message = "USD texture binding count exceeds uint32";
        return MR_INVALID_INDEX;
    }
    const std::uint32_t result =
        static_cast<std::uint32_t>(
            pack.textureBindings.size()
        );
    pack.textureBindings.push_back(binding);
    return result;
}

std::uint32_t importModelMaterial(
    MTKTextureLoader* textureLoader,
    MDLMaterial* material,
    const bool forceNeutralMaterial,
    const VisualAssetCookOptions& options,
    std::unordered_map<void*, std::uint32_t>& materialMap,
    ModelTextureCache& textureCache,
    VisualAssetPackV2& pack,
    std::string& message
) {
    void* key = (__bridge void*)material;
    if (const auto found = materialMap.find(key);
        found != materialMap.end()) {
        return found->second;
    }
    MDLMaterialProperty* const baseColorTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticBaseColor
        );
    MDLMaterialProperty* const roughnessTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticRoughness
        );
    MDLMaterialProperty* const normalTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticTangentSpaceNormal
        );
    MDLMaterialProperty* const occlusionTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticAmbientOcclusion
        );
    MDLMaterialProperty* const emissionTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticEmission
        );
    MDLMaterialProperty* const clearcoatTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticClearcoat
        );
    MDLMaterialProperty* const clearcoatGlossTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticClearcoatGloss
        );
    MDLMaterialProperty* const metallicTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticMetallic
        );
    MDLMaterialProperty* const opacityTexture =
        materialTextureProperty(
            material,
            MDLMaterialSemanticOpacity
        );
    MRVisualMaterialGPUV2 result{};
    result.baseColorAndOpacity = materialColor(
        material,
        MDLMaterialSemanticBaseColor,
        {1.0f, 1.0f, 1.0f, 1.0f}
    );
    // In USD Preview Surface a connected texture replaces the numeric socket
    // default. Model I/O exposes both properties, including the default 0.18
    // diffuse value, but multiplying them would incorrectly darken every
    // connected texture. glTF factor modulation is handled by its own import
    // path and does not pass through here.
    result.baseColorAndOpacity = baseColorTexture == nil
        ? materialColor(
              material,
              MDLMaterialSemanticBaseColor,
              {1.0f, 1.0f, 1.0f, 1.0f}
          )
        : mr_float4{1.0f, 1.0f, 1.0f, 1.0f};
    if (forceNeutralMaterial) {
        // Binary STL carries geometry but no interoperable PBR material. A
        // neutral presentation keeps source-derived URDF geometry legible
        // without inventing robot-specific appearance.
        result.baseColorAndOpacity = {0.58f, 0.61f, 0.66f, 1.0f};
    }
    result.baseColorAndOpacity.w *= materialScalar(
        material,
        MDLMaterialSemanticOpacity,
        1.0f
    );
    result.emissionAndStrength = materialColor(
        material,
        MDLMaterialSemanticEmission,
        emissionTexture == nil
            ? mr_float4{0.0f, 0.0f, 0.0f, 1.0f}
            : mr_float4{1.0f, 1.0f, 1.0f, 1.0f}
    );
    result.emissionAndStrength = emissionTexture == nil
        ? materialColor(
              material,
              MDLMaterialSemanticEmission,
              {0.0f, 0.0f, 0.0f, 1.0f}
          )
        : mr_float4{1.0f, 1.0f, 1.0f, 1.0f};
    if (forceNeutralMaterial) {
        result.emissionAndStrength = {0.58f, 0.61f, 0.66f, 1.0f};
    }
    result.surface = {
        std::clamp(
            materialScalar(
                material,
                MDLMaterialSemanticRoughness,
                1.0f
            ),
            0.0f,
            1.0f
        ),
        std::clamp(
            materialScalar(
                material,
                MDLMaterialSemanticMetallic,
                metallicTexture == nil ? 0.0f : 1.0f
            ),
            0.0f,
            1.0f
        ),
        1.0f,
        std::clamp(
            materialScalar(
                material,
                MDLMaterialSemanticAmbientOcclusionScale,
                1.0f
            ),
            0.0f,
            1.0f
        ),
    };
    const float clearcoat = std::clamp(
        materialScalar(
            material,
            MDLMaterialSemanticClearcoat,
            clearcoatTexture == nil ? 0.0f : 1.0f
        ),
        0.0f,
        1.0f
    );
    const float clearcoatGloss = std::clamp(
        materialScalar(
            material,
            MDLMaterialSemanticClearcoatGloss,
            1.0f
        ),
        0.0f,
        1.0f
    );
    const float ior = std::max(
        materialScalar(
            material,
            MDLMaterialSemanticMaterialIndexOfRefraction,
            1.5f
        ),
        1.0f
    );
    const float dielectricF0 =
        std::pow((ior - 1.0f) / (ior + 1.0f), 2.0f);
    result.coatingAndAlphaCutoff = {
        clearcoat,
        1.0f - clearcoatGloss,
        dielectricF0 / 0.04f,
        0.5f,
    };
    const auto texture = [&](
        MDLMaterialProperty* property,
        const bool srgb
    ) {
        return appendModelTextureBinding(
            textureLoader,
            property,
            srgb,
            options,
            textureCache,
            pack,
            message
        );
    };
    result.textureIndices0 = {
        texture(baseColorTexture, true),
        MR_INVALID_INDEX,
        texture(normalTexture, false),
        texture(occlusionTexture, false),
    };
    result.textureIndices1 = {
        texture(emissionTexture, true),
        texture(clearcoatTexture, false),
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    result.reserved = {
        texture(metallicTexture, false),
        texture(opacityTexture, false),
        texture(roughnessTexture, false),
        texture(clearcoatGlossTexture, false),
    };
    if (!message.empty()) {
        return MR_INVALID_INDEX;
    }
    MRVisualAlphaMode alphaMode =
        result.baseColorAndOpacity.w < 0.999f ||
                result.reserved.y != MR_INVALID_INDEX
        ? MR_VISUAL_ALPHA_BLEND
        : MR_VISUAL_ALPHA_OPAQUE;
    if (MDLMaterialProperty* authoredMode =
            materialPropertyContaining(material, "alphamode");
        authoredMode != nil) {
        std::string mode = utf8(authoredMode.stringValue);
        std::ranges::transform(
            mode,
            mode.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            }
        );
        alphaMode = mode == "mask"
            ? MR_VISUAL_ALPHA_MASK
            : mode == "blend"
                ? MR_VISUAL_ALPHA_BLEND
                : MR_VISUAL_ALPHA_OPAQUE;
    }
    if (MDLMaterialProperty* cutoff =
            materialPropertyContaining(material, "alphacutoff");
        cutoff != nil &&
        cutoff.type == MDLMaterialPropertyTypeFloat) {
        result.coatingAndAlphaCutoff.w =
            std::clamp(cutoff.floatValue, 0.0f, 1.0f);
    }
    const bool unlit = forceNeutralMaterial ||
        materialPropertyContaining(material, "unlit") != nil;
    result.flags = {
        alphaMode,
        (
            material.materialFace == MDLMaterialFaceDoubleSided
            ? MR_VISUAL_MATERIAL_DOUBLE_SIDED
            : 0u
        ) |
            (unlit ? MR_VISUAL_MATERIAL_UNLIT : 0u),
        0u,
        0u,
    };
    if (pack.materials.size() ==
        std::numeric_limits<std::uint32_t>::max()) {
        message = "USD material count exceeds uint32";
        return MR_INVALID_INDEX;
    }
    for (std::uint32_t index = 0u;
         index < pack.materials.size();
         ++index) {
        MRVisualMaterialGPUV2 existing =
            pack.materials[index];
        existing.flags.w = 0u;
        if (std::memcmp(
                &existing,
                &result,
                sizeof(result)
            ) == 0) {
            materialMap.emplace(key, index);
            return index;
        }
    }
    const std::uint32_t index =
        static_cast<std::uint32_t>(pack.materials.size());
    result.flags.w = index + 1u;
    pack.materials.push_back(result);
    materialMap.emplace(key, index);
    return index;
}

MDLVertexAttributeData* modelAttribute(
    MDLMesh* mesh,
    NSString* semantic,
    const NSUInteger occurrence,
    const MDLVertexFormat format
) {
    NSUInteger seen = 0u;
    for (MDLVertexAttribute* attribute in
         mesh.vertexDescriptor.attributes) {
        if (attribute.name == nil ||
            ![attribute.name hasPrefix:semantic]) {
            continue;
        }
        if (seen++ == occurrence) {
            return [mesh
                vertexAttributeDataForAttributeNamed:attribute.name
                                             asFormat:format];
        }
    }
    return nil;
}

template <std::size_t Count>
std::array<float, Count> readModelAttribute(
    MDLVertexAttributeData* attribute,
    const std::size_t vertex,
    const std::array<float, Count>& fallback
) {
    if (attribute == nil || attribute.dataStart == nullptr) {
        return fallback;
    }
    std::array<float, Count> result{};
    std::memcpy(
        result.data(),
        static_cast<const std::uint8_t*>(attribute.dataStart) +
            vertex * attribute.stride,
        Count * sizeof(float)
    );
    return result;
}

bool modelDirectionAttributeValid(
    MDLVertexAttributeData* attribute,
    const std::size_t vertexCount
) {
    if (attribute == nil || attribute.dataStart == nullptr ||
        attribute.stride < 3u * sizeof(float)) {
        return false;
    }
    for (std::size_t vertex = 0u;
         vertex < vertexCount;
         ++vertex) {
        std::array<float, 3u> value{};
        std::memcpy(
            value.data(),
            static_cast<const std::uint8_t*>(
                attribute.dataStart
            ) + vertex * attribute.stride,
            value.size() * sizeof(float)
        );
        const float lengthSquared =
            value[0] * value[0] +
            value[1] * value[1] +
            value[2] * value[2];
        if (!std::isfinite(value[0]) ||
            !std::isfinite(value[1]) ||
            !std::isfinite(value[2]) ||
            !(lengthSquared > 1.0e-12f)) {
            return false;
        }
    }
    return true;
}

std::string stableModelObjectPath(
    MDLObject* object,
    const std::uint32_t rootOrdinal
) {
    std::vector<std::string> segments;
    for (MDLObject* cursor = object;
         cursor != nil;
         cursor = cursor.parent) {
        std::string name = utf8(cursor.name);
        if (name.empty()) {
            name = utf8(NSStringFromClass(cursor.class));
        }
        std::uint32_t ordinal =
            cursor.parent == nil ? rootOrdinal : 0u;
        if (cursor.parent != nil &&
            cursor.parent.children != nil) {
            for (MDLObject* sibling in
                 cursor.parent.children.objects) {
                if (sibling == cursor) {
                    break;
                }
                const std::string siblingName =
                    utf8(sibling.name).empty()
                    ? utf8(NSStringFromClass(sibling.class))
                    : utf8(sibling.name);
                if (siblingName == name) {
                    ++ordinal;
                }
            }
        }
        segments.push_back(
            name + "[" + std::to_string(ordinal) + "]"
        );
    }
    std::ranges::reverse(segments);
    std::string result;
    for (const std::string& segment : segments) {
        result += "/" + segment;
    }
    return result.empty()
        ? "/mesh[" + std::to_string(rootOrdinal) + "]"
        : result;
}

struct ReusedModelGeometry {
    std::vector<std::uint32_t> firstIndices;
    std::vector<std::uint32_t> indexCounts;
};

struct VisualCookTransformGPU {
    mr_float4 positionRow0;
    mr_float4 positionRow1;
    mr_float4 positionRow2;
    mr_float4 normalRow0;
    mr_float4 normalRow1;
    mr_float4 normalRow2;
    mr_uint4 counts;
};

bool bakeResidualOnMetal(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const Matrix4& residual,
    const std::span<MRVisualVertexGPUV2> vertices,
    std::string& message
) {
    if (vertices.empty()) {
        return true;
    }
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:vertices.data()
                   length:vertices.size_bytes()
                  options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        message =
            "USD residual vertex staging allocation failed";
        return false;
    }
    simd_double3x3 linear{
        {
            {
                residual.value[0][0],
                residual.value[1][0],
                residual.value[2][0],
            },
            {
                residual.value[0][1],
                residual.value[1][1],
                residual.value[2][1],
            },
            {
                residual.value[0][2],
                residual.value[1][2],
                residual.value[2][2],
            },
        }
    };
    const simd_double3x3 normalMatrix =
        simd_transpose(simd_inverse(linear));
    VisualCookTransformGPU transform{
        {
            static_cast<float>(residual.value[0][0]),
            static_cast<float>(residual.value[0][1]),
            static_cast<float>(residual.value[0][2]),
            static_cast<float>(residual.value[0][3]),
        },
        {
            static_cast<float>(residual.value[1][0]),
            static_cast<float>(residual.value[1][1]),
            static_cast<float>(residual.value[1][2]),
            static_cast<float>(residual.value[1][3]),
        },
        {
            static_cast<float>(residual.value[2][0]),
            static_cast<float>(residual.value[2][1]),
            static_cast<float>(residual.value[2][2]),
            static_cast<float>(residual.value[2][3]),
        },
        {
            static_cast<float>(normalMatrix.columns[0][0]),
            static_cast<float>(normalMatrix.columns[1][0]),
            static_cast<float>(normalMatrix.columns[2][0]),
            0.0f,
        },
        {
            static_cast<float>(normalMatrix.columns[0][1]),
            static_cast<float>(normalMatrix.columns[1][1]),
            static_cast<float>(normalMatrix.columns[2][1]),
            0.0f,
        },
        {
            static_cast<float>(normalMatrix.columns[0][2]),
            static_cast<float>(normalMatrix.columns[1][2]),
            static_cast<float>(normalMatrix.columns[2][2]),
            0.0f,
        },
        {
            static_cast<std::uint32_t>(vertices.size()),
            determinant3(residual) < 0.0 ? 1u : 0u,
            0u,
            0u,
        },
    };
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
        message = "USD residual compute command allocation failed";
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0u atIndex:0u];
    [encoder setBytes:&transform
               length:sizeof(transform)
              atIndex:1u];
    const NSUInteger threadCount = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder dispatchThreads:MTLSizeMake(vertices.size(), 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(threadCount, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        message =
            "USD residual compute bake failed: " +
            utf8(command.error.localizedDescription);
        return false;
    }
    std::memcpy(
        vertices.data(),
        buffer.contents,
        vertices.size_bytes()
    );
    return true;
}

std::string hashModelGeometry(
    const std::span<const MRVisualVertexGPUV2> vertices,
    const std::span<const std::uint32_t> indices,
    const std::uint32_t vertexBase,
    const std::span<const MRVisualPrimitiveGPUV2> primitives
) {
    CC_SHA256_CTX context{};
    CC_SHA256_Init(&context);
    updateSha256(
        context,
        vertices.data(),
        vertices.size_bytes()
    );
    for (const std::uint32_t index : indices) {
        const std::uint32_t local = index - vertexBase;
        updateSha256(context, &local, sizeof(local));
    }
    for (const MRVisualPrimitiveGPUV2& primitive : primitives) {
        updateSha256(
            context,
            &primitive.geometry.y,
            sizeof(primitive.geometry.y)
        );
    }
    return finishSha256(context);
}

bool appendModelMesh(
    MDLMesh* objectMesh,
    const std::uint32_t objectOrdinal,
    const Matrix4& stageConversion,
    const bool forceNeutralMaterial,
    const VisualAssetCookOptions& options,
    std::unordered_map<void*, std::uint32_t>& materialMap,
    ModelTextureCache& textureCache,
    std::unordered_map<std::string, ReusedModelGeometry>&
        geometryMap,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    MTKTextureLoader* textureLoader,
    id<MTLComputePipelineState> residualPipeline,
    VisualAssetPackV2& pack,
    std::string& message
) {
    MDLMesh* mesh =
        [objectMesh.instance isKindOfClass:MDLMesh.class]
        ? static_cast<MDLMesh*>(objectMesh.instance)
        : objectMesh;
    if (mesh.vertexCount == 0u ||
        mesh.vertexCount >
            std::numeric_limits<std::uint32_t>::max()) {
        message = "USD mesh has an invalid vertex count";
        return false;
    }
    MDLVertexAttributeData* normals = modelAttribute(
        mesh,
        MDLVertexAttributeNormal,
        0u,
        MDLVertexFormatFloat3
    );
    if (!modelDirectionAttributeValid(
            normals,
            mesh.vertexCount
        )) {
        if (!options.generateNormals) {
            message =
                "USD mesh normals are absent or invalid and generation "
                "is disabled";
            return false;
        }
        [mesh addNormalsWithAttributeNamed:MDLVertexAttributeNormal
                           creaseThreshold:0.75f];
        normals = modelAttribute(
            mesh,
            MDLVertexAttributeNormal,
            0u,
            MDLVertexFormatFloat3
        );
        if (!modelDirectionAttributeValid(
                normals,
                mesh.vertexCount
            )) {
            message =
                "Model I/O could not generate valid USD mesh normals";
            return false;
        }
    }
    MDLVertexAttributeData* uv0 = modelAttribute(
        mesh,
        MDLVertexAttributeTextureCoordinate,
        0u,
        MDLVertexFormatFloat2
    );
    MDLVertexAttributeData* tangents = modelAttribute(
        mesh,
        MDLVertexAttributeTangent,
        0u,
        MDLVertexFormatFloat4
    );
    if (!modelDirectionAttributeValid(
            tangents,
            mesh.vertexCount
        ) &&
        options.generateTangents && uv0 != nil) {
        [mesh
            addOrthTanBasisForTextureCoordinateAttributeNamed:
                MDLVertexAttributeTextureCoordinate
            normalAttributeNamed:MDLVertexAttributeNormal
            tangentAttributeNamed:MDLVertexAttributeTangent];
        tangents = modelAttribute(
            mesh,
            MDLVertexAttributeTangent,
            0u,
            MDLVertexFormatFloat4
        );
    }
    MDLVertexAttributeData* positions = modelAttribute(
        mesh,
        MDLVertexAttributePosition,
        0u,
        MDLVertexFormatFloat3
    );
    MDLVertexAttributeData* uv1 = modelAttribute(
        mesh,
        MDLVertexAttributeTextureCoordinate,
        1u,
        MDLVertexFormatFloat2
    );
    MDLVertexAttributeData* colors = modelAttribute(
        mesh,
        MDLVertexAttributeColor,
        0u,
        MDLVertexFormatFloat4
    );
    if (positions == nil || normals == nil) {
        message = "USD mesh has no readable positions or normals";
        return false;
    }
    const Matrix4 world = multiply(
        stageConversion,
        matrixFromSimd(
            [MDLTransform
                globalTransformWithObject:objectMesh
                                    atTime:0.0]
        )
    );
    MRVisualInstanceGPUV2 instance{};
    Matrix4 residual;
    if (!decomposeRigidResidual(
            world,
            instance.translationAndScale,
            instance.orientation,
            residual
        )) {
        message =
            "USD mesh transform is singular or non-finite";
        return false;
    }
    const bool hasResidual =
        std::abs(residual.value[0][0] - 1.0) > 1.0e-8 ||
        std::abs(residual.value[1][1] - 1.0) > 1.0e-8 ||
        std::abs(residual.value[2][2] - 1.0) > 1.0e-8 ||
        std::abs(residual.value[0][1]) > 1.0e-8 ||
        std::abs(residual.value[0][2]) > 1.0e-8 ||
        std::abs(residual.value[1][0]) > 1.0e-8 ||
        std::abs(residual.value[1][2]) > 1.0e-8 ||
        std::abs(residual.value[2][0]) > 1.0e-8 ||
        std::abs(residual.value[2][1]) > 1.0e-8;
    const std::uint32_t vertexOffset =
        static_cast<std::uint32_t>(pack.vertices.size());
    if (mesh.vertexCount >
        std::numeric_limits<std::uint32_t>::max() -
            vertexOffset) {
        message = "USD vertex arena exceeds uint32";
        return false;
    }
    pack.vertices.reserve(
        pack.vertices.size() + mesh.vertexCount
    );
    for (std::size_t vertex = 0u;
         vertex < mesh.vertexCount;
         ++vertex) {
        std::array<float, 3u> position =
            readModelAttribute<3u>(
                positions,
                vertex,
                {0.0f, 0.0f, 0.0f}
            );
        std::array<float, 3u> normal =
            readModelAttribute<3u>(
                normals,
                vertex,
                {0.0f, 0.0f, 1.0f}
            );
        std::array<float, 4u> tangent =
            readModelAttribute<4u>(
                tangents,
                vertex,
                {1.0f, 0.0f, 0.0f, 1.0f}
            );
        normalize3(normal[0], normal[1], normal[2]);
        normalize3(tangent[0], tangent[1], tangent[2]);
        const auto finiteVector3 = [](const auto& value) {
            return std::isfinite(value[0]) &&
                std::isfinite(value[1]) &&
                std::isfinite(value[2]);
        };
        if (!finiteVector3(normal) ||
            normal[0] * normal[0] +
                    normal[1] * normal[1] +
                    normal[2] * normal[2] <=
                1.0e-12f) {
            normal = {0.0f, 0.0f, 1.0f};
        }
        normalize3(normal[0], normal[1], normal[2]);
        if (!finiteVector3(tangent)) {
            tangent[0] = 0.0f;
            tangent[1] = 0.0f;
            tangent[2] = 0.0f;
        }
        const float tangentProjection =
            tangent[0] * normal[0] +
            tangent[1] * normal[1] +
            tangent[2] * normal[2];
        tangent[0] -= tangentProjection * normal[0];
        tangent[1] -= tangentProjection * normal[1];
        tangent[2] -= tangentProjection * normal[2];
        const float tangentLengthSquared =
            tangent[0] * tangent[0] +
            tangent[1] * tangent[1] +
            tangent[2] * tangent[2];
        if (!(tangentLengthSquared > 1.0e-12f)) {
            const std::array<float, 3u> reference =
                std::abs(normal[2]) < 0.999f
                ? std::array<float, 3u>{0.0f, 0.0f, 1.0f}
                : std::array<float, 3u>{0.0f, 1.0f, 0.0f};
            tangent[0] =
                reference[1] * normal[2] -
                reference[2] * normal[1];
            tangent[1] =
                reference[2] * normal[0] -
                reference[0] * normal[2];
            tangent[2] =
                reference[0] * normal[1] -
                reference[1] * normal[0];
        }
        normalize3(tangent[0], tangent[1], tangent[2]);
        const std::array<float, 2u> firstUv =
            readModelAttribute<2u>(
                uv0,
                vertex,
                {0.0f, 0.0f}
            );
        const std::array<float, 2u> secondUv =
            readModelAttribute<2u>(
                uv1,
                vertex,
                firstUv
            );
        const std::array<float, 4u> color =
            readModelAttribute<4u>(
                colors,
                vertex,
                {1.0f, 1.0f, 1.0f, 1.0f}
            );
        MRVisualVertexGPUV2 result{};
        result.position = {
            position[0], position[1], position[2], 1.0f,
        };
        result.normalAndTangentSign = {
            normal[0], normal[1], normal[2],
            tangent[3] < 0.0f ? -1.0f : 1.0f,
        };
        result.tangent = {
            tangent[0], tangent[1], tangent[2], 0.0f,
        };
        result.texcoord01 = {
            firstUv[0], firstUv[1],
            secondUv[0], secondUv[1],
        };
        result.color = {
            color[0], color[1], color[2], color[3],
        };
        pack.vertices.push_back(result);
    }
    if (hasResidual &&
        !bakeResidualOnMetal(
            device,
            queue,
            residualPipeline,
            residual,
            std::span<MRVisualVertexGPUV2>{
                pack.vertices.data() + vertexOffset,
                pack.vertices.size() - vertexOffset,
            },
            message
        )) {
        return false;
    }

    const std::string nodeName = utf8(objectMesh.name);
    const std::string nodePath =
        stableModelObjectPath(objectMesh, objectOrdinal);
    AuthoredVisualBodyBinding body;
    if (!resolveVisualBodyBinding(
            options,
            nodeName,
            body,
            message
        )) {
        return false;
    }
    instance.binding = {
        0u,
        body.body,
        body.kind,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
    };
    instance.identity = {
        0u,
        0u,
        body.body,
        static_cast<std::uint32_t>(pack.instances.size() + 1u),
    };
    instance.geometry.x =
        static_cast<std::uint32_t>(pack.primitives.size());
    const std::uint32_t instanceIndex =
        static_cast<std::uint32_t>(pack.instances.size());
    pack.instances.push_back(instance);

    for (MDLSubmesh* source in mesh.submeshes) {
        MDLSubmesh* triangles =
            source.geometryType == MDLGeometryTypeTriangles &&
                source.indexType == MDLIndexBitDepthUInt32
            ? source
            : [[MDLSubmesh alloc]
                  initWithMDLSubmesh:source
                          indexType:MDLIndexBitDepthUInt32
                       geometryType:MDLGeometryTypeTriangles];
        if (triangles == nil || triangles.indexCount == 0u ||
            triangles.indexCount % 3u != 0u ||
            triangles.indexCount >
                std::numeric_limits<std::uint32_t>::max()) {
            message =
                "Model I/O could not triangulate a USD submesh";
            return false;
        }
        id<MDLMeshBuffer> indexBuffer =
            [triangles
                indexBufferAsIndexType:MDLIndexBitDepthUInt32];
        MDLMeshBufferMap* indexMap = [indexBuffer map];
        if (indexMap == nil || indexMap.bytes == nullptr) {
            message = "USD index buffer could not be mapped";
            return false;
        }
        const std::uint32_t firstIndex =
            static_cast<std::uint32_t>(pack.indices.size());
        if (triangles.indexCount >
            std::numeric_limits<std::uint32_t>::max() -
                firstIndex) {
            message = "USD index arena exceeds uint32";
            return false;
        }
        const auto* sourceIndices =
            static_cast<const std::uint32_t*>(indexMap.bytes);
        for (std::size_t index = 0u;
             index < triangles.indexCount;
             index += 3u) {
            std::array<std::uint32_t, 3u> local{
                sourceIndices[index],
                sourceIndices[index + 1u],
                sourceIndices[index + 2u],
            };
            if (std::ranges::any_of(
                    local,
                    [mesh](const std::uint32_t value) {
                        return value >= mesh.vertexCount;
                    }
                )) {
                message = "USD submesh index is out of range";
                return false;
            }
            if (determinant3(residual) < 0.0) {
                std::swap(local[1], local[2]);
            }
            for (const std::uint32_t value : local) {
                pack.indices.push_back(vertexOffset + value);
            }
        }
        const std::uint32_t materialIndex =
            importModelMaterial(
                textureLoader,
                triangles.material,
                forceNeutralMaterial,
                options,
                materialMap,
                textureCache,
                pack,
                message
            );
        if (!message.empty() ||
            materialIndex == MR_INVALID_INDEX) {
            return false;
        }
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry = {
            firstIndex,
            static_cast<std::uint32_t>(triangles.indexCount),
            materialIndex,
            instanceIndex,
        };
        primitive.identity = {
            0u,
            0u,
            instance.identity.z,
            static_cast<std::uint32_t>(
                pack.primitives.size() + 1u
            ),
        };
        primitive.boundsMinimum = {
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            1.0f,
        };
        primitive.boundsMaximum = {
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            1.0f,
        };
        for (std::size_t index = firstIndex;
             index < pack.indices.size();
             ++index) {
            const MRVisualVertexGPUV2& vertex =
                pack.vertices[pack.indices[index]];
            primitive.boundsMinimum.x = std::min(
                primitive.boundsMinimum.x,
                vertex.position.x
            );
            primitive.boundsMinimum.y = std::min(
                primitive.boundsMinimum.y,
                vertex.position.y
            );
            primitive.boundsMinimum.z = std::min(
                primitive.boundsMinimum.z,
                vertex.position.z
            );
            primitive.boundsMaximum.x = std::max(
                primitive.boundsMaximum.x,
                vertex.position.x
            );
            primitive.boundsMaximum.y = std::max(
                primitive.boundsMaximum.y,
                vertex.position.y
            );
            primitive.boundsMaximum.z = std::max(
                primitive.boundsMaximum.z,
                vertex.position.z
            );
        }
        pack.primitives.push_back(primitive);
    }
    const std::uint32_t firstPrimitive =
        pack.instances.back().geometry.x;
    const std::uint32_t primitiveCount =
        static_cast<std::uint32_t>(pack.primitives.size()) -
        firstPrimitive;
    const std::uint32_t firstMeshIndex =
        primitiveCount == 0u
        ? static_cast<std::uint32_t>(pack.indices.size())
        : pack.primitives[firstPrimitive].geometry.x;
    const std::string geometryHash = hashModelGeometry(
        std::span<const MRVisualVertexGPUV2>{
            pack.vertices.data() + vertexOffset,
            pack.vertices.size() - vertexOffset,
        },
        std::span<const std::uint32_t>{
            pack.indices.data() + firstMeshIndex,
            pack.indices.size() - firstMeshIndex,
        },
        vertexOffset,
        std::span<const MRVisualPrimitiveGPUV2>{
            pack.primitives.data() + firstPrimitive,
            primitiveCount,
        }
    );
    if (const auto reused = geometryMap.find(geometryHash);
        reused != geometryMap.end()) {
        if (reused->second.firstIndices.size() != primitiveCount) {
            message =
                "canonical USD geometry hash has incompatible subsets";
            return false;
        }
        for (std::uint32_t primitive = 0u;
             primitive < primitiveCount;
             ++primitive) {
            MRVisualPrimitiveGPUV2& record =
                pack.primitives[firstPrimitive + primitive];
            if (record.geometry.y !=
                reused->second.indexCounts[primitive]) {
                message =
                    "canonical USD geometry subset count changed";
                return false;
            }
            record.geometry.x =
                reused->second.firstIndices[primitive];
        }
        pack.vertices.resize(vertexOffset);
        pack.indices.resize(firstMeshIndex);
    } else {
        ReusedModelGeometry stored;
        stored.firstIndices.reserve(primitiveCount);
        stored.indexCounts.reserve(primitiveCount);
        for (std::uint32_t primitive = 0u;
             primitive < primitiveCount;
             ++primitive) {
            const MRVisualPrimitiveGPUV2& record =
                pack.primitives[firstPrimitive + primitive];
            stored.firstIndices.push_back(record.geometry.x);
            stored.indexCounts.push_back(record.geometry.y);
        }
        geometryMap.emplace(geometryHash, std::move(stored));
    }
    pack.instances.back().geometry.y =
        primitiveCount;
    VisualSymbolicBindingV2 binding;
    binding.node = nodePath.empty()
        ? nodeName.empty()
            ? "usd_mesh_" + std::to_string(instanceIndex)
            : nodeName
        : nodePath;
    binding.instanceIndex = instanceIndex;
    if (body.kind != MR_VISUAL_BINDING_ASSET) {
        binding.link = nodeName;
        binding.bodyIndex = body.body;
        binding.binding =
            static_cast<MRVisualBindingKind>(body.kind);
    }
    pack.symbolicBindings.push_back(std::move(binding));
    return true;
}

MDLVertexDescriptor* canonicalModelVertexDescriptor() {
    MDLVertexDescriptor* descriptor =
        [[MDLVertexDescriptor alloc] init];
    descriptor.attributes[0] = [[MDLVertexAttribute alloc]
        initWithName:MDLVertexAttributePosition
              format:MDLVertexFormatFloat3
              offset:0u
         bufferIndex:0u];
    descriptor.attributes[1] = [[MDLVertexAttribute alloc]
        initWithName:MDLVertexAttributeNormal
              format:MDLVertexFormatFloat3
              offset:16u
         bufferIndex:0u];
    descriptor.attributes[2] = [[MDLVertexAttribute alloc]
        initWithName:MDLVertexAttributeTangent
              format:MDLVertexFormatFloat4
              offset:32u
         bufferIndex:0u];
    descriptor.attributes[3] = [[MDLVertexAttribute alloc]
        initWithName:MDLVertexAttributeTextureCoordinate
              format:MDLVertexFormatFloat2
              offset:48u
         bufferIndex:0u];
    descriptor.attributes[4] = [[MDLVertexAttribute alloc]
        initWithName:
            [MDLVertexAttributeTextureCoordinate
                stringByAppendingString:@"_1"]
              format:MDLVertexFormatFloat2
              offset:56u
         bufferIndex:0u];
    descriptor.attributes[5] = [[MDLVertexAttribute alloc]
        initWithName:MDLVertexAttributeColor
              format:MDLVertexFormatFloat4
              offset:64u
         bufferIndex:0u];
    descriptor.layouts[0] = [[MDLVertexBufferLayout alloc]
        initWithStride:sizeof(MRVisualVertexGPUV2)];
    return descriptor;
}

VisualAssetCookDiagnostics cookModelIOAsset(
    const std::filesystem::path& source,
    std::string sourceHash,
    VisualAssetPackV2& output,
    const VisualAssetCookOptions& options,
    const bool sourceIsUSD
) {
    VisualAssetCookDiagnostics diagnostics;
    std::string message;
    UsdStageMetadata metadata;
    if (sourceIsUSD &&
        !readUsdStageMetadata(source, metadata, message)) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            std::move(message)
        );
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::internalFailure,
            "Metal device is unavailable for Model I/O cooking"
        );
    }
    id<MTLCommandQueue> cookQueue = [device newCommandQueue];
    NSError* metalError = nil;
    const std::filesystem::path metallibPath{
        METALROBO_DEFAULT_METALLIB
    };
    id<MTLLibrary> library = [device
        newLibraryWithURL:
            [NSURL fileURLWithPath:@(metallibPath.string().c_str())]
                   error:&metalError];
    id<MTLFunction> residualFunction = [library
        newFunctionWithName:@"mr_visual_cook_bake_residual_v3"];
    id<MTLComputePipelineState> residualPipeline =
        residualFunction == nil
        ? nil
        : [device
              newComputePipelineStateWithFunction:residualFunction
                                            error:&metalError];
    if (cookQueue == nil || library == nil ||
        residualPipeline == nil) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::internalFailure,
            "Model I/O residual Metal pipeline is unavailable: " +
                utf8(metalError.localizedDescription)
        );
    }
    MTKMeshBufferAllocator* allocator =
        [[MTKMeshBufferAllocator alloc] initWithDevice:device];
    MTKTextureLoader* textureLoader =
        [[MTKTextureLoader alloc] initWithDevice:device];
    NSURL* url = [NSURL
        fileURLWithPath:@(source.string().c_str())];
    MDLAsset* asset = [[MDLAsset alloc]
        initWithURL:url
  vertexDescriptor:canonicalModelVertexDescriptor()
   bufferAllocator:allocator];
    if (asset == nil || asset.count == 0u) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            "Model I/O could not compose the visual source"
        );
    }
    // USDZ requires Model I/O package resolution. Loose USD keeps authored
    // texture URLs intact so appendModelTextureBinding can decode their bytes
    // directly and deterministically.
    if (source.extension() == ".usdz") {
        [asset loadTextures];
    }
    NSArray<MDLObject*>* meshObjects =
        [asset childObjectsOfClass:MDLMesh.class];
    if (meshObjects.count == 0u) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidGeometry,
            "visual source contains no triangle meshes"
        );
    }

    VisualAssetPackV2 candidate;
    candidate.id = options.id.empty()
        ? source.stem().string()
        : options.id;
    candidate.sourceUri = source.string();
    candidate.sourceContentHash = std::move(sourceHash);
    candidate.license = options.license;
    candidate.preprocessingProvenance =
        options.preprocessingProvenance +
        ";input=" + std::string{sourceIsUSD ? "usd" : "stl"} +
        ";importer=ModelIO+MTKMeshBufferAllocator" +
        ";metersPerUnit=" +
        std::to_string(sourceIsUSD ? metadata.metersPerUnit : 1.0) +
        ";upAxis=" + std::string{sourceIsUSD ? metadata.upAxis : 'Z'} +
        ";defaultPrim=" +
        (sourceIsUSD ? metadata.defaultPrim : std::string{}) +
        ";modelio=" + frameworkVersion(MDLAsset.class) +
        ";sdk=" +
        std::to_string(__MAC_OS_X_VERSION_MAX_ALLOWED) +
        ";cook-kernel=" METALROBO_VISUAL_ASSET_COOK_KERNEL_HASH +
        ";residual-transform=metal-compute-polar-v1" +
        ";canonical-vertex-layout=mrvisualvertexgpuv2" +
        ";normals=" +
        (options.generateNormals ? "generate-if-missing" : "authored") +
        ";tangents=" +
        (options.generateTangents ? "generate-if-missing" : "authored") +
        ";mips=" +
        (options.generateMipmaps
             ? "semantic-linear-box-v2"
             : "none");
    Matrix4 conversion{};
    if (sourceIsUSD) {
        conversion = usdCoordinateTransform(metadata);
    } else {
        for (std::size_t axis = 0u; axis < 4u; ++axis) {
            conversion.value[axis][axis] = 1.0;
        }
    }
    std::unordered_map<void*, std::uint32_t> materialMap;
    ModelTextureCache textureCache;
    std::unordered_map<std::string, ReusedModelGeometry>
        geometryMap;
    for (std::uint32_t objectOrdinal = 0u;
         objectOrdinal < meshObjects.count;
         ++objectOrdinal) {
        MDLObject* object = meshObjects[objectOrdinal];
        if (![object isKindOfClass:MDLMesh.class] ||
            object.hidden) {
            continue;
        }
        if (!appendModelMesh(
                static_cast<MDLMesh*>(object),
                objectOrdinal,
                conversion,
                !sourceIsUSD,
                options,
                    materialMap,
                    textureCache,
                    geometryMap,
                    device,
                    cookQueue,
                    textureLoader,
                    residualPipeline,
                candidate,
                message
            )) {
            return reject(
                std::move(diagnostics),
                message.find("material") != std::string::npos ||
                        message.find("texture") != std::string::npos
                    ? VisualAssetCookStatus::invalidMaterial
                    : VisualAssetCookStatus::invalidGeometry,
                std::move(message)
            );
        }
    }
    if (candidate.instances.empty()) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidGeometry,
            "visual source has no visible mesh instances"
        );
    }
    CC_SHA256_CTX dependencyContext{};
    CC_SHA256_Init(&dependencyContext);
    for (const VisualTextureImageV2& texture :
         candidate.textures) {
        updateSha256(
            dependencyContext,
            texture.contentHash.data(),
            texture.contentHash.size()
        );
    }
    candidate.preprocessingProvenance +=
        ";dependency-set=" +
        finishSha256(dependencyContext) +
        ";texture-transform=mdl-sampler-affine-v1";
    candidate.contentHash =
        computeVisualAssetPackContentHash(candidate);
    if (!candidate.valid(&message)) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            std::move(message)
        );
    }
    diagnostics.vertexCount =
        static_cast<std::uint32_t>(candidate.vertices.size());
    diagnostics.indexCount =
        static_cast<std::uint32_t>(candidate.indices.size());
    diagnostics.primitiveCount =
        static_cast<std::uint32_t>(candidate.primitives.size());
    diagnostics.instanceCount =
        static_cast<std::uint32_t>(candidate.instances.size());
    diagnostics.materialCount =
        static_cast<std::uint32_t>(candidate.materials.size());
    diagnostics.textureCount =
        static_cast<std::uint32_t>(candidate.textures.size());
    diagnostics.sourceHash = candidate.sourceContentHash;
    diagnostics.packHash = candidate.contentHash;
    output = std::move(candidate);
    return diagnostics;
}

} // namespace

VisualAssetCookDiagnostics cookVisualAsset(
    const std::filesystem::path& source,
    VisualAssetPackV2& output,
    const VisualAssetCookOptions& options
) {
    VisualAssetCookDiagnostics diagnostics;
    try {
        std::string extension = source.extension().string();
        std::ranges::transform(
            extension,
            extension.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            }
        );
        if (extension == ".glb" || extension == ".gltf") {
            const auto bytes = readBytes(source);
            if (!bytes.has_value()) {
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::ioFailure,
                    "visual source could not be read"
                );
            }
            return cookGltf(source, *bytes, output, options);
        }
        if (extension == ".usd" || extension == ".usda" ||
            extension == ".usdc" || extension == ".usdz") {
            std::string sourceHash = sha256File(source);
            if (sourceHash.empty()) {
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::ioFailure,
                    "USD source could not be hashed"
                );
            }
            return cookModelIOAsset(
                source,
                std::move(sourceHash),
                output,
                options,
                true
            );
        }
        if (extension == ".stl") {
            std::string sourceHash = sha256File(source);
            if (sourceHash.empty()) {
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::ioFailure,
                    "STL source could not be hashed"
                );
            }
            return cookModelIOAsset(
                source,
                std::move(sourceHash),
                output,
                options,
                false
            );
        }
        if (extension == ".dae") {
            return reject(
                std::move(diagnostics),
                VisualAssetCookStatus::unsupportedFormat,
                "DAE is an offline legacy input; convert it to GLB and "
                "record the converter version in preprocessing provenance"
            );
        }
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::unsupportedFormat,
            "visual source must be GLB/glTF, STL, or a Model I/O USD asset"
        );
    } catch (const std::bad_alloc&) {
        return reject(
            {},
            VisualAssetCookStatus::capacityOverflow,
            "visual asset cook exhausted host memory"
        );
    } catch (const std::exception& error) {
        return reject(
            {},
            VisualAssetCookStatus::internalFailure,
            error.what()
        );
    }
}

VisualAssetCookDiagnostics cookUrdfVisualDescription(
    const std::filesystem::path& urdf,
    std::vector<VisualAssetPackV2>& output,
    const VisualAssetCookOptions& options
) {
    VisualAssetCookDiagnostics diagnostics;
    const auto bytes = readBytes(urdf);
    if (!bytes.has_value()) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::ioFailure,
            "URDF visual description could not be read"
        );
    }
    xmlDoc* document = xmlReadMemory(
        reinterpret_cast<const char*>(bytes->data()),
        static_cast<int>(bytes->size()),
        urdf.string().c_str(),
        nullptr,
        XML_PARSE_NONET
    );
    if (document == nullptr) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::malformedAsset,
            "URDF visual description is malformed"
        );
    }
    std::vector<VisualAssetPackV2> candidate;
    xmlNode* robot = xmlDocGetRootElement(document);
    for (xmlNode* link : xmlChildren(robot, "link")) {
        const std::string linkName =
            xmlProperty(link, "name").value_or("");
        if (!options.linkBodyIndices.empty() &&
            !options.linkBodyIndices.contains(linkName)) {
            continue;
        }
        for (xmlNode* visual : xmlChildren(link, "visual")) {
            xmlNode* geometry = xmlChild(visual, "geometry");
            xmlNode* mesh = xmlChild(geometry, "mesh");
            if (mesh == nullptr) {
                xmlFreeDoc(document);
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::unsupportedFeature,
                    "URDF primitive visuals must be authored into the GLB "
                    "presentation asset"
                );
            }
            const std::string filename =
                xmlProperty(mesh, "filename").value_or("");
            if (filename.empty() ||
                filename.starts_with("package://")) {
                xmlFreeDoc(document);
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::invalidBinding,
                    "URDF visual mesh needs a local path resolved before "
                    "cooking"
                );
            }
            std::filesystem::path source{filename};
            if (source.is_relative()) {
                source = urdf.parent_path() / source;
            }
            VisualAssetCookOptions visualOptions = options;
            visualOptions.id =
                options.id.empty()
                ? linkName + "_visual_" +
                      std::to_string(candidate.size())
                : options.id + "_" + linkName + "_" +
                      std::to_string(candidate.size());
            if (const auto body =
                    options.linkBodyIndices.find(linkName);
                body != options.linkBodyIndices.end()) {
                visualOptions.linkBodyIndices.clear();
            }
            VisualAssetPackV2 pack;
            VisualAssetCookDiagnostics result =
                cookVisualAsset(source, pack, visualOptions);
            if (!result.succeeded()) {
                xmlFreeDoc(document);
                return result;
            }
            const auto body =
                options.linkBodyIndices.find(linkName);
            Matrix4 authoredTransform;
            if (!urdfVisualTransform(
                    visual,
                    mesh,
                    authoredTransform
                )) {
                xmlFreeDoc(document);
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::invalidBinding,
                    "URDF visual origin or mesh scale is invalid"
                );
            }
            if (const auto centerOfMass =
                    options.linkCenterOfMassOffsets.find(linkName);
                centerOfMass != options.linkCenterOfMassOffsets.end()) {
                // Physics state describes the body COM, whereas URDF visual
                // coordinates are relative to the link origin.  Convert the
                // mesh from link coordinates into the COM-centred runtime
                // frame before applying the authored visual origin/scale.
                Matrix4 originFromCenterOfMass{};
                originFromCenterOfMass.value[0][0] = 1.0;
                originFromCenterOfMass.value[1][1] = 1.0;
                originFromCenterOfMass.value[2][2] = 1.0;
                originFromCenterOfMass.value[3][3] = 1.0;
                originFromCenterOfMass.value[0][3] =
                    -centerOfMass->second.x;
                originFromCenterOfMass.value[1][3] =
                    -centerOfMass->second.y;
                originFromCenterOfMass.value[2][3] =
                    -centerOfMass->second.z;
                authoredTransform = multiply(
                    originFromCenterOfMass,
                    authoredTransform
                );
            }
            for (std::uint32_t instanceIndex = 0u;
                 instanceIndex < pack.instances.size();
                 ++instanceIndex) {
                MRVisualInstanceGPUV2& instance =
                    pack.instances[instanceIndex];
                const Matrix4 transformed = multiply(
                    authoredTransform,
                    visualInstanceTransform(instance)
                );
                if (!decomposeUniform(
                        transformed,
                        instance.translationAndScale,
                        instance.orientation
                    )) {
                    xmlFreeDoc(document);
                    return reject(
                        std::move(diagnostics),
                        VisualAssetCookStatus::invalidBinding,
                        "URDF visual transform has non-uniform scale"
                    );
                }
                if (body != options.linkBodyIndices.end()) {
                    instance.binding.y = body->second;
                    instance.binding.z =
                        MR_VISUAL_BINDING_ARTICULATED_LINK;
                    instance.identity.z = body->second;
                }
                VisualSymbolicBindingV2 binding;
                binding.node =
                    linkName + "/" +
                    std::to_string(instanceIndex);
                binding.link = body == options.linkBodyIndices.end()
                    ? std::string{}
                    : linkName;
                binding.instanceIndex = instanceIndex;
                binding.bodyIndex =
                    body == options.linkBodyIndices.end()
                    ? MR_INVALID_INDEX
                    : body->second;
                binding.binding =
                    body == options.linkBodyIndices.end()
                    ? MR_VISUAL_BINDING_ASSET
                    : MR_VISUAL_BINDING_ARTICULATED_LINK;
                pack.symbolicBindings.push_back(
                    std::move(binding)
                );
            }
            pack.preprocessingProvenance +=
                ";urdf=" + sha256(*bytes) +
                ";link=" + linkName;
            pack.contentHash =
                computeVisualAssetPackContentHash(pack);
            candidate.push_back(std::move(pack));
            diagnostics.vertexCount += result.vertexCount;
            diagnostics.indexCount += result.indexCount;
            diagnostics.primitiveCount += result.primitiveCount;
            diagnostics.instanceCount += result.instanceCount;
            diagnostics.materialCount += result.materialCount;
            diagnostics.textureCount += result.textureCount;
        }
    }
    xmlFreeDoc(document);
    if (candidate.empty()) {
        return reject(
            std::move(diagnostics),
            VisualAssetCookStatus::invalidGeometry,
            "URDF contains no mesh visual elements"
        );
    }
    diagnostics.sourceHash = sha256(*bytes);
    diagnostics.packHash =
        "collection:" + std::to_string(candidate.size());
    output = std::move(candidate);
    return diagnostics;
}

} // namespace metalrobo
