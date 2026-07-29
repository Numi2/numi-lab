#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <ModelIO/ModelIO.h>

#include "metalrobo/VisualPresentation.hpp"

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
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

std::string sha256(const std::span<const std::uint8_t> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(
        bytes.data(),
        static_cast<CC_LONG>(bytes.size()),
        digest.data()
    );
    constexpr std::array digits{
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    };
    std::string result = "sha256:";
    result.reserve(result.size() + digest.size() * 2u);
    for (const unsigned char byte : digest) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 15u]);
    }
    return result;
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

bool decodeTexture(
    const std::vector<std::uint8_t>& encoded,
    const bool srgb,
    const bool mipmaps,
    const std::string& id,
    VisualTextureImageV1& output
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
    output.id = id;
    output.contentHash = sha256(encoded);
    output.width = static_cast<std::uint32_t>(width);
    output.height = static_cast<std::uint32_t>(height);
    output.flags = srgb ? MR_VISUAL_TEXTURE_SRGB : 0u;
    std::uint32_t levelWidth = output.width;
    std::uint32_t levelHeight = output.height;
    while (true) {
        output.mipTexelOffsets.push_back(
            static_cast<std::uint32_t>(output.rgba8.size() / 4u)
        );
        output.rgba8.insert(
            output.rgba8.end(),
            level.begin(),
            level.end()
        );
        if (!mipmaps ||
            (levelWidth == 1u && levelHeight == 1u) ||
            output.mipTexelOffsets.size() == 9u) {
            break;
        }
        const std::uint32_t nextWidth =
            std::max(1u, levelWidth / 2u);
        const std::uint32_t nextHeight =
            std::max(1u, levelHeight / 2u);
        std::vector<std::uint8_t> next(
            static_cast<std::size_t>(nextWidth) * nextHeight * 4u
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
    return true;
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
        if (tangent[0] == 0.0f &&
            tangent[1] == 0.0f &&
            tangent[2] == 0.0f) {
            tangent = std::abs(normal.z) < 0.999f
                ? std::array{-normal.y, normal.x, 0.0f}
                : std::array{1.0f, 0.0f, 0.0f};
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

std::uint32_t textureForMaterial(
    const GltfDocument& document,
    const std::uint32_t textureIndex,
    const bool srgb,
    const VisualAssetCookOptions& options,
    std::map<std::pair<std::uint32_t, bool>, std::uint32_t>& textureMap,
    VisualAssetPackV1& pack,
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
    VisualTextureImageV1 decoded;
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

bool importMaterials(
    const GltfDocument& document,
    const VisualAssetCookOptions& options,
    VisualAssetPackV1& pack,
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
            const std::uint32_t source =
                textureReference(record);
            const std::uint32_t result = textureForMaterial(
                document,
                source,
                isSrgb,
                options,
                textureMap,
                pack,
                message
            );
            if (source != MR_INVALID_INDEX &&
                result == MR_INVALID_INDEX &&
                message.empty()) {
                message = "glTF material texture is invalid";
            }
            return result;
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
    VisualAssetPackV1& pack,
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
    VisualAssetPackV1& pack,
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
        const auto body = options.linkBodyIndices.find(nodeName);
        instance.binding = {
            0u,
            body == options.linkBodyIndices.end()
                ? MR_INVALID_INDEX
                : body->second,
            body == options.linkBodyIndices.end()
                ? MR_VISUAL_BINDING_ASSET
                : MR_VISUAL_BINDING_ARTICULATED_LINK,
            MR_VISUAL_INSTANCE_CASTS_SHADOW |
                MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
        };
        instance.identity = {
            0u,
            0u,
            body == options.linkBodyIndices.end()
                ? MR_INVALID_INDEX
                : body->second,
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
        VisualSymbolicBindingV1 binding;
        binding.node = nodeName.empty()
            ? "node_" + std::to_string(nodeIndex)
            : nodeName;
        binding.instanceIndex = instanceIndex;
        if (body != options.linkBodyIndices.end()) {
            binding.link = nodeName;
            binding.bodyIndex = body->second;
            binding.binding =
                MR_VISUAL_BINDING_ARTICULATED_LINK;
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
    VisualAssetPackV1& pack,
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
    VisualAssetPackV1& output,
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
    VisualAssetPackV1 candidate;
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

} // namespace

VisualAssetCookDiagnostics cookVisualAsset(
    const std::filesystem::path& source,
    VisualAssetPackV1& output,
    const VisualAssetCookOptions& options
) {
    VisualAssetCookDiagnostics diagnostics;
    try {
        const auto bytes = readBytes(source);
        if (!bytes.has_value()) {
            return reject(
                std::move(diagnostics),
                VisualAssetCookStatus::ioFailure,
                "visual source could not be read"
            );
        }
        std::string extension = source.extension().string();
        std::ranges::transform(
            extension,
            extension.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            }
        );
        if (extension == ".glb" || extension == ".gltf") {
            return cookGltf(source, *bytes, output, options);
        }
        if (extension == ".usd" || extension == ".usda" ||
            extension == ".usdc" || extension == ".usdz") {
            // Model I/O is the authoritative USD importer. Its runtime mesh
            // conversion is deliberately separate from the glTF parser so
            // every path still terminates in the same immutable pack.
            NSURL* url = [NSURL
                fileURLWithPath:@(source.string().c_str())];
            MDLAsset* asset = [[MDLAsset alloc] initWithURL:url];
            if (asset == nil || asset.count == 0u) {
                return reject(
                    std::move(diagnostics),
                    VisualAssetCookStatus::malformedAsset,
                    "Model I/O could not import the USD asset"
                );
            }
            return reject(
                std::move(diagnostics),
                VisualAssetCookStatus::unsupportedFeature,
                "USD was validated by Model I/O, but this asset requires "
                "export to GLB before the deterministic V1 pack cook"
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
            "visual source must be GLB/glTF or a Model I/O USD asset"
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
    std::vector<VisualAssetPackV1>& output,
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
    std::vector<VisualAssetPackV1> candidate;
    xmlNode* robot = xmlDocGetRootElement(document);
    for (xmlNode* link : xmlChildren(robot, "link")) {
        const std::string linkName =
            xmlProperty(link, "name").value_or("");
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
            VisualAssetPackV1 pack;
            VisualAssetCookDiagnostics result =
                cookVisualAsset(source, pack, visualOptions);
            if (!result.succeeded()) {
                xmlFreeDoc(document);
                return result;
            }
            const auto body =
                options.linkBodyIndices.find(linkName);
            for (std::uint32_t instanceIndex = 0u;
                 instanceIndex < pack.instances.size();
                 ++instanceIndex) {
                MRVisualInstanceGPUV2& instance =
                    pack.instances[instanceIndex];
                if (body != options.linkBodyIndices.end()) {
                    instance.binding.y = body->second;
                    instance.binding.z =
                        MR_VISUAL_BINDING_ARTICULATED_LINK;
                    instance.identity.z = body->second;
                }
                VisualSymbolicBindingV1 binding;
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
