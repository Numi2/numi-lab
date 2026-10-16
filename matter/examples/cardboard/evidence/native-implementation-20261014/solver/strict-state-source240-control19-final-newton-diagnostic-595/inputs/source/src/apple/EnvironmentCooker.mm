#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "metalrobo/VisualPresentation.hpp"
#include "VisualKernelHashes.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ranges>
#include <sstream>
#include <span>
#include <string>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

VisualAssetCookDiagnostics rejectEnvironment(
    VisualAssetCookDiagnostics diagnostics,
    const VisualAssetCookStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

std::string text(NSString* value) {
    return value == nil || value.UTF8String == nullptr
        ? std::string{}
        : std::string{value.UTF8String};
}

std::string errorText(NSError* error) {
    return error == nil
        ? "unknown Apple framework error"
        : text(error.localizedDescription);
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
    return text(version);
}

std::string hexDigest(
    const std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH>& digest
) {
    std::ostringstream stream;
    stream << "sha256:";
    for (const std::uint8_t value : digest) {
        stream << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<unsigned>(value);
    }
    return stream.str();
}

std::string sha256(
    const std::span<const std::uint8_t> bytes
) {
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(
        bytes.data(),
        static_cast<CC_LONG>(bytes.size()),
        digest.data()
    );
    return hexDigest(digest);
}

std::string sha256File(
    const std::filesystem::path& path
) {
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
            CC_SHA256_Update(
                &context,
                buffer.data(),
                static_cast<CC_LONG>(count)
            );
        }
    }
    if (!stream.eof()) {
        return {};
    }
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return hexDigest(digest);
}

std::uint32_t highestPowerOfTwo(
    const std::uint32_t value
) {
    if (value == 0u) {
        return 0u;
    }
    return 1u << (31u - std::countl_zero(value));
}

std::uint32_t mipCount(const std::uint32_t size) {
    return size == 0u
        ? 0u
        : 32u - std::countl_zero(size);
}

std::uint32_t bytesPerPixel(
    const VisualTexturePixelFormatV2 format
) {
    return format == VisualTexturePixelFormatV2::rgba16Float
        ? 8u
        : 4u;
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name,
    std::string& message
) {
    id<MTLFunction> function =
        [library newFunctionWithName:name];
    if (function == nil) {
        message =
            "environment cook kernel is absent: " + text(name);
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function
                                              error:&error];
    if (result == nil) {
        message =
            "environment cook pipeline failed: " +
            errorText(error);
    }
    return result;
}

bool completed(
    id<MTLCommandBuffer> command,
    std::string& message
) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        message =
            "environment GPU command failed: " +
            errorText(command.error);
        return false;
    }
    return true;
}

void dispatchCube(
    id<MTLComputeCommandEncoder> encoder,
    const std::uint32_t size,
    id<MTLComputePipelineState> state
) {
    const NSUInteger width =
        std::min<NSUInteger>(
            8u,
            state.threadExecutionWidth
        );
    const NSUInteger height = std::min<NSUInteger>(
        8u,
        std::max<NSUInteger>(
            1u,
            state.maxTotalThreadsPerThreadgroup / width
        )
    );
    [encoder dispatchThreads:MTLSizeMake(size, size, 6u)
       threadsPerThreadgroup:MTLSizeMake(width, height, 1u)];
}

bool readTexture(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLTexture> source,
    const std::string& textureId,
    const VisualTexturePixelFormatV2 pixelFormat,
    VisualTextureImageV2& output,
    std::string& message
) {
    const std::uint32_t slices =
        source.textureType == MTLTextureTypeCube ? 6u : 1u;
    const std::uint32_t levels =
        static_cast<std::uint32_t>(source.mipmapLevelCount);
    const std::uint32_t pixelBytes =
        bytesPerPixel(pixelFormat);
    std::size_t maximumBytes = 0u;
    std::size_t totalBytes = 0u;
    for (std::uint32_t level = 0u; level < levels; ++level) {
        const std::uint32_t width =
            std::max<std::uint32_t>(
                static_cast<std::uint32_t>(source.width) >> level,
                1u
            );
        const std::uint32_t height =
            std::max<std::uint32_t>(
                static_cast<std::uint32_t>(source.height) >> level,
                1u
            );
        const std::uint32_t row =
            (width * pixelBytes + 255u) & ~255u;
        maximumBytes = std::max<std::size_t>(
            maximumBytes,
            static_cast<std::size_t>(row) * height
        );
        const std::size_t imageBytes =
            static_cast<std::size_t>(row) * height;
        if (imageBytes >
            std::numeric_limits<std::size_t>::max() /
                std::max(slices, 1u) ||
            imageBytes * slices >
                std::numeric_limits<std::size_t>::max() -
                    totalBytes) {
            message =
                "environment texture payload exceeds host address space";
            return false;
        }
        totalBytes += imageBytes * slices;
    }
    const std::array<id<MTLBuffer>, 2u> staging{
        [device
            newBufferWithLength:std::max<std::size_t>(
                                    maximumBytes,
                                    256u
                                )
                       options:MTLResourceStorageModeShared],
        [device
            newBufferWithLength:std::max<std::size_t>(
                                    maximumBytes,
                                    256u
                                )
                       options:MTLResourceStorageModeShared],
    };
    if (staging[0] == nil || staging[1] == nil) {
        message =
            "environment readback staging allocation failed";
        return false;
    }
    output = {};
    output.id = textureId;
    output.width = static_cast<std::uint32_t>(source.width);
    output.height = static_cast<std::uint32_t>(source.height);
    output.mipCount = levels;
    output.arrayLength = slices;
    output.pixelFormat = pixelFormat;
    output.dimension = slices == 6u
        ? VisualTextureDimensionV2::cube
        : VisualTextureDimensionV2::texture2D;
    output.data.reserve(totalBytes);
    std::array<id<MTLCommandBuffer>, 2u> pending{
        nil,
        nil,
    };
    std::array<VisualTextureSubresourceV2, 2u>
        pendingSubresources{};
    std::array<std::uint64_t, 2u> pendingSequences{
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
    };
    const auto drain = [&](const std::uint32_t slot) {
        id<MTLCommandBuffer> command = pending[slot];
        if (command == nil) {
            return true;
        }
        [command waitUntilCompleted];
        if (command.status !=
            MTLCommandBufferStatusCompleted) {
            message =
                "environment GPU readback failed: " +
                errorText(command.error);
            return false;
        }
        VisualTextureSubresourceV2 subresource =
            pendingSubresources[slot];
        subresource.dataOffset = output.data.size();
        const auto* first =
            static_cast<const std::uint8_t*>(
                staging[slot].contents
            );
        output.data.insert(
            output.data.end(),
            first,
            first + subresource.dataSize
        );
        output.subresources.push_back(subresource);
        pending[slot] = nil;
        pendingSequences[slot] =
            std::numeric_limits<std::uint64_t>::max();
        return true;
    };
    std::uint64_t submission = 0u;
    for (std::uint32_t slice = 0u;
         slice < slices;
         ++slice) {
        for (std::uint32_t level = 0u;
             level < levels;
             ++level) {
            const std::uint32_t width =
                std::max(output.width >> level, 1u);
            const std::uint32_t height =
                std::max(output.height >> level, 1u);
            const std::uint32_t row =
                (width * pixelBytes + 255u) & ~255u;
            const std::uint32_t imageBytes = row * height;
            const std::uint32_t slot =
                static_cast<std::uint32_t>(submission & 1u);
            if (!drain(slot)) {
                return false;
            }
            id<MTLBuffer> buffer = staging[slot];
            std::memset(buffer.contents, 0, imageBytes);
            id<MTLCommandBuffer> command =
                [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit =
                [command blitCommandEncoder];
            [blit
                copyFromTexture:source
                    sourceSlice:slice
                    sourceLevel:level
                   sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                     sourceSize:MTLSizeMake(width, height, 1u)
                       toBuffer:buffer
              destinationOffset:0u
         destinationBytesPerRow:row
       destinationBytesPerImage:imageBytes];
            [blit endEncoding];
            pending[slot] = command;
            pendingSubresources[slot] = {
                level,
                slice,
                width,
                height,
                0u,
                imageBytes,
                row,
                imageBytes,
            };
            pendingSequences[slot] = submission++;
            [command commit];
        }
    }
    while (pending[0] != nil || pending[1] != nil) {
        const std::uint32_t slot =
            pendingSequences[0] <= pendingSequences[1]
            ? 0u
            : 1u;
        if (!drain(slot)) {
            return false;
        }
    }
    output.contentHash = sha256(output.data);
    return output.valid(&message);
}

} // namespace

VisualAssetCookDiagnostics cookVisualEnvironment(
    const std::filesystem::path& source,
    VisualEnvironmentPackV2& output,
    const VisualEnvironmentCookOptions& options
) {
    VisualAssetCookDiagnostics diagnostics;
    @autoreleasepool {
        const std::string sourceHash = sha256File(source);
        if (sourceHash.empty()) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::ioFailure,
                "HDR environment source could not be read"
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
        if (extension != ".hdr" && extension != ".exr") {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::unsupportedFormat,
                "environment source must be Radiance HDR or OpenEXR"
            );
        }
        if (options.sourceColorSpace != "auto" &&
            options.sourceColorSpace != "linear-rec709" &&
            options.sourceColorSpace != "acescg") {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::invalidTexture,
                "source color space must be auto, linear-rec709, or acescg"
            );
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                "Metal device is unavailable for environment cooking"
            );
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                "Metal command queue is unavailable"
            );
        }
        NSURL* sourceUrl =
            [NSURL fileURLWithPath:@(source.string().c_str())];
        bool hasEmbeddedColorProfile = false;
        CGImageSourceRef imageSource =
            CGImageSourceCreateWithURL(
                (__bridge CFURLRef)sourceUrl,
                nullptr
            );
        if (imageSource != nullptr) {
            CFDictionaryRef properties =
                CGImageSourceCopyPropertiesAtIndex(
                    imageSource,
                    0u,
                    nullptr
                );
            if (properties != nullptr) {
                CFTypeRef profileName = CFDictionaryGetValue(
                    properties,
                    kCGImagePropertyProfileName
                );
                hasEmbeddedColorProfile =
                    profileName != nullptr &&
                    CFGetTypeID(profileName) ==
                        CFStringGetTypeID() &&
                    CFStringGetLength(
                        static_cast<CFStringRef>(profileName)
                    ) != 0;
                CFRelease(properties);
            }
            CFRelease(imageSource);
        }
        CGColorSpaceRef explicitSource = nullptr;
        if (options.sourceColorSpace == "linear-rec709") {
            explicitSource = CGColorSpaceCreateWithName(
                kCGColorSpaceExtendedLinearSRGB
            );
        } else if (options.sourceColorSpace == "acescg") {
            explicitSource = CGColorSpaceCreateWithName(
                kCGColorSpaceACESCGLinear
            );
        } else if (!hasEmbeddedColorProfile) {
            explicitSource = CGColorSpaceCreateWithName(
                kCGColorSpaceExtendedLinearSRGB
            );
        }
        NSDictionary* imageOptions = explicitSource == nullptr
            ? @{}
            : @{
                  kCIImageColorSpace:
                      (__bridge id)explicitSource,
              };
        CIImage* image = [CIImage
            imageWithContentsOfURL:sourceUrl
                          options:imageOptions];
        if (explicitSource != nullptr) {
            CGColorSpaceRelease(explicitSource);
        }
        if (image == nil || CGRectIsEmpty(image.extent)) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::invalidTexture,
                "Image I/O/Core Image could not decode the HDR source"
            );
        }
        const CGRect extent = CGRectIntegral(image.extent);
        const std::uint32_t sourceWidth =
            static_cast<std::uint32_t>(extent.size.width);
        const std::uint32_t sourceHeight =
            static_cast<std::uint32_t>(extent.size.height);
        if (sourceWidth == 0u || sourceHeight == 0u ||
            sourceWidth < sourceHeight) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::invalidTexture,
                "environment image must be a horizontal equirectangular map"
            );
        }
        image = [image imageByApplyingTransform:
            CGAffineTransformMakeTranslation(
                -extent.origin.x,
                -extent.origin.y
            )];
        CGColorSpaceRef linearRec709 =
            CGColorSpaceCreateWithName(
                kCGColorSpaceExtendedLinearSRGB
            );
        CIContext* context = [CIContext
            contextWithMTLCommandQueue:queue
                              options:@{
                                  kCIContextWorkingColorSpace:
                                      (__bridge id)linearRec709,
                                  kCIContextOutputColorSpace:
                                      (__bridge id)linearRec709,
                                  kCIContextWorkingFormat:
                                      @(kCIFormatRGBAh),
                                  kCIContextCacheIntermediates:
                                      @NO,
                              }];
        MTLTextureDescriptor* equirectDescriptor =
            [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:
                    MTLPixelFormatRGBA16Float
                                         width:sourceWidth
                                        height:sourceHeight
                                     mipmapped:NO];
        equirectDescriptor.storageMode =
            MTLStorageModePrivate;
        equirectDescriptor.usage =
            MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite;
        id<MTLTexture> equirect =
            [device newTextureWithDescriptor:equirectDescriptor];
        if (equirect == nil) {
            CGColorSpaceRelease(linearRec709);
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::capacityOverflow,
                "HDR equirectangular texture allocation failed"
            );
        }
        id<MTLCommandBuffer> imageCommand =
            [queue commandBuffer];
        [context
            render:image
            toMTLTexture:equirect
            commandBuffer:imageCommand
            bounds:CGRectMake(
                0.0,
                0.0,
                sourceWidth,
                sourceHeight
            )
            colorSpace:linearRec709];
        CGColorSpaceRelease(linearRec709);
        std::string message;
        if (!completed(imageCommand, message)) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                std::move(message)
            );
        }

        const std::uint32_t derivedFaceSize = std::clamp(
            highestPowerOfTwo(
                std::max(sourceWidth / 4u, 1u)
            ),
            64u,
            1024u
        );
        const std::uint32_t faceSize =
            options.faceSize == 0u
            ? derivedFaceSize
            : highestPowerOfTwo(options.faceSize);
        if (faceSize < 16u || faceSize > 1024u) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::invalidTexture,
                "environment face size must resolve to 16..1024"
            );
        }
        const std::filesystem::path metallibPath{
            METALROBO_DEFAULT_METALLIB
        };
        NSError* libraryError = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:
                [NSURL
                    fileURLWithPath:
                        @(metallibPath.string().c_str())]
                       error:&libraryError];
        if (library == nil) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                "environment metallib could not be loaded: " +
                    errorText(libraryError)
            );
        }
        id<MTLComputePipelineState> equirectPipeline =
            pipeline(
                device,
                library,
                @"mr_environment_equirect_to_cube",
                message
            );
        id<MTLComputePipelineState> diffusePipeline =
            pipeline(
                device,
                library,
                @"mr_environment_diffuse_irradiance",
                message
            );
        id<MTLComputePipelineState> specularPipeline =
            pipeline(
                device,
                library,
                @"mr_environment_prefilter_specular",
                message
            );
        id<MTLComputePipelineState> brdfPipeline =
            pipeline(
                device,
                library,
                @"mr_environment_integrate_brdf",
                message
            );
        if (equirectPipeline == nil || diffusePipeline == nil ||
            specularPipeline == nil || brdfPipeline == nil) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                std::move(message)
            );
        }

        MTLTextureDescriptor* sourceCubeDescriptor =
            [MTLTextureDescriptor
                textureCubeDescriptorWithPixelFormat:
                    MTLPixelFormatRGBA16Float
                                            size:faceSize
                                       mipmapped:YES];
        sourceCubeDescriptor.storageMode = MTLStorageModePrivate;
        sourceCubeDescriptor.usage =
            MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite |
            MTLTextureUsagePixelFormatView;
        id<MTLTexture> sourceCube =
            [device newTextureWithDescriptor:sourceCubeDescriptor];
        MTLTextureDescriptor* diffuseDescriptor =
            [MTLTextureDescriptor
                textureCubeDescriptorWithPixelFormat:
                    MTLPixelFormatRG11B10Float
                                            size:64u
                                       mipmapped:NO];
        diffuseDescriptor.storageMode = MTLStorageModePrivate;
        diffuseDescriptor.usage =
            MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite;
        id<MTLTexture> diffuse =
            [device newTextureWithDescriptor:diffuseDescriptor];
        MTLTextureDescriptor* specularDescriptor =
            [MTLTextureDescriptor
                textureCubeDescriptorWithPixelFormat:
                    MTLPixelFormatRG11B10Float
                                            size:faceSize
                                       mipmapped:YES];
        specularDescriptor.storageMode = MTLStorageModePrivate;
        specularDescriptor.usage =
            MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite |
            MTLTextureUsagePixelFormatView;
        id<MTLTexture> specular =
            [device newTextureWithDescriptor:specularDescriptor];
        MTLTextureDescriptor* brdfDescriptor =
            [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:
                    MTLPixelFormatRG16Float
                                         width:256u
                                        height:256u
                                     mipmapped:NO];
        brdfDescriptor.storageMode = MTLStorageModePrivate;
        brdfDescriptor.usage =
            MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite;
        id<MTLTexture> brdf =
            [device newTextureWithDescriptor:brdfDescriptor];
        if (sourceCube == nil || diffuse == nil ||
            specular == nil || brdf == nil) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::capacityOverflow,
                "environment derived texture allocation failed"
            );
        }

        id<MTLCommandBuffer> cubeCommand =
            [queue commandBuffer];
        id<MTLComputeCommandEncoder> cubeEncoder =
            [cubeCommand computeCommandEncoder];
        [cubeEncoder setComputePipelineState:equirectPipeline];
        [cubeEncoder setTexture:equirect atIndex:0u];
        [cubeEncoder setTexture:sourceCube atIndex:1u];
        dispatchCube(cubeEncoder, faceSize, equirectPipeline);
        [cubeEncoder endEncoding];
        id<MTLBlitCommandEncoder> mipBlit =
            [cubeCommand blitCommandEncoder];
        [mipBlit generateMipmapsForTexture:sourceCube];
        [mipBlit endEncoding];
        if (!completed(cubeCommand, message)) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                std::move(message)
            );
        }
        equirect = nil;
        [context clearCaches];
        image = nil;

        id<MTLCommandBuffer> derivedCommand =
            [queue commandBuffer];
        id<MTLComputeCommandEncoder> convolution =
            [derivedCommand computeCommandEncoder];
        std::uint32_t samples = 1024u;
        [convolution setComputePipelineState:diffusePipeline];
        [convolution setTexture:sourceCube atIndex:0u];
        [convolution setTexture:diffuse atIndex:1u];
        [convolution
            setBytes:&samples
              length:sizeof(samples)
             atIndex:0u];
        dispatchCube(convolution, 64u, diffusePipeline);

        const std::uint32_t specularMips =
            mipCount(faceSize);
        NSMutableArray<id<MTLTexture>>* specularViews =
            [[NSMutableArray alloc]
                initWithCapacity:specularMips];
        for (std::uint32_t level = 0u;
             level < specularMips;
             ++level) {
            const std::uint32_t levelSize =
                std::max(faceSize >> level, 1u);
            const std::uint32_t sampleCount =
                level == 0u
                ? 1u
                : std::clamp(
                      64u << std::min(level - 1u, 4u),
                      64u,
                      1024u
                  );
            const float parameters[4]{
                static_cast<float>(levelSize),
                static_cast<float>(sampleCount),
                specularMips <= 1u
                    ? 0.0f
                    : static_cast<float>(level) /
                          static_cast<float>(
                              specularMips - 1u
                          ),
                static_cast<float>(faceSize),
            };
            id<MTLTexture> view = [specular
                newTextureViewWithPixelFormat:
                    MTLPixelFormatRG11B10Float
                                textureType:MTLTextureTypeCube
                                     levels:NSMakeRange(level, 1u)
                                     slices:NSMakeRange(0u, 6u)];
            [specularViews addObject:view];
            [convolution
                setComputePipelineState:specularPipeline];
            [convolution setTexture:sourceCube atIndex:0u];
            [convolution setTexture:view atIndex:1u];
            [convolution
                setBytes:parameters
                  length:sizeof(parameters)
                 atIndex:0u];
            dispatchCube(
                convolution,
                levelSize,
                specularPipeline
            );
        }
        samples = 1024u;
        [convolution setComputePipelineState:brdfPipeline];
        [convolution setTexture:brdf atIndex:0u];
        [convolution
            setBytes:&samples
              length:sizeof(samples)
             atIndex:0u];
        const MTLSize brdfThreads =
            MTLSizeMake(8u, 8u, 1u);
        [convolution
            dispatchThreads:MTLSizeMake(256u, 256u, 1u)
            threadsPerThreadgroup:brdfThreads];
        [convolution endEncoding];
        if (!completed(derivedCommand, message)) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::internalFailure,
                std::move(message)
            );
        }
        sourceCube = nil;
        specularViews = nil;

        VisualEnvironmentPackV2 candidate;
        candidate.id = options.id.empty()
            ? source.stem().string()
            : options.id;
        candidate.sourceUri = source.string();
        candidate.sourceContentHash = sourceHash;
        candidate.sourceColorSpace =
            options.sourceColorSpace == "auto"
            ? hasEmbeddedColorProfile
                ? "embedded-profile"
                : "linear-rec709"
            : options.sourceColorSpace;
        candidate.preprocessingProvenance =
            "metalrobo_environment_cook/v3;"
            "working=extended-linear-rec709;"
            "diffuse=cosine-hammersley-1024;"
            "specular=heitz-vndf-solid-angle-lod;"
            "brdf=split-sum-ggx-1024;"
            "storage=rg11b10f+rg16f;"
            "coreimage=" + frameworkVersion(CIContext.class) +
            ";sdk=" +
            std::to_string(__MAC_OS_X_VERSION_MAX_ALLOWED) +
            ";kernel=" METALROBO_ENVIRONMENT_KERNEL_HASH +
            ";samples=diffuse:1024,specular:64-1024,"
            "dfg:1024";
        candidate.specularFaceSize = faceSize;
        candidate.diffuseFaceSize = 64u;
        candidate.brdfLutSize = 256u;
        if (!readTexture(
                device,
                queue,
                diffuse,
                candidate.id + ".diffuse",
                VisualTexturePixelFormatV2::rg11b10Float,
                candidate.diffuseIrradiance,
                message
            ) ||
            !readTexture(
                device,
                queue,
                specular,
                candidate.id + ".specular",
                VisualTexturePixelFormatV2::rg11b10Float,
                candidate.prefilteredSpecular,
                message
            ) ||
            !readTexture(
                device,
                queue,
                brdf,
                "metalrobo.dfg.ggxcorr.v1",
                VisualTexturePixelFormatV2::rg16Float,
                candidate.brdfLut,
                message
            )) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::writeFailure,
                std::move(message)
            );
        }
        diffuse = nil;
        specular = nil;
        brdf = nil;
        candidate.contentHash =
            computeVisualEnvironmentPackContentHash(candidate);
        if (!candidate.valid(&message)) {
            return rejectEnvironment(
                std::move(diagnostics),
                VisualAssetCookStatus::invalidTexture,
                std::move(message)
            );
        }
        diagnostics.textureCount = 3u;
        diagnostics.sourceHash = candidate.sourceContentHash;
        diagnostics.packHash = candidate.contentHash;
        output = std::move(candidate);
        return diagnostics;
    }
}

} // namespace metalrobo
