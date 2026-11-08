#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/rod_gpu_shared.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 4 || (std::string(argv[3]) != "exact" &&
                          std::string(argv[3]) != "corrupt")) {
            std::cerr << "usage: roundtrip INPUT OUTPUT exact|corrupt\n";
            return 2;
        }
        const std::string mode(argv[3]);
        const std::filesystem::path destination(argv[2]);
        const std::filesystem::path temporary = destination.string() + ".tmp";
        if (std::filesystem::exists(destination) || std::filesystem::exists(temporary)) {
            std::cerr << "output_error=already_exists\n"; return 2;
        }
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() <= 0 ||
            static_cast<std::uint64_t>(input.tellg()) > 64ull * 1024ull * 1024ull) {
            std::cerr << "input_error=invalid_size\n"; return 3;
        }
        const std::size_t size = static_cast<std::size_t>(input.tellg());
        if (size % sizeof(MRRodToolWitnessGPU) != 0u) {
            std::cerr << "input_error=truncated_record\n"; return 3;
        }
        std::vector<std::uint8_t> expected(size);
        input.seekg(0);
        input.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(size));
        if (!input || static_cast<std::size_t>(input.gcount()) != size) {
            std::cerr << "input_error=truncated_read\n"; return 3;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || size > device.maxBufferLength) {
            std::cerr << "device_error=unavailable_or_capacity\n"; return 4;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> upload = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> privateArena = [device newBufferWithLength:size options:MTLResourceStorageModePrivate];
        id<MTLBuffer> readback = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (queue == nil || upload == nil || privateArena == nil || readback == nil ||
            upload.contents == nullptr || readback.contents == nullptr ||
            upload.length < size || privateArena.length < size || readback.length < size) {
            std::cerr << "device_error=allocation_or_mapping\n"; return 4;
        }
        std::memcpy(upload.contents, expected.data(), size);
        if (mode == "corrupt") static_cast<std::uint8_t*>(upload.contents)[size / 2u] ^= 1u;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            std::cerr << "device_error=encoder\n"; return 4;
        }
        [blit copyFromBuffer:upload sourceOffset:0 toBuffer:privateArena destinationOffset:0 size:size];
        [blit copyFromBuffer:privateArena sourceOffset:0 toBuffer:readback destinationOffset:0 size:size];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted || readback.contents == nullptr || readback.length < size) {
            std::cerr << "device_error=command_or_readback\n"; return 5;
        }
        const bool equal = std::memcmp(expected.data(), readback.contents, size) == 0;
        std::cout << "device_roundtrip mode=" << mode << " bytes=" << size
                  << " records=" << size / sizeof(MRRodToolWitnessGPU)
                  << " bytes_equal=" << equal << "\n";
        if (!equal) return 6;
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char*>(readback.contents), static_cast<std::streamsize>(size));
        output.close();
        if (!output || std::rename(temporary.c_str(), destination.c_str()) != 0) {
            std::cerr << "output_error=write_or_rename\n"; return 7;
        }
        return 0;
    }
}
