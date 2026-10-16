#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::vector<std::uint8_t> readExact(const std::string& path, std::size_t size) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open input: " + path);
    std::vector<std::uint8_t> bytes(size);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    require(input.gcount() == static_cast<std::streamsize>(size), "input has wrong byte count: " + path);
    char extra = 0;
    input.read(&extra, 1);
    require(input.gcount() == 0, "input exceeds expected byte count: " + path);
    return bytes;
}
id<MTLBuffer> makeBuffer(id<MTLDevice> device,
                         const std::vector<std::uint8_t>& bytes) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes.data()
        length:bytes.size() options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal input buffer allocation failed");
    return buffer;
}
void writeExclusive(const std::string& path, const void* bytes, std::size_t size) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(fd >= 0, "output must be fresh; open failed: " + path + " errno=" + std::to_string(errno));
    const auto* cursor = static_cast<const std::uint8_t*>(bytes);
    std::size_t remaining = size;
    while (remaining != 0u) {
        const ssize_t written = ::write(fd, cursor, remaining);
        if (written < 0 && errno == EINTR) continue;
        require(written > 0, "output write failed");
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
    require(::fsync(fd) == 0, "output fsync failed");
    require(::close(fd) == 0, "output close failed");
}
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 4, "usage: checker <metallib> <case-input-dir> <fresh-output.bin>");
            const std::string libraryPath(argv[1]);
            const std::string caseDir(argv[2]);
            const std::uint32_t contacts = 3u;
            const std::uint32_t trials = 8u;
            const std::uint32_t nodes = 36u;
            auto contactBytes = readExact(caseDir + "/contacts.bin", contacts * 432u);
            auto acceptedBytes = readExact(caseDir + "/accepted-nodes.bin", nodes * 80u);
            auto candidateBytes = readExact(caseDir + "/candidate-nodes.bin", nodes * 80u);
            auto correctionBytes = readExact(caseDir + "/correction.bin", nodes * 16u);
            auto alphaBytes = readExact(caseDir + "/alphas.bin", contacts * trials * sizeof(float));
            auto timestepBytes = readExact(caseDir + "/timestep.bin", sizeof(float));
            std::vector<std::uint8_t> output(std::size_t(contacts) * trials * 14u * 16u, 0u);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            NSURL* libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath.c_str()]];
            id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
            require(library != nil, error ? [[error localizedDescription] UTF8String] : "cannot load metallib");
            id<MTLFunction> function = [library newFunctionWithName:
                @"numi_matter_metal::nm_contact_barrier_trial_replay"];
            require(function != nil, "replay kernel missing from library");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline != nil, error ? [[error localizedDescription] UTF8String] : "cannot create pipeline");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "cannot create command queue");

            id<MTLBuffer> buffers[] = {
                makeBuffer(device, contactBytes), makeBuffer(device, acceptedBytes),
                makeBuffer(device, candidateBytes), makeBuffer(device, correctionBytes),
                makeBuffer(device, alphaBytes), makeBuffer(device, timestepBytes)
            };
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:output.size()
                options:MTLResourceStorageModeShared];
            require(outputBuffer != nil, "Metal output buffer allocation failed");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command != nil, "command buffer allocation failed");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder != nil, "compute encoder allocation failed");
            [encoder setComputePipelineState:pipeline];
            for (NSUInteger i = 0; i < 5; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            [encoder setBuffer:outputBuffer offset:0 atIndex:5];
            [encoder setBuffer:buffers[5] offset:0 atIndex:9];
            [encoder setBytes:&contacts length:sizeof(contacts) atIndex:6];
            [encoder setBytes:&trials length:sizeof(trials) atIndex:7];
            [encoder setBytes:&nodes length:sizeof(nodes) atIndex:8];
            const NSUInteger total = NSUInteger(contacts) * trials;
            const NSUInteger width = std::max<NSUInteger>(1u, std::min<NSUInteger>(
                pipeline.maxTotalThreadsPerThreadgroup, 32u));
            [encoder dispatchThreads:MTLSizeMake(total, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                command.error ? [[command.error localizedDescription] UTF8String] : "GPU replay command failed");
            std::memcpy(output.data(), outputBuffer.contents, output.size());
            writeExclusive(argv[3], output.data(), output.size());
            std::cout << "diagnostic_kernel_completed=1\n"
                      << "contacts=" << contacts << "\ntrials_per_contact=" << trials
                      << "\noutput_bytes=" << output.size()
                      << "\nphysical_transactions=0\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "contact barrier replay failed: " << exception.what() << "\n";
            return 2;
        }
    }
}
