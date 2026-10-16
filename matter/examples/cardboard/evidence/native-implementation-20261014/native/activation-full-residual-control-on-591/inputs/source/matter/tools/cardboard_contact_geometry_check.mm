#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

#ifndef NUMI_CARDBOARD_GEOMETRY_METALLIB
#error NUMI_CARDBOARD_GEOMETRY_METALLIB must identify the built geometry library
#endif

namespace {
using Vector = std::array<double, 3>;
constexpr std::size_t kRows = 96u;
constexpr std::size_t kWordsPerRow = 64u;
constexpr std::size_t kBytes = kRows * kWordsPerRow * sizeof(float);
void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
Vector add(Vector a, Vector b) {
    for (unsigned i = 0u; i < 3u; ++i) a[i] += b[i];
    return a;
}
Vector subtract(Vector a, Vector b) {
    for (unsigned i = 0u; i < 3u; ++i) a[i] -= b[i];
    return a;
}
Vector scale(Vector a, double s) {
    for (double& x : a) x *= s;
    return a;
}
double dot(Vector a, Vector b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
double norm(Vector a) { return std::sqrt(dot(a, a)); }
Vector vectorAt(const float* row, unsigned slot) {
    return {row[slot*4u], row[slot*4u+1u], row[slot*4u+2u]};
}
double maximumDifference(Vector a, Vector b) {
    double value = 0.0;
    for (unsigned i = 0u; i < 3u; ++i)
        value = std::max(value, std::abs(a[i]-b[i]));
    return value;
}
double nearestError(const float* row) {
    const unsigned kind = static_cast<unsigned>(row[0]);
    const double size = row[1];
    require(kind < 6u && size > 0.0, "bad nearest-feature case");
    const Vector query = vectorAt(row, 1u);
    const Vector a = vectorAt(row, 2u), b = vectorAt(row, 3u);
    const Vector c = vectorAt(row, 4u), d = vectorAt(row, 5u);
    Vector parameters{}, first{}, second{};
    if (kind == 0u) {
        const double v = (query[0]-a[0])/(b[0]-a[0]);
        const double w = (query[1]-a[1])/(c[1]-a[1]);
        parameters = {1.0-v-w, v, w};
        first = {query[0], query[1], a[2]};
    } else if (kind == 1u) {
        const double v = (query[0]-a[0])/(b[0]-a[0]);
        parameters = {1.0-v, v, 0.0};
        first = {query[0], a[1], a[2]};
    } else if (kind == 2u) {
        parameters = {0.0, 0.0, 1.0};
        first = c;
    } else {
        double s = 0.0, t = 0.0;
        if (kind == 3u) {
            s = (c[0]-a[0])/(b[0]-a[0]);
            t = (a[1]-c[1])/(d[1]-c[1]);
        } else if (kind == 4u) {
            s = (c[0]-a[0])/(b[0]-a[0]);
        } else {
            t = (a[0]-c[0])/(d[0]-c[0]);
        }
        parameters = {s, t, 0.0};
        first = add(a, scale(subtract(b,a),s));
        second = add(c, scale(subtract(d,c),t));
    }
    double error = std::max(maximumDifference(vectorAt(row,7u),parameters),
        norm(subtract(vectorAt(row,9u),first))/size);
    if (kind >= 3u)
        error = std::max(error,norm(subtract(vectorAt(row,11u),second))/size);
    return error;
}
std::array<double, 2> mollifierError(const float* row) {
    const Vector first = vectorAt(row,1u), second = vectorAt(row,2u);
    const double firstLength = norm(first), secondLength = norm(second);
    require(firstLength > 0.0 && secondLength > 0.0, "invalid mollifier case");
    const Vector u = scale(first,1.0/firstLength), v = scale(second,1.0/secondLength);
    const double cosine = dot(u,v);
    const Vector cross{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    const double ratio = std::clamp(dot(cross,cross)/1.0e-3,0.0,1.0);
    const double expected = ratio*(2.0-ratio);
    Vector gradientFirst{}, gradientSecond{};
    if (ratio > 0.0 && ratio < 1.0) {
        const double multiplier = 4.0*(1.0-ratio)*cosine/1.0e-3;
        gradientFirst = scale(subtract(scale(u,cosine),v),multiplier/firstLength);
        gradientSecond = scale(subtract(scale(v,cosine),u),multiplier/secondLength);
    }
    const double gradientError = std::max(
        norm(subtract(vectorAt(row,3u),gradientFirst))/std::max(norm(gradientFirst),1.0/firstLength),
        norm(subtract(vectorAt(row,4u),gradientSecond))/std::max(norm(gradientSecond),1.0/secondLength));
    return {std::abs(row[20u]-expected),gradientError};
}
void writeExclusive(const std::string& path,const void* data,std::size_t bytes) {
    const int fd = ::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(fd >= 0,"output must be fresh");
    const auto* cursor = static_cast<const char*>(data);
    while (bytes != 0u) {
        const ssize_t count = ::write(fd,cursor,bytes);
        require(count > 0,"output write failed");
        cursor += count;
        bytes -= static_cast<std::size_t>(count);
    }
    require(::fsync(fd)==0,"output fsync failed");
    require(::close(fd)==0,"output close failed");
}
}

int main(int argc,const char** argv) {
    @autoreleasepool {
        try {
            require(argc==1 || (argc==3 && std::string(argv[1])=="--output"),
                "usage: cardboard-contact-geometry-check [--output fresh.bin]");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device!=nil,"no Metal device");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
                @NUMI_CARDBOARD_GEOMETRY_METALLIB] error:&error];
            require(library!=nil,"cannot load geometry library");
            id<MTLFunction> function = [library newFunctionWithName:
                @"numi_matter_metal::nm_contact_scaled_geometry_replay"];
            require(function!=nil,"missing geometry kernel");
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"cannot create geometry pipeline");
            id<MTLBuffer> output = [device newBufferWithLength:kBytes options:MTLResourceStorageModeShared];
            require(output!=nil,"geometry arena allocation failed");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue!=nil,"cannot create command queue");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command!=nil,"cannot create command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder!=nil,"cannot create encoder");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(kRows,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"geometry GPU command failed");
            const auto* words = static_cast<const float*>(output.contents);
            if (argc==3) writeExclusive(argv[2],words,kBytes);
            double nearestMaximum = 0.0, valueMaximum = 0.0, gradientMaximum = 0.0;
            for (std::size_t i = 0u; i < kRows; ++i) {
                const float* row = words+i*kWordsPerRow;
                for (unsigned j = 0u; j < kWordsPerRow; ++j)
                    require(std::isfinite(row[j]),"nonfinite geometry output");
                if (i < 48u) nearestMaximum = std::max(nearestMaximum,nearestError(row));
                else {
                    const auto errors = mollifierError(row);
                    valueMaximum = std::max(valueMaximum,errors[0]);
                    gradientMaximum = std::max(gradientMaximum,errors[1]);
                }
            }
            std::printf("{\"nearest_cases\":48,\"mollifier_cases\":48,\"nearest_error\":%.17g,\"mollifier_value_error\":%.17g,\"mollifier_gradient_error\":%.17g,\"physical_transactions\":0}\n",nearestMaximum,valueMaximum,gradientMaximum);
            require(nearestMaximum <= 1.0e-3 && valueMaximum <= 1.0e-3 && gradientMaximum <= 1.0e-3,
                "native contact geometry exceeded the 1e-3 analytical gate");
            return 0;
        } catch (const std::exception& ex) {
            std::fprintf(stderr,"cardboard contact geometry check failed: %s\n",ex.what());
            return 2;
        }
    }
}
