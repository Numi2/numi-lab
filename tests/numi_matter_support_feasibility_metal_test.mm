#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include "numi/matter/support_feasibility.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void complete(id<MTLCommandBuffer> command) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
    [command commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC))) {
        std::cerr << "Production support-kernel regression timed out\n";
        std::_Exit(2);
    }
    require(command.status == MTLCommandBufferStatusCompleted, "Metal command failed");
}

void exercise(id<MTLDevice> device, id<MTLLibrary> library, float impulse, unsigned row) {
    constexpr unsigned environments = 2, contacts = 33;
    require(row < contacts, "Test row out of range");
    std::array<id<MTLBuffer>, 31> b{};
    for (auto& buffer : b) {
        buffer = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
        require(buffer != nil, "Buffer allocation failed");
        std::memset(buffer.contents, 0, buffer.length);
    }
    auto* dispatch = static_cast<NMMatterDispatchGPU*>(b[0].contents);
    dispatch->environmentCount = environments;
    auto* layout = static_cast<NMFGMRESLayoutGPU*>(b[30].contents);
    layout->supportContactCount = contacts;
    layout->unknownCount = environments * contacts;
    auto* solver = static_cast<NMMixedSolverGPU*>(b[1].contents);
    solver->regularization.x = 0.001f;
    solver->residualTolerances.x = 0.001f;
    auto* residual = static_cast<nm_float4*>(b[3].contents);
    auto* states = static_cast<NMFGMRESStateGPU*>(b[8].contents);
    for (unsigned environment = 0; environment < environments; ++environment) {
        residual[environment * contacts].x = 0.0002f;
        states[environment].nonlinear = {0.0002f, 0.0002f, 0.0f, 0.0f};
    }
    *static_cast<unsigned*>(b[16].contents) = 2; // Completed earlier corrections.
    *static_cast<float*>(b[15].contents) = 0.25f;
    auto* histories = static_cast<nm_float4*>(b[19].contents);
    histories[contacts + row].w = impulse;
    const bool feasible = numi_matter_contact::normalImpulseFeasible(impulse);

    // Use the production argument-layout anchor, including otherwise unused
    // contact pointers. No body, FEM, or MPM rows are fabricated by this test.
    id<MTLFunction> anchor = [library newFunctionWithName:@"nm_primal_contact_argument_layout"];
    require(anchor != nil, "Contact argument-layout kernel missing");
    id<MTLArgumentEncoder> arguments = [anchor newArgumentEncoderWithBufferIndex:0];
    require(arguments != nil && arguments.encodedLength <= b[13].length,
            "Contact argument layout does not fit");
    [arguments setArgumentBuffer:b[13] offset:0];
    for (unsigned index = 0; index < 15; ++index)
        [arguments setBuffer:b[20] offset:0 atIndex:index];

    NSError* error = nil;
    id<MTLFunction> begin = [library newFunctionWithName:@"nm_fgmres_begin"];
    require(begin != nil, "Production FGMRES begin kernel missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:begin error:&error];
    require(pipeline != nil && pipeline.threadExecutionWidth == 32, "FGMRES pipeline unavailable");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "Command queue unavailable");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil, "Command allocation failed");
    [encoder setComputePipelineState:pipeline];
    for (unsigned index = 0; index < 20; ++index)
        [encoder setBuffer:b[index] offset:0 atIndex:index];
    [encoder setBuffer:b[30] offset:0 atIndex:30];
    [encoder useResource:b[20] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(environments, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    complete(command);
    require(states[0].nonlinear.w == 1.0f, "Feasible neighboring environment did not converge");
    require((states[1].nonlinear.w == 1.0f) == feasible,
            "Production Newton stopping disagrees with support feasibility");
    require((states[1].diagnostics.z == 1.0f) == feasible,
            "Linear solve disabled for an infeasible residual-small iterate");

    id<MTLFunction> certify = [library newFunctionWithName:@"nm_human_support_certify"];
    require(certify != nil, "Production support certificate kernel missing");
    pipeline = [device newComputePipelineStateWithFunction:certify error:&error];
    require(pipeline != nil, "Certificate pipeline unavailable");
    command = [queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    const std::array<unsigned, 6> map{0, 1, 3, 8, 9, 19};
    for (unsigned index = 0; index < map.size(); ++index)
        [encoder setBuffer:b[map[index]] offset:0 atIndex:index];
    [encoder setBuffer:b[30] offset:0 atIndex:30];
    [encoder dispatchThreadgroups:MTLSizeMake(environments, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    complete(command);
    const auto* status = static_cast<const NMMatterStatusGPU*>(b[9].contents);
    require(status[0].code == 0, "Certificate contaminated neighboring environment");
    require((status[1].code == 0) == feasible, "Final certificate predicate changed");
    if (!feasible) {
        require(status[1].code == NM_STATUS_CONTACT_FAILURE && status[1].failingIndex == row,
                "Certificate lost failing contact row");
        require(status[1].diagnostics.y == impulse ||
                (std::isnan(status[1].diagnostics.y) && std::isnan(impulse)),
                "Certificate lost invalid impulse value");
    }
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: support-feasibility-metal /path/NumiMatter.metallib");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::cerr << "UNVERIFIED: no Metal device; no production kernel executed\n";
                return 77;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library != nil, "Cannot load production Matter metallib");
            const float infinity = std::numeric_limits<float>::infinity();
            const std::array<float, 8> values{0.0f, 1.0f, -1.0e-7f,
                std::nextafter(-1.0e-7f, -infinity), -0.000127762556f,
                infinity, -infinity, std::numeric_limits<float>::quiet_NaN()};
            for (float value : values)
                for (unsigned row : {0u, 31u, 32u}) exercise(device, library, value, row);
            std::cout << "PASS: 24 production-kernel cases, two environments each; "
                         "stopping/certificate agreement and failure diagnostics\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
