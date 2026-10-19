#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <numi/matter/shared.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>


static id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* p, NSUInteger n) {
    return [device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> makeZeroBuffer(id<MTLDevice> device, NSUInteger n) {
    std::vector<std::uint8_t> z(n, 0);
    return makeBuffer(device, z.data(), n);
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil) { std::cerr << "missing function " << [name UTF8String] << "\n"; return nil; }
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
    if (p == nil) std::cerr << "pipeline error " << [[error description] UTF8String] << "\n";
    return p;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: predicate_probe MatterSource.metal\n"; return 2; }
    @autoreleasepool {
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
        if (source == nil) { std::cerr << "source read failed: " << [[error description] UTF8String] << "\n"; return 2; }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) { std::cerr << "no Metal device\n"; return 2; }
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion4_0;
        options.optimizationLevel = MTLLibraryOptimizationLevelDefault;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) { std::cerr << "Metal source compile failed: " << [[error description] UTF8String] << "\n"; return 3; }
        auto build = pipeline(device, library, @"numi_matter_metal::nm_contact_build_surface_primitives");
        auto pair = pipeline(device, library, @"numi_matter_metal::nm_outer_contact_test_pair_eligibility");
        if (build == nil || pair == nil) return 4;

        NMMatterDispatchGPU dispatch{};
        dispatch.abiVersion = NM_MATTER_ABI_VERSION;
        dispatch.flags = NM_MATTER_STATIC_FEM_SURFACE_MAP;
        dispatch.environmentCount = 1;
        dispatch.objectCount = 2;
        dispatch.femNodeCount = 8;
        dispatch.tetrahedronCount = 2;
        dispatch.surfaceFaceCount = 3;
        dispatch.gravityAndTimestep.w = 0.001f;
        dispatch.numericalLimits = {0.001f, 1.0f, 1.0e-6f, 1.0e8f};
        NMMicrostepGPU micro{};
        micro.microtickCount = 1;
        micro.time = {0.001f, 1000.0f, 0.001f, 0.001f};

        std::array<NMContinuumObjectGPU, 2> objects{};
        for (std::uint32_t i = 0; i < objects.size(); ++i) {
            objects[i].representation = NM_REPRESENTATION_FEM;
            objects[i].flags = NM_OBJECT_ACTIVE;
            objects[i].solver = {0u, 0u, 0u, 0u};
        }
        std::array<NMTetrahedronGPU, 2> tets{};
        std::array<NMFEMNodeStateGPU, 8> nodes{};
        const std::array<std::array<float,3>,8> xyz{{
            {{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}},
            {{3,0,0}}, {{4,0,0}}, {{3,1,0}}, {{3,0,1}}
        }};
        for (std::uint32_t i = 0; i < 8; ++i) {
            nodes[i].positionAndMass = {xyz[i][0], xyz[i][1], xyz[i][2], 1.0f};
            nodes[i].velocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
            nodes[i].restAndFixed = {xyz[i][0], xyz[i][1], xyz[i][2], 0.0f};
        }
        for (std::uint32_t i = 0; i < 2; ++i) {
            tets[i].nodes = {4u*i, 4u*i+1u, 4u*i+2u, 4u*i+3u};
            tets[i].identity = {0u, i, 0u, NM_OBJECT_ACTIVE};
        }
        std::array<NMSchedulerStateGPU, 2> schedulers{};
        std::array<NMAdaptiveStateGPU, 2> adaptive{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            schedulers[i].numerical.w = 1.0f;
            schedulers[i].activeExponent = 0;
            adaptive[i].activeRepresentation = NM_REPRESENTATION_FEM;
        }
        std::array<NMFEMSurfaceFaceGPU, 3> faces{};
        faces[0].adjacency = {0u, NM_INVALID_INDEX, 0u, 0u};
        faces[0].sides = {3u, NM_INVALID_INDEX, 110u, 0u}; // outer lateral representative
        faces[1].adjacency = {0u, NM_INVALID_INDEX, 0u, 0u};
        faces[1].sides = {0u, NM_INVALID_INDEX, 111u, 0u}; // cap representative
        faces[2].adjacency = {1u, NM_INVALID_INDEX, 1u, 0u};
        faces[2].sides = {3u, NM_INVALID_INDEX, 210u, 0u}; // opposing FEM target

        id<MTLBuffer> bDispatch = makeBuffer(device, &dispatch, sizeof(dispatch));
        id<MTLBuffer> bMicro = makeBuffer(device, &micro, sizeof(micro));
        id<MTLBuffer> bObjects = makeBuffer(device, objects.data(), sizeof(objects));
        id<MTLBuffer> bTets = makeBuffer(device, tets.data(), sizeof(tets));
        id<MTLBuffer> bNodes = makeBuffer(device, nodes.data(), sizeof(nodes));
        id<MTLBuffer> bSchedulers = makeBuffer(device, schedulers.data(), sizeof(schedulers));
        id<MTLBuffer> bAdaptive = makeBuffer(device, adaptive.data(), sizeof(adaptive));
        id<MTLBuffer> bFaces = makeBuffer(device, faces.data(), sizeof(faces));
        id<MTLBuffer> bPrimitives = makeZeroBuffer(device, sizeof(NMContinuumSurfacePrimitiveGPU) * 3);
        id<MTLBuffer> bOut = makeZeroBuffer(device, sizeof(nm_uint4) * 2);
        id<MTLBuffer> bGridNodes = makeZeroBuffer(device, sizeof(NMGridNodeStateGPU));
        id<MTLBuffer> bGrids = makeZeroBuffer(device, sizeof(NMMPMGridGPU));
        id<MTLBuffer> bGridGenerations = makeZeroBuffer(device, sizeof(std::uint32_t));
        id<MTLBuffer> bIncidence = makeZeroBuffer(device, sizeof(std::uint32_t));
        id<MTLBuffer> bRanges = makeZeroBuffer(device, sizeof(NMIncidenceRangeGPU) * 8);
        id<MTLBuffer> bTopology = makeZeroBuffer(device, sizeof(NMFEMTopologyNodeGPU) * 8);
        id<MTLBuffer> bCohRefs = makeZeroBuffer(device, sizeof(nm_uint4));
        id<MTLBuffer> bCohFaces = makeZeroBuffer(device, sizeof(NMCohesiveFaceGPU));
        std::uint32_t includePrivateBase = 0;

        auto run = [&](bool maskCap) -> bool {
            auto* faceBytes = static_cast<NMFEMSurfaceFaceGPU*>(bFaces.contents);
            faceBytes[1].sides.w = maskCap ? NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT : 0u;
            id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
            [enc setComputePipelineState:build];
            [enc setBuffer:bDispatch offset:0 atIndex:0];
            [enc setBuffer:bMicro offset:0 atIndex:1];
            [enc setBuffer:bObjects offset:0 atIndex:2];
            [enc setBuffer:bTets offset:0 atIndex:3];
            [enc setBuffer:bNodes offset:0 atIndex:4];
            [enc setBuffer:bSchedulers offset:0 atIndex:5];
            [enc setBuffer:bGridNodes offset:0 atIndex:6];
            [enc setBuffer:bGrids offset:0 atIndex:7];
            [enc setBuffer:bGridGenerations offset:0 atIndex:8];
            [enc setBuffer:bAdaptive offset:0 atIndex:9];
            [enc setBuffer:bIncidence offset:0 atIndex:10];
            [enc setBuffer:bRanges offset:0 atIndex:11];
            [enc setBuffer:bPrimitives offset:0 atIndex:12];
            [enc setBuffer:bFaces offset:0 atIndex:13];
            [enc setBuffer:bNodes offset:0 atIndex:14];
            [enc setBytes:&includePrivateBase length:sizeof(includePrivateBase) atIndex:15];
            [enc dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc setComputePipelineState:pair];
            [enc setBuffer:bDispatch offset:0 atIndex:0];
            [enc setBuffer:bPrimitives offset:0 atIndex:1];
            [enc setBuffer:bObjects offset:0 atIndex:2];
            [enc setBuffer:bTopology offset:0 atIndex:3];
            [enc setBuffer:bOut offset:0 atIndex:4];
            [enc setBuffer:bRanges offset:0 atIndex:5];
            [enc setBuffer:bCohRefs offset:0 atIndex:6];
            [enc setBuffer:bCohFaces offset:0 atIndex:7];
            [enc dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                std::cerr << "Metal dispatch failed: " << [[command.error description] UTF8String] << "\n";
                return false;
            }
            auto* prim = static_cast<NMContinuumSurfacePrimitiveGPU*>(bPrimitives.contents);
            auto* out = static_cast<nm_uint4*>(bOut.contents);
            std::cout << "case=" << (maskCap ? "cap_masked" : "default_unmasked")
                      << " outer_primitive_flags=" << prim[0].identity.w
                      << " cap_primitive_flags=" << prim[1].identity.w
                      << " target_flags=" << prim[2].identity.w
                      << " outer_pair_fwd_rev=" << out[0].x << "," << out[0].y
                      << " cap_pair_fwd_rev=" << out[1].x << "," << out[1].y << "\n";
            const bool outerOk = out[0].x == 1u && out[0].y == 1u;
            const bool capOk = maskCap ? (out[1].x == 0u && out[1].y == 0u)
                                       : (out[1].x == 1u && out[1].y == 1u);
            const bool flagsOk = prim[0].identity.w == NM_TOPOLOGY_ACTIVE &&
                prim[2].identity.w == NM_TOPOLOGY_ACTIVE &&
                prim[1].identity.w == (maskCap ? NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT : NM_TOPOLOGY_ACTIVE);
            return outerOk && capOk && flagsOk;
        };
        const bool defaultOk = run(false);
        const bool selectorOk = run(true);
        std::cout << "result=" << (defaultOk && selectorOk ? "PASS" : "FAIL") << "\n";
        return defaultOk && selectorOk ? 0 : 1;
    }
}
