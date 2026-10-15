#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "numi/matter/shared.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct InputHeader {
    std::uint32_t version;
    std::uint32_t nodeCount;
    std::uint32_t tetrahedronCount;
    std::uint32_t modeCount;
    std::uint32_t patchCount;
    std::uint32_t patchNodeCount;
    std::uint32_t dofCount;
    std::uint32_t matrixFloatCount;
    std::uint32_t pivotCount;
    std::uint32_t sourceEnvironmentCount;
    std::uint32_t selectedEnvironment;
};

struct ArrayView {
    const char* data = nullptr;
    std::size_t size = 0u;
};

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<char> readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    need(stream.good(), "cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string sha256(const char* bytes, const std::size_t count) {
    need(count <= static_cast<std::size_t>(UINT32_MAX), "SHA input exceeds CommonCrypto size");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(reinterpret_cast<const unsigned char*>(bytes),
                   static_cast<CC_LONG>(count), digest) != nullptr,
         "CommonCrypto SHA-256 failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(2u * CC_SHA256_DIGEST_LENGTH);
    for (const unsigned char value : digest) {
        result.push_back(hex[value >> 4]);
        result.push_back(hex[value & 0x0fu]);
    }
    return result;
}

template <typename T>
ArrayView takeArray(const std::vector<char>& input, std::size_t& cursor,
                    const std::uint64_t count, const char* label) {
    need(count <= SIZE_MAX / sizeof(T), std::string(label) + " byte count overflow");
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
    need(cursor <= input.size() && bytes <= input.size() - cursor,
         std::string("truncated input array ") + label);
    ArrayView result{input.data() + cursor, bytes};
    cursor += bytes;
    return result;
}

id<MTLBuffer> inputBuffer(id<MTLDevice> device, const ArrayView view,
                          const char* label) {
    need(view.size > 0u, std::string("empty input buffer ") + label);
    id<MTLBuffer> buffer = [device newBufferWithBytes:view.data
                                               length:view.size
                                              options:MTLResourceStorageModeShared];
    need(buffer != nil, std::string("Metal input buffer allocation failed: ") + label);
    return buffer;
}

id<MTLBuffer> outputBuffer(id<MTLDevice> device, const std::size_t bytes,
                           const char* label) {
    need(bytes > 0u, std::string("zero-sized output buffer ") + label);
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    need(buffer != nil, std::string("Metal output buffer allocation failed: ") + label);
    return buffer;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         NSString* unqualified) {
    NSString* qualified = [@"numi_matter_metal::" stringByAppendingString:unqualified];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    need(function != nil, "missing production function " + std::string(qualified.UTF8String));
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    const char* localizedError = error.localizedDescription != nil
        ? error.localizedDescription.UTF8String : "unknown Metal error";
    need(pipeline != nil,
         "cannot create production pipeline " + std::string(qualified.UTF8String) +
         ": " + localizedError);
    return pipeline;
}

void setCount(id<MTLComputeCommandEncoder> encoder, const std::uint32_t value,
              const NSUInteger index) {
    [encoder setBytes:&value length:sizeof(value) atIndex:index];
}

void setBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer,
               const NSUInteger index) {
    [encoder setBuffer:buffer offset:0u atIndex:index];
}

void appendBytes(std::vector<char>& output, const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    output.insert(output.end(), bytes, bytes + size);
}

void writeFile(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    need(stream.good(), "cannot write " + path.string());
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    need(stream.good(), "failed closing " + path.string());
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            need(argc == 4,
                 "usage: selector5_graph_kernels INPUT.bin OUTPUT.bin NumiMatterGraphSchwarz.metallib");
            const std::filesystem::path inputPath(argv[1]);
            const std::filesystem::path outputPath(argv[2]);
            const std::filesystem::path metallibPath(argv[3]);
            const std::vector<char> input = readFile(inputPath);
            need(input.size() >= 8u + sizeof(InputHeader), "input bundle is too short");
            need(std::memcmp(input.data(), "NMSGKC1\0", 8u) == 0,
                 "input bundle magic mismatch");
            InputHeader header{};
            std::memcpy(&header, input.data() + 8u, sizeof(header));
            need(header.version == 1u && header.nodeCount > 0u &&
                 header.tetrahedronCount > 0u && header.modeCount > 0u &&
                 header.patchCount > 0u && header.patchNodeCount > 0u &&
                 header.dofCount > 0u && header.matrixFloatCount > 0u &&
                 header.pivotCount > 0u && header.sourceEnvironmentCount == 2u &&
                 header.selectedEnvironment == 0u,
                 "input dimensions or source selection are invalid");

            std::size_t cursor = 8u + sizeof(InputHeader);
            const ArrayView rapLUView = takeArray<float>(input, cursor,
                static_cast<std::uint64_t>(header.modeCount) * header.modeCount, "rap_lu");
            const ArrayView rapPivotsView = takeArray<std::uint32_t>(input, cursor,
                header.modeCount, "rap_pivots");
            const ArrayView rapValidView = takeArray<std::uint32_t>(input, cursor, 1u, "rap_valid");
            const ArrayView basisView = takeArray<nm_float4>(input, cursor, header.nodeCount, "basis");
            const ArrayView regionalBasisView = takeArray<nm_float4>(input, cursor,
                static_cast<std::uint64_t>(header.modeCount) * header.nodeCount, "regional_basis");
            const ArrayView regionalAPView = takeArray<nm_float4>(input, cursor,
                static_cast<std::uint64_t>(header.modeCount) * header.nodeCount, "regional_ap");
            const ArrayView patchesView = takeArray<nm_uint4>(input, cursor, header.patchCount, "patches");
            const ArrayView patchNodesView = takeArray<std::uint32_t>(input, cursor,
                header.patchNodeCount, "patch_nodes");
            const ArrayView nodePatchOffsetsView = takeArray<std::uint32_t>(input, cursor,
                static_cast<std::uint64_t>(header.nodeCount) + 1u, "node_patch_offsets");
            const ArrayView nodePatchValuesView = takeArray<std::uint32_t>(input, cursor,
                header.patchNodeCount, "node_patch_values");
            const ArrayView nodeCoverageView = takeArray<std::uint32_t>(input, cursor,
                header.nodeCount, "node_coverage");
            const ArrayView patchMatricesView = takeArray<float>(input, cursor,
                header.matrixFloatCount, "patch_matrices");
            need(cursor == input.size(), "input bundle has trailing or unaccounted bytes");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            need(device != nil, "no default Metal device");
            NSError* libraryError = nil;
            NSString* metallibURLPath = [NSString stringWithUTF8String:metallibPath.c_str()];
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:metallibURLPath] error:&libraryError];
            const char* localizedLibraryError = libraryError.localizedDescription != nil
                ? libraryError.localizedDescription.UTF8String : "unknown Metal error";
            need(library != nil,
                 std::string("cannot load graph-Schwarz metallib: ") +
                 localizedLibraryError);

            const auto projectPipeline = makePipeline(device, library,
                @"nm_fgmres_regional_project_rhs");
            const auto rapSolvePipeline = makePipeline(device, library,
                @"nm_fgmres_regional_solve_rap");
            const auto factorPipeline = makePipeline(device, library,
                @"nm_fgmres_graph_factor_patches");
            const auto solvePipeline = makePipeline(device, library,
                @"nm_fgmres_graph_solve_patches");
            const auto applyPipeline = makePipeline(device, library,
                @"nm_fgmres_graph_apply_patches");

            id<MTLBuffer> dispatchBuffer = outputBuffer(device,
                sizeof(NMMatterDispatchGPU), "dispatch");
            NMMatterDispatchGPU dispatch{};
            dispatch.abiVersion = NM_MATTER_ABI_VERSION;
            dispatch.environmentCount = 1u;
            dispatch.femNodeCount = header.nodeCount;
            dispatch.tetrahedronCount = header.tetrahedronCount;
            std::memcpy(dispatchBuffer.contents, &dispatch, sizeof(dispatch));

            id<MTLBuffer> rapLU = inputBuffer(device, rapLUView, "rap_lu");
            id<MTLBuffer> rapPivots = inputBuffer(device, rapPivotsView, "rap_pivots");
            id<MTLBuffer> rapValid = inputBuffer(device, rapValidView, "rap_valid");
            id<MTLBuffer> basis = inputBuffer(device, basisView, "basis");
            id<MTLBuffer> regionalBasis = inputBuffer(device, regionalBasisView, "regional_basis");
            id<MTLBuffer> regionalAP = inputBuffer(device, regionalAPView, "regional_ap");
            id<MTLBuffer> patches = inputBuffer(device, patchesView, "patches");
            id<MTLBuffer> patchNodes = inputBuffer(device, patchNodesView, "patch_nodes");
            id<MTLBuffer> nodePatchOffsets = inputBuffer(device, nodePatchOffsetsView,
                                                          "node_patch_offsets");
            id<MTLBuffer> nodePatchValues = inputBuffer(device, nodePatchValuesView,
                                                         "node_patch_values");
            id<MTLBuffer> nodeCoverage = inputBuffer(device, nodeCoverageView, "node_coverage");
            id<MTLBuffer> patchMatrices = inputBuffer(device, patchMatricesView, "patch_matrices");

            id<MTLBuffer> coarseRHS = outputBuffer(device,
                sizeof(float) * header.modeCount, "coarse_rhs");
            id<MTLBuffer> coefficients = outputBuffer(device,
                sizeof(float) * header.modeCount, "coefficients");
            id<MTLBuffer> pivots = outputBuffer(device,
                sizeof(std::uint32_t) * header.pivotCount, "patch_pivots");
            std::memset(pivots.contents, 0, pivots.length);
            id<MTLBuffer> factorValid = outputBuffer(device,
                sizeof(std::uint32_t) * header.patchCount, "patch_factor_valid");
            std::memset(factorValid.contents, 0, factorValid.length);
            id<MTLBuffer> corrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "patch_corrections");
            id<MTLBuffer> preconditioned = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "preconditioned");
            id<MTLBuffer> statusBuffer = outputBuffer(device,
                sizeof(NMMatterStatusGPU), "status");
            std::memset(statusBuffer.contents, 0, statusBuffer.length);
            id<MTLBuffer> statesBuffer = outputBuffer(device,
                sizeof(NMFGMRESStateGPU), "fgmres_state");
            std::memset(statesBuffer.contents, 0, statesBuffer.length);

            id<MTLCommandQueue> queue = [device newCommandQueue];
            need(queue != nil, "cannot create Metal command queue");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            need(command != nil, "cannot create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            need(encoder != nil, "cannot create Metal compute encoder");

            // Production coarse projection c0 = P^T b.
            [encoder setComputePipelineState:projectPipeline];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, regionalBasis, 1u);
            setBuffer(encoder, basis, 2u);
            setBuffer(encoder, coarseRHS, 3u);
            setCount(encoder, header.modeCount, 4u);
            [encoder dispatchThreadgroups:MTLSizeMake(header.modeCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Production pivoted RAP solve c = (P^T A P)^-1 P^T b.
            [encoder setComputePipelineState:rapSolvePipeline];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, rapLU, 1u);
            setBuffer(encoder, rapPivots, 2u);
            setBuffer(encoder, rapValid, 3u);
            setBuffer(encoder, coarseRHS, 4u);
            setBuffer(encoder, coefficients, 5u);
            setCount(encoder, header.modeCount, 6u);
            [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Production in-place pivoted LU for each actual-mesh principal patch.
            [encoder setComputePipelineState:factorPipeline];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, patches, 1u);
            setBuffer(encoder, patchMatrices, 2u);
            setBuffer(encoder, pivots, 3u);
            setBuffer(encoder, factorValid, 4u);
            setBuffer(encoder, statusBuffer, 5u);
            setCount(encoder, header.patchCount, 6u);
            setCount(encoder, header.matrixFloatCount, 7u);
            setCount(encoder, header.pivotCount, 8u);
            [encoder dispatchThreadgroups:MTLSizeMake(header.patchCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Production local correction solve against b - AP c.
            [encoder setComputePipelineState:solvePipeline];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, patches, 1u);
            setBuffer(encoder, patchNodes, 2u);
            setBuffer(encoder, basis, 3u);
            setBuffer(encoder, regionalBasis, 4u);
            setBuffer(encoder, regionalAP, 5u);
            setBuffer(encoder, coefficients, 6u);
            setBuffer(encoder, patchMatrices, 7u);
            setBuffer(encoder, pivots, 8u);
            setBuffer(encoder, factorValid, 9u);
            setBuffer(encoder, corrections, 10u);
            setBuffer(encoder, statusBuffer, 11u);
            setCount(encoder, header.patchCount, 12u);
            setCount(encoder, header.patchNodeCount, 13u);
            setCount(encoder, header.matrixFloatCount, 14u);
            setCount(encoder, header.pivotCount, 15u);
            setCount(encoder, header.modeCount, 16u);
            [encoder dispatchThreadgroups:MTLSizeMake(header.patchCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Production global gather: coarse P c plus overlap-averaged patch corrections.
            [encoder setComputePipelineState:applyPipeline];
            setBuffer(encoder, dispatchBuffer, 0u);
            setBuffer(encoder, patchNodes, 1u);
            setBuffer(encoder, nodePatchOffsets, 2u);
            setBuffer(encoder, nodePatchValues, 3u);
            setBuffer(encoder, nodeCoverage, 4u);
            setBuffer(encoder, regionalBasis, 5u);
            setBuffer(encoder, coefficients, 6u);
            setBuffer(encoder, corrections, 7u);
            setBuffer(encoder, preconditioned, 8u);
            setBuffer(encoder, statesBuffer, 9u);
            setCount(encoder, header.patchNodeCount, 10u);
            setCount(encoder, header.modeCount, 11u);
            [encoder dispatchThreads:MTLSizeMake(header.nodeCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            const char* localizedCommandError = command.error.localizedDescription != nil
                ? command.error.localizedDescription.UTF8String : "unknown command error";
            need(command.status == MTLCommandBufferStatusCompleted,
                 std::string("Metal graph kernel command failed: ") + localizedCommandError);

            NMMatterStatusGPU status{};
            std::memcpy(&status, statusBuffer.contents, sizeof(status));
            need(status.code == 0u,
                 "production graph kernel status code " + std::to_string(status.code));

            std::vector<char> output;
            appendBytes(output, "NMGKO001", 8u);
            appendBytes(output, &header, sizeof(header));
            appendBytes(output, coarseRHS.contents, coarseRHS.length);
            appendBytes(output, coefficients.contents, coefficients.length);
            appendBytes(output, patchMatrices.contents, patchMatrices.length);
            appendBytes(output, pivots.contents, pivots.length);
            appendBytes(output, factorValid.contents, factorValid.length);
            appendBytes(output, corrections.contents, corrections.length);
            appendBytes(output, preconditioned.contents, preconditioned.length);
            appendBytes(output, statusBuffer.contents, statusBuffer.length);
            writeFile(outputPath, output);

            const std::string inputHash = sha256(input.data(), input.size());
            const std::string outputHash = sha256(output.data(), output.size());
            std::cout << "selector5_production_kernel_check=complete"
                      << " device=\"" << device.name.UTF8String << "\""
                      << " source_environment=" << header.selectedEnvironment
                      << " nodes=" << header.nodeCount
                      << " patches=" << header.patchCount
                      << " patch_nodes=" << header.patchNodeCount
                      << " input_sha256=" << inputHash
                      << " output_sha256=" << outputHash
                      << " kernels=nm_fgmres_regional_project_rhs,nm_fgmres_regional_solve_rap,"
                         "nm_fgmres_graph_factor_patches,nm_fgmres_graph_solve_patches,"
                         "nm_fgmres_graph_apply_patches\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "selector5_production_kernel_check=fail message=" << error.what() << '\n';
            return 1;
        }
    }
}
