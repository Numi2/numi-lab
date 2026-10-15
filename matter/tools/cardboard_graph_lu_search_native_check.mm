#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "numi/matter/shared.h"

#include <array>
#include <cmath>
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

struct OutputHeader {
    std::uint32_t version;
    std::uint32_t nodeCount;
    std::uint32_t patchCount;
    std::uint32_t patchNodeCount;
    std::uint32_t modeCount;
    std::uint32_t matrixFloatCount;
    std::uint32_t pivotCount;
    std::uint32_t statusCount;
};

struct FactorParityHeader {
    std::uint32_t version;
    std::uint32_t matrixByteCount;
    std::uint32_t pivotByteCount;
    std::uint32_t validByteCount;
    std::uint32_t statusByteCount;
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

id<MTLBuffer> bytesBuffer(id<MTLDevice> device, const void* bytes,
                          const std::size_t size, const char* label) {
    need(size > 0u, std::string("empty Metal bytes buffer ") + label);
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
                                               length:size
                                              options:MTLResourceStorageModeShared];
    need(buffer != nil, std::string("Metal bytes allocation failed: ") + label);
    return buffer;
}

id<MTLBuffer> outputBuffer(id<MTLDevice> device, const std::size_t bytes,
                           const char* label) {
    need(bytes > 0u, std::string("zero-sized output buffer ") + label);
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    need(buffer != nil, std::string("Metal output allocation failed: ") + label);
    std::memset(buffer.contents, 0, bytes);
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

void encodeProject(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> dispatch, id<MTLBuffer> regionalBasis,
                   id<MTLBuffer> basis, id<MTLBuffer> coarseRHS,
                   std::uint32_t modeCount) {
    [encoder setComputePipelineState:pipeline];
    setBuffer(encoder, dispatch, 0u);
    setBuffer(encoder, regionalBasis, 1u);
    setBuffer(encoder, basis, 2u);
    setBuffer(encoder, coarseRHS, 3u);
    setCount(encoder, modeCount, 4u);
    setCount(encoder, modeCount, 5u); // modeStride equals modeCount in this fixture.
    [encoder dispatchThreadgroups:MTLSizeMake(modeCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void encodeRAPSolve(id<MTLComputeCommandEncoder> encoder,
                    id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> dispatch, id<MTLBuffer> rapLU,
                    id<MTLBuffer> rapPivots, id<MTLBuffer> rapValid,
                    id<MTLBuffer> coarseRHS, id<MTLBuffer> coefficients,
                    std::uint32_t modeCount) {
    [encoder setComputePipelineState:pipeline];
    setBuffer(encoder, dispatch, 0u);
    setBuffer(encoder, rapLU, 1u);
    setBuffer(encoder, rapPivots, 2u);
    setBuffer(encoder, rapValid, 3u);
    setBuffer(encoder, coarseRHS, 4u);
    setBuffer(encoder, coefficients, 5u);
    setCount(encoder, modeCount, 6u);
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void encodeFactor(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> dispatch, id<MTLBuffer> patches,
                  id<MTLBuffer> matrices, id<MTLBuffer> pivots,
                  id<MTLBuffer> factorValid, id<MTLBuffer> status,
                  id<MTLBuffer> originalMatrices,
                  id<MTLBuffer> fgmresStates,
                  id<MTLBuffer> priorFactorValid,
                  const InputHeader& header,
                  const std::uint32_t parallelSearch = 0u) {
    [encoder setComputePipelineState:pipeline];
    setBuffer(encoder, dispatch, 0u);
    setBuffer(encoder, patches, 1u);
    setBuffer(encoder, matrices, 2u);
    setBuffer(encoder, pivots, 3u);
    setBuffer(encoder, factorValid, 4u);
    setBuffer(encoder, status, 5u);
    setCount(encoder, header.patchCount, 6u);
    setCount(encoder, header.matrixFloatCount, 7u);
    setCount(encoder, header.pivotCount, 8u);
    setBuffer(encoder, originalMatrices, 9u);
    setBuffer(encoder, fgmresStates, 10u);
    const std::uint32_t reuseCompletedTailAllowed = 0u;
    setCount(encoder, reuseCompletedTailAllowed, 11u);
    setBuffer(encoder, priorFactorValid, 12u);
    setCount(encoder, parallelSearch, 13u);
    [encoder dispatchThreadgroups:MTLSizeMake(header.patchCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void encodeSolve(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> dispatch, id<MTLBuffer> patches,
                 id<MTLBuffer> patchNodes, id<MTLBuffer> basis,
                 id<MTLBuffer> regionalBasis, id<MTLBuffer> regionalAP,
                 id<MTLBuffer> coefficients, id<MTLBuffer> matrices,
                 id<MTLBuffer> pivots, id<MTLBuffer> factorValid,
                 id<MTLBuffer> corrections, id<MTLBuffer> status,
                 id<MTLBuffer> states, id<MTLBuffer> originalMatrices,
                 id<MTLBuffer> candidate, std::uint32_t refinementMode,
                 const InputHeader& header) {
    [encoder setComputePipelineState:pipeline];
    setBuffer(encoder, dispatch, 0u);
    setBuffer(encoder, patches, 1u);
    setBuffer(encoder, patchNodes, 2u);
    setBuffer(encoder, basis, 3u);
    setBuffer(encoder, regionalBasis, 4u);
    setBuffer(encoder, regionalAP, 5u);
    setBuffer(encoder, coefficients, 6u);
    setBuffer(encoder, matrices, 7u);
    setBuffer(encoder, pivots, 8u);
    setBuffer(encoder, factorValid, 9u);
    setBuffer(encoder, corrections, 10u);
    setBuffer(encoder, status, 11u);
    setCount(encoder, header.patchCount, 12u);
    setCount(encoder, header.patchNodeCount, 13u);
    setCount(encoder, header.matrixFloatCount, 14u);
    setCount(encoder, header.pivotCount, 15u);
    setCount(encoder, header.modeCount, 16u);
    setBuffer(encoder, states, 17u);
    setBuffer(encoder, originalMatrices, 18u);
    setCount(encoder, refinementMode, 19u);
    setBuffer(encoder, candidate, 20u);
    setCount(encoder, header.modeCount, 21u);
    [encoder dispatchThreadgroups:MTLSizeMake(header.patchCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void encodeApply(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> dispatch, id<MTLBuffer> patchNodes,
                 id<MTLBuffer> nodePatchOffsets, id<MTLBuffer> nodePatchValues,
                 id<MTLBuffer> nodeCoverage, id<MTLBuffer> regionalBasis,
                 id<MTLBuffer> coefficients, id<MTLBuffer> corrections,
                 id<MTLBuffer> preconditioned, id<MTLBuffer> states,
                 id<MTLBuffer> candidate, std::uint32_t patchNodeCount,
                 std::uint32_t modeCount, std::uint32_t nodeCount) {
    [encoder setComputePipelineState:pipeline];
    setBuffer(encoder, dispatch, 0u);
    setBuffer(encoder, patchNodes, 1u);
    setBuffer(encoder, nodePatchOffsets, 2u);
    setBuffer(encoder, nodePatchValues, 3u);
    setBuffer(encoder, nodeCoverage, 4u);
    setBuffer(encoder, regionalBasis, 5u);
    setBuffer(encoder, coefficients, 6u);
    setBuffer(encoder, corrections, 7u);
    setBuffer(encoder, preconditioned, 8u);
    setBuffer(encoder, states, 9u);
    setCount(encoder, patchNodeCount, 10u);
    setCount(encoder, modeCount, 11u);
    setBuffer(encoder, candidate, 12u);
    setCount(encoder, modeCount, 13u); // modeStride equals modeCount here.
    [encoder dispatchThreads:MTLSizeMake(nodeCount, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

bool allBitwiseZero(const id<MTLBuffer> buffer) {
    const auto* words = static_cast<const std::uint32_t*>(buffer.contents);
    const std::size_t count = buffer.length / sizeof(std::uint32_t);
    for (std::size_t i = 0; i < count; ++i)
        if (words[i] != 0u) return false;
    return true;
}

struct NumericZeroSummary {
    std::size_t valueCount = 0u;
    std::size_t nonfiniteCount = 0u;
    std::size_t nonzeroCount = 0u;
    std::size_t negativeZeroCount = 0u;
    std::size_t firstNonzeroIndex = SIZE_MAX;
    std::uint32_t firstNonzeroBits = 0u;
};

NumericZeroSummary summarizeNumericZero(const id<MTLBuffer> buffer) {
    NumericZeroSummary summary;
    summary.valueCount = buffer.length / sizeof(float);
    const auto* values = static_cast<const float*>(buffer.contents);
    for (std::size_t index = 0u; index < summary.valueCount; ++index) {
        std::uint32_t bits = 0u;
        std::memcpy(&bits, values + index, sizeof(bits));
        const float value = values[index];
        if (!std::isfinite(value)) {
            ++summary.nonfiniteCount;
        } else if (value == 0.0f) {
            if ((bits & 0x80000000u) != 0u) ++summary.negativeZeroCount;
        } else {
            if (summary.firstNonzeroIndex == SIZE_MAX) {
                summary.firstNonzeroIndex = index;
                summary.firstNonzeroBits = bits;
            }
            ++summary.nonzeroCount;
        }
    }
    return summary;
}

bool numericZero(const NumericZeroSummary& summary) {
    return summary.nonfiniteCount == 0u && summary.nonzeroCount == 0u;
}

void printNumericZeroSummary(const char* label,
                             const NumericZeroSummary& summary) {
    std::cout << "\"" << label << "\":{\"value_count\":"
        << summary.valueCount << ",\"nonfinite_count\":"
        << summary.nonfiniteCount << ",\"nonzero_count\":"
        << summary.nonzeroCount << ",\"negative_zero_count\":"
        << summary.negativeZeroCount << ",\"first_nonzero_index\":";
    if (summary.firstNonzeroIndex == SIZE_MAX) std::cout << "null";
    else std::cout << summary.firstNonzeroIndex;
    std::cout << ",\"first_nonzero_bits\":";
    if (summary.firstNonzeroIndex == SIZE_MAX) std::cout << "null";
    else std::cout << summary.firstNonzeroBits;
    std::cout << "}";
}

bool rangeBitwiseZero(const id<MTLBuffer> buffer, const std::size_t byteOffset,
                      const std::size_t byteCount) {
    if (byteOffset > buffer.length || byteCount > buffer.length - byteOffset ||
        (byteOffset % sizeof(std::uint32_t)) != 0u ||
        (byteCount % sizeof(std::uint32_t)) != 0u) return false;
    const auto* words = static_cast<const std::uint32_t*>(buffer.contents) +
        byteOffset / sizeof(std::uint32_t);
    for (std::size_t i = 0; i < byteCount / sizeof(std::uint32_t); ++i)
        if (words[i] != 0u) return false;
    return true;
}

std::uint32_t statusCode(const id<MTLBuffer> buffer) {
    NMMatterStatusGPU status{};
    std::memcpy(&status, buffer.contents, sizeof(status));
    return status.code;
}

void appendBuffer(std::vector<char>& output, id<MTLBuffer> buffer) {
    appendBytes(output, buffer.contents, buffer.length);
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            need(argc == 4,
                 "usage: selector5_refinement_kernels INPUT.bin OUTPUT.bin NumiMatterGraphSchwarz.metallib");
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
            need(library != nil, std::string("cannot load graph-Schwarz metallib: ") +
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
            std::vector<NMFEMNodeStateGPU> graphCandidate(header.nodeCount);
            for (NMFEMNodeStateGPU& node : graphCandidate) {
                node.positionAndMass.w = 1.0f;
                node.velocityAndInverseMass.w = 1.0f;
            }
            id<MTLBuffer> candidateBuffer = outputBuffer(device,
                graphCandidate.size() * sizeof(NMFEMNodeStateGPU),
                "all-free graph candidate nodes");
            std::memcpy(candidateBuffer.contents, graphCandidate.data(),
                        candidateBuffer.length);
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
            id<MTLBuffer> originalMatrices = inputBuffer(device, patchMatricesView,
                                                          "original_patch_matrices");
            id<MTLBuffer> patchMatrices = inputBuffer(device, patchMatricesView, "patch_matrices");

            // Singular-factor control: exact zero matrix in patch zero. This is a
            // negative contract test and does not alter the active source run.
            std::vector<char> singularSource(patchMatricesView.data,
                                              patchMatricesView.data + patchMatricesView.size);
            nm_uint4 firstPatch{};
            std::memcpy(&firstPatch,
                        patchesView.data,
                        sizeof(firstPatch));
            const std::uint64_t singularN = static_cast<std::uint64_t>(firstPatch.y) * 3u;
            const std::uint64_t singularCount = singularN * singularN;
            need(firstPatch.y > 0u && firstPatch.z <= header.matrixFloatCount &&
                 singularCount <= header.matrixFloatCount - firstPatch.z,
                 "first patch matrix span is invalid");
            std::memset(singularSource.data() + firstPatch.z * sizeof(float), 0,
                        static_cast<std::size_t>(singularCount) * sizeof(float));
            id<MTLBuffer> singularMatrices = bytesBuffer(device, singularSource.data(),
                singularSource.size(), "singular_patch_matrices");

            id<MTLBuffer> coarseRHS = outputBuffer(device,
                sizeof(float) * header.modeCount, "coarse_rhs");
            id<MTLBuffer> coefficients = outputBuffer(device,
                sizeof(float) * header.modeCount, "coefficients");
            id<MTLBuffer> pivots = outputBuffer(device,
                sizeof(std::uint32_t) * header.pivotCount, "patch_pivots");
            id<MTLBuffer> factorValid = outputBuffer(device,
                sizeof(std::uint32_t) * header.patchCount, "patch_factor_valid");
            id<MTLBuffer> factorStatus = outputBuffer(device,
                sizeof(NMMatterStatusGPU), "factor_status");
            id<MTLBuffer> parallelMatrices = bytesBuffer(device,
                patchMatricesView.data, patchMatricesView.size,
                "parallel_search_patch_matrices");
            id<MTLBuffer> parallelOriginalMatrices = inputBuffer(device,
                patchMatricesView, "parallel_search_original_matrices");
            id<MTLBuffer> parallelPivots = outputBuffer(device,
                sizeof(std::uint32_t) * header.pivotCount,
                "parallel_search_patch_pivots");
            id<MTLBuffer> parallelFactorValid = outputBuffer(device,
                sizeof(std::uint32_t) * header.patchCount,
                "parallel_search_factor_valid");
            id<MTLBuffer> parallelFactorStatus = outputBuffer(device,
                sizeof(NMMatterStatusGPU), "parallel_search_factor_status");
            std::memset(pivots.contents, 0, pivots.length);
            std::memset(factorValid.contents, 0, factorValid.length);
            std::memset(factorStatus.contents, 0, factorStatus.length);
            std::memset(parallelPivots.contents, 0, parallelPivots.length);
            std::memset(parallelFactorValid.contents, 0,
                        parallelFactorValid.length);
            std::memset(parallelFactorStatus.contents, 0,
                        parallelFactorStatus.length);

            auto newStatus = [&]() {
                return outputBuffer(device, sizeof(NMMatterStatusGPU), "scenario_status");
            };
            id<MTLBuffer> mode0Status = newStatus();
            id<MTLBuffer> mode1Status = newStatus();
            id<MTLBuffer> zeroMode0Status = newStatus();
            id<MTLBuffer> zeroStatus = newStatus();
            id<MTLBuffer> completedStatus = newStatus();
            id<MTLBuffer> invalidFactorStatus = newStatus();
            id<MTLBuffer> invalidModeStatus = newStatus();
            id<MTLBuffer> singularFactorStatus = newStatus();
            id<MTLBuffer> singularSolveStatus = newStatus();

            id<MTLBuffer> corrections0 = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "mode0_corrections");
            id<MTLBuffer> corrections1 = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "mode1_corrections");
            id<MTLBuffer> zeroMode0Corrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "zero_rhs_mode0_corrections");
            id<MTLBuffer> zeroCorrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "zero_rhs_corrections");
            id<MTLBuffer> completedCorrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "completed_tail_corrections");
            id<MTLBuffer> invalidFactorCorrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "invalid_factor_corrections");
            id<MTLBuffer> invalidModeCorrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "invalid_mode_corrections");
            id<MTLBuffer> singularCorrections = outputBuffer(device,
                sizeof(nm_float4) * header.patchNodeCount, "singular_factor_corrections");

            id<MTLBuffer> preconditioned0 = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "mode0_preconditioned");
            id<MTLBuffer> preconditioned1 = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "mode1_preconditioned");
            id<MTLBuffer> zeroMode0Preconditioned = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "zero_rhs_mode0_preconditioned");
            id<MTLBuffer> zeroPreconditioned = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "zero_rhs_preconditioned");
            id<MTLBuffer> completedPreconditioned = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "completed_tail_preconditioned");

            id<MTLBuffer> activeState = outputBuffer(device,
                sizeof(NMFGMRESStateGPU), "active_state");
            id<MTLBuffer> completedState = outputBuffer(device,
                sizeof(NMFGMRESStateGPU), "completed_state");
            auto* activeStateCPU = static_cast<NMFGMRESStateGPU*>(activeState.contents);
            std::memset(activeStateCPU, 0, sizeof(*activeStateCPU));
            auto* completedStateCPU = static_cast<NMFGMRESStateGPU*>(completedState.contents);
            std::memset(completedStateCPU, 0, sizeof(*completedStateCPU));
            completedStateCPU->diagnostics.z = 1.0f;
            id<MTLBuffer> zeroBasis = outputBuffer(device,
                sizeof(nm_float4) * header.nodeCount, "zero_basis");
            id<MTLBuffer> zeroAP = outputBuffer(device,
                sizeof(nm_float4) * header.modeCount * header.nodeCount, "zero_regional_ap");
            id<MTLBuffer> zeroCoefficients = outputBuffer(device,
                sizeof(float) * header.modeCount, "zero_coefficients");
            id<MTLBuffer> invalidFactorValid = outputBuffer(device,
                sizeof(std::uint32_t) * header.patchCount, "invalid_factor_valid");
            id<MTLBuffer> singularPivots = outputBuffer(device,
                sizeof(std::uint32_t) * header.pivotCount, "singular_pivots");
            id<MTLBuffer> singularFactorValid = outputBuffer(device,
                sizeof(std::uint32_t) * header.patchCount, "singular_factor_valid");
            std::memset(singularFactorValid.contents, 0,
                        singularFactorValid.length);
            id<MTLBuffer> singularOriginalMatrices = inputBuffer(device, patchMatricesView,
                                                                  "singular_original_matrices");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            need(queue != nil, "cannot create Metal command queue");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            need(command != nil, "cannot create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            need(encoder != nil, "cannot create Metal compute encoder");

            encodeProject(encoder, projectPipeline, dispatchBuffer, regionalBasis,
                          basis, coarseRHS, header.modeCount);
            encodeRAPSolve(encoder, rapSolvePipeline, dispatchBuffer, rapLU,
                           rapPivots, rapValid, coarseRHS, coefficients,
                           header.modeCount);
            encodeFactor(encoder, factorPipeline, dispatchBuffer, patches,
                         patchMatrices, pivots, factorValid, factorStatus,
                         originalMatrices, activeState, factorValid, header, 0u);
            encodeFactor(encoder, factorPipeline, dispatchBuffer, patches,
                         parallelMatrices, parallelPivots, parallelFactorValid,
                         parallelFactorStatus, parallelOriginalMatrices,
                         activeState, parallelFactorValid, header, 1u);

            // Both modes use the same source matrix, pivoted LU, pivot vector,
            // coarse coefficients, and RHS. The only changed binding is mode.
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, patchMatrices,
                        pivots, factorValid, corrections0, mode0Status, activeState,
                        originalMatrices, candidateBuffer, 0u, header);
            encodeApply(encoder, applyPipeline, dispatchBuffer, patchNodes,
                        nodePatchOffsets, nodePatchValues, nodeCoverage, regionalBasis,
                        coefficients, corrections0, preconditioned0, activeState, candidateBuffer,
                        header.patchNodeCount, header.modeCount, header.nodeCount);
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, patchMatrices,
                        pivots, factorValid, corrections1, mode1Status, activeState,
                        originalMatrices, candidateBuffer, 1u, header);
            encodeApply(encoder, applyPipeline, dispatchBuffer, patchNodes,
                        nodePatchOffsets, nodePatchValues, nodeCoverage, regionalBasis,
                        coefficients, corrections1, preconditioned1, activeState, candidateBuffer,
                        header.patchNodeCount, header.modeCount, header.nodeCount);

            // Exact zero RHS exercises the no-op refinement branch end to end.
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        zeroBasis, regionalBasis, zeroAP, zeroCoefficients, patchMatrices,
                        pivots, factorValid, zeroMode0Corrections, zeroMode0Status,
                        activeState, originalMatrices, candidateBuffer, 0u, header);
            encodeApply(encoder, applyPipeline, dispatchBuffer, patchNodes,
                        nodePatchOffsets, nodePatchValues, nodeCoverage, regionalBasis,
                        zeroCoefficients, zeroMode0Corrections, zeroMode0Preconditioned,
                        activeState, candidateBuffer, header.patchNodeCount, header.modeCount,
                        header.nodeCount);
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        zeroBasis, regionalBasis, zeroAP, zeroCoefficients, patchMatrices,
                        pivots, factorValid, zeroCorrections, zeroStatus, activeState,
                        originalMatrices, candidateBuffer, 1u, header);
            encodeApply(encoder, applyPipeline, dispatchBuffer, patchNodes,
                        nodePatchOffsets, nodePatchValues, nodeCoverage, regionalBasis,
                        zeroCoefficients, zeroCorrections, zeroPreconditioned, activeState, candidateBuffer,
                        header.patchNodeCount, header.modeCount, header.nodeCount);

            // The completed-cycle latch is tested independently of the solve mode.
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, patchMatrices,
                        pivots, factorValid, completedCorrections, completedStatus,
                        completedState, originalMatrices, candidateBuffer, 1u, header);
            encodeApply(encoder, applyPipeline, dispatchBuffer, patchNodes,
                        nodePatchOffsets, nodePatchValues, nodeCoverage, regionalBasis,
                        coefficients, completedCorrections, completedPreconditioned,
                        completedState, candidateBuffer, header.patchNodeCount, header.modeCount,
                        header.nodeCount);

            // Explicit invalid-factor and invalid-mode admission controls.
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, patchMatrices,
                        pivots, invalidFactorValid, invalidFactorCorrections,
                        invalidFactorStatus, activeState, originalMatrices, candidateBuffer, 1u, header);
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, patchMatrices,
                        pivots, factorValid, invalidModeCorrections,
                        invalidModeStatus, activeState, originalMatrices, candidateBuffer, 2u, header);

            // Separate factor/solve path with one authored singular principal block.
            encodeFactor(encoder, factorPipeline, dispatchBuffer, patches,
                         singularMatrices, singularPivots, singularFactorValid,
                         singularFactorStatus, singularOriginalMatrices,
                         activeState, singularFactorValid, header, 0u);
            encodeSolve(encoder, solvePipeline, dispatchBuffer, patches, patchNodes,
                        basis, regionalBasis, regionalAP, coefficients, singularMatrices,
                        singularPivots, singularFactorValid, singularCorrections,
                        singularSolveStatus, activeState, singularOriginalMatrices,
                        candidateBuffer, 1u, header);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            const char* localizedCommandError = command.error.localizedDescription != nil
                ? command.error.localizedDescription.UTF8String : "unknown command error";
            need(command.status == MTLCommandBufferStatusCompleted,
                 std::string("Metal refinement checker command failed: ") +
                 localizedCommandError);

            // Save exact source-derived factor outputs before the parity gate.
            // The sidecar retains both modes even if an equality check fails.
            need(patchMatrices.length <= UINT32_MAX && pivots.length <= UINT32_MAX &&
                 factorValid.length <= UINT32_MAX && factorStatus.length <= UINT32_MAX,
                 "factor parity sidecar exceeds its checked format");
            FactorParityHeader parityHeader{
                1u, static_cast<std::uint32_t>(patchMatrices.length),
                static_cast<std::uint32_t>(pivots.length),
                static_cast<std::uint32_t>(factorValid.length),
                static_cast<std::uint32_t>(factorStatus.length)};
            std::vector<char> parity;
            const std::filesystem::path parityPath(outputPath.string() +
                ".factor-search-parity.bin");
            appendBytes(parity, "NMSFSP01", 8u);
            appendBytes(parity, &parityHeader, sizeof(parityHeader));
            appendBuffer(parity, patchMatrices);
            appendBuffer(parity, pivots);
            appendBuffer(parity, factorValid);
            appendBuffer(parity, originalMatrices);
            appendBuffer(parity, factorStatus);
            appendBuffer(parity, parallelMatrices);
            appendBuffer(parity, parallelPivots);
            appendBuffer(parity, parallelFactorValid);
            appendBuffer(parity, parallelOriginalMatrices);
            appendBuffer(parity, parallelFactorStatus);
            const bool matrixParity = std::memcmp(patchMatrices.contents,
                parallelMatrices.contents, patchMatrices.length) == 0;
            const bool pivotParity = std::memcmp(pivots.contents,
                parallelPivots.contents, pivots.length) == 0;
            const bool validParity = std::memcmp(factorValid.contents,
                parallelFactorValid.contents, factorValid.length) == 0;
            const bool originalParity = std::memcmp(originalMatrices.contents,
                parallelOriginalMatrices.contents, originalMatrices.length) == 0;
            const bool statusParity = std::memcmp(factorStatus.contents,
                parallelFactorStatus.contents, factorStatus.length) == 0;
            writeFile(parityPath, parity);
            const std::string parityHash = sha256(parity.data(), parity.size());
            const std::filesystem::path parityManifestPath(outputPath.string() +
                ".factor-search-parity.json");
            {
                std::ofstream manifest(parityManifestPath, std::ios::binary | std::ios::trunc);
                need(manifest.good(), "cannot create factor parity manifest");
                manifest << "{\n"
                         << "  \"schema\": \"numi.matter.graph_lu_search_parity.v1\",\n"
                         << "  \"input_sha256\": \"" << sha256(input.data(), input.size()) << "\",\n"
                         << "  \"parity_payload_sha256\": \"" << parityHash << "\",\n"
                         << "  \"legacy_vs_parallel\": {\n"
                         << "    \"lu_matrix_bitwise_equal\": " << (matrixParity ? "true" : "false") << ",\n"
                         << "    \"pivot_vector_bitwise_equal\": " << (pivotParity ? "true" : "false") << ",\n"
                         << "    \"factor_valid_bitwise_equal\": " << (validParity ? "true" : "false") << ",\n"
                         << "    \"original_matrix_bitwise_equal\": " << (originalParity ? "true" : "false") << ",\n"
                         << "    \"status_bytes_equal\": " << (statusParity ? "true" : "false") << "\n"
                         << "  },\n"
                         << "  \"status_code_legacy\": " << statusCode(factorStatus) << ",\n"
                         << "  \"status_code_parallel\": " << statusCode(parallelFactorStatus) << "\n"
                         << "}\n";
                manifest.close();
                need(manifest.good(), "failed writing factor parity manifest");
            }

            // Persist the complete readback before any semantic gate throws.
            // This keeps diagnostic evidence for a failed checker invocation.
            OutputHeader outputHeader{
                1u, header.nodeCount, header.patchCount, header.patchNodeCount,
                header.modeCount, header.matrixFloatCount, header.pivotCount, 10u};
            std::vector<char> output;
            appendBytes(output, "NMSRC001", 8u);
            appendBytes(output, &outputHeader, sizeof(outputHeader));
            appendBuffer(output, coarseRHS);
            appendBuffer(output, coefficients);
            appendBuffer(output, factorValid);
            appendBuffer(output, patchMatrices);
            appendBuffer(output, pivots);
            appendBuffer(output, corrections0);
            appendBuffer(output, corrections1);
            appendBuffer(output, preconditioned0);
            appendBuffer(output, preconditioned1);
            appendBuffer(output, zeroMode0Corrections);
            appendBuffer(output, zeroMode0Preconditioned);
            appendBuffer(output, zeroCorrections);
            appendBuffer(output, zeroPreconditioned);
            appendBuffer(output, completedCorrections);
            appendBuffer(output, completedPreconditioned);
            appendBuffer(output, invalidFactorCorrections);
            appendBuffer(output, invalidModeCorrections);
            appendBuffer(output, singularFactorValid);
            appendBuffer(output, singularMatrices);
            appendBuffer(output, singularPivots);
            appendBuffer(output, singularCorrections);
            appendBuffer(output, factorStatus);
            appendBuffer(output, mode0Status);
            appendBuffer(output, mode1Status);
            appendBuffer(output, zeroMode0Status);
            appendBuffer(output, zeroStatus);
            appendBuffer(output, completedStatus);
            appendBuffer(output, invalidFactorStatus);
            appendBuffer(output, invalidModeStatus);
            appendBuffer(output, singularFactorStatus);
            appendBuffer(output, singularSolveStatus);
            writeFile(outputPath, output);

            const NumericZeroSummary zeroMode0CorrectionSummary =
                summarizeNumericZero(zeroMode0Corrections);
            const NumericZeroSummary zeroMode0PreconditionedSummary =
                summarizeNumericZero(zeroMode0Preconditioned);
            const NumericZeroSummary zeroCorrectionSummary =
                summarizeNumericZero(zeroCorrections);
            const NumericZeroSummary zeroPreconditionedSummary =
                summarizeNumericZero(zeroPreconditioned);
            std::cout << "selector5_zero_rhs_diagnostics={";
            printNumericZeroSummary("mode0_corrections",
                                    zeroMode0CorrectionSummary);
            std::cout << ',';
            printNumericZeroSummary("mode0_preconditioned",
                                    zeroMode0PreconditionedSummary);
            std::cout << ',';
            printNumericZeroSummary("refined_corrections",
                                    zeroCorrectionSummary);
            std::cout << ',';
            printNumericZeroSummary("refined_preconditioned",
                                    zeroPreconditionedSummary);
            std::cout << "}\n" << std::flush;

            // Contract gates are deterministic and separate from the local residual
            // accuracy gate computed by verify_refinement_output.py.
            need(matrixParity && pivotParity && validParity && originalParity &&
                 statusParity,
                 "parallel graph LU differs from legacy source-derived factor outputs");
            need(statusCode(factorStatus) == NM_STATUS_SUCCESS &&
                 statusCode(parallelFactorStatus) == NM_STATUS_SUCCESS,
                 "valid active patch factorization unexpectedly failed");
            need(statusCode(mode0Status) == NM_STATUS_SUCCESS &&
                 statusCode(mode1Status) == NM_STATUS_SUCCESS,
                 "active refinement mode reported a solver status");
            need(statusCode(zeroMode0Status) == NM_STATUS_SUCCESS &&
                 numericZero(zeroMode0CorrectionSummary) &&
                 numericZero(zeroMode0PreconditionedSummary),
                 "zero-RHS mode-0 control contained nonzero or nonfinite output");
            need(statusCode(zeroStatus) == NM_STATUS_SUCCESS &&
                 numericZero(zeroCorrectionSummary) &&
                 numericZero(zeroPreconditionedSummary),
                 "zero-RHS refinement path contained nonzero or nonfinite output");
            need(statusCode(completedStatus) == NM_STATUS_SUCCESS &&
                 allBitwiseZero(completedCorrections) &&
                 allBitwiseZero(completedPreconditioned),
                 "completed-cycle guard did not zero solve and apply outputs");
            need(statusCode(invalidFactorStatus) == NM_STATUS_LINEAR_SOLVER_FAILURE &&
                 allBitwiseZero(invalidFactorCorrections),
                 "invalid-factor path did not fail closed with zero correction");
            need(statusCode(invalidModeStatus) == NM_STATUS_INVALID_DISPATCH &&
                 allBitwiseZero(invalidModeCorrections),
                 "unsupported refinement mode did not fail closed");
            need(statusCode(singularFactorStatus) == NM_STATUS_LINEAR_SOLVER_FAILURE &&
                 statusCode(singularSolveStatus) == NM_STATUS_LINEAR_SOLVER_FAILURE &&
                 rangeBitwiseZero(singularCorrections,
                    static_cast<std::size_t>(firstPatch.x) * sizeof(nm_float4),
                    static_cast<std::size_t>(firstPatch.y) * sizeof(nm_float4)),
                 "singular factor path did not fail closed");
            const auto* singularValidCPU =
                static_cast<const std::uint32_t*>(singularFactorValid.contents);
            need(singularValidCPU[0] == 0u,
                 "authored zero principal matrix was incorrectly admitted as a valid factor");

            std::cout << "selector5_refinement_checker=complete"
                      << " device=\"" << device.name.UTF8String << "\""
                      << " source_environment=" << header.selectedEnvironment
                      << " nodes=" << header.nodeCount
                      << " patches=" << header.patchCount
                      << " patch_nodes=" << header.patchNodeCount
                      << " active_modes=0,1 zero_rhs=pass completed_tail=pass"
                      << " factor_search_parity=pass legacy_vs_parallel=bitwise"
                      << " matrix=" << (matrixParity ? "equal" : "different")
                      << " pivots=" << (pivotParity ? "equal" : "different")
                      << " factor_valid=" << (validParity ? "equal" : "different")
                      << " original_matrix=" << (originalParity ? "equal" : "different")
                      << " status=" << (statusParity ? "equal" : "different")
                      << " parity_sidecar=" << parityPath.string()
                      << " parity_manifest=" << parityManifestPath.string()
                      << " parity_sha256=" << parityHash
                      << " invalid_factor=pass invalid_mode=pass singular_factor=pass"
                      << " input_sha256=" << sha256(input.data(), input.size())
                      << " output_sha256=" << sha256(output.data(), output.size())
                      << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "selector5_refinement_checker=fail message=" << error.what() << '\n';
            return 1;
        }
    }
}
