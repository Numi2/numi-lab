#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "numi/matter/shared.h"

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

static_assert(sizeof(NMDeformableVTCCDAuditGPU) == 336u);
static_assert(alignof(NMDeformableVTCCDAuditGPU) == 16u);
static_assert(sizeof(NMDeformableContactFailureGPU) == 384u);
static_assert(alignof(NMDeformableContactFailureGPU) == 16u);

namespace {
using Float4 = std::array<float, 4>;
constexpr std::array<std::uint32_t, 4> kBudgets{16u, 32u, 64u, 128u};
constexpr std::uint32_t kTraceStride = 128u;
constexpr char kSourcePhysicsFingerprint[] = "915f50b10edc2e46";
constexpr char kObservedProgramFingerprint[] = "28f5c6c60a7f247f";
constexpr char kTrial21ManifestSHA[] = "a821cf7e1e166327059258a953aaf519c8ef7e8ca9f0c344f1b6681185845e64";
constexpr char kTrial21PayloadSHA[] = "eb6286c09081577e226eec6deb11021b4e483e104583214216b6b77c441190b5";
constexpr char kTrial21ObserverSHA[] = "ed119deb040fcc91e33e4eafe73e948cdd34946960c4e1ecee2decc6d600d097";
constexpr char kTrial22ManifestSHA[] = "41eb361744412ff47f338bc563a43cd940c29019a3c32c3c4de83e61677d220c";
constexpr char kTrial22PayloadSHA[] = "d1c3e4d3c74d3bf75a72ade703557b4cfa9a70d62f1f51683fffbcff8546bd74";
constexpr char kTrial22ObserverSHA[] = "981fb3efe37dd2463b86e756b5a312ce33c619c8a93110c7ea4580a16d5c2277";

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string sha256(const void* bytes, const std::size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    CC_SHA256(bytes, static_cast<CC_LONG>(size), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::string jsonQuote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20u) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(ch) << std::dec;
                } else out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

std::uint64_t number(NSDictionary* object, NSString* key) {
    id value = object[key];
    require([value isKindOfClass:[NSNumber class]],
        std::string("missing numeric JSON field ") + [key UTF8String]);
    return [value unsignedLongLongValue];
}

std::string stringValue(NSDictionary* object, NSString* key) {
    id value = object[key];
    require([value isKindOfClass:[NSString class]],
        std::string("missing string JSON field ") + [key UTF8String]);
    return [value UTF8String];
}

struct CapturedInput {
    std::string capturePath;
    std::string captureManifestSHA;
    std::string payloadSHA;
    std::string observerSHA;
    std::string deviceFingerprint;
    NMDeformableVTCCDAuditGPU record{};
};

CapturedInput loadCapturedInput(const std::string& manifestPath,
                               const char* expectedManifestSHA,
                               const char* expectedPayloadSHA,
                               const char* expectedObserverSHA) {
    NSString* manifestNS = [NSString stringWithUTF8String:manifestPath.c_str()];
    NSError* error = nil;
    NSData* manifestData = [NSData dataWithContentsOfFile:manifestNS
        options:NSDataReadingMappedIfSafe error:&error];
    require(manifestData != nil, error != nil
        ? std::string("cannot read capture manifest: ") + [[error localizedDescription] UTF8String]
        : "cannot read capture manifest");
    id parsed = [NSJSONSerialization JSONObjectWithData:manifestData
        options:NSJSONReadingFragmentsAllowed error:&error];
    require(parsed != nil && [parsed isKindOfClass:[NSDictionary class]], error != nil
        ? std::string("capture manifest is not strict JSON: ") + [[error localizedDescription] UTF8String]
        : "capture manifest is not a JSON object");
    NSDictionary* root = (NSDictionary*)parsed;
    require(stringValue(root, @"schema") == "numi.matter.coupled_armijo_trial_capture.v1",
        "unexpected capture schema");
    require(stringValue(root, @"source_physics_fingerprint") == kSourcePhysicsFingerprint,
        "capture source-physics fingerprint is not the retained 405/406 source");
    const std::string deviceFingerprint = stringValue(root, @"device_program_fingerprint");
    require(deviceFingerprint == kObservedProgramFingerprint,
        "capture is not the retained observer source122 program");

    NSDictionary* payloadSpec = root[@"payload"];
    require([payloadSpec isKindOfClass:[NSDictionary class]], "capture payload descriptor missing");
    const std::string payloadLeaf = stringValue(payloadSpec, @"file");
    require(payloadLeaf.find('/') == std::string::npos && payloadLeaf.find('\\') == std::string::npos,
        "payload filename must be an adjacent basename");
    NSString* payloadPath = [[manifestNS stringByDeletingLastPathComponent]
        stringByAppendingPathComponent:[NSString stringWithUTF8String:payloadLeaf.c_str()]];
    NSData* payload = [NSData dataWithContentsOfFile:payloadPath
        options:NSDataReadingMappedIfSafe error:&error];
    require(payload != nil, error != nil
        ? std::string("cannot read adjacent capture payload: ") + [[error localizedDescription] UTF8String]
        : "cannot read adjacent capture payload");
    const std::size_t payloadBytes = [payload length];
    const auto* payloadData = static_cast<const std::uint8_t*>([payload bytes]);
    const std::string payloadSHA = sha256(payloadData, payloadBytes);
    require(number(payloadSpec, @"byte_count") == payloadBytes, "payload byte extent mismatch");
    require(stringValue(payloadSpec, @"sha256") == payloadSHA, "payload SHA-256 mismatch");
    require(number(root, @"array_count") == [root[@"arrays"] count], "array count mismatch");

    NSDictionary* observer = nil;
    for (id entry in root[@"arrays"]) {
        require([entry isKindOfClass:[NSDictionary class]], "malformed array descriptor");
        NSDictionary* spec = (NSDictionary*)entry;
        if ([spec[@"name"] isEqual:@"actual_vertex_triangle_ccd_call"]) {
            require(observer == nil, "duplicate exact-call observer array");
            observer = spec;
        }
    }
    require(observer != nil, "exact-call observer array is absent");
    const std::uint64_t offset = number(observer, @"offset");
    const std::uint64_t byteCount = number(observer, @"byte_count");
    require(number(observer, @"count") == 1u && number(observer, @"element_bytes") == 336u &&
        byteCount == sizeof(NMDeformableVTCCDAuditGPU), "observer array has unexpected layout");
    require(offset <= payloadBytes && byteCount <= payloadBytes - offset,
        "observer array exceeds payload bounds");
    const std::string observerSHA = sha256(payloadData + offset, static_cast<std::size_t>(byteCount));
    require(stringValue(observer, @"sha256") == observerSHA, "observer array SHA-256 mismatch");

    CapturedInput result;
    result.capturePath = manifestPath;
    result.captureManifestSHA = sha256([manifestData bytes], [manifestData length]);
    result.payloadSHA = payloadSHA;
    result.observerSHA = observerSHA;
    result.deviceFingerprint = deviceFingerprint;
    require(result.captureManifestSHA == expectedManifestSHA,
        "capture manifest hash differs from exact retained input");
    require(result.payloadSHA == expectedPayloadSHA,
        "capture payload hash differs from exact retained input");
    require(result.observerSHA == expectedObserverSHA,
        "observer record hash differs from exact retained input");
    std::memcpy(&result.record, payloadData + offset, sizeof(result.record));
    require(result.record.attempt.x == 1u && result.record.attempt.y == 0u &&
        result.record.attempt.z == 173u && result.record.attempt.w == 5u,
        "observer is not the selected environment0 slot173 direction1 vertex2 call");
    require(result.record.candidateIdentity.x == 36u && result.record.candidateIdentity.y == 123u &&
        result.record.candidateIdentity.z == 173u && result.record.candidateIdentity.w == 1u,
        "observer candidate identity differs from retained 405/406 pair");
    require(result.record.parameters.w == 9.0f, "observer is not Newton iteration 9");
    return result;
}

struct HumanFailureInput {
    std::string sourcePath;
    std::string headerLine;
    std::vector<std::string> sourceLines;
    NMDeformableContactFailureGPU record{};
};

float parseLogFloat(const float value, const char* field) {
    require(std::isfinite(value), std::string("nonfinite human failure field ") + field);
    return value;
}

HumanFailureInput loadHumanFailureInput(const std::string& logPath) {
    std::ifstream input(logPath);
    require(input.good(), "cannot open --human-failure log");
    HumanFailureInput result;
    result.sourcePath = logPath;
    bool haveHeader = false;
    std::uint32_t primitiveMask = 0u;
    std::uint32_t vertexMask = 0u;
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("deformable_contact_failure ", 0u) == 0u) {
            require(!haveHeader, "human failure log contains multiple failure records");
            unsigned step=0u, microtick=0u, slot=0u, primitiveA=0u, primitiveB=0u;
            float thickness=0.0f, dtA=0.0f, dtB=0.0f, newton=0.0f;
            float dx=0.0f, dy=0.0f, dz=0.0f, dw=0.0f;
            const int count = std::sscanf(line.c_str(),
                "deformable_contact_failure step=%u microtick=%u slot=%u primitive_a=%u primitive_b=%u thickness=%f dt_a=%f dt_b=%f newton=%f diagnostics=[%f,%f,%f,%f]",
                &step, &microtick, &slot, &primitiveA, &primitiveB,
                &thickness, &dtA, &dtB, &newton, &dx, &dy, &dz, &dw);
            require(count == 13, "malformed deformable_contact_failure line");
            const bool isVertexTriangle = dw == 3.0f || dw == 4.0f;
            const bool isEdgeEdge = dw >= 0.0f && dw <= 2.0f &&
                dw == std::floor(dw);
            require(isVertexTriangle || isEdgeEdge,
                "human failure is neither VTCCD direction 3/4 nor EECCD edge index 0..2");
            require(dz >= 0.0f && dz <= 2.0f && dz == std::floor(dz),
                "human VTCCD vertex or EECCD edge index is not 0..2");
            require(dx == static_cast<float>(primitiveA) &&
                dy == static_cast<float>(primitiveB),
                "diagnostic primitive ids disagree with failure record");
            result.headerLine = line;
            result.sourceLines.push_back(line);
            result.record.attempt = {1u, step, microtick, slot};
            result.record.primitivePair = {primitiveA, primitiveB, 0u, 0u};
            result.record.diagnostics = {
                parseLogFloat(dx, "diagnostics.x"),
                parseLogFloat(dy, "diagnostics.y"),
                parseLogFloat(dz, "diagnostics.z"),
                parseLogFloat(dw, "diagnostics.w")
            };
            result.record.parameters = {
                parseLogFloat(thickness, "thickness"),
                parseLogFloat(dtA, "dt_a"),
                parseLogFloat(dtB, "dt_b"),
                parseLogFloat(newton, "newton")
            };
            require(thickness >= 0.0f, "negative contact thickness");
            haveHeader = true;
            continue;
        }
        if (!haveHeader) continue;
        if (line.rfind("contact_primitive ", 0u) == 0u) {
            unsigned side=0u, object=0u, kind=0u, source=0u;
            const int count = std::sscanf(line.c_str(),
                "contact_primitive side=%u object=%u kind=%u source=%u",
                &side, &object, &kind, &source);
            require(count == 4 && side < 2u, "malformed contact_primitive line");
            require((primitiveMask & (1u << side)) == 0u,
                "duplicate contact_primitive side");
            result.sourceLines.push_back(line);
            NMContinuumSurfacePrimitiveGPU& primitive = result.record.primitives[side];
            primitive.nodesAndObject.w = object;
            primitive.identity.x = source;
            primitive.identity.y = kind;
            primitiveMask |= 1u << side;
            continue;
        }
        if (line.rfind("contact_vertex ", 0u) == 0u) {
            unsigned side=0u, corner=0u, node=0u;
            float sx=0.0f, sy=0.0f, sz=0.0f;
            float fx=0.0f, fy=0.0f, fz=0.0f;
            const int count = std::sscanf(line.c_str(),
                "contact_vertex side=%u corner=%u node=%u start=%f,%f,%f finish=%f,%f,%f",
                &side, &corner, &node, &sx, &sy, &sz, &fx, &fy, &fz);
            require(count == 9 && side < 2u && corner < 3u,
                "malformed contact_vertex line");
            const std::uint32_t local = 3u * side + corner;
            require((vertexMask & (1u << local)) == 0u,
                "duplicate contact_vertex endpoint");
            result.sourceLines.push_back(line);
            result.record.start[local] = {
                parseLogFloat(sx, "start.x"), parseLogFloat(sy, "start.y"),
                parseLogFloat(sz, "start.z"), 0.0f
            };
            result.record.finish[local] = {
                parseLogFloat(fx, "finish.x"), parseLogFloat(fy, "finish.y"),
                parseLogFloat(fz, "finish.z"), 0.0f
            };
            NMContinuumSurfacePrimitiveGPU& primitive = result.record.primitives[side];
            if (corner == 0u) primitive.nodesAndObject.x = node;
            else if (corner == 1u) primitive.nodesAndObject.y = node;
            else primitive.nodesAndObject.z = node;
            vertexMask |= 1u << local;
        }
    }
    require(haveHeader, "log has no deformable_contact_failure record");
    require(primitiveMask == 3u && vertexMask == 0x3fu &&
        result.sourceLines.size() == 9u,
        "human failure record is missing or duplicates primitive/endpoint rows");
    return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> library) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        @"numi_matter_metal::nm_contact_vtccd_iteration_budget_replay"];
    require(function != nil, "iteration-budget replay kernel missing from metallib");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error != nil
        ? std::string("cannot compile replay pipeline: ") + [[error localizedDescription] UTF8String]
        : "cannot compile replay pipeline");
    return pipeline;
}

id<MTLComputePipelineState> makeHumanFailurePipeline(
    id<MTLDevice> device, id<MTLLibrary> library) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        @"numi_matter_metal::nm_contact_human_failure_vtccd_replay"];
    require(function != nil, "human-failure VTCCD replay kernel missing from metallib");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error != nil
        ? std::string("cannot compile human-failure pipeline: ") +
            [[error.localizedDescription description] UTF8String]
        : "cannot compile human-failure pipeline");
    return pipeline;
}

id<MTLComputePipelineState> makeHumanEdgeFailurePipeline(
    id<MTLDevice> device, id<MTLLibrary> library) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        @"numi_matter_metal::nm_contact_human_failure_eeccd_replay"];
    require(function != nil, "human-failure EECCD replay kernel missing from metallib");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error != nil
        ? std::string("cannot compile EECCD failure replay pipeline: ") +
            [[error.localizedDescription description] UTF8String]
        : "cannot compile EECCD failure replay pipeline");
    return pipeline;
}

void runHumanEdgeFailure(id<MTLDevice> device, id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         const HumanFailureInput& input,
                         const bool requireSeparated) {
    require(input.record.diagnostics.z >= 0.0f &&
        input.record.diagnostics.z <= 2.0f &&
        input.record.diagnostics.w >= 0.0f &&
        input.record.diagnostics.w <= 2.0f,
        "--human-ee-failure requires edge indices in diagnostics.z/w");
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:&input.record
        length:sizeof(input.record) options:MTLResourceStorageModeShared];
    std::array<Float4, 18> summaries{};
    id<MTLBuffer> summaryBuffer = [device newBufferWithBytes:summaries.data()
        length:sizeof(summaries) options:MTLResourceStorageModeShared];
    require(inputBuffer != nil && summaryBuffer != nil,
        "cannot allocate human EECCD replay buffers");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "Metal command buffer unavailable");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "Metal compute encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0u atIndex:0u];
    [encoder setBuffer:summaryBuffer offset:0u atIndex:1u];
    [encoder dispatchThreads:MTLSizeMake(9u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(9u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("Metal EECCD failure replay failed: ") +
                [[command.error.localizedDescription description] UTF8String]
            : "Metal EECCD failure replay did not complete");
    const auto* output = static_cast<const Float4*>([summaryBuffer contents]);
    require(output != nullptr, "human EECCD replay output is not CPU-readable");
    const char* names[9] = {
        "production_separating_plane", "legacy", "separating_plane",
        "interval_64", "interval_256", "interval_1024", "interval_96",
        "compensated_swept_projection", "thickness_plus_0.2nm_hit_control"
    };
    if (requireSeparated) {
        require(output[0][0] == 1.0f && output[0][1] == 0.0f,
            "exact witness was not certified clear by production EECCD");
        require(output[14][0] == 1.0f,
            "exact witness lacks the compensated swept projection certificate");
        require(output[16][0] == 1.0f && output[16][1] == 1.0f,
            "nearby true-contact control was incorrectly certified clear");
    }
    std::cout << std::setprecision(9)
        << "{\"schema\":\"numi.matter.human_eeccd_failure_replay.v3\","
        << "\"source_log\":" << jsonQuote(input.sourcePath)
        << ",\"control_step\":" << input.record.attempt.y
        << ",\"microtick\":" << input.record.attempt.z
        << ",\"candidate_slot\":" << input.record.attempt.w
        << ",\"primitive_pair\":[" << input.record.primitivePair.x << ','
        << input.record.primitivePair.y << "],\"edge_indices\":["
        << input.record.diagnostics.z << ',' << input.record.diagnostics.w
        << "],\"thickness_m\":" << input.record.parameters.x
        << ",\"capture_lines\":[";
    for (std::size_t i = 0u; i < input.sourceLines.size(); ++i)
        std::cout << (i == 0u ? "" : ",") << jsonQuote(input.sourceLines[i]);
    std::cout << "],\"stages\":[";
    for (std::uint32_t i = 0u; i < 9u; ++i) {
        const Float4 row = output[2u * i];
        const Float4 detail = output[2u * i + 1u];
        require(std::isfinite(row[0]) && (row[0] == 0.0f || row[0] == 1.0f) &&
            std::isfinite(row[1]) && (row[1] == 0.0f || row[1] == 1.0f) &&
            std::isfinite(row[2]), "invalid EECCD failure replay result");
        require(std::isfinite(detail[0]) && std::isfinite(detail[1]) &&
            std::isfinite(detail[2]) && detail[0] >= 0.0f &&
            detail[1] >= 0.0f,
            "invalid EECCD interval search details");
        std::cout << (i == 0u ? "" : ",") << "{\"stage\":"
            << jsonQuote(names[i]) << ",\"certified\":"
            << (row[0] != 0.0f ? "true" : "false") << ",\"hit\":"
            << (row[1] != 0.0f ? "true" : "false") << ",\"toi\":" << row[2]
            << ",\"gap_to_effective_thickness_m\":";
        if (std::isfinite(row[3])) std::cout << row[3];
        else std::cout << "null";
        std::cout << ",\"interval_evaluations\":" << detail[0]
            << ",\"unresolved_intervals\":" << detail[1]
            << ",\"evaluation_cap\":" << detail[2] << '}';
    }
    std::cout << "]}\n";
}


using EEPoint = std::array<float, 3>;

NMDeformableContactFailureGPU makeEEIntervalRecord(
    const float thickness, const EEPoint p0, const EEPoint p1,
    const EEPoint q0, const EEPoint q1) {
    NMDeformableContactFailureGPU record{};
    record.attempt = {1u, 0u, 0u, 0u};
    record.primitivePair = {0u, 1u, 0u, 0u};
    record.diagnostics = {0.0f, 1.0f, 2.0f, 0.0f};
    record.parameters = {thickness, 0.0f, 0.0f, 0.0f};
    const EEPoint zero{0.0f, 0.0f, 0.0f};
    for (std::uint32_t i = 0u; i < 6u; ++i) {
        record.start[i] = {zero[0], zero[1], zero[2], 0.0f};
        record.finish[i] = record.start[i];
    }
    const auto setVertex = [&](const std::uint32_t index, const EEPoint point) {
        record.start[index] = {point[0], point[1], point[2], 0.0f};
        record.finish[index] = record.start[index];
    };
    // diagnostics.z/w select edge 2=(corner2,corner0) and edge 0=(corner0,corner1).
    setVertex(2u, p0);
    setVertex(0u, p1);
    setVertex(3u, q0);
    setVertex(4u, q1);
    return record;
}

std::array<Float4, 14> runEEIntervalCase(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const NMDeformableContactFailureGPU& record) {
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:&record
        length:sizeof(record) options:MTLResourceStorageModeShared];
    std::array<Float4, 14> initial{};
    id<MTLBuffer> summaryBuffer = [device newBufferWithBytes:initial.data()
        length:sizeof(initial) options:MTLResourceStorageModeShared];
    require(inputBuffer != nil && summaryBuffer != nil,
        "cannot allocate EE interval regression buffers");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "Metal command buffer unavailable");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "Metal compute encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0u atIndex:0u];
    [encoder setBuffer:summaryBuffer offset:0u atIndex:1u];
    [encoder dispatchThreads:MTLSizeMake(7u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(7u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("EE interval regression failed: ") +
                [[command.error.localizedDescription description] UTF8String]
            : "EE interval regression did not complete");
    const auto* output = static_cast<const Float4*>([summaryBuffer contents]);
    require(output != nullptr, "EE interval regression output is not CPU-readable");
    std::array<Float4, 14> result{};
    for (std::size_t i = 0u; i < result.size(); ++i) result[i] = output[i];
    return result;
}

void requireEEStage(const std::array<Float4, 14>& rows,
                    const std::uint32_t stage, const bool certified,
                    const bool hit, const char* name) {
    const Float4& row = rows[2u * stage];
    const Float4& detail = rows[2u * stage + 1u];
    require(std::isfinite(row[0]) && row[0] == (certified ? 1.0f : 0.0f) &&
        row[1] == (hit ? 1.0f : 0.0f),
        std::string("EE interval regression contract failed: ") + name);
    require(std::isfinite(detail[0]) && detail[0] >= 0.0f &&
        detail[0] <= detail[2] && (stage < 3u || detail[2] > 0.0f),
        std::string("EE interval regression work bound failed: ") + name);
}

void runEEIntervalRegressions(
    id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const NMDeformableContactFailureGPU& captured) {
    const auto capturedRows = runEEIntervalCase(device, queue, pipeline, captured);
    requireEEStage(capturedRows, 0u, true, false, "production captured clear");
    requireEEStage(capturedRows, 3u, false, false, "64-evaluation cap stays bounded");
    requireEEStage(capturedRows, 6u, true, false, "96-evaluation cap certifies witness");

    NMDeformableContactFailureGPU reversed = captured;
    std::swap(reversed.start[2], reversed.start[0]);
    std::swap(reversed.finish[2], reversed.finish[0]);
    std::swap(reversed.start[3], reversed.start[4]);
    std::swap(reversed.finish[3], reversed.finish[4]);
    requireEEStage(runEEIntervalCase(device, queue, pipeline, reversed),
        6u, true, false, "reversed endpoint orientation");

    NMDeformableContactFailureGPU sidePermuted = captured;
    for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
        std::swap(sidePermuted.start[corner], sidePermuted.start[corner + 3u]);
        std::swap(sidePermuted.finish[corner], sidePermuted.finish[corner + 3u]);
    }
    sidePermuted.diagnostics.z = 0.0f;
    sidePermuted.diagnostics.w = 2.0f;
    requireEEStage(runEEIntervalCase(device, queue, pipeline, sidePermuted),
        6u, true, false, "primitive-side permutation");

    const float thickness = 1.0e-5f;
    const auto crossing = makeEEIntervalRecord(
        thickness, EEPoint{-1.0e-3f, 0.0f, 0.0f},
        EEPoint{1.0e-3f, 0.0f, 0.0f},
        EEPoint{0.0f, -1.0e-3f, 0.0f},
        EEPoint{0.0f, 1.0e-3f, 0.0f});
    requireEEStage(runEEIntervalCase(device, queue, pipeline, crossing),
        6u, true, true, "static true hit");

    const float roundoff = 8.0f * 1.1920928955078125e-7f * 1.0e-3f;
    const float ambiguousGap = thickness + 1.5f * roundoff;
    const auto nearParallel = makeEEIntervalRecord(
        thickness, EEPoint{-1.0e-3f, 0.0f, 0.0f},
        EEPoint{1.0e-3f, 0.0f, 0.0f},
        EEPoint{-1.0e-3f, ambiguousGap, 0.0f},
        EEPoint{1.0e-3f, ambiguousGap, 0.0f});
    requireEEStage(runEEIntervalCase(device, queue, pipeline, nearParallel),
        6u, false, false, "near-parallel uncertainty remains fail-closed");

    const EEPoint offset{0.125f, -0.25f, 0.5f};
    const auto translate = [&](const EEPoint point) {
        return EEPoint{point[0] + offset[0], point[1] + offset[1],
            point[2] + offset[2]};
    };
    auto translatedPermuted = makeEEIntervalRecord(
        thickness, translate(EEPoint{1.0e-3f, 0.0f, 0.0f}),
        translate(EEPoint{-1.0e-3f, 0.0f, 0.0f}),
        translate(EEPoint{1.0e-3f, 3.0e-5f, 0.0f}),
        translate(EEPoint{-1.0e-3f, 3.0e-5f, 0.0f}));
    requireEEStage(runEEIntervalCase(device, queue, pipeline, translatedPermuted),
        6u, true, false, "translated and endpoint-permuted clear sweep");

    std::cout << "PASS EE CCD interval cap96: captured clear, retained cap64 "
        << "uncertainty, true hit, near-parallel fail-closed, reversal, "
        << "primitive permutation, and translated clear cases\n";
}

void runHumanFailure(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline,
                     const HumanFailureInput& input, const bool requireSeparatedWitness,
                     const bool requireHitControl = false) {
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:&input.record
        length:sizeof(input.record) options:MTLResourceStorageModeShared];
    std::array<Float4, 4> summaries{};
    id<MTLBuffer> summaryBuffer = [device newBufferWithBytes:summaries.data()
        length:sizeof(summaries) options:MTLResourceStorageModeShared];
    require(inputBuffer != nil && summaryBuffer != nil,
        "cannot allocate human-failure replay buffers");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "Metal command buffer unavailable");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "Metal compute encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0u atIndex:0u];
    [encoder setBuffer:summaryBuffer offset:0u atIndex:1u];
    [encoder dispatchThreads:MTLSizeMake(4u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(4u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("Metal human-failure replay failed: ") +
                [[command.error.localizedDescription description] UTF8String]
            : "Metal human-failure replay did not complete");
    const auto* output = static_cast<const Float4*>([summaryBuffer contents]);
    require(output != nullptr, "human-failure replay output is not CPU-readable");
    const char* names[4] = {
        "production_separating_plane", "legacy", "budget16_separating_plane",
        "interval_fallback"
    };
    const std::uint32_t packedFeature =
        (static_cast<std::uint32_t>(input.record.diagnostics.w) - 3u) * 3u +
        static_cast<std::uint32_t>(input.record.diagnostics.z);
    std::cout << std::setprecision(9)
        << "{\"schema\":\"numi.matter.human_vtccd_failure_replay.v2\","
        << "\"source_log\":" << jsonQuote(input.sourcePath)
        << ",\"control_step\":" << input.record.attempt.y
        << ",\"microtick\":" << input.record.attempt.z
        << ",\"candidate_slot\":" << input.record.attempt.w
        << ",\"primitive_pair\":[" << input.record.primitivePair.x << ','
        << input.record.primitivePair.y << "],\"captured_failure_line\":"
        << jsonQuote(input.headerLine) << ",\"capture_lines\":[";
    for (std::size_t i = 0u; i < input.sourceLines.size(); ++i)
        std::cout << (i == 0u ? "" : ",") << jsonQuote(input.sourceLines[i]);
    std::cout << "],\"direction_code\":"
        << input.record.diagnostics.w << ",\"vertex_index\":"
        << input.record.diagnostics.z << ",\"packed_feature\":"
        << packedFeature << ",\"thickness_m\":" << input.record.parameters.x
        << ",\"stages\":[";
    for (std::uint32_t i = 0u; i < 4u; ++i) {
        const Float4 row = output[i];
        require(std::isfinite(row[0]) && (row[0] == 0.0f || row[0] == 1.0f) &&
            std::isfinite(row[1]) && (row[1] == 0.0f || row[1] == 1.0f) &&
            std::isfinite(row[2]), "invalid human-failure replay result");
        if (requireSeparatedWitness && (i == 0u || i == 3u)) {
            require(row[0] == 1.0f && row[1] == 0.0f && row[2] == 1.0f &&
                std::isfinite(row[3]) && row[3] > 0.0f,
                "captured near-clear sweep lacks a physical-thickness clear certificate");
        }
        if (requireHitControl && (i == 0u || i == 3u)) {
            constexpr float epsilon = 1.1920928955078125e-7f;
            constexpr float syntheticCoordinateScale = 1.0e-3f;
            const float acceptedRoundoff = 8.0f * epsilon *
                std::max(input.record.parameters.x, syntheticCoordinateScale);
            if (!(row[0] == 1.0f && row[1] == 1.0f &&
                    std::isfinite(row[3]) && row[3] <= acceptedRoundoff)) {
                std::cerr << std::setprecision(9) << "VT_HIT_CONTROL stage="
                    << i << " certified=" << row[0] << " hit=" << row[1]
                    << " toi=" << row[2] << " gap=" << row[3]
                    << " allowed_roundoff=" << acceptedRoundoff << "\n";
                throw std::runtime_error("penetrating VT control was not certified as a hit");
            }
        }
        std::cout << (i == 0u ? "" : ",") << "{\"stage\":"
            << jsonQuote(names[i]) << ",\"certified\":"
            << (row[0] != 0.0f ? "true" : "false") << ",\"hit\":"
            << (row[1] != 0.0f ? "true" : "false") << ",\"toi\":" << row[2]
            << ",\"gap_to_physical_thickness_m\":";
        if (std::isfinite(row[3])) std::cout << row[3];
        else std::cout << "null";
        std::cout << '}';
    }
    std::cout << "]}\n";
}

void runOne(id<MTLDevice> device, id<MTLCommandQueue> queue,
            id<MTLComputePipelineState> pipeline, const CapturedInput& input,
            std::ostream& output, const bool first) {
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:&input.record
        length:sizeof(input.record) options:MTLResourceStorageModeShared];
    require(inputBuffer != nil, "cannot allocate captured observer input");
    id<MTLBuffer> budgetsBuffer = [device newBufferWithBytes:kBudgets.data()
        length:sizeof(kBudgets) options:MTLResourceStorageModeShared];
    require(budgetsBuffer != nil, "cannot allocate iteration budget inputs");
    std::array<Float4, 8> initialSummaries{};
    std::array<Float4, 4u * kTraceStride> initialTrace{};
    id<MTLBuffer> summaryBuffer = [device newBufferWithBytes:initialSummaries.data()
        length:sizeof(initialSummaries) options:MTLResourceStorageModeShared];
    id<MTLBuffer> traceBuffer = [device newBufferWithBytes:initialTrace.data()
        length:sizeof(initialTrace) options:MTLResourceStorageModeShared];
    require(summaryBuffer != nil && traceBuffer != nil, "cannot allocate bounded replay outputs");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "Metal command buffer unavailable");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "Metal compute encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0u atIndex:0u];
    [encoder setBuffer:budgetsBuffer offset:0u atIndex:1u];
    [encoder setBuffer:summaryBuffer offset:0u atIndex:2u];
    [encoder setBuffer:traceBuffer offset:0u atIndex:3u];
    [encoder dispatchThreads:MTLSizeMake(4u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(4u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("Metal replay failed: ") + [[command.error localizedDescription] UTF8String]
            : "Metal replay did not complete");

    const auto* summaries = static_cast<const Float4*>([summaryBuffer contents]);
    const auto* trace = static_cast<const Float4*>([traceBuffer contents]);
    require(summaries != nullptr && trace != nullptr, "Metal replay outputs are not CPU-readable");
    std::array<float, 4> capturedReturn{};
    std::memcpy(capturedReturn.data(), &input.record.result, sizeof(capturedReturn));
    output << (first ? "" : ",") << "{\"capture\":" << jsonQuote(input.capturePath)
        << ",\"capture_manifest_sha256\":" << jsonQuote(input.captureManifestSHA)
        << ",\"payload_sha256\":" << jsonQuote(input.payloadSHA)
        << ",\"observer_sha256\":" << jsonQuote(input.observerSHA)
        << ",\"source_physics_fingerprint\":" << jsonQuote(kSourcePhysicsFingerprint)
        << ",\"input_device_program_fingerprint\":" << jsonQuote(input.deviceFingerprint)
        << ",\"captured_return\":[" << capturedReturn[0] << ',' << capturedReturn[1]
        << ',' << capturedReturn[2] << ',' << capturedReturn[3] << "],\"budgets\":[";

    for (std::size_t slot = 0u; slot < kBudgets.size(); ++slot) {
        const Float4 summary = summaries[2u * slot];
        const Float4 metadata = summaries[2u * slot + 1u];
        const std::uint32_t budget = kBudgets[slot];
        require(metadata[1] == static_cast<float>(budget) && metadata[2] == 5.0f && metadata[3] == 1.0f,
            "Metal kernel rejected captured VT input or budget slot");
        require(std::isfinite(summary[0]) && std::isfinite(summary[1]) &&
            std::isfinite(summary[2]) && std::isfinite(summary[3]), "nonfinite replay summary");
        require(summary[0] == 0.0f || summary[1] != 0.0f,
            "replay returned a hit without certification");
        const std::uint32_t executed = static_cast<std::uint32_t>(metadata[0]);
        require(executed > 0u && executed <= budget && metadata[0] == static_cast<float>(executed),
            "invalid replay iteration count");
        std::uint32_t finalFlags = 0u;
        if (slot == 0u) {
            require(std::memcmp(&summary, capturedReturn.data(), sizeof(summary)) == 0,
                "budget16 replay is not bitwise equal to the captured production result");
        }
        output << (slot == 0u ? "" : ",") << "{\"budget\":" << budget
            << ",\"return\":[" << summary[0] << ',' << summary[1] << ','
            << summary[2] << ',' << summary[3] << "],\"iterations\":" << executed
            << ",\"certified\":" << (summary[1] != 0.0f ? "true" : "false")
            << ",\"hit\":" << (summary[0] != 0.0f ? "true" : "false")
            << ",\"exhausted_uncertified\":"
            << ((summary[1] == 0.0f && executed == budget) ? "true" : "false")
            << ",\"trace\":[";
        for (std::uint32_t it = 0u; it < executed; ++it) {
            const Float4 row = trace[slot * kTraceStride + it];
            require(std::isfinite(row[0]) && std::isfinite(row[1]) &&
                std::isfinite(row[2]) && std::isfinite(row[3]), "nonfinite iteration trace");
            const std::uint32_t flags = static_cast<std::uint32_t>(row[3]);
            require(row[3] == static_cast<float>(flags) && flags <= 3u &&
                ((flags & 2u) == 0u || (flags & 1u) != 0u), "invalid iteration terminal flags");
            if (it + 1u < executed) require(flags == 0u,
                "a nonterminal CCD sample was marked certified/hit");
            finalFlags = flags;
            output << (it == 0u ? "" : ",") << "{\"iteration\":" << it
                << ",\"time_fraction\":" << row[0] << ",\"gap_m\":" << row[1]
                << ",\"separation_m\":" << row[2]
                << ",\"certified\":" << ((flags & 1u) != 0u ? "true" : "false")
                << ",\"hit\":" << ((flags & 2u) != 0u ? "true" : "false") << '}';
        }
        require(((finalFlags & 1u) != 0u) == (summary[1] != 0.0f) &&
            ((finalFlags & 2u) != 0u) == (summary[0] != 0.0f),
            "terminal trace flags disagree with helper return");
        for (std::uint32_t it = executed; it < kTraceStride; ++it) {
            const Float4 row = trace[slot * kTraceStride + it];
            require(row == Float4{}, "trace wrote beyond actual iteration count");
        }
        output << "]}";
    }
    output << "]}";
}
} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            if (argc == 3 && std::string(argv[1]) == "--ee-interval-regression") {
                const HumanFailureInput input = loadHumanFailureInput(argv[2]);
                require(input.record.diagnostics.z >= 0.0f &&
                    input.record.diagnostics.z <= 2.0f &&
                    input.record.diagnostics.w >= 0.0f &&
                    input.record.diagnostics.w <= 2.0f,
                    "--ee-interval-regression requires an EE failure record");
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                require(device != nil, "Metal device unavailable");
                NSError* error = nil;
                NSString* libraryPath = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
                id<MTLLibrary> library = [device newLibraryWithURL:
                    [NSURL fileURLWithPath:libraryPath] error:&error];
                require(library != nil, error != nil
                    ? std::string("cannot load Matter metallib: ") +
                        [[error.localizedDescription description] UTF8String]
                    : "cannot load Matter metallib");
                id<MTLCommandQueue> queue = [device newCommandQueue];
                require(queue != nil, "Metal command queue unavailable");
                runEEIntervalRegressions(device, queue,
                    makeHumanEdgeFailurePipeline(device, library), input.record);
                return 0;
            }
            if ((argc == 3 || argc == 4) &&
                std::string(argv[1]) == "--human-ee-failure") {
                require(argc == 3 || std::string(argv[3]) == "--require-separated",
                    "unknown human-ee-failure option");
                const HumanFailureInput input = loadHumanFailureInput(argv[2]);
                require(input.record.diagnostics.z >= 0.0f &&
                    input.record.diagnostics.z <= 2.0f &&
                    input.record.diagnostics.w >= 0.0f &&
                    input.record.diagnostics.w <= 2.0f,
                    "input is not an edge-edge failure record");
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                require(device != nil, "Metal device unavailable");
                NSError* error = nil;
                NSString* libraryPath = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
                id<MTLLibrary> library = [device newLibraryWithURL:
                    [NSURL fileURLWithPath:libraryPath] error:&error];
                require(library != nil, error != nil
                    ? std::string("cannot load Matter metallib: ") +
                        [[error.localizedDescription description] UTF8String]
                    : "cannot load Matter metallib");
                id<MTLCommandQueue> queue = [device newCommandQueue];
                require(queue != nil, "Metal command queue unavailable");
                runHumanEdgeFailure(device, queue,
                    makeHumanEdgeFailurePipeline(device, library), input,
                    argc == 4);
                return 0;
            }
            if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--human-failure") {
                require(argc == 3 || std::string(argv[3]) == "--require-separated",
                    "unknown human-failure option");
                const HumanFailureInput input = loadHumanFailureInput(argv[2]);
                require(input.record.diagnostics.w == 3.0f ||
                    input.record.diagnostics.w == 4.0f,
                    "--human-failure requires a VTCCD failure record");
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                require(device != nil, "Metal device unavailable");
                NSError* error = nil;
                NSString* libraryPath = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
                id<MTLLibrary> library = [device newLibraryWithURL:
                    [NSURL fileURLWithPath:libraryPath] error:&error];
                require(library != nil, error != nil
                    ? std::string("cannot load Matter metallib: ") +
                        [[error.localizedDescription description] UTF8String]
                    : "cannot load Matter metallib");
                id<MTLCommandQueue> queue = [device newCommandQueue];
                require(queue != nil, "Metal command queue unavailable");
                id<MTLComputePipelineState> pipeline =
                    makeHumanFailurePipeline(device, library);
                runHumanFailure(device, queue, pipeline, input, argc == 4);
                HumanFailureInput penetrating;
                penetrating.sourcePath = "synthetic VTCCD penetrating control";
                penetrating.record.attempt = {1u, 0u, 0u, 0u};
                penetrating.record.diagnostics = {0.0f, 0.0f, 1.0f, 4.0f};
                penetrating.record.parameters = {1.0e-5f, 0.0f, 0.0f, 0.0f};
                penetrating.record.start[0] = {-1.0e-3f, -1.0e-3f, 0.0f, 0.0f};
                penetrating.record.start[1] = { 1.0e-3f, -1.0e-3f, 0.0f, 0.0f};
                penetrating.record.start[2] = { 0.0f,     1.0e-3f, 0.0f, 0.0f};
                for (std::uint32_t corner = 0u; corner < 3u; ++corner)
                    penetrating.record.finish[corner] = penetrating.record.start[corner];
                penetrating.record.start[3] = {0.0f, 0.0f, 1.0e-3f, 0.0f};
                penetrating.record.finish[3] = penetrating.record.start[3];
                penetrating.record.start[4] = {0.0f, 0.0f, 2.0e-5f, 0.0f};
                penetrating.record.finish[4] = {0.0f, 0.0f, -2.0e-5f, 0.0f};
                penetrating.record.start[5] = {0.0f, 0.0f, 1.0e-3f, 0.0f};
                penetrating.record.finish[5] = penetrating.record.start[5];
                runHumanFailure(device, queue, pipeline, penetrating, false, true);
                std::cout << "PASS VTCCD physical-clear witness and penetrating control\n";
                return 0;
            }
            require(argc == 3, "usage: checker <trial21-capture.json> <trial22-capture.json> | --human-failure LOG [--require-separated] | --human-ee-failure LOG [--require-separated] | --ee-interval-regression LOG");
            const CapturedInput trial21 = loadCapturedInput(argv[1],
                kTrial21ManifestSHA, kTrial21PayloadSHA, kTrial21ObserverSHA);
            const CapturedInput trial22 = loadCapturedInput(argv[2],
                kTrial22ManifestSHA, kTrial22PayloadSHA, kTrial22ObserverSHA);
            require(trial21.record.result.x == 0.0f && trial21.record.result.y == 0.0f,
                "trial21 input no longer records the retained uncertified result");
            require(trial22.record.result.x == 1.0f && trial22.record.result.y == 1.0f,
                "trial22 input no longer records the retained certified hit");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            NSString* libraryPath = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath]
                error:&error];
            require(library != nil, error != nil
                ? std::string("cannot load Matter metallib: ") + [[error localizedDescription] UTF8String]
                : "cannot load Matter metallib");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            id<MTLComputePipelineState> pipeline = makePipeline(device, library);
            std::cout << std::setprecision(10)
                << "{\"schema\":\"numi.matter.vtccd_iteration_budget_replay.v1\","
                << "\"budgets\":[16,32,64,128],\"trace_stride\":128,"
                << "\"trace_lanes\":[\"time\",\"gap_to_effective_thickness_m\",\"separation_m\",\"flags_bit0_certified_bit1_hit\"],"
                << "\"production_default_budget\":16,\"production_tolerance_changed\":false,\"captures\":[";
            runOne(device, queue, pipeline, trial21, std::cout, true);
            runOne(device, queue, pipeline, trial22, std::cout, false);
            std::cout << "]}\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "FAIL " << exception.what() << '\n';
            return 1;
        }
    }
}
