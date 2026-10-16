#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
            require(argc == 3, "usage: checker <trial21-capture.json> <trial22-capture.json>");
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
