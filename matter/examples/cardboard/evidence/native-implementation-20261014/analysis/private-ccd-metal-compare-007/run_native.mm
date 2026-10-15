#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Float4 = std::array<float, 4>;
constexpr std::uint32_t kCaseCount = 4u;
constexpr std::uint32_t kFloat4PerCase = 9u;
constexpr std::uint32_t kDiagnosticRows = 5u;
constexpr std::uint32_t kTraceRowsPerIteration = 3u;
constexpr std::uint32_t kTraceIterations = 17u;
constexpr char kExpectedContactSHA[] =
    "3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132";
constexpr char kExpectedFrozenManifestSHA[] =
    "5fb22c61c280f576f15677c8cf6dbbc174beceaa225973242f6d53f9c7957446";
constexpr char kExpectedCasesSHA[] =
    "80eaab55f3d471934382eaec6165c2c5a717abafb94cf08dd75f7bb69c42d3f9";
constexpr char kExpectedGeneratedMetalSHA[] =
    "b9298fcd0eed812944296c6880ecbd4fac2ebc7306d67bdc319be3a20130a270";
constexpr char kExpectedExtractedFunctionSHA[] =
    "a76e6024194848f935521b10ed72fb4b2938a321636bf06faf7ca02e07fb84c9";
constexpr char kExpectedInstrumentedFunctionSHA[] =
    "a810aa0e22334f9eb3819bb5cc17f2830b02ef1e87e3cb4accca0d4661782518";
constexpr char kExpectedInstrumented16SHA[] =
    "96613fc6f1ea3eeb2eda91259994fb40468a2ebc33825b6c528151d3fd5ed3d3";
constexpr char kExpectedInstrumented17SHA[] =
    "62260b72d5b810148dbc6638e73ba316dfae93904afd8a0ca74176b990538391";
constexpr char kExpectedCasesSchema[] =
    "numi.matter.private_ee_ccd_native_cases.v1";
constexpr char kExpectedBuildSchema[] =
    "numi.matter.private_ee_ccd_metal_diagnostic_build.v2";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint8_t> readBytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open " + path);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), {});
}

std::string sha256(const void* bytes, std::size_t length) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    CC_SHA256(bytes, static_cast<CC_LONG>(length), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << unsigned(byte);
    return out.str();
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
    return sha256(bytes.data(), bytes.size());
}

std::string jsonStringField(const std::string& text, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const auto keyAt = text.find(marker);
    require(keyAt != std::string::npos, "build manifest missing " + key);
    const auto colon = text.find(':', keyAt + marker.size());
    require(colon != std::string::npos, "malformed build manifest field " + key);
    const auto open = text.find('"', colon + 1u);
    require(open != std::string::npos, "build manifest field is not a string: " + key);
    const auto close = text.find('"', open + 1u);
    require(close != std::string::npos, "unterminated build manifest field " + key);
    return text.substr(open + 1u, close - open - 1u);
}

// Read a string field from the root JSON object. The capture-case manifest
// repeats payload_sha256 inside each capture entry, so a first-match substring
// search can bind the runner to one source payload instead of the packed corpus.
std::string jsonRootStringField(const std::string& text, const std::string& key) {
    auto skipWhitespace = [&text](std::size_t& at) {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' ||
               text[at] == '\r' || text[at] == '\n')) ++at;
    };
    auto parseString = [&text](std::size_t& at) {
        require(at < text.size() && text[at] == '"', "malformed JSON string field");
        ++at;
        std::string value;
        while (at < text.size()) {
            const char c = text[at++];
            if (c == '"') return value;
            if (c == '\\') {
                require(at < text.size(), "unterminated JSON escape");
                const char escaped = text[at++];
                switch (escaped) {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case '/': value.push_back('/'); break;
                    case 'b': value.push_back('\b'); break;
                    case 'f': value.push_back('\f'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default: throw std::runtime_error("unsupported escaped character in JSON field");
                }
            } else {
                value.push_back(c);
            }
        }
        throw std::runtime_error("unterminated JSON string");
    };

    std::size_t at = 0u;
    unsigned depth = 0u;
    while (at < text.size()) {
        const char c = text[at];
        if (c == '{' || c == '[') {
            ++depth;
            ++at;
            continue;
        }
        if (c == '}' || c == ']') {
            if (depth > 0u) --depth;
            ++at;
            continue;
        }
        if (c != '"') {
            ++at;
            continue;
        }
        const unsigned stringDepth = depth;
        const std::string token = parseString(at);
        const std::size_t afterString = at;
        skipWhitespace(at);
        if (stringDepth == 1u && at < text.size() && text[at] == ':' && token == key) {
            ++at;
            skipWhitespace(at);
            return parseString(at);
        }
        // parseString leaves the scanner at the end of the token; whitespace
        // was inspected only to determine whether it is an object key.
        if (afterString < at) at = afterString;
    }
    throw std::runtime_error("root JSON object missing string field " + key);
}

void jsonFloat(std::ostream& out, float value) {
    if (std::isfinite(value)) out << std::setprecision(9) << value;
    else out << "null";
}

void jsonFloat4(std::ostream& out, const Float4& v) {
    out << '[';
    for (std::size_t i = 0; i < 4; ++i) {
        if (i) out << ',';
        jsonFloat(out, v[i]);
    }
    out << ']';
}

const char* classification(const Float4& result);

void validateReportJSON(const std::string& text) {
    NSData* bytes = [NSData dataWithBytes:text.data() length:text.size()];
    NSError* error = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:bytes
        options:NSJSONReadingFragmentsAllowed error:&error];
    require(parsed != nil && [parsed isKindOfClass:[NSDictionary class]],
        error ? std::string("generated report is not valid JSON: ") +
            std::string([[error localizedDescription] UTF8String])
              : "generated report root is not a JSON object");
    NSDictionary* root = (NSDictionary*)parsed;
    require([[root objectForKey:@"schema"] isEqual:@"numi.matter.private_ee_ccd_metal_diagnostic_result.v2"],
        "generated report has unexpected schema");
    id casesValue = [root objectForKey:@"cases"];
    require(casesValue != nil && [casesValue isKindOfClass:[NSArray class]] &&
            [(NSArray*)casesValue count] == kCaseCount,
        "generated report does not contain exactly four cases");
    for (id value in (NSArray*)casesValue) {
        require([value isKindOfClass:[NSDictionary class]],
            "generated report case is not an object");
        NSDictionary* item = (NSDictionary*)value;
        id instrumentedValue = [item objectForKey:@"instrumented"];
        require(instrumentedValue != nil &&
                [instrumentedValue isKindOfClass:[NSDictionary class]],
            "generated report case lacks instrumented result");
        id iterationsValue = [(NSDictionary*)instrumentedValue objectForKey:@"iterations"];
        require(iterationsValue != nil && [iterationsValue isKindOfClass:[NSArray class]],
            "generated report case lacks iteration array");
        NSArray* iterations = (NSArray*)iterationsValue;
        for (NSUInteger i = 0; i < [iterations count]; ++i)
            require([[iterations objectAtIndex:i] isKindOfClass:[NSDictionary class]],
                "generated iteration is not an object");
    }
}

std::string serializeReport(
    const std::string& contactSHA,
    const std::string& frozenManifestSHA,
    const std::string& generatedSHA,
    const std::vector<std::uint8_t>& buildBytes,
    const std::vector<std::uint8_t>& casesManifestBytes,
    const std::vector<std::uint8_t>& caseBytes,
    const std::string& metallibSHA,
    bool gpuResult,
    const std::vector<Float4>& cases,
    const std::vector<Float4>& baseline,
    const std::vector<Float4>& instrumented,
    const std::vector<Float4>& trace) {
    require(cases.size() == std::size_t(kCaseCount) * kFloat4PerCase &&
            baseline.size() == kCaseCount &&
            instrumented.size() == std::size_t(kCaseCount) * kDiagnosticRows &&
            trace.size() == std::size_t(kCaseCount) * kTraceIterations * kTraceRowsPerIteration,
        "report serializer received an incomplete array");
    std::ostringstream json;
    json << std::setprecision(9)
        << "{\"schema\":\"numi.matter.private_ee_ccd_metal_diagnostic_result.v2\",";
    json << "\"contact_source_sha256\":\"" << contactSHA << "\",";
    json << "\"frozen_source_manifest_sha256\":\"" << frozenManifestSHA << "\",";
    json << "\"generated_metal_sha256\":\"" << generatedSHA << "\",";
    json << "\"build_manifest_sha256\":\"" << sha256(buildBytes) << "\",";
    json << "\"case_manifest_sha256\":\"" << sha256(casesManifestBytes) << "\",";
    json << "\"case_payload_sha256\":\"" << sha256(caseBytes) << "\",";
    json << "\"metallib_sha256\":\"" << metallibSHA << "\",";
    json << "\"gpu_result\":" << (gpuResult ? "true" : "false")
        << ",\"solver_acceptance_evaluated\":false,\"physics_acceptance_evaluated\":false,";
    json << "\"qualification\":\"diagnostic only; CCD classification does not evaluate Armijo or transaction acceptance\",";
    json << "\"cases\":[";
    for (std::uint32_t i = 0; i < kCaseCount; ++i) {
        if (i) json << ',';
        const auto& h = cases[i * kFloat4PerCase];
        const auto& d0 = instrumented[i * kDiagnosticRows];
        const auto& n0 = instrumented[i * kDiagnosticRows + 1u];
        const auto& p0 = instrumented[i * kDiagnosticRows + 2u];
        const auto& wp = instrumented[i * kDiagnosticRows + 3u];
        const auto& wq = instrumented[i * kDiagnosticRows + 4u];
        const bool baseParity = std::memcmp(&baseline[i], &d0, sizeof(Float4)) == 0;
        const std::uint32_t iterations = std::isfinite(wp[3]) && wp[3] > 0
            ? std::min<std::uint32_t>(kTraceIterations,
                static_cast<std::uint32_t>(wp[3])) : 0u;
        json << "{\"index\":" << i
            << ",\"capture_case\":" << static_cast<std::uint32_t>(h[3])
            << ",\"budget\":" << static_cast<std::uint32_t>(h[2])
            << ",\"production_budget16\":{\"hit\":" << (baseline[i][0] > 0.5f ? "true" : "false")
            << ",\"certified\":" << (baseline[i][1] > 0.5f ? "true" : "false")
            << ",\"classification\":\"" << classification(baseline[i]) << "\""
            << ",\"toi\":"; jsonFloat(json, baseline[i][2]);
        json << ",\"separation_m\":"; jsonFloat(json, baseline[i][3]);
        json << "},\"instrumented\":{\"hit\":" << (d0[0] > 0.5f ? "true" : "false")
            << ",\"certified\":" << (d0[1] > 0.5f ? "true" : "false")
            << ",\"classification\":\"" << classification(d0) << "\""
            << ",\"toi\":"; jsonFloat(json, d0[2]);
        json << ",\"separation_m\":"; jsonFloat(json, d0[3]);
        json << ",\"normal\":"; jsonFloat4(json, n0);
        json << ",\"midpoint_and_t\":"; jsonFloat4(json, p0);
        json << ",\"closest_p_and_s\":"; jsonFloat4(json, wp);
        json << ",\"closest_q_and_budget\":"; jsonFloat4(json, wq);
        json << ",\"budget16_bitwise_matches_production\":"
            << ((h[2] == 16.0f && baseParity) ? "true" :
                (h[2] == 16.0f ? "false" : "null"));
        json << ",\"iterations\":[";
        for (std::uint32_t iter = 0; iter < iterations; ++iter) {
            if (iter) json << ',';
            const std::size_t traceBase =
                std::size_t(i) * kTraceIterations * kTraceRowsPerIteration +
                std::size_t(iter) * kTraceRowsPerIteration;
            json << "{\"iteration\":" << iter << ",\"time_distance_effective_thickness_speed\":";
            jsonFloat4(json, trace[traceBase]);
            json << ",\"closest_p_and_s\":"; jsonFloat4(json, trace[traceBase + 1u]);
            json << ",\"closest_q_and_t\":"; jsonFloat4(json, trace[traceBase + 2u]);
            json << '}';
        }
        // Close iterations, instrumented result, and this case object.
        json << "]}}";
    }
    json << "]}\n";
    const std::string text = json.str();
    validateReportJSON(text);
    return text;
}

const char* classification(const Float4& result) {
    if (result[1] <= 0.5f) return "uncertified_fail_closed";
    return result[0] > 0.5f ? "certified_hit" : "certified_miss";
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         std::size_t length) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
        length:length options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal buffer allocation failed");
    return buffer;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil, "missing kernel " + std::string([name UTF8String]));
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, error ? std::string([[error localizedDescription] UTF8String])
                                : "pipeline creation failed");
    return result;
}

std::vector<Float4> dispatch(id<MTLDevice> device, id<MTLCommandQueue> queue,
                             id<MTLComputePipelineState> pso,
                             id<MTLBuffer> input, std::uint32_t count,
                             std::uint32_t rowsPerCase,
                             id<MTLBuffer> trace = nil) {
    std::vector<Float4> output(std::size_t(count) * rowsPerCase, Float4{});
    id<MTLBuffer> outputBuffer = makeBuffer(device, output.data(),
        output.size() * sizeof(Float4));
    id<MTLBuffer> countBuffer = makeBuffer(device, &count, sizeof(count));
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBuffer:countBuffer offset:0 atIndex:2];
    if (trace) [encoder setBuffer:trace offset:0 atIndex:3];
    const NSUInteger width = std::min<NSUInteger>(
        pso.threadExecutionWidth, static_cast<NSUInteger>(count));
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error ? std::string([[command.error localizedDescription] UTF8String])
                      : "Metal dispatch did not complete");
    std::memcpy(output.data(), outputBuffer.contents,
        output.size() * sizeof(Float4));
    return output;
}

struct Args {
    std::string metallib, cases, casesManifest, buildManifest, output;
    bool validateInputsOnly = false;
    bool validateReportOnly = false;
};

Args parseArgs(int argc, const char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--validate-inputs-only") {
            a.validateInputsOnly = true;
            continue;
        }
        if (key == "--validate-report-only") {
            a.validateReportOnly = true;
            continue;
        }
        require(i + 1 < argc, "missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--metallib") a.metallib = value;
        else if (key == "--cases") a.cases = value;
        else if (key == "--cases-manifest") a.casesManifest = value;
        else if (key == "--build-manifest") a.buildManifest = value;
        else if (key == "--output") a.output = value;
        else throw std::runtime_error("unknown argument " + key);
    }
    require(!a.metallib.empty() && !a.cases.empty() && !a.casesManifest.empty() &&
            !a.buildManifest.empty(),
        "usage: run_native --metallib PATH --cases PATH --cases-manifest PATH "
        "--build-manifest PATH [--output PATH]");
    return a;
}

} // namespace

int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            const Args args = parseArgs(argc, argv);
            const auto caseBytes = readBytes(args.cases);
            require(caseBytes.size() == kCaseCount * kFloat4PerCase * sizeof(Float4),
                "captured case payload has wrong byte length");
            require(caseBytes.size() % alignof(Float4) == 0,
                "case payload alignment is invalid");
            std::vector<Float4> cases(caseBytes.size() / sizeof(Float4));
            std::memcpy(cases.data(), caseBytes.data(), caseBytes.size());
            const auto casesManifestBytes = readBytes(args.casesManifest);
            const std::string casesManifestText(casesManifestBytes.begin(), casesManifestBytes.end());
            const std::string expectedCasesSHA = jsonRootStringField(casesManifestText, "payload_sha256");
            require(jsonRootStringField(casesManifestText, "schema") == kExpectedCasesSchema &&
                    jsonRootStringField(casesManifestText, "contact_source_sha256") == kExpectedContactSHA &&
                    jsonRootStringField(casesManifestText, "frozen_source_manifest_sha256") == kExpectedFrozenManifestSHA,
                "case manifest has unexpected schema or frozen source binding");
            require(expectedCasesSHA == kExpectedCasesSHA,
                "case manifest does not describe the retained 333/334 input corpus");
            require(sha256(caseBytes) == expectedCasesSHA,
                "packed cases do not match their source-bound manifest");
            const auto buildBytes = readBytes(args.buildManifest);
            const std::string buildText(buildBytes.begin(), buildBytes.end());
            const std::string contactSHA = jsonStringField(buildText, "contact_source_sha256");
            const std::string frozenManifestSHA =
                jsonStringField(buildText, "frozen_source_manifest_sha256");
            const std::string generatedSHA = jsonStringField(buildText, "generated_metal_sha256");
            require(jsonStringField(buildText, "schema") == kExpectedBuildSchema &&
                    jsonStringField(buildText, "extracted_production_function_sha256") == kExpectedExtractedFunctionSHA &&
                    jsonStringField(buildText, "instrumented_function_sha256") == kExpectedInstrumentedFunctionSHA &&
                    jsonStringField(buildText, "instrumented_16_function_sha256") == kExpectedInstrumented16SHA &&
                    jsonStringField(buildText, "instrumented_17_function_sha256") == kExpectedInstrumented17SHA &&
                    generatedSHA == kExpectedGeneratedMetalSHA,
                "diagnostic build manifest differs from the reviewed source extraction");
            require(contactSHA == kExpectedContactSHA,
                "diagnostic build manifest is not bound to frozen source 099");
            require(frozenManifestSHA == kExpectedFrozenManifestSHA &&
                    jsonStringField(buildText, "frozen_source_parent_revision") ==
                        "35bfb518c542a306f563cc364ef8496837068b9a",
                "diagnostic build manifest has unexpected full-source identity");
            (void)generatedSHA;
            for (std::uint32_t i = 0; i < kCaseCount; ++i) {
                const Float4& h = cases[i * kFloat4PerCase];
                require(h[0] == 1.0f && h[1] > 0.0f &&
                        (h[2] == 16.0f || h[2] == 17.0f) &&
                        (h[3] == 333.0f || h[3] == 334.0f),
                    "unexpected diagnostic case header");
            }

            // This mode exercises the same source and manifest admission path
            // without creating a Metal device or dispatching a command buffer.
            if (args.validateInputsOnly) {
                std::cout << "PASS source-bound case/build input validation; no Metal device or dispatch\n";
                return 0;
            }

            if (args.validateReportOnly) {
                std::vector<Float4> syntheticBaseline(kCaseCount);
                std::vector<Float4> syntheticInstrumented(
                    std::size_t(kCaseCount) * kDiagnosticRows);
                std::vector<Float4> syntheticTrace(
                    std::size_t(kCaseCount) * kTraceIterations * kTraceRowsPerIteration);
                for (std::uint32_t i = 0; i < kCaseCount; ++i) {
                    syntheticBaseline[i] = Float4{0.0f, 1.0f, 0.25f, 1.0e-5f};
                    auto* rows = syntheticInstrumented.data() +
                        std::size_t(i) * kDiagnosticRows;
                    rows[0] = syntheticBaseline[i];
                    rows[1] = Float4{0.0f, 0.0f, 1.0f, 0.0f};
                    rows[2] = Float4{0.0f, 0.0f, 0.0f, 0.5f};
                    rows[3] = Float4{0.001f, 0.5f, 1.0e-5f, 2.0f};
                    rows[4] = Float4{0.002f, 0.5f, 1.0e-5f, 17.0f};
                    for (std::uint32_t iteration = 0; iteration < 2u; ++iteration) {
                        const std::size_t base =
                            std::size_t(i) * kTraceIterations * kTraceRowsPerIteration +
                            std::size_t(iteration) * kTraceRowsPerIteration;
                        syntheticTrace[base] = Float4{float(iteration), 1.0e-5f, 1.0e-5f, 1.0e-4f};
                        syntheticTrace[base + 1u] = Float4{0.1f, 0.2f, 0.3f, 0.4f};
                        syntheticTrace[base + 2u] = Float4{0.5f, 0.6f, 0.7f, 1.0f};
                    }
                }
                const std::string generatedSHA = jsonStringField(buildText, "generated_metal_sha256");
                const std::string text = serializeReport(
                    contactSHA, frozenManifestSHA, generatedSHA,
                    buildBytes, casesManifestBytes, caseBytes,
                    std::string(64u, '0'), false, cases, syntheticBaseline,
                    syntheticInstrumented, syntheticTrace);
                if (!args.output.empty()) {
                    std::ofstream out(args.output, std::ios::binary | std::ios::trunc);
                    require(bool(out), "cannot open report-only output JSON");
                    out.write(text.data(), static_cast<std::streamsize>(text.size()));
                    require(bool(out), "failed writing report-only JSON");
                }
                std::cout << "PASS full four-case JSON serialization and NSJSONSerialization validation; no Metal device or dispatch\n";
                return 0;
            }

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:args.metallib.c_str()]]
                error:&error];
            require(library != nil, error
                ? std::string([[error localizedDescription] UTF8String])
                : "could not load diagnostic metallib");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            id<MTLBuffer> input = makeBuffer(device, cases.data(),
                cases.size() * sizeof(Float4));
            std::vector<Float4> zeroTrace(std::size_t(kCaseCount) * kTraceIterations *
                kTraceRowsPerIteration, Float4{});
            id<MTLBuffer> trace = makeBuffer(device, zeroTrace.data(),
                zeroTrace.size() * sizeof(Float4));

            const auto baseline = dispatch(device, queue,
                pipeline(device, library, @"numi_matter_metal::nm_contact_ccd_regression"),
                input, kCaseCount, 1u);
            const auto instrumented = dispatch(device, queue,
                pipeline(device, library,
                    @"numi_matter_metal::nm_contact_ee_ccd_budget_diagnostic"),
                input, kCaseCount, kDiagnosticRows, trace);
            std::memcpy(zeroTrace.data(), trace.contents,
                zeroTrace.size() * sizeof(Float4));

            const std::string text = serializeReport(
                contactSHA, frozenManifestSHA, generatedSHA,
                buildBytes, casesManifestBytes, caseBytes,
                sha256(readBytes(args.metallib)), true, cases,
                baseline, instrumented, zeroTrace);
            if (!args.output.empty()) {
                std::ofstream out(args.output, std::ios::binary | std::ios::trunc);
                require(bool(out), "cannot open output JSON");
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                require(bool(out), "failed writing output JSON");
            } else {
                std::cout << text;
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL native captured EE diagnostic: " << error.what() << '\n';
            return 1;
        }
    }
}
