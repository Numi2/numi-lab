#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/NumiHumanInitialState.hpp"
#include "metalrobo/NumiHumanRuntimeIdentity.hpp"
#include "metalrobo/NumiHumanSupport.hpp"
#include "numi/matter/matter.hpp"
#include "metalrobo/NeuronCultureArtifacts.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/mrnx_bridge_v1.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "metalrobo/numanx_human_io_gpu.h"
#include "numi/matter/shared.h"
#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <sys/stdio.h>
#include <unistd.h>

#ifndef MRNX_FULLBODY_RIGID
#error MRNX_FULLBODY_RIGID is required
#endif
#ifndef MRNX_FULLBODY_MUSCLE
#error MRNX_FULLBODY_MUSCLE is required
#endif
#ifndef MRNX_FULLBODY_SUPPORT_CONTACT
#error MRNX_FULLBODY_SUPPORT_CONTACT is required
#endif
#ifndef MRNX_METALROBO_METALLIB
#error MRNX_METALROBO_METALLIB is required
#endif
#ifndef MRNX_MATTER_METALLIB
#error MRNX_MATTER_METALLIB is required
#endif
#ifndef MRNX_MATTER_MATERIAL
#error MRNX_MATTER_MATERIAL is required
#endif

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
std::uint64_t equalityFingerprint(const std::vector<std::uint8_t>& bytes) {
    return metalrobo::numiHumanRuntimePayloadFingerprint(
        std::as_bytes(std::span(bytes)));
}

std::uint64_t constrainedFingerprint(std::uint64_t base, const std::vector<std::uint8_t>& bytes) {
    return metalrobo::numiHumanRuntimeAppendPayloadOwner(
        base, "NHEQ2", equalityFingerprint(bytes));
}

constexpr std::uint64_t kStartMicros = 1'000u;
constexpr std::uint64_t kDurationMicros = 2'000u;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string supportSHA256Hex(
    const metalrobo::NumiHumanSupportPayloadIdentity& identity
) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(identity.sha256.size() * 2u);
    for (const std::uint8_t byte : identity.sha256) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 0x0fu]);
    }
    return result;
}

std::array<std::uint8_t, 32u> parseSupportSHA256Hex(
    const std::string_view text
) {
    require(text.size() == 64u, "support SHA-256 must contain 64 hex digits");
    const auto nibble = [](const char value) -> std::optional<std::uint8_t> {
        if (value >= '0' && value <= '9')
            return static_cast<std::uint8_t>(value - '0');
        if (value >= 'a' && value <= 'f')
            return static_cast<std::uint8_t>(value - 'a' + 10);
        if (value >= 'A' && value <= 'F')
            return static_cast<std::uint8_t>(value - 'A' + 10);
        return std::nullopt;
    };
    std::array<std::uint8_t, 32u> result{};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        const auto high = nibble(text[index * 2u]);
        const auto low = nibble(text[index * 2u + 1u]);
        require(high.has_value() && low.has_value(),
            "support SHA-256 contains a non-hex character");
        result[index] = static_cast<std::uint8_t>((*high << 4u) | *low);
    }
    return result;
}

std::uint64_t parseUnsignedIdentityField(
    const std::string_view text, const std::uint64_t maximum
) {
    require(!text.empty(), "support identity integer is empty");
    std::uint64_t result = 0u;
    for (const char value : text) {
        require(value >= '0' && value <= '9',
            "support identity integer contains a non-digit");
        const std::uint64_t digit = static_cast<std::uint64_t>(value - '0');
        require(result <= (maximum - digit) / 10u,
            "support identity integer exceeds its ABI domain");
        result = result * 10u + digit;
    }
    return result;
}

double parseExactFiniteDecimal(const std::string_view text) {
    require(!text.empty(), "support force is empty");
    const std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(owned.c_str(), &end);
    require(end == owned.c_str() + owned.size() && errno != ERANGE &&
        std::isfinite(value),
        "support force is not one complete finite decimal");
    return value;
}

void mixU32(std::uint64_t& hash, const std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixFloat(std::uint64_t& hash, const float value) noexcept {
    mixU32(hash, std::bit_cast<std::uint32_t>(value));
}

std::vector<std::uint8_t> readPayloadBytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "provenance payload did not open");
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(!bytes.empty() && !input.bad(),
            "provenance payload did not read exactly");
    return bytes;
}

class ScopedDescriptor final {
public:
    explicit ScopedDescriptor(const int descriptor) noexcept
        : descriptor_(descriptor) {}
    ~ScopedDescriptor() {
        if (descriptor_ >= 0) (void)::close(descriptor_);
    }
    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;
    [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
    int descriptor_ = -1;
};

int openNoFollow(
    const std::filesystem::path& path,
    const int flags,
    const char* label
) {
    int descriptor = -1;
    do {
        descriptor = ::open(path.c_str(), flags | O_NOFOLLOW);
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) {
        throw std::runtime_error(
            std::string("could not open ") + label + ": " +
            std::strerror(errno));
    }
    return descriptor;
}

void fullSyncDescriptor(const int descriptor, const char* label) {
    int result = 0;
    do {
        result = ::fcntl(descriptor, F_FULLFSYNC);
    } while (result != 0 && errno == EINTR);
    const int savedErrno = errno;
    if (result != 0) {
        throw std::runtime_error(
            std::string("could not durably sync ") + label + ": " +
            std::strerror(savedErrno));
    }
}

void syncPublishedPath(
    const std::filesystem::path& path,
    const char* label
) {
    const ScopedDescriptor descriptor(
        openNoFollow(path, O_RDONLY, label));
    fullSyncDescriptor(descriptor.get(), label);
}

void requirePinnedDirectoryEntry(
    const int parentDescriptor,
    const std::filesystem::path& leaf,
    const int directoryDescriptor
) {
    struct stat pinned{};
    struct stat entry{};
    require(
        leaf.has_filename() && leaf == leaf.filename() &&
            ::fstat(directoryDescriptor, &pinned) == 0 &&
            ::fstatat(
                parentDescriptor, leaf.c_str(), &entry,
                AT_SYMLINK_NOFOLLOW) == 0 &&
            S_ISDIR(pinned.st_mode) && S_ISDIR(entry.st_mode) &&
            pinned.st_dev == entry.st_dev && pinned.st_ino == entry.st_ino,
        "prepared fixture staging entry no longer matches its pinned directory");
}

void qualifyHumanSupportNewtonAdmission(id<MTLDevice> device) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
        [NSString stringWithUTF8String:MRNX_MATTER_METALLIB]] error:&error];
    require(library != nil, "support Newton admission library failed");
    id<MTLFunction> function = [library newFunctionWithName:@"numi_matter_metal::nm_fgmres_begin"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "support Newton admission pipeline failed");
    id<MTLFunction> layoutFunction = [library newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
    id<MTLArgumentEncoder> arguments = [layoutFunction newArgumentEncoderWithBufferIndex:0u];
    id<MTLBuffer> argumentBuffer = [device newBufferWithLength:arguments.encodedLength options:MTLResourceStorageModeShared];
    [arguments setArgumentBuffer:argumentBuffer offset:0u];
    id<MTLFunction> certifyFunction = [library newFunctionWithName:@"numi_matter_metal::nm_human_support_certify"];
    id<MTLComputePipelineState> certifyPipeline = [device newComputePipelineStateWithFunction:certifyFunction error:&error];
    require(certifyPipeline != nil, "support Newton admission certificate pipeline failed");
    std::array<id<MTLBuffer>,22u> buffers;
    for (NSUInteger i=0; i<buffers.size(); ++i) {
        buffers[i] = [device newBufferWithLength:4096u options:MTLResourceStorageModeShared];
        require(buffers[i] != nil, "support Newton admission allocation failed");
        std::memset(buffers[i].contents, 0, 4096u);
    }
    for (NSUInteger i=0;i<15u;++i) [arguments setBuffer:buffers[2] offset:0u atIndex:i];
    NMMatterDispatchGPU dispatch{}; dispatch.environmentCount=1u;
    NMFGMRESLayoutGPU layout{}; layout.supportContactCount=2u;
    NMMixedSolverGPU solver{}; solver.residualTolerances.x=0.001f;
    NMHumanSupportDispatchGPU support{};
    support.groundNormal = {0.0f,1.0f,0.0f,0.0f};
    const std::uint32_t restart=0u, iteration=2u;
    const float forcing=0.25f;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    struct Case {
        const char* name;
        nm_float4 history;
        float friction;
        bool admissible;
        std::uint32_t rows=2u, environments=1u;
    };
    const float nan=std::numeric_limits<float>::quiet_NaN();
    const float infinity=std::numeric_limits<float>::infinity();
    const Case cases[] = {
        {"negative normal",{0,0,0,-0.0001f},0.5f,false},
        {"historical negative boundary",{0,0,0,-0.0000001f},0.5f,false},
        {"small negative normal",{0,0,0,-0.00000005f},0.5f,false},
        {"positive normal",{0,0,0,0.25f},0.5f,true},
        {"zero history",{0,0,0,0},0.5f,true},
        {"normal NaN",{0,0,0,nan},0.5f,false},
        {"normal positive infinity",{0,0,0,infinity},0.5f,false},
        {"normal negative infinity",{0,0,0,-infinity},0.5f,false},
        {"tangent NaN",{nan,0,0,0.25f},0.5f,false},
        {"tangent infinity",{0,0,infinity,0.25f},0.5f,false},
        {"cone boundary",{0.125f,0,0,0.25f},0.5f,true},
        {"outside cone",{0.126f,0,0,0.25f},0.5f,false},
        {"nonorthogonal tangent",{0,0.01f,0,0.25f},0.5f,false},
        {"zero normal with tangent",{0.00001f,0,0,0},0.5f,false},
        {"frictionless normal",{0,0,0,0.25f},0.0f,true},
        {"frictionless tangent",{0.00001f,0,0,0.25f},0.0f,false},
        {"negative friction",{0,0,0,0.25f},-0.5f,false},
        {"nonfinite friction",{0,0,0,0.25f},nan,false},
        {"second environment second SIMD pass negative",{0,0,0,-0.0001f},0.5f,false,40u,2u},
        {"second environment second SIMD pass cone",{0.126f,0,0,0.25f},0.5f,false,40u,2u},
        {"second environment second SIMD pass valid",{0.125f,0,0,0.25f},0.5f,true,40u,2u},
        {"no support rows",{0,0,0,0},0.5f,true,0u,2u},
    };
    for (const auto test : cases) {
        dispatch.environmentCount=test.environments;
        layout.supportContactCount=test.rows;
        support.contactCount=test.rows;
        auto* state=static_cast<NMFGMRESStateGPU*>(buffers[8].contents);
        auto* residual=static_cast<nm_float4*>(buffers[3].contents);
        auto* history=static_cast<nm_float4*>(buffers[19].contents);
        auto* contacts=static_cast<NMHumanSupportContactGPU*>(buffers[21].contents);
        std::memset(buffers[3].contents,0,4096u);
        std::memset(buffers[9].contents,0,4096u);
        for (std::uint32_t row=0;row<test.rows;++row) {
            contacts[row]={};
            contacts[row].frictionSlopAndStabilization.x=0.5f;
        }
        for (std::uint32_t environment=0;environment<test.environments;++environment) {
            state[environment]={}; state[environment].nonlinear={1.0f,1.0f,0.0f,0.0f};
            for (std::uint32_t row=0;row<test.rows;++row)
                history[environment*test.rows+row]={0,0,0,0.25f};
            if (test.rows) residual[environment*test.rows]={0.0001f,0,0,0};
        }
        if (test.rows) {
            history[test.environments*test.rows-1u]=test.history;
            contacts[test.rows-1u].frictionSlopAndStabilization.x=test.friction;
            // Contact parameters are shared across environments. Keep the
            // preceding environment admissible even for invalid friction.
            if (!std::isfinite(test.friction) || test.friction<0.0f)
                require(test.environments==1u,"invalid-friction fixture must use one environment");
        }
        const std::vector<nm_float4> originalHistory(
            history,history+test.environments*test.rows);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0u atIndex:i];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&solver length:sizeof(solver) atIndex:1u];
        [encoder setBuffer:argumentBuffer offset:0u atIndex:13u];
        [encoder useResource:buffers[2] usage:MTLResourceUsageRead];
        [encoder setBytes:&restart length:sizeof(restart) atIndex:14u];
        [encoder setBytes:&forcing length:sizeof(forcing) atIndex:15u];
        [encoder setBytes:&iteration length:sizeof(iteration) atIndex:16u];
        [encoder setBytes:&support length:sizeof(support) atIndex:20u];
        [encoder setBytes:&layout length:sizeof(layout) atIndex:30u];
        [encoder dispatchThreadgroups:MTLSizeMake(test.environments,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted, "support Newton admission kernel failed");
        for (std::uint32_t environment=0;environment<test.environments;++environment) {
            const bool admissible=test.rows==0u || environment+1u<test.environments ||
                test.admissible;
            require((state[environment].nonlinear.w>0.5f)==admissible &&
                    (state[environment].diagnostics.z>0.5f)==admissible,
                test.name);
        }
        // Run the actual publication certificate over the same bytes, so a
        // future change to its contact law cannot silently drift from stopping.
        command=[queue commandBuffer];
        encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:certifyPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&solver length:sizeof(solver) atIndex:1u];
        [encoder setBuffer:buffers[3] offset:0u atIndex:2u];
        [encoder setBuffer:buffers[8] offset:0u atIndex:3u];
        [encoder setBuffer:buffers[9] offset:0u atIndex:4u];
        [encoder setBuffer:buffers[19] offset:0u atIndex:5u];
        [encoder setBytes:&support length:sizeof(support) atIndex:6u];
        [encoder setBuffer:buffers[21] offset:0u atIndex:7u];
        [encoder setBytes:&layout length:sizeof(layout) atIndex:30u];
        [encoder dispatchThreadgroups:MTLSizeMake(test.environments,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"support Newton admission certificate failed");
        const auto* statuses=static_cast<const NMMatterStatusGPU*>(buffers[9].contents);
        for (std::uint32_t environment=0;environment<test.environments;++environment)
            require((statuses[environment].code==NM_STATUS_SUCCESS)==
                (state[environment].nonlinear.w>0.5f),
                "Newton stopping disagrees with actual publication certificate");
        if (test.rows) require(std::memcmp(history,originalHistory.data(),
            originalHistory.size()*sizeof(nm_float4))==0,"Newton admission mutated physical impulses");
    }
    std::puts("numanx_support_newton_admission=pass cases=22 negative=continues cone=certified impulse=unchanged");
}


void qualifyHumanSupportKKT(id<MTLDevice> device, unsigned shape = 0) {
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:MRNX_MATTER_METALLIB]]
                       error:&libraryError];
    require(library != nil, "Matter support qualification library failed");
    const auto pipeline = [&](NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:
            [@"numi_matter_metal::" stringByAppendingString:name]];
        NSError* error = nil;
        id<MTLComputePipelineState> result = function == nil ? nil :
            [device newComputePipelineStateWithFunction:function error:&error];
        require(result != nil, "Matter support qualification pipeline failed");
        return result;
    };
    id<MTLComputePipelineState> checkpointPipeline =
        pipeline(@"nm_human_support_checkpoint");
    id<MTLComputePipelineState> evaluatePipeline =
        pipeline(@"nm_human_support_evaluate");
    id<MTLComputePipelineState> residualPipeline =
        pipeline(@"nm_human_support_accumulate_rigid_residual");
    id<MTLComputePipelineState> operatorPipeline =
        pipeline(@"nm_fgmres_apply_human_support");
    auto dualPipeline = pipeline(@"nm_fgmres_apply_support_dual");
    id<MTLComputePipelineState> commitPipeline =
        pipeline(@"nm_human_support_commit");
    id<MTLComputePipelineState> rollbackPipeline =
        pipeline(@"nm_human_support_rollback");

    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    dispatch.objectCount = 1u;
    dispatch.rigidGeneralizedCapacity = 1u;
    NMFGMRESLayoutGPU layout{};
    layout.supportContactCount = 1u;
    layout.supportBase = 1u;
    layout.unknownCount = 2u;
    NMHumanSupportDispatchGPU support{};
    support.contactCount = 1u;
    support.articulatedNv = 1u;
    support.bodyCount = 1u;
    support.bodyStride = 1u;
    support.groundPointAndTimestep.w = 0.01f;
    support.groundNormal.y = 1.0f;
    NMHumanSupportContactGPU contact{};
    contact.identity = {0u, 7u, 0u, 0u};
    contact.localPoint.y = -0.01f;
    contact.frictionSlopAndStabilization = {0.5f, 0.02f, 0.2f, 0.0f};
    MRBodyStateGPU body{};
    body.orientation.w = 1.0f;
    body.linearVelocityAndInverseMass.y = -1.0f;
    if (shape) {
        contact.identity.w = 1u;
        contact.localPoint = {0.0f,0.0f,0.0f,0.01f};
        body.orientation = {0.0f,0.0f,std::sqrt(0.5f),std::sqrt(0.5f)};
        body.angularVelocity.z = 2.0f;
        if (shape==2) {
            contact.identity.w=2u;contact.localPoint.w=0;
            contact.supportRadii={0.02f,0.01f,0.03f,0};
            contact.supportOrientation={0,0,std::sqrt(0.5f),std::sqrt(0.5f)};
        }
    }
    // The delta alone is deliberately different from the total velocity.
    // Support must include the free predictor carried by candidateBodies.
    const float generalized = -0.25f;
    // The scalar test coordinate has the explicitly supplied point tangent.
    // Shape geometry and source-compiled curved Jacobians are also covered by
    // the full-body primitive oracle; this test isolates NCP linearization.
    const std::array<float, 3u> jacobian{shape ? -0.02f : 0.0f, 1.0f, 0.0f};
    const float freeVelocity = -0.75f;
    const nm_float4 zero4{};
    const NMMatterStatusGPU success{};
    NMMatterStatusGPU failure{};
    failure.code = NM_STATUS_CONTACT_FAILURE;
    NMFGMRESStateGPU fgmres{};
    nm_float4 direction{};
    direction.x = 2.0f;
    const std::array<nm_float4,2> directionRows{direction,{0.4f,0.3f,0.2f,0}};
    const std::array<nm_float4,2> zeroRows{};
    const NMHumanSupportKKTGPU zeroKKT{};

    const auto buffer = [&](const void* bytes, const NSUInteger length) {
        id<MTLBuffer> result = [device newBufferWithBytes:bytes length:length
            options:MTLResourceStorageModeShared];
        require(result != nil, "Matter support qualification buffer failed");
        return result;
    };
    id<MTLBuffer> contacts = buffer(&contact, sizeof(contact));
    id<MTLBuffer> bodies = buffer(&body, sizeof(body));
    // This direct NCP fixture intentionally exercises legacy coordinates.
    // Metal still requires valid bindings for the disabled paired inputs.
    id<MTLBuffer> bodyPositionLow = buffer(&zero4, sizeof(zero4));
    const std::uint32_t compensatedTranslation = 0u;
    id<MTLBuffer> generalizedBuffer = buffer(&generalized, sizeof(generalized));
    id<MTLBuffer> jacobians = buffer(jacobian.data(), sizeof(jacobian));
    // Keep the direct apply fixture inside the authored Coulomb cone. The
    // runtime now rejects an already-infeasible accepted history instead of
    // carrying it forward as this legacy probe once did for the ellipsoid.
    const nm_float4 initialHistory{
        shape==2 ? 0.18f : 0.0f, 0.0f, 0.0f, 0.5f};
    id<MTLBuffer> acceptedHistory = buffer(&initialHistory, sizeof(initialHistory));
    id<MTLBuffer> candidateHistory = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> checkpointHistory = buffer(&zero4, sizeof(zero4));
    const NMHumanSupportConsequenceGPU zeroConsequence{};
    id<MTLBuffer> acceptedConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    id<MTLBuffer> candidateConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    id<MTLBuffer> checkpointConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    const NMContactSampleGPU zeroSample{};
    id<MTLBuffer> samples = buffer(&zeroSample, sizeof(zeroSample));
    id<MTLBuffer> successStatus = buffer(&success, sizeof(success));
    id<MTLBuffer> failureStatus = buffer(&failure, sizeof(failure));
    id<MTLBuffer> residual = buffer(zeroRows.data(), sizeof(zeroRows));
    id<MTLBuffer> kkt = buffer(&zeroKKT,sizeof(zeroKKT));
    id<MTLBuffer> freeBuffer = buffer(&freeVelocity,sizeof(freeVelocity));
    id<MTLBuffer> directionBuffer = buffer(directionRows.data(), sizeof(directionRows));
    id<MTLBuffer> work = buffer(zeroRows.data(), sizeof(zeroRows));
    id<MTLBuffer> fgmresState = buffer(&fgmres, sizeof(fgmres));
    const std::uint32_t zeroWorkingSet = 0u;
    id<MTLBuffer> supportWorkingSet =
        buffer(&zeroWorkingSet, sizeof(zeroWorkingSet));
    id<MTLBuffer> supportChanged =
        buffer(&zeroWorkingSet, sizeof(zeroWorkingSet));
    id<MTLBuffer> supportConeWorkingSet =
        buffer(&zeroWorkingSet, sizeof(zeroWorkingSet));
    id<MTLBuffer> supportConeTarget = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> committedHistory = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> committedConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(queue != nil && command != nil,
            "Matter support qualification command failed");
    const auto encodeOne = [&](id<MTLComputePipelineState> state,
                               const auto& bind) {
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        require(encoder != nil, "Matter support qualification encoder failed");
        [encoder setComputePipelineState:state];
        bind(encoder);
        if (state == evaluatePipeline) {
            [encoder setBuffer:bodyPositionLow offset:0u atIndex:15u];
            [encoder setBuffer:bodyPositionLow offset:0u atIndex:16u];
            [encoder setBytes:&compensatedTranslation
                length:sizeof(compensatedTranslation) atIndex:17u];
        }
        [encoder setBytes:&layout length:sizeof(layout) atIndex:30u];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
    };
    const std::uint32_t count = 1u;
    encodeOne(checkpointPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:2u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:3u];
        [encoder setBuffer:checkpointHistory offset:0u atIndex:4u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:5u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:6u];
        [encoder setBuffer:checkpointConsequence offset:0u atIndex:7u];
    });
    encodeOne(evaluatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:contacts offset:0u atIndex:2u];
        [encoder setBuffer:bodies offset:0u atIndex:3u];
        [encoder setBuffer:generalizedBuffer offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:7u];
        [encoder setBuffer:samples offset:0u atIndex:8u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:9u];
        [encoder setBuffer:successStatus offset:0u atIndex:10u];
        [encoder setBuffer:bodies offset:0u atIndex:11u];
        [encoder setBuffer:kkt offset:0u atIndex:12u];
        [encoder setBuffer:residual offset:0u atIndex:13u];
        [encoder setBuffer:freeBuffer offset:0u atIndex:14u];
    });
    encodeOne(residualPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:samples offset:0u atIndex:2u];
        [encoder setBuffer:jacobians offset:0u atIndex:3u];
        [encoder setBuffer:residual offset:0u atIndex:4u];
    });
    encodeOne(operatorPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:directionBuffer offset:0u atIndex:2u];
        [encoder setBuffer:work offset:0u atIndex:3u];
        [encoder setBuffer:samples offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:fgmresState offset:0u atIndex:6u];
    });
    encodeOne(dualPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:directionBuffer offset:0u atIndex:2u];
        [encoder setBuffer:work offset:0u atIndex:3u];
        [encoder setBuffer:kkt offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:fgmresState offset:0u atIndex:6u];
    });
    const nm_float4 alpha{0.25f,0,0,0};
    auto alphaBuffer = buffer(&alpha,sizeof(alpha));
    encodeOne(pipeline(@"nm_human_support_apply_solution"), [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:directionBuffer offset:0u atIndex:2u];
        [encoder setBuffer:alphaBuffer offset:0u atIndex:3u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:4u];
        [encoder setBuffer:supportWorkingSet offset:0u atIndex:5u];
        [encoder setBuffer:supportChanged offset:0u atIndex:6u];
        [encoder setBuffer:successStatus offset:0u atIndex:7u];
        [encoder setBuffer:contacts offset:0u atIndex:8u];
        [encoder setBuffer:supportConeWorkingSet offset:0u atIndex:9u];
        [encoder setBuffer:supportConeTarget offset:0u atIndex:10u];
    });
    encodeOne(evaluatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:contacts offset:0u atIndex:2u];
        [encoder setBuffer:bodies offset:0u atIndex:3u];
        [encoder setBuffer:generalizedBuffer offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:7u];
        [encoder setBuffer:samples offset:0u atIndex:8u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:9u];
        [encoder setBuffer:successStatus offset:0u atIndex:10u];
        [encoder setBuffer:bodies offset:0u atIndex:11u];
        [encoder setBuffer:kkt offset:0u atIndex:12u];
        [encoder setBuffer:residual offset:0u atIndex:13u];
        [encoder setBuffer:freeBuffer offset:0u atIndex:14u];
    });
    encodeOne(commitPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:successStatus offset:0u atIndex:2u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:3u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:4u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:5u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:6u];
    });
    id<MTLBlitCommandEncoder> snapshot = [command blitCommandEncoder];
    require(snapshot != nil, "Matter support qualification snapshot failed");
    [snapshot copyFromBuffer:acceptedHistory sourceOffset:0u
        toBuffer:committedHistory destinationOffset:0u size:sizeof(nm_float4)];
    [snapshot copyFromBuffer:acceptedConsequence sourceOffset:0u
        toBuffer:committedConsequence destinationOffset:0u
        size:sizeof(NMHumanSupportConsequenceGPU)];
    [snapshot endEncoding];
    encodeOne(rollbackPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:failureStatus offset:0u atIndex:2u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:3u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:4u];
        [encoder setBuffer:checkpointHistory offset:0u atIndex:5u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:6u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:7u];
        [encoder setBuffer:checkpointConsequence offset:0u atIndex:8u];
    });
    // Central differences of the assembled dual residual perturb both
    // velocity/position and independent impulse. Initial gap remains frozen.
    // No finite-difference helper uses the stored projection derivative.
    constexpr float epsilon = 1.0e-3f;
    const auto perturbSupport = [&](const float sign) {
        MRBodyStateGPU perturbed = body;
        perturbed.position.y += sign * epsilon * direction.x *
            support.groundPointAndTimestep.w;
        perturbed.linearVelocityAndInverseMass.y += sign * epsilon * direction.x;
        id<MTLBuffer> perturbedBodies = buffer(&perturbed, sizeof(perturbed));
        perturbed.position.x += sign * epsilon * direction.x *
            jacobian[0] * support.groundPointAndTimestep.w;
        // Refresh after completing the candidate geometry perturbation.
        perturbedBodies = buffer(&perturbed, sizeof(perturbed));
        const float perturbedGeneralized = generalized + sign * epsilon * direction.x;
        auto perturbedGeneralizedBuffer = buffer(&perturbedGeneralized,sizeof(perturbedGeneralized));
        const nm_float4 perturbedHistory{initialHistory.x+sign*epsilon*directionRows[1].x,0,
            sign*epsilon*directionRows[1].z,
            initialHistory.w+sign*epsilon*directionRows[1].y};
        id<MTLBuffer> historyOut = buffer(&perturbedHistory, sizeof(perturbedHistory));
        auto kktOut = buffer(&zeroKKT,sizeof(zeroKKT));
        auto residualOut = buffer(zeroRows.data(),sizeof(zeroRows));
        id<MTLBuffer> sampleOut = buffer(&zeroSample, sizeof(zeroSample));
        id<MTLBuffer> consequenceOut = buffer(&zeroConsequence, sizeof(zeroConsequence));
        encodeOne(evaluatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBytes:&support length:sizeof(support) atIndex:1u];
            [encoder setBuffer:contacts offset:0u atIndex:2u];
            [encoder setBuffer:perturbedBodies offset:0u atIndex:3u];
            [encoder setBuffer:perturbedGeneralizedBuffer offset:0u atIndex:4u];
            [encoder setBuffer:jacobians offset:0u atIndex:5u];
            [encoder setBuffer:historyOut offset:0u atIndex:7u];
            [encoder setBuffer:sampleOut offset:0u atIndex:8u];
            [encoder setBuffer:consequenceOut offset:0u atIndex:9u];
            [encoder setBuffer:successStatus offset:0u atIndex:10u];
            [encoder setBuffer:bodies offset:0u atIndex:11u];
            [encoder setBuffer:kktOut offset:0u atIndex:12u];
            [encoder setBuffer:residualOut offset:0u atIndex:13u];
            [encoder setBuffer:freeBuffer offset:0u atIndex:14u];
        });
        return kktOut;
    };
    id<MTLBuffer> plusSample = perturbSupport(1.0f);
    id<MTLBuffer> minusSample = perturbSupport(-1.0f);
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "Matter support qualification GPU command failed");

    const auto& committed =
        *static_cast<const NMHumanSupportConsequenceGPU*>(
            committedConsequence.contents);
    const auto& history =
        *static_cast<const nm_float4*>(committedHistory.contents);
    const auto& residualValue =
        *static_cast<const nm_float4*>(residual.contents);
    const auto& operatorValue =
        *static_cast<const nm_float4*>(work.contents);
    const auto& rolledHistory =
        *static_cast<const nm_float4*>(acceptedHistory.contents);
    const auto& rolledConsequence =
        *static_cast<const NMHumanSupportConsequenceGPU*>(
            acceptedConsequence.contents);
    require(committed.identity.x == 0u && committed.identity.y == 7u &&
                committed.identity.w ==
                    NM_HUMAN_SUPPORT_CONSEQUENCE_VERSION &&
                (committed.identity.z & NM_CONTACT_VALID) != 0u &&
                std::abs(committed.impulseAndNormal.w - 0.575f) < 1.0e-5f &&
                history.w == committed.impulseAndNormal.w &&
                std::abs(residualValue.x-(0.5f+jacobian[0]*initialHistory.x))<1.0e-6f &&
                std::abs(operatorValue.x + jacobian[0]*directionRows[1].x +
                    directionRows[1].y)<1.0e-6f,
            "Matter support J^T lambda or independent dual action is wrong");
    if (shape) require(std::abs(committed.pointAndSeparation.y+0.01f)<1.0e-7f &&
        std::abs(committed.tangentVelocityAndImpulse.x-0.02f)<1.0e-6f &&
        std::abs(committed.impulseAndNormal.x-(initialHistory.x+0.1f))<1.0e-6f,
        "Matter sphere surface/friction moment arm is wrong");
    const auto plus = static_cast<const NMHumanSupportKKTGPU*>(plusSample.contents)->residual;
    const auto minus = static_cast<const NMHumanSupportKKTGPU*>(minusSample.contents)->residual;
    const auto dual = static_cast<const nm_float4*>(work.contents)[1];
    const float error = std::max({
        std::abs(dual.x+(plus.x-minus.x)/(2*epsilon)),
        std::abs(dual.y+(plus.y-minus.y)/(2*epsilon)),
        std::abs(dual.z+(plus.z-minus.z)/(2*epsilon))});
    require(error<3.0e-4f, "support NCP action disagrees with finite difference");
    std::cout << "SUPPORT shape=" << shape << " total_velocity=-1 delta_velocity=-0.25"
              << " dual_fd_error=" << error << '\n';
    require(rolledHistory.x == initialHistory.x && rolledHistory.y == 0.0f &&
                rolledHistory.z == 0.0f && rolledHistory.w == initialHistory.w &&
                rolledConsequence.identity.x == 0u &&
                rolledConsequence.identity.y == 0u &&
                rolledConsequence.identity.z == 0u &&
                rolledConsequence.identity.w == 0u,
            "Matter support checkpoint rollback did not restore exact bytes");
}

std::uint64_t fullBodySourceFingerprint(
    const std::vector<std::uint8_t>& rigid,
    const std::vector<std::uint8_t>& muscle,
    const std::vector<std::uint8_t>& support
) noexcept {
    return metalrobo::numiHumanRuntimeBaseSourceFingerprint(
        std::as_bytes(std::span(rigid)),
        std::as_bytes(std::span(muscle)),
        std::as_bytes(std::span(support)));
}

std::uint64_t motorOutputFingerprint(
    const MRNumanXBrainMotorOutputHeaderGPU& header,
    const float* excitation
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION);
    mixU32(hash, header.formatVersion);
    mixU32(hash, header.flags);
    mixU64(hash, header.timestampMicroseconds);
    mixU64(hash, header.brainGeneration);
    mixU64(hash, header.profileFingerprint);
    mixU64(hash, header.protectiveCommandFingerprint);
    mixU32(hash, header.muscleCount);
    mixU32(hash, header.environmentIdentifier);
    mixFloat(hash, header.motorInhibition);
    mixFloat(hash, header.autonomicArousal);
    mixU32(hash, header.actuatorCommandKind);
    mixU32(hash, header.reserved);
    mixFloat(hash, header.outputMinimum);
    mixFloat(hash, header.outputMaximum);
    for (std::uint32_t index = 0u; index < header.muscleCount; ++index) {
        mixFloat(hash, excitation[index]);
    }
    return hash;
}

std::uint64_t readyGateFingerprint(
    const MRNumanXBrainMotorReadyGateGPU& gate
) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&gate);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u; index < 152u; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

std::uint64_t timingFingerprint(
    const mrnx_candidate_timing_v1& timing
) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&timing);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u;
         index < offsetof(mrnx_candidate_timing_v1, timing_fingerprint);
         ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

mrnx_metal_range_v1 range(
    id<MTLBuffer> buffer,
    const std::uint32_t elementType,
    const std::uint32_t elementBytes
) noexcept {
    mrnx_metal_range_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.metal_buffer = (__bridge void*)buffer;
    result.gpu_address = buffer.gpuAddress;
    result.byte_count = buffer.length;
    result.element_type = elementType;
    result.element_byte_count = elementBytes;
    return result;
}

struct Completion {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<std::uint32_t> status{0u};
    mrnx_prepared_v1* prepared = nullptr;
    mrnx_candidate_v1* candidate = nullptr;
    mrnx_root_v1 root{};
};

void settled(
    void* raw,
    mrnx_prepared_v1* prepared,
    mrnx_candidate_v1* candidate,
    const mrnx_completion_v1* completion,
    const mrnx_root_v1* root
) {
    auto* capture = static_cast<Completion*>(raw);
    if (capture == nullptr || completion == nullptr) return;
    if (prepared != nullptr) mrnx_bridge_v1_prepared_retain(prepared);
    if (candidate != nullptr) mrnx_bridge_v1_candidate_retain(candidate);
    capture->prepared = prepared;
    capture->candidate = candidate;
    if (root != nullptr) capture->root = *root;
    capture->status.store(completion->status, std::memory_order_release);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
}

struct GenerationLatchCapture {
    std::atomic<std::uint32_t> count{0u};
};

bool generationLatch(
    void* raw,
    const std::uint64_t
) {
    auto* capture = static_cast<GenerationLatchCapture*>(raw);
    if (capture == nullptr) return false;
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
    return true;
}

void waitForCompletion(Completion& completion, const unsigned timeoutSeconds=10u) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(timeoutSeconds);
    while (completion.count.load(std::memory_order_acquire) == 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(
                stderr,
                "full-body root exceeded terminal callback deadline\n");
            std::fflush(stderr);
            // The accepted asynchronous begin borrows this stack capture.
            // Exit without unwinding so a lost or late callback cannot write
            // into an expired context.
            std::_Exit(124);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(
        completion.count.load(std::memory_order_acquire) == 1u,
        "full-body root did not settle exactly once");
}

void appendSyntheticVascularOwner(numi::matter::WorldSource& world) {
    // This owner is deliberately source-independent. It proves package
    // admission and identity/mass/momentum wiring on the real fullbody
    // topology; it is not anatomical, subject-calibrated, or behavioral data.
    world.vascular.contentIdentity = {
        0x4e554d495f424c44ull, 0x534d4153535f5631ull,
        0x464958545552455full, 0x31325f3555535f31ull};
    world.vascular.sourceIdentity = {
        0x53594e5448455449ull, 0x435f424c4f4f445full,
        0x4d4543485f5631ull, 0x0000000000000001ull};
    world.vascular.authoredIdentity = {
        0x4d41545445525f46ull, 0x554c4c424f44595full,
        0x424c4f4f445f4f57ull, 0x4e45525f5631ull};
    world.vascular.compartments.push_back({
        2u, "fixture:blood-proximal", 1.0e-6, 1.0e-6,
        std::getenv("MRNX_SYNTHETIC_PROXIMAL_PRESSURE") ? std::strtod(std::getenv("MRNX_SYNTHETIC_PROXIMAL_PRESSURE"), nullptr) : 0.25,
        1.0e-9, 0.0, 1.0e-6, 1.0e-5, {},
        numi::matter::VascularStorageKind::absoluteVolume,
        numi::matter::VascularPressureLaw::linearCompliance});
    world.vascular.compartments.push_back({
        3u, "fixture:blood-distal", 1.0e-6, 1.0e-6,
        0.0, 1.0e-9, 0.0, 1.0e-6, 1.0e-5, {},
        numi::matter::VascularStorageKind::absoluteVolume,
        numi::matter::VascularPressureLaw::linearCompliance});
    world.vascular.connections.push_back({
        4u, 2u, 3u, 1.0e9, 1.0e-3, 0.0, 1.0e-6, 1.0, 1.0e-5,
        numi::matter::VascularFlowLaw::resistanceInertance});
    numi::matter::VascularTissueSource bloodOwner;
    bloodOwner.stableIdentifier = 5u;
    bloodOwner.anatomicalIdentifier = "fixture:blood-owner";
    bloodOwner.volume = 1.0e-6;
    bloodOwner.objectIndex = 0u;
    bloodOwner.bloodCompartment = 2u;
    bloodOwner.bloodDensity = 1060.0;
    bloodOwner.bloodMomentumTransfer = true;
    bloodOwner.pressureFromCompartment = 2u;
    bloodOwner.pressureToCompartment = 3u;
    bloodOwner.pressureDirection = {0.0, 0.0, 1.0};
    bloodOwner.pressureArea = 1.0e-4;
    bloodOwner.femRegion = {{0u, 0.25}, {1u, 0.25}, {2u, 0.25}, {3u, 0.25}};
    world.vascular.tissues.push_back(std::move(bloodOwner));
}

numi::matter::WorldSource authoredFixtureWorld(const std::vector<float>& preparedQ = {}, const bool includeVascular = false) {
    // Decode only the existing rigid ABI needed by this qualification fixture.
    // Production asset parsing remains in the native runtime owner.
    std::ifstream input(MRNX_FULLBODY_RIGID, std::ios::binary);
    std::array<std::uint32_t, 20u> header{};
    input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    require(input.good() && header[5u] == 157u && header[7u] == 129u &&
                header[8u] == 128u, "unexpected full-body fixture header");
    auto read = [&]<typename T>(T& value) {
        input.read(reinterpret_cast<char*>(&value), sizeof(T));
        require(input.good(), "truncated rigid fixture");
    };
    metalrobo::EngineModel model;
    read(model.world);
    model.articulations.resize(1u);
    read(model.articulations[0u]);
    model.bodies.resize(header[5u]);
    model.joints.resize(header[6u]);
    model.dofs.resize(header[8u]);
    model.defaultQ.resize(header[7u]);
    model.defaultV.resize(header[8u]);
    for (auto& value : model.bodies) read(value);
    for (auto& value : model.joints) read(value);
    for (auto& value : model.dofs) read(value);
    for (auto& value : model.defaultQ) read(value);
    for (auto& value : model.defaultV) read(value);
    require(preparedQ.empty() || preparedQ.size() == model.defaultQ.size(), "prepared fixture pose dimensions");
    const auto& q = preparedQ.empty() ? model.defaultQ : preparedQ;
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(model.bodies.size());
    require(metalrobo::computeArticulatedBodyKinematics(model, 0u,
                std::vector<double>(q.begin(), q.end()),
                std::vector<double>(model.defaultV.begin(), model.defaultV.end()),
                bodies).succeeded(), "authored fixture kinematics failed");
    const auto material = numi::matter::parseMatterFile(MRNX_MATTER_MATERIAL);
    require(material.succeeded(), "authored fixture material failed");
    numi::matter::WorldSource world;
    world.frameTimestep = static_cast<double>(kDurationMicros) / 1'000'000.0;
    world.gravity = {model.world.gravityAndTimestep.x,
                     model.world.gravityAndTimestep.y,
                     model.world.gravityAndTimestep.z};
    world.articulatedDofCapacity = 160u;
    world.articulatedQCapacity = 161u;
    world.mixedSolver.newtonIterations = 16u;
    world.mixedSolver.relativeResidual = 5.0e-3;
    world.materials.push_back(material.material);
    // Three disjoint tiny samples exercise a package wider than the old
    // four-node constant and the ten support queries. No anatomical claim.
    for (std::uint32_t objectIndex = 0u; objectIndex < 3u; ++objectIndex) {
        numi::matter::ObjectSource object;
        object.name = "authored_fixture_" + std::to_string(objectIndex);
        object.representation = numi::matter::Representation::fem;
        object.mixedFEM = false;
        object.characteristicLength = 0.01;
        const std::array<std::array<double, 3u>, 4u> offsets{{
            {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
            {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01}}};
        const auto& body = bodies[0u];
        const auto& q = body.orientation;
        for (std::uint32_t node = 0u; node < 4u; ++node) {
            auto local = offsets[node];
            local[0] += 0.03 * objectIndex;
            const std::array<double, 3u> t{
                2.0 * (q[1] * local[2] - q[2] * local[1]),
                2.0 * (q[2] * local[0] - q[0] * local[2]),
                2.0 * (q[0] * local[1] - q[1] * local[0])};
            object.femNodes.push_back({
                body.centerOfMassPosition[0] + local[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1],
                body.centerOfMassPosition[1] + local[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
                body.centerOfMassPosition[2] + local[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]});
            numi::matter::FEMHumanAttachmentSource attachment;
            attachment.node = node;
            attachment.bodyIndex = 0u;
            attachment.stableIdentifier = 1000u + 4u * objectIndex + node;
            attachment.localPoint = local;
            object.femHumanAttachments.push_back(attachment);
        }
        object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
        world.objects.push_back(object);
    }
    if (includeVascular) appendSyntheticVascularOwner(world);
    return world;
}

void qualifyFullBodyVascularAdmission() {
    @autoreleasepool {
        const auto source = authoredFixtureWorld({}, true);
        numi::matter::CompileOptions options;
        options.maximumRateExponent = 0u;
        const auto compiled = numi::matter::compileWorld(source, options);
        require(compiled.succeeded(), "fullbody vascular source failed to compile");
        std::string error;
        require(numi::matter::validateCompiledWorldLayout(compiled.world, &error),
                error.c_str());
        const auto& world = compiled.world;
        const auto& vascular = world.vascular;
        require(world.dispatch.objectCount == 3u &&
                    world.dispatch.femNodeCount == 12u &&
                    world.dispatch.femHumanAttachmentCount == 12u &&
                    vascular.layout.counts.x == 2u &&
                    vascular.layout.counts.y == 1u &&
                    vascular.layout.counts.w == 1u &&
                    vascular.layout.ranges.z == 3u &&
                    vascular.tissues.size() == 1u &&
                    vascular.tissueBindings.size() == 4u,
                "fullbody vascular package counts are not canonical");
        const auto& owner = vascular.tissues.front();
        require(owner.identity.z == 0u &&
                    owner.identity.w ==
                        (1u | NM_VASCULAR_TISSUE_MOMENTUM_TRANSFER) &&
                    owner.region.x == 0u && owner.region.y == 4u &&
                    owner.region.z == 1u && owner.region.w == 2u &&
                    owner.physical.y == 1060.0f &&
                    std::abs(owner.physical.z - 1.06e-3f) < 1.0e-8f,
                "fullbody vascular owner mass/momentum contract is not cooked");
        const auto package = std::filesystem::temp_directory_path() /
            ("numanx-fullbody-vascular-" + std::to_string(getpid()) +
             ".nmatterpack");
        require(numi::matter::writePackage(compiled, package, &error),
                error.c_str());
        numi::matter::CompiledWorld reloaded;
        require(numi::matter::readPackage(package, reloaded, nullptr, &error),
                error.c_str());
        require(numi::matter::validateCompiledWorldLayout(reloaded, &error),
                error.c_str());
        require(reloaded.fingerprint == world.fingerprint &&
                    reloaded.vascular.tissues.size() == 1u &&
                    reloaded.vascular.tissues.front().identity.w ==
                        owner.identity.w,
                "fullbody vascular package replay changed owner identity");
        std::filesystem::remove(package);
        std::printf(
            "numanx_fullbody_vascular_admission=pass bodies=157 nq=129 nv=128 "
            "objects=3 fem_nodes=12 vascular_compartments=2 connections=1 "
            "tissue_owner=1 blood_density=1060kg_m3 momentum_transfer=explicit "
            "package_replay=bitwise dynamics=unqualified anatomy=unqualified "
            "subject_calibration=unqualified\n");
    }
}

void qualifyTouchAggregation(id<MTLDevice> device) {
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@MRNX_METALROBO_METALLIB] error:&error];
    require(library!=nil,"touch aggregation library unavailable");
    id<MTLFunction> function=[library newFunctionWithName:@"numanx_human_aggregate_support"];
    require(function!=nil,"touch aggregation kernel missing");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil,"touch aggregation pipeline failed");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    std::vector<MRNumanXHumanSupportConsequenceGPU> rows(18u);
    std::vector<mr_uint4> mapping;
    unsigned first=0u;
    for(unsigned receptor=0;receptor<10;++receptor) {
        const unsigned count=receptor<8?2u:1u;
        mapping.push_back({first,count,100u+receptor,200u+receptor});
        for(unsigned j=0;j<count;++j) {
            const unsigned index=first+j;auto& r=rows[index];
            r.identity={index,200u+receptor,1u,MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION};
            r.pointAndSeparation={float(index),0,0,-float(index)/100.0f};
            r.impulseAndNormal={0,0,float(index+1),float(index+1)};
            r.tangentVelocityAndImpulse={float(index),0,0,float(index+1)/4.0f};
        }
        first+=count;
    }
    id<MTLBuffer> input=[device newBufferWithBytes:rows.data() length:rows.size()*sizeof(rows[0]) options:MTLResourceStorageModeShared];
    id<MTLBuffer> map=[device newBufferWithBytes:mapping.data() length:mapping.size()*sizeof(mapping[0]) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:10u*sizeof(rows[0]) options:MTLResourceStorageModeShared];
    require(input!=nil&&map!=nil&&output!=nil&&queue!=nil,"touch aggregation allocation failed");
    const auto run=[&]() {
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:map offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        const mr_uint4 dispatch{18u,10u,0u,0u};
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(10,1,1) threadsPerThreadgroup:MTLSizeMake(10,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"touch aggregation GPU failure");
        std::vector<MRNumanXHumanSupportConsequenceGPU> values(10u);
        std::memcpy(values.data(),output.contents,output.length);return values;
    };
    const auto result=run();
    double total=0,weighted=0,expectedTotal=0,expectedWeighted=0;
    for(const auto& row:rows) {expectedTotal+=row.impulseAndNormal.w;expectedWeighted+=row.pointAndSeparation.x*row.impulseAndNormal.w;}
    for(unsigned i=0;i<10;++i) {
        const auto& m=mapping[i];const auto& r=result[i];
        require(r.identity.x==i&&r.identity.y==m.w&&r.identity.w==1u,"touch geometry identity lost");
        float impulse=0,friction=0,minimum=INFINITY;
        for(unsigned j=m.x;j<m.x+m.y;++j) {impulse+=rows[j].impulseAndNormal.w;friction+=rows[j].tangentVelocityAndImpulse.w;minimum=std::min(minimum,rows[j].pointAndSeparation.w);}
        require(r.impulseAndNormal.w==impulse&&r.tangentVelocityAndImpulse.w==friction&&r.pointAndSeparation.w==minimum,"touch row load/gap aggregation changed");
        total+=r.impulseAndNormal.w;weighted+=double(r.pointAndSeparation.x)*r.impulseAndNormal.w;
    }
    require(total==expectedTotal&&std::abs(weighted-expectedWeighted)<1e-4,"touch total impulse or centre of pressure lost");
    const auto replay=run();require(std::memcmp(result.data(),replay.data(),10u*sizeof(rows[0]))==0,"touch aggregation replay drift");
    auto* raw=static_cast<MRNumanXHumanSupportConsequenceGPU*>(input.contents);
    for(unsigned mutation=0;mutation<4;++mutation) {
        std::memcpy(input.contents,rows.data(),input.length);
        if(mutation==0)raw[1].identity.y^=1u;
        if(mutation==1)raw[1].impulseAndNormal.w=-1;
        if(mutation==2)raw[1].pointAndSeparation.x=NAN;
        if(mutation==3)raw[1].identity.x=18u;
        const auto invalid=run();
        require(invalid[0].identity.w==0u&&invalid[1].identity.w==1u,"bad row did not invalidate only its receptor");
    }
    std::memcpy(input.contents,rows.data(),input.length);
    raw[1].impulseAndNormal.w=-1.0e-12f;
    const auto signedZero=run();
    require(signedZero[0].identity.w==MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION &&
        signedZero[0].impulseAndNormal.w==rows[0].impulseAndNormal.w,
        "sub-ulp negative unloaded support impulse was not clamped");
    std::printf("numanx_touch_aggregation=pass rows=18 receptors=10 impulse=%.9g centre_of_pressure=conserved replay=bitwise malformed_rows=4\n",total);
}

// Qualification fixture only: import a saved native compiler state and cook
// the existing three tiny pelvis samples in that pose. No tissue registration,
// calibration or dynamics is performed by this construction helper.
int writePreparedStanceFixture(const char* certificate, const char* output,
    const char* contacts, const char* equalities, const char* limits,
    const std::uint64_t timestepMicroseconds = 100u, const bool importInitialState = false,
    const std::uint32_t newtonIterations = 16u,
    const std::uint64_t timestepNanoseconds = 0u,
    const std::uint32_t fgmresRestart = NM_MIXED_FGMRES_DEFAULT_RESTART,
    const std::uint32_t fgmresIterations = NM_MIXED_FGMRES_ITERATIONS,
    const double relativeResidual = 5.0e-3) {
    @autoreleasepool {
        require(timestepMicroseconds <= 1'000'000u && timestepNanoseconds <= 1'000'000'000u,
            "prepared fixture timestep exceeds one second");
        require(timestepNanoseconds != 0u || timestepMicroseconds != 0u,
            "prepared fixture timestep must be positive");
        require(timestepNanoseconds == 0u || timestepMicroseconds == 0u ||
            timestepNanoseconds == timestepMicroseconds * 1000u,
            "prepared fixture clock units disagree");
        const auto exactNanoseconds = timestepNanoseconds != 0u
            ? timestepNanoseconds : timestepMicroseconds * 1000u;
        const auto rigid = readPayloadBytes(MRNX_FULLBODY_RIGID);
        require(rigid.size() >= 80u, "truncated source rigid fixture");
        const std::uint32_t rigidBodyCount =
            std::uint32_t(rigid[20]) |
            (std::uint32_t(rigid[21]) << 8u) |
            (std::uint32_t(rigid[22]) << 16u) |
            (std::uint32_t(rigid[23]) << 24u);
        std::array<std::uint8_t, 32u> sourceSHA{};
        std::copy_n(rigid.begin()+48u, 32u, sourceSHA.begin());
        const auto supportBytes = readPayloadBytes(contacts);
        require(supportBytes.size() >= sizeof(metalrobo::NumiHumanSupportHeader) &&
            supportBytes.size() <= std::numeric_limits<CC_LONG>::max(),
            "support payload cannot be identified");
        metalrobo::NumiHumanSupportHeader rawSupportHeader{};
        std::memcpy(&rawSupportHeader, supportBytes.data(), sizeof(rawSupportHeader));
        metalrobo::NumiHumanSupportPayload supportPayload;
        std::string supportError;
        require(metalrobo::decodeNumiHumanSupportPayload(
            std::as_bytes(std::span(supportBytes)), rigidBodyCount,
            sourceSHA, supportPayload, supportError), supportError.c_str());
        metalrobo::NumiHumanSupportPayloadIdentity supportIdentity;
        CC_SHA256(supportBytes.data(), static_cast<CC_LONG>(supportBytes.size()),
            supportIdentity.sha256.data());
        supportIdentity.byteCount = supportBytes.size();
        supportIdentity.payloadABI = rawSupportHeader.payloadAbi;
        supportIdentity.sourceRecordCount = rawSupportHeader.contactCount;
        supportIdentity.expandedRowCount =
            static_cast<std::uint32_t>(supportPayload.contacts.size());
        metalrobo::NumiHumanInitialState initial;
        if (importInitialState) {
            const auto bytes = readPayloadBytes(certificate);
            std::string error;
            const bool decoded = metalrobo::decodeNumiHumanInitialState(
                {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()},
                MRNX_FULL_BODY_NQ, MRNX_FULL_BODY_NV, MRNX_FULL_BODY_MUSCLE_COUNT,
                sourceSHA, supportIdentity, initial, error);
            require(decoded, error.c_str());
        } else {
        std::ifstream log(certificate);
        require(log.good(), "could not open native stance certificate");
        std::string line, qText, muscleText, compliantText;
        std::vector<double> loggedSupportNormalForces;
        unsigned qCount=0, muscleCount=0, compliantCount=0, supportLineCount=0;
        while(std::getline(log,line)) {
            const std::string qPrefix="compiled_equilibrium_q=";
            const std::string musclePrefix="compiled_equilibrium_muscles=";
            const std::string compliantPrefix="source_compliant_equilibrium=";
            if(line.starts_with(qPrefix)) {qText=line.substr(qPrefix.size());++qCount;}
            if(line.starts_with(musclePrefix)) {muscleText=line.substr(musclePrefix.size());++muscleCount;}
            if(line.starts_with(compliantPrefix)) {compliantText=line.substr(compliantPrefix.size());++compliantCount;}
            if(line.starts_with("numi_human_whole_body_support_wrench=ok")) {
                ++supportLineCount;
                metalrobo::NumiHumanSupportPayloadIdentity loggedSupportIdentity;
                unsigned supportSHACount=0u, supportByteCount=0u,
                    supportABICount=0u, supportSourceCount=0u,
                    supportExpandedCount=0u;
                std::istringstream fields(line);
                for (std::string token; fields >> token;) {
                    constexpr std::string_view shaPrefix="support_sha256=";
                    constexpr std::string_view bytesPrefix="support_bytes=";
                    constexpr std::string_view abiPrefix="support_abi=";
                    constexpr std::string_view sourcePrefix="support_source_records=";
                    constexpr std::string_view expandedPrefix="support_expanded_rows=";
                    if (token.starts_with(shaPrefix)) {
                        require(++supportSHACount==1u,
                            "duplicate support SHA-256 in certificate");
                        loggedSupportIdentity.sha256=parseSupportSHA256Hex(
                            std::string_view(token).substr(shaPrefix.size()));
                        continue;
                    }
                    if (token.starts_with(bytesPrefix)) {
                        require(++supportByteCount==1u,
                            "duplicate support byte count in certificate");
                        loggedSupportIdentity.byteCount=parseUnsignedIdentityField(
                            std::string_view(token).substr(bytesPrefix.size()),
                            std::numeric_limits<CC_LONG>::max());
                        continue;
                    }
                    if (token.starts_with(abiPrefix)) {
                        require(++supportABICount==1u,
                            "duplicate support ABI in certificate");
                        loggedSupportIdentity.payloadABI=static_cast<std::uint32_t>(
                            parseUnsignedIdentityField(
                                std::string_view(token).substr(abiPrefix.size()),
                                std::numeric_limits<std::uint32_t>::max()));
                        continue;
                    }
                    if (token.starts_with(sourcePrefix)) {
                        require(++supportSourceCount==1u,
                            "duplicate support source count in certificate");
                        loggedSupportIdentity.sourceRecordCount=static_cast<std::uint32_t>(
                            parseUnsignedIdentityField(
                                std::string_view(token).substr(sourcePrefix.size()),
                                std::numeric_limits<std::uint32_t>::max()));
                        continue;
                    }
                    if (token.starts_with(expandedPrefix)) {
                        require(++supportExpandedCount==1u,
                            "duplicate support expanded count in certificate");
                        loggedSupportIdentity.expandedRowCount=static_cast<std::uint32_t>(
                            parseUnsignedIdentityField(
                                std::string_view(token).substr(expandedPrefix.size()),
                                std::numeric_limits<std::uint32_t>::max()));
                        continue;
                    }
                    constexpr std::string_view prefix="contact_";
                    constexpr std::string_view infix="_normal_force_n=";
                    if (!token.starts_with(prefix)) continue;
                    const auto split=token.find(infix);
                    if (split==std::string::npos) continue;
                    const auto indexText=token.substr(prefix.size(),split-prefix.size());
                    require(!indexText.empty() && indexText.find_first_not_of("0123456789")==std::string::npos,
                        "invalid support force row identity in certificate");
                    const auto index=std::stoull(indexText);
                    require(index<=std::numeric_limits<std::uint32_t>::max(),
                        "support force row exceeds ABI capacity");
                    if (loggedSupportNormalForces.size()<=index)
                        loggedSupportNormalForces.resize(index+1u,
                            std::numeric_limits<double>::quiet_NaN());
                    require(std::isnan(loggedSupportNormalForces[index]),
                        "duplicate support force row in certificate");
                    loggedSupportNormalForces[index]=parseExactFiniteDecimal(
                        std::string_view(token).substr(split+infix.size()));
                }
                require(supportSHACount==1u && supportByteCount==1u &&
                    supportABICount==1u && supportSourceCount==1u &&
                    supportExpandedCount==1u,
                    "certificate lacks exact support payload identity");
                require(loggedSupportIdentity==supportIdentity,
                    "certificate support payload identity disagrees with supplied NHCNT");
            }
        }
        require(qCount==1 && muscleCount==1,"certificate state missing or duplicated");
        const auto parse=[](const std::string& text) -> id {
            NSData* data=[NSData dataWithBytes:text.data() length:text.size()];
            NSError* error=nil;
            id result=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            require(result!=nil && error==nil,"invalid native state JSON");return result;
        };
        id qJSON=parse(qText), muscleJSON=parse(muscleText);
        if (compliantCount != 0u) {
            require(compliantCount==1u && supportLineCount==0u,
                "certificate support history source is duplicated");
            id compliantJSON=parse(compliantText);
            require([compliantJSON isKindOfClass:[NSDictionary class]] &&
                [compliantJSON[@"schema"] isEqual:@"numi.human.source-compliant-equilibrium.v1"],
                "source-compliant certificate schema is invalid");
            const auto unsignedJSON=[](id value, const std::uint64_t maximum) {
                require([value isKindOfClass:[NSNumber class]] &&
                    CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),
                    "support identity field is not numeric");
                const double scalar=[value doubleValue];
                require(std::isfinite(scalar) && scalar>=0.0 &&
                    std::floor(scalar)==scalar,
                    "support identity field is not an unsigned integer");
                const auto parsed=[value unsignedLongLongValue];
                require(parsed<=maximum && static_cast<double>(parsed)==scalar,
                    "support identity field exceeds its ABI domain");
                return static_cast<std::uint64_t>(parsed);
            };
            id shaJSON=compliantJSON[@"support_sha256"];
            require([shaJSON isKindOfClass:[NSString class]],
                "source-compliant certificate lacks support SHA-256");
            const char* shaText=[shaJSON UTF8String];
            require(shaText!=nullptr,
                "source-compliant support SHA-256 is not UTF-8");
            metalrobo::NumiHumanSupportPayloadIdentity loggedSupportIdentity;
            loggedSupportIdentity.sha256=parseSupportSHA256Hex(shaText);
            loggedSupportIdentity.byteCount=unsignedJSON(
                compliantJSON[@"support_bytes"],
                std::numeric_limits<CC_LONG>::max());
            loggedSupportIdentity.payloadABI=static_cast<std::uint32_t>(unsignedJSON(
                compliantJSON[@"support_abi"],
                std::numeric_limits<std::uint32_t>::max()));
            loggedSupportIdentity.sourceRecordCount=static_cast<std::uint32_t>(unsignedJSON(
                compliantJSON[@"support_source_records"],
                std::numeric_limits<std::uint32_t>::max()));
            loggedSupportIdentity.expandedRowCount=static_cast<std::uint32_t>(unsignedJSON(
                compliantJSON[@"support_expanded_rows"],
                std::numeric_limits<std::uint32_t>::max()));
            require(loggedSupportIdentity==supportIdentity,
                "source-compliant support identity disagrees with supplied NHCNT");
            id forceJSON=compliantJSON[@"support_normal_force"];
            require([forceJSON isKindOfClass:[NSArray class]],
                "source-compliant certificate lacks support normal force rows");
            loggedSupportNormalForces.reserve([forceJSON count]);
            for (id value in forceJSON) {
                require([value isKindOfClass:[NSNumber class]] &&
                    CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),
                    "support normal force is not numeric");
                loggedSupportNormalForces.push_back([value doubleValue]);
            }
        } else {
            require(supportLineCount==1u,
                "certificate lacks one support force history source");
        }
        require([qJSON isKindOfClass:[NSArray class]] && [qJSON count]==MRNX_FULL_BODY_NQ &&
            [muscleJSON isKindOfClass:[NSDictionary class]] &&
            [muscleJSON[@"schema"] isEqual:@"numi.human.offline-muscle-state.v1"],"native state schema mismatch");
        NSArray* activation=muscleJSON[@"activation_fp32"];
        NSArray* fiber=muscleJSON[@"reference_fiber_length_m"];
        require([activation isKindOfClass:[NSArray class]] && [fiber isKindOfClass:[NSArray class]] &&
            activation.count==MRNX_FULL_BODY_MUSCLE_COUNT && fiber.count==MRNX_FULL_BODY_MUSCLE_COUNT,
            "native muscle state dimensions mismatch");
        const auto scalar=[](id x) {
            require([x isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)x)!=CFBooleanGetTypeID(),
                "state component is not numeric");
            const float value=[x floatValue];require(std::isfinite(value),"nonfinite prepared state");return value;
        };
        for(id x in qJSON) initial.q.push_back(scalar(x));
        initial.v.assign(MRNX_FULL_BODY_NV,0.0f);
        for(unsigned i=0;i<MRNX_FULL_BODY_MUSCLE_COUNT;++i) {
            const float a=scalar(activation[i]);
            MRMujocoMuscleStateGPU state{};state.excitationAndActivation={a,a,scalar(fiber[i]),0.0f};
            initial.muscles.push_back(state);
        }
        require(loggedSupportNormalForces.size()==supportIdentity.expandedRowCount,
            "certificate support forces do not cover the expanded NHCNT rows");
        metalrobo::NumiHumanPreparedSupportHistory prepared;
        prepared.support=supportIdentity;
        prepared.rows.reserve(loggedSupportNormalForces.size());
        const double timestepSeconds=double(exactNanoseconds)*1.0e-9;
        for (const double force : loggedSupportNormalForces) {
            require(std::isfinite(force) && force>=0.0,
                "certificate support normal force is invalid");
            const float impulse=static_cast<float>(force*timestepSeconds);
            require(std::isfinite(impulse) && impulse>=0.0f,
                "prepared support impulse is not FP32-representable");
            prepared.rows.push_back({0.0f,0.0f,0.0f,impulse});
        }
        initial.preparedSupportHistory=std::move(prepared);
        }
        require(newtonIterations > 0u && newtonIterations <= 128u, "invalid prepared Newton iteration budget");
        require(
            fgmresRestart > 0u &&
                fgmresRestart <= NM_MIXED_FGMRES_RESTART &&
                fgmresIterations >= fgmresRestart &&
                fgmresIterations <= NM_MIXED_FGMRES_MAX_ITERATIONS &&
                fgmresIterations <=
                    std::numeric_limits<std::uint32_t>::max() -
                        fgmresRestart + 1u &&
                std::isfinite(relativeResidual) &&
                relativeResidual > 0.0 &&
                std::isfinite(static_cast<float>(relativeResidual)) &&
                static_cast<float>(relativeResidual) > 0.0f,
            "invalid prepared FGMRES or residual policy");
        const bool includeVascular = std::getenv("MRNX_INCLUDE_SYNTHETIC_VASCULAR") != nullptr;
        auto world=authoredFixtureWorld(initial.q, includeVascular);world.frameTimestep=exactNanoseconds*1.0e-9;
        world.mixedSolver.newtonIterations = newtonIterations;
        world.mixedSolver.fgmresRestart = fgmresRestart;
        world.mixedSolver.fgmresIterations = fgmresIterations;
        world.mixedSolver.relativeResidual = relativeResidual;
        numi::matter::CompileOptions options;options.maximumRateExponent=0u;
        const auto compiled=numi::matter::compileWorld(world,options);
        require(compiled.succeeded(),"prepared fixture world failed to compile");
        const float compiledRelativeResidual =
            compiled.world.mixedSolver.residualTolerances.x;
        require(
            std::isfinite(compiledRelativeResidual) &&
                compiledRelativeResidual > 0.0f,
            "compiled prepared relative residual is invalid");
        std::copy_n(rigid.begin()+48u,32u,initial.sourceArchiveSHA256.begin());
        const auto base=fullBodySourceFingerprint(rigid,readPayloadBytes(MRNX_FULLBODY_MUSCLE),readPayloadBytes(contacts));
        auto source=constrainedFingerprint(base,readPayloadBytes(equalities));
        source=metalrobo::numiHumanRuntimeAppendPayloadOwner(
            source,"NHLIM1",equalityFingerprint(readPayloadBytes(limits)));
        const auto expectedSource = source;
        require(!importInitialState || initial.humanSourceFingerprint == expectedSource,
            "imported prepared state composed source mismatch");
        initial.humanSourceFingerprint=expectedSource;
        initial.worldFingerprint=compiled.world.fingerprint;
        initial.timestepMicroseconds=exactNanoseconds % 1000u == 0u ? exactNanoseconds / 1000u : 0u;
        // Preserve an imported state's root-extension representation. Exact
        // clock migration does not require inventing a compensated root;
        // newly authored stance state still receives one when its clock does.
        if (timestepNanoseconds != 0u || initial.timestepNanoseconds != 0u || initial.rootTranslation) {
            initial.timestepNanoseconds=exactNanoseconds;
            if (!importInitialState && !initial.rootTranslation) initial.rootTranslation=mrCompensatedTranslationFromProjection(
                {initial.q[0],initial.q[1],initial.q[2],0.0f});
        }
        require(metalrobo::numiHumanInitialStateTimestepNanoseconds(initial) == exactNanoseconds,
            "prepared fixture exact clock mismatch");
        std::string error;std::vector<std::byte> bytes;
        const bool encoded=metalrobo::encodeNumiHumanInitialState(initial,bytes,error);
        require(encoded,error.c_str());
        const std::filesystem::path suppliedDirectory(output);
        require(!suppliedDirectory.empty(),
            "prepared fixture output path is empty");
        for (const auto& component : suppliedDirectory) {
            require(
                component != "..",
                "prepared fixture output may not contain .. components");
        }
        std::filesystem::path directory =
            std::filesystem::absolute(suppliedDirectory).lexically_normal();
        if (!directory.has_filename() &&
            directory != directory.root_path()) {
            directory = directory.parent_path();
        }
        require(
            directory.has_filename() &&
                directory != directory.root_path(),
            "prepared fixture output must name a non-root directory");
        std::error_code pathError;
        auto parent = std::filesystem::canonical(
            directory.parent_path(), pathError);
        require(
            !pathError && std::filesystem::is_directory(parent, pathError) &&
                !pathError,
            "prepared fixture output parent is not an existing directory");
        directory = parent / directory.filename();
        const ScopedDescriptor parentDescriptor(
            openNoFollow(
                parent, O_RDONLY | O_DIRECTORY,
                "prepared fixture parent directory"));
        const auto destinationStatus =
            std::filesystem::symlink_status(directory, pathError);
        require(
            (!pathError || pathError == std::errc::no_such_file_or_directory) &&
                destinationStatus.type() == std::filesystem::file_type::not_found,
            "prepared fixture output already exists or cannot be inspected");

        std::filesystem::path staging;
        bool published = false;
        const auto durationName=timestepNanoseconds != 0u
            ? std::to_string(exactNanoseconds)+"ns" : std::to_string(initial.timestepMicroseconds)+"us";
        const std::vector<std::uint8_t> raw(reinterpret_cast<const std::uint8_t*>(bytes.data()),
            reinterpret_cast<const std::uint8_t*>(bytes.data())+bytes.size());
        const std::string supportSHAHex=supportSHA256Hex(supportIdentity);
        try {
            std::string stagingTemplate =
                (parent / ("." + directory.filename().string() +
                           ".staging.XXXXXX")).string();
            std::vector<char> mutableTemplate(
                stagingTemplate.begin(), stagingTemplate.end());
            mutableTemplate.push_back('\0');
            const char* created = ::mkdtemp(mutableTemplate.data());
            require(created != nullptr,
                "could not create prepared fixture staging directory");
            staging = std::filesystem::path(created);
            const ScopedDescriptor stagingDescriptor(
                openNoFollow(
                    staging, O_RDONLY | O_DIRECTORY,
                    "prepared fixture staging directory"));
            requirePinnedDirectoryEntry(
                parentDescriptor.get(), staging.filename(),
                stagingDescriptor.get());

            const auto packagePath =
                staging/("prepared-"+durationName+".nmatterpack");
            const auto statePath = staging/"prepared.nhinit";
            const auto receiptPath = staging/"prepared.json";
            const bool saved=numi::matter::writePackage(compiled,packagePath,&error);
            require(saved,error.c_str());
            std::ofstream stateFile(statePath,std::ios::binary);
            stateFile.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            stateFile.close();
            require(stateFile.good(),"could not write prepared state");
            std::ostringstream receiptText;
            receiptText.imbue(std::locale::classic());
            receiptText << "{\"schema\":\"numi.human.prepared-stance-fixture.v2\",\"human_source_fp\":\"" << std::hex << base
                << "\",\"composed_human_source_fp\":\"" << initial.humanSourceFingerprint
                << "\",\"world_fp\":\"" << initial.worldFingerprint
                << "\",\"initial_state_fp\":\"" << equalityFingerprint(raw)
                << "\",\"support_sha256\":\"" << supportSHAHex
                << "\",\"support_bytes\":" << std::dec << supportIdentity.byteCount
                << ",\"support_abi\":" << supportIdentity.payloadABI
                << ",\"support_source_records\":" << supportIdentity.sourceRecordCount
                << ",\"support_expanded_rows\":" << supportIdentity.expandedRowCount
                << ",\"solver_newton_iterations\":" << newtonIterations
                << ",\"solver_fgmres_restart\":" << fgmresRestart
                << ",\"solver_fgmres_iterations\":" << fgmresIterations
                << ",\"solver_relative_residual_requested\":" << std::setprecision(17)
                << relativeResidual
                << ",\"solver_relative_residual_fp32\":"
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << compiledRelativeResidual
                << ",\"solver_relative_residual_fp32_bits\":\"0x"
                << std::hex << std::setw(8) << std::setfill('0')
                << std::bit_cast<std::uint32_t>(compiledRelativeResidual)
                << std::dec << std::setfill(' ')
                << "\",\"scope\":\"three tiny pelvis samples; no anatomical tissue or sustained behavior qualification\"}\n";
            const std::string expectedReceipt = receiptText.str();
            std::ofstream receipt(receiptPath, std::ios::binary);
            receipt.write(expectedReceipt.data(), expectedReceipt.size());
            receipt.close();
            require(receipt.good(),"could not write prepared fixture identity");

            numi::matter::CompiledWorld reloadedWorld;
            error.clear();
            require(
                numi::matter::readPackage(
                    packagePath, reloadedWorld, nullptr, &error) &&
                    numi::matter::validateCompiledWorldLayout(
                        reloadedWorld, &error),
                error.c_str());
            require(
                reloadedWorld.fingerprint == compiled.world.fingerprint &&
                    reloadedWorld.mixedSolver.nonlinearIterations.x ==
                        newtonIterations &&
                    reloadedWorld.mixedSolver.nonlinearIterations.y ==
                        fgmresRestart &&
                    reloadedWorld.mixedSolver.nonlinearIterations.z ==
                        fgmresIterations &&
                    std::bit_cast<std::uint32_t>(
                        reloadedWorld.mixedSolver.residualTolerances.x) ==
                        std::bit_cast<std::uint32_t>(compiledRelativeResidual),
                "prepared Matter package readback changed solver identity");

            const auto statePathText = statePath.string();
            const auto reloadedStateBytes =
                readPayloadBytes(statePathText.c_str());
            metalrobo::NumiHumanInitialState reloadedInitial;
            error.clear();
            require(
                metalrobo::decodeNumiHumanInitialState(
                    {reinterpret_cast<const std::byte*>(
                         reloadedStateBytes.data()),
                     reloadedStateBytes.size()},
                    MRNX_FULL_BODY_NQ, MRNX_FULL_BODY_NV,
                    MRNX_FULL_BODY_MUSCLE_COUNT, sourceSHA, supportIdentity,
                    reloadedInitial, error),
                error.c_str());
            std::vector<std::byte> reencodedState;
            error.clear();
            require(
                metalrobo::encodeNumiHumanInitialState(
                    reloadedInitial, reencodedState, error),
                error.c_str());
            require(
                reencodedState == bytes,
                "prepared initial-state readback changed bytes");
            const auto receiptPathText = receiptPath.string();
            const auto reloadedReceipt =
                readPayloadBytes(receiptPathText.c_str());
            require(
                reloadedReceipt.size() == expectedReceipt.size() &&
                    std::memcmp(
                        reloadedReceipt.data(), expectedReceipt.data(),
                        expectedReceipt.size()) == 0,
                "prepared fixture receipt readback changed bytes");
            NSData* receiptData = [NSData
                dataWithBytes:reloadedReceipt.data()
                length:reloadedReceipt.size()];
            NSError* receiptJSONError = nil;
            id receiptDocument = [NSJSONSerialization
                JSONObjectWithData:receiptData
                options:0
                error:&receiptJSONError];
            require(
                receiptJSONError == nil &&
                    [receiptDocument isKindOfClass:[NSDictionary class]],
                "prepared fixture receipt is not valid JSON");
            NSDictionary* receiptObject = (NSDictionary*)receiptDocument;
            NSNumber* receiptResidual =
                receiptObject[@"solver_relative_residual_fp32"];
            NSString* receiptResidualBits =
                receiptObject[@"solver_relative_residual_fp32_bits"];
            const std::uint32_t compiledResidualBits =
                std::bit_cast<std::uint32_t>(compiledRelativeResidual);
            NSString* expectedResidualBits = [NSString
                stringWithFormat:@"0x%08x", compiledResidualBits];
            require(
                [receiptObject[@"schema"] isEqualToString:
                    @"numi.human.prepared-stance-fixture.v2"] &&
                    [receiptResidual isKindOfClass:[NSNumber class]] &&
                    std::bit_cast<std::uint32_t>(receiptResidual.floatValue) ==
                        compiledResidualBits &&
                    [receiptResidualBits isKindOfClass:[NSString class]] &&
                    [receiptResidualBits isEqualToString:expectedResidualBits],
                "prepared fixture receipt changed executable FP32 policy");

            syncPublishedPath(packagePath, "prepared Matter package");
            syncPublishedPath(statePath, "prepared initial state");
            syncPublishedPath(receiptPath, "prepared fixture receipt");
            fullSyncDescriptor(
                stagingDescriptor.get(),
                "prepared fixture staging directory");
            requirePinnedDirectoryEntry(
                parentDescriptor.get(), staging.filename(),
                stagingDescriptor.get());
            if (::renameatx_np(
                    parentDescriptor.get(),
                    staging.filename().c_str(),
                    parentDescriptor.get(),
                    directory.filename().c_str(),
                    RENAME_EXCL | RENAME_NOFOLLOW_ANY) != 0) {
                const std::string message =
                    "could not publish prepared fixture without replacement: " +
                    std::string(std::strerror(errno));
                throw std::runtime_error(message);
            }
            published = true;
            staging.clear();
            fullSyncDescriptor(
                parentDescriptor.get(),
                "prepared fixture parent directory");
        } catch (...) {
            if (published) {
                std::fprintf(
                    stderr,
                    "prepared fixture is visible at %s but final directory "
                    "durability is uncertain\n",
                    directory.c_str());
            } else if (!staging.empty()) {
                // Preserve a failed attempt for inspection. Recursive cleanup
                // through a pathname would be unsafe if a same-UID process
                // renamed or replaced an ancestor after mkdtemp returned.
                std::fprintf(
                    stderr,
                    "prepared fixture staging retained after failure: %s\n",
                    staging.c_str());
            }
            throw;
        }
        std::cout << "prepared_stance_fixture=compiled nq=129 nv=128 muscles=416 objects=3 attachments=12\n";
        return 0;
    }
}

mrnx_runtime_v1* makeAuthoredRuntime(
    const mrnx_runtime_config_v2& base,
    mrnx_runtime_info_v1& info,
    const bool sourceEqualities
) {
    auto source = authoredFixtureWorld();
    const auto package = std::filesystem::temp_directory_path() /
        ("numanx-authored-" + std::to_string(getpid()) + ".nmatterpack");
    const auto packagePath = package.string();
    mrnx_runtime_config_v3 config{};
    config.abi_version = MRNX_RUNTIME_CONFIG_ABI_V3;
    config.struct_size = sizeof(config);
    config.runtime = base;
    config.runtime.matter_material_path = nullptr;
    config.matter_world_package_path = packagePath.c_str();
    config.expected_model_source_fingerprint = fullBodySourceFingerprint(
        readPayloadBytes(MRNX_FULLBODY_RIGID),
        readPayloadBytes(MRNX_FULLBODY_MUSCLE),
        readPayloadBytes(MRNX_FULLBODY_SUPPORT_CONTACT));
    const auto cook = [&](const numi::matter::WorldSource& world) {
        numi::matter::CompileOptions options;
        options.maximumRateExponent = 0u;
        const auto compiled = numi::matter::compileWorld(world, options);
        std::string error;
        require(compiled.succeeded() &&
                    numi::matter::writePackage(compiled, package, &error),
                "authored fixture package failed");
        config.expected_matter_world_fingerprint = compiled.world.fingerprint;
    };
    const auto reject = [&](const mrnx_runtime_config_v3& candidate,
                            const mrnx_runtime_status_v1 status) {
        mrnx_runtime_info_v1 failed{};
        auto* runtime = mrnx_bridge_v1_runtime_create_v3(&candidate, &failed);
        require(runtime == nullptr && failed.status == status,
                "invalid authored world was admitted or misclassified");
    };
    cook(source);
    auto changed = config;
    changed.expected_model_source_fingerprint ^= 1u;
    reject(changed, MRNX_RUNTIME_ASSET_FAILURE_V1);
    changed = config;
    changed.expected_matter_world_fingerprint ^= 1u;
    reject(changed, MRNX_RUNTIME_ASSET_FAILURE_V1);
    changed = config;
    changed.runtime.matter_material_path = MRNX_MATTER_MATERIAL;
    reject(changed, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
    changed = config;
    changed.matter_world_package_path = "/nonexistent-numanx-world.nmatterpack";
    reject(changed, MRNX_RUNTIME_ASSET_FAILURE_V1);
    changed = config;
    changed.runtime.timestep_microseconds += 1u;
    reject(changed, MRNX_RUNTIME_ASSET_FAILURE_V1);
    auto bad = source;
    bad.gravity[2] += 1.0;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad = source;
    bad.objects[0u].femHumanAttachments[0u].localPoint[0] += 0.01;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad = source;
    bad.objects[0u].femHumanAttachments[0u].bodyIndex = 157u;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad = source;
    bad.objects[0u].femInitialVelocity[0] = 0.1;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad = source;
    bad.deterministic = false;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad = source;
    bad.materials[0u].mixed.maximumActiveTension = 100.0;
    cook(bad);
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    cook(source);
    {
        std::fstream corrupted(package, std::ios::binary | std::ios::in | std::ios::out);
        char byte = 0;
        corrupted.read(&byte, 1);
        byte ^= 1;
        corrupted.seekp(0);
        corrupted.write(&byte, 1);
    }
    reject(config, MRNX_RUNTIME_ASSET_FAILURE_V1);
    cook(source);
    mrnx_runtime_v1* runtime = nullptr;
    std::filesystem::path equalityTemporary;
    if (sourceEqualities) {
        const char* equalityPath = std::getenv("MRNX_JOINT_EQUALITIES");
        require(equalityPath != nullptr, "MRNX_JOINT_EQUALITIES must identify NHEQ2");
        const auto equalityBytes = readPayloadBytes(equalityPath);
        equalityTemporary = package.string()+".nheq";
        const auto equalityPathString = equalityTemporary.string();
        const auto writeEquality = [&](const std::vector<std::uint8_t>& bytes) {
            std::ofstream output(equalityTemporary, std::ios::binary);
            output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            require(output.good(), "could not retain equality fixture");
        };
        writeEquality(equalityBytes);
        mrnx_runtime_config_v4 constrained{};
        constrained.abi_version=MRNX_RUNTIME_CONFIG_ABI_V4;
        constrained.struct_size=sizeof(constrained);
        constrained.runtime=config;
        constrained.joint_equality_payload_path=equalityPathString.c_str();
        constrained.expected_joint_equality_fingerprint=equalityFingerprint(equalityBytes);
        const auto rejectEquality = [&](const mrnx_runtime_config_v4& candidate,
                                        const mrnx_runtime_status_v1 expected) {
            mrnx_runtime_info_v1 rejected{};
            auto* unexpected=mrnx_bridge_v1_runtime_create_v4(&candidate,&rejected);
            require(unexpected==nullptr && rejected.status==expected,
                    "invalid NHEQ2 source was admitted or misclassified");
        };
        auto invalid=constrained;
        invalid.abi_version=3u;
        rejectEquality(invalid,MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        invalid=constrained;invalid.runtime.struct_size-=8u;
        rejectEquality(invalid,MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        invalid=constrained;invalid.expected_joint_equality_fingerprint^=1u;
        rejectEquality(invalid,MRNX_RUNTIME_ASSET_FAILURE_V1);
        for (const std::size_t offset : {std::size_t(4u),std::size_t(16u),std::size_t(32u),std::size_t(48u)}) {
            auto bytes=equalityBytes;bytes[offset]^=1u;writeEquality(bytes);
            invalid=constrained;invalid.expected_joint_equality_fingerprint=equalityFingerprint(bytes);
            rejectEquality(invalid,MRNX_RUNTIME_ASSET_FAILURE_V1);
        }
        auto truncated=equalityBytes;truncated.pop_back();writeEquality(truncated);
        invalid=constrained;invalid.expected_joint_equality_fingerprint=equalityFingerprint(truncated);
        rejectEquality(invalid,MRNX_RUNTIME_ASSET_FAILURE_V1);
        writeEquality(equalityBytes);
        runtime=mrnx_bridge_v1_runtime_create_v4(&constrained,&info);
        require(runtime != nullptr, "valid source-compliant authored world was rejected");
        require(info.model_source_fingerprint == constrainedFingerprint(
                    config.expected_model_source_fingerprint,equalityBytes),
                "NHEQ2 source is absent from Human program identity");
        std::filesystem::remove(equalityTemporary);
        std::printf("numanx_source_equalities_admission=pass rows=51 negative_cases=8 source_fp=%016llx\n",
                    static_cast<unsigned long long>(info.model_source_fingerprint));
    } else {
        runtime=mrnx_bridge_v1_runtime_create_v3(&config,&info);
    }
    require(runtime != nullptr, "valid authored world was rejected");
    mrnx_runtime_world_info_v1 worldInfo{};
    worldInfo.abi_version = MRNX_BRIDGE_ABI_V1;
    worldInfo.struct_size = sizeof(worldInfo);
    require(mrnx_bridge_v1_runtime_copy_world_info(runtime, &worldInfo) &&
        worldInfo.authored_package == 1u && worldInfo.object_count == 3u &&
        worldInfo.fem_node_count == 12u && worldInfo.fem_attachment_count == 12u &&
        worldInfo.world_fingerprint == config.expected_matter_world_fingerprint &&
        worldInfo.physics_fingerprint != 0u,
        "authored world metadata does not identify the loaded package");
    // Rewriting/removing the file after construction must not change retained
    // runtime state or its proof authority.
    std::filesystem::remove(package);
    if (const char* retained = std::getenv("MRNX_AUTHORED_FIXTURE_DIR")) {
        const std::filesystem::path directory(retained);
        std::filesystem::create_directories(directory);
        source.frameTimestep = 0.0001;
        numi::matter::CompileOptions options;
        options.maximumRateExponent = 0u;
        const auto compiled = numi::matter::compileWorld(source, options);
        std::string error;
        require(compiled.succeeded() && numi::matter::writePackage(compiled,
                    directory / "authored-100us.nmatterpack", &error),
                "retaining authored Swift fixture failed");
        std::ofstream receipt(directory / "authored-100us.json");
        receipt << "{\"human_source_fp\":\"" << std::hex
                << config.expected_model_source_fingerprint
                << "\",\"matter_world_fp\":\"" << compiled.world.fingerprint
                << "\",\"timestep_microseconds\":100}\n";
        require(receipt.good(), "retaining authored fixture identity failed");
    }
    std::printf("numanx_authored_world_admission=pass negative_cases=12 objects=3 attachments=12\n");
    return runtime;
}

// Exact-clock qualification uses the complete authored v7 construction
// contract. The package/state pair is generated by either exact-nanosecond
// fixture authoring mode with the same immutable source payloads supplied
// here; no legacy microsecond field is allowed to participate in admission.
mrnx_runtime_v1* makeExactRuntime(
    const mrnx_runtime_config_v2& base,
    mrnx_runtime_info_v1& info
) {
    const char* packagePath = std::getenv("MRNX_EXACT_WORLD");
    const char* initialPath = std::getenv("MRNX_EXACT_INITIAL_STATE");
    const char* supportPath = std::getenv("MRNX_EXACT_SUPPORT_CONTACT");
    const char* equalityPath = std::getenv("MRNX_EXACT_JOINT_EQUALITIES");
    const char* limitPath = std::getenv("MRNX_EXACT_JOINT_LIMITS");
    require(packagePath && initialPath && supportPath && equalityPath && limitPath,
        "exact clock needs MRNX_EXACT_WORLD/INITIAL_STATE/SUPPORT_CONTACT/JOINT_EQUALITIES/JOINT_LIMITS");
    numi::matter::CompiledWorld world;
    std::string error;
    require(numi::matter::readPackage(packagePath, world, nullptr, &error),
        "exact authored package could not be read");
    const auto rigid = readPayloadBytes(MRNX_FULLBODY_RIGID);
    const auto muscle = readPayloadBytes(MRNX_FULLBODY_MUSCLE);
    const auto support = readPayloadBytes(supportPath);
    const auto equality = readPayloadBytes(equalityPath);
    const auto limits = readPayloadBytes(limitPath);
    const auto initial = readPayloadBytes(initialPath);
    const auto baseSource = fullBodySourceFingerprint(rigid, muscle, support);
    auto source = constrainedFingerprint(baseSource, equality);
    source = metalrobo::numiHumanRuntimeAppendPayloadOwner(
        source, "NHLIM1", equalityFingerprint(limits));
    const auto initialFingerprint = equalityFingerprint(initial);
    source = metalrobo::numiHumanRuntimeAppendInitialState(
        source, initialFingerprint);

    mrnx_runtime_config_v8 config{};
    config.abi_version = MRNX_RUNTIME_CONFIG_ABI_V8;
    config.struct_size = sizeof(config);
    config.timestep_nanoseconds = 12'500u;
    auto& v7 = config.runtime;
    v7.abi_version = MRNX_RUNTIME_CONFIG_ABI_V7;
    v7.struct_size = sizeof(v7);
    v7.initial_state_payload_path = initialPath;
    v7.expected_initial_state_fingerprint = equalityFingerprint(initial);
    auto& v6 = v7.runtime;
    v6.abi_version = MRNX_RUNTIME_CONFIG_ABI_V6;
    v6.struct_size = sizeof(v6);
    v6.joint_limit_payload_path = limitPath;
    v6.expected_joint_limit_fingerprint = equalityFingerprint(limits);
    auto& v4 = v6.runtime;
    v4.abi_version = MRNX_RUNTIME_CONFIG_ABI_V4;
    v4.struct_size = sizeof(v4);
    v4.joint_equality_payload_path = equalityPath;
    v4.expected_joint_equality_fingerprint = equalityFingerprint(equality);
    auto& v3 = v4.runtime;
    v3.abi_version = MRNX_RUNTIME_CONFIG_ABI_V3;
    v3.struct_size = sizeof(v3);
    v3.runtime = base;
    v3.runtime.timestep_microseconds = 0u;
    v3.runtime.support_contact_payload_path = supportPath;
    v3.runtime.matter_material_path = nullptr;
    v3.matter_world_package_path = packagePath;
    // Authored-world admission precedes the optional NHEQ2/NHLIM1 owners;
    // the base Human source is therefore the v3 compatibility key.
    v3.expected_model_source_fingerprint = baseSource;
    v3.expected_matter_world_fingerprint = world.fingerprint;
    auto* runtime = mrnx_bridge_v1_runtime_create_v8(&config, &info);
    require(runtime != nullptr && info.status == MRNX_RUNTIME_READY_V1,
        "exact authored v8 runtime was rejected");
    require(info.model_source_fingerprint == source,
        "exact initial-state source identity is not retained");
    return runtime;
}

std::uint64_t costalFingerprint(std::uint64_t source, const std::uint64_t world) {
    for(const auto byte:std::array<unsigned char,8>{'N','H','T','M','A','S','S','1'}) {
        source^=byte;source*=kFnvPrime;
    }
    mixU64(source,equalityFingerprint(readPayloadBytes(std::getenv("MRNX_COSTAL_BINDING"))));
    mixU64(source,world);return source;
}

mrnx_runtime_v1* makeCostalRuntime(const mrnx_runtime_config_v2& base,
                                  mrnx_runtime_info_v1& info) {
    const char* package=std::getenv("MRNX_COSTAL_WORLD");
    const char* binding=std::getenv("MRNX_COSTAL_BINDING");
    const char* cartilage=std::getenv("MRNX_COSTAL_PAYLOAD");
    const char* equality=std::getenv("MRNX_JOINT_EQUALITIES");
    require(package&&binding&&cartilage&&equality,"costal runtime needs MRNX_COSTAL_WORLD/BINDING/PAYLOAD and MRNX_JOINT_EQUALITIES");
    numi::matter::CompiledWorld world;std::string error;
    require(numi::matter::readPackage(package,world,nullptr,&error),"could not read registered costal package");
    mrnx_runtime_config_v5 config{};
    config.abi_version=MRNX_RUNTIME_CONFIG_ABI_V5;config.struct_size=sizeof(config);
    config.costal_cartilage_payload_path=cartilage;config.costal_binding_payload_path=binding;
    config.expected_costal_binding_fingerprint=equalityFingerprint(readPayloadBytes(binding));
    auto& v4=config.runtime;v4.abi_version=MRNX_RUNTIME_CONFIG_ABI_V4;v4.struct_size=sizeof(v4);
    v4.joint_equality_payload_path=equality;v4.expected_joint_equality_fingerprint=equalityFingerprint(readPayloadBytes(equality));
    auto& v3=v4.runtime;v3.abi_version=MRNX_RUNTIME_CONFIG_ABI_V3;v3.struct_size=sizeof(v3);
    v3.runtime=base;v3.runtime.matter_material_path=nullptr;v3.matter_world_package_path=package;
    v3.expected_matter_world_fingerprint=world.fingerprint;
    v3.expected_model_source_fingerprint=fullBodySourceFingerprint(readPayloadBytes(MRNX_FULLBODY_RIGID),
        readPayloadBytes(MRNX_FULLBODY_MUSCLE),readPayloadBytes(MRNX_FULLBODY_SUPPORT_CONTACT));
    const auto reject=[&](const mrnx_runtime_config_v5& bad,const mrnx_runtime_status_v1 status) {
        mrnx_runtime_info_v1 failed{};
        auto* result=mrnx_bridge_v1_runtime_create_v5(&bad,&failed);
        require(result==nullptr&&failed.status==status,"invalid costal mass ownership was admitted");
    };
    auto bad=config;bad.struct_size-=8;reject(bad,MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
    bad=config;bad.expected_costal_binding_fingerprint^=1;reject(bad,MRNX_RUNTIME_ASSET_FAILURE_V1);
    bad=config;bad.costal_cartilage_payload_path=MRNX_FULLBODY_RIGID;reject(bad,MRNX_RUNTIME_ASSET_FAILURE_V1);
    // A legacy v4 cannot silently add the tissue mass; it sees incompatible
    // local frames and must reject before persistent runtime allocation.
    mrnx_runtime_info_v1 legacy{};
    require(mrnx_bridge_v1_runtime_create_v4(&v4,&legacy)==nullptr&&legacy.status==MRNX_RUNTIME_ASSET_FAILURE_V1,
            "legacy runtime admitted a mass-partitioned tissue package");
    auto* result=mrnx_bridge_v1_runtime_create_v5(&config,&info);
    require(result!=nullptr,"registered costal mass runtime was rejected");
    require(info.model_source_fingerprint==costalFingerprint(constrainedFingerprint(
        v3.expected_model_source_fingerprint,readPayloadBytes(equality)),world.fingerprint),
        "tissue mass ownership is absent from runtime identity");
    std::printf("numanx_costal_mass_admission=pass nodes=%zu tets=%zu attachments=%zu negative_cases=4 source_fp=%016llx\n",
        world.fem.nodes.size(),world.fem.tetrahedra.size(),world.fem.humanAttachments.size(),
        static_cast<unsigned long long>(info.model_source_fingerprint));
    return result;
}

int run(const bool authored, const bool sourceEqualities, const bool costalTissue,
        const bool exactClock = false) {
    const std::uint64_t durationMicros=costalTissue?10u:kDurationMicros;
    const std::uint64_t startTimestamp = exactClock ? 1'000'000u : kStartMicros;
    const std::uint64_t durationTicks = exactClock ? 12'500u : durationMicros;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        qualifyHumanSupportKKT(device);
        qualifyHumanSupportKKT(device, 1u);
        qualifyHumanSupportKKT(device, 2u);
        const auto culturePack = metalrobo::makePotterReferenceCulture(
            1000u, 50000u, 2056u);
        metalrobo::CompiledNeuronCulture compiledCulture;
        require(metalrobo::compileNeuronCulture(
                    culturePack, compiledCulture).succeeded(),
                "full-body culture pack did not compile");
        const auto culturePath = std::filesystem::temp_directory_path() /
            ("numanx-fullbody-culture-" + std::to_string(getpid()) +
             ".nculture");
        require(metalrobo::writeCompiledNeuronCulture(
                    compiledCulture, culturePath).succeeded(),
                "full-body culture artifact publication failed");
        const auto culturePathString = culturePath.string();
        mrnx_runtime_config_v2 config{};
        config.abi_version = MRNX_RUNTIME_CONFIG_ABI_V2;
        config.struct_size = sizeof(config);
        config.metal_device = (__bridge void*)device;
        config.rigid_payload_path = MRNX_FULLBODY_RIGID;
        config.muscle_payload_path = MRNX_FULLBODY_MUSCLE;
        config.support_contact_payload_path = MRNX_FULLBODY_SUPPORT_CONTACT;
        config.visual_pack_path = MRNX_FULLBODY_VISUAL_PACK;
        config.vision_profile_path = MRNX_FULLBODY_VISION_PROFILE;
        config.metalrobo_metallib_path = MRNX_METALROBO_METALLIB;
        config.matter_metallib_path = MRNX_MATTER_METALLIB;
        config.matter_material_path = MRNX_MATTER_MATERIAL;
        config.timestep_microseconds = exactClock ? 0u : durationMicros;
        config.maximum_retained_bytes = (costalTissue?2ull:1ull) * 1024ull * 1024ull * 1024ull;
        config.transaction_slot_count = 2u;
        config.culture_pack_path = culturePathString.c_str();
        config.culture_window_ticks = 100u;
        config.culture_current_per_newton = 4.0f;
        if (!exactClock) {
            auto mismatchedSupport = config;
            mismatchedSupport.support_contact_payload_path = MRNX_FULLBODY_MUSCLE;
            mrnx_runtime_info_v1 mismatchedInfo{};
            mismatchedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            mismatchedInfo.struct_size = sizeof(mismatchedInfo);
            mrnx_runtime_v1* mismatchedRuntime =
                mrnx_bridge_v1_runtime_create_v2(
                    &mismatchedSupport, &mismatchedInfo);
            require(
                mismatchedRuntime == nullptr &&
                    mismatchedInfo.status == MRNX_RUNTIME_ASSET_FAILURE_V1,
                "mismatched source support-contact authority was admitted");
        }
        mrnx_runtime_info_v1 info{};
        mrnx_runtime_v1* runtime = exactClock ? makeExactRuntime(config, info) :
            costalTissue ? makeCostalRuntime(config,info) : authored
            ? makeAuthoredRuntime(config, info, sourceEqualities)
            : mrnx_bridge_v1_runtime_create_v2(&config, &info);
        if (runtime == nullptr || info.status != MRNX_RUNTIME_READY_V1) {
            std::fprintf(
                stderr, "full-body runtime status=%u\n", info.status);
        }
        require(runtime != nullptr && info.status == MRNX_RUNTIME_READY_V1,
                "full-body runtime creation failed");
        if (exactClock) {
            mrnx_exact_clock_info_v1 clockInfo{};
            clockInfo.abi_version = MRNX_EXACT_CLOCK_INFO_ABI_V1;
            clockInfo.struct_size = sizeof(clockInfo);
            require(mrnx_bridge_v1_runtime_copy_exact_clock(runtime, &clockInfo) &&
                    clockInfo.timestep_nanoseconds == 12'500u &&
                    clockInfo.clock_quantum_nanoseconds == 1u &&
                    clockInfo.published_timestamp_nanoseconds == 0u &&
                    clockInfo.publication_epoch == 0u,
                "exact clock admission did not retain the canonical 12.5 us timeline");
        }
        mrnx_runtime_world_info_v1 worldInfo{};
        worldInfo.abi_version = MRNX_BRIDGE_ABI_V1;
        worldInfo.struct_size = sizeof(worldInfo);
        require(mrnx_bridge_v1_runtime_copy_world_info(runtime, &worldInfo) &&
                    worldInfo.authored_package == ((authored || exactClock) ? 1u : 0u),
                "runtime world kind was not reported exactly");
        require(info.q_coordinate_count == 129u && info.dof_count == 128u &&
                    info.muscle_count == 416u && info.body_count == 157u &&
                    info.accepted_state_proof_program_fingerprint != 0u,
                "full-body runtime provenance is wrong");
        const auto rigidPayload = readPayloadBytes(MRNX_FULLBODY_RIGID);
        const auto musclePayload = readPayloadBytes(MRNX_FULLBODY_MUSCLE);
        const auto supportPayload =
            readPayloadBytes(MRNX_FULLBODY_SUPPORT_CONTACT);
        auto expectedSource=sourceEqualities ?
            constrainedFingerprint(fullBodySourceFingerprint(rigidPayload,musclePayload,supportPayload),
                readPayloadBytes(std::getenv("MRNX_JOINT_EQUALITIES"))) :
            fullBodySourceFingerprint(rigidPayload,musclePayload,supportPayload);
        if(costalTissue)expectedSource=costalFingerprint(expectedSource,worldInfo.world_fingerprint);
        if (exactClock) {
            const auto supportExact = readPayloadBytes(std::getenv("MRNX_EXACT_SUPPORT_CONTACT"));
            const auto equalityExact = readPayloadBytes(std::getenv("MRNX_EXACT_JOINT_EQUALITIES"));
            const auto limitsExact = readPayloadBytes(std::getenv("MRNX_EXACT_JOINT_LIMITS"));
            const auto initialExact = readPayloadBytes(std::getenv("MRNX_EXACT_INITIAL_STATE"));
            expectedSource = constrainedFingerprint(
                fullBodySourceFingerprint(rigidPayload, musclePayload, supportExact),
                equalityExact);
            expectedSource = metalrobo::numiHumanRuntimeAppendPayloadOwner(
                expectedSource,"NHLIM1",equalityFingerprint(limitsExact));
            const auto initialFingerprint = equalityFingerprint(initialExact);
            expectedSource = metalrobo::numiHumanRuntimeAppendInitialState(
                expectedSource,initialFingerprint);
        }
        require(
            info.model_source_fingerprint == expectedSource,
            "runtime model fingerprint does not bind exact source payloads");
        auto mutatedSupportPayload = supportPayload;
        require(
            mutatedSupportPayload.size() > 56u,
            "support payload is too small for its NHCNT1 header");
        mutatedSupportPayload[56u] ^= 1u;
        require(
            info.model_source_fingerprint != fullBodySourceFingerprint(
                rigidPayload, musclePayload, mutatedSupportPayload),
            "support-contact payload mutation did not change model identity");

        id<MTLBuffer> headerBuffer = [device
            newBufferWithLength:sizeof(MRNumanXBrainMotorOutputHeaderGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> excitationBuffer = [device
            newBufferWithLength:416u * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> autonomicBuffer = [device
            newBufferWithLength:MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> activeBuffer = [device
            newBufferWithLength:
                MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gateBuffer = [device
            newBufferWithLength:sizeof(MRNumanXBrainMotorReadyGateGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLSharedEvent> readyEvent = [device newSharedEvent];
        require(headerBuffer != nil && excitationBuffer != nil &&
                    autonomicBuffer != nil && activeBuffer != nil &&
                    gateBuffer != nil && readyEvent != nil,
                "fixture Metal resources unavailable");
        auto* excitation = static_cast<float*>(excitationBuffer.contents);
        for (std::uint32_t index = 0u; index < 416u; ++index) {
            excitation[index] = 0.05f +
                0.1f * static_cast<float>(index % 7u) / 6.0f;
        }
        std::memset(autonomicBuffer.contents, 0, autonomicBuffer.length);
        std::memset(activeBuffer.contents, 0, activeBuffer.length);

        mrnx_physical_root_request_v1 request{};
        request.abi_version = MRNX_BRIDGE_ABI_V1;
        request.struct_size = sizeof(request);
        request.root.format_version =
            MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION;
        request.root.environment_identifier = 0u;
        request.root.episode_identifier = 1u;
        request.root.control_step_identifier = 1u;
        request.root.parameter_version_fingerprint = 2u;
        request.root.base_brain_generation = 0u;
        request.root.base_physics_generation = 0u;
        request.root.committed_timestamp_microseconds = startTimestamp;
        request.root.target_timestamp_microseconds =
            startTimestamp + durationTicks;
        request.root.shadow_generation = 1u;
        request.root.random_counter_generation = 3u;
        MRNumanXBrainJointTransactionToken nativeRoot{};
        std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
        request.root.transaction_fingerprint =
            metalrobo::metalNumanXBrainJointTransactionFingerprint(nativeRoot);

        request.substep.transaction_fingerprint =
            request.root.transaction_fingerprint;
        request.substep.substep_index = 0u;
        request.substep.attempt_index = 0u;
        request.substep.start_timestamp_microseconds = startTimestamp;
        request.substep.duration_microseconds = durationTicks;
        request.substep.candidate_timestamp_microseconds =
            startTimestamp + durationTicks;
        request.substep.shadow_generation = request.root.shadow_generation;
        request.substep.random_counter_generation =
            request.root.random_counter_generation;
        MRNumanXBrainJointSubstepToken nativeSubstep{};
        std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
        request.substep.substep_fingerprint =
            metalrobo::metalNumanXBrainJointSubstepFingerprint(nativeSubstep);

        request.candidate.format_version =
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION;
        request.candidate.flags =
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        request.candidate.transaction_fingerprint =
            request.root.transaction_fingerprint;
        request.candidate.substep_fingerprint =
            request.substep.substep_fingerprint;
        request.candidate.accepted_brain_timestamp_microseconds =
            startTimestamp;
        request.candidate.brain_generation = request.root.shadow_generation;
        request.candidate.motor_profile_fingerprint = 4u;
        request.candidate.motor_output_header_gpu_address =
            headerBuffer.gpuAddress;
        request.candidate.muscle_excitation_gpu_address =
            excitationBuffer.gpuAddress;
        request.candidate.random_counter_generation =
            request.root.random_counter_generation;
        request.candidate.motor_output_header_byte_count =
            headerBuffer.length;
        request.candidate.muscle_excitation_byte_count =
            excitationBuffer.length;
        request.candidate.muscle_count = 416u;
        request.candidate.environment_identifier = 0u;
        request.candidate.autonomic_command_gpu_address =
            autonomicBuffer.gpuAddress;
        request.candidate.autonomic_command_byte_count =
            autonomicBuffer.length;
        request.candidate.autonomic_command_count = 1u;
        request.candidate.active_sensing_command_gpu_address =
            activeBuffer.gpuAddress;
        request.candidate.active_sensing_command_byte_count =
            activeBuffer.length;
        request.candidate.active_sensing_command_count = 1u;
        request.candidate.actuator_command_kind =
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
        request.candidate.species_template_fingerprint = 5u;
        request.candidate.compiled_species_template_fingerprint = 6u;
        MRNumanXBrainMotorCandidate nativeCandidate{};
        std::memcpy(
            &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
        request.candidate.candidate_fingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                nativeCandidate);

        mrnx_physical_root_request_v3 exactRequest{};
        MRNumanXBrainJointTransactionTokenV2 nativeExactRoot{};
        MRNumanXBrainJointSubstepTokenV2 nativeExactSubstep{};
        MRNumanXBrainMotorCandidateV2 nativeExactCandidate{};
        if (exactClock) {
            exactRequest.abi_version = MRNX_PHYSICAL_ROOT_REQUEST_ABI_V3;
            exactRequest.struct_size = sizeof(exactRequest);
            exactRequest.root.format_version =
                MRNX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
            exactRequest.root.environment_identifier = 0u;
            exactRequest.root.episode_identifier = 1u;
            exactRequest.root.control_step_identifier = 1u;
            exactRequest.root.parameter_version_fingerprint = 2u;
            exactRequest.root.base_brain_generation = 0u;
            exactRequest.root.base_physics_generation = 0u;
            exactRequest.root.committed_timestamp_nanoseconds =
                startTimestamp;
            exactRequest.root.target_timestamp_nanoseconds =
                startTimestamp + durationTicks;
            exactRequest.root.shadow_generation = 1u;
            exactRequest.root.random_counter_generation = 3u;
            exactRequest.root.clock_domain =
                MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
            exactRequest.root.clock_quantum_nanoseconds =
                MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
            std::memcpy(
                &nativeExactRoot, &exactRequest.root,
                sizeof(nativeExactRoot));
            exactRequest.root.transaction_fingerprint =
                metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(
                    nativeExactRoot);

            exactRequest.substep.transaction_fingerprint =
                exactRequest.root.transaction_fingerprint;
            exactRequest.substep.substep_index = 0u;
            exactRequest.substep.attempt_index = 0u;
            exactRequest.substep.start_timestamp_nanoseconds = startTimestamp;
            exactRequest.substep.duration_nanoseconds = durationTicks;
            exactRequest.substep.candidate_timestamp_nanoseconds =
                startTimestamp + durationTicks;
            exactRequest.substep.shadow_generation =
                exactRequest.root.shadow_generation;
            exactRequest.substep.random_counter_generation =
                exactRequest.root.random_counter_generation;
            exactRequest.substep.clock_domain =
                exactRequest.root.clock_domain;
            exactRequest.substep.clock_quantum_nanoseconds =
                exactRequest.root.clock_quantum_nanoseconds;
            std::memcpy(
                &nativeExactSubstep, &exactRequest.substep,
                sizeof(nativeExactSubstep));
            exactRequest.substep.substep_fingerprint =
                metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(
                    nativeExactSubstep);

            exactRequest.candidate.format_version =
                MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
            exactRequest.candidate.flags =
                MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
                MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
            exactRequest.candidate.transaction_fingerprint =
                exactRequest.root.transaction_fingerprint;
            exactRequest.candidate.substep_fingerprint =
                exactRequest.substep.substep_fingerprint;
            exactRequest.candidate.accepted_brain_timestamp_nanoseconds =
                startTimestamp;
            exactRequest.candidate.brain_generation =
                exactRequest.root.shadow_generation;
            exactRequest.candidate.motor_profile_fingerprint = 4u;
            exactRequest.candidate.motor_output_header_gpu_address =
                headerBuffer.gpuAddress;
            exactRequest.candidate.muscle_excitation_gpu_address =
                excitationBuffer.gpuAddress;
            exactRequest.candidate.random_counter_generation =
                exactRequest.root.random_counter_generation;
            exactRequest.candidate.motor_output_header_byte_count =
                headerBuffer.length;
            exactRequest.candidate.muscle_excitation_byte_count =
                excitationBuffer.length;
            exactRequest.candidate.muscle_count = 416u;
            exactRequest.candidate.environment_identifier = 0u;
            exactRequest.candidate.autonomic_command_gpu_address =
                autonomicBuffer.gpuAddress;
            exactRequest.candidate.autonomic_command_byte_count =
                autonomicBuffer.length;
            exactRequest.candidate.autonomic_command_count = 1u;
            exactRequest.candidate.active_sensing_command_gpu_address =
                activeBuffer.gpuAddress;
            exactRequest.candidate.active_sensing_command_byte_count =
                activeBuffer.length;
            exactRequest.candidate.active_sensing_command_count = 1u;
            exactRequest.candidate.actuator_command_kind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            exactRequest.candidate.clock_domain =
                MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
            exactRequest.candidate.species_template_fingerprint = 5u;
            exactRequest.candidate.compiled_species_template_fingerprint = 6u;
            std::memcpy(
                &nativeExactCandidate, &exactRequest.candidate,
                sizeof(nativeExactCandidate));
            exactRequest.candidate.candidate_fingerprint =
                metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
                    nativeExactCandidate);
            require(
                exactRequest.root.transaction_fingerprint !=
                    request.root.transaction_fingerprint &&
                exactRequest.substep.substep_fingerprint !=
                    request.substep.substep_fingerprint &&
                exactRequest.candidate.candidate_fingerprint !=
                    request.candidate.candidate_fingerprint,
                "exact-clock fingerprints were not domain-separated from v1");
        }

        auto* header = static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
            headerBuffer.contents);
        auto* exactHeader =
            static_cast<MRNumanXBrainMotorOutputHeaderGPUV2*>(
                headerBuffer.contents);
        if (exactClock) {
            *exactHeader = {};
            exactHeader->formatVersion =
                MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
            exactHeader->flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
            exactHeader->timestampNanoseconds = startTimestamp;
            exactHeader->brainGeneration =
                exactRequest.root.shadow_generation;
            exactHeader->profileFingerprint =
                exactRequest.candidate.motor_profile_fingerprint;
            exactHeader->protectiveCommandFingerprint = 7u;
            exactHeader->muscleCount = 416u;
            exactHeader->environmentIdentifier = 0u;
            exactHeader->motorInhibition = 0.1f;
            exactHeader->autonomicArousal = 0.2f;
            exactHeader->actuatorCommandKind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            exactHeader->clockDomain =
                MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
            exactHeader->outputMinimum = 0.0f;
            exactHeader->outputMaximum = 1.0f;
            exactHeader->outputFingerprint =
                metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
                    *exactHeader, excitation, 416u);
        } else {
            *header = {};
            header->formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION;
            header->flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
            header->timestampMicroseconds = startTimestamp;
            header->brainGeneration = request.root.shadow_generation;
            header->profileFingerprint =
                request.candidate.motor_profile_fingerprint;
            header->protectiveCommandFingerprint = 7u;
            header->muscleCount = 416u;
            header->environmentIdentifier = 0u;
            header->motorInhibition = 0.1f;
            header->autonomicArousal = 0.2f;
            header->actuatorCommandKind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            header->outputMinimum = 0.0f;
            header->outputMaximum = 1.0f;
            header->outputFingerprint = motorOutputFingerprint(
                *header, excitation);
        }

        auto* gate = static_cast<MRNumanXBrainMotorReadyGateGPU*>(
            gateBuffer.contents);
        auto* exactGate = static_cast<MRNumanXBrainMotorReadyGateGPUV2*>(
            gateBuffer.contents);
        if (exactClock) {
            *exactGate = {};
            exactGate->abiVersion =
                MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
            exactGate->structBytes = sizeof(*exactGate);
            exactGate->status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
            exactGate->environment = 0u;
            exactGate->substepIndex = 0u;
            exactGate->attemptIndex = 0u;
            exactGate->muscleCount = 416u;
            exactGate->actuatorCommandKind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            exactGate->controlStep =
                exactRequest.root.control_step_identifier;
            exactGate->transactionFingerprint =
                exactRequest.root.transaction_fingerprint;
            exactGate->substepFingerprint =
                exactRequest.substep.substep_fingerprint;
            exactGate->candidateFingerprint =
                exactRequest.candidate.candidate_fingerprint;
            exactGate->motorOutputFingerprint =
                exactHeader->outputFingerprint;
            exactGate->motorProfileFingerprint =
                exactRequest.candidate.motor_profile_fingerprint;
            exactGate->brainGeneration =
                exactRequest.root.shadow_generation;
            exactGate->acceptedBrainTimestampNanoseconds = startTimestamp;
            exactGate->randomCounterGeneration =
                exactRequest.root.random_counter_generation;
            exactGate->speciesTemplateFingerprint =
                exactRequest.candidate.species_template_fingerprint;
            exactGate->compiledSpeciesTemplateFingerprint =
                exactRequest.candidate.compiled_species_template_fingerprint;
            exactGate->brainProgramFingerprint = 8u;
            exactGate->fastProgramFingerprint = 9u;
            exactGate->decisionGateFingerprint = 10u;
            exactGate->clockDomain =
                MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
            exactGate->clockQuantumNanoseconds =
                MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
            exactGate->gateFingerprint =
                metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(
                    *exactGate);
        } else {
            *gate = {};
            gate->abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION;
            gate->structBytes = sizeof(*gate);
            gate->status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
            gate->environment = 0u;
            gate->substepIndex = 0u;
            gate->attemptIndex = 0u;
            gate->muscleCount = 416u;
            gate->actuatorCommandKind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            gate->controlStep = request.root.control_step_identifier;
            gate->transactionFingerprint =
                request.root.transaction_fingerprint;
            gate->substepFingerprint = request.substep.substep_fingerprint;
            gate->candidateFingerprint =
                request.candidate.candidate_fingerprint;
            gate->motorOutputFingerprint = header->outputFingerprint;
            gate->motorProfileFingerprint =
                request.candidate.motor_profile_fingerprint;
            gate->brainGeneration = request.root.shadow_generation;
            gate->acceptedBrainTimestampMicroseconds = startTimestamp;
            gate->randomCounterGeneration =
                request.root.random_counter_generation;
            gate->speciesTemplateFingerprint =
                request.candidate.species_template_fingerprint;
            gate->compiledSpeciesTemplateFingerprint =
                request.candidate.compiled_species_template_fingerprint;
            gate->brainProgramFingerprint = 8u;
            gate->fastProgramFingerprint = 9u;
            gate->decisionGateFingerprint = 10u;
            gate->gateFingerprint = readyGateFingerprint(*gate);
        }

        request.motor_header = range(
            headerBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.muscle_excitation = range(
            excitationBuffer, MRNX_ELEMENT_FLOAT32_V1, sizeof(float));
        request.autonomic_command = range(
            autonomicBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.active_sensing_command = range(
            activeBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.motor_ready_gate = range(
            gateBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.motor_ready.abi_version = MRNX_BRIDGE_ABI_V1;
        request.motor_ready.struct_size = sizeof(request.motor_ready);
        request.motor_ready.shared_event = (__bridge void*)readyEvent;
        request.motor_ready.value = 1u;
        request.motor_ready.device_registry_id = device.registryID;

        if (exactClock) {
            exactRequest.motor_header = range(
                headerBuffer,
                MRNX_ELEMENT_BRAIN_MOTOR_OUTPUT_HEADER_V2,
                sizeof(MRNumanXBrainMotorOutputHeaderGPUV2));
            exactRequest.muscle_excitation = request.muscle_excitation;
            exactRequest.autonomic_command = request.autonomic_command;
            exactRequest.active_sensing_command =
                request.active_sensing_command;
            exactRequest.motor_ready_gate = range(
                gateBuffer,
                MRNX_ELEMENT_BRAIN_MOTOR_READY_GATE_V2,
                sizeof(MRNumanXBrainMotorReadyGateGPUV2));
            exactRequest.motor_ready = request.motor_ready;
        }

        std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
        std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
        std::memcpy(
            &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
        require(request.root.transaction_fingerprint ==
                    metalrobo::metalNumanXBrainJointTransactionFingerprint(
                        nativeRoot),
                "fixture root fingerprint mismatch");
        require(request.substep.substep_fingerprint ==
                    metalrobo::metalNumanXBrainJointSubstepFingerprint(
                        nativeSubstep),
                "fixture substep fingerprint mismatch");
        require(request.candidate.candidate_fingerprint ==
                    metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                        nativeCandidate),
                "fixture candidate fingerprint mismatch");

        if (exactClock) {
            std::memcpy(
                &nativeExactRoot, &exactRequest.root,
                sizeof(nativeExactRoot));
            std::memcpy(
                &nativeExactSubstep, &exactRequest.substep,
                sizeof(nativeExactSubstep));
            std::memcpy(
                &nativeExactCandidate, &exactRequest.candidate,
                sizeof(nativeExactCandidate));
            require(
                metalrobo::metalNumanXBrainJointTransactionV2Valid(
                    nativeExactRoot) &&
                metalrobo::metalNumanXBrainJointSubstepV2Valid(
                    nativeExactRoot, nativeExactSubstep) &&
                metalrobo::metalNumanXBrainMotorCandidateV2Valid(
                    nativeExactRoot, nativeExactSubstep,
                    nativeExactCandidate) &&
                metalrobo::metalNumanXBrainMotorOutputV2Valid(
                    nativeExactCandidate, *exactHeader, excitation, 416u) &&
                metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                    nativeExactRoot, nativeExactSubstep,
                    nativeExactCandidate, *exactHeader, *exactGate),
                "coherent exact-clock v2 fixture failed CPU validation");
        }

        if (exactClock) {
            Completion rejected{};
            require(!mrnx_bridge_v1_runtime_begin_physical_root(
                        runtime, &request, &rejected, &settled),
                    "legacy physical-root entry admitted an exact-clock runtime");
            require(
                rejected.count.load(std::memory_order_acquire) == 0u &&
                    readyEvent.signaledValue == 0u,
                "legacy exact-runtime rejection invoked completion or event");

            mrnx_physical_root_request_v2 legacyExactRequest{};
            std::memcpy(
                &legacyExactRequest, &request, sizeof(legacyExactRequest));
            legacyExactRequest.abi_version =
                MRNX_PHYSICAL_ROOT_REQUEST_ABI_V2;
            legacyExactRequest.struct_size = sizeof(legacyExactRequest);
            legacyExactRequest.motor_header.metal_buffer =
                reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x201u));
            legacyExactRequest.motor_ready.shared_event =
                reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x202u));
            require(!mrnx_bridge_v1_runtime_begin_physical_root_v2(
                        runtime, &legacyExactRequest, &rejected, &settled),
                    "published all-v1 request-v2 entered an exact runtime");
            mrnx_runtime_info_v1 legacyV2Info{};
            legacyV2Info.abi_version = MRNX_BRIDGE_ABI_V1;
            legacyV2Info.struct_size = sizeof(legacyV2Info);
            require(
                mrnx_bridge_v1_runtime_copy_info(
                    runtime, &legacyV2Info) &&
                legacyV2Info.status == MRNX_RUNTIME_INVALID_REQUEST_V1 &&
                legacyV2Info.request_failure_stage ==
                    MRNX_REQUEST_FAILURE_STAGE_LEGACY_EXACT_V2_UNROUTABLE &&
                rejected.count.load(std::memory_order_acquire) == 0u &&
                readyEvent.signaledValue == 0u,
                "published request-v2 did not fail before resource admission");

            const auto requireV3RejectedAtStage = [
                &rejected, runtime, readyEvent
            ](
                const mrnx_physical_root_request_v3& rejectedRequest,
                const std::uint32_t expectedStage,
                const char* admittedMessage,
                const char* stageMessage
            ) {
                require(!mrnx_bridge_v1_runtime_begin_physical_root_v3(
                            runtime, &rejectedRequest, &rejected, &settled),
                        admittedMessage);
                mrnx_runtime_info_v1 rejectedInfo{};
                rejectedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
                rejectedInfo.struct_size = sizeof(rejectedInfo);
                require(
                    mrnx_bridge_v1_runtime_copy_info(
                        runtime, &rejectedInfo) &&
                    rejectedInfo.status == MRNX_RUNTIME_INVALID_REQUEST_V1 &&
                    rejectedInfo.request_failure_stage == expectedStage &&
                    rejected.count.load(std::memory_order_acquire) == 0u &&
                    readyEvent.signaledValue == 0u,
                    stageMessage);
            };

            auto overflowExact = exactRequest;
            overflowExact.root.control_step_identifier =
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::uint32_t>::max()) + 1u;
            requireV3RejectedAtStage(
                overflowExact, 1u,
                "UInt64 control-step overflow was admitted on exact clock",
                "control-step overflow missed request-v3 stage 1");

            auto legacyRoot = exactRequest;
            std::memcpy(
                &legacyRoot.root, &request.root, sizeof(legacyRoot.root));
            requireV3RejectedAtStage(
                legacyRoot, 10u,
                "request-v3 admitted a legacy joint root",
                "legacy joint root missed request-v3 stage 10");

            auto mixedSubstep = exactRequest;
            mixedSubstep.substep.clock_domain = 0u;
            std::memcpy(
                &nativeExactSubstep, &mixedSubstep.substep,
                sizeof(nativeExactSubstep));
            mixedSubstep.substep.substep_fingerprint =
                metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(
                    nativeExactSubstep);
            requireV3RejectedAtStage(
                mixedSubstep, 2u,
                "request-v3 admitted a mixed-domain substep",
                "mixed-domain substep missed request-v3 stage 2");

            auto legacyCandidate = exactRequest;
            legacyCandidate.candidate.format_version =
                MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V1;
            legacyCandidate.candidate.clock_domain = 0u;
            std::memcpy(
                &nativeExactCandidate, &legacyCandidate.candidate,
                sizeof(nativeExactCandidate));
            legacyCandidate.candidate.candidate_fingerprint =
                metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
                    nativeExactCandidate);
            requireV3RejectedAtStage(
                legacyCandidate, 3u,
                "request-v3 admitted a legacy motor candidate",
                "legacy motor candidate missed request-v3 stage 3");

            auto misalignedCandidate = exactRequest;
            misalignedCandidate.candidate.motor_output_header_gpu_address +=
                8u;
            std::memcpy(
                &nativeExactCandidate, &misalignedCandidate.candidate,
                sizeof(nativeExactCandidate));
            misalignedCandidate.candidate.candidate_fingerprint =
                metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
                    nativeExactCandidate);
            requireV3RejectedAtStage(
                misalignedCandidate, 3u,
                "request-v3 admitted a misaligned candidate motor header",
                "misaligned candidate missed request-v3 stage 3");

            auto legacyHeaderDeclaration = exactRequest;
            legacyHeaderDeclaration.motor_header.element_type =
                MRNX_ELEMENT_RAW_BYTES_V1;
            requireV3RejectedAtStage(
                legacyHeaderDeclaration, 41u,
                "request-v3 admitted a legacy motor-header declaration",
                "legacy motor header missed request-v3 stage 41");

            auto wrongHeaderElementSize = exactRequest;
            wrongHeaderElementSize.motor_header.element_byte_count = 1u;
            requireV3RejectedAtStage(
                wrongHeaderElementSize, 41u,
                "request-v3 admitted a scalar-sized typed motor header",
                "typed motor-header size missed request-v3 stage 41");

            auto misalignedHeaderAddress = exactRequest;
            misalignedHeaderAddress.motor_header.gpu_address += 8u;
            requireV3RejectedAtStage(
                misalignedHeaderAddress, 41u,
                "request-v3 admitted a misaligned motor-header address",
                "motor-header address alignment missed request-v3 stage 41");

            auto misalignedHeaderOffset = exactRequest;
            misalignedHeaderOffset.motor_header.byte_offset = 8u;
            requireV3RejectedAtStage(
                misalignedHeaderOffset, 41u,
                "request-v3 admitted a misaligned motor-header offset",
                "motor-header offset alignment missed request-v3 stage 41");

            auto legacyGateDeclaration = exactRequest;
            legacyGateDeclaration.motor_ready_gate.element_type =
                MRNX_ELEMENT_RAW_BYTES_V1;
            requireV3RejectedAtStage(
                legacyGateDeclaration, 45u,
                "request-v3 admitted a legacy ready-gate declaration",
                "legacy ready gate missed request-v3 stage 45");

            auto wrongGateElementSize = exactRequest;
            wrongGateElementSize.motor_ready_gate.element_byte_count = 1u;
            requireV3RejectedAtStage(
                wrongGateElementSize, 45u,
                "request-v3 admitted a scalar-sized typed ready gate",
                "typed ready-gate size missed request-v3 stage 45");

            auto misalignedGateAddress = exactRequest;
            misalignedGateAddress.motor_ready_gate.gpu_address += 8u;
            requireV3RejectedAtStage(
                misalignedGateAddress, 45u,
                "request-v3 admitted a misaligned ready-gate address",
                "ready-gate address alignment missed request-v3 stage 45");

            auto misalignedGateOffset = exactRequest;
            misalignedGateOffset.motor_ready_gate.byte_offset = 8u;
            requireV3RejectedAtStage(
                misalignedGateOffset, 45u,
                "request-v3 admitted a misaligned ready-gate offset",
                "ready-gate offset alignment missed request-v3 stage 45");

            mrnx_bridge_v1_runtime_drop(runtime);
            std::filesystem::remove(culturePath);
            std::printf(
                "numanx_fullbody_bridge_probe=pass clock=exact-nanoseconds "
                "timestep=12500ns scalar_negatives=validated mixed_v1=rejected "
                "execution_probe=numanx_exact_runtime_v3_lifecycle_probe "
                "root_fp=%llu substep_fp=%llu "
                "candidate_fp=%llu\n",
                static_cast<unsigned long long>(
                    exactRequest.root.transaction_fingerprint),
                static_cast<unsigned long long>(
                    exactRequest.substep.substep_fingerprint),
                static_cast<unsigned long long>(
                    exactRequest.candidate.candidate_fingerprint));
            return 0;
        }

        auto overflow = request;
        overflow.root.control_step_identifier =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) + 1u;
        require(!mrnx_bridge_v1_runtime_begin_physical_root(
                    runtime, &overflow, nullptr, &settled),
                "UInt64 control-step overflow was admitted");
        Completion completion{};
        const bool began = mrnx_bridge_v1_runtime_begin_physical_root(
            runtime, &request, &completion, &settled);
        if (!began) {
            mrnx_runtime_info_v1 failedInfo{};
            failedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            failedInfo.struct_size = sizeof(failedInfo);
            (void)mrnx_bridge_v1_runtime_copy_info(runtime, &failedInfo);
            throw std::runtime_error(
                "full-body physical root was not armed status=" +
                std::to_string(failedInfo.status) + " stage=" +
                std::to_string(failedInfo.request_failure_stage));
        }
        require(completion.count.load(std::memory_order_acquire) == 0u,
                "physical root ignored the unsignaled motor-ready event");
        readyEvent.signaledValue = 1u;
        waitForCompletion(completion,costalTissue?60u:10u);
        if (completion.status.load(std::memory_order_acquire) !=
                MRNX_COMPLETION_READY_V1 ||
            completion.prepared == nullptr || completion.candidate == nullptr ||
            completion.root.q_coordinate_count != 129u ||
            completion.root.dof_count != 128u) {
            mrnx_runtime_info_v1 failedInfo{};
            failedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            failedInfo.struct_size = sizeof(failedInfo);
            (void)mrnx_bridge_v1_runtime_copy_info(runtime, &failedInfo);
            throw std::runtime_error(
                "real full-body physical root did not prepare status=" +
                std::to_string(completion.status.load(
                    std::memory_order_acquire)) + " stage=" +
                std::to_string(failedInfo.request_failure_stage) +
                " prepared=" +
                std::to_string(completion.prepared != nullptr) +
                " candidate=" +
                std::to_string(completion.candidate != nullptr) +
                " nq=" +
                std::to_string(completion.root.q_coordinate_count) +
                " nv=" + std::to_string(completion.root.dof_count));
        }
        mrnx_wire_lease_v1 physicalGate{};
        physicalGate.abi_version = MRNX_BRIDGE_ABI_V1;
        physicalGate.struct_size = sizeof(physicalGate);
        require(mrnx_bridge_v1_prepared_copy_physical_gate(
                    completion.prepared, &physicalGate) &&
                    physicalGate.record.byte_count == 64u &&
                    physicalGate.ready.value != 0u,
                "prepared physical gate is unavailable");
        mrnx_culture_prepared_view_v1 culturePrepared{};
        culturePrepared.abi_version = MRNX_CULTURE_PREPARED_VIEW_ABI_V1;
        culturePrepared.struct_size = sizeof(culturePrepared);
        require(mrnx_bridge_v1_prepared_copy_culture_view(
                    completion.prepared, &culturePrepared) &&
                    culturePrepared.culture_fingerprint ==
                        compiledCulture.fingerprint() &&
                    culturePrepared.accepted_generation == 0u &&
                    culturePrepared.prepared_generation == 1u &&
                    culturePrepared.source_root_fingerprint ==
                        completion.root.transaction_fingerprint &&
                    culturePrepared.receipt_fingerprint != 0u &&
                    culturePrepared.ready.value != 0u,
                "prepared culture receipt is unavailable or stale");
        mrnx_candidate_view_v1 sensor{};
        sensor.abi_version = MRNX_BRIDGE_ABI_V1;
        sensor.struct_size = sizeof(sensor);
        mrnx_candidate_channel_v1 proprioception{};
        proprioception.abi_version = MRNX_BRIDGE_ABI_V1;
        proprioception.struct_size = sizeof(proprioception);
        mrnx_candidate_channel_v1 interoception{};
        interoception.abi_version = MRNX_BRIDGE_ABI_V1;
        interoception.struct_size = sizeof(interoception);
        mrnx_candidate_channel_v1 supplemental[5]{};
        for (auto& channel : supplemental) {
            channel.abi_version = MRNX_BRIDGE_ABI_V1;
            channel.struct_size = sizeof(channel);
        }
        mrnx_candidate_timing_v1 timing{};
        timing.abi_version = MRNX_BRIDGE_ABI_V1;
        timing.struct_size = sizeof(timing);
        require(mrnx_bridge_v1_candidate_copy_view(
                    completion.candidate, &sensor) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 0u, &proprioception) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 1u, &interoception) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 2u, &supplemental[0]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 3u, &supplemental[1]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 4u, &supplemental[2]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 5u, &supplemental[3]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 6u, &supplemental[4]) &&
                    mrnx_bridge_v1_candidate_copy_timing(
                        completion.candidate, &timing) &&
                    sensor.channel_count == 7u &&
                    sensor.accepted_brain_generation == 1u &&
                    sensor.key.sensor_generation == 1u &&
                    proprioception.modality ==
                        MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1 &&
                    proprioception.receptor_count == 416u &&
                    proprioception.feature_dimension == 10u &&
                    proprioception.receptor_timestamp_microseconds ==
                        startTimestamp &&
                    interoception.modality ==
                        MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1 &&
                    interoception.receptor_count == 416u &&
                    interoception.feature_dimension == 6u &&
                    interoception.receptor_timestamp_microseconds ==
                        startTimestamp &&
                    supplemental[0].modality ==
                        MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1 &&
                    supplemental[0].receptor_count == 128u &&
                    supplemental[0].feature_dimension == 7u &&
                    supplemental[1].modality ==
                        MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1 &&
                    supplemental[1].receptor_count == 1u &&
                    supplemental[1].feature_dimension == 22u &&
                    supplemental[2].modality ==
                        MRNX_CANDIDATE_MODALITY_AUDITION_V1 &&
                    supplemental[2].receptor_count == 24u &&
                    supplemental[2].feature_dimension == 8u &&
                    supplemental[3].modality ==
                        MRNX_CANDIDATE_MODALITY_VISION_V1 &&
                    supplemental[3].receptor_count == 64u * 48u &&
                    supplemental[3].feature_dimension == 8u &&
                    supplemental[4].modality ==
                        MRNX_CANDIDATE_MODALITY_TOUCH_V1 &&
                    supplemental[4].receptor_count == 10u &&
                    supplemental[4].feature_dimension == 7u &&
                    timing.capture_timestamp_microseconds == startTimestamp &&
                    timing.delivery_timestamp_microseconds ==
                        startTimestamp + durationTicks &&
                    timing.latency_microseconds == durationTicks &&
                    timing.sample_interval_microseconds == durationTicks &&
                    timing.timing_fingerprint == timingFingerprint(timing),
                "causal HumanIO candidate is not the exact full-body view");
        __unsafe_unretained id<MTLBuffer> touchValues =
            (__bridge id<MTLBuffer>)supplemental[4].values.metal_buffer;
        __unsafe_unretained id<MTLBuffer> touchValidity =
            (__bridge id<MTLBuffer>)supplemental[4].validity.metal_buffer;
        require(touchValues != nil && touchValidity != nil &&
                    supplemental[4].values.byte_count ==
                        10u * 7u * sizeof(float) &&
                    supplemental[4].validity.byte_count ==
                        10u * sizeof(std::uint32_t),
                "touch consequence buffers have the wrong exact layout");
        id<MTLBuffer> touchReadback = [device
            newBufferWithLength:supplemental[4].values.byte_count
                + supplemental[4].validity.byte_count +
                    sizeof(MRNumanXAcceptedPhysicsStateTokenGPU)
                         options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> readbackQueue = [device newCommandQueue];
        id<MTLCommandBuffer> readbackCommand =
            [readbackQueue commandBuffer];
        id<MTLBlitCommandEncoder> readback =
            [readbackCommand blitCommandEncoder];
        require(touchReadback != nil && readbackQueue != nil &&
                    readbackCommand != nil && readback != nil,
                "touch consequence diagnostic readback allocation failed");
        [readback copyFromBuffer:touchValues
                    sourceOffset:supplemental[4].values.byte_offset
                        toBuffer:touchReadback destinationOffset:0u
                            size:supplemental[4].values.byte_count];
        [readback copyFromBuffer:touchValidity
                    sourceOffset:supplemental[4].validity.byte_offset
                        toBuffer:touchReadback
               destinationOffset:supplemental[4].values.byte_count
                            size:supplemental[4].validity.byte_count];
        __unsafe_unretained id<MTLBuffer> physicalTokenBuffer =
            (__bridge id<MTLBuffer>)physicalGate.record.metal_buffer;
        [readback copyFromBuffer:physicalTokenBuffer
                    sourceOffset:physicalGate.record.byte_offset
                        toBuffer:touchReadback
               destinationOffset:supplemental[4].values.byte_count +
                    supplemental[4].validity.byte_count
                            size:sizeof(
                                MRNumanXAcceptedPhysicsStateTokenGPU)];
        [readback endEncoding];
        [readbackCommand commit];
        [readbackCommand waitUntilCompleted];
        require(readbackCommand.status == MTLCommandBufferStatusCompleted,
                "touch consequence diagnostic readback failed");
        const auto* touch = static_cast<const float*>(touchReadback.contents);
        const auto* validity = reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::uint8_t*>(touchReadback.contents) +
            supplemental[4].values.byte_count);
        const auto* acceptedToken =
            reinterpret_cast<const MRNumanXAcceptedPhysicsStateTokenGPU*>(
                static_cast<const std::uint8_t*>(touchReadback.contents) +
                supplemental[4].values.byte_count +
                supplemental[4].validity.byte_count);
        require(acceptedToken->transactionFingerprint ==
                    completion.root.transaction_fingerprint &&
                    acceptedToken->substepFingerprint != 0u &&
                    acceptedToken->physicsStateFingerprint != 0u &&
                    acceptedToken->physicsGeneration == 1u &&
                    acceptedToken->environmentIdentifier == 0u &&
                    acceptedToken->reserved == 0u &&
                    acceptedToken->tokenFingerprint != 0u,
                "accepted token did not bind the proved Human/Matter state");
        bool observedSupportGeometry = false;
        for (std::uint32_t row = 0u; row < 10u; ++row) {
            const float separation = touch[row * 7u + 3u];
            const float normalForce = touch[row * 7u + 4u];
            const float frictionForce = touch[row * 7u + 5u];
            const float slipSpeed = touch[row * 7u + 6u];
            const bool rowValid = validity[row] ==
                    MR_NUMANX_HUMAN_TOUCH_VALIDITY_ALL &&
                std::isfinite(separation) &&
                std::isfinite(normalForce) && normalForce >= 0.0f &&
                std::isfinite(frictionForce) && frictionForce >= 0.0f &&
                std::isfinite(slipSpeed) && slipSpeed >= 0.0f;
            if (!rowValid) {
                std::fprintf(stderr,
                    "support_row=%u validity=%u separation=%g normal=%g "
                    "friction=%g slip=%g\n", row, validity[row], separation,
                    normalForce, frictionForce, slipSpeed);
            }
            require(rowValid, "Matter support consequence is invalid");
            observedSupportGeometry = observedSupportGeometry ||
                std::abs(separation) < 0.1f;
        }
        require(observedSupportGeometry,
                "NHCNT rows produced no solver-owned support geometry");
        mrnx_aggregate_snapshot_v1 aggregate{};
        aggregate.abi_version = MRNX_BRIDGE_ABI_V1;
        aggregate.struct_size = sizeof(aggregate);
        require(!mrnx_bridge_v1_runtime_copy_aggregate_snapshot(
                    runtime, &aggregate),
                "unpublished full-body root escaped the aggregate reader");
        mrnx_aggregate_snapshot_v2 aggregateV2{};
        aggregateV2.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V2;
        aggregateV2.struct_size = sizeof(aggregateV2);
        mrnx_aggregate_snapshot_v3 aggregateV3{};
        aggregateV3.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V3;
        aggregateV3.struct_size = sizeof(aggregateV3);
        mrnx_aggregate_snapshot_v4 aggregateV4{};
        aggregateV4.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V4;
        aggregateV4.struct_size = sizeof(aggregateV4);
        require(!mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v2(
                    runtime, &aggregateV2) &&
                    !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v3(
                        runtime, &aggregateV3) &&
                    !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v4(
                        runtime, &aggregateV4),
                "unpublished root escaped an extensible aggregate reader");
        mrnx_publication_v2 exactPublication{};
        exactPublication.abi_version = MRNX_PUBLICATION_ABI_V2;
        exactPublication.struct_size = sizeof(exactPublication);
        GenerationLatchCapture exactLatch{};
        require(
            mrnx_bridge_v1_release_accepted_v2(
                completion.prepared, &exactPublication, &exactLatch,
                &generationLatch) == MRNX_PUBLICATION_REJECTED_V1 &&
                exactLatch.count.load(std::memory_order_acquire) == 0u,
            "legacy root admitted an exact publication operation");
        require(mrnx_bridge_v1_quarantine_timeout(completion.prepared),
                "prepared-only qualification did not quarantine on timeout");
        mrnx_bridge_v1_candidate_drop(completion.candidate);
        mrnx_bridge_v1_prepared_drop(completion.prepared);
        mrnx_bridge_v1_runtime_drop(runtime);
        std::filesystem::remove(culturePath);
        std::printf(
            "numanx_fullbody_bridge_probe=pass bodies=%u nq=%u nv=%u "
            "muscles=%u motor_wait=ordered physical=prepared "
            "sensor=unpublished cross_family_release=rejected "
            "timeout=quarantined clock=%s timestep=%s\n",
            info.body_count, info.q_coordinate_count, info.dof_count,
            info.muscle_count, exactClock ? "exact-nanoseconds" : "legacy-microseconds",
            exactClock ? "12500ns" : (std::to_string(durationMicros) + "us").c_str());
        return 0;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--support-newton-admission") {
            @autoreleasepool {
                qualifyHumanSupportNewtonAdmission(MTLCreateSystemDefaultDevice());
                return 0;
            }
        }
        if (argc==2 && std::string(argv[1])=="--touch-aggregation-only") {
            @autoreleasepool {qualifyTouchAggregation(MTLCreateSystemDefaultDevice());return 0;}
        }
        const std::string fixtureMode = argc >= 2 ? argv[1] : "";
        const bool certificateFixture =
            fixtureMode == "--prepared-stance-fixture" ||
            fixtureMode == "--prepared-stance-fixture-ns";
        const bool importedStateFixture =
            fixtureMode == "--prepared-state-fixture" ||
            fixtureMode == "--prepared-state-fixture-ns";
        if (certificateFixture || importedStateFixture) {
            require(
                argc==7 || argc==8 || argc==9 || argc==12,
                "fixture usage: MODE SOURCE OUTPUT CONTACTS EQUALITIES LIMITS "
                "[TIMESTEP [NEWTON [FGMRES_RESTART FGMRES_ITERATIONS "
                "RELATIVE_RESIDUAL]]]");
            const bool exactNs =
                fixtureMode == "--prepared-stance-fixture-ns" ||
                fixtureMode == "--prepared-state-fixture-ns";
            require(!exactNs || argc >= 8, "exact-ns fixture requires an explicit nanosecond timestep");
            std::uint64_t timestep = 100u;
            if (argc >= 8) {
                const std::string value(argv[7]);
                require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "prepared fixture timestep must be an unsigned integer");
                timestep = std::stoull(value);
            }
            std::uint32_t iterations = 16u;
            if (argc == 9 || argc == 12) {
                const std::string value(argv[8]);
                require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "Newton iteration budget must be an unsigned integer");
                const auto parsed = std::stoull(value);
                require(parsed > 0u && parsed <= 128u, "invalid Newton iteration budget");
                iterations = static_cast<std::uint32_t>(parsed);
            }
            std::uint32_t fgmresRestart = NM_MIXED_FGMRES_DEFAULT_RESTART;
            std::uint32_t fgmresIterations = NM_MIXED_FGMRES_ITERATIONS;
            double relativeResidual = 5.0e-3;
            if (argc == 12) {
                const auto parseBudget = [](const char* raw,
                                            const char* message) {
                    const std::string value(raw);
                    require(
                        !value.empty() &&
                            value.find_first_not_of("0123456789") ==
                                std::string::npos,
                        message);
                    const auto parsed = std::stoull(value);
                    require(
                        parsed > 0u &&
                            parsed <=
                                std::numeric_limits<std::uint32_t>::max(),
                        message);
                    return static_cast<std::uint32_t>(parsed);
                };
                fgmresRestart = parseBudget(
                    argv[9],
                    "FGMRES restart must be a positive unsigned integer");
                fgmresIterations = parseBudget(
                    argv[10],
                    "FGMRES iterations must be a positive unsigned integer");
                const std::string residualText(argv[11]);
                std::size_t residualEnd = 0u;
                relativeResidual = std::stod(residualText, &residualEnd);
                require(
                    residualEnd == residualText.size() &&
                        std::isfinite(relativeResidual) &&
                        relativeResidual > 0.0,
                    "relative residual must be finite and positive");
            }
            return writePreparedStanceFixture(argv[2],argv[3],argv[4],argv[5],argv[6],exactNs?0u:timestep,
                importedStateFixture,iterations,exactNs?timestep:0u,
                fgmresRestart,fgmresIterations,relativeResidual);
        }
        if (argc==2 && std::string(argv[1])=="--support-only") {
            @autoreleasepool {
                id<MTLDevice> device=MTLCreateSystemDefaultDevice();
                require(device!=nil,"Metal device unavailable");
                for (unsigned shape=0;shape<3;++shape) qualifyHumanSupportKKT(device,shape);
                std::puts("numanx_support_surface_probe=pass shapes=point,sphere,ellipsoid rollback=exact");
                return 0;
            }
        }
        if (argc == 2 && std::string(argv[1]) == "--vascular-fullbody-admission") {
            qualifyFullBodyVascularAdmission();
            return 0;
        }
        require(argc == 1 || (argc == 2 &&
                    (std::string(argv[1]) == "--authored-world" ||
                     std::string(argv[1]) == "--source-equalities" ||
                     std::string(argv[1]) == "--costal-tissue" ||
                     std::string(argv[1]) == "--exact-clock")),
                "usage: numanx_fullbody_bridge_probe [--authored-world|--source-equalities|--costal-tissue|--exact-clock|--support-only|--vascular-fullbody-admission]");
        const bool costal=argc==2&&std::string(argv[1])=="--costal-tissue";
        const bool exactClock=argc==2&&std::string(argv[1])=="--exact-clock";
        return run(argc == 2 && !exactClock,
            costal||(argc == 2 && std::string(argv[1]) == "--source-equalities"),
            costal, exactClock);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
