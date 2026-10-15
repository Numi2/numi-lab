#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#define main preserved_free_box_angular_main
#include "/Users/home/cardboard-implementation-20261014/snapshots/source-boundary-reset-order-101/matter/tools/cardboard_free_box_angular_native_check.mm"
#undef main

#include "numi/matter/matter.hpp"

namespace {
using namespace numi::matter;

#pragma pack(push, 1)
struct PayloadHeader {
    char magic[8];
    std::uint32_t version, environments, nodes, tetrahedra, stateStride, parameterCount;
    std::uint64_t worldFingerprint;
    float dt;
    std::uint32_t reserved;
};
struct TetAudit {
    std::uint32_t nodes[4];
    float inverseRest[9];
    float volume;
    std::uint32_t identity[4];
    float frame[4];
};
#pragma pack(pop)
static_assert(sizeof(PayloadHeader) == 48);
static_assert(sizeof(TetAudit) == 88);

struct Payload {
    PayloadHeader header{};
    std::vector<NMFEMNodeStateGPU> candidate;
    std::vector<nm_float4> direction;
    std::vector<nm_float4> rhs;
    std::vector<float> acceptedHistory;
    std::vector<float> tangent;
    std::vector<std::uint32_t> tangentValid;
    std::vector<float> parameters;
    std::vector<nm_float4> capturedFreshAction;
    std::vector<NMFEMNodeStateGPU> acceptedNodes;
    std::vector<TetAudit> tetAudit;
};

std::string floatWords(const float* values, std::size_t count) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) stream << ',';
        stream << std::setw(8) << std::bit_cast<std::uint32_t>(values[i]);
    }
    return stream.str();
}

bool exactFiniteFloatWords(const float* left, const float* right, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
            std::bit_cast<std::uint32_t>(left[i]) !=
                std::bit_cast<std::uint32_t>(right[i])) {
            return false;
        }
    }
    return true;
}

void testExactFiniteFloatWords() {
    const float positiveZero[3] = {0.0f, 1.0f, 0.0f};
    const float negativeZero[3] = {-0.0f, 1.0f, 0.0f};
    need(!exactFiniteFloatWords(positiveZero, negativeZero, 3u),
         "bitwise immutable-field comparison must distinguish signed zero");
    need(exactFiniteFloatWords(negativeZero, negativeZero, 3u),
         "bitwise immutable-field comparison must accept identical signed-zero words");
    const float adjacent[3] = {0.0f, std::nextafter(1.0f, 2.0f), 0.0f};
    need(!exactFiniteFloatWords(positiveZero, adjacent, 3u),
         "bitwise comparison must reject an adjacent nonzero float");
    const float nonfinite[3] = {0.0f, 1.0f,
                                std::numeric_limits<float>::quiet_NaN()};
    need(!exactFiniteFloatWords(positiveZero, nonfinite, 3u),
         "bitwise comparison must reject nonfinite immutable fields");
}

void readExact(std::ifstream& stream, void* destination, std::size_t bytes,
               const char* label) {
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    need(static_cast<std::size_t>(stream.gcount()) == bytes,
         std::string("truncated replay payload at ") + label);
}

template<class T>
std::vector<T> readVector(std::ifstream& stream, std::size_t count, const char* label) {
    std::vector<T> result(count);
    readExact(stream, result.data(), count * sizeof(T), label);
    return result;
}

Payload readPayload(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    need(stream.good(), "cannot open replay payload");
    Payload p;
    readExact(stream, &p.header, sizeof(p.header), "header");
    need(std::memcmp(p.header.magic, "NMFDSWP1", 8u) == 0 && p.header.version == 1u,
         "unsupported replay payload signature/version");
    const std::size_t e = p.header.environments;
    const std::size_t n = p.header.nodes;
    const std::size_t t = p.header.tetrahedra;
    const std::size_t s = p.header.stateStride;
    const std::size_t q = p.header.parameterCount;
    need(e == 2u && n == 882u && t == 1920u && s == 15u && q == 59u,
         "replay payload dimensions differ from the source125 selected capture");
    p.candidate = readVector<NMFEMNodeStateGPU>(stream, e * n, "candidate nodes");
    p.direction = readVector<nm_float4>(stream, e * n, "correction vector");
    p.rhs = readVector<nm_float4>(stream, e * n, "pre-FGMRES RHS");
    p.acceptedHistory = readVector<float>(stream, e * t * s, "accepted material history");
    p.tangent = readVector<float>(stream, e * t * 81u, "cached tangent");
    p.tangentValid = readVector<std::uint32_t>(stream, e * t, "cache-valid flags");
    p.parameters = readVector<float>(stream, e * q, "environment parameters");
    p.capturedFreshAction = readVector<nm_float4>(stream, e * n,
                                                   "captured production action");
    p.tetAudit = readVector<TetAudit>(stream, e * t, "tetrahedron package audit");
    p.acceptedNodes = readVector<NMFEMNodeStateGPU>(stream, e * n, "accepted nodes");
    need(stream.peek() == std::ifstream::traits_type::eof(),
         "unexpected trailing bytes in replay payload");
    need(std::all_of(p.tangentValid.begin(), p.tangentValid.end(),
                     [](std::uint32_t value) { return value == 1u; }),
         "replay requires all captured cache entries valid");
    need(std::all_of(p.candidate.begin(), p.candidate.end(), [](const auto& node) {
        const float* fields = reinterpret_cast<const float*>(&node);
        for (std::size_t i = 0u; i < sizeof(node) / sizeof(float); ++i)
            if (!std::isfinite(fields[i])) return false;
        return true;
    }), "candidate node buffer contains nonfinite values");
    return p;
}

struct ContactBuffers {
    id<MTLBuffer> argument = nil;
    std::vector<id<MTLBuffer>> owned;
};

ContactBuffers zeroContactArguments(Device& device,
                                   const CompiledWorld& world,
                                   id<MTLArgumentEncoder> encoder) {
    ContactBuffers out;
    const auto& d = world.dispatch;
    auto keep = [&](id<MTLBuffer> buffer) {
        need(buffer != nil, "zero-contact buffer allocation failed");
        out.owned.push_back(buffer);
        return buffer;
    };
    const std::size_t unified = std::size_t(d.gridNodeCount) + d.femNodeCount;
    id<MTLBuffer> args[15] = {
        keep(makeBuffer(device.gpu, std::vector<NMContactPairGPU>{})),
        keep(makeBuffer(device.gpu, std::vector<NMContactSampleGPU>{})),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>{})),
        keep(makeBuffer(device.gpu,
            std::vector<NMIncidenceRangeGPU>(std::size_t(d.environmentCount) * unified))),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>{})),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>{})),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>(d.environmentCount))),
        keep(makeBuffer(device.gpu,
            std::vector<NMDeformableContactGPU>(
                std::size_t(d.environmentCount) * d.deformableContactCapacity))),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>{})),
        keep(makeBuffer(device.gpu,
            std::vector<NMIncidenceRangeGPU>(std::size_t(d.environmentCount) * unified))),
        keep(makeBuffer(device.gpu,
            std::vector<std::uint32_t>(std::max<std::size_t>(1u,
                std::size_t(d.environmentCount) * d.gridNodeCount), NM_INVALID_INDEX))),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>(d.environmentCount))),
        keep(makeBuffer(device.gpu, world.contact.rigidProxies)),
        keep(makeBuffer(device.gpu, std::vector<NMRigidStateGPU>{})),
        keep(makeBuffer(device.gpu, std::vector<std::uint32_t>(d.environmentCount)))};
    out.argument = [device.gpu newBufferWithLength:encoder.encodedLength
                                           options:MTLResourceStorageModeShared];
    need(out.argument != nil, "zero-contact argument buffer allocation failed");
    [encoder setArgumentBuffer:out.argument offset:0u];
    for (NSUInteger i = 0u; i < 15u; ++i)
        [encoder setBuffer:args[i] offset:0u atIndex:i];
    return out;
}

std::vector<NMTetrahedronGPU> repeatTetrahedra(const CompiledWorld& world,
                                                std::uint32_t environments) {
    std::vector<NMTetrahedronGPU> result;
    result.reserve(std::size_t(environments) * world.fem.tetrahedra.size());
    for (std::uint32_t e = 0u; e < environments; ++e)
        result.insert(result.end(), world.fem.tetrahedra.begin(), world.fem.tetrahedra.end());
    return result;
}

std::vector<std::uint32_t> repeatIncidence(const CompiledWorld& world,
                                          std::uint32_t environments) {
    std::vector<std::uint32_t> result;
    result.reserve(std::size_t(environments) * world.fem.nodeIncidence.size());
    for (std::uint32_t e = 0u; e < environments; ++e)
        result.insert(result.end(), world.fem.nodeIncidence.begin(), world.fem.nodeIncidence.end());
    return result;
}

std::vector<NMIncidenceRangeGPU> repeatRanges(const CompiledWorld& world,
                                              std::uint32_t environments) {
    std::vector<NMIncidenceRangeGPU> result;
    result.reserve(std::size_t(environments) * world.fem.nodeRanges.size());
    const auto perEnvironmentIncidence = static_cast<std::uint32_t>(world.fem.nodeIncidence.size());
    for (std::uint32_t e = 0u; e < environments; ++e) {
        for (auto range : world.fem.nodeRanges) {
            range.first += e * perEnvironmentIncidence;
            result.push_back(range);
        }
    }
    return result;
}

template<class T>
std::vector<T> repeatObjectRows(const std::vector<T>& input,
                                std::uint32_t environments,
                                const char* label) {
    need(!input.empty(), std::string("compiled ") + label + " array is empty");
    std::vector<T> result;
    result.reserve(std::size_t(environments) * input.size());
    for (std::uint32_t e = 0u; e < environments; ++e)
        result.insert(result.end(), input.begin(), input.end());
    return result;
}

void checkCompiledBinding(const CompiledWorld& world, const Payload& p) {
    const auto& h = p.header;
    const auto& d = world.dispatch;
    need(world.fingerprint == h.worldFingerprint,
         "compiled package fingerprint differs from rich-capture world fingerprint");
    need(d.environmentCount == h.environments && d.femNodeCount == h.nodes &&
         d.tetrahedronCount == h.tetrahedra && d.materialStateStride == h.stateStride &&
         d.parameterCount == h.parameterCount,
         "compiled package dispatch dimensions differ from captured inputs");
    need(d.contactPairCount == 0u && d.rigidProxyCount == 0u &&
         d.fieldBoundaryCount == 0u && d.cohesiveFaceCount == 0u &&
         d.rigidGeneralizedCapacity == 0u,
         "this replay is bounded to the captured uncoupled FEM linear slice");
    need(world.objects.size() == d.objectCount && d.objectCount == 1u &&
         world.fem.nodes.size() == h.nodes && world.fem.tetrahedra.size() == h.tetrahedra,
         "compiled FEM topology shape unsupported by the replay");
    need(world.parameters.size() == h.parameterCount && world.schedulers.size() == d.objectCount &&
         world.adaptive.size() == d.objectCount,
         "compiled parameter/scheduler/adaptive shape unsupported");
    need(world.materials.size() == d.materialCount &&
         world.mixedSolver.executionBudgets.w == 1u,
         "compiled material/cache policy differs from source125 run");
    const auto& object = world.objects[0];
    need(object.representation == NM_REPRESENTATION_FEM &&
         (object.flags & NM_OBJECT_ACTIVE) != 0u,
         "compiled object is not one active FEM object");
    const auto& scheduler = world.schedulers[0];
    const auto& adaptive = world.adaptive[0];
    need(adaptive.activeRepresentation == object.representation &&
         scheduler.numerical.w > 0.0f,
         "captured FEM object is not scheduled active");
    const std::uint32_t activeExponent = std::min(scheduler.activeExponent,
                                                  d.maximumRateExponent);
    const float compiledDt = d.gravityAndTimestep.w /
        float(1u << activeExponent);
    need(std::bit_cast<std::uint32_t>(compiledDt) == std::bit_cast<std::uint32_t>(p.header.dt),
         "capture timestep does not match compiled active scheduler timestep");
    need(world.fem.nodeRanges.size() == h.nodes && !world.fem.nodeIncidence.empty(),
         "compiled FEM incidence arrays are malformed");

    for (std::uint32_t e = 0u; e < h.environments; ++e) {
        for (std::uint32_t t = 0u; t < h.tetrahedra; ++t) {
            const TetAudit& captured = p.tetAudit[std::size_t(e) * h.tetrahedra + t];
            const auto& compiled = world.fem.tetrahedra[t];
            const std::uint32_t compiledNodes[4] = {
                compiled.nodes.x, compiled.nodes.y, compiled.nodes.z, compiled.nodes.w};
            const float compiledInv[9] = {
                compiled.inverseRestRow0.x, compiled.inverseRestRow0.y,
                compiled.inverseRestRow0.z, compiled.inverseRestRow1.x,
                compiled.inverseRestRow1.y, compiled.inverseRestRow1.z,
                compiled.inverseRestRow2.x, compiled.inverseRestRow2.y,
                compiled.inverseRestRow2.z};
            const std::uint32_t idWords[4] = {compiled.identity.x, compiled.identity.y,
                compiled.identity.z, compiled.identity.w};
            const float frame[4] = {compiled.materialFrameRotation.x,
                compiled.materialFrameRotation.y, compiled.materialFrameRotation.z,
                compiled.materialFrameRotation.w};
            need(std::memcmp(captured.nodes, compiledNodes, sizeof(compiledNodes)) == 0,
                 "capture/package connectivity mismatch env=" + std::to_string(e) +
                 " tet=" + std::to_string(t));
            need(exactFiniteFloatWords(captured.inverseRest, compiledInv, 9u),
                 "capture/package inverse-rest mismatch env=" + std::to_string(e) +
                 " tet=" + std::to_string(t) + " captured=" + floatWords(captured.inverseRest, 9u) +
                 " compiled=" + floatWords(compiledInv, 9u));
            need(std::bit_cast<std::uint32_t>(captured.volume) ==
                    std::bit_cast<std::uint32_t>(compiled.inverseRestRow0.w),
                 "capture/package volume mismatch env=" + std::to_string(e) +
                 " tet=" + std::to_string(t));
            need(std::memcmp(captured.identity, idWords, sizeof(idWords)) == 0,
                 "capture/package identity mismatch env=" + std::to_string(e) +
                 " tet=" + std::to_string(t));
            need(exactFiniteFloatWords(captured.frame, frame, 4u),
                 "capture/package material-frame mismatch env=" + std::to_string(e) +
                 " tet=" + std::to_string(t) + " captured=" + floatWords(captured.frame, 4u) +
                 " compiled=" + floatWords(frame, 4u));
        }
        for (std::uint32_t i = 0u; i < h.parameterCount; ++i) {
            const float captured = p.parameters[std::size_t(e) * h.parameterCount + i];
            const float compiled = world.parameters[i].valueAndBounds.x;
            need(std::bit_cast<std::uint32_t>(captured) ==
                 std::bit_cast<std::uint32_t>(compiled),
                 "captured environment parameter differs from compiled package");
        }
    }
}

struct NativeResult {
    std::vector<NMFEMElementVectorGPU> elements;
    std::vector<nm_float4> action;
    std::vector<NMMatterStatusGPU> status;
};

NativeResult evaluateNative(Device& device,
                            id<MTLComputePipelineState> femOperator,
                            id<MTLComputePipelineState> gatherNodes,
                            const CompiledWorld& sourceWorld,
                            const Payload& p,
                            id<MTLArgumentEncoder> contactEncoder,
                            const std::uint32_t useCache,
                            const bool massOnly = false,
                            const bool trueResidual = true,
                            const bool requireSuccess = true) {
    CompiledWorld world = sourceWorld;
    const std::uint32_t eCount = p.header.environments;
    const std::uint32_t nCount = p.header.nodes;
    const std::uint32_t tCount = p.header.tetrahedra;
    world.dispatch.gravityAndTimestep.w = p.header.dt;
    NMMicrostepGPU micro{};
    micro.microtickCount = 1u;
    micro.microtick = 0u;
    micro.flags = NM_MICROSTEP_FGMRES_OPERATOR |
        (trueResidual ? NM_MICROSTEP_FGMRES_TRUE_RESIDUAL : 0u);
    micro.time = {p.header.dt, 1.0f / p.header.dt, 0.0f, p.header.dt};

    const auto nodeIncidence = repeatIncidence(world, eCount);
    const auto nodeRanges = repeatRanges(world, eCount);
    const auto tetrahedra = repeatTetrahedra(world, eCount);
    const auto schedulers = repeatObjectRows(world.schedulers, eCount, "scheduler");
    const auto adaptive = repeatObjectRows(world.adaptive, eCount, "adaptive");
    const auto objectB = makeBuffer(device.gpu, world.objects);
    const auto materialB = makeBuffer(device.gpu, world.materials);
    const auto programB = makeBuffer(device.gpu, world.scalarPrograms);
    const auto instructionB = makeBuffer(device.gpu, world.instructions);
    const auto parameterB = makeBuffer(device.gpu, p.parameters);
    const auto nodeB = makeBuffer(device.gpu, p.candidate);
    const auto tetB = makeBuffer(device.gpu, tetrahedra);
    const auto historyB = makeBuffer(device.gpu, p.acceptedHistory);
    const auto directionB = makeBuffer(device.gpu, p.direction);
    const auto schedulerB = makeBuffer(device.gpu, schedulers);
    const auto adaptiveB = makeBuffer(device.gpu, adaptive);
    const auto mixedB = makeBuffer(device.gpu, world.mixedMaterials);
    const auto fieldsB = makeBuffer(device.gpu, world.fem.fields);
    const auto learnedB = makeBuffer(device.gpu, world.learnedMaterials);
    const auto layersB = makeBuffer(device.gpu, world.learnedLayers);
    const auto weightsB = makeBuffer(device.gpu, world.learnedWeights);
    const auto solverB = makeBuffer(device.gpu, world.mixedSolver);
    std::vector<NMFGMRESStateGPU> latchedStates(eCount);
    for (auto& state : latchedStates) state.diagnostics.z = 1.0f;
    const auto fgmresB = makeBuffer(device.gpu, latchedStates);
    std::vector<nm_float4> fallbackMetrics(eCount);
    fallbackMetrics[0].w = 3.0f; fallbackMetrics[1].w = 1.0f;
    const auto fallbackMetricsB = makeBuffer(device.gpu, fallbackMetrics);
    const auto cacheB = makeBuffer(device.gpu, p.tangent);
    const auto cacheValidB = makeBuffer(device.gpu, p.tangentValid);
    const auto useCacheB = makeBuffer(device.gpu, useCache);
    const auto statusB = makeBuffer(device.gpu,
        std::vector<NMMatterStatusGPU>(eCount));
    const auto elementB = makeBuffer(device.gpu,
        std::vector<NMFEMElementVectorGPU>(std::size_t(eCount) * tCount));
    auto dispatch = world.dispatch;
    const auto contact = zeroContactArguments(device, world, contactEncoder);

    auto command = [device.queue commandBuffer];
    auto encoder = [command computeCommandEncoder];
    need(command != nil && encoder != nil, "native element operator encoder failed");
    [encoder setComputePipelineState:femOperator];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
    id<MTLBuffer> args[27] = {};
    args[2] = objectB; args[3] = materialB; args[4] = programB;
    args[5] = instructionB; args[6] = parameterB; args[7] = nodeB;
    args[8] = tetB; args[9] = historyB; args[10] = directionB;
    args[11] = schedulerB; args[12] = adaptiveB; args[13] = elementB;
    args[14] = statusB; args[16] = mixedB; args[17] = fieldsB;
    args[18] = learnedB; args[19] = layersB; args[20] = weightsB;
    args[21] = solverB; args[22] = fgmresB; args[23] = cacheB;
    args[24] = cacheValidB; args[25] = useCacheB; args[26] = fallbackMetricsB;
    for (NSUInteger i = 2u; i < 27u; ++i)
        if (i != 15u) [encoder setBuffer:args[i] offset:0u atIndex:i];
    [encoder dispatchThreads:MTLSizeMake(std::size_t(eCount) * tCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
    [encoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    need(command.status == MTLCommandBufferStatusCompleted,
         "native element operator command did not complete");
    const auto* statuses = static_cast<const NMMatterStatusGPU*>(statusB.contents);
    for (std::uint32_t e = 0u; e < eCount; ++e)
        need(!requireSuccess || statuses[e].code == NM_STATUS_SUCCESS,
             "native element operator returned status " + std::to_string(statuses[e].code));

    // Isolate the unchanged gather mass term after checking the real bulk action.
    // The normal cached/fresh paths never change this buffer.
    if (massOnly) std::memset(elementB.contents, 0, elementB.length);

    // Production nodal gather: M*delta_v - dt^2*dF, with the authored fixed
    // rows zeroed. No rigid/contact or field rows exist in this captured pack.
    auto workB = makeBuffer(device.gpu,
        std::vector<nm_float4>(std::size_t(2u) * eCount * nCount +
            std::size_t(eCount) * (dispatch.mpmActiveNodeCapacity +
                                   dispatch.rigidGeneralizedCapacity)));
    auto incidenceB = makeBuffer(device.gpu, nodeIncidence);
    auto rangesB = makeBuffer(device.gpu, nodeRanges);
    command = [device.queue commandBuffer]; encoder = [command computeCommandEncoder];
    need(command != nil && encoder != nil, "native nodal gather encoder failed");
    [encoder setComputePipelineState:gatherNodes];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
    [encoder setBuffer:objectB offset:0u atIndex:2u];
    [encoder setBuffer:nodeB offset:0u atIndex:3u];
    [encoder setBuffer:tetB offset:0u atIndex:4u];
    [encoder setBuffer:elementB offset:0u atIndex:5u];
    [encoder setBuffer:incidenceB offset:0u atIndex:6u];
    [encoder setBuffer:rangesB offset:0u atIndex:7u];
    [encoder setBuffer:schedulerB offset:0u atIndex:8u];
    [encoder setBuffer:adaptiveB offset:0u atIndex:9u];
    [encoder setBuffer:directionB offset:0u atIndex:10u];
    [encoder setBuffer:workB offset:0u atIndex:11u];
    [encoder setBuffer:contact.argument offset:0u atIndex:12u];
    [encoder setBuffer:fgmresB offset:0u atIndex:13u];
    const std::uint32_t articulatedNv = 0u;
    [encoder setBytes:&articulatedNv length:sizeof(articulatedNv) atIndex:14u];
    const auto pointJacobiansB = makeBuffer(device.gpu, std::vector<float>{0.0f});
    [encoder setBuffer:pointJacobiansB offset:0u atIndex:15u];
    [encoder dispatchThreads:MTLSizeMake(std::size_t(eCount) * nCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
    [encoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    need(command.status == MTLCommandBufferStatusCompleted,
         "native nodal gather command did not complete");
    const auto* work = static_cast<const nm_float4*>(workB.contents);
    NativeResult result;
    const auto* elementValues = static_cast<const NMFEMElementVectorGPU*>(elementB.contents);
    result.elements.assign(elementValues, elementValues + std::size_t(eCount) * tCount);
    result.action.assign(work, work + std::size_t(eCount) * nCount);
    result.status.assign(statuses, statuses + eCount);
    return result;
}


void save(const std::filesystem::path& path, id<MTLBuffer> buffer) {
    std::ofstream out(path,std::ios::binary); need(out.good(),"output unavailable");
    out.write(static_cast<const char*>(buffer.contents),buffer.length);
    need(out.good(),"output write failed");
}
template<class T> void saveVector(const std::filesystem::path& path,const std::vector<T>& values) {
    std::ofstream out(path,std::ios::binary);need(out.good(),"output unavailable");
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));need(out.good(),"output write failed");
}
void kernelCheck(Device& device,const CompiledWorld& world,const Payload& p,const std::filesystem::path& output) {
    const auto E=p.header.environments,N=p.header.nodes;
    const auto count=std::size_t(E)*N;
    auto dispatch=world.dispatch;
    std::vector<nm_float4> metrics(E);metrics[0]={1.0f,2.0f,0.0f,3.0f};metrics[1]={4.0f,5.0f,0.75f,1.0f};
    const auto metricsB=makeBuffer(device.gpu,metrics);
    const auto nodesB=makeBuffer(device.gpu,p.candidate);
    const auto rhsB=makeBuffer(device.gpu,p.rhs);
    const auto baseB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    const auto emptyB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    const auto directionB=makeBuffer(device.gpu,std::vector<nm_float4>(count));
    auto invoke=[&](NSString* name,NSUInteger rows,auto bind){
        auto command=[device.queue commandBuffer];auto encoder=[command computeCommandEncoder];
        need(command&&encoder,"kernel encoder failed");[encoder setComputePipelineState:device.makePipeline(name)];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(rows,1u,1u) threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,"kernel command did not complete");
    };
    invoke(@"numi_matter_metal::nm_fgmres_capture_base_cauchy_rhs",count,[&](id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:rhsB offset:0u atIndex:1u];[enc setBuffer:nodesB offset:0u atIndex:2u];[enc setBuffer:baseB offset:0u atIndex:3u];
    });save(output/"base-rhs.bin",baseB);
    auto prepare=[&](id<MTLBuffer> gradient,std::uint32_t useGradient){
        invoke(@"numi_matter_metal::nm_fgmres_prepare_base_cauchy_action",count,[&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:metricsB offset:0u atIndex:1u];[enc setBuffer:baseB offset:0u atIndex:2u];[enc setBuffer:gradient offset:0u atIndex:3u];
            [enc setBuffer:nodesB offset:0u atIndex:4u];[enc setBuffer:directionB offset:0u atIndex:5u];[enc setBytes:&useGradient length:sizeof(useGradient) atIndex:6u];
        });
        const auto* d=static_cast<const nm_float4*>(directionB.contents);return std::vector<nm_float4>(d,d+count);
    };
    Payload working=p;working.direction=prepare(emptyB,0u);save(output/"transpose-input.bin",directionB);
    auto fem=device.makePipeline(@"numi_matter_metal::nm_fem_apply_operator_elements");auto gather=device.makePipeline(@"numi_matter_metal::nm_fgmres_gather_nodes");
    auto layoutFunction=[device.library newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
    need(layoutFunction!=nil,"contact layout absent");auto contactEncoder=[layoutFunction newArgumentEncoderWithBufferIndex:0u];
    auto transpose=evaluateNative(device,fem,gather,world,working,contactEncoder,3u);saveVector(output/"gradient.bin",transpose.action);
    auto gradientB=makeBuffer(device.gpu,transpose.action);
    auto latch=evaluateNative(device,fem,gather,world,working,contactEncoder,3u,false,false);saveVector(output/"latch-control.bin",latch.action);
    Payload invalidUnselected=working;invalidUnselected.tangentValid[p.header.tetrahedra]=0u;
    auto unselected=evaluateNative(device,fem,gather,world,invalidUnselected,contactEncoder,3u);saveVector(output/"unselected-invalid-cache.bin",unselected.action);
    Payload invalidSelected=working;invalidSelected.tangentValid[0]=0u;
    auto selected=evaluateNative(device,fem,gather,world,invalidSelected,contactEncoder,3u,false,true,false);saveVector(output/"selected-invalid-cache-status.bin",selected.status);
    need(selected.status[0].code!=NM_STATUS_SUCCESS&&selected.status[1].code==NM_STATUS_SUCCESS,"strict cache mask failure");
    working.direction=prepare(gradientB,1u);save(output/"forward-input.bin",directionB);
    auto forward=evaluateNative(device,fem,gather,world,working,contactEncoder,4u);saveVector(output/"gradient-action.bin",forward.action);
    auto forwardB=makeBuffer(device.gpu,forward.action);
    std::vector<nm_float4> solution(2u*count,nm_float4{0.125f,-0.25f,0.5f,0.0f});
    const auto solutionB=makeBuffer(device.gpu,solution);
    std::vector<nm_float4> line(E,nm_float4{0.25f,3.0f,0.5f,48.0f});
    const auto lineB=makeBuffer(device.gpu,line);const auto alphaB=makeBuffer(device.gpu,std::vector<float>(E,0.75f));
    const auto traceB=makeBuffer(device.gpu,std::vector<nm_float4>(E));
    std::vector<NMFGMRESStateGPU> states(E);for(auto& s:states)s.diagnostics.z=1.0f;
    const auto statesB=makeBuffer(device.gpu,states);const auto statusB=makeBuffer(device.gpu,std::vector<NMMatterStatusGPU>(E));
    const auto trialStatusB=makeBuffer(device.gpu,std::vector<NMMatterStatusGPU>(E));
    NMFGMRESLayoutGPU layout{};layout.unknownCount=static_cast<std::uint32_t>(2u*count);
    const std::uint32_t iteration=48u;
    invoke(@"numi_matter_metal::nm_fgmres_build_base_cauchy_direction",E,[&](id<MTLComputeCommandEncoder> enc){
        [enc setBytes:&world.mixedSolver length:sizeof(world.mixedSolver) atIndex:1u];
        id<MTLBuffer> buffers[]={metricsB,nodesB,gradientB,forwardB,solutionB,lineB,alphaB,traceB,statesB,statusB,trialStatusB};
        for(NSUInteger i=0u;i<11u;++i)[enc setBuffer:buffers[i] offset:0u atIndex:i+2u];
        [enc setBytes:&iteration length:sizeof(iteration) atIndex:13u];[enc setBytes:&layout length:sizeof(layout) atIndex:30u];
    });
    save(output/"solution.bin",solutionB);save(output/"metrics.bin",metricsB);save(output/"line-search.bin",lineB);save(output/"environment-alpha.bin",alphaB);
    save(output/"fallback-trace.bin",traceB);save(output/"states.bin",statesB);save(output/"statuses.bin",statusB);save(output/"trial-statuses.bin",trialStatusB);
    need(std::memcmp(nodesB.contents,p.candidate.data(),nodesB.length)==0,"candidate inputs changed");
    need(std::memcmp(rhsB.contents,p.rhs.data(),rhsB.length)==0,"RHS inputs changed");
    const auto* actual=static_cast<const nm_float4*>(solutionB.contents);
    need(std::memcmp(actual+N,solution.data()+N,N*sizeof(nm_float4))==0,"unselected solution changed");
    need(std::memcmp(actual+count+N,solution.data()+count+N,N*sizeof(nm_float4))==0,"unselected field rows changed");
    need(std::memcmp(static_cast<const nm_float4*>(metricsB.contents)+1,metrics.data()+1,sizeof(nm_float4))==0,"unselected metrics changed");
    need(std::memcmp(static_cast<const NMFGMRESStateGPU*>(statesB.contents)+1,states.data()+1,sizeof(NMFGMRESStateGPU))==0,"unselected state changed");
    need(std::memcmp(static_cast<const nm_float4*>(lineB.contents)+1,line.data()+1,sizeof(nm_float4))==0,"unselected line search changed");
    const auto* statuses=static_cast<const NMMatterStatusGPU*>(statusB.contents);need(statuses[0].code==0u&&statuses[1].code==0u,"direction status failure");
    std::ofstream report(output/"native-contracts.json");report<<"{\"native_commands_complete\":true,\"unselected_environment_unchanged\":true,\"immutable_inputs_unchanged\":true,\"strict_cache_mask_control\":true,\"physical_transactions\":0}\n";
}
} // namespace
int main(int argc,char** argv){@autoreleasepool{try{
    testExactFiniteFloatWords();need(argc==4,"usage: source133 kernel check <package> <payload> <output>");auto p=readPayload(argv[2]);CompiledWorld world;std::string error;
    need(readPackage(argv[1],world,nullptr,&error),"package rejected: "+error);checkCompiledBinding(world,p);
    std::filesystem::create_directories(argv[3]);Device device;kernelCheck(device,world,p,argv[3]);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
