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
                            const bool massOnly = false) {
    CompiledWorld world = sourceWorld;
    const std::uint32_t eCount = p.header.environments;
    const std::uint32_t nCount = p.header.nodes;
    const std::uint32_t tCount = p.header.tetrahedra;
    world.dispatch.gravityAndTimestep.w = p.header.dt;
    NMMicrostepGPU micro{};
    micro.microtickCount = 1u;
    micro.microtick = 0u;
    micro.flags = NM_MICROSTEP_FGMRES_OPERATOR;
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
    const auto fgmresB = makeBuffer(device.gpu,
        std::vector<NMFGMRESStateGPU>(eCount));
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
    id<MTLBuffer> args[26] = {};
    args[2] = objectB; args[3] = materialB; args[4] = programB;
    args[5] = instructionB; args[6] = parameterB; args[7] = nodeB;
    args[8] = tetB; args[9] = historyB; args[10] = directionB;
    args[11] = schedulerB; args[12] = adaptiveB; args[13] = elementB;
    args[14] = statusB; args[16] = mixedB; args[17] = fieldsB;
    args[18] = learnedB; args[19] = layersB; args[20] = weightsB;
    args[21] = solverB; args[22] = fgmresB; args[23] = cacheB;
    args[24] = cacheValidB; args[25] = useCacheB;
    for (NSUInteger i = 2u; i < 26u; ++i)
        if (i != 15u) [encoder setBuffer:args[i] offset:0u atIndex:i];
    [encoder dispatchThreads:MTLSizeMake(std::size_t(eCount) * tCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
    [encoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    need(command.status == MTLCommandBufferStatusCompleted,
         "native element operator command did not complete");
    const auto* statuses = static_cast<const NMMatterStatusGPU*>(statusB.contents);
    for (std::uint32_t e = 0u; e < eCount; ++e)
        need(statuses[e].code == NM_STATUS_SUCCESS,
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


struct ResidualResult {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMElementVectorGPU> forces;
    std::vector<nm_float4> residual;
    std::vector<NMMatterStatusGPU> statuses;
};

ResidualResult evaluateResidual(Device& device, const CompiledWorld& world,
                                const Payload& p, float epsilon, bool negative) {
    const auto e = p.header.environments, n = p.header.nodes, t = p.header.tetrahedra;
    auto dispatch = world.dispatch;
    dispatch.gravityAndTimestep.w = p.header.dt;
    NMMicrostepGPU micro{};
    micro.microtickCount = 1u; micro.solverIteration = 48u;
    micro.time = {p.header.dt, 1.0f / p.header.dt, 0.0f, p.header.dt};
    const auto objectB = makeBuffer(device.gpu, world.objects);
    const auto materialsB = makeBuffer(device.gpu, world.materials);
    const auto programsB = makeBuffer(device.gpu, world.scalarPrograms);
    const auto instructionsB = makeBuffer(device.gpu, world.instructions);
    const auto parametersB = makeBuffer(device.gpu, p.parameters);
    const auto acceptedB = makeBuffer(device.gpu, p.acceptedNodes);
    const auto candidateB = makeBuffer(device.gpu, p.candidate);
    auto signedDirection = p.direction;
    if (negative) for (auto& d : signedDirection) { d.x = -d.x; d.y = -d.y; d.z = -d.z; }
    const auto directionB = makeBuffer(device.gpu, signedDirection);
    const auto rangesB = makeBuffer(device.gpu, repeatRanges(world, e));
    const auto incidenceB = makeBuffer(device.gpu, repeatIncidence(world, e));
    const auto schedulersB = makeBuffer(device.gpu, repeatObjectRows(world.schedulers, e, "scheduler"));
    const auto adaptiveB = makeBuffer(device.gpu, repeatObjectRows(world.adaptive, e, "adaptive"));
    const auto tetB = makeBuffer(device.gpu, repeatTetrahedra(world, e));
    const auto historyB = makeBuffer(device.gpu, p.acceptedHistory);
    const auto statusB = makeBuffer(device.gpu, std::vector<NMMatterStatusGPU>(e));
    const auto trialB = makeBuffer(device.gpu, std::vector<NMFEMNodeStateGPU>(std::size_t(e)*n));
    const auto forceB = makeBuffer(device.gpu, std::vector<NMFEMElementVectorGPU>(std::size_t(e)*t));
    std::vector<nm_float4> metrics(e); for (auto& m : metrics) m.z = epsilon;
    const auto metricsB = makeBuffer(device.gpu, metrics);
    const auto mixedB = makeBuffer(device.gpu, world.mixedMaterials);
    const auto fieldsB = makeBuffer(device.gpu, world.fem.fields);
    const auto learnedB = makeBuffer(device.gpu, world.learnedMaterials);
    const auto layersB = makeBuffer(device.gpu, world.learnedLayers);
    const auto weightsB = makeBuffer(device.gpu, world.learnedWeights);
    const auto solverB = makeBuffer(device.gpu, world.mixedSolver);
    const auto residualB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto solutionB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto zB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto dB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto aB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto reactionsB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    const auto externalB = makeBuffer(device.gpu, std::vector<nm_float4>(std::size_t(e)*n));
    auto command = [device.queue commandBuffer];
    auto encode = [&](NSString* name, NSUInteger count, auto bind) {
        auto encoder = [command computeCommandEncoder];
        auto pipeline = device.makePipeline(name);
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
        bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(count,1u,1u) threadsPerThreadgroup:MTLSizeMake(64u,1u,1u)];
        [encoder endEncoding];
    };
    encode(@"numi_matter_metal::nm_fem_armijo_prepare_candidate",std::size_t(e)*n,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,rangesB,schedulersB,adaptiveB,acceptedB,candidateB,directionB,metricsB,statusB,trialB};
        for(NSUInteger i=0;i<10u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_armijo_validate_geometry",e,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,materialsB,tetB,schedulersB,adaptiveB,trialB,statusB,metricsB};
        for(NSUInteger i=0;i<8u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_internal_forces",std::size_t(e)*t,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,materialsB,programsB,instructionsB,parametersB,trialB,tetB,historyB,schedulersB,adaptiveB,forceB,statusB,mixedB,fieldsB,learnedB,layersB,weightsB};
        for(NSUInteger i=0;i<17u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
    });
    encode(@"numi_matter_metal::nm_fem_build_mechanical_residual",std::size_t(e)*n,[&](id<MTLComputeCommandEncoder> enc){
        id<MTLBuffer> args[]={objectB,acceptedB,trialB,tetB,forceB,incidenceB,rangesB,schedulersB,adaptiveB,solutionB,residualB,zB,dB,aB,statusB,solverB,mixedB,fieldsB};
        for(NSUInteger i=0;i<18u;++i)[enc setBuffer:args[i] offset:0u atIndex:i+2u];
        const std::uint32_t preserve=1u,hasExternal=0u;
        [enc setBytes:&preserve length:sizeof(preserve) atIndex:20u];
        [enc setBuffer:externalB offset:0u atIndex:21u];
        [enc setBytes:&hasExternal length:sizeof(hasExternal) atIndex:22u];
        [enc setBuffer:reactionsB offset:0u atIndex:23u];
    });
    [command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,"native residual command did not complete");
    ResidualResult r;
    const auto* nodes=static_cast<const NMFEMNodeStateGPU*>(trialB.contents);r.nodes.assign(nodes,nodes+std::size_t(e)*n);
    const auto* forces=static_cast<const NMFEMElementVectorGPU*>(forceB.contents);r.forces.assign(forces,forces+std::size_t(e)*t);
    const auto* res=static_cast<const nm_float4*>(residualB.contents);r.residual.assign(res,res+std::size_t(e)*n);
    const auto* status=static_cast<const NMMatterStatusGPU*>(statusB.contents);r.statuses.assign(status,status+e);
    need(std::memcmp(historyB.contents,p.acceptedHistory.data(),historyB.length)==0,"accepted material history mutated");
    need(std::memcmp(acceptedB.contents,p.acceptedNodes.data(),acceptedB.length)==0,"accepted node history mutated");
    need(std::memcmp(candidateB.contents,p.candidate.data(),candidateB.length)==0,"center candidate mutated");
    return r;
}

struct Metrics {
    double actionL2 = 0.0;
    double otherActionL2 = 0.0;
    double cachedFreshDifferenceL2 = 0.0;
    double cachedCapturedDifferenceL2 = 0.0;
    double freshCapturedDifferenceL2 = 0.0;
    double residualL2 = 0.0;
    double otherResidualL2 = 0.0;
    double residualL2AllRows = 0.0;
    double otherResidualL2AllRows = 0.0;
    double bL2 = 0.0;
    double directionL2 = 0.0;
    double maxAction = 0.0;
    double maxCacheFreshDifference = 0.0;
    std::uint32_t freeNodes = 0u;
};

Metrics metricsFor(std::uint32_t environment, const Payload& p,
                   const NativeResult& result, const NativeResult& other,
                   const std::vector<nm_float4>& capturedFresh) {
    Metrics m;
    const std::size_t n = p.header.nodes;
    double residualAll2 = 0.0;
    double residualFree2 = 0.0;
    double b2 = 0.0;
    double dir2 = 0.0;
    double action2 = 0.0;
    double otherAction2 = 0.0;
    double diff2 = 0.0;
    double cachedCapturedDiff2 = 0.0;
    double freshCapturedDiff2 = 0.0;
    double otherResidualAll2 = 0.0;
    double otherResidualFree2 = 0.0;
    for (std::size_t i = 0u; i < n; ++i) {
        const std::size_t index = std::size_t(environment) * n + i;
        const auto& node = p.candidate[index];
        const float constraint = node.restAndFixed.w;
        const bool fixed = constraint == 1.0f || constraint == 3.0f;
        const float* action = &result.action[index].x;
        const float* otherAction = &other.action[index].x;
        const float* b = &p.rhs[index].x;
        const float* direction = &p.direction[index].x;
        double localResidual2 = 0.0;
        double localOtherResidual2 = 0.0;
        for (unsigned c = 0u; c < 3u; ++c) {
            const double r = double(b[c]) - double(action[c]);
            localResidual2 += r * r;
            const double otherResidual = double(b[c]) - double(otherAction[c]);
            localOtherResidual2 += otherResidual * otherResidual;
            b2 += double(b[c]) * double(b[c]);
            dir2 += double(direction[c]) * double(direction[c]);
            action2 += double(action[c]) * double(action[c]);
            otherAction2 += double(otherAction[c]) * double(otherAction[c]);
            const double d = double(action[c]) - double(otherAction[c]);
            diff2 += d * d;
            const double ccd = double(action[c]) - double((&capturedFresh[index].x)[c]);
            const double fcd = double(otherAction[c]) - double((&capturedFresh[index].x)[c]);
            cachedCapturedDiff2 += ccd * ccd;
            freshCapturedDiff2 += fcd * fcd;
            m.maxAction = std::max(m.maxAction, std::abs(double(action[c])));
            m.maxCacheFreshDifference = std::max(m.maxCacheFreshDifference, std::abs(d));
        }
        residualAll2 += localResidual2;
        otherResidualAll2 += localOtherResidual2;
        if (!fixed) {
            residualFree2 += localResidual2;
            otherResidualFree2 += localOtherResidual2;
            ++m.freeNodes;
        }
    }
    m.actionL2 = std::sqrt(action2);
    m.otherActionL2 = std::sqrt(otherAction2);
    m.cachedFreshDifferenceL2 = std::sqrt(diff2);
    m.cachedCapturedDifferenceL2 = std::sqrt(cachedCapturedDiff2);
    m.freshCapturedDifferenceL2 = std::sqrt(freshCapturedDiff2);
    m.residualL2 = std::sqrt(residualFree2);
    m.otherResidualL2 = std::sqrt(otherResidualFree2);
    m.residualL2AllRows = std::sqrt(residualAll2);
    m.otherResidualL2AllRows = std::sqrt(otherResidualAll2);
    m.bL2 = std::sqrt(b2);
    m.directionL2 = std::sqrt(dir2);
    return m;
}

void writeFloat4File(const std::string& path, const std::vector<nm_float4>& values) {
    std::ofstream out(path, std::ios::binary);
    need(out.good(), "cannot open output vector file");
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(nm_float4)));
    need(out.good(), "failed writing output vector file");
}

void writeElementFile(const std::string& path, const std::vector<NMFEMElementVectorGPU>& values) {
    static_assert(sizeof(NMFEMElementVectorGPU) == 64u);
    std::ofstream out(path, std::ios::binary);
    need(out.good(), "cannot open element output file");
    out.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(NMFEMElementVectorGPU)));
    need(out.good(), "failed writing element output file");
}

[[maybe_unused]] void writeReport(const std::string& path, const Payload& p,
                 const Metrics& env0, const Metrics& env1) {
    std::ofstream out(path);
    need(out.good(), "cannot open JSON report");
    out << std::setprecision(17)
        << "{\n  \"schema\":\"numi.cardboard.native-cached-operator-replay.v1\",\n"
        << "  \"source125_world_fingerprint_hex\":\"" << std::hex
        << p.header.worldFingerprint << std::dec << "\",\n"
        << "  \"capture_selection\":{\"control_step\":1,\"newton_iteration\":48},\n"
        << "  \"physical_transactions\":0,\n"
        << "  \"operator_path\":\"production nm_fem_apply_operator_elements plus nm_fgmres_gather_nodes; same captured candidate, accepted history, parameters, direction, and dt; useCache=1 versus useCache=0\",\n"
        << "  \"residual_convention\":\"captured pre-FGMRES RHS b minus replayed A*delta_v; free FEM xyz rows\",\n"
        << "  \"dt_seconds\":" << p.header.dt << ",\n"
        << "  \"environment_0\":{\"action_l2_Ns\":" << env0.actionL2
        << ",\"fresh_action_l2_Ns\":" << env0.otherActionL2
        << ",\"cached_vs_fresh_action_difference_l2_Ns\":" << env0.cachedFreshDifferenceL2
        << ",\"cached_vs_fresh_action_difference_relative_to_cached\":"
        << (env0.actionL2 > 0.0 ? env0.cachedFreshDifferenceL2 / env0.actionL2 : 0.0)
        << ",\"cached_vs_captured_action_difference_l2_Ns\":" << env0.cachedCapturedDifferenceL2
        << ",\"fresh_vs_captured_action_difference_l2_Ns\":" << env0.freshCapturedDifferenceL2
        << ",\"cached_rhs_minus_Ax_l2_Ns\":" << env0.residualL2
        << ",\"fresh_rhs_minus_Ax_l2_Ns\":" << env0.otherResidualL2
        << ",\"cached_rhs_minus_Ax_all_rows_l2_Ns\":" << env0.residualL2AllRows
        << ",\"fresh_rhs_minus_Ax_all_rows_l2_Ns\":" << env0.otherResidualL2AllRows
        << ",\"rhs_l2_Ns\":" << env0.bL2
        << ",\"correction_l2_m_per_s\":" << env0.directionL2
        << ",\"maximum_cached_action_component_Ns\":" << env0.maxAction
        << ",\"maximum_cached_fresh_component_difference_Ns\":" << env0.maxCacheFreshDifference
        << ",\"free_nodes\":" << env0.freeNodes << "},\n"
        << "  \"environment_1\":{\"action_l2_Ns\":" << env1.actionL2
        << ",\"fresh_action_l2_Ns\":" << env1.otherActionL2
        << ",\"cached_vs_fresh_action_difference_l2_Ns\":" << env1.cachedFreshDifferenceL2
        << ",\"cached_vs_captured_action_difference_l2_Ns\":" << env1.cachedCapturedDifferenceL2
        << ",\"fresh_vs_captured_action_difference_l2_Ns\":" << env1.freshCapturedDifferenceL2
        << ",\"cached_rhs_minus_Ax_l2_Ns\":" << env1.residualL2
        << ",\"fresh_rhs_minus_Ax_l2_Ns\":" << env1.otherResidualL2
        << ",\"rhs_l2_Ns\":" << env1.bL2
        << ",\"correction_l2_m_per_s\":" << env1.directionL2
        << ",\"free_nodes\":" << env1.freeNodes << "},\n"
        << "  \"limitations\":[\"Rejected Newton candidate; diagnostic only, not an accepted physical state.\",\"Zero rigid proxies/contact pairs/fields/cohesive faces in this compiled world; this is the complete mechanical FEM operator slice for this source-bound case.\",\"No noise floor or solver convergence claim is inferred.\"]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            need(argc == 4 || argc == 5,
                 "usage: cached_operator_replay <compiled.nmatterpack> <replay-input.bin> <output-dir> [--preflight-only]");
            const bool preflightOnly = argc == 5;
            need(!preflightOnly || std::string(argv[4]) == "--preflight-only",
                 "the only supported optional argument is --preflight-only");
            testExactFiniteFloatWords();
            const std::string packagePath = argv[1];
            const std::string payloadPath = argv[2];
            const std::string outputDir = argv[3];
            Payload p = readPayload(payloadPath);
            CompiledWorld world;
            std::string error;
            need(readPackage(packagePath, world, nullptr, &error),
                 "compiled package rejected: " + error);
            checkCompiledBinding(world, p);
            if (preflightOnly) {
                std::cout << "compiled package and captured immutable tetrahedron fields match "
                          << "for " << (p.header.environments * p.header.tetrahedra)
                          << " records; exact-float sentinel tests pass; "
                             "no Metal device or pipeline was created\n";
                return 0;
            }

            Device device;
            auto femOperator = device.makePipeline(
                @"numi_matter_metal::nm_fem_apply_operator_elements");
            auto gatherNodes = device.makePipeline(
                @"numi_matter_metal::nm_fgmres_gather_nodes");
            id<MTLFunction> layout = [device.library
                newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
            need(layout != nil, "production primal contact argument layout missing");
            id<MTLArgumentEncoder> contactEncoder =
                [layout newArgumentEncoderWithBufferIndex:0u];
            need(contactEncoder != nil, "cannot construct zero-contact arguments");

            NativeResult cached = evaluateNative(device, femOperator, gatherNodes,
                world, p, contactEncoder, 1u);
            NativeResult fresh = evaluateNative(device, femOperator, gatherNodes,
                world, p, contactEncoder, 0u);
            NativeResult mass = evaluateNative(device, femOperator, gatherNodes,
                world, p, contactEncoder, 1u, true);
            const Metrics env0 = metricsFor(0u, p, cached, fresh, fresh.action);
            const Metrics env1 = metricsFor(1u, p, cached, fresh, fresh.action);
            std::filesystem::create_directories(outputDir);
            writeFloat4File(outputDir + "/cached-action.float4.bin", cached.action);
            writeFloat4File(outputDir + "/fresh-action.float4.bin", fresh.action);

            writeElementFile(outputDir + "/cached-elements.float4x4.bin", cached.elements);
            writeElementFile(outputDir + "/fresh-elements.float4x4.bin", fresh.elements);
            writeFloat4File(outputDir + "/mass-only-action.float4.bin", mass.action);
            (void)env1; // Historical captured action belongs to original Newton direction and is deliberately not compared.

            std::ofstream summary(outputDir + "/residual-sweep.jsonl");
            summary << std::setprecision(17);
            const float epsilons[]={0.0f,1.0f,0.5f,0.25f,0.125f,0.0625f,0.03125f,2.0f};
            for(std::size_t k=0;k<std::size(epsilons);++k)for(unsigned side=0;side<(k==0?1u:2u);++side){
                auto r=evaluateResidual(device,world,p,epsilons[k],side!=0u);
                const auto prefix=outputDir+"/sample-"+std::to_string(k)+"-"+std::to_string(side);
                const auto* nodeWords=reinterpret_cast<const nm_float4*>(r.nodes.data());
                writeFloat4File(prefix+"-nodes.bin",std::vector<nm_float4>(nodeWords,nodeWords+r.nodes.size()*5u));
                writeFloat4File(prefix+"-residual.bin",r.residual);
                writeElementFile(prefix+"-forces.bin",r.forces);
                summary<<"{\"sample\":"<<k<<",\"side\":"<<side<<",\"epsilon\":"<<epsilons[k]<<",\"status\":["<<r.statuses[0].code<<","<<r.statuses[1].code<<"],\"history_unchanged\":true}\n";
                summary.flush();
            }

            std::cout << "native alternative direction/residual replay complete; transactions=0; "
                      << "env0 cache/fresh relative action delta="
                      << (env0.actionL2 > 0.0 ? env0.cachedFreshDifferenceL2 / env0.actionL2 : 0.0)
                      << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "native cached operator replay rejected: " << error.what() << "\n";
            return 2;
        }
    }
}
