#pragma once
#include "metalrobo/engine_types.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace metalrobo {
// NHCNT1 retains fixed material witnesses. NHCNT2 stores authored spheres,
// capsules and ellipsoids. A capsule compiles to its two endpoint-sphere plane constraints:
// together they exactly bound the full primitive and retain a parallel line
// contact's two moment arms. contactCount below is the expanded row count.
struct NumiHumanSupportHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
    float groundPointX = 0.0f;
    float groundPointY = 0.0f;
    float groundPointZ = 0.0f;
    float groundNormalX = 0.0f;
    float groundNormalY = 0.0f;
    float groundNormalZ = 1.0f;
    float groundFriction = 0.0f;
};

struct NumiHumanSupportContact {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t sourceGeometryIndex = MR_INVALID_INDEX;
    float localPointX = 0.0f;
    float localPointY = 0.0f;
    float localPointZ = 0.0f;
    float worldWitnessX = 0.0f;
    float worldWitnessY = 0.0f;
    float worldWitnessZ = 0.0f;
    float friction = 0.0f;
    float defaultSignedPlaneDistance = 0.0f;
    float reserved0 = 0.0f;
    float reserved1 = 0.0f;
    float supportRadius = 0.0f;
    std::array<float, 3> supportRadii{};
    std::array<float, 4> supportOrientation{};
};


struct NumiHumanSupportPayload {
    NumiHumanSupportHeader header;
    std::vector<NumiHumanSupportContact> contacts;
};
MRArticulatedPointImpulseGPU compileNumiHumanSupportQuery(
    const NumiHumanSupportHeader&, const NumiHumanSupportContact&);
// Fail closed and preserve output on malformed/foreign/truncated payloads.
bool decodeNumiHumanSupportPayload(std::span<const std::byte> bytes,
    std::uint32_t expectedBodyCount,
    const std::array<std::uint8_t, 32>& expectedSource,
    NumiHumanSupportPayload& output, std::string& error);
} // namespace metalrobo
