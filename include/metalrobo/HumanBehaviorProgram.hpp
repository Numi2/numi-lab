#pragma once
#include "metalrobo/human_behavior_gpu.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {
struct HumanBehaviorBodyBinding {
    std::uint32_t sourceBodyRecordIndex = 0, coreBodyIndex = 0;
    std::array<double,3> sourceOriginInOriginalCOMFrame{};
    std::array<double,4> sourceInertialQuaternionBody{0,0,0,1};
};
struct HumanBehaviorProgramSource {
    std::uint32_t bodyCount = 0, nq = 0, nv = 0, task = 0;
    std::uint64_t timestepNanoseconds = 0;
    // archive, rigid, Human manifest, source export, authored criteria, catalog
    std::array<std::array<std::uint8_t,32>,6> sourceSHA256{};
    std::array<HumanBehaviorBodyBinding,3> bodies{}; // root, trunk, velocity
    std::array<double,16> numerical{};
    std::vector<std::array<std::uint32_t,2>> forbiddenBodies;
};
struct HumanBehaviorCompileBinding {
    std::array<std::uint8_t,32> sourceArchiveSHA256{}, rigidSHA256{};
    std::uint32_t bodyCount = 0, nq = 0, nv = 0;
    std::uint64_t timestepNanoseconds = 0;
    std::vector<std::array<std::uint32_t,2>> sourceToCore;
    // Exact remainingCOMOffsetM from the mechanical owner's actual compiler.
    // Every body, including unchanged bodies, has an explicit vector.
    std::vector<std::array<double,3>> cookedCOMOffset;
};
struct CompiledHumanBehaviorProgram {
    MRHumanBehaviorProgramGPU gpu{};
    std::vector<std::uint32_t> forbiddenBodies;
    std::array<std::array<std::uint8_t,32>,6> sourceSHA256{};
    bool fullBehaviorEvidenceSupported = false; // owners must supply coverage
};
// Construction only; output is unchanged on failure. Strict length/fields.
[[nodiscard]] bool decodeHumanBehaviorProgram(std::span<const std::byte> bytes,
    HumanBehaviorProgramSource& output, std::string& error);
[[nodiscard]] bool compileHumanBehaviorProgram(const HumanBehaviorProgramSource& source,
    const HumanBehaviorCompileBinding& binding, CompiledHumanBehaviorProgram& output, std::string& error);
} // namespace metalrobo
