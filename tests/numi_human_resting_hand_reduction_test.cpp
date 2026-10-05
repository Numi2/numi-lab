#include "metalrobo/NumiHumanRestingHandReduction.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace metalrobo;
    std::size_t checks = 0;
    const auto require = [&](bool ok, const char* reason) {
        ++checks;
        if (!ok) throw std::runtime_error(reason);
    };
    NumiHumanJointEqualityPayload source;
    source.nq = 129; source.nv = 128;
    source.sourceSha256 = {0x28,0x0d,0x29,0x7a,0xa4,0x96,0xac,0xcc,
        0xf3,0xf1,0xc5,0x37,0x3a,0x13,0x04,0xd2,0x3f,0x95,0x69,0x36,
        0x2c,0x2d,0x69,0x60,0x91,0x01,0x28,0xbf,0xba,0x14,0x49,0x75};
    source.records.resize(51);
    for (auto& row : source.records) row.indices = {12,11,7,6};
    std::vector<float> q(129, 0.f);
    std::vector<MRDofPropertiesGPU> dofs(128);
    for (std::uint32_t v = 6; v < 128; ++v) {
        dofs[v].qIndex = v + 1; dofs[v].vIndex = v;
        dofs[v].flags = MR_DOF_FLAG_POSITION_LIMIT;
        dofs[v].limits = {-1,1,0,0};
    }
    q[43] = .15f; q[81] = -.2f;
    std::string error;
    std::vector<MRNumiHumanJointEqualityGPU> output;
    require(compileNumiHumanRestingHandReduction(source,q,dofs,output,error), "valid reduction rejected");
    require(output.size() == 91 && error.empty(), "reduction count/error");
    require(std::memcmp(source.records.data(), output.data(), 51*sizeof(output[0])) == 0,
            "source equalities changed");
    for (std::size_t i = 51; i < output.size(); ++i) {
        const auto& row = output[i];
        require(row.indices.x >= 43 && row.indices.y >= 42, "root/wrist constrained");
        require(row.indices.z == MR_INVALID_INDEX && row.indices.w == MR_INVALID_INDEX,
                "derived row is not an internal fixed coordinate");
        require(row.referencesAndCoefficients0.x == q[row.indices.x], "wrong reference target");
    }
    const auto accepted = output;
    const auto rejected = [&](const auto& s, const auto& state, const auto& coordinates) {
        require(!compileNumiHumanRestingHandReduction(s,state,coordinates,output,error), "invalid reduction admitted");
        require(output.size() == accepted.size() &&
            std::memcmp(output.data(),accepted.data(),output.size()*sizeof(output[0])) == 0 &&
            !error.empty(), "rejection changed output or omitted reason");
    };
    auto badSource = source; badSource.sourceSha256[0] ^= 1; rejected(badSource,q,dofs);
    badSource = source; badSource.nv = 127; rejected(badSource,q,dofs);
    badSource = source; badSource.records.pop_back(); rejected(badSource,q,dofs);
    badSource = source; badSource.records[0].indices.x = 43; rejected(badSource,q,dofs);
    badSource = source; badSource.records[0].indices.z = 81; rejected(badSource,q,dofs);
    auto badQ=q;badQ[43]=std::numeric_limits<float>::quiet_NaN();rejected(source,badQ,dofs);
    badQ=q;badQ[81]=2;rejected(source,badQ,dofs);
    auto badDofs=dofs;badDofs[42].qIndex=42;rejected(source,q,badDofs);
    badDofs=dofs;badDofs[80].flags=0;rejected(source,q,badDofs);
    std::cout << "resting_hand_reduction checks=" << checks << " passed\n";
}
