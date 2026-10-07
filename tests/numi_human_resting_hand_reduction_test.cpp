#include "metalrobo/NumiHumanRestingHandReduction.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
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
    source.records[0].indices = {10,9,MR_INVALID_INDEX,MR_INVALID_INDEX};
    source.records[1].indices = {11,10,MR_INVALID_INDEX,MR_INVALID_INDEX};
    std::vector<std::uint32_t> coupledDependents;
    for (std::uint32_t q = 7; q <= 42; ++q)
        if (q != 10 && q != 11) coupledDependents.push_back(q);
    for (std::uint32_t q = 63; q <= 77; ++q)
        coupledDependents.push_back(q);
    require(coupledDependents.size() == 49, "fixture coupled row count");
    for (std::size_t i = 0; i < coupledDependents.size(); ++i) {
        const std::uint32_t q = coupledDependents[i];
        const std::uint32_t masterQ = 126u + static_cast<std::uint32_t>(i % 3u);
        source.records[i + 2u].indices = {q, q - 1u, masterQ, masterQ - 1u};
    }
    std::vector<float> q(129, 0.f);
    std::vector<MRDofPropertiesGPU> dofs(128);
    for (std::uint32_t v = 6; v < 128; ++v) {
        dofs[v].qIndex = v + 1; dofs[v].vIndex = v;
        dofs[v].articulationIndex = 0;
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

    std::vector<MRDofPropertiesGPU> eliminatedDofs;
    std::vector<NumiHumanRestingFixedBoundReceiptRecord> receipt;
    require(compileNumiHumanRestingFixedBoundElimination(
                source, q, dofs, output, eliminatedDofs, receipt, error),
            "fixed-bound elimination rejected admitted program");
    require(error.empty() && receipt.size() == 42, "fixed-bound receipt count/error");
    require(std::memcmp(source.records.data(), output.data(), 51*sizeof(output[0])) == 0,
            "elimination changed source equality rows");
    std::size_t eliminatedCount = 0, retainedCount = 0;
    for (const auto& record : receipt) {
        require(record.originalFlags == MR_DOF_FLAG_POSITION_LIMIT,
                "receipt lost original position-limit flag");
        require(record.sourceLower == -1 && record.sourceUpper == 1,
                "receipt changed source interval");
        if (record.action == static_cast<std::uint32_t>(
                NumiHumanRestingFixedBoundAction::eliminated)) {
            ++eliminatedCount;
            require(record.derivedFlags == 0, "implied bound flag was not cleared");
        } else {
            ++retainedCount;
            require(false, "inside-interval fixture retained a fixed bound");
        }
    }
    require(eliminatedCount == 42 && retainedCount == 0,
            "implied bound actions");
    for (std::uint32_t v = 0; v < 128; ++v) {
        const bool hand = (v >= 42 && v <= 61) || (v >= 80 && v <= 99);
        const bool sourceFixed = v == 9 || v == 10;
        const bool coupled = std::find(coupledDependents.begin(),
            coupledDependents.end(), dofs[v].qIndex) != coupledDependents.end();
        if (hand || sourceFixed)
            require((eliminatedDofs[v].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0,
                    "fixed coordinate bound remains");
        if (coupled)
            require((eliminatedDofs[v].flags & MR_DOF_FLAG_POSITION_LIMIT) != 0,
                    "coupled dependent bound was eliminated");
    }

    auto outsideSource = source;
    outsideSource.records[0].referencesAndCoefficients0.x = 3.0f;
    std::vector<MRNumiHumanJointEqualityGPU> outsideEqualities;
    require(compileNumiHumanRestingHandReduction(
                outsideSource, q, dofs, outsideEqualities, error),
            "outside-interval equality fixture rejected before retention test");
    std::vector<MRDofPropertiesGPU> outsideDofs;
    std::vector<NumiHumanRestingFixedBoundReceiptRecord> outsideReceipt;
    require(compileNumiHumanRestingFixedBoundElimination(
                outsideSource, q, dofs, outsideEqualities, outsideDofs,
                outsideReceipt, error),
            "outside-interval bound should remain available to the solver");
    require((outsideDofs[9].flags & MR_DOF_FLAG_POSITION_LIMIT) != 0,
            "outside-interval fixed bound was cleared");
    require(outsideReceipt.size() == 42 &&
            outsideReceipt[0].vIndex == 9 &&
            outsideReceipt[0].action == static_cast<std::uint32_t>(
                NumiHumanRestingFixedBoundAction::retainedOutsideSourceInterval),
            "outside-interval retention was not recorded");
    require(outsideReceipt[0].derivedFlags == outsideReceipt[0].originalFlags,
            "outside-interval retained flags changed");

    const auto preservesOutputsOnFailure = [&](const auto& invalidSource,
            const auto& invalidEqualities, const auto& invalidDofs,
            const char* reason) {
        auto untouchedDofs = invalidDofs;
        std::vector<NumiHumanRestingFixedBoundReceiptRecord> untouchedReceipt(1);
        untouchedReceipt[0].vIndex = 77u;
        const auto beforeReceipt = untouchedReceipt;
        const bool accepted = compileNumiHumanRestingFixedBoundElimination(
            invalidSource, q, invalidDofs, invalidEqualities, untouchedDofs,
            untouchedReceipt, error);
        require(!accepted, reason);
        require(untouchedDofs.size() == invalidDofs.size() &&
                std::memcmp(untouchedDofs.data(), invalidDofs.data(),
                    invalidDofs.size()*sizeof(MRDofPropertiesGPU)) == 0,
                "failed elimination mutated the model DoFs");
        require(untouchedReceipt.size() == beforeReceipt.size() &&
                std::memcmp(untouchedReceipt.data(), beforeReceipt.data(),
                    sizeof(beforeReceipt[0])) == 0,
                "failed elimination mutated the receipt");
        require(!error.empty(), "failed elimination omitted a reason");
    };

    auto zeroWidthDofs = dofs;
    zeroWidthDofs[9].limits = {0.0f, 0.0f, 0.0f, 0.0f};
    preservesOutputsOnFailure(source, output, zeroWidthDofs,
        "zero-width source interval was accepted for fixed-bound elimination");

    auto malformedSource = source;
    malformedSource.records[0].indices.y = 8u;
    std::vector<MRNumiHumanJointEqualityGPU> malformedEqualities;
    require(compileNumiHumanRestingHandReduction(
                malformedSource, q, dofs, malformedEqualities, error),
            "malformed q/v fixture did not reach elimination validation");
    preservesOutputsOnFailure(malformedSource, malformedEqualities, dofs,
        "mismatched fixed equality q/v indices were admitted");

    auto malformedCoupledSource = source;
    malformedCoupledSource.records[2].indices.w = MR_INVALID_INDEX;
    std::vector<MRNumiHumanJointEqualityGPU> malformedCoupledEqualities;
    require(compileNumiHumanRestingHandReduction(
                malformedCoupledSource, q, dofs, malformedCoupledEqualities, error),
            "malformed coupled fixture did not reach elimination validation");
    preservesOutputsOnFailure(malformedCoupledSource, malformedCoupledEqualities,
        dofs, "malformed fixed/coupled master pair was admitted");

    auto badCompiled = output;
    badCompiled.back().indices.y -= 1u;
    preservesOutputsOnFailure(source, badCompiled, dofs,
        "unrecognized equality program was admitted");

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
