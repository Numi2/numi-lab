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
            require(eliminatedDofs[record.vIndex].limits.x == 0.0f &&
                    eliminatedDofs[record.vIndex].limits.y == 0.0f,
                    "inactive derived position interval was not cleared");
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
    require(outsideDofs[9].limits.x == -1.0f &&
            outsideDofs[9].limits.y == 1.0f,
            "outside-interval source limits changed");

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
    // Source-specific linear POSITION_LIMIT consolidation fixture. The fake
    // rows exercise the compiler's exact topology gate and independent F32
    // endpoint logic; the runtime-only flag remains off in ordinary tests.
    auto linearSource = source;
    linearSource.records.assign(51u, MRNumiHumanJointEqualityGPU{});
    linearSource.records[13].indices = {
        10u, 9u, MR_INVALID_INDEX, MR_INVALID_INDEX};
    linearSource.records[14].indices = {
        11u, 10u, MR_INVALID_INDEX, MR_INVALID_INDEX};
    auto linearSourceDofs = dofs;
    for (std::uint32_t v = 0u; v < linearSourceDofs.size(); ++v) {
        linearSourceDofs[v].limits.z = 2.0f + static_cast<float>(v);
        linearSourceDofs[v].limits.w = 3.0f + static_cast<float>(v);
    }
    for (const auto& expected : detail::kNumiHumanRestingLinearBoundSourceRows) {
        auto& row = linearSource.records[expected.equalityIndex];
        row.indices = {expected.dependentQIndex,
            expected.dependentQIndex - 1u, expected.masterQIndex,
            expected.masterQIndex - 1u};
        const float slope = (expected.equalityIndex % 2u) == 0u
            ? 1.25f : -1.5f;
        row.referencesAndCoefficients0 = {0.0f, 0.0f, 0.0f, slope};
        row.coefficients1 = {0.0f, 0.0f, 0.0f, 0.0f};
        auto& dependent = linearSourceDofs[expected.dependentQIndex - 1u];
        dependent.limits.x = expected.equalityIndex == 1u ? -0.1f : -0.4f;
        dependent.limits.y = expected.equalityIndex == 1u ? 0.2f : 0.6f;
        auto& master = linearSourceDofs[expected.masterQIndex - 1u];
        master.limits.x = -2.0f;
        master.limits.y = 2.0f;
    }
    for (std::uint32_t i = 0u; i < 14u; ++i) {
        const std::uint32_t rowIndex = 37u + i;
        const std::uint32_t dependentQ = 104u + i;
        const std::uint32_t masterQ = 120u + (i % 8u);
        auto& row = linearSource.records[rowIndex];
        row.indices = {dependentQ, dependentQ - 1u,
            masterQ, masterQ - 1u};
        row.referencesAndCoefficients0 = {0.0f, 0.0f, 0.01f, 0.2f};
        row.coefficients1 = {0.1f, 0.0f, 0.0f, 0.0f};
    }
    const auto originalLinearSourceDofs = linearSourceDofs;
    const auto originalLinearSourceRows = linearSource.records;
    std::vector<MRNumiHumanJointEqualityGPU> linearEqualities;
    require(compileNumiHumanRestingHandReduction(
                linearSource, q, linearSourceDofs, linearEqualities, error),
            "linear-bound source fixture failed rigid-hand admission");
    const auto sourcePayloadSHA =
        kNumiHumanRestingReferenceNHEQPayloadSHA256;
    std::vector<MRDofPropertiesGPU> linearDerivedDofs;
    std::vector<NumiHumanRestingLinearBoundReceiptRecord> linearReceipt;
    require(compileNumiHumanRestingLinearBoundConsolidation(
                linearSource, sourcePayloadSHA, q, linearSourceDofs,
                linearSourceDofs, linearEqualities, linearDerivedDofs,
                linearReceipt, error),
            "valid linear source bounds were not consolidated");
    require(error.empty() && linearReceipt.size() == 35u,
            "linear-bound receipt count/error");
    require(std::memcmp(linearSourceDofs.data(),
                originalLinearSourceDofs.data(),
                linearSourceDofs.size() * sizeof(MRDofPropertiesGPU)) == 0 &&
            std::memcmp(linearSource.records.data(),
                originalLinearSourceRows.data(),
                linearSource.records.size() *
                    sizeof(MRNumiHumanJointEqualityGPU)) == 0,
            "linear-bound consolidation mutated source records or DoFs");
    std::array<std::uint8_t, 128u> seenLinearMasters{};
    std::size_t linearMasterCount = 0u;
    bool sawNegativeSlope = false;
    for (const auto& record : linearReceipt) {
        const auto& equality = linearSource.records[record.equalityIndex];
        const auto& originalDependent =
            linearSourceDofs[record.dependentVIndex];
        const auto& derivedDependent = linearDerivedDofs[record.dependentVIndex];
        const auto& originalMaster = linearSourceDofs[record.masterVIndex];
        const auto& derivedMaster = linearDerivedDofs[record.masterVIndex];
        sawNegativeSlope = sawNegativeSlope ||
            equality.referencesAndCoefficients0.w < 0.0f;
        require((derivedDependent.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u &&
                derivedDependent.limits.x == 0.0f &&
                derivedDependent.limits.y == 0.0f &&
                derivedDependent.limits.z == originalDependent.limits.z &&
                derivedDependent.limits.w == originalDependent.limits.w,
                "dependent limit consolidation changed non-position metadata");
        require(derivedMaster.flags == originalMaster.flags &&
                derivedMaster.limits.x == record.derivedMasterLower &&
                derivedMaster.limits.y == record.derivedMasterUpper &&
                derivedMaster.limits.z == originalMaster.limits.z &&
                derivedMaster.limits.w == originalMaster.limits.w,
                "master interval consolidation changed flags or velocity/effort bounds");
        if (seenLinearMasters[record.masterVIndex] == 0u) {
            seenLinearMasters[record.masterVIndex] = 1u;
            ++linearMasterCount;
        } else {
            require(derivedMaster.limits.x == record.derivedMasterLower &&
                    derivedMaster.limits.y == record.derivedMasterUpper,
                    "shared master did not receive one common intersection");
        }
        float lowerTarget = 0.0f;
        float upperTarget = 0.0f;
        require(detail::evaluateLinearSourceTargetF32(
                    equality, record.inverseMasterLower, lowerTarget) &&
                detail::evaluateLinearSourceTargetF32(
                    equality, record.inverseMasterUpper, upperTarget) &&
                lowerTarget >= record.sourceLower &&
                lowerTarget <= record.sourceUpper &&
                upperTarget >= record.sourceLower &&
                upperTarget <= record.sourceUpper,
                "inverse endpoint does not satisfy the exact source interval");
        if (record.inverseMasterLower > record.sourceMasterLower) {
            const float prior = std::nextafter(
                record.inverseMasterLower,
                -std::numeric_limits<float>::infinity());
            float target = 0.0f;
            require(detail::evaluateLinearSourceTargetF32(
                        equality, prior, target) &&
                    (target < record.sourceLower || target > record.sourceUpper),
                    "lower inverse endpoint has a preceding representable source-valid value");
        }
        if (record.inverseMasterUpper < record.sourceMasterUpper) {
            const float next = std::nextafter(
                record.inverseMasterUpper,
                std::numeric_limits<float>::infinity());
            float target = 0.0f;
            require(detail::evaluateLinearSourceTargetF32(
                        equality, next, target) &&
                    (target < record.sourceLower || target > record.sourceUpper),
                    "upper inverse endpoint has a following representable source-valid value");
        }
    }
    require(linearMasterCount == 7u && sawNegativeSlope,
            "linear-bound compiler missed shared or negative-slope masters");

    const auto rejectsLinearWithoutMutation = [&](const auto& invalidSource,
            const auto& invalidPayloadSHA, const auto& invalidDofSource,
            const auto& invalidEqualities, const char* reason) {
        auto untouchedDofs = linearDerivedDofs;
        std::vector<NumiHumanRestingLinearBoundReceiptRecord> untouchedReceipt(1u);
        untouchedReceipt[0].equalityIndex = 99u;
        untouchedReceipt[0].inverseMasterLower = -123.0f;
        const auto beforeDofs = untouchedDofs;
        const auto beforeReceipt = untouchedReceipt;
        const bool acceptedLinear = compileNumiHumanRestingLinearBoundConsolidation(
            invalidSource, invalidPayloadSHA, q, invalidDofSource,
            invalidDofSource, invalidEqualities, untouchedDofs,
            untouchedReceipt, error);
        require(!acceptedLinear, reason);
        require(untouchedDofs.size() == beforeDofs.size() &&
                std::memcmp(untouchedDofs.data(), beforeDofs.data(),
                    beforeDofs.size() * sizeof(MRDofPropertiesGPU)) == 0,
                "failed linear-bound compilation mutated derived DoFs");
        require(untouchedReceipt.size() == beforeReceipt.size() &&
                std::memcmp(untouchedReceipt.data(), beforeReceipt.data(),
                    sizeof(beforeReceipt[0])) == 0,
                "failed linear-bound compilation mutated its receipt");
        require(!error.empty(), "failed linear-bound compilation omitted a reason");
    };
    auto unknownPayloadSHA = sourcePayloadSHA;
    unknownPayloadSHA[0] ^= 1u;
    rejectsLinearWithoutMutation(linearSource, unknownPayloadSHA,
        linearSourceDofs, linearEqualities,
        "unknown NHEQ payload was admitted for bound consolidation");
    auto zeroSlopeSource = linearSource;
    zeroSlopeSource.records[0].referencesAndCoefficients0.w = 0.0f;
    std::vector<MRNumiHumanJointEqualityGPU> zeroSlopeEqualities;
    require(compileNumiHumanRestingHandReduction(
                zeroSlopeSource, q, linearSourceDofs,
                zeroSlopeEqualities, error),
            "zero-slope fixture failed before source-bound validation");
    rejectsLinearWithoutMutation(zeroSlopeSource, sourcePayloadSHA,
        linearSourceDofs, zeroSlopeEqualities,
        "zero-slope bounded dependency was admitted");
    auto nonfiniteSource = linearSource;
    nonfiniteSource.records[0].referencesAndCoefficients0.w =
        std::numeric_limits<float>::infinity();
    std::vector<MRNumiHumanJointEqualityGPU> nonfiniteEqualities;
    require(compileNumiHumanRestingHandReduction(
                nonfiniteSource, q, linearSourceDofs,
                nonfiniteEqualities, error),
            "nonfinite-slope fixture failed before source-bound validation");
    rejectsLinearWithoutMutation(nonfiniteSource, sourcePayloadSHA,
        linearSourceDofs, nonfiniteEqualities,
        "nonfinite bounded dependency was admitted");
    auto missingMasterDofs = linearSourceDofs;
    missingMasterDofs[6].flags &= ~MR_DOF_FLAG_POSITION_LIMIT;
    std::vector<MRNumiHumanJointEqualityGPU> missingMasterEqualities;
    require(compileNumiHumanRestingHandReduction(
                linearSource, q, missingMasterDofs,
                missingMasterEqualities, error),
            "missing-master fixture failed before bound validation");
    rejectsLinearWithoutMutation(linearSource, sourcePayloadSHA,
        missingMasterDofs, missingMasterEqualities,
        "source master without an active interval was admitted");
    auto chainedSource = linearSource;
    chainedSource.records[37].indices.x = 7u;
    chainedSource.records[37].indices.y = 6u;
    std::vector<MRNumiHumanJointEqualityGPU> chainedEqualities;
    require(compileNumiHumanRestingHandReduction(
                chainedSource, q, linearSourceDofs,
                chainedEqualities, error),
            "chained-master fixture failed before topology validation");
    rejectsLinearWithoutMutation(chainedSource, sourcePayloadSHA,
        linearSourceDofs, chainedEqualities,
        "a master that is itself a dependent was admitted");
    auto degenerateDofs = linearSourceDofs;
    const float smallest = std::numeric_limits<float>::denorm_min();
    degenerateDofs[6].limits.x = 0.0f;
    degenerateDofs[6].limits.y = 2.0f * smallest;
    degenerateDofs[11].limits.x = 1.0e-7f;
    degenerateDofs[11].limits.y = 1.5e-7f;
    auto degenerateSource = linearSource;
    degenerateSource.records[0].referencesAndCoefficients0.w = 1.0e38f;
    std::vector<MRNumiHumanJointEqualityGPU> degenerateEqualities;
    require(compileNumiHumanRestingHandReduction(
                degenerateSource, q, degenerateDofs,
                degenerateEqualities, error),
            "degenerate-interval fixture failed before endpoint validation");
    rejectsLinearWithoutMutation(degenerateSource, sourcePayloadSHA,
        degenerateDofs, degenerateEqualities,
        "single-F32 inverse interval was admitted as a valid source bound");

    std::cout << "resting_hand_reduction checks=" << checks << " passed\n";
}
