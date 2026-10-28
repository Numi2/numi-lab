#include "NumiHumanConservativeSpatialMassTransport.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

static bool hasError(const ConservativeSpatialMassTransportInput& input,
                     std::string_view expected) {
    ConservativeSpatialMassTransportResult result;
    std::string error;
    return !assembleConservativeSpatialMassTransport(input, result, error) &&
           error.find(expected) != std::string::npos;
}

static ConservativeSpatialMassTransportInput fixtureInput() {
    static const std::vector<SpatialMassTransportPoint> parentPoints{
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0}, {0.0, 0.02, 0.0}, {0.02, 0.02, 0.0}};
    static const std::vector<double> parentVolumes{1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6};
    static const std::vector<double> parentMasses{1.0e-3, 1.0e-3, 9.0e-4, 9.0e-4};
    static const std::vector<std::uint32_t> parentMaterials{15u, 15u, 16u, 16u};
    static const std::vector<std::uint32_t> parentDonors{2u, 2u, 3u, 3u};
    static const std::vector<std::uint32_t> parentRegions{101u, 101u, 202u, 202u};
    static const std::vector<std::uint32_t> parentSourceOffsets{0u, 2u, 4u, 6u, 8u};
    static const std::vector<std::uint32_t> parentSourceDonors{54u, 61u, 54u, 61u,
                                                                54u, 61u, 54u, 61u};
    static const std::vector<double> parentSourceMasses{7.5e-4, 2.5e-4, 2.5e-4, 7.5e-4,
                                                        4.5e-4, 4.5e-4, 4.5e-4, 4.5e-4};
    static const std::vector<SpatialMassTransportPoint> childPoints{
        {0.0001, 0.0, 0.0}, {0.0099, 0.0, 0.0},
        {0.0001, 0.02, 0.0}, {0.0199, 0.02, 0.0}};
    static const std::vector<double> childVolumes{1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6};
    static const std::vector<std::uint32_t> childMaterials{15u, 15u, 16u, 16u};
    static const std::vector<std::uint32_t> childDonors{2u, 2u, 3u, 3u};
    static const std::vector<std::uint32_t> childRegions{101u, 101u, 202u, 202u};
    static const std::vector<double> lower = [] {
        std::vector<double> x(17u, std::numeric_limits<double>::quiet_NaN());
        x[15] = 900.0; x[16] = 850.0; return x;
    }();
    static const std::vector<double> upper = [] {
        std::vector<double> x(17u, std::numeric_limits<double>::quiet_NaN());
        x[15] = 1100.0; x[16] = 950.0; return x;
    }();
    static const std::string hashA(64u, '1'), hashB(64u, '2'), hashC(64u, '3'),
                             hashD(64u, '4'), hashE(64u, '5');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld = parentPoints;
    in.parentReferenceVolumeM3 = parentVolumes;
    in.parentMassKg = parentMasses;
    in.parentMaterialIndices = parentMaterials;
    in.parentDonorBodyIndices = parentDonors;
    in.parentAnatomicalRegionIndices = parentRegions;
    in.parentSourceDonorOffsets = parentSourceOffsets;
    in.parentSourceDonorBodyIndices = parentSourceDonors;
    in.parentSourceDonorMassKg = parentSourceMasses;
    in.childCentroidsWorld = childPoints;
    in.childReferenceVolumeM3 = childVolumes;
    in.childMaterialIndices = childMaterials;
    in.childDonorBodyIndices = childDonors;
    in.childAnatomicalRegionIndices = childRegions;
    in.materialDensityLowerKgM3 = lower;
    in.materialDensityUpperKgM3 = upper;
    in.parentMassPacketSHA256 = hashA;
    in.parentSourceContributionSHA256 = hashB;
    in.childMeshSHA256 = hashC;
    in.regionLabelSHA256 = hashD;
    in.coordinateFrameSHA256 = hashE;
    in.maximumTransportDistanceM = 0.02;
    return in;
}

int main() {
    auto input = fixtureInput();
    const std::vector<std::array<std::uint32_t, 4>> parentCellNodes{
        {0u, 1u, 2u, 3u}, {4u, 5u, 6u, 7u},
        {8u, 9u, 10u, 11u}, {12u, 13u, 14u, 15u}};
    const std::vector<std::array<std::uint32_t, 4>> cornerSourceDonors{
        {54u, 54u, 54u, 61u}, {54u, 61u, 61u, 61u},
        {54u, 54u, 61u, 61u}, {54u, 61u, 54u, 61u}};
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> nodeSourceRows;
    nodeSourceRows.reserve(16u);
    for (std::size_t cell = 0; cell < parentCellNodes.size(); ++cell)
        for (std::size_t corner = 0; corner < 4u; ++corner)
            nodeSourceRows.push_back({.parentNodeIndex = parentCellNodes[cell][corner],
                .materialIndex = input.parentMaterialIndices[cell],
                .sourceDonorBodyIndex = cornerSourceDonors[cell][corner],
                .massKg = input.parentMassKg[cell] * 0.25});
    std::vector<std::uint32_t> reconstructedOffsets, reconstructedDonors;
    std::vector<double> reconstructedMasses;
    std::string error;
    if (!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            input.parentMassKg, input.parentMaterialIndices, parentCellNodes,
            nodeSourceRows, input.absoluteMassToleranceKg, input.relativeClosureTolerance,
            reconstructedOffsets, reconstructedDonors, reconstructedMasses, error) ||
        reconstructedOffsets != std::vector<std::uint32_t>(input.parentSourceDonorOffsets.begin(),
                                                           input.parentSourceDonorOffsets.end()) ||
        reconstructedDonors != std::vector<std::uint32_t>(input.parentSourceDonorBodyIndices.begin(),
                                                           input.parentSourceDonorBodyIndices.end()) ||
        reconstructedMasses != std::vector<double>(input.parentSourceDonorMassKg.begin(),
                                                    input.parentSourceDonorMassKg.end())) {
        std::cerr << "actual node source rows did not reconstruct deterministic cell donor splits: "
                  << error << '\n';
        return 21;
    }
    auto missingNodeSource = nodeSourceRows;
    missingNodeSource.pop_back();
    if (assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            input.parentMassKg, input.parentMaterialIndices, parentCellNodes,
            missingNodeSource, input.absoluteMassToleranceKg, input.relativeClosureTolerance,
            reconstructedOffsets, reconstructedDonors, reconstructedMasses, error) ||
        error.find("omits a cell corner owner") == std::string::npos) {
        std::cerr << "source-donor reconstruction accepted a missing node owner\n";
        return 22;
    }
    auto splitNodeSource = nodeSourceRows;
    const auto firstNode = splitNodeSource.front();
    splitNodeSource.front().massKg *= 0.65;
    auto secondOwner = firstNode;
    secondOwner.sourceDonorBodyIndex = 99u;
    secondOwner.massKg *= 0.35;
    splitNodeSource.push_back(secondOwner);
    if (!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            input.parentMassKg, input.parentMaterialIndices, parentCellNodes,
            splitNodeSource, input.absoluteMassToleranceKg, input.relativeClosureTolerance,
            reconstructedOffsets, reconstructedDonors, reconstructedMasses, error) ||
        reconstructedOffsets[1] != 3u || reconstructedDonors[0] != 54u ||
        reconstructedDonors[1] != 61u || reconstructedDonors[2] != 99u) {
        std::cerr << "source-donor reconstruction did not preserve supplied multi-donor proportions: "
                  << error << "\n";
        return 23;
    }
    ConservativeSpatialMassTransportResult result, repeated;
    if (!assembleConservativeSpatialMassTransport(input, result, error)) {
        std::cerr << "valid spatial transport fixture rejected: " << error << '\n';
        return 1;
    }
    if (result.childMassKg.size() != 4u || result.transportEdgeCount != 4u ||
        result.groupClosure.size() != 2u || result.childParentRows.size() != 4u ||
        result.childSourceDonorOffsets.size() != 5u ||
        result.childSourceDonorBodyIndices.size() != 8u ||
        result.sourceDonorClosure.size() != 4u ||
        result.maximumSourceDonorMassClosureErrorKg != 0.0 ||
        result.receiptSHA256.size() != 64u ||
        result.maximumParentMassClosureErrorKg != 0.0 ||
        result.maximumParentMeasureClosureErrorM3 != 0.0 ||
        result.maximumChildFractionSumError > 1.0e-14 ||
        result.maximumTransportDistanceUsedM > input.maximumTransportDistanceM) {
        std::cerr << "spatial transport fixture closure or boundedness failed"
                  << " cells=" << result.childMassKg.size()
                  << " edges=" << result.transportEdgeCount
                  << " groups=" << result.groupClosure.size()
                  << " rows=" << result.childParentRows.size()
                  << " sha=" << result.receiptSHA256.size()
                  << " parent_mass_err=" << result.maximumParentMassClosureErrorKg
                  << " parent_measure_err=" << result.maximumParentMeasureClosureErrorM3
                  << " fraction_err=" << result.maximumChildFractionSumError
                  << " max_distance=" << result.maximumTransportDistanceUsedM << '\n';
        return 2;
    }
    for (std::size_t c = 0; c < result.childMassKg.size(); ++c) {
        if (result.childParentOffsets[c + 1u] - result.childParentOffsets[c] != 1u ||
            result.childParentRows[result.childParentOffsets[c]] != c ||
            std::abs(result.childMassKg[c] - input.parentMassKg[c]) > 1.0e-15 ||
            std::abs(result.childImpliedDensityKgM3[c] -
                     input.parentMassKg[c] / input.childReferenceVolumeM3[c]) > 1.0e-10) {
            std::cerr << "spatial transport failed deterministic nearest bounded partition at child " << c << '\n';
            return 3;
        }
    }
    if (!assembleConservativeSpatialMassTransport(input, repeated, error) ||
        !verifyConservativeSpatialMassTransport(input, result, error) ||
        !verifyRecordedConservativeSpatialMassTransport(input, result, error) ||
        repeated.receiptSHA256 != result.receiptSHA256 ||
        repeated.childParentRows != result.childParentRows ||
        repeated.childParentTransportMassKg != result.childParentTransportMassKg) {
        std::cerr << "spatial transport receipt or flow is not deterministic\n";
        return 4;
    }
    auto alteredWorkReceipt = result;
    ++alteredWorkReceipt.augmentationCount;
    if (verifyRecordedConservativeSpatialMassTransport(input, alteredWorkReceipt, error) ||
        error.find("receipt digest") == std::string::npos) {
        std::cerr << "recorded flow receipt did not bind the solver work count\n";
        return 24;
    }
    for (std::size_t c = 0; c < result.childMassKg.size(); ++c) {
        double sourceMass = 0.0;
        std::uint32_t previousDonor = 0u;
        for (std::uint32_t row = result.childSourceDonorOffsets[c];
             row < result.childSourceDonorOffsets[c + 1u]; ++row) {
            if (row > result.childSourceDonorOffsets[c] &&
                result.childSourceDonorBodyIndices[row] <= previousDonor) {
                std::cerr << "child source-donor lineage rows are not sorted/unique\n";
                return 16;
            }
            previousDonor = result.childSourceDonorBodyIndices[row];
            sourceMass += result.childSourceDonorMassKg[row];
        }
        if (std::abs(sourceMass - result.childMassKg[c]) > 1.0e-15) {
            std::cerr << "child source-donor lineage does not close child mass\n";
            return 17;
        }
    }
    for (const auto& closure : result.sourceDonorClosure) {
        if (closure.parentMassKg != closure.childMassKg) {
            std::cerr << "source-donor mass closure failed\n";
            return 18;
        }
    }
    for (const auto& closure : result.groupClosure) {
        if (closure.parentMassKg != closure.childMassKg ||
            closure.childMassKg < closure.childMinimumCapacityKg ||
            closure.childMassKg > closure.childMaximumCapacityKg) {
            std::cerr << "material/donor/region group closure failed\n";
            return 5;
        }
    }

    auto tampered = result;
    tampered.childParentTransportMassKg[0] *= 0.5;
    if (verifyConservativeSpatialMassTransport(input, tampered, error)) {
        std::cerr << "spatial transport admitted a modified flow under the original receipt\n";
        return 15;
    }
    tampered = result;
    tampered.childSourceDonorMassKg[0] *= 0.5;
    if (verifyConservativeSpatialMassTransport(input, tampered, error) ||
        verifyRecordedConservativeSpatialMassTransport(input, tampered, error)) {
        std::cerr << "spatial transport admitted modified child source-donor lineage\n";
        return 19;
    }

    tampered = result;
    tampered.childParentTransportDistanceM[0] += 1.0e-6;
    if (verifyRecordedConservativeSpatialMassTransport(input, tampered, error)) {
        std::cerr << "recorded-flow admission accepted a distance inconsistent with current cell centroids\n";
        return 24;
    }

    auto changed = fixtureInput();
    auto changedPoint = std::vector<SpatialMassTransportPoint>(changed.childCentroidsWorld.begin(),
                                                               changed.childCentroidsWorld.end());
    changedPoint[0][0] += 0.00001;
    changed.childCentroidsWorld = changedPoint;
    if (verifyRecordedConservativeSpatialMassTransport(changed, result, error)) {
        std::cerr << "recorded-flow admission accepted altered current arrays under the old receipt\n";
        return 25;
    }
    ConservativeSpatialMassTransportResult changedResult;
    if (!assembleConservativeSpatialMassTransport(changed, changedResult, error) ||
        changedResult.receiptSHA256 == result.receiptSHA256) {
        std::cerr << "transport receipt failed to bind child coordinates\n";
        return 6;
    }

    auto regionMismatch = fixtureInput();
    auto regions = std::vector<std::uint32_t>(regionMismatch.childAnatomicalRegionIndices.begin(),
                                               regionMismatch.childAnatomicalRegionIndices.end());
    regions[1] = 999u;
    regionMismatch.childAnatomicalRegionIndices = regions;
    if (!hasError(regionMismatch, "child anatomical/material/donor group has no parent supply")) {
        std::cerr << "transport crossed or silently broadened an anatomical region\n";
        return 7;
    }

    auto donorMismatch = fixtureInput();
    auto donors = std::vector<std::uint32_t>(donorMismatch.childDonorBodyIndices.begin(),
                                              donorMismatch.childDonorBodyIndices.end());
    donors[1] = 77u;
    donorMismatch.childDonorBodyIndices = donors;
    if (!hasError(donorMismatch, "child anatomical/material/donor group has no parent supply")) {
        std::cerr << "transport crossed or silently broadened a source donor\n";
        return 8;
    }

    auto distanceMismatch = fixtureInput();
    distanceMismatch.maximumTransportDistanceM = 1.0e-5;
    if (!hasError(distanceMismatch, "no same-material/donor/region child within")) {
        std::cerr << "transport exceeded its bounded distance or fell back to nearest outside the bound\n";
        return 9;
    }

    auto densityMismatch = fixtureInput();
    auto childVolumes = std::vector<double>(densityMismatch.childReferenceVolumeM3.begin(),
                                             densityMismatch.childReferenceVolumeM3.end());
    childVolumes[0] = 0.5e-6;
    childVolumes[1] = 0.5e-6;
    densityMismatch.childReferenceVolumeM3 = childVolumes;
    if (!hasError(densityMismatch, "outside child volume times actual material density bounds")) {
        std::cerr << "transport renormalized mass beyond actual density capacities\n";
        return 10;
    }

    auto edgeBudget = fixtureInput();
    edgeBudget.maximumAllowedEdges = 1u;
    if (!hasError(edgeBudget, "allowed-edge count exceeds its pinned budget")) {
        std::cerr << "transport silently truncated its allowed-edge set\n";
        return 11;
    }

    auto badSourceSplit = fixtureInput();
    auto invalidSourceMasses = std::vector<double>(badSourceSplit.parentSourceDonorMassKg.begin(),
                                                   badSourceSplit.parentSourceDonorMassKg.end());
    invalidSourceMasses[0] *= 0.5;
    badSourceSplit.parentSourceDonorMassKg = invalidSourceMasses;
    if (!hasError(badSourceSplit, "do not close the authoritative cell mass")) {
        std::cerr << "transport accepted a source-donor split that misses parent cell mass\n";
        return 20;
    }

    auto badHash = fixtureInput();
    badHash.coordinateFrameSHA256 = "not-a-sha";
    if (!hasError(badHash, "requires pinned parent, source, child, region and coordinate-frame")) {
        std::cerr << "transport accepted missing coordinate-frame provenance\n";
        return 12;
    }

    auto nonuniform = fixtureInput();
    auto nonuniformChild = std::vector<SpatialMassTransportPoint>(nonuniform.childCentroidsWorld.begin(),
                                                                  nonuniform.childCentroidsWorld.end());
    nonuniformChild[0] = {0.0, 0.0, 0.0};
    nonuniformChild[1] = {0.0, 0.0, 0.0};
    nonuniform.childCentroidsWorld = nonuniformChild;
    nonuniform.maximumTransportDistanceM = 0.02;
    nonuniform.maximumAllowedEdges = 100u;
    ConservativeSpatialMassTransportResult splitResult;
    if (!assembleConservativeSpatialMassTransport(nonuniform, splitResult, error) ||
        splitResult.childParentRows.size() < 4u ||
        splitResult.maximumParentMassClosureErrorKg > 1.0e-13) {
        std::cerr << "feasible coupled transport failed: " << error << '\n';
        return 13;
    }
    for (const auto& closure : splitResult.groupClosure)
        if (std::abs(closure.parentMassKg - closure.childMassKg) > 1.0e-13) {
            std::cerr << "split transport lost material/donor/region mass\n";
            return 14;
        }

    // Total group capacity can be valid while a spatial component has too
    // little supply for its density floor. Keep that partition in the failure
    // witness and leave the prior accepted result unchanged.
    auto disconnected = fixtureInput();
    const std::vector<double> disconnectedVolumes{1.2e-6,0.8e-6,1.0e-6,1.0e-6};
    disconnected.childReferenceVolumeM3 = disconnectedVolumes;
    disconnected.maximumTransportDistanceM = 0.001;
    auto untouched = result;
    const auto previousReceipt = untouched.receiptSHA256;
    if (assembleConservativeSpatialMassTransport(disconnected,untouched,error) ||
        error.find("same-material/donor/region and distance bounds") == std::string::npos ||
        error.find("material=15 donor=2 region=101") == std::string::npos ||
        error.find("supply_kg=") == std::string::npos ||
        error.find("sent_kg=") == std::string::npos ||
        error.find("distance_m=0.001") == std::string::npos ||
        untouched.receiptSHA256 != previousReceipt ||
        untouched.childMassKg != result.childMassKg) {
        std::cerr << "infeasible spatial partition lost its witness or mutated output: " << error << '\n';
        return 19;
    }

    std::cout << "conservative spatial mass transport PASS"
              << " cells=4 edges=" << result.transportEdgeCount
              << " groups=" << result.groupClosure.size()
              << " min_cost_distance_m=" << result.massWeightedMeanTransportDistanceM
              << " parent_mass_closure_kg=" << result.maximumParentMassClosureErrorKg
              << " parent_measure_closure_m3=" << result.maximumParentMeasureClosureErrorM3
              << " source_donor_closure_kg=" << result.maximumSourceDonorMassClosureErrorKg
              << " density_range_kg_m3=" << result.minimumChildImpliedDensityKgM3
              << ':' << result.maximumChildImpliedDensityKgM3
              << " receipt_sha256=" << result.receiptSHA256 << '\n';
    return 0;
}
