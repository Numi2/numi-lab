#include "NumiHumanConservativeSpatialMassTransport.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

struct OwnedSparseFixture {
    std::vector<SpatialMassTransportPoint> parentPoints, childPoints;
    std::vector<double> parentVolumes, parentMasses, childVolumes;
    std::vector<std::uint32_t> parentMaterials, parentDonors, parentRegions;
    std::vector<std::uint32_t> sourceOffsets, sourceDonors;
    std::vector<double> sourceMasses;
    std::vector<std::uint32_t> childMaterials, childDonors, childRegions;
    std::vector<double> lower, upper;
    std::string parentHash = std::string(64u, '1');
    std::string sourceHash = std::string(64u, '2');
    std::string childHash = std::string(64u, '3');
    std::string labelHash = std::string(64u, '4');
    std::string frameHash = std::string(64u, '5');
    double radius = 1.0;
    std::size_t edgeBudget = 100u;
    std::size_t augmentationBudget = 300000u;

    OwnedSparseFixture(std::vector<SpatialMassTransportPoint> p,
                       std::vector<SpatialMassTransportPoint> c, double r)
        : parentPoints(std::move(p)), childPoints(std::move(c)),
          parentVolumes(parentPoints.size(), 1.0e-6),
          parentMasses(parentPoints.size(), 1.0e-3),
          parentMaterials(parentPoints.size(), 15u),
          parentDonors(parentPoints.size(), 2u),
          parentRegions(parentPoints.size(), 101u),
          sourceOffsets(parentPoints.size() + 1u), sourceDonors(parentPoints.size(), 54u),
          sourceMasses(parentPoints.size(), 1.0e-3),
          childMaterials(childPoints.size(), 15u),
          childDonors(childPoints.size(), 2u),
          childRegions(childPoints.size(), 101u),
          lower(17u, std::numeric_limits<double>::quiet_NaN()),
          upper(17u, std::numeric_limits<double>::quiet_NaN()), radius(r) {
        childVolumes.assign(childPoints.size(), 1.0e-6);
        for (std::size_t i = 0; i < sourceOffsets.size(); ++i)
            sourceOffsets[i] = static_cast<std::uint32_t>(i);
        lower[15] = 900.0;
        upper[15] = 1100.0;
    }

    ConservativeSpatialMassTransportInput view() const {
        ConservativeSpatialMassTransportInput in;
        in.parentCentroidsWorld = parentPoints;
        in.parentReferenceVolumeM3 = parentVolumes;
        in.parentMassKg = parentMasses;
        in.parentMaterialIndices = parentMaterials;
        in.parentDonorBodyIndices = parentDonors;
        in.parentAnatomicalRegionIndices = parentRegions;
        in.parentSourceDonorOffsets = sourceOffsets;
        in.parentSourceDonorBodyIndices = sourceDonors;
        in.parentSourceDonorMassKg = sourceMasses;
        in.childCentroidsWorld = childPoints;
        in.childReferenceVolumeM3 = childVolumes;
        in.childMaterialIndices = childMaterials;
        in.childDonorBodyIndices = childDonors;
        in.childAnatomicalRegionIndices = childRegions;
        in.materialDensityLowerKgM3 = lower;
        in.materialDensityUpperKgM3 = upper;
        in.parentMassPacketSHA256 = parentHash;
        in.parentSourceContributionSHA256 = sourceHash;
        in.childMeshSHA256 = childHash;
        in.regionLabelSHA256 = labelHash;
        in.coordinateFrameSHA256 = frameHash;
        in.maximumTransportDistanceM = radius;
        in.maximumAllowedEdges = edgeBudget;
        in.maximumAugmentations = augmentationBudget;
        return in;
    }
};

static double normalizedSquaredTransportCost(
        const ConservativeSpatialMassTransportResult& result, double radius) {
    double cost = 0.0;
    for (std::size_t c = 0; c + 1u < result.childParentOffsets.size(); ++c) {
        for (std::uint32_t row = result.childParentOffsets[c];
             row < result.childParentOffsets[c + 1u]; ++row) {
            const double d = result.childParentTransportDistanceM[row];
            cost += result.childParentTransportMassKg[row] * d*d / (radius*radius);
        }
    }
    return cost;
}

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
    {
        struct OrderedFlow {
            std::uint32_t parent = 0u, child = 0u;
            double massKg = 0.0, measureM3 = 0.0;
        };
        std::vector<OrderedFlow> inserted{
            {2u, 0u, 1.0e-3, 1.0e-6},
            {0u, 0u, 1.0e-19, 1.0e-22},
            {1u, 0u, 1.0e-19, 1.0e-22}};
        const double insertionOrderMass =
            (inserted[0].massKg + inserted[1].massKg) + inserted[2].massKg;
        std::vector<double> replayMass, replayMeasure;
        if (!spatial_mass_transport_detail::sortAndAccumulateChildFlowSums(
                inserted, 1u, replayMass, replayMeasure) ||
            inserted[0].parent != 0u || inserted[1].parent != 1u ||
            inserted[2].parent != 2u || replayMass[0] != 1.0e-19 + 1.0e-19 + 1.0e-3 ||
            replayMeasure[0] != 1.0e-22 + 1.0e-22 + 1.0e-6 ||
            replayMass[0] == insertionOrderMass) {
            std::cerr << "sparse child-flow accumulation did not match canonical CSR order "
                         "for three unequal, noncanonically inserted contributions\n";
            return 58;
        }
    }
    {
        using spatial_mass_transport_detail::EdgeRef;
        using spatial_mass_transport_detail::MinCostFlow;
        constexpr double tinyChildMass = 2.7976994668771821e-7;
        MinCostFlow smallFlow(4u);
        smallFlow.addEdge(0u, 1u, 1.0e-4, 0.0);
        const auto edge = smallFlow.addEdge(1u, 2u, 1.0e-4, 0.0);
        smallFlow.addEdge(2u, 3u, tinyChildMass, 0.0);
        std::size_t augmentations = 0u;
        double sent = 0.0;
        std::string error;
        if (!smallFlow.run(0u, 3u, tinyChildMass, 10u, augmentations, sent, error) ||
            sent != tinyChildMass ||
            smallFlow.reverseFlow(EdgeRef{0u, 0u, 1u, edge, 0.0}) != tinyChildMass ||
            smallFlow.flow(EdgeRef{0u, 0u, 1u, edge, 0.0}) == tinyChildMass) {
            std::cerr << "small sparse flow was not retained independently of large "
                         "initial parent capacity\n";
            return 36;
        }
    }
    {
        using spatial_mass_transport_detail::ExactDoubleSum;
        ExactDoubleSum sourceMass;
        sourceMass.add(0.1);
        for (std::size_t i = 0; i < 1000u; ++i) sourceMass.add(2.0e-18);
        ExactDoubleSum surplus = sourceMass;
        surplus.add(-0.1);
        const double downwardSurplus = surplus.roundedDownPositive();
        ExactDoubleSum accounted = sourceMass;
        accounted.add(-0.1);
        accounted.add(-downwardSurplus);
        if (!sourceMass.valid() || !surplus.valid() || surplus.sign() <= 0 ||
            !std::isfinite(downwardSurplus) || downwardSurplus <= 0.0 ||
            !accounted.valid() || accounted.sign() < 0) {
            std::cerr << "sparse surplus rounding exceeded the exact source-minus-floor budget\n";
            return 37;
        }
    }
    {
        using spatial_mass_transport_detail::EdgeRef;
        using spatial_mass_transport_detail::MinCostFlow;
        constexpr std::size_t tinyParents = 1000u;
        constexpr double tinyMass = 2.0e-18;
        constexpr double mandatoryFloor = 0.1;
        const std::uint32_t source = 0u;
        const std::uint32_t firstParent = 1u;
        const std::uint32_t largeParent = firstParent + static_cast<std::uint32_t>(tinyParents);
        const std::uint32_t child = largeParent + 1u;
        const std::uint32_t sink = child + 1u;
        MinCostFlow manySmall(static_cast<std::size_t>(sink) + 1u);
        for (std::uint32_t i = 0; i < tinyParents; ++i) {
            const auto parent = firstParent + i;
            manySmall.addEdge(source, parent, tinyMass, 0.0);
            manySmall.addEdge(parent, child, tinyMass, 0.0);
        }
        manySmall.addEdge(source, largeParent, mandatoryFloor, 0.0);
        manySmall.addEdge(largeParent, child, mandatoryFloor, 0.0);
        const auto floorEdge = manySmall.addEdge(child, sink, mandatoryFloor, 0.0);
        std::size_t augmentations = 0u;
        double sent = 0.0;
        std::string error;
        if (!manySmall.run(source, sink, mandatoryFloor, tinyParents + 10u,
                           augmentations, sent, error, nullptr, false, nullptr,
                           0.0, true) ||
            manySmall.remainingCapacity(child, floorEdge) != 0.0 ||
            manySmall.reverseFlow(EdgeRef{0u, 0u, child, floorEdge, 0.0}) != mandatoryFloor ||
            !manySmall.hasResidualSourceCapacity(source) ||
            augmentations <= tinyParents || sent < mandatoryFloor - 1.0e-16) {
            double sourceResidual = 0.0;
            std::size_t residualEdges = 0u;
            for (std::uint32_t i = 0; i <= tinyParents; ++i) {
                const double remaining = manySmall.remainingCapacity(source, i);
                if (remaining > 0.0) { sourceResidual += remaining; ++residualEdges; }
            }
            std::cerr << std::setprecision(17)
                         << "sparse completion did not saturate the mandatory floor after "
                         "many small augmentations: " << error
                      << " sent=" << sent << " aug=" << augmentations
                      << " floor_remaining=" << manySmall.remainingCapacity(child, floorEdge)
                      << " source_remaining=" << sourceResidual
                      << " residual_edges=" << residualEdges
                      << " floor_flow=" << manySmall.reverseFlow(
                             EdgeRef{0u, 0u, child, floorEdge, 0.0})
                      << std::endl;
            return 38;
        }
    }
    {
        using spatial_mass_transport_detail::EdgeRef;
        using spatial_mass_transport_detail::MinCostFlow;
        constexpr double tinyMass = 2.0e-18;
        constexpr double floor = 0.1;
        const double nearFloor = std::nextafter(floor, 0.0);
        MinCostFlow cancellation(6u);
        cancellation.addEdge(0u, 1u, tinyMass, 0.0);
        cancellation.addEdge(1u, 4u, tinyMass, 0.0);
        cancellation.addEdge(0u, 2u, nearFloor, 0.0);
        cancellation.addEdge(2u, 4u, nearFloor, 0.0);
        cancellation.addEdge(0u, 3u, 1.0e-16, 0.0);
        cancellation.addEdge(3u, 4u, 1.0e-16, 0.0);
        const auto floorEdge = cancellation.addEdge(4u, 5u, floor, 0.0);
        std::size_t augmentations = 0u;
        double sent = 0.0;
        std::string error;
        if (!cancellation.run(0u, 5u, floor, 16u, augmentations, sent, error,
                              nullptr, false, nullptr, 0.0, true) ||
            cancellation.remainingCapacity(4u, floorEdge) != 0.0 ||
            cancellation.reverseFlow(EdgeRef{0u, 0u, 4u, floorEdge, 0.0}) != floor ||
            !cancellation.hasResidualSourceCapacity(0u) || augmentations < 3u) {
            std::cerr << std::setprecision(17)
                      << "sparse hi/lo residual lost a correction through large cancellation: "
                      << error << " sent=" << sent << " aug=" << augmentations
                      << " floor_flow=" << cancellation.reverseFlow(
                             EdgeRef{0u, 0u, 4u, floorEdge, 0.0})
                      << " floor_remaining=" << cancellation.remainingCapacity(4u, floorEdge)
                      << std::endl;
            return 39;
        }
    }
    {
        using spatial_mass_transport_detail::EdgeRef;
        using spatial_mass_transport_detail::MinCostFlow;
        MinCostFlow incremental(4u);
        incremental.addEdge(0u, 1u, 1.0, 0.0);
        const auto firstColumn = incremental.addEdge(1u, 2u, 1.0, 0.5);
        incremental.addEdge(2u, 3u, 1.0, 0.0);
        std::size_t augmentations = 0u;
        double sent = 0.0;
        std::string incrementalError;
        std::vector<double> dual;
        if (!incremental.run(0u, 3u, 1.0, 300u, augmentations, sent,
                             incrementalError, &dual, true, nullptr, 0.0, true) ||
            std::abs(sent - 1.0) > 1.0e-15) {
            std::cerr << "incremental column fixture could not establish its initial flow: "
                      << incrementalError << '\n';
            return 32;
        }
        std::uint32_t lastColumn = spatial_mass_transport_detail::kInvalid;
        for (std::size_t i = 1u; i <= 32u; ++i) {
            const double cost = 0.5 - static_cast<double>(i) / 64.0;
            if (!incremental.addColumnWithOptimalityRepair(
                    1u, 2u, 1.0, cost, dual, 300u, augmentations,
                    lastColumn, incrementalError) ||
                !incremental.residualReducedCostsNonnegative(dual, 1.0e-12,
                                                              incrementalError)) {
                std::cerr << "incremental column insertion lost its residual dual after "
                          << i << " insertions: " << incrementalError << '\n';
                return 33;
            }
        }
        const auto firstFlow = incremental.flow(EdgeRef{0u, 0u, 1u, firstColumn, 0.0});
        const auto lastFlow = incremental.flow(EdgeRef{0u, 0u, 1u, lastColumn, 0.0});
        if (augmentations != 33u || firstFlow != 0.0 || lastFlow != 1.0) {
            std::cerr << "incremental columns failed to reoptimize one retained flow "
                         "without restarting it\n";
            return 34;
        }

        const std::vector<double> smallGauge{0.125, 0.375, 0.625, 0.875};
        const double commonOffset = std::ldexp(1.0, 40);
        std::vector<double> shiftedGauge = smallGauge;
        for (auto& value : shiftedGauge) value += commonOffset;
        const double smallReduced = incremental.reducedCost(1u, 2u, 0.25, smallGauge);
        const double shiftedReduced = incremental.reducedCost(1u, 2u, 0.25, shiftedGauge);
        if (smallReduced != 0.0 || shiftedReduced != smallReduced) {
            std::cerr << "reduced-cost evaluation changed under a representable common "
                         "potential offset\n";
            return 35;
        }
    }
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
    if (result.receiptSHA256 !=
        "9e7a09107d50412732551462b9eb7dae7dcd8d39e0f5180078d13a2eb3163c96") {
        std::cerr << "dense-path receipt changed from the pre-sparse owner baseline\\n";
        return 30;
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
    if (!hasError(edgeBudget, "sparse seed exceeds its pinned edge budget")) {
        std::cerr << "transport silently truncated its allowed-edge set\n";
        return 11;
    }

    // A 3x3 geometric assignment where the global minimum uses a parent's
    // third-nearest admissible column. Two-neighbor seeding is feasible but
    // suboptimal; sparse pricing must discover the omitted negative reduced
    // cost column and match the complete dense graph.
    OwnedSparseFixture pricing(
        {{-1.1165707282527606, 0.8907405935581916, 0.38954359053015475},
         {-1.1961322028105807, 0.7504209914329159, 0.25329830570039436},
         {-0.9336157717673905,-0.22187878718069287, 1.9706943205416514}},
        {{-1.4360573094831204,-1.053497591019691, 0.3752449571409362},
         { 1.3597123033165000, 1.6868391130867888,-0.40333112014312666},
         { 1.0754457208618984,-1.5259818399516214,-0.4553717768811505}},
        4.0);
    pricing.edgeBudget = 8u;
    auto pricingSparseInput = pricing.view();
    ConservativeSpatialMassTransportResult sparseOptimal, sparseRepeated, denseOptimal;
    std::string sparseError;
    if (!assembleConservativeSpatialMassTransport(pricingSparseInput, sparseOptimal, sparseError)) {
        std::cerr << "sparse column-generation pricing fixture failed: " << sparseError << '\n';
        return 24;
    }
    pricing.edgeBudget = 32u;
    auto pricingDenseInput = pricing.view();
    if (!assembleConservativeSpatialMassTransport(pricingDenseInput, denseOptimal, sparseError) ||
        !assembleConservativeSpatialMassTransport(pricingSparseInput, sparseRepeated, sparseError) ||
        sparseOptimal.childDominantParentRows != denseOptimal.childDominantParentRows ||
        std::abs(normalizedSquaredTransportCost(sparseOptimal, pricing.radius) -
                 normalizedSquaredTransportCost(denseOptimal, pricing.radius)) > 1.0e-12 ||
        std::abs(normalizedSquaredTransportCost(sparseOptimal, pricing.radius) -
                 0.0013110258129016244) > 1.0e-11 ||
        sparseRepeated.receiptSHA256 != sparseOptimal.receiptSHA256 ||
        sparseRepeated.childParentRows != sparseOptimal.childParentRows ||
        !verifyRecordedConservativeSpatialMassTransport(
            pricingSparseInput, sparseOptimal, sparseError)) {
        std::cerr << "sparse pricing did not certify deterministic dense-optimal transport: "
                  << sparseError << '\n';
        return 25;
    }

    // A four-child cut where two fixed parents consume the sparse seed's
    // nearest columns. The only feasible extension crosses the residual
    // min-cut through a third candidate; one unrelated omitted edge remains
    // outside the active-edge budget.
    OwnedSparseFixture cutExpansion(
        {{0.0,0.0,0.0}, {0.9,0.0,0.0}, {-0.9,0.0,0.0}, {0.0,-0.9,0.0}},
        {{0.9,0.0,0.0}, {-0.9,0.0,0.0}, {0.0,0.9,0.0}, {0.0,-0.9,0.0}},
        1.0);
    cutExpansion.edgeBudget = 6u;
    auto cutInput = cutExpansion.view();
    ConservativeSpatialMassTransportResult cutResult;
    if (!assembleConservativeSpatialMassTransport(cutInput, cutResult, sparseError) ||
        cutResult.childDominantParentRows != std::vector<std::uint32_t>{1u,2u,0u,3u} ||
        cutResult.maximumParentMassClosureErrorKg > 1.0e-13) {
        std::cerr << "sparse residual min-cut expansion failed a feasible local graph: "
                  << sparseError << '\n';
        return 26;
    }

    // The full candidate graph has seven edges, while the six-edge sparse
    // budget can cover the five seeds and the sole min-cut edge needed for
    // feasibility. One further omitted edge remains unmaterialized.
    auto cutBudget = cutExpansion.view();
    cutExpansion.edgeBudget = 5u;
    cutBudget = cutExpansion.view();
    auto preservedCut = denseOptimal;
    const auto preservedCutReceipt = preservedCut.receiptSHA256;
    if (assembleConservativeSpatialMassTransport(cutBudget, preservedCut, sparseError) ||
        sparseError.find("edge budget") == std::string::npos ||
        preservedCut.receiptSHA256 != preservedCutReceipt ||
        preservedCut.childMassKg != denseOptimal.childMassKg) {
        std::cerr << "sparse edge budget was exceeded or failure mutated the caller result: "
                  << sparseError << '\n';
        return 28;
    }

    // A second label group is spatially disconnected and has enough aggregate
    // capacity, but its only reachable child cannot carry the fixed parents'
    // mass within its upper density. The preceding dense group pushes the
    // combined graph over budget, so sparse fallback must retain the exact
    // same-label/distance infeasibility rather than use the other group's room.
    OwnedSparseFixture infeasibleSparse(
        {{-0.22331414565055212, 0.17814811871163832, 0.07790871810603095},
         {-0.23922644056211614, 0.15008419828658318, 0.05065966114007887},
         {-0.1867231543534781,-0.044375757436138574,0.3941388641083303},
         {10.0,0.0,0.0}, {10.0,0.0,0.0}},
        {{-0.2872114618966241,-0.2106995182039382, 0.07504899142818725},
         { 0.2719424606633000, 0.3373678226173578,-0.08066622402862533},
         { 0.2150890941723797,-0.3051963679903243,-0.0910743553762301},
         {10.0,0.0,0.0}, {12.0,0.0,0.0}},
        1.0);
    infeasibleSparse.edgeBudget = 9u;
    infeasibleSparse.parentRegions[3] = 202u;
    infeasibleSparse.parentRegions[4] = 202u;
    infeasibleSparse.childRegions[3] = 202u;
    infeasibleSparse.childRegions[4] = 202u;
    infeasibleSparse.childVolumes[3] = 0.9e-6;
    infeasibleSparse.childVolumes[4] = 1.1e-6;
    auto infeasibleSparseInput = infeasibleSparse.view();
    auto preservedInfeasible = denseOptimal;
    const auto preservedInfeasibleReceipt = preservedInfeasible.receiptSHA256;
    if (assembleConservativeSpatialMassTransport(infeasibleSparseInput,
                                                  preservedInfeasible, sparseError) ||
        sparseError.find("infeasible under same-material/donor/region and distance bounds") ==
            std::string::npos ||
        sparseError.find("material=15 donor=2 region=202") == std::string::npos ||
        preservedInfeasible.receiptSHA256 != preservedInfeasibleReceipt ||
        preservedInfeasible.childMassKg != denseOptimal.childMassKg) {
        std::cerr << "sparse fallback did not preserve a genuine capacity/distance infeasibility: "
                  << sparseError << '\n';
        return 29;
    }

    // Overflow occurs only in the second label group, after the first dense
    // group has already spent augmentations. The fallback replay must carry
    // that spent work forward; otherwise it could reset the global budget and
    // accept a run whose total solver augmentations exceed the pinned limit.
    OwnedSparseFixture laterOverflow(
        {{-1.1165707282527606, 0.8907405935581916, 0.38954359053015475},
         {-1.1961322028105807, 0.7504209914329159, 0.25329830570039436},
         {-0.9336157717673905,-0.22187878718069287, 1.9706943205416514},
         {20.0,0.0,0.0}},
        {{-1.4360573094831204,-1.053497591019691, 0.3752449571409362},
         { 1.3597123033165000, 1.6868391130867888,-0.40333112014312666},
         { 1.0754457208618984,-1.5259818399516214,-0.4553717768811505},
         {20.0,0.0,0.0}},
        4.0);
    laterOverflow.edgeBudget = 9u;
    laterOverflow.augmentationBudget = 6u;
    laterOverflow.parentRegions[3] = 202u;
    laterOverflow.childRegions[3] = 202u;
    const auto laterOverflowInput = laterOverflow.view();
    auto preservedLater = denseOptimal;
    const auto preservedLaterReceipt = preservedLater.receiptSHA256;
    if (assembleConservativeSpatialMassTransport(laterOverflowInput, preservedLater,
                                                  sparseError) ||
        sparseError.find("augmentation budget") == std::string::npos ||
        preservedLater.receiptSHA256 != preservedLaterReceipt ||
        preservedLater.childMassKg != denseOptimal.childMassKg) {
        std::cerr << "later-group sparse fallback reset already-spent augmentation work: "
                  << sparseError << '\n';
        return 31;
    }

    // Both the seed graph and later pricing must obey the existing aggregate
    // augmentation budget. Failure is atomic and leaves the caller's output.
    auto exhausted = pricing.view();
    exhausted.maximumAllowedEdges = 8u;
    exhausted.maximumAugmentations = 3u;
    auto preserved = denseOptimal;
    const auto preservedReceipt = preserved.receiptSHA256;
    if (assembleConservativeSpatialMassTransport(exhausted, preserved, sparseError) ||
        sparseError.find("augmentation budget") == std::string::npos ||
        preserved.receiptSHA256 != preservedReceipt ||
        preserved.childMassKg != denseOptimal.childMassKg) {
        std::cerr << "sparse retries exceeded or bypassed the pinned augmentation budget: "
                  << sparseError << '\n';
        return 27;
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
        error.find("cut_parent_rows=[1]") == std::string::npos ||
        error.find("cut_child_rows=[1]") == std::string::npos ||
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
