#pragma once

// Deterministic, bounded transfer of authoritative parent cell mass to a new
// child-cell partition. This is a mass transport coupling, not a geometric
// overlap test or a reference-coordinate pullback.
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <map>
#include <queue>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

using SpatialMassTransportPoint = std::array<double, 3>;

enum class UnpairedVolumeMassTransferKind : std::uint8_t {
    none = 0u,
    conservativeSpatialMassTransport = 1u
};

struct ConservativeSpatialMassTransportInput {
    std::span<const SpatialMassTransportPoint> parentCentroidsWorld;
    std::span<const double> parentReferenceVolumeM3;
    std::span<const double> parentMassKg;
    std::span<const std::uint32_t> parentMaterialIndices;
    std::span<const std::uint32_t> parentDonorBodyIndices;
    std::span<const std::uint32_t> parentAnatomicalRegionIndices;
    // Cooked source-donor split for every parent cell. Each CSR row is built
    // from the actual parent-node source contributions and sums to that cell's
    // authoritative mass; it remains separate from the cell's geometric donor.
    std::span<const std::uint32_t> parentSourceDonorOffsets;
    std::span<const std::uint32_t> parentSourceDonorBodyIndices;
    std::span<const double> parentSourceDonorMassKg;

    std::span<const SpatialMassTransportPoint> childCentroidsWorld;
    std::span<const double> childReferenceVolumeM3;
    std::span<const std::uint32_t> childMaterialIndices;
    std::span<const std::uint32_t> childDonorBodyIndices;
    std::span<const std::uint32_t> childAnatomicalRegionIndices;

    // Bounds come from the owning material definitions, indexed by material ID.
    std::span<const double> materialDensityLowerKgM3;
    std::span<const double> materialDensityUpperKgM3;

    // All five digests are supplied by the owning artifact inputs. The helper
    // also hashes the exact numeric arrays it receives and includes both sets
    // of hashes in the result receipt.
    std::string_view parentMassPacketSHA256;
    std::string_view parentSourceContributionSHA256;
    std::string_view childMeshSHA256;
    std::string_view regionLabelSHA256;
    std::string_view coordinateFrameSHA256;

    double maximumTransportDistanceM = 0.0;
    double relativeClosureTolerance = 1.0e-10;
    double absoluteMassToleranceKg = 1.0e-13;
    double absoluteMeasureToleranceM3 = 1.0e-15;
    std::size_t maximumAllowedEdges = 3'000'000u;
    std::size_t maximumAugmentations = 300'000u;
};

struct ConservativeSpatialMassTransportSourceDonorClosure {
    std::uint32_t materialIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceDonorBodyIndex = std::numeric_limits<std::uint32_t>::max();
    double parentMassKg = 0.0;
    double childMassKg = 0.0;
};

struct ConservativeSpatialMassTransportGroupClosure {
    std::uint32_t materialIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t donorBodyIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t anatomicalRegionIndex = std::numeric_limits<std::uint32_t>::max();
    double parentMassKg = 0.0;
    double childMassKg = 0.0;
    double childMinimumCapacityKg = 0.0;
    double childMaximumCapacityKg = 0.0;
};

struct ConservativeSpatialMassTransportParentNodeSourceMass {
    std::uint32_t parentNodeIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t materialIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceDonorBodyIndex = std::numeric_limits<std::uint32_t>::max();
    double massKg = 0.0;
};

// Reconstruct each parent cell's donor split from the actual compiled nodal
// source rows. When a node/material has several recorded source donors, their
// measured node-level shares apportion each cooked tetrahedron corner mass.
// The reconstructed corner shares must reproduce the complete node/material/
// donor source-mass packet within its declared cook-rounding tolerance.
[[nodiscard]] inline bool assembleConservativeSpatialMassTransportParentSourceDonorSplits(
        const std::span<const double> parentCellMassKg,
        const std::span<const std::uint32_t> parentCellMaterialIndices,
        const std::span<const std::array<std::uint32_t, 4>> parentCellNodeIndices,
        const std::span<const ConservativeSpatialMassTransportParentNodeSourceMass> nodeSourceMass,
        const double absoluteMassToleranceKg,
        const double relativeMassTolerance,
        std::vector<std::uint32_t>& sourceDonorOffsets,
        std::vector<std::uint32_t>& sourceDonorBodyIndices,
        std::vector<double>& sourceDonorMassKg,
        std::string& error) {
    using NodeMaterial = std::pair<std::uint32_t, std::uint32_t>;
    using SourceKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    if (parentCellMassKg.empty() || parentCellMassKg.size() > invalid ||
        parentCellMaterialIndices.size() != parentCellMassKg.size() ||
        parentCellNodeIndices.size() != parentCellMassKg.size() || nodeSourceMass.empty() ||
        nodeSourceMass.size() > invalid || !std::isfinite(absoluteMassToleranceKg) ||
        absoluteMassToleranceKg < 0.0 || !std::isfinite(relativeMassTolerance) ||
        relativeMassTolerance < 0.0)
        return fail("parent source-donor reconstruction arrays or tolerances are invalid");

    std::map<NodeMaterial, std::vector<std::pair<std::uint32_t, double>>> sourcesByNodeMaterial;
    std::map<NodeMaterial, double> sourceTotalByNodeMaterial;
    std::map<SourceKey, double> suppliedMassBySource;
    for (const auto& row : nodeSourceMass) {
        if (row.parentNodeIndex == invalid || row.materialIndex == invalid ||
            row.sourceDonorBodyIndex == invalid || !std::isfinite(row.massKg) ||
            row.massKg <= 0.0)
            return fail("actual parent node source-mass row is invalid");
        const SourceKey key{row.parentNodeIndex, row.materialIndex, row.sourceDonorBodyIndex};
        if (!suppliedMassBySource.emplace(key, row.massKg).second)
            return fail("actual parent node source-mass packet has duplicate node/material/donor keys");
        const NodeMaterial nodeMaterial{row.parentNodeIndex, row.materialIndex};
        sourcesByNodeMaterial[nodeMaterial].push_back({row.sourceDonorBodyIndex, row.massKg});
        sourceTotalByNodeMaterial[nodeMaterial] += row.massKg;
    }

    std::map<SourceKey, double> expectedMassBySource;
    std::vector<std::map<std::uint32_t, double>> splitByCell(parentCellMassKg.size());
    for (std::size_t cell = 0; cell < parentCellMassKg.size(); ++cell) {
        const double mass = parentCellMassKg[cell];
        const auto material = parentCellMaterialIndices[cell];
        if (!std::isfinite(mass) || mass <= 0.0 || material == invalid)
            return fail("authoritative parent cell mass or material is invalid");
        const double cornerMass = mass * 0.25;
        auto nodes = parentCellNodeIndices[cell];
        std::sort(nodes.begin(), nodes.end());
        if (std::adjacent_find(nodes.begin(), nodes.end()) != nodes.end() || nodes.front() == invalid)
            return fail("authoritative parent cell has invalid or repeated corner node rows");
        for (const auto node : parentCellNodeIndices[cell]) {
            const NodeMaterial nodeMaterial{node, material};
            const auto owners = sourcesByNodeMaterial.find(nodeMaterial);
            const auto total = sourceTotalByNodeMaterial.find(nodeMaterial);
            if (owners == sourcesByNodeMaterial.end() || owners->second.empty() ||
                total == sourceTotalByNodeMaterial.end() || !std::isfinite(total->second) ||
                total->second <= 0.0)
                return fail("actual parent node source-mass packet omits a cell corner owner");
            for (const auto& [donor, sourceMass] : owners->second) {
                const double fraction = sourceMass / total->second;
                const double cornerSourceMass = cornerMass * fraction;
                expectedMassBySource[{node, material, donor}] += cornerSourceMass;
                splitByCell[cell][donor] += cornerSourceMass;
            }
        }
    }
    if (expectedMassBySource.size() != suppliedMassBySource.size())
        return fail("actual parent node source-mass key set differs from cell-corner reconstruction");
    for (const auto& [key, expected] : expectedMassBySource) {
        const auto found = suppliedMassBySource.find(key);
        if (found == suppliedMassBySource.end())
            return fail("actual parent node source-mass key set differs from cell-corner reconstruction");
        if (std::abs(found->second - expected) >
            std::max(absoluteMassToleranceKg, relativeMassTolerance * expected))
            return fail("actual parent node source mass differs from cooked parent cell corner shares");
    }

    std::vector<std::uint32_t> offsets(parentCellMassKg.size() + 1u, 0u);
    std::vector<std::uint32_t> donors;
    std::vector<double> masses;
    for (std::size_t cell = 0; cell < splitByCell.size(); ++cell) {
        offsets[cell] = static_cast<std::uint32_t>(donors.size());
        double splitMass = 0.0;
        for (const auto& [donor, mass] : splitByCell[cell]) {
            if (donor == invalid || !std::isfinite(mass) || mass <= 0.0 ||
                donors.size() >= invalid)
                return fail("reconstructed parent source-donor CSR contains an invalid row or exceeds range");
            donors.push_back(donor);
            masses.push_back(mass);
            splitMass += mass;
        }
        if (std::abs(splitMass - parentCellMassKg[cell]) >
            std::max(absoluteMassToleranceKg, relativeMassTolerance * parentCellMassKg[cell]))
            return fail("reconstructed parent source-donor split does not close cell mass");
        offsets[cell + 1u] = static_cast<std::uint32_t>(donors.size());
    }
    sourceDonorOffsets = std::move(offsets);
    sourceDonorBodyIndices = std::move(donors);
    sourceDonorMassKg = std::move(masses);
    error.clear();
    return true;
}

struct ConservativeSpatialMassTransportResult {
    UnpairedVolumeMassTransferKind method =
        UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    // Rows are child cells. Parent rows are sorted and unique within each row.
    std::vector<std::uint32_t> childParentOffsets;
    std::vector<std::uint32_t> childParentRows;
    std::vector<double> childParentTransportMassKg;
    // This is the source-parent reference measure carried by each mass flow:
    // q_cp / (parentMass_p / parentReferenceVolume_p). It is not a geometric
    // intersection volume and must retain method=conservativeSpatialMassTransport.
    std::vector<double> childParentTransportMeasureM3;
    std::vector<double> childParentTransportDistanceM;
    std::vector<double> childParentVolumeFractions;
    std::vector<double> childAncestryMeasureVolumeM3;
    std::vector<double> childMassKg;
    std::vector<double> childImpliedDensityKgM3;
    std::vector<std::uint32_t> childDominantParentRows;
    std::vector<double> parentMassTransferredKg;
    std::vector<double> parentMeasureTransferredM3;
    std::vector<ConservativeSpatialMassTransportGroupClosure> groupClosure;
    // Per-child source donor composition, in sorted CSR rows. These are
    // lineage allocations reconstructed from each parent split and edge flow.
    std::vector<std::uint32_t> childSourceDonorOffsets;
    std::vector<std::uint32_t> childSourceDonorBodyIndices;
    std::vector<double> childSourceDonorMassKg;
    std::vector<ConservativeSpatialMassTransportSourceDonorClosure> sourceDonorClosure;

    std::size_t transportEdgeCount = 0u;
    std::size_t augmentationCount = 0u;
    double maximumParentMassClosureErrorKg = 0.0;
    double maximumParentMeasureClosureErrorM3 = 0.0;
    double maximumSourceDonorMassClosureErrorKg = 0.0;
    double maximumChildFractionSumError = 0.0;
    double maximumTransportDistanceUsedM = 0.0;
    double massWeightedMeanTransportDistanceM = 0.0;
    double minimumChildImpliedDensityKgM3 = 0.0;
    double maximumChildImpliedDensityKgM3 = 0.0;

    std::string parentDataSHA256;
    std::string childDataSHA256;
    std::string materialBoundsSHA256;
    std::string parentMassPacketSHA256;
    std::string parentSourceContributionSHA256;
    std::string childMeshSHA256;
    std::string regionLabelSHA256;
    std::string coordinateFrameSHA256;
    double maximumTransportDistanceM = 0.0;
    double relativeClosureTolerance = 0.0;
    double absoluteMassToleranceKg = 0.0;
    double absoluteMeasureToleranceM3 = 0.0;
    std::size_t maximumAllowedEdges = 0u;
    std::size_t maximumAugmentations = 0u;
    std::string receiptSHA256;
};

namespace spatial_mass_transport_detail {

inline constexpr std::uint32_t kInvalid = std::numeric_limits<std::uint32_t>::max();
inline constexpr double kFlowEpsilonKg = 1.0e-18;

// Exact expansion accumulator for sums of binary64 capacities. The expansion
// is used only by the sparse fallback to keep the source/floor budget
// conservative; dense receipt arithmetic remains unchanged.
class ExactDoubleSum {
public:
    void add(double value) {
        if (!std::isfinite(value)) {
            valid_ = false;
            return;
        }
        std::size_t write = 0u;
        for (const double partial : partials_) {
            double other = partial;
            if (std::abs(value) < std::abs(other)) std::swap(value, other);
            const double high = value + other;
            const double virtualOther = high - value;
            const double low = other - virtualOther;
            if (low != 0.0) partials_[write++] = low;
            value = high;
        }
        partials_.resize(write);
        if (value != 0.0) partials_.push_back(value);
    }

    [[nodiscard]] bool valid() const { return valid_; }

    [[nodiscard]] int sign() const {
        if (!valid_) return 0;
        for (auto it = partials_.rbegin(); it != partials_.rend(); ++it)
            if (*it != 0.0) return *it > 0.0 ? 1 : -1;
        return 0;
    }

    [[nodiscard]] double rounded() const {
        double value = 0.0;
        for (const double partial : partials_) value += partial;
        return value;
    }

    // Return the greatest binary64 value known not to exceed this positive
    // exact expansion. Binary-searching the ordered positive bit patterns
    // avoids assuming that long double has more precision than double.
    [[nodiscard]] double roundedDownPositive() const {
        if (sign() <= 0) return 0.0;
        const double approximation = rounded();
        if (!std::isfinite(approximation) || approximation <= 0.0) return 0.0;
        const auto exactMinus = [&](const double candidate) {
            ExactDoubleSum difference = *this;
            difference.add(-candidate);
            return difference.sign();
        };
        std::uint64_t low = std::bit_cast<std::uint64_t>(0.0);
        std::uint64_t high = std::bit_cast<std::uint64_t>(approximation);
        if (exactMinus(approximation) >= 0) {
            low = high;
            const double next = std::nextafter(approximation,
                                               std::numeric_limits<double>::infinity());
            if (!std::isfinite(next)) return approximation;
            high = std::bit_cast<std::uint64_t>(next);
            while (high - low > 1u) {
                const auto middle = low + (high - low) / 2u;
                const double candidate = std::bit_cast<double>(middle);
                if (exactMinus(candidate) >= 0) low = middle;
                else high = middle;
            }
            return std::bit_cast<double>(low);
        }
        while (high - low > 1u) {
            const auto middle = low + (high - low) / 2u;
            const double candidate = std::bit_cast<double>(middle);
            if (exactMinus(candidate) >= 0) low = middle;
            else high = middle;
        }
        return std::bit_cast<double>(low);
    }

private:
    std::vector<double> partials_;
    bool valid_ = true;
};


inline double densityBoundRoundoff(const double lower, const double upper,
                                   const double density) {
    return 64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(lower), std::abs(upper), std::abs(density)});
}

inline bool validSHA256(const std::string_view value) {
    return value.size() == 64u && std::all_of(value.begin(), value.end(), [](const char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

class SHA256Builder {
public:
    SHA256Builder() { CC_SHA256_Init(&context_); }
    void bytes(const void* data, const std::size_t size) {
        const auto* cursor = static_cast<const std::uint8_t*>(data);
        std::size_t remaining = size;
        while (remaining != 0u) {
            const auto block = static_cast<CC_LONG>(std::min<std::size_t>(remaining, 1u << 20u));
            CC_SHA256_Update(&context_, cursor, block);
            cursor += block;
            remaining -= block;
        }
    }
    void string(const std::string_view value) {
        u64(value.size());
        bytes(value.data(), value.size());
    }
    void u32(const std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (unsigned i = 0; i < encoded.size(); ++i)
            encoded[i] = static_cast<std::uint8_t>(value >> (8u * (3u - i)));
        bytes(encoded.data(), encoded.size());
    }
    void u64(const std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (unsigned i = 0; i < encoded.size(); ++i)
            encoded[i] = static_cast<std::uint8_t>(value >> (8u * (7u - i)));
        bytes(encoded.data(), encoded.size());
    }
    void real(const double value) { u64(std::bit_cast<std::uint64_t>(value)); }
    [[nodiscard]] std::string finish() {
        std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
        CC_SHA256_Final(digest.data(), &context_);
        constexpr char hex[] = "0123456789abcdef";
        std::string result;
        result.resize(2u * digest.size());
        for (std::size_t i = 0; i < digest.size(); ++i) {
            result[2u * i] = hex[digest[i] >> 4u];
            result[2u * i + 1u] = hex[digest[i] & 0x0fu];
        }
        return result;
    }
private:
    CC_SHA256_CTX context_{};
};

using GroupKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
struct GridKey {
    std::int64_t x = 0, y = 0, z = 0;
    friend bool operator<(const GridKey& a, const GridKey& b) {
        return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
    }
};
struct Candidate {
    std::uint32_t child = 0u;
    double distanceM = 0.0;
};

template <typename FlowRecord>
inline bool sortAndAccumulateChildFlowSums(
        std::vector<FlowRecord>& flows, const std::size_t childCount,
        std::vector<double>& childMassKg,
        std::vector<double>& childMeasureM3) {
    childMassKg.assign(childCount, 0.0);
    childMeasureM3.assign(childCount, 0.0);
    std::sort(flows.begin(), flows.end(), [](const auto& a, const auto& b) {
        if (a.child != b.child) return a.child < b.child;
        return a.parent < b.parent;
    });
    for (const auto& flow : flows) {
        if (flow.child >= childCount || !std::isfinite(flow.massKg) ||
            flow.massKg <= 0.0 || !std::isfinite(flow.measureM3) ||
            flow.measureM3 <= 0.0)
            return false;
        childMassKg[flow.child] += flow.massKg;
        childMeasureM3[flow.child] += flow.measureM3;
        if (!std::isfinite(childMassKg[flow.child]) ||
            !std::isfinite(childMeasureM3[flow.child]))
            return false;
    }
    return true;
}

inline bool gridKey(const SpatialMassTransportPoint& point, const double width,
                    GridKey& out) {
    for (const double coordinate : point)
        if (!std::isfinite(coordinate)) return false;
    const auto cell = [width](const double coordinate, std::int64_t& value) {
        const double raw = std::floor(coordinate / width);
        if (!std::isfinite(raw) ||
            raw < static_cast<double>(std::numeric_limits<std::int64_t>::min() + 2) ||
            raw > static_cast<double>(std::numeric_limits<std::int64_t>::max() - 2))
            return false;
        value = static_cast<std::int64_t>(raw);
        return true;
    };
    return cell(point[0], out.x) && cell(point[1], out.y) && cell(point[2], out.z);
}

inline double distance(const SpatialMassTransportPoint& a,
                       const SpatialMassTransportPoint& b) {
    const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return std::sqrt(x*x + y*y + z*z);
}

struct FlowEdge {
    std::uint32_t to = 0u;
    std::uint32_t reverse = 0u;
    double capacity = 0.0;
    double initialCapacity = 0.0;
    double cost = 0.0;
    // Kahan correction for sparse-only capacity updates. The represented
    // residual is capacity - capacityCorrection.
    double capacityCorrection = 0.0;
};
struct EdgeRef {
    std::uint32_t parent = 0u;
    std::uint32_t child = 0u;
    std::uint32_t from = 0u;
    std::uint32_t edge = 0u;
    double distanceM = 0.0;
};

class MinCostFlow {
    [[nodiscard]] static double stableReducedCost(const double cost,
                                                   const double fromPotential,
                                                   const double toPotential) {
        return static_cast<double>(static_cast<long double>(cost) +
            (static_cast<long double>(fromPotential) -
             static_cast<long double>(toPotential)));
    }

    static void twoSum(const double a, const double b,
                       double& sum, double& error) {
        sum = a + b;
        const double bVirtual = sum - a;
        error = (a - (sum - bVirtual)) + (b - bVirtual);
    }

    static void addCapacity(FlowEdge& edge, const double delta) {
        // Keep the exact low part when a much smaller augmentation is added
        // to a large residual. Kahan alone loses that correction if the next
        // delta nearly cancels the high part.
        double highSum = 0.0, highError = 0.0;
        twoSum(edge.capacity, delta, highSum, highError);
        const double oldLow = -edge.capacityCorrection;
        double lowSum = 0.0, lowError = 0.0;
        twoSum(highError, oldLow, lowSum, lowError);
        double normalizedHigh = 0.0, normalizedLow = 0.0;
        twoSum(highSum, lowSum, normalizedHigh, normalizedLow);
        double tail = 0.0, tailError = 0.0;
        twoSum(normalizedLow, lowError, tail, tailError);
        double finalHigh = 0.0, finalLow = 0.0;
        twoSum(normalizedHigh, tail, finalHigh, finalLow);
        edge.capacity = finalHigh;
        // The final transform's low error is below binary64 double-double
        // precision for the bounded mass scales used here. Fold its residual
        // into the stored low part with a second error-free transform.
        double storedLow = 0.0, storedTail = 0.0;
        twoSum(finalLow, tailError, storedLow, storedTail);
        edge.capacityCorrection = -(storedLow + storedTail);
    }

    [[nodiscard]] static bool hasPositiveCapacity(const FlowEdge& edge) {
        if (edge.capacity > 0.0) return true;
        return edge.capacity == 0.0 && edge.capacityCorrection < 0.0;
    }

    [[nodiscard]] static double pathCapacity(const FlowEdge& edge) {
        if (edge.capacity > 0.0) {
            return edge.capacityCorrection > 0.0
                ? std::nextafter(edge.capacity, 0.0) : edge.capacity;
        }
        return edge.capacity == 0.0 && edge.capacityCorrection < 0.0
            ? -edge.capacityCorrection : 0.0;
    }

    [[nodiscard]] static double roundedCapacity(const FlowEdge& edge) {
        return edge.capacity - edge.capacityCorrection;
    }

    static bool normalizePotentialGauge(std::vector<double>& potential) {
        if (potential.empty()) return true;
        const auto minimum = std::min_element(potential.begin(), potential.end());
        const long double gauge = static_cast<long double>(*minimum);
        for (auto& value : potential) {
            value = static_cast<double>(static_cast<long double>(value) - gauge);
            if (!std::isfinite(value)) return false;
        }
        return true;
    }
public:
    explicit MinCostFlow(const std::size_t nodeCount) : graph_(nodeCount) {}
    std::uint32_t addEdge(const std::uint32_t from, const std::uint32_t to,
                          const double capacity, const double cost) {
        const auto index = static_cast<std::uint32_t>(graph_[from].size());
        const auto reverseIndex = static_cast<std::uint32_t>(graph_[to].size());
        graph_[from].push_back({to, reverseIndex, capacity, capacity, cost});
        graph_[to].push_back({from, index, 0.0, 0.0, -cost});
        return index;
    }

    bool run(const std::uint32_t source, const std::uint32_t sink,
             const double required, const std::size_t maximumAugmentations,
             std::size_t& augmentationCount, double& sent,
             std::string& error, std::vector<double>* finalPotential = nullptr,
             const bool updateUnreachablePotentials = false,
             const std::vector<double>* initialPotential = nullptr,
             const double initialSent = 0.0,
             const bool stableSparseArithmetic = false) {
        const auto fail = [&](const char* message) { error = message; return false; };
        const std::size_t n = graph_.size();
        std::vector<double> potential(n, 0.0), distance(n);
        std::vector<std::uint32_t> previousNode(n), previousEdge(n);
        if ((initialPotential && initialPotential->size() != n) ||
            !std::isfinite(initialSent) || initialSent < 0.0 ||
            (!stableSparseArithmetic && initialSent > required + kFlowEpsilonKg))
            return fail("spatial transport continuation state is invalid");
        if (initialPotential) potential = *initialPotential;
        augmentationCount = 0u;
        if (stableSparseArithmetic) compensatedSparseCapacities_ = true;
        sent = stableSparseArithmetic ? sentFromSource(source) : initialSent;
        while (stableSparseArithmetic ? hasResidualSourceCapacity(source)
                                     : required - sent > kFlowEpsilonKg) {
            if (augmentationCount >= maximumAugmentations)
                return fail("spatial transport exceeded its pinned augmentation budget");
            std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
            std::fill(previousNode.begin(), previousNode.end(), kInvalid);
            std::fill(previousEdge.begin(), previousEdge.end(), kInvalid);
            using QueueItem = std::pair<double, std::uint32_t>;
            std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
            distance[source] = 0.0;
            queue.push({0.0, source});
            while (!queue.empty()) {
                const auto [knownDistance, node] = queue.top();
                queue.pop();
                if (knownDistance > distance[node] +
                        (stableSparseArithmetic ? 0.0 : 1.0e-15)) continue;
                for (std::uint32_t edgeIndex = 0; edgeIndex < graph_[node].size(); ++edgeIndex) {
                    const auto& edge = graph_[node][edgeIndex];
                    if (stableSparseArithmetic ? !hasPositiveCapacity(edge)
                                               : edge.capacity <= kFlowEpsilonKg) continue;
                    double reduced = stableSparseArithmetic
                        ? stableReducedCost(edge.cost, potential[node], potential[edge.to])
                        : edge.cost + potential[node] - potential[edge.to];
                    if (reduced < 0.0 && reduced > -1.0e-12) reduced = 0.0;
                    if (reduced < 0.0)
                        return fail("spatial transport min-cost reduced edge became negative");
                    const double candidate = knownDistance + reduced;
                    if (candidate < distance[edge.to] -
                            (stableSparseArithmetic ? 0.0 : 1.0e-15)) {
                        distance[edge.to] = candidate;
                        previousNode[edge.to] = node;
                        previousEdge[edge.to] = edgeIndex;
                        queue.push({candidate, edge.to});
                    }
                }
            }
            if (previousNode[sink] == kInvalid) break;
            if (updateUnreachablePotentials) {
                double maximumFiniteDistance = 0.0;
                for (const auto value : distance)
                    if (std::isfinite(value))
                        maximumFiniteDistance = std::max(maximumFiniteDistance, value);
                // Assigning the largest finite label to unreachable nodes keeps
                // every residual reduced cost feasible globally: an edge from
                // an unreachable node into the reached set gains a nonnegative
                // amount, while a reached-to-unreachable residual edge cannot
                // exist after Dijkstra has exhausted the queue.
                for (std::size_t node = 0; node < n; ++node)
                    potential[node] = static_cast<double>(
                        static_cast<long double>(potential[node]) +
                        (std::isfinite(distance[node]) ? distance[node] : maximumFiniteDistance));
            } else {
                for (std::size_t node = 0; node < n; ++node)
                    if (std::isfinite(distance[node])) potential[node] += distance[node];
            }
            if (stableSparseArithmetic && !normalizePotentialGauge(potential))
                return fail("spatial transport sparse potential gauge normalization failed");
            double amount = stableSparseArithmetic
                ? std::numeric_limits<double>::infinity() : required - sent;
            for (std::uint32_t node = sink; node != source; node = previousNode[node]) {
                if (previousNode[node] == kInvalid)
                    return fail("spatial transport residual path is incomplete");
                const auto& pathEdge =
                    graph_[previousNode[node]][previousEdge[node]];
                amount = std::min(amount, stableSparseArithmetic
                    ? pathCapacity(pathEdge) : pathEdge.capacity);
            }
            if (!std::isfinite(amount) ||
                (stableSparseArithmetic ? amount <= 0.0 : amount <= kFlowEpsilonKg))
                return fail("spatial transport residual path has no positive mass capacity");
            for (std::uint32_t node = sink; node != source; node = previousNode[node]) {
                auto& edge = graph_[previousNode[node]][previousEdge[node]];
                auto& reverse = graph_[node][edge.reverse];
                if (stableSparseArithmetic) {
                    addCapacity(edge, -amount);
                    addCapacity(reverse, amount);
                } else {
                    edge.capacity -= amount;
                    reverse.capacity += amount;
                }
            }
            sent = stableSparseArithmetic ? sentFromSource(source) : sent + amount;
            ++augmentationCount;
        }
        if (finalPotential) *finalPotential = potential;
        error.clear();
        return true;
    }

    // Add one previously omitted admissible parent-child edge while retaining
    // the current min-cost flow. If its reduced cost is negative, a shortest
    // residual path from child back to parent either closes a negative cycle
    // (which is augmented) or provides the dual update that makes the new edge
    // nonnegative. The search excludes both directions of the newly inserted
    // edge so it cannot certify a zero-length edge/reverse-edge cycle.
    bool addColumnWithOptimalityRepair(
            const std::uint32_t from, const std::uint32_t to,
            const double capacity, const double cost,
            std::vector<double>& potential,
            const std::size_t maximumAugmentations,
            std::size_t& augmentationCount,
            std::uint32_t& insertedEdgeIndex,
            std::string& error) {
        const auto fail = [&](const char* message) { error = message; return false; };
        if (from >= graph_.size() || to >= graph_.size() || from == to ||
            potential.size() != graph_.size() || !std::isfinite(capacity) ||
            capacity <= 0.0 || !std::isfinite(cost) || cost < 0.0)
            return fail("spatial transport inserted column is invalid");
        compensatedSparseCapacities_ = true;
        insertedEdgeIndex = addEdge(from, to, capacity, cost);
        const auto insertedReverseIndex = graph_[from][insertedEdgeIndex].reverse;
        std::vector<double> distance(graph_.size());
        std::vector<std::uint32_t> previousNode(graph_.size()), previousEdge(graph_.size());
        std::vector<std::pair<std::uint32_t, std::uint32_t>> lastCycleReverseEdges;
        std::size_t cycleNumber = 0u;
        while (hasPositiveCapacity(graph_[from][insertedEdgeIndex])) {
            const double reducedColumn = stableReducedCost(cost, potential[from], potential[to]);
            if (!std::isfinite(reducedColumn))
                return fail("spatial transport inserted column has nonfinite reduced cost");
            if (reducedColumn >= -1.0e-12) break;

            std::fill(distance.begin(), distance.end(),
                      std::numeric_limits<double>::infinity());
            std::fill(previousNode.begin(), previousNode.end(), kInvalid);
            std::fill(previousEdge.begin(), previousEdge.end(), kInvalid);
            using QueueItem = std::pair<double, std::uint32_t>;
            std::priority_queue<QueueItem, std::vector<QueueItem>,
                                std::greater<QueueItem>> queue;
            distance[to] = 0.0;
            queue.push({0.0, to});
            while (!queue.empty()) {
                const auto [knownDistance, node] = queue.top();
                queue.pop();
                if (knownDistance > distance[node]) continue;
                for (std::uint32_t edgeIndex = 0;
                     edgeIndex < graph_[node].size(); ++edgeIndex) {
                    if ((node == from && edgeIndex == insertedEdgeIndex) ||
                        (node == to && edgeIndex == insertedReverseIndex))
                        continue;
                    const auto& edge = graph_[node][edgeIndex];
                    if (!hasPositiveCapacity(edge)) continue;
                    double reduced = stableReducedCost(
                        edge.cost, potential[node], potential[edge.to]);
                    if (reduced < 0.0 && reduced > -1.0e-12) reduced = 0.0;
                    if (!std::isfinite(reduced) || reduced < 0.0) {
                        std::ostringstream witness;
                        const bool reverseOfLatestCycle = std::find(
                            lastCycleReverseEdges.begin(), lastCycleReverseEdges.end(),
                            std::pair<std::uint32_t, std::uint32_t>{node, edgeIndex}) !=
                                lastCycleReverseEdges.end();
                        witness << std::setprecision(17)
                            << "spatial transport residual dual became infeasible during column insertion"
                            << " [inserted_from=" << from << " inserted_to=" << to
                            << " inserted_reduced=" << reducedColumn
                            << " source=" << node << " edge=" << edgeIndex
                            << " target=" << edge.to << " reduced=" << reduced
                            << " residual_capacity=" << edge.capacity
                            << " initial_capacity=" << edge.initialCapacity
                            << " edge_cost=" << edge.cost
                            << " potential_from=" << potential[node]
                            << " potential_to=" << potential[edge.to]
                            << " cycle_number=" << cycleNumber
                            << " latest_cycle_reverse_arc_count=" << lastCycleReverseEdges.size()
                            << " is_reverse_of_latest_cycle=" << (reverseOfLatestCycle ? 1 : 0)
                            << " is_inserted_column_reverse="
                            << ((node == to && edgeIndex == insertedReverseIndex) ? 1 : 0) << ']';
                        error = witness.str();
                        return false;
                    }
                    const double candidateDistance = knownDistance + reduced;
                    if (candidateDistance < distance[edge.to]) {
                        distance[edge.to] = candidateDistance;
                        previousNode[edge.to] = node;
                        previousEdge[edge.to] = edgeIndex;
                        queue.push({candidateDistance, edge.to});
                    }
                }
            }

            const bool pathExists = std::isfinite(distance[from]);
            const double dualStepLimit =
                !pathExists || distance[from] >= -reducedColumn - 1.0e-12
                    ? -reducedColumn : distance[from];
            if (!std::isfinite(dualStepLimit) || dualStepLimit < 0.0)
                return fail("spatial transport column-insertion dual step is invalid");
            for (std::size_t node = 0; node < graph_.size(); ++node) {
                const double step = std::isfinite(distance[node])
                    ? std::min(distance[node], dualStepLimit) : dualStepLimit;
                potential[node] = static_cast<double>(
                    static_cast<long double>(potential[node]) + step);
                if (!std::isfinite(potential[node]))
                    return fail("spatial transport column-insertion potential became nonfinite");
            }
            if (!normalizePotentialGauge(potential))
                return fail("spatial transport column-insertion potential gauge normalization failed");
            const double repairedReducedColumn =
                stableReducedCost(cost, potential[from], potential[to]);
            if (!std::isfinite(repairedReducedColumn))
                return fail("spatial transport column-insertion dual became nonfinite");
            if (repairedReducedColumn >= -1.0e-12) break;
            if (!pathExists)
                return fail("spatial transport negative column has no residual return path");
            if (augmentationCount >= maximumAugmentations)
                return fail("spatial transport exceeded its pinned augmentation budget during column-cycle repair");

            std::vector<std::pair<std::uint32_t, std::uint32_t>> path;
            double amount = pathCapacity(graph_[from][insertedEdgeIndex]);
            for (std::uint32_t node = from; node != to; node = previousNode[node]) {
                if (previousNode[node] == kInvalid)
                    return fail("spatial transport column cycle path is incomplete");
                const auto pathFrom = previousNode[node];
                const auto pathEdgeIndex = previousEdge[node];
                amount = std::min(amount,
                    pathCapacity(graph_[pathFrom][pathEdgeIndex]));
                path.emplace_back(pathFrom, pathEdgeIndex);
            }
            if (!std::isfinite(amount) || amount <= 0.0)
                return fail("spatial transport negative column cycle has no positive residual capacity");
            auto& inserted = graph_[from][insertedEdgeIndex];
            addCapacity(inserted, -amount);
            addCapacity(graph_[to][inserted.reverse], amount);
            lastCycleReverseEdges.clear();
            lastCycleReverseEdges.emplace_back(to, insertedReverseIndex);
            for (const auto [pathFrom, pathEdgeIndex] : path) {
                auto& edge = graph_[pathFrom][pathEdgeIndex];
                const auto pathTo = edge.to;
                const auto reverseIndex = edge.reverse;
                addCapacity(edge, -amount);
                addCapacity(graph_[pathTo][reverseIndex], amount);
                lastCycleReverseEdges.emplace_back(pathTo, reverseIndex);
            }
            ++cycleNumber;
            ++augmentationCount;
        }
        error.clear();
        return true;
    }

    [[nodiscard]] bool hasResidualSourceCapacity(const std::uint32_t source) const {
        if (source >= graph_.size()) return false;
        for (const auto& edge : graph_[source])
            if (edge.initialCapacity > 0.0 && hasPositiveCapacity(edge)) return true;
        return false;
    }

    [[nodiscard]] double sentFromSource(const std::uint32_t source) const {
        if (source >= graph_.size()) return 0.0;
        double sum = 0.0;
        double compensation = 0.0;
        for (const auto& edge : graph_[source]) {
            if (edge.initialCapacity <= 0.0) continue;
            const double value = roundedCapacity(graph_[edge.to][edge.reverse]);
            const double adjusted = value - compensation;
            const double next = sum + adjusted;
            compensation = (next - sum) - adjusted;
            sum = next;
        }
        return sum;
    }

    [[nodiscard]] double remainingCapacity(const std::uint32_t from,
                                           const std::uint32_t edge) const {
        if (from >= graph_.size() || edge >= graph_[from].size())
            return std::numeric_limits<double>::quiet_NaN();
        return roundedCapacity(graph_[from][edge]);
    }

    [[nodiscard]] bool hasResidualCapacity(const std::uint32_t from,
                                           const std::uint32_t edge) const {
        return from < graph_.size() && edge < graph_[from].size() &&
               hasPositiveCapacity(graph_[from][edge]);
    }

    [[nodiscard]] double reducedCost(const std::uint32_t from,
                                    const std::uint32_t to,
                                    const double cost,
                                    const std::vector<double>& potential) const {
        return stableReducedCost(cost, potential[from], potential[to]);
    }

    [[nodiscard]] bool residualReducedCostsNonnegative(
            const std::vector<double>& potential, const double tolerance,
            std::string& error) const {
        if (potential.size() != graph_.size() || !std::isfinite(tolerance) ||
            tolerance < 0.0) {
            error = "spatial transport dual certificate has invalid dimensions or tolerance";
            return false;
        }
        for (std::size_t from = 0; from < graph_.size(); ++from) {
            for (const auto& edge : graph_[from]) {
                if (!hasPositiveCapacity(edge)) continue;
                const double reduced = stableReducedCost(
                    edge.cost, potential[from], potential[edge.to]);
                if (!std::isfinite(reduced) || reduced < -tolerance) {
                    error = "spatial transport residual dual certificate has a negative reduced-cost arc";
                    return false;
                }
            }
        }
        error.clear();
        return true;
    }

    // Failure-only min-cut witness. It observes the same residual capacities
    // used by run(); it changes neither costs, augmentations nor accepted mass.
    [[nodiscard]] std::vector<bool> residualReachable(const std::uint32_t source,
                                                      const bool includeSubEpsilon = false) const {
        std::vector<bool> reached(graph_.size(), false);
        std::vector<std::uint32_t> queue{source};
        reached[source] = true;
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            for (const auto& edge : graph_[queue[cursor]]) {
                if ((includeSubEpsilon ? !hasPositiveCapacity(edge)
                                       : edge.capacity <= kFlowEpsilonKg) ||
                    reached[edge.to]) continue;
                reached[edge.to] = true;
                queue.push_back(edge.to);
            }
        }
        return reached;
    }

    [[nodiscard]] double flow(const EdgeRef& reference) const {
        const auto& edge = graph_[reference.from][reference.edge];
        return compensatedSparseCapacities_
            ? reverseFlow(reference) : edge.initialCapacity - edge.capacity;
    }
    [[nodiscard]] double reverseFlow(const EdgeRef& reference) const {
        const auto& edge = graph_[reference.from][reference.edge];
        return roundedCapacity(graph_[edge.to][edge.reverse]);
    }
private:
    std::vector<std::vector<FlowEdge>> graph_;
    bool compensatedSparseCapacities_ = false;
};

inline void hashInputArrays(const ConservativeSpatialMassTransportInput& in,
                            std::string& parentHash, std::string& childHash,
                            std::string& boundHash) {
    SHA256Builder parent;
    parent.string("conservativeSpatialMassTransport.parent.v1");
    parent.u64(in.parentMassKg.size());
    for (std::size_t i = 0; i < in.parentMassKg.size(); ++i) {
        parent.u32(static_cast<std::uint32_t>(i));
        parent.u32(in.parentMaterialIndices[i]);
        parent.u32(in.parentDonorBodyIndices[i]);
        parent.u32(in.parentAnatomicalRegionIndices[i]);
        parent.real(in.parentReferenceVolumeM3[i]);
        parent.real(in.parentMassKg[i]);
        for (const auto coordinate : in.parentCentroidsWorld[i]) parent.real(coordinate);
        parent.u32(in.parentSourceDonorOffsets[i]);
        parent.u32(in.parentSourceDonorOffsets[i + 1u]);
        for (std::uint32_t row = in.parentSourceDonorOffsets[i];
             row < in.parentSourceDonorOffsets[i + 1u]; ++row) {
            parent.u32(in.parentSourceDonorBodyIndices[row]);
            parent.real(in.parentSourceDonorMassKg[row]);
        }
    }
    parentHash = parent.finish();

    SHA256Builder child;
    child.string("conservativeSpatialMassTransport.child.v1");
    child.u64(in.childReferenceVolumeM3.size());
    for (std::size_t i = 0; i < in.childReferenceVolumeM3.size(); ++i) {
        child.u32(static_cast<std::uint32_t>(i));
        child.u32(in.childMaterialIndices[i]);
        child.u32(in.childDonorBodyIndices[i]);
        child.u32(in.childAnatomicalRegionIndices[i]);
        child.real(in.childReferenceVolumeM3[i]);
        for (const auto coordinate : in.childCentroidsWorld[i]) child.real(coordinate);
    }
    childHash = child.finish();

    std::set<std::uint32_t> materials;
    materials.insert(in.parentMaterialIndices.begin(), in.parentMaterialIndices.end());
    materials.insert(in.childMaterialIndices.begin(), in.childMaterialIndices.end());
    SHA256Builder bounds;
    bounds.string("conservativeSpatialMassTransport.material-bounds.v1");
    bounds.u64(materials.size());
    for (const auto material : materials) {
        bounds.u32(material);
        bounds.real(in.materialDensityLowerKgM3[material]);
        bounds.real(in.materialDensityUpperKgM3[material]);
    }
    boundHash = bounds.finish();
}

inline std::string receiptSHA256(const ConservativeSpatialMassTransportInput& in,
                                 const ConservativeSpatialMassTransportResult& result) {
    SHA256Builder receipt;
    receipt.string("conservativeSpatialMassTransport.receipt.v1");
    receipt.u32(static_cast<std::uint32_t>(result.method));
    receipt.string(in.parentMassPacketSHA256);
    receipt.string(in.parentSourceContributionSHA256);
    receipt.string(in.childMeshSHA256);
    receipt.string(in.regionLabelSHA256);
    receipt.string(in.coordinateFrameSHA256);
    receipt.string(result.parentDataSHA256);
    receipt.string(result.childDataSHA256);
    receipt.string(result.materialBoundsSHA256);
    receipt.real(in.maximumTransportDistanceM);
    receipt.real(in.relativeClosureTolerance);
    receipt.real(in.absoluteMassToleranceKg);
    receipt.real(in.absoluteMeasureToleranceM3);
    receipt.u64(in.maximumAllowedEdges);
    receipt.u64(in.maximumAugmentations);
    receipt.u64(result.transportEdgeCount);
    receipt.u64(result.augmentationCount);
    for (std::size_t e = 0; e < result.childParentRows.size(); ++e) {
        const auto child = std::upper_bound(result.childParentOffsets.begin(),
                                            result.childParentOffsets.end(),
                                            static_cast<std::uint32_t>(e)) -
                           result.childParentOffsets.begin() - 1;
        receipt.u32(static_cast<std::uint32_t>(child));
        receipt.u32(result.childParentRows[e]);
        receipt.real(result.childParentTransportMassKg[e]);
        receipt.real(result.childParentTransportMeasureM3[e]);
        receipt.real(result.childParentTransportDistanceM[e]);
    }
    receipt.u64(result.childSourceDonorBodyIndices.size());
    for (std::size_t c = 0; c + 1u < result.childSourceDonorOffsets.size(); ++c) {
        receipt.u32(result.childSourceDonorOffsets[c]);
        receipt.u32(result.childSourceDonorOffsets[c + 1u]);
        for (std::uint32_t row = result.childSourceDonorOffsets[c];
             row < result.childSourceDonorOffsets[c + 1u]; ++row) {
            receipt.u32(result.childSourceDonorBodyIndices[row]);
            receipt.real(result.childSourceDonorMassKg[row]);
        }
    }
    for (const auto& source : result.sourceDonorClosure) {
        receipt.u32(source.materialIndex);
        receipt.u32(source.sourceDonorBodyIndex);
        receipt.real(source.parentMassKg);
        receipt.real(source.childMassKg);
    }
    for (const auto& group : result.groupClosure) {
        receipt.u32(group.materialIndex);
        receipt.u32(group.donorBodyIndex);
        receipt.u32(group.anatomicalRegionIndex);
        receipt.real(group.parentMassKg);
        receipt.real(group.childMassKg);
        receipt.real(group.childMinimumCapacityKg);
        receipt.real(group.childMaximumCapacityKg);
    }
    return receipt.finish();
}

} // namespace spatial_mass_transport_detail

// The coupling is an initialization-time mass transfer constrained to the same
// material, registered donor body and explicit anatomical region. A child cell
// may receive only from parents within maximumTransportDistanceM. Child demand
// is bounded by its emitted reference volume times the owning material density
// bounds. The deterministic min-cost flow minimizes mass-weighted squared
// centroid distance among feasible couplings. The dense path is unchanged for
// graphs within budget; overflow uses sparse min-cut expansion and globally
// feasible residual-dual pricing. Both paths retain the same label, distance,
// density, edge and aggregate augmentation limits.
[[nodiscard]] inline bool assembleConservativeSpatialMassTransportImpl(
        const ConservativeSpatialMassTransportInput& in,
        ConservativeSpatialMassTransportResult& out,
        std::string& error, const bool sparseMode,
        const std::size_t priorAugmentationCount) {
    using namespace spatial_mass_transport_detail;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    const std::size_t np = in.parentMassKg.size();
    const std::size_t nc = in.childReferenceVolumeM3.size();
    if (np == 0u || nc == 0u || np > std::numeric_limits<std::uint32_t>::max() ||
        nc > std::numeric_limits<std::uint32_t>::max() ||
        in.parentCentroidsWorld.size() != np || in.parentReferenceVolumeM3.size() != np ||
        in.parentMaterialIndices.size() != np || in.parentDonorBodyIndices.size() != np ||
        in.parentAnatomicalRegionIndices.size() != np ||
        in.parentSourceDonorOffsets.size() != np + 1u ||
        in.parentSourceDonorOffsets.empty() || in.parentSourceDonorOffsets.front() != 0u ||
        in.parentSourceDonorOffsets.back() != in.parentSourceDonorBodyIndices.size() ||
        in.parentSourceDonorBodyIndices.size() != in.parentSourceDonorMassKg.size() ||
        in.childCentroidsWorld.size() != nc || in.childMaterialIndices.size() != nc ||
        in.childDonorBodyIndices.size() != nc || in.childAnatomicalRegionIndices.size() != nc ||
        in.materialDensityLowerKgM3.empty() ||
        in.materialDensityUpperKgM3.size() != in.materialDensityLowerKgM3.size() ||
        !std::isfinite(in.maximumTransportDistanceM) || in.maximumTransportDistanceM <= 0.0 ||
        !std::isfinite(in.relativeClosureTolerance) || in.relativeClosureTolerance < 0.0 ||
        !std::isfinite(in.absoluteMassToleranceKg) || in.absoluteMassToleranceKg < 0.0 ||
        !std::isfinite(in.absoluteMeasureToleranceM3) || in.absoluteMeasureToleranceM3 < 0.0 ||
        in.maximumAllowedEdges == 0u || in.maximumAugmentations == 0u)
        return fail("spatial transport input arrays, distance, tolerances or budgets are invalid");
    if (!validSHA256(in.parentMassPacketSHA256) ||
        !validSHA256(in.parentSourceContributionSHA256) ||
        !validSHA256(in.childMeshSHA256) || !validSHA256(in.regionLabelSHA256) ||
        !validSHA256(in.coordinateFrameSHA256))
        return fail("spatial transport requires pinned parent, source, child, region and coordinate-frame SHA-256 inputs");

    for (std::size_t p = 0; p < np; ++p) {
        const auto material = in.parentMaterialIndices[p];
        if (material >= in.materialDensityLowerKgM3.size() ||
            in.parentDonorBodyIndices[p] == kInvalid ||
            in.parentAnatomicalRegionIndices[p] == kInvalid ||
            !std::isfinite(in.parentReferenceVolumeM3[p]) || in.parentReferenceVolumeM3[p] <= 0.0 ||
            !std::isfinite(in.parentMassKg[p]) || in.parentMassKg[p] <= 0.0 ||
            !std::all_of(in.parentCentroidsWorld[p].begin(), in.parentCentroidsWorld[p].end(),
                         [](double value) { return std::isfinite(value); }))
            return fail("spatial transport parent row has invalid volume, mass, material, donor, region or centroid");
        const double lower = in.materialDensityLowerKgM3[material];
        const double upper = in.materialDensityUpperKgM3[material];
        const double density = in.parentMassKg[p] / in.parentReferenceVolumeM3[p];
        const double densityTolerance = densityBoundRoundoff(lower, upper, density);
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower > upper ||
            !std::isfinite(density) || density < lower - densityTolerance ||
            density > upper + densityTolerance)
            return fail("authoritative parent cell density is outside the owning material bounds");
        const auto begin = in.parentSourceDonorOffsets[p];
        const auto end = in.parentSourceDonorOffsets[p + 1u];
        if (begin >= end || end > in.parentSourceDonorBodyIndices.size())
            return fail("spatial transport parent source-donor CSR row is empty or out of range");
        double sourceMass = 0.0;
        std::uint32_t previousDonor = 0u;
        for (std::uint32_t row = begin; row < end; ++row) {
            const auto sourceDonor = in.parentSourceDonorBodyIndices[row];
            const double mass = in.parentSourceDonorMassKg[row];
            if (sourceDonor == kInvalid || (row > begin && sourceDonor <= previousDonor) ||
                !std::isfinite(mass) || mass <= 0.0)
                return fail("spatial transport parent source-donor split is invalid, unsorted or duplicated");
            previousDonor = sourceDonor;
            sourceMass += mass;
        }
        if (std::abs(sourceMass - in.parentMassKg[p]) >
            std::max(in.absoluteMassToleranceKg,
                     in.relativeClosureTolerance * in.parentMassKg[p]))
            return fail("actual parent source-donor contributions do not close the authoritative cell mass");
    }
    for (std::size_t c = 0; c < nc; ++c) {
        const auto material = in.childMaterialIndices[c];
        if (material >= in.materialDensityLowerKgM3.size() ||
            in.childDonorBodyIndices[c] == kInvalid ||
            in.childAnatomicalRegionIndices[c] == kInvalid ||
            !std::isfinite(in.childReferenceVolumeM3[c]) || in.childReferenceVolumeM3[c] <= 0.0 ||
            !std::all_of(in.childCentroidsWorld[c].begin(), in.childCentroidsWorld[c].end(),
                         [](double value) { return std::isfinite(value); }))
            return fail("spatial transport child row has invalid volume, material, donor, region or centroid");
        const double lower = in.materialDensityLowerKgM3[material];
        const double upper = in.materialDensityUpperKgM3[material];
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower > upper)
            return fail("spatial transport child material density bounds are invalid");
    }

    ConservativeSpatialMassTransportResult next;
    next.augmentationCount = priorAugmentationCount;
    next.method = UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    next.parentMassPacketSHA256 = std::string(in.parentMassPacketSHA256);
    next.parentSourceContributionSHA256 = std::string(in.parentSourceContributionSHA256);
    next.childMeshSHA256 = std::string(in.childMeshSHA256);
    next.regionLabelSHA256 = std::string(in.regionLabelSHA256);
    next.coordinateFrameSHA256 = std::string(in.coordinateFrameSHA256);
    next.maximumTransportDistanceM = in.maximumTransportDistanceM;
    next.relativeClosureTolerance = in.relativeClosureTolerance;
    next.absoluteMassToleranceKg = in.absoluteMassToleranceKg;
    next.absoluteMeasureToleranceM3 = in.absoluteMeasureToleranceM3;
    next.maximumAllowedEdges = in.maximumAllowedEdges;
    next.maximumAugmentations = in.maximumAugmentations;
    hashInputArrays(in, next.parentDataSHA256, next.childDataSHA256,
                    next.materialBoundsSHA256);
    next.parentMassTransferredKg.assign(np, 0.0);
    next.parentMeasureTransferredM3.assign(np, 0.0);
    next.childMassKg.assign(nc, 0.0);
    next.childImpliedDensityKgM3.assign(nc, 0.0);
    next.childAncestryMeasureVolumeM3.assign(nc, 0.0);
    next.childDominantParentRows.assign(nc, kInvalid);
    next.childSourceDonorOffsets.assign(nc + 1u, 0u);
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> parentSourceMassByDonor;
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> childSourceMassByDonor;
    std::vector<std::map<std::uint32_t, double>> childSourceMassByCell(nc);
    for (std::size_t p = 0; p < np; ++p) {
        const auto keyMaterial = in.parentMaterialIndices[p];
        for (std::uint32_t row = in.parentSourceDonorOffsets[p];
             row < in.parentSourceDonorOffsets[p + 1u]; ++row)
            parentSourceMassByDonor[{keyMaterial, in.parentSourceDonorBodyIndices[row]}] +=
                in.parentSourceDonorMassKg[row];
    }

    using GridBucket = std::pair<GroupKey, GridKey>;
    std::map<GroupKey, std::vector<std::uint32_t>> parentsByGroup, childrenByGroup;
    std::map<GridBucket, std::vector<std::uint32_t>> childrenByBucket;
    std::vector<GridKey> parentGrid(np), childGrid(nc);
    for (std::size_t p = 0; p < np; ++p) {
        const GroupKey key{in.parentMaterialIndices[p], in.parentDonorBodyIndices[p],
                           in.parentAnatomicalRegionIndices[p]};
        parentsByGroup[key].push_back(static_cast<std::uint32_t>(p));
        if (!gridKey(in.parentCentroidsWorld[p], in.maximumTransportDistanceM, parentGrid[p]))
            return fail("spatial transport parent centroid cannot be indexed by the bounded search grid");
    }
    for (std::size_t c = 0; c < nc; ++c) {
        const GroupKey key{in.childMaterialIndices[c], in.childDonorBodyIndices[c],
                           in.childAnatomicalRegionIndices[c]};
        childrenByGroup[key].push_back(static_cast<std::uint32_t>(c));
        if (!gridKey(in.childCentroidsWorld[c], in.maximumTransportDistanceM, childGrid[c]))
            return fail("spatial transport child centroid cannot be indexed by the bounded search grid");
        childrenByBucket[{key, childGrid[c]}].push_back(static_cast<std::uint32_t>(c));
    }
    for (const auto& [key, children] : childrenByGroup) {
        (void)children;
        if (!parentsByGroup.contains(key))
            return fail("spatial transport child anatomical/material/donor group has no parent supply");
    }
    for (const auto& [key, parents] : parentsByGroup) {
        (void)parents;
        if (!childrenByGroup.contains(key))
            return fail("spatial transport parent anatomical/material/donor supply has no child demand");
    }

    struct FlowRecord {
        std::uint32_t parent = 0u, child = 0u;
        double massKg = 0.0, measureM3 = 0.0, distanceM = 0.0;
    };
    std::vector<FlowRecord> flows;
    std::size_t allCandidateEdges = 0u;
    const double radiusSquared = in.maximumTransportDistanceM * in.maximumTransportDistanceM;
    if (!sparseMode) {
        // Decide whether the dense path fits before doing any group flow work.
        // This prevents a dense-prefix solve from being replayed and counted a
        // second time when a later group crosses the edge budget.
        std::size_t denseCandidateEdges = 0u;
        bool denseBudgetExceeded = false;
        for (const auto& [groupKey, groupParents] : parentsByGroup) {
            for (const auto p : groupParents) {
                const auto base = parentGrid[p];
                for (int dx = -1; dx <= 1 && !denseBudgetExceeded; ++dx)
                    for (int dy = -1; dy <= 1 && !denseBudgetExceeded; ++dy)
                        for (int dz = -1; dz <= 1 && !denseBudgetExceeded; ++dz) {
                            const GridKey grid{base.x + dx, base.y + dy, base.z + dz};
                            const auto found = childrenByBucket.find({groupKey, grid});
                            if (found == childrenByBucket.end()) continue;
                            for (const auto c : found->second) {
                                const double d = distance(in.parentCentroidsWorld[p],
                                                          in.childCentroidsWorld[c]);
                                if (d > in.maximumTransportDistanceM ||
                                    d*d > radiusSquared * (1.0 + 1.0e-14))
                                    continue;
                                if (denseCandidateEdges >= in.maximumAllowedEdges) {
                                    denseBudgetExceeded = true;
                                    break;
                                }
                                ++denseCandidateEdges;
                            }
                        }
                if (denseBudgetExceeded) break;
            }
            if (denseBudgetExceeded) break;
        }
        if (denseBudgetExceeded)
            return assembleConservativeSpatialMassTransportImpl(
                in, out, error, true, priorAugmentationCount);
    }
    for (const auto& [group, parentRows] : parentsByGroup) {
        // A failed preparation must retain the exact conservation partition.
        // This is diagnostic context only; capacities and policy are unchanged.
        const auto groupWitness = [&](const double supply, const double minimum,
                const double maximum, const double sent) {
            std::ostringstream context;
            context << std::setprecision(17)
                << " [material=" << std::get<0>(group)
                << " donor=" << std::get<1>(group)
                << " region=" << std::get<2>(group)
                << " parent_rows=" << parentRows.size()
                << " child_rows=" << childrenByGroup.at(group).size()
                << " supply_kg=" << supply << " sent_kg=" << sent
                << " child_min_kg=" << minimum << " child_max_kg=" << maximum
                << " distance_m=" << in.maximumTransportDistanceM << ']';
            return context.str();
        };
        const auto& childRows = childrenByGroup[group];
        long double supplyLong = 0.0L, minimumLong = 0.0L, maximumLong = 0.0L;
        spatial_mass_transport_detail::ExactDoubleSum exactSupply, exactMinimum, exactMaximum;
        std::vector<double> exactChildMinimum, exactChildMaximum;
        exactChildMinimum.reserve(childRows.size());
        exactChildMaximum.reserve(childRows.size());
        for (const auto p : parentRows) {
            supplyLong += in.parentMassKg[p];
            exactSupply.add(in.parentMassKg[p]);
        }
        for (const auto c : childRows) {
            const double childMinimum = in.childReferenceVolumeM3[c] *
                in.materialDensityLowerKgM3[in.childMaterialIndices[c]];
            const double childMaximum = in.childReferenceVolumeM3[c] *
                in.materialDensityUpperKgM3[in.childMaterialIndices[c]];
            minimumLong += childMinimum;
            maximumLong += childMaximum;
            exactMinimum.add(childMinimum);
            exactMaximum.add(childMaximum);
            exactChildMinimum.push_back(childMinimum);
            exactChildMaximum.push_back(childMaximum);
        }
        const double supply = sparseMode ? exactSupply.rounded()
                                         : static_cast<double>(supplyLong);
        const double minimum = sparseMode ? exactMinimum.rounded()
                                          : static_cast<double>(minimumLong);
        const double maximum = sparseMode ? exactMaximum.rounded()
                                          : static_cast<double>(maximumLong);
        spatial_mass_transport_detail::ExactDoubleSum sourceMinusMinimum = exactSupply;
        spatial_mass_transport_detail::ExactDoubleSum sourceMinusMaximum = exactSupply;
        for (const auto value : exactChildMinimum) sourceMinusMinimum.add(-value);
        for (const auto value : exactChildMaximum) sourceMinusMaximum.add(-value);
        const bool sparseBoundsValid = exactSupply.valid() && exactMinimum.valid() &&
            exactMaximum.valid() && sourceMinusMinimum.valid() && sourceMinusMaximum.valid() &&
            sourceMinusMinimum.sign() >= 0 && sourceMinusMaximum.sign() <= 0;
        if (!std::isfinite(supply) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
            (sparseMode ? !sparseBoundsValid : (supply < minimum || supply > maximum)))
            return fail("spatial transport parent supply is outside child volume times actual material density bounds for material/donor/region group" +
                groupWitness(supply,minimum,maximum,0.0));

        struct ParentCandidates {
            std::uint32_t parent = 0u;
            std::vector<Candidate> candidates;
        };
        std::vector<ParentCandidates> candidates;
        candidates.reserve(parentRows.size());
        std::size_t groupCandidateEdges = 0u;
        for (const auto p : parentRows) {
            ParentCandidates row;
            row.parent = p;
            const auto base = parentGrid[p];
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const GridKey grid{base.x + dx, base.y + dy, base.z + dz};
                        const auto found = childrenByBucket.find({group, grid});
                        if (found == childrenByBucket.end()) continue;
                        for (const auto c : found->second) {
                            const double d = distance(in.parentCentroidsWorld[p],
                                                      in.childCentroidsWorld[c]);
                            if (d <= in.maximumTransportDistanceM && d*d <= radiusSquared * (1.0 + 1.0e-14))
                                row.candidates.push_back({c, d});
                        }
                    }
            std::sort(row.candidates.begin(), row.candidates.end(), [](const auto& a, const auto& b) {
                if (a.distanceM != b.distanceM) return a.distanceM < b.distanceM;
                return a.child < b.child;
            });
            row.candidates.erase(std::unique(row.candidates.begin(), row.candidates.end(),
                [](const auto& a, const auto& b) { return a.child == b.child; }), row.candidates.end());
            if (row.candidates.empty())
                return fail("spatial transport parent has no same-material/donor/region child within the pinned distance bound");
            if (sparseMode) {
                // The full set is generated only for one parent at a time.
                // Keep two deterministic seed columns; every omitted column is
                // later considered by the exact reduced-cost pricing scan.
                if (row.candidates.size() > 2u) row.candidates.resize(2u);
                const std::size_t remaining =
                    in.maximumAllowedEdges - std::min(allCandidateEdges, in.maximumAllowedEdges);
                if (groupCandidateEdges > remaining ||
                    row.candidates.size() > remaining - groupCandidateEdges)
                    return fail("spatial transport sparse seed exceeds its pinned edge budget");
                groupCandidateEdges += row.candidates.size();
            } else {
                if (row.candidates.size() > in.maximumAllowedEdges -
                        std::min(allCandidateEdges, in.maximumAllowedEdges))
                    return assembleConservativeSpatialMassTransportImpl(
                        in, out, error, true, next.augmentationCount);
                allCandidateEdges += row.candidates.size();
                groupCandidateEdges += row.candidates.size();
            }
            candidates.push_back(std::move(row));
        }

        const std::uint32_t parentCount = static_cast<std::uint32_t>(parentRows.size());
        const std::uint32_t childCount = static_cast<std::uint32_t>(childRows.size());
        const std::uint32_t parentBase = 0u;
        const std::uint32_t childBase = parentCount;
        const std::uint32_t sink = parentCount + childCount;
        const std::uint32_t superSource = sink + 1u;
        const std::uint32_t superSink = sink + 2u;
        MinCostFlow network(static_cast<std::size_t>(superSink) + 1u);
        std::map<std::uint32_t, std::uint32_t> localChild;
        std::vector<EdgeRef> references;
        struct MandatoryFloorArc {
            std::uint32_t childRow = 0u;
            std::uint32_t from = 0u;
            std::uint32_t edge = 0u;
            double requiredKg = 0.0;
        };
        std::vector<MandatoryFloorArc> mandatoryFloorArcs;
        std::vector<std::unordered_set<std::uint32_t>> activeChildByParent(parentCount);
        if (sparseMode) {
            for (std::uint32_t i = 0; i < parentCount; ++i)
                for (const auto& candidate : candidates[i].candidates)
                    activeChildByParent[i].insert(candidate.child);
        }

        long double childMinimumLong = 0.0L;
        std::vector<double> childMinimumByLocal(childCount), childMaximumByLocal(childCount);
        for (std::uint32_t i = 0; i < childCount; ++i) {
            const auto c = childRows[i];
            const double childMinimum = in.childReferenceVolumeM3[c] *
                in.materialDensityLowerKgM3[in.childMaterialIndices[c]];
            const double childMaximum = in.childReferenceVolumeM3[c] *
                in.materialDensityUpperKgM3[in.childMaterialIndices[c]];
            if (!std::isfinite(childMinimum) || !std::isfinite(childMaximum) ||
                childMinimum <= 0.0 || childMaximum < childMinimum)
                return fail("spatial transport child density capacity is invalid");
            childMinimumByLocal[i] = childMinimum;
            childMaximumByLocal[i] = childMaximum;
            childMinimumLong += childMinimum;
        }
        double residualCapacity = supply - static_cast<double>(childMinimumLong);
        if (sparseMode) {
            auto exactResidualCapacity = exactSupply;
            for (const auto childMinimum : childMinimumByLocal)
                exactResidualCapacity.add(-childMinimum);
            if (!exactResidualCapacity.valid() || exactResidualCapacity.sign() < 0)
                return fail("spatial transport mandatory child density floors exceed exact parent mass supply");
            residualCapacity = exactResidualCapacity.roundedDownPositive();
        }
        if (residualCapacity < 0.0)
            return fail("spatial transport mandatory child density floors exceed parent mass supply");

        const auto rebuildNetwork = [&]() {
            network = MinCostFlow(static_cast<std::size_t>(superSink) + 1u);
            localChild.clear();
            for (std::uint32_t i = 0; i < childCount; ++i) localChild[childRows[i]] = i;
            references.clear();
            references.reserve(groupCandidateEdges);
            mandatoryFloorArcs.clear();
            mandatoryFloorArcs.reserve(childCount);
            for (std::uint32_t i = 0; i < parentCount; ++i)
                network.addEdge(superSource, parentBase + i, in.parentMassKg[parentRows[i]], 0.0);
            for (std::uint32_t i = 0; i < parentCount; ++i) {
                const auto p = parentRows[i];
                for (const auto& candidate : candidates[i].candidates) {
                    const auto cLocal = localChild.at(candidate.child);
                    const double cost = (candidate.distanceM * candidate.distanceM) / radiusSquared;
                    const auto edge = network.addEdge(parentBase + i, childBase + cLocal,
                                                      in.parentMassKg[p], cost);
                    references.push_back({p, candidate.child, parentBase + i, edge,
                                          candidate.distanceM});
                }
            }
            for (std::uint32_t i = 0; i < childCount; ++i) {
                const double childMinimum = childMinimumByLocal[i];
                const double childMaximum = childMaximumByLocal[i];
                const auto floorEdge = network.addEdge(childBase + i, superSink,
                                                       childMinimum, 0.0);
                mandatoryFloorArcs.push_back({childRows[i], childBase + i,
                                              floorEdge, childMinimum});
                network.addEdge(childBase + i, sink, childMaximum - childMinimum, 0.0);
            }
            network.addEdge(sink, superSink, residualCapacity, 0.0);
        };
        rebuildNetwork();

        const auto cutWitness = [&](const std::vector<bool>& reachable) {
            std::ostringstream cut;
            cut << " [cut_parent_rows=[";
            bool first = true;
            for (std::uint32_t i = 0; i < parentCount; ++i) {
                if (!reachable[parentBase + i]) continue;
                if (!first) cut << ',';
                first = false;
                cut << parentRows[i];
            }
            cut << "] cut_child_rows=[";
            first = true;
            for (std::uint32_t i = 0; i < childCount; ++i) {
                if (!reachable[childBase + i]) continue;
                if (!first) cut << ',';
                first = false;
                cut << childRows[i];
            }
            cut << "]]";
            return cut.str();
        };
        const auto enumerateAdmissible = [&](const std::uint32_t parent,
                                              const auto& callback) {
            const auto base = parentGrid[parent];
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const GridKey grid{base.x + dx, base.y + dy, base.z + dz};
                        const auto found = childrenByBucket.find({group, grid});
                        if (found == childrenByBucket.end()) continue;
                        for (const auto c : found->second) {
                            const double d = distance(in.parentCentroidsWorld[parent],
                                                      in.childCentroidsWorld[c]);
                            if (d <= in.maximumTransportDistanceM &&
                                d*d <= radiusSquared * (1.0 + 1.0e-14))
                                callback(Candidate{c, d});
                        }
                    }
        };

        if (!sparseMode) {
            std::size_t augmentations = 0u;
            double sent = 0.0;
            std::string flowError;
            const auto remainingAugmentations = in.maximumAugmentations -
                std::min(next.augmentationCount, in.maximumAugmentations);
            if (remainingAugmentations == 0u)
                return fail("spatial transport exceeded its global pinned augmentation budget");
            if (!network.run(superSource, superSink, supply, remainingAugmentations,
                             augmentations, sent, flowError))
                return fail("spatial transport min-cost flow failed: " + flowError);
            next.augmentationCount += augmentations;
            if (std::abs(sent - supply) >
                std::max(in.absoluteMassToleranceKg, in.relativeClosureTolerance * supply)) {
                const auto reachable = network.residualReachable(superSource);
                return fail("spatial transport is infeasible under same-material/donor/region and distance bounds" +
                    groupWitness(supply,minimum,maximum,sent) + cutWitness(reachable));
            }
        } else {
            struct PricedColumn {
                std::uint32_t parentLocal = 0u;
                Candidate candidate;
                double reducedCost = 0.0;
            };
            const auto columnLess = [](const PricedColumn& a, const PricedColumn& b) {
                if (a.reducedCost != b.reducedCost) return a.reducedCost < b.reducedCost;
                if (a.parentLocal != b.parentLocal) return a.parentLocal < b.parentLocal;
                return a.candidate.child < b.candidate.child;
            };
            constexpr std::size_t kColumnBatch = 256u;
            bool optimalityCertified = false;
            std::vector<double> dual;
            double sent = 0.0;
            std::size_t pricingRound = 0u;
            std::size_t cycleAugmentations = 0u;
            std::size_t insertedColumns = 0u;
            const auto progress = [&](const std::string_view stage) {
                std::ostringstream context;
                context << " [stage=" << stage
                    << " material=" << std::get<0>(group)
                    << " donor=" << std::get<1>(group)
                    << " region=" << std::get<2>(group)
                    << " round=" << pricingRound
                    << " cumulative_augmentations=" << next.augmentationCount
                    << " cycle_augmentations=" << cycleAugmentations
                    << " inserted_columns=" << insertedColumns
                    << " active_edges=" << groupCandidateEdges << ']';
                return context.str();
            };
            const auto hasUnsaturatedFloor = [&]() {
                return std::any_of(mandatoryFloorArcs.begin(), mandatoryFloorArcs.end(),
                    [&](const MandatoryFloorArc& arc) {
                        return network.hasResidualCapacity(arc.from, arc.edge);
                    });
            };
            const auto floorWitness = [&]() {
                std::ostringstream witness;
                for (const auto& arc : mandatoryFloorArcs) {
                    const double remaining = network.remainingCapacity(arc.from, arc.edge);
                    if (!network.hasResidualCapacity(arc.from, arc.edge)) continue;
                    witness << std::setprecision(17)
                        << " [unsaturated_floor_child=" << arc.childRow
                        << " required_kg=" << arc.requiredKg
                        << " residual_kg=" << remaining << ']';
                    break;
                }
                return witness.str();
            };
            const auto runRestrictedMaster = [&](const std::string_view stage) {
                if (!network.hasResidualSourceCapacity(superSource)) return true;
                const auto remainingAugmentations = in.maximumAugmentations -
                    std::min(next.augmentationCount, in.maximumAugmentations);
                if (remainingAugmentations == 0u) {
                    error = "spatial transport exceeded its global pinned augmentation budget" +
                            progress(stage);
                    return false;
                }
                std::size_t augmentations = 0u;
                double updatedSent = sent;
                std::string flowError;
                std::vector<double> updatedDual;
                const auto* initialDual = dual.empty() ? nullptr : &dual;
                if (!network.run(superSource, superSink, supply, remainingAugmentations,
                                 augmentations, updatedSent, flowError, &updatedDual, true,
                                 initialDual, sent, true)) {
                    next.augmentationCount += augmentations;
                    error = "spatial transport sparse restricted min-cost flow failed: " +
                            flowError + progress(stage);
                    return false;
                }
                next.augmentationCount += augmentations;
                sent = updatedSent;
                dual = std::move(updatedDual);
                return true;
            };
            const auto insertColumn = [&](const std::uint32_t parentLocal,
                                          const Candidate& candidate,
                                          const std::string_view stage) {
                const auto p = parentRows[parentLocal];
                const auto childFound = localChild.find(candidate.child);
                if (childFound == localChild.end()) {
                    error = "spatial transport sparse column references a child outside its group" +
                            progress(stage);
                    return false;
                }
                const double cost = (candidate.distanceM * candidate.distanceM) / radiusSquared;
                std::uint32_t edgeIndex = kInvalid;
                const auto before = next.augmentationCount;
                std::string columnError;
                if (!network.addColumnWithOptimalityRepair(
                        parentBase + parentLocal, childBase + childFound->second,
                        in.parentMassKg[p], cost, dual, in.maximumAugmentations,
                        next.augmentationCount, edgeIndex, columnError)) {
                    cycleAugmentations += next.augmentationCount - before;
                    error = "spatial transport sparse column insertion failed: " +
                            columnError + progress(stage);
                    return false;
                }
                cycleAugmentations += next.augmentationCount - before;
                references.push_back({p, candidate.child, parentBase + parentLocal,
                                      edgeIndex, candidate.distanceM});
                auto& row = candidates[parentLocal].candidates;
                const auto position = std::lower_bound(row.begin(), row.end(), candidate,
                    [](const Candidate& a, const Candidate& b) {
                        return a.distanceM != b.distanceM ? a.distanceM < b.distanceM
                                                           : a.child < b.child;
                    });
                row.insert(position, candidate);
                activeChildByParent[parentLocal].insert(candidate.child);
                ++groupCandidateEdges;
                ++insertedColumns;
                return true;
            };

            if (!runRestrictedMaster("initial-restricted-master")) return false;
            while (!optimalityCertified) {
                ++pricingRound;
                const double closureTolerance =
                    std::max(in.absoluteMassToleranceKg, in.relativeClosureTolerance * supply);
                if (std::abs(sent - supply) > closureTolerance || hasUnsaturatedFloor()) {
                    const auto reachable = network.residualReachable(superSource, true);
                    std::vector<PricedColumn> additions;
                    const std::size_t remainingEdges = in.maximumAllowedEdges -
                        std::min(in.maximumAllowedEdges, allCandidateEdges + groupCandidateEdges);
                    const std::size_t batchLimit = std::min(kColumnBatch, remainingEdges);
                    bool omittedCutCrossing = false;
                    for (std::uint32_t i = 0; i < parentCount; ++i) {
                        if (batchLimit > 0u && additions.size() >= batchLimit) break;
                        const auto p = parentRows[i];
                        if (!reachable[parentBase + i]) continue;
                        enumerateAdmissible(p, [&](const Candidate& candidate) {
                            if (activeChildByParent[i].contains(candidate.child)) return;
                            const auto child = localChild.find(candidate.child);
                            if (child == localChild.end() ||
                                reachable[childBase + child->second]) return;
                            omittedCutCrossing = true;
                            if (additions.size() < batchLimit)
                                additions.push_back({i, candidate, 0.0});
                        });
                    }
                    if (additions.empty()) {
                        if (omittedCutCrossing)
                            return fail("spatial transport sparse edge budget prevents min-cut feasibility expansion" +
                                groupWitness(supply,minimum,maximum,sent) + progress("min-cut"));
                        if (hasUnsaturatedFloor())
                            return fail("spatial transport is infeasible under same-material/donor/region and distance bounds: sparse solve cannot saturate a mandatory child density floor" +
                                floorWitness() + groupWitness(supply,minimum,maximum,sent) +
                                progress("min-cut"));
                        return fail("spatial transport is infeasible under same-material/donor/region and distance bounds" +
                            groupWitness(supply,minimum,maximum,sent) + progress("min-cut"));
                    }
                    std::sort(additions.begin(), additions.end(), [&](const auto& a, const auto& b) {
                        if (a.candidate.distanceM != b.candidate.distanceM)
                            return a.candidate.distanceM < b.candidate.distanceM;
                        if (parentRows[a.parentLocal] != parentRows[b.parentLocal])
                            return parentRows[a.parentLocal] < parentRows[b.parentLocal];
                        return a.candidate.child < b.candidate.child;
                    });
                    for (const auto& addition : additions) {
                        if (!insertColumn(addition.parentLocal, addition.candidate, "min-cut-column"))
                            return false;
                    }
                    if (!runRestrictedMaster("min-cut-resume")) return false;
                    continue;
                }

                std::string flowError;
                if (!network.residualReducedCostsNonnegative(dual, 1.0e-12, flowError))
                    return fail("spatial transport sparse solve lacks a globally feasible residual dual: " +
                                flowError + progress("dual-check"));
                std::vector<PricedColumn> negativeColumns;
                negativeColumns.reserve(kColumnBatch);
                bool invalidReducedCost = false;
                for (std::uint32_t i = 0; i < parentCount; ++i) {
                    const auto p = parentRows[i];
                    enumerateAdmissible(p, [&](const Candidate& candidate) {
                        if (activeChildByParent[i].contains(candidate.child)) return;
                        const auto child = localChild.find(candidate.child);
                        if (child == localChild.end()) return;
                        const double cost = (candidate.distanceM * candidate.distanceM) / radiusSquared;
                        const double reduced = network.reducedCost(
                            parentBase + i, childBase + child->second, cost, dual);
                        if (!std::isfinite(reduced)) {
                            invalidReducedCost = true;
                            return;
                        }
                        if (reduced >= -1.0e-12) return;
                        PricedColumn column{i, candidate, reduced};
                        if (negativeColumns.size() < kColumnBatch) {
                            negativeColumns.push_back(column);
                            std::push_heap(negativeColumns.begin(), negativeColumns.end(), columnLess);
                        } else if (columnLess(column, negativeColumns.front())) {
                            std::pop_heap(negativeColumns.begin(), negativeColumns.end(), columnLess);
                            negativeColumns.back() = column;
                            std::push_heap(negativeColumns.begin(), negativeColumns.end(), columnLess);
                        }
                    });
                }
                if (invalidReducedCost)
                    return fail("spatial transport sparse pricing computed a nonfinite omitted-edge reduced cost" +
                                progress("pricing"));
                if (negativeColumns.empty()) {
                    optimalityCertified = true;
                    break;
                }
                std::sort(negativeColumns.begin(), negativeColumns.end(), columnLess);
                const std::size_t remainingEdges = in.maximumAllowedEdges -
                    std::min(in.maximumAllowedEdges, allCandidateEdges + groupCandidateEdges);
                if (remainingEdges == 0u)
                    return fail("spatial transport sparse edge budget prevents full reduced-cost optimality certification" +
                                progress("pricing"));
                if (negativeColumns.size() > remainingEdges)
                    negativeColumns.resize(remainingEdges);
                for (const auto& addition : negativeColumns) {
                    if (!insertColumn(addition.parentLocal, addition.candidate, "priced-column"))
                        return false;
                }
                if (!runRestrictedMaster("pricing-resume")) return false;
            }
            if (sparseMode && hasUnsaturatedFloor())
                return fail("spatial transport sparse optimality left a mandatory child density floor unsaturated" +
                            floorWitness() + groupWitness(supply,minimum,maximum,sent));
            allCandidateEdges += groupCandidateEdges;
        }
        double groupOutputMass = 0.0;
        for (const auto& reference : references) {
            // Dense receipt arithmetic remains unchanged. Sparse flows use
            // the reverse residual capacity, which retains small edge masses
            // without subtracting them from a much larger parent capacity.
            const double mass = sparseMode ? network.reverseFlow(reference)
                                           : network.flow(reference);
            if (sparseMode ? mass <= 0.0 : mass <= kFlowEpsilonKg) continue;
            const auto p = reference.parent;
            const auto c = reference.child;
            const double parentDensity = in.parentMassKg[p] / in.parentReferenceVolumeM3[p];
            const double measure = mass / parentDensity;
            if (!std::isfinite(mass) || !std::isfinite(measure) || measure <= 0.0 ||
                reference.distanceM > in.maximumTransportDistanceM)
                return fail("spatial transport flow produced an invalid mass, source measure or distance");
            flows.push_back({p, c, mass, measure, reference.distanceM});
            next.parentMassTransferredKg[p] += mass;
            next.parentMeasureTransferredM3[p] += measure;
            next.childMassKg[c] += mass;
            next.childAncestryMeasureVolumeM3[c] += measure;
            const double parentFraction = mass / in.parentMassKg[p];
            for (std::uint32_t sourceRow = in.parentSourceDonorOffsets[p];
                 sourceRow < in.parentSourceDonorOffsets[p + 1u]; ++sourceRow) {
                const auto sourceDonor = in.parentSourceDonorBodyIndices[sourceRow];
                const double sourceMass =
                    in.parentSourceDonorMassKg[sourceRow] * parentFraction;
                childSourceMassByDonor[{in.parentMaterialIndices[p], sourceDonor}] +=
                    sourceMass;
                childSourceMassByCell[c][sourceDonor] += sourceMass;
            }
            groupOutputMass += mass;
            next.maximumTransportDistanceUsedM =
                std::max(next.maximumTransportDistanceUsedM, reference.distanceM);
        }
        next.groupClosure.push_back({
            .materialIndex = std::get<0>(group),
            .donorBodyIndex = std::get<1>(group),
            .anatomicalRegionIndex = std::get<2>(group),
            .parentMassKg = supply,
            .childMassKg = groupOutputMass,
            .childMinimumCapacityKg = minimum,
            .childMaximumCapacityKg = maximum});
    }
    if (allCandidateEdges > in.maximumAllowedEdges)
        return fail("spatial transport allowed-edge count exceeds its pinned budget");
    if (sparseMode) {
        // The verifier reconstructs each child mass from canonical CSR order.
        // Sparse columns can be inserted after their peers during pricing, so
        // replay those sums only after sorting to make the bound and recorded
        // density decisions use identical floating-point addition order.
        if (!sortAndAccumulateChildFlowSums(flows, nc, next.childMassKg,
                                            next.childAncestryMeasureVolumeM3))
            return fail("spatial transport sparse canonical child-flow accumulation is invalid");
    } else {
        std::sort(flows.begin(), flows.end(), [](const auto& a, const auto& b) {
            if (a.child != b.child) return a.child < b.child;
            return a.parent < b.parent;
        });
    }
    // A child-parent pair is unique because every parent and child has one
    // material/donor/region key; duplicates indicate malformed grouping.
    for (std::size_t i = 1; i < flows.size(); ++i)
        if (flows[i-1].child == flows[i].child && flows[i-1].parent == flows[i].parent)
            return fail("spatial transport produced duplicate child-parent rows");

    next.childParentOffsets.assign(nc + 1u, 0u);
    next.childParentOffsets.reserve(nc + 1u);
    next.childParentRows.reserve(flows.size());
    next.childParentTransportMassKg.reserve(flows.size());
    next.childParentTransportMeasureM3.reserve(flows.size());
    next.childParentTransportDistanceM.reserve(flows.size());
    next.childParentVolumeFractions.reserve(flows.size());
    std::vector<std::uint32_t> flowCountByChild(nc, 0u);
    for (const auto& flow : flows) ++flowCountByChild[flow.child];
    std::size_t flowAt = 0u;
    long double weightedDistance = 0.0L, totalMass = 0.0L;
    for (std::size_t c = 0; c < nc; ++c) {
        if (flowCountByChild[c] == 0u || next.childAncestryMeasureVolumeM3[c] <= 0.0 ||
            next.childMassKg[c] <= 0.0)
            return fail("spatial transport left a child cell without a positive bounded mass assignment");
        next.childParentOffsets[c] = static_cast<std::uint32_t>(next.childParentRows.size());
        double fractionSum = 0.0, bestMeasure = -1.0;
        std::uint32_t dominant = kInvalid;
        while (flowAt < flows.size() && flows[flowAt].child == c) {
            const auto& flow = flows[flowAt++];
            const double fraction = flow.measureM3 / next.childAncestryMeasureVolumeM3[c];
            if (!std::isfinite(fraction) || fraction <= 0.0)
                return fail("spatial transport child-parent source measure fraction is invalid");
            next.childParentRows.push_back(flow.parent);
            next.childParentTransportMassKg.push_back(flow.massKg);
            next.childParentTransportMeasureM3.push_back(flow.measureM3);
            next.childParentTransportDistanceM.push_back(flow.distanceM);
            next.childParentVolumeFractions.push_back(fraction);
            fractionSum += fraction;
            if (flow.measureM3 > bestMeasure ||
                (flow.measureM3 == bestMeasure && flow.parent < dominant)) {
                bestMeasure = flow.measureM3;
                dominant = flow.parent;
            }
            weightedDistance += static_cast<long double>(flow.massKg) * flow.distanceM;
            totalMass += flow.massKg;
        }
        next.childParentOffsets[c + 1u] = static_cast<std::uint32_t>(next.childParentRows.size());
        next.childSourceDonorOffsets[c] =
            static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        double sourceMassForChild = 0.0;
        for (const auto& [sourceDonor, sourceMass] : childSourceMassByCell[c]) {
            if (sourceDonor == kInvalid || !std::isfinite(sourceMass) || sourceMass <= 0.0 ||
                next.childSourceDonorBodyIndices.size() >=
                    std::numeric_limits<std::uint32_t>::max())
                return fail("spatial transport child source-donor output row is invalid or exceeds CSR range");
            next.childSourceDonorBodyIndices.push_back(sourceDonor);
            next.childSourceDonorMassKg.push_back(sourceMass);
            sourceMassForChild += sourceMass;
        }
        next.childSourceDonorOffsets[c + 1u] =
            static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        if (std::abs(sourceMassForChild - next.childMassKg[c]) >
            std::max(in.absoluteMassToleranceKg,
                     in.relativeClosureTolerance * next.childMassKg[c]))
            return fail("spatial transport child source-donor rows do not close child mass");
        next.maximumChildFractionSumError = std::max(next.maximumChildFractionSumError,
                                                     std::abs(fractionSum - 1.0));
        if (std::abs(fractionSum - 1.0) > 5.0e-13)
            return fail("spatial transport child-parent measure fractions do not sum to one");
        next.childDominantParentRows[c] = dominant;
        const double density = next.childMassKg[c] / in.childReferenceVolumeM3[c];
        const auto material = in.childMaterialIndices[c];
        const double densityTolerance = densityBoundRoundoff(
            in.materialDensityLowerKgM3[material], in.materialDensityUpperKgM3[material], density);
        if (!std::isfinite(density) ||
            density < in.materialDensityLowerKgM3[material] - densityTolerance ||
            density > in.materialDensityUpperKgM3[material] + densityTolerance) {
            const double lowerMass = in.childReferenceVolumeM3[c] *
                in.materialDensityLowerKgM3[material];
            std::ostringstream diagnostic;
            diagnostic << std::setprecision(17)
                << "spatial transport child cell mass implies density outside actual material bounds at child "
                << c << " (density=" << density
                << ", mass_kg=" << next.childMassKg[c]
                << ", volume_m3=" << in.childReferenceVolumeM3[c]
                << ", lower_mass_kg=" << lowerMass
                << ", lower_mass_deficit_kg=" << (lowerMass - next.childMassKg[c])
                << ", density_tolerance=" << densityTolerance
                << ", bounds=" << in.materialDensityLowerKgM3[material] << ':'
                << in.materialDensityUpperKgM3[material]
                << ", contribution_rows="
                << (next.childParentOffsets[c + 1u] - next.childParentOffsets[c])
                << ", contributors=[";
            bool firstContributor = true;
            for (std::uint32_t row = next.childParentOffsets[c];
                 row < next.childParentOffsets[c + 1u]; ++row) {
                if (!firstContributor) diagnostic << ';';
                firstContributor = false;
                diagnostic << next.childParentRows[row] << ','
                    << next.childParentTransportMassKg[row] << ','
                    << next.childParentTransportMeasureM3[row] << ','
                    << next.childParentTransportDistanceM[row];
            }
            diagnostic << "])";
            return fail(diagnostic.str());
        }
        next.childImpliedDensityKgM3[c] = density;
    }
    next.childParentOffsets[nc] = static_cast<std::uint32_t>(next.childParentRows.size());
    next.childSourceDonorOffsets[nc] =
        static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
    next.transportEdgeCount = next.childParentRows.size();
    next.massWeightedMeanTransportDistanceM = totalMass > 0.0L
        ? static_cast<double>(weightedDistance / totalMass) : 0.0;
    next.minimumChildImpliedDensityKgM3 = std::numeric_limits<double>::infinity();
    next.maximumChildImpliedDensityKgM3 = 0.0;
    for (const auto density : next.childImpliedDensityKgM3) {
        next.minimumChildImpliedDensityKgM3 =
            std::min(next.minimumChildImpliedDensityKgM3, density);
        next.maximumChildImpliedDensityKgM3 =
            std::max(next.maximumChildImpliedDensityKgM3, density);
    }

    for (std::size_t p = 0; p < np; ++p) {
        const double massError = std::abs(next.parentMassTransferredKg[p] - in.parentMassKg[p]);
        const double measureError = std::abs(next.parentMeasureTransferredM3[p] - in.parentReferenceVolumeM3[p]);
        next.maximumParentMassClosureErrorKg = std::max(next.maximumParentMassClosureErrorKg, massError);
        next.maximumParentMeasureClosureErrorM3 = std::max(next.maximumParentMeasureClosureErrorM3, measureError);
        if (massError > std::max(in.absoluteMassToleranceKg,
                                 in.relativeClosureTolerance * in.parentMassKg[p]) ||
            measureError > std::max(in.absoluteMeasureToleranceM3,
                                    in.relativeClosureTolerance * in.parentReferenceVolumeM3[p]))
            return fail("spatial transport parent supply or source reference measure does not close");
    }
    for (const auto& group : next.groupClosure) {
        if (std::abs(group.parentMassKg - group.childMassKg) >
            std::max(in.absoluteMassToleranceKg,
                     in.relativeClosureTolerance * group.parentMassKg))
            return fail("spatial transport material/donor/anatomical-region mass does not close");
    }
    std::set<std::pair<std::uint32_t, std::uint32_t>> sourceDonorKeys;
    for (const auto& [key, mass] : parentSourceMassByDonor) { (void)mass; sourceDonorKeys.insert(key); }
    for (const auto& [key, mass] : childSourceMassByDonor) { (void)mass; sourceDonorKeys.insert(key); }
    for (const auto& key : sourceDonorKeys) {
        const double parentMass = parentSourceMassByDonor[key];
        const double childMass = childSourceMassByDonor[key];
        const double closureError = std::abs(parentMass - childMass);
        next.maximumSourceDonorMassClosureErrorKg =
            std::max(next.maximumSourceDonorMassClosureErrorKg, closureError);
        if (!std::isfinite(parentMass) || !std::isfinite(childMass) ||
            closureError > std::max(in.absoluteMassToleranceKg,
                                    in.relativeClosureTolerance * parentMass))
            return fail("spatial transport source-donor mass does not close");
        next.sourceDonorClosure.push_back({.materialIndex = key.first,
            .sourceDonorBodyIndex = key.second, .parentMassKg = parentMass,
            .childMassKg = childMass});
    }

    next.receiptSHA256 = receiptSHA256(in, next);
    out = std::move(next);
    error.clear();
    return true;
}

[[nodiscard]] inline bool assembleConservativeSpatialMassTransport(
        const ConservativeSpatialMassTransportInput& in,
        ConservativeSpatialMassTransportResult& out,
        std::string& error) {
    return assembleConservativeSpatialMassTransportImpl(in, out, error, false, 0u);
}

// Verify a recorded spatial coupling against freshly derived source and child
// cell data without rerunning min-cost flow. This checks edge feasibility,
// source measures, mass/measure columns, child density capacities, source-donor
// composition, material/donor/region closures, and both numeric and receipt hashes.
[[nodiscard]] inline bool verifyRecordedConservativeSpatialMassTransport(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialMassTransportResult& supplied,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    const std::size_t np = in.parentMassKg.size();
    const std::size_t nc = in.childReferenceVolumeM3.size();
    if (np == 0u || nc == 0u || np > std::numeric_limits<std::uint32_t>::max() ||
        nc > std::numeric_limits<std::uint32_t>::max() ||
        in.parentCentroidsWorld.size() != np || in.parentReferenceVolumeM3.size() != np ||
        in.parentMaterialIndices.size() != np || in.parentDonorBodyIndices.size() != np ||
        in.parentAnatomicalRegionIndices.size() != np ||
        in.parentSourceDonorOffsets.size() != np + 1u ||
        in.parentSourceDonorOffsets.empty() || in.parentSourceDonorOffsets.front() != 0u ||
        in.parentSourceDonorOffsets.back() != in.parentSourceDonorBodyIndices.size() ||
        in.parentSourceDonorBodyIndices.size() != in.parentSourceDonorMassKg.size() ||
        in.childCentroidsWorld.size() != nc || in.childMaterialIndices.size() != nc ||
        in.childDonorBodyIndices.size() != nc || in.childAnatomicalRegionIndices.size() != nc ||
        in.materialDensityLowerKgM3.empty() ||
        in.materialDensityUpperKgM3.size() != in.materialDensityLowerKgM3.size() ||
        !std::isfinite(in.maximumTransportDistanceM) || in.maximumTransportDistanceM <= 0.0 ||
        !std::isfinite(in.relativeClosureTolerance) || in.relativeClosureTolerance < 0.0 ||
        !std::isfinite(in.absoluteMassToleranceKg) || in.absoluteMassToleranceKg < 0.0 ||
        !std::isfinite(in.absoluteMeasureToleranceM3) || in.absoluteMeasureToleranceM3 < 0.0 ||
        in.maximumAllowedEdges == 0u || in.maximumAugmentations == 0u)
        return fail("recorded spatial transport input arrays, distance, tolerances or budgets are invalid");
    if (!validSHA256(in.parentMassPacketSHA256) ||
        !validSHA256(in.parentSourceContributionSHA256) ||
        !validSHA256(in.childMeshSHA256) || !validSHA256(in.regionLabelSHA256) ||
        !validSHA256(in.coordinateFrameSHA256))
        return fail("recorded spatial transport requires pinned parent, source, child, region and coordinate-frame SHA-256 inputs");
    if (supplied.method != UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport ||
        supplied.parentMassPacketSHA256 != in.parentMassPacketSHA256 ||
        supplied.parentSourceContributionSHA256 != in.parentSourceContributionSHA256 ||
        supplied.childMeshSHA256 != in.childMeshSHA256 ||
        supplied.regionLabelSHA256 != in.regionLabelSHA256 ||
        supplied.coordinateFrameSHA256 != in.coordinateFrameSHA256 ||
        supplied.maximumTransportDistanceM != in.maximumTransportDistanceM ||
        supplied.relativeClosureTolerance != in.relativeClosureTolerance ||
        supplied.absoluteMassToleranceKg != in.absoluteMassToleranceKg ||
        supplied.absoluteMeasureToleranceM3 != in.absoluteMeasureToleranceM3 ||
        supplied.maximumAllowedEdges != in.maximumAllowedEdges ||
        supplied.maximumAugmentations != in.maximumAugmentations)
        return fail("recorded spatial transport policy or external input hashes differ from current admission inputs");

    std::string parentHash, childHash, boundsHash;
    hashInputArrays(in, parentHash, childHash, boundsHash);
    if (supplied.parentDataSHA256 != parentHash)
        return fail("recorded spatial transport numeric parent mass/source/geometry arrays changed");
    if (supplied.childDataSHA256 != childHash)
        return fail("recorded spatial transport numeric child geometry/region arrays changed");
    if (supplied.materialBoundsSHA256 != boundsHash)
        return fail("recorded spatial transport actual material density bounds changed");
    if (supplied.childParentOffsets.size() != nc + 1u ||
        supplied.childParentOffsets.empty() || supplied.childParentOffsets.front() != 0u ||
        supplied.childParentOffsets.back() != supplied.childParentRows.size() ||
        supplied.childParentRows.empty() ||
        supplied.childParentTransportMassKg.size() != supplied.childParentRows.size() ||
        supplied.childParentTransportMeasureM3.size() != supplied.childParentRows.size() ||
        supplied.childParentTransportDistanceM.size() != supplied.childParentRows.size() ||
        supplied.childParentVolumeFractions.size() != supplied.childParentRows.size() ||
        supplied.parentMassTransferredKg.size() != np ||
        supplied.parentMeasureTransferredM3.size() != np ||
        supplied.childAncestryMeasureVolumeM3.size() != nc || supplied.childMassKg.size() != nc ||
        supplied.childImpliedDensityKgM3.size() != nc ||
        supplied.childDominantParentRows.size() != nc ||
        supplied.childSourceDonorOffsets.size() != nc + 1u ||
        supplied.childSourceDonorOffsets.empty() || supplied.childSourceDonorOffsets.front() != 0u ||
        supplied.childSourceDonorOffsets.back() != supplied.childSourceDonorBodyIndices.size() ||
        supplied.childSourceDonorBodyIndices.size() != supplied.childSourceDonorMassKg.size() ||
        supplied.transportEdgeCount != supplied.childParentRows.size() ||
        supplied.transportEdgeCount > in.maximumAllowedEdges ||
        supplied.augmentationCount == 0u || supplied.augmentationCount > in.maximumAugmentations)
        return fail("recorded spatial transport CSR, edge or augmentation counts are invalid");

    const auto withinMass = [&](double a, double b) {
        return std::isfinite(a) && std::isfinite(b) &&
            std::abs(a - b) <= std::max(in.absoluteMassToleranceKg,
                                        in.relativeClosureTolerance * std::max(std::abs(a), std::abs(b)));
    };
    const auto withinMeasure = [&](double a, double b) {
        return std::isfinite(a) && std::isfinite(b) &&
            std::abs(a - b) <= std::max(in.absoluteMeasureToleranceM3,
                                        in.relativeClosureTolerance * std::max(std::abs(a), std::abs(b)));
    };
    std::vector<double> parentMass(np, 0.0), parentMeasure(np, 0.0);
    std::vector<double> childMass(nc, 0.0), childMeasure(nc, 0.0);
    std::vector<std::map<std::uint32_t, double>> childSourceByCell(nc);
    std::map<GroupKey, double> parentGroupMass, childGroupMass;
    std::map<GroupKey, double> groupMinCapacity, groupMaxCapacity;
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> parentSourceByDonor,
        childSourceByDonor;
    std::vector<std::uint32_t> parentSourceEnd(np, 0u);
    for (std::size_t p = 0; p < np; ++p) {
        const auto material = in.parentMaterialIndices[p];
        if (material >= in.materialDensityLowerKgM3.size() ||
            in.parentDonorBodyIndices[p] == kInvalid ||
            in.parentAnatomicalRegionIndices[p] == kInvalid ||
            !std::isfinite(in.parentReferenceVolumeM3[p]) || in.parentReferenceVolumeM3[p] <= 0.0 ||
            !std::isfinite(in.parentMassKg[p]) || in.parentMassKg[p] <= 0.0 ||
            !std::all_of(in.parentCentroidsWorld[p].begin(), in.parentCentroidsWorld[p].end(),
                         [](double value) { return std::isfinite(value); }))
            return fail("recorded spatial transport current parent cell is invalid");
        const double lower = in.materialDensityLowerKgM3[material];
        const double upper = in.materialDensityUpperKgM3[material];
        const double density = in.parentMassKg[p] / in.parentReferenceVolumeM3[p];
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower > upper ||
            !std::isfinite(density) || density < lower || density > upper)
            return fail("recorded spatial transport authoritative parent density is outside material bounds");
        const auto begin = in.parentSourceDonorOffsets[p];
        const auto end = in.parentSourceDonorOffsets[p + 1u];
        if (begin >= end || end > in.parentSourceDonorBodyIndices.size())
            return fail("recorded spatial transport parent source-donor CSR row is empty or out of range");
        double rowMass = 0.0;
        std::uint32_t previous = 0u;
        for (std::uint32_t row = begin; row < end; ++row) {
            const auto donor = in.parentSourceDonorBodyIndices[row];
            const double mass = in.parentSourceDonorMassKg[row];
            if (donor == kInvalid || (row > begin && donor <= previous) ||
                !std::isfinite(mass) || mass <= 0.0)
                return fail("recorded spatial transport parent source-donor CSR row is invalid");
            previous = donor;
            rowMass += mass;
            parentSourceByDonor[{material, donor}] += mass;
        }
        if (!withinMass(rowMass, in.parentMassKg[p]))
            return fail("recorded spatial transport actual source donors do not close parent cell mass");
        parentSourceEnd[p] = end;
        parentGroupMass[{material, in.parentDonorBodyIndices[p],
                         in.parentAnatomicalRegionIndices[p]}] += in.parentMassKg[p];
    }
    for (std::size_t c = 0; c < nc; ++c) {
        const auto material = in.childMaterialIndices[c];
        if (material >= in.materialDensityLowerKgM3.size() ||
            in.childDonorBodyIndices[c] == kInvalid ||
            in.childAnatomicalRegionIndices[c] == kInvalid ||
            !std::isfinite(in.childReferenceVolumeM3[c]) || in.childReferenceVolumeM3[c] <= 0.0 ||
            !std::all_of(in.childCentroidsWorld[c].begin(), in.childCentroidsWorld[c].end(),
                         [](double value) { return std::isfinite(value); }))
            return fail("recorded spatial transport current child cell is invalid");
        const double lower = in.materialDensityLowerKgM3[material];
        const double upper = in.materialDensityUpperKgM3[material];
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower > upper)
            return fail("recorded spatial transport current child material bounds are invalid");
        const GroupKey key{material, in.childDonorBodyIndices[c],
                           in.childAnatomicalRegionIndices[c]};
        const double minimum = in.childReferenceVolumeM3[c] * lower;
        const double maximum = in.childReferenceVolumeM3[c] * upper;
        if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum <= 0.0 || maximum < minimum)
            return fail("recorded spatial transport current child density capacity is invalid");
        groupMinCapacity[key] += minimum;
        groupMaxCapacity[key] += maximum;
    }

    double maxDistance = 0.0;
    long double weightedDistance = 0.0L, totalMass = 0.0L;
    double maxFractionError = 0.0;
    std::vector<std::uint32_t> dominantRows(nc, kInvalid);
    for (std::size_t c = 0; c < nc; ++c) {
        const auto begin = supplied.childParentOffsets[c];
        const auto end = supplied.childParentOffsets[c + 1u];
        if (begin >= end || end > supplied.childParentRows.size())
            return fail("recorded spatial transport child mass row is empty or out of range");
        std::uint32_t prior = 0u;
        double bestMeasure = -1.0;
        for (std::uint32_t e = begin; e < end; ++e) {
            const auto p = supplied.childParentRows[e];
            const double mass = supplied.childParentTransportMassKg[e];
            const double measure = supplied.childParentTransportMeasureM3[e];
            const double recordedDistance = supplied.childParentTransportDistanceM[e];
            if (p >= np || (e > begin && p <= prior) || !std::isfinite(mass) || mass <= 0.0 ||
                !std::isfinite(measure) || measure <= 0.0 ||
                in.parentMaterialIndices[p] != in.childMaterialIndices[c] ||
                in.parentDonorBodyIndices[p] != in.childDonorBodyIndices[c] ||
                in.parentAnatomicalRegionIndices[p] != in.childAnatomicalRegionIndices[c])
                return fail("recorded spatial transport edge has invalid or cross-material/donor/region ancestry");
            prior = p;
            const double actualDistance = distance(in.parentCentroidsWorld[p], in.childCentroidsWorld[c]);
            if (!std::isfinite(recordedDistance) || recordedDistance != actualDistance ||
                actualDistance > in.maximumTransportDistanceM)
                return fail("recorded spatial transport edge distance differs from current cells or exceeds its bound");
            const double parentDensity = in.parentMassKg[p] / in.parentReferenceVolumeM3[p];
            const double expectedMeasure = mass / parentDensity;
            if (measure != expectedMeasure)
                return fail("recorded spatial transport source measure differs from mass at current parent density");
            parentMass[p] += mass;
            parentMeasure[p] += measure;
            childMass[c] += mass;
            childMeasure[c] += measure;
            const GroupKey group{in.parentMaterialIndices[p], in.parentDonorBodyIndices[p],
                                 in.parentAnatomicalRegionIndices[p]};
            childGroupMass[group] += mass;
            const double parentFraction = mass / in.parentMassKg[p];
            for (std::uint32_t source = in.parentSourceDonorOffsets[p];
                 source < parentSourceEnd[p]; ++source) {
                const auto donor = in.parentSourceDonorBodyIndices[source];
                const double sourceMass = in.parentSourceDonorMassKg[source] * parentFraction;
                childSourceByCell[c][donor] += sourceMass;
                childSourceByDonor[{in.parentMaterialIndices[p], donor}] += sourceMass;
            }
            maxDistance = std::max(maxDistance, actualDistance);
            weightedDistance += static_cast<long double>(mass) * actualDistance;
            totalMass += mass;
            if (measure > bestMeasure || (measure == bestMeasure && p < dominantRows[c])) {
                bestMeasure = measure;
                dominantRows[c] = p;
            }
        }
    }

    std::vector<double> expectedFractions(supplied.childParentRows.size(), 0.0);
    double minDensity = std::numeric_limits<double>::infinity(), maxDensity = 0.0;
    for (std::size_t c = 0; c < nc; ++c) {
        if (!(childMass[c] > 0.0 && childMeasure[c] > 0.0) ||
            !withinMass(childMass[c], supplied.childMassKg[c]) ||
            !withinMeasure(childMeasure[c], supplied.childAncestryMeasureVolumeM3[c]))
            return fail("recorded spatial transport child row mass or measure does not close");
        double fractionSum = 0.0;
        for (std::uint32_t e = supplied.childParentOffsets[c];
             e < supplied.childParentOffsets[c + 1u]; ++e) {
            expectedFractions[e] = supplied.childParentTransportMeasureM3[e] / childMeasure[c];
            fractionSum += expectedFractions[e];
            if (!std::isfinite(supplied.childParentVolumeFractions[e]) ||
                std::abs(supplied.childParentVolumeFractions[e] - expectedFractions[e]) > 5.0e-13)
                return fail("recorded spatial transport child-parent measure fractions are inconsistent");
        }
        const double fractionError = std::abs(fractionSum - 1.0);
        maxFractionError = std::max(maxFractionError, fractionError);
        if (fractionError > 5.0e-13 || supplied.childDominantParentRows[c] != dominantRows[c])
            return fail("recorded spatial transport child fractions or dominant source row are invalid");
        const double density = childMass[c] / in.childReferenceVolumeM3[c];
        const auto material = in.childMaterialIndices[c];
        const double densityTolerance = densityBoundRoundoff(
            in.materialDensityLowerKgM3[material], in.materialDensityUpperKgM3[material], density);
        if (!std::isfinite(density) ||
            density < in.materialDensityLowerKgM3[material] - densityTolerance ||
            density > in.materialDensityUpperKgM3[material] + densityTolerance ||
            supplied.childImpliedDensityKgM3[c] != density)
            return fail("recorded spatial transport current child density is outside actual material bounds");
        minDensity = std::min(minDensity, density);
        maxDensity = std::max(maxDensity, density);
        const auto sourceBegin = supplied.childSourceDonorOffsets[c];
        const auto sourceEnd = supplied.childSourceDonorOffsets[c + 1u];
        if (sourceBegin >= sourceEnd || sourceEnd > supplied.childSourceDonorBodyIndices.size())
            return fail("recorded spatial transport child source-donor CSR row is empty or out of range");
        std::uint32_t previousDonor = 0u;
        double sourceMass = 0.0;
        auto expected = childSourceByCell[c].begin();
        for (std::uint32_t row = sourceBegin; row < sourceEnd; ++row) {
            const auto donor = supplied.childSourceDonorBodyIndices[row];
            const double mass = supplied.childSourceDonorMassKg[row];
            if (donor == kInvalid || (row > sourceBegin && donor <= previousDonor) ||
                !std::isfinite(mass) || mass <= 0.0 || expected == childSourceByCell[c].end() ||
                donor != expected->first || !withinMass(mass, expected->second))
                return fail("recorded spatial transport child source-donor lineage differs from its parent flows");
            previousDonor = donor;
            sourceMass += mass;
            ++expected;
        }
        if (expected != childSourceByCell[c].end() || !withinMass(sourceMass, childMass[c]))
            return fail("recorded spatial transport child source-donor mass does not close child mass");
    }
    if (supplied.childSourceDonorOffsets[nc] != supplied.childSourceDonorBodyIndices.size())
        return fail("recorded spatial transport child source-donor CSR terminal offset is invalid");

    double maxParentMassError = 0.0, maxParentMeasureError = 0.0;
    for (std::size_t p = 0; p < np; ++p) {
        const double massError = std::abs(parentMass[p] - in.parentMassKg[p]);
        const double measureError = std::abs(parentMeasure[p] - in.parentReferenceVolumeM3[p]);
        maxParentMassError = std::max(maxParentMassError, massError);
        maxParentMeasureError = std::max(maxParentMeasureError, measureError);
        if (!withinMass(parentMass[p], in.parentMassKg[p]) ||
            !withinMeasure(parentMeasure[p], in.parentReferenceVolumeM3[p]) ||
            !withinMass(parentMass[p], supplied.parentMassTransferredKg[p]) ||
            !withinMeasure(parentMeasure[p], supplied.parentMeasureTransferredM3[p]))
            return fail("recorded spatial transport parent supply or reference measure column does not close");
    }
    if (!withinMass(supplied.maximumParentMassClosureErrorKg, maxParentMassError) ||
        !withinMeasure(supplied.maximumParentMeasureClosureErrorM3, maxParentMeasureError) ||
        !std::isfinite(supplied.maximumChildFractionSumError) ||
        std::abs(supplied.maximumChildFractionSumError - maxFractionError) > 5.0e-13 ||
        supplied.maximumTransportDistanceUsedM != maxDistance ||
        supplied.massWeightedMeanTransportDistanceM !=
            (totalMass > 0.0L ? static_cast<double>(weightedDistance / totalMass) : 0.0L) ||
        supplied.minimumChildImpliedDensityKgM3 != minDensity ||
        supplied.maximumChildImpliedDensityKgM3 != maxDensity)
        return fail("recorded spatial transport closure or summary metrics differ from its flows");

    if (parentGroupMass.size() != childGroupMass.size() ||
        parentGroupMass.size() != groupMinCapacity.size() ||
        parentGroupMass.size() != groupMaxCapacity.size() ||
        supplied.groupClosure.size() != parentGroupMass.size())
        return fail("recorded spatial transport material/donor/region group set differs from current cells");
    std::size_t groupRow = 0u;
    for (const auto& [key, parentGroup] : parentGroupMass) {
        const auto childFound = childGroupMass.find(key);
        const double childGroup = childFound == childGroupMass.end() ? 0.0 : childFound->second;
        const auto& recorded = supplied.groupClosure[groupRow++];
        const double minimum = groupMinCapacity[key], maximum = groupMaxCapacity[key];
        if (childFound == childGroupMass.end() || !withinMass(parentGroup, childGroup) ||
            !withinMass(recorded.parentMassKg, parentGroup) ||
            !withinMass(recorded.childMassKg, childGroup) ||
            !withinMass(recorded.childMinimumCapacityKg, minimum) ||
            !withinMass(recorded.childMaximumCapacityKg, maximum) ||
            recorded.materialIndex != std::get<0>(key) ||
            recorded.donorBodyIndex != std::get<1>(key) ||
            recorded.anatomicalRegionIndex != std::get<2>(key) ||
            childGroup < minimum - std::max(in.absoluteMassToleranceKg,
                in.relativeClosureTolerance * std::max(std::abs(childGroup), std::abs(minimum))) ||
            childGroup > maximum + std::max(in.absoluteMassToleranceKg,
                in.relativeClosureTolerance * std::max(std::abs(childGroup), std::abs(maximum))))
            return fail("recorded spatial transport material/donor/region capacity or mass closure fails");
    }
    if (parentSourceByDonor.size() != childSourceByDonor.size() ||
        supplied.sourceDonorClosure.size() != parentSourceByDonor.size())
        return fail("recorded spatial transport source-donor set differs from current source ledger");
    double maxSourceError = 0.0;
    std::size_t sourceRow = 0u;
    for (const auto& [key, parentSource] : parentSourceByDonor) {
        const auto childFound = childSourceByDonor.find(key);
        const double childSource = childFound == childSourceByDonor.end() ? 0.0 : childFound->second;
        const double sourceError = std::abs(parentSource - childSource);
        maxSourceError = std::max(maxSourceError, sourceError);
        const auto& recorded = supplied.sourceDonorClosure[sourceRow++];
        if (childFound == childSourceByDonor.end() || !withinMass(parentSource, childSource) ||
            recorded.materialIndex != key.first || recorded.sourceDonorBodyIndex != key.second ||
            !withinMass(recorded.parentMassKg, parentSource) ||
            !withinMass(recorded.childMassKg, childSource))
            return fail("recorded spatial transport material/source-donor mass closure fails");
    }
    if (!withinMass(supplied.maximumSourceDonorMassClosureErrorKg, maxSourceError))
        return fail("recorded spatial transport maximum source-donor closure error differs from flows");
    if (supplied.receiptSHA256 != receiptSHA256(in, supplied))
        return fail("recorded spatial transport receipt digest does not bind the validated flow and source ledger");
    error.clear();
    return true;
}

// Rebuild the deterministic coupling from the pinned inputs and compare every
// emitted mass, measure, row and digest. Native admission should call this on
// the receipt-bound arrays before passing them to the existing unpaired source
// preparation path.
[[nodiscard]] inline bool verifyConservativeSpatialMassTransport(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialMassTransportResult& supplied,
        std::string& error) {
    ConservativeSpatialMassTransportResult expected;
    if (!assembleConservativeSpatialMassTransport(in, expected, error)) return false;
    if (supplied.method != expected.method ||
        supplied.childParentOffsets != expected.childParentOffsets ||
        supplied.childParentRows != expected.childParentRows ||
        supplied.childParentTransportMassKg != expected.childParentTransportMassKg ||
        supplied.childParentTransportMeasureM3 != expected.childParentTransportMeasureM3 ||
        supplied.childParentTransportDistanceM != expected.childParentTransportDistanceM ||
        supplied.childParentVolumeFractions != expected.childParentVolumeFractions ||
        supplied.childSourceDonorOffsets != expected.childSourceDonorOffsets ||
        supplied.childSourceDonorBodyIndices != expected.childSourceDonorBodyIndices ||
        supplied.childSourceDonorMassKg != expected.childSourceDonorMassKg ||
        supplied.childAncestryMeasureVolumeM3 != expected.childAncestryMeasureVolumeM3 ||
        supplied.childMassKg != expected.childMassKg ||
        supplied.childImpliedDensityKgM3 != expected.childImpliedDensityKgM3 ||
        supplied.childDominantParentRows != expected.childDominantParentRows ||
        supplied.parentMassTransferredKg != expected.parentMassTransferredKg ||
        supplied.parentMeasureTransferredM3 != expected.parentMeasureTransferredM3 ||
        supplied.transportEdgeCount != expected.transportEdgeCount ||
        supplied.augmentationCount != expected.augmentationCount ||
        supplied.maximumParentMassClosureErrorKg != expected.maximumParentMassClosureErrorKg ||
        supplied.maximumParentMeasureClosureErrorM3 != expected.maximumParentMeasureClosureErrorM3 ||
        supplied.maximumSourceDonorMassClosureErrorKg != expected.maximumSourceDonorMassClosureErrorKg ||
        supplied.maximumChildFractionSumError != expected.maximumChildFractionSumError ||
        supplied.maximumTransportDistanceUsedM != expected.maximumTransportDistanceUsedM ||
        supplied.massWeightedMeanTransportDistanceM != expected.massWeightedMeanTransportDistanceM ||
        supplied.minimumChildImpliedDensityKgM3 != expected.minimumChildImpliedDensityKgM3 ||
        supplied.maximumChildImpliedDensityKgM3 != expected.maximumChildImpliedDensityKgM3 ||
        supplied.parentDataSHA256 != expected.parentDataSHA256 ||
        supplied.childDataSHA256 != expected.childDataSHA256 ||
        supplied.materialBoundsSHA256 != expected.materialBoundsSHA256 ||
        supplied.parentMassPacketSHA256 != expected.parentMassPacketSHA256 ||
        supplied.parentSourceContributionSHA256 != expected.parentSourceContributionSHA256 ||
        supplied.childMeshSHA256 != expected.childMeshSHA256 ||
        supplied.regionLabelSHA256 != expected.regionLabelSHA256 ||
        supplied.coordinateFrameSHA256 != expected.coordinateFrameSHA256 ||
        supplied.maximumTransportDistanceM != expected.maximumTransportDistanceM ||
        supplied.relativeClosureTolerance != expected.relativeClosureTolerance ||
        supplied.absoluteMassToleranceKg != expected.absoluteMassToleranceKg ||
        supplied.absoluteMeasureToleranceM3 != expected.absoluteMeasureToleranceM3 ||
        supplied.maximumAllowedEdges != expected.maximumAllowedEdges ||
        supplied.maximumAugmentations != expected.maximumAugmentations ||
        supplied.receiptSHA256 != expected.receiptSHA256 ||
        supplied.groupClosure.size() != expected.groupClosure.size() ||
        supplied.sourceDonorClosure.size() != expected.sourceDonorClosure.size()) {
        error = "spatial transport receipt or emitted flow arrays differ from deterministic reconstruction";
        return false;
    }
    for (std::size_t i = 0; i < supplied.groupClosure.size(); ++i) {
        const auto& a = supplied.groupClosure[i];
        const auto& b = expected.groupClosure[i];
        if (a.materialIndex != b.materialIndex || a.donorBodyIndex != b.donorBodyIndex ||
            a.anatomicalRegionIndex != b.anatomicalRegionIndex ||
            a.parentMassKg != b.parentMassKg || a.childMassKg != b.childMassKg ||
            a.childMinimumCapacityKg != b.childMinimumCapacityKg ||
            a.childMaximumCapacityKg != b.childMaximumCapacityKg) {
            error = "spatial transport material/donor/region receipt closure differs from deterministic reconstruction";
            return false;
        }
    }
    for (std::size_t i = 0; i < supplied.sourceDonorClosure.size(); ++i) {
        const auto& a = supplied.sourceDonorClosure[i];
        const auto& b = expected.sourceDonorClosure[i];
        if (a.materialIndex != b.materialIndex ||
            a.sourceDonorBodyIndex != b.sourceDonorBodyIndex ||
            a.parentMassKg != b.parentMassKg || a.childMassKg != b.childMassKg) {
            error = "spatial transport source-donor closure differs from deterministic reconstruction";
            return false;
        }
    }
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
