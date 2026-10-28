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
#include <map>
#include <queue>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
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
};
struct EdgeRef {
    std::uint32_t parent = 0u;
    std::uint32_t child = 0u;
    std::uint32_t from = 0u;
    std::uint32_t edge = 0u;
    double distanceM = 0.0;
};

class MinCostFlow {
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
             std::string& error) {
        const auto fail = [&](const char* message) { error = message; return false; };
        const std::size_t n = graph_.size();
        std::vector<double> potential(n, 0.0), distance(n);
        std::vector<std::uint32_t> previousNode(n), previousEdge(n);
        augmentationCount = 0u;
        sent = 0.0;
        while (required - sent > kFlowEpsilonKg) {
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
                if (knownDistance > distance[node] + 1.0e-15) continue;
                for (std::uint32_t edgeIndex = 0; edgeIndex < graph_[node].size(); ++edgeIndex) {
                    const auto& edge = graph_[node][edgeIndex];
                    if (edge.capacity <= kFlowEpsilonKg) continue;
                    double reduced = edge.cost + potential[node] - potential[edge.to];
                    if (reduced < 0.0 && reduced > -1.0e-12) reduced = 0.0;
                    if (reduced < 0.0)
                        return fail("spatial transport min-cost reduced edge became negative");
                    const double candidate = knownDistance + reduced;
                    if (candidate < distance[edge.to] - 1.0e-15) {
                        distance[edge.to] = candidate;
                        previousNode[edge.to] = node;
                        previousEdge[edge.to] = edgeIndex;
                        queue.push({candidate, edge.to});
                    }
                }
            }
            if (previousNode[sink] == kInvalid) break;
            for (std::size_t node = 0; node < n; ++node)
                if (std::isfinite(distance[node])) potential[node] += distance[node];
            double amount = required - sent;
            for (std::uint32_t node = sink; node != source; node = previousNode[node]) {
                if (previousNode[node] == kInvalid)
                    return fail("spatial transport residual path is incomplete");
                amount = std::min(amount,
                    graph_[previousNode[node]][previousEdge[node]].capacity);
            }
            if (!std::isfinite(amount) || amount <= kFlowEpsilonKg)
                return fail("spatial transport residual path has no positive mass capacity");
            for (std::uint32_t node = sink; node != source; node = previousNode[node]) {
                auto& edge = graph_[previousNode[node]][previousEdge[node]];
                auto& reverse = graph_[node][edge.reverse];
                edge.capacity -= amount;
                reverse.capacity += amount;
            }
            sent += amount;
            ++augmentationCount;
        }
        error.clear();
        return true;
    }

    [[nodiscard]] double flow(const EdgeRef& reference) const {
        const auto& edge = graph_[reference.from][reference.edge];
        return edge.initialCapacity - edge.capacity;
    }
private:
    std::vector<std::vector<FlowEdge>> graph_;
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
// centroid distance among feasible couplings; infeasible capacities, region
// coverage, distance, edge-count or augmentation limits fail without expanding
// the candidate set or renormalizing density.
[[nodiscard]] inline bool assembleConservativeSpatialMassTransport(
        const ConservativeSpatialMassTransportInput& in,
        ConservativeSpatialMassTransportResult& out,
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
    for (const auto& [group, parentRows] : parentsByGroup) {
        const auto& childRows = childrenByGroup[group];
        long double supplyLong = 0.0L, minimumLong = 0.0L, maximumLong = 0.0L;
        for (const auto p : parentRows) supplyLong += in.parentMassKg[p];
        for (const auto c : childRows) {
            minimumLong += static_cast<long double>(in.childReferenceVolumeM3[c]) *
                           in.materialDensityLowerKgM3[in.childMaterialIndices[c]];
            maximumLong += static_cast<long double>(in.childReferenceVolumeM3[c]) *
                           in.materialDensityUpperKgM3[in.childMaterialIndices[c]];
        }
        const double supply = static_cast<double>(supplyLong);
        const double minimum = static_cast<double>(minimumLong);
        const double maximum = static_cast<double>(maximumLong);
        if (!std::isfinite(supply) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
            supply < minimum || supply > maximum)
            return fail("spatial transport parent supply is outside child volume times actual material density bounds for material/donor/region group");

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
            if (row.candidates.size() > in.maximumAllowedEdges -
                    std::min(allCandidateEdges, in.maximumAllowedEdges))
                return fail("spatial transport allowed-edge count exceeds its pinned budget; no candidate edges were truncated");
            allCandidateEdges += row.candidates.size();
            groupCandidateEdges += row.candidates.size();
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
        for (std::uint32_t i = 0; i < childCount; ++i) localChild[childRows[i]] = i;
        for (std::uint32_t i = 0; i < parentCount; ++i)
            network.addEdge(superSource, parentBase + i, in.parentMassKg[parentRows[i]], 0.0);
        std::vector<EdgeRef> references;
        references.reserve(groupCandidateEdges);
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
        long double childMinimumLong = 0.0L;
        for (std::uint32_t i = 0; i < childCount; ++i) {
            const auto c = childRows[i];
            const double childMinimum = in.childReferenceVolumeM3[c] *
                in.materialDensityLowerKgM3[in.childMaterialIndices[c]];
            const double childMaximum = in.childReferenceVolumeM3[c] *
                in.materialDensityUpperKgM3[in.childMaterialIndices[c]];
            if (!std::isfinite(childMinimum) || !std::isfinite(childMaximum) ||
                childMinimum <= 0.0 || childMaximum < childMinimum)
                return fail("spatial transport child density capacity is invalid");
            childMinimumLong += childMinimum;
            network.addEdge(childBase + i, superSink, childMinimum, 0.0);
            network.addEdge(childBase + i, sink, childMaximum - childMinimum, 0.0);
        }
        const double residualCapacity = supply - static_cast<double>(childMinimumLong);
        if (residualCapacity < 0.0)
            return fail("spatial transport mandatory child density floors exceed parent mass supply");
        network.addEdge(sink, superSink, residualCapacity, 0.0);

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
            std::max(in.absoluteMassToleranceKg, in.relativeClosureTolerance * supply))
            return fail("spatial transport is infeasible under same-material/donor/region and distance bounds");

        double groupOutputMass = 0.0;
        for (const auto& reference : references) {
            const double mass = network.flow(reference);
            if (mass <= kFlowEpsilonKg) continue;
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
    std::sort(flows.begin(), flows.end(), [](const auto& a, const auto& b) {
        if (a.child != b.child) return a.child < b.child;
        return a.parent < b.parent;
    });
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
            density > in.materialDensityUpperKgM3[material] + densityTolerance)
            return fail("spatial transport child cell mass implies density outside actual material bounds at child "+
                std::to_string(c)+" (density="+std::to_string(density)+", bounds="+
                std::to_string(in.materialDensityLowerKgM3[material])+":"+
                std::to_string(in.materialDensityUpperKgM3[material])+")");
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
