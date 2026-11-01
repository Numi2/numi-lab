#define NUMI_MINCOSTFLOW_COMPACTION_TEST 1
#include "NumiHumanConservativeSpatialMassTransport.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using namespace metalrobo::human::coarse_tissue::spatial_mass_transport_detail;

namespace metalrobo::human::coarse_tissue::spatial_mass_transport_detail {
struct MinCostFlowCompactionTestAccess {
    static const std::vector<std::vector<FlowEdge>>& graph(const MinCostFlow& net) {
        return net.graph_;
    }

    static std::vector<std::vector<FlowEdge>>& graph(MinCostFlow& net) {
        return net.graph_;
    }

    static bool sameRawSurvivor(const FlowEdge& before, const FlowEdge& after,
                                const std::uint32_t expectedReverse) {
        return before.to == after.to &&
            after.reverse == expectedReverse &&
            std::memcmp(&before.capacity, &after.capacity, sizeof(double)) == 0 &&
            std::memcmp(&before.initialCapacity, &after.initialCapacity, sizeof(double)) == 0 &&
            std::memcmp(&before.cost, &after.cost, sizeof(double)) == 0 &&
            std::memcmp(&before.capacityCorrection, &after.capacityCorrection, sizeof(double)) == 0;
    }

    static bool reciprocal(const MinCostFlow& net) {
        const auto& rows = graph(net);
        for (std::size_t from = 0; from < rows.size(); ++from) {
            for (std::size_t index = 0; index < rows[from].size(); ++index) {
                const auto& edge = rows[from][index];
                if (edge.to >= rows.size() || edge.reverse >= rows[edge.to].size()) return false;
                const auto& reverse = rows[edge.to][edge.reverse];
                if (reverse.to != from || reverse.reverse != index) return false;
            }
        }
        return true;
    }
};
}

static bool require(const bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << "\n";
    return false;
}

static EdgeRef ref(const std::uint32_t from, const std::uint32_t edge,
                   const std::uint32_t parent, const std::uint32_t child) {
    return EdgeRef{parent, child, from, edge, 0.0};
}

static bool exactZeroAndTinyResidualControl() {
    MinCostFlow net(5u);
    const auto zeroA = net.addEdge(0u, 1u, 1.25, 0.125);
    const auto tiny = net.addEdge(0u, 1u, 2.5, 0.25);
    const auto zeroB = net.addEdge(0u, 1u, 3.75, 0.375);
    const auto mandatory = net.addEdge(1u, 2u, 4.25, 0.5);
    const auto zeroLonelyRowPair = net.addEdge(3u, 4u, 0.75, 0.125);

    // Model a represented positive tiny residual whose high part is zero.
    // The pair remains reciprocal and its raw fields must survive unchanged.
    auto& rows = MinCostFlowCompactionTestAccess::graph(net);
    rows[0u][tiny].capacityCorrection = 1e-300;
    rows[1u][rows[0u][tiny].reverse].capacity = 0.0;
    rows[1u][rows[0u][tiny].reverse].capacityCorrection = -1e-300;

    const auto before = rows;
    const std::vector<EdgeRef> candidates{
        ref(0u, zeroA, 10u, 20u),
        ref(0u, zeroA, 10u, 20u), // duplicate handle is idempotent
        ref(0u, tiny, 11u, 21u),
        ref(0u, zeroB, 12u, 22u),
        ref(3u, zeroLonelyRowPair, 13u, 23u)
    };
    EdgeIndexCompactionMap map;
    std::vector<std::uint8_t> retained;
    CandidateEdgeCompactionStats stats;
    std::string error;
    if (!require(net.compactUnusedCandidateForwardPairs(
            candidates, map, retained, error, &stats), error.c_str())) return false;

    if (!require(retained == std::vector<std::uint8_t>({0u, 0u, 1u, 0u, 0u}),
                 "candidate retained bits include duplicate/compensated pair semantics")) return false;
    if (!require(stats.uniqueCandidatePairs == 3u && stats.removedPairs == 3u,
                 "three unique exact-zero candidate pairs removed")) return false;
    if (!require(stats.directedEdgesBefore == 10u && stats.directedEdgesAfter == 4u,
                 "only the selected two reciprocal pairs were removed")) return false;
    if (!require(map.wasRemoved(0u, zeroA) && map.wasRemoved(0u, zeroB) &&
                 map.wasRemoved(3u, zeroLonelyRowPair),
                 "exact-zero candidate indexes map to removed")) return false;
    if (!require(stats.rowsReleased == 2u &&
                 stats.adjacencyCapacityAfter < stats.adjacencyCapacityBefore,
                 "empty endpoint adjacency rows release their allocated capacity")) return false;

    std::uint32_t mapped = 99u;
    if (!require(map.remapIndex(0u, tiny, mapped) && mapped == 0u,
                 "retained candidate index shifts to compact row index")) return false;
    if (!require(map.remapIndex(1u, mandatory, mapped) && mapped == 1u,
                 "noncandidate zero-flow mandatory arc remains and is remapped")) return false;
    if (!require(!map.remapIndex(0u, 100u, mapped),
                 "out-of-range old adjacency index rejects")) return false;

    const auto& after = MinCostFlowCompactionTestAccess::graph(net);
    for (std::size_t from = 0; from < before.size(); ++from) {
        for (std::size_t old = 0; old < before[from].size(); ++old) {
            std::uint32_t newIndex = 0u;
            if (!map.remapIndex(static_cast<std::uint32_t>(from),
                                static_cast<std::uint32_t>(old), newIndex)) continue;
            const auto& prior = before[from][old];
            std::uint32_t expectedReverse = 0u;
            if (!require(map.remapIndex(prior.to, prior.reverse, expectedReverse),
                         "retained reverse endpoint has a map")) return false;
            if (!require(MinCostFlowCompactionTestAccess::sameRawSurvivor(
                    prior, after[from][newIndex], expectedReverse),
                         "survivor raw capacities, correction, initial capacity, and cost stay bit-exact")) return false;
        }
    }
    if (!require(MinCostFlowCompactionTestAccess::reciprocal(net),
                 "all compacted reverse indexes remain reciprocal")) return false;
    if (!require(net.hasResidualCapacity(0u, 0u),
                 "compensated tiny residual pair remains traversable")) return false;

    EdgeRef tinyRef = candidates[2u];
    if (!require(map.remapReference(tinyRef) && tinyRef.edge == 0u &&
                 tinyRef.parent == 11u && tinyRef.child == 21u,
                 "reference remap changes only adjacency edge index")) return false;
    EdgeRef removedRef = candidates[0u];
    if (!require(!map.remapReference(removedRef),
                 "removed candidate reference is rejected for caller filtering")) return false;
    return true;
}

struct ParityHandles {
    std::uint32_t sourceP1 = 0u;
    std::uint32_t sourceP2 = 0u;
    std::uint32_t candidateUnused = 0u;
    std::uint32_t candidateUsed = 0u;
    std::uint32_t p2ToC1 = 0u;
    std::uint32_t p2ToC2 = 0u;
    std::uint32_t c1ToSink = 0u;
    std::uint32_t c2ToSink = 0u;
    std::vector<EdgeRef> candidates;
};

static ParityHandles makeParityNetwork(MinCostFlow& net) {
    ParityHandles h;
    h.sourceP1 = net.addEdge(0u, 1u, 1.0, 0.0);
    h.sourceP2 = net.addEdge(0u, 2u, 1.0, 0.0);
    h.candidateUnused = net.addEdge(1u, 3u, 1.0, 2.0);
    h.candidateUsed = net.addEdge(1u, 3u, 1.0, 0.0);
    h.p2ToC1 = net.addEdge(2u, 3u, 1.0, 4.0);
    h.p2ToC2 = net.addEdge(2u, 4u, 1.0, 0.0);
    h.c1ToSink = net.addEdge(3u, 5u, 1.0, 0.0);
    h.c2ToSink = net.addEdge(4u, 5u, 1.0, 0.0);
    h.candidates = {
        ref(1u, h.candidateUnused, 100u, 300u),
        ref(1u, h.candidateUsed, 100u, 300u)
    };
    return h;
}

static bool solveOneMinCostUnit(MinCostFlow& net, const ParityHandles& h,
                                double& sent) {
    std::size_t augmentations = 0u;
    std::string error;
    const auto ok = net.run(0u, 5u, 1.0, 16u, augmentations, sent, error);
    return require(ok, error.c_str()) &&
           require(std::abs(sent - 1.0) < 1e-14, "initial min-cost unit sent") &&
           require(net.flow(ref(1u, h.candidateUsed, 100u, 300u)) > 0.999999,
                   "fixture routes initial flow through the cheaper parallel candidate") &&
           require(net.flow(ref(1u, h.candidateUnused, 100u, 300u)) == 0.0,
                   "candidate selected for compaction has exact zero flow");
}

static bool continuationParityControl() {
    MinCostFlow baseline(6u), compacted(6u);
    auto hb = makeParityNetwork(baseline);
    auto hc = makeParityNetwork(compacted);
    double initialBaseline = 0.0, initialCompacted = 0.0;
    if (!solveOneMinCostUnit(baseline, hb, initialBaseline) ||
        !solveOneMinCostUnit(compacted, hc, initialCompacted)) return false;

    EdgeIndexCompactionMap map;
    std::vector<std::uint8_t> retained;
    std::string error;
    if (!require(compacted.compactUnusedCandidateForwardPairs(
            hc.candidates, map, retained, error), error.c_str())) return false;
    if (!require(retained == std::vector<std::uint8_t>({0u, 1u}),
                 "zero-flow parallel candidate removed while used parallel candidate retained")) return false;

    auto remap = [&](const std::uint32_t from, std::uint32_t& index) {
        std::uint32_t mapped = 0u;
        if (!map.remapHandle(from, index, mapped)) return false;
        index = mapped;
        return true;
    };
    // These exercise caller-owned source, minimum, maximum, and candidate
    // reference handles after a row loses its first parallel edge.
    if (!require(remap(0u, hc.sourceP1) && remap(0u, hc.sourceP2),
                 "source-arc handle remap")) return false;
    if (!require(remap(2u, hc.p2ToC1) && remap(2u, hc.p2ToC2),
                 "retained candidate-row handle remap")) return false;
    if (!require(remap(3u, hc.c1ToSink) && remap(4u, hc.c2ToSink),
                 "child min/max arc handle remap")) return false;
    std::vector<EdgeRef> retainedReferences;
    for (std::size_t i = 0; i < hc.candidates.size(); ++i) {
        if (!retained[i]) continue;
        auto reference = hc.candidates[i];
        if (!require(map.remapReference(reference),
                     "retained EdgeRef remaps")) return false;
        retainedReferences.push_back(reference);
    }
    if (!require(retainedReferences.size() == 1u &&
                 retainedReferences.front().edge == 1u,
                 "caller filters removed reference and remaps retained reference")) return false;

    std::size_t augBaseline = 0u, augCompacted = 0u;
    double sentBaseline = 0.0, sentCompacted = 0.0;
    if (!require(baseline.runFeasible(0u, 5u, 16u, augBaseline,
                                     sentBaseline, error), error.c_str())) return false;
    if (!require(compacted.runFeasible(0u, 5u, 16u, augCompacted,
                                      sentCompacted, error), error.c_str())) return false;
    if (!require(initialBaseline == initialCompacted &&
                 sentBaseline == sentCompacted && sentCompacted == 2.0,
                 "max-flow continuation sent mass is identical")) return false;
    if (!require(std::abs(baseline.flow(ref(0u, hb.sourceP1, 0u, 1u)) -
                          compacted.flow(ref(0u, hc.sourceP1, 0u, 1u))) < 1e-14 &&
                 std::abs(baseline.flow(ref(0u, hb.sourceP2, 0u, 2u)) -
                          compacted.flow(ref(0u, hc.sourceP2, 0u, 2u))) < 1e-14,
                 "source flow handles retain identical continuation")) return false;
    if (!require(std::abs(baseline.flow(ref(1u, hb.candidateUsed, 1u, 3u)) -
                          compacted.flow(retainedReferences.front())) < 1e-14,
                 "used candidate flow is unchanged after index remap")) return false;
    if (!require(std::abs(baseline.flow(ref(3u, hb.c1ToSink, 3u, 5u)) -
                          compacted.flow(ref(3u, hc.c1ToSink, 3u, 5u))) < 1e-14 &&
                 std::abs(baseline.flow(ref(4u, hb.c2ToSink, 4u, 5u)) -
                          compacted.flow(ref(4u, hc.c2ToSink, 4u, 5u))) < 1e-14,
                 "minimum/maximum arc flow handles remain equivalent")) return false;
    return MinCostFlowCompactionTestAccess::reciprocal(compacted);
}

int main() {
    if (!exactZeroAndTinyResidualControl()) return 1;
    if (!continuationParityControl()) return 1;
    std::cout << "PASS exact-zero/compensated-tiny/duplicate-parallel/remap/maxflow-continuation controls\n";
    return 0;
}
