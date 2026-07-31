#pragma once

#include <cstdint>

namespace metalrobo {

// Fixed per-environment capacities for the device-resident contact graph.
// Zero fields request the compiler's topology-derived envelope. A task may
// author smaller operational capacities; overflow remains transactional and
// reports the exact required count instead of truncating physics.
struct MetalWorldCapacityProfile {
    std::uint32_t candidatePairs = 0u;
    std::uint32_t rawContacts = 0u;
    std::uint32_t manifolds = 0u;
    std::uint32_t constraintBlocks = 0u;
    std::uint32_t constraintRows = 0u;
    std::uint32_t islands = 0u;
    std::uint32_t hardConvexPairs = 0u;
    std::uint32_t meshTriangleCandidates = 0u;
    std::uint32_t solverTiles = 0u;
    std::uint32_t spillRows = 0u;
    std::uint32_t ccdCandidates = 0u;
    std::uint32_t ccdEvents = 0u;
    std::uint32_t endpointRuntimeRecords = 0u;
    std::uint32_t articulationPointQueries = 0u;
    std::uint32_t rodCandidatePairs = 0u;
    std::uint32_t rodRawContacts = 0u;
    std::uint32_t rodManifolds = 0u;
    std::uint32_t rodCCDEvents = 0u;
    std::uint32_t qualityGeneralizedVelocities = 0u;
    std::uint32_t qualityRows = 0u;
    std::uint32_t qualityKrylovVectors = 0u;
    std::uint32_t qualityDirectTiles = 0u;
    std::uint32_t dynamicNodes = 0u;
    std::uint32_t islandNodeReferences = 0u;
    std::uint32_t islandConstraintReferences = 0u;
    std::uint32_t rodFactorBlocks = 0u;
    std::uint32_t operatorVelocityElements = 0u;
};

} // namespace metalrobo
