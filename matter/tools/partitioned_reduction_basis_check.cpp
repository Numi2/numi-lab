#include "numi/matter/partitioned_reduction_basis.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {
using Position = std::array<double, 3>;
using Tet = std::array<std::uint32_t, 4>;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "partitioned reduction basis check failed: " << message << '\n';
    std::exit(1);
}

void require(const bool condition, const std::string& message) {
    if (!condition) fail(message);
}

bool near(const double a, const double b, const double tolerance = 2.0e-5) {
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

std::uint32_t nodeIndex(const std::uint32_t x, const std::uint32_t y,
                        const std::uint32_t z) {
    constexpr std::uint32_t nx = 5u;
    constexpr std::uint32_t ny = 4u;
    return x + nx * (y + ny * z);
}

void addTetraGrid(std::vector<Tet>& tetrahedra,
                  std::vector<std::uint32_t>& labels) {
    for (std::uint32_t z = 0u; z < 3u; ++z) {
        for (std::uint32_t y = 0u; y < 3u; ++y) {
            for (std::uint32_t x = 0u; x < 4u; ++x) {
                const auto v000 = nodeIndex(x, y, z);
                const auto v100 = nodeIndex(x + 1u, y, z);
                const auto v010 = nodeIndex(x, y + 1u, z);
                const auto v110 = nodeIndex(x + 1u, y + 1u, z);
                const auto v001 = nodeIndex(x, y, z + 1u);
                const auto v101 = nodeIndex(x + 1u, y, z + 1u);
                const auto v011 = nodeIndex(x, y + 1u, z + 1u);
                const auto v111 = nodeIndex(x + 1u, y + 1u, z + 1u);
                const std::array<Tet, 6> cellTets{{
                    {v000, v100, v110, v111},
                    {v000, v110, v010, v111},
                    {v000, v010, v011, v111},
                    {v000, v011, v001, v111},
                    {v000, v001, v101, v111},
                    {v000, v101, v100, v111},
                }};
                const std::uint32_t sourceRegion = x < 2u ? 10u : 20u;
                for (const Tet& tet : cellTets) {
                    tetrahedra.push_back(tet);
                    labels.push_back(sourceRegion);
                }
            }
        }
    }
}

Position cross(const Position& a, const Position& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

void runValidCase() {
    constexpr std::uint32_t nx = 5u;
    constexpr std::uint32_t ny = 4u;
    constexpr std::uint32_t nz = 4u;
    constexpr std::size_t nodeCount = nx * ny * nz;
    std::vector<Position> positions(nodeCount);
    std::vector<double> masses(nodeCount);
    std::vector<std::uint8_t> fixed(nodeCount, 0u);
    for (std::uint32_t z = 0u; z < nz; ++z) {
        for (std::uint32_t y = 0u; y < ny; ++y) {
            for (std::uint32_t x = 0u; x < nx; ++x) {
                const std::size_t node = nodeIndex(x, y, z);
                positions[node] = {0.5 * x, 0.75 * y, 1.25 * z};
                masses[node] = 1.0 + 0.001 * static_cast<double>(node);
            }
        }
    }
    std::vector<Tet> tetrahedra;
    std::vector<std::uint32_t> labels;
    addTetraGrid(tetrahedra, labels);
    const std::vector<std::uint32_t> anchors{
        nodeIndex(2u, 1u, 1u), nodeIndex(0u, 3u, 3u), nodeIndex(4u, 0u, 0u)};
    const std::uint32_t fixedNode = nodeIndex(4u, 3u, 3u);
    fixed[fixedNode] = 1u;
    const auto sourcePositions = positions;
    const auto sourceTetrahedra = tetrahedra;
    const auto sourceLabels = labels;
    const auto sourceAnchors = anchors;

    const numi::matter::PartitionedReductionBasisInput input{
        positions, masses, fixed, tetrahedra, labels, anchors};
    const numi::matter::PartitionedReductionBasis basis =
        numi::matter::buildPartitionedReductionBasis(input);
    require(basis.valid(), basis.error);
    require(basis.role ==
                numi::matter::PartitionedReductionBasisRole::preconditionerOnly,
            "basis role is not explicitly preconditioner-only");
    require(!basis.runtimeCoordinateMapAvailable,
            "builder falsely advertises runtime coordinate projection");
    require(basis.nodeCount == nodeCount && basis.anchorNodes == anchors,
            "source node count or anchor identity was not retained");
    require(basis.modeCount == 21u && basis.regions.size() == 2u,
            "unexpected regional-plus-anchor basis size");
    require(near(basis.regions[0].volume, 8.4375, 1.0e-12) &&
                near(basis.regions[1].volume, 8.4375, 1.0e-12),
            "regional source volumes were not preserved in basis construction");
    require(basis.modeCount * basis.minimumCoordinateReductionFactor <=
                basis.freeCoordinateCount,
            "accepted basis misses the requested 4:1 coordinate reduction");
    require(basis.values.size() == basis.modeCount * nodeCount,
            "basis does not use RegionalGalerkinBasis-compatible mode-major storage");
    require(positions == sourcePositions && tetrahedra == sourceTetrahedra &&
                labels == sourceLabels && anchors == sourceAnchors,
            "builder modified pinned source mesh, labels, or anchors");

    require(basis.partitionCount == basis.regions.size() + anchors.size(),
            "partition count does not include one local partition per anchor");
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        double sum = 0.0;
        for (std::uint32_t partition = 0u;
             partition < basis.partitionCount; ++partition) {
            const double weight = basis.partitionWeights[
                node * basis.partitionCount + partition];
            require(weight >= -1.0e-14 && std::isfinite(weight),
                    "partition weight is negative or non-finite");
            sum += weight;
        }
        require(near(sum, fixed[node] == 0u ? 1.0 : 0.0, 1.0e-12),
                "partition of unity failed on a source node");
    }
    for (std::size_t anchor = 0u; anchor < anchors.size(); ++anchor) {
        const std::uint32_t node = anchors[anchor];
        require(near(basis.partitionWeights[
                         node * basis.partitionCount + basis.regions.size() + anchor],
                     1.0, 1.0e-12),
                "anchor node lacks its one-hot local partition");
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            bool found = false;
            for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
                const auto& descriptor = basis.modes[mode];
                if (descriptor.kind !=
                        numi::matter::PartitionedReductionModeKind::anchorTranslation ||
                    descriptor.anchorNode != node || descriptor.component != axis)
                    continue;
                found = true;
                for (std::uint32_t other = 0u; other < basis.nodeCount; ++other) {
                    const auto& value = basis.values[
                        static_cast<std::size_t>(mode) * basis.nodeCount + other];
                    const double expected = other == node ? 1.0 : 0.0;
                    const double actual = axis == 0u ? value.x :
                        axis == 1u ? value.y : value.z;
                    require(near(actual, expected, 1.0e-7),
                            "anchor mode is not identity-local at the source anchor");
                }
            }
            require(found, "an authored free anchor lost a local Cartesian mode");
        }
    }

    const Position translation{0.3, -0.7, 1.1};
    const Position angularVelocity{-0.2, 0.4, 0.6};
    std::vector<double> q(basis.modeCount, 0.0);
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        const auto& descriptor = basis.modes[mode];
        if (descriptor.kind ==
                numi::matter::PartitionedReductionModeKind::regionalTranslation) {
            const auto found = std::find_if(
                basis.regions.begin(), basis.regions.end(),
                [&](const auto& region) {
                    return region.sourceLabel == descriptor.regionLabel;
                });
            require(found != basis.regions.end(), "regional translation lost its source label");
            const Position centerRotation = cross(angularVelocity, found->center);
            q[mode] = translation[descriptor.component] +
                centerRotation[descriptor.component];
        } else if (descriptor.kind ==
                   numi::matter::PartitionedReductionModeKind::regionalRotation) {
            q[mode] = descriptor.scale * angularVelocity[descriptor.component];
        } else {
            const Position anchorRotation = cross(
                angularVelocity, positions[descriptor.anchorNode]);
            q[mode] = translation[descriptor.component] +
                anchorRotation[descriptor.component];
        }
    }
    std::vector<Position> displacement(nodeCount);
    require(numi::matter::combinePartitionedPreconditionerModes(basis, q, displacement),
            "basis combination rejected dimensionally correct input");
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        if (fixed[node] != 0u) {
            require(near(displacement[node][0], 0.0) &&
                        near(displacement[node][1], 0.0) &&
                        near(displacement[node][2], 0.0),
                    "fixed source node moved under basis prolongation");
            continue;
        }
        const Position rotation = cross(angularVelocity, positions[node]);
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            require(near(displacement[node][axis],
                         translation[axis] + rotation[axis]),
                    "partitioned basis does not reproduce an exact global rigid motion");
    }

    const std::size_t modeCount = basis.modeCount;
    for (std::size_t row = 0u; row < modeCount; ++row) {
        for (std::size_t column = 0u; column < modeCount; ++column) {
            double expected = 0.0;
            for (std::size_t node = 0u; node < nodeCount; ++node) {
                const auto& a = basis.values[row * nodeCount + node];
                const auto& b = basis.values[column * nodeCount + node];
                expected += masses[node] * (static_cast<double>(a.x) * b.x +
                    static_cast<double>(a.y) * b.y + static_cast<double>(a.z) * b.z);
            }
            require(near(basis.projectedMass[row * modeCount + column], expected,
                         2.0e-12),
                    "projected mass differs from direct B^T M B assembly");
        }
    }

    std::vector<Position> force(nodeCount);
    for (std::size_t node = 0u; node < nodeCount; ++node)
        force[node] = {0.13 * (node + 1u), -0.07 * (node + 2u),
                       0.11 * (node % 9u)};
    std::vector<double> reducedForce(modeCount, 0.0);
    require(numi::matter::projectPartitionedNodalForce(
                basis, force, reducedForce),
            "B^T f rejected dimensionally correct input");
    for (std::size_t mode = 0u; mode < modeCount; ++mode) {
        double expected = 0.0;
        for (std::size_t node = 0u; node < nodeCount; ++node) {
            const auto& b = basis.values[mode * nodeCount + node];
            expected += b.x * force[node][0] + b.y * force[node][1] +
                b.z * force[node][2];
        }
        require(near(reducedForce[mode], expected, 2.0e-12),
                "force projection differs from direct B^T f assembly");
    }
    double reducedWork = 0.0;
    for (std::size_t mode = 0u; mode < modeCount; ++mode)
        reducedWork += q[mode] * reducedForce[mode];
    double nodalWork = 0.0;
    for (std::size_t node = 0u; node < nodeCount; ++node)
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            nodalWork += displacement[node][axis] * force[node][axis];
    require(near(reducedWork, nodalWork, 2.0e-12),
            "B^T f fails virtual-work equality");
}

void runRejectCase() {
    std::vector<Position> positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                    {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    std::vector<double> masses(4u, 1.0);
    std::vector<std::uint8_t> fixed(4u, 0u);
    std::vector<Tet> tetrahedra{{0u, 1u, 2u, 3u}};
    std::vector<std::uint32_t> labels{7u};
    std::vector<std::uint32_t> duplicateAnchors{0u, 0u};
    const numi::matter::PartitionedReductionBasisInput input{
        positions, masses, fixed, tetrahedra, labels, duplicateAnchors};
    const auto basis = numi::matter::buildPartitionedReductionBasis(input);
    require(!basis.valid() && basis.error.find("duplicated") != std::string::npos,
            "invalid duplicate anchor list was not rejected");
}

} // namespace

int main() {
    runValidCase();
    runRejectCase();
    std::cout << "partitioned reduction basis: source preservation, partition unity, "
                 "rigid motion, projected mass, B^T f, virtual work, and reject case passed\n";
    return 0;
}
