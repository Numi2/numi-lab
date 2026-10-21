#include "numi/matter/partitioned_reduction_basis.hpp"
#include "numi/matter/compiled_world_partitioned_reduction.hpp"
#include "numi/matter/partitioned_reduction_mass_whitening.hpp"
#include "numi/matter/partitioned_reduction_sparse_support.hpp"

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

void verifyMassWhiteningCase(
    numi::matter::PartitionedReductionBasis basis,
    std::span<const double> masses,
    std::uint32_t expectedModesPerRegion);

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

    verifyMassWhiteningCase(basis, masses, 6u);
    auto affineBasis = basis;
    require(numi::matter::compiled_reduction_detail::finalizeAffineModes(
                affineBasis, 4u, 512u, positions, masses, fixed),
            "compiled affine fixture could not construct the Human twelve-mode blocks");
    verifyMassWhiteningCase(std::move(affineBasis), masses, 12u);
}

void verifyMassWhiteningCase(
    numi::matter::PartitionedReductionBasis basis,
    const std::span<const double> masses,
    const std::uint32_t expectedModesPerRegion
) {
    using namespace numi::matter;
    const PartitionedReductionBasis original = basis;
    std::vector<std::uint32_t> regionalMap;
    for (const auto& region : basis.regions) {
        std::uint32_t retained = 0u;
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const auto& descriptor = basis.modes[mode];
            if (descriptor.kind == PartitionedReductionModeKind::anchorTranslation ||
                descriptor.regionLabel != region.sourceLabel) continue;
            regionalMap.push_back(mode);
            ++retained;
        }
        require(retained == expectedModesPerRegion,
                "mass whitening fixture has the wrong regional mode count");
    }
    const auto supportBefore = buildPartitionedReductionSparseSupport(
        std::span<const nm_float4>(basis.values), basis.nodeCount, basis.modeCount,
        std::span<const std::uint32_t>(regionalMap),
        static_cast<std::uint32_t>(basis.regions.size()), expectedModesPerRegion);
    require(supportBefore.valid(), supportBefore.error);

    const auto report = massWhitenPartitionedReductionBasis(basis, masses);
    require(report.valid(), report.error);
    require(report.regions.size() == basis.regions.size(),
            "mass whitening did not return one factor per source region");
    require(report.maximumRegionalIdentityResidual <= 5.0e-4,
            "mass whitening roundoff residual exceeds the production bound");

    // Independent dense oracle covers cross-region overlap and exact-zero
    // supports. Bound compiler contraction differences by the absolute sum
    // of terms, so cancellation does not hide an omitted cross term.
    for (std::uint32_t row = 0u; row < basis.modeCount; ++row) {
        for (std::uint32_t column = 0u; column < basis.modeCount; ++column) {
            double expected = 0.0;
            double absoluteSum = 0.0;
            for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
                const auto& a = basis.values[
                    static_cast<std::size_t>(row) * basis.nodeCount + node];
                const auto& b = basis.values[
                    static_cast<std::size_t>(column) * basis.nodeCount + node];
                const double term = masses[node] *
                    (static_cast<double>(a.x) * b.x +
                     static_cast<double>(a.y) * b.y +
                     static_cast<double>(a.z) * b.z);
                expected += term;
                absoluteSum += std::abs(term);
            }
            const double actual = basis.projectedMass[
                static_cast<std::size_t>(row) * basis.modeCount + column];
            const double roundingBound = 32.0 *
                std::numeric_limits<double>::epsilon() *
                std::max(absoluteSum, std::numeric_limits<double>::min());
            require(std::abs(expected - actual) <= roundingBound,
                    "sparse whitened full Gram differs from dense source-node oracle");
        }
    }

    const auto supportAfter = buildPartitionedReductionSparseSupport(
        std::span<const nm_float4>(basis.values), basis.nodeCount, basis.modeCount,
        std::span<const std::uint32_t>(regionalMap),
        static_cast<std::uint32_t>(basis.regions.size()), expectedModesPerRegion);
    require(supportAfter.valid(), supportAfter.error);
    require(supportAfter.regionSupportRows == supportBefore.regionSupportRows,
            "mass whitening changed the exact anatomical source-row support");

    for (const auto& block : report.regions) {
        const std::size_t count = block.modeIndices.size();
        require(count == expectedModesPerRegion &&
                    block.choleskyLower.size() == count * count,
                "mass whitening factor dimensions do not match its region");
        for (std::size_t row = 0u; row < count; ++row) {
            for (std::size_t column = 0u; column < count; ++column) {
                const double expected = row == column ? 1.0 : 0.0;
                const double actual = basis.projectedMass[
                    static_cast<std::size_t>(block.modeIndices[row]) *
                        basis.modeCount + block.modeIndices[column]];
                require(near(actual, expected, 5.0e-4),
                        "regional B^T M B is not identity after whitening");
            }
        }

        // B_original = B_whitened L^T, so every original column remains in
        // exactly the same regional span.
        for (std::size_t originalColumn = 0u; originalColumn < count;
             ++originalColumn) {
            for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
                std::array<double, 3> reconstructed{};
                for (std::size_t rebasedColumn = 0u; rebasedColumn < count;
                     ++rebasedColumn) {
                    const double factor = block.choleskyLower[
                        originalColumn * count + rebasedColumn];
                    const nm_float4& value = basis.values[
                        static_cast<std::size_t>(block.modeIndices[rebasedColumn]) *
                            basis.nodeCount + node];
                    reconstructed[0] += factor * value.x;
                    reconstructed[1] += factor * value.y;
                    reconstructed[2] += factor * value.z;
                }
                const nm_float4& expected = original.values[
                    static_cast<std::size_t>(block.modeIndices[originalColumn]) *
                        basis.nodeCount + node];
                require(near(reconstructed[0], expected.x, 3.0e-4) &&
                            near(reconstructed[1], expected.y, 3.0e-4) &&
                            near(reconstructed[2], expected.z, 3.0e-4),
                        "mass whitening changed the regional basis span");
            }
        }
    }

    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        if (basis.modes[mode].kind !=
                PartitionedReductionModeKind::anchorTranslation) continue;
        for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
            const nm_float4& before = original.values[
                static_cast<std::size_t>(mode) * basis.nodeCount + node];
            const nm_float4& after = basis.values[
                static_cast<std::size_t>(mode) * basis.nodeCount + node];
            require(before.x == after.x && before.y == after.y &&
                        before.z == after.z && before.w == after.w,
                    "mass whitening changed a sensitive-anchor identity mode");
        }
    }
    for (const std::uint32_t node : basis.anchorNodes) {
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            if (basis.modes[mode].kind ==
                    PartitionedReductionModeKind::anchorTranslation) continue;
            const nm_float4& value = basis.values[
                static_cast<std::size_t>(mode) * basis.nodeCount + node];
            require(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f,
                    "mass whitening introduced regional motion at a sensitive anchor");
        }
    }

    std::vector<Position> force(basis.nodeCount);
    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node)
        force[node] = {0.03 * (node + 1u), -0.017 * (node + 2u),
                       0.011 * (node % 13u)};
    std::vector<double> q(basis.modeCount);
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode)
        q[mode] = 0.007 * static_cast<double>(mode + 1u) - 0.03;
    std::vector<double> reducedForce(basis.modeCount, 0.0);
    std::vector<Position> displacement(basis.nodeCount);
    require(projectPartitionedNodalForce(basis, force, reducedForce) &&
                combinePartitionedPreconditionerModes(basis, q, displacement),
            "mass-whitened basis rejected force or coordinate mapping");
    double reducedWork = 0.0;
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode)
        reducedWork += q[mode] * reducedForce[mode];
    double nodalWork = 0.0;
    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node)
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            nodalWork += displacement[node][axis] * force[node][axis];
    require(near(reducedWork, nodalWork, 2.0e-12),
            "mass-whitened B^T f violates virtual-work equality");

    // A singular regional block must fail transactionally.
    PartitionedReductionBasis singular = original;
    const auto& firstBlock = report.regions.front();
    const std::uint32_t first = firstBlock.modeIndices[0];
    const std::uint32_t duplicate = firstBlock.modeIndices[1];
    for (std::uint32_t node = 0u; node < singular.nodeCount; ++node)
        singular.values[static_cast<std::size_t>(duplicate) *
            singular.nodeCount + node] =
            singular.values[static_cast<std::size_t>(first) *
                singular.nodeCount + node];
    const auto singularValues = singular.values;
    const auto singularMass = singular.projectedMass;
    const auto singularReport =
        massWhitenPartitionedReductionBasis(singular, masses);
    require(!singularReport.valid(),
            "rank-deficient regional mass Gram matrix was admitted");
    require(singular.values.size() == singularValues.size() &&
                std::equal(singular.values.begin(), singular.values.end(),
                           singularValues.begin(),
                           [](const nm_float4& a, const nm_float4& b) {
                               return a.x == b.x && a.y == b.y &&
                                   a.z == b.z && a.w == b.w;
                           }) && singular.projectedMass == singularMass,
            "failed mass whitening partially mutated the original basis");
}

void runHybridIndexMapCases() {
    using numi::matter::validatePartitionedHybridIndexMaps;
    const std::vector<std::uint32_t> nodes{1u, 4u, 8u};
    const std::vector<std::uint32_t> modes{0u, 1u, 2u, 3u};
    require(validatePartitionedHybridIndexMaps(nodes, 3u, 10u, modes,
                4u, 2u, 2u).empty(),
            "valid ascending node map and complete regional mode map were rejected");

    const std::vector<std::uint32_t> duplicateNodes{1u, 4u, 4u};
    require(!validatePartitionedHybridIndexMaps(duplicateNodes, 3u, 10u,
                modes, 4u, 2u, 2u).empty(),
            "duplicate source nodes were admitted into a race-prone lift map");
    const std::vector<std::uint32_t> unsortedNodes{1u, 8u, 4u};
    require(!validatePartitionedHybridIndexMaps(unsortedNodes, 3u, 10u,
                modes, 4u, 2u, 2u).empty(),
            "unsorted source rows were admitted");
    const std::vector<std::uint32_t> outOfRangeNodes{1u, 4u, 10u};
    require(!validatePartitionedHybridIndexMaps(outOfRangeNodes, 3u, 10u,
                modes, 4u, 2u, 2u).empty(),
            "out-of-range source rows were admitted");

    const std::vector<std::uint32_t> incompleteModes{0u, 1u, 2u};
    require(!validatePartitionedHybridIndexMaps(nodes, 3u, 10u,
                incompleteModes, 4u, 2u, 2u).empty(),
            "incomplete regional map left a hybrid coordinate without a region");
    const std::vector<std::uint32_t> duplicateModes{0u, 1u, 2u, 2u};
    require(!validatePartitionedHybridIndexMaps(nodes, 3u, 10u,
                duplicateModes, 4u, 2u, 2u).empty(),
            "duplicate regional modes were admitted");
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
    runHybridIndexMapCases();
    runRejectCase();
    std::cout << "partitioned reduction basis: source preservation, partition unity, "
                 "rigid motion, projected mass, B^T f, virtual work, hybrid index-map validation, "
                 "regional mass whitening, span, anchor preservation, and reject cases passed\n";
    return 0;
}
