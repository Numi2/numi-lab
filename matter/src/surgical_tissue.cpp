#include "numi/matter/surgical_tissue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::matter {
namespace {

constexpr double kMinimumDimension = 1.0e-9;

[[nodiscard]] bool finitePositive(const JejunalScalar& value) {
    return value.value > 0.0 && std::isfinite(value.value);
}

[[nodiscard]] double tetrahedronVolume(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b,
    const std::array<double, 3>& c,
    const std::array<double, 3>& d
) {
    const std::array<double, 3> ab{
        b[0] - a[0], b[1] - a[1], b[2] - a[2],
    };
    const std::array<double, 3> ac{
        c[0] - a[0], c[1] - a[1], c[2] - a[2],
    };
    const std::array<double, 3> ad{
        d[0] - a[0], d[1] - a[1], d[2] - a[2],
    };
    const std::array<double, 3> cross{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    return (
        cross[0] * ad[0] +
        cross[1] * ad[1] +
        cross[2] * ad[2]
    ) / 6.0;
}

[[nodiscard]] bool setParameter(
    MaterialProgram& material,
    const std::string_view name,
    const double value
) {
    const auto found = std::find_if(
        material.parameters.begin(),
        material.parameters.end(),
        [&](const Parameter& parameter) {
            return parameter.name == name;
        }
    );
    if (found == material.parameters.end() ||
        !std::isfinite(value) ||
        value < found->lower ||
        value > found->upper) {
        return false;
    }
    found->defaultValue = value;
    return true;
}

} // namespace

std::string_view jejunalValueBasisName(
    const JejunalValueBasis basis
) noexcept {
    switch (basis) {
    case JejunalValueBasis::belliniPorcineBiaxialStudy:
        return "sourced:Bellini-porcine-jejunum-biaxial";
    case JejunalValueBasis::derivedGeometry:
        return "derived-from-authored-geometry";
    case JejunalValueBasis::researchDefault:
        return "explicit-research-default";
    }
    return "invalid";
}

std::string_view jejunalValueSourceReference(
    const JejunalValueBasis basis
) noexcept {
    switch (basis) {
    case JejunalValueBasis::belliniPorcineBiaxialStudy:
        return "https://doi.org/10.1016/j.jmbbm.2011.05.030";
    case JejunalValueBasis::derivedGeometry:
        return "structured tetrahedral closure-coupon construction";
    case JejunalValueBasis::researchDefault:
        return "MetalRobo research default; specimen calibration required";
    }
    return "invalid";
}

bool configurePorcineJejunumFungMaterial(
    MaterialProgram& material,
    const PorcineJejunumFungSpec& spec,
    std::string* error
) {
    MaterialProgram staged = material;
    const bool configured =
        staged.name == "porcine_jejunum_fung" &&
        setParameter(staged, "density", spec.densityKgPerM3.value) &&
        setParameter(staged, "fung_c", spec.fungCpa.value) &&
        setParameter(
            staged,
            "a_longitudinal",
            spec.longitudinalCoefficient.value
        ) &&
        setParameter(
            staged,
            "a_circumferential",
            spec.circumferentialCoefficient.value
        ) &&
        setParameter(
            staged,
            "a_coupling",
            spec.couplingCoefficient.value
        ) &&
        setParameter(
            staged,
            "ground_shear",
            spec.groundShearPa.value
        ) &&
        setParameter(
            staged,
            "numerical_viscosity",
            spec.numericalViscosityPaS.value
        ) &&
        finitePositive(spec.bulkModulusPa);
    if (!configured) {
        if (error != nullptr) {
            *error =
                "material is not the expected porcine jejunum Fung "
                "program or a calibrated value is outside its range";
        }
        return false;
    }
    staged.mixed.bulkModulus = spec.bulkModulusPa.value;
    material = std::move(staged);
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

PorcineJejunumFungResponse evaluatePorcineJejunumFungResponse(
    const PorcineJejunumFungSpec& spec,
    const double longitudinalStretch,
    const double circumferentialStretch
) {
    if (!finitePositive(spec.fungCpa) ||
        !finitePositive(spec.longitudinalCoefficient) ||
        !finitePositive(spec.circumferentialCoefficient) ||
        !finitePositive(spec.couplingCoefficient) ||
        !(longitudinalStretch > 0.0) ||
        !(circumferentialStretch > 0.0) ||
        !std::isfinite(longitudinalStretch) ||
        !std::isfinite(circumferentialStretch)) {
        throw std::invalid_argument(
            "porcine jejunum Fung response input is invalid"
        );
    }
    const double longitudinalGreen =
        0.5 * (longitudinalStretch * longitudinalStretch - 1.0);
    const double circumferentialGreen =
        0.5 * (
            circumferentialStretch * circumferentialStretch - 1.0
        );
    const double q =
        spec.longitudinalCoefficient.value *
            longitudinalGreen * longitudinalGreen +
        spec.circumferentialCoefficient.value *
            circumferentialGreen * circumferentialGreen +
        2.0 * spec.couplingCoefficient.value *
            longitudinalGreen * circumferentialGreen;
    if (q > 80.0 || !std::isfinite(q)) {
        throw std::overflow_error(
            "porcine jejunum Fung response exceeds the calibrated domain"
        );
    }
    const double exponential = std::exp(q);
    PorcineJejunumFungResponse result;
    result.energyDensityPa =
        0.5 * spec.fungCpa.value * (exponential - 1.0);
    result.longitudinalSecondPiolaPa =
        spec.fungCpa.value * exponential * (
            spec.longitudinalCoefficient.value * longitudinalGreen +
            spec.couplingCoefficient.value * circumferentialGreen
        );
    result.circumferentialSecondPiolaPa =
        spec.fungCpa.value * exponential * (
            spec.circumferentialCoefficient.value *
                circumferentialGreen +
            spec.couplingCoefficient.value * longitudinalGreen
        );
    return result;
}

PorcineJejunumClosureCoupon makePorcineJejunumClosureCoupon(
    const std::uint32_t materialIndex,
    const PorcineJejunumFungSpec& spec
) {
    if (!finitePositive(spec.lengthM) ||
        !finitePositive(spec.widthM) ||
        !finitePositive(spec.thicknessM) ||
        !finitePositive(spec.incisionLengthM) ||
        !finitePositive(spec.incisionGapM) ||
        !finitePositive(spec.densityKgPerM3) ||
        spec.incisionLengthM.value >= spec.lengthM.value ||
        spec.incisionGapM.value >=
            spec.widthM.value /
                static_cast<double>(spec.circumferentialCells) ||
        spec.longitudinalCells < 4u ||
        spec.circumferentialCells < 4u ||
        spec.circumferentialCells % 2u != 0u ||
        spec.throughThicknessCells < 1u ||
        spec.longitudinalCells > 256u ||
        spec.circumferentialCells > 256u ||
        spec.throughThicknessCells > 16u) {
        throw std::invalid_argument(
            "porcine jejunum closure-coupon specification is invalid"
        );
    }

    const std::uint32_t nx = spec.longitudinalCells;
    const std::uint32_t ny = spec.circumferentialCells;
    const std::uint32_t nz = spec.throughThicknessCells;
    const std::uint32_t seamY = ny / 2u;
    const double dx = spec.lengthM.value / static_cast<double>(nx);
    const double dy = spec.widthM.value / static_cast<double>(ny);
    const double dz = spec.thicknessM.value / static_cast<double>(nz);
    const auto validateCoordinates = [](
        const std::vector<double>& positions,
        const std::uint32_t cells,
        const double extent
    ) {
        if (positions.empty()) {
            return;
        }
        if (positions.size() != static_cast<std::size_t>(cells) + 1u ||
            !std::isfinite(positions.front()) ||
            !std::isfinite(positions.back()) ||
            std::abs(positions.front() + 0.5 * extent) > 1.0e-12 ||
            std::abs(positions.back() - 0.5 * extent) > 1.0e-12) {
            throw std::invalid_argument(
                "surgical-tissue mesh coordinates do not match their extent"
            );
        }
        for (std::size_t index = 0u; index < positions.size(); ++index) {
            if (!std::isfinite(positions[index]) ||
                (index != 0u &&
                 !(positions[index] > positions[index - 1u]))) {
                throw std::invalid_argument(
                    "surgical-tissue mesh coordinates must be finite and "
                    "strictly increasing"
                );
            }
        }
    };
    validateCoordinates(
        spec.longitudinalNodePositionsM, nx, spec.lengthM.value
    );
    validateCoordinates(
        spec.circumferentialNodePositionsM, ny, spec.widthM.value
    );
    validateCoordinates(
        spec.thicknessNodePositionsM, nz, spec.thicknessM.value
    );
    const auto minSpacing = [](const std::vector<double>& positions,
                               const double uniformSpacing) {
        if (positions.empty()) {
            return uniformSpacing;
        }
        double result = std::numeric_limits<double>::infinity();
        for (std::size_t index = 1u; index < positions.size(); ++index) {
            result = std::min(
                result, positions[index] - positions[index - 1u]
            );
        }
        return result;
    };
    const double minimumCellSpacing = std::min({
        minSpacing(spec.longitudinalNodePositionsM, dx),
        minSpacing(spec.circumferentialNodePositionsM, dy),
        minSpacing(spec.thicknessNodePositionsM, dz),
    });
    std::uint32_t incisionCellCount = std::clamp(
        static_cast<std::uint32_t>(std::llround(
            spec.incisionLengthM.value / dx
        )),
        2u,
        nx - 2u
    );
    std::uint32_t incisionBegin = (nx - incisionCellCount) / 2u;
    std::uint32_t incisionEnd = incisionBegin + incisionCellCount;
    if (!spec.longitudinalNodePositionsM.empty()) {
        const auto findExactNode = [](
            const std::vector<double>& positions,
            const double target
        ) -> std::uint32_t {
            const auto found = std::lower_bound(
                positions.begin(), positions.end(), target
            );
            if (found == positions.end() ||
                std::abs(*found - target) > 1.0e-12) {
                throw std::invalid_argument(
                    "nonuniform surgical mesh must place a node at each "
                    "incision endpoint"
                );
            }
            return static_cast<std::uint32_t>(
                std::distance(positions.begin(), found)
            );
        };
        const double halfIncision = 0.5 * spec.incisionLengthM.value;
        incisionBegin = findExactNode(
            spec.longitudinalNodePositionsM, -halfIncision
        );
        incisionEnd = findExactNode(
            spec.longitudinalNodePositionsM, halfIncision
        );
        incisionCellCount = incisionEnd - incisionBegin;
        if (incisionBegin < 2u || incisionEnd > nx - 2u ||
            incisionCellCount < 2u) {
            throw std::invalid_argument(
                "nonuniform surgical mesh incision lacks end support"
            );
        }
    }
    if (!spec.circumferentialNodePositionsM.empty() &&
        std::abs(spec.circumferentialNodePositionsM[seamY]) > 1.0e-12) {
        throw std::invalid_argument(
            "nonuniform circumferential mesh must center the wound seam"
        );
    }
    const auto longitudinalPosition = [&](const std::uint32_t ix) {
        if (!spec.longitudinalNodePositionsM.empty()) {
            return spec.longitudinalNodePositionsM[ix];
        }
        const double halfLength = 0.5 * spec.lengthM.value;
        const double halfIncision = 0.5 * spec.incisionLengthM.value;
        if (ix <= incisionBegin) {
            return -halfLength +
                (halfLength - halfIncision) *
                    static_cast<double>(ix) /
                    static_cast<double>(incisionBegin);
        }
        if (ix <= incisionEnd) {
            return -halfIncision + spec.incisionLengthM.value *
                static_cast<double>(ix - incisionBegin) /
                static_cast<double>(incisionCellCount);
        }
        return halfIncision +
            (halfLength - halfIncision) *
                static_cast<double>(ix - incisionEnd) /
                static_cast<double>(nx - incisionEnd);
    };

    PorcineJejunumClosureCoupon result;
    result.spec = spec;
    ObjectSource& object = result.object;
    object.name = "porcine_jejunum_enterotomy_coupon";
    object.materialIndex = materialIndex;
    object.representation = Representation::fem;
    object.twoWayCoupling = true;
    object.characteristicLength = minimumCellSpacing;
    object.mixedFEM = true;
    object.mutationPolicy.enabled = true;
    object.mutationPolicy.cohesiveFracture = true;
    // Automatic puncture remains disabled until needle/tissue impulse has a
    // specimen-specific calibration; explicit puncture commands are reserved.
    object.mutationPolicy.punctureImpulseThreshold = 0.0;

    const std::size_t baseNodeCount =
        static_cast<std::size_t>(nx + 1u) *
        static_cast<std::size_t>(ny + 1u) *
        static_cast<std::size_t>(nz + 1u);
    object.femNodes.reserve(
        baseNodeCount +
        static_cast<std::size_t>(incisionCellCount - 1u) * (nz + 1u)
    );
    std::vector<std::uint32_t> baseIndices(baseNodeCount);
    const auto flat = [=](
        const std::uint32_t ix,
        const std::uint32_t iy,
        const std::uint32_t iz
    ) {
        return (
            static_cast<std::size_t>(iz) * (ny + 1u) + iy
        ) * (nx + 1u) + ix;
    };
    for (std::uint32_t iz = 0u; iz <= nz; ++iz) {
        for (std::uint32_t iy = 0u; iy <= ny; ++iy) {
            for (std::uint32_t ix = 0u; ix <= nx; ++ix) {
                std::array<double, 3> position{
                    longitudinalPosition(ix),
                    spec.circumferentialNodePositionsM.empty()
                        ? -0.5 * spec.widthM.value + dy * iy
                        : spec.circumferentialNodePositionsM[iy],
                    spec.thicknessNodePositionsM.empty()
                        ? -0.5 * spec.thicknessM.value + dz * iz
                        : spec.thicknessNodePositionsM[iz],
                };
                const bool lowerIncisionLip =
                    iy == seamY &&
                    ix > incisionBegin && ix < incisionEnd;
                if (lowerIncisionLip) {
                    position[1] = -0.5 * spec.incisionGapM.value;
                }
                baseIndices[flat(ix, iy, iz)] =
                    static_cast<std::uint32_t>(object.femNodes.size());
                object.femNodes.push_back(position);
            }
        }
    }

    std::vector<std::uint32_t> upperSeam(
        static_cast<std::size_t>(nx + 1u) * (nz + 1u),
        NM_INVALID_INDEX
    );
    const auto seamFlat = [=](
        const std::uint32_t ix,
        const std::uint32_t iz
    ) {
        return static_cast<std::size_t>(iz) * (nx + 1u) + ix;
    };
    for (std::uint32_t ix = incisionBegin + 1u;
         ix < incisionEnd;
         ++ix) {
        for (std::uint32_t iz = 0u; iz <= nz; ++iz) {
            std::array<double, 3> position =
                object.femNodes[baseIndices[flat(ix, seamY, iz)]];
            position[1] = 0.5 * spec.incisionGapM.value;
            upperSeam[seamFlat(ix, iz)] =
                static_cast<std::uint32_t>(object.femNodes.size());
            object.femNodes.push_back(position);
            result.metadata.incisionLipNodePairs.push_back({
                baseIndices[flat(ix, seamY, iz)],
                upperSeam[seamFlat(ix, iz)],
            });
            ++result.metadata.duplicatedIncisionNodeCount;
        }
    }

    const auto node = [&](const std::uint32_t ix,
                          const std::uint32_t iy,
                          const std::uint32_t iz,
                          const bool upperCell) {
        if (upperCell && iy == seamY &&
            ix > incisionBegin && ix < incisionEnd) {
            return upperSeam[seamFlat(ix, iz)];
        }
        return baseIndices[flat(ix, iy, iz)];
    };
    object.tetrahedra.reserve(
        static_cast<std::size_t>(nx) * ny * nz * 6u
    );
    constexpr std::array<std::array<std::uint32_t, 4>, 6> kTets{{
        {{0u, 1u, 3u, 7u}},
        {{0u, 3u, 2u, 7u}},
        {{0u, 2u, 6u, 7u}},
        {{0u, 6u, 4u, 7u}},
        {{0u, 4u, 5u, 7u}},
        {{0u, 5u, 1u, 7u}},
    }};
    double minimumVolume = std::numeric_limits<double>::infinity();
    for (std::uint32_t iz = 0u; iz < nz; ++iz) {
        for (std::uint32_t iy = 0u; iy < ny; ++iy) {
            const bool upperCell = iy >= seamY;
            for (std::uint32_t ix = 0u; ix < nx; ++ix) {
                const std::array<std::uint32_t, 8> corners{{
                    node(ix, iy, iz, upperCell),
                    node(ix + 1u, iy, iz, upperCell),
                    node(ix, iy + 1u, iz, upperCell),
                    node(ix + 1u, iy + 1u, iz, upperCell),
                    node(ix, iy, iz + 1u, upperCell),
                    node(ix + 1u, iy, iz + 1u, upperCell),
                    node(ix, iy + 1u, iz + 1u, upperCell),
                    node(ix + 1u, iy + 1u, iz + 1u, upperCell),
                }};
                for (const auto& local : kTets) {
                    TetrahedronSource tetrahedron{{
                        corners[local[0]],
                        corners[local[1]],
                        corners[local[2]],
                        corners[local[3]],
                    }};
                    const double volume = tetrahedronVolume(
                        object.femNodes[tetrahedron.nodes[0]],
                        object.femNodes[tetrahedron.nodes[1]],
                        object.femNodes[tetrahedron.nodes[2]],
                        object.femNodes[tetrahedron.nodes[3]]
                    );
                    if (!(volume > kMinimumDimension * kMinimumDimension *
                                      kMinimumDimension) ||
                        !std::isfinite(volume)) {
                        throw std::logic_error(
                            "porcine jejunum mesh contains an inverted or "
                            "degenerate tetrahedron"
                        );
                    }
                    minimumVolume = std::min(minimumVolume, volume);
                    object.tetrahedra.push_back(tetrahedron);
                }
            }
        }
    }

    // Contact is restricted to the actual tetrahedral boundary. Cooking every
    // node would expose interior quadrature support as a collision surface and
    // let a needle or tool contact material that is still buried in the wall.
    constexpr std::array<std::array<std::uint32_t, 3>, 4> kFaces{{
        {{1u, 2u, 3u}},
        {{0u, 3u, 2u}},
        {{0u, 1u, 3u}},
        {{0u, 2u, 1u}},
    }};
    std::map<std::array<std::uint32_t, 3>, std::uint32_t> faceIncidence;
    for (const TetrahedronSource& tetrahedron : object.tetrahedra) {
        for (const auto& localFace : kFaces) {
            std::array<std::uint32_t, 3> key{
                tetrahedron.nodes[localFace[0]],
                tetrahedron.nodes[localFace[1]],
                tetrahedron.nodes[localFace[2]],
            };
            std::ranges::sort(key);
            std::uint32_t& incidence = faceIncidence[key];
            ++incidence;
            if (incidence > 2u) {
                throw std::logic_error(
                    "porcine jejunum mesh contains a non-manifold face"
                );
            }
        }
    }
    std::vector<bool> boundaryNode(object.femNodes.size(), false);
    for (const auto& [face, incidence] : faceIncidence) {
        if (incidence != 1u) {
            continue;
        }
        for (const std::uint32_t nodeIndex : face) {
            boundaryNode[nodeIndex] = true;
        }
    }
    for (std::uint32_t index = 0u; index < boundaryNode.size(); ++index) {
        if (boundaryNode[index]) {
            object.femContactNodes.push_back(index);
        }
    }
    if (spec.fixLongitudinalEnds) {
        for (std::uint32_t iz = 0u; iz <= nz; ++iz) {
            for (std::uint32_t iy = 0u; iy <= ny; ++iy) {
                object.femFixedNodes.push_back(
                    baseIndices[flat(0u, iy, iz)]
                );
                object.femFixedNodes.push_back(
                    baseIndices[flat(nx, iy, iz)]
                );
            }
        }
        std::ranges::sort(object.femFixedNodes);
        const auto unique = std::ranges::unique(object.femFixedNodes);
        object.femFixedNodes.erase(unique.begin(), unique.end());
    }

    const std::uint32_t nodeCount =
        static_cast<std::uint32_t>(object.femNodes.size());
    const std::uint32_t tetrahedronCount =
        static_cast<std::uint32_t>(object.tetrahedra.size());
    object.femCapacity.nodes = nodeCount + 64u;
    object.femCapacity.tetrahedra = tetrahedronCount + 128u;
    // A conforming tetrahedral volume has fewer than 2*T internal faces.
    // Reserve that topology-derived bound rather than a blanket factor.
    object.femCapacity.cohesiveFaces = 2u * tetrahedronCount;
    object.femCapacity.punctureChannels = 64u;
    object.femCapacity.mutationCommands = 64u;

    result.metadata.nodeCount = nodeCount;
    result.metadata.tetrahedronCount = tetrahedronCount;
    result.metadata.fixedNodeCount =
        static_cast<std::uint32_t>(object.femFixedNodes.size());
    result.metadata.minimumRestTetrahedronVolumeM3 = minimumVolume;
    result.metadata.fidelityBoundary =
        "Source-parameterized porcine jejunal planar hyperelasticity; "
        "research 3-D regularization, density, incision and fixture; no "
        "patient, perfusion, layered histology, failure or puncture "
        "calibration.";
    return result;
}

SyntheticSkinWoundCoupon makeSyntheticSkinWoundCoupon(
    const std::uint32_t materialIndex,
    const SyntheticSkinWoundSpec& spec
) {
    // The established incision mesher is geometric. Its jejunal response law
    // is configured separately and is never copied into this skin specimen.
    PorcineJejunumFungSpec geometry;
    geometry.lengthM = {spec.lengthM, JejunalValueBasis::researchDefault};
    geometry.widthM = {spec.widthM, JejunalValueBasis::researchDefault};
    geometry.thicknessM = {
        spec.thicknessM, JejunalValueBasis::researchDefault
    };
    geometry.incisionLengthM = {
        spec.incisionLengthM, JejunalValueBasis::researchDefault
    };
    geometry.incisionGapM = {
        spec.incisionGapM, JejunalValueBasis::researchDefault
    };
    geometry.densityKgPerM3 = {
        spec.densityKgPerM3, JejunalValueBasis::researchDefault
    };
    geometry.longitudinalCells = spec.longitudinalCells;
    geometry.circumferentialCells = spec.transverseCells;
    geometry.throughThicknessCells = spec.throughThicknessCells;
    geometry.fixLongitudinalEnds = spec.fixLongitudinalEnds;
    geometry.longitudinalNodePositionsM = spec.longitudinalNodePositionsM;
    geometry.circumferentialNodePositionsM = spec.transverseNodePositionsM;
    geometry.thicknessNodePositionsM = spec.thicknessNodePositionsM;
    PorcineJejunumClosureCoupon mesh =
        makePorcineJejunumClosureCoupon(materialIndex, geometry);

    SyntheticSkinWoundCoupon result;
    result.object = std::move(mesh.object);
    result.spec = spec;
    result.metadata = std::move(mesh.metadata);
    result.object.name = "synthetic_skin_wound_coupon";
    result.metadata.fidelityBoundary =
        "Homogeneous synthetic skin-like material and authored incision "
        "geometry; no measured human-skin fit, layered histology, puncture "
        "force calibration, tear threshold, or clinical claim.";
    return result;
}

} // namespace numi::matter
