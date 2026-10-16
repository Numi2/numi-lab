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
    const std::uint32_t incisionCellCount = std::clamp(
        static_cast<std::uint32_t>(std::llround(
            spec.incisionLengthM.value / dx
        )),
        2u,
        nx - 2u
    );
    const std::uint32_t incisionBegin =
        (nx - incisionCellCount) / 2u;
    const std::uint32_t incisionEnd =
        incisionBegin + incisionCellCount;
    const auto longitudinalPosition = [&](const std::uint32_t ix) {
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
    object.characteristicLength = std::min({dx, dy, dz});
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
                    -0.5 * spec.widthM.value + dy * iy,
                    -0.5 * spec.thicknessM.value + dz * iz,
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

} // namespace numi::matter
