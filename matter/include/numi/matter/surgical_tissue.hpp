#pragma once

#include "numi/matter/matter.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace numi::matter {

enum class JejunalValueBasis : std::uint32_t {
    belliniPorcineBiaxialStudy = 0u,
    derivedGeometry = 1u,
    researchDefault = 2u,
};

struct JejunalScalar {
    double value = 0.0;
    JejunalValueBasis basis = JejunalValueBasis::researchDefault;
};

[[nodiscard]] std::string_view jejunalValueBasisName(
    JejunalValueBasis basis
) noexcept;

[[nodiscard]] std::string_view jejunalValueSourceReference(
    JejunalValueBasis basis
) noexcept;

// Porcine jejunal-wall calibration from Bellini et al., J Mech Behav Biomed
// Mater 2011. The published planar Fung coefficients and mean wall thickness
// are source values. Coupon extent, incision, density, volumetric penalty and
// missing 3-D shear regularization are kept visibly separate as simulator
// choices; this is not an identified patient or a whole-organ model.
struct PorcineJejunumFungSpec {
    JejunalScalar lengthM{
        0.030,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar widthM{
        0.024,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar thicknessM{
        0.00077,
        JejunalValueBasis::belliniPorcineBiaxialStudy,
    };
    JejunalScalar incisionLengthM{
        0.016,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar incisionGapM{
        0.00060,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar densityKgPerM3{
        1000.0,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar fungCpa{
        690.0,
        JejunalValueBasis::belliniPorcineBiaxialStudy,
    };
    // Bellini's first (11) axis is circumferential and second (22) axis is
    // longitudinal. Keep that source convention explicit while mapping the
    // coupon's x column to longitudinal and y column to circumferential.
    JejunalScalar longitudinalCoefficient{
        81.2,
        JejunalValueBasis::belliniPorcineBiaxialStudy,
    };
    JejunalScalar circumferentialCoefficient{
        72.4,
        JejunalValueBasis::belliniPorcineBiaxialStudy,
    };
    JejunalScalar couplingCoefficient{
        19.7,
        JejunalValueBasis::belliniPorcineBiaxialStudy,
    };
    JejunalScalar groundShearPa{
        250.0,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar numericalViscosityPaS{
        5.0,
        JejunalValueBasis::researchDefault,
    };
    JejunalScalar bulkModulusPa{
        150000.0,
        JejunalValueBasis::researchDefault,
    };
    std::uint32_t longitudinalCells = 18u;
    std::uint32_t circumferentialCells = 16u;
    std::uint32_t throughThicknessCells = 2u;
    bool fixLongitudinalEnds = false;
};

struct PorcineJejunumFungResponse {
    double energyDensityPa = 0.0;
    double longitudinalSecondPiolaPa = 0.0;
    double circumferentialSecondPiolaPa = 0.0;
};

struct PorcineJejunumClosureMetadata {
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t duplicatedIncisionNodeCount = 0u;
    std::uint32_t fixedNodeCount = 0u;
    std::array<double, 3> longitudinalAxis{1.0, 0.0, 0.0};
    std::array<double, 3> circumferentialAxis{0.0, 1.0, 0.0};
    std::array<double, 3> thicknessAxis{0.0, 0.0, 1.0};
    double minimumRestTetrahedronVolumeM3 = 0.0;
    // Matching lower/upper lip nodes, ordered longitudinally then through
    // thickness. These are stable future needle-bite targets, not stitches.
    std::vector<std::array<std::uint32_t, 2>> incisionLipNodePairs;
    std::string fidelityBoundary;
};

struct PorcineJejunumClosureCoupon {
    ObjectSource object;
    PorcineJejunumFungSpec spec;
    PorcineJejunumClosureMetadata metadata;
};

// Applies the source coefficients and explicit 3-D regularization settings to
// a parsed porcine_jejunum_fung material. Publication is transactional.
[[nodiscard]] bool configurePorcineJejunumFungMaterial(
    MaterialProgram& material,
    const PorcineJejunumFungSpec& spec,
    std::string* error = nullptr
);

// Analytic planar calibration oracle for diagonal longitudinal/
// circumferential stretches. It evaluates only the published Fung term, not
// the explicit 3-D numerical regularizer.
[[nodiscard]] PorcineJejunumFungResponse
evaluatePorcineJejunumFungResponse(
    const PorcineJejunumFungSpec& spec,
    double longitudinalStretch,
    double circumferentialStretch
);

// Builds an incision-bearing, two-layer-or-better tetrahedral wall coupon.
// The incision is an actual duplicated free surface with a finite initial gap,
// and the returned object reserves bounded cohesive/puncture growth for later
// needle/tissue work. No stitch or hidden constraint joins the two lips.
[[nodiscard]] PorcineJejunumClosureCoupon
makePorcineJejunumClosureCoupon(
    std::uint32_t materialIndex,
    const PorcineJejunumFungSpec& spec = {}
);

} // namespace numi::matter
