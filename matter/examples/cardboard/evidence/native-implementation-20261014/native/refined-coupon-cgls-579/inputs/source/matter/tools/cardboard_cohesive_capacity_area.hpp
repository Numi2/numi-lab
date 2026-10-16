#pragma once
#include <array>
#include <cmath>
#include <cstdint>
namespace numi::cardboard {
// The native mixed-mode face uses three equal reference-area quadrature
// weights. face.state.x is their MAXIMUM damage, not a capacity-area integral.
struct CohesiveQuadratureCapacityArea {
    bool valid = false;
    double failedReferenceAreaM2 = 0.0;
    std::uint32_t failedQuadraturePoints = 0u;
    bool wholeFaceFailed = false;
};
inline CohesiveQuadratureCapacityArea cohesiveQuadratureCapacityArea(
    const double referenceAreaM2, const std::array<double, 3>& pointDamage) {
    CohesiveQuadratureCapacityArea result;
    if (!(std::isfinite(referenceAreaM2) && referenceAreaM2 > 0.0))
        return result;
    for (const double value : pointDamage) {
        if (!(std::isfinite(value) && value >= 0.0 && value <= 1.0))
            return {};
        if (value == 1.0) ++result.failedQuadraturePoints;
    }
    result.failedReferenceAreaM2 = referenceAreaM2 *
        (static_cast<double>(result.failedQuadraturePoints) / 3.0);
    result.wholeFaceFailed = result.failedQuadraturePoints == 3u;
    result.valid = true;
    return result;
}
}
