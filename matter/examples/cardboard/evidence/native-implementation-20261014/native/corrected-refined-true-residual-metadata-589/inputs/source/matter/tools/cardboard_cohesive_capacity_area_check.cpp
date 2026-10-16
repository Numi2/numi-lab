#include "cardboard_cohesive_capacity_area.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using numi::cardboard::cohesiveQuadratureCapacityArea;
namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void near(double a, double b) { require(std::abs(a-b) < 1e-14, "reference area mismatch"); }
}
int main() {
    try {
        const auto intact = cohesiveQuadratureCapacityArea(6.0, {0.0,0.0,0.0});
        require(intact.valid && intact.failedQuadraturePoints==0u &&
                !intact.wholeFaceFailed, "intact face failed");
        near(intact.failedReferenceAreaM2, 0.0);
        // A triangular face with three equal native quadrature cells:
        // failure of only one cell removes 2 of its 6 m^2 reference area,
        // although the native face maximum is already 1.
        const auto partial = cohesiveQuadratureCapacityArea(6.0, {0.2,1.0,0.7});
        require(partial.valid && partial.failedQuadraturePoints==1u &&
                !partial.wholeFaceFailed, "nonuniform face counted wholly failed");
        near(partial.failedReferenceAreaM2, 2.0);
        const auto two = cohesiveQuadratureCapacityArea(6.0, {1.0,1.0,0.999999});
        require(two.valid && two.failedQuadraturePoints==2u &&
                !two.wholeFaceFailed, "near-complete damage was rounded to failure");
        near(two.failedReferenceAreaM2, 4.0);
        const auto full = cohesiveQuadratureCapacityArea(6.0, {1.0,1.0,1.0});
        require(full.valid && full.wholeFaceFailed, "complete failure omitted");
        near(full.failedReferenceAreaM2, 6.0);
        // Refining a uniformly failed cell into two half-area cells leaves
        // the integrated failed reference area unchanged.
        const auto left = cohesiveQuadratureCapacityArea(3.0, {1.0,1.0,1.0});
        const auto right = cohesiveQuadratureCapacityArea(3.0, {1.0,1.0,1.0});
        near(left.failedReferenceAreaM2+right.failedReferenceAreaM2, full.failedReferenceAreaM2);
        for (const double d : {-0.1,1.1,std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity()})
            require(!cohesiveQuadratureCapacityArea(1.0,{0.0,d,1.0}).valid,
                    "invalid history admitted");
        for (const double a : {0.0,-1.0,std::numeric_limits<double>::infinity()})
            require(!cohesiveQuadratureCapacityArea(a,{0.0,0.0,0.0}).valid,
                    "invalid area admitted");
        std::cout << "cohesive capacity area: nonuniform quadrature, refinement and invalid-history checks passed; CPU observer only\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
