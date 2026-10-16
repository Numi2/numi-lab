#include "cardboard_constituent_volume_damage.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using numi::cardboard::damage_volume::PaperConstituentDamageVolume;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance = 1e-14) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            "unexpected volume-damage result");
}
}

int main() {
    try {
        const std::array<double, 4> areasA{1.0, 3.0, 2.0, 2.0};
        const std::array<double, 4> damageA{0.0, 0.5, 1.0, 0.0};
        const std::array<double, 4> equalAreas{1.0, 1.0, 1.0, 1.0};
        const std::array<double, 4> damageB{0.75, 0.75, 0.75, 0.75};

        PaperConstituentDamageVolume coarse;
        require(coarse.addTetrahedron(1.0, areasA, damageA, 0.5),
                "valid heterogeneous tet rejected");
        require(coarse.addTetrahedron(3.0, equalAreas, damageB, 0.5),
                "valid second tet rejected");
        double index = -1.0;
        double thresholdFraction = -1.0;
        require(coarse.finish(index, thresholdFraction), "valid aggregate rejected");
        near(index, 0.671875);
        near(thresholdFraction, 0.75);
        near(coarse.referenceVolumeM3, 4.0);
        require(coarse.tetrahedronCount == 2u, "tetrahedron count mismatch");

        // Subdivide the uniform volume-3 constituent into three equal reference
        // volume cells. Both the volume-weighted index and thresholded volume
        // fraction must remain invariant.
        PaperConstituentDamageVolume refined;
        require(refined.addTetrahedron(1.0, areasA, damageA, 0.5),
                "refined heterogeneous tet rejected");
        for (int i = 0; i < 3; ++i)
            require(refined.addTetrahedron(1.0, equalAreas, damageB, 0.5),
                    "subdivided uniform tet rejected");
        double refinedIndex = -1.0;
        double refinedThresholdFraction = -1.0;
        require(refined.finish(refinedIndex, refinedThresholdFraction),
                "refined aggregate rejected");
        near(refinedIndex, index);
        near(refinedThresholdFraction, thresholdFraction);

        const std::array<double, 4> boundaryDamage{0.5, 0.5, 0.5, 0.5};
        PaperConstituentDamageVolume boundary;
        require(boundary.addTetrahedron(2.0, equalAreas, boundaryDamage, 0.5),
                "threshold-boundary tet rejected");
        double boundaryIndex = -1.0;
        double boundaryFraction = -1.0;
        require(boundary.finish(boundaryIndex, boundaryFraction),
                "threshold-boundary aggregate rejected");
        near(boundaryIndex, 0.5);
        near(boundaryFraction, 1.0);

        PaperConstituentDamageVolume empty;
        require(!empty.finish(index, thresholdFraction), "empty aggregate admitted");
        const std::array<double, 4> badArea{1.0, 0.0, 1.0, 1.0};
        const std::array<double, 4> badDamage{0.0, 0.1, 1.0001, 0.0};
        PaperConstituentDamageVolume invalid;
        require(!invalid.addTetrahedron(1.0, badArea, damageA, 0.5),
                "zero plane area admitted");
        require(!invalid.addTetrahedron(1.0, equalAreas, badDamage, 0.5),
                "out-of-range plane damage admitted");
        require(!invalid.addTetrahedron(
                    std::numeric_limits<double>::quiet_NaN(), equalAreas, damageA, 0.5),
                "nonfinite reference volume admitted");
        require(!invalid.addTetrahedron(1.0, equalAreas, damageA, 1.1),
                "invalid threshold admitted");
        std::cout << "constituent volume damage index: heterogeneous weighting, subdivision invariance, threshold boundary, and invalid-input checks passed; CPU observer only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
