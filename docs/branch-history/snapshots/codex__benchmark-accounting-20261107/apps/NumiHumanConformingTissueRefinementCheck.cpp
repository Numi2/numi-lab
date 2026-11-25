// Synthetic offline regression for the refinement verifier. This does not
// authenticate the retained anatomical assets or qualify runtime mechanics.
#include "NumiHumanConformingTissueRefinement.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
using Point = std::array<double, 3>;
using Tet = std::array<std::uint32_t, 4>;

struct Fixture {
    std::vector<Point> source{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
    std::vector<Tet> sourceTets{{0,1,2,3},{0,1,4,2}};
    std::vector<std::array<std::uint32_t,2>> edges{{0,1}};
    std::vector<Point> refined{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1},{0.5,0,0}};
    std::vector<Tet> child{{0,5,2,3},{0,5,4,2},{5,1,2,3},{5,1,4,2}};
    std::vector<std::uint32_t> parent{0,1,0,1};
    std::vector<double> fractions{0.5,0.5,0.5,0.5};
    std::vector<std::uint32_t> offsets{0,1,2,3,4,5,7};
    std::vector<std::uint32_t> rows{0,1,2,3,4,0,1};
    std::vector<double> weights{1,1,1,1,1,0.5,0.5};
    std::vector<std::uint32_t> sourceMaterial{0,1}, childMaterial{0,1,0,1};
    std::vector<std::uint32_t> sourceDonor{0,1}, childDonor{0,1,0,1};
    std::vector<std::uint32_t> sourceRegion{10,11}, childRegion{10,11,10,11};
    std::vector<std::uint32_t> sourceFace{20,21}, childFace{20,21,20,21};
    std::vector<double> volumes{1.0/6.0,1.0/6.0}, mass{1.0/3.0,2.0/3.0};
    std::vector<double> lower{1.9,3.9}, upper{2.1,4.1};
    std::string hash = std::string(64, 'a');

    ConformingTissueRefinementInput input() const {
        return {source,sourceTets,edges,refined,child,parent,fractions,
            offsets,rows,weights,sourceMaterial,childMaterial,
            sourceDonor,childDonor,sourceRegion,childRegion,sourceFace,childFace,
            volumes,mass,lower,upper,hash,hash,hash,hash,hash,2};
    }
};

bool near(double a, double b) { return std::abs(a-b) <= 1e-13; }

bool run() {
    Fixture f;
    ConformingTissueRefinementResult result;
    std::string error;
    if (!verifyConformingTissueRefinement(f.input(), result, error)) {
        std::cerr << "valid shared-edge fixture rejected: " << error << '\n';
        return false;
    }
    const std::vector<double> expectedMass{1.0/6.0,1.0/3.0,1.0/6.0,1.0/3.0};
    if (result.childMassKg.size() != expectedMass.size() ||
        result.verificationSHA256.size() != 64 ||
        !near(result.maximumParentMassClosureErrorKg, 0) ||
        !near(result.maximumParentMeasureClosureErrorM3, 0)) {
        std::cerr << "mass closure or proof fingerprint mismatch\n";
        return false;
    }
    for (std::size_t i=0;i<expectedMass.size();++i) {
        if (!near(result.childMassKg[i], expectedMass[i]) ||
            !near(result.childReferenceVolumeM3[i], 1.0/12.0) ||
            result.childParentRows[i] != f.parent[i]) {
            std::cerr << "wrong child mass, volume or parent at " << i << '\n';
            return false;
        }
    }
    const auto retainedHash = result.verificationSHA256;
    const auto retainedMass = result.childMassKg;
    const auto reject = [&](const char* label) {
        error.clear();
        if (verifyConformingTissueRefinement(f.input(), result, error) ||
            error.empty() || result.verificationSHA256 != retainedHash ||
            result.childMassKg != retainedMass) {
            std::cerr << label << " was not rejected atomically\n";
            return false;
        }
        return true;
    };
    f.child[1] = f.sourceTets[1]; // A partial star is not a conforming split.
    if (!reject("partial shared-edge star")) return false;
    f.child[1] = {0,5,4,2};
    f.refined[5][0] = 0.51;
    if (!reject("changed midpoint")) return false;
    f.refined[5][0] = 0.5;
    f.weights[5] = 0.25;
    if (!reject("changed ancestry")) return false;
    f.weights[5] = 0.5;
    f.childRegion[2] = 11;
    if (!reject("changed region")) return false;
    f.childRegion[2] = 10;
    f.mass[0] = 1.0;
    if (!reject("out-of-range density")) return false;
    f.mass[0] = 1.0/3.0;
    f.hash = "not-a-sha";
    if (!reject("unpinned input")) return false;
    std::cout << "conforming tissue refinement synthetic shared-edge and rejection checks passed\n";
    return true;
}
} // namespace

int main() { return run() ? 0 : 1; }
