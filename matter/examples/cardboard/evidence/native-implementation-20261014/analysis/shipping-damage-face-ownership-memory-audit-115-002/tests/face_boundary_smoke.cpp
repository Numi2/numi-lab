#include "cardboard_fracture_regularization.hpp"
#include "cardboard_fracture_regularization_original.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace numi::cardboard;

static bool sameBits(double a,double b) {
    return std::bit_cast<std::uint64_t>(a)==std::bit_cast<std::uint64_t>(b);
}
static void requireSameGeometry(const TetCrackPlaneGeometry& patched,
                                const baseline::cardboard::TetCrackPlaneGeometry& original) {
    if (!sameBits(patched.referenceVolumeM3,original.referenceVolumeM3) ||
        !sameBits(patched.referenceAreaM2,original.referenceAreaM2) ||
        patched.characteristicLengthM.has_value()!=original.characteristicLengthM.has_value() ||
        patched.cutPolygonReferenceM.size()!=original.cutPolygonReferenceM.size() ||
        !sameBits(patched.referenceUnitNormal.x,original.referenceUnitNormal.x) ||
        !sameBits(patched.referenceUnitNormal.y,original.referenceUnitNormal.y) ||
        !sameBits(patched.referenceUnitNormal.z,original.referenceUnitNormal.z))
        throw std::runtime_error("default geometry arithmetic differs from baseline");
    if (patched.characteristicLengthM &&
        !sameBits(*patched.characteristicLengthM,*original.characteristicLengthM))
        throw std::runtime_error("default characteristic length differs from baseline");
    for (std::size_t i=0;i<patched.cutPolygonReferenceM.size();++i) {
        const auto a=patched.cutPolygonReferenceM[i];
        const auto b=original.cutPolygonReferenceM[i];
        if (!sameBits(a.x,b.x)||!sameBits(a.y,b.y)||!sameBits(a.z,b.z))
            throw std::runtime_error("default cut polygon differs from baseline");
    }
}

int main() {
    const TetCrackPlane plane{{0,0,1},FractureNormalFrame::Reference,0.0,{},1.0e-12};
    auto vertices = [](double h) {
        return std::array<FractureVec3,4>{{{0,0,0},{1,0,0},{0,1,0},{0,0,h}}};
    };
    const auto baseline = vertices(1.0);
    const auto baselineClass = classifyTetCrackPlane(baseline,plane);
    if (!tetCrackPlaneClearsEveryFullFace(baselineClass)) return 1;
    const std::array<baseline::cardboard::FractureVec3,4> originalVertices{{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1}}};
    const auto originalOrdinary=baseline::cardboard::computeTetCrackPlaneGeometry(
        originalVertices,baseline::cardboard::TetCrackPlane{{0,0,1},
            baseline::cardboard::FractureNormalFrame::Reference,0.0,{},1.0e-12});
    requireSameGeometry(computeTetCrackPlaneGeometry(baseline,plane),originalOrdinary);

    const auto sliver = vertices(2.0e-12);
    const auto sliverClass = classifyTetCrackPlane(sliver,plane);
    if (!(sliverClass.referenceVolumeM3 > 128.0*std::numeric_limits<double>::epsilon()*
          sliverClass.edgeScaleM*sliverClass.edgeScaleM*sliverClass.edgeScaleM)) return 2;
    if (tetCrackPlaneClearsEveryFullFace(sliverClass)) return 3;
    if (!computeTetCrackPlaneGeometry(sliver,plane).hasPositiveAreaCut()) return 4;
    bool rejected = false;
    try {
        (void)computeTetCrackPlaneGeometry(
            sliver,plane,FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 5;

    const double boundaryHeight = 4.0*baselineClass.distanceToleranceM;
    const auto boundary = vertices(boundaryHeight);
    const auto boundaryClass = classifyTetCrackPlane(boundary,plane);
    if (std::abs(boundaryClass.signedDistanceM[0]) != boundaryClass.distanceToleranceM) return 6;
    if (tetCrackPlaneClearsEveryFullFace(boundaryClass)) return 7;
    const auto outside = vertices(std::nextafter(boundaryHeight,
                                                std::numeric_limits<double>::infinity()));
    if (!tetCrackPlaneClearsEveryFullFace(classifyTetCrackPlane(outside,plane))) return 8;

    TetCrackPlane rotated{{1,0,0},FractureNormalFrame::ElementMaterial,0.0,{},1.0e-12};
    rotated.materialToReference.value={0,-1,0,1,0,0,0,0,1};
    const auto rotatedClass=classifyTetCrackPlane(baseline,rotated);
    if (!tetCrackPlaneClearsEveryFullFace(rotatedClass)) return 9;
    (void)computeTetCrackPlaneGeometry(
        baseline,rotated,FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance);
    const std::array<baseline::cardboard::FractureVec3,4> originalRotatedVertices{{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1}}};
    baseline::cardboard::TetCrackPlane originalRotated{{1,0,0},
        baseline::cardboard::FractureNormalFrame::ElementMaterial,0.0,{},1.0e-12};
    originalRotated.materialToReference.value={0,-1,0,1,0,0,0,0,1};
    requireSameGeometry(computeTetCrackPlaneGeometry(baseline,rotated),
        baseline::cardboard::computeTetCrackPlaneGeometry(
            originalRotatedVertices,originalRotated));
    std::cout << "face_boundary_smoke_passed\n";
}
