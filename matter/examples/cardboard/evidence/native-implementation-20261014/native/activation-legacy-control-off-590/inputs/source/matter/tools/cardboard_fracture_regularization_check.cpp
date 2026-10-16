#include "cardboard_fracture_regularization.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace numi::cardboard;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double actual, double expected, double absoluteTolerance,
          double relativeTolerance = 1.0e-12) {
    return std::abs(actual - expected) <=
           absoluteTolerance + relativeTolerance * std::abs(expected);
}

template<class F>
void expectInvalid(F&& action, const std::string& message) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

std::array<FractureVec3, 4> unitRightTet() {
    return {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
             {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

TetCrackPlane zCut(double offset) {
    TetCrackPlane plane;
    plane.normal = {0.0, 0.0, 1.0};
    plane.offsetFromCentroidM = offset;
    return plane;
}

FractureVec3 rotateY90(FractureVec3 p) {
    return {p.z, p.y, -p.x};
}

void testAnalyticCutsAndEnergy() {
    const auto tet = unitRightTet();
    const auto horizontal = computeTetCrackPlaneGeometry(tet, zCut(0.0));
    const double expectedVolume = 1.0 / 6.0;
    const double expectedArea = 9.0 / 32.0;
    require(horizontal.cutPolygonReferenceM.size() == 3u &&
            near(horizontal.referenceVolumeM3, expectedVolume, 1.0e-14) &&
            near(horizontal.referenceAreaM2, expectedArea, 1.0e-14) &&
            horizontal.characteristicLengthM.has_value() &&
            near(*horizontal.characteristicLengthM, 16.0 / 27.0, 1.0e-14),
            "centroid z-plane did not produce the analytic triangle, V/A");

    TetCrackPlane diagonal;
    diagonal.normal = {1.0, 1.0, 1.0};
    diagonal.offsetFromCentroidM =
        0.5 / std::sqrt(3.0) - std::sqrt(3.0) / 4.0;
    // The plane equation is x+y+z=0.5; the offset is measured along unit n
    // from the tet centroid (1/4,1/4,1/4).
    const auto diagonalCut = computeTetCrackPlaneGeometry(tet, diagonal);
    const double diagonalArea = std::sqrt(3.0) / 8.0;
    require(diagonalCut.cutPolygonReferenceM.size() == 3u &&
            near(diagonalCut.referenceAreaM2, diagonalArea, 1.0e-14) &&
            near(*diagonalCut.characteristicLengthM,
                 4.0 / (3.0 * std::sqrt(3.0)), 1.0e-14),
            "oblique plane did not produce the analytic triangle, V/A");

    TetCrackPlane quadrilateral;
    quadrilateral.normal = {1.0, 1.0, 0.0};
    const auto quadCut = computeTetCrackPlaneGeometry(tet, quadrilateral);
    require(quadCut.cutPolygonReferenceM.size() == 4u &&
            near(quadCut.referenceAreaM2, std::sqrt(2.0) / 4.0, 1.0e-14),
            "two-positive/two-negative cut did not produce the analytic quadrilateral");

    TetCrackPlane facePlane;
    facePlane.normal = {0.0, 0.0, 1.0};
    facePlane.offsetFromCentroidM = -0.25; // coincident with the z=0 face
    const auto faceCut = computeTetCrackPlaneGeometry(tet, facePlane);
    require(faceCut.cutPolygonReferenceM.size() == 3u &&
            near(faceCut.referenceAreaM2, 0.5, 1.0e-14),
            "plane coincident with a tetrahedron face did not return that element-local area");

    constexpr double gc = 120.0; // J/m^2; test fixture only, not a paper value.
    const auto budget = makeTetFractureEnergyBudget(horizontal, gc);
    require(near(budget.dissipatedEnergyJ, gc * expectedArea, 1.0e-14) &&
            near(budget.equivalentVolumetricDissipationJPerM3,
                 gc / *horizontal.characteristicLengthM, 1.0e-12),
            "fracture budget is inconsistent with Gc*A=(Gc/l)*V");
}

void testFramesAndVertexWinding() {
    const auto tet = unitRightTet();
    auto referenceX = zCut(0.0);
    referenceX.normal = {1.0, 0.0, 0.0};
    const auto xCut = computeTetCrackPlaneGeometry(tet, referenceX);

    TetCrackPlane materialZ;
    materialZ.normal = {0.0, 0.0, 1.0};
    materialZ.normalFrame = FractureNormalFrame::ElementMaterial;
    materialZ.offsetFromCentroidM = 0.0;
    materialZ.materialToReference.value = {
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
    };
    const auto materialCut = computeTetCrackPlaneGeometry(tet, materialZ);
    require(near(materialCut.referenceAreaM2, xCut.referenceAreaM2, 1.0e-14) &&
            near(*materialCut.characteristicLengthM,
                 *xCut.characteristicLengthM, 1.0e-14) &&
            near(materialCut.referenceUnitNormal.x, 1.0, 1.0e-14) &&
            near(materialCut.referenceUnitNormal.z, 0.0, 1.0e-14),
            "material-frame normal was not rotated to reference coordinates");

    auto reversedTet = tet;
    std::swap(reversedTet[1], reversedTet[2]);
    const auto reversed = computeTetCrackPlaneGeometry(reversedTet, zCut(0.0));
    const auto original = computeTetCrackPlaneGeometry(tet, zCut(0.0));
    require(near(reversed.referenceVolumeM3, original.referenceVolumeM3, 1.0e-14) &&
            near(reversed.referenceAreaM2, original.referenceAreaM2, 1.0e-14) &&
            near(*reversed.characteristicLengthM,
                 *original.characteristicLengthM, 1.0e-14),
            "tetrahedron vertex winding changed positive reference geometry");

    auto rotatedTet = tet;
    for (auto& vertex : rotatedTet) vertex = rotateY90(vertex);
    TetCrackPlane rotatedZ = zCut(0.0);
    rotatedZ.normal = {1.0, 0.0, 0.0}; // R_y(90) maps reference z to reference x.
    const auto rotated = computeTetCrackPlaneGeometry(rotatedTet, rotatedZ);
    require(near(rotated.referenceVolumeM3, original.referenceVolumeM3, 1.0e-14) &&
            near(rotated.referenceAreaM2, original.referenceAreaM2, 1.0e-14) &&
            near(*rotated.characteristicLengthM,
                 *original.characteristicLengthM, 1.0e-14),
            "rigidly rotating the tet and crack normal changed V/A");

    auto reflected = materialZ;
    reflected.materialToReference.value[0] = -1.0;
    expectInvalid([&] { (void)computeTetCrackPlaneGeometry(tet, reflected); },
                  "reflected material frame was admitted as a proper rotation");
}

void testScalingAndSubdivision() {
    const auto baseTet = unitRightTet();
    const auto base = computeTetCrackPlaneGeometry(baseTet, zCut(0.0));
    constexpr double scale = 2.5;
    auto scaledTet = baseTet;
    for (auto& vertex : scaledTet) vertex = vertex * scale;
    TetCrackPlane scaledPlane = zCut(0.0);
    scaledPlane.offsetFromCentroidM = 0.0;
    const auto scaled = computeTetCrackPlaneGeometry(scaledTet, scaledPlane);
    require(near(scaled.referenceVolumeM3,
                 base.referenceVolumeM3 * scale * scale * scale, 1.0e-12) &&
            near(scaled.referenceAreaM2,
                 base.referenceAreaM2 * scale * scale, 1.0e-12) &&
            near(*scaled.characteristicLengthM,
                 *base.characteristicLengthM * scale, 1.0e-12),
            "uniform geometry scaling did not follow V~s^3, A~s^2, l~s");

    FractureVec3 centroid{0.25, 0.25, 0.25};
    constexpr std::array<std::array<std::size_t, 3>, 4> faces{{
        {{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}},
    }};
    double subVolume = 0.0;
    double subArea = 0.0;
    double subEnergy = 0.0;
    constexpr double gc = 83.0;
    for (const auto& face : faces) {
        const std::array<FractureVec3, 4> subTet{{
            centroid, baseTet[face[0]], baseTet[face[1]], baseTet[face[2]]}};
        FractureVec3 subCentroid{};
        for (FractureVec3 point : subTet) subCentroid = subCentroid + point;
        subCentroid = subCentroid / 4.0;
        // Keep the same physical plane z=1/4 across the partition; do not
        // silently recenter a different crack plane in each child element.
        const auto geometry = computeTetCrackPlaneGeometry(
            subTet, zCut(0.25 - subCentroid.z));
        subVolume += geometry.referenceVolumeM3;
        subArea += geometry.referenceAreaM2;
        subEnergy += makeTetFractureEnergyBudget(geometry, gc).dissipatedEnergyJ;
    }
    require(near(subVolume, base.referenceVolumeM3, 1.0e-14) &&
            near(subArea, base.referenceAreaM2, 1.0e-14) &&
            near(subEnergy, gc * base.referenceAreaM2, 1.0e-12),
            "centroid subdivision did not conserve volume, cut area, and Gc*A");

    TetCrackPlane outside = zCut(1.0);
    const auto noCut = computeTetCrackPlaneGeometry(baseTet, outside);
    require(!noCut.hasPositiveAreaCut() && noCut.referenceAreaM2 == 0.0 &&
            !noCut.characteristicLengthM &&
            makeTetFractureEnergyBudget(noCut, gc).dissipatedEnergyJ == 0.0,
            "plane outside the tet did not produce an explicit inactive result");
}

void testFailClosedInputs() {
    const auto tet = unitRightTet();
    TetCrackPlane zeroNormal = zCut(0.0);
    zeroNormal.normal = {};
    expectInvalid([&] { (void)computeTetCrackPlaneGeometry(tet, zeroNormal); },
                  "zero crack-plane normal was admitted");

    auto flatTet = tet;
    flatTet[3] = flatTet[0];
    expectInvalid([&] { (void)computeTetCrackPlaneGeometry(flatTet, zCut(0.0)); },
                  "zero-volume tetrahedron was admitted");

    const auto geometry = computeTetCrackPlaneGeometry(tet, zCut(0.0));
    expectInvalid([&] { (void)makeTetFractureEnergyBudget(geometry, -1.0); },
                  "negative fracture energy release rate was admitted");
    TetCrackPlaneGeometry malformed = geometry;
    malformed.referenceAreaM2 = std::numeric_limits<double>::quiet_NaN();
    expectInvalid([&] { (void)makeTetFractureEnergyBudget(malformed, 1.0); },
                  "nonfinite caller-supplied fracture geometry was admitted");
    malformed = geometry;
    malformed.characteristicLengthM = *geometry.characteristicLengthM * 2.0;
    expectInvalid([&] { (void)makeTetFractureEnergyBudget(malformed, 1.0); },
                  "energy budget accepted a characteristic length inconsistent with V/A");
}

} // namespace

int main() {
    try {
        testAnalyticCutsAndEnergy();
        testFramesAndVertexWinding();
        testScalingAndSubdivision();
        testFailClosedInputs();
        std::cout << "cardboard_tet_fracture_reference_geometry_passed "
                     "energy_fixture_gc_is_synthetic_not_calibration "
                     "plane_area_is_element_local_and_reference_configuration\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
