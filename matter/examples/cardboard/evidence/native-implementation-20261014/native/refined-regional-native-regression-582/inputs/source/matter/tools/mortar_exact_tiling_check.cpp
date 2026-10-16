#include "numi/matter/mortar_shared_edge_guard.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
unsigned checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

NMMortarGuardPoint2 point(float x, float y) { return {x, y}; }
NMMortarGuardTriangle2 triangle(NMMortarGuardPoint2 a,
                                NMMortarGuardPoint2 b,
                                NMMortarGuardPoint2 c) {
    return {a, b, c};
}
NMMortarGuardPoint3 point3(float x, float y, float z) { return {x, y, z}; }

NMMortarGuardPoint3 mapToParent(
    const NMMortarGuardPoint2 pointValue,
    const NMMortarGuardPoint3 parent[3]
) {
    const float weight0 = pointValue.x;
    const float weight1 = pointValue.y;
    const float weight2 = 1.0f - weight0 - weight1;
    return point3(
        weight0 * parent[0].x + weight1 * parent[1].x + weight2 * parent[2].x,
        weight0 * parent[0].y + weight1 * parent[1].y + weight2 * parent[2].y,
        weight0 * parent[0].z + weight1 * parent[1].z + weight2 * parent[2].z);
}

NMMortarGuardPoint2 permuteBarycentricPoint(
    const NMMortarGuardPoint2 pointValue,
    const unsigned int rightIndexForLeft[3]
) {
    const float weights[3] = {
        pointValue.x, pointValue.y, 1.0f - pointValue.x - pointValue.y};
    float rightWeights[3] = {};
    for (unsigned int leftIndex = 0u; leftIndex < 3u; ++leftIndex)
        rightWeights[rightIndexForLeft[leftIndex]] = weights[leftIndex];
    return point(rightWeights[0], rightWeights[1]);
}

NMMortarGuardTriangle2 permuteCell(
    const NMMortarGuardTriangle2 cell,
    const unsigned int rightIndexForLeft[3]
) {
    return triangle(
        permuteBarycentricPoint(cell.p0, rightIndexForLeft),
        permuteBarycentricPoint(cell.p1, rightIndexForLeft),
        permuteBarycentricPoint(cell.p2, rightIndexForLeft));
}

NMMortarGuardTriangleSweep makePatchSweep(
    const unsigned int patchID,
    const NMMortarGuardTriangle2 cell,
    const NMMortarGuardPoint3 parentStart[3],
    const NMMortarGuardPoint3 parentFinish[3]
) {
    const NMMortarGuardPoint2 cellPoints[3] = {
        cell.p0, cell.p1, cell.p2};
    NMMortarGuardTriangleSweep sweep{};
    sweep.patchID = patchID;
    for (unsigned int vertex = 0u; vertex < 3u; ++vertex) {
        sweep.start[vertex] = mapToParent(cellPoints[vertex], parentStart);
        sweep.finish[vertex] = mapToParent(cellPoints[vertex], parentFinish);
    }
    return sweep;
}

void makePatchSweeps(
    const NMMortarGuardTriangle2 cells[4],
    const unsigned int patchIDs[4],
    const unsigned int count,
    const NMMortarGuardPoint3 parentStart[3],
    const NMMortarGuardPoint3 parentFinish[3],
    NMMortarGuardTriangleSweep output[4]
) {
    for (unsigned int index = 0u; index < count; ++index)
        output[index] = makePatchSweep(
            patchIDs[index], cells[index], parentStart, parentFinish);
}

constexpr float q = 1.0f / 3.0f;
const NMMortarGuardPoint2 e0 = point(1.0f, 0.0f);
const NMMortarGuardPoint2 e1 = point(0.0f, 1.0f);
const NMMortarGuardPoint2 e2 = point(0.0f, 0.0f);
const NMMortarGuardPoint2 center = point(q, q);

void testExactFanAndTiles() {
    const auto fan0 = triangle(e0, e1, center);
    const auto fan1 = triangle(e1, e2, center);
    const auto fan2 = triangle(e2, e0, center);
    const auto zero = triangle(e2, e2, e2);
    require(nmMortarGuardExactTilingOfParent(
                fan0, fan1, fan2, zero, 3u),
            "exact three-sector fan was not certified");

    const auto m01 = point(0.5f, 0.5f);
    const auto m12 = point(0.0f, 0.5f);
    const auto m20 = point(0.5f, 0.0f);
    const auto tile0 = triangle(e0, m01, m20);
    const auto tile1 = triangle(m01, e1, m12);
    const auto tile2 = triangle(m20, m12, e2);
    const auto tile3 = triangle(m01, m12, m20);
    require(nmMortarGuardExactTilingOfParent(
                tile0, tile1, tile2, tile3, 4u),
            "exact four-tile partition was not certified");
    require(!nmMortarGuardExactTilingOfParent(
                fan0, fan1, zero, zero, 2u),
            "incomplete parent cover was certified");
    require(!nmMortarGuardExactTilingOfParent(
                tile0, tile1, tile1, tile3, 4u),
            "positive-area overlap was certified as a tiling");
    const auto sameSide = triangle(e0, m01, center);
    require(!nmMortarGuardExactTilingOfParent(
                tile0, sameSide, tile2, tile3, 4u),
            "same-side triangles were accepted as an oriented tiling");
    const auto reversedFan1 = triangle(center, e2, e1);
    require(nmMortarGuardExactTilingOfParent(
                fan0, reversedFan1, fan2, zero, 3u),
            "orientation normalization rejected a complete cover");
    const auto shiftedCenter = point(q, std::nextafter(q, 1.0f));
    const auto mismatchedFan2 = triangle(e2, e0, shiftedCenter);
    require(!nmMortarGuardExactTilingOfParent(
                fan0, fan1, mismatchedFan2, zero, 3u),
            "mismatched fan vertex was rounded into a closed cover");
    const auto nonQ25 = triangle(e0, e1, point(0.1f, 0.2f));
    require(!nmMortarGuardExactTilingOfParent(
                fan0, fan1, nonQ25, zero, 3u),
            "non-Q25 map was rounded into an exact tiling");
}

void testExactPairedFeatures() {
    const auto centerQ = point(q, q);
    const auto edgeA0 = point(0.0f, 0.0f);
    const auto edgeA1 = point(q, q);
    const auto edgeB0 = point(1.0f, 0.0f);
    const auto edgeB1 = point(q, q);
    require(nmMortarGuardSameSegmentParameterQ25(
                centerQ, edgeA0, edgeA1, centerQ, edgeB0, edgeB1),
            "exact shared endpoint was not recognized");
    const float oneULP = std::nextafter(0.5f, 1.0f);
    require(!nmMortarGuardSameSegmentParameterQ25(
                point(0.5f, 0.5f), point(0.0f, 0.0f), point(0.5f, 0.5f),
                point(oneULP, 0.5f), point(0.0f, 0.0f), point(0.5f, 0.5f)),
            "one-ULP mismatched edge witness was treated as paired");
    require(!nmMortarGuardSameSegmentParameterQ25(
                point(0.25f, std::nextafter(0.25f, 1.0f)),
                point(0.0f, 0.0f), point(0.5f, 0.5f),
                point(0.25f, 0.25f),
                point(0.0f, 0.0f), point(0.5f, 0.5f)),
            "near-edge off-line witness was treated as paired");

    // The production witness gate requires exact Q25 edge membership and an
    // exact same segment parameter.  A 4 nm in-plane shift on a 4 mm edge is
    // a 1e-6 parameter mismatch, which the former 8e-6 tolerance accepted.
    const auto fullEdge0 = point(0.0f, 0.0f);
    const auto fullEdge1 = point(1.0f, 0.0f);
    require(nmMortarGuardSameSegmentParameterExactOrQ25(
                point(0.5f, 0.0f), fullEdge0, fullEdge1,
                point(0.5f, 0.0f), fullEdge0, fullEdge1),
            "exact same-edge Q25 witness was rejected");
    require(!nmMortarGuardSameSegmentParameterExactOrQ25(
                point(0.5f, 0.0f), fullEdge0, fullEdge1,
                point(0.500001f, 0.0f), fullEdge0, fullEdge1),
            "4 nm off-edge Q25 witness was treated as paired");
    require(nmMortarGuardSameSegmentParameterExactOrQ25(
                edgeA1, edgeA0, edgeA1,
                edgeB1, edgeB0, edgeB1),
            "exact corresponding non-Q25 endpoint was rejected");
    require(!nmMortarGuardSameSegmentParameterExactOrQ25(
                fullEdge0, fullEdge0, fullEdge1,
                point(0.000001f, 0.0f), fullEdge0, fullEdge1),
            "4 nm one-vertex offset was treated as a paired endpoint");

    const float worldOffset = 4.0e-9f;
    require(worldOffset != 0.0f,
            "4 nm authored world offset vanished in FP32");
    const NMMortarGuardPoint3 exactLeftStart = {0.0f, 0.0f, 0.0f};
    const NMMortarGuardPoint3 exactLeftEnd = {0.004f, 0.0f, 0.0f};
    const NMMortarGuardPoint3 exactRightStart = {0.0f, 0.0f, 0.0f};
    const NMMortarGuardPoint3 exactRightEnd = {0.004f, 0.0f, 0.0f};
    require(nmMortarGuardSamePairedWorldSegment(
                exactLeftStart, exactLeftEnd,
                exactRightStart, exactRightEnd),
            "exact coincident physical seam was rejected");
    const NMMortarGuardPoint3 offsetRightStart = {
        worldOffset, 0.0f, 0.0f};
    require(!nmMortarGuardSamePairedWorldSegment(
                exactLeftStart, exactLeftEnd,
                offsetRightStart, exactRightEnd),
            "4 nm one-endpoint physical gap was treated as a paired seam");
    const NMMortarGuardPoint3 offsetRightEnd = {
        exactRightEnd.x, worldOffset, exactRightEnd.z};
    require(!nmMortarGuardSamePairedWorldSegment(
                exactLeftStart, exactLeftEnd,
                exactRightStart, offsetRightEnd),
            "4 nm other-endpoint physical gap was treated as a paired seam");

    const float infinity = std::numeric_limits<float>::infinity();
    const float notANumber = std::numeric_limits<float>::quiet_NaN();
    const NMMortarGuardPoint2 infinitePoint = point(infinity, 0.0f);
    const NMMortarGuardPoint2 infiniteStart = point(infinity, 0.0f);
    const NMMortarGuardPoint2 infiniteEnd = point(infinity, 1.0f);
    require(!nmMortarGuardSameSegmentParameterExactOrQ25(
                infinitePoint, infiniteStart, infiniteEnd,
                infinitePoint, infiniteStart, infiniteEnd),
            "infinite exact edge endpoint passed paired-parameter gate");
    const NMMortarGuardPoint3 infiniteWorld = {
        infinity, 0.0f, 0.0f};
    require(!nmMortarGuardSamePairedWorldSegment(
                infiniteWorld, infiniteWorld,
                infiniteWorld, infiniteWorld),
            "infinite paired world segment passed equality gate");
    require(!nmMortarGuardExactPairedVertex(
                infiniteWorld, infiniteWorld,
                infiniteWorld, infiniteWorld),
            "infinite paired mapped vertex passed equality gate");
    const NMMortarGuardPoint3 nanWorld = {
        notANumber, 0.0f, 0.0f};
    require(!nmMortarGuardSamePairedWorldSegment(
                nanWorld, exactLeftEnd, nanWorld, exactRightEnd),
            "NaN paired world endpoint passed equality gate");
}
void testWholeParentSweepCertificate() {
    const NMMortarGuardPoint3 leftStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, -1.0f, 0.0f)};
    const NMMortarGuardPoint3 leftFinish[3] = {
        point3(0.25f, 0.5f, 0.0f), point3(1.25f, 0.5f, 0.0f),
        point3(0.25f, -0.5f, 0.0f)};
    const NMMortarGuardPoint3 rightStart[3] = {
        leftStart[2], leftStart[0], leftStart[1]};
    const NMMortarGuardPoint3 rightFinish[3] = {
        leftFinish[2], leftFinish[0], leftFinish[1]};
    require(nmMortarGuardSameTriangleSweep(
                leftStart, leftFinish, rightStart, rightFinish),
            "permuted identical parent sweeps were not certified");
    require(nmMortarGuardExactDifferenceEqual(
                1.0f, 0.0f, 1.0f, 0.0f),
            "identical FP32 translations did not compare equal");
    require(!nmMortarGuardExactDifferenceEqual(
                1.0f, 0.0f, 1.0f, 0x1p-30f),
            "rounded subtraction hid a nonzero translation difference");
    require(!nmMortarGuardExactDifferenceEqual(
                1.0f, 0.0f, 1.0f,
                std::numeric_limits<float>::denorm_min()),
            "subnormal endpoint was admitted to exact translation proof");
    require(!nmMortarGuardExactDifferenceEqual(
                1.0e-21f, 0.0f, 1.0e-21f, 0.0f),
            "sub-1e-20 endpoint was admitted to exact translation proof");
    require(!nmMortarGuardExactDifferenceEqual(
                1.0e21f, 0.0f, 1.0e21f, 0.0f),
            "over-1e20 endpoint was admitted to exact translation proof");
    const auto full0 = triangle(point(1.0f, 0.0f), point(0.0f, 1.0f),
                                point(1.0f / 3.0f, 1.0f / 3.0f));
    const auto full1 = triangle(point(0.0f, 1.0f), point(0.0f, 0.0f),
                                point(1.0f / 3.0f, 1.0f / 3.0f));
    const auto full2 = triangle(point(0.0f, 0.0f), point(1.0f, 0.0f),
                                point(1.0f / 3.0f, 1.0f / 3.0f));
    const auto noTriangle = triangle(point(0.0f, 0.0f), point(0.0f, 0.0f),
                                     point(0.0f, 0.0f));
    const bool bothSidesFullyCovered =
        nmMortarGuardExactTilingOfParent(full0, full1, full2, noTriangle, 3u) &&
        nmMortarGuardExactTilingOfParent(full2, full0, full1, noTriangle, 3u);
    require(bothSidesFullyCovered && nmMortarGuardSameTriangleSweep(
                leftStart, leftFinish, rightStart, rightFinish),
            "whole bonded-pair skip certificate rejected valid exact case");
    const bool incompleteOtherSide =
        nmMortarGuardExactTilingOfParent(full0, full1, noTriangle, noTriangle, 2u);
    require(!incompleteOtherSide || !nmMortarGuardSameTriangleSweep(
                leftStart, leftFinish, rightStart, rightFinish),
            "whole bonded-pair gate accepted incomplete coverage");

    auto displacedFinish = rightFinish[1];
    displacedFinish.x = std::nextafter(displacedFinish.x, 1.0f);
    const NMMortarGuardPoint3 oneULPFinish[3] = {
        rightFinish[0], displacedFinish, rightFinish[2]};
    require(!nmMortarGuardSameTriangleSweep(
                leftStart, leftFinish, rightStart, oneULPFinish),
            "one-ULP endpoint mismatch passed whole-sweep identity");
    const NMMortarGuardPoint3 degenerate[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(2.0f, 0.0f, 0.0f)};
    require(!nmMortarGuardSameTriangleSweep(
                degenerate, degenerate, degenerate, degenerate),
            "degenerate parent triangle passed sweep identity");

    const NMMortarGuardPoint3 collapseStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 collapseFinish[3] = {
        collapseStart[0], collapseStart[2], collapseStart[1]};
    require(nmMortarGuardNondegenerateTriangle3(
                collapseStart[0], collapseStart[1], collapseStart[2]) &&
                nmMortarGuardNondegenerateTriangle3(
                    collapseFinish[0], collapseFinish[1], collapseFinish[2]),
            "mid-sweep collapse control endpoints are not nondegenerate");
    const NMMortarGuardPoint3 collapseMidpoint[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(0.5f, 0.5f, 0.0f),
        point3(0.5f, 0.5f, 0.0f)};
    require(!nmMortarGuardNondegenerateTriangle3(
                collapseMidpoint[0], collapseMidpoint[1],
                collapseMidpoint[2]),
            "adversarial triangle did not collapse at mid-sweep");
    require(!nmMortarGuardSameTriangleSweep(
                collapseStart, collapseFinish,
                collapseStart, collapseFinish),
            "endpoint-nondegenerate mid-sweep collapse passed guard");

    // Adversarial vertex/triangle and edge/edge path. At t=0 the two
    // triangles meet only at the paired vertex (0,0,0). At t=.2 the moving
    // triangle's first vertex is strictly inside the stationary triangle and
    // its first edge crosses the stationary top edge away from the seam.
    // A point-only initial-witness exemption can exit before seeing those
    // different barycentric witnesses; a whole-parent certificate rejects it.
    const NMMortarGuardPoint3 adversarialStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(0.0f, 1.0f, 0.0f),
        point3(-1.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 adversarialFinish[3] = {
        point3(0.5f, -1.0f, 0.0f), point3(-0.5f, 1.0f, 0.0f),
        point3(-1.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 leftStationaryFinish[3] = {
        leftStart[0], leftStart[1], leftStart[2]};
    require(!nmMortarGuardSameTriangleSweep(
                leftStart, leftStationaryFinish,
                adversarialStart, adversarialFinish),
            "deforming paired seam passed whole-parent trajectory guard");
    const float t = 0.2f;
    const float qx = (1.0f - t) * adversarialStart[0].x +
        t * adversarialFinish[0].x;
    const float qy = (1.0f - t) * adversarialStart[0].y +
        t * adversarialFinish[0].y;
    require(qx > 0.0f && qy < 0.0f && qx - qy < 1.0f,
            "adversarial moving vertex is not inside stationary triangle");
    const float rx = (1.0f - t) * adversarialStart[1].x +
        t * adversarialFinish[1].x;
    const float ry = (1.0f - t) * adversarialStart[1].y +
        t * adversarialFinish[1].y;
    const float edgeParameter = -qy / (ry - qy);
    const float crossingX = qx + edgeParameter * (rx - qx);
    require(edgeParameter > 0.0f && edgeParameter < 1.0f &&
                crossingX > 0.0f && crossingX < 1.0f,
            "adversarial moving edge does not cross away from paired vertex");
}

void testPatchIdentityAcrossParentPermutation() {
    const auto m01 = point(0.5f, 0.5f);
    const auto m12 = point(0.0f, 0.5f);
    const auto m20 = point(0.5f, 0.0f);
    const NMMortarGuardTriangle2 leftCells[4] = {
        triangle(e0, m01, m20), triangle(m01, e1, m12),
        triangle(m20, m12, e2), triangle(m01, m12, m20)};
    const unsigned int leftPatchIDs[4] = {20u, 21u, 22u, 23u};
    const unsigned int samePatchIDs[4] = {20u, 21u, 22u, 23u};
    const NMMortarGuardPoint3 leftStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 leftFinish[3] = {
        point3(0.25f, -0.5f, 0.0f), point3(1.25f, -0.5f, 0.0f),
        point3(0.25f, 0.5f, 0.0f)};
    const unsigned int rotatedParentMap[3] = {2u, 0u, 1u};
    NMMortarGuardPoint3 rightStart[3] = {};
    NMMortarGuardPoint3 rightFinish[3] = {};
    for (unsigned int leftIndex = 0u; leftIndex < 3u; ++leftIndex) {
        rightStart[rotatedParentMap[leftIndex]] = leftStart[leftIndex];
        rightFinish[rotatedParentMap[leftIndex]] = leftFinish[leftIndex];
    }
    NMMortarGuardTriangleSweep leftSweeps[4] = {};
    NMMortarGuardTriangleSweep rightSweeps[4] = {};
    makePatchSweeps(leftCells, leftPatchIDs, 4u, leftStart, leftFinish,
                    leftSweeps);
    NMMortarGuardTriangle2 rotatedCells[4] = {};
    for (unsigned int index = 0u; index < 4u; ++index)
        rotatedCells[index] = permuteCell(leftCells[index], rotatedParentMap);
    makePatchSweeps(rotatedCells, samePatchIDs, 4u, rightStart, rightFinish,
                    rightSweeps);
    require(nmMortarGuardSamePatchTriangleSweeps(
                leftSweeps, 4u, rightSweeps, 4u),
            "identical per-patch partitions were not paired");

    const unsigned int swappedPatchIDs[4] = {21u, 20u, 22u, 23u};
    NMMortarGuardTriangleSweep swappedIDRight[4] = {};
    makePatchSweeps(rotatedCells, swappedPatchIDs, 4u, rightStart, rightFinish,
                    swappedIDRight);
    require(!nmMortarGuardSamePatchTriangleSweeps(
                leftSweeps, 4u, swappedIDRight, 4u),
            "cross-wired patch identities passed the skip proof");

    NMMortarGuardTriangle2 crossWiredCells[4] = {
        rotatedCells[1], rotatedCells[0], rotatedCells[2], rotatedCells[3]};
    NMMortarGuardTriangleSweep crossWiredRight[4] = {};
    makePatchSweeps(crossWiredCells, samePatchIDs, 4u, rightStart, rightFinish,
                    crossWiredRight);
    require(!nmMortarGuardSamePatchTriangleSweeps(
                leftSweeps, 4u, crossWiredRight, 4u),
            "same IDs over different physical cells passed the skip proof");

    const NMMortarGuardPoint3 changedFinish[3] = {
        rightFinish[0], point3(std::nextafter(rightFinish[1].x, 2.0f),
                               rightFinish[1].y, rightFinish[1].z),
        rightFinish[2]};
    NMMortarGuardTriangleSweep oneULPRight[4] = {};
    makePatchSweeps(rotatedCells, samePatchIDs, 4u, rightStart, changedFinish,
                    oneULPRight);
    require(!nmMortarGuardSamePatchTriangleSweeps(
                leftSweeps, 4u, oneULPRight, 4u),
            "one-ULP patch endpoint mismatch passed the skip proof");

    const unsigned int invalidIDs[4] = {20u, 21u, 22u, 0xffffffffu};
    NMMortarGuardTriangleSweep invalidRight[4] = {};
    makePatchSweeps(rotatedCells, invalidIDs, 4u, rightStart, rightFinish,
                    invalidRight);
    require(!nmMortarGuardSamePatchTriangleSweeps(
                leftSweeps, 4u, invalidRight, 4u),
            "invalid cohesive patch identity passed the skip proof");
}

void testAxisAlignedNormalRelativeParentSweep() {
    const NMMortarGuardPoint3 leftStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 leftFinish[3] = {
        point3(0.25f, 0.5f, 0.125f), point3(1.25f, 0.5f, 0.125f),
        point3(0.25f, 1.5f, 0.125f)};
    const NMMortarGuardPoint3 rightStart[3] = {
        point3(0.0f, 0.0f, 0.01f), point3(1.0f, 0.0f, 0.01f),
        point3(0.0f, 1.0f, 0.01f)};
    const NMMortarGuardPoint3 rightFinish[3] = {
        point3(0.25f, 0.5f, 0.125f), point3(1.25f, 0.5f, 0.125f),
        point3(0.25f, 1.5f, 0.125f)};
    unsigned int permutation[3] = {0u, 0u, 0u};
    require(nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, rightStart, rightFinish, permutation),
            "parallel parents with exact common projected motion were rejected");
    const NMMortarGuardPoint3 openingFinish[3] = {
        point3(0.25f, 0.5f, 0.25f), point3(1.25f, 0.5f, 0.25f),
        point3(0.25f, 1.5f, 0.25f)};
    require(nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, rightStart, openingFinish, permutation),
            "parallel parents with exact normal opening were rejected");

    constexpr float smallEdge = 0x1p-10f;
    constexpr float lateralOffset = 0x1p-28f;
    const NMMortarGuardPoint3 smallLeftStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(smallEdge, 0.0f, 0.0f),
        point3(0.0f, smallEdge, 0.0f)};
    const NMMortarGuardPoint3 smallLeftFinish[3] = {
        point3(0x1p-20f, 0x1p-20f, 0x1p-18f),
        point3(smallEdge + 0x1p-20f, 0x1p-20f, 0x1p-18f),
        point3(0x1p-20f, smallEdge + 0x1p-20f, 0x1p-18f)};
    NMMortarGuardPoint3 smallRightStart[3] = {
        smallLeftStart[0], smallLeftStart[1], smallLeftStart[2]};
    NMMortarGuardPoint3 smallRightFinish[3] = {
        smallLeftFinish[0], smallLeftFinish[1], smallLeftFinish[2]};
    for (unsigned int index = 0u; index < 3u; ++index) {
        smallRightStart[index].x += lateralOffset;
        smallRightFinish[index].x += lateralOffset;
    }
    require(nmMortarGuardTriangleHasExactCommonTranslation(
                smallRightStart, smallRightFinish),
            "exact 3.725 nm lateral translation fixture is not uniform");
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                smallLeftStart, smallLeftFinish,
                smallRightStart, smallRightFinish, permutation),
            "exact 3.725 nm lateral offset passed normal-only certificate");

    const NMMortarGuardTriangle2 fan[4] = {
        triangle(point(1.0f, 0.0f), point(0.5f, 0.5f), point(0.5f, 0.0f)),
        triangle(point(0.5f, 0.5f), point(0.0f, 1.0f), point(0.0f, 0.5f)),
        triangle(point(0.5f, 0.0f), point(0.0f, 0.5f), point(0.0f, 0.0f)),
        triangle(point(0.5f, 0.5f), point(0.0f, 0.5f), point(0.5f, 0.0f))};
    const unsigned int ids[4] = {10u, 11u, 12u, 13u};
    NMMortarGuardTriangleSweep leftCells[4] = {};
    NMMortarGuardTriangleSweep rightCells[4] = {};
    require(nmMortarGuardExactTilingOfParent(
                fan[0], fan[1], fan[2], fan[3], 4u),
            "dyadic four-cell partition is not an exact parent cover");
    makePatchSweeps(fan, ids, 4u, leftStart, leftFinish, leftCells);
    makePatchSweeps(fan, ids, 4u, rightStart, rightFinish, rightCells);
    require(nmMortarGuardSameProjectedPatchTriangleSweeps(
                leftCells, 4u, rightCells, 4u),
            "same-ID projected cohesive cell sweeps were rejected");

    const NMMortarGuardTriangle2 fan3[4] = {
        triangle(point(1.0f, 0.0f), point(0.0f, 1.0f), point(q, q)),
        triangle(point(0.0f, 1.0f), point(0.0f, 0.0f), point(q, q)),
        triangle(point(0.0f, 0.0f), point(1.0f, 0.0f), point(q, q)),
        triangle(point(0.0f, 0.0f), point(0.0f, 0.0f), point(0.0f, 0.0f))};
    const NMMortarGuardPoint3 stationaryXYFinish[3] = {
        point3(0.0f, 0.0f, 0.125f), point3(1.0f, 0.0f, 0.125f),
        point3(0.0f, 1.0f, 0.125f)};
    const NMMortarGuardPoint3 rightOpeningFinish[3] = {
        point3(0.0f, 0.0f, 0.25f), point3(1.0f, 0.0f, 0.25f),
        point3(0.0f, 1.0f, 0.25f)};
    const unsigned int fan3IDs[4] = {30u, 31u, 32u, 0xffffffffu};
    NMMortarGuardTriangleSweep fan3Left[4] = {};
    NMMortarGuardTriangleSweep fan3Right[4] = {};
    makePatchSweeps(fan3, fan3IDs, 3u, leftStart,
                    stationaryXYFinish, fan3Left);
    makePatchSweeps(fan3, fan3IDs, 3u, rightStart,
                    rightOpeningFinish, fan3Right);
    require(nmMortarGuardSameProjectedPatchTriangleSweeps(
                fan3Left, 3u, fan3Right, 3u),
            "three-patch common-vertex fan projected sweeps were rejected");

    // A normal-only sweep may pass through zero gap and end with the two
    // planes interpenetrating. The geometric skip is still valid only because
    // the complete projected parent and patch tilings remain identical; the
    // active cohesive law owns normal tension and compression over that full
    // interface. This fixture verifies the full endpoint trajectory contract,
    // not a material-force or native contact qualification.
    const NMMortarGuardPoint3 crossingLeftStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 crossingLeftFinish[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 crossingRightStart[3] = {
        point3(0.0f, 0.0f, 0.125f), point3(1.0f, 0.0f, 0.125f),
        point3(0.0f, 1.0f, 0.125f)};
    const NMMortarGuardPoint3 crossingRightFinish[3] = {
        point3(0.0f, 0.0f, -0.125f), point3(1.0f, 0.0f, -0.125f),
        point3(0.0f, 1.0f, -0.125f)};
    const unsigned int crossingIDs[4] = {60u, 61u, 62u, 0xffffffffu};
    NMMortarGuardTriangleSweep crossingLeftCells[4] = {};
    NMMortarGuardTriangleSweep crossingRightCells[4] = {};
    makePatchSweeps(fan3, crossingIDs, 3u, crossingLeftStart,
                    crossingLeftFinish, crossingLeftCells);
    makePatchSweeps(fan3, crossingIDs, 3u, crossingRightStart,
                    crossingRightFinish, crossingRightCells);
    unsigned int crossingPermutation[3] = {0u, 0u, 0u};
    require(crossingRightStart[0].z > crossingLeftStart[0].z &&
                crossingRightFinish[0].z < crossingLeftFinish[0].z &&
                nmMortarGuardSameAxisAlignedNormalParentSweep(
                    crossingLeftStart, crossingLeftFinish,
                    crossingRightStart, crossingRightFinish,
                    crossingPermutation) &&
                nmMortarGuardSameProjectedPatchTriangleSweeps(
                    crossingLeftCells, 3u, crossingRightCells, 3u),
            "fully covered normal closure/crossing lost whole-sweep certificate");

    const NMMortarGuardPoint2 midpoint = point(0.5f, 0.5f);
    const NMMortarGuardTriangle2 split2[4] = {
        triangle(point(1.0f, 0.0f), midpoint, point(0.0f, 0.0f)),
        triangle(midpoint, point(0.0f, 1.0f), point(0.0f, 0.0f)),
        fan3[3], fan3[3]};
    const unsigned int split2IDs[4] = {40u, 41u, 0xffffffffu, 0xffffffffu};
    NMMortarGuardTriangleSweep split2Left[4] = {};
    NMMortarGuardTriangleSweep split2Right[4] = {};
    makePatchSweeps(split2, split2IDs, 2u, leftStart,
                    stationaryXYFinish, split2Left);
    makePatchSweeps(split2, split2IDs, 2u, rightStart,
                    rightOpeningFinish, split2Right);
    require(nmMortarGuardExactTilingOfParent(
                split2[0], split2[1], split2[2], split2[3], 2u) &&
                nmMortarGuardSameProjectedPatchTriangleSweeps(
                    split2Left, 2u, split2Right, 2u),
            "two-patch diagonal partition was not certified");

    NMMortarGuardPoint3 lateralStart[3] = {
        rightStart[0], rightStart[1], rightStart[2]};
    NMMortarGuardPoint3 lateralFinish[3] = {
        rightFinish[0], rightFinish[1], rightFinish[2]};
    for (unsigned int vertex = 0u; vertex < 3u; ++vertex) {
        lateralStart[vertex].x += 4.0e-9f;
        lateralFinish[vertex].x += 4.0e-9f;
    }
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, lateralStart, lateralFinish, permutation),
            "4 nm lateral offset passed normal-only parent certificate");

    NMMortarGuardPoint3 shearedFinish[3] = {
        rightFinish[0], rightFinish[1], rightFinish[2]};
    shearedFinish[1].x += 0.125f;
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, rightStart, shearedFinish, permutation),
            "projected shear passed normal-only parent certificate");

    NMMortarGuardPoint3 nonplanarStart[3] = {
        rightStart[0], rightStart[1], rightStart[2]};
    NMMortarGuardPoint3 nonplanarFinish[3] = {
        rightFinish[0], rightFinish[1], rightFinish[2]};
    nonplanarStart[2].z += 0.25f;
    nonplanarFinish[2].z += 0.25f;
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, nonplanarStart, nonplanarFinish,
                permutation),
            "non-horizontal parent passed axis-aligned proof");

    const NMMortarGuardPoint3 collapsingFinish[3] = {
        rightFinish[0], rightFinish[2], rightFinish[1]};
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, rightStart, collapsingFinish,
                permutation),
            "mid-sweep projected collapse passed normal-only proof");
    const NMMortarGuardPoint3 commonCollapseStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 commonCollapseFinish[3] = {
        point3(0.25f, 0.5f, 0.125f), point3(0.25f, 1.5f, 0.125f),
        point3(1.25f, 0.5f, 0.125f)};
    const NMMortarGuardPoint3 commonCollapseRightStart[3] = {
        commonCollapseStart[0], commonCollapseStart[1], commonCollapseStart[2]};
    const NMMortarGuardPoint3 commonCollapseRightFinish[3] = {
        commonCollapseFinish[0], commonCollapseFinish[1], commonCollapseFinish[2]};
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                commonCollapseStart, commonCollapseFinish,
                commonCollapseRightStart, commonCollapseRightFinish, permutation),
            "identical parent paths that collapse at mid-sweep were certified");

    const unsigned int crossWiredIDs[4] = {11u, 10u, 12u, 13u};
    NMMortarGuardTriangleSweep crossWiredCells[4] = {};
    makePatchSweeps(fan, crossWiredIDs, 4u,
                    rightStart, rightFinish, crossWiredCells);
    require(!nmMortarGuardSameProjectedPatchTriangleSweeps(
                leftCells, 4u, crossWiredCells, 4u),
            "cross-wired patch IDs passed projected seam certificate");

    NMMortarGuardTriangleSweep inconsistentLeft[4] = {};
    NMMortarGuardTriangleSweep inconsistentRight[4] = {};
    inconsistentLeft[0].patchID = 50u;
    inconsistentLeft[1].patchID = 51u;
    inconsistentRight[0].patchID = 50u;
    inconsistentRight[1].patchID = 51u;
    const NMMortarGuardPoint3 firstCellStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 firstCellFinish[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 1.0f, 0.0f)};
    const NMMortarGuardPoint3 secondCellStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(0.0f, 1.0f, 0.0f),
        point3(-1.0f, 0.0f, 0.0f)};
    const NMMortarGuardPoint3 secondCellFinish[3] = {
        point3(0.125f, 0.0f, 0.0f), point3(0.125f, 1.0f, 0.0f),
        point3(-0.875f, 0.0f, 0.0f)};
    for (unsigned int vertex = 0u; vertex < 3u; ++vertex) {
        inconsistentLeft[0].start[vertex] = firstCellStart[vertex];
        inconsistentLeft[0].finish[vertex] = firstCellFinish[vertex];
        inconsistentLeft[1].start[vertex] = secondCellStart[vertex];
        inconsistentLeft[1].finish[vertex] = secondCellFinish[vertex];
        inconsistentRight[0] = inconsistentLeft[0];
        inconsistentRight[1] = inconsistentLeft[1];
    }
    require(!nmMortarGuardSameProjectedPatchTriangleSweeps(
                inconsistentLeft, 2u, inconsistentRight, 2u),
            "shared projected start vertices with divergent finish paths were accepted");

    const NMMortarGuardPoint3 rotatedStart[3] = {
        point3(0.0f, 0.0f, 0.0f), point3(1.0f, 0.0f, 0.0f),
        point3(0.0f, 0.0f, 1.0f)};
    const NMMortarGuardPoint3 rotatedFinish[3] = {
        point3(0.25f, 0.5f, 0.0f), point3(1.25f, 0.5f, 0.0f),
        point3(0.25f, 0.5f, 1.0f)};
    require(!nmMortarGuardSameAxisAlignedNormalParentSweep(
                leftStart, leftFinish, rotatedStart, rotatedFinish,
                permutation),
            "unsupported non-horizontal relative pose passed certificate");
}

}

int main() {
    try {
        testExactFanAndTiles();
        testExactPairedFeatures();
        testWholeParentSweepCertificate();
        testPatchIdentityAcrossParentPermutation();
        testAxisAlignedNormalRelativeParentSweep();
        std::cout << "mortar_exact_tiling_checks=" << checks
                  << " exact_fan_and_tiles=true exact_feature_identity=true"
                     " whole_parent_sweep_certificate=true"
                     " patch_identity=true axis_normal_relative=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
