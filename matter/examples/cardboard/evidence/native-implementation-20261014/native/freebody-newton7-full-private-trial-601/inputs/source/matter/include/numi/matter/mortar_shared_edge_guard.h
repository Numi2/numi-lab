#ifndef NUMI_MATTER_MORTAR_SHARED_EDGE_GUARD_H
#define NUMI_MATTER_MORTAR_SHARED_EDGE_GUARD_H

#ifdef __METAL_VERSION__
#define NM_MORTAR_THREAD_POINTER thread
#else
#define NM_MORTAR_THREAD_POINTER
#endif

// Conservative exact-map predicates shared by the Metal contact path and its
// CPU controls. Uncertain, degenerate, or nonmatching cases return false so
// the ordinary contact/CCD path remains responsible for them.
struct NMMortarGuardPoint3 {
    float x;
    float y;
    float z;
};

struct NMMortarGuardPoint2 {
    float x;
    float y;
};

struct NMMortarGuardTriangle2 {
    NMMortarGuardPoint2 p0;
    NMMortarGuardPoint2 p1;
    NMMortarGuardPoint2 p2;
};

struct NMMortarGuardTriangleSweep {
    unsigned int patchID;
    NMMortarGuardPoint3 start[3];
    NMMortarGuardPoint3 finish[3];
};

inline float nmMortarGuardAbs(const float value) {
    return value < 0.0f ? -value : value;
}

inline bool nmMortarGuardFinite(const float value) {
    return value == value &&
        nmMortarGuardAbs(value) <= 3.402823466e+38f;
}

// The contact partition uses barycentric FP32 input maps. For the common
// rational maps used by authoring (including thirds and halves), a Q25
// integer representation is exact and leaves a signed 64-bit determinant
// budget: each coordinate is at most 2^25, so an orientation determinant is
// strictly below 2^51. Maps that do not lie exactly on this grid are not
// rounded; the fast exact-cover proof simply declines them.
inline bool nmMortarGuardBaryQ25(
    const float value, NM_MORTAR_THREAD_POINTER long* result) {
    if (!nmMortarGuardFinite(value) || value < 0.0f || value > 1.0f ||
        result == nullptr) return false;
    const float scaled = value * 33554432.0f;
    const long integer = long(scaled);
    if (float(integer) != scaled) return false;
    *result = integer;
    return true;
}

inline bool nmMortarGuardPointQ25(
    const NMMortarGuardPoint2 point,
    NM_MORTAR_THREAD_POINTER long* x,
    NM_MORTAR_THREAD_POINTER long* y
) {
    return nmMortarGuardBaryQ25(point.x, x) &&
        nmMortarGuardBaryQ25(point.y, y) &&
        *x + *y <= 33554432l;
}

inline long nmMortarGuardOrientQ25(
    const long ax, const long ay,
    const long bx, const long by,
    const long cx, const long cy
) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

inline bool nmMortarGuardPointOnSegmentQ25(
    const long px, const long py,
    const long ax, const long ay,
    const long bx, const long by,
    NM_MORTAR_THREAD_POINTER long* numerator,
    NM_MORTAR_THREAD_POINTER long* denominator
) {
    const long dx = bx - ax;
    const long dy = by - ay;
    const long lengthSquared = dx * dx + dy * dy;
    const long projection = (px - ax) * dx + (py - ay) * dy;
    if (lengthSquared <= 0 ||
        nmMortarGuardOrientQ25(ax, ay, bx, by, px, py) != 0 ||
        projection < 0 || projection > lengthSquared) return false;
    *numerator = projection;
    *denominator = lengthSquared;
    return true;
}

inline bool nmMortarGuardEqualFractionsQ25(
    long firstNumerator, long firstDenominator,
    long secondNumerator, long secondDenominator
) {
    if (firstDenominator <= 0 || secondDenominator <= 0 ||
        firstNumerator < 0 || secondNumerator < 0) return false;
    // Continued-fraction comparison avoids overflowing a 64-bit cross
    // product when barycentric edge lengths approach the Q25 bound.
    for (unsigned int iteration = 0u; iteration < 128u; ++iteration) {
        const long firstWhole = firstNumerator / firstDenominator;
        const long secondWhole = secondNumerator / secondDenominator;
        if (firstWhole != secondWhole) return false;
        const long firstRemainder = firstNumerator % firstDenominator;
        const long secondRemainder = secondNumerator % secondDenominator;
        if (firstRemainder == 0 || secondRemainder == 0)
            return firstRemainder == 0 && secondRemainder == 0;
        firstNumerator = firstDenominator;
        firstDenominator = firstRemainder;
        secondNumerator = secondDenominator;
        secondDenominator = secondRemainder;
    }
    return false;
}

inline bool nmMortarGuardSameSegmentParameterQ25(
    const NMMortarGuardPoint2 leftPoint,
    const NMMortarGuardPoint2 leftStart,
    const NMMortarGuardPoint2 leftEnd,
    const NMMortarGuardPoint2 rightPoint,
    const NMMortarGuardPoint2 rightStart,
    const NMMortarGuardPoint2 rightEnd
) {
    long px, py, ax, ay, bx, by;
    long qx, qy, cx, cy, dx, dy;
    if (!nmMortarGuardPointQ25(leftPoint, &px, &py) ||
        !nmMortarGuardPointQ25(leftStart, &ax, &ay) ||
        !nmMortarGuardPointQ25(leftEnd, &bx, &by) ||
        !nmMortarGuardPointQ25(rightPoint, &qx, &qy) ||
        !nmMortarGuardPointQ25(rightStart, &cx, &cy) ||
        !nmMortarGuardPointQ25(rightEnd, &dx, &dy)) return false;
    long leftNumerator, leftDenominator;
    long rightNumerator, rightDenominator;
    return nmMortarGuardPointOnSegmentQ25(
            px, py, ax, ay, bx, by,
            &leftNumerator, &leftDenominator) &&
        nmMortarGuardPointOnSegmentQ25(
            qx, qy, cx, cy, dx, dy,
            &rightNumerator, &rightDenominator) &&
        nmMortarGuardEqualFractionsQ25(
            leftNumerator, leftDenominator,
            rightNumerator, rightDenominator);
}

// Boundary witnesses may be one of the exact mapped endpoints even when an
// endpoint (for example 1/3) is not representable on the Q25 grid. Admit that
// case only when both sides identify the same corresponding endpoint exactly;
// interior witnesses still require exact Q25 segment and parameter equality.
inline bool nmMortarGuardSameSegmentParameterExactOrQ25(
    const NMMortarGuardPoint2 leftPoint,
    const NMMortarGuardPoint2 leftStart,
    const NMMortarGuardPoint2 leftEnd,
    const NMMortarGuardPoint2 rightPoint,
    const NMMortarGuardPoint2 rightStart,
    const NMMortarGuardPoint2 rightEnd
) {
    const NMMortarGuardPoint2 points[6] = {
        leftPoint, leftStart, leftEnd,
        rightPoint, rightStart, rightEnd};
    for (unsigned int index = 0u; index < 6u; ++index)
        if (!nmMortarGuardFinite(points[index].x) ||
            !nmMortarGuardFinite(points[index].y)) return false;
    if (leftPoint.x == leftStart.x && leftPoint.y == leftStart.y &&
        rightPoint.x == rightStart.x && rightPoint.y == rightStart.y)
        return true;
    if (leftPoint.x == leftEnd.x && leftPoint.y == leftEnd.y &&
        rightPoint.x == rightEnd.x && rightPoint.y == rightEnd.y)
        return true;
    return nmMortarGuardSameSegmentParameterQ25(
        leftPoint, leftStart, leftEnd,
        rightPoint, rightStart, rightEnd);
}

inline bool nmMortarGuardSeparatedQ25(
    const long ax, const long ay,
    const long bx, const long by,
    const long aInteriorSign,
    const long b0x, const long b0y,
    const long b1x, const long b1y,
    const long b2x, const long b2y
) {
    const long s0 = nmMortarGuardOrientQ25(ax, ay, bx, by, b0x, b0y) *
        aInteriorSign;
    const long s1 = nmMortarGuardOrientQ25(ax, ay, bx, by, b1x, b1y) *
        aInteriorSign;
    const long s2 = nmMortarGuardOrientQ25(ax, ay, bx, by, b2x, b2y) *
        aInteriorSign;
    return s0 <= 0 && s1 <= 0 && s2 <= 0;
}

inline bool nmMortarGuardNoPositiveOverlapQ25(
    NM_MORTAR_THREAD_POINTER const long* a,
    NM_MORTAR_THREAD_POINTER const long* b
) {
    const long areaA = nmMortarGuardOrientQ25(
        a[0], a[1], a[2], a[3], a[4], a[5]);
    const long areaB = nmMortarGuardOrientQ25(
        b[0], b[1], b[2], b[3], b[4], b[5]);
    if (areaA == 0 || areaB == 0) return false;
    const long signA = areaA > 0 ? 1 : -1;
    const long signB = areaB > 0 ? 1 : -1;
    for (unsigned int edge = 0; edge < 3u; ++edge) {
        const unsigned int next = (edge + 1u) % 3u;
        if (nmMortarGuardSeparatedQ25(
                a[2u * edge], a[2u * edge + 1u],
                a[2u * next], a[2u * next + 1u], signA,
                b[0], b[1], b[2], b[3], b[4], b[5]) ||
            nmMortarGuardSeparatedQ25(
                b[2u * edge], b[2u * edge + 1u],
                b[2u * next], b[2u * next + 1u], signB,
                a[0], a[1], a[2], a[3], a[4], a[5])) return true;
    }
    return false;
}

inline bool nmMortarGuardExactTilingOfParent(
    const NMMortarGuardTriangle2 triangle0,
    const NMMortarGuardTriangle2 triangle1,
    const NMMortarGuardTriangle2 triangle2,
    const NMMortarGuardTriangle2 triangle3,
    const unsigned int triangleCount
) {
    if (triangleCount < 2u || triangleCount > 4u) return false;
    const NMMortarGuardTriangle2 triangles[4] = {
        triangle0, triangle1, triangle2, triangle3};
    long coordinates[4][6] = {};
    constexpr long parentScale = 33554432l;
    long edgeCoordinates[12][4] = {};
    bool edgeMatched[12] = {};
    long boundaryStart[3][12] = {};
    long boundaryEnd[3][12] = {};
    unsigned int boundaryCount[3] = {};
    unsigned int edgeCount = 0u;
    for (unsigned int triangle = 0; triangle < triangleCount; ++triangle) {
        const NMMortarGuardPoint2 vertices[3] = {
            triangles[triangle].p0,
            triangles[triangle].p1,
            triangles[triangle].p2};
        for (unsigned int vertexIndex = 0; vertexIndex < 3u; ++vertexIndex) {
            long x, y;
            if (!nmMortarGuardBaryQ25(vertices[vertexIndex].x, &x) ||
                !nmMortarGuardBaryQ25(vertices[vertexIndex].y, &y) ||
                x + y > parentScale) return false;
            coordinates[triangle][2u * vertexIndex] = x;
            coordinates[triangle][2u * vertexIndex + 1u] = y;
        }
        long signedArea = nmMortarGuardOrientQ25(
            coordinates[triangle][0], coordinates[triangle][1],
            coordinates[triangle][2], coordinates[triangle][3],
            coordinates[triangle][4], coordinates[triangle][5]);
        if (signedArea == 0) return false;
        if (signedArea < 0) {
            const long swapX = coordinates[triangle][2];
            const long swapY = coordinates[triangle][3];
            coordinates[triangle][2] = coordinates[triangle][4];
            coordinates[triangle][3] = coordinates[triangle][5];
            coordinates[triangle][4] = swapX;
            coordinates[triangle][5] = swapY;
            signedArea = -signedArea;
        }
        for (unsigned int localEdge = 0u; localEdge < 3u; ++localEdge) {
            const unsigned int next = (localEdge + 1u) % 3u;
            edgeCoordinates[edgeCount][0] =
                coordinates[triangle][2u * localEdge];
            edgeCoordinates[edgeCount][1] =
                coordinates[triangle][2u * localEdge + 1u];
            edgeCoordinates[edgeCount][2] =
                coordinates[triangle][2u * next];
            edgeCoordinates[edgeCount][3] =
                coordinates[triangle][2u * next + 1u];
            ++edgeCount;
        }
    }
    // Every interior edge must cancel with exactly one reversed edge. The
    // remaining directed edges must tile the three parent edges without a
    // gap or overlap and retain the parent's counter-clockwise winding.
    for (unsigned int edge = 0u; edge < edgeCount; ++edge) {
        if (edgeMatched[edge]) continue;
        for (unsigned int other = edge + 1u; other < edgeCount; ++other) {
            const bool reversed =
                edgeCoordinates[edge][0] == edgeCoordinates[other][2] &&
                edgeCoordinates[edge][1] == edgeCoordinates[other][3] &&
                edgeCoordinates[edge][2] == edgeCoordinates[other][0] &&
                edgeCoordinates[edge][3] == edgeCoordinates[other][1];
            if (reversed) {
                edgeMatched[edge] = true;
                edgeMatched[other] = true;
                break;
            }
        }
        if (edgeMatched[edge]) continue;
        const long ax = edgeCoordinates[edge][0];
        const long ay = edgeCoordinates[edge][1];
        const long bx = edgeCoordinates[edge][2];
        const long by = edgeCoordinates[edge][3];
        unsigned int side = 3u;
        long intervalStart = 0;
        long intervalEnd = 0;
        if (ay == 0 && by == 0 && ax < bx) {
            side = 0u;
            intervalStart = ax;
            intervalEnd = bx;
        } else if (ax == 0 && bx == 0 && ay > by) {
            side = 1u;
            intervalStart = by;
            intervalEnd = ay;
        } else if (ax + ay == parentScale &&
                   bx + by == parentScale && ay < by) {
            side = 2u;
            intervalStart = ay;
            intervalEnd = by;
        }
        if (side >= 3u || intervalStart < 0 ||
            intervalEnd > parentScale || intervalStart >= intervalEnd ||
            boundaryCount[side] >= edgeCount) return false;
        const unsigned int slot = boundaryCount[side]++;
        boundaryStart[side][slot] = intervalStart;
        boundaryEnd[side][slot] = intervalEnd;
        edgeMatched[edge] = true;
    }
    for (unsigned int side = 0u; side < 3u; ++side) {
        const unsigned int count = boundaryCount[side];
        if (count == 0u) return false;
        // At most twelve directed edges exist; insertion sort is bounded and
        // keeps this proof deterministic without temporary dynamic storage.
        for (unsigned int i = 1u; i < count; ++i) {
            const long start = boundaryStart[side][i];
            const long end = boundaryEnd[side][i];
            unsigned int j = i;
            while (j > 0u && boundaryStart[side][j - 1u] > start) {
                boundaryStart[side][j] = boundaryStart[side][j - 1u];
                boundaryEnd[side][j] = boundaryEnd[side][j - 1u];
                --j;
            }
            boundaryStart[side][j] = start;
            boundaryEnd[side][j] = end;
        }
        if (boundaryStart[side][0] != 0) return false;
        for (unsigned int i = 1u; i < count; ++i)
            if (boundaryStart[side][i] != boundaryEnd[side][i - 1u]) return false;
        if (boundaryEnd[side][count - 1u] != parentScale)
            return false;
    }
    for (unsigned int left = 0; left < triangleCount; ++left)
        for (unsigned int right = left + 1u; right < triangleCount; ++right)
            if (!nmMortarGuardNoPositiveOverlapQ25(
                    coordinates[left], coordinates[right])) return false;
    // Directed edge cancellation plus exact boundary coverage proves the
    // oriented 2-chain is the parent. Pairwise SAT separately rules out a
    // positive-area overlap; no floating-point area sum participates.
    return true;
}

inline bool nmMortarGuardSamePoint(
    const NMMortarGuardPoint3 a,
    const NMMortarGuardPoint3 b
) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool nmMortarGuardFinitePoint3(const NMMortarGuardPoint3 point) {
    return nmMortarGuardFinite(point.x) &&
        nmMortarGuardFinite(point.y) &&
        nmMortarGuardFinite(point.z);
}

inline bool nmMortarGuardSamePairedWorldSegment(
    const NMMortarGuardPoint3 leftStart,
    const NMMortarGuardPoint3 leftEnd,
    const NMMortarGuardPoint3 rightStart,
    const NMMortarGuardPoint3 rightEnd
) {
    return nmMortarGuardFinitePoint3(leftStart) &&
        nmMortarGuardFinitePoint3(leftEnd) &&
        nmMortarGuardFinitePoint3(rightStart) &&
        nmMortarGuardFinitePoint3(rightEnd) &&
        nmMortarGuardSamePoint(leftStart, rightStart) &&
        nmMortarGuardSamePoint(leftEnd, rightEnd);
}

// A page-boundary correspondence is admitted only when the complete mapped
// 3D barycentric vertex agrees on both sides. Near matches are not proof of
// the same authored seam and must remain contact candidates.
inline bool nmMortarGuardExactPairedVertex(
    const NMMortarGuardPoint3 leftA,
    const NMMortarGuardPoint3 rightA,
    const NMMortarGuardPoint3 leftB,
    const NMMortarGuardPoint3 rightB
) {
    return nmMortarGuardFinitePoint3(leftA) &&
        nmMortarGuardFinitePoint3(rightA) &&
        nmMortarGuardFinitePoint3(leftB) &&
        nmMortarGuardFinitePoint3(rightB) &&
        nmMortarGuardSamePoint(leftA, leftB) &&
        nmMortarGuardSamePoint(rightA, rightB);
}

inline float nmMortarGuardOrientation(
    const NMMortarGuardPoint3 a,
    const NMMortarGuardPoint3 b,
    const NMMortarGuardPoint3 c
) {
    const float edgeX = b.x - a.x;
    const float edgeY = b.y - a.y;
    const float pointX = c.x - a.x;
    const float pointY = c.y - a.y;
    return edgeX * pointY - edgeY * pointX;
}

inline float nmMortarGuardOrientationErrorBound(
    const NMMortarGuardPoint3 a,
    const NMMortarGuardPoint3 b,
    const NMMortarGuardPoint3 c
) {
    // This bounds the rounded coordinate differences, two products and final
    // subtraction with ample binary32 margin. If a determinant is not outside
    // this bound, its sign is not used to suppress a partition rejection.
    constexpr float epsilon = 1.1920928955078125e-7f;
    const float xScale = (nmMortarGuardAbs(a.x) + nmMortarGuardAbs(b.x)) *
        (nmMortarGuardAbs(a.y) + nmMortarGuardAbs(c.y));
    const float yScale = (nmMortarGuardAbs(a.y) + nmMortarGuardAbs(b.y)) *
        (nmMortarGuardAbs(a.x) + nmMortarGuardAbs(c.x));
    return 32.0f * epsilon * (xScale + yScale);
}

inline bool nmMortarGuardStrictOppositeSides(
    const NMMortarGuardPoint3 edgeA,
    const NMMortarGuardPoint3 edgeB,
    const NMMortarGuardPoint3 pointA,
    const NMMortarGuardPoint3 pointB
) {
    const float sideA = nmMortarGuardOrientation(edgeA, edgeB, pointA);
    const float sideB = nmMortarGuardOrientation(edgeA, edgeB, pointB);
    const float boundA = nmMortarGuardOrientationErrorBound(
        edgeA, edgeB, pointA);
    const float boundB = nmMortarGuardOrientationErrorBound(
        edgeA, edgeB, pointB);
    if (!nmMortarGuardFinite(sideA) || !nmMortarGuardFinite(sideB) ||
        !nmMortarGuardFinite(boundA) || !nmMortarGuardFinite(boundB))
        return false;
    return (sideA > boundA && sideB < -boundB) ||
        (sideA < -boundA && sideB > boundB);
}

inline bool nmMortarExactOppositeSharedEdge(
    const NMMortarGuardPoint3 a0,
    const NMMortarGuardPoint3 a1,
    const NMMortarGuardPoint3 a2,
    const NMMortarGuardPoint3 b0,
    const NMMortarGuardPoint3 b1,
    const NMMortarGuardPoint3 b2
) {
    const NMMortarGuardPoint3 triangleA[3] = {a0, a1, a2};
    const NMMortarGuardPoint3 triangleB[3] = {b0, b1, b2};
    unsigned int matchedA[2] = {0u, 0u};
    bool usedB[3] = {false, false, false};
    unsigned int matchCount = 0u;
    for (unsigned int a = 0u; a < 3u; ++a) {
        for (unsigned int b = 0u; b < 3u; ++b) {
            if (!usedB[b] && nmMortarGuardSamePoint(triangleA[a], triangleB[b])) {
                if (matchCount >= 2u) return false;
                matchedA[matchCount] = a;
                usedB[b] = true;
                ++matchCount;
                break;
            }
        }
    }
    if (matchCount != 2u ||
        (triangleA[matchedA[0]].x == triangleA[matchedA[1]].x &&
         triangleA[matchedA[0]].y == triangleA[matchedA[1]].y))
        return false;

    unsigned int thirdA = 3u, thirdB = 3u;
    for (unsigned int i = 0u; i < 3u; ++i) {
        if (i != matchedA[0] && i != matchedA[1]) thirdA = i;
        if (!usedB[i]) thirdB = i;
    }
    if (thirdA >= 3u || thirdB >= 3u) return false;
    return nmMortarGuardStrictOppositeSides(
        triangleA[matchedA[0]], triangleA[matchedA[1]],
        triangleA[thirdA], triangleB[thirdB]);
}

inline bool nmMortarGuardNondegenerateTriangle3(
    const NMMortarGuardPoint3 a,
    const NMMortarGuardPoint3 b,
    const NMMortarGuardPoint3 c
) {
    const float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
    const float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;
    const float nx = aby * acz - abz * acy;
    const float ny = abz * acx - abx * acz;
    const float nz = abx * acy - aby * acx;
    const float areaSquared = nx * nx + ny * ny + nz * nz;
    return nmMortarGuardFinite(areaSquared) && areaSquared > 0.0f;
}

// Return the canonical two-float expansion of a+b. Under the repository's
// non-fast floating-point mode, TwoSum preserves the exact real sum of two
// finite FP32 values as (sum,error), unless the rounded sum overflows.
inline bool nmMortarGuardTwoSum(
    const float a,
    const float b,
    NM_MORTAR_THREAD_POINTER float* sum,
    NM_MORTAR_THREAD_POINTER float* error
) {
    if (!nmMortarGuardFinite(a) || !nmMortarGuardFinite(b) ||
        sum == nullptr || error == nullptr) return false;
    const float rounded = a + b;
    if (!nmMortarGuardFinite(rounded)) return false;
    const float bVirtual = rounded - a;
    const float aVirtual = rounded - bVirtual;
    const float bRoundoff = b - bVirtual;
    const float aRoundoff = a - aVirtual;
    const float residual = aRoundoff + bRoundoff;
    if (!nmMortarGuardFinite(residual)) return false;
    *sum = rounded;
    *error = residual;
    return true;
}

// The Metal path must not depend on subnormal preservation for the exact
// two-word translation comparison. Keep endpoint coordinates in a bounded
// normal range (zero remains valid); any out-of-range authored geometry
// declines the whole-parent optimization and falls back to CCD.
inline bool nmMortarGuardSafeTwoSumOperand(const float value) {
    if (!nmMortarGuardFinite(value)) return false;
    if (value == 0.0f) return true;
    const float magnitude = nmMortarGuardAbs(value);
    return magnitude >= 1.0e-20f && magnitude <= 1.0e20f;
}

inline bool nmMortarGuardExactDifferenceEqual(
    const float firstFinish,
    const float firstStart,
    const float secondFinish,
    const float secondStart
) {
    float firstHigh, firstLow, secondHigh, secondLow;
    return nmMortarGuardSafeTwoSumOperand(firstFinish) &&
        nmMortarGuardSafeTwoSumOperand(firstStart) &&
        nmMortarGuardSafeTwoSumOperand(secondFinish) &&
        nmMortarGuardSafeTwoSumOperand(secondStart) &&
        nmMortarGuardTwoSum(firstFinish, -firstStart,
                               &firstHigh, &firstLow) &&
        nmMortarGuardTwoSum(secondFinish, -secondStart,
                            &secondHigh, &secondLow) &&
        firstHigh == secondHigh && firstLow == secondLow;
}

inline bool nmMortarGuardTriangleHasExactCommonTranslation(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* start,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* finish
) {
    if (start == nullptr || finish == nullptr) return false;
    for (unsigned int vertexIndex = 0u; vertexIndex < 3u; ++vertexIndex) {
        if (!nmMortarGuardFinite(start[vertexIndex].x) ||
            !nmMortarGuardFinite(start[vertexIndex].y) ||
            !nmMortarGuardFinite(start[vertexIndex].z) ||
            !nmMortarGuardFinite(finish[vertexIndex].x) ||
            !nmMortarGuardFinite(finish[vertexIndex].y) ||
            !nmMortarGuardFinite(finish[vertexIndex].z)) return false;
    }
    for (unsigned int vertexIndex = 1u; vertexIndex < 3u; ++vertexIndex) {
        if (!nmMortarGuardExactDifferenceEqual(
                finish[0].x, start[0].x,
                finish[vertexIndex].x, start[vertexIndex].x) ||
            !nmMortarGuardExactDifferenceEqual(
                finish[0].y, start[0].y,
                finish[vertexIndex].y, start[vertexIndex].y) ||
            !nmMortarGuardExactDifferenceEqual(
                finish[0].z, start[0].z,
                finish[vertexIndex].z, start[vertexIndex].z)) return false;
    }
    return true;
}

// Certify that two parent triangles describe the same complete physical sweep.
// Vertices may be stored in different order, but the same permutation must
// match exact FP32 positions at both endpoints. Since the caller's trajectory
// is linear between those endpoints, corresponding vertices and therefore
// every corresponding barycentric material point coincide for all t in [0,1].
// Nonfinite, duplicate-start, or unmatched vertices fail closed.
inline bool nmMortarGuardSameTriangleSweepWithPermutation(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftFinish,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightFinish,
    NM_MORTAR_THREAD_POINTER unsigned int* rightIndexForLeft
) {
    if (leftStart == nullptr || leftFinish == nullptr ||
        rightStart == nullptr || rightFinish == nullptr ||
        rightIndexForLeft == nullptr) return false;
    if (!nmMortarGuardNondegenerateTriangle3(
            leftStart[0], leftStart[1], leftStart[2]) ||
        !nmMortarGuardNondegenerateTriangle3(
            leftFinish[0], leftFinish[1], leftFinish[2]) ||
        !nmMortarGuardNondegenerateTriangle3(
            rightStart[0], rightStart[1], rightStart[2]) ||
        !nmMortarGuardNondegenerateTriangle3(
            rightFinish[0], rightFinish[1], rightFinish[2])) return false;
    // Endpoint nondegeneracy alone is insufficient: a deforming triangle can
    // collapse between the endpoints, invalidating disjoint-cell reasoning.
    // The whole-parent contact skip currently admits only exact common
    // translations, which keep each parent shape and area constant throughout
    // the linear sweep. Other motions remain on the ordinary CCD path.
    if (!nmMortarGuardTriangleHasExactCommonTranslation(
            leftStart, leftFinish) ||
        !nmMortarGuardTriangleHasExactCommonTranslation(
            rightStart, rightFinish)) return false;
    bool matchedRight[3] = {false, false, false};
    for (unsigned int i = 0u; i < 3u; ++i) {
        const NMMortarGuardPoint3 points[4] = {
            leftStart[i], leftFinish[i], rightStart[i], rightFinish[i]};
        for (unsigned int k = 0u; k < 4u; ++k)
            if (!nmMortarGuardFinite(points[k].x) ||
                !nmMortarGuardFinite(points[k].y) ||
                !nmMortarGuardFinite(points[k].z)) return false;
        for (unsigned int prior = 0u; prior < i; ++prior)
            if (nmMortarGuardSamePoint(leftStart[i], leftStart[prior]) ||
                nmMortarGuardSamePoint(rightStart[i], rightStart[prior]))
                return false;

        unsigned int match = 3u;
        for (unsigned int j = 0u; j < 3u; ++j) {
            if (nmMortarGuardSamePoint(leftStart[i], rightStart[j])) {
                if (match != 3u || matchedRight[j]) return false;
                match = j;
            }
        }
        if (match >= 3u ||
            !nmMortarGuardSamePoint(leftFinish[i], rightFinish[match]))
            return false;
        rightIndexForLeft[i] = match;
        matchedRight[match] = true;
    }
    return matchedRight[0] && matchedRight[1] && matchedRight[2];
}

inline bool nmMortarGuardSameTriangleSweep(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftFinish,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightFinish
) {
    unsigned int rightIndexForLeft[3] = {0u, 0u, 0u};
    return nmMortarGuardSameTriangleSweepWithPermutation(
        leftStart, leftFinish, rightStart, rightFinish, rightIndexForLeft);
}

inline bool nmMortarGuardSameProjectedPoint(
    const NMMortarGuardPoint3 left,
    const NMMortarGuardPoint3 right
) {
    return nmMortarGuardFinite(left.x) && nmMortarGuardFinite(left.y) &&
        nmMortarGuardFinite(right.x) && nmMortarGuardFinite(right.y) &&
        left.x == right.x && left.y == right.y;
}

inline bool nmMortarGuardRobustProjectedTriangle(
    const NMMortarGuardPoint3 a,
    const NMMortarGuardPoint3 b,
    const NMMortarGuardPoint3 c
) {
    if (!nmMortarGuardFinite(a.x) || !nmMortarGuardFinite(a.y) ||
        !nmMortarGuardFinite(b.x) || !nmMortarGuardFinite(b.y) ||
        !nmMortarGuardFinite(c.x) || !nmMortarGuardFinite(c.y)) return false;
    const float orientation = nmMortarGuardOrientation(a, b, c);
    const float error = nmMortarGuardOrientationErrorBound(a, b, c);
    return nmMortarGuardFinite(orientation) && nmMortarGuardFinite(error) &&
        nmMortarGuardAbs(orientation) > error;
}

// A whole-pair certificate for a narrow axis-aligned case. Each parent must
// remain a horizontal triangle under exact common translation, its projected
// start/end footprint must be nondegenerate, and the two parents must have
// identical projected endpoint vertices under one exact permutation. The
// parent z planes may translate independently, so only relative normal
// closure/opening is admitted; lateral slip, shear, rotation, nonplanarity,
// and uncertain coordinates return false for ordinary CCD.
inline bool nmMortarGuardSameAxisAlignedNormalParentSweep(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* leftFinish,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightStart,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardPoint3* rightFinish,
    NM_MORTAR_THREAD_POINTER unsigned int* rightIndexForLeft
) {
    if (leftStart == nullptr || leftFinish == nullptr ||
        rightStart == nullptr || rightFinish == nullptr ||
        rightIndexForLeft == nullptr ||
        !nmMortarGuardTriangleHasExactCommonTranslation(leftStart, leftFinish) ||
        !nmMortarGuardTriangleHasExactCommonTranslation(rightStart, rightFinish))
        return false;
    for (unsigned int vertexIndex = 1u; vertexIndex < 3u; ++vertexIndex) {
        if (leftStart[vertexIndex].z != leftStart[0].z ||
            leftFinish[vertexIndex].z != leftFinish[0].z ||
            rightStart[vertexIndex].z != rightStart[0].z ||
            rightFinish[vertexIndex].z != rightFinish[0].z) return false;
    }
    if (!nmMortarGuardRobustProjectedTriangle(
            leftStart[0], leftStart[1], leftStart[2]) ||
        !nmMortarGuardRobustProjectedTriangle(
            rightStart[0], rightStart[1], rightStart[2])) return false;

    bool matchedRight[3] = {false, false, false};
    for (unsigned int leftVertex = 0u; leftVertex < 3u; ++leftVertex) {
        for (unsigned int prior = 0u; prior < leftVertex; ++prior) {
            if (nmMortarGuardSameProjectedPoint(
                    leftStart[leftVertex], leftStart[prior]) ||
                nmMortarGuardSameProjectedPoint(
                    rightStart[leftVertex], rightStart[prior])) return false;
        }
        unsigned int match = 3u;
        for (unsigned int rightVertex = 0u; rightVertex < 3u; ++rightVertex) {
            if (!nmMortarGuardSameProjectedPoint(
                    leftStart[leftVertex], rightStart[rightVertex])) continue;
            if (match != 3u || matchedRight[rightVertex]) return false;
            match = rightVertex;
        }
        if (match >= 3u ||
            !nmMortarGuardSameProjectedPoint(
                leftFinish[leftVertex], rightFinish[match])) return false;
        rightIndexForLeft[leftVertex] = match;
        matchedRight[match] = true;
    }
    return matchedRight[0] && matchedRight[1] && matchedRight[2];
}

inline bool nmMortarGuardSamePatchTriangleSweeps(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* leftCells,
    const unsigned int leftCount,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* rightCells,
    const unsigned int rightCount
);

inline bool nmMortarGuardProjectedPartitionSharedVerticesAreCoherent(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* cells,
    const unsigned int count
) {
    if (cells == nullptr || count < 2u || count > 4u) return false;
    for (unsigned int firstCell = 0u; firstCell < count; ++firstCell) {
        if (cells[firstCell].patchID == 0xffffffffu) return false;
        for (unsigned int firstVertex = 0u; firstVertex < 3u; ++firstVertex) {
            const NMMortarGuardPoint3 firstStart =
                cells[firstCell].start[firstVertex];
            const NMMortarGuardPoint3 firstFinish =
                cells[firstCell].finish[firstVertex];
            if (!nmMortarGuardFinite(firstStart.x) ||
                !nmMortarGuardFinite(firstStart.y) ||
                !nmMortarGuardFinite(firstFinish.x) ||
                !nmMortarGuardFinite(firstFinish.y)) return false;
            for (unsigned int secondCell = firstCell;
                 secondCell < count; ++secondCell) {
                const unsigned int vertexStart =
                    secondCell == firstCell ? firstVertex + 1u : 0u;
                for (unsigned int secondVertex = vertexStart;
                     secondVertex < 3u; ++secondVertex) {
                    const NMMortarGuardPoint3 secondStart =
                        cells[secondCell].start[secondVertex];
                    const NMMortarGuardPoint3 secondFinish =
                        cells[secondCell].finish[secondVertex];
                    if (!nmMortarGuardFinite(secondStart.x) ||
                        !nmMortarGuardFinite(secondStart.y) ||
                        !nmMortarGuardFinite(secondFinish.x) ||
                        !nmMortarGuardFinite(secondFinish.y)) return false;
                    if (nmMortarGuardSameProjectedPoint(firstStart, secondStart) &&
                        !nmMortarGuardSameProjectedPoint(
                            firstFinish, secondFinish)) return false;
                }
            }
        }
    }
    return true;
}

inline bool nmMortarGuardSameProjectedPatchTriangleSweeps(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* leftCells,
    const unsigned int leftCount,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* rightCells,
    const unsigned int rightCount
) {
    if (leftCells == nullptr || rightCells == nullptr || leftCount < 2u ||
        leftCount > 4u || rightCount != leftCount) return false;
    NMMortarGuardTriangleSweep leftProjected[4] = {};
    NMMortarGuardTriangleSweep rightProjected[4] = {};
    for (unsigned int cell = 0u; cell < leftCount; ++cell) {
        leftProjected[cell].patchID = leftCells[cell].patchID;
        rightProjected[cell].patchID = rightCells[cell].patchID;
        NMMortarGuardPoint3 leftStartPoints[3];
        NMMortarGuardPoint3 leftFinishPoints[3];
        NMMortarGuardPoint3 rightStartPoints[3];
        NMMortarGuardPoint3 rightFinishPoints[3];
        for (unsigned int vertexIndex = 0u; vertexIndex < 3u; ++vertexIndex) {
            leftProjected[cell].start[vertexIndex] = {
                leftCells[cell].start[vertexIndex].x,
                leftCells[cell].start[vertexIndex].y, 0.0f};
            leftProjected[cell].finish[vertexIndex] = {
                leftCells[cell].finish[vertexIndex].x,
                leftCells[cell].finish[vertexIndex].y, 0.0f};
            rightProjected[cell].start[vertexIndex] = {
                rightCells[cell].start[vertexIndex].x,
                rightCells[cell].start[vertexIndex].y, 0.0f};
            rightProjected[cell].finish[vertexIndex] = {
                rightCells[cell].finish[vertexIndex].x,
                rightCells[cell].finish[vertexIndex].y, 0.0f};
            leftStartPoints[vertexIndex] = leftProjected[cell].start[vertexIndex];
            leftFinishPoints[vertexIndex] = leftProjected[cell].finish[vertexIndex];
            rightStartPoints[vertexIndex] = rightProjected[cell].start[vertexIndex];
            rightFinishPoints[vertexIndex] = rightProjected[cell].finish[vertexIndex];
        }
        if (!nmMortarGuardRobustProjectedTriangle(
                leftStartPoints[0], leftStartPoints[1], leftStartPoints[2]) ||
            !nmMortarGuardRobustProjectedTriangle(
                leftFinishPoints[0], leftFinishPoints[1], leftFinishPoints[2]) ||
            !nmMortarGuardRobustProjectedTriangle(
                rightStartPoints[0], rightStartPoints[1], rightStartPoints[2]) ||
            !nmMortarGuardRobustProjectedTriangle(
                rightFinishPoints[0], rightFinishPoints[1], rightFinishPoints[2]))
            return false;
    }
    return nmMortarGuardProjectedPartitionSharedVerticesAreCoherent(
            leftProjected, leftCount) &&
        nmMortarGuardProjectedPartitionSharedVerticesAreCoherent(
            rightProjected, rightCount) &&
        nmMortarGuardSamePatchTriangleSweeps(
        leftProjected, leftCount, rightProjected, rightCount);
}

inline bool nmMortarGuardSamePatchTriangleSweeps(
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* leftCells,
    const unsigned int leftCount,
    NM_MORTAR_THREAD_POINTER const NMMortarGuardTriangleSweep* rightCells,
    const unsigned int rightCount
) {
    if (leftCells == nullptr || rightCells == nullptr || leftCount < 2u ||
        leftCount > 4u || rightCount != leftCount) return false;
    bool matchedRight[4] = {false, false, false, false};
    for (unsigned int leftIndex = 0u; leftIndex < leftCount; ++leftIndex) {
        if (leftCells[leftIndex].patchID == 0xffffffffu) return false;
        unsigned int match = rightCount;
        for (unsigned int rightIndex = 0u; rightIndex < rightCount; ++rightIndex) {
            if (matchedRight[rightIndex] ||
                leftCells[leftIndex].patchID != rightCells[rightIndex].patchID)
                continue;
            if (nmMortarGuardSameTriangleSweep(
                    leftCells[leftIndex].start, leftCells[leftIndex].finish,
                    rightCells[rightIndex].start,
                    rightCells[rightIndex].finish)) {
                if (match != rightCount) return false;
                match = rightIndex;
            }
        }
        if (match >= rightCount) return false;
        matchedRight[match] = true;
    }
    for (unsigned int rightIndex = 0u; rightIndex < rightCount; ++rightIndex)
        if (!matchedRight[rightIndex] ||
            rightCells[rightIndex].patchID == 0xffffffffu) return false;
    return true;
}

#endif

#undef NM_MORTAR_THREAD_POINTER
