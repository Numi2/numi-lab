#include "/Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099/matter/src/matter_kernels.metal"
namespace numi_matter_metal {
inline bool edgeEdgeCCD16Diagnostic(
    const float3 p00,
    const float3 p01,
    const float3 p10,
    const float3 p11,
    const float3 q00,
    const float3 q01,
    const float3 q10,
    const float3 q11,
    const float thickness,
    thread float& toi,
    thread float3& normal,
    thread float2& parameters,
    thread float3& point,
    thread float& separation,
    thread bool& certified,
    thread uint& iterationsExecuted,
    thread float3& witnessP,
    thread float3& witnessQ,
    thread float2& witnessParameters,
    device float4* iterationTrace,
    const uint traceOffset
) {
    certified = false;
    iterationsExecuted = 0u;
    normal = float3(0.0f);
    parameters = float2(0.0f);
    point = float3(0.0f);
    witnessP = float3(0.0f);
    witnessQ = float3(0.0f);
    witnessParameters = float2(0.0f);
    toi = 0.0f;
    separation = INFINITY;
    float coordinateScale = maximumAbsoluteCoordinate(p00);
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p11));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q00));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q11));
    const float distanceRoundoff =
        ccdDistanceRoundoff(thickness, coordinateScale);
    const float effectiveThickness = thickness + distanceRoundoff;
    const float speedBound = max(length(p01 - p00), length(p11 - p10)) +
        max(length(q01 - q00), length(q11 - q10));
    float time = 0.0f;
    for (uint iteration = 0u; iteration < 16u; ++iteration) {
        const float3 p0 = mix(p00, p01, time);
        const float3 p1 = mix(p10, p11, time);
        const float3 q0 = mix(q00, q01, time);
        const float3 q1 = mix(q10, q11, time);
        float s, t;
        float3 pointP, pointQ;
        closestPointsOnSegments(p0, p1, q0, q1, s, t, pointP, pointQ);
        iterationsExecuted = iteration + 1u;
        witnessP = pointP;
        witnessQ = pointQ;
        witnessParameters = float2(s, t);
        const uint trace = traceOffset + 3u * iteration;
        const float3 delta = pointP - pointQ;
        const float distance = length(delta);
        iterationTrace[trace] = float4(time, distance, effectiveThickness, speedBound);
        iterationTrace[trace + 1u] = float4(pointP, s);
        iterationTrace[trace + 2u] = float4(pointQ, t);
        toi = time;
        separation = distance;
        if (distance <= effectiveThickness) {
            float3 fallback = cross(p1 - p0, q1 - q0);
            if (length(fallback) <= kGeometryFloor)
                fallback = (p01 - p00) - (q01 - q00);
            normal = distance > kGeometryFloor
                ? delta / distance
                : normalize(fallback + float3(kGeometryFloor));
            toi = time;
            parameters = float2(s, t);
            point = 0.5f * (pointP + pointQ);
            separation = distance;
            certified = true;
            return finite3(normal);
        }
        const float3 firstStart[2] = {p00, p10}, firstFinish[2] = {p01, p11};
        const float3 secondStart[2] = {q00, q10}, secondFinish[2] = {q01, q11};
        if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                secondStart, secondFinish, 2u, delta, effectiveThickness)) {
            certified = true;
            return false;
        }
        if (iteration == 0u) {
            const float3 axes[2] = {cross(p10-p00, q10-q00), cross(p11-p01, q11-q01)};
            for (uint candidateAxis = 0u; candidateAxis < 2u; ++candidateAxis) {
                float3 axis = axes[candidateAxis];
                if (dot(axis, firstStart[0] - secondStart[0]) < 0.0f) axis = -axis;
                if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                        secondStart, secondFinish, 2u, axis, effectiveThickness)) {
                    certified = true;
                    return false;
                }
            }
        }
        if (!(speedBound > kGeometryFloor)) {
            certified = true;
            return false;
        }
        const float step = max(
            (distance - effectiveThickness) / speedBound, 1.0e-6f);
        time += step;
        if (time > 1.0f + 8.0f * kEpsilon) {
            certified = true;
            return false;
        }
        time = min(time, 1.0f);
    }
    return false;
}
inline bool edgeEdgeCCD17Diagnostic(
    const float3 p00,
    const float3 p01,
    const float3 p10,
    const float3 p11,
    const float3 q00,
    const float3 q01,
    const float3 q10,
    const float3 q11,
    const float thickness,
    thread float& toi,
    thread float3& normal,
    thread float2& parameters,
    thread float3& point,
    thread float& separation,
    thread bool& certified,
    thread uint& iterationsExecuted,
    thread float3& witnessP,
    thread float3& witnessQ,
    thread float2& witnessParameters,
    device float4* iterationTrace,
    const uint traceOffset
) {
    certified = false;
    iterationsExecuted = 0u;
    normal = float3(0.0f);
    parameters = float2(0.0f);
    point = float3(0.0f);
    witnessP = float3(0.0f);
    witnessQ = float3(0.0f);
    witnessParameters = float2(0.0f);
    toi = 0.0f;
    separation = INFINITY;
    float coordinateScale = maximumAbsoluteCoordinate(p00);
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p11));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q00));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q11));
    const float distanceRoundoff =
        ccdDistanceRoundoff(thickness, coordinateScale);
    const float effectiveThickness = thickness + distanceRoundoff;
    const float speedBound = max(length(p01 - p00), length(p11 - p10)) +
        max(length(q01 - q00), length(q11 - q10));
    float time = 0.0f;
    for (uint iteration = 0u; iteration < 17u; ++iteration) {
        const float3 p0 = mix(p00, p01, time);
        const float3 p1 = mix(p10, p11, time);
        const float3 q0 = mix(q00, q01, time);
        const float3 q1 = mix(q10, q11, time);
        float s, t;
        float3 pointP, pointQ;
        closestPointsOnSegments(p0, p1, q0, q1, s, t, pointP, pointQ);
        iterationsExecuted = iteration + 1u;
        witnessP = pointP;
        witnessQ = pointQ;
        witnessParameters = float2(s, t);
        const uint trace = traceOffset + 3u * iteration;
        const float3 delta = pointP - pointQ;
        const float distance = length(delta);
        iterationTrace[trace] = float4(time, distance, effectiveThickness, speedBound);
        iterationTrace[trace + 1u] = float4(pointP, s);
        iterationTrace[trace + 2u] = float4(pointQ, t);
        toi = time;
        separation = distance;
        if (distance <= effectiveThickness) {
            float3 fallback = cross(p1 - p0, q1 - q0);
            if (length(fallback) <= kGeometryFloor)
                fallback = (p01 - p00) - (q01 - q00);
            normal = distance > kGeometryFloor
                ? delta / distance
                : normalize(fallback + float3(kGeometryFloor));
            toi = time;
            parameters = float2(s, t);
            point = 0.5f * (pointP + pointQ);
            separation = distance;
            certified = true;
            return finite3(normal);
        }
        const float3 firstStart[2] = {p00, p10}, firstFinish[2] = {p01, p11};
        const float3 secondStart[2] = {q00, q10}, secondFinish[2] = {q01, q11};
        if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                secondStart, secondFinish, 2u, delta, effectiveThickness)) {
            certified = true;
            return false;
        }
        if (iteration == 0u) {
            const float3 axes[2] = {cross(p10-p00, q10-q00), cross(p11-p01, q11-q01)};
            for (uint candidateAxis = 0u; candidateAxis < 2u; ++candidateAxis) {
                float3 axis = axes[candidateAxis];
                if (dot(axis, firstStart[0] - secondStart[0]) < 0.0f) axis = -axis;
                if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                        secondStart, secondFinish, 2u, axis, effectiveThickness)) {
                    certified = true;
                    return false;
                }
            }
        }
        if (!(speedBound > kGeometryFloor)) {
            certified = true;
            return false;
        }
        const float step = max(
            (distance - effectiveThickness) / speedBound, 1.0e-6f);
        time += step;
        if (time > 1.0f + 8.0f * kEpsilon) {
            certified = true;
            return false;
        }
        time = min(time, 1.0f);
    }
    return false;
}

kernel void nm_contact_ee_ccd_budget_diagnostic(
    device const float4* cases [[buffer(0)]],
    device float4* results [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device float4* iterationTrace [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const uint base = 9u * index;
    const float budgetWord = cases[base].z;
    const bool valid = cases[base].x == 1.0f &&
        (budgetWord == 16.0f || budgetWord == 17.0f) &&
        isfinite(cases[base].y) && cases[base].y > 0.0f;
    const uint out = 5u * index;
    if (!valid) {
        results[out] = float4(0.0f);
        results[out + 1u] = float4(0.0f);
        results[out + 2u] = float4(0.0f);
        results[out + 3u] = float4(0.0f);
        results[out + 4u] = float4(0.0f);
        return;
    }
    float toi = 0.0f, separation = INFINITY;
    float3 normal = float3(0.0f), point = float3(0.0f);
    float2 parameters = float2(0.0f), witnessParameters = float2(0.0f);
    float3 witnessP = float3(0.0f), witnessQ = float3(0.0f);
    uint iterationsExecuted = 0u;
    bool certified = false;
    const uint traceBase = index * 51u;
    bool hit = false;
    if (budgetWord == 16.0f) {
        hit = edgeEdgeCCD16Diagnostic(
            cases[base+1u].xyz, cases[base+5u].xyz,
            cases[base+2u].xyz, cases[base+6u].xyz,
            cases[base+3u].xyz, cases[base+7u].xyz,
            cases[base+4u].xyz, cases[base+8u].xyz,
            cases[base].y, toi, normal, parameters, point, separation, certified,
            iterationsExecuted, witnessP, witnessQ, witnessParameters,
            iterationTrace, traceBase);
    } else {
        hit = edgeEdgeCCD17Diagnostic(
            cases[base+1u].xyz, cases[base+5u].xyz,
            cases[base+2u].xyz, cases[base+6u].xyz,
            cases[base+3u].xyz, cases[base+7u].xyz,
            cases[base+4u].xyz, cases[base+8u].xyz,
            cases[base].y, toi, normal, parameters, point, separation, certified,
            iterationsExecuted, witnessP, witnessQ, witnessParameters,
            iterationTrace, traceBase);
    }
    results[out] = float4(hit ? 1.0f : 0.0f,
        certified ? 1.0f : 0.0f, toi, separation);
    results[out + 1u] = float4(normal, witnessParameters.x);
    results[out + 2u] = float4(point, witnessParameters.y);
    results[out + 3u] = float4(witnessP, float(iterationsExecuted));
    results[out + 4u] = float4(witnessQ, budgetWord);
}

} // namespace numi_matter_metal
