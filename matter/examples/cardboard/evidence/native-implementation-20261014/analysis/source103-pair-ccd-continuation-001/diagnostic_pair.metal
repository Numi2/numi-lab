// Standalone diagnostic translation unit. The production kernels and exact
// source-103 CCD routines come from the frozen matter_kernels.metal include.
#include "matter_kernels.metal"

namespace numi_matter_metal {

inline bool vertexTriangleCCDTraceClone(
    const float3 vertex0, const float3 vertex1,
    const float3 a0, const float3 a1,
    const float3 b0, const float3 b1,
    const float3 c0, const float3 c1,
    const float thickness,
    thread float& toi, thread float3& normal,
    thread float3& barycentric, thread float3& point,
    thread float& separation, thread bool& certified,
    device float4* trace, const uint traceBase,
    thread uint& iterationCount, thread uint& terminal,
    thread float& speedBoundOut, thread float& effectiveThicknessOut
) {
    certified = false; toi = 0.0f; separation = INFINITY;
    iterationCount = 0u; terminal = 0u;
    float coordinateScale = maximumAbsoluteCoordinate(vertex0);
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(vertex1));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(a0));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(a1));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(b0));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(b1));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(c0));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(c1));
    const float distanceRoundoff = ccdDistanceRoundoff(thickness, coordinateScale);
    const float effectiveThickness = thickness + distanceRoundoff;
    effectiveThicknessOut = effectiveThickness;
    const float speedBound = length(vertex1 - vertex0) + max(
        length(a1 - a0), max(length(b1 - b0), length(c1 - c0)));
    speedBoundOut = speedBound;
    float time = 0.0f;
    for (uint iteration = 0u; iteration < 16u; ++iteration) {
        const float3 vertexPoint = mix(vertex0, vertex1, time);
        const float3 a = mix(a0, a1, time);
        const float3 b = mix(b0, b1, time);
        const float3 c = mix(c0, c1, time);
        float3 bary;
        const float3 trianglePoint = closestPointOnTriangle(vertexPoint, a, b, c, bary);
        const float3 delta = vertexPoint - trianglePoint;
        const float distance = length(delta);
        toi = time; separation = distance; iterationCount = iteration + 1u;
        trace[traceBase + iteration] = float4(time, distance, 0.0f, 0.0f);
        if (distance <= effectiveThickness) {
            float3 fallback = cross(b - a, c - a);
            if (length(fallback) <= kGeometryFloor)
                fallback = (vertex1 - vertex0) -
                    ((a1 + b1 + c1) - (a0 + b0 + c0)) / 3.0f;
            normal = distance > kGeometryFloor
                ? delta / distance
                : normalize(fallback + float3(kGeometryFloor));
            toi = time; barycentric = bary; point = trianglePoint;
            separation = distance; certified = true;
            terminal = 1u;
            trace[traceBase + iteration].w = float(terminal);
            return finite3(normal);
        }
        const float3 firstStart[1] = {vertex0}, firstFinish[1] = {vertex1};
        const float3 secondStart[3] = {a0, b0, c0};
        const float3 secondFinish[3] = {a1, b1, c1};
        if (sweptFeaturesSeparated(firstStart, firstFinish, 1u,
                secondStart, secondFinish, 3u, delta, effectiveThickness)) {
            certified = true; terminal = 2u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        if (iteration == 0u) {
            const float3 axes[2] = {cross(b0-a0, c0-a0), cross(b1-a1, c1-a1)};
            for (uint candidateAxis = 0u; candidateAxis < 2u; ++candidateAxis) {
                float3 axis = axes[candidateAxis];
                if (dot(axis, firstStart[0] - secondStart[0]) < 0.0f) axis = -axis;
                if (sweptFeaturesSeparated(firstStart, firstFinish, 1u,
                        secondStart, secondFinish, 3u, axis, effectiveThickness)) {
                    certified = true; terminal = 3u;
                    trace[traceBase + iteration].w = float(terminal);
                    return false;
                }
            }
        }
        if (!(speedBound > kGeometryFloor)) {
            certified = true; terminal = 4u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        const float step = max((distance - effectiveThickness) / speedBound, 1.0e-6f);
        trace[traceBase + iteration].z = step;
        time += step;
        if (time > 1.0f + 8.0f * kEpsilon) {
            certified = true; terminal = 5u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        time = min(time, 1.0f);
    }
    terminal = 6u;
    return false;
}

inline bool edgeEdgeCCDTraceClone(
    const float3 p00, const float3 p01, const float3 p10, const float3 p11,
    const float3 q00, const float3 q01, const float3 q10, const float3 q11,
    const float thickness,
    thread float& toi, thread float3& normal, thread float2& parameters,
    thread float3& point, thread float& separation, thread bool& certified,
    device float4* trace, const uint traceBase,
    thread uint& iterationCount, thread uint& terminal,
    thread float& speedBoundOut, thread float& effectiveThicknessOut
) {
    certified = false; toi = 0.0f; separation = INFINITY;
    iterationCount = 0u; terminal = 0u;
    float coordinateScale = maximumAbsoluteCoordinate(p00);
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(p11));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q00));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q01));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q10));
    coordinateScale = max(coordinateScale, maximumAbsoluteCoordinate(q11));
    const float distanceRoundoff = ccdDistanceRoundoff(thickness, coordinateScale);
    const float effectiveThickness = thickness + distanceRoundoff;
    effectiveThicknessOut = effectiveThickness;
    const float speedBound = max(length(p01 - p00), length(p11 - p10)) +
        max(length(q01 - q00), length(q11 - q10));
    speedBoundOut = speedBound;
    float time = 0.0f;
    for (uint iteration = 0u; iteration < 33u; ++iteration) {
        const float3 p0 = mix(p00, p01, time);
        const float3 p1 = mix(p10, p11, time);
        const float3 q0 = mix(q00, q01, time);
        const float3 q1 = mix(q10, q11, time);
        float s, t; float3 pointP, pointQ;
        closestPointsOnSegments(p0, p1, q0, q1, s, t, pointP, pointQ);
        const float3 delta = pointP - pointQ;
        const float distance = length(delta);
        toi = time; separation = distance; iterationCount = iteration + 1u;
        trace[traceBase + iteration] = float4(time, distance, 0.0f, 0.0f);
        if (distance <= effectiveThickness) {
            float3 fallback = cross(p1 - p0, q1 - q0);
            if (length(fallback) <= kGeometryFloor)
                fallback = (p01 - p00) - (q01 - q00);
            normal = distance > kGeometryFloor
                ? delta / distance
                : normalize(fallback + float3(kGeometryFloor));
            toi = time; parameters = float2(s, t);
            point = 0.5f * (pointP + pointQ);
            separation = distance; certified = true; terminal = 1u;
            trace[traceBase + iteration].w = float(terminal);
            return finite3(normal);
        }
        const float3 firstStart[2] = {p00, p10}, firstFinish[2] = {p01, p11};
        const float3 secondStart[2] = {q00, q10}, secondFinish[2] = {q01, q11};
        if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                secondStart, secondFinish, 2u, delta, effectiveThickness)) {
            certified = true; terminal = 2u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        if (iteration == 0u) {
            const float3 axes[2] = {cross(p10-p00, q10-q00), cross(p11-p01, q11-q01)};
            for (uint candidateAxis = 0u; candidateAxis < 2u; ++candidateAxis) {
                float3 axis = axes[candidateAxis];
                if (dot(axis, firstStart[0] - secondStart[0]) < 0.0f) axis = -axis;
                if (sweptFeaturesSeparated(firstStart, firstFinish, 2u,
                        secondStart, secondFinish, 2u, axis, effectiveThickness)) {
                    certified = true; terminal = 3u;
                    trace[traceBase + iteration].w = float(terminal);
                    return false;
                }
            }
        }
        if (!(speedBound > kGeometryFloor)) {
            certified = true; terminal = 4u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        const float step = max((distance - effectiveThickness) / speedBound, 1.0e-6f);
        trace[traceBase + iteration].z = step;
        time += step;
        if (time > 1.0f + 8.0f * kEpsilon) {
            certified = true; terminal = 5u;
            trace[traceBase + iteration].w = float(terminal);
            return false;
        }
        time = min(time, 1.0f);
    }
    terminal = 6u;
    return false;
}

// Every capture gives two triangles. Input case records match
// nm_contact_ccd_regression: header + four starts + four finishes. This kernel
// directly calls the source-103 production routines and also a trace clone.
kernel void nm_cardboard_source103_pair_ccd_diagnostic(
    device const float4* cases [[buffer(0)]],
    device float4* summaries [[buffer(1)]],
    device float4* iterationTrace [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const uint inputBase = 9u * index;
    const uint summaryBase = 9u * index;
    const uint traceBase = 33u * index;
    const float4 header = cases[inputBase];
    const float thickness = header.y;
    const uint kind = uint(header.x);
    float hitToi=0.0f, hitSeparation=0.0f;
    float3 normal=float3(0.0f), point=float3(0.0f), bary=float3(0.0f);
    float2 parameters=float2(0.0f);
    bool certified=false;
    for (uint it=0u;it<33u;++it) iterationTrace[traceBase+it]=float4(0.0f);
    bool hit=false;
    if (kind==0u) {
        hit=vertexTriangleCCD(cases[inputBase+1u].xyz,cases[inputBase+5u].xyz,
            cases[inputBase+2u].xyz,cases[inputBase+6u].xyz,
            cases[inputBase+3u].xyz,cases[inputBase+7u].xyz,
            cases[inputBase+4u].xyz,cases[inputBase+8u].xyz,thickness,
            hitToi,normal,bary,point,hitSeparation,certified);
    } else {
        hit=edgeEdgeCCD(cases[inputBase+1u].xyz,cases[inputBase+5u].xyz,
            cases[inputBase+2u].xyz,cases[inputBase+6u].xyz,
            cases[inputBase+3u].xyz,cases[inputBase+7u].xyz,
            cases[inputBase+4u].xyz,cases[inputBase+8u].xyz,thickness,
            hitToi,normal,parameters,point,hitSeparation,certified);
    }
    summaries[summaryBase+0u]=float4(hit?1.0f:0.0f,certified?1.0f:0.0f,hitToi,hitSeparation);
    summaries[summaryBase+1u]=float4(normal,kind==0u?bary.x:parameters.x);
    summaries[summaryBase+2u]=float4(point,kind==0u?bary.y:parameters.y);
    summaries[summaryBase+3u]=float4(kind==0u?bary.z:0.0f,0.0f,0.0f,0.0f);

    float cloneToi=0.0f,cloneSeparation=0.0f;
    float3 cloneNormal=float3(0.0f),clonePoint=float3(0.0f),cloneBary=float3(0.0f);
    float2 cloneParameters=float2(0.0f);
    bool cloneCertified=false,cloneHit=false;
    uint iterations=0u,terminal=0u;
    float speedBound=0.0f,effectiveThickness=0.0f;
    if (kind==0u) {
        cloneHit=vertexTriangleCCDTraceClone(
            cases[inputBase+1u].xyz,cases[inputBase+5u].xyz,
            cases[inputBase+2u].xyz,cases[inputBase+6u].xyz,
            cases[inputBase+3u].xyz,cases[inputBase+7u].xyz,
            cases[inputBase+4u].xyz,cases[inputBase+8u].xyz,thickness,
            cloneToi,cloneNormal,cloneBary,clonePoint,cloneSeparation,cloneCertified,
            iterationTrace,traceBase,iterations,terminal,speedBound,effectiveThickness);
    } else {
        cloneHit=edgeEdgeCCDTraceClone(
            cases[inputBase+1u].xyz,cases[inputBase+5u].xyz,
            cases[inputBase+2u].xyz,cases[inputBase+6u].xyz,
            cases[inputBase+3u].xyz,cases[inputBase+7u].xyz,
            cases[inputBase+4u].xyz,cases[inputBase+8u].xyz,thickness,
            cloneToi,cloneNormal,cloneParameters,clonePoint,cloneSeparation,cloneCertified,
            iterationTrace,traceBase,iterations,terminal,speedBound,effectiveThickness);
    }
    summaries[summaryBase+4u]=float4(cloneHit?1.0f:0.0f,cloneCertified?1.0f:0.0f,cloneToi,cloneSeparation);
    summaries[summaryBase+5u]=float4(cloneNormal,kind==0u?cloneBary.x:cloneParameters.x);
    summaries[summaryBase+6u]=float4(clonePoint,kind==0u?cloneBary.y:cloneParameters.y);
    summaries[summaryBase+7u]=float4(kind==0u?cloneBary.z:0.0f,float(iterations),float(terminal),effectiveThickness);
    summaries[summaryBase+8u]=float4(speedBound,0.0f,0.0f,0.0f);
}

} // namespace numi_matter_metal
