#include "matter_kernels.metal"
namespace numi_matter_metal {
inline float3 scaledClosestPointOnTriangle(
    const float3 point,
    const float3 a,
    const float3 b,
    const float3 c,
    thread float3& barycentric
) {
    // Nearest-point coordinates are dimensionless. A fixed physical floor on
    // dot products (m^2) and area determinants (m^4) changes the answer under
    // mesh refinement. Normalize before the Voronoi-region tests instead.
    const float3 worldAB = b - a;
    const float3 worldAC = c - a;
    const float3 worldAP = point - a;
    const float scale = max(maximumAbsoluteCoordinate(worldAB),
        max(maximumAbsoluteCoordinate(worldAC),
            maximumAbsoluteCoordinate(worldAP)));
    if (!isfinite(scale)) {
        barycentric = float3(NAN);
        return float3(NAN);
    }
    if (!(scale > 0.0f)) {
        barycentric = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float3 ab = worldAB / scale;
    const float3 ac = worldAC / scale;
    const float3 ap = worldAP / scale;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        barycentric = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float3 bp = (point - b) / scale;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        barycentric = float3(0.0f, 1.0f, 0.0f);
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float denominator = d1 - d3;
        const float v = denominator > 0.0f ? d1 / denominator : 0.0f;
        barycentric = float3(1.0f - v, v, 0.0f);
        return a + v * worldAB;
    }
    const float3 cp = (point - c) / scale;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        barycentric = float3(0.0f, 0.0f, 1.0f);
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float denominator = d2 - d6;
        const float w = denominator > 0.0f ? d2 / denominator : 0.0f;
        barycentric = float3(1.0f - w, 0.0f, w);
        return a + w * worldAC;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const float denominator = (d4 - d3) + (d5 - d6);
        const float w = denominator > 0.0f ? (d4 - d3) / denominator : 0.0f;
        barycentric = float3(0.0f, 1.0f - w, w);
        return b + w * (c - b);
    }
    const float denominator = va + vb + vc;
    if (denominator > 0.0f) {
        const float v = vb / denominator;
        const float w = vc / denominator;
        barycentric = float3(1.0f - v - w, v, w);
        return a + v * worldAB + w * worldAC;
    }
    // Exact degeneracy has no face interior. Select the nearest of its three
    // closed segments; no unit-dependent epsilon changes the feature domain.
    float best = INFINITY;
    float3 bestBary = float3(1.0f, 0.0f, 0.0f);
    const float3 vertex[3] = {float3(0.0f), ab, ac};
    for (uint edge = 0u; edge < 3u; ++edge) {
        const uint next = (edge + 1u) % 3u;
        const float3 direction = vertex[next] - vertex[edge];
        const float norm2 = dot(direction, direction);
        const float t = norm2 > 0.0f
            ? clamp(dot(ap - vertex[edge], direction) / norm2, 0.0f, 1.0f)
            : 0.0f;
        const float3 offset = ap - (vertex[edge] + t * direction);
        const float distance2 = dot(offset, offset);
        if (distance2 < best) {
            best = distance2;
            bestBary = float3(0.0f);
            bestBary[edge] = 1.0f - t;
            bestBary[next] = t;
        }
    }
    barycentric = bestBary;
    return a + bestBary.y * worldAB + bestBary.z * worldAC;
}

inline void scaledClosestPointsOnSegments(
    const float3 p0,
    const float3 p1,
    const float3 q0,
    const float3 q1,
    thread float& s,
    thread float& t,
    thread float3& pointP,
    thread float3& pointQ
) {
    const float3 worldDP = p1 - p0;
    const float3 worldDQ = q1 - q0;
    const float3 worldR = p0 - q0;
    const float scale = max(maximumAbsoluteCoordinate(worldDP),
        max(maximumAbsoluteCoordinate(worldDQ),
            maximumAbsoluteCoordinate(worldR)));
    if (!isfinite(scale)) {
        s = t = NAN;
        pointP = pointQ = float3(NAN);
        return;
    }
    if (!(scale > 0.0f)) {
        s = t = 0.0f;
        pointP = p0;
        pointQ = q0;
        return;
    }
    const float3 dP = worldDP / scale;
    const float3 dQ = worldDQ / scale;
    const float3 r = worldR / scale;
    const float a = dot(dP, dP);
    const float e = dot(dQ, dQ);
    const float f = dot(dQ, r);
    if (!(a > 0.0f) && !(e > 0.0f)) {
        s = t = 0.0f;
    } else if (!(a > 0.0f)) {
        s = 0.0f;
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        const float c = dot(dP, r);
        if (!(e > 0.0f)) {
            t = 0.0f;
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            const float b = dot(dP, dQ);
            const float3 perpendicular = cross(dP, dQ);
            const float denominator = dot(perpendicular, perpendicular);
            s = denominator > 0.0f
                ? clamp(dot(cross(dQ, r), perpendicular) / denominator, 0.0f, 1.0f)
                : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    pointP = p0 + s * worldDP;
    pointQ = q0 + t * worldDQ;
}

kernel void nm_contact_scaled_geometry_replay(device float4* output [[buffer(0)]], uint global [[thread_position_in_grid]]) {
    if (global>=48u) return;
    const uint kind=global%6u;
    const uint scaleIndex=(global/6u)%4u;
    const float scales[4]={0.0001f,0.0005f,0.002f,0.02f};
    const float s=scales[scaleIndex];
    const float shift=global<24u?0.0f:0.03f;
    const float3 origin(shift,-0.5f*shift,0.25f*shift);
    float3 point=origin,a=origin,b=origin,c=origin,q1=origin;
    float3 oldBary(0.0f),newBary(0.0f),oldP(0.0f),newP(0.0f),oldQ(0.0f),newQ(0.0f);
    if (kind<3u) {
        a=origin;b=origin+float3(s,0,0);c=origin+float3(0,s,0);
        if(kind==0u)point=origin+float3(0.25f*s,0.25f*s,0.1f*s);
        if(kind==1u)point=origin+float3(0.5f*s,-0.2f*s,0.1f*s);
        if(kind==2u)point=origin+float3(-0.2f*s,1.2f*s,0.1f*s);
        oldP=closestPointOnTriangle(point,a,b,c,oldBary);
        newP=scaledClosestPointOnTriangle(point,a,b,c,newBary);
    } else {
        if(kind==3u){a=origin+float3(-0.5f*s,0,0);b=origin+float3(0.5f*s,0,0);c=origin+float3(0,-0.5f*s,0.1f*s);q1=origin+float3(0,0.5f*s,0.1f*s);}
        if(kind==4u){a=origin;b=origin+float3(s,0,0);c=origin+float3(0.25f*s,0.1f*s,0);q1=origin+float3(0.75f*s,0.1f*s,0);}
        if(kind==5u){a=b=origin;c=origin+float3(-0.5f*s,0.2f*s,0);q1=origin+float3(0.5f*s,0.2f*s,0);}
        float oldS,oldT,newS,newT;
        closestPointsOnSegments(a,b,c,q1,oldS,oldT,oldP,oldQ);
        scaledClosestPointsOnSegments(a,b,c,q1,newS,newT,newP,newQ);
        oldBary=float3(oldS,oldT,0);newBary=float3(newS,newT,0);
    }
    const uint base=global*16u;
    for(uint j=0;j<16u;++j)output[base+j]=float4(0.0f);
    output[base]=float4(float(kind),s,shift,float(global));
    output[base+1u]=float4(point,1);output[base+2u]=float4(a,1);
    output[base+3u]=float4(b,1);output[base+4u]=float4(c,1);output[base+5u]=float4(q1,1);
    output[base+6u]=float4(oldBary,1);output[base+7u]=float4(newBary,1);
    output[base+8u]=float4(oldP,1);output[base+9u]=float4(newP,1);
    output[base+10u]=float4(oldQ,1);output[base+11u]=float4(newQ,1);
}
}
