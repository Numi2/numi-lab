#ifndef METALROBO_VISUAL_PBR_METAL
#define METALROBO_VISUAL_PBR_METAL

#include <metal_stdlib>

namespace metalrobo_pbr {

using namespace metal;

inline float distributionGGX(
    const float noH,
    const float perceptualRoughness
) {
    const float alpha =
        max(perceptualRoughness * perceptualRoughness, 0.0025f);
    const float alphaSquared = alpha * alpha;
    const float denominator =
        noH * noH * (alphaSquared - 1.0f) + 1.0f;
    return alphaSquared /
        max(M_PI_F * denominator * denominator, 1.0e-7f);
}

inline float visibilitySmithGGX(
    const float noV,
    const float noL,
    const float perceptualRoughness
) {
    const float alpha =
        max(perceptualRoughness * perceptualRoughness, 0.0025f);
    const float alphaSquared = alpha * alpha;
    const float lambdaV =
        noL * sqrt(max(
            noV * noV * (1.0f - alphaSquared) +
                alphaSquared,
            0.0f
        ));
    const float lambdaL =
        noV * sqrt(max(
            noL * noL * (1.0f - alphaSquared) +
                alphaSquared,
            0.0f
        ));
    return 0.5f / max(lambdaV + lambdaL, 1.0e-7f);
}

inline float smithG1GGX(
    const float noX,
    const float perceptualRoughness
) {
    const float alpha =
        max(perceptualRoughness * perceptualRoughness, 0.0025f);
    const float alphaSquared = alpha * alpha;
    return 2.0f * noX /
        max(
            noX + sqrt(
                alphaSquared +
                (1.0f - alphaSquared) * noX * noX
            ),
            1.0e-7f
        );
}

inline float3 fresnelSchlick(
    const float voH,
    const float3 f0
) {
    const float factor =
        pow(clamp(1.0f - voH, 0.0f, 1.0f), 5.0f);
    return f0 + (1.0f - f0) * factor;
}

inline float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) |
        ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) |
        ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) |
        ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) |
        ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

inline float2 hammersley(
    const uint index,
    const uint count
) {
    return float2(
        (float(index) + 0.5f) / float(max(count, 1u)),
        radicalInverseVdC(index)
    );
}

inline void basis(
    const float3 normal,
    thread float3& tangent,
    thread float3& bitangent
) {
    const float sign = copysign(1.0f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    tangent = float3(
        1.0f + sign * normal.x * normal.x * a,
        sign * b,
        -sign * normal.x
    );
    bitangent = float3(
        b,
        sign + normal.y * normal.y * a,
        -normal.y
    );
}

// Eric Heitz, "Sampling the GGX Distribution of Visible Normals".
inline float3 sampleGGXVNDF(
    const float3 view,
    const float alpha,
    const float2 sample
) {
    const float3 stretchedView = normalize(
        float3(alpha * view.xy, view.z)
    );
    float3 tangent;
    float3 bitangent;
    basis(stretchedView, tangent, bitangent);
    const float radius = sqrt(sample.x);
    const float phi = 2.0f * M_PI_F * sample.y;
    const float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    const float s = 0.5f * (1.0f + stretchedView.z);
    t2 = mix(
        sqrt(max(0.0f, 1.0f - t1 * t1)),
        t2,
        s
    );
    const float3 normal = t1 * tangent + t2 * bitangent +
        sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) *
            stretchedView;
    return normalize(float3(
        alpha * normal.xy,
        max(normal.z, 0.0f)
    ));
}

inline float3 multiscatterSpecular(
    const float3 prefilteredRadiance,
    const float3 irradiance,
    const float3 f0,
    const float2 dfg
) {
    const float3 fssEss = f0 * dfg.x + dfg.y;
    const float ess = max(dfg.x + dfg.y, 1.0e-5f);
    const float ems = 1.0f - ess;
    const float3 favg = f0 + (1.0f - f0) / 21.0f;
    const float3 fms =
        fssEss * favg /
        max(1.0f - ems * favg, float3(1.0e-5f));
    return prefilteredRadiance * fssEss +
        irradiance * M_1_PI_F * fms * ems;
}

inline float3 multiscatterDiffuseEnergy(
    const float3 f0,
    const float2 dfg
) {
    const float3 fssEss = f0 * dfg.x + dfg.y;
    const float ess = max(dfg.x + dfg.y, 1.0e-5f);
    const float ems = 1.0f - ess;
    const float3 favg = f0 + (1.0f - f0) / 21.0f;
    const float3 fms =
        fssEss * favg /
        max(1.0f - ems * favg, float3(1.0e-5f));
    return max(1.0f - (fssEss + fms * ems), 0.0f);
}

} // namespace metalrobo_pbr

#endif
